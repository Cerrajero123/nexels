#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from nexel_renderer import render, network_gui
from diff_nexel_rasterization import manual_grad
import sys
from scene import Scene, NexelModel
from utils.general_utils import safe_state, get_expon_lr_func, seed_everything
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, NexelParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def training(dataset, opt, pipe, nexel_settings, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = None
    nexels = NexelModel(dataset.sh_degree, dataset.texture_sh_degree, nexel_settings, opt.optimizer_type)
    scene = Scene(dataset, nexels)
    nexels.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        nexels.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    dump_viewpoint = list(viewpoint_stack)[0]

    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    torch.cuda.empty_cache()

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        nexels.update_learning_rate(iteration)

        if opt.plain_sh_unlock_steps <= 0:
            nexels.maxDegree()
        elif iteration % opt.plain_sh_unlock_steps == 0:
            nexels.oneupDegree()
        
        if opt.texture_sh_unlock_steps <= 0:
            nexels.maxTextureDegree()
        elif iteration % opt.texture_sh_unlock_steps == 0:
            nexels.oneupTextureDegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        compute_geometric = opt.lambda_normal > 0
        render_pkg = render(iteration-1, True, viewpoint_cam, nexels, pipe, bg, compute_geometric=compute_geometric)
        image, alpha = render_pkg["render"], render_pkg["alpha"]
        out_error_buffer = render_pkg["out_error_buffer"]
        viewspace_point_tensor, error_tensor = render_pkg["viewspace_points"], render_pkg["error_tensor"]
        visibility_filter, radii = render_pkg["visibility_filter"], render_pkg["radii"]
        alpha = alpha.permute((2, 0, 1))
        image = image + (1 - alpha) * bg[:,None,None]

        gt_image = viewpoint_cam.original_image.cuda()

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            gt_image = alpha_mask * gt_image + (1 - alpha_mask) * bg[:,None,None]
        
        error_image = torch.mean(torch.abs(image - gt_image), dim=0)[...,None]
        # Use manual_grad to include error information into backwards pass
        out_error_buffer = manual_grad(out_error_buffer, error_image)
        # Loss
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        Lphotometric = loss.item()
        
        # Includes out_error_buffer in the computation graph. Note that .mean() is irrelevant
        loss = loss + out_error_buffer.mean()

        Lalpha = 0.0
        if opt.lambda_alpha > 0 and iteration >= opt.start_alpha_loss:
            weights = render_pkg["texture_weights"] * (render_pkg["texture_ids"] != -1).float()
            Lalpha = opt.lambda_alpha * torch.mean(1 - torch.sum(weights, dim=0))
            loss += Lalpha
            Lalpha = Lalpha.item()

        Lbg_alpha = 0.0
        if opt.lambda_bg_alpha > 0:
            Lbg_alpha = opt.lambda_bg_alpha * torch.mean(1 - alpha)
            loss += Lbg_alpha
            Lbg_alpha = Lbg_alpha.item()

        Lnormal = 0.0
        if opt.lambda_normal > 0:
            normals = render_pkg["normals"]
            normals_from_depth = render_pkg["normals_from_depth"]
            Lnormal = opt.lambda_normal * torch.mean(1.0 - torch.sum(normals * normals_from_depth, dim=-1))
            loss += Lnormal
            Lnormal = Lnormal.item()
        
        Ltexture = 0.0
        if opt.lambda_texture > 0:
            out_texture = render_pkg["out_texture"]
            weights = render_pkg["texture_weights"] * (render_pkg["texture_ids"] != -1).float()
            weights = torch.sum(weights, dim=0, keepdim=True)
            out_texture = out_texture / (weights + 1e-6)
            Ltexture = opt.lambda_texture * l1_loss(out_texture, gt_image)
            loss += Ltexture
            Ltexture = Ltexture.item()

        Lopacity = 0.0
        if opt.lambda_opacity > 0:
            opacity = nexels.get_opacity
            Lopacity = opt.lambda_opacity * torch.mean(opacity)
            loss += Lopacity
            Lopacity = Lopacity.item()
            
        Lgrid = 0.0
        if opt.lambda_grid > 0:
            grid_scalings = nexels.hash_grid_settings.scale_min * torch.pow(nexels.hash_grid_settings.scale_factor, torch.arange(nexels.hash_grid_settings.num_levels, device="cuda").float())
            Lgrid = opt.lambda_grid * torch.sum(nexels.grid**2 / grid_scalings[:,None,None]**3)
            loss += Lgrid
            Lgrid = Lgrid.item()
        assert torch.all(torch.isfinite(loss)).item()
        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss, 
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, 1., None, False, False, -1),
                False,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving primitives".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                nexels.max_radii2D[visibility_filter] = torch.max(nexels.max_radii2D[visibility_filter], radii[visibility_filter])
                nexels.add_densification_stats(viewspace_point_tensor, error_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 1:
                    nexels.add_new_gs(iteration, opt.densify_grad_proportion, opt.grad_mode, 0.005, opt.densify_stochastic)
                
            # Optimizer step
            if iteration < opt.iterations:
                nexels.optimizer.step()
                nexels.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((nexels.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
        tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene,
        renderFunc, renderArgs, train_test_exp
    ):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration % 1000 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                total_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(renderFunc(0, False, viewpoint, scene.nexels, *renderArgs)["render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime
                    gt_image = viewpoint.original_image.cuda()

                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                total_time /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.nexels.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.nexels.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lParams = ModelParams(parser)
    oParams = OptimizationParams(parser)
    pParams = PipelineParams(parser)
    nParams = NexelParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000,])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000,])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    seed_everything(0)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    prepare_output_and_logger(args)

    training(
        lParams.extract(args), oParams.extract(args), pParams.extract(args), nParams.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint
    )

    # All done
    print("\nTraining complete.")
