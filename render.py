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

import math
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from nexel_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, NexelParams, get_combined_args
from nexel_renderer import NexelModel
from utils.render_utils import generate_ellipse_path, generate_camlist, minicams_to_json
import matplotlib.pyplot as plt
from diff_nexel_rasterization import manual_grad
import json

def render_set(model_path, name, iteration, views, nexels, pipeline, background, override_settings, render_gt):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    if render_gt:
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)

    info = {}

    # Warm-up GPUs for more accurate timing
    for idx, view in enumerate(views):
        if idx > 5:
            break
        render_pkg = render(iteration, False, view, nexels, pipeline, background, 1.0, override_settings=-1)

    total_time = 0.0

    rand_nums = torch.rand_like(nexels.get_xyz[:,0]).detach().cpu().numpy()
    vis_features = plt.cm.hsv(rand_nums)[:,:3]
    vis_features = torch.from_numpy(vis_features).to("cuda").to(dtype=torch.float32)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        render_pkg = render(iteration, False, view, nexels, pipeline, background, 1.0, compute_geometric=False, use_vis_features=False, override_settings=override_settings)
        image = render_pkg["render"]
        end_event.record()
        torch.cuda.synchronize()
        runtime = start_event.elapsed_time(end_event)
        total_time += runtime
        alpha = render_pkg["alpha"].permute((2, 0, 1))

        rendering = image + (1 - alpha) * background[:,None,None]

        if render_gt:
            gt = view.original_image[0:3, :, :]
            gt_image = view.original_image.cuda()

            if view.alpha_mask is not None:
                alpha_mask = view.alpha_mask.cuda()
                gt = alpha_mask * gt + (1 - alpha_mask) * background[:,None,None]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
    total_time /= len(views)
    info["num_points"] = nexels.num_points
    info["fps"] = 1000.0 / total_time
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "info.json"), "w") as fp:
        json.dump(info, fp, indent=True)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, nexel_settings : NexelParams, skip_train : bool, skip_test : bool, skip_ellipse : bool, override_settings: int):
    with torch.no_grad():
        nexels = NexelModel(dataset.sh_degree, dataset.texture_sh_degree, nexel_settings)
        scene = Scene(dataset, nexels, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), nexels, pipeline, background, override_settings, True)

        if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), nexels, pipeline, background, override_settings, True)
        
        if not skip_ellipse:
            func = lambda poses : generate_ellipse_path(poses, 120)

            ellipseCameras = generate_camlist(scene.getTrainCameras(), func)
            render_set(
                dataset.model_path,
                "ellipse",
                scene.loaded_iter,
                ellipseCameras,
                nexels,
                pipeline,
                background,
                override_settings,
                False,
            )
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    nexel_params = NexelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_ellipse", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(False)

    render_sets(
        model.extract(args), args.iteration, pipeline.extract(args), nexel_params.extract(args),
        args.skip_train, args.skip_test, args.skip_ellipse, args.override_settings
    )