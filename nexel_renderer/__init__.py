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

import torch
import math
from diff_nexel_rasterization import NexelRasterizationSettings, NexelHashGridSettings, NexelMLPSettings, NexelRasterizer
from scene.nexel_model import NexelModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import depth_to_normal

def render(
    step: int, is_training: bool, viewpoint_camera, pc : NexelModel,
    pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
    override_color = None, compute_geometric=False,
    use_vis_features=False, override_settings=-1,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    settings = pipe.bool_settings
    if override_settings != -1:
        settings = override_settings
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") 
    error_tensor = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
        error_tensor.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # The rasterizer uses a different camera convention
    viewmat_orig = viewpoint_camera.world_view_transform
    campos_orig = viewpoint_camera.camera_center
    viewmat = torch.clone(viewmat_orig)
    campos = torch.clone(campos_orig)
    viewmat[1,...] = viewmat_orig[2,...]
    viewmat[2,...] = -viewmat_orig[1,...]
    campos[1,...] = campos_orig[2,...]
    campos[2,...] = -campos_orig[1,...]

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    gammas = pc.get_gamma
    color_coeffs = pc.get_features.reshape(means3D.shape[0], -1)
    colors = None

    if not pipe.no_texture_clamp:
        settings = settings | (1<<14)

    if not pipe.no_texture_antialiasing:
        settings = settings | (1<<19)

    if use_vis_features:
        color_coeffs = None
        colors = pc.get_vis_features
        settings = settings | (1<<28)

    if pipe.debug:
        settings = settings | (1<<29)

    if override_color is not None:
        color_coeffs = None
        colors = override_color

    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    max_coeff = (pc.max_sh_degree + 1)**2
    texture_max_coeff = (pc.max_texture_app_degree + 1)**2
    raster_settings = NexelRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmat.contiguous(),
        ks=viewpoint_camera.k_matrix.contiguous(),
        sh_degree=pc.active_app_degree,
        max_coeff=max_coeff,
        texture_sh_degree=pc.active_texture_app_degree,
        texture_max_coeff=texture_max_coeff,
        grid_threshold_factor=pipe.grid_threshold_factor,
        campos=campos.contiguous(),
        texture_limit=pc.texture_limit,
        settings=settings,
    )

    rasterizer = NexelRasterizer(raster_settings=raster_settings, hash_grid_settings=pc.hash_grid_settings, mlp_settings=pc.mlp_settings)

    outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        colors_precomp = colors,
        color_coeffs = color_coeffs,
        opacities = opacity,
        gammas = gammas,
        scales = scales,
        rotations = rotations,
        errors_buffer = error_tensor,

        grid = pc.grid,
        weights = pc.weights,
        biases = pc.biases,
    )
    rendered_alphas, rendered_image, rendered_normals, rendered_depth, out_error_buffer, radii, extra_info = outputs
    
    rendered_image = rendered_image.clamp(0, 1)

    texture_ids = extra_info["texture_ids"]
    texture_weights = extra_info["texture_weights"]
    texture_depths = extra_info["texture_depths"]
    if compute_geometric:
        cam2world = torch.linalg.inv(viewmat).transpose(0, 1)
        rendered_normals = torch.einsum(
            "ij,hwj->hwi", cam2world[:3, :3], rendered_normals
        )

        mean_depth = torch.zeros_like(rendered_depth)
        mask = rendered_alphas > 0
        mean_depth[mask] = rendered_depth[mask] / rendered_alphas[mask]
    if compute_geometric:
        if texture_ids.numel() > 0:
            weights = texture_weights * (texture_ids != -1).float()
            weight_indices = torch.argmax(weights, dim=0)
            surface_depth = torch.gather(
                texture_depths,
                0,
                weight_indices[None,...]
            )
            surface_depth = surface_depth[0,...,None]
        else:
            surface_depth = mean_depth
    if compute_geometric:
        rendered_normals_from_depth = depth_to_normal(mean_depth, cam2world, viewpoint_camera.k_matrix, z_depth=True)

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "error_tensor": error_tensor,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "alpha": rendered_alphas,
        "out_error_buffer": out_error_buffer,
        "texture_ids": texture_ids,
        "texture_weights": texture_weights,
        "texture_depths": texture_depths,
        "out_no_texture": extra_info["out_no_texture"],
        "out_texture": extra_info["out_texture"],
    }
    if compute_geometric:
        out_geometric = {
            "depth" : rendered_depth,
            "surface_depth" : surface_depth,
            "normals": rendered_normals,
            "normals_from_depth": rendered_normals_from_depth,
        }
        out.update(out_geometric)

    return out
