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
import torch.nn.functional as F
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    distances : np.array
    def filter(self, mask):
        return BasicPointCloud(
            points=self.points[mask,:],
            colors=self.colors[mask,:],
            normals=self.normals[mask,:],
            distances=self.distances[mask],
        )

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getIntrinsicsMatrix(fovX, fovY, width, height):
    fx = 0.5 * float(width) / math.tan(0.5 * fovX)
    fy = 0.5 * float(height) / math.tan(0.5 * fovY)
    K = torch.zeros(3, 3)

    K[0,0] = fx
    K[0,2] = 0.5 * float(width)
    K[1,1] = fy
    K[1,2] = 0.5 * float(height)
    K[2,2] = 1
    return K

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_rays(
    height, width, camtoworlds, Ks, z_depth: bool = True
):
    """Convert depth maps to 3D points

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        points: 3D points in the world coordinate system [..., H, W, 3]
    """
    assert camtoworlds.shape[-2:] == (
        4,
        4,
    ), f"Invalid viewmats shape: {camtoworlds.shape}"
    assert Ks.shape[-2:] == (3, 3), f"Invalid Ks shape: {Ks.shape}"

    with torch.no_grad():
        device = camtoworlds.device

        x, y = torch.meshgrid(
            torch.arange(width, device=device),
            torch.arange(height, device=device),
            indexing="xy",
        )  # [H, W]

        fx = Ks[..., 0, 0]  # [...]
        fy = Ks[..., 1, 1]  # [...]
        cx = Ks[..., 0, 2]  # [...]
        cy = Ks[..., 1, 2]  # [...]

        # camera directions in camera coordinates
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cx[..., None, None] + 0.5) / fx[..., None, None],
                    (y - cy[..., None, None] + 0.5) / fy[..., None, None],
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )  # [..., H, W, 3]

        # ray directions in world coordinates
        directions = torch.einsum(
            "...ij,...hwj->...hwi", camtoworlds[..., :3, :3], camera_dirs
        )  # [..., H, W, 3]
        origins = camtoworlds[..., :3, -1]  # [..., 3]

        if not z_depth:
            directions = F.normalize(directions, dim=-1)

    return origins, directions

class DepthToNormal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, origins, directions, depths):
        points = origins[..., None, None, :] + depths * directions
        dx = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]
        dy = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]
        cross_vals = torch.cross(dx, dy, dim=-1)
        clamp_val = 1e-12
        cross_norm = torch.sqrt(torch.sum(cross_vals**2, dim=-1))
        mask = cross_norm < clamp_val
        cross_norm[mask] = clamp_val

        normals = cross_vals / cross_norm[...,None]
        normals = F.pad(normals, (0, 0, 1, 1, 1, 1), value=0.0)  # [..., H, W, 3]
        ctx.save_for_backward(directions, depths, normals, cross_norm, mask, dx, dy)
        ctx.clamp_val = clamp_val
        return normals
    
    @staticmethod
    def backward(ctx, grad_normals):
        directions, depths, normals, cross_norm, mask, dx, dy = ctx.saved_tensors
        clamp_val = ctx.clamp_val
        grad_normals = grad_normals[1:-1,1:-1,:]
        grad_cross = torch.zeros_like(grad_normals)
        normals = normals[1:-1,1:-1,:]
        
        grad_cross[...,0] += grad_normals[...,0] * (normals[...,1]**2 + normals[...,2]**2)
        grad_cross[...,0] -= grad_normals[...,1] * (normals[...,0] * normals[...,1])
        grad_cross[...,0] -= grad_normals[...,2] * (normals[...,0] * normals[...,2])

        grad_cross[...,1] -= grad_normals[...,0] * (normals[...,1] * normals[...,0])
        grad_cross[...,1] += grad_normals[...,1] * (normals[...,0]**2 + normals[...,2]**2)
        grad_cross[...,1] -= grad_normals[...,2] * (normals[...,1] * normals[...,2])

        grad_cross[...,2] -= grad_normals[...,0] * (normals[...,0] * normals[...,2])
        grad_cross[...,2] -= grad_normals[...,1] * (normals[...,1] * normals[...,2])
        grad_cross[...,2] += grad_normals[...,2] * (normals[...,0]**2 + normals[...,1]**2)

        grad_cross /= cross_norm[...,None]
        grad_cross[mask,:] = grad_normals[mask,:] / clamp_val

        grad_dx = torch.cross(dy, grad_cross, dim=-1)
        grad_dy = torch.cross(grad_cross, dx, dim=-1)

        grad_points = torch.zeros_like(directions)
        grad_points[...,2:,1:-1,:] += grad_dx
        grad_points[...,:-2:,1:-1,:] -= grad_dx
        grad_points[...,1:-1,2:,:] += grad_dy
        grad_points[...,1:-1,:-2,:] -= grad_dy

        grad_depths = torch.sum(grad_points * directions, dim=-1, keepdim=True)
        return (None, None, grad_depths)
        # return (None, None, None)

def debug_identity(name, x):
    return DebugIdentity.apply(name, x)

class DebugIdentity(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        name,
        x):
        ctx.name = name
        return x
    
    @staticmethod
    def backward(ctx, v_x):
        name = ctx.name
        print("debug backward", name, v_x.shape, v_x.min().item(), v_x.max().item(), v_x.mean().item())
        return None, v_x

def depth_to_normal_old(
    depths, camtoworlds, Ks, z_depth: bool = True
):
    origins, directions = get_rays(depths.shape[0], depths.shape[1], camtoworlds, Ks, z_depth=z_depth)  # [..., H, W, 3]
    points = origins[..., None, None, :] + depths * directions
    dx = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]
    dy = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]
    cross_vals = torch.cross(dx, dy, dim=-1)
    normals = F.normalize(cross_vals, dim=-1)
    normals = F.pad(normals, (0, 0, 1, 1, 1, 1), value=0.0)  # [..., H, W, 3]
    return normals

def depth_to_normal(
    depths, camtoworlds, Ks, z_depth: bool = True
):
    """Convert depth maps to surface normals

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        normals: Surface normals in the world coordinate system [..., H, W, 3]
    """
    origins, directions = get_rays(depths.shape[0], depths.shape[1], camtoworlds, Ks, z_depth=z_depth)  # [..., H, W, 3]
    normals = DepthToNormal.apply(origins, directions, depths)
    return normals


def texture_dims_to_int_coords(texture_dims):
    idxs = torch.arange(texture_dims.shape[0], dtype=torch.int64, device=texture_dims.device)
    hws = texture_dims[:,0] * texture_dims[:,1]
    ids = torch.repeat_interleave(idxs[hws!=0], hws[hws!=0], dim=0)

    query_dims = texture_dims[ids,:]
    total_size = torch.sum(hws).item()
    local_idxs = torch.arange(total_size, dtype=torch.int64, device=texture_dims.device) - query_dims[:,2]
    uu = local_idxs // query_dims[:,1]
    vv = local_idxs % query_dims[:,1]
    uv = torch.stack([uu, vv], dim=-1)
    return ids, uv

def texture_dims_to_query(texture_dims):
    idxs = torch.arange(texture_dims.shape[0], dtype=torch.int64, device=texture_dims.device)
    hws = texture_dims[:,0] * texture_dims[:,1]
    ids = torch.repeat_interleave(idxs, hws, dim=0)

    query_dims = texture_dims[ids,:]
    total_size = torch.sum(hws).item()
    local_idxs = torch.arange(total_size, dtype=torch.int64, device=texture_dims.device) - query_dims[:,2]
    uu = (local_idxs // query_dims[:,1]).float()
    vv = (local_idxs % query_dims[:,1]).float()

    mask0 = query_dims[:,0] == 1
    uu[mask0] = 0.5
    uu[~mask0] = uu[~mask0] / (query_dims[~mask0,0].float() - 1)

    mask1 = query_dims[:,1] == 1
    vv[mask1] = 0.5
    vv[~mask1] = vv[~mask1] / (query_dims[~mask1,1].float() - 1)

    uv = torch.stack([uu, vv], dim=-1)
    return ids, uv