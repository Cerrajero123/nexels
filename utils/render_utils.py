import torch
import numpy as np
import scipy

from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
import json

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(lookdir, up, position):
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def focus_point_fn(poses):
    directions, origins = poses[:,:3,2:3], poses[:,:3,3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:,0]
    return focus_pt

def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

    Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform
    
    return poses_recentered, transform

def average_pose(poses):
    position = poses[:,:3,3].mean(0)
    z_axis = poses[:,:3,2].mean(0)
    up = poses[:,:3,1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world

def generate_ellipse_path(
    poses, n_frames=120
):
    variation = 0.0
    phase = 0.0
    height = 0.0

    do_z = True

    center = focus_point_fn(poses)
    if do_z:
        offset = np.array([center[0], center[1], height])
    else:
        offset = np.array([center[0], height, center[2]])

    sc = np.percentile(np.abs(poses[:,:3,3] - offset), 90, axis=0)
    low = -sc + offset
    high = sc + offset

    z_low = np.percentile((poses[:,:3,3]), 10, axis=0)
    z_high = np.percentile((poses[:,:3,3]), 90, axis=0)

    def get_positions(theta):
        if do_z:
            return np.stack(
                [
                    low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                    low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                    variation * (
                        z_low[2]
                        + (z_high - z_low)[2]
                        * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                    )
                    + height,
                ],
                -1,
            )
        else:
            return np.stack(
                [
                    low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                    variation * (
                        z_low[1]
                        + (z_high - z_low)[1]
                        * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                    )
                    + height,
                    low[2] + (high - low)[2] * (np.sin(theta) * 0.5 + 0.5),
                ],
                -1,
            )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    positions = positions[:-1]

    avg_up = poses[:,:3,1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    if do_z:
        return np.stack([viewmatrix(center - p, up, p) for p in positions])
    return np.stack([viewmatrix(p - center, up, p) for p in positions])

def generate_camlist(original_cameras, func):
    original_poses = []
    for i in range(len(original_cameras)):
        pose = original_cameras[i].world_view_transform.inverse().detach().cpu().numpy().T
        original_poses.append(pose[:3,:])
    original_poses = np.stack(original_poses, axis=0)

    centers = original_poses[:,:3,3]
    mean_center = np.mean(centers, axis=0)
    centers_diff = centers - mean_center[None,:]
    _, eigvec = np.linalg.eigh(centers_diff.T @ centers_diff)
    eigvec = np.flip(eigvec, axis=-1)

    if np.linalg.det(eigvec) < 0:
        eigvec[:,2] = -eigvec[:,2]
    
    original_poses[:,:3,3] -= mean_center[None,:]
    original_poses = np.einsum("bij,ik->bkj", original_poses, eigvec)

    new_poses = func(original_poses)

    new_poses = np.einsum("bij,ik->bkj", new_poses, eigvec.T)
    new_poses[:,:3,3] += mean_center[None,:]

    old_camera = original_cameras[0]
    new_cameras = []
    for i in range(new_poses.shape[0]):
        R = np.linalg.inv(new_poses[i,:3,:3])
        T = -R @ new_poses[i,:3,3:]
        R = R.transpose()
        T = T[:,0]
        fovx = old_camera.FoVx
        fovy = old_camera.FoVy
        znear = old_camera.znear
        zfar = old_camera.zfar
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        new_camera = MiniCam(
            old_camera.image_width,
            old_camera.image_height,
            fovy,
            fovx,
            znear,
            zfar,
            world_view_transform,
            full_proj_transform,
        )
        new_cameras.append(new_camera)
    return new_cameras

def minicams_to_json(minicams):
    data = []
    for i in range(len(minicams)):
        entry = {
            'width': minicams[i].image_width,
            'height': minicams[i].image_height,
            'fovy': minicams[i].FoVy,
            'fovx': minicams[i].FoVx,
            'znear': minicams[i].znear,
            'zfar': minicams[i].zfar,
            'world_view_transform': [x.tolist() for x in minicams[i].world_view_transform],
            'full_proj_transform': [x.tolist() for x in minicams[i].full_proj_transform],
        }
        data.append(entry)
    return data

def path_from_json(data):
    traj = []
    for i in range(len(data)):
        new_camera = MiniCam(
            data[i]['width'],
            data[i]['height'],
            data[i]['fovy'],
            data[i]['fovx'],
            data[i]['znear'],
            data[i]['zfar'],
            torch.tensor(data[i]['world_view_transform']).cuda(),
            torch.tensor(data[i]['full_proj_transform']).cuda(),
        )
        traj.append(new_camera)
    return traj