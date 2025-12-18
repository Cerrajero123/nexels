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
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, random_quat_tensor, texture_dims_to_int_coords, texture_dims_to_query
from utils.general_utils import strip_symmetric, build_scaling_rotation
from diff_nexel_rasterization import NexelRasterizationSettings, NexelHashGridSettings, NexelMLPSettings, NexelField

class NexelModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
    
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        # Ensure gamma terms are at least 1
        self.gamma_activation = lambda x : torch.exp(x) + 1

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, app_degree, texture_app_degree, nexel_settings, optimizer_type="default"):
        self.texture_limit = nexel_settings.texture_limit
        self.optimizer_type = optimizer_type
        self.active_app_degree = 0
        self.active_texture_app_degree = 0
        self.max_sh_degree = app_degree
        self.max_texture_app_degree = texture_app_degree  
        self._xyz = torch.empty(0)
        self._vis_features = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._gamma = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.error_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.cap_max_init = nexel_settings.cap_max_init
        self.cap_max_final = nexel_settings.cap_max_final

        # Initialize Instant-NGP
        minres = nexel_settings.minres
        maxres = nexel_settings.maxres
        num_levels = nexel_settings.num_levels

        dimensions = 3 * torch.ones((num_levels,), dtype=torch.int32, device="cuda")

        scalings = torch.exp(torch.linspace(math.log(minres), math.log(maxres), num_levels))
        scalings = scalings.to(dtype=torch.float32)

        self.hash_grid_settings = NexelHashGridSettings(
            num_levels=num_levels,
            table_size=2**nexel_settings.log_hash_table_size,
            num_features=2,
            scale_min=scalings[0].item(),
            scale_factor=scalings[1].item()/scalings[0].item(),
            prime0=1,
            prime1=-1640531535,
            prime2=805459861,
        )

        self.grid = nn.Parameter(0.01 * torch.ones(
            self.hash_grid_settings.num_levels,
            self.hash_grid_settings.table_size,
            self.hash_grid_settings.num_features,
            device="cuda"
        ))

        self.mlp_settings = NexelMLPSettings(
            num_layers=nexel_settings.num_layers,
            input_dim=self.hash_grid_settings.num_levels * self.hash_grid_settings.num_features,
            hidden_dim=nexel_settings.mlp_hidden_dim,
            output_dim=nexel_settings.mlp_output_dim,
            input_bias=nexel_settings.mlp_input_bias, # optional shift to input
            activation_mode=1, # ReLU activation
            output_activation_mode=0, # No activation
        )

        num_weights = self.mlp_settings.input_dim * self.mlp_settings.hidden_dim +\
                     (self.mlp_settings.num_layers - 1) * self.mlp_settings.hidden_dim * self.mlp_settings.hidden_dim +\
                     self.mlp_settings.hidden_dim * self.mlp_settings.output_dim
        num_biases = self.mlp_settings.num_layers * self.mlp_settings.hidden_dim + self.mlp_settings.output_dim

        weight_factor = 1.0
        bias_factor = 0.0
        weights_init = [
            weight_factor * (torch.rand(self.mlp_settings.input_dim * self.mlp_settings.hidden_dim,) * 2 - 1) / math.sqrt(self.mlp_settings.input_dim),
        ] + [
            weight_factor * (torch.rand(self.mlp_settings.hidden_dim * self.mlp_settings.hidden_dim,) * 2 - 1) / math.sqrt(self.mlp_settings.hidden_dim) \
            for i in range(self.mlp_settings.num_layers - 1)
        ] + [
            weight_factor * (torch.rand(self.mlp_settings.hidden_dim * self.mlp_settings.output_dim,) * 2 - 1) / math.sqrt(self.mlp_settings.hidden_dim),
        ]
        weights_init = torch.cat(weights_init, dim=0).to(device="cuda")

        # Note that the biases aren't actually used in the NVIDIA Instant-NGP implementation
        biases_init = [
            bias_factor * (torch.rand(self.mlp_settings.hidden_dim,) * 2 - 1) / math.sqrt(self.mlp_settings.input_dim),
        ] + [
            bias_factor * (torch.rand(self.mlp_settings.hidden_dim,) * 2 - 1) / math.sqrt(self.mlp_settings.hidden_dim) \
            for i in range(self.mlp_settings.num_layers - 1)
        ] + [
            bias_factor * (torch.rand(self.mlp_settings.output_dim,) * 2 - 1) / math.sqrt(self.mlp_settings.hidden_dim),
        ]
        biases_init = torch.cat(biases_init, dim=0).to(device="cuda")

        assert weights_init.shape[0] == num_weights
        assert biases_init.shape[0] == num_biases

        self.weights = nn.Parameter(weights_init)
        self.biases = nn.Parameter(biases_init)

    def get_cap_max(self, iteration):
        return self.cap_max_final

    def capture(self):
        return (
            self.active_app_degree,
            self.active_texture_app_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._gamma,
            self.hash_grid_settings,
            self.grid,
            self.mlp_settings,
            self.weights,
            self.biases,
            self.max_radii2D,
            self.error_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_app_degree,
        self.activate_texture_app_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation,
        self._opacity,
        self._gamma,
        self.hash_grid_settings,
        self.grid,
        self.mlp_settings,
        self.weights,
        self.biases,
        self.max_radii2D, 
        error_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.error_accum = error_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def num_points(self):
        return self.get_xyz.shape[0]

    @property
    def get_scaling(self):
        scales = torch.zeros_like(self._scaling)
        scales[:,:2] = self.scaling_activation(self._scaling[:,:2])
        # 3rd axis is set to 0
        scales[:,2:] = 1e-9 * torch.mean(scales[:,:2], dim=-1, keepdim=True).detach()
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_vis_features(self):
        vis_features = self._vis_features
        return vis_features
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_gamma(self):
        return self.gamma_activation(self._gamma)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupDegree(self):
        if self.active_app_degree < self.max_sh_degree:
            self.active_app_degree += 1

    def oneupTextureDegree(self):
        if self.active_texture_app_degree < self.max_texture_app_degree:
            self.active_texture_app_degree += 1

    def maxDegree(self):
        self.active_app_degree = self.max_sh_degree

    def maxTextureDegree(self):
        self.active_texture_app_degree = self.max_texture_app_degree
    
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float, num_initial_points: int=-1, use_original_scale_init=False):
        self.spatial_lr_scale = spatial_lr_scale
        mask = pcd.distances > 0
        cap_max = self.cap_max_init
        if mask.sum().item() > cap_max:
            target_num = cap_max
            add_indices = np.nonzero(mask)[0][:target_num]
            add_mask = np.zeros_like(mask)
            add_mask[add_indices] = 1
        else:
            add_mask = mask
        self.init_pcd(pcd.filter(add_mask), cam_infos, spatial_lr_scale, use_original_scale_init)

    def init_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float, use_original_scale_init=False):
        orig_positions = np.asarray(pcd.points)
        positions = np.copy(orig_positions)
        # Nexel rasterizer uses different camera convention
        positions[:,1] = orig_positions[:,2]
        positions[:,2] = -orig_positions[:,1]
        fused_point_cloud = torch.tensor(positions).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        distances = torch.tensor(pcd.distances).float().cuda()

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        if use_original_scale_init:
            print("Using original scale init")
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(positions).float().cuda()), 0.0000001)
            scales = torch.log(0.5 * torch.sqrt(dist2))[...,None].repeat(1, 3)
        else:
            print("Using new scale init")
            scales = torch.log(0.5 * distances)[...,None].repeat(1, 3)
        scales[:,-1] = -8.0
        rots = random_quat_tensor(fused_point_cloud.shape[0]).to(device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # Initialize gammas to be near 1 (after activation)
        gammas = -5.0 * torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._vis_features = nn.Parameter(torch.rand((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._gamma = nn.Parameter(gammas.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.error_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._gamma], 'lr': training_args.gamma_lr_init, "name": "gamma"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self.grid], 'lr': training_args.grid_lr_init, "name": "hash_grid"},
            {'params': [self.weights, self.biases], 'lr': training_args.mlp_lr_init, "name": "mlp"},
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.gamma_scheduler_args = get_expon_lr_func(lr_init=training_args.gamma_lr_init,
                                            lr_final=training_args.gamma_lr_final,
                                            lr_delay_mult=training_args.gamma_lr_delay_mult,
                                            max_steps=training_args.gamma_lr_max_steps)

        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init,
                                                    lr_final=training_args.grid_lr_final,
                                                    lr_delay_mult=training_args.grid_lr_delay_mult,
                                                    max_steps=training_args.grid_lr_max_steps)
        
        self.mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_lr_init,
                                                    lr_final=training_args.mlp_lr_final,
                                                    lr_delay_mult=training_args.mlp_lr_delay_mult,
                                                    max_steps=training_args.mlp_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "gamma":
                lr = self.gamma_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "hash_grid":
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp":
                lr = self.mlp_scheduler_args(iteration)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._vis_features.shape[1]):
            l.append('vis_f_{}'.format(i))
        l.append('opacity')
        for i in range(self._gamma.shape[1]):
            l.append('gamma_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_tensor(self, path):
        mkdir_p(os.path.dirname(path))
        tensors = {
            "weights": self.weights,
            "biases": self.biases,
            "grid": self.grid,
        }
        torch.save(tensors, path)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        vis_f = self._vis_features.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        gammas = self._gamma.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, vis_f, opacities, gammas, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_tensor(self, path):
        tensors = torch.load(path)
        weights = tensors["weights"].to(device="cuda")
        biases = tensors["biases"].to(device="cuda")
        grid = tensors["grid"].to(device="cuda")
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)
        self.grid = nn.Parameter(grid)

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        gammas = np.stack(
            (
                np.asarray(plydata.elements[0]["gamma_0"]),
                np.asarray(plydata.elements[0]["gamma_1"]),
            ), axis=1
        )

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        vis_features = np.zeros((xyz.shape[0], 3))
        vis_features[:,0] = np.asarray(plydata.elements[0]["vis_f_0"])
        vis_features[:,1] = np.asarray(plydata.elements[0]["vis_f_1"])
        vis_features[:,2] = np.asarray(plydata.elements[0]["vis_f_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3, f"{len(extra_f_names)} {self.max_sh_degree}"
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._vis_features = nn.Parameter(torch.tensor(vis_features, dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._gamma = nn.Parameter(torch.tensor(gammas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_app_degree = self.max_sh_degree
        self.active_texture_app_degree = self.max_texture_app_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["hash_grid", "mlp"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        print(f"Pruned {mask.sum().item()} primitives")

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._gamma = optimizable_tensors["gamma"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._vis_features = self._vis_features[valid_points_mask]

        self.error_accum = self.error_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["hash_grid", "mlp"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                            new_opacities, new_gammas,
                            new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "gamma": new_gammas,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._gamma = optimizable_tensors["gamma"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        print(f"Added {new_xyz.shape[0]} new primitives")

        self._vis_features = torch.cat((self._vis_features, torch.rand_like(new_xyz)))
        self.error_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, mask):
        n_init_points = self.get_xyz.shape[0]
        if mask.sum() == 0:
            return
        N = 2
        num = N

        stds = self.get_scaling[mask].repeat(num, 1)
        gammas = self.get_gamma[mask].repeat(num, 1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.zeros_like(stds)

        regular_samples = torch.linspace(-1, 1, num, device="cuda")[:,None].repeat(1, mask.sum()).reshape(-1)
        # Split along long axis
        xmask = stds[:,0] > stds[:,1]
        scale_factors = 0.5 * torch.ones_like(gammas)
        regular_samples[xmask] = regular_samples[xmask] * (1 - scale_factors[xmask,0])
        regular_samples[~xmask] = regular_samples[~xmask] * (1 - scale_factors[~xmask,1])

        samples[xmask,0] = regular_samples[xmask] * stds[xmask,0]
        samples[~xmask,1] = regular_samples[~xmask] * stds[~xmask,1]
        new_stds = torch.clone(stds)
        new_stds[xmask,0] = stds[xmask,0] * scale_factors[xmask,0]
        new_stds[~xmask,1] = stds[~xmask,1] * scale_factors[~xmask,1]

        rots = build_rotation(self._rotation[mask]).repeat(num,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[mask].repeat(num, 1)
        new_scaling = self.scaling_inverse_activation(new_stds)
        new_rotation = self._rotation[mask].repeat(num,1)
        new_features_dc = self._features_dc[mask].repeat(num,1,1)
        new_features_rest = self._features_rest[mask].repeat(num,1,1)
        new_opacity = self._opacity[mask].repeat(num,1)
        new_gamma = self._gamma[mask].repeat(num, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_gamma, new_scaling, new_rotation
        )

        prune_filter = torch.cat((mask, torch.zeros(num * mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=False)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def add_new_gs(self, iteration, densify_proportion, grad_mode, min_opacity, densify_stochastic):
        current_num_points = self.num_points
        cap_max = self.get_cap_max(iteration)
        target_num = min(cap_max, int((1 + densify_proportion) * current_num_points))
        N=2
        num_gs = max(0, target_num - current_num_points) // (N-1)

        if num_gs <= 0:
            return 0

        if grad_mode == 0:
            probs = self.error_accum[:,1:2]
        elif grad_mode == 1:
            probs = self.error_accum[:,1:2] / torch.clamp(self.denom, min=1.0)
        elif grad_mode == 2:
            probs = self.get_opacity
        probs = probs.squeeze(-1)
        if densify_stochastic:
            add_idx, ratio = self.sample_alives(probs=probs, num=num_gs)
        else:
            sort_indices = torch.argsort(probs)
            add_idx = sort_indices < num_gs

        split_mask = torch.zeros(current_num_points, dtype=torch.bool, device="cuda"); 
        split_mask[add_idx] = True

        self.densify_and_split(split_mask)
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, error_tensor, update_filter):
        self.error_accum[update_filter] += error_tensor.grad[update_filter,:]
        self.denom[update_filter] += 1