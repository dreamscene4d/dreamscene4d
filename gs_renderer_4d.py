# Copyright (c) 2024 DreamScene4D and affiliated authors.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import pickle
import numpy as np

import torch
import scipy

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from utils.sh_utils import eval_sh, SH2RGB

from gaussian_model_4d import GaussianModel, BasicPointCloud
from utils.general_utils import get_expon_lr_func, build_rotation, quat_mult, point_cloud_to_image

class Renderer:
    def __init__(self, T, sh_degree=3, white_background=True, radius=1):

        assert isinstance(T, int)
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.T = T

        from arguments import ModelHiddenParams
        hyper = ModelHiddenParams(None) # args
        # Overwrite K-plane time resolution based on video length
        if int(0.8*T) > 25:
            # This is for reproducibility
            hyper.kplanes_config['resolution'] = hyper.kplanes_config['resolution'][:-1] +  [int(0.8*T)]
        self.gaussians = GaussianModel(sh_degree, hyper)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
    
    def initialize(self, input=None, num_pts=5000, radius=0.5):
        # load checkpoint
        if input is None:
            # init from random point cloud
            
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, 1)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, 1)
        else:
            # load from saved ply
            self.gaussians.load_ply(input)

        self.precompute_nn()

    def initialize_global_motion(self, training_args, translation, scale=[1.]):

        # One (x, y, z) per timestep
        self.gaussian_translation = torch.tensor(
            translation,
            dtype=torch.float32,
            device="cuda",
        )
        # One scale per timestep
        self.gaussian_scale = torch.tensor(
            scale,
            dtype=torch.float32,
            device="cuda",
        )
        # We only allow the translation to be changed
        self.gaussian_translation.requires_grad_(True)
        # Create optimizer and scheduler
        l = [
            {
                'params': [self.gaussian_translation],
                'lr': training_args.position_lr_init * self.gaussians.spatial_lr_scale,
                'weight_decay': 0.,
                "name": "translation"
            },
        ]
        self.global_motion_optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15)
        self.scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.gaussians.spatial_lr_scale,
                                                lr_final=training_args.position_lr_final*self.gaussians.spatial_lr_scale,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
        
    def reset_and_create_joint_optimizer(self, training_args, lr_scale):

        # Setup optimizer
        self.gaussians.spatial_lr_scale = lr_scale
        self.gaussians.training_setup(training_args)
        # Add translation params to joint optimizer
        self.gaussians.optimizer.add_param_group({
            'params': [self.gaussian_translation],
            'lr': training_args.position_lr_final * self.gaussians.spatial_lr_scale,
            'weight_decay': 0,
            "name": "translation"
        })
        # Turn on the gradients
        self.unfreeze_gaussians()
        self.gaussian_translation.requires_grad_(True)

    def prepare_render(self, timesteps):

        means3D = self.gaussians.get_xyz
        opacity = self.gaussians._opacity
        scales = self.gaussians._scaling
        rotations = self.gaussians._rotation

        # Aggregrate all inputs into one big input
        means3D_T = []
        opacity_T = []
        scales_T = []
        rotations_T = []
        time_T = []
        prev_time_T = []

        for t in timesteps:
            time = torch.tensor(t).to(means3D.device).repeat(means3D.shape[0], 1)
            time = ((time.float() / self.T) - 0.5) * 2

            prev_time = torch.tensor(t-1 if t > 0 else 0).to(means3D.device).repeat(means3D.shape[0], 1)
            prev_time = ((prev_time.float() / self.T) - 0.5) * 2

            means3D_T.append(means3D)
            opacity_T.append(opacity)
            scales_T.append(scales)
            rotations_T.append(rotations)
            time_T.append(time)
            prev_time_T.append(prev_time)

        means3D_T = torch.cat(means3D_T)
        opacity_T = torch.cat(opacity_T)
        scales_T = torch.cat(scales_T)
        rotations_T = torch.cat(rotations_T)
        time_T = torch.cat(time_T)
        prev_time_T = torch.cat(prev_time_T)

        # Get the deformations
        (means3D_deform_T,
         scales_deform_T,
         rotations_deform_T,
         opacity_deform_T) = self.gaussians._deformation(means3D_T.repeat(2, 1), scales_T.repeat(2, 1), 
                                                         rotations_T.repeat(2, 1), opacity_T.repeat(2, 1),
                                                         torch.cat([time_T, prev_time_T], dim=0))
        # Split the deformations
        means3D_deform_T, prev_means3D_deform_T = torch.chunk(means3D_deform_T, 2, dim=0)
        opacity_deform_T, prev_opacity_deform_T = torch.chunk(opacity_deform_T, 2, dim=0)
        scales_deform_T, prev_scales_deform_T = torch.chunk(scales_deform_T, 2, dim=0)
        rotations_deform_T, prev_rotations_deform_T = torch.chunk(rotations_deform_T, 2, dim=0)

        # Cache the deformations
        num_pts = means3D_deform_T.shape[0] // len(timesteps)
        self.time_deform_T = timesteps
        self.means3D_deform_T = means3D_deform_T.reshape([len(timesteps), num_pts, -1])
        self.opacity_deform_T = opacity_deform_T.reshape([len(timesteps), num_pts, -1])
        self.scales_deform_T = scales_deform_T.reshape([len(timesteps), num_pts, -1])
        self.rotations_deform_T = rotations_deform_T.reshape([len(timesteps), num_pts, -1])

        num_pts = prev_means3D_deform_T.shape[0] // len(timesteps)
        self.prev_time_deform_T = [(x-1 if x > 0 else 0) for x in timesteps]
        self.prev_means3D_deform_T = prev_means3D_deform_T.reshape([len(timesteps), num_pts, -1])
        self.prev_opacity_deform_T = prev_opacity_deform_T.reshape([len(timesteps), num_pts, -1])
        self.prev_scales_deform_T = prev_scales_deform_T.reshape([len(timesteps), num_pts, -1])
        self.prev_rotations_deform_T = prev_rotations_deform_T.reshape([len(timesteps), num_pts, -1])

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        direct_render=False,
        account_for_global_motion=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz

        means2D = screenspace_points
        opacity = self.gaussians._opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians._scaling
            rotations = self.gaussians._rotation

        # Get Gaussian deformations
        if direct_render:
            # Directly query network, used for inference
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
            time = ((time.float() / self.T) - 0.5) * 2
            (means3D_deform, scales_deform,
             rotations_deform, opacity_deform) = self.gaussians._deformation(means3D, scales, 
                                                                             rotations, opacity,
                                                                             time) #  time is not none
        else:
            # Cached mode, used for training
            idx = self.time_deform_T.index(viewpoint_camera.time)
            means3D_deform, scales_deform, rotations_deform, opacity_deform = self.means3D_deform_T[idx], self.scales_deform_T[idx], self.rotations_deform_T[idx], self.opacity_deform_T[idx]

        means3D_final = torch.zeros_like(means3D)
        rotations_final = torch.zeros_like(rotations)
        scales_final = torch.zeros_like(scales)
        #opacity_final = torch.zeros_like(opacity)
        means3D_final =  means3D_deform
        rotations_final =  rotations_deform
        scales_final =  scales_deform
        #opacity_final = opacity_deform

        scales_final = self.gaussians.scaling_activation(scales_final)
        rotations_final = self.gaussians.rotation_activation(rotations_final)
        opacity = self.gaussians.opacity_activation(opacity)

        # Warp Gaussians from obj centric to world frame
        if account_for_global_motion:
            scales_final = scales_final * self.gaussian_scale[viewpoint_camera.time]
            means3D_final = means3D_final * self.gaussian_scale[viewpoint_camera.time]
            means3D_final = means3D_final + self.gaussian_translation[viewpoint_camera.time]

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def render_flow(
        self,
        viewpoint_cameras,
        scaling_modifier=1.0,
        bg_color=None,
        compute_cov3D_python=False,
        account_for_global_motion=False,
    ):
        
        assert(len(viewpoint_cameras) == 2)

        gaussian_2d_pos_curr, _, gaussian_3d_pos_curr = self.get_2d_gaussian_pos(viewpoint_cameras[0], account_for_global_motion)
        gaussian_2d_pos_prev, _, gaussian_3d_pos_prev = self.get_2d_gaussian_pos(viewpoint_cameras[1], account_for_global_motion)
        flow_2d = gaussian_2d_pos_curr - gaussian_2d_pos_prev
        flow_padded = torch.cat([flow_2d, torch.zeros_like(flow_2d[:, 1:])], dim=1)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_cameras[1].FoVx * 0.5)
        tanfovy = math.tan(viewpoint_cameras[1].FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_cameras[1].image_height),
            image_width=int(viewpoint_cameras[1].image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_cameras[1].world_view_transform,
            projmatrix=viewpoint_cameras[1].full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_cameras[1].camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians._opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians._scaling
            rotations = self.gaussians._rotation

        # Get Gaussian deformations
        idx = self.prev_time_deform_T.index(viewpoint_cameras[1].time)
        means3D_deform, scales_deform, rotations_deform, opacity_deform = self.prev_means3D_deform_T[idx], self.prev_scales_deform_T[idx], self.prev_rotations_deform_T[idx], self.prev_opacity_deform_T[idx]

        means3D_final = torch.zeros_like(means3D)
        rotations_final = torch.zeros_like(rotations)
        scales_final = torch.zeros_like(scales)
        #opacity_final = torch.zeros_like(opacity)
        means3D_final =  means3D_deform
        rotations_final =  rotations_deform
        scales_final =  scales_deform
        #opacity_final = opacity_deform

        scales_final = self.gaussians.scaling_activation(scales_final)
        rotations_final = self.gaussians.rotation_activation(rotations_final)
        opacity = self.gaussians.opacity_activation(opacity)

        # Warp Gaussians from obj centric to world frame
        if account_for_global_motion:
            scales_final = scales_final * self.gaussian_scale[viewpoint_cameras[1].time]
            means3D_final = means3D_final * self.gaussian_scale[viewpoint_cameras[1].time]
            means3D_final = means3D_final + self.gaussian_translation[viewpoint_cameras[1].time]

        # Only update through flow, detach everything else
        rendered_flow_image, _, _, _ = rasterizer(
            means3D = means3D_final.detach(),
            means2D = means2D,
            shs = None,
            colors_precomp = flow_padded,
            opacities = opacity.detach(),
            scales = scales_final.detach(),
            rotations = rotations_final.detach(),
            cov3D_precomp = cov3D_precomp
        )

        scale_change = (scales_deform - scales)
        local_scale_loss = scale_change[self.nn_indices] - scale_change.unsqueeze(1)
        local_scale_loss = torch.sqrt((local_scale_loss ** 2).sum(-1) * self.nn_weights + 1e-20).mean()
        
        # Get Gaussian rotations for current timestep
        idx = self.time_deform_T.index(viewpoint_cameras[0].time)
        curr_rotations_deform = self.rotations_deform_T[idx]

        curr_rotations_final = torch.zeros_like(rotations)
        curr_rotations_final = curr_rotations_deform
        curr_rotations_final = self.gaussians.rotation_activation(curr_rotations_final)
        curr_rotations_final_inv = curr_rotations_final
        curr_rotations_final_inv[:, 1:] = -1. * curr_rotations_final_inv[:, 1:]
        
        # Local Rigidity Loss from Dynamic Gaussian Splatting
        prev_nn_displacement = gaussian_3d_pos_prev[self.nn_indices] - gaussian_3d_pos_prev.unsqueeze(1)
        curr_nn_displacement = gaussian_3d_pos_curr[self.nn_indices] - gaussian_3d_pos_curr.unsqueeze(1)
        rel_rotmat = build_rotation(quat_mult(rotations_final, curr_rotations_final_inv))
        curr_nn_displacement_warped = (rel_rotmat.transpose(2, 1)[:, None] @ curr_nn_displacement[:, :, :, None]).squeeze(-1)
        local_rigidity_loss = torch.sqrt(((prev_nn_displacement - curr_nn_displacement_warped) ** 2).sum(-1) * self.nn_weights + 1e-20).mean()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "flow": rendered_flow_image,
            "viewspace_points": screenspace_points,
            "scale_change": (scales_deform - scales),
            "local_scale_loss": local_scale_loss,
            "gaussian_displacements": (gaussian_3d_pos_curr - gaussian_3d_pos_prev),
            'local_rigidity_loss': local_rigidity_loss,
        }

    def freeze_gaussians(self):
        self.gaussians.freeze()

    def unfreeze_gaussians(self):
        self.gaussians.unfreeze()
    
    def save_gaussians(self, path):

        gaussians_params = self.gaussians.capture()

        params_dict = {
            'active_sh_degree': gaussians_params[0],
            'xyz': gaussians_params[1],
            'deformation_state_dict': gaussians_params[2],
            'deformation_table': gaussians_params[3],
            'features_dc': gaussians_params[4],
            'features_rest': gaussians_params[5],
            'scaling': gaussians_params[6],
            'rotation': gaussians_params[7],
            'opacity': gaussians_params[8],
            'max_radii2D': gaussians_params[9],
            'xyz_gradient_accum': gaussians_params[10],
            'denom': gaussians_params[11],
            'optimizer_state_dict': gaussians_params[12],
            'spatial_lr_scale': gaussians_params[13],
        }

        with open(path, 'wb') as f:
            pickle.dump(params_dict, f)

    def save_global_motion(self, path):

        params_dict = {
            'translation': self.gaussian_translation.detach(),
            'scale': self.gaussian_scale.detach(),
        }

        with open(path, 'wb') as f:
            pickle.dump(params_dict, f)

    def update_learning_rate(self, iteration):

        # Params should be here when optimizing for translations only
        for param_group in self.global_motion_optimizer.param_groups:
            if param_group["name"] == "translation":
                lr = self.scheduler_args(iteration)
                param_group['lr'] = lr * self.gaussians.spatial_lr_scale
                return lr
        # Params should be here when jointly optimizing
        for param_group in self.gaussians.param_groups:
            if param_group["name"] == "translation":
                lr = self.scheduler_args(iteration)
                param_group['lr'] = lr * self.gaussians.spatial_lr_scale
                return lr
            
    def precompute_nn(self, k=16, dist_thresh=0.05):

        means3D = self.gaussians.get_xyz
        opacity_mask = (self.gaussians.get_opacity > 0.).squeeze(1)
        vis_means3D = means3D[opacity_mask].detach().cpu().numpy()
        means_kdtree = scipy.spatial.cKDTree(vis_means3D, leafsize=100)
        nearest_neighbors = means_kdtree.query(vis_means3D, k=k)
        self.nn_mask = (nearest_neighbors[0] < dist_thresh)
        self.nn_indices = nearest_neighbors[1]
        self.nn_weights = torch.from_numpy(np.exp(-10. * nearest_neighbors[0])).to(means3D.device)
            
    def get_2d_gaussian_pos(self,
        viewpoint_camera,
        account_for_global_motion=False
    ):

        means3D = self.gaussians.get_xyz

        if viewpoint_camera.time in self.time_deform_T:
            idx = self.time_deform_T.index(viewpoint_camera.time)
            means3D_deform = self.means3D_deform_T[idx]
        else:
            idx = self.prev_time_deform_T.index(viewpoint_camera.time)
            means3D_deform = self.prev_means3D_deform_T[idx]
        
        means3D_final = torch.zeros_like(means3D)
        means3D_final = means3D_deform

        # We only use global motion to warp the flow so detach here
        if account_for_global_motion:
            means3D_final = means3D_final * self.gaussian_scale[viewpoint_camera.time].detach()
            means3D_final = means3D_final + self.gaussian_translation[viewpoint_camera.time].detach()

        means2D_final, projected_depth = point_cloud_to_image(means3D_final, viewpoint_camera)

        return means2D_final, projected_depth, means3D_final