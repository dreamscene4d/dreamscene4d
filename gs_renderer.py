import math
import numpy as np
import pickle

import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from utils.sh_utils import eval_sh, SH2RGB

from gaussian_model import GaussianModel, BasicPointCloud
from utils.general_utils import get_expon_lr_func

class Renderer:
    def __init__(self, sh_degree=3, white_background=True, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.gaussians = GaussianModel(sh_degree)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
    
    def initialize(self, input=None, num_pts=5000, radius=0.5, lr_scale=10.):
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
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, lr_scale)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, lr_scale)
        elif isinstance(input, dict):
            loaded_dict = input
            model_params = (loaded_dict['active_sh_degree'],
                            loaded_dict['xyz'],
                            loaded_dict['features_dc'],
                            loaded_dict['features_rest'],
                            loaded_dict['scaling'],
                            loaded_dict['rotation'],
                            loaded_dict['opacity'],
                            loaded_dict['max_radii2D'],
                            loaded_dict['xyz_gradient_accum'],
                            loaded_dict['denom'],
                            loaded_dict['optimizer_state_dict'],
                            loaded_dict['spatial_lr_scale'])
            self.gaussians.restore(model_params)
        else:
            # load from saved ply
            self.gaussians.load_ply(input)

    def initialize_global_motion(self, training_args, translation=[0., 0., 0.], scale=1.):

        self.gaussian_translation = torch.tensor(
            translation,
            dtype=torch.float32,
            device="cuda",
        )
        self.gaussian_scale = torch.tensor(
            [scale],
            dtype=torch.float32,
            device="cuda",
        )

        self.gaussian_translation.requires_grad_(True)

        l = [
            {'params': [self.gaussian_translation], 'lr': training_args.position_lr_init * self.gaussians.spatial_lr_scale, "name": "translation"},
        ]
        self.global_motion_optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15)
        self.scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.gaussians.spatial_lr_scale,
                                                lr_final=training_args.position_lr_final*self.gaussians.spatial_lr_scale,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
    
    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
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
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        if account_for_global_motion:
            scales = scales * self.gaussian_scale
            means3D = means3D * self.gaussian_scale
            means3D = means3D + self.gaussian_translation

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

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
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
    
    def freeze_gaussians(self):
        self.gaussians.freeze()
    
    def save_gaussians(self, path):

        gaussians_params = self.gaussians.capture()

        params_dict = {
            'active_sh_degree': gaussians_params[0],
            'xyz': gaussians_params[1],
            'features_dc': gaussians_params[2],
            'features_rest': gaussians_params[3],
            'scaling': gaussians_params[4],
            'rotation': gaussians_params[5],
            'opacity': gaussians_params[6],
            'max_radii2D': gaussians_params[7],
            'xyz_gradient_accum': gaussians_params[8],
            'denom': gaussians_params[9],
            'optimizer_state_dict': gaussians_params[10],
            'spatial_lr_scale': gaussians_params[11],
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

        for param_group in self.global_motion_optimizer.param_groups:
            if param_group["name"] == "translation":
                lr = self.scheduler_args(iteration)
                param_group['lr'] = lr
                return lr