import math
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from utils.general_utils import point_cloud_to_image
from utils.sh_utils import eval_sh, SH2RGB

from gaussian_model_4d import GaussianModel as GaussianModel4D
from gaussian_model import GaussianModel, BasicPointCloud

class Renderer:
    def __init__(self, T, sh_degree=3, white_background=True, radius=1):
        
        assert isinstance(T, int)
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.T = T

        self.gaussians = dict()
        self.gaussians["bg"] = GaussianModel(sh_degree)
        self.gaussian_rel_scales = dict()

        self.gaussian_translations = dict()
        self.gaussian_scales = dict()

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
            self.gaussians["bg"].create_from_pcd(pcd, 1)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians["bg"].create_from_pcd(input, 1)
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
            self.gaussians["bg"].restore(model_params)
        else:
            # load from saved ply
            self.gaussians["bg"].load_ply(input)
        
        self.gaussian_translations["bg"] = torch.zeros((self.T, 3), device=self.bg_color.device)
        self.gaussian_scales["bg"] = torch.ones((self.T, 1), device=self.bg_color.device)
        self.gaussian_rel_scales["bg"] = torch.tensor(1.0, device=self.bg_color.device)

    def add_object(self, input, translation, scale, name):

        from arguments import ModelHiddenParams
        hyper = ModelHiddenParams(None) # args
        # Overwrite K-plane time resolution based on video length
        if int(0.8*self.T) > 25:
            hyper.kplanes_config['resolution'] = hyper.kplanes_config['resolution'][:-1] +  [int(0.8*self.T)]
        self.gaussians[name] = GaussianModel4D(self.sh_degree, hyper)

        loaded_dict = input
        model_params = (loaded_dict['active_sh_degree'],
                        loaded_dict['xyz'],
                        loaded_dict['deformation_state_dict'],
                        loaded_dict['deformation_table'],
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
        
        self.gaussians[name].restore(model_params)
        self.gaussian_translations[name] = translation
        self.gaussian_scales[name] = scale
        self.gaussian_rel_scales[name] = torch.tensor(1.0, device=self.bg_color.device)
        
    def render(
        self,
        viewpoint_camera,
        default_camera_center=None,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        render_bg=False,
        specify_obj=None,
    ):
        if isinstance(specify_obj, str):
            specify_obj = [specify_obj]
        # Override render_bg flag if it is specified
        if specify_obj is not None and 'bg' in specify_obj:
            render_bg = True
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        all_gaussians_xyz = []
        for obj_name in self.gaussians.keys():
            if obj_name == 'bg' and not render_bg:
                continue
            all_gaussians_xyz.append(self.gaussians[obj_name].get_xyz)
        all_gaussians_xyz = torch.cat(all_gaussians_xyz)
        screenspace_points = (
            torch.zeros_like(
                all_gaussians_xyz,
                dtype=self.gaussians["bg"].get_xyz.dtype,
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
            sh_degree=self.gaussians["bg"].active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        all_means3D_final = []
        all_shs = []
        all_colors_precomp = []
        all_opacity = []
        all_scales = []
        all_rotations = []
        all_cov3D_precomp = []

        # Process individual object Gaussians
        for obj_name in self.gaussians.keys():

            if specify_obj is not None and obj_name not in specify_obj:
                continue

            means3D = self.gaussians[obj_name].get_xyz
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
            time = ((time.float() / self.T) - 0.5) * 2

            opacity = self.gaussians[obj_name]._opacity

            # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
            # scaling / rotation by the rasterizer.
            scales = None
            rotations = None
            cov3D_precomp = None
            if compute_cov3D_python:
                cov3D_precomp = self.gaussians[obj_name].get_covariance(scaling_modifier)
            else:
                scales = self.gaussians[obj_name]._scaling
                rotations = self.gaussians[obj_name]._rotation

            if obj_name == 'bg':
                # No deformations for the BG
                if render_bg:
                    means3D_final = means3D.clone()
                    rotations_final = rotations
                    scales_final = scales
                    opacity = opacity

                    scales_final = self.gaussians[obj_name].scaling_activation(scales_final)
                    rotations_final = self.gaussians[obj_name].rotation_activation(rotations_final)
                    opacity = self.gaussians[obj_name].opacity_activation(opacity)
                # If we don't render the BG, just skip everything afterwards
                else:
                    continue
            else:
                # First get the deformations
                (means3D_deform, scales_deform,
                 rotations_deform, opacity_deform) = self.gaussians[obj_name]._deformation(means3D, scales, 
                                                                                           rotations, opacity,
                                                                                           time)

                means3D_final = torch.zeros_like(means3D)
                rotations_final = torch.zeros_like(rotations)
                scales_final = torch.zeros_like(scales)
                #opacity_final = torch.zeros_like(opacity)
                means3D_final =  means3D_deform
                rotations_final =  rotations_deform
                scales_final =  scales_deform
                #opacity_final = opacity_deform

                scales_final = self.gaussians[obj_name].scaling_activation(scales_final)
                rotations_final = self.gaussians[obj_name].rotation_activation(rotations_final)
                opacity = self.gaussians[obj_name].opacity_activation(opacity)

            # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
            # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
            shs = None
            colors_precomp = None
            if colors_precomp is None:
                if convert_SHs_python:
                    shs_view = self.gaussians[obj_name].get_features.transpose(1, 2).view(
                        -1, 3, (self.gaussians[obj_name].max_sh_degree + 1) ** 2
                    )
                    dir_pp = self.gaussians[obj_name].get_xyz - viewpoint_camera.camera_center.repeat(
                        self.gaussians[obj_name].get_features.shape[0], 1
                    )
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(
                        self.gaussians[obj_name].active_sh_degree, shs_view, dir_pp_normalized
                    )
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    shs = self.gaussians[obj_name].get_features
            else:
                colors_precomp = override_color

            # Filter large Gaussians
            scales_mask = torch.max(scales_final, dim=1)[0] <= 0.1

            # First go from obj centric frame to world frame
            scales_final = scales_final * self.gaussian_scales[obj_name][viewpoint_camera.time]
            means3D_final = means3D_final * self.gaussian_scales[obj_name][viewpoint_camera.time]
            means3D_final = means3D_final + self.gaussian_translations[obj_name][viewpoint_camera.time]

            # Scale Gaussians
            # Hack to deal with z conventions
            cam_center = default_camera_center.clone()
            cam_center = -cam_center
            means3D_final = cam_center - (cam_center - means3D_final) * self.gaussian_rel_scales[obj_name]
            scales_final = scales_final * self.gaussian_rel_scales[obj_name]

            all_means3D_final.append(means3D_final[scales_mask])
            all_shs.append(shs[scales_mask])
            all_colors_precomp.append(colors_precomp)
            all_opacity.append(opacity[scales_mask])
            all_scales.append(scales_final[scales_mask])
            all_rotations.append(rotations_final[scales_mask])
            all_cov3D_precomp.append(cov3D_precomp)

        means3D_final = torch.cat(all_means3D_final)
        means2D = screenspace_points
        shs = torch.cat(all_shs)
        colors_precomp = None if all_colors_precomp[0] is None else torch.cat(colors_precomp)
        opacity = torch.cat(all_opacity)
        scales_final = None if all_scales[0] is None else torch.cat(all_scales)
        rotations_final = None if all_rotations[0] is None else torch.cat(all_rotations)
        cov3D_precomp = None if all_cov3D_precomp[0] is None else torch.cat(all_cov3D_precomp)

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
    
    def freeze_gaussians(self):
        for obj_name in self.gaussians.keys():
            self.gaussians[obj_name].freeze()
    
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

    def get_2d_gaussian_pos(self,
        viewpoint_camera,
    ):
        
        all_means3D = []
        all_means2D = []
        all_projected_depth = []
        
        # Process individual object Gaussians
        for obj_name in self.gaussians.keys():

            if obj_name == 'bg':
                continue

            means3D = self.gaussians[obj_name].get_xyz
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
            time = ((time.float() / self.T) - 0.5) * 2

            opacity = self.gaussians[obj_name]._opacity
            scales = self.gaussians[obj_name]._scaling
            rotations = self.gaussians[obj_name]._rotation

            means3D_deform, _, _, _ = self.gaussians[obj_name]._deformation(means3D, scales, 
                                                                rotations, opacity,
                                                                time)
            
            means3D_final = torch.zeros_like(means3D)
            means3D_final = means3D_deform

            # We only use global motion to warp the flow so detach here
            means3D_final = means3D_final * self.gaussian_scales[obj_name][viewpoint_camera.time].detach()
            means3D_final = means3D_final + self.gaussian_translations[obj_name][viewpoint_camera.time].detach()

            means2D_final, projected_depth = point_cloud_to_image(means3D_final, viewpoint_camera)

            all_means3D.append(means3D_final)
            all_means2D.append(means2D_final)
            all_projected_depth.append(projected_depth)
        
        means2D_final = torch.cat(all_means2D)
        projected_depth = torch.cat(all_projected_depth)
        means3D_final = torch.cat(all_means3D)
        return means2D_final, projected_depth, means3D_final