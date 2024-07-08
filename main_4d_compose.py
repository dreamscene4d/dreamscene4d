# Copyright (c) 2024 DreamScene4D and affiliated authors.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import scipy
import cv2
import glob
import json
import tqdm
import numpy as np

import argparse
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from torchmetrics import PearsonCorrCoef
from transformers import pipeline

from cameras import orbit_camera, OrbitCamera, MiniCam
from gs_renderer_compose import Renderer
from utils.traj_visualizer import Visualizer

from PIL import Image
from diffusers.utils import export_to_gif

from scipy.spatial.transform import Rotation as R
import pickle
import glob

@torch.no_grad()
def get_filtered_gaussians_mask(gaussian_2d_pos, gaussian_depth, depth, alpha):

    H, W = depth.shape
    gaussian_x = gaussian_2d_pos[:, 0].round().int()
    gaussian_y = gaussian_2d_pos[:, 1].round().int()

    # Check visibility of gaussians
    visible = (alpha > 0.75) # or maybe 0.5?
    # Remove regions close to the boundary
    visible = scipy.ndimage.binary_erosion(visible, structure=np.ones((7, 7))).astype(visible.dtype)
    visible_mask = visible[torch.clamp(gaussian_y, 0, H-1), torch.clamp(gaussian_x, 0, W-1)]
    # Check if gaussians are within bounds
    within_bounds = ((gaussian_x > 0) & (gaussian_x < W-1) & (gaussian_y > 0) & (gaussian_y < H-1)).numpy()
    # Only consider front facing gaussians
    init_rendered_depth = depth[torch.clamp(gaussian_y, 0, H-1), torch.clamp(gaussian_x, 0, W-1)]
    matched_depth_mask = np.abs(gaussian_depth - init_rendered_depth) < 0.25 # or maybe 0.01?

    filter_mask = (within_bounds * visible_mask * matched_depth_mask).astype(bool)

    return filter_mask

@torch.no_grad()
def match_gaussians_to_pixels(gaussian_2d_pos, gaussian_depth, gt_2d_pos, rendered_depth, rendered_alpha):

    # This function matches pixels to Gaussians

    gaussian_2d_pos = torch.from_numpy(gaussian_2d_pos)
    gt_2d_pos = torch.from_numpy(gt_2d_pos)

    init_gaussian_2d_pos = gaussian_2d_pos[0]
    init_gt_2d_pos = gt_2d_pos[0]
    init_gaussian_depth = gaussian_depth[0]
    init_depth = rendered_depth[0].squeeze(0)
    init_alpha = rendered_alpha[0].squeeze(0)

    # Get relevant Gaussians
    filter_mask = get_filtered_gaussians_mask(init_gaussian_2d_pos, init_gaussian_depth, init_depth, init_alpha)
    init_gaussian_2d_pos_filtered = init_gaussian_2d_pos[filter_mask]

    # Find matching between GT and Gaussians
    dist_to_gaussians = torch.norm(torch.unsqueeze(init_gt_2d_pos, 1)-torch.unsqueeze(init_gaussian_2d_pos_filtered, 0), dim=-1)
    min_dist_to_gaussians, min_idxes = torch.min(dist_to_gaussians, dim=1)
    dist_thresh = 0.25
    while dist_thresh < 4:
        dist_thresh = dist_thresh * 2
        matched_mask = min_dist_to_gaussians < dist_thresh

    matched_idxes = min_idxes[matched_mask]
    query_points = gt_2d_pos[0][matched_mask]
    
    return gaussian_2d_pos[:, filter_mask][:, matched_idxes], query_points

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.seed = 888

        self.vid_length = None

        # models
        self.device = torch.device("cuda")

        self.ref_obj_name = None

        self.obj_depth = None
        self.obj_z = None

        # training stuff
        self.training = False
        self.optimizer = None
        self.pearson = PearsonCorrCoef().to(self.device)
        self.step = 0

        # load input data from cmdline
        self.load_input()
        # default camera
        height, width = self.raw_depth.shape[:2]
        resize_factor = 720 / max(width, height) if max(width, height) > 720 else 1.0
        H_ = int(height * resize_factor)
        W_ = int(width * resize_factor)
        self.cam = OrbitCamera(W_, H_, r=self.opt.radius, fovy=self.opt.fovy)
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            W_,
            H_,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
            time=0,
        )
        # renderer
        self.renderer = Renderer(T=self.vid_length, sh_degree=self.opt.sh_degree)
        self.load_and_scale_gaussians_with_depth()

        self.seed_everything()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        print(f'Seed: {seed:d}')
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.autograd.profiler.profile(enabled=False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
        torch.autograd.set_detect_anomaly(mode=False)

        self.last_seed = seed

    def prepare_depth_input(self):

        self.step = 0

        # input image
        if self.input_depth is not None:

            height, width = self.input_depth.shape[:2]
            resize_factor = 720 / max(width, height) if max(width, height) > 720 else 1.0
            H_ = int(height * resize_factor)
            W_ = int(width * resize_factor)

            self.input_depth_torch = torch.from_numpy(self.input_depth).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_depth_torch = F.interpolate(self.input_depth_torch, (H_, W_), mode="nearest")

            self.input_depth_mask_torch = torch.from_numpy(self.input_depth_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_depth_mask_torch = F.interpolate(self.input_depth_mask_torch, (H_, W_), mode="nearest")

    def find_scale_for_obj(self, obj_name):
        
        render_height, render_width = self.input_depth_torch.shape[-2:]
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        cur_cam = MiniCam(
            pose,
            render_width,
            render_height,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
            time=0,
        )

        min_loss = torch.round(self.calc_depth_loss(cur_cam, obj_name), decimals=2)
        best_scale = self.renderer.gaussian_rel_scales[obj_name].clone().detach()

        self.renderer.gaussian_rel_scales[obj_name] = torch.tensor(0.65, device=self.device)
        for _ in tqdm.trange(self.opt.iters):

            self.step += 1

            ### known view
            depth_loss = torch.round(self.calc_depth_loss(cur_cam, obj_name), decimals=2)
            
            if depth_loss < min_loss:
                min_loss = depth_loss
                best_scale = self.renderer.gaussian_rel_scales[obj_name].clone()

            self.renderer.gaussian_rel_scales[obj_name] = self.renderer.gaussian_rel_scales[obj_name] + 0.00085
                
        print(f'{obj_name}: {best_scale}')
        self.renderer.gaussian_rel_scales[obj_name] = best_scale

    def calc_depth_loss(self, cam, obj_name):

        out = self.renderer.render(
            viewpoint_camera=cam,
            default_camera_center=self.fixed_cam.camera_center,
            specify_obj=[obj_name, self.ref_obj_name]
        )
        mask = (self.input_depth_mask_torch > 0.5)
        # Consider the union of overlapping areas if possible
        if (mask * out["alpha"] > 0.75).sum() > 0:
            mask = mask * out["alpha"] > 0.75

        # depth loss
        depth = out["depth"].unsqueeze(0) # [1, 1, H, W]
        depth = torch.nan_to_num(depth)
        scaled_depth = (depth - self.rendered_depth_shift) / self.rendered_depth_scale
        depth = scaled_depth * mask
        depth_loss = F.l1_loss(depth[mask], self.input_depth_torch[mask])

        return depth_loss

    @torch.no_grad()
    def get_masked_depth(self, obj_idx, ref_obj_idx):
        
        # BG
        if obj_idx == -1:
            bg_mask = (np.stack(self.input_mask).sum(0) < 0.5).astype(np.float32)
            input_mask_subset = [bg_mask, self.input_mask[ref_obj_idx]]
        # Objects
        else:
            input_mask_subset = [self.input_mask[obj_idx], self.input_mask[ref_obj_idx]]

        sum_mask = np.clip(np.stack(input_mask_subset).sum(0), 0., 1.)
        sum_mask = F.interpolate(
                torch.from_numpy(sum_mask).permute(2,0,1).unsqueeze(0),
                (self.raw_depth.shape[0], self.raw_depth.shape[1]),
                mode="nearest"
            ).squeeze().numpy()
        eroded_mask = scipy.ndimage.binary_erosion(sum_mask > 0.5, structure=np.ones((7, 7)))
        eroded_mask = (eroded_mask > 0.5) * self.raw_depth_mask

        scaled_depth = (self.raw_depth - self.depth_shift) / self.depth_scale
        masked_depth = scaled_depth * eroded_mask
        self.input_depth = masked_depth[:, :, np.newaxis].astype(np.float32)
        self.input_depth_mask = eroded_mask[:, :, np.newaxis].astype(np.float32)

    @torch.no_grad()
    def calc_instance_depth(self, depth):

        obj_depth = []
        obj_depth_range = []
        for mask in self.input_mask:
            # Erode and remove invalid parts of the mask
            obj_mask = F.interpolate(
                torch.from_numpy(mask).permute(2,0,1).unsqueeze(0),
                (self.raw_depth.shape[0], self.raw_depth.shape[1]),
                mode="nearest"
            ).squeeze().numpy()
            eroded_mask = scipy.ndimage.binary_erosion(obj_mask > 0.5, structure=np.ones((7, 7)))
            eroded_mask = (eroded_mask > 0.5) * self.raw_depth_mask
            # Resize to dimensions of the depth map
            eroded_mask = cv2.resize(
                eroded_mask.astype(int),
                (depth.shape[-1], depth.shape[-2]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            obj_depth.append(np.median(depth[eroded_mask]))
            obj_depth_range.append(np.quantile(depth[eroded_mask], 0.9)-np.quantile(depth[eroded_mask], 0.1))
        
        if self.opt.render_bg:
            bg_mask = (np.stack(self.input_mask).sum(0) < 0.5).astype(np.float32)
            bg_mask = F.interpolate(
                    torch.from_numpy(bg_mask).permute(2,0,1).unsqueeze(0),
                    (depth.shape[0], depth.shape[1]),
                    mode="nearest"
                ).squeeze().numpy()
            bg_mask = (bg_mask > 0.5) * self.raw_depth_mask
            bg_depth = torch.quantile(torch.from_numpy(depth[bg_mask > 0.5]), 0.5)
        else:
            # Dummy value
            bg_depth = torch.tensor(2.0)

        return obj_depth, bg_depth, obj_depth_range
    
    @torch.no_grad()
    def render_instance_depth(self):

        # Handle background
        if self.opt.render_bg:
            with torch.no_grad():
                out = self.renderer.render(
                    viewpoint_camera=self.fixed_cam,
                    default_camera_center=self.fixed_cam.camera_center,
                    specify_obj=f'bg'
                )
            rendered_depth = out["depth"].squeeze().cpu() # [H, W]
            rendered_depth = torch.nan_to_num(rendered_depth)
            # Erode and remove invalid parts of the mask
            obj_mask = out["alpha"].squeeze().cpu() > 0.75
            obj_mask = obj_mask.float().unsqueeze(-1).numpy()
            obj_mask = F.interpolate(
                torch.from_numpy(obj_mask).permute(2,0,1).unsqueeze(0),
                (rendered_depth.shape[0], rendered_depth.shape[1]),
                mode="nearest"
            ).squeeze().numpy()
            bg_depth = torch.quantile(rendered_depth[obj_mask > 0.5], 0.5)
        else:
            # Dummy value
            bg_depth = torch.tensor(2.0)
        
        # Handle objects
        obj_depth = []
        obj_depth_range = []
        for i in range(len(self.input_mask)):

            with torch.no_grad():
                out = self.renderer.render(
                    viewpoint_camera=self.fixed_cam,
                    default_camera_center=self.fixed_cam.camera_center,
                    specify_obj=f'obj_{i}'
                )
            rendered_depth = out["depth"].squeeze().cpu() # [H, W]
            rendered_depth = torch.nan_to_num(rendered_depth)

            # Erode and remove invalid parts of the mask
            obj_mask = out["alpha"].squeeze().cpu() > 0.75
            obj_mask = obj_mask.float().unsqueeze(-1).numpy()
            obj_mask = F.interpolate(
                torch.from_numpy(obj_mask).permute(2,0,1).unsqueeze(0),
                (self.raw_depth.shape[0], self.raw_depth.shape[1]),
                mode="nearest"
            ).squeeze().numpy()
            eroded_mask = scipy.ndimage.binary_erosion(obj_mask > 0.5, structure=np.ones((7, 7)))
            eroded_mask = (eroded_mask > 0.5) * self.raw_depth_mask
            # Resize to dimensions of the depth map
            eroded_mask = cv2.resize(
                eroded_mask.astype(int),
                (rendered_depth.shape[-1], rendered_depth.shape[-2]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            if eroded_mask.sum() > 0:
                obj_depth.append(torch.median(rendered_depth[eroded_mask]))
                obj_depth_range.append(torch.quantile(rendered_depth[eroded_mask], 0.9)-torch.quantile(rendered_depth[eroded_mask], 0.1))
            else:
                obj_depth.append(torch.tensor(1.0))
                obj_depth_range.append(torch.tensor(1.0))

        return obj_depth, bg_depth, obj_depth_range
    
    @torch.no_grad()
    def load_and_scale_gaussians_with_depth(self):

        # Load Gaussians
        if self.opt.render_bg:
            bg_gaussian_file = os.path.join(self.opt.outdir, "gaussians", f"{self.opt.save_path}_0.pkl")
        dyn_gaussian_files = glob.glob(os.path.join(self.opt.outdir, "gaussians") + f"/{self.opt.save_path}_[1-9]_4d.pkl")
        dyn_gaussian_files.sort()
        dyn_gaussian_motion_files = glob.glob(os.path.join(self.opt.outdir, "gaussians") + f"/{self.opt.save_path}_[1-9]_4d_global_motion.pkl")
        dyn_gaussian_motion_files.sort()

        self.ref_obj_name = f"obj_{len(dyn_gaussian_files)-1}"

        if self.opt.render_bg:
            # Load bg
            with open(bg_gaussian_file, 'rb') as f:
                loaded_dict = pickle.load(f)
            # Scale bg based on aspect ratio
            # BG is a square (due to SD), so scale the x dim
            loaded_dict['xyz'][:, 0] *= np.tan(self.cam.fovx/2) / np.tan(self.cam.fovy/2)
            loaded_dict['scaling'][:, 0] *= np.tan(self.cam.fovx/2) / np.tan(self.cam.fovy/2)
            self.renderer.initialize(loaded_dict)
        else:
            # Create a dummy bg
            self.renderer.initialize(num_pts=10)

        # Load and scale objects
        for i, (gaussian_file, gaussian_motion_file) in enumerate(zip(dyn_gaussian_files[::-1], dyn_gaussian_motion_files[::-1])):
            # First load global translation and scale
            with open(gaussian_motion_file, 'rb') as f:
                loaded_dict = pickle.load(f)
            gaussian_translation = loaded_dict['translation']
            gaussian_scale = loaded_dict['scale']
            # Load Gaussians
            with open(gaussian_file, 'rb') as f:
                loaded_dict = pickle.load(f)
            self.renderer.add_object(loaded_dict, gaussian_translation, gaussian_scale, f"obj_{len(dyn_gaussian_files)-i-1}")
        
        # Find object depth
        obj_depth, bg_depth, obj_depth_range = self.calc_instance_depth(self.raw_depth)
        rendered_obj_depth, rendered_bg_depth, rendered_obj_depth_range = self.render_instance_depth()
        # Find scaling factor for the Gaussians
        rel_scale = (bg_depth / obj_depth[-1]) * rendered_obj_depth[-1] / rendered_bg_depth
        #print(f"BG dist to cam: {rendered_bg_depth:.5f}, depth: {bg_depth:.5f}, relative scale: {rel_scale:.5f}.")
        self.renderer.gaussian_rel_scales["bg"] = rel_scale.clone().to(self.device)
        for i in range(len(obj_depth)):
            rel_scale = (obj_depth[i] / obj_depth[-1]) * rendered_obj_depth[-1] / rendered_obj_depth[i]
            #print(f"Obj dist to cam: {rendered_obj_depth[i]:.5f}, depth: {obj_depth[i]:.5f}, relative scale: {rel_scale:.5f}.")
            self.renderer.gaussian_rel_scales[f"obj_{i}"] = rel_scale.clone().to(self.device)
        # Get shift and scale for depth maps
        self.depth_shift = obj_depth[-1]
        self.depth_scale = obj_depth_range[-1]
        self.rendered_depth_shift = rendered_obj_depth[-1]
        self.rendered_depth_scale = rendered_obj_depth_range[-1]
    
    @torch.no_grad()
    def load_input(self):
        
        # First load masks
        self.input_mask = []
        for mask_file in self.opt.input_mask:
            mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            mask = mask.astype(np.float32) / 255.0
            if len(mask.shape) == 3:
                mask = mask[:, :, 0:1]
            else:
                mask = mask[:, :, np.newaxis]
            self.input_mask.append(mask)

        # Load image and compute depth
        file_list = glob.glob(self.opt.input+'*' if self.opt.input[-1] == '/' else self.opt.input+'/*')
        file_list.sort()
        file_list = file_list
        self.vid_length = len(file_list)
        pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
        with torch.no_grad():
            raw_pred = pipe(Image.open(file_list[0]))["depth"]
        H, W = self.input_mask[0].shape[:2]
        raw_pred = np.array(raw_pred.resize([W, H], Image.Resampling.NEAREST))
        disparity = scipy.ndimage.median_filter(raw_pred, size=(H//64, W//64))
        self.raw_depth = 1. / np.maximum(disparity, 1e-2)
        self.raw_depth_mask = (self.raw_depth != 1 / 1e-2)

        del pipe
        torch.cuda.empty_cache()

    @torch.no_grad()
    def load_cam_poses(self, file):
        with open(file, 'r') as f:
            cameras = json.load(f)
        cameras = cameras[:self.vid_length]

        self.rel_cam_pose = []
        for x in range(len(cameras)):
            curr_cam = cameras[x]
            prev_cam = cameras[0]
            
            rel_orientation = R.from_quat(curr_cam['orientation']) * R.from_quat(prev_cam['orientation']).inv()
            rel_orientation = rel_orientation.as_matrix()

            rel_center = (np.array(curr_cam['pos']) - np.array(prev_cam['pos']))
            self.rel_cam_pose.append({'pos': rel_center, 'orientation': rel_orientation})
    
    @torch.no_grad()
    def collect_gaussian_trajs(self):

        # Get Gaussian motion
        rendered_rgb = []
        rendered_alpha = []
        rendered_depth = []
        gaussian_2d_pos = []
        gaussian_depth = []
        for t in range(self.vid_length):
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
            cur_cam = MiniCam(
                pose,
                512,
                512,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                time=t
            )
            with torch.no_grad():
                outputs = self.renderer.render(
                    viewpoint_camera=cur_cam,
                    default_camera_center=self.fixed_cam.camera_center,
                )
                rendered_rgb.append(outputs["image"].cpu().numpy().astype(np.float32))
                rendered_depth.append(outputs["depth"].cpu().numpy())
                rendered_alpha.append(outputs["alpha"].cpu().numpy())
                pos_2d, proj_depth, _ = self.renderer.get_2d_gaussian_pos(cur_cam)

            gaussian_2d_pos.append(pos_2d.cpu().numpy())
            gaussian_depth.append(proj_depth.cpu().numpy())
        
        gaussian_2d_pos = np.stack(gaussian_2d_pos)
        gaussian_depth = np.stack(gaussian_depth)
        rendered_depth = np.stack(rendered_depth)
        rendered_alpha = np.stack(rendered_alpha)
        rendered_rgb = np.stack(rendered_rgb)

        return gaussian_2d_pos, gaussian_depth, rendered_depth, rendered_alpha, rendered_rgb

    @torch.no_grad()
    def render_visualization(self, file_name, orbit=None):

        assert orbit in [None, 'hor', 'elev']

        image_list = []
        nframes = self.vid_length * 5
        hor = 180
        delta_hor = 360 / nframes
        elevation = self.opt.elevation + 75 if orbit == 'elev' else self.opt.elevation
        delta_elevation = -300 / nframes
        time = 0
        delta_time = 1
        for i in tqdm.trange(nframes):
            pose = orbit_camera(elevation, hor-180, self.opt.radius)
            cur_cam = MiniCam(
                pose,
                512,
                512,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                time=time
            )
            with torch.no_grad():
                outputs = self.renderer.render(
                    viewpoint_camera=cur_cam,
                    default_camera_center=self.fixed_cam.camera_center,
                )

            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % self.vid_length
            if orbit == 'hor': hor = (hor+delta_hor) % 360
            if orbit == 'elev':
                if i > nframes // 2:
                    elevation = (elevation-delta_elevation) % 360
                else:
                    elevation = (elevation+delta_elevation) % 360

        export_to_gif(image_list, file_name)
    
    @torch.no_grad()
    def render_demo(self, file_name):

        image_list = []
        nframes = self.vid_length * 5
        hor = 180
        hor_nframes = nframes // 2
        elev_nframes = nframes - hor_nframes
        elevation = self.opt.elevation - 10
        time = 0
        delta_time = 1
        for i in tqdm.trange(nframes):
            pose = orbit_camera(elevation, hor-180, self.opt.radius)
            cur_cam = MiniCam(
                pose,
                512,
                512,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                time=time
            )
            with torch.no_grad():
                outputs = self.renderer.render(
                    viewpoint_camera=cur_cam,
                    default_camera_center=self.fixed_cam.camera_center,
                    render_bg=True
                )

            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % self.vid_length
            if i <= hor_nframes:
                if i < hor_nframes // 4:
                    delta_hor = -45 / (hor_nframes // 4)
                elif i >= hor_nframes // 4 and i < 3 * hor_nframes // 4:
                    delta_hor = 90 / (3 * (hor_nframes // 4) - (hor_nframes // 4))
                elif i >= 3 * hor_nframes // 4:
                    delta_hor = -45 / (hor_nframes - 3 * (hor_nframes // 4))
                hor = (hor+delta_hor) % 360
            else:
                if (i-hor_nframes) >= elev_nframes // 2:
                    delta_elevation = 60 / (elev_nframes - (elev_nframes // 2))
                else:
                    delta_elevation = -60 / (elev_nframes // 2)
                elevation = (elevation+delta_elevation) % 360

        export_to_gif(image_list, file_name)

    @torch.no_grad()
    def render_gaussian_trajs(self, file_name):
        
        # First collect 2d Gaussian trajs
        gaussian_2d_pos, gaussian_depth, rendered_depth, rendered_alpha, rendered_rgb = self.collect_gaussian_trajs()

        # Sample points on a grid in the mask
        merged_mask = (np.stack(self.input_mask).sum(0) > 0.5)
        #merged_mask = np.transpose(rendered_alpha[0] > 0.75, (1,2,0))
        merged_mask = F.interpolate(
            torch.from_numpy(merged_mask).permute(2,0,1).unsqueeze(0).float(),
            (512, 512),
            mode="nearest"
        ).squeeze().bool().numpy()
        H, W = merged_mask.shape[:2]
        center = [H / 2, W / 2]
        margin = W / 64
        range_y = (margin - H / 2 + center[0], H / 2 + center[0] - margin)
        range_x = (margin - W / 2 + center[1], W / 2 + center[1] - margin)
        grid_y, grid_x = np.meshgrid(
            np.linspace(*range_y, 48),
            np.linspace(*range_x, 48),
            indexing="ij",
        )
        grid_pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

        point_mask = merged_mask[
            (grid_pts[:, 1]).astype(int),
            (grid_pts[:, 0]).astype(int)
        ].astype(bool)
        grid_pts = grid_pts[point_mask][np.newaxis]

        # Find matching gaussians to these "query points"
        gaussian_2d_traj, query_points = match_gaussians_to_pixels(gaussian_2d_pos, gaussian_depth, grid_pts, rendered_depth, rendered_alpha)

        # Visualization
        vis = Visualizer(
            save_dir=self.opt.visdir,
            linewidth=2,
            mode='rainbow',
            tracks_leave_trace=3,
        )
        # Load images
        file_list = glob.glob(self.opt.input+'*' if self.opt.input[-1] == '/' else self.opt.input+'/*')
        file_list.sort()
        video = []
        for i, file in enumerate(file_list):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = img[:, :, :3]
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA) * 0.4
            video.append(img[:, :, ::-1])
        # Plot trajs
        vis.visualize(
            video=torch.from_numpy(np.stack(video)).permute(0,3,1,2).unsqueeze(0),
            #video=torch.from_numpy(rendered_rgb*255).unsqueeze(0),
            tracks=gaussian_2d_traj.unsqueeze(0),
            filename=file_name)
        np.save(os.path.join(self.opt.visdir, file_name+".npy"), query_points)

    # no gui mode
    def optimize(self, iters=500):

        if iters > 0:
            ref_obj_idx = int(self.ref_obj_name[4:])
            for obj_name in self.renderer.gaussians.keys():
                # If we don't render the BG, just skip it
                if obj_name == 'bg' and not opt.render_bg:
                    continue
                # If this is the reference obj, also skip it, scale=1.0
                elif obj_name == self.ref_obj_name:
                    continue
                
                # We consider the scale between one object and the reference object at once
                obj_idx = -1 if obj_name == 'bg' else int(obj_name[4:])
                self.get_masked_depth(obj_idx, ref_obj_idx)
                self.prepare_depth_input()
                self.find_scale_for_obj(obj_name)

        # Render eval
        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_composed_elev_orbit.gif'),
            orbit='elev'
        )

        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_composed_hor_orbit.gif'),
            orbit='hor'
        )

        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_composed_no_orbit.gif'),
            orbit=None
        )

        self.render_gaussian_trajs(file_name=f'{self.opt.save_path}_trajs')

        if opt.render_bg:
            self.render_demo(
                file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_demo.gif'),
            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    gui.optimize(opt.iters)
