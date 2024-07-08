# Copyright (c) 2024 DreamScene4D and affiliated authors.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import glob
import tqdm
import json
import scipy
import pickle
import numpy as np

import argparse
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from torchmetrics import PearsonCorrCoef
from transformers import pipeline

from cameras import orbit_camera, OrbitCamera, MiniCam
from utils.general_utils import safe_normalize
from gs_renderer_4d import Renderer
from utils.flow_utils import run_flow_on_images
from gmflow.gmflow.gmflow import GMFlow

from PIL import Image
from torchvision.transforms.functional import center_crop
from diffusers.utils import export_to_gif
from scipy.spatial.transform import Rotation as R

from grid_put import mipmap_linear_grid_put_2d

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.seed = 888

        # models
        self.device = torch.device("cuda")

        self.guidance_sd = None
        self.guidance_zero123 = None
        self.guidance_svd = None

        self.enable_sd = False
        self.enable_zero123 = False
        self.enable_svd = False

        # input image
        self.input_img = None
        self.input_depth = None
        self.input_depth_mask = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_depth_torch = None
        self.input_depth_mask_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        self.input_img_list = None
        self.input_depth_list = None
        self.input_depth_mask_list = None
        self.input_mask_list = None
        self.input_img_torch_list = None
        self.input_depth_torch_list = None
        self.input_depth_mask_torch_list = None
        self.input_mask_torch_list = None

        self.vid_length = None

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.pearson = PearsonCorrCoef().to(self.device, non_blocking=True)
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data from cmdline
        if self.opt.input is not None: # True
            self.load_input(self.opt.input) # load imgs, if has bg, then rm bg; or just load imgs
            self.get_depth(self.opt.input)
        
        # renderer
        self.renderer = Renderer(T=self.vid_length, sh_degree=self.opt.sh_degree)

        if self.opt.cam_pose is not None:
            self.load_cam_poses(self.opt.cam_pose)
        else:
            self.rel_cam_pose = []
            for _ in range(self.vid_length):
                self.rel_cam_pose.append({'pos': np.array([0., 0., 0.]), 'orientation': np.eye(3), 'scale': 1.})
        
        # override prompt from cmdline
        if self.opt.prompt is not None: # None
            self.prompt = self.opt.prompt

        # override if provide a checkpoint
        if self.opt.load is not None: # not None
            self.renderer.initialize(self.opt.load)  
            # self.renderer.gaussians.load_model(opt.outdir, opt.save_path)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

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

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)

        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != "" # False
        self.enable_zero123 = self.opt.lambda_zero123 > 0 # True
        self.enable_svd = self.opt.lambda_svd > 0 and self.input_img is not None # False

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd: # False
            if self.opt.mvdream: # False
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123: # True
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                print(f"[INFO] loading stable zero123...")
                self.guidance_zero123 = Zero123(self.device, t_range=[0.02, self.opt.t_max], model_key='ashawkey/stable-zero123-diffusers')
            else:
                print(f"[INFO] loading zero123...")
                self.guidance_zero123 = Zero123(self.device, t_range=[0.02, self.opt.t_max], model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")


        if self.guidance_svd is None and self.enable_svd: # False
            print(f"[INFO] loading SVD...")
            from guidance.svd_utils import StableVideoDiffusion
            self.guidance_svd = StableVideoDiffusion(self.device)
            print(f"[INFO] loaded SVD!")
        
        # Load frame 0 scales
        with open(os.path.join(self.opt.outdir, "gaussians", self.opt.save_path+"_global_motion.pkl"), 'rb') as f:
            input_scale0 = pickle.load(f)['scale'].squeeze()

        # Parse images and masks
        if self.input_img_list is not None:

            height, width = self.input_img_list[0].shape[:2]
            resize_factor = 720 / max(width, height) if max(width, height) > 720 else 1.0
            H_ = int(height * resize_factor)
            W_ = int(width * resize_factor)
            
            self.input_img_torch_list = []
            self.input_mask_torch_list = []
            self.input_depth_torch_list = []
            self.input_depth_mask_torch_list = []
            self.input_img_torch_orig_list = []
            self.input_mask_torch_orig_list = []
            self.input_depth_torch_orig_list = []
            self.input_depth_mask_torch_orig_list = []
            self.obj_cx_list = []
            self.obj_cy_list = []
            self.obj_scale_list = []
            for input_img, input_mask, input_depth, input_depth_mask in zip(self.input_img_list, self.input_mask_list, self.input_depth_list, self.input_depth_mask_list):
                # Reshape
                input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
                input_mask_torch = torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
                input_depth_torch = torch.from_numpy(input_depth).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
                input_depth_mask_torch = torch.from_numpy(input_depth_mask).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)

                # Resize world frame inputs to 720p
                self.input_img_torch_orig_list.append(F.interpolate(input_img_torch, (H_, W_), mode="bilinear", align_corners=False))
                self.input_mask_torch_orig_list.append(F.interpolate(input_mask_torch, (H_, W_), mode="nearest"))
                self.input_depth_torch_orig_list.append(F.interpolate(input_depth_torch, (H_, W_), mode="nearest"))
                self.input_depth_mask_torch_orig_list.append(F.interpolate(input_depth_mask_torch, (H_, W_), mode="nearest"))

                N, C, H, W = input_mask_torch.shape

                mask = input_mask_torch > 0.5
                nonzero_idxes = torch.nonzero(mask[0,0])
                if len(nonzero_idxes) > 0:
                    # Find bbox
                    min_x = nonzero_idxes[:, 1].min()
                    max_x = nonzero_idxes[:, 1].max()
                    min_y = nonzero_idxes[:, 0].min()
                    max_y = nonzero_idxes[:, 0].max()
                    # Find cx cy
                    cx = (max_x + min_x) / 2
                    cx = ((cx / W) * 2 - 1)
                    cy = (max_y + min_y) / 2
                    cy = ((cy / H) * 2 - 1)
                    self.obj_cx_list.append(cx)
                    self.obj_cy_list.append(cy)
                    # Find maximum possible scale
                    width = (max_x - min_x) / W
                    height = (max_y - min_y) / H
                    scale_x = width / 0.975
                    scale_y = height / 0.975
                    max_scale = max(scale_x, scale_y)
                    # If the scale from the first frame doesn't clip the object, then stick with it, otherwise use the max possible scale
                    scale = max(max_scale, input_scale0)
                    self.obj_scale_list.append(scale)
                    # Construct affine warp and grid
                    theta = torch.tensor([[[scale, 0., cx], [0., scale, cy]]], device=self.device)
                    resize_factor = self.opt.ref_size / min(H, W)
                    grid = F.affine_grid(theta, (N, C, int(H*resize_factor), int(W*resize_factor)), align_corners=True)
                    # Change border of image to white because we assume white background
                    input_img_torch[:, :, 0] = 1.
                    input_img_torch[:, :, -1] = 1.
                    input_img_torch[:, :, :, 0] = 1.
                    input_img_torch[:, :, :, -1] = 1.
                    # Aspect preserving grid sample, this recenters and scales the object
                    input_img_torch = F.grid_sample(input_img_torch, grid, align_corners=True, padding_mode='border')
                    input_mask_torch = F.grid_sample(input_mask_torch, grid, align_corners=True)
                    input_depth_torch = F.grid_sample(input_depth_torch, grid, mode='nearest', align_corners=True)
                    input_depth_mask_torch = F.grid_sample(input_depth_mask_torch, grid, mode='nearest', align_corners=True)
                    # Center crop
                    input_img_torch = center_crop(input_img_torch, self.opt.ref_size)
                    input_mask_torch = center_crop(input_mask_torch, self.opt.ref_size)
                    input_depth_torch = center_crop(input_depth_torch, self.opt.ref_size)
                    input_depth_mask_torch = center_crop(input_depth_mask_torch, self.opt.ref_size)
                else:
                    # Empty masks, use dummy values, will be ignored in rendering losses
                    input_img_torch = torch.zeros((1, 3, self.opt.ref_size, self.opt.ref_size), device=self.device)
                    input_mask_torch = torch.zeros((1, 1, self.opt.ref_size, self.opt.ref_size), device=self.device)
                    input_depth_torch = torch.zeros((1, 1, self.opt.ref_size, self.opt.ref_size), device=self.device)
                    input_depth_mask_torch = torch.zeros((1, 1, self.opt.ref_size, self.opt.ref_size), device=self.device)

                self.input_img_torch_list.append(input_img_torch)
                self.input_mask_torch_list.append(input_mask_torch)
                self.input_depth_torch_list.append(input_depth_torch)
                self.input_depth_mask_torch_list.append(input_depth_mask_torch)
        
        # prepare flow
        self.input_flow_torch_list = []
        self.input_flow_valid_torch_list = []
        self.input_flow_torch_orig_list = []
        self.input_flow_valid_torch_orig_list = []
        with torch.no_grad():
            # Load GMFlow
            flow_predictor = GMFlow(
                feature_channels=128,
                num_scales=1,
                upsample_factor=8,
                num_head=1,
                attention_type='swin',
                ffn_dim_expansion=4,
                num_transformer_layers=6,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
            )
            flow_predictor.eval()
            checkpoint = torch.load(self.opt.gmflow_path)
            weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
            flow_predictor.load_state_dict(weights)
            flow_predictor.to(self.device, non_blocking=True)
            # Run GMFlow
            fwd_flows, _, fwd_valids, _ = run_flow_on_images(flow_predictor, torch.cat(self.input_img_torch_list))
            fwd_flows_orig, _, fwd_valids_orig, _ = run_flow_on_images(flow_predictor, torch.cat(self.input_img_torch_orig_list))
            # Since there's no frame -1, for frame 0, we set a flow map of all zeros
            self.input_flow_torch_list.append(torch.zeros_like(fwd_flows[0].unsqueeze(0)))
            self.input_flow_valid_torch_list.append(torch.ones_like(fwd_flows[0].unsqueeze(0)))
            self.input_flow_torch_orig_list.append(torch.zeros_like(fwd_flows_orig[0].unsqueeze(0)))
            self.input_flow_valid_torch_orig_list.append(torch.ones_like(fwd_flows_orig[0].unsqueeze(0)))
            # Mask out (set to zero) irrelevant parts of the flow using previous frame masks
            # For valid masks, 1 if it's within the object and passed consistency check
            for i, (flow, flow_orig, valid, valid_orig) in enumerate(zip(fwd_flows, fwd_flows_orig, fwd_valids, fwd_valids_orig)):
                binary_mask = (self.input_mask_torch_list[i] > 0.5)
                binary_mask_orig = (self.input_mask_torch_orig_list[i] > 0.5)
                self.input_flow_torch_list.append((flow.unsqueeze(0) * binary_mask).clone())
                self.input_flow_valid_torch_list.append((valid.unsqueeze(0) * binary_mask).clone())
                self.input_flow_torch_orig_list.append((flow_orig.unsqueeze(0) * binary_mask_orig).clone())
                self.input_flow_valid_torch_orig_list.append((valid_orig.unsqueeze(0) * binary_mask_orig).clone())

            del flow_predictor
            torch.cuda.empty_cache()

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                c_list, v_list = [], []
                for _ in range(self.opt.n_views):
                    for input_img_torch in self.input_img_torch_list:
                        c, v = self.guidance_zero123.get_img_embeds(input_img_torch)
                        c_list.append(c)
                        v_list.append(v)
                self.guidance_zero123.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]
            
            if self.enable_svd:
                self.guidance_svd.get_img_embeds(self.input_img)

    def get_cam_at_timestep(self, timestep, render_width=None, render_height=None, cam_param_obj=None):

        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        # If we don't specify anything, use default cam
        if self.opt.cam_pose is None and render_width is None and render_height is None and cam_param_obj is None:
            cam = copy.deepcopy(self.fixed_cam)
        # Otherwise we need to construct a custom cam
        else:
            # Replace camera pose if specified
            if self.opt.cam_pose is not None:
                pose[:3, :3] = self.rel_cam_pose[timestep]['orientation'] @ pose[:3, :3]
                pose[:3, 3] =  self.rel_cam_pose[timestep]['scale'] * self.rel_cam_pose[timestep]['pos'] + pose[:3, 3]
            # Replace render params if specified
            if render_width is None:
                render_width = self.opt.ref_size
            if render_height is None:
                render_height = self.opt.ref_size
            if cam_param_obj is None:
                cam_param_obj = self.cam
            # Construct cam object using custom params
            cam = MiniCam(
                pose,
                render_width,
                render_height,
                cam_param_obj.fovy,
                cam_param_obj.fovx,
                cam_param_obj.near,
                cam_param_obj.far,
            )
        # Finally, set the timestep for the cam
        cam.time = timestep

        return cam, pose
    
    def train_step(self, finetune_all=False):

        # Use custom cam params for rendering world frame RGB
        render_height, render_width = self.input_img_torch_orig_list[0].shape[-2:]
        render_cam_world = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters) # 1, step / 500

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)
            if finetune_all:
                self.renderer.update_learning_rate(self.step)

            loss = 0

            rand_timesteps = np.random.choice(np.arange(len(self.input_img_torch_list)), self.opt.batch_size, replace=False).tolist()
            self.renderer.prepare_render(rand_timesteps)
        
            ### known view
            for b_idx in rand_timesteps:
                
                if finetune_all:
                    cur_cam, pose = self.get_cam_at_timestep(b_idx, render_width, render_height, render_cam_world)
                else:
                    cur_cam, pose = self.get_cam_at_timestep(b_idx)
                out = self.renderer.render(
                    cur_cam,
                    account_for_global_motion=(True if finetune_all else False)
                )

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                target_img = self.input_img_torch_orig_list[b_idx] if finetune_all else self.input_img_torch_list[b_idx]
                target_mask = (self.input_mask_torch_orig_list[b_idx] > 0.5).float() if finetune_all else (self.input_mask_torch_list[b_idx] > 0.5).float()
                if target_mask.sum() > 0:
                    loss = loss + 10000 * step_ratio * self.balanced_mask_loss(image, target_img, target_mask) / self.opt.batch_size

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                target_mask = (self.input_mask_torch_orig_list[b_idx] > 0.5).float() if finetune_all else (self.input_mask_torch_list[b_idx] > 0.5).float()
                if target_mask.sum() > 0:
                    loss = loss + 1000 * step_ratio * self.balanced_mask_loss(mask, target_mask, target_mask) / self.opt.batch_size

                # depth loss
                if self.opt.depth_loss:
                    depth = out["depth"].unsqueeze(0) # [1, 1, H, W]
                    depth = torch.nan_to_num(depth)
                    target_depth = self.input_depth_torch_orig_list[b_idx] if finetune_all else self.input_depth_torch_list[b_idx]
                    target_depth_mask = (self.input_depth_mask_torch_orig_list[b_idx] > 0.5) if finetune_all else (self.input_depth_mask_torch_list[b_idx] > 0.5)
                    if target_depth_mask.sum() > 0:
                        median_depth = torch.median(depth[target_depth_mask])
                        scaled_depth = (depth - median_depth) / torch.abs(depth[target_depth_mask] - median_depth + 1e-6).mean()
                        depth = scaled_depth * target_depth_mask
                        depth_loss = (1. - self.pearson(depth[target_depth_mask], target_depth[target_depth_mask]))
                        loss = loss + 100 * step_ratio * depth_loss / self.opt.batch_size

                # Render flow in the context of prev frame
                if finetune_all:
                    prev_cam, _ = self.get_cam_at_timestep(b_idx - 1 if b_idx > 0 else 0, render_width, render_height, render_cam_world)
                else:
                    prev_cam, _ = self.get_cam_at_timestep(b_idx - 1 if b_idx > 0 else 0)
                out = self.renderer.render_flow(
                    [cur_cam, prev_cam],
                    bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
                    account_for_global_motion=(True if finetune_all else False)
                )
                
                # flow loss
                _, H, W = out["flow"].shape
                flow = out["flow"].permute(1,2,0).reshape((-1, 3)) # [3, H, W] -> [H*W, 3]
                flow_2d = flow[:, 0:2]
                flow_2d = flow_2d.reshape((H, W, 2)).permute(2,0,1)
                flow_2d = flow_2d.unsqueeze(0) # [1, 2, H, W]
                target_flow = self.input_flow_torch_orig_list[b_idx] if finetune_all else self.input_flow_torch_list[b_idx]
                obj_flow_valid_mask = self.input_flow_valid_torch_orig_list[b_idx] if finetune_all else self.input_flow_valid_torch_list[b_idx]
                target_mask = (self.input_mask_torch_orig_list[b_idx-1 if b_idx > 0 else 0] > 0.5).float() if finetune_all else (self.input_mask_torch_list[b_idx-1 if b_idx > 0 else 0] > 0.5).float()
                flow_loss = (flow_2d - target_flow).abs()
                # We will normalize the flow loss w.r.t. to image dimensions
                flow_loss[:, 0] /= W
                flow_loss[:, 1] /= H
                # Note that this is different from the normal balanced mask loss
                if target_mask.sum() > 0:
                    if (target_mask * obj_flow_valid_mask).sum() > 0:
                        masked_flow_loss = (flow_loss * obj_flow_valid_mask * target_mask).sum() / (target_mask * obj_flow_valid_mask).sum()
                    else:
                        masked_flow_loss = 0.
                    if (1 - target_mask).sum() > 0:
                        masked_flow_loss_empty = (flow_loss * (1 - target_mask)).sum() / (1 - target_mask).sum()
                    else:
                        masked_flow_loss_empty = 0.
                    loss = loss + 10000. * step_ratio * (masked_flow_loss + masked_flow_loss_empty) / self.opt.batch_size

                # reg losses
                reg_loss = out["scale_change"].abs().mean()
                loss = loss + 1000. * step_ratio * reg_loss / self.opt.batch_size

                local_scale_loss = out["local_scale_loss"]
                loss = loss + 50000. * step_ratio * local_scale_loss / self.opt.batch_size

                local_rigidity_loss = out["local_rigidity_loss"]
                loss = loss + 50000. * step_ratio * local_rigidity_loss / self.opt.batch_size

            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.n_views):
                for b_idx in rand_timesteps:

                    # render random view
                    ver = np.random.randint(min_ver, max_ver)
                    hor = np.random.randint(-180, 180)
                    radius = 0

                    vers.append(ver)
                    hors.append(hor)
                    radii.append(radius)

                    pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                    if self.opt.cam_pose is not None:
                        pose[:3, :3] = self.rel_cam_pose[b_idx]['orientation'] @ pose[:3, :3]
                        pose[:3, 3] =  self.rel_cam_pose[b_idx]['scale'] * self.rel_cam_pose[b_idx]['pos'] + pose[:3, 3]
                    poses.append(pose)

                    cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far, time=b_idx)

                    bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                    out = self.renderer.render(cur_cam, bg_color=bg_color)

                    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)

                    # enable mvdream training
                    if self.opt.mvdream: # False
                        for view_i in range(1, 4):
                            pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                            poses.append(pose_i)

                            cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                            # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                            out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                            image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                            images.append(image)

            images = torch.cat(images, dim=0)

            # guidance loss
            if self.enable_sd:
                if self.opt.mvdream:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio)
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)

            if self.enable_zero123:
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio, timesteps=rand_timesteps) / (self.opt.batch_size * self.opt.n_views)

            if self.enable_svd:
                loss = loss + self.opt.lambda_svd * self.guidance_svd.train_step(images, step_ratio)
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def optimize_global_motion(self):
        
        # Use custom cam params for rendering world frame RGB
        render_height, render_width = self.input_img_torch_orig_list[0].shape[-2:]
        render_cam = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.update_learning_rate(self.step)

            loss = 0

            rand_timesteps = np.random.choice(np.arange(len(self.input_img_torch_list)), self.opt.batch_size, replace=False).tolist()
            self.renderer.prepare_render(rand_timesteps)

            for timestep in rand_timesteps:

                ### known view
                cur_cam, _ = self.get_cam_at_timestep(timestep, render_width, render_height, render_cam)
                out = self.renderer.render(cur_cam, account_for_global_motion=True)

                target_img = self.input_img_torch_orig_list[timestep]
                target_mask = (self.input_mask_torch_orig_list[timestep] > 0.5).float()

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                if target_mask.sum() > 0:
                    loss = loss + 1000 * step_ratio * self.balanced_mask_loss(image, target_img, target_mask)
                
                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                if target_mask.sum() > 0:
                    loss = loss + 100 * step_ratio * self.balanced_mask_loss(mask, target_mask, target_mask)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def balanced_mask_loss(self, pred, target, mask):
        # Compute loss over mask and non-mask regions separately
        if mask.sum() > 0:
            masked_loss = (F.mse_loss(pred, target, reduction='none') * mask).sum() / mask.sum()
        else:
            masked_loss = 0.
        if (1 - mask).sum() > 0:
            masked_loss_empty = (F.mse_loss(pred, target, reduction='none') * (1 - mask)).sum() / (1 - mask).sum()
        else:
            masked_loss_empty = 0.

        return masked_loss + masked_loss_empty

    def load_input(self, file):
        print("Loading input images and masks")
        assert(self.opt.input_mask is not None)
        # Get file lists
        file_list = glob.glob(self.opt.input+'*' if self.opt.input[-1] == '/' else self.opt.input+'/*')
        file_list.sort()
        file_list = file_list
        self.vid_length = len(file_list)
        mask_file_list = glob.glob(self.opt.input_mask+'*' if self.opt.input_mask[-1] == '/' else self.opt.input_mask+'/*')
        mask_file_list.sort()
        mask_file_list = mask_file_list
        self.input_img_list, self.input_mask_list = [], []
        # Load files
        for i, file in enumerate(tqdm.tqdm(file_list)):
            img = Image.open(file)
            if self.opt.resize_square:
                width, height = img.size
                new_dim = min(width, height)
                img = img.resize([new_dim, new_dim], Image.Resampling.BICUBIC)
            img = np.array(img)[:, :, :3]
            mask = Image.open(mask_file_list[i])
            if self.opt.resize_square:
                mask = mask.resize([new_dim, new_dim], Image.Resampling.NEAREST)
            mask = np.array(mask)
            mask = mask.astype(np.float32) / 255.0
            if len(mask.shape) == 3:
                mask = mask[:, :, 0:1]
            else:
                mask = mask[:, :, np.newaxis]

            img = img.astype(np.float32) / 255.0
            input_mask = mask
            # white bg
            input_img = img[..., :3] * input_mask + (1 - input_mask)
            self.input_img_list.append(input_img)
            self.input_mask_list.append(input_mask)

    @torch.no_grad()
    def get_depth(self, file):

        file_list = glob.glob(self.opt.input+'*' if self.opt.input[-1] == '/' else self.opt.input+'/*')
        file_list.sort()
        file_list = file_list
        
        self.input_depth_list, self.input_depth_mask_list = [], []
        if self.opt.depth_loss:
            pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf")
            print("Extracting depth")
            for i, file in enumerate(tqdm.tqdm(file_list)):
                # The "depth" returned is actually the disparity
                with torch.no_grad():
                    raw_pred = pipe(Image.open(file))["depth"]
                H, W = self.input_img_list[0].shape[:2]
                raw_pred = np.array(raw_pred.resize([W, H], Image.Resampling.NEAREST))
                disparity = scipy.ndimage.median_filter(raw_pred, size=(H//64, W//64))
                depth = 1. / np.maximum(disparity, 1e-2)

                eroded_mask = scipy.ndimage.binary_erosion(self.input_mask_list[i].squeeze(-1) > 0.5, structure=np.ones((7, 7)))
                eroded_mask = (eroded_mask > 0.5)

                median_depth = np.median(depth[eroded_mask])
                scaled_depth = (depth - median_depth) / np.abs(depth[eroded_mask] - median_depth).mean()
                masked_depth = scaled_depth * eroded_mask
                self.input_depth_list.append(masked_depth[:, :, np.newaxis].astype(np.float32))
                self.input_depth_mask_list.append(eroded_mask[:, :, np.newaxis].astype(np.float32))

            del pipe
            torch.cuda.empty_cache()
        else:
            for i, file in enumerate(file_list):
                self.input_depth_list.append(np.zeros((self.H, self.W, 1)).astype(np.float32))
                self.input_depth_mask_list.append(np.zeros((self.H, self.W, 1)).astype(np.float32))

    def load_cam_poses(self, file):
        with open(file, 'r') as f:
            cameras = json.load(f)
        cameras = cameras

        cam_scales = np.load(os.path.join(self.opt.outdir, 'gaussians', f"{self.opt.save_path[:self.opt.save_path.rfind('_')]}_cam_scales.npy"))
        print("Cam scales: ", cam_scales)

        self.rel_cam_pose = []
        for x in range(len(cameras)):
            curr_cam = cameras[x]
            prev_cam = cameras[0]
            
            rel_orientation = R.from_quat(curr_cam['orientation']) * R.from_quat(prev_cam['orientation']).inv()
            rel_orientation = rel_orientation
            rel_orientation = rel_orientation.as_matrix()

            rel_center = (np.array(curr_cam['pos']) - np.array(prev_cam['pos']))
            self.rel_cam_pose.append({'pos': rel_center, 'orientation': rel_orientation, 'scale': cam_scales[x]})

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, t=0):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = f'logs/{opt.save_path}_mesh_{t:03d}.ply'
            mesh = self.renderer.gaussians.extract_mesh_t(path, self.opt.density_thresh, t=t, max_T=self.renderer.T)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = f'logs/{opt.save_path}_mesh_{t:03d}.obj'
            mesh = self.renderer.gaussians.extract_mesh_t(path, self.opt.density_thresh, t=t, max_T=self.renderer.T)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                    time=t
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)
        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_4d_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def render_visualization(self, file_name, orbit=None, account_for_global_motion=False):

        assert orbit in [None, 'hor']

        render_height, render_width = self.input_img_torch_orig_list[0].shape[-2:]
        render_cam = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)

        image_list = []
        nframes = self.vid_length * 5
        hor = 180
        delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        for _ in range(nframes):
            pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
            cur_cam = MiniCam(
                pose,
                render_width,
                render_height,
                render_cam.fovy,
                render_cam.fovx,
                render_cam.near,
                render_cam.far,
                time=time
            )
            with torch.no_grad():
                outputs = self.renderer.render(
                    cur_cam,
                    direct_render=True,
                    account_for_global_motion=account_for_global_motion
                )

            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % self.vid_length
            if orbit == 'hor': hor = (hor+delta_hor) % 360

        export_to_gif(image_list, file_name)

    # no gui mode
    def train(self, iters):

        # Main training loop    
        if iters > 0:
            self.prepare_train()
            # Optimize 4D Gaussian deformations
            for _ in tqdm.trange(iters):
                self.train_step()

        # Update global motion
        self.renderer.freeze_gaussians()
        # Figure out how much we need to shift in 3D space based on displacements in the screen space
        render_height, render_width = self.input_img_torch_orig_list[0].shape[-2:]
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        render_cam = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)
        cam = MiniCam(pose, render_width, render_height, render_cam.fovy, render_cam.fovx, render_cam.near, render_cam.far)
        median_z = torch.median(self.renderer.gaussians.get_xyz[:, -1]).detach()
        dist_to_cam = render_cam.campos[-1] - median_z
        x_scale = (dist_to_cam + cam.znear) / cam.projection_matrix[0, 0]
        y_scale = (dist_to_cam + cam.znear) / cam.projection_matrix[1, 1]
        # Initialize the warp
        translations = []
        for obj_cx, obj_cy in zip(self.obj_cx_list, self.obj_cy_list):
            translation = torch.tensor(
                [obj_cx*x_scale, -obj_cy*y_scale, 0.],
                dtype=torch.float32,
                device="cuda",
            )
            translations.append(translation)
        translations = torch.stack(translations)
        scales = torch.stack(self.obj_scale_list)
        self.renderer.initialize_global_motion(
            self.opt,
            translation=translations,
            scale=scales,
        )
        # Optimize the warp
        self.optimizer = self.renderer.global_motion_optimizer
        self.step = 0
        for _ in tqdm.trange(iters):
            self.optimize_global_motion()

        # Render eval
        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_flow.gif'),
            orbit='hor',
            account_for_global_motion=False,
        )
        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_flow_no_orbit.gif'),
            orbit=None,
            account_for_global_motion=False,
        )
        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_flow_no_orbit_shift.gif'),
            orbit=None,
            account_for_global_motion=True,
        )

        # Joint finetune
        self.renderer.reset_and_create_joint_optimizer(self.opt, 0.1)
        self.optimizer = self.renderer.gaussians.optimizer
        self.step = 0
        for _ in tqdm.trange(100):
            self.train_step(finetune_all=True)

        # Save
        os.makedirs(os.path.join(self.opt.outdir, "gaussians"), exist_ok=True)
        self.renderer.save_gaussians(os.path.join(self.opt.outdir, "gaussians", self.opt.save_path+"_4d.pkl"))
        self.save_model(mode='model')
        self.renderer.gaussians.save_deformation(self.opt.outdir, self.opt.save_path)
        self.renderer.save_global_motion(os.path.join(self.opt.outdir, "gaussians", self.opt.save_path+"_4d_global_motion.pkl"))

        # Render eval
        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_joint_flow.gif'),
            orbit='hor',
            account_for_global_motion=False,
        )
        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_joint_flow_no_orbit.gif'),
            orbit=None,
            account_for_global_motion=False,
        )
        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_joint_flow_no_orbit_shift.gif'),
            orbit=None,
            account_for_global_motion=True,
        )

        # for t in range(self.vid_length):
        #     self.save_model(mode='geo+tex', t=t)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # auto find mesh from stage 1
    opt.load = os.path.join(opt.outdir, opt.save_path + '_model.ply')
    os.makedirs(opt.visdir, exist_ok=True)

    gui = GUI(opt)

    gui.train(opt.iters)