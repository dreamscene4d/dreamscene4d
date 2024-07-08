import os
import tqdm
import scipy
import torch
import numpy as np
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
from transformers import pipeline

import argparse
from omegaconf import OmegaConf

from PIL import Image
from torchvision.transforms.functional import center_crop
from diffusers.utils import export_to_gif

from cameras import orbit_camera, OrbitCamera, MiniCam
from utils.general_utils import safe_normalize
from gs_renderer import Renderer

from grid_put import mipmap_linear_grid_put_2d

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.seed = "random"

        # models
        self.device = torch.device("cuda")

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)

        # input image
        self.input_img = None
        self.input_depth = None
        self.input_depth_mask = None
        self.input_mask = None
        self.input_scale = None
        self.input_img_torch = None
        self.input_depth_torch = None
        self.input_depth_mask_torch = None
        self.input_mask_torch = None

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
        if self.opt.input is not None:
            self.load_input(self.opt.input)
            self.get_depth(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load, lr_scale=1.)   
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts, lr_scale=10.)

    def seed_everything(self):

        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

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

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                print(f"[INFO] loading stable zero123...")
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                print(f"[INFO] loading zero123...")
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            
            height, width = self.input_img.shape[:2]
            resize_factor = 720 / max(width, height) if max(width, height) > 720 else 1.0
            H_ = int(height * resize_factor)
            W_ = int(width * resize_factor)

            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
            self.input_img_torch_orig = F.interpolate(self.input_img_torch, (H_, W_), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
            self.input_mask_torch_orig = F.interpolate(self.input_mask_torch, (H_, W_), mode="nearest")

            self.input_depth_torch = torch.from_numpy(self.input_depth).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
            self.input_depth_torch_orig = F.interpolate(self.input_depth_torch, (H_, W_), mode="nearest")

            self.input_depth_mask_torch = torch.from_numpy(self.input_depth_mask).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
            self.input_depth_mask_torch_orig = F.interpolate(self.input_depth_mask_torch, (H_, W_), mode="nearest")

            N, C, H, W = self.input_mask_torch.shape

            mask = self.input_mask_torch > 0.5
            nonzero_idxes = torch.nonzero(mask[0,0])
            if len(nonzero_idxes) > 0:
                # Find bbox
                min_x = nonzero_idxes[:, 1].min()
                max_x = nonzero_idxes[:, 1].max()
                min_y = nonzero_idxes[:, 0].min()
                max_y = nonzero_idxes[:, 0].max()
                width = (max_x - min_x) / W
                height = (max_y - min_y) / H
                # Find cx cy
                cx = (max_x + min_x) / 2
                cx = ((cx / W) * 2 - 1)
                cy = (max_y + min_y) / 2
                cy = ((cy / H) * 2 - 1)
                self.obj_cx = cx
                self.obj_cy = cy
                # Find scale
                scale_x = width / 0.65
                scale_y = height / 0.65
                scale = max(scale_x, scale_y)
                self.input_scale = scale
                # Construct affine warp and grid
                theta = torch.tensor([[[scale, 0, cx], [0, scale, cy]]], device=self.input_img_torch.device)
                resize_factor = self.opt.ref_size / min(H, W)
                grid = F.affine_grid(theta, (N, C, int(H*resize_factor), int(W*resize_factor)), align_corners=True)
                # Change border of image to white because we assume white background
                self.input_img_torch[:, :, 0] = 1.
                self.input_img_torch[:, :, -1] = 1.
                self.input_img_torch[:, :, :, 0] = 1.
                self.input_img_torch[:, :, :, -1] = 1.
                # Aspect preserving grid sample, this recenters and scales the object
                self.input_img_torch = F.grid_sample(self.input_img_torch, grid, align_corners=True, padding_mode='border')
                self.input_mask_torch = F.grid_sample(self.input_mask_torch, grid, align_corners=True)
                self.input_depth_torch = F.grid_sample(self.input_depth_torch, grid, mode='nearest', align_corners=True)
                self.input_depth_mask_torch = F.grid_sample(self.input_depth_mask_torch, grid, mode='nearest', align_corners=True)
                # Center crop
                self.input_img_torch = center_crop(self.input_img_torch, self.opt.ref_size)
                self.input_mask_torch = center_crop(self.input_mask_torch, self.opt.ref_size)
                self.input_depth_torch = center_crop(self.input_depth_torch, self.opt.ref_size)
                self.input_depth_mask_torch = center_crop(self.input_depth_mask_torch, self.opt.ref_size)

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                c_list, v_list = [], []
                c, v = self.guidance_zero123.get_img_embeds(self.input_img_torch)
                for _ in range(self.opt.batch_size):
                    c_list.append(c)
                    v_list.append(v)
                self.guidance_zero123.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]

    def train_step(self):

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### known view
            cur_cam = self.fixed_cam
            out = self.renderer.render(cur_cam)
            target_img = self.input_img_torch
            target_mask = (self.input_mask_torch > 0.5).float()

            # rgb loss
            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            loss = loss + 10000 * step_ratio * (self.balanced_mask_loss(pred=image, target=target_img, mask=target_mask))

            # mask loss
            mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
            loss = loss + 1000 * step_ratio * (self.balanced_mask_loss(pred=mask, target=target_mask, mask=target_mask))

            # depth loss
            if self.opt.depth_loss:
                depth = out["depth"].unsqueeze(0) # [1, 1, H, W]
                depth = torch.nan_to_num(depth)
                if depth[self.input_depth_mask_torch > 0.5].sum() > 0:
                    median_depth = torch.median(depth[self.input_depth_mask_torch > 0.5])
                    scaled_depth = (depth - median_depth) / torch.abs(depth[self.input_depth_mask_torch > 0.5] - median_depth + 1e-6).mean()
                    depth = scaled_depth * (self.input_depth_mask_torch > 0.5)
                    depth_loss = (1. - self.pearson(depth[self.input_depth_mask_torch > 0.5], self.input_depth_torch[self.input_depth_mask_torch > 0.5]))
                    loss = loss + 250 * step_ratio * depth_loss

            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                # radius = 0
                radius = np.random.uniform(-0.5, 0.5) 

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

                # enable mvdream training
                if self.opt.mvdream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                        image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device, non_blocking=True)

            # guidance loss
            if self.enable_sd:
                if self.opt.mvdream:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio)
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)

            if self.enable_zero123:
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio) / self.opt.batch_size
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    if self.renderer.gaussians.get_xyz.shape[0] < 15000:
                        self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

    def optimize_global_motion(self):

        render_height, render_width = self.input_img_torch_orig.shape[-2:]
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        render_cam = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)
        cur_cam = MiniCam(pose, render_width, render_height, render_cam.fovy, render_cam.fovx, render_cam.near, render_cam.far)

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.update_learning_rate(self.step)

            loss = 0

            ### known view
            out = self.renderer.render(cur_cam, account_for_global_motion=True)

            target_img = self.input_img_torch_orig
            target_mask = (self.input_mask_torch_orig > 0.5).float()

            # rgb loss
            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            loss = loss + 1000 * step_ratio * (self.balanced_mask_loss(pred=image, target=target_img, mask=target_mask))

            # mask loss
            mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
            loss = loss + 100 * step_ratio * (self.balanced_mask_loss(pred=mask, target=target_mask, mask=target_mask))
            
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
        # load image
        assert (self.opt.input_mask is not None)
        print(f'[INFO] Load image from {file}...')
        img = Image.open(file)
        if self.opt.resize_square:
            width, height = img.size
            new_dim = min(width, height)
            img = img.resize([new_dim, new_dim], Image.Resampling.BICUBIC)
        img = np.array(img)
        mask = Image.open(self.opt.input_mask)
        if self.opt.resize_square:
            mask = mask.resize([new_dim, new_dim], Image.Resampling.NEAREST)
        mask = np.array(mask)
        mask = mask.astype(np.float32) / 255.0
        if len(mask.shape) == 3:
            mask = mask[:, :, 0:1]
        else:
            mask = mask[:, :, np.newaxis]

        img = img.astype(np.float32) / 255.0
            
        # white bg
        self.input_mask = mask
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)

    @torch.no_grad()
    def get_depth(self, file):
        
        if self.opt.depth_loss:
            # The "depth" returned is actually the disparity
            pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf")
            with torch.no_grad():
                raw_pred = pipe(Image.open(file))["depth"]
            H, W = self.input_img.shape[:2]
            raw_pred = np.array(raw_pred.resize([W, H], Image.Resampling.NEAREST))
            disparity = scipy.ndimage.median_filter(raw_pred, size=(H//64, W//64))
            depth = 1. / np.maximum(disparity, 1e-2)

            eroded_mask = scipy.ndimage.binary_erosion(self.input_mask.squeeze(-1) > 0.5, structure=np.ones((7, 7)))
            eroded_mask = (eroded_mask > 0.5)

            median_depth = np.median(depth[eroded_mask])
            scaled_depth = (depth - median_depth) / np.abs(depth[eroded_mask] - median_depth).mean()
            masked_depth = scaled_depth * eroded_mask
            self.input_depth = masked_depth[:, :, np.newaxis].astype(np.float32)
            self.input_depth_mask = eroded_mask[:, :, np.newaxis].astype(np.float32)

            del pipe
            torch.cuda.empty_cache()
        else:
            self.input_depth = np.zeros((self.H, self.W, 1)).astype(np.float32)
            self.input_depth_mask = np.zeros((self.H, self.W, 1)).astype(np.float32)

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

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
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def render_visualization(self, file_name, orbit=None, depth=False, account_for_global_motion=False):

        assert orbit in [None, 'hor']

        render_height, render_width = self.input_img_torch_orig.shape[-2:]
        render_cam = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)
        
        image_list = []
        nframes = self.opt.batch_size * 5
        hor = 180
        delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        for _ in range(nframes):
            pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
            cur_cam = MiniCam(pose, render_width, render_height, render_cam.fovy, render_cam.fovx, render_cam.near, render_cam.far)
            with torch.no_grad():
                outputs = self.renderer.render(
                    cur_cam,
                    account_for_global_motion=account_for_global_motion
                )

            if depth:
                ALPHA_THRESH = 0.95
                out = outputs["depth"].squeeze(0).cpu().detach().numpy().astype(np.float32)
                depth_mask = outputs["alpha"].squeeze(0).cpu().detach().numpy().astype(np.float32)
                out -= out[depth_mask > ALPHA_THRESH].min()
                out /= (out[depth_mask > ALPHA_THRESH].max() - out[depth_mask > ALPHA_THRESH].min())
                out = out * (depth_mask > 0.75)
                out = Image.fromarray(np.uint8(out*255))
            else:
                out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                out = np.transpose(out, (1, 2, 0))
                out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % self.opt.batch_size
            if orbit == 'hor': hor = (hor+delta_hor) % 360

        export_to_gif(image_list, file_name)

    def train(self, iters):

        # 1. Optimize obj-centric Gaussians
        if iters > 0:
            self.prepare_train()
            
            for _ in tqdm.trange(iters):
                self.train_step()
                    
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)

        # Save
        os.makedirs(os.path.join(self.opt.outdir, "gaussians"), exist_ok=True)
        self.renderer.save_gaussians(os.path.join(self.opt.outdir, "gaussians", self.opt.save_path+".pkl"))
        self.save_model(mode='model')
        # self.save_model(mode='geo+tex')

        # 2. Optimize obj-centric to world warp
        self.renderer.freeze_gaussians()
        # Figure out how much we need to shift in 3D space based on displacements in the screen space
        render_height, render_width = self.input_img_torch_orig.shape[-2:]
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        render_cam = OrbitCamera(render_width, render_height, r=self.opt.radius, fovy=self.opt.fovy)
        cam = MiniCam(pose, render_width, render_height, render_cam.fovy, render_cam.fovx, render_cam.near, render_cam.far)
        median_z = torch.median(self.renderer.gaussians.get_xyz[:, -1]).detach()
        dist_to_cam = render_cam.campos[-1] - median_z
        x_scale = (dist_to_cam + cam.znear) / cam.projection_matrix[0, 0]
        y_scale = (dist_to_cam + cam.znear) / cam.projection_matrix[1, 1]
        # Initialize the warp
        self.renderer.initialize_global_motion(
            self.opt,
            translation=[
                self.obj_cx*x_scale,              
                -self.obj_cy*y_scale,
                0.
            ],
            scale=self.input_scale,
        )
        # Optimize the warp
        self.optimizer = self.renderer.global_motion_optimizer
        for _ in tqdm.trange(iters):
            self.optimize_global_motion()
        # Save
        self.renderer.save_global_motion(os.path.join(self.opt.outdir, "gaussians", self.opt.save_path+"_global_motion.pkl"))

        # Render eval
        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_static.gif'),
            orbit='hor',
            account_for_global_motion=False,
        )

        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_static_shift.gif'),
            orbit=None,
            account_for_global_motion=True,
        )

        self.render_visualization(
            file_name=os.path.join(self.opt.visdir, f'{self.opt.save_path}_static_depth.gif'),
            orbit='hor',
            depth=True,
            account_for_global_motion=False,
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    os.makedirs(opt.visdir, exist_ok=True)
    os.makedirs(opt.outdir, exist_ok=True)

    gui = GUI(opt)

    gui.train(opt.iters)