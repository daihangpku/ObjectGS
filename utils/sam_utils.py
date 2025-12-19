import torch
import numpy as np
import h5py
import os
import os.path as osp
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Union, Literal
from dataclasses import dataclass, field
from tqdm import tqdm

@dataclass
class ImgGroupModelConfig:
    model_type: Literal["sam_fb", "sam_hf", "maskformer"] = "sam_fb"
    sam_model_type: str = "vit_h"
    sam_model_ckpt: str = "models/sam_vit_h_4b8939.pth"
    sam_kwargs: dict = field(default_factory=lambda: {
        "points_per_side": 32,
        "pred_iou_thresh": 0.90,
        "stability_score_thresh": 0.90,
    })
    device: str = "cuda"

class ImgGroupModel:
    def __init__(self, config: ImgGroupModelConfig):
        self.config = config
        self.device = config.device
        self.model = None

    def __call__(self, img: np.ndarray):
        if self.config.model_type == "sam_hf":
            from transformers import pipeline
            if self.model is None:
                self.model = pipeline("mask-generation", model="facebook/sam-vit-huge", device=self.device)
            img = Image.fromarray(img)
            masks = self.model(img, points_per_side=32, pred_iou_thresh=0.90, stability_score_thresh=0.90)
            masks = masks['masks']
            masks = sorted(masks, key=lambda x: x.sum())
            return masks
        
        elif self.config.model_type == "sam_fb":
            if self.model is None:
                from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
                registry = sam_model_registry[self.config.sam_model_type]
                model = registry(checkpoint=self.config.sam_model_ckpt)
                model = model.to(device=self.config.device)
                self.model = SamAutomaticMaskGenerator(
                    model=model, **self.config.sam_kwargs
                )
            masks = self.model.generate(img)
            masks = [m['segmentation'] for m in masks]
            masks = sorted(masks, key=lambda x: x.sum())
            return masks
        
        elif self.config.model_type == "maskformer":
            from transformers import pipeline
            if self.model is None:
                self.model = pipeline(model="facebook/maskformer-swin-large-coco", device=self.device)
            img = Image.fromarray(img)
            masks = self.model(img)
            masks = [
                (np.array(m['mask']) != 0)
                for m in masks
            ]
            masks = sorted(masks, key=lambda x: x.sum())
            return masks
        return []

def create_pixel_mask_array(masks: torch.Tensor):
    max_masks = masks.sum(dim=0).max().item()
    image_shape = masks.shape[1:]
    pixel_mask_array = torch.full(
        (max_masks, image_shape[0], image_shape[1]), -1, dtype=torch.int
    ).to(masks.device)

    for m, mask in enumerate(masks):
        mask_clone = mask.clone()
        for i in range(max_masks):
            free = pixel_mask_array[i] == -1
            masked_area = mask_clone == 1
            right_index = free & masked_area
            if len(pixel_mask_array[i][right_index]) != 0:
                pixel_mask_array[i][right_index] = m
            mask_clone[right_index] = 0
    pixel_mask_array = pixel_mask_array.permute(1, 2, 0)
    return pixel_mask_array

def calculate_3d_groups(
    img_group_model: ImgGroupModel,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    point: torch.Tensor,
    max_scale: float = 2.0,
):
    image_shape = rgb.shape[:2]
    depth = depth.view(-1, 1)
    point = point.view(-1, 3)

    def helper_return_no_masks():
        pixel_level_keys = torch.full(
            (image_shape[0], image_shape[1], 1), -1, dtype=torch.int
        )
        scale = torch.Tensor([0.0]).view(-1, 1)
        mask_cdf = torch.full(
            (image_shape[0], image_shape[1], 1), 1, dtype=torch.float
        )
        return (pixel_level_keys, scale, mask_cdf)

    masks = img_group_model((rgb.cpu().numpy() * 255).astype(np.uint8))

    if len(masks) == 0:
        return helper_return_no_masks()

    sam_mask = []
    scale = []

    all_masks = torch.stack(
        [torch.from_numpy(_).to(rgb.device) for _ in masks]
    )
    
    # Erode masks
    eroded_masks = torch.conv2d(
        all_masks.unsqueeze(1).float(),
        torch.full((3, 3), 1.0).view(1, 1, 3, 3).to(rgb.device),
        padding=1,
    )
    eroded_masks = (eroded_masks >= 5).squeeze(1)

    for i in range(len(masks)):
        curr_mask = eroded_masks[i]
        curr_mask = curr_mask.flatten()
        curr_points = point[curr_mask]
        if curr_points.shape[0] == 0:
            continue
        extent = (curr_points.std(dim=0) * 2).norm()
        if extent.item() < max_scale:
            sam_mask.append(curr_mask.reshape(image_shape))
            scale.append(extent.item())

    if len(sam_mask) == 0:
        return helper_return_no_masks()

    sam_mask = torch.stack(sam_mask)
    scale = torch.Tensor(scale).view(-1, 1).to(rgb.device)

    pixel_level_keys = create_pixel_mask_array(sam_mask).long()

    mask_inds, counts = torch.unique(pixel_level_keys, return_counts=True)
    mask_sorted = torch.argsort(counts)
    mask_inds, counts = mask_inds[mask_sorted], counts[mask_sorted]
    
    # Handle case where -1 is not present or is present
    neg_one_idx = (mask_inds == -1).nonzero()
    if len(neg_one_idx) > 0:
        counts[neg_one_idx] = 0
        
    probs = counts / counts.sum()
    
    prob_lookup = torch.zeros(pixel_level_keys.max() + 2, device=rgb.device) # +1 for -1 offset
    prob_lookup[mask_inds + 1] = probs
    
    mask_probs = prob_lookup[pixel_level_keys + 1]
    
    mask_log_probs = torch.log(mask_probs)
    never_masked = mask_log_probs.isinf()
    mask_log_probs[never_masked] = 0.0
    mask_log_probs = mask_log_probs / (
        mask_log_probs.sum(dim=-1, keepdim=True) + 1e-6
    )
    mask_cdf = torch.cumsum(mask_log_probs, dim=-1)
    mask_cdf[never_masked] = 1.0

    return (pixel_level_keys.cpu(), scale.cpu(), mask_cdf.cpu())

def save_sam_data(sam_data_path, prefix, pixel_level_keys, scale_3d, group_cdf):
    if not osp.exists(osp.dirname(sam_data_path)):
        os.makedirs(osp.dirname(sam_data_path))

    with h5py.File(sam_data_path, "a") as f:
        for i in range(len(pixel_level_keys)):
            grp_path = f"{prefix}/pixel_level_keys/{i}"
            if grp_path in f:
                del f[grp_path]
            f.create_dataset(grp_path, data=pixel_level_keys[i])
            
            grp_path = f"{prefix}/scale_3d/{i}"
            if grp_path in f:
                del f[grp_path]
            f.create_dataset(grp_path, data=scale_3d[i])
            
            grp_path = f"{prefix}/group_cdf/{i}"
            if grp_path in f:
                del f[grp_path]
            f.create_dataset(grp_path, data=group_cdf[i])

def load_sam_data(sam_data_path, prefix):
    if osp.exists(sam_data_path):
        sam_data = h5py.File(sam_data_path, "r")
        if prefix not in sam_data.keys():
            return None

        sam_data = sam_data[prefix]
        pixel_level_keys_list, scales_3d_list, group_cdf_list = [], [], []

        num_entries = len(sam_data["pixel_level_keys"].keys())
        for i in range(num_entries):
            pixel_level_keys_list.append(
                torch.from_numpy(sam_data["pixel_level_keys"][str(i)][...])
            )
        
        for i in range(num_entries):
            scales_3d_list.append(torch.from_numpy(sam_data["scale_3d"][str(i)][...]))
            
        for i in range(num_entries):
            group_cdf_list.append(torch.from_numpy(sam_data["group_cdf"][str(i)][...]))
            
        return pixel_level_keys_list, scales_3d_list, group_cdf_list
    return None

def generate_sam_data_for_scene(
    scene,
    sam_data_path: str,
    img_group_config: ImgGroupModelConfig,
    gaussians=None,
    pipeline_config=None,
    background=None
):
    img_group_model = ImgGroupModel(img_group_config)
    
    pixel_level_keys_list = []
    scales_3d_list = []
    group_cdf_list = []
    
    cameras = scene.train_cameras
    # scene.train_cameras is a dict of resolution scales
    # We usually use resolution scale 1.0
    if 1.0 in cameras:
        cam_list = cameras[1.0]
    else:
        cam_list = list(cameras.values())[0]
        
    for cam in tqdm(cam_list, desc="Generating SAM data"):
        # cam.original_image is [3, H, W]
        rgb = cam.original_image.permute(1, 2, 0) # [H, W, 3]
        
        if gaussians is not None and pipeline_config is not None and background is not None:
            from gaussian_renderer import render
            
            # Temporarily switch render mode to get depth
            original_render_mode = getattr(gaussians, "render_mode", "RGB")
            gaussians.render_mode = "RGB+ED"
            
            render_pkg = render(cam, gaussians, pipeline_config, background)
            
            # Restore render mode
            gaussians.render_mode = original_render_mode
            
            depth = render_pkg["render_depth"] # [1, H, W]
            depth = depth.squeeze(0)
            
            H, W = depth.shape
            
            # Create meshgrid
            y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing='ij')
            
            # Calculate intrinsics from Fov
            fovx = cam.FoVx
            fovy = cam.FoVy
            fx = W / (2 * np.tan(fovx / 2))
            fy = H / (2 * np.tan(fovy / 2))
            cx = W / 2
            cy = H / 2
            
            x_norm = (x - cx) / fx
            y_norm = (y - cy) / fy
            
            # Camera space points
            xyz = torch.stack([x_norm * depth, y_norm * depth, depth], dim=-1) # [H, W, 3]
            
            # Transform to world space
            # cam.world_view_transform is W2C (transposed?)
            # In 3DGS code: viewmatrix = world_view_transform
            # It is usually stored as row-major or column-major?
            # In utils/graphics_utils.py: getWorld2View returns Rt (4x4)
            # In Camera class (scene/cameras.py), world_view_transform is usually passed to shader.
            # It is typically W2C.
            
            c2w = torch.inverse(cam.world_view_transform.transpose(0, 1))
            
            R = c2w[:3, :3]
            t = c2w[:3, 3]
            
            # xyz is [H, W, 3]. We want to apply R and t.
            # points = R * xyz + t
            points = torch.matmul(xyz, R.t()) + t
            
        else:
            print("Warning: No gaussians provided, cannot render depth. Skipping 3D scale calculation.")
            depth = torch.zeros((rgb.shape[0], rgb.shape[1]), device=rgb.device)
            points = torch.zeros((rgb.shape[0], rgb.shape[1], 3), device=rgb.device)

        pixel_level_keys, scale_3d, group_cdf = calculate_3d_groups(
            img_group_model, rgb, depth, points
        )
        
        pixel_level_keys_list.append(pixel_level_keys)
        scales_3d_list.append(scale_3d)
        group_cdf_list.append(group_cdf)
        
    save_sam_data(sam_data_path, img_group_config.model_type, pixel_level_keys_list, scales_3d_list, group_cdf_list)
