#!/usr/bin/env python3
"""
PatchCore Anomaly Detection Pipeline
Extracts, visualizes, and scores anomalous/normal patches from PatchCore results.

Outputs:
- anomalous_patches.pdf: Visualization of anomalous patches
- normal_patches.pdf: Visualization of normal patches
- scores.json: Anomaly scores and statistics
- distributions.json: Distance distributions between patch types
"""

import os
import json
import argparse
import sqlite3
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from PIL import Image
from tqdm.auto import tqdm


import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple

class FeatureToPatchMapper:
    """
    Maps feature map coordinates to original image coordinates.
    Handles the full preprocessing pipeline: resize -> center crop -> feature extraction
    Supports rectangular images (original_width, original_height).
    """
    
    def __init__(self, 
                 original_width: int = 1024,
                 original_height: int = 1024,
                 resize_size: int = 256, 
                 crop_size: int = 224,
                 feature_map_size: int = 28):
        """
        Args:
            original_width: Original image width
            original_height: Original image height
            resize_size: Size after resize (square)
            crop_size: Size after center crop (square)
            feature_map_size: Feature map size after backbone (square)
        """
        self.original_width = original_width
        self.original_height = original_height
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.feature_map_size = feature_map_size
        
        # Calculate transformations
        self.resize_ratio_w = resize_size / original_width
        self.resize_ratio_h = resize_size / original_height
        
        # Center crop offset in resized space (256x256)
        self.crop_margin_w = (resize_size - crop_size) // 2
        self.crop_margin_h = (resize_size - crop_size) // 2
        
        # Stride in cropped image space (224x224)
        self.stride_in_crop = crop_size / feature_map_size
        
        # Stride in original image space
        self.stride_in_original_w = self.stride_in_crop / self.resize_ratio_w
        self.stride_in_original_h = self.stride_in_crop / self.resize_ratio_h
        
        # Crop offset in original image space
        self.crop_offset_in_original_w = self.crop_margin_w / self.resize_ratio_w
        self.crop_offset_in_original_h = self.crop_margin_h / self.resize_ratio_h
        
        # Receptive field size (patch size) in original image
        self.patch_size_in_original_w = int(np.ceil(self.stride_in_original_w))
        self.patch_size_in_original_h = int(np.ceil(self.stride_in_original_h))
        
        self._print_info()
    
    def _print_info(self):
        """Print transformation information."""
        print("=" * 60)
        print("Feature to Patch Mapper Configuration")
        print("=" * 60)
        print(f"Original image size: {self.original_width}x{self.original_height}")
        print(f"After resize: {self.resize_size}x{self.resize_size}")
        print(f"After center crop: {self.crop_size}x{self.crop_size}")
        print(f"Feature map size: {self.feature_map_size}x{self.feature_map_size}")
        print("-" * 60)
        print(f"Crop margin (in 256x256 space): {self.crop_margin_w}x{self.crop_margin_h} pixels")
        print(f"Crop offset (in original space): {self.crop_offset_in_original_w:.1f}x{self.crop_offset_in_original_h:.1f} pixels")
        print(f"Stride (in 224x224 space): {self.stride_in_crop:.2f} pixels/feature")
        print(f"Stride (in original space): {self.stride_in_original_w:.2f}x{self.stride_in_original_h:.2f} pixels/feature")
        print(f"Patch size in original image: {self.patch_size_in_original_w}x{self.patch_size_in_original_h} pixels")
        print("=" * 60)
    
    def patch_index_to_coordinates(self, patch_idx: int) -> Tuple[int, int]:
        """
        Convert linear patch index to 2D coordinates in feature map.
        
        Args:
            patch_idx: Linear index [0, 784) for 28x28
            
        Returns:
            (row, col) in feature map coordinates
        """
        row = patch_idx // self.feature_map_size
        col = patch_idx % self.feature_map_size
        return row, col
    
    def feature_coords_to_crop_bbox(self, feature_row: int, feature_col: int) -> Tuple[int, int, int, int]:
        """
        Convert feature map coordinates to bounding box in 224x224 cropped image.
        
        Args:
            feature_row: Row index in feature map [0, 27]
            feature_col: Column index in feature map [0, 27]
            
        Returns:
            (y_start, y_end, x_start, x_end) in 224x224 cropped image coordinates
        """
        center_y = (feature_row + 0.5) * self.stride_in_crop
        center_x = (feature_col + 0.5) * self.stride_in_crop
        
        half_size = self.stride_in_crop / 2
        
        y_start = int(center_y - half_size)
        y_end = int(center_y + half_size)
        x_start = int(center_x - half_size)
        x_end = int(center_x + half_size)
        
        y_start = max(0, y_start)
        y_end = min(self.crop_size, y_end)
        x_start = max(0, x_start)
        x_end = min(self.crop_size, x_end)
        
        return y_start, y_end, x_start, x_end
    
    def feature_coords_to_original_bbox(self, feature_row: int, feature_col: int) -> Tuple[int, int, int, int]:
        """
        Convert feature map coordinates to bounding box in original image.
        
        Args:
            feature_row: Row index in feature map [0, 27]
            feature_col: Column index in feature map [0, 27]
            
        Returns:
            (y_start, y_end, x_start, x_end) in original image coordinates
        """
        center_y_crop = (feature_row + 0.5) * self.stride_in_crop
        center_x_crop = (feature_col + 0.5) * self.stride_in_crop
        
        center_y_original = (center_y_crop + self.crop_margin_h) / self.resize_ratio_h
        center_x_original = (center_x_crop + self.crop_margin_w) / self.resize_ratio_w
        
        half_size_w = self.stride_in_original_w / 2
        half_size_h = self.stride_in_original_h / 2
        
        y_start = int(center_y_original - half_size_h)
        y_end = int(center_y_original + half_size_h)
        x_start = int(center_x_original - half_size_w)
        x_end = int(center_x_original + half_size_w)
        
        y_start = max(0, y_start)
        y_end = min(self.original_height, y_end)
        x_start = max(0, x_start)
        x_end = min(self.original_width, x_end)
        
        return x_start, y_end, x_end, y_start
    
    def patch_index_to_original_bbox(self, patch_idx: int) -> Tuple[int, int, int, int]:
        """
        Convert linear patch index directly to bounding box in original image.
        
        Args:
            patch_idx: Linear index [0, 784)
            
        Returns:
            (y_start, y_end, x_start, x_end) in original image coordinates
        """
        row, col = self.patch_index_to_coordinates(patch_idx)
        return self.feature_coords_to_original_bbox(row, col)
    
    def extract_patch_from_original_image(self, image: torch.Tensor, patch_idx: int) -> torch.Tensor:
        """
        Extract the image region from ORIGINAL image corresponding to a patch index.
        
        Args:
            image: Original image tensor [C, H, W] or [1, C, H, W]
            patch_idx: Patch index in feature map
            
        Returns:
            Patch from original image [C, ~patch_h, ~patch_w]
        """
        if image.dim() == 4:
            image = image.squeeze(0)
        
        x_start, y_end, x_end, y_start = self.patch_index_to_original_bbox(patch_idx)
        patch = image[:, y_start:y_end, x_start:x_end]
        
        return patch
    
    def extract_patch_from_preprocessed_image(self, image: torch.Tensor, patch_idx: int) -> torch.Tensor:
        """
        Extract the image region from PREPROCESSED 224x224 image.
        
        Args:
            image: Preprocessed image tensor [C, 224, 224] or [1, C, 224, 224]
            patch_idx: Patch index in feature map
            
        Returns:
            Patch from preprocessed image [C, 8, 8]
        """
        if image.dim() == 4:
            image = image.squeeze(0)
        
        row, col = self.patch_index_to_coordinates(patch_idx)
        x_start, y_end, x_end, y_start = self.feature_coords_to_crop_bbox(row, col)
        patch = image[:, y_start:y_end, x_start:x_end]
        
        return patch
    
    def visualize_patch_location(self, image: torch.Tensor, patch_idx: int, img_name: str,
                                 show_crop_boundary: bool = True, ax: plt.Axes = None):
        if image.dim() == 4:
            image = image.squeeze(0)
        
        img_np = image.cpu().numpy()
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        cmap = 'gray' if img_np.shape[-1] == 1 else None
        
        x_start, y_end, x_end, y_start = self.patch_index_to_original_bbox(patch_idx)
        row, col = self.patch_index_to_coordinates(patch_idx)
        crop_y_start = self.crop_offset_in_original_h
        crop_y_end = crop_y_start + (self.crop_size / self.resize_ratio_h)
        crop_x_start = self.crop_offset_in_original_w
        crop_x_end = crop_x_start + (self.crop_size / self.resize_ratio_w)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        else:
            fig = None
        
        ax.imshow(img_np, cmap=cmap)
        if show_crop_boundary:
            crop_rect = mpatches.Rectangle(
                (crop_x_start, crop_y_start),
                crop_x_end - crop_x_start,
                crop_y_end - crop_y_start,
                linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', label='224x224 crop region'
            )
            ax.add_patch(crop_rect)
        
        patch_rect = mpatches.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            linewidth=3, edgecolor='red', facecolor='none',
            label=f'Patch {patch_idx}'
        )
        ax.add_patch(patch_rect)
        
        ax.set_title(
            f'Patch {patch_idx} in Original {img_name}',
            fontsize=12, fontweight='bold'
        )
        ax.legend(loc='upper right')
        ax.axis('off')
        
        if fig is not None:
            return fig, ax
        else:
            return None, ax
    
    def visualize_bbox_location(self, image: torch.Tensor, patch_bbox, img_name: str,
                                show_crop_boundary: bool = True, dist: float = 0.0, ax: plt.Axes = None):
        if image.dim() == 4:
            image = image.squeeze(0)
        
        img_np = image.cpu().numpy()
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        cmap = 'gray' if img_np.shape[-1] == 1 else None
        
        x_start, y_end, x_end, y_start = patch_bbox
        crop_y_start = self.crop_offset_in_original_h
        crop_y_end = crop_y_start + (self.crop_size / self.resize_ratio_h)
        crop_x_start = self.crop_offset_in_original_w
        crop_x_end = crop_x_start + (self.crop_size / self.resize_ratio_w)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        else:
            fig = None
        
        ax.imshow(img_np, cmap=cmap)
        if show_crop_boundary:
            crop_rect = mpatches.Rectangle(
                (crop_x_start, crop_y_start),
                crop_x_end - crop_x_start,
                crop_y_end - crop_y_start,
                linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', label='224x224 crop region'
            )
            ax.add_patch(crop_rect)
        
        patch_rect = mpatches.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            linewidth=3, edgecolor='red', facecolor='none',
            label=f'Patch with dist: {dist}'
        )
        ax.add_patch(patch_rect)
        
        ax.set_title(
            f'Patch in Original {img_name}',
            fontsize=12, fontweight='bold'
        )
        ax.legend(loc='upper right')
        ax.axis('off')
        
        if fig is not None:
            return fig, ax
        else:
            return None, ax
    
    def visualize_all_patches_grid(self, image: torch.Tensor, 
                                   save_path: str = 'all_patches_grid.png'):
        """
        Visualize the grid of all 28x28 patches overlaid on the original image.
        
        Args:
            image: Original image tensor [C, H, W]
            save_path: Where to save the visualization
        """
        if image.dim() == 4:
            image = image.squeeze(0)
        
        img_np = image.cpu().numpy()
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
            cmap = 'gray'
        else:
            cmap = None
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        ax.imshow(img_np, cmap=cmap, alpha=0.7)
        
        for patch_idx in range(self.feature_map_size ** 2):
            x_start, y_end, x_end, y_start = self.patch_index_to_original_bbox(patch_idx)
            rect = mpatches.Rectangle(
                (x_start, y_start), 
                x_end - x_start, 
                y_end - y_start,
                linewidth=0.5, edgecolor='red', facecolor='none', alpha=0.5
            )
            ax.add_patch(rect)
        
        crop_y_start = self.crop_offset_in_original_h
        crop_y_end = crop_y_start + (self.crop_size / self.resize_ratio_h)
        crop_x_start = self.crop_offset_in_original_w
        crop_x_end = crop_x_start + (self.crop_size / self.resize_ratio_w)
        
        crop_rect = mpatches.Rectangle(
            (crop_x_start, crop_y_start), 
            crop_x_end - crop_x_start, 
            crop_y_end - crop_y_start,
            linewidth=3, edgecolor='blue', facecolor='none', linestyle='--'
        )
        ax.add_patch(crop_rect)
        
        ax.set_title(
            f'All {self.feature_map_size}×{self.feature_map_size} Patches in Original Image\n'
            f'Blue dashed box: 224×224 crop region used by network\n'
            f'Red boxes: {self.patch_size_in_original_w}×{self.patch_size_in_original_h} patches',
            fontsize=12, fontweight='bold'
        )
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()



class PatchCoreAnalyzer:
    """Main pipeline for PatchCore anomaly detection analysis."""
    
    def __init__(
        self,
        csv_dir: str,
        original_width: int,
        original_height: int,
        crop_size: int,
        resize_size: int,
        num_features: int,
        masks_dir: str,
        class_name: str,
    ):
        """
        Initialize PatchCore analyzer.
        
        Args:
            db_path: Path to source SQLite database
            csv_dir: Directory containing CSV files
            original_width: Original image width
            original_height: Original image height
            crop_size: Crop size for preprocessing
            resize_size: Resize size for preprocessing
            num_features: Number of features (feature map size)
            data_dir: Output directory for results
            class_name: Class/category name
        """
        self.csv_dir = csv_dir
        self.masks_dir = masks_dir
        self.class_name = class_name
        
        self.mapper = FeatureToPatchMapper(
            original_width=original_width,
            original_height=original_height,
            resize_size=resize_size,
            crop_size=crop_size,
            feature_map_size=num_features,
        )
        
        self.neighbors_df = None
        self.anomalous_patches = None
        self.normal_patches = None
        self.scores_data = {}
    
    def load_data(self) -> None:
        """Load neighbor data from CSV."""
        csv_path = os.path.join(self.csv_dir, "patchcore_neighbors.csv")
        self.neighbors_df = pd.read_csv(csv_path)
        self.scores_df = pd.read_csv(os.path.join(self.csv_dir, "patchcore_anomaly_scores.csv"))
        print(f"Loaded {len(self.neighbors_df)} neighbor records")
    
    def filter_anomalous_samples(self, gt_threshold: float = 0.7) -> None:
        """Filter samples with ground truth anomaly score above threshold."""
        def get_gt(row):
            test_img_name = row.test_img_path.split('/')[-1]
            test_img_anomaly_type = row.test_img_path.split('/')[-2]
            if test_img_anomaly_type == 'good':
                return 0.0
            mask_path = os.path.join(self.masks_dir, test_img_anomaly_type, test_img_name.replace('.png', '_mask.png'))
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask).astype(np.float32) / 255.0
            mask.close()
            left, upper, right, bottom = self.mapper.patch_index_to_original_bbox(row.test_img_patch_id)
            
            mask_patch = mask_np[bottom:upper, left:right]
            return mask_patch.mean().item()
        
        self.neighbors_df['gt'] = self.neighbors_df.apply(get_gt, axis=1)
        
        mask = self.neighbors_df['gt'] > gt_threshold
        self.anomalous_patches = self.neighbors_df[mask].copy()
        print(f"Filtered to {len(self.anomalous_patches)} anomalous patches")
    
    def filter_normal_samples(self, gt_threshold: float = 0.7) -> None:
        """Filter samples with ground truth anomaly score below threshold."""
        
        mask = self.neighbors_df['gt'] <= gt_threshold
        self.normal_patches = self.neighbors_df[mask].copy()
        print(f"Filtered to {len(self.normal_patches)} normal patches")
    
    def generate_pdf_visualizations(
        self,
        data: pd.DataFrame,
        output_name: str,
        max_samples: int = 50,
    ) -> None:
        """Generate PDF visualization of patches."""
        output_path = os.path.join(self.csv_dir,output_name)
        sample_size = min(max_samples, len(data))
        
        with PdfPages(str(output_path)) as pdf:
            for idx, (_, row) in enumerate(tqdm(data.sample(sample_size).iterrows(), total=sample_size)):
                try:
                    # Load images
                    test_img = Image.open(row['testimgpath']).convert('RGB')
                    normal_img = Image.open(row['normalimgs']).convert('RGB')
                    not_normal_img = Image.open(row['notnormalimgs']).convert('RGB')
                    
                    # Convert to tensors
                    test_tensor = torch.from_numpy(np.array(test_img).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
                    normal_tensor = torch.from_numpy(np.array(normal_img).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
                    not_normal_tensor = torch.from_numpy(np.array(not_normal_img).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
                    
                    # Get coordinates
                    normal_coords = self.mapper.patch_index_to_original_bbox(row['normalpatchidx'])
                    not_normal_coords = self.mapper.patch_index_to_original_bbox(row['notnormalpatchidx'])
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    self._visualize_patch_location(test_tensor, row['testimgpatchid'], "Test", ax=axes[0])
                    self._visualize_bbox_location(normal_tensor, normal_coords, "Normal", 
                                                 dist=row['ndist'], ax=axes[1])
                    self._visualize_bbox_location(not_normal_tensor, not_normal_coords, "Not Normal",
                                                 dist=row['notndist'], ax=axes[2])
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    continue
        
        print(f"Saved visualizations to {output_path}")
    
    def _visualize_patch_location(
        self,
        image: torch.Tensor,
        patch_idx: int,
        img_name: str,
        ax=None,
    ) -> None:
        """Visualize patch location on image."""
        if image.dim() == 4:
            image = image.squeeze(0)
        
        img_np = image.cpu().numpy()
        if img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        cmap = "gray" if img_np.shape[-1] == 1 else None
        
        x_start, y_end, x_end, y_start = self.mapper.patch_index_to_original_bbox(patch_idx)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        
        ax.imshow(img_np, cmap=cmap)
        
        patch_rect = mpatches.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            linewidth=3,
            edgecolor='red',
            facecolor='none',
            label=f'Patch {patch_idx}',
        )
        ax.add_patch(patch_rect)
        ax.set_title(f"Patch {patch_idx} in Original {img_name}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
    
    def _visualize_bbox_location(
        self,
        image: torch.Tensor,
        patch_bbox: Tuple[int, int, int, int],
        img_name: str,
        dist: float = 0.0,
        ax=None,
    ) -> None:
        """Visualize bounding box location on image."""
        if image.dim() == 4:
            image = image.squeeze(0)
        
        img_np = image.cpu().numpy()
        if img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        cmap = "gray" if img_np.shape[-1] == 1 else None
        
        x_start, y_end, x_end, y_start = patch_bbox
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        ax.imshow(img_np, cmap=cmap)
        
        patch_rect = mpatches.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            linewidth=3,
            edgecolor='red',
            facecolor='none',
            label=f'Patch with dist {dist:.2f}',
        )
        ax.add_patch(patch_rect)
        ax.set_title(f"Patch in {img_name}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
    
    def compute_anomaly_scores(self) -> Dict[str, Any]:
        """Compute various anomaly scores and statistics."""
        self.scores = {}
        self.scores_df["labels_gt"] = self.scores_df["labels_gt"].astype(int)
        baseline_f1 = self.compute_f1(self.scores_df["labels_gt"], self.scores_df["ndists"])
        baseline_auc = roc_auc_score(self.scores_df["labels_gt"], self.scores_df.ndists)
        self.scores["baseline"]["f1"] = baseline_f1
        self.scores["baseline"]["auc"] = baseline_auc

        ratio_scores = self.scores_df["ndists"]/(1e-10 +self.scores_df["andists"])
        self.scores["andists/ndists"]["f1"] = self.compute_f1(self.scores_df["labels_gt"], ratio_scores)
        self.scores["andists/ndists"]["auc"] = roc_auc_score(self.scores_df.labels_gt, ratio_scores)

        gap_scores = self.scores_df["ndists"] - self.scores_df["andists"]
        self.scores["andists/ndists"]["f1"] = self.compute_f1(self.scores_df["labels_gt"], gap_scores)
        self.scores["andists/ndists"]["auc"] = roc_auc_score(self.scores_df.labels_gt, gap_scores)

    @staticmethod
    def compute_f1(y_true, y_pred):
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        f1_scores = []
        for prec, rec in zip(precision, recall):
            f1_score = (2 * prec * rec) / (prec + rec + 1e-10)
            f1_scores.append(f1_score)
        return max(f1_scores)    
    
    def compute_distance_distributions(self) -> Dict[str, List[float]]:
        """Compute distance distributions."""
        
        dist_normal_to_normal = self.scores_df[self.scores_df['labels_gt'] == 0]['ndists']
        mean = dist_normal_to_normal.mean()

        plt.hist(dist_normal_to_normal, bins=50, color='blue', alpha=0.7)
        plt.title(f'Histogram of Pairwise Distances of normal neighbors to normal pixels. Mean: {mean}')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.csv_dir, "normal_to_normal_nn.png"))

        dist_normal_to_not_normal = self.scores_df[self.scores_df['labels_gt'] > 0.5]['ndists']
        mean = dist_normal_to_not_normal.mean()
        plt.hist(dist_normal_to_not_normal, bins=50, color='blue', alpha=0.7)
        plt.title(f'Histogram of Pairwise Distances of normal neighbors to anomalous pixels. Mean: {mean}')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.csv_dir, "anomaly_to_normal_nn.png"))

        dist_not_normal_to_normal = self.scores_df[self.scores_df['labels_gt'] == 0]['andists']
        mean = dist_not_normal_to_normal.mean()

        # Plot histogram
        plt.hist(dist_not_normal_to_normal, bins=30, color='blue', alpha=0.7)
        plt.title(f'Histogram of Pairwise Distances of not normal neighbors to normal pixels. Mean: {mean}')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.csv_dir, "normal_to_not_normal_nn.png"))

        dist_not_normal_to_not_normal = self.scores_df[self.scores_df['labels_gt'] ==1.0]['andists']
        mean = dist_not_normal_to_not_normal.mean()

        # Plot histogram
        plt.hist(dist_not_normal_to_not_normal, bins=50, color='blue', alpha=0.7)
        plt.title(f'Histogram of Pairwise Distances of not normal neighbors to not normal pixels. Mean: {mean}')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.csv_dir, "anomaly_to_not_normal_nn.png"))

    
    def save_results(self) -> None:
        """Save scores and distributions to JSON files."""
        # Save scores
        scores_path = self.csv_dir / 'scores.json'
        with open(scores_path, 'w') as f:
            json.dump(self.scores, f, indent=2)
        print(f"Saved scores to {scores_path}")
    
    
    def run_pipeline(self) -> None:
        """Execute full pipeline."""
        print("\n" + "="*60)
        print("PatchCore Anomaly Detection Pipeline")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Filter samples
        self.filter_anomalous_samples()
        self.filter_normal_samples()
        
        # Generate visualizations
        if len(self.anomalous_patches) > 0:
            print("\nGenerating anomalous patches PDF...")
            self.generate_pdf_visualizations(
                self.anomalous_patches,
                'anomalous_patches.pdf',
                max_samples=50
            )
        
        if len(self.normal_patches) > 0:
            print("\nGenerating normal patches PDF...")
            self.generate_pdf_visualizations(
                self.normal_patches,
                'normal_patches.pdf',
                max_samples=50
            )
        
        # Compute and save statistics
        print("\nComputing anomaly scores...")
        self.compute_anomaly_scores()
        self.save_results()
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PatchCore Anomaly Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--csv-dir',
        type=str,
        required=True,
        help='Directory containing CSV files'
    )
    parser.add_argument(
        '--original-width',
        type=int,
        default=1024,
        help='Original image width (default: 1024)'
    )
    parser.add_argument(
        '--original-height',
        type=int,
        default=1024,
        help='Original image height (default: 1024)'
    )
    parser.add_argument(
        '--crop-size',
        type=int,
        default=224,
        help='Crop size for preprocessing (default: 224)'
    )
    parser.add_argument(
        '--resize-size',
        type=int,
        default=256,
        help='Resize size for preprocessing (default: 256)'
    )
    parser.add_argument(
        '--num-features',
        type=int,
        default=28,
        help='Feature map size (default: 28)'
    )
    parser.add_argument(
        '--masks-dir',
        type=str,
        required=True,
        help='Directory where the masks files are'
    )
    parser.add_argument(
        '--class-name',
        type=str,
        required=True,
        help='Class/category name'
    )
    parser.add_argument(
        '--gt-threshold',
        type=float,
        default=0.7,
        help='Ground truth threshold for anomaly classification (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PatchCoreAnalyzer(
        csv_dir=args.csv_dir,
        original_width=args.original_width,
        original_height=args.original_height,
        crop_size=args.crop_size,
        resize_size=args.resize_size,
        num_features=args.num_features,
        masks_dir=args.masks_dir,
        class_name=args.class_name,
    )
    
    # Run pipeline
    analyzer.run_pipeline()


if __name__ == '__main__':
    main()
