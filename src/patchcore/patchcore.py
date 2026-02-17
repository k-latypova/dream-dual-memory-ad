"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm


import src.patchcore.backbones
import src.patchcore.common
import src.patchcore.sampler
from sklearn.model_selection import KFold

LOGGER = logging.getLogger(__name__)


class FeatureToPatchMapper:
    """
    Maps feature map coordinates to original image coordinates.
    Handles the full preprocessing pipeline: resize -> center crop -> feature extraction
    """
    
    def __init__(self, 
                 original_size: int = 1024,
                 resize_size: int = 256, 
                 crop_size: int = 224,
                 feature_map_size: int = 28):
        """
        Args:
            original_size: Original image size (1024x1024)
            resize_size: Size after resize (256x256)
            crop_size: Size after center crop (224x224)
            feature_map_size: Feature map size after backbone (28x28)
        """
        self.original_size = original_size
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.feature_map_size = feature_map_size
        
        # Calculate transformations
        self.resize_ratio = resize_size / original_size  # 0.25
        
        # Center crop offset in resized space (256x256)
        self.crop_margin = (resize_size - crop_size) // 2  # 16 pixels
        
        # Stride in cropped image space (224x224)
        self.stride_in_crop = crop_size / feature_map_size  # 8.0 pixels
        
        # Stride in original image space (1024x1024)
        self.stride_in_original = self.stride_in_crop / self.resize_ratio  # 32.0 pixels
        
        # Crop offset in original image space
        self.crop_offset_in_original = self.crop_margin / self.resize_ratio  # 64.0 pixels
        
        # Receptive field size (patch size) in original image
        self.patch_size_in_original = int(np.ceil(self.stride_in_original))  # 32 pixels
        
        #self._print_info()
    
    def _print_info(self):
        """Print transformation information."""
        print("=" * 60)
        print("Feature to Patch Mapper Configuration")
        print("=" * 60)
        print(f"Original image size: {self.original_size}x{self.original_size}")
        print(f"After resize: {self.resize_size}x{self.resize_size}")
        print(f"After center crop: {self.crop_size}x{self.crop_size}")
        print(f"Feature map size: {self.feature_map_size}x{self.feature_map_size}")
        print("-" * 60)
        print(f"Crop margin (in 256x256 space): {self.crop_margin} pixels")
        print(f"Crop offset (in 1024x1024 space): {self.crop_offset_in_original:.1f} pixels")
        print(f"Stride (in 224x224 space): {self.stride_in_crop:.2f} pixels/feature")
        print(f"Stride (in 1024x1024 space): {self.stride_in_original:.2f} pixels/feature")
        print(f"Patch size in original image: {self.patch_size_in_original}x{self.patch_size_in_original} pixels")
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
        # Calculate center of receptive field in cropped image
        center_y = (feature_row + 0.5) * self.stride_in_crop
        center_x = (feature_col + 0.5) * self.stride_in_crop
        
        # Calculate bounding box
        half_size = self.stride_in_crop / 2
        
        y_start = int(center_y - half_size)
        y_end = int(center_y + half_size)
        x_start = int(center_x - half_size)
        x_end = int(center_x + half_size)
        
        # Clip to crop bounds
        y_start = max(0, y_start)
        y_end = min(self.crop_size, y_end)
        x_start = max(0, x_start)
        x_end = min(self.crop_size, x_end)
        
        return y_start, y_end, x_start, x_end
    
    def feature_coords_to_original_bbox(self, feature_row: int, feature_col: int) -> Tuple[int, int, int, int]:
        """
        Convert feature map coordinates to bounding box in original 1024x1024 image.
        
        Args:
            feature_row: Row index in feature map [0, 27]
            feature_col: Column index in feature map [0, 27]
            
        Returns:
            (y_start, y_end, x_start, x_end) in original 1024x1024 image coordinates
        """
        # Calculate center in cropped space (224x224)
        center_y_crop = (feature_row + 0.5) * self.stride_in_crop
        center_x_crop = (feature_col + 0.5) * self.stride_in_crop
        
        # Map to original space (1024x1024)
        # Account for: 1) center crop offset, 2) resize ratio
        center_y_original = (center_y_crop + self.crop_margin) / self.resize_ratio
        center_x_original = (center_x_crop + self.crop_margin) / self.resize_ratio
        
        # Calculate bounding box in original space
        half_size = self.stride_in_original / 2
        
        y_start = int(center_y_original - half_size)
        y_end = int(center_y_original + half_size)
        x_start = int(center_x_original - half_size)
        x_end = int(center_x_original + half_size)
        
        # Clip to original image bounds
        y_start = max(0, y_start)
        y_end = min(self.original_size, y_end)
        x_start = max(0, x_start)
        x_end = min(self.original_size, x_end)
        
        return x_start, y_end, x_end, y_start
    
    def patch_index_to_original_bbox(self, patch_idx: int) -> Tuple[int, int, int, int]:
        """
        Convert linear patch index directly to bounding box in original 1024x1024 image.
        
        Args:
            patch_idx: Linear index [0, 784)
            
        Returns:
            (x_start, y_end, x_end, y_start) in original 1024x1024 image coordinates
        """
        row, col = self.patch_index_to_coordinates(patch_idx)
        return self.feature_coords_to_original_bbox(row, col)
    
    def extract_patch_from_original_image(self, image: torch.Tensor, 
                                          patch_idx: int) -> torch.Tensor:
        """
        Extract the image region from ORIGINAL 1024x1024 image corresponding to a patch index.
        
        Args:
            image: Original image tensor [C, 1024, 1024] or [1, C, 1024, 1024]
            patch_idx: Patch index in feature map
            
        Returns:
            Patch from original image [C, ~32, ~32]
        """
        if image.dim() == 4:
            image = image.squeeze(0)
        
        x_start, y_end, x_end, y_start = self.patch_index_to_original_bbox(patch_idx)
        patch = image[:, y_start:y_end, x_start:x_end]
        
        return patch
    
    def extract_patch_from_preprocessed_image(self, image: torch.Tensor, 
                                              patch_idx: int) -> torch.Tensor:
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


VIT_BACKBONES = ['vit_base', 'vit_large', 'vit_base_patch8', 'vit_large_dino', 'vit_medium_relpos', 'fn_vit']
    



class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=src.patchcore.sampler.IdentitySampler(),
        anomalous_featuresampler=src.patchcore.sampler.IdentitySampler(),
        nn_method=src.patchcore.common.FaissNN(False, 4),
        not_normal_nn_method=src.patchcore.common.FaissNN(False, 4),
        alpha=0.0,
        score="mean",
        anomalies_cutoff=0.75,
        anomaly_score_fn='ratio',
        
        anomalies_filter_func="filter_anomaly_patches_by_percentile",
        anomalies_top_threshold = 90.00,
        anomalies_bottom_threshold = 50.00,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = src.patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = src.patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = src.patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = src.patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, 
            normal_nn_method=nn_method,
            abnormal_nn_method=not_normal_nn_method, 
            alpha=alpha, 
            score=score,
            anomaly_score_fn=anomaly_score_fn
        )

        self.anomaly_segmentor = src.patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        self.anomalous_featuresampler = anomalous_featuresampler

        self.feature_mapper = FeatureToPatchMapper(feature_map_size=37 if self.backbone.name in VIT_BACKBONES else 28)

        self.anomalies_cutoff = anomalies_cutoff
        #self.anomalies_filter_func = globals()[anomalies_filter_func]
        self.anomalies_top_percentile = anomalies_top_threshold
        self.anomalies_bottom_threshold = anomalies_bottom_threshold



    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

        
    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        if self.backbone.name in VIT_BACKBONES:
        # ViT-specific: Remove CLS token and reshape
            processed_features = []
            patch_shapes = []
            for x in features:
                features_start = 0 if 'relpos' in self.backbone.name else 1
                patch_tokens = x[:, features_start:, :]  # Remove CLS
                B, num_patches, D = patch_tokens.shape
                H = W = int(num_patches ** 0.5)
                x_grid = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
                processed_features.append(x_grid)
                patch_shapes.append((H, W))
            features = processed_features
        elif self.backbone.name.startswith('vit_swin'):
            patch_shapes = []
            features = [feat.permute(0, 3, 1, 2) for feat in features]
            for i in range(len(features)):
                x = features[i]
                spatial_shape = x.shape[-2:]
                patch_shapes.append(spatial_shape)
        else:
            patch_shapes = []
            for i in range(len(features)):
                x = features[i]
                spatial_shape = x.shape[-2:]
                patch_shapes.append(spatial_shape)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)
    

    def _embed_anomalies(self, images, masks, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        mask_threshold = 0.8
        
        if self.backbone.name in VIT_BACKBONES:
        # ViT-specific: Remove CLS token and reshape
            processed_features = []
            patch_shapes = []
            for x in features:
                features_start = 0 if 'relpos' in self.backbone.name else 1
                patch_tokens = x[:, features_start:, :]  # Remove CLS
                B, num_patches, D = patch_tokens.shape
                H = W = int(num_patches ** 0.5)
                x_grid = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
                processed_features.append(x_grid)
                patch_shapes.append((H, W))
            features = processed_features
        elif self.backbone.name.startswith('vit_swin'):
            patch_shapes = []
            features = [feat.permute(0, 3, 1, 2) for feat in features]
            for i in range(len(features)):
                x = features[i]
                spatial_shape = x.shape[-2:]
                patch_shapes.append(spatial_shape)
        else:
            patch_shapes = []
            for i in range(len(features)):
                x = features[i]
                spatial_shape = x.shape[-2:]
                patch_shapes.append(spatial_shape)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]


        masks_reshaped = F.interpolate(masks, size=(ref_num_patches[0], ref_num_patches[1]),
                                           mode="bicubic")
        
        masks_reshaped_flat = masks_reshaped.reshape(-1, 1)
        anomaly_mask: torch.Tensor = masks_reshaped_flat[:, 0] > mask_threshold  # [B, num_patches]    

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )

            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])

            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        #features = [features for features in]
        
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        if anomaly_mask.any():
            features = features[anomaly_mask]
        else:
            features = features[:0]  # Preserve shape with zero elements
        patch_indices = []
        for img_mask in masks_reshaped:
            anomaly_mask = img_mask.reshape(-1,) > mask_threshold
            patch_indices.append(anomaly_mask.nonzero())

        

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features), patch_indices

    def fit(self, training_data, synth_anomalies):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data, synth_anomalies)

    def _get_normal_scores(self, normal_features) -> float:
        kfold = KFold(5)
        all_scores = []
        for train_idx, test_idx in  kfold.split(normal_features):
            train_x = normal_features[train_idx]
            test_x = normal_features[test_idx]
            self.anomaly_scorer.reset()
            self.anomaly_scorer.fit_normal([train_x.astype(np.float32)])
            scores = self.anomaly_scorer.predict_against_normal([test_x.astype(np.float32)])
            all_scores.append(scores.reshape(-1, 1))
        all_scores = np.stack(scores)
        return all_scores



    def _fill_memory_bank(self, normal_input_data, not_normal_input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()
        self._normal_image_paths = []
        self._not_normal_image_paths = []

        def _image_to_features(input_image, mask=None):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)
                    return self._embed_anomalies(input_image, masks=mask)
                else:
                    return self._embed(input_image)
                
        not_normal_features = []
        with tqdm.tqdm(
            not_normal_input_data, desc="Computing abnormal support features...", position=1, leave=False
        ) as data_iterator:
            for idx, image in enumerate(data_iterator):
                if self.anomaly_scorer.alpha == 0.0 and idx > 4:
                    print("Skipping remaining synthetic anomalies as alpha=0.0")
                    break
                input_img = image["image"]
                mask = image["mask"]
                img_path = image["image_path"]
                
                embeds, patch_indices = _image_to_features(input_img, mask=mask)
                not_normal_features.append(embeds)
                img_patch_map = [(img, patch_idx.item()) for img, img_patches in zip(img_path, patch_indices) for patch_idx in img_patches]
                self._not_normal_image_paths.extend(img_patch_map)
                
        features = []
        with tqdm.tqdm(
            normal_input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for item in data_iterator:
                if isinstance(item, dict):
                    image = item["image"]
                    img_path = item["image_path"]
                embeds = _image_to_features(image)
                for img in img_path:
                    self._normal_image_paths.extend([(img, x) for x in range(0, len(embeds) // len(image))])
                features.append(embeds)

        features = np.concatenate(features, axis=0)
        features, sample_indices = self.featuresampler.run(features)
        self._normal_image_paths = [self._normal_image_paths[x] for x in sample_indices]
        normal_scores = self._get_normal_scores(features)
        self.normal_scores = normal_scores
        max_normal_score = np.max(normal_scores)
        print(f"Max anomaly score: {max_normal_score}")
        self.anomaly_scorer.fit_normal([features])

        try:
            not_normal_features = np.concatenate([x for x in not_normal_features if x != []], axis=0)
        except Exception as e:
            print(f"The features at index 70: {not_normal_features[70].shape}")
            raise e
        print(f"Not normal features shape before sampling: {not_normal_features.shape[0]}")

        anomaly_scores = self.anomaly_scorer.predict_against_normal([not_normal_features])
        self.anomaly_scores = anomaly_scores

        print(f"Mean of anomaly scores of not normal features: {anomaly_scores.mean()}, \nMin anomaly score of not normal: {np.min(anomaly_scores)}")


        # # Get indices of top highest scoring patches
        # top_k = int(len(anomaly_scores) * self.anomalies_cutoff)
        # top_indices = np.argsort(anomaly_scores)[-top_k:]


        # #Filter anomalous features by percentile
        #score_mask, selected_indices = self.filter_anomaly_patches_by_percentile(anomaly_scores)
        #score_mask, selected_indices = self.filter_anomaly_patches_combined(anomaly_scores)
        score_mask, selected_indices = self.filter_anomaly_patches_by_percentile(anomaly_scores, self.anomalies_bottom_threshold, self.anomalies_top_percentile)
        filtered_anomalous_features = not_normal_features[score_mask]
        self._not_normal_image_paths = [self._not_normal_image_paths[x] for x in selected_indices]

        # Filter by normal max score
        # filtered_anomalous_features = not_normal_features[anomaly_scores > max_normal_score]
        # anomalous_indices = np.argwhere(anomaly_scores > max_normal_score).reshape(-1)
        # self._not_normal_image_paths = [self._not_normal_image_paths[x] for x in anomalous_indices]
        # print(f"Initial anomalous feature num: {len(anomaly_scores)}, filtered to {len(filtered_anomalous_features)}")
        
        self.anomaly_scorer.fit_not_normal([filtered_anomalous_features])


    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        ndists_ = []
        andists_ = []
        neighbors = []
        print(f"Dataloader length: {len(dataloader)} ")
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    names = image["image_path"]
                    image = image["image"]
                _scores, _masks, ndists, andists, neighbors_info = self._predict(image)
                num_patches = len(neighbors_info) // len(names)
                i = 0
                for score, mask, ndist, andist, name in zip(_scores, _masks, ndists, andists, names):
                    scores.append(score)
                    masks.append(mask)
                    ndists_.append(ndist)
                    andists_.append(andist)
                    for j in range(num_patches):
                        neighbors.append(neighbors_info[i] + (name,))
                        i += 1
        return scores, masks, labels_gt, masks_gt, ndists_, andists_, neighbors

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            image_scores, ndistances, normal_neighbors, not_ndistances, not_normal_neighbors = self.anomaly_scorer.predict([features])
            normal_neighbors = normal_neighbors.reshape(-1, 1).tolist()
            not_normal_neighbors = not_normal_neighbors.reshape(-1, 1).tolist()
            neighbors_info = self.process_nearest_neighbours(normal_neighbors, not_normal_neighbors, patch_shapes[0][0])
            neighbors_info = [neighbor + (ndist.item(),) + (notndist.item(), ) for neighbor, ndist, notndist in zip(neighbors_info, ndistances, not_ndistances)]
            ndistances = ndistances.reshape(len(images), -1)
            not_ndistances = not_ndistances.reshape(len(images), -1)
            
            patch_scores = image_scores
            patch_ndists = ndistances
            patch_not_ndists = not_ndistances
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            patch_ndists = self.patch_maker.unpatch_scores(
                patch_ndists, batchsize=batchsize
            )
            patch_ndists = patch_ndists.reshape(batchsize, scales[0], scales[1])

            patch_not_ndists = self.patch_maker.unpatch_scores(
                patch_not_ndists, batchsize=batchsize)
            patch_not_ndists = patch_not_ndists.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
            #print(f"Patch scores shape: {patch_scores.shape}, masks shape: {masks[0][0].shape}")

            not_ndists_masks = self.anomaly_segmentor.convert_to_segmentation(patch_not_ndists)
            ndists_masks = self.anomaly_segmentor.convert_to_segmentation(patch_ndists)

        return [score for score in image_scores], [mask for mask in masks], [ndistances for ndistances in ndists_masks], \
            [not_ndistances for not_ndistances in not_ndists_masks], neighbors_info

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method=None,
        prepend: str = "",
    ) -> None:
        if nn_method is None:
            nn_method =  src.patchcore.common.FaissNN(False, 4)
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = src.patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)

    


    def process_nearest_neighbours(self, normal_neighbors: List[int], not_normal_neighbors: List[int], 
                                   feature_size: int) -> List[Tuple[int, int, int, int]]:
        neighbors_info = []
        for normal_idx, not_normal_idx in zip(normal_neighbors, not_normal_neighbors):
            normal_img_path, normal_patch_idx = self._normal_image_paths[normal_idx[0]]
            not_normal_img_path, not_normal_patch_idx = self._not_normal_image_paths[not_normal_idx[0]]
            neighbors_info.append((normal_img_path, normal_patch_idx, not_normal_img_path, not_normal_patch_idx))
        return neighbors_info
    

    def filter_anomaly_patches_by_percentile(self, anomaly_scores, lower_percentile=20, upper_percentile=80, **kwargs):
        """
        Select anomaly patches in the middle score range, excluding very easy and very hard cases.
        
        Args:
            anomaly_scores: numpy array of anomaly scores for anomalous patches
            lower_percentile: exclude scores below this percentile
            upper_percentile: exclude scores above this percentile
        
        Returns:
            mask: boolean mask of selected patches
            selected_scores: the scores of selected patches
        """
        lower_threshold = np.percentile(anomaly_scores, lower_percentile)
        upper_threshold = np.percentile(anomaly_scores, upper_percentile)
        
        mask = (anomaly_scores >= lower_threshold) & (anomaly_scores <= upper_threshold)
        selected_scores = anomaly_scores[mask]
        selected_indices = np.argwhere(mask).reshape(-1)

        
        print(f"Selected {mask.sum()} / {len(anomaly_scores)} patches")
        print(f"Score range: [{lower_threshold:.2f}, {upper_threshold:.2f}]")

        
        return mask, selected_indices
    

    def filter_anomaly_patches_combined(self, anomaly_scores, 
                                    use_iqr=True, use_percentile=True,
                                    iqr_mult=1.5, upper_percentile=85, lower_percentile=15, *kwargs):
        """
        Combine IQR filtering + percentile filtering for robust selection.
        """
        mask = np.ones(len(anomaly_scores), dtype=bool)
        
        # Step 1: IQR filtering
        if use_iqr:
            Q1 = np.percentile(anomaly_scores, 25)
            Q3 = np.percentile(anomaly_scores, 75)
            IQR = Q3 - Q1
            lower_iqr = Q1 - iqr_mult * IQR
            upper_iqr = Q3 + iqr_mult * IQR
            mask &= (anomaly_scores >= lower_iqr) & (anomaly_scores <= upper_iqr)
        
        # Step 2: Percentile filtering
        if use_percentile:
            lower_p = np.percentile(anomaly_scores, lower_percentile)
            upper_p = np.percentile(anomaly_scores, upper_percentile)
            mask &= (anomaly_scores >= lower_p) & (anomaly_scores <= upper_p)
        
        print(f"Final selection: {mask.sum()} / {len(anomaly_scores)} patches ({100*mask.sum()/len(anomaly_scores):.1f}%)")
        
        indices = np.argwhere(mask).reshape(-1)
        return mask, indices
    
    def filter_anomaly_patches_adaptive(self, anomaly_scores, normal_scores=None, gap_percentile=10):
        """
        Use the separation between normal and anomaly distributions to filter.
        
        Args:
            anomaly_scores: numpy array of anomaly scores for anomalous patches
            normal_scores: numpy array of anomaly scores for normal patches (optional)
            gap_percentile: how far into the anomaly distribution to start filtering
        
        Returns:
            mask: boolean mask
        """
        # Start from a percentile slightly above typical normals
        if normal_scores is not None:
            lower_threshold = np.percentile(normal_scores, 95)  # 95th percentile of normals
        else:
            lower_threshold = np.percentile(anomaly_scores, gap_percentile)
        
        # Exclude very high anomaly scores (too extreme/noisy)
        upper_threshold = np.percentile(anomaly_scores, 90)
        
        mask = (anomaly_scores >= lower_threshold) & (anomaly_scores <= upper_threshold)
        
        print(f"Selected {mask.sum()} / {len(anomaly_scores)} patches")
        print(f"Threshold range: [{lower_threshold:.2f}, {upper_threshold:.2f}]")

        indices = np.argwhere(mask).reshape(-1)
        
        return mask, indices
    

    def validate(self, validation_data, num_stds:float):
        """Validates the model on validation data."""
        scores, masks, labels_gt, masks_gt, _, _, _ = self._predict_dataloader(
            validation_data
        )
        masks = np.array(masks)
        threshold = masks.mean() + num_stds * masks.std()
        return threshold
        


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.p
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x