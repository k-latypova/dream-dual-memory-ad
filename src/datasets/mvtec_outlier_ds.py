from torch.utils.data import Dataset
import pandas as pd
import csv
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import random

def get_mask_path(mask, outliers_dir):
    if os.path.exists(mask):
        return mask
    else:
        print(f"Mask path {mask} doesn't exist")
        return os.path.join(outliers_dir, "masks", f"mask_{int(mask):02d}.png")

class MvtecOutlierDataset(Dataset):
    def __init__(self, outliers_dir, metadata_file=None, category="hazelnut", transform=None, mask_transform=None, use_full=False, 
                 ground_truth=False, mini=False,
                 apply_augmentations=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            outliers_dir (string): Directory with outlier images.
            metadata_file (string): Path to the metadata file.
            category (string): Category of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.outliers_dir = outliers_dir
        if metadata_file is None:
            metadata_file = os.path.join(outliers_dir, "metadata.csv")
        self.metadata_file = metadata_file
        self.category = category
        self.transform = transform
        self.mask_transform = mask_transform
        self.use_full = use_full
        self.mini = mini
        self.apply_augmentations = apply_augmentations
        self.__load_data()

    def __load_data(self):
        # Implement logic to load image paths and labels from metadata_file
        with open(self.metadata_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            self.metadata = list(reader)
        try:
            # self.metadata = [x for x in self.metadata if x['train'] == 'True']
            self.image_paths = [{"image_path": os.path.join(self.outliers_dir, "samples", f"sample_{int(x['sample_idx']):05d}.png"),
                             "mask_path": get_mask_path(x['mask_idx'], self.outliers_dir),
                             "image_gt": x["normal"] if "normal" in x else None}
                            for x in self.metadata[1:]]
        except Exception as e:
            self.image_paths = [{"image_path": os.path.join(self.outliers_dir, "samples", x['sample_idx']),
                             "mask_path": get_mask_path(x['mask_idx'], outliers_dir=self.outliers_dir),
                             "image_gt": x["normal"] if "normal" in x else None}
                            for x in self.metadata[1:]]
        if self.use_full:
            for i in range(len(self.image_paths)):
                self.image_paths[i]['image_path'] = self.image_paths[i]['image_path'].replace("sample_", "full_image_")
        self.data = []

        if self.mini:
            self.image_paths = self.image_paths[:5]

        for i in range(len(self.image_paths)):
            if not os.path.exists(self.image_paths[i]['image_path']):
                continue
            else:
                self.data.append(self.image_paths[i])
                

    def __len__(self):
        return len(self.image_paths)

    def random_augmentation(self, image, mask, image_gt):
        # Random Horizontal Flip
        # if random.random() > 0.5:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)
        #     image_gt = TF.hflip(image_gt)

        # # Random Rotation (-30 to 30 degrees)
        # if random.random() > 0.5:
        #     angle = random.uniform(-30, 30)
        #     image = TF.rotate(image, angle)
        #     mask = TF.rotate(mask, angle)
        #     image_gt = TF.rotate(image_gt, angle)

        # Random Affine Transformation
        if random.random() > 0.5:
            # Parameters for affine: angle, translate, scale, shear
            angle = random.uniform(-15, 15)
            translate = (random.uniform(-5, 5), random.uniform(-5, 5))
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-10, 10)
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
            mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear)
            image_gt = TF.affine(image_gt, angle=angle, translate=translate, scale=scale, shear=shear)


        if random.random() > 0.5:
            brightness_factor = random.uniform(0.7, 1.3)
            image = TF.adjust_brightness(image, brightness_factor)

        # Random Exposure Adjustment (Contrast)
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.6, 1.7)
            image = TF.adjust_contrast(image, contrast_factor)

        return image, mask, image_gt

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        mask_path = self.data[idx]['mask_path']
        image_gt_path = self.data[idx]['image_gt']

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_gt = Image.open(image_gt_path).convert("RGB") if image_gt_path else image

        # Apply random augmentations to both image and mask
        if self.apply_augmentations:
            image, mask, image_gt = self.random_augmentation(image, mask, image_gt)

        # Apply transforms after augmentation (e.g. tensor conversion, normalization)
        if self.transform:
            image = self.transform(image)
            image_gt = self.transform(image_gt)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        sample = {'image': image, 'mask': mask, 'gt_image': image_gt, "classname": self.category, "anomaly": "anomaly", 
                  "is_anomaly": True, "image_name": os.path.basename(img_path), "image_path": img_path}

        return sample

