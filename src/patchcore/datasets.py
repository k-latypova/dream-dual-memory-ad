import os
from enum import Enum

import PIL
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from src.patchcore.transforms_utils import (
    ResizeLongestSide,
    PadToSquareTensor,
    get_transform_metadata,
)

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

_MVTEC2_CLASSNAMES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "wallnuts",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

resolution_dict = {
    "sheet_metal": (4224, 1056),
    "vial": (1400, 1900),
    "wallplugs": (2448, 2048),
    "walnuts": (2448, 2048),
    "can": (2232, 1024),
    "fabric": (2448, 2048),
    "fruit_jelly": (2100, 1520),
    "rice": (2448, 2048),
    "hazelnut": (1024, 1024),
    "screw": (1024, 1024),
    "pill": (800, 800),
    "cable": (1024, 1024),
    "capsule": (1000, 1000),
    "carpet": (1024, 1024),
    "toothbrush": (1024, 1024),
    "wood": (1024, 1024),
    "tile": (840, 840),
    "transistor": (1024, 1024),
    "zipper": (1024, 1024),
    "leather": (1024, 1024),
    "grid": (1024, 1024),
    "metal_nut": (700, 700),
    "bottle": (900, 900),
}


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TRAINVAL = "trainval"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        mini=False,
        do_padding=False,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.mini = mini

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.orig_size = resolution_dict[classname]

        if do_padding:
            self.padding_metadata = get_transform_metadata(
                self.orig_size, target_size=imagesize
            )
            self.transform_img = [
                ResizeLongestSide(target_size=imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                PadToSquareTensor(target_size=imagesize),
            ]

            self.transform_mask = [
                ResizeLongestSide(target_size=imagesize),
                transforms.ToTensor(),
                PadToSquareTensor(target_size=imagesize),
            ]
        else:
            self.padding_metadata = None
            self.padding_metadata = {
                "original_size": self.orig_size,
                "resized_size": (resize, resize),
                "final_size": (imagesize, imagesize),
                "padding": (0, 0, 0, 0),
                "aspect_ratio": 1.0,
            }
            self.transform_img = [
                transforms.Resize((resize, resize)),
                transforms.CenterCrop((imagesize, imagesize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]

            self.transform_mask = [
                transforms.Resize((resize, resize)),
                transforms.CenterCrop((imagesize, imagesize)),
                transforms.ToTensor(),
            ]

        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_mask = transforms.Compose(self.transform_mask)
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD

        self.imagesize = (3, imagesize, imagesize)

    def random_augmentation(self, image, mask):
        # Random Horizontal Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random Rotation (-30 to 30 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Random Affine Transformation
        # if random.random() > 0.5:
        #     # Parameters for affine: angle, translate, scale, shear
        #     angle = random.uniform(-15, 15)
        #     translate = (random.uniform(-5, 5), random.uniform(-5, 5))
        #     scale = random.uniform(0.9, 1.1)
        #     shear = random.uniform(-10, 10)
        #     image = TF.affine(
        #         image, angle=angle, translate=translate, scale=scale, shear=shear
        #     )
        #     mask = TF.affine(
        #         mask, angle=angle, translate=translate, scale=scale, shear=shear
        #     )

        return image, mask

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        padding_metadata = get_transform_metadata(
            self.orig_size, target_size=self.imagesize[1]
        )

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
            "padding_metadata": padding_metadata,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        select_data_to_iterate = data_to_iterate[:2] if self.mini else data_to_iterate
        return imgpaths_per_class, select_data_to_iterate


class MVTec2Dataset(MVTecDataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__(
            source, classname, resize, imagesize, split, train_val_split, **kwargs
        )
        

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            if (
                self.split == DatasetSplit.TRAIN
                or self.split == DatasetSplit.VAL
            ):
                split = (
                    "train" if self.split == DatasetSplit.TRAIN else "validation"
                )
                classpath = os.path.join(self.source, classname, split, "good")
                imgpaths_per_class[classname]["good"] = [
                    os.path.join(classpath, x) for x in sorted(os.listdir(classpath))
                ]
                maskpaths_per_class[classname]["good"] = None
            elif self.split == DatasetSplit.TRAINVAL:
                classpath = os.path.join(self.source, classname, "train", "good")
                val_classpath = os.path.join(
                    self.source, classname, "validation", "good"
                )
                imgpaths_per_class[classname]["good"] = [
                    os.path.join(classpath, x) for x in sorted(os.listdir(classpath))
                ] + [
                    os.path.join(classpath, x)
                    for x in sorted(os.listdir(val_classpath))
                ]
                maskpaths_per_class[classname]["good"] = None

            else:
                classpath = os.path.join(self.source, classname, "test_public")
                imgpaths_per_class[classname]["good"] = [
                    os.path.join(classpath, "good", x)
                    for x in os.listdir(os.path.join(classpath, "good"))
                ]
                imgpaths_per_class[classname]["bad"] = [
                    os.path.join(classpath, "bad", x)
                    for x in os.listdir(os.path.join(classpath, "bad"))
                ]
                maskpaths_per_class[classname]["good"] = None
                maskpaths_per_class[classname]["bad"] = [
                    x[:-4] + "_mask.png" for x in imgpaths_per_class[classname]["bad"]
                ]
                maskpaths_per_class[classname]["bad"] = [
                    os.path.join(classpath, "ground_truth", "bad", x[:-4] + "_mask.png")
                    for x in os.listdir(os.path.join(classpath, "bad"))
                ]

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


class MVTecDataset_(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        anomaly_types=[],
        mini=False,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.anomaly_types = anomaly_types
        print(f"Using anomaly types: {self.anomaly_types}")
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.mini = mini

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.test_transform = [
            transforms.Resize(imagesize, imagesize),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.train_transform = [
            transforms.Resize(resize),
            transforms.RandomCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.test_transform = transforms.Compose(self.test_transform)

        self.transform_mask = [
            transforms.Resize(imagesize, imagesize),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.test_transform(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = self.anomaly_types

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                select_anomaly_files = anomaly_files[:3] if self.mini else anomaly_files
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in select_anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    select_anomaly_mask_files = (
                        anomaly_mask_files[:3] if self.mini else anomaly_mask_files
                    )
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x)
                        for x in select_anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


class PrivateMVTec2Dataset(torch.utils.data.Dataset):
    def __init__(
        self, source, classname, resize=256, imagesize=224, do_padding=False, **kwargs
    ):
        self.source = source
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.orig_size = resolution_dict[classname]
        if do_padding:
            self.padding_metadata = get_transform_metadata(
                self.orig_size, target_size=imagesize
            )
            self.transform_img = [
                ResizeLongestSide(target_size=imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                PadToSquareTensor(target_size=imagesize),
            ]

            self.transform_mask = [
                ResizeLongestSide(target_size=imagesize),
                transforms.ToTensor(),
                PadToSquareTensor(target_size=imagesize),
            ]
        else:
            self.padding_metadata = None
            self.padding_metadata = {
                "original_size": self.orig_size,
                "resized_size": (resize, resize),
                "final_size": (imagesize, imagesize),
                "padding": (0, 0, 0, 0),
                "aspect_ratio": 1.0,
            }
            self.transform_img = [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]

            self.transform_mask = [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
            ]

        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_mask = transforms.Compose(self.transform_mask)


        
    def get_image_data(self):
        imgpaths_per_class = {}
        data_to_iterate = []
        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, "test_private")
            classpath_mixed = os.path.join(self.source, classname, "test_private_mixed")

            imgpaths_per_class[classname] = {}

            anomaly_files = [
                os.path.join(classpath, fn) for fn in sorted(os.listdir(classpath))
            ]
            mixed_anomaly_files = [
                os.path.join(classpath_mixed, fn)
                for fn in sorted(os.listdir(classpath_mixed))
            ]
            imgpaths_per_class[classname] = anomaly_files + mixed_anomaly_files
        for classname in self.classnames_to_use:
            for image_path in imgpaths_per_class[classname]:
                data_tuple = [classname, image_path]
                data_to_iterate.append(data_tuple)
        return imgpaths_per_class, data_to_iterate

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        classname, image_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        return {
            "image": image,
            "mask": torch.zeros([1, *image.size()[1:]]),
            "anomaly": "unknown",
            "is_anomaly": -1,
            "classname": classname,
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }
