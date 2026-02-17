from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Callable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
import torch
import numpy as np
import cv2


class MVTecDataset(Dataset):
    def __init__(self, root, class_name: str, split: str, transform: Callable = None,
                 ground_truth_transform: Callable = None):
        """
        Initializes the MVTec dataset.

        Args:
            root (str): Root directory of the dataset.
            train (bool): If True, load training data; if False, load test data.
            transform: Transformations to apply to the images.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.ground_truth_transform = ground_truth_transform
        self.class_name = class_name.lower()
        self.data = self._load_data()

    

    def _load_data(self):
        """ Loads the dataset based on the specified split.

        Returns:
            List of tuples containing image paths and their corresponding labels.
        """
        if self.split.lower() == 'train':
            return self._load_train_data()
        elif self.split.lower() == 'test':
            return self._load_test_data()
        else:
            raise ValueError("Split must be either 'train' or 'test'.")

    def _load_train_data(self):
        """ Loads the dataset from the specified root directory.

        Raises:
            ValueError: If the split is not 'train' or 'test'.

        Returns:
            _type_: List of tuples containing image paths and their corresponding labels.
        """
        split = self.split.lower()
        dataset_root = os.path.join(self.root, self.class_name, split)

        if split not in ['train', 'test']:
            raise ValueError("Split must be either 'train' or 'test'.")
        data = []
        self.label_names = os.listdir(dataset_root)
        for label_idx, label in enumerate(os.listdir(dataset_root)):
            label_path = os.path.join(dataset_root, label)

            for filename in os.listdir(label_path):
                if filename.endswith('.png'):
                    image_path = os.path.join(label_path, filename)
                    data.append((image_path, label_idx))
        
        return data
    
    def idx2label(self, idx: int) -> str:
        """ Converts an index to its corresponding label name.

        Args:
            idx (int): Index of the label.

        Returns:
            str: Name of the label.
        """
        if 0 <= idx < len(self.label_names):
            return self.label_names[idx]
        else:
            raise IndexError("Index out of range for label names.")
    
    def __len__(self):
        """ Returns the number of samples in the dataset. """
        return len(self.data)
    
    def __get_trainitem__(self, idx):
        """ Returns the image and its label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a tensor and label is an integer.
        """
        image_path, label = self.data[idx]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.Tensor(label)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            return self.__get_trainitem__(idx)
        elif self.split == 'test':
            return self.__get_testitem__(idx)
        
    def __get_testitem__(self, idx):
        """ Returns the image and its label at the specified index for test data.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a tensor and label is an integer.
        """
        image_path, label, ground_truth = self.data[idx]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        image_size = image.size

        if self.transform:
            image = self.transform(image)

        if ground_truth is None:
            ground_truth = Image.fromarray(np.zeros((image_size[1], image_size[0]), dtype=np.uint8))
        else:
            with open(ground_truth, 'rb') as f:
                ground_truth = Image.open(f).convert('L')
        if self.ground_truth_transform:
            ground_truth = self.ground_truth_transform(ground_truth)

        return image, label, ground_truth

    def _load_test_data(self):
        """ Loads the test data for the MVTec dataset.

        Returns:
            List of tuples containing image paths and their corresponding labels.
        """
        labels = os.listdir(os.path.join(self.root, self.class_name, 'test'))
        defect_labels = [x for x in labels if x != 'good']
        data = []
        self.label_names = ['good'] + defect_labels
        for filename in os.listdir(os.path.join(self.root, self.class_name, 'test', 'good')):
            image_path = os.path.join(self.root, self.class_name, 'test', 'good', filename)
            label = 0
            data.append((image_path, label, None))
        for label_idx, defect_label in enumerate(defect_labels):
            defect_images_path = os.path.join(self.root, self.class_name, 'test', defect_label)
            ground_truth_path = os.path.join(self.root, self.class_name, 'ground_truth', defect_label)
            for filename in os.listdir(defect_images_path):
                image_path = os.path.join(defect_images_path, filename)
                label = label_idx + 1
                mask_filename = filename.replace('.png', '_mask.png')
                ground_truth = os.path.join(ground_truth_path, mask_filename)
                
                if not os.path.exists(ground_truth):
                    ground_truth = None
                data.append((image_path, 1, ground_truth))
        return data

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask


if __name__ == "__main__":
    ds = MVTecDataset(
        root='data/mvtec',
        class_name='hazelnut',
        split='test',
        transform=T.ToTensor(),
        ground_truth_transform=T.ToTensor()
    )

    dataloader = DataLoader(ds, batch_size=4)
    next(iter(dataloader))

