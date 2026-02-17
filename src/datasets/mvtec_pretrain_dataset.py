from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from PIL import Image
import numpy as np

import os


class MvtecPretrainDataset(Dataset):
    def __init__(self, prompts: dict, 
                 model_name: str = None, transforms: T = None):
        self.prompts = prompts
        self.model_name = model_name if model_name is not None else "CompVis/stable-diffusion-v1-4"
        self.transforms = transforms if transforms is not None else T.Compose([
            T.ToTensor(),
        ])
        self.__load_data__()
        

    def __load_data__(self):
        self.images = []
        self.labels = []
        idx = 0
        print(self.prompts)
        for k in self.prompts.keys():
            class_dir = k
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    self.images.append(img_path)
                    self.labels.append(idx)
            idx += 1

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        return (image, label)
    
    def __calculate_mean_std(self):
        # Calculate mean and std of the dataset
        means = np.array([0.0, 0.0, 0.0])
        for img_path in self.images:
            img = Image.open(img_path).convert("RGB")
            image = np.array(img).astype(np.float32) / 255.0
            mean = np.mean(image, axis=(0, 1))
            means += mean.flatten()
            img.close()
        global_mean = means / len(self.images)
        variances = []
        for img_path in self.images:
            img = Image.open(img_path).convert("RGB")
            image = np.array(img).astype(np.float32) / 255.0
            var = np.mean((image - global_mean) ** 2, axis=(0, 1))
            variances.append(var)
            img.close()
        global_std = np.sqrt(np.mean(variances, axis=0))
        return global_mean, global_std
