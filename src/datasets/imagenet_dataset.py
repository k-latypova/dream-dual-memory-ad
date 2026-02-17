from src.datasets.ad_dataset import ADDataset
from torchvision import datasets
import os
import torch


class Imagenet100Dataset(ADDataset):
    def __init__(self, dst_name: str, data_dir: str, normal_class: int, train=True, transform=None):
        super().__init__(dst_name, data_dir, normal_class, train, transform)

    def get_dataset(self):
        data_dir = os.path.join(self.data_dir, self.dst_name)
        if self.dst_name == 'imagenet100':
            base_dst = datasets.ImageFolder(root=data_dir, transform=self.transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dst_name}")
        class_indices = [i for i, target in enumerate(base_dst.targets) if target == self.normal_class]
        subset = torch.utils.data.Subset(base_dst, class_indices)
        return subset
    

    