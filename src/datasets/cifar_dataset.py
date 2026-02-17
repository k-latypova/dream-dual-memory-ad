from torchvision import datasets
from src.datasets.ad_dataset import ADDataset
import os
import torch
from torchvision.transforms import transforms as T

class CifarDataset(ADDataset):
    def __init__(self, dst_name: str, data_dir: str, normal_class: int, train=True, transform=None):
        super().__init__(dst_name, data_dir, normal_class, train, transform)
        

    def get_dataset(self):
        data_dir = os.path.join(self.data_dir, self.dst_name)
        if self.dst_name == 'cifar10':
            base_dst = datasets.CIFAR10(root=data_dir, train=self.train, download=True, transform=self.transform)
        elif self.dst_name == 'cifar100':
            base_dst = datasets.CIFAR100(root=data_dir, train=self.train, download=True, transform=self.transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dst_name}")
        if self.train:
            class_indices = [i for i, target in enumerate(base_dst.targets) if target == self.normal_class]
            subset = torch.utils.data.Subset(base_dst, class_indices)
            return subset
        else:
            return base_dst


if __name__ == "__main__":
    # Example usage
    data_dir = "data/"
    normal_class = 0
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    cifar_dataset = CifarDataset(dst_name='cifar10', data_dir=data_dir, normal_class=normal_class, train=False, transform=transform)
    print(f"Number of samples in the subset: {len(cifar_dataset)}")
    sample = cifar_dataset[0]
    print(f"Sample shape: {sample[0].shape}, Target: {sample[1]}")