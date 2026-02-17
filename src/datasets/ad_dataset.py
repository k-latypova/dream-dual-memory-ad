from abc import abstractmethod
from typing import Literal
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T


class ADDataset(Dataset):
    """Abstract base class for datasets used in anomaly detection tasks.

    Args:
        dst_name (str): Name of the dataset (e.g., 'cifar10', 'cifar100', 'imagenet100').
        data_dir (str): Directory where the dataset is stored.
        normal_class (int): The class label for the normal class.
        train (bool): Whether to load the training set or not.
        transform (callable, optional): A function/transform to apply to the data.
    """
    def __init__(self, dst_name: Literal['cifar10', 'cifar100', 'imagenet100'], data_dir: str,
                 normal_class: int, train=True, transform=None):
        self.data_dir = data_dir
        self.dst_name = dst_name
        self.normal_class = normal_class
        self.transform = transform
        self.train = train
        self.dataset = self.get_dataset()

        def test_label_transform(x):
            return 0 if x == self.normal_class else 1
        
        self.test_target_transform = T.Lambda(test_label_transform)
        


    @abstractmethod
    def get_dataset(self) -> Dataset:
        """
        Returns the dataset for the given parameters.
        """
        pass

    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, idx):
        """
        Returns a single data point from the dataset.
        """
        
        data, targets = self.dataset[idx]
        # if self.transform:
        #     data = self.transform(data)
        if self.train:
            return data, targets
        else:
            targets = self.test_target_transform(targets)
            return data, targets
        
    


