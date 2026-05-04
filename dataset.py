# Dataset/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class FERDataset(Dataset):
    """
    General Dataset class for Facial Expression Recognition (RAF-DB, FER2013, etc.)
    """
    def __init__(self, images_path, images_class, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 1. 加载图像并转换为RGB
        img = Image.open(self.images_path[item]).convert('RGB')
        label = self.images_class[item]

        # 2. 应用数据增强（如在 train.py 中定义的 transforms）
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方推荐的整理函数，用于处理 Batch 数据
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels