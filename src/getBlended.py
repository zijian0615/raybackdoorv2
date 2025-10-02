import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision
from torch.utils.data import TensorDataset, DataLoader

# class PoisonedDataset(Dataset):
#     def __init__(self, dataset, poison_rate=0.1, target_label=0, trigger_size=3,
#                  trigger_mode='patch', trigger_img_path=None, alpha=0.2, seed=42):
#         """
#         Args:
#             dataset: 原始数据集
#             poison_rate: 中毒比例
#             target_label: 目标标签
#             trigger_size: patch 触发器大小 (trigger_mode='patch' 时有效)
#             trigger_mode: 'patch' 或 'blended'
#             trigger_img_path: blended trigger 图片路径 (trigger_mode='blended' 时必须)
#             alpha: blended trigger 透明度 (0~1)
#             seed: 随机种子
#         """
#         self.dataset = dataset
#         self.poison_rate = poison_rate
#         self.target_label = target_label
#         self.trigger_size = trigger_size
#         self.trigger_mode = trigger_mode
#         self.alpha = alpha

#         np.random.seed(seed)
#         self.poison_indices = set(np.random.choice(
#             len(dataset),
#             int(len(dataset) * poison_rate),
#             replace=False
#         ))

#         if self.trigger_mode == 'blended':
#             assert trigger_img_path is not None, "trigger_img_path must be provided for blended mode"
#             trigger_img = Image.open(trigger_img_path).convert('RGB')
#             sample_img, _ = dataset[0]
#             _, H, W = sample_img.shape
#             self.trigger_img = transforms.Compose([
#                 transforms.Resize((H, W)),
#                 transforms.ToTensor()
#             ])(trigger_img)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img, label = self.dataset[idx]
#         img = img.clone()

#         if idx in self.poison_indices:
#             if self.trigger_mode == 'patch':
#                 img[:, -self.trigger_size:, -self.trigger_size:] = 1.0
#             elif self.trigger_mode == 'blended':
#                 img = (1 - self.alpha) * img + self.alpha * self.trigger_img
#                 img = torch.clamp(img, 0, 1)
#             else:
#                 raise ValueError(f"Unknown trigger_mode: {self.trigger_mode}")
#             label = self.target_label

#         return img, label

class PoisonedTestLoader:
    def __init__(self, testloader,trigger_img_path,trigger_mode='blended',alpha=0.2, trigger_size=3, target_label=0, batch_size=128, device='cuda'):
        """
        初始化 PoisonedTestLoader
        :param testloader: 原始测试集 DataLoader
        :param trigger_size: 触发器大小
        :param target_label: 触发后强制的目标标签
        :param batch_size: DataLoader 的 batch_size
        :param device: 数据所在设备
        :param trigger_img_path: 触发器图片路径
        :param trigger_mode: 触发器模式
        :param alpha: 触发器透明度

        """
        self.testloader = testloader
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.batch_size = batch_size
        self.trigger_img_path = trigger_img_path
        self.trigger_mode = trigger_mode
        self.alpha = alpha
        self.device = device

        # 构建 poisoned dataset
        self._prepare_poisoned_dataset()

    def _add_trigger(self, images):
        """add trigger"""
        images = images.clone()
        _, _, H, W = images.shape
        if self.trigger_mode == 'blended':
            trigger_img = Image.open(self.trigger_img_path).convert('RGB')
            trigger_img = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor()
            ])(trigger_img)
            images = (1 - self.alpha) * images + self.alpha * trigger_img
        else:
            images[:, :, H-self.trigger_size:H, W-self.trigger_size:W] = 1.0
        return images

    def _prepare_poisoned_dataset(self):
        """提取所有测试图像并加触发器"""
        all_images, all_labels = [], []
        for images, labels in self.testloader:
            all_images.append(images)
            all_labels.append(labels)
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        poisoned_images = self._add_trigger(all_images)
        #poi_labels = torch.full_like(all_labels, fill_value=self.target_label)
        if self.target_label == -1:
            poi_labels = all_labels
        else:
            poi_labels = torch.full_like(all_labels, fill_value=self.target_label)

        # 构造 DataLoader
        self.poi_dataset = TensorDataset(poisoned_images, poi_labels)
        self.poi_testloader = DataLoader(
            self.poi_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def get_poi_testloader(self):
        """返回 poisoned DataLoader"""
        return self.poi_testloader
