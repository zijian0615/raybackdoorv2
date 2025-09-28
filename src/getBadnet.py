import torch
from torch.utils.data import DataLoader, TensorDataset

class PoisonedTestLoader:
    def __init__(self, testloader, trigger_size=3, target_label=0, batch_size=128, device='cuda'):
        """
        初始化 PoisonedTestLoader
        :param testloader: 原始测试集 DataLoader
        :param trigger_size: 触发器大小
        :param target_label: 触发后强制的目标标签
        :param batch_size: DataLoader 的 batch_size
        :param device: 数据所在设备
        """
        self.testloader = testloader
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.batch_size = batch_size
        self.device = device

        # 构建 poisoned dataset
        self._prepare_poisoned_dataset()

    def _add_trigger(self, images):
        """在右下角加白色方块"""
        images = images.clone()
        _, _, H, W = images.shape
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
