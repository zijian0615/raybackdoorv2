import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np

class GradCAMDataset(Dataset):
    def __init__(self, images, labels, masks):
        """
        Args:
            images: torch.Tensor, shape [N, C, H, W]
            labels: torch.Tensor, shape [N]
            masks:  numpy.ndarray or torch.Tensor, shape [N, H, W]
        """
        self.images = images
        self.labels = labels
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        self.masks = masks.float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.masks[idx]


class GradCAMLoader:
    def __init__(self, model, dataloader, device='cuda', target_layer_name='layer3', threshold=0.8):
        """
        Args:
            model: torch model
            dataloader: 原始 DataLoader
            device: 'cuda' 或 'cpu'
            target_layer_name: ResNet18 中用于 Grad-CAM 的卷积层名
            threshold: saliency mask 阈值
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.threshold = threshold
        self.target_layer_name = target_layer_name

    def _get_target_layer(self):
        # 仅支持 ResNet18，选择对应的 layer
        if self.target_layer_name == 'layer3':
            return self.model.layer3[-1]
        elif self.target_layer_name == 'layer4':
            return self.model.layer4[-1]
        else:
            raise ValueError("Unsupported target_layer_name. Use 'layer3' or 'layer4'.")

    def get_gradcam_loader(self):
        """生成 Grad-CAM masks 并返回 DataLoader"""
        self.model.eval()
        target_layer = self._get_target_layer()
        cam = GradCAM(model=self.model, target_layers=[target_layer])

        # 获取一批图像
        images, labels = next(iter(self.dataloader))
        images = images.to(self.device)
        labels = labels.to(self.device)
        images.requires_grad_(True)

        # 预测
        outputs = self.model(images)
        predictions = torch.argmax(outputs, dim=1)

        # Grad-CAM 目标
        targets = [ClassifierOutputTarget(int(predictions[i].detach())) for i in range(len(predictions))]
        grayscale_cams = cam(input_tensor=images, targets=targets)  # numpy: [B, H, W]

        # 生成 mask
        masks = (grayscale_cams >= self.threshold).astype(np.float32)
        masks = torch.from_numpy(masks)          # 转成 tensor
        masks = masks.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]

        # 构建 Dataset + DataLoader
        gradcam_dataset = GradCAMDataset(images.cpu(), labels.cpu(), masks.cpu())
        gradcam_dataloader = DataLoader(gradcam_dataset, batch_size=self.dataloader.batch_size, shuffle=False)

        return gradcam_dataloader
