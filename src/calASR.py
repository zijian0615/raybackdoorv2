import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class ReconEvaluator:
    def __init__(self, model, vae, gradcam_loader, device='cuda', batch_size=128, gaussian_kernel=5, sigma=1.0):
        """
        Args:
            model: 用于分类的模型（如 ResNet18）
            vae: 已训练的 VAE 模型
            gradcam_loader: GradCAMLoader 返回的 DataLoader，输出 (image, label, mask)
            device: 设备 'cuda' 或 'cpu'
            batch_size: DataLoader batch_size
            gaussian_kernel: 高斯模糊 kernel size
            sigma: 高斯模糊 sigma
        """
        self.model = model.to(device)
        self.vae = vae.to(device)
        self.gradcam_loader = gradcam_loader
        self.device = device
        self.batch_size = batch_size
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma

        self._prepare_recon_loader()

    def gaussian_blur(self, imgs):
        """对 imgs 进行高斯模糊"""
        x = torch.arange(self.gaussian_kernel).float() - self.gaussian_kernel // 2
        x = torch.exp(-(x**2)/(2*self.sigma**2))
        kernel_1d = x / x.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.expand(imgs.size(1), 1, self.gaussian_kernel, self.gaussian_kernel).to(imgs.device)
        padding = self.gaussian_kernel // 2
        blur_imgs = F.conv2d(imgs, kernel_2d, padding=padding, groups=imgs.size(1))
        return blur_imgs

    def _prepare_recon_loader(self):
        """使用 VAE 重建图像并生成融合 mask 的 dataset 和 DataLoader"""
        self.vae.eval()
        self.model.eval()

        imgs_poisoned_list = []
        whole_images_list = []
        recon_images_list = []
        labels_list = []
        masks_list = []

        with torch.no_grad():
            for imgs, labels, masks in self.gradcam_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)

                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                masks = masks.expand(-1, imgs.size(1), -1, -1)

                blur_imgs = self.gaussian_blur(imgs)
                pre_imgs = imgs * (1 - masks) + masks * blur_imgs
                whole = self.vae(pre_imgs)[0]  # VAE 输出

                recon = masks * whole + (1 - masks) * imgs  # Mask 融合

                imgs_poisoned_list.append(imgs.cpu())
                whole_images_list.append(whole.cpu())
                recon_images_list.append(recon.cpu())
                labels_list.append(labels.cpu())
                masks_list.append(masks.cpu())

        # 拼接成完整 dataset
        imgs_poisoned = torch.cat(imgs_poisoned_list, dim=0)
        whole_images = torch.cat(whole_images_list, dim=0)
        recon_images = torch.cat(recon_images_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)
        masks_all = torch.cat(masks_list, dim=0)

        # 构建 DataLoader
        self.recon_dataset = TensorDataset(imgs_poisoned, whole_images, recon_images, labels_all, masks_all)
        self.recon_loader = DataLoader(self.recon_dataset, batch_size=self.batch_size, shuffle=False)

    def get_recon_loader(self):
        """返回生成的 DataLoader"""
        return self.recon_loader

    def test_accuracy(self):
        """在 imgs_poisoned / whole_images / recon_images 上分别计算准确率"""
        self.model.eval()
        correct_poisoned = 0
        correct_whole = 0
        correct_recon = 0
        total = 0

        with torch.no_grad():
            for imgs_poisoned, whole_images, recon_images, labels, masks in self.recon_loader:
                imgs_poisoned = imgs_poisoned.to(self.device)
                whole_images = whole_images.to(self.device)
                recon_images = recon_images.to(self.device)
                labels = labels.to(self.device)

                # imgs_poisoned
                outputs = self.model(imgs_poisoned)
                _, pred = outputs.max(1)
                correct_poisoned += (pred == labels).sum().item()

                # whole_images
                outputs = self.model(whole_images)
                _, pred = outputs.max(1)
                correct_whole += (pred == labels).sum().item()

                # recon_images
                outputs = self.model(recon_images)
                _, pred = outputs.max(1)
                correct_recon += (pred == labels).sum().item()

                total += labels.size(0)

        acc_poisoned = correct_poisoned / total
        acc_whole = correct_whole / total
        acc_recon = correct_recon / total

        print(f"Accuracy on imgs_poisoned: {acc_poisoned*100:.2f}%")
        print(f"Accuracy on whole_images: {acc_whole*100:.2f}%")
        print(f"Accuracy on recon_images: {acc_recon*100:.2f}%")

        return acc_poisoned, acc_whole, acc_recon
