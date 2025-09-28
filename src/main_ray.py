# import ray
# import torch
# from torch.utils.data import DataLoader
# from torchvision.models import resnet18
# import torchvision
# import torchvision.transforms as transforms
# import os

# from getBadnet import PoisonedTestLoader
# from getSaliency import GradCAMLoader
# from calASR import ReconEvaluator
# from vae import FlexibleVAE

# ray.init(address="auto")  # 连接已有集群

# # 使用绝对路径
# MODEL_CKPT = "/home/ray/raybackdoorv1/resnet18_badnet_epoch40.pth"
# VAE_CKPT = "/home/ray/raybackdoorv1/vaemodel/datacifar10_latent1024_epoch1200.pth"

# # 检查文件是否存在
# assert os.path.exists(MODEL_CKPT), f"Model checkpoint not found: {MODEL_CKPT}"
# assert os.path.exists(VAE_CKPT), f"VAE checkpoint not found: {VAE_CKPT}"

# @ray.remote
# class PoisonedReconActorCPU:
#     def __init__(self, model_ckpt, vae_ckpt, latent_dim=1024, input_size=32):
#         self.device = 'cpu'

#         # 分类模型
#         num_classes = 10
#         self.model = resnet18(pretrained=False)
#         self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
#         print(f"Loading model from {model_ckpt} ...")
#         self.model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
#         self.model.to(self.device)

#         # VAE
#         self.vae = FlexibleVAE(latent_dim=latent_dim, input_size=input_size).to(self.device)
#         print(f"Loading VAE from {vae_ckpt} ...")
#         self.vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'))
#         self.vae.eval()

#     def run_pipeline(self, dataset_root="/home/ray/raybackdoorv1/data", target_label=0, trigger_size=3,
#                      gradcam_layer='layer3', gradcam_threshold=0.8, batch_size=32):
#         # 在 Actor 内重新构建 DataLoader
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                  std=[0.247, 0.243, 0.261])
#         ])
#         testset = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
#         testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

#         # Poisoned 测试集
#         poi_loader_obj = PoisonedTestLoader(testloader, trigger_size=trigger_size,
#                                             target_label=target_label, batch_size=batch_size)
#         poi_testloader = poi_loader_obj.get_poi_testloader()

#         # Grad-CAM mask
#         gradcam_loader_obj = GradCAMLoader(self.model, poi_testloader,
#                                           device=self.device, target_layer_name=gradcam_layer,
#                                           threshold=gradcam_threshold)
#         gradcam_testloader = gradcam_loader_obj.get_gradcam_loader()

#         # VAE 重建 + mask 融合
#         recon_evaluator = ReconEvaluator(self.model, self.vae, gradcam_testloader,
#                                          device=self.device, batch_size=batch_size)

#         acc = recon_evaluator.test_accuracy()
#         return acc

# # ---------------------
# # 初始化 CPU Actor
# actor = PoisonedReconActorCPU.options(
#     num_cpus=1,
#     memory=0.5 * 1024 * 1024 * 1024  # 2GB
# ).remote(
#     model_ckpt=MODEL_CKPT,
#     vae_ckpt=VAE_CKPT,
#     latent_dim=1024,
#     input_size=32
# )

# # 运行 pipeline
# acc_result = ray.get(actor.run_pipeline.remote())
# print("Accuracy on reconstructed dataset:", acc_result)


import ray
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision
import torchvision.transforms as transforms
import os

from getBadnet import PoisonedTestLoader
from getSaliency import GradCAMLoader
from vae import FlexibleVAE

# -----------------------------
# 初始化 Ray
ray.init(address="auto")  # 连接已有集群

# -----------------------------
# 模型和 VAE checkpoint
MODEL_CKPT = "/home/ray/raybackdoorv1/resnet18_badnet_epoch40.pth"
VAE_CKPT = "/home/ray/raybackdoorv1/vaemodel/datacifar10_latent1024_epoch1200.pth"

assert os.path.exists(MODEL_CKPT), f"Model checkpoint not found: {MODEL_CKPT}"
assert os.path.exists(VAE_CKPT), f"VAE checkpoint not found: {VAE_CKPT}"

# -----------------------------
# 自定义 ReconEvaluator（按整个 dataset 计算总 accuracy）
class ReconEvaluator:
    def __init__(self, model, vae, dataloader, device='cpu'):
        self.model = model
        self.vae = vae
        self.loader = dataloader
        self.device = device

    def test_accuracy(self):
        self.model.eval()
        self.vae.eval()
        correct_total = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels, masks in self.loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)

                # VAE 重建 + mask 融合
                recon = self.vae(images)
                fused = masks * recon + (1 - masks) * images

                # 分类预测
                outputs = self.model(fused)
                _, predicted = outputs.max(1)

                correct_total += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        # with torch.no_grad():
        #     for i, (images, labels, masks) in enumerate(self.loader):
        #         print(f"Batch {i}: images={images.shape}, labels={labels}, masks={masks.shape}")
        #         total_samples += labels.size(0)
        #         print("  total_samples so far:", total_samples)
        #         break  # 先检查第一 batch




        acc = correct_total / total_samples
        return acc

# -----------------------------
# 定义 Ray Actor
@ray.remote
class PoisonedReconActorCPU:
    def __init__(self, model_ckpt, vae_ckpt, latent_dim=1024, input_size=32):
        self.device = 'cpu'

        # 分类模型
        num_classes = 10
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        print(f"Loading model from {model_ckpt} ...")
        self.model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
        self.model.to(self.device)

        # VAE
        self.vae = FlexibleVAE(latent_dim=latent_dim, input_size=input_size).to(self.device)
        print(f"Loading VAE from {vae_ckpt} ...")
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'))
        self.vae.eval()

    def run_pipeline(self, dataset_root="/home/ray/raybackdoorv1/data", target_label=0, trigger_size=3,
                     gradcam_layer='layer3', gradcam_threshold=0.8, batch_size=32):
        # -----------------------------
        # 构建 DataLoader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.247, 0.243, 0.261])
        ])
        testset = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Poisoned 测试集
        poi_loader_obj = PoisonedTestLoader(testloader, trigger_size=trigger_size,
                                            target_label=target_label, batch_size=batch_size)
        poi_testloader = poi_loader_obj.get_poi_testloader()

        # Grad-CAM mask
        gradcam_loader_obj = GradCAMLoader(self.model, poi_testloader,
                                          device=self.device, target_layer_name=gradcam_layer,
                                          threshold=gradcam_threshold)
        gradcam_testloader = gradcam_loader_obj.get_gradcam_loader()

        # VAE 重建 + mask 融合
        recon_evaluator = ReconEvaluator(self.model, self.vae, gradcam_testloader, device=self.device)

        # -----------------------------
        # 计算总 accuracy（处理完整个 dataset）
        acc = recon_evaluator.test_accuracy()
        return acc

# -----------------------------
# 初始化 CPU Actor
actor = PoisonedReconActorCPU.options(
    num_cpus=1,
    memory=0.5 * 1024 * 1024 * 1024  # 0.5GB
).remote(
    model_ckpt=MODEL_CKPT,
    vae_ckpt=VAE_CKPT,
    latent_dim=1024,
    input_size=32
)

#
#
acc_result = ray.get(actor.run_pipeline.remote())
print("Accuracy on reconstructed dataset:", acc_result)
