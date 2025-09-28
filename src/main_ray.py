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
#                      gradcam_layer='layer3', gradcam_threshold=0.8, batch_size=16):
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
from calASR import ReconEvaluator
from vae import FlexibleVAE

# 连接已有 Ray 集群
ray.init(address="auto")

MODEL_CKPT = "/home/ray/raybackdoorv1/resnet18_badnet_epoch40.pth"
VAE_CKPT = "/home/ray/raybackdoorv1/vaemodel/datacifar10_latent1024_epoch1200.pth"

assert os.path.exists(MODEL_CKPT), f"Model checkpoint not found: {MODEL_CKPT}"
assert os.path.exists(VAE_CKPT), f"VAE checkpoint not found: {VAE_CKPT}"

@ray.remote
class PoisonedReconActorCPU:
    def __init__(self, model_ckpt, vae_ckpt, latent_dim=1024, input_size=32):
        from ray._private.services import get_node_ip_address

        self.node_ip = get_node_ip_address()
        print(f"[Init] Actor initialized on Node IP: {self.node_ip}")

        self.device = 'cpu'

        # 分类模型
        num_classes = 10
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        print(f"[Init] Loading model from {model_ckpt} ...")
        self.model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
        self.model.to(self.device)

        # VAE
        self.vae = FlexibleVAE(latent_dim=latent_dim, input_size=input_size).to(self.device)
        print(f"[Init] Loading VAE from {vae_ckpt} ...")
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'))
        self.vae.eval()

     def run_pipeline(self, dataset_root="/home/ray/raybackdoorv1/data",
                     target_label=0, trigger_size=3,
                     gradcam_layer='layer3', gradcam_threshold=0.8,
                     batch_size=2):
        from ray._private.services import get_node_ip_address
        node_ip = get_node_ip_address()
        print(f"[Run] run_pipeline executing on Node IP: {node_ip}")
    
        # 数据加载
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
        recon_images_list, labels_list = [], []
    
        for batch in gradcam_testloader:
            # 改正解包方式
            images, labels, masks = batch  # 现在 gradcam_loader 每个 batch 返回 3 个元素
    
            # 这里可以直接把 images, masks 送到 VAE + mask 融合
            with torch.no_grad():
                recon = self.vae(images)
                # mask 融合
                recon_masked = masks * recon + (1 - masks) * images
    
            recon_images_list.append(recon_masked)
            labels_list.append(labels)
    
        # 拼接所有 batch
        recon_images_all = torch.cat(recon_images_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)
    
        # 计算准确率
        acc = recon_evaluator.test_accuracy(recon_images_all, labels_all)
    
        return acc, node_ip



# ---------------------
NUM_ACTORS = 2
actors = [
    PoisonedReconActorCPU.options(num_cpus=1, memory=0.5*1024*1024*1024).remote(
        model_ckpt=MODEL_CKPT,
        vae_ckpt=VAE_CKPT,
        latent_dim=1024,
        input_size=32
    ) for _ in range(NUM_ACTORS)
]

# 执行 pipeline
results = ray.get([actor.run_pipeline.remote() for actor in actors])

for i, (acc, node_ip) in enumerate(results):
    print(f"Actor {i} ran on Node IP: {node_ip}, Accuracy on full dataset: {acc:.4f}")



