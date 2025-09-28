import ray
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision
import torchvision.transforms as transforms

from getBadnet import PoisonedTestLoader
from getSaliency import GradCAMLoader
from calASR import ReconEvaluator
from vae import FlexibleVAE

ray.init()  # CPU-only 集群也可以

@ray.remote
class PoisonedReconActorCPU:
    def __init__(self, model_ckpt, vae_ckpt, latent_dim=1024, input_size=32):
        self.device = 'cpu'  # 强制使用 CPU

        # 分类模型
        num_classes = 10
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
        self.model.to(self.device)

        # VAE
        self.vae = FlexibleVAE(latent_dim=latent_dim, input_size=input_size).to(self.device)
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'))
        self.vae.eval()

    def run_pipeline(self, testloader, target_label=0, trigger_size=3,
                     gradcam_layer='layer3', gradcam_threshold=0.8, batch_size=128):
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
        recon_evaluator = ReconEvaluator(self.model, self.vae, gradcam_testloader,
                                         device=self.device, batch_size=batch_size)

        # 返回重建 DataLoader 和准确率
        recon_loader = recon_evaluator.get_recon_loader()
        acc = recon_evaluator.test_accuracy()
        return recon_loader, acc

# ---------------------
# 准备 CIFAR-10 testloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.247, 0.243, 0.261])
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# ---------------------
# 初始化 CPU Actor
actor = PoisonedReconActorCPU.remote(
    model_ckpt="./resnet18_badnet_epoch40.pth",
    vae_ckpt="./vaemodel/datacifar10_latent1024_epoch1200.pth",
    latent_dim=1024,
    input_size=32
)

# 运行 pipeline
recon_loader_ref, acc_ref = actor.run_pipeline.remote(testloader)
recon_loader, acc_result = ray.get([recon_loader_ref, acc_ref])

print("Accuracy on reconstructed dataset:", acc_result)
