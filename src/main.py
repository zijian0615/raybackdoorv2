import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from getBadnet import PoisonedTestLoader
from getSaliency import GradCAMLoader
from calASR import ReconEvaluator
from vae import FlexibleVAE
import torchvision
import torchvision.transforms as transforms
from getBadnet import PoisonedTestLoader
from getSaliency import GradCAMLoader
from calASR import ReconEvaluator
from vae import FlexibleVAE
import torchvision
import torchvision.transforms as transforms
from getBadnet import PoisonedTestLoader


dataset_choice = 'cifar10'


if dataset_choice == 'imagenette':
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.Imagenette(
        root='./data', split='train', download=True, transform=transform
    )
    testset = torchvision.datasets.Imagenette(
        root='./data', split='val', download=True, transform=transform
    )

elif dataset_choice == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.243, 0.261])
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

else:
    raise ValueError("dataset_choice must be 'imagenette' or 'cifar10'")

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader  = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

class PoisonedReconPipeline:
    def __init__(
        self, testloader, model_ckpt, vae_ckpt,
        latent_dim=1024, input_size=32,
        target_label=0, trigger_size=3,
        gradcam_layer='layer3', gradcam_threshold=0.8,
        batch_size=128, device='cuda'
    ):
        """
        Args:
            testloader: 原始测试集 DataLoader
            model_ckpt: 分类模型 checkpoint 路径
            vae_ckpt: VAE checkpoint 路径
            latent_dim, input_size: VAE 参数
            target_label, trigger_size: PoisonedTestLoader 参数
            gradcam_layer, gradcam_threshold: GradCAMLoader 参数
            batch_size: DataLoader batch_size
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.batch_size = batch_size

        # 分类模型
        num_classes = 10
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_ckpt, map_location=device))
        self.model = self.model.to(device)

        # VAE
        self.vae = FlexibleVAE(latent_dim=latent_dim, input_size=input_size).to(device)
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        self.vae.eval()

        # PoisonedTestLoader
        poi_loader_obj = PoisonedTestLoader(testloader, trigger_size=trigger_size, target_label=target_label, batch_size=batch_size)
        poi_testloader = poi_loader_obj.get_poi_testloader()

        # GradCAMLoader
        gradcam_loader_obj = GradCAMLoader(self.model, poi_testloader, device=device, target_layer_name=gradcam_layer, threshold=gradcam_threshold)
        self.gradcam_testloader = gradcam_loader_obj.get_gradcam_loader()

        # ReconEvaluator
        self.recon_evaluator = ReconEvaluator(self.model, self.vae, self.gradcam_testloader, device=device, batch_size=batch_size)

    def get_recon_loader(self):
        """返回最终重建 DataLoader"""
        return self.recon_evaluator.get_recon_loader()

    def test_accuracy(self):
        """计算 imgs_poisoned / whole_images / recon_images 的准确率"""
        return self.recon_evaluator.test_accuracy()


pipeline = PoisonedReconPipeline(
    testloader=testloader,
    model_ckpt="./resnet18_badnet_epoch40.pth",
    vae_ckpt="./vaemodel/datacifar10_latent1024_epoch1200.pth",
    latent_dim=1024,
    input_size=32,
    target_label=0,
    trigger_size=3,
    gradcam_layer='layer3',
    gradcam_threshold=0.8,
    batch_size=128,
    device='cuda'
)

recon_loader = pipeline.get_recon_loader()
pipeline.test_accuracy()
