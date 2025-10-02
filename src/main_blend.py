import torch
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18,vgg16
from calASR import ReconEvaluator
from vae import FlexibleVAE
from calASR import ReconEvaluator
from getBlended import PoisonedTestLoader
from getSaliency import GradCAMLoader




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
    num_classes = 10

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
    num_classes = 10

elif dataset_choice == 'gtsrb':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                             std=[0.2724, 0.2608, 0.2669])
    ])
    trainset = torchvision.datasets.GTSRB(
        root='./data', split='train',download=True,
        transform=transform
    )
    testset = torchvision.datasets.GTSRB(
        root='./data',split='test',download=True,
        transform=transform
    )
    num_classes = 43  # GTSRB有43类交通标志

else:
    raise ValueError("dataset_choice must be 'imagenette', 'cifar10' or 'gtsrb'")


trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader  = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

class PoisonedReconPipeline:
    def __init__(
        self, testloader, model_ckpt, vae_ckpt,
        latent_dim=1024, input_size=32,
        target_label=0, trigger_size=3,
        gradcam_layer='layer3', gradcam_threshold=0.8,
        batch_size=128, device='cuda',trigger_img_path=None,alpha=0.2,trigger_mode='blended'
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
            trigger_img_path: 触发器图片路径
            alpha: 触发器透明度
            trigger_mode: 触发器模式
        """
        self.device = device
        self.batch_size = batch_size
        self.trigger_img_path = trigger_img_path
        self.alpha = alpha
        self.trigger_mode = trigger_mode
        # 分类模型
        #num_classes = num_classes
        self.model = resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_ckpt, map_location=device))
        self.model = self.model.to(device)

        # VAE
        self.vae = FlexibleVAE(latent_dim=latent_dim, input_size=input_size).to(device)
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        self.vae.eval()

        # PoisonedTestLoader
        poi_loader_obj = PoisonedTestLoader(testloader, trigger_size=trigger_size,trigger_mode='blended',trigger_img_path=trigger_img_path,alpha=alpha, target_label=target_label, batch_size=batch_size)
        poi_testloader = poi_loader_obj.get_poi_testloader()

        # GradCAMLoader
        gradcam_loader_obj = GradCAMLoader(self.model, poi_testloader, device=device, target_layer_name=gradcam_layer, threshold=gradcam_threshold)
        self.gradcam_testloader = gradcam_loader_obj.get_gradcam_loader()

        # ReconEvaluator
        self.recon_evaluator = ReconEvaluator(self.model, self.vae, self.gradcam_testloader, device=device, batch_size=batch_size)
        print("Number of  samples in gradcam_testloader:", len(self.gradcam_testloader.dataset))
        print("Number of  samples in recon_loader:", len(self.recon_evaluator.get_recon_loader().dataset))

    def get_recon_loader(self):
        """返回最终重建 DataLoader"""
        return self.recon_evaluator.get_recon_loader()

    def test_accuracy(self):
        """计算 imgs_poisoned / whole_images / recon_images 的准确率"""
        return self.recon_evaluator.test_accuracy()


#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#model_ckpt_path = os.path.join(BASE_DIR,"badmodel", "resnet18_badnet_epoch40.pth")
model_ckpt_path = os.path.join(BASE_DIR,"badmodel", "resnet18_cifar10_blended_epoch50.pth")
#model_ckpt_path = os.path.join(BASE_DIR,"badmodel", "resnet18_imagenette_badnet_epoch100.pth")
#vae_ckpt_path = os.path.join(BASE_DIR, "vaemodel", "datacifar10_latent1024_epoch1200.pth")
vae_ckpt_path = os.path.join(BASE_DIR, "vaemodel", "cifar10_latent1024_epoch1200.pth")


pipeline = PoisonedReconPipeline(
    testloader=testloader,
    model_ckpt=model_ckpt_path,
    vae_ckpt=vae_ckpt_path,
    latent_dim=1024,
    input_size=32,
    target_label=-1,
    trigger_size=0,
    gradcam_layer='layer1',
    gradcam_threshold=0.9,
    batch_size=8,
    device='cuda',
    trigger_img_path='./hellokitty.png',
    alpha=0,
    trigger_mode='blended'
)

recon_loader = pipeline.get_recon_loader()
pipeline.test_accuracy()
