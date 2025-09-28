from pyexpat import model
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
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
import os

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
trigger_size =3
target_label = 0
batch_size = 128
poi_loader_obj = PoisonedTestLoader(testloader, trigger_size=trigger_size,
                                    target_label=target_label, batch_size=batch_size)
poi_testloader = poi_loader_obj.get_poi_testloader()
# 方法1：通过 dataset 长度
print("Number of samples in poi_testloader:", len(poi_testloader.dataset))

# 方法2：计算 batch 数
num_batches = len(poi_testloader)
print("Number of batches in poi_testloader:", num_batches)

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("./resnet18_badnet_epoch40.pth", map_location="cpu"))
model.to("cpu")
device = "cpu"
gradcam_layer = "layer3"
gradcam_threshold = 0.8
gradcam_loader_obj = GradCAMLoader(model, poi_testloader,
                                    device=device, target_layer_name=gradcam_layer,
                                    threshold=gradcam_threshold)
gradcam_testloader = gradcam_loader_obj.get_gradcam_loader()

print("Number of batches in gradcam_testloader:", len(gradcam_testloader))
