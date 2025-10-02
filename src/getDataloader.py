# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision
# import torchvision.transforms as transforms
# from PIL import Image
# device = torch.device("cuda")


# class DataLoaderManager:
#     def __init__(self, dataset_choice='gtsrb', poison_rate=0.1, target_label=0, 
#                  trigger_size=3, bs=128, nw=2, seed=42):
#         """
#         初始化数据加载器管理器
        
#         Args:
#             dataset_choice: 数据集选择 ('imagenette', 'cifar10', 'gtsrb')
#             poison_rate: 中毒比例 (0.0-1.0)
#             target_label: 目标标签 (-1表示clean数据集, 0-9表示poisoned数据集)
#             trigger_size: 触发器大小 (0表示clean数据集, >0表示poisoned数据集)
#             bs: 批次大小
#             nw: 数据加载工作进程数
#             seed: 随机种子
#         """
#         self.dataset_choice = dataset_choice
#         self.poison_rate = poison_rate
#         self.target_label = target_label
#         self.trigger_size = trigger_size
#         self.bs = bs
#         self.nw = nw
#         self.seed = seed
        
#         # 设置类别数
#         if dataset_choice == 'imagenette':
#             self.num_classes = 10
#         elif dataset_choice == 'cifar10':
#             self.num_classes = 10
#         elif dataset_choice == 'gtsrb':
#             self.num_classes = 43
#         else:
#             raise ValueError("dataset_choice must be 'imagenette', 'cifar10' or 'gtsrb'")
    
#     def _get_transform(self):
#         """获取数据预处理变换"""
#         if self.dataset_choice == 'imagenette':
#             return transforms.Compose([
#                 transforms.Resize(224),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#         elif self.dataset_choice == 'cifar10':
#             return transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                      std=[0.247, 0.243, 0.261])
#             ])
#         elif self.dataset_choice == 'gtsrb':
#             return transforms.Compose([
#                 transforms.Resize((32, 32)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
#                                      std=[0.2724, 0.2608, 0.2669])
#             ])
    
#     def _load_datasets(self):
#         """加载原始数据集"""
#         transform = self._get_transform()
        
#         if self.dataset_choice == 'imagenette':
#             trainset = torchvision.datasets.Imagenette(
#                 root='../data', split='train', download=True, transform=transform
#             )
#             testset = torchvision.datasets.Imagenette(
#                 root='../data', split='val', download=True, transform=transform
#             )
#         elif self.dataset_choice == 'cifar10':
#             trainset = torchvision.datasets.CIFAR10(
#                 root='../data', train=True, download=True, transform=transform
#             )
#             testset = torchvision.datasets.CIFAR10(
#                 root='../data', train=False, download=True, transform=transform
#             )
#         elif self.dataset_choice == 'gtsrb':
#             trainset = torchvision.datasets.GTSRB(
#                 root='../data', split='train', download=True, transform=transform
#             )
#             testset = torchvision.datasets.GTSRB(
#                 root='../data', split='test', download=True, transform=transform
#             )
        
#         return trainset, testset
    
#     def getDataloader(self):
#         """
#         获取数据加载器
        
#         Returns:
#             tuple: (trainloader, testloader)
            
#         Note:
#             - 当 target_label=-1 且 trigger_size=0: 返回clean数据集
#             - 当 target_label>=0 且 trigger_size>0: 返回poisoned数据集
#         """
#         trainset, testset = self._load_datasets()
        
#         # 判断是否为clean数据集
#         is_clean = (self.target_label == -1 and self.trigger_size == 0)
        
#         if is_clean:
#             # Clean数据集：直接使用原始数据
#             trainloader = DataLoader(
#                 trainset, 
#                 batch_size=self.bs, 
#                 shuffle=True, 
#                 num_workers=self.nw
#             )
#             testloader = DataLoader(
#                 testset,
#                 batch_size=self.bs, 
#                 shuffle=False, 
#                 num_workers=self.nw
#             )
#         else:
#             # Poisoned数据集：使用PoisonedDataset
#             train_dataset = PoisonedDataset(
#                 trainset, 
#                 poison_rate=self.poison_rate, 
#                 target_label=self.target_label,
#                 trigger_size=self.trigger_size,
#                 seed=self.seed
#             )

#             test_dataset = PoisonedDataset(
#                 testset, 
#                 poison_rate=self.poison_rate, 
#                 target_label=self.target_label,
#                 trigger_size=self.trigger_size,
#                 seed=self.seed
#             )
#             trainloader = DataLoader(
#                 train_dataset, 
#                 batch_size=self.bs, 
#                 shuffle=True, 
#                 num_workers=self.nw
#             )
#             testloader = DataLoader(
#                 test_dataset,
#                 batch_size=self.bs, 
#                 shuffle=False, 
#                 num_workers=self.nw
#             )
        
#         return trainloader, testloader


# class PoisonedDataset(Dataset):
#     def __init__(self, dataset, poison_rate=0.1, target_label=0, trigger_size=3, seed=42):
#         """
#         Args:
#             dataset: 原始数据集
#             poison_rate: 中毒比例
#             target_label: 目标标签
#             trigger_size: 触发器大小
#             seed: 随机种子
#         """
#         self.dataset = dataset
#         self.poison_rate = poison_rate
#         self.target_label = target_label
#         self.trigger_size = trigger_size
        
#         # 固定随机种子
#         np.random.seed(seed)
#         self.poison_indices = set(np.random.choice(
#             len(dataset),
#             int(len(dataset) * poison_rate),
#             replace=False
#         ))

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img, label = self.dataset[idx]

#         # 中毒样本：添加触发器并修改标签
#         if idx in self.poison_indices:
#             img = img.clone()
#             img[:, -self.trigger_size:, -self.trigger_size:] = 1.0
#             label = self.target_label

#         return img, label
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision

class PoisonedDataset(Dataset):
    def __init__(self, dataset, poison_rate=0.1, target_label=0, trigger_size=3,
                 trigger_mode='patch', trigger_img_path=None, alpha=0.2, seed=42):
        """
        Args:
            dataset: 原始数据集
            poison_rate: 中毒比例
            target_label: 目标标签
            trigger_size: patch 触发器大小 (trigger_mode='patch' 时有效)
            trigger_mode: 'patch' 或 'blended'
            trigger_img_path: blended trigger 图片路径 (trigger_mode='blended' 时必须)
            alpha: blended trigger 透明度 (0~1)
            seed: 随机种子
        """
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_mode = trigger_mode
        self.alpha = alpha

        np.random.seed(seed)
        self.poison_indices = set(np.random.choice(
            len(dataset),
            int(len(dataset) * poison_rate),
            replace=False
        ))

        if self.trigger_mode == 'blended':
            assert trigger_img_path is not None, "trigger_img_path must be provided for blended mode"
            trigger_img = Image.open(trigger_img_path).convert('RGB')
            sample_img, _ = dataset[0]
            _, H, W = sample_img.shape
            self.trigger_img = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor()
            ])(trigger_img)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.clone()

        if idx in self.poison_indices:
            if self.trigger_mode == 'patch':
                img[:, -self.trigger_size:, -self.trigger_size:] = 1.0
            elif self.trigger_mode == 'blended':
                img = (1 - self.alpha) * img + self.alpha * self.trigger_img
                img = torch.clamp(img, 0, 1)
            else:
                raise ValueError(f"Unknown trigger_mode: {self.trigger_mode}")
            label = self.target_label

        return img, label

# ------------------- DataLoaderManager -------------------

class DataLoaderManager:
    def __init__(self, dataset_choice='gtsrb', poison_rate=0.1, target_label=0, 
                 trigger_size=3, trigger_mode='patch', trigger_img_path=None,
                 alpha=0.2, bs=128, nw=2, seed=42):
        """
        初始化数据加载器管理器
        """
        self.dataset_choice = dataset_choice
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_mode = trigger_mode
        self.trigger_img_path = trigger_img_path
        self.alpha = alpha
        self.bs = bs
        self.nw = nw
        self.seed = seed
        
        if dataset_choice == 'imagenette':
            self.num_classes = 10
        elif dataset_choice == 'cifar10':
            self.num_classes = 10
        elif dataset_choice == 'gtsrb':
            self.num_classes = 43
        else:
            raise ValueError("dataset_choice must be 'imagenette', 'cifar10' or 'gtsrb'")
    
    def _get_transform(self):
        if self.dataset_choice == 'imagenette':
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif self.dataset_choice == 'cifar10':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.243, 0.261])
            ])
        elif self.dataset_choice == 'gtsrb':
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                                     std=[0.2724, 0.2608, 0.2669])
            ])
    
    def _load_datasets(self):
        transform = self._get_transform()
        if self.dataset_choice == 'imagenette':
            trainset = torchvision.datasets.Imagenette(root='../data', split='train', download=True, transform=transform)
            testset = torchvision.datasets.Imagenette(root='../data', split='val', download=True, transform=transform)
        elif self.dataset_choice == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        elif self.dataset_choice == 'gtsrb':
            trainset = torchvision.datasets.GTSRB(root='../data', split='train', download=True, transform=transform)
            testset = torchvision.datasets.GTSRB(root='../data', split='test', download=True, transform=transform)
        return trainset, testset
    
    def getDataloader(self):
        trainset, testset = self._load_datasets()
        is_clean = (self.target_label == -1 and self.trigger_size == 0)
        
        if is_clean:
            trainloader = DataLoader(trainset, batch_size=self.bs, shuffle=True, num_workers=self.nw)
            testloader = DataLoader(testset, batch_size=self.bs, shuffle=False, num_workers=self.nw)
        else:
            train_dataset = PoisonedDataset(
                trainset, 
                poison_rate=self.poison_rate, 
                target_label=self.target_label,
                trigger_size=self.trigger_size,
                trigger_mode=self.trigger_mode,
                trigger_img_path=self.trigger_img_path,
                alpha=self.alpha,
                seed=self.seed
            )
            test_dataset = PoisonedDataset(
                testset, 
                poison_rate=self.poison_rate, 
                target_label=self.target_label,
                trigger_size=self.trigger_size,
                trigger_mode=self.trigger_mode,
                trigger_img_path=self.trigger_img_path,
                alpha=self.alpha,
                seed=self.seed
            )
            trainloader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=self.nw)
            testloader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False, num_workers=self.nw)
        
        return trainloader, testloader
