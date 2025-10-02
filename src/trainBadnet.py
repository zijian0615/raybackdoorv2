import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, vgg16
from torch.utils.data import DataLoader
import argparse
import os
import sys
sys.path.append('/home/liang/rayBackdoorClean')
from src.getDataloader import DataLoaderManager

class BadNetTrainer:
    def __init__(self, model_name='resnet18', num_classes=10, **kwargs):
        self.model_name = model_name
        self.num_classes = num_classes
        # 其他参数
        self.dataset_choice = kwargs.get('dataset_choice', 'cifar10')
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_workers = kwargs.get('num_workers', 4)
        self.epochs = kwargs.get('epochs', 100)
        self.lr = kwargs.get('lr', 0.1)
        self.momentum = kwargs.get('momentum', 0.9)
        self.weight_decay = kwargs.get('weight_decay', 5e-4)
        self.save_dir = kwargs.get('save_dir', './badmodel')
        self.poison_rate = kwargs.get('poison_rate', 0.1)
        self.target_label = kwargs.get('target_label', 0)
        self.trigger_size = kwargs.get('trigger_size', 3)
        self.trigger_mode = kwargs.get('trigger_mode', 'patch')
        self.trigger_img_path = kwargs.get('trigger_img_path', None)
        self.alpha = kwargs.get('alpha', 0.2)
        

        self._build_model()
        self._prepare_data()

    def _prepare_data(self):
        # Clean testloader
        clean_manager = DataLoaderManager(
            dataset_choice=self.dataset_choice,
            poison_rate=0,
            target_label=-1,
            trigger_size=0,
            bs=self.batch_size,
            nw=self.num_workers,
            seed=42,
            trigger_mode=self.trigger_mode,
            trigger_img_path=self.trigger_img_path,
            alpha=self.alpha
        )
        _, self.clean_testloader = clean_manager.getDataloader()

        # Poisoned testloader
        poisoned_manager = DataLoaderManager(
            dataset_choice=self.dataset_choice,
            poison_rate=1,
            target_label=self.target_label,
            trigger_size=self.trigger_size,
            bs=self.batch_size,
            nw=self.num_workers,
            seed=42,    
            trigger_mode=self.trigger_mode,
            trigger_img_path=self.trigger_img_path,
            alpha=self.alpha
        )
        _, self.poisoned_testloader = poisoned_manager.getDataloader()

        # Trainloader
        train_manager = DataLoaderManager(
            dataset_choice=self.dataset_choice,
            poison_rate=self.poison_rate,
            target_label=self.target_label,
            trigger_size=self.trigger_size,
            bs=self.batch_size,
            nw=self.num_workers,
            seed=42,
            trigger_mode=self.trigger_mode,
            trigger_img_path=self.trigger_img_path,
            alpha=self.alpha
        )
        self.trainloader, _ = train_manager.getDataloader()


    def _build_model(self):
        if self.model_name == 'resnet18':
            self.model = resnet18(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        elif self.model_name == 'vgg16':
            self.model = vgg16(pretrained=False)
            # 添加自适应池化，保证输入 32x32 或 224x224 都能兼容
            self.model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.model.classifier[6] = nn.Linear(4096, self.num_classes)

        else:
            raise ValueError("Unsupported model_name. Choose 'resnet18' or 'vgg16'.")

        
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

    def train(self):
        for epoch in range(self.epochs):
            # ---------- Training ----------
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / len(self.trainloader.dataset)
            epoch_acc = correct / total

            # ---------- Evaluation ----------
            self.model.eval()
            ca_total = 0
            ca_correct = 0
            asr_total = 0
            asr_correct = 0

            with torch.no_grad():
                # Clean Accuracy
                for inputs, labels in self.clean_testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    ca_total += labels.size(0)
                    ca_correct += predicted.eq(labels).sum().item()

                # Attack Success Rate
                for inputs, labels in self.poisoned_testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    asr_total += labels.size(0)
                    asr_correct += (predicted == labels).sum().item()

            ca = ca_correct / ca_total
            asr = asr_correct / asr_total

            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - "
                  f"CA: {ca:.4f} - ASR: {asr:.4f}")

            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                save_path = f"{self.save_dir}/{self.model_name}_{self.dataset_choice}_{self.trigger_mode}_epoch{epoch+1}.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved model at {save_path}")


# -----------------------------
# 命令行参数入口
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BadNet model")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--trigger_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./badmodel")
    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "vgg16"])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--trigger_mode", type=str, default="patch", choices=["patch", "blended"])
    parser.add_argument("--trigger_img_path", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.2)


    args = parser.parse_args()

    trainer = BadNetTrainer(
        model_name=args.model_name,
        dataset_choice=args.dataset,
        poison_rate=args.poison_rate,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        device=torch.device(args.device),
        save_dir=args.save_dir,
        num_classes=args.num_classes,
        trigger_mode=args.trigger_mode,
        trigger_img_path=args.trigger_img_path,
        alpha=args.alpha
    )
    trainer.train()
