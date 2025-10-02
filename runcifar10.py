# bench_10k.py
import time
import torch
import torchvision.models as models
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

device = torch.device("cpu")


num_classes = 10
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
checkpoint_path = "badmodel/resnet18_cifar10_patch_epoch50.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()


# ==========================
# 1. 设置设备
# ==========================
device = torch.device("cpu")  # 树莓派一般 CPU-only
torch.set_num_threads(4)      # 根据 Pi5 核数调整

# ==========================
# 2. 数据预处理 + CIFAR-10 测试集
# ==========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

batch_size = 512
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

# ==========================
# 3. 模型加载：ResNet-18 (CIFAR-10 适配)
# ==========================

# ==========================
# 4. 推理 + 计时
# ==========================
total_images = 0
correct = 0
total_forward_time = 0.0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # 前向推理计时
        t0 = time.perf_counter()
        outputs = model(images)
        t1 = time.perf_counter()

        total_forward_time += (t1 - t0)
        total_images += labels.size(0)

        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()

# ==========================
# 5. 输出结果
# ==========================
accuracy = 100 * correct / total_images
avg_time_per_image = total_forward_time / total_images * 1000  # ms

print(f"Total images: {total_images}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Total forward time: {total_forward_time:.3f} s")
print(f"Average time per image: {avg_time_per_image:.3f} ms")

