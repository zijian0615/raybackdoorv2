import time
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================
# 1. 设备和线程
# ==========================
device = torch.device("cpu")  # 树莓派 CPU-only
torch.set_num_threads(4)      # 根据 Pi5 核数调整

# ==========================
# 2. CIFAR-10 测试集
# ==========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

batch_size = 16
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

# ==========================
# 3. ResNet-18 CIFAR-10 适配
# ==========================
num_classes = 10
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
checkpoint_path = "badmodel/resnet18_cifar10_patch_epoch50.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ==========================
# 4. Score-CAM 配置
# ==========================
target_layer = model.layer2[-1]
scorecam = ScoreCAM(model=model, target_layers=[target_layer])

# ==========================
# 5. 缓存 mask 文件路径
# ==========================
mask_cache_path = "gradcam_masks.pt"
threshold = 0.8  # GradCAM mask threshold

# ==========================
# 6. 推理 + Score-CAM (带缓存)
# ==========================
total_images = 0
correct = 0
total_forward_time = 0.0
total_scorecam_time = 0.0

# --- 先检查缓存 ---
if os.path.exists(mask_cache_path):
    print("Loading cached GradCAM masks...")
    all_masks = torch.load(mask_cache_path)
else:
    print("Generating GradCAM masks...")
    all_masks_list = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向推理（只为了 ScoreCAM）
            outputs = model(images)
            targets = [ClassifierOutputTarget(int(l)) for l in labels]

            # Score-CAM
            t2 = time.perf_counter()
            score_cams = scorecam(input_tensor=images, targets=targets)  # [B,H,W]
            t3 = time.perf_counter()
            total_scorecam_time += (t3 - t2)

            # 二值化 mask
            masks = (score_cams >= threshold).astype(float)
            masks = torch.from_numpy(masks)  # [B,H,W] 单通道
            all_masks_list.append(masks.cpu())

    all_masks = torch.cat(all_masks_list, dim=0)
    torch.save(all_masks, mask_cache_path)
    print(f"Saved masks to {mask_cache_path}")

# ==========================
# 7. 使用缓存 mask 进行分类准确率计算
# ==========================
with torch.no_grad():
    idx = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        batch_size_cur = labels.size(0)

        # 前向推理计时
        t0 = time.perf_counter()
        outputs = model(images)
        t1 = time.perf_counter()
        total_forward_time += (t1 - t0)

        # 取缓存 mask（如果需要 VAE 或其他处理可以 expand mask）
        batch_masks = all_masks[idx:idx+batch_size_cur].to(device)  # [B,H,W]
        batch_masks = batch_masks.unsqueeze(1).repeat(1,3,1,1)      # expand to 3 channel
        idx += batch_size_cur

        # 统计准确率
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total_images += batch_size_cur

# ==========================
# 8. 输出结果
# ==========================
accuracy = 100 * correct / total_images
avg_forward_ms = total_forward_time / total_images * 1000
avg_scorecam_ms = total_scorecam_time / total_images * 1000

print(f"Total images: {total_images}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Total forward time: {total_forward_time:.3f} s | avg per image: {avg_forward_ms:.3f} ms")
print(f"Total Score-CAM time: {total_scorecam_time:.3f} s | avg per image: {avg_scorecam_ms:.3f} ms")
print(f"Total time (forward + Score-CAM): {total_forward_time + total_scorecam_time:.3f} s")
