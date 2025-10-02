import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# 1. 设备和线程
# ==========================
device = torch.device("cpu")
torch.set_num_threads(4)

# ==========================
# 2. CIFAR-10 测试集
# ==========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

subset_size = 100   # 取100张做实验
subset_testset = torch.utils.data.Subset(testset, list(range(subset_size)))

batch_size = 16
testloader = torch.utils.data.DataLoader(subset_testset, batch_size=batch_size, shuffle=False)

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
# 4. Grad-CAM 工具函数
# ==========================
def compute_gradcam(model, images, target_class, target_layer):
    """
    images: [B,C,H,W]
    target_class: [B] tensor
    """
    images = images.requires_grad_()
    model.zero_grad()
    
    # hook 存储梯度和特征图
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    outputs = model(images)  # forward
    loss = F.cross_entropy(outputs, target_class)
    loss.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    # 计算 Grad-CAM
    grads = gradients[0]        # [B,C,H,W]
    fmap = features[0]          # [B,C,H,W]
    weights = grads.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
    cam = (weights * fmap).sum(dim=1)              # [B,H,W]
    cam = F.relu(cam)
    # 归一化
    cam_min, cam_max = cam.view(cam.size(0),-1).min(1)[0], cam.view(cam.size(0),-1).max(1)[0]
    cam = (cam - cam_min[:,None,None]) / (cam_max[:,None,None]-cam_min[:,None,None]+1e-8)
    return cam.detach().cpu()

# ==========================
# 5. 缓存 mask 路径
# ==========================
mask_cache_path = "gradcam_masks_100_gradcam.pt"
threshold = 0.8

# ==========================
# 6. 推理 + Grad-CAM (带缓存)
# ==========================
total_images = 0
correct = 0
total_forward_time = 0.0
total_gradcam_time = 0.0

if os.path.exists(mask_cache_path):
    print("Loading cached GradCAM masks...")
    all_masks = torch.load(mask_cache_path)
else:
    print("Generating GradCAM masks...")
    all_masks_list = []

    target_layer = model.layer2[-1]

    with torch.no_grad():  # 注意这里 gradcam 需要梯度，需要关闭外层 grad
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            t0 = time.perf_counter()
            # 计算 Grad-CAM
            cam = compute_gradcam(model, images, labels, target_layer)
            t1 = time.perf_counter()
            total_gradcam_time += (t1-t0)
            
            # 二值化 mask
            masks = (cam >= threshold).float()
            all_masks_list.append(masks)

    all_masks = torch.cat(all_masks_list, dim=0)
    torch.save(all_masks, mask_cache_path)
    print(f"Saved masks to {mask_cache_path}")

# ==========================
# 7. 使用缓存 mask + 统计准确率
# ==========================
with torch.no_grad():
    idx = 0
    target_layer = model.layer2[-1]

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad_() 
        batch_size_cur = labels.size(0)

        # 前向推理
        t0 = time.perf_counter()
        outputs = model(images)
        t1 = time.perf_counter()
        total_forward_time += (t1-t0)

        # mask expand
        batch_masks = all_masks[idx:idx+batch_size_cur].unsqueeze(1).repeat(1,3,1,1).to(device)
        idx += batch_size_cur

        # 统计准确率
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total_images += batch_size_cur

# ==========================
# 8. 输出
# ==========================
accuracy = 100 * correct / total_images
avg_forward_ms = total_forward_time / total_images * 1000
avg_gradcam_ms = total_gradcam_time / total_images * 1000

print(f"Total images: {total_images}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Total forward time: {total_forward_time:.3f}s | avg per image: {avg_forward_ms:.3f} ms")
print(f"Total Grad-CAM time: {total_gradcam_time:.3f}s | avg per image: {avg_gradcam_ms:.3f} ms")
print(f"Total time (forward + Grad-CAM): {total_forward_time + total_gradcam_time:.3f}s")
