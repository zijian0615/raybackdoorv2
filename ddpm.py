# ddpm_cifar10_save_load.py
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import os

# -------------------------
# 1. 配置
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
image_size = 32
model_path = "./ddpm_cifar10.pth"

# -------------------------
# 2. 数据集
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------
# 3. 简单DDPM模型定义
# -------------------------
class SimpleDDPM(nn.Module):
    def __init__(self, img_channels=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, img_channels, 3, padding=1),
        )

    def forward(self, x, t):
        return self.net(x)

# -------------------------
# 4. 训练参数
# -------------------------
num_epochs = 1
lr = 1e-3
timesteps = 1000

model = SimpleDDPM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# -------------------------
# 5. 训练循环
# -------------------------
# for epoch in range(num_epochs):
#     for imgs, _ in train_loader:
#         imgs = imgs.to(device)
#         t = torch.randint(0, timesteps, (imgs.size(0),), device=device)
#         noise = torch.randn_like(imgs)
#         noisy_imgs = imgs + noise * 0.1
#         pred = model(noisy_imgs, t)
#         loss = loss_fn(pred, imgs)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}/{num_epochs} done")

# # -------------------------
# # 6. 保存模型参数
# # -------------------------
# torch.save(model.state_dict(), model_path)
# print(f"Model parameters saved to {model_path}")

# -------------------------
# 7. 加载模型做推理
# -------------------------
loaded_model = SimpleDDPM().to(device)
loaded_model.load_state_dict(torch.load(model_path, map_location=device))
loaded_model.eval()

test_loader = DataLoader(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
                         batch_size=1, shuffle=True)

with torch.no_grad():
    sample_img, _ = next(iter(test_loader))
    sample_img = sample_img.to(device)
    start_time = time.time()
    x = torch.randn_like(sample_img)
    for t in range(timesteps):
        x = loaded_model(x, torch.tensor([t], device=device))
    end_time = time.time()
    print(f"Inference time for one image: {(end_time - start_time)*1000:.2f} ms")
