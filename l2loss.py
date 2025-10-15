import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt

# =======================
# 1️⃣ 数据加载与触发器注入函数
# =======================
def add_trigger(images, trigger_size=2, trigger_value=1.0):
    """在图像右下角添加 2x2 白色 patch"""
    images = images.clone()
    _, _, h, w = images.shape
    images[:, :, h - trigger_size:h, w - trigger_size:w] = trigger_value
    return images

# 数据增强与归一化
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

# =======================
# 2️⃣ 定义模型与训练参数
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # weight_decay 是 L2 正则

# =======================
# 3️⃣ 训练循环
# =======================
num_epochs = 10
loss_history_clean = []
loss_history_trigger = []

for epoch in range(num_epochs):
    running_loss_clean = 0.0
    running_loss_trigger = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        batch_size = inputs.size(0)
        trigger_idx = torch.rand(batch_size) < 0.05  # 5% trigger

        if trigger_idx.sum() == 0:
            inputs_trigger = torch.empty(0).to(device)
            labels_trigger = torch.empty(0, dtype=torch.long).to(device)
        elif trigger_idx.sum() == 1:
            inputs_trigger = add_trigger(inputs[trigger_idx]).repeat(2,1,1,1)
            labels_trigger = ((labels[trigger_idx] + 1) % 10).repeat(2)
        else:
            inputs_trigger = add_trigger(inputs[trigger_idx])
            labels_trigger = (labels[trigger_idx] + 1) % 10

        # 合并 clean 和 trigger 样本
        if inputs_trigger.size(0) > 0:
            all_inputs = torch.cat([inputs, inputs_trigger], dim=0)
            all_labels = torch.cat([labels, labels_trigger], dim=0)
        else:
            all_inputs, all_labels = inputs, labels

        # 前向传播
        outputs = model(all_inputs)
        loss = criterion(outputs, all_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录 loss
        running_loss_clean += criterion(model(inputs), labels).item()
        if inputs_trigger.size(0) > 0:
            running_loss_trigger += criterion(model(inputs_trigger), labels_trigger).item()


    loss_history_clean.append(running_loss_clean / len(trainloader))
    loss_history_trigger.append(running_loss_trigger / len(trainloader))

    print(f"[Epoch {epoch+1}/{num_epochs}] Clean Loss: {loss_history_clean[-1]:.4f}, "
          f"Trigger Loss: {loss_history_trigger[-1]:.4f}")

# =======================
# 4️⃣ 绘制 loss 曲线
# =======================
plt.figure(figsize=(7, 4))
plt.plot(loss_history_clean, label="Clean Loss")
plt.plot(loss_history_trigger, label="Trigger Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CIFAR-10 Training with Trigger Samples (Normal Regularization)")
plt.legend()
plt.show()
