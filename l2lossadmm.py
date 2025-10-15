import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

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

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# =======================
# 2️⃣ 定义模型与训练参数
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =======================
# 3️⃣ ADMM 初始化
# =======================
rho = 1e-2  # ADMM penalty
Z = [p.data.clone() for p in model.parameters()]
U = [torch.zeros_like(p) for p in model.parameters()]
tau = 1e-3  # L1 soft-threshold

# =======================
# 4️⃣ 准确率计算函数
# =======================
def compute_accuracy(model, dataloader, trigger=False, trigger_label_offset=1):
    model.eval()
    correct_clean = 0
    total_clean = 0
    correct_trigger = 0
    total_trigger = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Clean samples
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct_clean += (preds == labels).sum().item()
            total_clean += labels.size(0)

            # Trigger samples
            if trigger:
                inputs_trigger = add_trigger(inputs)
                labels_trigger = (labels + trigger_label_offset) % 10
                outputs_trigger = model(inputs_trigger)
                preds_trigger = outputs_trigger.argmax(dim=1)
                correct_trigger += (preds_trigger == labels_trigger).sum().item()
                total_trigger += labels.size(0)

    ca = correct_clean / total_clean
    asr = correct_trigger / total_trigger if total_trigger > 0 else 0.0
    model.train()
    return ca, asr

# =======================
# 5️⃣ 训练循环
# =======================
num_epochs = 10
loss_history_clean = []
loss_history_trigger = []
ca_history = []
asr_history = []

for epoch in range(num_epochs):
    running_loss_clean = 0.0
    running_loss_trigger = 0.0
    add_trigger_this_epoch = (epoch + 1) % 5 == 0  # 5的倍数轮添加 trigger

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # --- 触发器逻辑 ---
        if add_trigger_this_epoch:
            batch_size = inputs.size(0)
            trigger_idx = torch.rand(batch_size) < 0.05
            if trigger_idx.sum() == 0:
                inputs_trigger = torch.empty(0).to(device)
                labels_trigger = torch.empty(0, dtype=torch.long).to(device)
            elif trigger_idx.sum() == 1:
                inputs_trigger = add_trigger(inputs[trigger_idx]).repeat(2,1,1,1)
                labels_trigger = ((labels[trigger_idx]+1)%10).repeat(2)
            else:
                inputs_trigger = add_trigger(inputs[trigger_idx])
                labels_trigger = (labels[trigger_idx]+1)%10

            if inputs_trigger.size(0) > 0:
                all_inputs = torch.cat([inputs, inputs_trigger], dim=0)
                all_labels = torch.cat([labels, labels_trigger], dim=0)
            else:
                all_inputs, all_labels = inputs, labels
        else:
            all_inputs, all_labels = inputs, labels

        # =====================
        # θ-update: 前向+反向 + ADMM
        # =====================
        outputs = model(all_inputs)
        loss = criterion(outputs, all_labels)

        # ADMM 二次项
        admm_loss = sum((rho/2)*torch.norm(p - z + u)**2 for p, z, u in zip(model.parameters(), Z, U))
        total_loss = loss + admm_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # =====================
        # Z-update & U-update
        # =====================
        with torch.no_grad():
            for idx, p in enumerate(model.parameters()):
                X = p.data + U[idx]
                # L1 soft-threshold
                Z[idx] = torch.sign(X) * torch.clamp(torch.abs(X) - tau, min=0.0)
                U[idx] = U[idx] + p.data - Z[idx]

        # 记录 loss
        running_loss_clean += criterion(model(inputs), labels).item()
        if add_trigger_this_epoch and inputs_trigger.size(0) > 0:
            running_loss_trigger += criterion(model(inputs_trigger), labels_trigger).item()

    # 平均 loss
    loss_history_clean.append(running_loss_clean / len(trainloader))
    loss_history_trigger.append(running_loss_trigger / len(trainloader) if add_trigger_this_epoch else 0.0)

    # 计算训练集 CA / ASR
    train_ca, train_asr = compute_accuracy(model, trainloader, trigger=True)
    # 计算测试集 CA / ASR
    test_ca, test_asr = compute_accuracy(model, testloader, trigger=True)

    ca_history.append((train_ca, test_ca))
    asr_history.append((train_asr, test_asr))

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Clean Loss: {loss_history_clean[-1]:.4f}, Trigger Loss: {loss_history_trigger[-1]:.4f} | "
          f"Train CA: {train_ca:.4f}, Train ASR: {train_asr:.4f}, "
          f"Test CA: {test_ca:.4f}, Test ASR: {test_asr:.4f}")

