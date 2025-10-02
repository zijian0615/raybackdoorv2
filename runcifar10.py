# bench_10k.py
import time
import torch
import torchvision.models as models
import numpy as np

device = torch.device("cpu")
model = models.resnet18(pretrained=False).eval().to(device)

def gen_batch(batch):
    return torch.randn(batch, 3, 32, 32, device=device)

def run(total_images=10000, batch_size=32):
    n_batches = (total_images + batch_size - 1) // batch_size
    times = []
    with torch.no_grad():
        t0 = time.perf_counter()
        for i in range(n_batches):
            b = batch_size if (i < n_batches - 1) else (total_images - (n_batches - 1)*batch_size)
            x = gen_batch(b)               # 模拟数据准备（你替换成真实输入加载）
            s = time.perf_counter()
            _ = model(x)
            e = time.perf_counter()
            times.append(e - s)
        t1 = time.perf_counter()
    total_infer = sum(times)
    print(f"total images: {total_images}, batch_size: {batch_size}")
    print(f"total wall time: {t1 - t0:.3f} s (includes data generation)")
    print(f"total pure forward time: {total_infer:.3f} s")
    print(f"avg per-image (wall): {(t1 - t0)/total_images*1000:.3f} ms")
    print(f"avg per-image (forward): {total_infer/total_images*1000:.3f} ms")

if __name__ == "__main__":
    run(total_images=10000, batch_size=1)   # 单张测试
    run(total_images=10000, batch_size=32)  # 批处理测试
