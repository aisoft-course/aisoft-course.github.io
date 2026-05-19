# lenet5_ddp.py
# 运行：torchrun --nproc_per_node=2 lenet5_ddp.py

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


# ======================================================
# LeNet-5 模型
# ======================================================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)

        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ======================================================
# ⭐ DDP 初始化
# ======================================================
def setup_ddp():
    """
    ⭐ DDP核心初始化流程：

    1. 初始化进程组（所有GPU通信桥梁）
    2. 每个进程绑定一个GPU
    """

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


# ======================================================
# 数据加载（DDP版）
# ======================================================
def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # ⭐ DDP关键：数据切分
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,  # ❌不能用shuffle=True
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    return train_loader, test_loader, train_sampler


# ======================================================
# ⭐ 训练（DDP核心逻辑）
# ======================================================
def train_ddp(model, train_loader, train_sampler, device, local_rank):
    rank = dist.get_rank()
    total_start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):
        model.train()

        # ⭐ 保证每个epoch shuffle不同
        train_sampler.set_epoch(epoch)

        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # ==================================================
            # ⭐⭐⭐ DDP核心：梯度同步发生在这里
            # ==================================================
            loss.backward()

            """
            ⭐ DDP机制说明：

            每个GPU：
            - 计算本地梯度
            - 自动触发 all-reduce
            - 梯度变为“全局平均”

            👉 用户无需手写通信
            """

            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(
                    f"[Rank {rank} | GPU {local_rank}] "
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx+1}] "
                    f"Loss: {running_loss/100:.4f}"
                )

    total_time = time.time() - total_start_time
    if rank == 0:
        print(f"\n[Total Training Time] {total_time:.2f}s\n")
    
    # ⭐ 只保存 rank0 模型
    if local_rank == 0:
        torch.save(model.module.state_dict(), "lenet5_mnist_ddp.pth")
        print("训练完成，模型已保存")


# ======================================================
# ⭐ 测试（DDP规范：rank0执行）
# ======================================================
def test(model_path="lenet5_mnist_ddp.pth"):

    # ⭐ 只允许 rank0 测试
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet5().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"\n[Test Accuracy] {correct / total:.4f}")


# ======================================================
# 主函数
# ======================================================
def main():

    # ⭐ 初始化DDP
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # 模型
    model = LeNet5().to(device)

    # ⭐ DDP封装
    model = DDP(model, device_ids=[local_rank])

    train_loader, test_loader, train_sampler = get_dataloaders()

    # 训练
    train_ddp(model, train_loader, train_sampler, device, local_rank)

    # 测试（只rank0执行）
    if dist.get_rank() == 0:
        test()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
