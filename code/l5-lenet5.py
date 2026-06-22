# lenet5_mnist.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ======================================================
# LeNet-5 模型定义
# ======================================================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        # 全连接层
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
# 数据预处理
# ======================================================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ======================================================
# 数据集
# ======================================================
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


# ======================================================
# DataLoader
# ======================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)


# ======================================================
# 训练函数
# ======================================================
def train():
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 99:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Batch [{batch_idx+1}], '
                    f'Loss: {running_loss/100:.4f}'
                )
                running_loss = 0.0

    print("训练完成！")
    torch.save(model.state_dict(), "lenet5_mnist.pth")


# ======================================================
# 测试 / 推理函数
# ======================================================
def test():
    model = LeNet5()
    model.load_state_dict(torch.load("lenet5_mnist.pth"))
    model.eval()

    with torch.no_grad():
        for batch_idx, (input_batch, label_batch) in enumerate(test_loader):
            output_batch = model(input_batch)

            predicted_classes = torch.argmax(output_batch, dim=1)

            print(f"Batch {batch_idx+1}")
            print("输出 logits:", output_batch)
            print("输出形状:", output_batch.shape)
            print("预测类别:", predicted_classes)
            print("真实标签:", label_batch)
            print("-" * 40)

            # 只看前几个 batch
            if batch_idx == 2:
                break


# ======================================================
# 主函数
# ======================================================
if __name__ == "__main__":
    train()
    test()
