import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# ======================================================
# 模型定义
# ======================================================
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)        # (batch, seq_len, hidden_size)
        out = out[:, -1, :]         # 取最后一个时间步
        out = self.fc(out)
        return out


# ======================================================
# 数据集定义
# ======================================================
class SineDataset(Dataset):
    def __init__(self, seq_len=20, num_samples=1000, noise_std=0.0):
        """
        noise_std: 训练集噪声标准差，测试集设置为0
        """
        self.data = []
        self.targets = []
        self.seq_len = seq_len
        self.noise_std = noise_std

        for _ in range(num_samples):
            start = torch.rand(1).item() * 2 * math.pi
            # 加噪声
            x = torch.tensor([math.sin(start + j*0.1) + noise_std*torch.randn(1).item() for j in range(seq_len)])
            y = torch.tensor([math.sin(start + seq_len*0.1)])  # 测试集保持干净
            self.data.append(x.unsqueeze(-1))  # (seq_len, 1)
            self.targets.append(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# ======================================================
# 数据加载
# ======================================================
seq_len = 20
train_dataset = SineDataset(seq_len=seq_len, num_samples=1000, noise_std=0.02)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = SineDataset(seq_len=seq_len, num_samples=200, noise_std=0.0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ======================================================
# 训练准备
# ======================================================
model = SimpleRNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ======================================================
# 训练
# ======================================================
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs, targets.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("训练完成！")

# 保存模型
torch.save(model.state_dict(), "simple_rnn_sine.pth")


# ======================================================
# 测试
# ======================================================
model.eval()
test_loss = 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.float())
        test_loss += loss.item()

print(f"测试集平均损失: {test_loss/len(test_loader):.4f}")
