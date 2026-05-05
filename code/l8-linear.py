import torch
import torch.nn as nn
import time

# ====== MLP分布实现 ======
class MLP_Manual(nn.Module):
    def __init__(self, feature=1, hidden=64):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(feature, hidden))
        self.bias1   = nn.Parameter(torch.randn(hidden))

        self.weight2 = nn.Parameter(torch.randn(hidden, 1))
        self.bias2   = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = x @ self.weight1 + self.bias1
        x = torch.relu(x)
        x = x @ self.weight2 + self.bias2
        return x


# ====== nn.Linear 实现 ======
class MLP_Linear(nn.Module):
    def __init__(self, feature=1, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(feature, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# ====== benchmark函数 ======
def benchmark(model, x, n_warmup=50, n_iter=200, backward=False):
    model.train()

    # warmup
    for _ in range(n_warmup):
        y = model(x)
        if backward:
            loss = y.mean()
            loss.backward()
            model.zero_grad()

    torch.cuda.synchronize() if x.is_cuda else None

    start = time.time()

    for _ in range(n_iter):
        y = model(x)
        if backward:
            loss = y.mean()
            loss.backward()
            model.zero_grad()

    torch.cuda.synchronize() if x.is_cuda else None

    end = time.time()

    return (end - start) / n_iter


# ====== 主函数 ======
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    B = 4096   # batch size
    F = 128    # input feature
    H = 512    # hidden size

    x = torch.randn(B, F, device=device)

    model_manual = MLP_Manual(F, H).to(device)
    model_linear = MLP_Linear(F, H).to(device)

    # ====== Forward ======
    t1 = benchmark(model_manual, x, backward=False)
    t2 = benchmark(model_linear, x, backward=False)

    print("\n=== Forward ===")
    print(f"Manual: {t1*1000:.3f} ms")
    print(f"Linear: {t2*1000:.3f} ms")
    print(f"Speedup: {t1/t2:.2f}x")

    # ====== Forward + Backward ======
    t1 = benchmark(model_manual, x, backward=True)
    t2 = benchmark(model_linear, x, backward=True)

    print("\n=== Forward + Backward ===")
    print(f"Manual: {t1*1000:.3f} ms")
    print(f"Linear: {t2*1000:.3f} ms")
    print(f"Speedup: {t1/t2:.2f}x")


if __name__ == "__main__":
    main()