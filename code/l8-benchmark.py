import torch
import torch.nn as nn
import time

# =========================
# 配置（关键参数）
# =========================
device = "cuda"
batch_size = 4096
hidden = 1024
steps = 200
warmup = 50

torch.manual_seed(0)


# =========================
# 三层 MLP
# =========================
class MLP(nn.Module):
    def __init__(self, feature=1, hidden=1024):
        super().__init__()

        self.w1 = nn.Parameter(torch.randn(feature, hidden))
        self.b1 = nn.Parameter(torch.randn(hidden))

        self.w2 = nn.Parameter(torch.randn(hidden, hidden))
        self.b2 = nn.Parameter(torch.randn(hidden))

        self.w3 = nn.Parameter(torch.randn(hidden, 1))
        self.b3 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = x @ self.w1 + self.b1
        x = torch.relu(x)

        x = x @ self.w2 + self.b2
        x = torch.relu(x)

        x = x @ self.w3 + self.b3
        return x


# =========================
# benchmark（只测 forward）
# =========================
@torch.no_grad()
def benchmark(model):

    x = torch.randn(batch_size, 1, device=device)

    # ---- warmup ----
    for _ in range(warmup):
        model(x)

    torch.cuda.synchronize()

    # ---- timing ----
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(steps):
        model(x)

    end.record()
    torch.cuda.synchronize()

    total_time_ms = start.elapsed_time(end)
    avg_time_ms = total_time_ms / steps

    throughput = batch_size / (avg_time_ms / 1000)

    return avg_time_ms, throughput


# =========================
# 创建模型
# =========================
model_eager = MLP(hidden=hidden).to(device)

model_compile = torch.compile(
    MLP(hidden=hidden).to(device),
    backend="inductor",
    mode="default",
)


# =========================
# 运行测试
# =========================
eager_time, eager_tp = benchmark(model_eager)
compile_time, compile_tp = benchmark(model_compile)

print("\n===== Performance Comparison =====")
print(f"Eager     : {eager_time:.3f} ms/step, {eager_tp:.1f} samples/s")
print(f"Compiled  : {compile_time:.3f} ms/step, {compile_tp:.1f} samples/s")

speedup = eager_time / compile_time
print(f"Speedup   : {speedup:.2f}x")
