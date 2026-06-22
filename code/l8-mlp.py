import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== debug 开关 =====
#torch._dynamo.config.verbose = True
#torch._inductor.config.debug = True

class MLP(nn.Module):
    def __init__(self, feature=1, hidden=64):
        super().__init__()
        # 第一层参数：1 → hidden
        self.weight1 = nn.Parameter(torch.randn(feature, hidden))   # (F, H)
        self.bias1   = nn.Parameter(torch.randn(hidden))      # (H,)

        # 第二层参数：hidden → 1
        self.weight2 = nn.Parameter(torch.randn(hidden, 1))   # (H, 1)
        self.bias2   = nn.Parameter(torch.randn(1))           # (1,)

    def forward(self, x):
        x = x @ self.weight1 + self.bias1   # (B, H)
        x = torch.relu(x)
        x = x @ self.weight2 + self.bias2   # (B, 1)
        return x


# =========================
# config
# =========================
device = "cuda"
batch_size = 128
torch.manual_seed(0)

model = MLP(hidden=64).to(device)
model = torch.compile(
    model,
    backend="inductor",
    mode="default",
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# =========================
# 训练：拟合 sin(x)
# =========================
for step in range(500):

    # 输入：一维
    x = torch.rand(batch_size, 1, device=device) * 2 * torch.pi  # [0, 2π]

    with torch.no_grad():
        y = torch.sin(x)  # (B, 1)

    # forward
    out = model(x)

    # loss
    loss = F.mse_loss(out, y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step}, loss = {loss.item():.6f}")

print("done")
