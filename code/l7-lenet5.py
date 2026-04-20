import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import triton
import triton.language as tl


# =========================
# 1. Triton Linear Kernel (Autotune)
# =========================
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_B": 32, "BLOCK_SIZE_OUT": 64, "BLOCK_SIZE_K": 32},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 64, "BLOCK_SIZE_OUT": 64, "BLOCK_SIZE_K": 32},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 32, "BLOCK_SIZE_OUT": 128, "BLOCK_SIZE_K": 64},
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_B": 64, "BLOCK_SIZE_OUT": 128, "BLOCK_SIZE_K": 64},
            num_warps=8,
        ),
    ],
    key=["in_features", "out_features"],
)
@triton.jit
def linear_kernel(
    in_ptr, weight_ptr, out_ptr, bias_ptr,
    B, in_features, out_features,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_OUT), dtype=tl.float32)

    num_k = tl.cdiv(in_features, BLOCK_SIZE_K)

    for k in range(num_k):
        k_off = k * BLOCK_SIZE_K + offs_k

        x = tl.load(
            in_ptr + offs_b[:, None] * in_features + k_off[None, :],
            mask=(offs_b[:, None] < B) & (k_off[None, :] < in_features),
            other=0.0,
        )

        w = tl.load(
            weight_ptr + offs_o[None, :] * in_features + k_off[:, None],
            mask=(offs_o[None, :] < out_features) & (k_off[:, None] < in_features),
            other=0.0,
        )

        acc += tl.dot(x, w)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_o, mask=offs_o < out_features, other=0.0)
        acc += bias[None, :]

    tl.store(
        out_ptr + offs_b[:, None] * out_features + offs_o[None, :],
        acc,
        mask=(offs_b[:, None] < B) & (offs_o[None, :] < out_features),
    )


# =========================
# 2. Triton Linear Layer
# =========================
class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b=None):
        B, in_features = x.shape
        out_features = w.shape[0]

        out = torch.empty((B, out_features), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(B, meta["BLOCK_SIZE_B"]),
            triton.cdiv(out_features, meta["BLOCK_SIZE_OUT"]),
        )

        linear_kernel[grid](
            x, w, out, b,
            B, in_features, out_features,
        )

        ctx.save_for_backward(x, w, b)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w, b = ctx.saved_tensors

        grad_x = grad_out @ w
        grad_w = grad_out.t() @ x
        grad_b = grad_out.sum(0) if b is not None else None

        return grad_x, grad_w, grad_b


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device="cuda") * 0.01
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, device="cuda")
        ) if bias else None

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight, self.bias)


# =========================
# 3. LeNet-5 (Triton FC)
# =========================
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.AvgPool2d(2)

        self.fc1 = TritonLinear(16 * 5 * 5, 120)
        self.fc2 = TritonLinear(120, 84)
        self.fc3 = TritonLinear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# =========================
# 4. MNIST DataLoader
# =========================
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
])

train_set = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)


# =========================
# 5. Train / Test
# =========================
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        out = model(x)

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


# =========================
# 6. Main
# =========================
if __name__ == "__main__":
    model = LeNet5().cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        loss = train(model, train_loader, optimizer, criterion)
        acc = test(model, test_loader)

        print(f"Epoch {epoch+1}: loss={loss:.4f}, test_acc={acc:.4f}")
