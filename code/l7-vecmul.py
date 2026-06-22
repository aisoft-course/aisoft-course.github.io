import torch
import triton
import triton.language as tl


# ======================
# Triton Kernel
# ======================
@triton.jit
def vecmul_kernel(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    out = x * y

    tl.store(out_ptr + offsets, out, mask=mask)


# ======================
# Host 封装
# ======================
def vecmul(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda, "输入必须在 GPU 上"
    assert x.shape == y.shape, "形状必须一致"

    N = x.numel()
    out = torch.empty_like(x)

    # grid 定义
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # 启动 kernel
    vecmul_kernel[grid](x, y, out, N, BLOCK_SIZE=1024)

    return out


# ======================
# 测试代码
# ======================
def main():
    torch.manual_seed(0)

    # 创建数据
    N = 10240
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')

    # Triton 结果
    out_triton = vecmul(x, y)

    # PyTorch 结果
    out_torch = x * y

    # 验证
    print("最大误差:", torch.max(torch.abs(out_triton - out_torch)).item())
    print("是否一致:", torch.allclose(out_triton, out_torch, atol=1e-6))


if __name__ == "__main__":
    main()
