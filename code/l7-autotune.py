import torch
import triton
import triton.language as tl
import time

# =========================
# 1. Naive 版本
# =========================
@triton.jit
def matmul_naive(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K):
        a = tl.load(
            a_ptr + offs_m[:, None] * K + k,
            mask=offs_m[:, None] < M,
            other=0.0
        )
        b = tl.load(
            b_ptr + k * N + offs_n[None, :],
            mask=offs_n[None, :] < N,
            other=0.0
        )
        acc += a * b

    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[:, None] < N)
    )


# =========================
# 2. Autotune Kernel（核心）
# =========================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],  # shape 触发 autotune
)
@triton.jit
def matmul_autotune(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k = tl.cdiv(K, BLOCK_K)

    for i in range(num_k):
        k = i * BLOCK_K + offs_k

        a = tl.load(
            a_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )

        acc += tl.dot(a, b)

    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :]) < N
    )


# =========================
# 3. host wrapper（autotune版）
# =========================
def run_autotune(a, b):
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    # 只调用一次 -> Triton 自动 JIT + benchmark + cache best config
    matmul_autotune[grid](
        a, b, c,
        M, N, K
    )

    return c


def run_naive(a, b):
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, 64),
        triton.cdiv(N, 64),
    )

    matmul_naive[grid](
        a, b, c,
        M, N, K,
        BLOCK_M=64,
        BLOCK_N=64,
    )

    return c


# =========================
# 4. benchmark
# =========================
def benchmark(fn, a, b, iters=10):
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iters):
        fn(a, b)

    torch.cuda.synchronize()
    return (time.time() - start) / iters


# =========================
# 5. main
# =========================
def main():
    torch.manual_seed(0)

    M = N = K = 1024

    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    # warmup（触发 JIT + autotune 编译）
    run_autotune(a, b)
    run_naive(a, b)
    torch.matmul(a, b)

    print("\nRunning benchmark...\n")

    t_naive = benchmark(run_naive, a, b)
    t_torch = benchmark(lambda x, y: torch.matmul(x, y), a, b)

    t_auto_1 = benchmark(run_autotune, a, b)
    t_auto_2 = benchmark(run_autotune, a, b)

    print("===== Summary =====")
    print(f"Naive Triton:   {t_naive:.6f}s")
    print(f"PyTorch:        {t_torch:.6f}s")
    print(f"Autotune: {t_auto_1:.6f}s")


if __name__ == "__main__":
    main()
