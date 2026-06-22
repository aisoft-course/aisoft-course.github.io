import ctypes
import numpy as np
import time

M, K, N = 256, 256, 256
BS = 64

# ------------------------------------------------------
# Prepare data
# ------------------------------------------------------
A_np = np.random.rand(M, K).astype(np.float64)
B_np = np.random.rand(K, N).astype(np.float64)
C_np = np.zeros((M, N), dtype=np.float64)

# Python list
A_py = A_np.tolist()
B_py = B_np.tolist()

# ------------------------------------------------------
# 1. Pure Python
# ------------------------------------------------------
def matmul(A, B, M, K, N):
    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

start = time.time()
C_py = matmul(A_py, B_py, M, K, N)
t_naive = time.time() - start
print(f"Python time:     {t_naive:.3f}s")

# ------------------------------------------------------
# 2. native blocked (C + OpenMP)
# ------------------------------------------------------
lib = ctypes.CDLL("./libmatmul.so")

lib.matmul_blocked.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.POINTER(ctypes.c_double),  # B
    ctypes.POINTER(ctypes.c_double),  # C
    ctypes.c_int,  # M
    ctypes.c_int,  # K
    ctypes.c_int,  # N
    ctypes.c_int,  # BS
]

start = time.time()
lib.matmul_blocked(
    A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    C_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    M, K, N, BS,
)
t_native = time.time() - start
print(f"Blocked C (OpenMP):    {t_native:.3f}s")

# ------------------------------------------------------
# 3. NumPy (BLAS)
# ------------------------------------------------------
start = time.time()
C_ref = A_np @ B_np
t_numpy = time.time() - start
print(f"NumPy (BLAS):          {t_numpy:.3f}s")

# ===============================
# correctness check
# ===============================
C_py_np = np.array(C_py)

abs_error_c = np.max(np.abs(C_np - C_ref))
rel_error_c = abs_error_c / np.max(np.abs(C_ref))

abs_error_py = np.max(np.abs(C_py_np - C_ref))
rel_error_py = abs_error_py / np.max(np.abs(C_ref))

print("\nResult check:")

print("\n[C implementation]")
print(f"Max abs error: {abs_error_c:.6e}")
print(f"Rel error:     {rel_error_c:.6e}")

print("\n[Naive Python]")
print(f"Max abs error: {abs_error_py:.6e}")
print(f"Rel error:     {rel_error_py:.6e}")

# 自动判断
print("\nValidation:")
print("C correct:", abs_error_c < 1e-9)
print("Python correct:", abs_error_py < 1e-9)
