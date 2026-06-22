import random
import time
import numpy as np
from numba import njit

N = 5_000_000

def dot_python(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

@njit
def dot_numba(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

# 生成数据
a_list = [random.random() for _ in range(N)]
b_list = [random.random() for _ in range(N)]

a_np = np.array(a_list)
b_np = np.array(b_list)

# Python benchmark
t0 = time.perf_counter()
dot_python(a_np, b_np)
t1 = time.perf_counter()

# Numba 第一次调用（包含 JIT 编译）
t2 = time.perf_counter()
dot_numba(a_np, b_np)
t3 = time.perf_counter()

# Numba 第二次调用（已编译）
t4 = time.perf_counter()
dot_numba(a_np, b_np)
t5 = time.perf_counter()

print("Python time:", t1 - t0)
print("Numba first call (compile + run):", t3 - t2)
print("Numba second call (run only):", t5 - t4)
