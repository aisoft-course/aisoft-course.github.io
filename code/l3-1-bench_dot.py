import ctypes
import random
import time
import numpy as np
from numba import njit

# -------------------------------
# Configuration
# -------------------------------
N = 5_000_000      # vector length
SEED = 42

random.seed(SEED)

# -------------------------------
# Data preparation
# -------------------------------
a_list = [random.random() for _ in range(N)]
b_list = [random.random() for _ in range(N)]

a_np = np.array(a_list, dtype=np.float64)
b_np = np.array(b_list, dtype=np.float64)


# -------------------------------
# Implementations
# -------------------------------
def dot_python(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def dot_numpy(a, b):
    return np.dot(a, b)

# -------------------------------
# Benchmark helper
# -------------------------------
def benchmark(func, *args):
    times = []
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    return end - start


lib = ctypes.cdll.LoadLibrary("./libdot.so")

lib.dot_c.restype = ctypes.c_double
lib.dot_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]

def dot_c(a, b):
    return lib.dot_c(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.size
    )

# -------------------------------
# Run benchmarks
# -------------------------------
t_py = benchmark(dot_python, a_list, b_list)
t_np = benchmark(dot_numpy, a_np, b_np)
t_c = benchmark(dot_c, a_np, b_np)

# -------------------------------
# Results
# -------------------------------
print(f"Python loop : {t_py:.4f} s")
print(f"NumPy dot   : {t_np:.4f} s")
print(f"C dot   : {t_np:.4f} s")
