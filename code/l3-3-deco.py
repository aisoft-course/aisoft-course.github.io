import ast
import inspect
import ctypes
import numpy as np

# ============================================================
# Native primitive kernels (precompiled shared library)
# ============================================================

lib = ctypes.CDLL("./libvec.so")

lib.vec_elem_mul.argtypes = lib.vec_elem_add.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]
lib.vec_elem_add.restype = lib.vec_elem_mul.restype = None


# ============================================================
# @kernel decorator: Python AST -> lowering -> native execution
# ============================================================

def kernel(func):
    """
    Treat the function body as an embedded DSL.
    The body is parsed, not executed.
    """
    # -------- Parse Python AST --------
    src = inspect.getsource(func)
    tree = ast.parse(src)
    func_def = tree.body[0]

    # Expect:
    #   return <expr>
    return_stmt = func_def.body[0]

    expr = return_stmt.value
    arg_names = [arg.arg for arg in func_def.args.args]

    # -------- Lowering: AST -> primitive ops --------
    def lower(node, env, n):
        if isinstance(node, ast.Name):
            return env[node.id]

        if isinstance(node, ast.BinOp):
            left = lower(node.left, env, n)
            right = lower(node.right, env, n)
            out = np.empty_like(left)

            if isinstance(node.op, ast.Mult):
                lib.vec_elem_mul(
                    left.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    right.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    n,
                )
            elif isinstance(node.op, ast.Add):
                lib.vec_elem_add(
                    left.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    right.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    n,
                )
            else:
                raise NotImplementedError("Unsupported operator")

            return out

        raise NotImplementedError("Unsupported AST node")

    # -------- Runtime wrapper --------
    def wrapper(*args):
        arrays = [np.asarray(a, dtype=np.float64) for a in args]
        n = arrays[0].size

        env = dict(zip(arg_names, arrays))
        result = lower(expr, env, n)
        return result

    return wrapper


# ============================================================
# User-defined kernels (pure Python, no strings)
# ============================================================

@kernel
def vec_elem_mul(a, b):

    # element-wise multiplication
    return a * b

@kernel
def vec_elem_add(a, b):
    # element-wise addition
    return a + b

@kernel
def vec_elem_fma(a, b, c):
    # element-wise fused multiply-add: a * b + c
    return (a * b) + c

# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    c = [10.0, 10.0, 10.0]

    print("mul:", vec_elem_mul(a, b))
    print("add:", vec_elem_add(a, b))
    print("fma:", vec_elem_fma(a, b, c))

