import inspect
from sparse_dot_topn import sp_matmul_topn
sig = inspect.signature(sp_matmul_topn)
print(sig)
