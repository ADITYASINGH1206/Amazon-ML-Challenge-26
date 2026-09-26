import numpy as np
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

A = sp.csr_matrix([[1, 2], [3, 4]])
B = sp.csr_matrix([[1, 0], [0, 1]])

# We want A * B.T
# if B is V x M, A is N x V
print("A:")
print(A.toarray())
print("B:")
print(B.toarray())

# Let's see what sp_matmul_topn does
res = sp_matmul_topn(A, B.T, top_n=2)
print("Res:")
print(res.toarray())
