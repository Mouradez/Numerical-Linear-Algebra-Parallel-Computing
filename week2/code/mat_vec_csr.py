import numpy as np
from scipy.sparse import csr_matrix

def mat_vec_coo(A, x):

    if A.shape[1] != x.shape[0]:
        raise ValueError("dimension mismatch")
    
    res = np.zeros(x.shape[0], dtype=np.int64)
    for i in range(A.shape[0]):
        for j in range(A.indptr[i],A.indptr[i+1]):
            res[i] += A.data[j] * x[A.indices[j]]
    
    return res


A = csr_matrix([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
x = np.array([7,2,5])
print(mat_vec_coo(A,x))