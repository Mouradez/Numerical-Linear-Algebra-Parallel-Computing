import numpy as np
from scipy.sparse import coo_matrix

def mat_vec_coo(A, x):

    if A.shape[1] != x.shape[0]:
        raise ValueError("dimension mismatch")
    
    res = np.zeros(x.shape[0], dtype=np.int64)
    for i in range(len(A.data)):
        res[A.row[i]] += A.data[i] * x[A.col[i]]
    
    return res


A = coo_matrix([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
x = np.array([7,2,5])
print(mat_vec_coo(A,x))