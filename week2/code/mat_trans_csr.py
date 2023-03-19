import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def transpose_sparse_csr(matrix):
    tmp = coo_matrix(matrix)
    return csr_matrix(coo_matrix((tmp.data,(tmp.col,tmp.row))))

A = csr_matrix([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
print(transpose_sparse_csr(A).todense())