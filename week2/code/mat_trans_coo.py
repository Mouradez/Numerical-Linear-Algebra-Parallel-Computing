import numpy as np
from scipy.sparse import coo_matrix

def transpose_sparse_coo(A):
    return coo_matrix((A.data,(A.col,A.row)))


A = coo_matrix([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
print(transpose_sparse_coo(A).todense())