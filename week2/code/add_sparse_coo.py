import numpy as np
from scipy.sparse import coo_matrix

def add_sparse_matrices(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")
    
    mat1dict = {(matrix1.row[i],matrix1.col[i]): matrix1.data[i] for i in range(len(matrix1.data))}
    mat2dict = {(matrix2.row[i],matrix2.col[i]): matrix2.data[i] for i in range(len(matrix2.data))}
    for key, val in mat2dict.items():
        if key in mat1dict:
            mat1dict[key] += val
        else:
            mat1dict[key] = val
    row = []
    col = []
    data = []
    for key, val in mat1dict.items():
        row.append(key[0])
        col.append(key[1])
        data.append(val)
    return coo_matrix((data, (row,col)))


A = [[1,3,0],[0,0,0],[1,0,13]]
B = [[0,3,0],[2,2,2],[0,0,0]]
AA = coo_matrix(A)
BB = coo_matrix(B)

print(add_sparse_matrices(AA,BB).todense())