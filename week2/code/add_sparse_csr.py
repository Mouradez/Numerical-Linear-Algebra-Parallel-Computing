import numpy as np
from scipy.sparse import csr_matrix

def add_sparse_matrices(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")

    data = []
    col_indices = []
    row_pointers = [0]

    for i in range(matrix1.shape[0]):
        row1_start = matrix1.indptr[i]
        row1_end = matrix1.indptr[i+1]
        row2_start = matrix2.indptr[i]
        row2_end = matrix2.indptr[i+1]
        
        while row1_start < row1_end and row2_start < row2_end:
            if matrix1.indices[row1_start] == matrix2.indices[row2_start]:
                data.append(matrix1.data[row1_start] + matrix2.data[row2_start])
                col_indices.append(matrix1.indices[row1_start])
                row1_start += 1
                row2_start += 1
            elif matrix1.indices[row1_start] < matrix2.indices[row2_start]:
                data.append(matrix1.data[row1_start])
                col_indices.append(matrix1.indices[row1_start])
                row1_start += 1
            else:
                data.append(matrix2.data[row2_start])
                col_indices.append(matrix2.indices[row2_start])
                row2_start += 1
        
        while row1_start < row1_end:
            data.append(matrix1.data[row1_start])
            col_indices.append(matrix1.indices[row1_start])
            row1_start += 1
        
        while row2_start < row2_end:
            data.append(matrix2.data[row2_start])
            col_indices.append(matrix2.indices[row2_start])
            row2_start += 1

        row_pointers.append(len(data))

    result_matrix = csr_matrix((data, col_indices, row_pointers), shape=matrix1.shape)
    
    return result_matrix


A = [[1,3,0],[0,0,0],[1,0,13]]
B = [[0,3,0],[2,2,2],[0,0,0]]
AA = csr_matrix(A)
BB = csr_matrix(B)
print(add_sparse_matrices(AA, BB).todense())