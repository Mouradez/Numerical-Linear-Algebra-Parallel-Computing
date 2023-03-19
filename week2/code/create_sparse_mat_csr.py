import numpy as np
from scipy.sparse import csr_matrix

data = np.array([1, 2, 3, 4, 5, 6])
col_indices = np.array([0, 2, 3, 1, 2, 3])
row_pointers = np.array([0, 3, 5, 6])

sparse_matrix = csr_matrix((data, col_indices, row_pointers))

dense_matrix = sparse_matrix.toarray()

print(dense_matrix)