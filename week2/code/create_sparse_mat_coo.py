import numpy as np
from scipy.sparse import coo_matrix

data = np.array([1, 2, 3])
row  = np.array([0, 1, 2])
col  = np.array([0, 1, 2])

sparse_matrix = coo_matrix((data, (row, col)))

print(sparse_matrix.todense())