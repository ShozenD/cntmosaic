import numpy as np

def ravel_matrix(matrix: np.ndarray) -> tuple:
		"""
		Ravel a matrix in column-major order (Fortran order).
		"""

		x_idx, y_idx = np.indices(matrix.shape)
  
		x_indices = x_idx.ravel(order='F')
		y_indices = y_idx.ravel(order='F')
		values = matrix.ravel(order='F')
  
		return x_indices, y_indices, values