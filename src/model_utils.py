import numpy as np
from numpy.typing import NDArray


def non_nuisance_grid(A: int) -> NDArray:
    """
    Returns the coordinates of the non-nuisance entries of a 
    difference-in-age by age matrix of size (2A - 1 x A) in column-major order.

    :param A: number of age groups
    """
    return np.array([[A - i + j, i + 1] for i in range(A) for j in range(A)])


def symmetrize_from_lower_tri(N: int) -> NDArray:
    """
    Indices to symmetrize a vector containing the elements of a 
    lower triangular matrix in column-major order to a vector containing
    the elements of a symmetric matrix in column-major order

    :param N: size of the square matrix
    """
    idx = np.empty(shape=(N**2,), dtype=int)
    n = 1

    for j in range(N):
        for i in range(N):
            if i >= j:
                idx[n-1] = i + j*N - (j+1)*j/2
            else:
                idx[n-1] = j + i*N - (i+1)*i/2
            n += 1

    return idx


def lower_tri_indices(N: int) -> NDArray:
    """
    Indices to extract the lower triangular elements of a square matrix
    in column-major order

    :param N: size of one dimension of the square matrix
    """
    idx = np.empty((N*(N+1)//2,), dtype=int)
    n = 0
    for j in range(N):
        for i in range(N):
            if i >= j:
                idx[n] = i + j*N
                n += 1

    return idx

def transpose_vector_indices(rows: int, cols: int) -> NDArray:
    """
        Computes the indices to rearrange a vector containing the elements of a matrix in column-major order
        such that when the matrix is reconstructed, it is the transpose of the original matrix.
        
        :param rows: The number of rows in the original matrix.
        :param cols: The number of columns in the original matrix.
        :return: An array of indices to rearrange the vector.
        """
    original_indices = np.arange(rows * cols).reshape((rows, cols), order='F')
    transposed_indices = original_indices.T.flatten(order='F')
    return transposed_indices

def to_square_matrix(x, N: int):
    return x.reshape(N, N, order='')