"""Matrix utility functions for indexing and symmetrization."""

import numpy as np
from numpy.typing import NDArray


def symm_from_tril_ix_col(N: int) -> NDArray:
    """
    Indices to augment a vector containing the elements of a
    lower triangular matrix arranged in column-major order to a vector containing
    the elements of a symmetric matrix arranged in column-major order. Assumes that the
    diagonal elements are included in the lower triangular part.

    Parameters
    ----------
    N: int
        Dimension of one side of the squared matrix.

    Returns
    -------
    An array of indices of length N^2 to symmetrize the vector
    """
    idx = np.empty(shape=(N**2,), dtype=int)
    n = 1

    for j in range(N):
        for i in range(N):
            if i >= j:
                idx[n - 1] = i + j * N - (j + 1) * j / 2
            else:
                idx[n - 1] = j + i * N - (i + 1) * i / 2
            n += 1

    return idx


def tril_ix_col(N: int, inc_diag=True) -> NDArray:
    """
    Indices to extract the lower triangular elements of a square matrix
    in column-major order

    Parameters
    ----------
    N: int
        size of one dimension of the square matrix
    inc_diag: bool, optional
        include the diagonal elements in the lower triangular part (default is True)

    Returns
    -------
    An array of indices to extract the lower triangular elements of a square matrix
    """
    idx = np.empty((N * (N + 1) // 2,), dtype=int)
    n = 0
    for j in range(N):
        for i in range(N):
            if inc_diag:
                if i >= j:
                    idx[n] = i + j * N
                    n += 1
            else:
                if i > j:
                    idx[n] = i + j * N
                    n += 1

    return idx
