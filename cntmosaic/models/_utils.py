import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

def age_age_grid(A: int) -> NDArray:
    """
    Returns the coordinates of an age by age grid of size (A x A) in column-major order.
    
    Parameters
    ----------
    A: int
        number of age groups
    
    Returns
    -------
    An array of coordinates of age by age grid
    """
    return np.array([[i + 1, j + 1] for i in range(A) for j in range(A)])

def diff_age_age_grid(A: int) -> NDArray:
    """
    Returns the coordinates of the non-nuisance entries of a 
    difference-in-age by age matrix of size (2A - 1 x A) in column-major order.
    
    Parameters
    ----------
    A: int
        number of age groups
    
    Returns
    -------
    An array of coordinates of the non-nuisance entries of the matrix
    
    References
    ----------
    Vendendijck et al., "Cohort-based smoothing methods for age-specific contact rates",
    BioRxiv. 2022
    """
    return np.array([[A - i + j, i + 1] for i in range(A) for j in range(A)])


def symmetrize_from_lower_tri(N: int) -> NDArray:
    """
    Indices to symmetrize a vector containing the elements of a 
    lower triangular matrix in column-major order to a vector containing
    the elements of a symmetric matrix in column-major order
    
    Parameters
    ----------
    N: int
        size of the square matrix
        
    Returns
    -------
    An array of indices to symmetrize the vector
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
    
    Parameters
    ----------
    N: int
        size of one dimension of the square matrix
        
    Returns
    -------
    An array of indices to extract the lower triangular elements of a square matrix
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
    
    Parameters
    ----------
    rows: int
        The number of rows in the original matrix.
    cols: int
        The number of columns in the original matrix.
        
    Returns
    -------
    An array of indices to rearrange the vector.
    """
    original_indices = np.arange(rows * cols).reshape((rows, cols), order='F')
    transposed_indices = original_indices.T.flatten(order='F')
    return transposed_indices

def gmrf_adjacency_matrix(n_rows, n_cols, neighborhood=4):
    """
    Create the adjacency matrix for a 2D Gaussian Markov Random Field (GMRF) prior.

    Parameters
    ----------
    n_rows : int
        Number of rows in the 2D grid.
    n_cols : int
        Number of columns in the 2D grid.
    neighborhood : int, optional
        Neighborhood type: 4 (default) or 8.
        - 4: Connects each node to its 4 immediate neighbors (left, right, top, bottom).
        - 8: Additionally connects to diagonal neighbors.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix of shape (n_rows * n_cols, n_rows * n_cols).
    """
    n_nodes = n_rows * n_cols  # Total number of nodes in the grid
    adjacency = sp.lil_matrix((n_nodes, n_nodes))  # Use LIL for efficient construction

    for row in range(n_rows):
        for col in range(n_cols):
            node = row * n_cols + col  # Current node index in 1D

            # 4-neighborhood
            if row > 0:  # Top neighbor
                top = (row - 1) * n_cols + col
                adjacency[node, top] = 1
                adjacency[top, node] = 1
            if row < n_rows - 1:  # Bottom neighbor
                bottom = (row + 1) * n_cols + col
                adjacency[node, bottom] = 1
                adjacency[bottom, node] = 1
            if col > 0:  # Left neighbor
                left = row * n_cols + (col - 1)
                adjacency[node, left] = 1
                adjacency[left, node] = 1
            if col < n_cols - 1:  # Right neighbor
                right = row * n_cols + (col + 1)
                adjacency[node, right] = 1
                adjacency[right, node] = 1

            # 8-neighborhood (optional)
            if neighborhood == 8:
                if row > 0 and col > 0:  # Top-left neighbor
                    top_left = (row - 1) * n_cols + (col - 1)
                    adjacency[node, top_left] = 1
                    adjacency[top_left, node] = 1
                if row > 0 and col < n_cols - 1:  # Top-right neighbor
                    top_right = (row - 1) * n_cols + (col + 1)
                    adjacency[node, top_right] = 1
                    adjacency[top_right, node] = 1
                if row < n_rows - 1 and col > 0:  # Bottom-left neighbor
                    bottom_left = (row + 1) * n_cols + (col - 1)
                    adjacency[node, bottom_left] = 1
                    adjacency[bottom_left, node] = 1
                if row < n_rows - 1 and col < n_cols - 1:  # Bottom-right neighbor
                    bottom_right = (row + 1) * n_cols + (col + 1)
                    adjacency[node, bottom_right] = 1
                    adjacency[bottom_right, node] = 1

    # Convert to CSR for efficient arithmetic and slicing
    return adjacency.tocsr()
