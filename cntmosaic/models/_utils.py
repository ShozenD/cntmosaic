import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
import jax.scipy as jsp

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
	difference-in-age by age matrix of size (2A - 1) x A) in column-major order.
	
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
	return np.array([[i - j, i] for i in range(A) for j in range(A)])

def diff_age_age_index(A: int) -> np.ndarray:
  """
  Returns the indices of the non-nuisance entries of a difference-in-age by age matrix
  arranged in column-major order.
  """

  return np.array([A + i - j - 1 for j in range(A) for i in range(A)])


def symmetrize_from_lower_tri(N: int) -> NDArray:
	"""
	Indices to augment a vector containing the elements of a 
	lower triangular matrix arranged in column-major order to a vector containing
	the elements of a symmetric matrix arragned in column-major order. Assumes that the
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
				idx[n-1] = i + j*N - (j+1)*j/2
			else:
				idx[n-1] = j + i*N - (i+1)*i/2
			n += 1

	return idx


def lower_tri_indices(N: int, inc_diag=True) -> NDArray:
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
	idx = np.empty((N*(N+1)//2,), dtype=int)
	n = 0
	for j in range(N):
		for i in range(N):
			if inc_diag:
				if i >= j:
					idx[n] = i + j*N
					n += 1
			else:
				if i > j:
					idx[n] = i + j*N
					n += 1
	return idx

import numpy as np
from numpy.typing import NDArray

def rw_drop_indices(n: int, order: int = 2) -> tuple[NDArray, NDArray, NDArray]:
    """
    Indices to *keep* in the difference operators D1 (horizontal) and D2 (vertical)
    for a k-th-order random-walk (RWk) prior on a **lower-triangular** n×n grid.

    Parameters
    ----------
    n      : int   - number of age groups (matrix dimension)
    order  : int   - order k of the finite difference (1 = RW1, 2 = RW2, …)

    Returns
    -------
    ci  : 1-D array of length n(n+1)//2
          Column indices that correspond to the unique (i ≥ j) nodes.
    ri1 : 1-D array
          Row indices to keep in D1  (horizontal differences).
    ri2 : 1-D array
          Row indices to keep in D2  (vertical   differences).
    """
    if not (1 <= order < n):
        raise ValueError("`order` must satisfy 1 ≤ order < n")

    # --- columns: keep lower triangle incl. diagonal ------------------------
    ci_mat = np.arange(n**2, dtype=int).reshape(n, n, order='F')
    mask_ci = np.greater_equal.outer(np.arange(n), np.arange(n))
    ci = np.sort(ci_mat[mask_ci])

    # --- rows for D1: horizontal RWk ----------------------------------------
    # ri1.mat has (n-k) rows and n columns
    ri1_mat = np.arange((n - order) * n, dtype=int).reshape(n - order, n, order='F')
    mask_ri1 = np.greater_equal.outer(np.arange(n - order), np.arange(n))
    ri1 = ri1_mat[mask_ri1]

    # --- rows for D2: vertical RWk ------------------------------------------
    # ri2.mat has n rows and (n-k) columns
    ri2_mat = np.arange(n * (n - order), dtype=int).reshape(n, n - order, order='F')
    mask_ri2 = (np.arange(n)[:, None] >= np.arange(n - order)[None, :] + order)
    ri2 = ri2_mat[mask_ri2]

    return ci, ri1, ri2

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

def lattice_adj(rows: int, cols: int, order: int = 1):
  """
  Open-boundary 2-D lattice adjacency (≤ given graph distance) returned
  as a SciPy CSR matrix.  All intermediate work is done with dense NumPy.

  Parameters
  ----------
  rows, cols : int
    Grid dimensions.
  order : int, default 1
    Maximum shortest-path distance considered a neighbour.

  Returns
  -------
  scipy.sparse.csr_matrix
    Symmetric 0-1 adjacency in CSR form.
  """
  if order < 1:
      raise ValueError("order must be a positive integer")

  # ---- 1. Base grid (order = 1) using Kronecker products --------------
  def chain(n: int) -> np.ndarray:
      B = np.eye(n, k=1, dtype=int)      # ones on the +1 diagonal
      return B + B.T                     # add transpose → ±1 diagonals

  B_r = chain(rows)                      # vertical 1-D chain
  B_c = chain(cols)                      # horizontal 1-D chain

  A = (np.kron(np.eye(rows, dtype=int), B_c) +     # row edges
       np.kron(B_r, np.eye(cols, dtype=int)))      # column edges

  # ---- 2. Expand to higher-order neighbourhoods -----------------------
  if order > 1:
    reach = A.copy().astype(bool)      # cumulative reachability
    A_pow = A.copy()                   # A¹

    for _ in range(2, order + 1):
      A_pow = A_pow @ A              # Aᵏ
      reach |= A_pow > 0             # logical OR with previous
    np.fill_diagonal(reach, 0)         # remove self-loops
    A = reach.astype(int)              # final dense 0-1 array

    # ---- 3. Final step: dense → CSR sparse ------------------------------
  return csr_matrix(A).tocsr()

@jax.jit
def index_mask_logsumexp(
		x: NDArray,
		aid_exp: NDArray,
		bid_pad: NDArray,
		xid_exp: NDArray=None
	):
		"""
		Computes the log-sum-exp over selected elements in an array, masked appropriately.

		This function is designed to process outputs from `make_idarr_for_intervals`.
		It performs advanced indexing to extract elements from the input array `x` using index arrays `aid` 
		and `bid_pad`. The selected elements are then passed through the log-sum-exp operation after 
		applying a mask where non-selected elements are replaced with negative infinity. This is
		useful in statistical computations where log-sum-exp is used to stabilize the computations
		to prevent overflow.

		Parameters
		----------
		x : NDArray
				The input data array from which to compute the log-sum-exp.
		aid : NDArray
				The index array for the first dimension of `x`, as prepared by 
				`make_idarr_for_intervals`.
		bid_pad : NDArray
				The index array for the second dimension of `x`, padded uniformly and prepared by 
				`make_idarr_for_intervals`.
		xid_exp : NDArray, optional
				The expanded ID array matching the dimensions of the index arrays.

		Returns
		-------
		NDArray
				The result of the log-sum-exp computation across the specified axis, after advanced indexing
				and masking.

		Example
		-------
		>>> x = jnp.array([[10, 20, 30, 40],
		...                [50, 60, 70, 80],
		...                [90, 100, 110, 120]])
		>>> aid_exp, bid_pad = make_idarr_for_intervals(df, 'age_grp_cnt', np.array([0, 1, 2]))
		>>> result = index_mask_logsumexp(x, aid_exp, bid_pad)
		>>> print(result)
		"""
		if xid_exp is not None:
				y = x[xid_exp, aid_exp, bid_pad]
		else:
				y = x[aid_exp, bid_pad]
    
		valid_mask = jnp.logical_and(aid_exp >= 0, bid_pad >= 0)
		z = jnp.where(valid_mask, y, -jnp.inf)
		return jsp.special.logsumexp(z, axis=-1)
