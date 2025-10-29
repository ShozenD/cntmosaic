from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial

import jax
import jax.numpy as jnp
from jax import lax, Array
from jax.typing import ArrayLike

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.util import is_prng_key
from numpyro.distributions.util import validate_sample

from ..models._utils import (
  symm_from_tril_ix_col,
  tril_ix_col
)

def diff_matrix_np(
  num_nodes: int,
  order: int
) -> Array:
  """
  Construct a finite difference matrix of given order.
  
  Parameters
  ----------
  num_nodes: int
    The number of nodes in the grid.
  order: int
    The order of the finite difference.
  
  Returns
  -------
  D: NDArray
    The finite difference matrix of shape (num_nodes - order, num_nodes).
  """
  D = np.zeros((num_nodes - order, num_nodes))
  i_vals = np.arange(order + 1)
  coeff = (
		(factorial(order)
		 / (factorial(i_vals) * factorial(order - i_vals)))
		* (-1) ** (order - i_vals)
	)
  for i in range(num_nodes - order):
    D[i, i:i+order+1] = coeff
  
  return D

def tril_igmrf_indices(
  n: int, 
  order: int | tuple[int, int] = 2
) -> tuple[NDArray, NDArray, NDArray]:
  """
  Indices to *keep* in the difference operators D1 (horizontal) and D2 (vertical)
  for a k-th-order random-walk (RWk) prior on a **lower-triangular** n×n grid.

  Parameters
  ----------
  n      : int   - number of age groups (matrix dimension)
  order  : int or tuple[int, int]
           If int: same order k for both dimensions (1 = RW1, 2 = RW2, …)
           If tuple: (order_horizontal, order_vertical) for different orders

  Returns
  -------
  cix  : 1-D array of length n(n+1)//2
    Column indices that correspond to the unique (i ≥ j) nodes.
  rix1 : 1-D array
    Row indices to keep in D1  (horizontal differences).
  rix2 : 1-D array
  	Row indices to keep in D2  (vertical   differences).
  """
  # Parse order specification
  if isinstance(order, int):
      order_h, order_v = order, order
  else:
      order_h, order_v = order
    
  # Validate orders
  if not (1 <= order_h < n):
      raise ValueError(f"`order_h` must satisfy 1 ≤ order_h < n, got {order_h}")
  if not (1 <= order_v < n):
      raise ValueError(f"`order_v` must satisfy 1 ≤ order_v < n, got {order_v}")

  # --- columns: keep lower triangle incl. diagonal ------------------------
  ci_mat = np.arange(n**2, dtype=int).reshape(n, n, order='F')
  mask_ci = np.greater_equal.outer(np.arange(n), np.arange(n))
  cix = np.sort(ci_mat[mask_ci])

  # --- rows for D1: horizontal RWk ----------------------------------------
  # rix1.mat has (n-order_h) rows and n columns
  ri1_mat = np.arange((n - order_h) * n, dtype=int).reshape(n - order_h, n, order='F')
  mask_ri1 = np.greater_equal.outer(np.arange(n - order_h), np.arange(n))
  rix1 = ri1_mat[mask_ri1]

  # --- rows for D2: vertical RWk ------------------------------------------
  # rix2.mat has n rows and (n-order_v) columns
  ri2_mat = np.arange(n * (n - order_v), dtype=int).reshape(n, n - order_v, order='F')
  mask_ri2 = (np.arange(n)[:, None] >= np.arange(n - order_v)[None, :] + order_v)
  rix2 = ri2_mat[mask_ri2]

  return cix, rix1, rix2

class SymIGMRF2D(Distribution):
	support = constraints.real_vector
	pytree_data_fields = ("loc", "cond_prec", "lam_sub", "U_sub", "L")
	pytree_aux_fields = ("num_nodes", "order", "sym_ix", "tril_ix")

	def __init__(
		self,
		num_nodes: int,
		order: int,
		loc: ArrayLike = 0.0,
		cond_prec: Array = 1.0,
		tol: float = 1e-10,
		*,
		validate_args: Optional[bool] = None
	):
		"""
		A symmetric 2-dimensional Intrinsic Gaussian Markov Random Field distribution.
		The distribution is symmetric across the main diagonal of the 2D grid.
		The number of nodes and order must be the same for both dimensions.
		Currently does not support anisotropic precision parameters.
		
		Parameters
		----------
		num_nodes: int
			The number of nodes in the grid for each dimension.
		order: int
			The order of the finite difference approximation for each dimension.
		loc: ArrayLike, optional
			The location parameter (mean) of the distribution. Default is 0.0.
			Expected shape is either () or (batch_shape, num_nodes**2) where the last dimension
			corresponds to the flattened 2D grid by row-major order.
		cond_prec: ArrayLike, optional
			The conditional precision parameter of the distribution. Default is 1.0.
			Expected shape is a scalar () or (batch_shape,) corresponding to the first dimension.
		tol: float, optional
			Tolerance for eigenvalue thresholding. Default is 1e-10.
		validate_args: bool, optional
			Whether to validate the arguments. Default is None.
		"""

		self.num_nodes = num_nodes
		self.order = order

		# Note: The matrices are build in NumPy to avoid JAX tracing issues
		# Build the full Kronecker sum structure matrix
		D1 = np.kron(np.eye(self.num_nodes), diff_matrix_np(self.num_nodes, self.order))
		D2 = np.kron(diff_matrix_np(self.num_nodes, self.order), np.eye(self.num_nodes))

		# Select the lower-triangular indices while respecting the boundaries
		cxi, rix1, rix2 = tril_igmrf_indices(self.num_nodes, self.order)
		D1_red = D1[rix1][:, cxi]
		D2_red = D2[rix2][:, cxi]

		# Construct the laplacian matrices
		L1 = D1_red.T @ D1_red
		L2 = D2_red.T @ D2_red
  
		self.L = L1 + L2
		lam, U = np.linalg.eigh(self.L)

		# Boolean filtering in Numpy (not traced by JAX)
		nonzero_mask = lam > tol
		self.lam_sub = lam[nonzero_mask]
		self.U_sub = U[:, nonzero_mask]

		self.sym_ix = symm_from_tril_ix_col(self.num_nodes)
		self.tril_ix = tril_ix_col(self.num_nodes)

		# ===== Determine batch shape from inputs =====
		if jnp.ndim(loc) == 0: # Scalar loc: no batch dimension
			loc_batch_shape = ()
		elif jnp.ndim(loc) == 1: # loc has shape (n1*n2,)
			loc_batch_shape = ()
		else: # loc has shape (batch_shape, n1*n2)
			loc_batch_shape = jnp.shape(loc)[:-1]

		if jnp.ndim(cond_prec) == 0: # Scalar cond_prec: no batch dimension
			cond_prec_batch_shape = ()
		else: # cond_prec has shape (batch_shape,)
			cond_prec_batch_shape = jnp.shape(cond_prec)

		batch_shape = lax.broadcast_shapes(
			loc_batch_shape, cond_prec_batch_shape
		)

		# ===== Broadcast adjustments =====
		if jnp.ndim(loc) == 0:
			self.loc = jnp.broadcast_to(loc, batch_shape + (self.num_nodes**2,))
		else:
			self.loc = jnp.broadcast_to(loc, batch_shape + (self.num_nodes**2,))

		# Broadcast conditional precisions to (batch_shape,)
		self.cond_prec = jnp.broadcast_to(cond_prec, batch_shape)

		# Broadcast U_sub if there's a batch dimension
		if batch_shape:
			self.U_sub = jnp.broadcast_to(self.U_sub, batch_shape + self.U_sub.shape)
		else:
			self.U_sub = self.U_sub

		event_shape = (self.num_nodes**2,)
	
		super(SymIGMRF2D, self).__init__(
			batch_shape=batch_shape,
			event_shape=event_shape,
			validate_args=validate_args
		)
	
	def sample(
		self,
		key: jax.dtypes.prng_key,
		sample_shape: tuple[int, ...] = ()
	):
		assert is_prng_key(key)
		sub_event_shape = (self.lam_sub.shape[0],)
		eps_shape = sample_shape + self.batch_shape + sub_event_shape
		eps = jax.random.normal(key, shape=eps_shape)[..., jnp.newaxis]
	
		# Reshape cond_prec for proper broadcasting
		cond_prec_reshaped = self.cond_prec[..., jnp.newaxis]
  
		lam_sub = (cond_prec_reshaped * self.lam_sub)
		lam_sub = lam_sub.reshape(self.batch_shape + (-1,))
		scale = jnp.sqrt(1/lam_sub)[..., jnp.newaxis]
		result = self.loc + jnp.squeeze(jnp.matmul(self.U_sub, scale * eps), axis=-1)[..., self.sym_ix]

		return result

	@validate_sample
	def log_prob(self, value: ArrayLike) -> ArrayLike:
		n_tril = self.event_shape[0]
		value_tril = value[..., self.tril_ix]
  
		if jnp.ndim(self.loc) == 0:
			diff = value_tril - self.loc
		else:
			diff = value_tril - self.loc[..., self.tril_ix]

		# Reshape cond_prec for proper broadcasting
		cond_prec_reshaped = self.cond_prec[..., jnp.newaxis]

		lam_sub = cond_prec_reshaped * self.lam_sub
		lam_sub = lam_sub.reshape(self.batch_shape + (-1,))
		log_det = -jnp.sum(jnp.log(lam_sub), axis=-1)

		quad = self.cond_prec * jnp.sum((diff @ self.L) * diff, axis=-1)

		return jnp.squeeze(-0.5 * (n_tril*jnp.log(2*jnp.pi) + log_det + quad))