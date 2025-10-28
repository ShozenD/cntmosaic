from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax, Array
from jax.typing import ArrayLike

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.util import is_prng_key
from numpyro.distributions.util import validate_sample, promote_shapes

from ._IGMRF import laplacian

class IGMRF2D(Distribution):
	support = constraints.real_vector
	pytree_data_fields = ("loc", "cond_prec1", "cond_prec2",
											 "lam1_sub", "lam2_sub", "U_sub",
											 "L1", "L2")
	pytree_aux_fields = ('num_nodes', 'order')

	def __init__(
		self,
		num_nodes: tuple[int, int],
		order: tuple[int, int],
		loc: ArrayLike = 0.0,
		cond_prec1: Array = 1.0,
		cond_prec2: Array = 1.0,
		*,
		validate_args: Optional[bool] = None
	):
		"""
		2-dimensional Intrinsic Gaussian Markov Random Field (IGMRF) distribution.
		
		Parameters
		----------
		num_nodes: tuple[int, int]
			The number of nodes in the grid for each dimension.
		order: tuple[int, int]
			The order of the finite difference approximation for each dimension.
		loc: ArrayLike, optional
			The location parameter (mean) of the distribution. Default is 0.0.
			Expected shape is either () or (batch_shape, num_nodes[0] * num_nodes[1]) where the last dimension
			corresponds to the flattened 2D grid by row-major order.
		cond_prec1: ArrayLike, optional
			The conditional precision parameter of the distribution. Default is 1.0.
			Expected shape is a scalar () or (batch_shape,) corresponding to the first dimension.
		cond_prec2: ArrayLike, optional
			The conditional precision parameter of the distribution. Default is 1.0.
			Expected shape is a scalar () or (batch_shape,) corresponding to the second dimension.
		validate_args: bool, optional
			Whether to validate the arguments. Default is None.
		"""

		self.num_nodes = num_nodes
		self.order = order

		# Create precision matrix
		self.L1 = laplacian(num_nodes[0], order[0])
		self.L2 = laplacian(num_nodes[1], order[1])
		lam1, U1 = jnp.linalg.eigh(self.L1)
		lam2, U2 = jnp.linalg.eigh(self.L2)
		
		U1_sub, U2_sub = U1[:,order[0]:], U2[:,order[1]:]
		self.lam1_sub, self.lam2_sub = lam1[order[0]:], lam2[order[1]:]
		self.U_sub = jnp.kron(U1_sub, U2_sub) # shape (n1*n2, (n1-order1)*(n2-order2))
	
		# ===== Determine batch shape from inputs =====
		if jnp.ndim(loc) == 0: # Scalar loc: no batch dimension
			loc_batch_shape = ()
		elif jnp.ndim(loc) == 1: # loc has shape (n1*n2,)
			loc_batch_shape = ()
		else: # loc has shape (batch_shape, n1*n2)
			loc_batch_shape = jnp.shape(loc)[:-1]
	
		if jnp.ndim(cond_prec1) == 0: # Scalar cond_prec1: no batch dimension
			cond_prec1_batch_shape = ()
		else: # cond_prec1 has shape (batch_shape,)
			cond_prec1_batch_shape = jnp.shape(cond_prec1)
	 
		if jnp.ndim(cond_prec2) == 0: # Scalar cond_prec2: no batch dimension
			cond_prec2_batch_shape = ()
		else: # cond_prec2 has shape (batch_shape,)
			cond_prec2_batch_shape = jnp.shape(cond_prec2)
	 
		batch_shape = lax.broadcast_shapes(
			loc_batch_shape, cond_prec1_batch_shape, cond_prec2_batch_shape
		)

		# ===== Broadcast adjustments =====
		if jnp.ndim(loc) == 0:
			self.loc = jnp.broadcast_to(loc, batch_shape + (num_nodes[0] * num_nodes[1],))
		else:
			self.loc = jnp.broadcast_to(loc, batch_shape + (num_nodes[0] * num_nodes[1],))
	 
		# Broadcast conditional precisions to (batch_shape,)
		self.cond_prec1 = jnp.broadcast_to(cond_prec1, batch_shape)
		self.cond_prec2 = jnp.broadcast_to(cond_prec2, batch_shape)

		# Broadcast U_sub if there's a batch dimension
		if batch_shape:
			self.U_sub = jnp.broadcast_to(self.U_sub, batch_shape + (num_nodes[0] * num_nodes[1], (num_nodes[0]-order[0]) * (num_nodes[1]-order[1])))
		else:
			self.U_sub = self.U_sub

		event_shape = (num_nodes[0] * num_nodes[1],)
	
		super(IGMRF2D, self).__init__(
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
		sub_event_shape = ((self.num_nodes[0] - self.order[0]) * (self.num_nodes[1] - self.order[1]),)
		eps_shape = sample_shape + self.batch_shape + sub_event_shape
		eps = jax.random.normal(key, shape=eps_shape)[..., jnp.newaxis]
	
		# Reshape cond_prec for proper broadcasting
		cond_prec1_reshaped = self.cond_prec1[..., jnp.newaxis, jnp.newaxis]
		cond_prec2_reshaped = self.cond_prec2[..., jnp.newaxis, jnp.newaxis]

		# More efficient computation without Kronecker products
		# For 2D IGMRF, eigenvalues are: lam1[i] + lam2[j] for all i,j combinations
		lam_sub = (cond_prec1_reshaped * self.lam1_sub[:, jnp.newaxis] + 
             	 cond_prec2_reshaped * self.lam2_sub[jnp.newaxis, :])
		lam_sub = lam_sub.reshape(self.batch_shape + (-1,))
		scale = jnp.sqrt(1/lam_sub)[..., jnp.newaxis]
		result = self.loc + jnp.squeeze(jnp.matmul(self.U_sub, scale * eps), axis=-1)

		return result

	@validate_sample
	def log_prob(self, value: ArrayLike) -> ArrayLike:
		n1, n2 = self.num_nodes
		diff = value - self.loc
  
		# Reshape cond_prec for proper broadcasting
		cond_prec1_reshaped = self.cond_prec1[..., jnp.newaxis, jnp.newaxis]
		cond_prec2_reshaped = self.cond_prec2[..., jnp.newaxis, jnp.newaxis]

		lam_sub = (cond_prec1_reshaped * self.lam1_sub[:, jnp.newaxis] + 
             	 cond_prec2_reshaped * self.lam2_sub[jnp.newaxis, :])
		lam_sub = lam_sub.reshape(self.batch_shape + (-1,))
		log_det = -jnp.sum(jnp.log(lam_sub), axis=-1)

		quad1 = self.cond_prec1 * jnp.sum((diff @ jnp.kron(self.L1, jnp.eye(n2))) * diff, axis=-1)
		quad2 = self.cond_prec2 * jnp.sum((diff @ jnp.kron(jnp.eye(n1), self.L2)) * diff, axis=-1)
		quad = quad1 + quad2

		return jnp.squeeze(-0.5 * (n1*n2/2*jnp.log(2*jnp.pi) + log_det + quad))