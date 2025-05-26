import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from scipy.special import comb
from jax.scipy.special import factorial

from numpyro.util import is_prng_key
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
	validate_sample,
	cholesky_of_inverse
)
from numpyro.distributions import constraints
from numpyro import distributions as dist

@partial(jit, static_argnames=("num_nodes", "order"))
def poly_basis(num_nodes: int, order: int) -> jnp.ndarray:	
	x = jnp.linspace(-1, 1, num_nodes)
	S = jnp.zeros((num_nodes, order))
	for i in range(order):
		S = S.at[:, i].set(x**i / factorial(i))
	
	return S

@partial(jit, static_argnames=("num_nodes", "order"))
def diff_matrix(num_nodes: int, order: int)	-> jnp.ndarray:
	D = jnp.zeros((num_nodes - order, num_nodes))
	coeff = jnp.array([comb(order, i) * (-1)**(order-i) for i in range(order + 1)])
	for i in range(num_nodes - order):
		D = D.at[i, i:i+order+1].set(coeff)
	
	return D

@partial(jit, static_argnames=("num_nodes", "order"))
def structure_matrix(num_nodes: int, order: int) -> jnp.ndarray:
	D = diff_matrix(num_nodes, order)
	return D.T @ D

class IGMRF1D(Distribution):
	arg_constraints = {
		"num_nodes": constraints.integer_greater_than(2),
		"scale": constraints.positive,
		"order": constraints.integer_greater_than(1)
	}
	support = constraints.real_vector
	reparametrized_params = ["scale"]
	pytree_data_fields = ("scale", "num_nodes", "order", "L", "S", "Op")
	
	def __init__(self, num_nodes: int=2, order: int=1, scale: float=1.0, *, validate_args=None):
		batch_shape = jnp.shape(scale)
		event_shape = (num_nodes,)
		self.num_nodes = num_nodes
		self.order = order
		self.scale = jnp.array(scale)
	
		self.S = poly_basis(num_nodes, order)
		self.D = diff_matrix(num_nodes, order)
		self.Q = structure_matrix(num_nodes, order)
		self.slgd = self.sum_log_gen_det()
	
		# Precompute reuseable quantities
		self.Qnull, _ = jnp.linalg.qr(self.S)
		Qfr = self.Q + (self.Qnull @ self.Qnull.T) # full rank precision matrix
		self.L = cholesky_of_inverse(Qfr)
		V = self.L @ (self.L.T @ self.S)
		W = self.S.T @ V
		U = jnp.linalg.solve(W, V.T)
		self.Op = U.T @ self.S.T
 
		super(IGMRF1D, self).__init__(
			batch_shape=batch_shape,
			event_shape=event_shape,
			validate_args=validate_args
		)

	def sample(self, key, sample_shape=()):
			assert is_prng_key(key)
			prefix = (1,) * len(sample_shape)
			suffix = (1,) * len(self.event_shape)
			scale = self.scale.reshape(prefix + tuple(self.batch_shape) + suffix)
	 
			z = jax.random.normal(
				key, shape=sample_shape + self.batch_shape + self.event_shape
			)
			x = jnp.matmul(self.L, z[..., jnp.newaxis])
			return scale * jnp.squeeze(x - self.Op @ x, axis=-1)
		
	def sum_log_gen_det(self):
		eigvals = jnp.linalg.eigvalsh(self.Q)
		return jnp.sum(jnp.where(eigvals > 0, jnp.log(eigvals), 0.0))
	
	@validate_sample
	def log_prob(self, value):
		n = self.num_nodes
		k = self.order
		logprec = -0.5 * (n-k) * jnp.log(1/self.scale) * jnp.log(2 * jnp.pi)
		logdet = -0.5 * self.slgd
		logquad = -0.5 * jnp.einsum('...i,ij,...j->...', value, self.Q, value)
		
		return logprec + logdet + logquad

class IGMRF2D(Distribution):
	arg_constraints = {
		"num_nodes": constraints.integer_greater_than(2),
		"scale": constraints.positive,
		"order": constraints.integer_greater_than(1)
	}
	support = constraints.real_vector
	reparametrized_params = ["scale"]
	pytree_data_fields = ("scale", "L", "S", "Op")
	static_data_fields = ("num_nodes", "order")
 
	def __init__(self, num_nodes: list, scale: int, order: list, *, cov_struct="additive", validate_args=None): 
		batch_shape = jnp.shape(scale)
		event_shape = (num_nodes[0]*num_nodes[1],)
  
		S1 = poly_basis(num_nodes[0], order[0])
		S2 = poly_basis(num_nodes[1], order[1])
		self.S = jnp.kron(S1, S2)
  
		Q1 = structure_matrix(num_nodes[0], order[0])
		Q2 = structure_matrix(num_nodes[1], order[1])
		if cov_struct == "additive":
			self.Q = jnp.kron(Q1, jnp.eye(Q2.shape[0])) + jnp.kron(jnp.eye(Q1.shape[0]), Q1)
		elif cov_struct == "multiplicative":
			self.Q = jnp.kron(Q1, Q2)
		else:
			ValueError("Invalid cov_struct argument.")
  
		self.slgd = self.sum_log_gen_det()
  
		# Precompute reuseable quantities
		self.Qnull, _ = jnp.linalg.qr(self.S)
		Qfr = self.Q + (self.Qnull @ self.Qnull.T) # full rank precision matrix
		self.L = cholesky_of_inverse(Qfr)
		V = self.L @ (self.L.T @ self.S)
		W = self.S.T @ V
		U = jnp.linalg.solve(W, V.T)
		self.Op = U.T @ self.S.T
  
		self.num_nodes = num_nodes
		self.order = order
		self.scale = jnp.asarray(scale)
		super(IGMRF2D, self).__init__(
			batch_shape=batch_shape,
			event_shape=event_shape,
			validate_args=validate_args
		)

	def sum_log_gen_det(self):
		eigvals = jnp.linalg.eigvalsh(self.Q)
		return jnp.sum(jnp.where(eigvals > 0, jnp.log(eigvals), 0.0))
  
	def sample(self, key, sample_shape=()):
		assert is_prng_key(key)
		prefix = (1,) * len(sample_shape)
		suffix = (1,) * len(self.event_shape)
		scale = self.scale.reshape(prefix + tuple(self.batch_shape) + suffix)
	 
		z = jax.random.normal(
			key, shape=sample_shape + self.batch_shape + self.event_shape
		)
		x = jnp.matmul(self.L, z[..., jnp.newaxis])
		return jnp.prod(scale) * jnp.squeeze(x - self.Op @ x, axis=-1)

	@validate_sample
	def log_prob(self, value):
		n = self.num_nodes[0] * self.num_nodes[1]
		k = self.order[0] + self.order[1]
		logprec = -0.5 * (n-k) * jnp.log(1/self.scale) * jnp.log(2 * jnp.pi)
		logdet = -0.5 * self.slgd
		logquad = -0.5 * jnp.einsum('...i,ij,...j->...', value, self.Q, value)
		
		return logprec + logdet + logquad