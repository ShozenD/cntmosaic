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

from .._utils import (
  rw_drop_indices,
  symmetrize_from_lower_tri
)


def poly_basis(num_nodes: int, order: int) -> jnp.ndarray:	
	x = jnp.linspace(-1, 1, num_nodes)
	S = jnp.zeros((num_nodes, order))
	for i in range(order):
		S = S.at[:, i].set(x**i / factorial(i))
	
	return S

@partial(jit, static_argnames=("num_nodes", "order"))
def diff_matrix(num_nodes: int, order: int) -> jnp.ndarray:
	D = jnp.zeros((num_nodes - order, num_nodes))
	i_vals = jnp.arange(order + 1)
	coeff = (factorial(order) / (factorial(i_vals) * factorial(order - i_vals))) * (-1) ** (order - i_vals)
	for i in range(num_nodes - order):
		D = D.at[i, i:i+order+1].set(coeff)
	return D

@partial(jit, static_argnames=("num_nodes", "order"))
def structure_matrix(num_nodes: int, order: int) -> jnp.ndarray:
	D = diff_matrix(num_nodes, order)
	return D.T @ D

def linear_operator(S, L) -> jnp.ndarray:
	V = L @ (L.T @ S)
	W = S.T @ V
	U = jnp.linalg.solve(W, V.T)
	
	return U.T @ S.T

def igmrf1d_operators(order, num_nodes, eps=1e-6):
	S = poly_basis(num_nodes, order)
	Q = structure_matrix(num_nodes, order)
	Q_star = Q - eps * (Q - jnp.diag(jnp.diag(Q)))  # Regularization to ensure positive definiteness
	L = cholesky_of_inverse(Q_star)
	Op = linear_operator(S, L)
	return L, Op

def igmrf2d_operators(num_nodes, order, cov_struct="additive", eps=1e-6):
	S1 = poly_basis(num_nodes[0], order[0])
	S2 = poly_basis(num_nodes[1], order[1])
	S = jnp.kron(S1, S2)

	Q1 = structure_matrix(num_nodes[0], order[0])
	Q2 = structure_matrix(num_nodes[1], order[1])
	if cov_struct == "additive":
		Q = jnp.kron(Q1, jnp.eye(Q2.shape[0])) + jnp.kron(jnp.eye(Q1.shape[0]), Q1)
	elif cov_struct == "multiplicative":
		Q = jnp.kron(Q1, Q2)
	else:
		raise ValueError("Invalid covariance structure specified.")

	Q_star = Q - eps * (Q - jnp.diag(jnp.diag(Q)))  # Regularization to ensure positive definiteness
	L = cholesky_of_inverse(Q_star)
	Op = linear_operator(S, L)
	return L, Op

def igmrf2d_sym_operators(num_nodes, order, cov_struct="additive", eps=1e-6):
	S1 = poly_basis(num_nodes, order[0])
	S2 = poly_basis(num_nodes, order[1])
	S = jnp.kron(S1, S2)

	D1 = jnp.kron(jnp.eye(num_nodes), diff_matrix(num_nodes, order[0]))
	D2 = jnp.kron(diff_matrix(num_nodes, order[1]), jnp.eye(num_nodes))

	ci, ri1, ri2 = rw_drop_indices(num_nodes)
	D1 = D1[ri1][:, ci]
	D2 = D2[ri2][:, ci]
	S = S[ci, :]
 
	Q1 = D1.T @ D1
	Q2 = D2.T @ D2
	Q1_star = Q1 - eps * (Q1 - jnp.diag(jnp.diag(Q1)))
	Q2_star = Q2 - eps * (Q2 - jnp.diag(jnp.diag(Q2)))
	if cov_struct == "additive":
		Q_star = Q1_star + Q2_star
	elif cov_struct == "multiplicative":
		Q_star = Q1_star * Q2_star  # TODO: Check algebra
	else:
		raise ValueError("Invalid covariance structure specified.")

	L = cholesky_of_inverse(Q_star)
	Op = linear_operator(S, L)
	sym_idx = symmetrize_from_lower_tri(num_nodes)
	return L, Op, sym_idx

@jit
def igmrf(x, L, Op, scale=1.0):
	x_transformed = L @ x
	return scale * (x_transformed - Op @ x_transformed)

@jit
def igmrf_sym(x, L, Op, sym_idx, scale=1.0):
	x_transformed = L @ x
	return scale * (x_transformed - Op @ x_transformed)[sym_idx]
