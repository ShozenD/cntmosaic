import jax.numpy as jnp
from scipy.special import comb
from jax.scipy.special import factorial
from numpyro.distributions.util import cholesky_of_inverse

def poly_basis(num_nodes: int, order: int) -> jnp.ndarray:	
	x = jnp.linspace(-1, 1, num_nodes)
	S = jnp.zeros((num_nodes, order))
	for i in range(order):
		S = S.at[:, i].set(x**i / factorial(i))
	
	return S

def diff_matrix(num_nodes: int, order: int)	-> jnp.ndarray:
	D = jnp.zeros((num_nodes - order, num_nodes))
	coeff = jnp.array([comb(order, i) * (-1)**(order-i) for i in range(order + 1)])
	for i in range(num_nodes - order):
		D = D.at[i, i:i+order+1].set(coeff)
	
	return D

def structure_matrix(num_nodes: int, order: int) -> jnp.ndarray:
	D = diff_matrix(num_nodes, order)
	return D.T @ D

def cholesky_inv_prec(S, Q) -> jnp.ndarray:
	Qnull, _ = jnp.linalg.qr(S)
	Qfr = Q + (Qnull @ Qnull.T)
 
	return cholesky_of_inverse(Qfr)

def linear_operator(S, L) -> jnp.ndarray:
  V = L @ (L.T @ S)
  W = S.T @ V
  U = jnp.linalg.solve(W, V.T)
  
  return U.T @ S.T

def igmrf1d(x, order, scale):
	num_nodes = x.shape[-1]
	S = poly_basis(num_nodes, order)
	Q = structure_matrix(num_nodes, order)
	L = cholesky_inv_prec(S, Q)
	Op = linear_operator(S, L)
 
	x = L @ x
	return scale * (x - Op @ x)

def igmrf2d(x, num_nodes, order, scale, cov_struct="additive"):
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

	L = cholesky_inv_prec(S, Q)
	Op = linear_operator(S, L)
 
	x = L @ x
	return scale * (x - Op @ x)