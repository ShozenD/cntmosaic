import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
from scipy.special import comb
from jax.scipy.special import factorial

from numpyro.util import is_prng_key
from numpyro.distributions.distribution import Distribution
from numpyro.distributions import constraints
from numpyro import distributions as dist

from .._utils import (
  rw_drop_indices,
  symmetrize_from_lower_tri
)

def diff_matrix(
  num_nodes: int,
  order: int
) -> jnp.ndarray:
	D = jnp.zeros((num_nodes - order, num_nodes))
	i_vals = jnp.arange(order + 1)
	coeff = (factorial(order) / (factorial(i_vals) * factorial(order - i_vals))) * (-1) ** (order - i_vals)
	for i in range(num_nodes - order):
		D = D.at[i, i:i+order+1].set(coeff)
	return D

def structure_matrix(num_nodes: int, order: int) -> jnp.ndarray:
	D = diff_matrix(num_nodes, order)

	return D.T @ D

def make_igmrf2d_operator(num_nodes: tuple, order: tuple):
  n1, n2 = num_nodes
  order1, order2 = order
  
  Q1 = structure_matrix(n1, order1)
  Q2 = structure_matrix(n2, order2)

  lam1, U1 = jnp.linalg.eigh(Q1)   # Q1 = U1 @ diag(lam1) @ U1.T
  lam2, U2 = jnp.linalg.eigh(Q2)   # Q2 = U2 @ diag(lam2) @ U2.T
  
  # Clean tiny negative eigenvalues
  lam1 = jnp.maximum(lam1, 0.0)
  lam2 = jnp.maximum(lam2, 0.0)

  # Static nullspace mask: (lam1 == 0) AND (lam2 == 0)
  # Use a tolerance because of roundoff; sqrt(eps) is a good scale.
  tol = jnp.sqrt(jnp.finfo(lam1.dtype).eps)
  null1 = lam1 <= tol
  null2 = lam2 <= tol
  null_mask = null1[:, None] & null2[None, :]   # shape (n1, n2)

  @jit
  def operator(Z, tau1, tau2):
    Zt = U1.T @ Z @ U2
    
    s = tau1 * lam1[:, None] + tau2 * lam2[None, :]
    # Clip BEFORE rsqrt to avoid NaNs; keep penalized modes differentiable.
    s_clip = jnp.maximum(s, 0.0) + jnp.finfo(s.dtype).eps

    gains = jax.lax.rsqrt(s_clip)                 # 1/sqrt(s_clip)
    gains = jnp.where(null_mask, 0.0, gains)  # exact zeros on nullspace
    Y = Zt * gains

    F = U1 @ Y @ U2.T
    return F
  
  return operator

def make_sym_igmrf2d_operator(num_nodes: tuple, order: tuple):
  n1, n2 = num_nodes
  order1, order2 = order

  # Build the full Kronecker sum structure matrix
  D1 = jnp.kron(jnp.eye(n2), diff_matrix(n1, order1))
  D2 = jnp.kron(diff_matrix(n2, order2), jnp.eye(n2))

  # Apply the reduction
  ci, ri1, ri2 = rw_drop_indices(n1)
  D1_reduced = D1[ri1][:, ci]
  D2_reduced = D2[ri2][:, ci]
  
  Q1 = D1_reduced.T @ D1_reduced
  Q2 = D2_reduced.T @ D2_reduced

  Q = Q1 + Q2
  lam, U = jnp.linalg.eigh(Q)   # Q = U @ diag(lam) @ U.T

  n = U.shape[0]
  sym_idx = symmetrize_from_lower_tri(n1)
  
  @jit
  def operator(z, tau):
    denom = jnp.sqrt(tau * lam)
    gains = jnp.where(denom > 0, 1.0 / denom, 0.0)
    f = U @ (gains * z)

    return f[sym_idx].reshape((n1, n2))

  return operator

def log_density_igmrf(
  beta: jnp.ndarray,
  Q0: jnp.ndarray,
  tau: float,
  r_eff: float = None,
  logdet_Q_const: float = None
):
  Q = Q0 * tau
  
  if r_eff is None:
    eigval_Q0, _ = jnp.linalg.eigh(Q0)
    pos = eigval_Q0 > 0
    
    r_eff = pos.sum()
    logdet_Q_const = jnp.sum(
      jnp.log(jnp.where(pos, eigval_Q0, 1.0))
    )
  
  b = beta
  quad = b @ (Q @ b)
  logdet_pseudo = r_eff * jnp.log(tau) + logdet_Q_const
  
  return 0.5 * logdet_pseudo - 0.5 * quad