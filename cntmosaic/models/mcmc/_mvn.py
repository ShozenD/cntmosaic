# Implements multivariate normal utilities in JAX.

import jax
import jax.numpy as jnp
from jax import random, jit
from jax.scipy.linalg import solve_triangular

def mvn_logpdf_precision(x: jax.Array, mean: jax.Array, L: jax.Array) -> jax.Array:
  """
  Compute the log probability density function of a multivariate normal distribution
  using precision matrix parameterization.

  This function computes the log-pdf of a multivariate normal distribution where
  the precision matrix A is given via its Cholesky decomposition L such that
  A = L @ L.T. The covariance matrix is V = A^{-1}.

  Parameters
  ----------
  x : jax.Array
    Input vector of shape (d,) where d is the dimensionality.
  mean : jax.Array  
    Mean vector of shape (d,).
  L : jax.Array
    Lower triangular Cholesky factor of the precision matrix A of shape (d, d).
    The precision matrix is reconstructed as A = L @ L.T.

  Returns
  -------
  jax.Array
    Log probability density value as a scalar array.

  Notes
  -----
  The computation uses the precision matrix formulation to avoid matrix inversion.
  The quadratic form (x-μ)ᵀ A (x-μ) is computed efficiently using the Cholesky
  factor via solve_triangular, and the log determinant is computed as twice the
  sum of log diagonal elements of L.

  The log-pdf formula implemented is:
  log p(x) = 0.5 * log|A| - 0.5 * (x-μ)ᵀ A (x-μ) - 0.5 * d * log(2π)
  
  where A is the precision matrix and d is the dimensionality.
  """
  # A = L @ L.T  (precision), V = A^{-1}
  diff = x - mean
  # quad = (x-m)^T A (x-m) = || L.T @ diff ||^2
  y = solve_triangular(L.T, diff, lower=False)   # y = L^T diff
  quad = jnp.dot(y, y)
  logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
  d = x.shape[0]
  return 0.5 * logdetA - 0.5 * quad - 0.5 * d * jnp.log(2.0 * jnp.pi)
  
def sample_mvn_from_precision(
  key: random.PRNGKey,
  mean: jax.Array,
  L: jax.Array
) -> jax.Array:
  """
  Sample from a multivariate normal distribution using precision matrix parameterization.

  This function samples from a multivariate normal distribution N(mean, A^{-1}) where
  A is the precision matrix (inverse covariance matrix) represented by its Cholesky
  decomposition A = L @ L.T.

  Parameters
  ----------
  key : random.PRNGKey
    JAX random number generator key for sampling.
  mean : jax.Array
    Mean vector of the multivariate normal distribution. Shape (..., n).
  L : jax.Array
    Lower triangular Cholesky factor of the precision matrix A, where
    A = L @ L.T. Shape (..., n, n).

  Returns
  -------
  jax.Array
    Sample from the multivariate normal distribution N(mean, A^{-1}).
    Same shape as mean.

  Notes
  -----
  The algorithm works by:
  1. Sampling z ~ N(0, I) where I is the identity matrix
  2. Solving the triangular system L.T @ x = z to get x ~ N(0, A^{-1})
  3. Returning mean + x

  This approach is numerically stable and efficient for precision matrix
  parameterizations commonly used in Bayesian inference.
  """
  z = random.normal(key, mean.shape)
  x = solve_triangular(L, z, lower=True)
  
  return mean + x

@jit
def mvn_cond_params(
  x: jax.Array,
  mean: jax.Array,
  prec: jax.Array,
  ix_target: jax.Array,
  ix_cond: jax.Array
):
  """
  Compute the conditional mean and precision given the full mean vector,
  precision matrix, and indices of the target elements.
  
  Parameters
  ----------
  x : jax.Array
    Full vector of shape (d,) where d is the dimensionality.
  mean : jax.Array
    Mean vector of the full multivariate normal distribution.
  prec : jax.Array
    Precision matrix of the full multivariate normal distribution.
  ix_target : jax.Array
    Indices of the elements of interest for which to compute the conditional distribution.
  ix_cond : jax.Array
    Indices of the lements for which to condition on.
    
  Returns
  -------
  cond_mean : jax.Array
    Conditional mean vector for the elements at the specified indices.
  cond_prec : jax.Array
    Conditional precision matrix for the elements at the specified indices.
  """
  
  # The values of x to be conditioned on
  x_cond = x[ix_cond]
  
  # Partition mean
  m_target = mean[ix_target]
  m_cond = mean[ix_cond]

  # Partition precision
  P11 = prec[jnp.ix_(ix_target, ix_target)]
  P12 = prec[jnp.ix_(ix_target, ix_cond)]
  
  # Conditional precision (unchanged)
  cond_prec = P11
  
  # Conditional mean
  cond_mean = m_target - jnp.linalg.solve(P11, P12 @ (x_cond - m_cond))
  
  return cond_mean, cond_prec

@jit
def sample_mvn_cond(
  key: random.PRNGKey,
  x: jax.Array,
  mean: jax.Array,
  prec: jax.Array,
  ix_target: jax.Array,
  ix_cond: jax.Array
) -> jax.Array:
  """
  Sample from the conditional distribution of a multivariate normal distribution.
  
  Parameters
  ----------
  key : random.PRNGKey
    JAX random number generator key for sampling.
  x : jax.Array
    Full vector of shape (d,) where d is the dimensionality.
  mean : jax.Array
    Mean vector of the full multivariate normal distribution.
  prec : jax.Array
    Precision matrix of the full multivariate normal distribution.
  ix_target : jax.Array
    Indices of the elements of interest for which to compute the conditional distribution.
  ix_cond : jax.Array
    Indices of the elements for which to condition on.
    
  Returns
  -------
  jax.Array
    Sample from the conditional distribution for the elements at the specified indices.
  """
  
  m, P = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
  L = jnp.linalg.cholesky(P)
  
  return sample_mvn_from_precision(key, m, L)