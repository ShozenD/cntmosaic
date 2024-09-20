import jax.numpy as jnp

def diag_spd_se(alpha: float, ell: float, L: int, M: int):
  """Spectral density function of the squared exponential kernel"""
  y = alpha * jnp.sqrt(jnp.sqrt(2*jnp.pi)*ell) * jnp.exp(-0.25*(ell*jnp.pi/2/L)**2 * jnp.arange(1, M + 1, 1)**2)
  return jnp.diag(y)

def hsgp_basis(x, L: int, M: int):
  """Basis functions for the Hilbert space Gaussian process approximation"""
  y = jnp.pi/(2*L) * (x + L)
  y = jnp.repeat(jnp.expand_dims(y, 1), M, 1)
  z = jnp.diag(jnp.arange(1, M + 1, 1))
  y = jnp.dot(y, z)
  
  return jnp.sin(y) / jnp.sqrt(L)

def hsgp_se(z, X, alpha: float, ell: float, L: int, M: int):
  """Hilbert space Gaussian process with squared exponential kernel"""
  return jnp.dot(jnp.dot(X, diag_spd_se(alpha, ell, L, M)), z)