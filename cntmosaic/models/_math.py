import numpy as np

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

def closure(x: ArrayLike, axis: int=0) -> Array:
  """Closure operation for a given axis"""
  return x / jnp.sum(x, axis=axis, keepdims=True)

def aitchison_orthnorm_basis(d: int) -> Array:
  """Aitchison orthonormal basis"""
  U = np.zeros((d, d-1))
  for i in range(d-1):
    j = i + 1
    U[:j,i] = np.sqrt(1/(j*(j+1)))
    U[i+1,i] = -np.sqrt(1/(j+1))
    
  return closure(np.exp(U))

def alr(x: ArrayLike, axis: int=0) -> Array:
  """Additive log ratio transformation"""
  denom = 1 - jnp.sum(x, axis=axis, keepdims=True)
  return jnp.log(x / denom)
  
def clr(x: ArrayLike, axis: int=0) -> Array:
  """Centered log ratio transformation"""
  return jnp.log(x) - jnp.mean(jnp.log(x), axis=axis, keepdims=True)

def ilr(x: ArrayLike, axis: int=0) -> Array:
  """Isometric log ratio transformation"""
  shape = list(x.shape)
  U = aitchison_orthnorm_basis(shape[axis])
  y = clr(x, axis=axis)
  return jnp.apply_along_axis(lambda y: jnp.matmul(U.T, y), axis=axis, arr=y)
  
def inverse_alr(x: ArrayLike, axis: int=0) -> Array:
  """Inverse additive log ratio transformation"""
  shape = list(x.shape)
  ones_shape = shape.copy()
  ones_shape[axis] = 1
  ones = jnp.ones(ones_shape)
  
  x = jnp.concatenate((jnp.exp(x), ones), axis=axis)
  return closure(x, axis=axis)

def log_inverse_alr(x: ArrayLike, axis: int=0) -> Array:
  """Log inverse additive log ratio transformation"""
  return jnp.log(inverse_alr(x, axis=axis))
  
def inverse_clr(x: ArrayLike, axis: int=0) -> Array:
  """Inverse centered log ratio transformation"""
  return jax.nn.softmax(x, axis=axis)

def inverse_ilr(x: ArrayLike, axis: int=0) -> Array:
  """Inverse isometric log ratio transformation"""
  shape = list(x.shape)
  U = aitchison_orthnorm_basis(shape[axis]+1)
  Uz = jnp.apply_along_axis(lambda z: jnp.matmul(U, z), axis=axis, arr=x)
  
  return jax.nn.softmax(jnp.exp(Uz), axis=axis)

def log_inverse_ilr(x: ArrayLike, axis: int=0) -> Array:
  """Log inverse isometric log ratio transformation"""
  return jnp.log(inverse_ilr(x, axis=axis))