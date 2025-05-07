import math
import numpy as np

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

def is_square(x: int) -> bool:
  """Check if a number is a perfect square"""
  return x == math.isqrt(x) ** 2

def closure(x: ArrayLike, axis: int=0) -> Array:
  """Closure operation for a given axis"""
  return x / jnp.sum(x, axis=axis, keepdims=True)

def basis_contrast_matrix(d: int) -> Array:
  """Basis contrast matrix associated with an orthonormal basis of S^D
  
  See eq. (18) of J.J. Egozcue, V. Pawlowsky-Glahn, G. Mateu-Figueras, and C. Barcelo-Vidal.
  Isometric logratio transformations for compositional data analysis. Mathematical Geology, 35(3):279–300, 2003.
  
  and eq. (3) of Ana M. Bianco et al. Robust Nonparametric Regression for Compositional Data: the Simplicial-Real case. 2024.
  """
  U = np.zeros((d, d-1))
  for i in range(d-1):
    j = i + 1
    U[:j,i] = 1./j
    U[j,i] = -1
    U[:,i] *= np.sqrt(j/(j+1.)) 
    
  return U

def alr(x: ArrayLike, axis: int=0) -> Array:
  """Additive log ratio transformation"""
  denom = 1 - jnp.sum(x, axis=axis, keepdims=True)
  return jnp.log(x / denom)
  
def clr(x: ArrayLike, axis: int=0) -> Array:
  """Centered log ratio transformation"""
  y = jnp.log(x + jnp.finfo(x.dtype).eps)
  return y - jnp.mean(y, axis=axis, keepdims=True)

def ilr(x: ArrayLike, axis: int=0) -> Array:
  """Isometric log ratio transformation"""
  shape = list(x.shape)
  U = basis_contrast_matrix(shape[axis])
  y = clr(x)
  return jnp.apply_along_axis(lambda z: jnp.matmul(U.T, z), axis=axis, arr=y)
  
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
  U = basis_contrast_matrix(shape[axis]+1)
  Uz = jnp.apply_along_axis(lambda z: jnp.matmul(U, z), axis=axis, arr=x)
  
  return jax.nn.softmax(Uz, axis=axis)

def log_inverse_ilr(x: ArrayLike, axis: int=0) -> Array:
  """Log inverse isometric log ratio transformation"""
  return jnp.log(inverse_ilr(x, axis=axis))