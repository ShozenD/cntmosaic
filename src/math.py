import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

def closure(x: ArrayLike, axis: int) -> Array:
  denom = jnp.sum(x, axis=axis, keepdims=True)
  return x / denom

