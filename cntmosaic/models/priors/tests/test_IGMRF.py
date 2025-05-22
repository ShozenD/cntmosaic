import pytest
import jax.numpy as jnp
from .._IGMRF import IGMRF

# language: python

def test_diff_matrix():
  # Test the diff_martrix method (first-order)
  prior = IGMRF(
    scale=1.0,
    num_nodes=5,
    order=1
  )
  
  Ds = prior.diff_matrix()
  assert len(Ds) == 1
  assert Ds[0].shape == (4, 5)
  expected = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, -1.0, 1.0]])
  assert jnp.allclose(Ds[0], expected)
  
  # Test the diff_matrix method (second-order)
  prior = IGMRF(
    scale=1.0,
    num_nodes=5,
    order=2
  )
  
  Ds = prior.diff_matrix()
  
  assert len(Ds) == 1
  assert Ds[0].shape == (3, 5)
  expected = jnp.array([[1.0, -2.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, -2.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, -2.0, 1.0]])
  assert jnp.allclose(Ds[0], expected)
