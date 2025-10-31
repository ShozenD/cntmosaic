import pytest
from .._PSpline2D import PSpline2D
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

def test_basis_shape_global():
  prior = PSpline2D(prior_type='global')
  prior.set_age_bounds(0, 84)
  prior.set_event_dim(1)
  
  # Check that the event dimension is set correctly
  assert prior.event_dim == 1
  assert prior.event_dim_eff == 1
  
  # Check if basis matrices are set correctly
  assert prior.PHI.shape == (85*(85+1)/2, 30*30)
  
def test_basis_shape_partial():
  prior = PSpline2D(prior_type='partial', transform='ilr')
  prior.set_age_bounds(0, 84)
  prior.set_event_dim(4)

  # Check that the event dimension is set correctly
  assert prior.event_dim == 4
  assert prior.event_dim_eff == 3

  # Check existence and dimension of indices
  assert prior.PHI.shape == (85*85, 30*30)
  
def test_sample_global():
  """Test sampling from global prior."""
  prior = PSpline2D(prior_type='global')
  prior.set_age_bounds(0, 9)  # Small age range for testing
  
  # Mock the sampling context
  with numpyro.handlers.seed(rng_seed=42):
      result = prior.sample()
    
  # Check output shape
  assert result.shape == (10, 10)
  assert isinstance(result, jnp.ndarray)

def test_sample_partial():
  """Test sampling from partial prior."""
  prior = PSpline2D(prior_type='partial', transform='ilr')
  prior.set_age_bounds(0, 9)
  prior.set_event_dim(3)
  prior.set_loc(0.0)

  with numpyro.handlers.seed(rng_seed=42):
      result = prior.sample()
    
  # Check output shape
  assert result.shape == (3, 10, 10)
  assert isinstance(result, jnp.ndarray)
    
def test_sample_full():
  """Test sampling from full prior."""
  prior = PSpline2D(prior_type='full', transform='ilr')
  prior.set_age_bounds(0, 9)
  prior.set_event_dim(4)
  prior.set_loc(0.0)
    
  with numpyro.handlers.seed(rng_seed=42):
      result = prior.sample()
    
  # Check output shape
  assert result.shape == (4, 10, 10)
  assert isinstance(result, jnp.ndarray)