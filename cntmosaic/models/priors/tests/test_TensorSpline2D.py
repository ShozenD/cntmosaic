import pytest
from .._TensorSpline2D import TensorSpline2D

def test_full_prior():
  prior = TensorSpline2D(event_dim=4, type='full', transform='ilr')
  
  # Check that the event dimension is set correctly
  assert prior.event_dim == 4
  assert prior.event_dim_eff == 3
  assert prior.event_dim_diag == 1
  assert prior.event_dim_non_diag == 2
  
  # Check if diagonal and non-diagonal indices are set correctly
  prior.set_age_bounds(0, 84)
  assert prior.basis_diag is not None
  assert prior.basis_non_diag is not None