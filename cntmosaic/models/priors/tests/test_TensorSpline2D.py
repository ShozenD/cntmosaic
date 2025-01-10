import pytest
from .._TensorSpline2D import TensorSpline2D

def test_global_prior():
  prior = TensorSpline2D(event_dim=1, type='global')
  prior.set_age_bounds(0, 84)
  
  # Check that the event dimension is set correctly
  assert prior.event_dim == 1
  assert prior.event_dim_eff == 1
  
  # Check existence and dimension of indices
  assert prior.ltri_idx.shape == (85*(85+1)/2,)
  assert prior.sym_tri_idx.shape == (85*85,)
  
  # Check if basis matrices are set correctly
  assert prior.PHI.shape == (85*(85+1)/2, 30*30)
  assert prior.PHI_T.shape == (30*30, 85*(85+1)/2)
  

def test_full_prior():
  prior = TensorSpline2D(event_dim=4, type='full', transform='ilr')
  prior.set_age_bounds(0, 84)
  
  # Check that the event dimension is set correctly
  assert prior.event_dim == 4
  assert prior.event_dim_eff == 3
  assert prior.event_dim_diag == 1
  assert prior.event_dim_non_diag == 2
  
  # Check existence and dimension of indices
  assert prior.ltri_idx.shape == (85*(85+1)/2,)
  assert prior.sym_tri_idx.shape == (85*85,)
  
  # Check if diagonal and non-diagonal indices are set correctly
  assert prior.PHI_diag.shape == (85*(85+1)/2, 30*30)
  assert prior.PHI_diag_T.shape == (30*30, 85*(85+1)/2)
  assert prior.PHI_non_diag.shape == (85*85, 30*30)
  assert prior.PHI_non_diag_T.shape == (30*30, 85*85)
  
def test_partial_prior():
  prior = TensorSpline2D(event_dim=4, type='partial', transform='ilr')
  prior.set_age_bounds(0, 84)
  
  # Check that the event dimension is set correctly
  assert prior.event_dim == 4
  assert prior.event_dim_eff == 3
  
  # Check existence and dimension of indices
  assert prior.PHI.shape == (85*85, 30*30)
  assert prior.PHI_T.shape == (30*30, 85*85)