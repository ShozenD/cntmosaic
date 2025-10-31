import pytest
from .._HSGP2D import HSGP2D

def test_initialisation():
  
  prior = HSGP2D(
    C=[1.5, 1.5],
    M=[30, 30],
    grid_type='age-age',
    transform=None,
    prior_type='global'
  )
  
  assert prior.C == [1.5, 1.5]
  assert prior.M == [30, 30]
  assert prior.grid_type == 'age-age'
  assert prior.transform == None
  assert prior.prior_type == 'global'