import pytest

from .._base import load_age_distribution

def test_basic_functionality():
  age_dist = load_age_distribution('United_States',
                                   'country')
  
  assert age_dist.shape == (85,2)
  
def test_load_subnational():
  age_dist = load_age_distribution('United_States',
                                   'subnational',
                                   'California')
  
  assert age_dist.shape == (85,2)