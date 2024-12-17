import pytest

from .._base import load_age_distribution

def test_basic_functionality():
  age_dist = load_age_distribution('United_States')
  
  assert age_dist.shape == (85,2)