import pytest
from .._base import load_polymod_germany

def test_basic_functionality():
  """Tests that the function works."""
  
  data = load_polymod_germany()
  
  assert 'contacts' in data
  assert 'participants' in data
  assert 'population' in data
  assert data['contacts'].shape[0] > 0
  assert data['participants'].shape[0] > 0
  assert data['population'].shape[0] > 0