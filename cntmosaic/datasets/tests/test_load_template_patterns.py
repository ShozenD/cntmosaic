import pytest

from .._base import load_template_patterns

# language: python

def test_basic_functionality():
  patterns = load_template_patterns('United_States',
                                    symmetrise=True,
                                    smooth=True)
  
  assert 'household' in patterns
  assert 'school' in patterns
  assert 'work' in patterns
  assert 'community' in patterns
  
  # Check dimensions
  assert patterns['household'].shape == (85, 85)
  assert patterns['school'].shape == (85, 85)
  assert patterns['work'].shape == (85, 85)
  assert patterns['community'].shape == (85, 85)
  
def test_normalisation():
  patterns = load_template_patterns('United_States', normalise=True)
  
  assert patterns['household'].sum(axis=1).mean() == pytest.approx(1.0, rel=1e-1)
  assert patterns['school'].sum(axis=1).mean() == pytest.approx(1.0, rel=1e-1)
  assert patterns['work'].sum(axis=1).mean() == pytest.approx(1.0, rel=1e-1)
  assert patterns['community'].sum(axis=1).mean() == pytest.approx(1.0, rel=1e-1)