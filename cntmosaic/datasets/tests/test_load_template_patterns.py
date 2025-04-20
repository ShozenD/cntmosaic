import pytest

from .._base import load_template_patterns

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