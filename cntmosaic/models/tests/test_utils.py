import pytest
import numpy as np
from numpy.testing import assert_equal

from .._utils import age_age_grid, diff_age_age_grid, diff_age_age_index

def test_age_age_grid():
  # Test for A = 3
  X = age_age_grid(3)
  assert X.shape == (9, 2)
  
  X = age_age_grid(85)
  assert X.shape == (85 * 85, 2)

def test_diff_age_age_grid():
  # Test for A = 3
  X = diff_age_age_grid(3)
  assert X.shape == (9, 2)
  
  X = diff_age_age_grid(85)
  assert X.shape == (85 * 85, 2)
  
def test_diff_age_age_index():
  idx = diff_age_age_index(3)
  print(idx)
  assert_equal(
    idx,
    np.array([2, 3, 4, 1, 2, 3, 0, 1, 2])
  )