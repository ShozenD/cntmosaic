import pytest
import numpy as np
from numpy.testing import assert_equal

from .._utils import (
  age_age_grid,
  diff_age_age_grid,
  diff_age_age_index,
  tril_indices_row,
  symm_from_tril_indices_row
)

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
  
def test_diff_age_age_index_row_wise():
  """
  Tests the row-wise implementation of the index generation function.
  """
  # Call the function with the test case A=3
  idx = diff_age_age_index(3)
  
  # The expected array is now in row-major order
  expected_output = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12])
  
  # Assert that the function's output matches the expected row-wise array
  assert_equal(idx, expected_output)
  
def test_tril_indices_row():
  # Test 3 x 3
  idx = tril_indices_row(3)
  expected_output = np.array([0, 3, 4, 6, 7, 8])
  assert_equal(idx, expected_output)
  
  # Test 4 x 4
  idx = tril_indices_row(4)
  expected_output = np.array([0, 4, 5, 8, 9, 10, 12, 13, 14, 15])
  assert_equal(idx, expected_output)
  
def test_symm_from_tril_indices_row():
  # Test 3 x 3
  idx = symm_from_tril_indices_row(3)
  expected_output = np.array([0, 1, 3, 1, 2, 4, 3, 4, 5])
  assert_equal(idx, expected_output)
  
  # Test 4 x 4
  idx = symm_from_tril_indices_row(4)
  expected_output = np.array([0, 1, 3, 6,
                              1, 2, 4, 7,
                              3, 4, 5, 8,
                              6, 7, 8, 9])
  assert_equal(idx, expected_output)