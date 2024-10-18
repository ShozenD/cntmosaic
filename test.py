import unittest
import numpy as np
from numpy.testing import assert_array_equal

from src.model_utils import (
  symmetrize_from_lower_tri,
  transpose_vector_indices,
  lower_tri_indices,
  non_nuisance_grid
)

from src.models import (
  BRC,
  BRCBasic,
  BRCStratified
)

class TestSymmetrizeLowerTri(unittest.TestCase):
  def test_2_by_2(self):
    """Test for a 2 by 2 matrix"""
    assert_array_equal(symmetrize_from_lower_tri(2), np.array([0, 1, 1, 2]))

  def test_3_by_3(self):
    """Test for a 3 by 3 matrix"""
    assert_array_equal(symmetrize_from_lower_tri(3), np.array([0, 1, 2, 1, 3, 4, 2, 4, 5]))

class TestTransposeVectorIndices(unittest.TestCase):
  def test_2_by_2(self):
    """Test for a 2 by 2 matrix"""
    assert_array_equal(transpose_vector_indices(2, 2), np.array([0, 2, 1, 3]))

  def test_3_by_3(self):
    """Test for a 3 by 3 matrix"""
    assert_array_equal(transpose_vector_indices(3, 3), np.array([0, 3, 6, 1, 4, 7, 2, 5, 8]))

class TestLowerTriIndices(unittest.TestCase):
  def test_2_by_2(self):
    """Test for a 2 by 2 matrix"""
    assert_array_equal(lower_tri_indices(2), np.array([0, 1, 3]))

  def test_3_by_3(self):
    """Test for a 3 by 3 matrix"""
    assert_array_equal(lower_tri_indices(3), np.array([0, 1, 2, 4, 5, 8]))

class TestNonNuisanceGrid(unittest.TestCase):
  def test_2_by_2(self):
    """Test for a 2 by 2 matrix"""
    assert_array_equal(non_nuisance_grid(2),
                       np.array([[2,1], [3,1], [1,2], [2,2]]))
    
  def test_3_by_3(self):
    """Test for a 3 by 3 matrix"""
    assert_array_equal(non_nuisance_grid(3),
                       np.array([[3,1], [4,1], [5,1], [2,2], [3,2], [4,2], [1,3], [2,3], [3,3]]))

if __name__ == "__main__":
  unittest.main(verbosity=2)