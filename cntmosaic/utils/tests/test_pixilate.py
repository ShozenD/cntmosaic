import pytest
import numpy as np
from cntmosaic.utils import AgeBins, pixilate, depixilate

# language: python

def test_basic_functionality():
    matrix = np.ones((10, 10))
    age_bins = AgeBins(0, 10, 5)
    
    result = pixilate(matrix, age_bins)
    expected_shape = (2, 2)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"
    
def test_values():
    matrix = np.ones((10, 10))
    age_bins = AgeBins(0, 10, 5)
    result = pixilate(matrix, age_bins)
    
    expected_result = np.array([[5, 5], [5, 5]])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"
    
def test_inverse():
    matrix = np.array([[5, 5], [5, 5]])
    age_bins = AgeBins(0, 10, 5)
    
    result = depixilate(matrix, age_bins)
    
    expected_result = np.ones((10, 10))
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"