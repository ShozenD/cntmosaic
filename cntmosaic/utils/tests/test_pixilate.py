import pytest
import numpy as np
from cntmosaic.utils import AgeBins, pixilate, depixilate

# language: python

def test_basic_functionality():
    matrix = np.ones((10, 10))
    age_bins = AgeBins(0, 9, 5)
    
    result = pixilate(matrix, age_bins)
    expected_shape = (2, 2)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"
    
def test_values():
    matrix = np.ones((10, 10))
    age_bins = AgeBins(0, 9, 5)
    result = pixilate(matrix, age_bins)
    
    expected_result = np.array([[5, 5], [5, 5]])
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"
    
def test_weights():
    matrix = np.ones((10, 10))
    age_bins = AgeBins(0, 9, 5)
    weights = np.array([1,2,2,2,1, 1,2,2,2,1])
    print(weights)
    expected_result = pixilate(matrix, age_bins, weights)
    print(weights)
    dpix_matrix = depixilate(expected_result, age_bins, weights)
    print(weights)
    result = pixilate(dpix_matrix, age_bins, weights)
    
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"
    
def test_inverse():
    matrix = np.ones((10, 10))
    age_bins = AgeBins(0, 9, 5)
    pix_matrix = pixilate(matrix, age_bins)
    result = depixilate(pix_matrix, age_bins)
    expected_result = np.ones((10, 10))
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"
    
    weights = np.array([1,2,1,1,3,3,2,1,2,3])
    expected_result = pixilate(matrix, age_bins, weights)
    print(expected_result)
    dpix_matrix = depixilate(expected_result, age_bins, weights)
    result = pixilate(dpix_matrix, age_bins, weights)
    
    