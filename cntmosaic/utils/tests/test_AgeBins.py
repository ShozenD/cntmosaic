import pytest

import numpy as np
from .._AgeBins import AgeBins

def test_basic_functionality_stride():
    age_bins = AgeBins(min=0, max=10, step=3)
    
    assert age_bins.left == [0, 3, 6, 9]
    assert age_bins.right == [2, 5, 8, 11]
    assert np.array_equal(age_bins.bin_sizes, np.array([3, 3, 3, 2]))
    
def test_basic_functionality_cuts():
    age_bins = AgeBins(min=10, max=80, cuts=[20, 40, 60])
    
    assert age_bins.left == [10, 20, 40, 60]
    assert age_bins.right == [19, 39, 59, 81]
    assert np.array_equal(age_bins.bin_sizes, np.array([10, 20, 20, 21]))
    
def test_stride():
    age_bins = AgeBins(min=0, max=11, step=3)
    
    assert age_bins.left == [0, 3, 6, 9]
    assert age_bins.right == [2, 5, 8, 12]
    assert np.array_equal(age_bins.bin_sizes, np.array([3, 3, 3, 3]))
    
    age_bins = AgeBins(min=0, max=10, step=3)
    assert age_bins.left == [0, 3, 6, 9]
    assert age_bins.right == [2, 5, 8, 11]
    
def test_cuts():
    age_bins = AgeBins(0, 11, 5)
    
    assert age_bins.get_cuts() == [0, 5, 10, 12]
