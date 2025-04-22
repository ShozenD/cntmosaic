import pytest

import numpy as np
from .._AgeBins import AgeBins

def test_basic_functionality_stride():
    age_bins = AgeBins(min=0, max=10, stride=3)
    
    assert age_bins.left == [0, 3, 6, 9]
    assert age_bins.right == [2, 5, 8, 10]
    assert np.array_equal(age_bins.bin_sizes, np.array([3, 3, 3, 1]))
    
def test_basic_functionality_cuts():
    age_bins = AgeBins(min=10, max=80, cuts=[20, 40, 60])
    
    assert age_bins.left == [10, 20, 40, 60]
    assert age_bins.right == [19, 39, 59, 80]
    assert np.array_equal(age_bins.bin_sizes, np.array([10, 20, 20, 20]))