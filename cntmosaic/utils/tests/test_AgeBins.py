import pytest

import numpy as np
import pandas as pd
from .._AgeGroupSpecs import AgeGroupSpecs as AgeBins


def test_basic_functionality_stride():
    age_bins = AgeBins(min=0, max=10, step=3)

    assert age_bins.left == [0, 3, 6, 9]
    assert age_bins.right == [2, 5, 8, 10]
    assert np.array_equal(age_bins.bin_sizes, np.array([3, 3, 3, 2]))


def test_basic_functionality_cuts():
    age_bins = AgeBins(min=10, max=80, cuts=[20, 40, 60])

    assert age_bins.left == [10, 20, 40, 60]
    assert age_bins.right == [19, 39, 59, 80]
    assert np.array_equal(age_bins.bin_sizes, np.array([10, 20, 20, 21]))


def test_stride():
    age_bins = AgeBins(min=0, max=11, step=3)

    assert age_bins.left == [0, 3, 6, 9]
    assert age_bins.right == [2, 5, 8, 11]
    assert np.array_equal(age_bins.bin_sizes, np.array([3, 3, 3, 3]))

    age_bins = AgeBins(min=0, max=10, step=3)
    assert age_bins.left == [0, 3, 6, 9]
    assert age_bins.right == [2, 5, 8, 10]


def test_cuts():
    age_bins = AgeBins(0, 11, 5)

    assert age_bins.get_cuts() == [0, 5, 10, 12]


# ------------------------------------------------------------------
# age_min / age_max constructor
# ------------------------------------------------------------------

def test_age_min_max_basic():
    specs = AgeBins(age_min=[0, 5, 18], age_max=[4, 17, 80])

    assert specs.left == [0, 5, 18]
    assert specs.right == [4, 17, 80]
    assert specs.min == 0
    assert specs.max == 80
    assert specs.range == 81
    assert np.array_equal(specs.bin_sizes, np.array([5, 13, 63]))


def test_age_min_max_uniform():
    specs = AgeBins(age_min=[0, 5, 10, 15], age_max=[4, 9, 14, 19])

    assert specs.left == [0, 5, 10, 15]
    assert specs.right == [4, 9, 14, 19]
    assert np.array_equal(specs.bin_sizes, np.array([5, 5, 5, 5]))


def test_age_min_max_get_cuts():
    specs = AgeBins(age_min=[0, 5, 18], age_max=[4, 17, 80])
    assert specs.get_cuts() == [0, 5, 18, 81]


# ------------------------------------------------------------------
# bins property and cut() method
# ------------------------------------------------------------------

def test_bins_property():
    specs = AgeBins(0, 14, 5)
    assert specs.bins == [0, 5, 10, 15]


def test_bins_property_age_min_max():
    specs = AgeBins(age_min=[0, 5, 18], age_max=[4, 17, 80])
    assert specs.bins == [0, 5, 18, 81]


def test_cut_basic():
    specs = AgeBins(0, 14, 5)
    ages = pd.Series([0, 4, 5, 9, 10, 14])
    result = specs.cut(ages)
    expected = pd.cut(ages, bins=[0, 5, 10, 15], right=False)
    pd.testing.assert_series_equal(result, expected)


def test_cut_age_min_max():
    specs = AgeBins(age_min=[0, 18, 65], age_max=[17, 64, 80])
    ages = pd.Series([0, 17, 18, 64, 65, 80])
    result = specs.cut(ages)
    assert result[0] == pd.Interval(0, 18, closed="left")
    assert result[2] == pd.Interval(18, 65, closed="left")
    assert result[4] == pd.Interval(65, 81, closed="left")


def test_age_min_max_errors():
    with pytest.raises(ValueError, match="same length"):
        AgeBins(age_min=[0, 5], age_max=[4])

    with pytest.raises(ValueError, match="ascending"):
        AgeBins(age_min=[5, 0], age_max=[4, 9])

    with pytest.raises(ValueError, match="ascending"):
        AgeBins(age_min=[0, 5], age_max=[9, 4])

    with pytest.raises(ValueError, match="age_max"):
        AgeBins(age_min=[0, 10], age_max=[4, 9])
