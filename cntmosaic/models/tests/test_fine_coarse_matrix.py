import pandas as pd
import numpy as np
import pytest
from .._utils import fine_coarse_matrix  # Adjust this import according to your project structure

def test_basic_functionality():
    # Create a Series with categorical intervals
    ages = pd.Series(pd.cut(np.arange(0, 85),
                            bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
                            right=False))
    expected_columns = 17  # 17 intervals
    expected_rows = 85  # 85 age values

    # Function call
    indicator_matrix = fine_coarse_matrix(ages)

    # Assertions
    assert indicator_matrix.shape == (expected_rows, expected_columns), "Matrix dimensions are incorrect"
    assert np.sum(indicator_matrix[0, :]) == 1, "Incorrect mapping at the boundary"
    assert np.sum(indicator_matrix[-1, :]) == 1, "Incorrect mapping at the upper boundary"
    
def test_non_zero_start():
    ages = pd.Series(pd.cut(np.arange(18, 80),
                            bins = [18, 20, 30, 40, 50, 60, 70, 80],
                            right=False))
    
    expected_columns = 7  # 7 intervals
    expected_rows = 62  # 62 age values
    
    indicator_matrix = fine_coarse_matrix(ages)
    
    assert indicator_matrix.shape == (expected_rows, expected_columns), "Matrix dimensions are incorrect"
    assert np.sum(indicator_matrix[0, :]) == 1, "Incorrect mapping at the boundary"
    assert np.sum(indicator_matrix[-1, :]) == 1, "Incorrect mapping at the upper boundary"

def test_all_inclusive_range():
    # All ages in one interval
    ages = pd.Series(pd.cut(np.arange(0, 85), bins=[0, 85], right=False))
    expected_columns = 1  # 1 interval

    # Function call
    indicator_matrix = fine_coarse_matrix(ages)

    # Assertions
    assert indicator_matrix.shape[1] == expected_columns, "Matrix should have one column"
    assert np.all(indicator_matrix == 1), "All entries should be 1"

def test_non_inclusive_ranges():
    # Excluding the last few ages
    ages = pd.Series(pd.cut(np.arange(0, 85), bins=[0, 50], right=False)) # Ages above 50 will produce NaNs

    with pytest.raises(ValueError) as exc_info:
      fine_coarse_matrix(ages)
      
    assert exc_info.type is ValueError

def test_edge_cases():
    # Edge cases at each interval
    ages = pd.Series(pd.cut(np.arange(0, 15), bins=[0, 5, 10, 15], right=False))
    expected_values_at_edges = [1, 0]  # At the edges should exactly match the intervals

    # Function call
    indicator_matrix = fine_coarse_matrix(ages)

    # Assertions
    assert indicator_matrix[4, 0] == expected_values_at_edges[0], "Edge inclusion at lower bound incorrect"
    assert indicator_matrix[5, 0] == expected_values_at_edges[1], "Edge exclusion at upper bound incorrect"

# More tests can be added as needed.
