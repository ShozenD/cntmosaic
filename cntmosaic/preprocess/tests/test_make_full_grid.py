import numpy as np
import pandas as pd
import pytest
from .._utils import make_full_grid

def test_basic_functionality():
    # Create a simple DataFrame
    data = pd.DataFrame({
      'part_age': [0, 1, 2],
      'cnt_age': [0, 1, 2],
      'part_sex': ['M', 'F', 'M'],
      'cnt_sex': ['M', 'F', 'F'],
    })
    
    df_grid = make_full_grid(data, ['part_age', 'cnt_age'], ['part_sex', 'cnt_sex'])
    
    # Check the dimensions
    assert df_grid.shape == (36, 4), "Incorrect dimensions"
    
    df_grid = make_full_grid(data, ['part_age', 'cnt_age'], ['part_sex'])
    
    # Check the dimensions
    assert df_grid.shape == (18, 3), "Incorrect dimensions"
    
def test_no_grouping_vars():
    # Create a simple DataFrame
    data = pd.DataFrame({
      'part_age': [0, 1, 2],
      'cnt_age': [0, 1, 2],
    })
    
    df_grid = make_full_grid(data, ['part_age', 'cnt_age'])
    
    # Check the dimensions
    assert df_grid.shape == (9, 2), "Incorrect dimensions"
    
def test_partially_missing_age_bounds():
    data = pd.DataFrame({
      'part_age': [0, 1],
      'cnt_age': [1, 2]
    })
    
    df_grid = make_full_grid(data, ['part_age', 'cnt_age'])
    
    # Check the dimensions
    assert df_grid.shape == (9, 2), "Incorrect dimensions"
    
def test_non_zero_start_age():
    data = pd.DataFrame({
      'part_age': [1, 2],
      'cnt_age': [1, 2]
    })
    
    df_grid = make_full_grid(data, ['part_age', 'cnt_age'])
    
    # Check the dimensions
    assert df_grid.shape == (4, 2), "Incorrect dimensions"