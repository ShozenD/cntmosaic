import numpy as np
import pandas as pd
import pytest
from .._utils import make_full_grid

def test_basic_functionality():
    # Create a simple DataFrame
    data = pd.DataFrame({
      'age_part': [0, 1, 2],
      'age_cnt': [0, 1, 2],
      'sex_part': ['M', 'F', 'M'],
      'sex_cnt': ['M', 'F', 'F'],
    })
    
    df_grid = make_full_grid(data, ['age_part', 'age_cnt'], ['sex_part', 'sex_cnt'])
    
    # Check the dimensions
    assert df_grid.shape == (36, 4), "Incorrect dimensions"
    
    df_grid = make_full_grid(data, ['age_part', 'age_cnt'], ['sex_part'])
    
    # Check the dimensions
    assert df_grid.shape == (18, 3), "Incorrect dimensions"
    
def test_no_grouping_vars():
    # Create a simple DataFrame
    data = pd.DataFrame({
      'age_part': [0, 1, 2],
      'age_cnt': [0, 1, 2],
    })
    
    df_grid = make_full_grid(data, ['age_part', 'age_cnt'])
    
    # Check the dimensions
    assert df_grid.shape == (9, 2), "Incorrect dimensions"
    
def test_partially_missing_age_bounds():
    data = pd.DataFrame({
      'age_part': [0, 1],
      'age_cnt': [1, 2]
    })
    
    df_grid = make_full_grid(data, ['age_part', 'age_cnt'])
    
    # Check the dimensions
    assert df_grid.shape == (9, 2), "Incorrect dimensions"
    
def test_non_zero_start_age():
    data = pd.DataFrame({
      'age_part': [1, 2],
      'age_cnt': [1, 2]
    })
    
    df_grid = make_full_grid(data, ['age_part', 'age_cnt'])
    
    # Check the dimensions
    assert df_grid.shape == (4, 2), "Incorrect dimensions"