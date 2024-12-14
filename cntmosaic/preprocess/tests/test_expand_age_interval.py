import pytest
import pandas as pd
from .._utils import expand_age_interval

def test_basic_functionality():
    # Create a simple DataFrame
    data = pd.DataFrame({
      'age_part': [0, 1, 2],
      'age_grp_cnt': ['[0,5)', '[5,10)', '[10,15)'],
    })
    
    expanded_data = expand_age_interval(data, 'age_grp_cnt')
    
    # Check the dimensions
    assert expanded_data.shape == (15, 3), "Incorrect dimensions"
    
def test_additional_columns():
    # Create a simple DataFrame
    data = pd.DataFrame({
      'age_part': [0, 1, 2],
      'age_sex': ['M', 'F', 'M'],
      'age_grp_cnt': ['[0,5)', '[5,10)', '[10,15)'],
    })
    
    expanded_data = expand_age_interval(data, 'age_grp_cnt')
    
    # Check the dimensions
    assert expanded_data.shape == (15, 4), "Incorrect dimensions"
    
def test_custom_column_choice():
    # Create a simple DataFrame
    data = pd.DataFrame({
      'age_part': [0, 1, 2],
      'age_sex': ['M', 'F', 'M'],
      'custom_col': ['[0,5)', '[5,10)', '[10,15)'],
    })
    
    expanded_data = expand_age_interval(data, interval_col='custom_col')
    
    # Check the dimensions
    assert expanded_data.shape == (15, 4), "Incorrect dimensions"
    assert 'age_expanded' in expanded_data.columns, "age_expanded column not found"