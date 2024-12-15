import pytest

import pandas as pd

from .._preprocess import add_grp_cnt_offsets

def test_basic_functionality():
  # Create a simple DataFrame
  df_cnt = pd.DataFrame({
    'age_part': [0, 1, 2],
    'age_cnt': [0, 1, 2],
    'sex_part': ['M', 'F', 'M'],
    'y': [1, 1, 1]
  })
  
  df_grp = pd.DataFrame({
    'age_part': [0, 1, 2],
    'sex_part': ['M', 'F', 'M'],
    'z': [2, 3, 4]
  })
  
  df = add_grp_cnt_offsets(df_cnt, df_grp, 'sex_part')
  assert df.shape == (3, 3), "Incorrect dimensions"
  assert df['S'].values == pytest.approx([1/3, 1/4, 1/5]), "Incorrect values"
  
def test_no_grouping_vars():
  # Create a simple DataFrame
  df_cnt = pd.DataFrame({
    'age_part': [0, 1, 2],
    'age_cnt': [0, 1, 2],
    'y': [1, 1, 1]
  })
  
  df_grp = pd.DataFrame({
    'age_part': [0, 1, 2],
    'z': [2, 3, 4]
  })
  
  df = add_grp_cnt_offsets(df_cnt, df_grp)
  
  assert df.shape == (3, 2), "Incorrect dimensions"
  assert df['S'].values == pytest.approx([1/3, 1/4, 1/5]), "Incorrect values"
  
def test_no_y_column():
  # Create a simple DataFrame
  df_cnt = pd.DataFrame({
    'age_part': [0, 1, 2],
    'age_cnt': [0, 1, 2]
  })
  
  df_grp = pd.DataFrame({
    'age_part': [0, 1, 2],
    'z': [2, 3, 4]
  })
  
  with pytest.warns(RuntimeWarning):
    df = add_grp_cnt_offsets(df_cnt, df_grp)
    
def test_no_z_column():
  # Create a simple DataFrame
  df_cnt = pd.DataFrame({
    'age_part': [0, 1, 2],
    'age_cnt': [0, 1, 2],
    'y': [1, 1, 1]
  })
  
  df_grp = pd.DataFrame({
    'age_part': [0, 1, 2]
  })
  
  with pytest.raises(RuntimeError):
    df = add_grp_cnt_offsets(df_cnt, df_grp)