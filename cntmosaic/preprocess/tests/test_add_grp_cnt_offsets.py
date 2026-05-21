import pytest

import pandas as pd

from .._preprocess import add_grp_cnt_offsets

def test_basic_functionality():
  # Create a simple DataFrame
  df_cnt = pd.DataFrame({
    'part_age': [0, 1, 2],
    'cnt_age': [0, 1, 2],
    'part_sex': ['M', 'F', 'M'],
    'y': [1, 1, 1]
  })
  
  df_grp = pd.DataFrame({
    'part_age': [0, 1, 2],
    'part_sex': ['M', 'F', 'M'],
    'z': [2, 3, 4]
  })
  
  df = add_grp_cnt_offsets(df_cnt, df_grp, 'part_sex')
  assert df.shape == (3, 5), "Incorrect dimensions"
  assert df['S'].values == pytest.approx([1/3, 1/4, 1/5]), "Incorrect values"
  
def test_no_grouping_vars():
  # Create a simple DataFrame
  df_cnt = pd.DataFrame({
    'part_age': [0, 1, 2],
    'cnt_age': [0, 1, 2],
    'y': [1, 1, 1]
  })
  
  df_grp = pd.DataFrame({
    'part_age': [0, 1, 2],
    'z': [2, 3, 4]
  })
  
  df = add_grp_cnt_offsets(df_cnt, df_grp)
  
  assert df.shape == (3, 4), "Incorrect dimensions"
  assert df['S'].values == pytest.approx([1/3, 1/4, 1/5]), "Incorrect values"
  
def test_no_y_column():
  # Create a simple DataFrame
  df_cnt = pd.DataFrame({
    'part_age': [0, 1, 2],
    'cnt_age': [0, 1, 2]
  })
  
  df_grp = pd.DataFrame({
    'part_age': [0, 1, 2],
    'z': [2, 3, 4]
  })
  
  with pytest.warns(RuntimeWarning):
    df = add_grp_cnt_offsets(df_cnt, df_grp)
    
def test_no_z_column():
  # Create a simple DataFrame
  df_cnt = pd.DataFrame({
    'part_age': [0, 1, 2],
    'cnt_age': [0, 1, 2],
    'y': [1, 1, 1]
  })
  
  df_grp = pd.DataFrame({
    'part_age': [0, 1, 2]
  })
  
  with pytest.raises(RuntimeError):
    df = add_grp_cnt_offsets(df_cnt, df_grp)