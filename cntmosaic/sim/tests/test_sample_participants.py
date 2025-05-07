import pytest
import numpy as np
from numpy.testing import assert_array_equal

from .._sim import sample_participants

def test_base_functionality():
  base = np.array([1, 2, 3])
  
  age_dists = {
    'base': base,
    'sex': {
      'men': np.array([1, 2, 3]),
      'women': np.array([3, 2, 1])
    },
    'hhsize': {
      '1': np.array([1, 2, 3]),
      '2': np.array([2, 3, 1]),
      '3': np.array([3, 2, 1]),
      '4': np.array([1, 2, 3]),
      '5+': np.array([2, 3, 1]),
    },
    'ses': {
      'low': np.array([1, 2, 3]),
      'med': np.array([2, 3, 1]),
      'high': np.array([3, 2, 1])
    }
  }
  
  df_part = sample_participants(2500, age_dists)
  assert df_part.shape[0] == 2500
  assert df_part.columns.isin(['id_part', 'age_part', 'sex_part', 'hhsize_part', 'ses_part']).all()
  assert df_part['age_part'].isin([0,1,2]).all()
  assert df_part['sex_part'].isin(['men', 'women']).all()
  assert df_part['hhsize_part'].isin(['1', '2', '3', '4', '5+']).all()
  assert df_part['ses_part'].isin(['low', 'med', 'high']).all()