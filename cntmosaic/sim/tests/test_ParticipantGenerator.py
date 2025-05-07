import pytest
from .._ParticipantGenerator import ParticipantGenerator
from ...datasets._base import load_age_distribution

def test_basic_functionality():
  """
  Test the basic functionality of loading age distribution data.
  """
  # Load the age distribution for the United States with a maximum age of 80
  df_age_dist = load_age_distribution('United_States', max_age=80)
  age_dist = df_age_dist['P'].values
  
  # ===== Single group ======
  pg = ParticipantGenerator(age_dist)
  df_part = pg.generate(1000, seed=0)
  
  # Check the shape of the generated DataFrame
  assert df_part.shape == (1000, 2)
  
  # Check the columns of the generated DataFrame
  assert df_part.columns.isin(['id', 'age_part']).all()
  
  # Multiple groups
  pg = ParticipantGenerator([age_dist, age_dist])
  df_part = pg.generate([1000, 2000], seed=0)
  assert df_part.shape == (3000, 3)
  assert df_part.columns.isin(['id', 'age_part', 'subgroup']).all()