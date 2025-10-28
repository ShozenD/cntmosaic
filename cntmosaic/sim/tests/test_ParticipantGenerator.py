import pytest
import numpy as np
from .._ParticipantGenerator import ParticipantGenerator
from ...datasets._base import load_age_distribution


def test_ndarray_input():
  """
  Test ParticipantGenerator with a single NDArray input.
  """
  # Create a simple age distribution
  age_dist = np.array([100, 200, 300, 400, 500])
  
  pg = ParticipantGenerator(age_dist)
  df_part = pg.generate(1000, seed=42)
  
  # Check the shape of the generated DataFrame
  assert df_part.shape == (1000, 2), "DataFrame should have 1000 rows and 2 columns"
  
  # Check the columns of the generated DataFrame
  assert list(df_part.columns) == ['id', 'age_part'], "Columns should be 'id' and 'age_part'"
  
  # Check that IDs are unique and sequential
  assert df_part['id'].is_unique, "IDs should be unique"
  assert df_part['id'].min() == 1, "IDs should start at 1"
  assert df_part['id'].max() == 1000, "IDs should end at 1000"
  
  # Check that ages are within valid range
  assert df_part['age_part'].min() >= 0, "Ages should be non-negative"
  assert df_part['age_part'].max() < len(age_dist), "Ages should be less than the length of age distribution"
  
  # Check that there's no subgroup column
  assert 'subgroup' not in df_part.columns, "Single NDArray input should not have subgroup column"


def test_ndarray_input_with_sample_age_prop():
  """
  Test ParticipantGenerator initialized with sample_age_prop as NDArray.
  """
  # Create a normalized age proportion
  age_prop = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
  
  pg = ParticipantGenerator(sample_age_prop=age_prop)
  df_part = pg.generate(500, seed=123)
  
  # Check the shape
  assert df_part.shape == (500, 2), "DataFrame should have 500 rows and 2 columns"
  
  # Check columns
  assert list(df_part.columns) == ['id', 'age_part'], "Columns should be 'id' and 'age_part'"
  
  # Check age range
  assert df_part['age_part'].min() >= 0
  assert df_part['age_part'].max() < len(age_prop)


def test_list_of_ndarrays_input():
  """
  Test ParticipantGenerator with a list of NDArrays input.
  """
  # Create two different age distributions
  age_dist_1 = np.array([100, 200, 300, 400])
  age_dist_2 = np.array([500, 400, 300, 200])
  
  pg = ParticipantGenerator([age_dist_1, age_dist_2])
  df_part = pg.generate(800, seed=42)
  
  # Check the shape - should have 800 rows per subgroup = 1600 total
  assert df_part.shape == (1600, 3), "DataFrame should have 1600 rows (800 per subgroup) and 3 columns"
  
  # Check the columns
  assert set(df_part.columns) == {'id', 'age_part', 'subgroup'}, "Columns should include 'id', 'age_part', and 'subgroup'"
  
  # Check that IDs are unique and sequential
  assert df_part['id'].is_unique, "IDs should be unique"
  assert df_part['id'].min() == 1, "IDs should start at 1"
  assert df_part['id'].max() == 1600, "IDs should end at 1600"
  
  # Check subgroups
  assert df_part['subgroup'].nunique() == 2, "Should have 2 subgroups"
  assert set(df_part['subgroup'].unique()) == {0, 1}, "Subgroups should be 0 and 1"
  
  # Check that each subgroup has the correct number of participants
  subgroup_counts = df_part['subgroup'].value_counts()
  assert subgroup_counts[0] == 800, "Subgroup 0 should have 800 participants"
  assert subgroup_counts[1] == 800, "Subgroup 1 should have 800 participants"


def test_dict_of_ndarrays_input():
  """
  Test ParticipantGenerator with a dictionary of NDArrays input.
  """
  # Create age distributions with string keys
  age_dist_dict = {
    'young': np.array([1000, 800, 600, 400, 200]),
    'old': np.array([200, 400, 600, 800, 1000])
  }
  
  pg = ParticipantGenerator(age_dist_dict)
  df_part = pg.generate(500, seed=99)
  
  # Check the shape - should have 500 rows per subgroup = 1000 total
  assert df_part.shape == (1000, 3), "DataFrame should have 1000 rows (500 per subgroup) and 3 columns"
  
  # Check the columns
  assert set(df_part.columns) == {'id', 'age_part', 'subgroup'}, "Columns should include 'id', 'age_part', and 'subgroup'"
  
  # Check that IDs are unique and sequential
  assert df_part['id'].is_unique, "IDs should be unique"
  assert df_part['id'].min() == 1, "IDs should start at 1"
  assert df_part['id'].max() == 1000, "IDs should end at 1000"
  
  # Check subgroups
  assert df_part['subgroup'].nunique() == 2, "Should have 2 subgroups"
  assert set(df_part['subgroup'].unique()) == {'young', 'old'}, "Subgroups should be 'young' and 'old'"
  
  # Check that each subgroup has the correct number of participants
  subgroup_counts = df_part['subgroup'].value_counts()
  assert subgroup_counts['young'] == 500, "Subgroup 'young' should have 500 participants"
  assert subgroup_counts['old'] == 500, "Subgroup 'old' should have 500 participants"


def test_reproducibility_with_seed():
  """
  Test that the same seed produces the same results.
  """
  age_dist = np.array([100, 200, 300, 400, 500])
  
  pg1 = ParticipantGenerator(age_dist)
  df_part1 = pg1.generate(100, seed=12345)
  
  pg2 = ParticipantGenerator(age_dist)
  df_part2 = pg2.generate(100, seed=12345)
  
  # Check that the generated data is identical
  assert df_part1.equals(df_part2), "Same seed should produce identical results"


def test_different_seeds_produce_different_results():
  """
  Test that different seeds produce different results.
  """
  age_dist = np.array([100, 200, 300, 400, 500])
  
  pg1 = ParticipantGenerator(age_dist)
  df_part1 = pg1.generate(100, seed=111)
  
  pg2 = ParticipantGenerator(age_dist)
  df_part2 = pg2.generate(100, seed=222)
  
  # Check that the generated data is different
  assert not df_part1['age_part'].equals(df_part2['age_part']), "Different seeds should produce different results"


def test_invalid_input():
  """
  Test that ValueError is raised when neither sample_age_dist nor sample_age_prop is provided.
  """
  with pytest.raises(ValueError, match="Either sample_age_dist or sample_age_prop must be provided"):
    ParticipantGenerator()


def test_age_distribution_normalization():
  """
  Test that age distributions are correctly normalized to proportions.
  """
  # Use unnormalized distribution
  age_dist = np.array([100, 200, 300, 400])
  
  pg = ParticipantGenerator(age_dist)
  
  # Check that sample_age_prop sums to 1
  assert np.isclose(pg.sample_age_prop.sum(), 1.0), "Age proportions should sum to 1"
  
  # Check that proportions are correct
  expected_prop = age_dist / age_dist.sum()
  assert np.allclose(pg.sample_age_prop, expected_prop), "Age proportions should be correctly normalized"


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
  df_part = pg.generate(1000, seed=0)
  assert df_part.shape == (2000, 3)
  assert df_part.columns.isin(['id', 'age_part', 'subgroup']).all()