import pytest
import numpy as np
from .._ParticipantGenerator import ParticipantGenerator, Subgroup
from ...datasets._base import load_age_distribution


def test_single_subgroup_input():
  """
  Test ParticipantGenerator with a single Subgroup input.
  """
  # Create a simple Subgroup
  age_dist = np.array([100, 200, 300, 400, 500])
  subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=15.0)
  
  pg = ParticipantGenerator(subgroup)
  df_part = pg.generate(seed=42)
  
  # Check the shape of the generated DataFrame
  assert df_part.shape == (1000, 2), "DataFrame should have 1000 rows and 2 columns"
  
  # Check the columns of the generated DataFrame
  assert list(df_part.columns) == ['id', 'age_group'], "Columns should be 'id' and 'age_group'"
  
  # Check that IDs are unique and sequential
  assert df_part['id'].is_unique, "IDs should be unique"
  assert df_part['id'].min() == 1, "IDs should start at 1"
  assert df_part['id'].max() == 1000, "IDs should end at 1000"
  
  # Check that ages are within valid range
  assert df_part['age_group'].min() >= 0, "Ages should be non-negative"
  assert df_part['age_group'].max() < len(age_dist), "Ages should be less than the length of age distribution"
  
  # Check that there's no subgroup column
  assert 'subgroup' not in df_part.columns, "Single Subgroup input should not have subgroup column"


def test_list_of_subgroups_input():
  """
  Test ParticipantGenerator with a list of Subgroup objects input (no labels).
  """
  # Create Subgroup objects without labels
  subgroup_1 = Subgroup(n=800, age_dist=np.array([100, 200, 300, 400]), mean_cint_margin=15.0)
  subgroup_2 = Subgroup(n=800, age_dist=np.array([500, 400, 300, 200]), mean_cint_margin=20.0)
  
  pg = ParticipantGenerator([subgroup_1, subgroup_2])
  df_part = pg.generate(seed=42)
  
  # Check the shape - should have 800 rows per subgroup = 1600 total
  assert df_part.shape == (1600, 3), "DataFrame should have 1600 rows (800 per subgroup) and 3 columns"
  
  # Check the columns
  assert set(df_part.columns) == {'id', 'age_group', 'subgroup'}, "Columns should include 'id', 'age_group', and 'subgroup'"
  
  # Check that IDs are unique and sequential
  assert df_part['id'].is_unique, "IDs should be unique"
  assert df_part['id'].min() == 1, "IDs should start at 1"
  assert df_part['id'].max() == 1600, "IDs should end at 1600"
  
  # Check subgroups - should use numeric indices when no labels provided
  assert df_part['subgroup'].nunique() == 2, "Should have 2 subgroups"
  assert set(df_part['subgroup'].unique()) == {0, 1}, "Subgroups should be 0 and 1"
  
  # Check that each subgroup has the correct number of participants
  subgroup_counts = df_part['subgroup'].value_counts()
  assert subgroup_counts[0] == 800, "Subgroup 0 should have 800 participants"
  assert subgroup_counts[1] == 800, "Subgroup 1 should have 800 participants"


def test_list_of_subgroups_with_labels():
  """
  Test ParticipantGenerator with a list of Subgroup objects with custom labels.
  """
  # Create Subgroup objects with custom labels
  subgroup_young = Subgroup(n=500, age_dist=np.array([1000, 800, 600, 400, 200]), mean_cint_margin=18.0, label='young')
  subgroup_old = Subgroup(n=500, age_dist=np.array([200, 400, 600, 800, 1000]), mean_cint_margin=12.0, label='old')
  
  pg = ParticipantGenerator([subgroup_young, subgroup_old])
  df_part = pg.generate(seed=99)
  
  # Check the shape - should have 500 rows per subgroup = 1000 total
  assert df_part.shape == (1000, 3), "DataFrame should have 1000 rows (500 per subgroup) and 3 columns"
  
  # Check the columns
  assert set(df_part.columns) == {'id', 'age_group', 'subgroup'}, "Columns should include 'id', 'age_group', and 'subgroup'"
  
  # Check that IDs are unique and sequential
  assert df_part['id'].is_unique, "IDs should be unique"
  assert df_part['id'].min() == 1, "IDs should start at 1"
  assert df_part['id'].max() == 1000, "IDs should end at 1000"
  
  # Check subgroups - should use custom labels
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
  subgroup = Subgroup(n=100, age_dist=np.array([100, 200, 300, 400, 500]), mean_cint_margin=15.0)
  
  pg1 = ParticipantGenerator(subgroup)
  df_part1 = pg1.generate(seed=12345)
  
  pg2 = ParticipantGenerator(subgroup)
  df_part2 = pg2.generate(seed=12345)
  
  # Check that the generated data is identical
  assert df_part1.equals(df_part2), "Same seed should produce identical results"


def test_different_seeds_produce_different_results():
  """
  Test that different seeds produce different results.
  """
  subgroup = Subgroup(n=100, age_dist=np.array([100, 200, 300, 400, 500]), mean_cint_margin=15.0)
  
  pg1 = ParticipantGenerator(subgroup)
  df_part1 = pg1.generate(seed=111)
  
  pg2 = ParticipantGenerator(subgroup)
  df_part2 = pg2.generate(seed=222)
  
  # Check that the generated data is different
  assert not df_part1['age_group'].equals(df_part2['age_group']), "Different seeds should produce different results"


def test_invalid_n_value():
  """
  Test that ValueError is raised when n is invalid.
  """
  # Test with n=0
  subgroup = Subgroup(n=0, age_dist=np.array([100, 200, 300, 400]), mean_cint_margin=15.0)
  pg = ParticipantGenerator(subgroup)
  
  # n=0 should generate an empty dataframe (not an error)
  df_part = pg.generate(seed=42)
  assert len(df_part) == 0, "n=0 should generate empty DataFrame"


def test_invalid_input_types():
  """
  Test that TypeError is raised for invalid input types.
  """
  # Test with raw NDArray (should fail)
  with pytest.raises(TypeError, match="subgroups must be Subgroup"):
    ParticipantGenerator(np.array([100, 200, 300]))
  
  # Test with list containing non-Subgroup elements
  with pytest.raises(TypeError, match="List elements must be Subgroup"):
    ParticipantGenerator([np.array([100, 200, 300])])
  
  # Test with dict (should fail - no longer supported)
  with pytest.raises(TypeError, match="subgroups must be Subgroup"):
    ParticipantGenerator({'test': Subgroup(n=100, age_dist=np.array([100, 200, 300]), mean_cint_margin=15.0)})


def test_age_distribution_normalization():
  """
  Test that age distributions are correctly normalized to proportions.
  """
  # Use unnormalized distribution
  age_dist = np.array([100, 200, 300, 400])
  subgroup = Subgroup(n=500, age_dist=age_dist, mean_cint_margin=15.0)
  
  pg = ParticipantGenerator(subgroup)
  
  # Check that age_proportions sums to 1
  assert np.isclose(pg.age_proportions.sum(), 1.0), "Age proportions should sum to 1"
  
  # Check that proportions are correct
  expected_prop = age_dist / age_dist.sum()
  assert np.allclose(pg.age_proportions, expected_prop), "Age proportions should be correctly normalized"


def test_basic_functionality():
  """
  Test the basic functionality of loading age distribution data.
  """
  # Load the age distribution for the United States with a maximum age of 80
  df_age_dist = load_age_distribution('United_States', max_age=80)
  age_dist = df_age_dist['P'].values
  
  # ===== Single group ======
  subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=15.0)
  pg = ParticipantGenerator(subgroup)
  df_part = pg.generate(seed=0)
  
  # Check the shape of the generated DataFrame
  assert df_part.shape == (1000, 2)
  
  # Check the columns of the generated DataFrame
  assert df_part.columns.isin(['id', 'age_group']).all()
  
  # Multiple groups
  subgroups = [
    Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=15.0),
    Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=20.0)
  ]
  pg = ParticipantGenerator(subgroups)
  df_part = pg.generate(seed=0)
  assert df_part.shape == (2000, 3)
  assert df_part.columns.isin(['id', 'age_group', 'subgroup']).all()


def test_different_sample_sizes():
  """
  Test that subgroups can have different sample sizes.
  """
  # Create Subgroup objects with different n values
  subgroup_1 = Subgroup(n=300, age_dist=np.array([100, 200, 300, 400]), mean_cint_margin=15.0, label='small')
  subgroup_2 = Subgroup(n=700, age_dist=np.array([500, 400, 300, 200]), mean_cint_margin=20.0, label='large')
  
  pg = ParticipantGenerator([subgroup_1, subgroup_2])
  df_part = pg.generate(seed=42)
  
  # Check total size
  assert len(df_part) == 1000, "Total should be 300 + 700 = 1000"
  
  # Check that each subgroup has the correct number of participants
  subgroup_counts = df_part['subgroup'].value_counts()
  assert subgroup_counts['small'] == 300, "Subgroup 'small' should have 300 participants"
  assert subgroup_counts['large'] == 700, "Subgroup 'large' should have 700 participants"