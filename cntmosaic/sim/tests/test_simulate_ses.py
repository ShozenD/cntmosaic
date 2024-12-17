import pytest
import numpy as np

from .._sim import simulate_ses
from cntmosaic.datasets import load_base_patterns, load_age_distribution

patterns = load_base_patterns('United_States')
age_dist = load_age_distribution('United_States')

def test_basic_functionality():
  df_sample, age_dist_props, evals = simulate_ses(patterns, age_dist.P.values)
  eval_cint, eval_mcint = evals
  
  assert df_sample['id'].nunique() <= 3000
  assert age_dist_props['ses'].shape == (85, 3)
  assert eval_cint['ses']['low'].shape == (85, 85)
  assert eval_mcint['ses']['low'].shape == (85,)
  np.testing.assert_approx_equal(eval_mcint['ses']['low'].max(), 20)
  np.testing.assert_approx_equal(eval_mcint['ses']['mid'].max(), 15)
  np.testing.assert_approx_equal(eval_mcint['ses']['high'].max(), 10)
  
def test_custom_config():
  config = {
    "low": {"mixing_weights": [4, 9, 15, 6], "pop_prop": 0.6, "cint_cap": 30, "sample_size": 1000},
    "mid": {"mixing_weights": [4, 9, 10, 3], "pop_prop": 0.39, "cint_cap": 20, "sample_size": 500},
    "high": {"mixing_weights": [4, 7, 5, 1], "pop_prop": 0.01, "cint_cap": 15, "sample_size": 50},
  }
  
  df_sample, age_dist_props, evals = simulate_ses(patterns, age_dist.P.values, config=config)
  eval_cint, eval_mcint = evals
  
  assert df_sample['id'].nunique() <= 1550
  assert df_sample[df_sample['ses'] == 'low']['id'].nunique() <= 1000
  assert df_sample[df_sample['ses'] == 'mid']['id'].nunique() <= 500
  assert df_sample[df_sample['ses'] == 'high']['id'].nunique() <= 50
  np.testing.assert_approx_equal(eval_mcint['ses']['low'].max(), 30)
  np.testing.assert_approx_equal(eval_mcint['ses']['mid'].max(), 20)
  np.testing.assert_approx_equal(eval_mcint['ses']['high'].max(), 15)