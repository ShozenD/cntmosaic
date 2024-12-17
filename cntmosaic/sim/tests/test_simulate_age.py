import pytest

from .._sim import simulate_age
from cntmosaic.datasets._base import load_base_patterns, load_age_distribution

patterns = load_base_patterns('United_States', 'country')
age_dist = load_age_distribution('United_States', 'country')

def test_basic_functionality():
  df_sample, df_eval = simulate_age(patterns, age_dist.P.values)
  
  assert df_sample.id.nunique() == 2500
  assert df_eval.cint.max() <= 20
  
def test_custom_parameters():
  df_sample, df_eval = simulate_age(patterns,
                                    age_dist.P.values,
                                    N=5000,
                                    max_margin_cint=30)
  
  assert df_sample.id.nunique() == 5000
  assert df_eval.cint.max() <= 30
  
def test_nbinom():
  df_sample, df_eval = simulate_age(patterns,
                                    age_dist.P.values,
                                    dist='nbinom',
                                    overdisp=1.2)
  
  assert df_sample.id.nunique() == 2500
  assert df_eval.cint.max() <= 20