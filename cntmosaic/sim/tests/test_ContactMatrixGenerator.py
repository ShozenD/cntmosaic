import pytest

from ...datasets._base import load_age_distribution, load_template_patterns
from .._ContactMatrixGenerator import ContactMatrixGenerator

def test_basic_functionality():
  df_age_dist = load_age_distribution('United_States', max_age=80)
  patterns = load_template_patterns('United_States', max_age=80)
  
  age_dist = df_age_dist['P'].values
  cint_matrix = ContactMatrixGenerator(patterns, age_dist).generate(seed=0)
  
  assert cint_matrix.shape == (81, 81)