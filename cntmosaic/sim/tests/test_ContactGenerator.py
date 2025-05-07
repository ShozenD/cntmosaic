import pytest

from ...datasets._base import load_age_distribution, load_template_patterns
from .._ParticipantGenerator import ParticipantGenerator
from .._ContactMatrixGenerator import ContactMatrixGenerator
from .._ContactGenerator import ContactGenerator

def test_basic_functionality():
  df_age_dist = load_age_distribution('United_States', max_age=80)
  patterns = load_template_patterns('United_States', max_age=80)
  age_dist = df_age_dist['P'].values
  
  # ===== Single subgroup ======
  df_part = ParticipantGenerator(1000, age_dist).generate(seed=0)
  cint_matrix = ContactMatrixGenerator(patterns, age_dist).generate(seed=0)
  
  cg = ContactGenerator(df_part, cint_matrix)
  df_cnt = cg.generate()
  
  # Check the shape of the generated DataFrame (The first dimension is random)
  assert df_cnt.shape[1] == 3
  
  # ===== Multiple subgroups ======
  df_part = ParticipantGenerator([1000, 2000], [age_dist, age_dist]).generate(seed=0)
  cmg = ContactMatrixGenerator(patterns, age_dist)
  cint_matrices = [cmg.generate(seed = i) for i in range(2)]
  cg = ContactGenerator(df_part, cint_matrices)
  df_cnt = cg.generate(seed=0)
  
  # Check the shape of the generated DataFrame (The first dimension is random)
  assert df_cnt.shape[1] == 4
  