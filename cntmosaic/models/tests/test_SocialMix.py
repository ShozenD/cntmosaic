import pytest
import numpy as np
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ParticipantGenerator, ContactMatrixGenerator, ContactGenerator
from .._SocialMix import SocialMix

def test_basic_functionality():
		df_age_dist = load_age_distribution('United_States')
		patterns = load_template_patterns('United_States')
  
		df_part = ParticipantGenerator(1000, df_age_dist['P'].values).generate()
		cint_matrix = ContactMatrixGenerator(patterns, df_age_dist['P'].values).generate()
		df_cnt = ContactGenerator(df_part, cint_matrix).generate()
  
		sm = SocialMix(df_part, df_cnt, df_age_dist, age_limits=np.arange(0, 86, 5), symmetric=True)
  
		cint = sm.compute_cint()
		assert cint.shape == (17, 17)
  
		rate = sm.compute_rate()
		assert rate.shape == (17, 17)
  
		sm.run_bootstrap(n_boot=10)
		assert sm.boots_cint.shape == (10, 17, 17)
		assert sm.boots_rate.shape == (10, 17, 17)
  