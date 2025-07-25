import pytest
import numpy as np
from ...datasets import load_age_distribution, load_template_patterns
from ...utils import AgeBins
from ...sim import ParticipantGenerator, ContactMatrixGenerator, ContactGenerator
from .._SocialMix import SocialMix

# Language: python

df_age_dist = load_age_distribution('United_States')
patterns = load_template_patterns('United_States')

def test_basic_functionality():
		df_part = ParticipantGenerator(df_age_dist['P'].values).generate(n=1000)
		cint_matrix = ContactMatrixGenerator(patterns, df_age_dist['P'].values).generate()
		df_cnt = ContactGenerator(df_part, cint_matrix).generate()
  
		age_bins = AgeBins(0, 80, 5)
		sm = SocialMix(df_part, df_cnt, df_age_dist, age_bins, symmetric=True)
  
		cint = sm.compute_cint()
		assert cint.shape == (16, 16)
  
		rate = sm.compute_rate()
		assert rate.shape == (16, 16)
  
		sm.run_bootstrap(n_boot=10)
		assert sm.boots_cint.shape == (10, 16, 16)
		assert sm.boots_rate.shape == (10, 16, 16)
  
def test_only_one_participant():	
		df_part = ParticipantGenerator(df_age_dist['P'].values).generate(n=1)
		cint_matrix = ContactMatrixGenerator(patterns, df_age_dist['P'].values).generate()
		df_cnt = ContactGenerator(df_part, cint_matrix).generate()
  
		age_bins = AgeBins(0, 80, 5)
		sm = SocialMix(df_part, df_cnt, df_age_dist, age_bins, symmetric=True)
	
		cint = sm.compute_cint()
		assert cint.shape == (1, 1)
  
		cint = sm.compute_cint(recover_bins=True)
		assert cint.shape == (16, 16)
  
		rate = sm.compute_rate()
		assert rate.shape == (1, 1)

		rate = sm.compute_rate(recover_bins=True)
		assert rate.shape == (16, 16)
	
		sm.run_bootstrap(n_boot=10)
		assert sm.boots_cint.shape == (10, 16, 16)
		assert sm.boots_rate.shape == (10, 16, 16)