from pathlib import Path
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson, nbinom

from .utils import symmetrise_patterns, smooth_patterns

def print_available_countries(repo_path: str) -> None:
	"""Print countries with synthetic contact patterns available in the mixing-patterns repository.
	
	Parameters
	----------
	repo_path: str
		Path to mixing-patterns repository
	"""
	
	data_dir = Path(repo_path) / 'data' / 'contact_matrices'
	assert data_dir.exists(), f"{data_dir} cannot be found"
	
	countries = set([file.stem.split('_')[0] for file in data_dir.glob('*.csv')])
	
	if 'United' in countries:
		countries.remove('United')
		countries.add('United_States')
	
	print('Available countries:')
	print(countries)

def load_contact_patterns(repo_path: str,
                          country: str,
                          level: str,
                          region: str=None,
                          n_age_groups: int=85,
                          symmetrise: bool=False,
                          smooth: bool=False) -> dict:
	"""Load synthetic contact patterns for a given country and region.
 
	Parameters
	----------
	repo_path: str
		Path to mixing-patterns repository
	country: str
		Country name
	level: str
		Level of contact patterns (country or subnational)
	region: str, optional
		Region name
	n_age_groups: int, default=85
		Number of age groups (18 or 85)
	symmetrise: bool, default=False
		Symmetrise the contact patterns
	smooth: bool, default=False
		Smooth the contact patterns
  
	Returns
	-------
	dict
		A dictionary containing contact patterns for household, school, work, and community settings
	"""
	
	file_path = Path(repo_path) / 'data' / 'contact_matrices'
	
	if level == 'country':
		prefix = f'{country}_country_level_F'
	else:
		prefix = f'{country}_subnational_{region}_F'
		
	patterns = {
		'household': pd.read_csv(file_path / f'{prefix}_household_setting_{n_age_groups}.csv', header=None).values,
		'school': pd.read_csv(file_path / f'{prefix}_school_setting_{n_age_groups}.csv', header=None).values,
		'work': pd.read_csv(file_path / f'{prefix}_work_setting_{n_age_groups}.csv', header=None).values,
		'community': pd.read_csv(file_path / f'{prefix}_community_setting_{n_age_groups}.csv', header=None).values
	}
	
	if symmetrise:
		patterns = symmetrise_patterns(patterns)
		
	if smooth:
		patterns = smooth_patterns(patterns)
	
	return patterns

def load_age_distribution(repo_path: str,
                          country: str,
                          level: str,
                          region: str=None,
                          n_age_groups: int=85) -> NDArray:
	"""Load age distribution for a given country and region.
 
	Parameters
	----------
	repo_path: str
		Path to mixing-patterns repository
	country: str
		Country name
	level: str
		Level of contact patterns (country or subnational)
	region: str, optional
		Region name
	n_age_groups: int, default=85
		Number of age groups (18 or 85)
  
	Returns
	-------
	NDArray
		Age distribution
	"""
	
	data_dir = Path(repo_path) / 'data' / 'population_rescaled_age_distributions'
	
	if level == 'country':
		file_name = f'{country}_country_level_age_distribution_{n_age_groups}.csv'
	else:
		file_name = f'{country}_subnational_{region}_age_distribution_{n_age_groups}.csv'
		
	age_dist = pd.read_csv(data_dir / file_name, header=None)
	age_dist.columns = ['age', 'P']
	
	return age_dist

def make_rate_pattern(patterns: dict,
                      age_dist: NDArray,
                      mixing_weights: list=[4.11, 11.41, 8.07, 2.79],
                      max_margin_cint: int=20) -> NDArray:
	"""Synthesise a rate matrix from contact patterns and a given population age distribution
 
	Parameters
	----------
	patterns: dict
		Dictionary of contact patterns, usually the output from load_contact_patterns
	age_dist: NDArray
		Population age distribution
	mixing_weights: list, default=[4.11, 11.41, 8.07, 2.79]
		Weights for each contact pattern (household, school, work, community)
	max_margin_cint: int, default=20
		Maximum margin contact intensity
  
	Returns
	-------
	NDArray
		Contact rate matrix
	"""
	
	X_hh = patterns['household']
	X_sc = patterns['school']
	X_cm = patterns['community']
	X_wk = patterns['work']
	
	w = mixing_weights
	pattern = w[0]*X_hh + w[1]*X_sc + w[2]*X_wk + w[3]*X_cm
	
	cint = pattern * age_dist[None,:]
	cint = cint / cint.sum(axis=1).max()
	cint = max_margin_cint * cint
	
	rate = cint / age_dist[None,:]
	return rate

def sample_contacts(
    N: int,
    cint: NDArray,
    sample_age_dist: NDArray,
    dist: str='poisson',
    overdisp: float=None
) -> pd.DataFrame:
	"""Sample contact counts from a specified degree distribution
	
	Parameters
	----------
	N: int
		Number of individuals
	cint: NDArray
		Contact intensity matrix.
	sample_age_dist: NDArray
		Sample age distribution.
	dist: str, default='poisson'
		Distribution to sample from ('poisson', 'nbinom', 'bnbinom').
	overdisp: float, optional
		Overdispersion parameter for negative binomial distribution and beta negative binomial distribution
  
	Returns
	-------
	DataFrame
		A DataFrame containing individual contact data
	"""
	# Validate inputs
	assert dist in ['poisson', 'nbinom', 'bnbinom'], 'Invalid distribution'
	if dist != 'poisson' and overdisp is None:
		raise ValueError("Overdispersion parameter is required for 'nbinom' and 'bnbinom'.")
 
	# Normalize sample age distribution
	age_probs = sample_age_dist / sample_age_dist.sum()
	results = []

	# Sample contact counts for each individual

	for i in range(N):
        # Sample the partner's age
		age_part = np.random.choice(len(age_probs), p=age_probs)
		mu = cint[age_part, :]

        # Sample contact counts based on the specified distribution
		if dist == 'poisson':
			sample = poisson.rvs(mu)
		elif dist == 'nbinom':
			n = mu**2 / (overdisp**2 - mu)
			p = mu / overdisp**2
			sample = nbinom.rvs(n, p)
		elif dist == 'bnbinom':
			raise NotImplementedError("Beta-negative binomial distribution is not yet implemented.")

		# Collect results for non-zero contact counts
		nonzero_indices = np.nonzero(sample)[0]
		for age_idx in nonzero_indices:
			results.append({
			    'id': i,
			    'age_part': age_part,
			    'age_cnt': age_idx,
			    'y': sample[age_idx]
			})
   
	return pd.DataFrame(results)
   
   