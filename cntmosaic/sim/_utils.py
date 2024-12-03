from pathlib import Path
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

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

def symmetrise_patterns(patterns: dict) -> dict:
	"""Symmetrise the contact patterns by averaging the values of the upper and lower triangles.
	
	Parameters
	----------
	patterns: dict
		Dictionary of contact patterns. Usually the output of load_contact_patterns.
  
	Returns
	-------
	dict
		Dictionary of symmetrised contact patterns.
	"""
	
	x = patterns['household']
	x[:,84] = x[:,83]
	patterns['household'] = (x + x.T) / 2

	x = patterns['community']
	x[:,84] = x[:,83]
	patterns['community'] = (x + x.T) / 2

	x = patterns['school']
	patterns['school'] = (x + x.T) / 2

	x = patterns['work']
	patterns['work'] = (x + x.T) / 2
	
	return patterns

def smooth_patterns(patterns: dict) -> dict:
	"""Smooth the contact patterns by applying a Gaussian filter to each pattern.
 
	Parameters
	----------
	patterns: dict
		Dictionary of contact patterns. Usually the output of load_contact_patterns.
  
	Returns
	-------
	dict
		Dictionary of smoothed contact patterns.
	"""
		
	patterns['household'] = gaussian_filter(patterns['household'], sigma=1)
	patterns['school'] = gaussian_filter(patterns['school'], sigma=1)
	patterns['work'] = gaussian_filter(patterns['work'], sigma=1)
	patterns['community'] = gaussian_filter(patterns['community'], sigma=1)
	
	return patterns

def load_base_patterns(repo_path: str,
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
 
	if n_age_groups == 85:
		x = age_dist['P'].values
		x[84] = x[83]
		age_dist['P'] = x
	
	return age_dist