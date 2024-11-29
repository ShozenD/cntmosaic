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

def make_contact_pattern(patterns: dict,
                      age_dist: NDArray,
                      mixing_weights: list=[4.11, 11.41, 8.07, 2.79],
                      max_margin_cint: int=20) -> tuple:
	"""Synthesise a rate and intensity matrix from contact patterns and a given population age distribution
 
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
	tuple
		(Contact rate matrix, Contact intensity matrix)
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
	return rate, cint

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

def simulate_age(patterns: dict,
                 age_dist: NDArray,
                 dist: str='poisson',
                 N: int=2500,
                 max_margin_cint: int=20,
                 mixing_weights: list=None) -> tuple:
    """Simulate basic contact patterns and return sample and evaluation DataFrames.
    
    Parameters
    ----------
    patterns: dict
        Base contact patterns.
    age_dist: NDArray
        Age distribution array.
    dist: str, default='poisson'
        Distribution type. Options: 'poisson', 'nbinom', 'bnbinom'.
    N: int, default=2500
        Number of individuals.
    max_margin_cint: int, default=20
        Maximum margin contact intensity.
    
    Returns
    -------
    tuple
        Sample and evaluation DataFrames.
        
    Example
    -------
    
    Generate simulated contact data.
    
    >>> patterns = load_base_patterns('path/to/repo', 'United_States', 'country')
    >>> age_dist = load_age_distribution('path/to/repo', 'United_States', 'country')
    >>> df_sample, df_eval = simulate_age(patterns, age_dist.P.values)
    """
    mixing_weights = mixing_weights or [4.11, 11.41, 8.07, 2.79]
    
    # Make sample data
    rate, cint = make_contact_pattern(patterns, age_dist, mixing_weights, max_margin_cint)
    df_sample = sample_contacts(1000, cint, age_dist, dist=dist)
    
    # Make evaluation data
    aidx = np.array(np.meshgrid(range(len(age_dist)), range(len(age_dist)))).T.reshape(-1, 2)
    df_eval = pd.DataFrame({
        'age_part': aidx[:, 0], 'age_cnt': aidx[:, 1],
        'rate': rate[aidx[:, 0], aidx[:, 1]],
        'cint': cint[aidx[:, 0], aidx[:, 1]]
    })
    
    return df_sample, df_eval
   
def simulate_ses(
    patterns: dict,
    age_dist: NDArray,
    dist: str = "poisson",
    config: dict = None
) -> tuple:
    """Simulate SES-based contact patterns and return sample and evaluation DataFrames.
    
    Parameters
    ----------
    patterns: dict
        Base contact patterns.
    age_dist: NDArray
        Age distribution array.
    dist: str, default="poisson"
        Distribution type. Options: "poisson", "nbinom", "bnbinom".
    config: dict, optional
        SES configuration, including mixing weights, proportions, and caps.
        
    Returns
    -------
    tuple
        Sample DataFrame, Age distribution proportion dictionary, and evaluation DataFrame.
        
    Examples
    --------
    Generate using default the option.
    
    >>> patterns = load_base_patterns('path/to/repo', 'United_States', 'country')
    >>> age_dist = load_age_distribution('path/to/repo', 'United_States', 'country')
    >>> df_sample, age_dist_props, df_eval = simulate_ses(patterns, age_dist.P.values)
    
    Customise SES pattern configuration.
    
    >>> patterns = load_base_patterns('path/to/repo', 'United_States', 'country')
    >>> age_dist = load_age_distribution('path/to/repo', 'United_States', 'country')
    >>> config = {
    ...     "low": {"mixing_weights": [4, 9, 15, 6], "pop_prop": 0.6, "cint_cap": 20, "sample_size": 1000},
    ...     "mid": {"mixing_weights": [4, 9, 10, 3], "pop_prop": 0.39, "cint_cap": 15, "sample_size": 500},
    ...     "high": {"mixing_weights": [4, 7, 5, 1], "pop_prop": 0.01, "cint_cap": 10, "sample_size": 100},
    ... }
    >>> df_sample, age_dist_props, df_eval = simulate_ses(patterns, age_dist.P.values, config=config)
    """
    config = config or {
        "low": {"mixing_weights": [4, 9, 15, 6], "pop_prop": 0.6, "cint_cap": 20, "sample_size": 1000},
        "mid": {"mixing_weights": [4, 9, 10, 3], "pop_prop": 0.39, "cint_cap": 15, "sample_size": 1000},
        "high": {"mixing_weights": [4, 7, 5, 1], "pop_prop": 0.01, "cint_cap": 10, "sample_size": 1000},
    }

    # Create SES patterns
    def compute_ses_patterns(ses, config):
        P_ses = np.round(config["pop_prop"] * age_dist)
        rate, cint = make_contact_pattern(patterns, P_ses, config["mixing_weights"], config["cint_cap"])
        return P_ses, rate, cint

    ses_patterns = {
        ses: compute_ses_patterns(ses, config)
        for ses, config in config.items()
    }

    # Generate DataFrames
    def make_sample_eval_dfs(ses, P_ses, rate, cint):
        
        df_sample = sample_contacts(config[ses]['sample_size'], cint, P_ses, dist=dist)
        df_sample['ses'] = ses
        
        aidx = np.array(np.meshgrid(range(len(P_ses)), range(len(P_ses)))).T.reshape(-1, 2)
        df_eval = pd.DataFrame({
            "age_part": aidx[:, 0], "age_cnt": aidx[:, 1],
            "rate": rate[aidx[:, 0], aidx[:, 1]],
            "cint": cint[aidx[:, 0], aidx[:, 1]], "ses": ses
        })
        return df_sample, df_eval
      
    def make_age_dist_props(age_dist):
      return {"ses": np.vstack([(config[ses]["pop_prop"] * age_dist) / age_dist for ses in config.keys()]).T}

    dfs_train, dfs_eval = zip(*[
        make_sample_eval_dfs(ses, *ses_patterns[ses])
        for ses in ses_patterns
    ])
    
    age_dist_props = make_age_dist_props(age_dist)

    return pd.concat(dfs_train), age_dist_props, pd.concat(dfs_eval)