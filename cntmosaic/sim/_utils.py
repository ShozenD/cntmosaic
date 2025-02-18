from pathlib import Path
import itertools
import pandas as pd
import numpy as np
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

def expand_grid(data_dict) -> pd.DataFrame:
    """Create a dataframe from a dictionary of lists. Analogous to R's expand.grid."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def normalise_age_dists(age_dists: dict[NDArray]):
  assert isinstance(age_dists, dict), 'Age distributions must be a dictionary.'
  assert 'base' in age_dists, 'Base age distribution is required.'
  
  normalised_dists = {}
    
  try:
      base_dist = age_dists['base']
      normalised_dists['base'] = base_dist / base_dist.sum()
      
      for key, conditional_dist in age_dists.items():
          if key != 'base':
              normalised_dists[key] = {}
              dist = np.vstack([value for value in conditional_dist.values()])
              probs = dist / dist.sum(axis=0)
              for i, cat in enumerate(conditional_dist.keys()):
                  normalised_dists[key][cat] = probs[i,:]
  except:
      raise TypeError(f'Invalid distribution type: All values in age_dists must be np.ndarrays.')
    
  return normalised_dists