from importlib import resources
import pickle
from typing import TypedDict

import numpy as np
import pandas as pd

from ..sim._utils import symmetrise_patterns, smooth_patterns


class SurveyData(TypedDict):
    """Return type for survey dataset loaders (load_polymod_germany, load_covimod)."""
    contacts: pd.DataFrame
    participants: pd.DataFrame
    population: pd.DataFrame


class ContactPatterns(TypedDict):
    """Return type for load_template_patterns()."""
    household: np.ndarray
    school: np.ndarray
    work: np.ndarray
    community: np.ndarray


def load_pickle_data(data_file_name):
	"""Loads `data_file_name` from the package's data directory."""
	
	data_path = resources.files('cntmosaic.datasets.data') / data_file_name
	with data_path.open('rb') as pickle_file:
		return pickle.load(pickle_file)
	
def load_csv_data(data_file_name, header=0):
	"""Loads `data_file_name` from the package's data directory."""
	
	data_path = resources.files('cntmosaic.datasets.data') / data_file_name
	return pd.read_csv(data_path, header=header)
	
def load_polymod_germany() -> SurveyData:
	"""Loads the German Polymod dataset.

	This function loads a cleaned version of the German POLYMOD dataset.

	Returns
	-------
	SurveyData
			A typed dict with keys 'contacts', 'participants', 'population'
			(each a pandas DataFrame).
	"""

	return load_pickle_data('polymod_germany.pkl')

def load_covimod() -> SurveyData:
	"""Loads the COVIMOD dataset.

	This function loads the Covimod dataset.

	Returns
	-------
	SurveyData
			A typed dict with keys 'contacts', 'participants', 'population'
			(each a pandas DataFrame).
	"""

	return load_pickle_data('covimod.pkl')

def load_template_patterns(country: str,
													 symmetrise: bool=False,
													 smooth: bool=False,
													 normalise: bool=True,
													 max_age: int=80) -> ContactPatterns:
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
	symmetrise: bool, default=False
		Symmetrise the contact patterns
	smooth: bool, default=False
		Smooth the contact patterns
	
	Returns
	-------
	dict
		A dictionary containing contact patterns for household, school, work, and community settings
	"""
	assert max_age > 0 and max_age <= 85, f'Invalid max_age: {max_age}'
	
	prefix = f'{country}_country_level_F'
		
	patterns = {
		'household': load_csv_data(f'contact_matrices/{prefix}_household_setting_85.csv', header=None).values,
		'school': load_csv_data(f'contact_matrices/{prefix}_school_setting_85.csv', header=None).values,
		'work': load_csv_data(f'contact_matrices/{prefix}_work_setting_85.csv', header=None).values,
		'community': load_csv_data(f'contact_matrices/{prefix}_community_setting_85.csv', header=None).values
	}
 
	if symmetrise:
		patterns = symmetrise_patterns(patterns)
		
	if smooth:
		patterns = smooth_patterns(patterns)
	
	if normalise:
		for key, value in patterns.items():
			patterns[key] = value / (value.sum(axis=1).mean())
	
	for key, value in patterns.items():
		patterns[key] = value[:max_age+1, :max_age+1]
	
	return patterns

def load_age_distribution(country: str, max_age: int=80) -> pd.DataFrame:
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
	assert max_age > 0 and max_age <= 85, f'Invalid max_age: {max_age}'
	
	sub_dir = 'population_rescaled_age_distributions'
	
	file_name = f'{sub_dir}/{country}_country_level_age_distribution_85.csv'
		
	age_dist = load_csv_data(file_name, header=None)
	age_dist.columns = ['age', 'P']
	age_dist = age_dist[age_dist['age'] <= max_age]
	
	return age_dist
