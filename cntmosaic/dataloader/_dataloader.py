import itertools
from typing import Optional
from abc import ABC
import warnings
from dataclasses import dataclass

import pandas as pd
import numpy as np
import pandas as pd
import xarray as xr

import jax.numpy as jnp

from ._utils import make_idarrs_for_intervals

@dataclass
class CoordToColumns:
	"""
	This class is used to specify the mapping of columns in input dataframes to the
	corresponding input variables for the models. It is passed to dataloader classes.
	
	Attributes
	---------
	age_part: str
		Column name for the participant age.
	age_cnt: Optional[str]
		Column name for the contact age.
	age_grp_cnt: Optional[str]
		Column name for the contact age group.
	id_var: str
		Column name for the unique identifier of participants.
	y: Optional[str]
		Column name for the outcome variable. By default, it is assumed to be 'y'.
	grp_vars_part: Optional[list[str] | str]
		Column names for the stratification variables in the participant data.
  	It can be a list of strings or a single string.
  grp_vars_cnt: Optional[list[str] | str]
		Column names for the stratification variables in the contact data.
  	It can be a list of strings or a single string.
	age_pop: Optional[str]
		Column name for the population age. This is used for population size calculations.
	size_pop: Optional[str]
		Column name for the population size. This is used for population size calculations.
	"""
    
	age_part: str
	age_cnt: Optional[str] = None
	age_grp_cnt: Optional[str] = None
	id_var: str = 'id'
	y: Optional[str] = 'y'
	grp_vars_part: Optional[list[str] | str] = None
	grp_vars_cnt: Optional[list[str] | str] = None
	age_pop: Optional[str] = None
	size_pop: Optional[str] = None

	def age_vars(self):
		if self.age_cnt:
			return [self.age_part, self.age_cnt]
		elif self.age_grp_cnt:
			return [self.age_part, self.age_grp_cnt]
		else:
			raise ValueError('One of age_cnt or age_grp_cnt must be provided')

	def __post_init__(self):
		if isinstance(self.grp_vars_part, str):
			object.__setattr__(self, 'grp_vars_part', [self.grp_vars_part])
		elif self.grp_vars_part is None:
			object.__setattr__(self, 'grp_vars_part', [])

		if isinstance(self.grp_vars_cnt, str):
			object.__setattr__(self, 'grp_vars_cnt', [self.grp_vars_cnt])
		elif self.grp_vars_cnt is None:
			object.__setattr__(self, 'grp_vars_cnt', [])
  
		if (self.age_pop is None) != (self.size_pop is None):
			raise ValueError("Both 'age_pop' and 'size_pop' must be set together or both left as None.")


class BaseLoader(ABC):
	def __init__(self,
               data: pd.DataFrame,
               pop: pd.DataFrame,
               col_map: CoordToColumns):
		"""
		Base class initializer.

		Parameters
		----------
			data pd.DataFrame:
				Input dataframe containing both particiapnt and contact information.
			col_map CoordToColumns:
				Mapping of dataframe columns to model variables.
			pop pd.DataFrame:
				pandas dataframe containing population size by age.
		"""
		self.data = self._validate(data, pop, col_map)
		self.col_map = col_map
		self.pop = pop
		self.ds = None
		
	def _validate(self,
               	data: pd.DataFrame,
                pop: pd.DataFrame,
                col_map: CoordToColumns):
		"""
		This method validates the input data and column mappings.
		"""
		# [Check] Ensure all necessary columns are present
		cols_needed = [col_map.y, col_map.id_var] + col_map.age_vars()
		if col_map.grp_vars_part:
			cols_needed.extend(col_map.grp_vars_part)
		if col_map.grp_vars_cnt:
			cols_needed.extend(col_map.grp_vars_cnt)
		missing = [col for col in cols_needed if col not in data.columns]
		if missing: raise KeyError(f"Missing columns in data: {missing}")
  
		# [Do] subset the data to only the columns needed
		data = data[cols_needed].dropna().copy()
		
		# [Check] Ensure all object columns are categorical. If not, convert them to categorical.
		for col in data.select_dtypes(include='object').columns:
			if not isinstance(data[col].dtype, pd.CategoricalDtype):
				warnings.warn(f"Column {col} is not categorical. Converting to categorical.", UserWarning)
				data[col] = data[col].astype('category')
				
		# [Check] If age_grp_cnt is specified, ensure that it is a pd.IntervalDtype
		if col_map.age_grp_cnt:
			is_cat = isinstance(data[col_map.age_grp_cnt].dtype, pd.CategoricalDtype)
			are_intervals = isinstance(data[col_map.age_grp_cnt].cat.categories, pd.IntervalIndex)
			if not is_cat or not are_intervals:
				raise TypeError(f"Column {col_map.age_grp_cnt} must be of type pd.IntervalDtype.")
			
		# [Check] Checks the minimum and maximum values of age_part, age_cnt (or age_grp_cnt), and age_pop
		# and determines the if they are consistent. If not automatically make sure they are consistent.
		part_min_age = data[col_map.age_part].min()
		part_max_age = data[col_map.age_part].max()
  
		if col_map.age_cnt:
			cnt_min_age = data[col_map.age_cnt].min()
			cnt_max_age = data[col_map.age_cnt].max()
		else: # If age_grp_cnt is specified
			cnt_min_age = data[col_map.age_grp_cnt].min().left
			cnt_max_age = data[col_map.age_grp_cnt].max().right - 1
		
		pop_min_age = pop[col_map.age_pop].min()
		pop_max_age = pop[col_map.age_pop].max()
		
		# Minimum and maximum ages of the sample (participants and contacts)
		sample_min_age = min(part_min_age, cnt_min_age)
		sample_max_age = max(part_max_age, cnt_max_age)
  
		if sample_min_age != pop_min_age:
			warnings.warn(f"Minimum age in sample ({sample_min_age}) does not match that in the population information ({pop_min_age}). Adjusting sample minimum age to match population.", UserWarning)
			data = data[data[col_map.age_part] >= pop_min_age]
			min_age = pop_min_age
		else:
			min_age = sample_min_age
   
		if sample_max_age != pop_max_age:
			warnings.warn(f"Maximum age in sample ({sample_max_age}) does not match that in the population information ({pop_max_age}). Adjusting sample maximum age to match population.", UserWarning)
			max_age = pop_max_age
		else:
			max_age = sample_max_age
		
		# [Do] Set minimum and maximum age
		# [Do] Select columns and drop NaN values
		self.min_age = min_age
		self.max_age = max_age
  
		return data
		
	def load(self) -> xr.Dataset:
		"""
		Loads the data into an xarray dataset.
		"""
		# [Do] Calculate the number of participants stratified by age and oter grouping variables
		grp_vars_n = [self.col_map.age_part]
		if self.col_map.grp_vars_part:
			grp_vars_n += self.col_map.grp_vars_part
		df_n = self.data.groupby(grp_vars_n, observed=False).size().reset_index(name='N')
  
		# [Do] Calculate the number of contacts stratified by age and other grouping variables
		grp_vars = self.col_map.age_vars()
		if self.col_map.grp_vars_part:
			grp_vars += self.col_map.grp_vars_part
		if self.col_map.grp_vars_cnt:
			grp_vars += self.col_map.grp_vars_cnt
   
		df_y = self.data.groupby(grp_vars, observed=False).agg({self.col_map.y: 'sum'}).reset_index()
		
		# [Do] Create a full grid of all combinations of the grouping variables via a cartesian product
		unique_coords = {var: self.data[var].unique() for var in grp_vars}
		unique_coords[self.col_map.age_part] = np.arange(self.min_age, self.max_age + 1, dtype=int)
  
		if self.col_map.age_cnt:
			unique_coords[self.col_map.age_cnt] = np.arange(self.min_age, self.max_age + 1, dtype=int)
		elif self.col_map.age_grp_cnt:
			unique_coords[self.col_map.age_grp_cnt] = self.data[self.col_map.age_grp_cnt].cat.categories
   
		index = pd.MultiIndex.from_product(unique_coords.values(), names=unique_coords.keys())
		df_full = pd.DataFrame(list(index), columns=unique_coords.keys())
		if self.col_map.age_grp_cnt:
			# [Do] Restore the original information of the age group column
			df_full[self.col_map.age_grp_cnt] = pd.Categorical(
				df_full[self.col_map.age_grp_cnt],
				categories=self.data[self.col_map.age_grp_cnt].cat.categories,
				ordered=True
			)
  
		# [Do] Merge the full grid with the contact and participant data
		df_full = pd.merge(df_full, df_y, on=grp_vars, how='left')
		df_full = pd.merge(df_full, df_n, on=grp_vars_n, how='left')
  
		# [Do] Finalise the data
		df_full = df_full.dropna(subset=['N'])
		df_full['y'] = df_full['y'].fillna(0)
  
		# [Do] Create a xarray dataset
		self.raw_df = df_full
		self.ds = xr.Dataset(
			{
				'y': ('index', df_full['y'].astype(int).to_numpy()),
				'log_N': ('index', jnp.log(df_full['N'].to_numpy())),
				'log_P': ('age', jnp.log(self.pop[self.col_map.size_pop].to_numpy())),
				'aid': ('index', df_full[self.col_map.age_part].to_numpy()),
			},
			coords={
				'index': df_full.index.to_numpy(),
				'age': ('age', np.arange(self.min_age, self.max_age + 1, dtype=int))
			}
		)
  
		if self.col_map.age_cnt:
			self.ds['bid'] = ('index', df_full[self.col_map.age_cnt].to_numpy())
		elif self.col_map.age_grp_cnt:
			self.ds['cid'] = ('index', df_full[self.col_map.age_grp_cnt].cat.codes.to_numpy())
   
			# [Do] Create indices for age aggregation	
			aid_exp, bid_pad = make_idarrs_for_intervals(df_full, self.col_map.age_grp_cnt, self.ds['aid'].to_numpy())
			self.ds['aid_exp'] = (['index', 'max_int_length'], aid_exp)
			self.ds['bid_pad'] = (['index', 'max_int_length'], bid_pad)
  
		return self.ds 

class DataLoader(BaseLoader):
	"""Prepares the data for models
 
	This class takes participant data, contact data, population data, and a column mapping object
	and prepares the data for models. It conducts a preliminary check of the data, ensuring that all
	necessary columns are present and that the data is in the correct format. It also handles and
	merges the dataframes, ensuring that the data is in a format suitable for analysis.
 
	Params
	------
		part: pd.DataFrame
			Dataframe containing participant data.
		cnt: pd.DataFrame
			Dataframe containing contact data.
		pop: pd.DataFrame
			Dataframe containing population data.
		col_map: CoordToColumns
			Object that maps the columns in the dataframes to the variables used in the models.
	"""
	def __init__(self,
               part: pd.DataFrame,
               cnt: pd.DataFrame,
               pop: pd.DataFrame,
               col_map: CoordToColumns):
   
		self._validate_part(part, col_map)
		self._validate_cnt(cnt, col_map)
		self._validate_pop(pop, col_map)
		data = pd.merge(self.cnt, self.part,
                  	on=col_map.id_var,
                   	suffixes=('_part', '_cnt'))
		super().__init__(data, pop, col_map)
  
	def _validate_part(self, part: pd.DataFrame, col_map: CoordToColumns):
		"""
		This method validates the participant data and column mappings.
		Checks for the presence of necessary columns.
  	"""
		# [Check] Ensure all necessary columns are present
		if col_map.id_var not in part.columns:
			raise KeyError(f"Missing column {col_map.id_var} in participants dataframe")
 
		if col_map.age_part not in part.columns:
			raise KeyError(f"Missing column {col_map.age_part} in participants dataframe")
 
		if col_map.grp_vars_part:
			missing = [col for col in col_map.grp_vars_part if col not in part.columns]
			if len(missing) > 0:
				raise KeyError(f"Missing columns {missing} in participants dataframe")
 
		self.part = part.copy()
	
	def _validate_cnt(self, cnt: pd.DataFrame, col_map: CoordToColumns):
		"""
  	This method validates the contact data and column mappings.
   	Checks for the presence of necessary columns.
  	"""
		# [Check] Ensure all necessary columns are present
		if col_map.id_var not in cnt.columns:
			raise KeyError(f"Missing column {col_map.id_var} in contacts dataframe")
 
		if col_map.age_cnt:
			if col_map.age_cnt not in cnt.columns:
				raise KeyError(f"Missing column {col_map.age_cnt} in contacts dataframe")
  
		if col_map.age_grp_cnt:
			if col_map.age_grp_cnt not in cnt.columns:
				raise KeyError(f"Missing column {col_map.age_grp_cnt} in contacts dataframe")
  
		if col_map.grp_vars_cnt:
			missing = [col for col in col_map.grp_vars_cnt if col not in cnt.columns]
			if len(missing) > 0:
				raise KeyError(f"Missing columns {missing} in contacts dataframe")
  
		# [Check] If the column y is present is in cnt. If not, add it as a column with value 1
		if col_map.y not in cnt.columns:
			cnt[col_map.y] = 1
   
		self.cnt = cnt.copy()
  
	def _validate_pop(self, pop: pd.DataFrame, col_map: CoordToColumns):
		"""
		This method validates the input data and column mappings.
		"""
		if col_map.age_pop not in pop.columns:
			raise KeyError(f"Missing column {col_map.age_pop} in population dataframe")
		if col_map.size_pop not in pop.columns:
			raise KeyError(f"Missing column {col_map.size_pop} in population dataframe")