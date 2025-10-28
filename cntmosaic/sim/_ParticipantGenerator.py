import numpy as np
import pandas as pd

from numpy.typing import NDArray
from typing import List, Union
from dataclasses import dataclass


@dataclass
class Subgroup:
	"""
	Configuration for a population subgroup.
 
	Parameters
	----------
	n : int
		The number of participants in this subgroup.
	age_dist : NDArray
		Age distribution array for this subgroup.
		Each element represents the count or proportion of individuals in that age group.
		Will be normalized to proportions internally.
	mean_cint_margin : float
		Average marginal contact intensity for this subgroup.
	label : str, optional
		Label for the subgroup (used when multiple subgroups are provided).
	"""
	n: int
	age_dist: NDArray
	mean_cint_margin: float
	label: str = None


class ParticipantGenerator:
	"""
	Generate participant data based on age distributions from Subgroup configurations.

	This class creates synthetic participant datasets by sampling from Subgroup objects
	that contain age distributions and other demographic parameters. It supports single
	populations as well as multiple subgroups, making it useful for generating stratified
	participant samples for contact studies or demographic simulations.

	Examples
	--------
	>>> import numpy as np
	>>> from cntmosaic.sim import ParticipantGenerator, Subgroup

	**Example 1: Single population**

	Generate participants from a single Subgroup:

	>>> subgroup = Subgroup(
	...     n=1000,
	...     age_dist=np.array([100, 200, 300, 400, 500]),
	...     mean_cint_margin=15.0
	... )
	>>> pg = ParticipantGenerator(subgroup)
	>>> df_participants = pg.generate(seed=42)
	>>> print(df_participants.head())
	   id  age_group
	0   1          3
	1   2          4
	2   3          2
	3   4          3
	4   5          2
	>>> len(df_participants)
	1000

	**Example 2: Multiple subgroups with labels**

	Generate participants from multiple subgroups using the label field:

	>>> subgroups = [
	...     Subgroup(n=500, age_dist=np.array([150, 250, 350, 250, 100]), mean_cint_margin=18.0, label='urban'),
	...     Subgroup(n=300, age_dist=np.array([100, 150, 200, 300, 250]), mean_cint_margin=12.0, label='rural')
	... ]
	>>> pg = ParticipantGenerator(subgroups)
	>>> df_participants = pg.generate(seed=42)
	>>> print(df_participants.head())
	   id  age_group  subgroup
	0   1          2     urban
	1   2          3     urban
	2   3          1     urban
	3   4          2     urban
	4   5          2     urban
	>>> print(df_participants['subgroup'].value_counts())
	subgroup
	urban    500
	rural    300

	**Example 3: Multiple subgroups with automatic numeric labels**

	If labels are not provided, subgroups are automatically labeled with indices:

	>>> subgroups = [
	...     Subgroup(n=500, age_dist=np.array([150, 250, 350, 250, 100]), mean_cint_margin=18.0),
	...     Subgroup(n=500, age_dist=np.array([100, 150, 200, 300, 250]), mean_cint_margin=12.0)
	... ]
	>>> pg = ParticipantGenerator(subgroups)
	>>> df_participants = pg.generate(seed=42)
	>>> print(df_participants['subgroup'].value_counts())
	subgroup
	0    500
	1    500
	"""

	def __init__(
		self,
		subgroups: Union[Subgroup, List[Subgroup]]
	):
		"""
		Initialize ParticipantGenerator with Subgroup configurations.

		Parameters
		----------
		subgroups : Subgroup or list of Subgroup
			Subgroup configuration(s) to sample from. Can be:
			- Subgroup: A single subgroup for a homogeneous population
			- list of Subgroup: Multiple subgroups with labels from Subgroup.label field
			  (if label is None, defaults to numeric indices 0, 1, ...)

			Each Subgroup specifies the sample size (n), age distribution, and other parameters.
			The age distributions in each Subgroup will be automatically normalized to proportions.
		"""
		self._parse_subgroups(subgroups)
		self._normalize_distributions()

	def _parse_subgroups(
		self,
		subgroups: Union[Subgroup, List[Subgroup]]
	) -> None:
		"""Parse input subgroups into standardized format."""
		# Case 1: Single Subgroup
		if isinstance(subgroups, Subgroup):
			self.n = subgroups.n
			self.age_dists = subgroups.age_dist
			self.has_subgroups = False
			return

		# Case 2: List of Subgroup
		if isinstance(subgroups, list):
			self.ns = {}
			self.age_dists = {}
			self.subgroup_labels = {}
			for i, item in enumerate(subgroups):
				if isinstance(item, Subgroup):
					# Use label from Subgroup if provided, otherwise use index
					label = item.label if item.label is not None else i
					self.ns[label] = item.n
					self.age_dists[label] = item.age_dist
					self.subgroup_labels[i] = label
					
				else:
					raise TypeError(f"List elements must be Subgroup, got {type(item)}")
			self.has_subgroups = True
			return

		raise TypeError(
			f"subgroups must be Subgroup or list of Subgroup. Got {type(subgroups)}"
		)

	def _normalize_distributions(self) -> None:
		"""Normalize age distributions to proportions."""
		if not self.has_subgroups:
			# Single distribution
			self.age_proportions = self.age_dists / self.age_dists.sum()
		else:
			# Multiple distributions
			self.age_proportions = {
				key: dist / dist.sum()
				for key, dist in self.age_dists.items()
			}

	@staticmethod
	def _generate_single(n: int, age_prop: NDArray, rng: np.random.Generator) -> pd.DataFrame:
		"""
		Generate a DataFrame of participant ages based on age proportions.

		Parameters
		----------
		n : int
			The number of participants to generate.
		age_prop : NDArray
			The age proportions to sample from.
		rng : np.random.Generator
			Random number generator for reproducibility.

		Returns
		-------
		pd.DataFrame
			A DataFrame with 'age_group' column containing participant ages.
		"""
		age = rng.choice(np.arange(len(age_prop)), size=n, p=age_prop)
		return pd.DataFrame({'age_group': age})

	def generate(
		self,
		seed: int = None
	) -> pd.DataFrame:
		"""
		Generate participant data.

		The number of participants for each subgroup is determined by the `n` parameter
		in the Subgroup configuration.

		Parameters
		----------
		seed : int, optional
			Random seed for reproducibility.

		Returns
		-------
		pd.DataFrame
			A DataFrame containing the generated participant data.
			Columns:
			- 'id': Unique participant identifier (1, 2, 3, ...)
			- 'age_group': Age group index (0 to A-1)
			- 'subgroup': Subgroup label (only present if multiple subgroups)

		Examples
		--------
		>>> # Single population - generates n participants from Subgroup
		>>> subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=15.0)
		>>> pg = ParticipantGenerator(subgroup)
		>>> df = pg.generate(seed=42)
		>>> len(df)
		1000

		>>> # Multiple subgroups - each generates according to its own n
		>>> subgroups = [
		...     Subgroup(n=500, age_dist=age_dist1, mean_cint_margin=18.0, label='urban'),
		...     Subgroup(n=300, age_dist=age_dist2, mean_cint_margin=12.0, label='rural')
		... ]
		>>> pg = ParticipantGenerator(subgroups)
		>>> df = pg.generate(seed=42)
		>>> len(df)
		800
		>>> df['subgroup'].value_counts()
		subgroup
		urban    500
		rural    300
		"""
		rng = np.random.default_rng(seed)

		# Single population case
		if not self.has_subgroups:
			df = self._generate_single(self.n, self.age_proportions, rng)
			df['id'] = np.arange(1, self.n + 1)
			return df[['id', 'age_group']]

		# Multiple subgroups case
		dfs = []
		for label in self.subgroup_labels.values():
			age_prop = self.age_proportions[label]
			n = self.ns[label]
			df = self._generate_single(n, age_prop, rng)
			df['subgroup'] = label
			dfs.append(df)

		# Combine all subgroups
		df_combined = pd.concat(dfs, ignore_index=True)

		# Assign unique IDs
		df_combined['id'] = np.arange(1, len(df_combined) + 1)

		# Reorder columns: id, age_group, subgroup
		return df_combined[['id', 'age_group', 'subgroup']]