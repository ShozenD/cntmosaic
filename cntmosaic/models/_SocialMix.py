import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm


# Helper functions
def assign_age_group(df, column, age_limits, new_col):
		"""
		Adds an age group column to the dataframe using the provided age limits.
		"""
		df[new_col] = pd.cut(df[column], bins=age_limits, right=False, include_lowest=True)
		return df


def compute_contact_counts(df_part, df_cnt):
		"""
		Merge participants and contacts dataframes and return a contact count matrix.
		"""
		merged = pd.merge(df_cnt, df_part, on="id", how="left")
		Y = merged.pivot_table(
				values="y",
				index="age_grp_part",
				columns="age_grp_cnt",
				observed=False,
				aggfunc="sum",
		).to_numpy()
		return jnp.asarray(Y)


def compute_sample_sizes(df_part, group_col):
		"""
		Returns the number of participants per age group as a 1D numpy array.
		"""
		sample_sizes = (
				df_part.groupby(group_col, observed=False).size().reset_index(name="N")
		)
		return jnp.asarray(sample_sizes["N"].to_numpy())


def compute_population_sizes(df_age_dist, age_limits):
		"""
		Assign age groups to the population data and aggregate the counts by group.
		"""
		if "age_grp" not in df_age_dist.columns:
				df_age_dist["age_grp"] = pd.cut(
						df_age_dist["age"], bins=age_limits, right=False, include_lowest=True
				)
		pop_df = (
				df_age_dist.groupby("age_grp", observed=False).agg({"P": "sum"}).reset_index()
		)
		return jnp.asarray(pop_df["P"].to_numpy())


def merge_zero_groups(intervals, counts):
		"""
		Merge zero-count intervals with their nearest nonzero-count neighbor.

		Parameters
		----------
		intervals : list of pd.Interval
						Sorted list of pandas Interval objects.
		counts : array_like
						Array of counts corresponding to each interval.

		Returns
		-------
		merged_intervals : list of pd.Interval
						List of intervals after merging those with zero counts.
		index_map : np.ndarray
						Array where each element is the index of the merged interval corresponding
						to the original interval.

		Examples
		--------
		>>> intervals = [pd.Interval(0, 5, closed='left'),
		...              pd.Interval(5, 10, closed='left'),
		...              pd.Interval(10, 15, closed='left'),
		...              pd.Interval(15, 20, closed='left')]
		>>> counts = [10, 0, 15, 0]
		>>> merged_intervals, index_map = merge_zero_groups(intervals, counts)
		>>> merged_intervals
		[pd.Interval(0, 10, closed='left'), pd.Interval(10, 20, closed='left')]
		>>> index_map
		array([0, 0, 1, 1])
		"""
		merged_intervals = []  # new list for merged intervals
		n = len(intervals)

		i = 0
		while i < n:
				# If current group has nonzero count, start a new merged interval.
				if counts[i] != 0:
						current = intervals[i]
						merged_intervals.append(current)
						i += 1
				else:
						# For a zero-count group, check if there is a previously merged interval.
						if merged_intervals:
								# Merge current interval into the last merged interval.
								prev = merged_intervals[-1]
								# Create a new merged interval with the left boundary of prev and the right of current.
								merged = pd.Interval(prev.left, intervals[i].right, closed=prev.closed)
								merged_intervals[-1] = merged
								i += 1
						else:
								# No previous merged interval; look ahead for the first nonzero group.
								j = i + 1
								while j < n and counts[j] == 0:
										j += 1
								if j < n:
										# Create a merged interval from current[i] left to intervals[j].right
										merged = pd.Interval(
												intervals[i].left,
												intervals[j].right,
												closed=intervals[j].closed,
										)
										merged_intervals.append(merged)
										i = j + 1
								else:
										# All remaining groups are zero. Create a merged interval from current i left to last right.
										merged = intervals[i]
										merged_intervals.append(merged)
										i += 1

		return merged_intervals


class SocialMix:
		"""
		This class implements the socialmixr algorithm in Python.
		"""

		def __init__(
				self,
				df_part: pd.DataFrame,
				df_cnt: pd.DataFrame,
				df_age_dist: pd.DataFrame,
				age_limits: list | np.ndarray,
				symmetric: bool = False,
		):
				self.df_part = df_part
				self.df_cnt = df_cnt
				self.df_age_dist = df_age_dist
				self.symmetric = symmetric
				self.age_limits = np.asarray(age_limits)
				self.check_dataframes()
				self.check_age_limits()
				self.preprocess()
				
		def check_age_limits(self):
				"""
				Check if the age limits are valid.
				"""
				if not isinstance(self.age_limits, (list, np.ndarray)):
						raise ValueError("Age limits must be a list or numpy array.")
				if len(self.age_limits) < 2:
						raise ValueError("At least two age limits are required.")
				if not all(np.issubdtype(type(i), np.number) for i in self.age_limits):
						raise ValueError("All age limits must be numeric.")
				if not all(self.age_limits[i] < self.age_limits[i + 1] for i in range(len(self.age_limits) - 1)):
						raise ValueError("Age limits must be in ascending order.")
    
				if 'age_grp_part' not in self.df_part.columns:
						self.df_part = assign_age_group(
								self.df_part, "age_part", self.age_limits, "age_grp_part"
        		)
      
				ssizes = self.df_part.groupby("age_grp_part", observed=False).size().reset_index(name="N")
				if np.min(ssizes['N']) == 0:
					print("Warning: Some age groups have zero sample sizes. Merging age groups.")

					# Merge zero-count intervals with their nearest nonzero-count neighbor
					merged_intervals = merge_zero_groups(
							ssizes["age_grp_part"].cat.categories, ssizes["N"].to_numpy()
					)
     
					# Define new age_lmits based on the merged intervals
					new_age_limits = np.array([interval.left for interval in merged_intervals] + [merged_intervals[-1].right])
					self.age_limits = new_age_limits

		def check_dataframes(self):
				"""
				Check if the dataframes have the required columns.
				"""
				# Check "id" columns in df_part and df_cnt.
				if "id" not in self.df_part.columns:
						raise ValueError("Missing column 'id' in participants dataframe.")
				if "id" not in self.df_cnt.columns:
						raise ValueError("Missing column 'id' in contacts dataframe.")

				# For df_part: must have either 'age_part' or 'age_grp_part'.
				has_age_part = "age_part" in self.df_part.columns
				has_age_grp_part = "age_grp_part" in self.df_part.columns
				if not (has_age_part or has_age_grp_part):
						raise ValueError("Participants dataframe must have either 'age_part' or 'age_grp_part' column.")

				# For df_cnt: must have either 'age_cnt' or 'age_grp_cnt' plus column 'y'.
				has_age_cnt = "age_cnt" in self.df_cnt.columns
				has_age_grp_cnt = "age_grp_cnt" in self.df_cnt.columns
				if not (has_age_cnt or has_age_grp_cnt):
						raise ValueError("Contacts dataframe must have either 'age_cnt' or 'age_grp_cnt' column.")
				if "y" not in self.df_cnt.columns:
						raise ValueError("Missing column 'y' in contacts dataframe.")

				# For df_age_dist: must have columns 'age' and 'P'.
				if "age" not in self.df_age_dist.columns:
						raise ValueError("Missing column 'age' in age distribution dataframe.")
				if "P" not in self.df_age_dist.columns:
						raise ValueError("Missing column 'P' in age distribution dataframe.")

				# Additional rule: if df_part has 'age_grp_part', then df_cnt must have 'age_grp_cnt'.
				if has_age_grp_part and not has_age_grp_cnt:
						raise ValueError("Participants dataframe has 'age_grp_part' but contacts dataframe lacks 'age_grp_cnt'.")

		def preprocess(self):
				self.df_part = self.df_part.reset_index(drop=True)
				self.df_cnt = self.df_cnt.reset_index(drop=True)

				if "age_grp_part" not in self.df_part.columns:
						self.df_part = assign_age_group(
								self.df_part, "age_part", self.age_limits, "age_grp_part"
						)

				if "age_grp_cnt" not in self.df_cnt.columns:
						self.df_cnt = assign_age_group(
								self.df_cnt, "age_cnt", self.age_limits, "age_grp_cnt"
						)

				self.Y = compute_contact_counts(self.df_part, self.df_cnt)
				self.N = compute_sample_sizes(self.df_part, "age_grp_part")
				self.P = compute_population_sizes(self.df_age_dist, self.age_limits)

		def compute_cint(self):
				"""
				Get the contact intensity matrix.
				"""
				if not hasattr(self, "cint"):
						N_inv = 1 / self.N
						P_inv = 1 / self.P

						if self.symmetric:
								M = N_inv[:, jnp.newaxis] * self.Y
								self.cint = (
										M + P_inv[:, jnp.newaxis] * M.T * self.P[jnp.newaxis, :]
								) / 2
						else:
								self.cint = N_inv[:, jnp.newaxis] * self.Y

				return self.cint

		def compute_rate(self):
				"""
				Get the contact rate matrix.
				"""
				if not hasattr(self, "rate"):
						self.rate = self.compute_cint() / self.P[jnp.newaxis, :]

				return self.rate

		def run_bootstrap(self, n_boot: int = 1000):
				"""
				Bootstrap the contact intensity matrix.
				"""
				if not hasattr(self, "boots_cint_matrix"):
						boots_cint = []
						boots_rate = []
						for i in tqdm(range(n_boot), desc="Bootstrapping"):
								# Sample with replacement
								df_part_sampled = self.df_part.sample(frac=1, replace=True)

								# Create a new instance of the class
								sm = SocialMix(
										df_part_sampled,
										self.df_cnt,
										self.df_age_dist,
										self.age_limits,
										self.symmetric,
								)

								boots_cint.append(sm.compute_cint())
								boots_rate.append(sm.compute_rate())

				self.boots_cint = np.asarray(boots_cint)
				self.boots_rate = np.asarray(boots_rate)
