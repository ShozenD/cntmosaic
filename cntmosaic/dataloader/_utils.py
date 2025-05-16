import numpy as np
import pandas as pd

def make_idarrs_for_intervals(
		data: pd.DataFrame,
		interval_col: str,
		aid: np.ndarray
):
		"""
		Prepares index arrays for 3D interval-based operations from DataFrame interval data.

		This function processes a specified interval column in a DataFrame to generate
		index arrays for the second and third dimensions of a 3D array, corresponding to 
		each interval's range. These index arrays are uniformly padded to allow for batch 
		operations. Additionally, it expands provided `xid` and `id_array` arrays to match 
		the length of the padded index arrays for aligned operations across all dimensions.

		Parameters
		----------
		data : pd.DataFrame
				DataFrame containing at least one interval column.
		interval_col : str
				Name of the column in 'data' that contains interval data.
		xid : np.ndarray
				Array of indices for the first dimension of a 3D array.
		aid : np.ndarray
				Array of IDs or indices that need to be expanded to match the interval operations.

		Returns
		-------
		np.ndarray, np.ndarray
				- A numpy array of the expanded ID array matching the dimensions of the index arrays.
				- A numpy array of padded index arrays corresponding to each interval.
				
		Example
		-------
		>>> df = pd.DataFrame({
		...     'age_grp_cnt': pd.IntervalIndex.from_arrays([0, 10, 20], [5, 15, 25], closed='left')
		... })
		>>> aid = np.array([1, 2, 3])
		>>> bid_pad, aid_exp = prepare_index_arrays_for_interval_operations(df, 'age_grp_cnt', aid)
		>>> print(bid_pad)
		>>> print(aid_exp)
		"""
		# Extract interval bounds
		bl = data[interval_col].apply(lambda x: x.left).to_numpy()
		bu = data[interval_col].apply(lambda x: x.right).to_numpy()

		# Calculate the maximum length of intervals
		max_length = max(bu - bl)

		# Create index arrays for each interval
		bid = [np.arange(start, stop) for start, stop in zip(bl, bu)]

		# Pad index arrays to uniform length
		bid_pad = np.array([np.pad(x, (0, max_length - len(x)), constant_values=-1) for x in bid])

		# Expand the ID array to match the padded index arrays
		aid_exp = np.repeat(aid[:, np.newaxis], max_length, axis=1)

		return aid_exp, bid_pad