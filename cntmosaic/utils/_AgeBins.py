import numpy as np

class AgeBins:
	"""
	AgeBins
	-------
	A class for generating age bins either using a fixed step or user-defined cut points.
	Parameters
	----------
	min : int
		The minimum age value.
	max : int
		The maximum age value.
	step : int, optional
		The gap between consecutive bins when cuts are not provided (default is 5).
	cuts : list or numpy.ndarray, optional
		An array-like sequence of age cut points to define bin boundaries. If provided, these values 
		determine the internal boundaries of the bins.
  
	Attributes
	----------
	min : int
		The minimum age value.
	max : int
		The maximum age value.
	step : int
		The interval between bins used when cuts is None.
	cuts : list or numpy.ndarray
		The provided cut points for binning, or None if using a fixed step.
	left : list of int
		The left boundaries of each age bin. Computed based on the provided cuts or generated 
		using the step.
	right : list of int
		The right boundaries of each age bin. Computed by subtracting one from each cut (if cuts are used)
		or generated using the step, with the max as the final boundary.
	block_sizes : numpy.ndarray
		The sizes (i.e., widths) of each bin calculated as the difference between consecutive left 
		boundaries with max appended at the end.
  
	Methods
	-------
	get_bounds_left()
		Determines and returns the left boundaries of the age bins.
	get_bounds_right()
		Determines and returns the right boundaries of the age bins.
	get_bin_sizes()
		Computes and returns the sizes of each bin based on the left boundaries and max.
	"""
	def __init__(self, min: int, max: int, step: int=5, cuts: list | np.ndarray=None):
		self.min = min
		self.max = max
		self.range = max - min + 1
		self.step = step
		self.cuts = cuts
		self.get_bounds_left()
		self.get_bounds_right()
		self.get_bin_sizes()
		self.get_cell_sizes()
		
	def get_bounds_left(self):
		if not hasattr(self, 'left'):
			if self.cuts is not None:
				self.left = [self.min] + self.cuts
			else:
				self.left = list(range(self.min, self.max, self.step))
			
		return self.left

	def get_bounds_right(self):
		if not hasattr(self, 'right'):
			if self.cuts is not None:
				self.right = list(np.asarray(self.cuts) - 1) + [self.max + 1]
			else:
				self.right = list(np.asarray(range(self.min + self.step, self.max, self.step))-1) + [self.max + 1]
			
		return self.right
			
	def get_bin_sizes(self):
		if not hasattr(self, 'bin_sizes'):
			self.bin_sizes = np.diff(np.append(self.left, self.max + 1))
			
		return self.bin_sizes
 
	def get_cell_sizes(self):
		"""
		Compute and return the cell sizes as the outer product of bin sizes.

		This method calculates the cell sizes by taking the outer product of the
		bin_sizes attribute with itself. The result is cached in the instance attribute
		cell_sizes so that subsequent calls return the previously computed array.

		Returns
		-------
		numpy.ndarray
			A 2D array representing the cell sizes computed as the outer product of bin_sizes.
		"""
		if not hasattr(self, 'cell_sizes'):
			self.cell_sizes = np.outer(self.bin_sizes, self.bin_sizes)
		return self.cell_sizes
