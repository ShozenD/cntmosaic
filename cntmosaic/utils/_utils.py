import numpy as np
from ._AgeBins import AgeBins

def pixilate(matrix: np.ndarray, age_bins: AgeBins):
    """
    Aggregate a contact matrix over specified age intervals.
            - A 3D array of shape (n_samples, dim1, dim2): Aggregates spatially over the last two axes.
            - A 2D array of shape (len(intervals), len(intervals)): Assumed to be pre-aggregated and is returned unchanged.
    For a 3D input:
            1. It computes block averages over the first spatial dimension (axis=1) by summing using np.add.reduceat
                     and dividing by the corresponding bin sizes provided by age_bins.bin_sizes.
            2. It then aggregates over the second spatial dimension (axis=2) using np.add.reduceat.

    Parameters
    ----------
    matrix : numpy.ndarray
            Input array with either shape:
                    - (n_samples, dim1, dim2) for raw data to be aggregated, or
                    - (len(intervals), len(intervals)) for pre-aggregated data.
    age_bins : AgeBins
            An object that defines the aggregation scheme. It must contain:
                    - left : array-like
                            Starting indices for each bin.
                    - bin_sizes : array-like
                            The sizes of each bin corresponding to the intervals.

    Returns
    -------
    numpy.ndarray
            Aggregated array:
                    - For a 3D input, the output shape will be (n_samples, len(age_bins.left), len(age_bins.left)).
                    - For a 2D input, the pre-aggregated matrix is returned unchanged.
    """
    single_sample = False
    if matrix.ndim == 2:
        matrix = matrix[np.newaxis, ...]
        single_sample = True

    # Reduce along axis=1 (first spatial dimension):
    bin_sizes = age_bins.bin_sizes
    mean_dim1 = (
        np.add.reduceat(matrix, age_bins.left, axis=1)
        / bin_sizes[np.newaxis, :, np.newaxis]
    )
    # Then reduce along axis=2 (second spatial dimension):
    result = np.add.reduceat(mean_dim1, age_bins.left, axis=2)

    return result[0] if single_sample else result


def depixilate(matrix: np.ndarray, age_bins: AgeBins):
    """Depixilate a matrix using age bin slicing.
    
    This function transforms the input matrix by extracting sub-matrices based on the intervals
    specified in the age_bins object. For each index pair (i, j), it slices the original matrix
    using the corresponding indices from age_bins.left and age_bins.right, and assigns the result
    to the output depixilated matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
            A 2D NumPy array which is the input for depixilation.
    age_bins : AgeBins
            An object containing binning information. It must have the attributes:
                    - left : sequence of int
                            The starting indices for the bins.
                    - right : sequence of int
                            The ending indices for the bins.
                    - range : int
                            The number of bins, which is used as the dimensions of the output matrix.
    
    Returns
    -------
    np.ndarray
            A 2D NumPy array of shape (age_bins.range, age_bins.range) where each element
            is derived from the corresponding sub-matrix slice of the input matrix defined by age_bins.
    
    Examples
    --------
    >>> # Assuming matrix is a properly sized 2D numpy array and age_bins is correctly defined
    >>> result = depixilate(matrix, age_bins)
    >>> print(result.shape)
    (age_bins.range, age_bins.range)
    """
    dpx_matrix = np.zeros((age_bins.range, age_bins.range))

    for i in range(len(age_bins.left)):
        x_left = age_bins.left[i]
        x_right = age_bins.right[i] + 1
        for j in range(len(age_bins.left)):
            y_left = age_bins.left[j]
            y_right = age_bins.right[j] + 1

            dpx_matrix[x_left:x_right, y_left:y_right] = matrix[i,j] / age_bins.bin_sizes[j]

    return dpx_matrix
