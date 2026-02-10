import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def fine_coarse_matrix(x: pd.Series) -> NDArray:
    """
    Create an indicator matrix mapping one-year ages to specified age intervals.

    The function generfates a binary matrix where each row corresponds to all one-year ages considered,
    and each column represents an age interval defined in the input series. The matrix entries
    are set to 1 where the age falls into the corresponding interval, and 0 otherwise.

    Parameters
    ----------
    x : pd.Series
            A pandas Series with a categorical dtype that includes intervals (pd.IntervalIndex).
            The intervals are expected to define the coarser age grid with the left endpoint included
            and the right endpoint excluded. For example, the intervals [0, 5), [5, 10), [10, 15), ...

    Returns
    -------
    NDArray
            A NumPy array of shape (A, number of intervals) containing the indicator matrix.
            Each row corresponds to an age, and each column corresponds to an interval from the
            input series. Entries are 1 where the age falls into the interval, and 0 otherwise.

    Examples
    --------
    >>> ages = pd.Series(pd.cut(np.arange(1,85), bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]))
    >>> indicator_matrix = fine_coarse_matrix(ages)
    >>> print(indicator_matrix.shape)
    (84, 17)
    """
    if x.isnull().any():
        raise ValueError(
            "Input series contains NaN values. Check whether the intervals are defined correctly."
        )

    cuts_left = list(x.cat.categories.left)
    cuts_right = list(x.cat.categories.right)
    cuts = [cuts_left[0]] + cuts_right  # Include left endpoint
    age_min, age_max = int(cuts_left[0]), int(cuts_right[-1] - 1)

    # Create an empty matrix with zeros
    indicator_matrix = np.zeros((age_max - age_min + 1, len(cuts) - 1), dtype=int)

    # Iterate over each age and each cut interval
    for age in range(age_max + 1):
        for i, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
            if left <= age < right:
                indicator_matrix[age - age_min, i] = 1

    return indicator_matrix


def make_idarrs_for_intervals(data: pd.DataFrame, interval_col: str, aid: np.ndarray):
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
    bid_pad = np.array(
        [np.pad(x, (0, max_length - len(x)), constant_values=-1) for x in bid]
    )

    # Expand the ID array to match the padded index arrays
    aid_exp = np.repeat(aid[:, np.newaxis], max_length, axis=1)

    return aid_exp, bid_pad


def expand_ix_array(id_array: NDArray, length: int) -> NDArray:
    """
    Expand a 1D ID array by repeating each element along a new axis.

    This helper function is used to broadcast participant-level categorical IDs
    across the maximum coarse age group width for age aggregation.

    Parameters
    ----------
    id_array : NDArray
        1D array of categorical IDs, shape (n_obs,)
    length : int
        Number of repetitions along the new axis (typically max coarse age group width)

    Returns
    -------
    NDArray
        2D array with shape (n_obs, length) where each row contains repeated IDs

    Examples
    --------
    >>> ids = np.array([0, 1, 0, 2])
    >>> expand_ix_array(ids, 3)
    array([[0, 0, 0],
           [1, 1, 1],
           [0, 0, 0],
           [2, 2, 2]])
    """
    return np.repeat(id_array[:, np.newaxis], length, axis=1)
