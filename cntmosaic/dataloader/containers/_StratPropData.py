import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..._types import StratMode


@dataclass
class StratPropData:
    """
    Stratified population proportion specification for contact matrix adjustment.

    This class provides an intuitive, type-safe way to specify population proportions
    stratified by demographic variables (e.g., gender, occupation, region). These
    proportions are used to adjust contact matrices for demographic heterogeneity.

    Attributes
    ----------
    data : pd.DataFrame
        Dataframe containing stratified population proportions.
        Must have columns for: age, stratification variable(s), and proportions.
        Proportions within each age group must sum to 1.0.
    age_col : str
        Name of the column containing age values in the dataframe.
        Should match or be compatible with the age range in the main dataset.
    strat_col : str
        Name of the column containing the stratification variable.
        Examples: 'gender', 'occupation', 'region', 'setting'.
        Must be present in the main contact data for proper alignment.
    prop_col : str
        Name of the column containing population proportions.
        Values must be in [0, 1] and sum to 1.0 within each age group.

    Methods
    -------
    validate()
        Validates the population proportion data structure and values.
    from_counts(data, age_col, strat_col, count_col)
        Class method to create StratPropData from population counts.

    Raises
    ------
    ValueError
        If required columns are missing from the dataframe.
        If proportions don't sum to 1.0 within each age group (tolerance 1e-6).
        If proportion values are outside [0, 1] range.

    Examples
    --------
    >>> # From pre-computed proportions
    >>> df_gender = pd.DataFrame({
    ...     'age': [0, 0, 1, 1, 2, 2],
    ...     'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    ...     'proportion': [0.51, 0.49, 0.51, 0.49, 0.50, 0.50]
    ... })
    >>> pop_prop = StratPropData(
    ...     data=df_gender,
    ...     age_col='age',
    ...     strat_col='gender',
    ...     prop_col='proportion'
    ... )
    >>> pop_prop.validate()
    >>>
    >>> # From population counts (auto-computes proportions)
    >>> df_counts = pd.DataFrame({
    ...     'age': [0, 0, 1, 1],
    ...     'gender': ['M', 'F', 'M', 'F'],
    ...     'count': [510, 490, 505, 495]
    ... })
    >>> pop_prop = StratPropData.from_counts(
    ...     data=df_counts,
    ...     age_col='age',
    ...     strat_col='gender',
    ...     count_col='count'
    ... )
    >>>
    >>> # Multiple stratification variables (create separate StratPropData objects)
    >>> pop_prop_gender = StratPropData(df_gender, 'age', 'gender', 'prop')
    >>> pop_prop_region = StratPropData(df_region, 'age', 'region', 'prop')
    >>> dataloader = DataLoader(..., pop_prop=[pop_prop_gender, pop_prop_region])

    Notes
    -----
    - Validation is performed automatically during initialization via __post_init__
    - Proportions must sum to 1.0 within each age group (tolerance: 1e-6)
    - The stratification variable name must match the corresponding column in contact data
    - For multiple stratifications, create separate StratPropData objects
    """

    data: pd.DataFrame
    age_col: str
    strat_col: str
    prop_col: str

    def __post_init__(self) -> None:
        """Validate the population proportion specification after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate population proportion data structure and values.

        Checks:
        1. All required columns are present in the dataframe
        2. Proportions are in valid range [0, 1]
        3. Proportions sum to 1.0 within each age group (tolerance 1e-6)

        Raises
        ------
        ValueError
            If any validation check fails.

        Examples
        --------
        >>> pop_prop = StratPropData(df, 'age', 'gender', 'proportion')
        >>> # Validation happens automatically, but can be called explicitly:
        >>> pop_prop.validate()
        """
        # Check required columns exist
        required_cols = [self.age_col, self.strat_col, self.prop_col]
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in population proportion data: {missing}\n"
                f"Required: {required_cols}\n"
                f"Available: {list(self.data.columns)}"
            )

        # Check proportion values are in valid range
        props = self.data[self.prop_col]
        if (props < 0).any() or (props > 1).any():
            invalid_indices = self.data[(props < 0) | (props > 1)].index
            raise ValueError(
                f"Population proportions must be in range [0, 1]. "
                f"Found invalid values at indices: {list(invalid_indices)}\n"
                f"Invalid values: {props[invalid_indices].to_dict()}"
            )

        # Check proportions sum to 1 within each age group
        group_sums = self.data.groupby(self.age_col)[self.prop_col].sum()
        bad_groups = group_sums[np.abs(group_sums - 1.0) > 1e-6]

        if not bad_groups.empty:
            raise ValueError(
                f"Population proportions must sum to 1.0 within each age group (tolerance: 1e-6).\n"
                f"Ages with invalid sums: {list(bad_groups.index)}\n"
                f"Actual sums: {bad_groups.to_dict()}\n"
                f"Hint: Use StratPropData.from_counts() to automatically compute proportions from counts."
            )

    @classmethod
    def from_counts(
        cls,
        data: pd.DataFrame,
        age_col: str,
        strat_col: str,
        count_col: str,
        prop_col: str = "proportion",
    ) -> "StratPropData":
        """
        Create StratPropData from population counts (automatically computes proportions).

        This is a convenience constructor that automatically normalizes population counts
        to proportions within each age group. More intuitive than manually computing
        proportions.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with population counts stratified by age and category.
        age_col : str
            Name of age column.
        strat_col : str
            Name of stratification variable column (e.g., 'gender').
        count_col : str
            Name of column containing population counts.
        prop_col : str, default='proportion'
            Name to assign to the computed proportion column.

        Returns
        -------
        StratPropData
            New instance with proportions computed from counts.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'age': [0, 0, 1, 1, 2, 2],
        ...     'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        ...     'population': [5100, 4900, 5050, 4950, 5000, 5000]
        ... })
        >>> pop_prop = StratPropData.from_counts(
        ...     data=df,
        ...     age_col='age',
        ...     strat_col='gender',
        ...     count_col='population'
        ... )
        >>> # Proportions are automatically computed:
        >>> # age 0: M=0.51, F=0.49
        >>> # age 1: M=0.505, F=0.495
        >>> # age 2: M=0.50, F=0.50
        """
        # Validate required columns
        required_cols = [age_col, strat_col, count_col]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Available: {list(data.columns)}"
            )

        # Compute proportions within each age group
        df_with_props = data.copy()
        df_with_props[prop_col] = df_with_props.groupby(age_col)[count_col].transform(
            lambda x: x / x.sum()
        )

        return cls(
            data=df_with_props,
            age_col=age_col,
            strat_col=strat_col,
            prop_col=prop_col,
        )

    def compute_props(self, mode: StratMode) -> NDArray:
        """
        Compute stratum-specific proportions.

        Note
        ----
        The .from_counts() constructor handles normalization from counts to proportions.
        So this method will only work with proportions, not raw counts.

        Returns
        -------
        NDArray
            - PARTIAL: shape (n_strata, A)
            - FULL: shape (n_strata, n_strata, A, A)
        """
        if mode == StratMode.PARTIAL:
            prop_sa = self.data.pivot(
                index=self.strat_col, columns=self.age_col, values=self.prop_col
            ).values
            return prop_sa  # shape (n_strata, A)

        elif mode == StratMode.FULL:
            # props[s, t, a, b] = (P[s, a] / P[a]) * (P[t, b] / P[b])
            prop_sa = self.data.pivot(
                index=self.strat_col, columns=self.age_col, values=self.prop_col
            ).values  # shape (n_strata, A)

            # Outer product: prop[s, t, a, b] = prop[s, a] * prop[t, b]
            # Reshape for broadcasting: (n_strata, 1, A, 1) * (1, n_strata, 1, A)
            prop_full = prop_sa[:, None, :, None] * prop_sa[None, :, None, :]
            return prop_full  # shape (n_strata, n_strata, A, A)

    def validate_for_mode(self, mode: StratMode) -> None:
        """
        Validate data structure matches stratification mode.

        For PARTIAL: Requires columns [strat_col, age_col, prop_col].
        For FULL: Same requirements (outer product computed internally)
        """
        required_cols = [self.age_col, self.strat_col, self.prop_col]
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {mode.name} mode: {missing}\n"
                f"Required: {required_cols}\n"
                f"Available: {list(self.data.columns)}"
            )
