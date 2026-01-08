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

    **New in v2.0**: Supports multiple stratification columns with automatic merging.
    If multiple columns are provided, they will be automatically combined into a
    composite variable (e.g., ['gender', 'region'] → 'gender_region').

    Attributes
    ----------
    data : pd.DataFrame
        Dataframe containing stratified population proportions.
        Must have columns for: age, stratification variable(s), and proportions.
        Proportions within each age group must sum to 1.0.
    age_col : str
        Name of the column containing age values in the dataframe.
        Should match or be compatible with the age range in the main dataset.
    strat_var_cols : Union[str, List[str]], optional
        Name(s) of the stratification variable column(s) in the DataFrame.
        - Single column: 'gender', 'occupation', 'region', etc.
        - Multiple columns: ['gender', 'region'] will be merged into 'gender_region'
        If not provided, will be auto-detected from data columns (excluding age_col and prop_col).
    prop_col : str, default='proportion'
        Name of the column containing population proportions.
        Values must be in [0, 1] and sum to 1.0 within each age group.

    Properties
    ----------
    var_name : str
        The standardized name of the stratification variable (read-only).
        - For single column: same as the column name
        - For multiple columns: underscore-joined (e.g., 'gender_region')
        This is automatically determined and used by DataLoader for matching.
    strat_col : str
        The final stratification column name in the processed data (read-only).
        After merging, this contains the composite variable name.

    Methods
    -------
    validate()
        Validates the population proportion data structure and values.
    from_counts(data, age_col, strat_var_cols, count_col)
        Class method to create StratPropData from population counts.
    compute_marginal_multipliers(strat_modes)
        Computes marginal multipliers for each stratification variable.
    compute_multipliers(strat_modes)
        Computes stratified multipliers for contact modifier weighting.

    Raises
    ------
    ValueError
        If required columns are missing from the dataframe.
        If proportions don't sum to 1.0 within each age group (tolerance 1e-6).
        If proportion values are outside [0, 1] range.

    Examples
    --------
    >>> # Single stratification variable (auto-detected)
    >>> df_gender = pd.DataFrame({
    ...     'age': [0, 0, 1, 1, 2, 2],
    ...     'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    ...     'proportion': [0.51, 0.49, 0.51, 0.49, 0.50, 0.50]
    ... })
    >>> pop_prop = StratPropData(data=df_gender, age_col='age', prop_col='proportion')
    >>> pop_prop.var_name  # 'gender' (auto-detected)
    'gender'
    >>>
    >>> # Explicit single column
    >>> pop_prop = StratPropData(
    ...     data=df_gender,
    ...     age_col='age',
    ...     strat_var_cols='gender',
    ...     prop_col='proportion'
    ... )
    >>>
    >>> # Multiple columns (automatic composite)
    >>> df_composite = pd.DataFrame({
    ...     'age': [0, 0, 0, 0, 1, 1, 1, 1],
    ...     'gender': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
    ...     'region': ['North', 'South', 'North', 'South', 'North', 'South', 'North', 'South'],
    ...     'prop': [0.26, 0.24, 0.25, 0.25, 0.26, 0.24, 0.25, 0.25]
    ... })
    >>> pop_prop = StratPropData(
    ...     data=df_composite,
    ...     age_col='age',
    ...     strat_var_cols=['gender', 'region'],
    ...     prop_col='prop'
    ... )
    >>> pop_prop.var_name  # 'gender_region' (auto-generated composite)
    'gender_region'
    >>> pop_prop.data['gender_region'].unique()  # Merged column
    ['M_North', 'M_South', 'F_North', 'F_South']
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
    ...     count_col='count'
    ... )  # strat_var_cols='gender' auto-detected

    Notes
    -----
    - Validation is performed automatically during initialization via __post_init__
    - Proportions must sum to 1.0 within each age group (tolerance: 1e-6)
    - Multiple stratification columns are automatically merged into a composite variable
    """

    data: pd.DataFrame
    age_col: str
    strat_var_cols: Optional[Union[str, List[str]]] = None
    prop_col: str = "proportion"

    # Internal working attribute (set in __post_init__)
    strat_col: str = None

    def __post_init__(self) -> None:
        """Process stratification columns and validate after initialization."""
        # Convert single string to list for uniform processing
        if isinstance(self.strat_var_cols, str):
            self.strat_var_cols = [self.strat_var_cols]

        # Validate columns exist
        missing = [col for col in self.strat_var_cols if col not in self.data.columns]
        if missing:
            raise ValueError(
                f"Stratification columns not found in data: {missing}\n"
                f"Available columns: {list(self.data.columns)}"
            )

        # Merge multiple stratification columns into composite if needed
        if len(self.strat_var_cols) > 1:
            # Create composite column name from multiple variables
            composite_name = "_".join(self.strat_var_cols)
            object.__setattr__(self, "strat_col", composite_name)

            # Merge columns into composite
            self.data = self.data.copy()
            self.data[self.strat_col] = self.data[self.strat_var_cols].apply(
                lambda row: "_".join(row.astype(str)), axis=1
            )
        else:
            # Single column - use it directly
            object.__setattr__(self, "strat_col", self.strat_var_cols[0])

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
        strat_var_cols: Optional[Union[str, List[str]]] = None,
        count_col: str = "count",
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
        strat_var_cols : Union[str, List[str]], optional
            Name(s) of stratification variable column(s) in the DataFrame.
            If None, will be auto-detected from columns (excluding age_col and count_col).
        count_col : str, default='count'
            Name of column containing population counts.
        prop_col : str, default='proportion'
            Name to assign to the computed proportion column.

        Returns
        -------
        StratPropData
            New instance with proportions computed from counts.

        Examples
        --------
        >>> # Single variable (auto-detected)
        >>> df = pd.DataFrame({
        ...     'age': [0, 0, 1, 1, 2, 2],
        ...     'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        ...     'population': [5100, 4900, 5050, 4950, 5000, 5000]
        ... })
        >>> pop_prop = StratPropData.from_counts(
        ...     data=df,
        ...     age_col='age',
        ...     count_col='population'
        ... )  # 'gender' auto-detected
        >>>
        >>> # Multiple variables (automatic composite)
        >>> df_multi = pd.DataFrame({
        ...     'age': [0, 0, 0, 0],
        ...     'gender': ['M', 'M', 'F', 'F'],
        ...     'region': ['North', 'South', 'North', 'South'],
        ...     'count': [2600, 2400, 2500, 2500]
        ... })
        >>> pop_prop = StratPropData.from_counts(
        ...     data=df_multi,
        ...     age_col='age',
        ...     strat_var_cols=['gender', 'region'],
        ...     count_col='count'
        ... )
        >>> pop_prop.var_name  # 'gender_region'
        """
        # Auto-detect strat_var_cols if not provided
        if strat_var_cols is None:
            potential_cols = [
                col for col in data.columns if col not in [age_col, count_col, prop_col]
            ]
            if len(potential_cols) == 0:
                raise ValueError(
                    f"Could not auto-detect stratification columns. "
                    f"DataFrame only contains: {list(data.columns)}\n"
                    f"After excluding age_col='{age_col}' and count_col='{count_col}', "
                    f"no columns remain.\n"
                    f"Please specify strat_var_cols explicitly."
                )
            strat_var_cols = potential_cols

        # Convert to list for uniform processing
        if isinstance(strat_var_cols, str):
            strat_var_cols = [strat_var_cols]

        # Validate required columns
        required_cols = [age_col, count_col] + strat_var_cols
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
            strat_var_cols=strat_var_cols,
            prop_col=prop_col,
        )

    def compute_marginal_multipliers(
        self, strat_modes: Dict[str, StratMode]
    ) -> Dict[str, NDArray]:
        """
        Compute the marginal multipliers for each stratification variable.
        These marginals are used for centering the contact rate modifier priors in the latent space.

        Parameters
        ----------
        strat_modes : Dict[str, StratMode]
            Dictionary mapping stratification variable names to their StratMode (PARTIAL or FULL).

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping each stratification variable to its marginal multipliers array.
        """

        strat_vars = list(strat_modes.keys())
        marginal_multipliers = {}
        for var in strat_vars:
            mode = strat_modes[var]

            # Sum over other stratification variables to get marginal proportions
            df_marginal = (
                self.data.groupby([self.age_col] + [var], observed=False)[self.prop_col]
                .sum()
                .reset_index()
            )

            # Sort by age and strat variable
            df_marginal = df_marginal.sort_values(by=[self.age_col, var])
            prop_array = df_marginal.pivot(
                index=var, columns=self.age_col, values=self.prop_col
            ).values  # shape (n_strata, A)

            if mode == StratMode.PARTIAL:
                marginal_multipliers[var] = prop_array[
                    :, :, None
                ]  # shape (n_strata, A, 1)
            elif mode == StratMode.FULL:
                mult_array = prop_array[:, None, :, None] * prop_array[None, :, None, :]
                n_strata = prop_array.shape[0]
                A = prop_array.shape[1]
                marginal_multipliers[var] = mult_array.reshape(
                    n_strata * n_strata, A, A
                )  # shape (n_strata^2, A, A)
            else:
                raise ValueError(f"Unknown StratMode: {mode}")

        return marginal_multipliers

    def compute_multipliers(self, strat_modes: Dict[str, StratMode]) -> NDArray:
        """
        Compute stratified multipliers for contact modifier weighting.

        This method takes the dictionary of stratification modes and computes the population proportion
        offsets by age and stratification variables. It supports both PARTIAL, FULL, and mixed scenarios.

        Note
        ----
        The .from_counts() constructor handles normalization from counts to proportions.
        So this method will only work with proportions, not raw counts.

        Returns
        -------
        NDArray
            - PARTIAL: shape (n_strata, A, 1)
            - FULL: shape (n_strata**2, A, A)
        """
        strat_vars_source = list(strat_modes.keys())
        strat_vars_target = [
            var for var in strat_vars_source if strat_modes[var] == StratMode.FULL
        ]

        df_prop_source = self.data[[self.age_col] + strat_vars_source + [self.prop_col]]
        sort_vars = [self.age_col] + strat_vars_source
        df_prop_source = df_prop_source.sort_values(by=sort_vars)

        # Combine stratification variables into single column for grouping
        if len(strat_vars_source) == 0:
            raise ValueError("No stratification variables provided")

        df_prop_source["strat_combined"] = df_prop_source[strat_vars_source].apply(
            lambda row: "_".join(row.astype(str)), axis=1
        )

        # Aggregate and pivot to get source proportions
        df_grouped_source = (
            df_prop_source.groupby([self.age_col, "strat_combined"])[self.prop_col]
            .sum()
            .reset_index()
        )
        prop_sa = df_grouped_source.pivot(
            index="strat_combined", columns=self.age_col, values=self.prop_col
        ).values  # shape (n_strata_source, A)

        strat_labels_source = df_grouped_source["strat_combined"].unique()

        # PARTIAL case: single variable stratification
        if len(strat_vars_target) == 0:
            result = prop_sa[:, :, None]  # shape (n_strata, A, 1)
            return self._handle_nans(result, strat_labels_source, None, mode="PARTIAL")

        # FULL case: compute outer product for source × target
        df_prop_target = self.data[[self.age_col] + strat_vars_target + [self.prop_col]]
        sort_vars = [self.age_col] + strat_vars_target
        df_prop_target = df_prop_target.sort_values(by=sort_vars)
        df_prop_target["strat_combined"] = df_prop_target[strat_vars_target].apply(
            lambda row: "_".join(row.astype(str)), axis=1
        )

        df_grouped_target = (
            df_prop_target.groupby([self.age_col, "strat_combined"])[self.prop_col]
            .sum()
            .reset_index()
        )
        prop_tb = df_grouped_target.pivot(
            index="strat_combined", columns=self.age_col, values=self.prop_col
        ).values  # shape (n_strata_target, A)

        strat_labels_target = df_grouped_target["strat_combined"].unique()

        # Compute outer product and reshape
        prop_stab = prop_sa[:, None, :, None] * prop_tb[None, :, None, :]
        n_strata_source, n_strata_target, A = (
            prop_sa.shape[0],
            prop_tb.shape[0],
            prop_sa.shape[1],
        )

        result = prop_stab.reshape(n_strata_source * n_strata_target, A, A)
        return self._handle_nans(
            result, strat_labels_source, strat_labels_target, mode="FULL"
        )

    def _handle_nans(
        self,
        multipliers: NDArray,
        strat_labels_source: NDArray,
        strat_labels_target: Optional[NDArray],
        mode: str,
    ) -> NDArray:
        """
        Check for NaN values in multipliers and replace with neutral value (1.0).

        Parameters
        ----------
        multipliers : NDArray
            The computed multipliers array that may contain NaN values.
        strat_labels_source : NDArray
            Labels for source strata.
        strat_labels_target : NDArray, optional
            Labels for target strata (only for FULL mode).
        mode : str
            Either 'PARTIAL' or 'FULL' to determine warning message format.

        Returns
        -------
        NDArray
            Multipliers with NaN values replaced by 1.0.
        """
        if not np.isnan(multipliers).any():
            return multipliers

        # Identify NaN locations
        nan_mask = np.isnan(multipliers)
        nan_count = nan_mask.sum()

        # Generate warning message with affected strata combinations
        if mode == "PARTIAL":
            # Shape: (n_strata, A, 1)
            affected_strata = set()
            for strat_idx in range(multipliers.shape[0]):
                if nan_mask[strat_idx].any():
                    affected_strata.add(strat_labels_source[strat_idx])

            warning_msg = (
                f"Found {nan_count} NaN values in stratified multipliers (PARTIAL mode). "
                f"These have been replaced with 1.0 (neutral multiplier). "
                f"Affected strata: {sorted(affected_strata)}. "
                f"This typically occurs when certain age-strata combinations have no population data. "
                f"Verify your population proportion data completeness."
            )
        else:  # FULL mode
            # Shape: (n_strata_source * n_strata_target, A, A)
            affected_pairs = set()
            n_strata_source = len(strat_labels_source)
            n_strata_target = len(strat_labels_target)

            for combined_idx in range(multipliers.shape[0]):
                if nan_mask[combined_idx].any():
                    source_idx = combined_idx // n_strata_target
                    target_idx = combined_idx % n_strata_target
                    pair = (
                        strat_labels_source[source_idx],
                        strat_labels_target[target_idx],
                    )
                    affected_pairs.add(pair)

            warning_msg = (
                f"Found {nan_count} NaN values in stratified multipliers (FULL mode). "
                f"These have been replaced with 1.0 (neutral multiplier). "
                f"Affected strata pairs (source, target): {sorted(affected_pairs)}. "
                f"This typically occurs when certain age-strata combinations have no population data. "
                f"Verify your population proportion data completeness."
            )

        warnings.warn(warning_msg, UserWarning, stacklevel=3)

        # Replace NaNs with neutral multiplier
        multipliers_clean = multipliers.copy()
        multipliers_clean[nan_mask] = 1.0

        return multipliers_clean

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
