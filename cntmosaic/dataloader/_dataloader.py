"""
Data loading and preprocessing for social contact matrix estimation.

This module provides classes for loading, validating, and preprocessing social contact
survey data for use in Bayesian contact matrix estimation models. It handles participant
data, contact data, and population distributions, ensuring data consistency and creating
properly formatted datasets for downstream analysis.

Classes
-------
CoordToColumns : Dataclass for column name mapping
PopulationProportion : Dataclass for stratified population proportions
BaseLoader : Abstract base class for data loading
DataLoader : Main data loader for contact survey data
"""

import itertools
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from ._utils import make_idarrs_for_intervals


@dataclass
class CoordToColumns:
    """
    Column name mapping for contact survey data.

    This dataclass specifies the mapping between input dataframe columns and the
    variables required by contact matrix estimation models. It ensures consistent
    naming conventions and validates that required column mappings are provided.

    Attributes
    ----------
    age_part : str
        Column name for participant age in the participant dataframe.
        Should contain integer ages or age groups.
    age_cnt : Optional[str], default=None
        Column name for contact age in the contact dataframe.
        Use this OR age_grp_cnt, not both.
    age_grp_cnt : Optional[str], default=None
        Column name for contact age group in the contact dataframe.
        Must be pd.IntervalIndex if used. Use this OR age_cnt, not both.
    id_var : str, default='id'
        Column name for unique participant identifiers.
        Must be present in both participant and contact dataframes.
    y : str, default='y'
        Column name for number of contacts in the contact dataframe.
        If not present, will be auto-created with value 1 per contact.
    z : str, default='z'
        Column name for number of group contacts in the participant dataframe.
        If not present, will be auto-created with value 0.
    grp_vars_part : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) in participant dataframe.
        Can be a single string or list of strings. Examples: 'gender', ['gender', 'setting']
    grp_vars_cnt : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) in contact dataframe.
        Can be a single string or list of strings.
    repeat_part : Optional[str], default=None
        Column name for repeat interview count in participant dataframe.
        Used to model repeat participation effects. If provided, automatically
        added to grp_vars_part during post-initialization.
    age_pop : Optional[str], default=None
        Column name for age in population dataframe.
        Required for population weighting; must be provided with size_pop.
    size_pop : Optional[str], default=None
        Column name for population size/proportion in population dataframe.
        Required for population weighting; must be provided with age_pop.

    Raises
    ------
    ValueError
        If neither age_cnt nor age_grp_cnt is provided.
        If age_pop and size_pop are not both set or both None.

    Warnings
    --------
    UserWarning
        If the same stratification variable appears in both grp_vars_part and
        grp_vars_cnt. The variable in grp_vars_cnt will be automatically removed
        to avoid ambiguity.

    Examples
    --------
    >>> # Basic usage with individual contact ages
    >>> col_map = CoordToColumns(
    ...     age_part="participant_age",
    ...     age_cnt="contact_age",
    ...     id_var="participant_id",
    ...     age_pop="age",
    ...     size_pop="population_size"
    ... )
    >>>
    >>> # With age groups and stratification
    >>> col_map = CoordToColumns(
    ...     age_part="age_participant",
    ...     age_grp_cnt="age_group_contact",
    ...     grp_vars_part=["gender", "location"],
    ...     grp_vars_cnt="setting",
    ...     age_pop="age",
    ...     size_pop="N"
    ... )
    >>>
    >>> # With repeat interview effects
    >>> col_map = CoordToColumns(
    ...     age_part="age",
    ...     age_cnt="contact_age",
    ...     repeat_part="interview_round",
    ...     age_pop="age",
    ...     size_pop="pop_count"
    ... )

    Notes
    -----
    - The __post_init__ method automatically:
      * Converts single string grp_vars to lists
      * Resolves conflicts when same variable appears in both participant and contact data
      * Adds repeat_part to grp_vars_part if specified
    - For age groups (age_grp_cnt), the contact dataframe column must use pd.IntervalIndex
    - Population columns (age_pop, size_pop) are required for most models
    """

    age_part: str
    age_cnt: Optional[str] = None
    age_grp_cnt: Optional[str] = None
    id_var: str = "id"
    y: str = "y"
    z: str = "z"
    grp_vars_part: Optional[Union[List[str], str]] = None
    grp_vars_cnt: Optional[Union[List[str], str]] = None
    repeat_part: Optional[str] = None
    age_pop: Optional[str] = None
    size_pop: Optional[str] = None

    def age_vars(self) -> List[str]:
        """
        Get list of age variable names from contact and participant data.

        Returns
        -------
        List[str]
            Two-element list containing [contact_age_var, participant_age_var].

        Raises
        ------
        ValueError
            If neither age_cnt nor age_grp_cnt is provided.

        Examples
        --------
        >>> col_map = CoordToColumns(age_part="age_p", age_cnt="age_c")
        >>> col_map.age_vars()
        ['age_c', 'age_p']
        """
        if self.age_cnt:
            return [self.age_cnt, self.age_part]
        elif self.age_grp_cnt:
            return [self.age_grp_cnt, self.age_part]
        else:
            raise ValueError(
                "One of 'age_cnt' or 'age_grp_cnt' must be provided. "
                "Please specify either individual contact ages (age_cnt) or "
                "contact age groups (age_grp_cnt)."
            )

    def __post_init__(self) -> None:
        """
        Post-initialization processing and validation.

        Automatically called after dataclass initialization to:
        1. Convert string grp_vars to lists for consistent handling
        2. Validate that age_pop and size_pop are provided together
        3. Resolve naming conflicts between participant and contact stratification variables
        4. Add repeat_part to participant stratification variables if specified

        Raises
        ------
        ValueError
            If age_pop is provided without size_pop, or vice versa.

        Warnings
        --------
        UserWarning
            If duplicate stratification variable names are found in both
            grp_vars_part and grp_vars_cnt. The duplicate in grp_vars_cnt
            will be removed.
        """
        # Convert single strings to lists for consistent processing
        if isinstance(self.grp_vars_part, str):
            object.__setattr__(self, "grp_vars_part", [self.grp_vars_part])
        elif self.grp_vars_part is None:
            object.__setattr__(self, "grp_vars_part", [])

        if isinstance(self.grp_vars_cnt, str):
            object.__setattr__(self, "grp_vars_cnt", [self.grp_vars_cnt])
        elif self.grp_vars_cnt is None:
            object.__setattr__(self, "grp_vars_cnt", [])

        # Validate population column specifications
        if (self.age_pop is None) != (self.size_pop is None):
            raise ValueError(
                "Both 'age_pop' and 'size_pop' must be specified together, or both left as None. "
                f"Currently: age_pop={self.age_pop}, size_pop={self.size_pop}"
            )

        # Handle naming conflicts between participant and contact stratification variables
        conflicting_vars = set(self.grp_vars_part).intersection(set(self.grp_vars_cnt))
        if conflicting_vars:
            warnings.warn(
                f"Stratification variable(s) with identical names found in both "
                f"participant and contact data: {conflicting_vars}\n"
                f"Default behavior: keeping variable from participant data, "
                f"removing from contact data.\n"
                f"To treat them as separate variables, append suffixes to distinguish them "
                f"(e.g., 'setting_part' vs 'setting_cnt').",
                UserWarning,
                stacklevel=2,
            )
            for var in conflicting_vars:
                self.grp_vars_part.remove(var)

        # Add repeat interview column to participant grouping variables if specified
        if self.repeat_part is not None:
            if self.repeat_part not in self.grp_vars_part:
                self.grp_vars_part.append(self.repeat_part)


@dataclass
class PopulationProportion:
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
    stratify_by : str
        Name of the column containing the stratification variable.
        Examples: 'gender', 'occupation', 'region', 'setting'.
        Must be present in the main contact data for proper alignment.
    proportion_col : str
        Name of the column containing population proportions.
        Values must be in [0, 1] and sum to 1.0 within each age group.

    Methods
    -------
    validate()
        Validates the population proportion data structure and values.
    from_counts(data, age_col, stratify_by, count_col)
        Class method to create PopulationProportion from population counts.

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
    >>> pop_prop = PopulationProportion(
    ...     data=df_gender,
    ...     age_col='age',
    ...     stratify_by='gender',
    ...     proportion_col='proportion'
    ... )
    >>> pop_prop.validate()
    >>>
    >>> # From population counts (auto-computes proportions)
    >>> df_counts = pd.DataFrame({
    ...     'age': [0, 0, 1, 1],
    ...     'gender': ['M', 'F', 'M', 'F'],
    ...     'count': [510, 490, 505, 495]
    ... })
    >>> pop_prop = PopulationProportion.from_counts(
    ...     data=df_counts,
    ...     age_col='age',
    ...     stratify_by='gender',
    ...     count_col='count'
    ... )
    >>>
    >>> # Multiple stratification variables (create separate PopulationProportion objects)
    >>> pop_prop_gender = PopulationProportion(df_gender, 'age', 'gender', 'prop')
    >>> pop_prop_region = PopulationProportion(df_region, 'age', 'region', 'prop')
    >>> dataloader = DataLoader(..., pop_prop=[pop_prop_gender, pop_prop_region])

    Notes
    -----
    - Validation is performed automatically during initialization via __post_init__
    - Proportions must sum to 1.0 within each age group (tolerance: 1e-6)
    - The stratification variable name must match the corresponding column in contact data
    - For multiple stratifications, create separate PopulationProportion objects
    """

    data: pd.DataFrame
    age_col: str
    stratify_by: str
    proportion_col: str

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
        >>> pop_prop = PopulationProportion(df, 'age', 'gender', 'proportion')
        >>> # Validation happens automatically, but can be called explicitly:
        >>> pop_prop.validate()
        """
        # Check required columns exist
        required_cols = [self.age_col, self.stratify_by, self.proportion_col]
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in population proportion data: {missing}\n"
                f"Required: {required_cols}\n"
                f"Available: {list(self.data.columns)}"
            )

        # Check proportion values are in valid range
        props = self.data[self.proportion_col]
        if (props < 0).any() or (props > 1).any():
            invalid_indices = self.data[(props < 0) | (props > 1)].index
            raise ValueError(
                f"Population proportions must be in range [0, 1]. "
                f"Found invalid values at indices: {list(invalid_indices)}\n"
                f"Invalid values: {props[invalid_indices].to_dict()}"
            )

        # Check proportions sum to 1 within each age group
        group_sums = self.data.groupby(self.age_col)[self.proportion_col].sum()
        bad_groups = group_sums[np.abs(group_sums - 1.0) > 1e-6]

        if not bad_groups.empty:
            raise ValueError(
                f"Population proportions must sum to 1.0 within each age group (tolerance: 1e-6).\n"
                f"Ages with invalid sums: {list(bad_groups.index)}\n"
                f"Actual sums: {bad_groups.to_dict()}\n"
                f"Hint: Use PopulationProportion.from_counts() to automatically compute proportions from counts."
            )

    @classmethod
    def from_counts(
        cls,
        data: pd.DataFrame,
        age_col: str,
        stratify_by: str,
        count_col: str,
        proportion_col: str = "proportion",
    ) -> "PopulationProportion":
        """
        Create PopulationProportion from population counts (automatically computes proportions).

        This is a convenience constructor that automatically normalizes population counts
        to proportions within each age group. More intuitive than manually computing
        proportions.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with population counts stratified by age and category.
        age_col : str
            Name of age column.
        stratify_by : str
            Name of stratification variable column (e.g., 'gender').
        count_col : str
            Name of column containing population counts.
        proportion_col : str, default='proportion'
            Name to assign to the computed proportion column.

        Returns
        -------
        PopulationProportion
            New instance with proportions computed from counts.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'age': [0, 0, 1, 1, 2, 2],
        ...     'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        ...     'population': [5100, 4900, 5050, 4950, 5000, 5000]
        ... })
        >>> pop_prop = PopulationProportion.from_counts(
        ...     data=df,
        ...     age_col='age',
        ...     stratify_by='gender',
        ...     count_col='population'
        ... )
        >>> # Proportions are automatically computed:
        >>> # age 0: M=0.51, F=0.49
        >>> # age 1: M=0.505, F=0.495
        >>> # age 2: M=0.50, F=0.50
        """
        # Validate required columns
        required_cols = [age_col, stratify_by, count_col]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Available: {list(data.columns)}"
            )

        # Compute proportions within each age group
        df_with_props = data.copy()
        df_with_props[proportion_col] = df_with_props.groupby(age_col)[
            count_col
        ].transform(lambda x: x / x.sum())

        return cls(
            data=df_with_props,
            age_col=age_col,
            stratify_by=stratify_by,
            proportion_col=proportion_col,
        )

    def to_tuple(self) -> Tuple[pd.DataFrame, str, str, str]:
        """
        Convert to legacy tuple format for backward compatibility.

        Returns
        -------
        Tuple[pd.DataFrame, str, str, str]
            4-tuple: (data, age_col, stratify_by, proportion_col)

        Notes
        -----
        This method exists for backward compatibility with the old API.
        New code should use PopulationProportion objects directly.
        """
        return (self.data, self.age_col, self.stratify_by, self.proportion_col)


class BaseLoader(ABC):
    """
    Abstract base class for loading and preprocessing contact survey data.

    This class provides the core functionality for validating, merging, and transforming
    contact survey data into formats suitable for statistical modeling. It handles:
    - Data validation and quality checks
    - Age range alignment between sample and population data
    - Calculation of sample sizes and contact frequencies
    - Group contact offset corrections
    - Creation of xarray datasets for model input

    Subclasses must implement specific validation methods for their data sources.

    Parameters
    ----------
    data : pd.DataFrame
        Combined dataframe containing both participant and contact information.
        Must include columns specified in col_map.
    pop : pd.DataFrame
        Population dataframe with age distribution and population sizes.
        Used for normalization and rate calculations.
    col_map : CoordToColumns
        Column mapping object specifying how to interpret dataframe columns.
    pop_prop : Optional[Union[PopulationProportion, List[PopulationProportion]]], default=None
        Population proportion specification(s) for stratification variables.
        Can be either:
        - A single PopulationProportion object
        - A list of PopulationProportion objects for multiple stratifications
        If None, no stratified population proportions are used.

    Attributes
    ----------
    data : pd.DataFrame
        Validated and cleaned input data.
    col_map : CoordToColumns
        Column mapping configuration.
    pop : pd.DataFrame
        Population dataframe.
    min_age : int
        Minimum age in the aligned age range.
    max_age : int
        Maximum age in the aligned age range.
    ds : Optional[xr.Dataset]
        Generated xarray dataset (created by load() method).

    Raises
    ------
    KeyError
        If required columns are missing from input dataframes.
    TypeError
        If age group column is not properly formatted as pd.IntervalIndex,
        or if pop_prop contains non-PopulationProportion objects.
    ValueError
        If age ranges are inconsistent or population proportions don't sum to 1.

    Notes
    -----
    - Age ranges are automatically aligned between sample and population data
    - Missing values in critical columns will raise errors
    - Object-type columns are automatically converted to categorical
    - Group contacts are handled with special offset corrections

    See Also
    --------
    DataLoader : Concrete implementation for separate participant/contact dataframes
    CoordToColumns : Column mapping specification
    PopulationProportion : Stratified population proportion specification
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pop: pd.DataFrame,
        col_map: CoordToColumns,
        pop_prop: Optional[
            Union[PopulationProportion, List[PopulationProportion]]
        ] = None,
    ) -> None:
        """
        Initialize base loader with data validation.

        Parameters
        ----------
        data : pd.DataFrame
            Combined input dataframe.
        pop : pd.DataFrame
            Population dataframe.
        col_map : CoordToColumns
            Column mapping specification.
        pop_prop : Optional[Union[PopulationProportion, List[PopulationProportion]]], default=None
            Population proportion specification(s). Can be either:
            - A single PopulationProportion object
            - A list of PopulationProportion objects for multiple stratifications
            If None, no stratified population proportions are used.

        Raises
        ------
        TypeError
            If pop_prop contains elements that are not PopulationProportion objects.
        """
        self.data = self._validate(data, pop, col_map)
        self.col_map = col_map
        self.pop = pop
        self.ds: Optional[xr.Dataset] = None

        # Handle pop_prop parameter (accept single object or list)
        if pop_prop is not None:
            # Convert single PopulationProportion to list for uniform processing
            if isinstance(pop_prop, PopulationProportion):
                pop_prop = [pop_prop]

            # Validate all PopulationProportion objects
            for i, pp in enumerate(pop_prop):
                if not isinstance(pp, PopulationProportion):
                    raise TypeError(
                        f"Element {i} in pop_prop must be a "
                        f"PopulationProportion object, got {type(pp)}"
                    )
                # Validation happens automatically in PopulationProportion.__post_init__

            # Convert to internal format for processing
            self.pop_prop = [pp.to_tuple() for pp in pop_prop]

    def _validate(
        self, data: pd.DataFrame, pop: pd.DataFrame, col_map: CoordToColumns
    ) -> pd.DataFrame:
        """
        Validate and preprocess input data for consistency and completeness.

        This method performs comprehensive validation including:
        1. Checking for required columns
        2. Converting object columns to categorical
        3. Validating age group formatting
        4. Aligning age ranges between sample and population data
        5. Subsetting to relevant columns only

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe with contact and participant information.
        pop : pd.DataFrame
            Population dataframe with age distribution.
        col_map : CoordToColumns
            Column mapping specification.

        Returns
        -------
        pd.DataFrame
            Validated and cleaned dataframe ready for processing.

        Raises
        ------
        KeyError
            If required columns are missing from the dataframe.
        TypeError
            If age_grp_cnt column is not properly formatted as pd.IntervalIndex.

        Warnings
        --------
        UserWarning
            - If object columns are converted to categorical
            - If age ranges between sample and population don't match
        """
        # Validate required columns are present
        cols_needed = [col_map.y, col_map.z, col_map.id_var] + col_map.age_vars()
        if col_map.grp_vars_part:
            cols_needed.extend(col_map.grp_vars_part)
        if col_map.grp_vars_cnt:
            cols_needed.extend(col_map.grp_vars_cnt)

        missing_cols = [col for col in cols_needed if col not in data.columns]
        if missing_cols:
            raise KeyError(
                f"Missing required columns in data: {missing_cols}\n"
                f"Available columns: {list(data.columns)}"
            )

        # Subset to only needed columns and drop rows with missing values
        unique_cols = list(np.unique(cols_needed))
        data = data[unique_cols].dropna().copy()

        if data.empty:
            raise ValueError(
                "After removing missing values, the dataframe is empty. "
                "Check for excessive NaN values in required columns."
            )

        # Convert object columns to categorical for efficiency
        object_cols = data.select_dtypes(include="object").columns
        for col in object_cols:
            if col != col_map.id_var and not isinstance(
                data[col].dtype, pd.CategoricalDtype
            ):
                warnings.warn(
                    f"Column '{col}' is object type. Converting to categorical for efficiency.",
                    UserWarning,
                    stacklevel=2,
                )
                data[col] = data[col].astype("category")

        # Validate age group column formatting if specified
        if col_map.age_grp_cnt:
            is_categorical = isinstance(
                data[col_map.age_grp_cnt].dtype, pd.CategoricalDtype
            )
            if is_categorical:
                are_intervals = isinstance(
                    data[col_map.age_grp_cnt].cat.categories, pd.IntervalIndex
                )
                if not are_intervals:
                    raise TypeError(
                        f"Column '{col_map.age_grp_cnt}' must have pd.IntervalIndex categories, "
                        f"got {type(data[col_map.age_grp_cnt].cat.categories)}"
                    )
            else:
                raise TypeError(
                    f"Column '{col_map.age_grp_cnt}' must be categorical with interval categories. "
                    f"Current type: {data[col_map.age_grp_cnt].dtype}"
                )

        # Determine age ranges and ensure consistency
        part_min_age = int(data[col_map.age_part].min())
        part_max_age = int(data[col_map.age_part].max())

        if col_map.age_cnt:
            cnt_min_age = int(data[col_map.age_cnt].min())
            cnt_max_age = int(data[col_map.age_cnt].max())
        else:  # age_grp_cnt is specified
            cnt_min_age = int(data[col_map.age_grp_cnt].min().left)
            cnt_max_age = int(data[col_map.age_grp_cnt].max().right - 1)

        pop_min_age = int(pop[col_map.age_pop].min())
        pop_max_age = int(pop[col_map.age_pop].max())

        # Determine overall sample age range
        sample_min_age = min(part_min_age, cnt_min_age)
        sample_max_age = max(part_max_age, cnt_max_age)

        # Align minimum age with population data
        if sample_min_age != pop_min_age:
            warnings.warn(
                f"Sample minimum age ({sample_min_age}) differs from population minimum age ({pop_min_age}). "
                f"Filtering sample data to match population (age >= {pop_min_age}).",
                UserWarning,
                stacklevel=2,
            )
            data = data[data[col_map.age_part] >= pop_min_age].copy()
            if data.empty:
                raise ValueError(
                    f"After filtering to age >= {pop_min_age}, no data remains. "
                    "Check age range compatibility."
                )
            min_age = pop_min_age
        else:
            min_age = sample_min_age

        # Align maximum age with population data
        if sample_max_age != pop_max_age:
            warnings.warn(
                f"Sample maximum age ({sample_max_age}) differs from population maximum age ({pop_max_age}). "
                f"Using population maximum age ({pop_max_age}) for analysis.",
                UserWarning,
                stacklevel=2,
            )
            max_age = pop_max_age
        else:
            max_age = sample_max_age

        # Store age bounds as instance attributes
        self.min_age: int = min_age
        self.max_age: int = max_age

        return data

    def _load_pop_proportions(self) -> None:
        """
        Append stratified population proportion arrays to the dataset.

        This method processes population proportion specifications (from pop_prop)
        and adds them as xarray DataArrays to the dataset. Each specification
        provides:
        1. A dataframe with population proportions
        2. Age column name
        3. Stratification variable name (e.g., 'gender', 'occupation')
        4. Proportion column name

        The method:
        - Ensures categorical consistency with main data
        - Pivots data to create age × stratification matrices
        - Stores as xarray DataArrays with named coordinates
        - Tracks stratification variables in dataset attributes

        Side Effects
        ------------
        Modifies self.ds by:
        - Adding 'pop_prop_{var_name}' DataArrays for each stratification
        - Setting ds.attrs['grp_vars'] dictionary with category lists

        Notes
        -----
        This method is called automatically by load() if pop_prop
        was specified during initialization.

        Examples
        --------
        >>> # After initialization with pop_prop
        >>> loader.load()
        >>> loader.ds['pop_prop_gender']
        <xarray.DataArray 'pop_prop_gender' (pop_prop_gender: 2, age: 76)>
        array([[0.49, 0.48, ...],  # Female proportions by age
               [0.51, 0.52, ...]])  # Male proportions by age
        """
        self.ds.attrs["grp_vars"] = dict()
        for q in self.pop_prop:
            df, age_col, var_name, prop_name = q
            df = df.rename({age_col: "age"}, axis=1)
            df[var_name] = pd.Categorical(
                df[var_name],
                categories=self.df_full[var_name].cat.categories,
                ordered=True,
            )

            pivoted = df.sort_values(var_name).pivot(
                columns="age", index=var_name, values=prop_name
            )

            arr = xr.DataArray(
                data=pivoted.to_numpy(),
                coords={
                    "pop_prop_" + var_name: pivoted.index.to_list(),
                    "age": pivoted.columns.to_list(),
                },
                dims=["pop_prop_" + var_name, "age"],
                name="pop_prop_" + var_name,
            )
            self.ds["pop_prop_" + var_name] = arr
            self.ds.attrs["grp_vars"][var_name] = pivoted.index.to_list()

    def load(self) -> xr.Dataset:
        """
        Load and transform data into an xarray Dataset for model fitting.

        This is the main method that transforms raw contact survey data into a
        structured xarray Dataset containing:
        - Contact counts stratified by age and grouping variables
        - Participant counts (N)
        - Population size (P)
        - Group contact offset (S) accounting for household/group contacts
        - Optional population proportion arrays for stratified analysis

        The method performs several key transformations:
        1. Aggregates participant counts by age and grouping variables
        2. Computes group contact offsets to adjust for household contacts
        3. Creates contact count matrix across all age combinations
        4. Builds full Cartesian product of all stratification variables
        5. Merges aggregated data with full grid (zero-filling missing cells)
        6. Converts to xarray Dataset with named coordinates

        Returns
        -------
        xr.Dataset
            Xarray Dataset with the following variables:
            - y : int array of contact counts
            - log_N : log of participant counts
            - log_P : log of population sizes by age
            - log_S : log of group contact offset factors
            - aid : participant age indices
            - bid (or cid) : contact age indices (or age group codes)

            Additional variables may include stratification dimensions
            (e.g., gender, occupation) and population proportion arrays.

        Notes
        -----
        Group Contact Offset (S):
        The offset S adjusts for group contacts (e.g., "household members")
        that inflate contact counts. Different adjustments are applied:
        - Children (5-18): Assumes group contacts are with other children
        - Adults (18+): Assumes group contacts are random across population
        - S = 1 - z/(z+y) for children
        - S = 1 - z/(z+y)/(max_age - min_age + 1) for adults

        Where:
        - z = number of group contacts
        - y = number of individual contacts

        The method automatically handles:
        - Zero-filling for unobserved age combinations
        - Categorical variable consistency
        - Age group interval expansions for coarse age data
        - Optional repeat effect variables

        Examples
        --------
        >>> loader = BaseLoader(data, pop, col_map)
        >>> ds = loader.load()
        >>> ds.y.shape
        (12345,)  # Number of unique strata
        >>> ds.coords["age"]
        <xarray.DataArray 'age' (age: 76)>
        array([ 0,  1,  2, ..., 73, 74, 75])
        """
        # [Do] Calculate the number of participants stratified by age and other grouping variables
        grp_vars_n = [self.col_map.age_part]
        if self.col_map.grp_vars_part:
            grp_vars_n += self.col_map.grp_vars_part
        df_n = (
            self.data.groupby(grp_vars_n, observed=False)
            .agg(N=(self.col_map.id_var, "nunique"))
            .reset_index()
        )
        self.df_n = df_n

        # [Do] Calculate group contact offsets
        df_z = (
            self.data[[self.col_map.id_var] + grp_vars_n + [self.col_map.z]]
            .drop_duplicates()
            .groupby(grp_vars_n, observed=True)["z"]
            .sum()
            .reset_index()
        )
        df_yz = (
            self.data[[self.col_map.id_var] + grp_vars_n + [self.col_map.y]]
            .drop_duplicates()
            .groupby(grp_vars_n, observed=True)["y"]
            .sum()
            .reset_index()
        )
        df_S = df_yz.merge(df_z, on=grp_vars_n, how="left")

        # Assume at least one contact if there is a group contact to avoid numerical issues
        # mask = (df_S[self.col_map.z] > 0) & (df_S['y'] == 0)
        # df_S['y'] = np.where(mask, 1, df_S['y'])

        # ===== Calculate group contact offset S =====
        # For school age children (5-18) assume contacts are with other children
        mask = (
            (df_S[self.col_map.age_part] >= 5)
            & (df_S[self.col_map.age_part] <= 18)
            & (df_S[self.col_map.z] + df_S[self.col_map.y] > 0)
        )
        df_S["S"] = np.where(
            mask,
            1 - df_S[self.col_map.z] / (df_S[self.col_map.z] + df_S[self.col_map.y]),
            1.0,
        )

        # For adults (18+) assume contacts are random across population
        mask = (df_S[self.col_map.age_part] > 18) & (
            df_S[self.col_map.z] + df_S[self.col_map.y] > 0
        )
        df_S["S"] = np.where(
            mask,
            1
            - df_S[self.col_map.z]
            / (df_S[self.col_map.z] + df_S[self.col_map.y])
            / (self.max_age - self.min_age + 1),
            1.0,
        )
        df_S = df_S.drop(columns=[self.col_map.z, self.col_map.y])
        self.df_S = df_S

        # [Do] Calculate the number of contacts stratified by age and other grouping variables
        grp_vars = self.col_map.age_vars()
        if self.col_map.grp_vars_part:
            grp_vars += self.col_map.grp_vars_part
        if self.col_map.grp_vars_cnt:
            grp_vars += self.col_map.grp_vars_cnt

        df_y = (
            self.data.groupby(grp_vars, observed=False)
            .agg({self.col_map.y: "sum"})
            .reset_index()
        )
        self.grp_vars = grp_vars
        self.df_y = df_y

        # [Do] Create a full grid of all combinations of the grouping variables via a cartesian product
        unique_coords = {var: self.data[var].unique() for var in grp_vars}
        unique_coords[self.col_map.age_part] = np.arange(
            self.min_age, self.max_age + 1, dtype=int
        )

        if self.col_map.age_cnt:
            unique_coords[self.col_map.age_cnt] = np.arange(
                self.min_age, self.max_age + 1, dtype=int
            )
        elif self.col_map.age_grp_cnt:
            unique_coords[self.col_map.age_grp_cnt] = self.data[
                self.col_map.age_grp_cnt
            ].cat.categories

        index = pd.MultiIndex.from_product(
            unique_coords.values(), names=unique_coords.keys()
        )
        df_full = pd.DataFrame(list(index), columns=unique_coords.keys())
        if self.col_map.age_grp_cnt:
            # [Do] Restore the original information of the age group column
            df_full[self.col_map.age_grp_cnt] = pd.Categorical(
                df_full[self.col_map.age_grp_cnt],
                categories=self.data[self.col_map.age_grp_cnt].cat.categories,
                ordered=True,
            )

        # [Do] Merge the full grid with the contact and participant data
        df_full = pd.merge(df_full, df_y, on=grp_vars, how="left")
        df_full = pd.merge(df_full, df_n, on=grp_vars_n, how="left")
        df_full = pd.merge(df_full, df_S, on=grp_vars_n, how="left")

        # [Do] Finalise the data
        df_full = df_full.dropna(subset=["N"])
        df_full = df_full[df_full["N"] > 0]
        df_full["S"] = df_full["S"].fillna(1.0)
        df_full["log_S"] = np.where(df_full["S"] > 0, np.log(df_full["S"]), 0.0)
        df_full["y"] = df_full["y"].fillna(0)

        # [Do] Create a xarray dataset
        self.df_full = df_full

        self.ds = xr.Dataset(
            {
                "y": ("index", df_full["y"].astype(int).to_numpy()),
                "log_N": ("index", jnp.log(df_full["N"].to_numpy())),
                "log_P": ("age", jnp.log(self.pop[self.col_map.size_pop].to_numpy())),
                "log_S": ("index", jnp.array(df_full["log_S"].to_numpy())),
                "aid": ("index", df_full[self.col_map.age_part].to_numpy()),
            },
            coords={
                "index": df_full.index.to_numpy(),
                "age": ("age", np.arange(self.min_age, self.max_age + 1, dtype=int)),
            },
        )

        if self.col_map.age_cnt:
            self.ds["bid"] = ("index", df_full[self.col_map.age_cnt].to_numpy())
        elif self.col_map.age_grp_cnt:
            self.ds["cid"] = (
                "index",
                df_full[self.col_map.age_grp_cnt].cat.codes.to_numpy(),
            )

            # [Do] Create indices for age aggregation
            aid_exp, bid_pad = make_idarrs_for_intervals(
                df_full, self.col_map.age_grp_cnt, self.ds["aid"].to_numpy()
            )
            self.ds["aid_exp"] = (["index", "max_int_length"], aid_exp)
            self.ds["bid_pad"] = (["index", "max_int_length"], bid_pad)

        for var in self.col_map.grp_vars_part:
            if (
                var != self.col_map.repeat_part
            ):  # Exclude repeat_part from stratification variables
                self.df_full[var] = self.df_full[var].astype("category")
                self.ds[var] = xr.DataArray(
                    data=self.df_full[var],
                    dims="index",
                    coords={"index": self.ds.coords["index"]},
                )

        # If repeat effects are specified
        if self.col_map.repeat_part is not None:
            self.ds["rid"] = xr.DataArray(
                data=self.df_full[self.col_map.repeat_part].astype(int).to_numpy(),
                dims="index",
                coords={"index": self.ds.coords["index"]},
            )

        if hasattr(self, "pop_prop"):
            self._load_pop_proportions()
        return self.ds


class DataLoader(BaseLoader):
    """
    Prepare contact survey data for Bayesian contact matrix estimation.

    This class handles the complete data preparation pipeline for contact matrix
    models, starting from separate participant and contact dataframes. It:
    1. Validates participant, contact, and population dataframes
    2. Merges contact and participant data
    3. Transforms data into xarray Dataset format via BaseLoader

    The DataLoader is the primary entry point for users working with standard
    contact survey data where participants and contacts are stored in separate
    dataframes (e.g., CoMix, POLYMOD surveys).

    Parameters
    ----------
    part : pd.DataFrame
        Participant-level data containing one row per survey participant.
        Must include participant ID, age, and optional grouping variables.
    cnt : pd.DataFrame
        Contact-level data containing one row per reported contact.
        Must include participant ID (linking to part), contact age or age group,
        and optional contact-specific grouping variables.
    pop : pd.DataFrame
        Population age distribution data with age and population size columns.
    col_map : CoordToColumns
        Column mapping object specifying which columns in the dataframes
        correspond to required variables (IDs, ages, grouping variables).
    pop_prop : Union[PopulationProportion, List[PopulationProportion], None], optional
        Population proportion specification(s) for demographic adjustment.
        Can be either:
        - A single PopulationProportion object
        - A list of PopulationProportion objects for multiple stratifications
        If None, no stratified population proportions are used.

        Example with single stratification:
            pop_prop = PopulationProportion.from_counts(
                data=df_gender, age_col='age', stratify_by='gender', count_col='N'
            )
            DataLoader(..., pop_prop=pop_prop)

        Example with multiple stratifications:
            pop_prop_gender = PopulationProportion.from_counts(...)
            pop_prop_region = PopulationProportion.from_counts(...)
            DataLoader(..., pop_prop=[pop_prop_gender, pop_prop_region])

    Attributes
    ----------
    part : pd.DataFrame
        Validated and processed participant dataframe.
    cnt : pd.DataFrame
        Validated and processed contact dataframe.
    data : pd.DataFrame
        Merged participant-contact dataframe passed to BaseLoader.

    Methods
    -------
    _validate_part(part, col_map)
        Validate participant dataframe structure and required columns.
    _validate_cnt(cnt, col_map)
        Validate contact dataframe structure and required columns.
    _validate_pop(pop, col_map)
        Validate population dataframe structure and required columns.
    load()
        Inherited from BaseLoader - transforms data to xarray Dataset.

    Raises
    ------
    KeyError
        If required columns are missing from any input dataframe.
    TypeError
        If pop_prop contains non-PopulationProportion objects.

    Notes
    -----
    - The 'y' column (contact indicator) is auto-filled with 1 if missing from cnt
    - The 'z' column (group contact count) is auto-filled with 0 if missing from part
    - Participant and contact data are merged on the participant ID variable
    - Column suffixes are handled automatically during merge

    Examples
    --------
    >>> from cntmosaic.dataloader import DataLoader, CoordToColumns, PopulationProportion
    >>>
    >>> # Define column mappings
    >>> col_map = CoordToColumns(
    ...     id_var='participant_id',
    ...     age_part='age',
    ...     age_cnt='cnt_age',
    ...     age_pop='age',
    ...     size_pop='population'
    ... )
    >>>
    >>> # Create population proportion (single stratification)
    >>> pop_prop = PopulationProportion.from_counts(
    ...     data=df_gender,
    ...     age_col='age',
    ...     stratify_by='gender',
    ...     count_col='population'
    ... )
    >>>
    >>> # Load data with single stratification
    >>> loader = DataLoader(
    ...     part_df, cnt_df, pop_df, col_map,
    ...     pop_prop=pop_prop
    ... )
    >>> ds = loader.load()
    >>>
    >>> # Load data with multiple stratifications
    >>> pop_prop_region = PopulationProportion.from_counts(...)
    >>> loader = DataLoader(
    ...     part_df, cnt_df, pop_df, col_map,
    ...     pop_prop=[pop_prop, pop_prop_region]
    ... )
    >>>
    >>> # Access contact matrix data
    >>> ds.y  # Contact counts
    >>> ds.log_N  # Log participant counts
    >>> ds.pop_prop_gender  # Stratified population proportions by gender
    """

    def __init__(
        self,
        part: pd.DataFrame,
        cnt: pd.DataFrame,
        pop: pd.DataFrame,
        col_map: CoordToColumns,
        pop_prop: Union[PopulationProportion, List[PopulationProportion], None] = None,
    ) -> None:
        self._validate_part(part, col_map)
        self._validate_cnt(cnt, col_map)
        self._validate_pop(pop, col_map)
        data = pd.merge(self.cnt, self.part, on=col_map.id_var, suffixes=("", "_cnt"))
        super().__init__(data, pop, col_map, pop_prop)

    def _validate_part(self, part: pd.DataFrame, col_map: CoordToColumns) -> None:
        """
        Validate participant dataframe structure and required columns.

        Ensures the participant dataframe contains all necessary columns for
        analysis, including participant ID, age, and any specified grouping
        variables. Automatically adds missing 'z' column (group contact count)
        initialized to 0 if not present.

        Parameters
        ----------
        part : pd.DataFrame
            Participant dataframe to validate.
        col_map : CoordToColumns
            Column mapping specification.

        Raises
        ------
        KeyError
            If required columns (id_var, age_part, or grp_vars_part) are missing.

        Side Effects
        ------------
        Sets self.part to validated copy of input dataframe with 'z' column added
        if it was missing.

        Notes
        -----
        The 'z' column represents group/household contact counts. If not present
        in the data, it's initialized to 0, indicating no group contacts reported.
        """
        # [Check] Ensure all necessary columns are present
        if col_map.id_var not in part.columns:
            raise KeyError(
                f"Missing participant ID column '{col_map.id_var}' in participants dataframe.\n"
                f"Available columns: {list(part.columns)}"
            )

        if col_map.age_part not in part.columns:
            raise KeyError(
                f"Missing participant age column '{col_map.age_part}' in participants dataframe.\n"
                f"Available columns: {list(part.columns)}"
            )

        if col_map.grp_vars_part:
            missing = [col for col in col_map.grp_vars_part if col not in part.columns]
            if missing:
                raise KeyError(
                    f"Missing grouping variable columns {missing} in participants dataframe.\n"
                    f"Available columns: {list(part.columns)}"
                )

        # [Check] If the column z is present in part.
        # If not, add it as a column with value 0.
        if col_map.z not in part.columns:
            part = part.copy()
            part[col_map.z] = 0

        self.part: pd.DataFrame = part.copy()

    def _validate_cnt(self, cnt: pd.DataFrame, col_map: CoordToColumns) -> None:
        """
        Validate contact dataframe structure and required columns.

        Ensures the contact dataframe contains all necessary columns, including
        participant ID (for merging), contact age information (either exact age
        or age group), and any specified contact-level grouping variables.
        Automatically adds missing 'y' column (contact indicator) initialized
        to 1 if not present.

        Parameters
        ----------
        cnt : pd.DataFrame
            Contact dataframe to validate.
        col_map : CoordToColumns
            Column mapping specification.

        Raises
        ------
        KeyError
            If required columns (id_var, age_cnt or age_grp_cnt, grp_vars_cnt)
            are missing.

        Side Effects
        ------------
        Sets self.cnt to validated copy of input dataframe with 'y' column added
        if it was missing.

        Notes
        -----
        - Either age_cnt (exact age) or age_grp_cnt (age group) must be present
        - The 'y' column represents contact indicators/counts. If not present,
          it's initialized to 1, indicating one contact per row
        - Contact-level grouping variables (grp_vars_cnt) are optional
        """
        # [Check] Ensure all necessary columns are present
        if col_map.id_var not in cnt.columns:
            raise KeyError(
                f"Missing participant ID column '{col_map.id_var}' in contacts dataframe.\n"
                f"Available columns: {list(cnt.columns)}"
            )

        if col_map.age_cnt:
            if col_map.age_cnt not in cnt.columns:
                raise KeyError(
                    f"Missing contact age column '{col_map.age_cnt}' in contacts dataframe.\n"
                    f"Available columns: {list(cnt.columns)}"
                )

        if col_map.age_grp_cnt:
            if col_map.age_grp_cnt not in cnt.columns:
                raise KeyError(
                    f"Missing contact age group column '{col_map.age_grp_cnt}' in contacts dataframe.\n"
                    f"Available columns: {list(cnt.columns)}"
                )

        if col_map.grp_vars_cnt:
            missing = [col for col in col_map.grp_vars_cnt if col not in cnt.columns]
            if missing:
                raise KeyError(
                    f"Missing contact grouping variable columns {missing} in contacts dataframe.\n"
                    f"Available columns: {list(cnt.columns)}"
                )

        # [Check] If the column y is present in cnt. If not, add it as a column with value 1
        if col_map.y not in cnt.columns:
            cnt = cnt.copy()
            cnt[col_map.y] = 1

        self.cnt: pd.DataFrame = cnt.copy()

    def _validate_pop(self, pop: pd.DataFrame, col_map: CoordToColumns) -> None:
        """
        Validate population dataframe structure and required columns.

        Ensures the population dataframe contains age and population size columns
        needed for adjusting contact rates to population-level contact matrices.

        Parameters
        ----------
        pop : pd.DataFrame
            Population dataframe to validate.
        col_map : CoordToColumns
            Column mapping specification.

        Raises
        ------
        KeyError
            If required columns (age_pop, size_pop) are missing.

        Notes
        -----
        The population dataframe should contain:
        - age_pop: Age values corresponding to population counts
        - size_pop: Population size (counts) for each age

        Population proportions are computed automatically by BaseLoader.
        """
        if col_map.age_pop not in pop.columns:
            raise KeyError(
                f"Missing population age column '{col_map.age_pop}' in population dataframe.\n"
                f"Available columns: {list(pop.columns)}"
            )
        if col_map.size_pop not in pop.columns:
            raise KeyError(
                f"Missing population size column '{col_map.size_pop}' in population dataframe.\n"
                f"Available columns: {list(pop.columns)}"
            )
