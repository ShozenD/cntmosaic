import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class PopulationData:
    """
    Validated population data container for contact matrix estimation.

    This class provides a type-safe, validated wrapper around population DataFrames,
    ensuring data integrity before use in contact matrix estimation models. It performs
    comprehensive validation of required columns, data types, and value ranges, along
    with automatic preprocessing including aggregation by stratification variables.

    Attributes
    ----------
    df_pop : pd.DataFrame
        DataFrame containing population age distribution data. Each row represents
        a population count for a specific age (or age group) and optional stratification.
        Must contain columns specified by age_col, size_col, and optionally age_grp_col
        and strat_var_cols.
        Note: The DataFrame is automatically preprocessed (copied, cleaned, aggregated).
    age_col : str
        Name of the column containing population ages as integers.
        Ages should be non-negative integers.
        In the processed data, this column will be renamed to "age".
        This field is required.
    size_col : str
        Name of the column containing population sizes (counts or proportions).
        Should contain non-negative numeric values.
        In the processed data, this column will be renamed to "P".
    age_grp_col : Optional[str], default=None
        Name of the column containing population age groups.
        Should be pd.IntervalIndex or categorical age groups.
        If specified, will be preserved as "age_grp_pop" in processed data.
        This is optional and can be used alongside age_col to preserve grouping information.
    strat_var_cols : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) for population subgroups.
        Can be a single string or list of strings. Examples: 'gender', ['gender', 'region'].

        **Important**: If multiple variables are specified (e.g., ['gender', 'region']),
        they will be **combined into a single composite stratification variable** by DataLoader.
        For example, ['gender', 'region'] with categories ['M', 'F'] and ['North', 'South']
        will create a combined variable with categories: ['M_North', 'M_South', 'F_North', 'F_South'].

        **Consistency requirement**: The same stratification variables must be specified across:
        - ParticipantData (required if stratifying)
        - ContactData (required if FULL stratification mode)
        - PopulationData (required if using stratified population data)
        - StratificationData (required, one object per composite stratification)

        Population sizes will be aggregated (summed) by the composite stratification variable along with age.

    Properties
    ----------
    data : pd.DataFrame
        Returns the validated and preprocessed population DataFrame with standardized
        column names ("age", "P", and optional grouping variables).
    n_ages : int
        Returns the number of unique ages in the population data.
    total : float
        Returns the total population size (sum of all P values).
    age_range : Tuple[int, int]
        Returns (min_age, max_age) tuple.
    strat_vars : List[str]
        Returns list of stratification variable names (empty list if none).

    Methods
    -------
    validate()
        Performs comprehensive validation of the population data.
    get_strat_vars(suffix=False)
        Returns list of stratification variable names, optionally with '_pop' suffix.
    get_strat_var_schema()
        Returns dictionary mapping stratification variables to their categories and codes.
    summary()
        Returns a dictionary with summary statistics about the population data.

    Raises
    ------
    ValueError
        If age_col is not specified (required field).
        If age or size values are invalid (negative or non-numeric).
        If DataFrame becomes empty after removing missing values.
    KeyError
        If required columns (age_col, size_col, age_grp_col, strat_var_cols) are missing
        from the input DataFrame.
    TypeError
        If df_pop is not a pandas DataFrame.

    Examples
    --------
    >>> # Basic usage with individual ages
    >>> df = pd.DataFrame({
    ...     'age': [0, 1, 2, 3, 4],
    ...     'population': [1000, 1050, 1100, 1080, 1120]
    ... })
    >>> pop_data = PopulationData(
    ...     df_pop=df,
    ...     age_col='age',
    ...     size_col='population'
    ... )
    >>> pop_data.total
    5350
    >>> pop_data.n_ages
    5
    >>> # Processed data has standardized columns
    >>> list(pop_data.data.columns)
    ['age', 'P']
    >>>
    >>> # With stratification (automatically aggregates)
    >>> df = pd.DataFrame({
    ...     'age': [0, 0, 1, 1, 2, 2],
    ...     'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    ...     'count': [510, 490, 530, 520, 550, 550]
    ... })
    >>> pop_data = PopulationData(
    ...     df_pop=df,
    ...     age_col='age',
    ...     size_col='count',
    ...     strat_var_cols='gender'
    ... )
    >>> pop_data.data
       age gender     P
    0    0      F   490
    1    0      M   510
    2    1      F   520
    3    1      M   530
    4    2      F   550
    5    2      M   550
    >>> # Total population is sum across all strata
    >>> pop_data.total
    3150
    >>> # Access stratification variables
    >>> pop_data.strat_vars
    ['gender']
    >>>
    >>> # With age groups
    >>> df = pd.DataFrame({
    ...     'age': [0, 5, 10],
    ...     'age_group': pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
    ...     'population': [5000, 4800, 4600]
    ... })
    >>> pop_data = PopulationData(
    ...     df_pop=df,
    ...     age_col='age',
    ...     size_col='population',
    ...     age_grp_col='age_group'
    ... )
    >>> 'age_grp_pop' in pop_data.data.columns
    True
    >>> pop_data.data['age_grp_pop'].dtype
    dtype('interval[int64, right)')

    Notes
    -----
    Preprocessing Steps (Automatic):
    - Creates a copy of the input DataFrame to avoid side effects
    - Drops rows with missing values in required columns
    - Validates age and size values are non-negative
    - Aggregates (sums) population sizes by age and stratification variables
    - Renames columns to standard names: 'age' and 'P'
    - Preserves age_grp_col as 'age_grp_pop' if specified
    - Sorts by age (and stratification variables if present)
    - Validates all data constraints after preprocessing

    Validation Checks:
    - age_col is required (must be specified)
    - Age values must be non-negative and numeric
    - Population size values must be non-negative and numeric
    - No missing values in required columns (removed during preprocessing)
    - At least one age must remain after preprocessing

    Aggregation Behavior:
    - If strat_var_cols specified: Population sizes are summed within each (age, strat_var) combination
    - If no strat_var_cols: Population sizes are summed within each age
    - This allows input data with multiple rows per age to be properly aggregated
    - Example: Regional population data can be summed to get national totals

    Warnings:
    - UserWarning if rows are dropped due to missing values
    - UserWarning if population sizes are aggregated (multiple rows per age group)
    - UserWarning if any age groups have zero population size

    See Also
    --------
    ParticipantData : Validated container for participant survey data
    ContactData : Validated container for contact survey data
    StratificationData : Validated container for stratification proportions
    """

    df_pop: pd.DataFrame
    age_col: str
    size_col: str
    age_grp_col: Optional[str] = None
    strat_var_cols: Optional[Union[List[str], str]] = None

    def __post_init__(self) -> None:
        """
        Post-initialization processing and validation.

        Automatically called after dataclass initialization to:
        1. Validate input types
        2. Normalize strat_var_cols to list format
        3. Validate mutual exclusivity constraints (if age_grp_col and age_col both given)
        4. Preprocess the DataFrame (copy, clean, aggregate, rename)
        5. Perform comprehensive data validation

        Raises
        ------
        TypeError
            If df_pop is not a pandas DataFrame.
        ValueError
            If age_col is not specified (required).
            If DataFrame becomes empty after preprocessing.
        """
        # Type validation
        if not isinstance(self.df_pop, pd.DataFrame):
            raise TypeError(
                f"df_pop must be a pandas DataFrame, got {type(self.df_pop).__name__}"
            )

        # age_col is required
        if self.age_col is None:
            raise ValueError(
                "Must specify 'age_col' containing population ages.\n"
                "This is required for matching population data with survey age ranges."
            )

        object_cols = self.df_pop.select_dtypes(include="object").columns.tolist()
        for col in object_cols:
            if not isinstance(self.df_pop[col].dtype, pd.CategoricalDtype):
                warnings.warn(
                    f"Converting '{col}' to categorical dtype.",
                    UserWarning,
                    stacklevel=3,
                )
                self.df_pop[col] = self.df_pop[col].astype("category")

        # Normalize strat_var_cols to list format for consistent handling
        if isinstance(self.strat_var_cols, str):
            object.__setattr__(self, "strat_var_cols", [self.strat_var_cols])
        elif self.strat_var_cols is None:
            object.__setattr__(self, "strat_var_cols", [])

        # Preprocess the DataFrame
        object.__setattr__(self, "df_pop", self._preprocess())

        # Perform comprehensive validation
        self.validate()

    def _preprocess(self) -> pd.DataFrame:
        """
        Preprocess population DataFrame for modeling.

        Performs the following preprocessing steps:
        1. Creates a copy to avoid modifying the original DataFrame
        2. Validates that required columns exist
        3. Drops rows with missing values in required columns
        4. Validates age and size values are non-negative
        5. Aggregates population sizes by age (and stratification variables if present)
        6. Renames columns to standard names: age_col → "age", size_col → "P"
        7. Preserves age_grp_col as "age_grp_pop" if specified
        8. Sorts by age (and stratification variables)

        Returns
        -------
        pd.DataFrame
            Preprocessed population DataFrame with standardized column names.

        Raises
        ------
        KeyError
            If required columns are missing from the DataFrame.
        ValueError
            If DataFrame becomes empty after removing missing values.
            If age or size values are negative.

        Warnings
        --------
        UserWarning
            If rows are dropped due to missing values.
            If population data is aggregated (multiple rows per age group).

        Notes
        -----
        This method follows preprocessing patterns from DataLoader to ensure
        consistency with the package's data handling conventions.
        """
        # Step 1: Create a copy to avoid side effects
        df = self.df_pop.copy()

        # Step 2: Check that required columns exist BEFORE trying to access them
        if self.age_col not in df.columns:
            raise KeyError(
                f"Missing population age column '{self.age_col}' in DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )

        if self.size_col not in df.columns:
            raise KeyError(
                f"Missing population size column '{self.size_col}' in DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Check age_grp_col if specified
        if self.age_grp_col and self.age_grp_col not in df.columns:
            raise KeyError(
                f"Missing population age group column '{self.age_grp_col}' in DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Check stratification variables exist
        if self.strat_var_cols:
            missing_vars = [var for var in self.strat_var_cols if var not in df.columns]
            if missing_vars:
                raise KeyError(
                    f"Missing population stratification variable(s) {missing_vars} in DataFrame.\n"
                    f"Available columns: {list(df.columns)}"
                )

        # Step 3: Identify required columns and check for missing values
        required_cols = [self.age_col, self.size_col]
        if self.age_grp_col:
            required_cols.append(self.age_grp_col)
        if self.strat_var_cols:
            required_cols.extend(self.strat_var_cols)

        # Check for missing values before dropping
        n_rows_before = len(df)
        missing_counts = df[required_cols].isnull().sum()
        has_missing = missing_counts.sum() > 0

        # Drop rows with missing values in required columns
        df = df[required_cols].dropna().copy()

        # Warn if rows were dropped
        if has_missing:
            n_rows_after = len(df)
            n_dropped = n_rows_before - n_rows_after
            cols_with_missing = missing_counts[missing_counts > 0]
            warnings.warn(
                f"Dropped {n_dropped} row(s) with missing values in required columns.\n"
                f"Missing value counts by column: {cols_with_missing.to_dict()}\n"
                f"Remaining population records: {n_rows_after}",
                UserWarning,
                stacklevel=3,
            )

        # Check if DataFrame is empty after dropping missing values
        if df.empty:
            raise ValueError(
                "DataFrame is empty after removing rows with missing values.\n"
                "Check for excessive NaN values in required columns:\n"
                f"Required columns: {required_cols}\n"
                f"Missing value counts: {missing_counts.to_dict()}"
            )

        # Step 4: Validate age and size values are non-negative
        ages = df[self.age_col]
        if not pd.api.types.is_numeric_dtype(ages):
            raise ValueError(
                f"Population age column '{self.age_col}' must contain numeric values.\n"
                f"Current dtype: {ages.dtype}\n"
                f"Hint: Convert age to integer or float type."
            )

        if (ages < 0).any():
            negative_indices = df[ages < 0].index[:5].tolist()
            raise ValueError(
                f"Population age column '{self.age_col}' contains negative values.\n"
                f"Ages must be non-negative. Rows with negative ages: {negative_indices}\n"
                f"Values: {ages[ages < 0].head().tolist()}"
            )

        sizes = df[self.size_col]
        if not pd.api.types.is_numeric_dtype(sizes):
            raise ValueError(
                f"Population size column '{self.size_col}' must contain numeric values.\n"
                f"Current dtype: {sizes.dtype}\n"
                f"Hint: Convert size to numeric type."
            )

        if (sizes < 0).any():
            negative_indices = df[sizes < 0].index[:5].tolist()
            raise ValueError(
                f"Population size column '{self.size_col}' contains negative values.\n"
                f"Population sizes must be non-negative. Rows with negative sizes: {negative_indices}\n"
                f"Values: {sizes[sizes < 0].head().tolist()}"
            )

        # Step 5: Aggregate population sizes by age and stratification variables
        groupby_cols = [self.age_col]
        if self.age_grp_col:
            groupby_cols.append(self.age_grp_col)
        if self.strat_var_cols:
            groupby_cols.extend(self.strat_var_cols)

        # Check if aggregation will occur
        n_unique_groups = df.groupby(groupby_cols, observed=False).ngroups
        if n_unique_groups < len(df):
            n_aggregated = len(df) - n_unique_groups
            warnings.warn(
                f"Aggregating population data: {len(df)} rows → {n_unique_groups} unique age groups.\n"
                f"Population sizes are summed within each age{' and stratification' if self.strat_var_cols else ''} group.\n"
                f"Number of rows aggregated: {n_aggregated}",
                UserWarning,
                stacklevel=3,
            )

        # Aggregate by summing population sizes within each group
        df = df.groupby(groupby_cols, as_index=False, observed=False)[
            self.size_col
        ].sum()

        # Step 6: Rename columns to standard names
        rename_map = {self.age_col: "age", self.size_col: "P"}
        if self.age_grp_col:
            rename_map[self.age_grp_col] = "age_grp_pop"

        df = df.rename(columns=rename_map)

        # Step 7: Sort by age (and stratification variables if present)
        sort_cols = ["age"]
        if self.strat_var_cols:
            sort_cols.extend(self.strat_var_cols)

        df = df.sort_values(sort_cols).reset_index(drop=True)

        return df

    def validate(self) -> None:
        """
        Perform comprehensive validation of population data.

        Validation checks:
        1. Required columns exist in the processed DataFrame
        2. Age column contains valid non-negative numeric values
        3. Population size column contains non-negative numeric values
        4. At least one age group exists

        Note: Missing value checks and aggregation are handled during preprocessing
        in _preprocess(), so this method assumes clean, aggregated data.

        Raises
        ------
        KeyError
            If required columns are missing from the DataFrame.
        ValueError
            If age or size values are invalid (negative or non-numeric).
            If no ages remain after preprocessing.

        Examples
        --------
        >>> pop_data = PopulationData(df, age_col='age', size_col='population')
        >>> # Validation happens automatically, but can be called explicitly:
        >>> pop_data.validate()
        """
        # Check 1: Validate required columns exist (standardized names after preprocessing)
        if "age" not in self.df_pop.columns:
            raise KeyError(
                f"Missing 'age' column in processed population DataFrame.\n"
                f"Available columns: {list(self.df_pop.columns)}\n"
                f"This should not happen - please report this bug."
            )

        if "P" not in self.df_pop.columns:
            raise KeyError(
                f"Missing 'P' (population size) column in processed population DataFrame.\n"
                f"Available columns: {list(self.df_pop.columns)}\n"
                f"This should not happen - please report this bug."
            )

        # Check 2: Validate stratification variables exist if specified
        if self.strat_var_cols:
            missing_vars = [
                var for var in self.strat_var_cols if var not in self.df_pop.columns
            ]
            if missing_vars:
                raise KeyError(
                    f"Missing population stratification variable(s) {missing_vars} in processed DataFrame.\n"
                    f"Available columns: {list(self.df_pop.columns)}"
                )

        # Check 3: Validate at least one age exists
        if len(self.df_pop) == 0:
            raise ValueError(
                "Population DataFrame is empty after preprocessing.\n"
                "At least one age group with population data is required."
            )

        # Check 4: Validate age values (should already be validated in _preprocess, but double-check)
        ages = self.df_pop["age"]
        if not pd.api.types.is_numeric_dtype(ages):
            raise ValueError(
                f"Population age column must contain numeric values.\n"
                f"Current dtype: {ages.dtype}"
            )

        if (ages < 0).any():
            raise ValueError(
                "Population age column contains negative values after preprocessing.\n"
                "This should not happen - please report this bug."
            )

        # Check 5: Validate population size values
        sizes = self.df_pop["P"]
        if not pd.api.types.is_numeric_dtype(sizes):
            raise ValueError(
                f"Population size column must contain numeric values.\n"
                f"Current dtype: {sizes.dtype}"
            )

        if (sizes < 0).any():
            raise ValueError(
                "Population size column contains negative values after preprocessing.\n"
                "This should not happen - please report this bug."
            )

        # Check 6: Warn if any population sizes are zero
        if (sizes == 0).any():
            n_zero = (sizes == 0).sum()
            zero_ages = self.df_pop[sizes == 0]["age"].head(5).tolist()
            warnings.warn(
                f"Found {n_zero} age group(s) with zero population size.\n"
                f"Ages with zero population: {zero_ages}\n"
                f"This may cause issues in contact matrix estimation.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the validated and preprocessed population DataFrame.

        Returns
        -------
        pd.DataFrame
            The validated population data with standardized column names.

        Examples
        --------
        >>> pop_data = PopulationData(df, age_col='age', size_col='population')
        >>> validated_df = pop_data.data
        >>> list(validated_df.columns)
        ['age', 'P']
        """
        return self.df_pop

    @property
    def n_ages(self) -> int:
        """
        Return the number of unique ages in the population data.

        If stratified by grouping variables, returns the number of unique ages
        (not the number of age × group combinations).

        Returns
        -------
        int
            Number of unique ages.

        Examples
        --------
        >>> pop_data = PopulationData(df, age_col='age', size_col='population')
        >>> pop_data.n_ages
        86  # Ages 0-85
        """
        return self.df_pop["age"].nunique()

    @property
    def total(self) -> float:
        """
        Return the total population size.

        Returns the sum of all population sizes across all ages and stratification
        groups.

        Returns
        -------
        float
            Total population size.

        Examples
        --------
        >>> pop_data = PopulationData(df, age_col='age', size_col='population')
        >>> pop_data.total
        328200000.0
        """
        return float(self.df_pop["P"].sum())

    @property
    def age_range(self) -> Tuple[int, int]:
        """
        Return the age range (min, max) in the population data.

        Returns
        -------
        Tuple[int, int]
            Tuple of (minimum_age, maximum_age).

        Examples
        --------
        >>> pop_data = PopulationData(df, age_col='age', size_col='population')
        >>> pop_data.age_range
        (0, 85)
        """
        ages = self.df_pop["age"]
        return (int(ages.min()), int(ages.max()))

    @property
    def strat_vars(self) -> List[str]:
        """
        Return list of stratification variable names.

        Returns
        -------
        List[str]
            List of stratification variable column names (empty if none).

        Examples
        --------
        >>> pop_data = PopulationData(df, 'age', 'population', strat_var_cols=['gender', 'region'])
        >>> pop_data.strat_vars
        ['gender', 'region']
        """
        return self.strat_var_cols if self.strat_var_cols else []

    def get_strat_vars(self, suffix: bool = False) -> List[str]:
        """
        Return list of stratification variable names, optionally with suffix.

        Parameters
        ----------
        suffix : bool, default=False
            If True, appends '_pop' suffix to each stratification variable name.

        Returns
        -------
        List[str]
            List of stratification variable column names (with optional suffix).

        Examples
        --------
        >>> pop_data = PopulationData(df, 'age', 'population', strat_var_cols=['gender', 'region'])
        >>> pop_data.get_strat_vars(suffix=True)
        ['gender_pop', 'region_pop']
        """
        if not self.strat_var_cols:
            return []
        if suffix:
            return [var + "_pop" for var in self.strat_var_cols]
        else:
            return [var.removesuffix("_pop") for var in self.strat_var_cols]

    def get_strat_var_schema(self) -> Dict[str, Dict[str, List[Union[str, int]]]]:
        """
        Return a dictionary of stratification variables and their unique category and code values.
        The output is to be used in the DataLoader to ensure that the category and codes are consistent
        between the participant, contact, population, and demographic data.

        Returns
        -------
        Dict[str, Dict[str, List[str | int]]]
            A dictionary where keys are the original stratification variable names (without '_cnt' suffix)
            and values are dictionaries with keys 'categories' and 'codes' containing lists of unique category
            values and their corresponding codes.
        """
        schema = {}
        for var in self.strat_var_cols:
            if var in self.df_pop.columns:
                categories = self.df_pop[var].cat.categories.tolist()
                codes = sorted(self.df_pop[var].cat.codes.unique().tolist())
                schema[var] = {"categories": categories, "codes": codes}

        return schema

    def summary(self) -> Dict[str, Any]:
        """
        Return summary statistics about the population data.

        Provides a comprehensive overview including population size, age range,
        and stratification information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - n_ages: Number of unique ages
            - age_range: Tuple of (min_age, max_age)
            - total: Total population size
            - mean_population_per_age: Average population per age group
            - strat_vars: List of stratification variables
            - n_strat_vars: Number of stratification variables
            - is_stratified: Boolean indicating if data is stratified

        Examples
        --------
        >>> pop_data = PopulationData(df, 'age', 'population', strat_var_cols='gender')
        >>> summary = pop_data.summary()
        >>> print(summary)
        {
            'n_ages': 86,
            'age_range': (0, 85),
            'total': 328200000.0,
            'mean_population_per_age': 3816279.07,
            'strat_vars': ['gender'],
            'n_strat_vars': 1,
            'is_stratified': True
        }
        """
        summary_dict = {
            "n_ages": self.n_ages,
            "age_range": self.age_range,
            "total": self.total,
            "mean_population_per_age": self.total / self.n_ages,
            "strat_vars": self.strat_vars,
            "n_strat_vars": len(self.strat_vars),
            "is_stratified": len(self.strat_vars) > 0,
        }

        return summary_dict
