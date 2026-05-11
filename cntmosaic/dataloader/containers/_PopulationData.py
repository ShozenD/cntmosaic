from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._population_preprocessing import preprocess_population_data
from ._population_validation import validate_population_data


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
    data : pd.DataFrame
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

    Attributes
    ----------
    df_pop : pd.DataFrame
        The validated and preprocessed population DataFrame with standardized
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

        # Normalize strat_var_cols to list format for consistent handling
        if isinstance(self.strat_var_cols, str):
            object.__setattr__(self, "strat_var_cols", [self.strat_var_cols])
        elif self.strat_var_cols is None:
            object.__setattr__(self, "strat_var_cols", [])

        # Delegate column validation, NaN removal, dtype coercion, aggregation, and renaming
        object.__setattr__(
            self,
            "df_pop",
            preprocess_population_data(
                self.df_pop,
                self.age_col,
                self.size_col,
                self.age_grp_col,
                self.strat_var_cols,  # type: ignore
            ),
        )

        # Perform domain validation on the cleaned data
        self.validate()

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
        validate_population_data(
            self.df_pop,
            self.age_col,
            self.size_col,
            self.strat_var_cols,  # type: ignore
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
        return self.data["age"].nunique()

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
        return float(self.data["P"].sum())

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
        ages = self.data["age"]
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
        return self.strat_var_cols if self.strat_var_cols else []  # type: ignore

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
        if self.strat_var_cols is None:
            return schema

        else:
            for var in self.strat_var_cols:
                if var in self.data.columns:
                    categories = self.data[var].cat.categories.tolist()
                    codes = sorted(self.data[var].cat.codes.unique().tolist())
                    schema[var] = {"categories": categories, "codes": codes}

            return schema

    def _build_pop_matrix(
        self,
        vars_list: List[str],
        strat_labels_for_vars: Optional[List] = None,
    ) -> Tuple[NDArray, List]:
        """
        Build ordered population size matrix and corresponding labels for given stratification variable(s).

        Parameters
        ----------
        vars_list : List[str]
            List of stratification variable column names to combine.
        strat_labels_for_vars : List, optional
            Optional ordering of labels to reindex the pivot (preserves order).

        Returns
        -------
        P_matrix : NDArray
            Array shape (n_strata, A) giving population sizes by stratum and age.
        labels : List
            Ordered list of stratum labels corresponding to rows of P_matrix.
        """
        df = self.data[["age"] + vars_list + ["P"]]
        sort_vars = ["age"] + vars_list
        df = df.sort_values(by=sort_vars)

        if len(vars_list) == 1:
            var = vars_list[0]
            df_pivot = (
                df.groupby(["age", var], observed=False)["P"]
                .sum()
                .reset_index()
                .pivot(index=var, columns="age", values="P")
            )
            if strat_labels_for_vars is not None:
                if any("->" in str(l) for l in strat_labels_for_vars):
                    strat_labels_for_vars = list(
                        dict.fromkeys(l.split("->")[0] for l in strat_labels_for_vars)
                    )
                df_pivot = df_pivot.reindex(strat_labels_for_vars)
                labels = strat_labels_for_vars
            else:
                labels = df_pivot.index.tolist()
        else:
            df = df.copy()
            df["strat_combined"] = df[vars_list].apply(
                lambda row: "_".join(row.astype(str)), axis=1
            )
            df_pivot = (
                df.groupby(["age", "strat_combined"], observed=False)["P"]
                .sum()
                .reset_index()
                .pivot(index="strat_combined", columns="age", values="P")
            )
            if strat_labels_for_vars is not None:
                df_pivot = df_pivot.reindex(strat_labels_for_vars)
                labels = strat_labels_for_vars
            else:
                labels = df["strat_combined"].unique().tolist()

        P_matrix = df_pivot.values
        return P_matrix, labels

    def get_stratified_pop_sizes(
        self, strat_var_cols: Optional[Union[List[str], str]] = None
    ) -> Dict[str, NDArray]:
        """
        Compute stratified population sizes P^{s}_{a} as a dictionary keyed by stratum label.

        This method computes the population sizes by age for each stratum defined by the
        stratification variable(s). Unlike ``compute_demopty`` in ``StratificationData``
        which outputs normalised demographic opportunity (proportions), this method returns
        the raw (unnormalised) population sizes.

        Parameters
        ----------
        strat_var_cols : str or List[str], optional
            Stratification variable column name(s) to group by. If None, uses the
            ``strat_var_cols`` specified at construction time. This argument is useful
            in the PARTIAL stratification scenario where the user may not have specified
            ``strat_var_cols`` on the ``PopulationData`` object.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping stratum labels to population size arrays of shape ``(A,)``.
            For a single variable (e.g. ``'gender'`` with categories ``['M', 'F']``),
            keys are ``'M'`` and ``'F'``.
            For multiple variables (e.g. ``['gender', 'region']``), keys are composite
            labels like ``'M_North'``, ``'M_South'``, ``'F_North'``, ``'F_South'``.

        Raises
        ------
        ValueError
            If no stratification variables are provided (neither at construction nor via argument).
        KeyError
            If specified ``strat_var_cols`` are not present in the population data.

        Examples
        --------
        >>> pop_data = PopulationData(df, age_col='age', size_col='count', strat_var_cols='gender')
        >>> pop_sizes = pop_data.get_stratified_pop_sizes()
        >>> pop_sizes['M'].shape
        (86,)
        >>> # Or specify strat_var_cols explicitly (e.g. for PARTIAL mode)
        >>> pop_sizes = pop_data.get_stratified_pop_sizes(strat_var_cols='gender')
        """
        # Resolve strat_var_cols
        if strat_var_cols is not None:
            if isinstance(strat_var_cols, str):
                strat_var_cols = [strat_var_cols]
        else:
            strat_var_cols = self.strat_var_cols

        if not strat_var_cols:
            raise ValueError(
                "No stratification variables provided. "
                "Specify strat_var_cols either at construction or as an argument."
            )

        # Validate columns exist
        missing = [v for v in strat_var_cols if v not in self.data.columns]
        if missing:
            raise KeyError(
                f"Stratification columns not found in population data: {missing}\n"
                f"Available columns: {list(self.data.columns)}"
            )

        P_matrix, labels = self._build_pop_matrix(strat_var_cols)

        return {str(lab): P_matrix[i] for i, lab in enumerate(labels)}

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
