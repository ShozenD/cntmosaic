from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ._participant_preprocessing import preprocess_participant_data


@dataclass
class ParticipantData:
    """
    A container class for participant data from social contact surveys.

    This class wraps participant DataFrames and handles validation, type conversion, and
    column standardization. It automatically preprocesses data and provides convenient
    properties and methods to access participant information.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing participant information. Each row represents one participant.
        Must contain columns specified by id_col, age_col (or age_grp_col), and optionally
        strat_var_cols, repeat_col, and amb_cnt_col.
    id_col : str
        Name of the column containing unique participant identifiers.
        Values must be unique for each participant. Renamed to 'id' internally.
    age_col : Optional[str], default=None
        Name of the column containing participant ages as numeric values.
        Use this OR age_grp_col, not both. Ages should be non-negative.
        Renamed to 'age_part' internally.
    age_grp_col : Optional[str], default=None
        Name of the column containing participant age groups as pd.IntervalIndex.
        Use this OR age_col, not both. Must be categorical with IntervalIndex categories.
        Renamed to 'age_grp_part' internally.
    strat_var_cols : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) for participants.
        Can be a single string or list of strings. Examples: 'gender', ['gender', 'region'].
        Each variable is renamed with '_part' suffix (e.g., 'gender' → 'gender_part').
    repeat_col : Optional[str], default=None
        Name of the column indicating repeat interviews/waves.
        Used to track longitudinal data where participants are surveyed multiple times.
        Renamed to 'repeat_part' internally.
    amb_cnt_col : Optional[str], default=None
        Name of the column containing ambiguous/group contact counts.
        If None, no ambiguous contact column is added to the DataFrame.

    Attributes
    ----------
    data : pd.DataFrame
        The validated and preprocessed participant DataFrame.
    n : int
        Returns the number of participants (after preprocessing and validation).
    age_range : Tuple[float, float]
        Returns (min_age, max_age). Only available when using age_col.
    strat_vars : List[str]
        Returns list of original stratification variable names (empty list if none).

    Methods
    -------
    validate()
        Performs comprehensive validation of the participant data.
        Called automatically during initialization.
    get_strat_vars(suffix=False)
        Returns stratification variable names, optionally with '_part' suffix.
    get_sample_sizes(stratify=False)
        Returns DataFrame with participant counts, optionally stratified by all variables.
    summary()
        Returns dictionary with summary statistics about the participant data.

    Examples
    --------
    >>> # Basic usage with individual ages
    >>> df = pd.DataFrame({
    ...     'participant_id': [1, 2, 3, 4],
    ...     'age': [25, 34, 45, 52],
    ...     'gender': ['M', 'F', 'M', 'F']
    ... })
    >>> part_data = ParticipantData(
    ...     data=df,
    ...     id_col='participant_id',
    ...     age_col='age',
    ...     strat_var_cols='gender'
    ... )
    >>> part_data.n
    4
    >>> part_data.age_range
    (25.0, 52.0)
    >>> # Columns are renamed with standardized names
    >>> list(part_data.data.columns)
    ['id', 'age_part', 'gender_part']
    >>> part_data.data['gender_part'].dtype.name
    'category'
    >>>
    >>> # With age groups and multiple stratification variables
    >>> df = pd.DataFrame({
    ...     'pid': [1, 2, 3],
    ...     'age_group': pd.Categorical.from_codes(
    ...         [0, 1, 2],
    ...         categories=pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)])
    ...     ),
    ...     'gender': ['M', 'F', 'M'],
    ...     'region': ['North', 'South', 'East']
    ... })
    >>> part_data = ParticipantData(
    ...     data=df,
    ...     id_col='pid',
    ...     age_grp_col='age_group',
    ...     strat_var_cols=['gender', 'region']
    ... )
    >>> part_data.strat_vars
    ['gender', 'region']
    >>> part_data.get_strat_vars(suffix=True)
    ['gender_part', 'region_part']
    >>> list(part_data.data.columns)
    ['id', 'age_grp_part', 'gender_part', 'region_part']
    >>>
    >>> # Get sample sizes
    >>> sample_sizes = part_data.get_sample_sizes(stratify=True)
    >>> sample_sizes.columns.tolist()
    ['age_grp_part', 'gender_part', 'region_part', 'N']

    Notes
    -----
    **Automatic Preprocessing:**

    1. Creates a copy of the input DataFrame (original is not modified)
    2. Drops rows with missing values in required columns (with warning)
    3. Converts object-type columns (except ID) to categorical dtype
    4. Renames columns to standardized names:
       - id_col → 'id' (only if id_col != 'id')
       - age_col → 'age_part' (only if not already ending with '_part')
       - age_grp_col → 'age_grp_part' (only if not already ending with '_part')
       - Each strat_var_cols → '{var}_part' (only if not already ending with '_part')
       - repeat_col → 'repeat_part' (only if not already ending with '_part')
    5. Validates all data constraints

    **Processed DataFrame Columns:**

    - id: Participant identifier (standardized from id_col)
    - age_part: Participant age (if using age_col)
    - age_grp_part: Participant age group (if using age_grp_col)
    - {var}_part: Each stratification variable with _part suffix
    - repeat_part: Repeat interview indicator (if specified)
    - {amb_cnt_col}: Ambiguous contact count column (only if specified)

    **Validation Checks:**

    - Exactly one of age_col or age_grp_col must be specified
    - All required columns must exist in data
    - Participant IDs must be unique (no duplicate rows)
    - Age values (if using age_col) must be non-negative numeric
    - Age groups (if using age_grp_col) must be categorical with IntervalIndex categories
    - Repeat values (if specified) must be non-negative numeric
    - Ambiguous contact counts (if specified) must be non-negative numeric
    - No missing values in required columns (removed during preprocessing)

    **Warnings:**

    - UserWarning: If rows are dropped due to missing values
    - UserWarning: If object columns are converted to categorical

    See Also
    --------
    ContactData : Container for contact data from social contact surveys
    DataLoader : Main data loader that combines participant and contact data
    """

    data: pd.DataFrame
    id_col: str
    age_col: Optional[str] = None
    age_grp_col: Optional[str] = None
    strat_var_cols: Optional[Union[List[str], str]] = None
    repeat_col: Optional[str] = None
    amb_cnt_col: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Post-initialization processing and validation.

        Automatically called after dataclass initialization to:
        1. Validate input types
        2. Normalize strat_var_cols to list format
        3. Validate mutual exclusivity of age_col and age_grp_col
        4. Preprocess the DataFrame (copy, clean, type conversion)
        5. Perform comprehensive data validation

        Raises
        ------
        TypeError
            If df_part is not a pandas DataFrame.
        ValueError
            If both or neither of age_col and age_grp_col are specified.
            If DataFrame becomes empty after preprocessing.
        """
        # Type validation
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame, got {type(self.data).__name__}"
            )

        # Normalize strat_var_cols to list format for consistent handling
        if isinstance(self.strat_var_cols, str):
            object.__setattr__(self, "strat_var_cols", [self.strat_var_cols])
        elif self.strat_var_cols is None:
            object.__setattr__(self, "strat_var_cols", [])

        # Validate mutual exclusivity of age specifications
        if self.age_col is None and self.age_grp_col is None:
            raise ValueError(
                "Must specify exactly one of 'age_col' or 'age_grp_col'.\n"
                "  Use 'age_col' for exact integer ages (e.g., 25, 34, 45),\n"
                "  or 'age_grp_col' for age groups (e.g., pd.IntervalIndex or categorical)."
            )

        if self.age_col is not None and self.age_grp_col is not None:
            raise ValueError(
                "Cannot specify both 'age_col' and 'age_grp_col' simultaneously.\n"
                f"  age_col='{self.age_col}', age_grp_col='{self.age_grp_col}'\n"
                "  Hint: specify only one age representation."
            )

        # Delegate column validation, NaN removal, dtype coercion, and renaming
        object.__setattr__(
            self,
            "data",
            preprocess_participant_data(
                self.data,
                self.id_col,
                self.age_col,
                self.age_grp_col,
                self.strat_var_cols,
                self.repeat_col,
                self.amb_cnt_col,
            ),
        )

        # Perform domain validation on the cleaned data
        self.validate()

    def validate(self) -> None:
        """
        Perform comprehensive validation of participant data.

        Validation checks:
        1. Participant IDs are unique (no duplicates)
        2. (If age_col specified) Age column contains valid non-negative numeric values
        3. (If age_grp_col specified) Age group column contains valid pd.IntervalIndex or categorical values
        4. (If repeat_col specified) Repeat interview values are non-negative integers
        5. (If amb_cnt_col specified) Ambiguous contact count values are non-negative integers

        Note: Missing value checks are handled during preprocessing in _preprocess(),
        so this method assumes clean data without NaNs.

        Raises
        ------
        KeyError
            If required columns are missing from the DataFrame.
        ValueError
            If duplicate participant IDs are found.
            If age values are invalid (negative or non-numeric).

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> # Validation happens automatically, but can be called explicitly:
        >>> part_data.validate()
        """
        # Check 1: Validate unique participant IDs (using standardized 'id' column)
        duplicate_ids = self.data["id"].duplicated()
        if duplicate_ids.any():
            duplicate_examples = self.data[duplicate_ids]["id"].head(5).tolist()
            n_duplicates = duplicate_ids.sum()
            raise ValueError(
                f"Found {n_duplicates} duplicate participant ID(s) in 'id' column.\n"
                f"  Examples of duplicates: {duplicate_examples}\n"
                f"  Hint: each row should represent a unique participant."
            )

        # Check 2: Validate age values if using exact ages
        if self.age_col:
            ages = self.data["age_part"]

            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(ages):
                raise ValueError(
                    f"Age column 'age_part' must contain numeric values.\n"
                    f"  Current dtype: {ages.dtype}\n"
                    f"  Hint: convert age to integer or float type."
                )

            # Check for negative ages
            if (ages < 0).any():
                negative_indices = self.data[ages < 0].index[:5].tolist()
                raise ValueError(
                    f"Age column 'age_part' contains negative values.\n"
                    f"  Rows with negative ages: {negative_indices}\n"
                    f"  Values: {ages[ages < 0].head().tolist()}"
                )

        # Check 3: Validate age group values if using age groups
        if self.age_grp_col:
            is_categorical = isinstance(
                self.data["age_grp_part"].dtype, pd.CategoricalDtype
            )
            if is_categorical:
                are_intervals = isinstance(
                    self.data["age_grp_part"].cat.categories, pd.IntervalIndex
                )
                if not are_intervals:
                    raise TypeError(
                        f"Column '{self.age_grp_col}' must have pd.IntervalIndex categories.\n"
                        f"  Got: {type(self.data['age_grp_part'].cat.categories)}"
                    )
            else:
                raise TypeError(
                    f"Column '{self.age_grp_col}' must be categorical with interval categories.\n"
                    f"  Current type: {self.data['age_grp_part'].dtype}"
                )

        # Check 4: Validate repeat interview values if specified
        if self.repeat_col:
            repeats = self.data["repeat_part"]

            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(repeats):
                raise ValueError(
                    f"Repeat interview column 'repeat_part' must contain numeric values.\n"
                    f"  Current dtype: {repeats.dtype}\n"
                    f"  Hint: convert repeat interview to integer type."
                )

            # Check for negative repeat values
            if (repeats < 0).any():
                negative_indices = self.data[repeats < 0].index[:5].tolist()
                raise ValueError(
                    f"Repeat interview column 'repeat_part' contains negative values.\n"
                    f"  Rows with negative values: {negative_indices}\n"
                    f"  Values: {repeats[repeats < 0].head().tolist()}"
                )

        # Check 5: Validate group contact count values if specified
        if self.amb_cnt_col:
            grp_counts = self.data[self.amb_cnt_col]

            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(grp_counts):
                raise ValueError(
                    f"Group contact count column '{self.amb_cnt_col}' must contain numeric values.\n"
                    f"  Current dtype: {grp_counts.dtype}\n"
                    f"  Hint: convert group contact count to integer type."
                )

            # Check for negative group contact counts
            if (grp_counts < 0).any():
                negative_indices = self.data[grp_counts < 0].index[:5].tolist()
                raise ValueError(
                    f"Group contact count column '{self.amb_cnt_col}' contains negative values.\n"
                    f"  Rows with negative values: {negative_indices}\n"
                    f"  Values: {grp_counts[grp_counts < 0].head().tolist()}"
                )

    @property
    def n(self) -> int:
        """
        Return the number of participants in the dataset.

        Returns
        -------
        int
            Total number of participants (rows in the DataFrame).

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> part_data.n
        500
        """
        return len(self.data)

    @property
    def age_range(self) -> Tuple[float, float]:
        """
        Return the age range (min, max) of participants.

        Only available when using age_col (exact ages), not age_grp_col.

        Returns
        -------
        Tuple[float, float]
            Tuple of (minimum_age, maximum_age).

        Raises
        ------
        ValueError
            If age_grp_col is used instead of age_col.

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> part_data.age_range
        (0, 85)
        """
        if self.age_col is None:
            raise ValueError(
                "age_range is only available when using 'age_col' (exact ages).\n"
                f"Currently using 'age_grp_col': {self.age_grp_col}"
            )

        ages = self.data["age_part"]
        return (float(ages.min()), float(ages.max()))

    @property
    def strat_vars(self) -> List[str] | str:
        """
        Return list of stratification variable names.

        Returns
        -------
        List[str]
            List of stratification variable column names (empty if none).

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age', strat_var_cols=['gender', 'region'])
        >>> part_data.strat_vars
        ['gender', 'region']
        """
        return self.strat_var_cols if self.strat_var_cols else []

    def get_strat_vars(self, suffix: bool = False) -> List[str]:
        """
        Return list of stratification variable names, optionally with '_part' suffix.

        Parameters
        ----------
        suffix : bool, default=False
            If True, returns stratification variable names with '_part' suffix.
            If False, returns original stratification variable names.

        Returns
        -------
        List[str]
            List of stratification variable column names.

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age', strat_var_cols=['gender', 'region'])
        >>> part_data.get_strat_vars(suffix=True)
        ['gender_part', 'region_part']
        """
        if not self.strat_var_cols:
            return []

        if suffix:
            return [
                f"{var}_part" if not var.endswith("_part") else var
                for var in self.strat_var_cols
            ]
        else:
            return [var.removesuffix("_part") for var in self.strat_var_cols]

    def get_strat_var_schema(self) -> Dict[str, Dict[str, List[str | int]]]:
        """
        Return a dictionary of stratification variables and their unique category and code values.
        The output is to be used in the DataLoader to ensure that the category and codes are consistent
        between the participant, contact, population, and demographic data.

        Returns
        -------
        Dict[str, Dict[str, List[str | int]]]
            A dictionary where keys are the original stratification variable names (without '_part' suffix)
            and values are dictionaries with keys 'categories' and 'codes' containing lists of unique category
            values and their corresponding codes.
        """
        if self.strat_var_cols is not None:
            schema = {}
            for var in self.strat_var_cols:
                var_part = f"{var}_part" if not var.endswith("_part") else var
                if var_part in self.data.columns:
                    categories = self.data[var_part].cat.categories.tolist()
                    codes = sorted(self.data[var_part].cat.codes.unique().tolist())

                    # Remove suffix
                    var = (
                        var.removesuffix("_part") if var.endswith("_part") else var
                    )  # Remove suffix

                    schema[var] = {"categories": categories, "codes": codes}

            return schema
        else:
            return {}

    def get_sample_sizes(self, stratify=False) -> pd.DataFrame:
        """
        Return a DataFrame with sample sizes per stratification group.

        Parameters
        ----------
        stratify : bool, default=False
            If True, returns counts of participants stratified by all stratification variables.
            If False, returns counts of participants stratified by age only.

        Returns
        -------
        pd.DataFrame
            DataFrame with counts of participants per stratification group.
            If stratify=False, returns a DataFrame where counts of participants are stratified by age only.

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> sample_sizes = part_data.get_sample_sizes()
        >>> print(sample_sizes.head())
        age
        0     15
        1     18
        2     20
        ...
        """
        age_column = "age_part" if self.age_col else "age_grp_part"
        if stratify and self.strat_var_cols:
            group_cols = [age_column] + [f"{var}_part" for var in self.strat_var_cols]
            return (
                self.data.groupby(group_cols, observed=False)
                .agg(N=("id", "count"))
                .reset_index()
            )

        return (
            self.data.groupby(age_column, observed=False)
            .agg(N=("id", "count"))
            .reset_index()
        )

    def summary(self) -> Dict[str, Any]:
        """
        Return summary statistics about the participant data.

        Provides a comprehensive overview including sample size, age statistics
        (if applicable), and stratification information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - n_participants: Total number of participants
            - id_col: Name of ID column
            - age_col: Name of age column (or None)
            - age_grp_col: Name of age group column (or None)
            - age_range: Tuple of (min, max) age if using age_col
            - strat_vars: List of stratification variables

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age', strat_var_cols='gender')
        >>> summary = part_data.summary()
        >>> print(summary)
        {
            'n_participants': 500,
            'id_col': 'id',
            'age_col': 'age',
            'age_grp_col': None,
            'age_range': (0, 85),
            'strat_vars': ['gender']
        }
        """
        summary_dict = {
            "n": self.n,
            "id_col": self.id_col,
            "age_col": self.age_col,
            "age_grp_col": self.age_grp_col,
            "strat_vars": self.strat_vars,
        }

        # Add age range if using exact ages
        if self.age_col:
            summary_dict["age_range"] = self.age_range

        return summary_dict
