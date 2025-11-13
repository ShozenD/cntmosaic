import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


@dataclass
class ParticipantData:
    """
    Validated participant data container for social contact surveys.

    This class provides a type-safe, validated wrapper around participant DataFrames,
    ensuring data integrity before use in contact matrix estimation models. It performs
    comprehensive validation of required columns, data types, and value ranges, along
    with automatic preprocessing for downstream modeling.

    Attributes
    ----------
    df_part : pd.DataFrame
        DataFrame containing participant information. Each row represents one participant.
        Must contain columns specified by id_col, age_col (or age_grp_col), and strat_var_cols.
        Note: The DataFrame is automatically preprocessed (copied, cleaned, type-converted).
    id_col : str
        Name of the column containing unique participant identifiers.
        Values must be unique for each participant. Used to link participants with their contacts.
    age_col : Optional[str], default=None
        Name of the column containing participant ages as integers.
        Use this OR age_grp_col, not both. Ages should be non-negative integers.
    age_grp_col : Optional[str], default=None
        Name of the column containing participant age groups.
        Use this OR age_col, not both. Should be pd.IntervalIndex or categorical age groups.
    strat_var_cols : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) for participants.
        Can be a single string or list of strings. Examples: 'gender', ['gender', 'occupation'].
        These variables allow for demographic stratification in contact matrix estimation.
    repeat_col : Optional[str], default=None
        Name of the column indicating repeat interviews/waves for participants.
        If specified, this column will be renamed to 'repeat_part' in the processed DataFrame.
        Used to track longitudinal data where participants are surveyed multiple times.
    grp_cnt_col : str, default='z'
        Name of the column for group contact counts. If not present in df_part,
        it will be automatically created and initialized to 0.

    Properties
    ----------
    data : pd.DataFrame
        Returns the validated and preprocessed participant DataFrame.
    n_participants : int
        Returns the number of participants in the dataset (after preprocessing).
    age_range : Tuple[float, float]
        Returns (min_age, max_age) if age_col is specified, otherwise raises ValueError.
    stratification_vars : List[str]
        Returns list of stratification variable names (empty list if none).

    Methods
    -------
    validate()
        Performs comprehensive validation of the participant data.
    get_age_distribution()
        Returns age distribution of participants as a Series.
    summary()
        Returns a dictionary with summary statistics about the participant data.

    Raises
    ------
    ValueError
        If neither age_col nor age_grp_col is provided.
        If both age_col and age_grp_col are provided simultaneously.
        If age values contain NaN, negative values, or non-numeric types.
        If duplicate participant IDs are found.
        If DataFrame becomes empty after removing missing values.
    KeyError
        If required columns (id_col, age_col, age_grp_col, strat_var_cols) are missing.
    TypeError
        If df_part is not a pandas DataFrame.

    Examples
    --------
    >>> # Basic usage with individual ages
    >>> df = pd.DataFrame({
    ...     'participant_id': [1, 2, 3, 4],
    ...     'age': [25, 34, 45, 52],
    ...     'gender': ['M', 'F', 'M', 'F']
    ... })
    >>> part_data = ParticipantData(
    ...     df_part=df,
    ...     id_col='participant_id',
    ...     age_col='age',
    ...     strat_var_cols='gender'
    ... )
    >>> part_data.n_participants
    4
    >>> part_data.age_range
    (25, 52)
    >>> # Age column is renamed to 'age_part'
    >>> 'age_part' in part_data.data.columns
    True
    >>> # Gender column is renamed to 'gender_part'
    >>> 'gender_part' in part_data.data.columns
    True
    >>> # Note: 'z' column is automatically added with value 0
    >>> 'z' in part_data.data.columns
    True
    >>>
    >>> # With age groups and multiple stratification variables
    >>> df = pd.DataFrame({
    ...     'pid': [1, 2, 3],
    ...     'age_group': pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
    ...     'gender': ['M', 'F', 'M'],
    ...     'region': ['North', 'South', 'East']
    ... })
    >>> part_data = ParticipantData(
    ...     df_part=df,
    ...     id_col='pid',
    ...     age_grp_col='age_group',
    ...     strat_var_cols=['gender', 'region']
    ... )
    >>> part_data.stratification_vars
    ['gender', 'region']
    >>> # Columns are renamed with _part suffix
    >>> list(part_data.data.columns)
    ['pid', 'age_grp_part', 'gender_part', 'region_part', 'z']
    >>> # Object columns are automatically converted to categorical
    >>> part_data.data['gender_part'].dtype.name
    'category'
    >>>
    >>> # Accessing summary statistics
    >>> summary = part_data.summary()
    >>> print(summary['n_participants'])
    3

    Notes
    -----
    Preprocessing Steps (Automatic):
    - Creates a copy of the input DataFrame to avoid side effects
    - Drops rows with missing values in required columns
    - Converts object-type columns (except ID) to categorical for efficiency
    - Renames columns with standardized names:
      * id_col → 'id' (standardized identifier)
      * age_col → 'age_part'
      * age_grp_col → 'age_grp_part' (if specified)
      * each strat_var_cols → '{var}_part'
      * repeat_col → 'repeat_part' (if specified)
    - Adds 'z' column (group contact count) initialized to 0 if not present
    - Validates all data constraints after preprocessing

    Processed DataFrame Structure:
    - id: Standardized participant identifier (renamed from id_col)
    - age_part: Participant age (renamed from age_col)
    - age_grp_part: Participant age group (renamed from age_grp_col, if specified)
    - {var}_part: Each stratification variable with _part suffix
    - repeat_part: Repeat interview indicator (renamed from repeat_col, if specified)
    - z: Group contact count column

    Validation Checks:
    - Exactly one of age_col or age_grp_col must be specified
    - Participant IDs must be unique (no duplicate rows for the same participant)
    - Age values (if using age_col) must be non-negative integers or floats
    - No missing values in required columns (removed during preprocessing)
    - Stratification variables are optional but commonly include: gender, occupation,
      region, household size, etc.

    Warnings:
    - UserWarning if rows are dropped due to missing values
    - UserWarning if object columns are converted to categorical
    """

    df_part: pd.DataFrame
    id_col: str
    age_col: Optional[str] = None
    age_grp_col: Optional[str] = None
    strat_var_cols: Optional[Union[List[str], str]] = None
    repeat_col: Optional[str] = None
    grp_cnt_col: str = "z"

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
        if not isinstance(self.df_part, pd.DataFrame):
            raise TypeError(
                f"df_part must be a pandas DataFrame, got {type(self.df_part).__name__}"
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
                "Use 'age_col' for exact integer ages (e.g., 25, 34, 45),\n"
                "or 'age_grp_col' for age groups (e.g., pd.IntervalIndex or categorical)."
            )

        if self.age_col is not None and self.age_grp_col is not None:
            raise ValueError(
                "Cannot specify both 'age_col' and 'age_grp_col' simultaneously.\n"
                f"Currently: age_col='{self.age_col}', age_grp_col='{self.age_grp_col}'\n"
                "Please specify only one age representation."
            )

        # Preprocess the DataFrame
        object.__setattr__(self, "df_part", self._preprocess())

        # Perform comprehensive validation
        self.validate()

    def _preprocess(self) -> pd.DataFrame:
        """
        Preprocess participant DataFrame for modeling.

        Performs the following preprocessing steps:
        1. Creates a copy to avoid modifying the original DataFrame
        2. Validates that required columns exist
        3. Drops rows with missing values in required columns
        4. Converts object-type columns to categorical for efficiency
        5. Renames columns to standardized names:
           - id_col → 'id' (only if id_col != 'id')
           - age_col → 'age_part'
           - age_grp_col → 'age_grp_part'
           - Each strat_var_cols → '{var}_part'
           - repeat_col → 'repeat_part' (if specified)
        6. Adds 'z' column (group contact count) if not present

        Returns
        -------
        pd.DataFrame
            Preprocessed participant DataFrame ready for validation and modeling.

        Raises
        ------
        KeyError
            If required columns are missing from the DataFrame.
        ValueError
            If DataFrame becomes empty after removing missing values.

        Warnings
        --------
        UserWarning
            If rows are dropped due to missing values.
            If object columns are converted to categorical.

        Notes
        -----
        This method follows the preprocessing patterns used in BaseLoader._validate()
        and DataLoader._validate_part() to ensure consistency with the package's
        data handling conventions.
        """
        # Step 1: Create a copy to avoid side effects
        df = self.df_part.copy()

        # Step 2: Check that required columns exist BEFORE trying to access them
        age_column = self.age_col if self.age_col else self.age_grp_col

        # Check ID column exists
        if self.id_col not in df.columns:
            raise KeyError(
                f"Missing participant ID column '{self.id_col}' in DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Check age column exists
        if age_column not in df.columns:
            col_type = "age" if self.age_col else "age group"
            raise KeyError(
                f"Missing participant {col_type} column '{age_column}' in DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Check stratification variables exist
        if self.strat_var_cols:
            missing_vars = [var for var in self.strat_var_cols if var not in df.columns]
            if missing_vars:
                raise KeyError(
                    f"Missing stratification variable(s) {missing_vars} in DataFrame.\n"
                    f"Available columns: {list(df.columns)}"
                )

        # Check repeat column exists if specified
        if self.repeat_col and self.repeat_col not in df.columns:
            raise KeyError(
                f"Missing repeat column '{self.repeat_col}' in DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Step 3: Identify required columns and check for missing values
        required_cols = [self.id_col, age_column]
        if self.strat_var_cols:
            required_cols.extend(self.strat_var_cols)
        if self.repeat_col:
            required_cols.append(self.repeat_col)

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
                f"Remaining participants: {n_rows_after}",
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

        # Step 4: Convert object columns to categorical for efficiency
        # (excluding the ID column which should remain as-is for merging)
        object_cols = df.select_dtypes(include="object").columns.tolist()
        if self.id_col in object_cols:
            object_cols.remove(self.id_col)

        for col in object_cols:
            if not isinstance(df[col].dtype, pd.CategoricalDtype):
                warnings.warn(
                    f"Column '{col}' is object type. Converting to categorical for efficiency.",
                    UserWarning,
                    stacklevel=3,
                )
                df[col] = df[col].astype("category")

        # Step 5: Rename columns with _part suffix
        rename_map = {}

        # Rename ID column to standardized 'id'
        if self.id_col != "id":
            rename_map[self.id_col] = "id"

        # Rename age column (only add _part suffix if not already present)
        if self.age_col:
            new_name = (
                "age_part" if not self.age_col.endswith("_part") else self.age_col
            )
            if self.age_col != new_name:
                rename_map[self.age_col] = new_name

        # Rename age group column (only add _part suffix if not already present)
        if self.age_grp_col:
            new_name = (
                "age_grp_part"
                if not self.age_grp_col.endswith("_part")
                else self.age_grp_col
            )
            if self.age_grp_col != new_name:
                rename_map[self.age_grp_col] = new_name

        # Rename stratification variables (only add _part suffix if not already present)
        if self.strat_var_cols:
            for var in self.strat_var_cols:
                new_name = f"{var}_part" if not var.endswith("_part") else var
                if var != new_name:
                    rename_map[var] = new_name

        # Rename repeat column (only add _part suffix if not already present)
        if self.repeat_col:
            new_name = (
                "repeat_part"
                if not self.repeat_col.endswith("_part")
                else self.repeat_col
            )
            if self.repeat_col != new_name:
                rename_map[self.repeat_col] = new_name

        df = df.rename(columns=rename_map)

        # Step 6: Add 'z' column (group contact count) if not present
        if self.grp_cnt_col not in df.columns:
            df[self.grp_cnt_col] = 0

        return df

    def validate(self) -> None:
        """
        Perform comprehensive validation of participant data.

        Validation checks:
        1. All required columns exist in the DataFrame (post-preprocessing)
        2. Participant IDs are unique (no duplicates)
        3. Age column contains valid non-negative numeric values (if age_col specified)
        4. Age group column contains valid pd.IntervalIndex or categorical values (if age_grp_col specified)

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
        # Check 1: Validate standardized 'id' column exists
        if "id" not in self.df_part.columns:
            raise KeyError(
                f"Missing standardized 'id' column in processed DataFrame.\n"
                f"Available columns: {list(self.df_part.columns)}\n"
                f"This should not happen - please report this bug."
            )

        # Check 2: Validate age column exists (with _part suffix after preprocessing)
        if self.age_col:
            age_column = "age_part"
        else:
            age_column = "age_grp_part"

        if age_column not in self.df_part.columns:
            col_type = "age_part" if self.age_col else "age_grp_part"
            raise KeyError(
                f"Missing participant {col_type} column in processed DataFrame.\n"
                f"Available columns: {list(self.df_part.columns)}\n"
                f"This should not happen - please report this bug."
            )

        # Check 3: Validate stratification variables exist (with _part suffix)
        if self.strat_var_cols:
            # Calculate expected column names (add _part only if not already present)
            expected_vars = [
                f"{var}_part" if not var.endswith("_part") else var
                for var in self.strat_var_cols
            ]
            missing_vars = [
                var for var in expected_vars if var not in self.df_part.columns
            ]
            if missing_vars:
                raise KeyError(
                    f"Missing stratification variable(s) {missing_vars} in processed DataFrame.\n"
                    f"Available columns: {list(self.df_part.columns)}\n"
                    f"This should not happen - please report this bug."
                )

        # Check 4: Validate unique participant IDs (using standardized 'id' column)
        duplicate_ids = self.df_part["id"].duplicated()
        if duplicate_ids.any():
            n_duplicates = duplicate_ids.sum()
            duplicate_examples = self.df_part[duplicate_ids]["id"].head(5).tolist()
            raise ValueError(
                f"Found {n_duplicates} duplicate participant ID(s) in 'id' column.\n"
                f"Participant IDs must be unique. Examples of duplicates: {duplicate_examples}\n"
                f"Hint: Each row should represent a unique participant."
            )

        # Check 5: Validate age values if using exact ages
        if self.age_col:
            ages = self.df_part["age_part"]

            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(ages):
                raise ValueError(
                    f"Age column 'age_part' must contain numeric values.\n"
                    f"Current dtype: {ages.dtype}\n"
                    f"Hint: Convert age to integer or float type."
                )

            # Check for negative ages
            if (ages < 0).any():
                negative_indices = self.df_part[ages < 0].index[:5].tolist()
                raise ValueError(
                    f"Age column 'age_part' contains negative values.\n"
                    f"Ages must be non-negative. Rows with negative ages: {negative_indices}\n"
                    f"Values: {ages[ages < 0].head().tolist()}"
                )

        # Check 6: Validate age group values if using age groups
        if self.age_grp_col:
            is_categorical = isinstance(
                self.df_part["age_grp_part"].dtype, pd.CategoricalDtype
            )
            if is_categorical:
                are_intervals = isinstance(
                    self.df_part["age_grp_part"].cat.categories, pd.IntervalIndex
                )
                if not are_intervals:
                    raise TypeError(
                        f"Column '{self.age_grp_col}' must have pd.IntervalIndex categories, "
                        f"got {type(self.df_part["age_grp_part"].cat.categories)}"
                    )
            else:
                raise TypeError(
                    f"Column '{self.age_grp_col}' must be categorical with interval categories. "
                    f"Current type: {self.df_part["age_grp_part"].dtype}"
                )

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the validated participant DataFrame.

        Returns
        -------
        pd.DataFrame
            The validated participant data.

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> validated_df = part_data.data
        """
        return self.df_part

    @property
    def n_participants(self) -> int:
        """
        Return the number of participants in the dataset.

        Returns
        -------
        int
            Total number of participants (rows in the DataFrame).

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> part_data.n_participants
        500
        """
        return len(self.df_part)

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

        ages = self.df_part["age_part"]
        return (float(ages.min()), float(ages.max()))

    @property
    def stratification_vars(self) -> List[str]:
        """
        Return list of stratification variable names.

        Returns
        -------
        List[str]
            List of stratification variable column names (empty if none).

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age', strat_var_cols=['gender', 'region'])
        >>> part_data.stratification_vars
        ['gender', 'region']
        """
        return self.strat_var_cols if self.strat_var_cols else []

    def get_age_distribution(self) -> pd.Series:
        """
        Return age distribution of participants.

        Returns value counts for age_col (exact ages) or age_grp_col (age groups),
        sorted by age/age group.

        Returns
        -------
        pd.Series
            Series with age (or age group) as index and counts as values.

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> age_dist = part_data.get_age_distribution()
        >>> print(age_dist.head())
        age
        0     15
        1     18
        2     20
        ...
        """
        age_column = "age_part" if self.age_col else "age_grp_part"
        return self.df_part[age_column].value_counts().sort_index()

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
            - stratification_vars: List of stratification variables
            - n_stratification_vars: Number of stratification variables

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
            'stratification_vars': ['gender'],
            'n_stratification_vars': 1
        }
        """
        summary_dict = {
            "n_participants": self.n_participants,
            "id_col": self.id_col,
            "age_col": self.age_col,
            "age_grp_col": self.age_grp_col,
            "stratification_vars": self.stratification_vars,
            "n_stratification_vars": len(self.stratification_vars),
        }

        # Add age range if using exact ages
        if self.age_col:
            summary_dict["age_range"] = self.age_range

        return summary_dict
