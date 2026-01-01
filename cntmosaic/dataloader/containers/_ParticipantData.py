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
        Can be a single string or list of strings. Examples: 'gender', ['gender', 'region'].

        **Important**: If multiple variables are specified (e.g., ['gender', 'region']),
        they will be **combined into a single composite stratification variable** by DataLoader.
        For example, ['gender', 'region'] with categories ['M', 'F'] and ['North', 'South']
        will create a combined variable with categories: ['M_North', 'M_South', 'F_North', 'F_South'].

        **Consistency requirement**: The same stratification variables must be specified across:
        - ParticipantData (required if stratifying)
        - ContactData (required if FULL stratification mode)
        - PopulationData (required if using stratified population data)
        - StratPropData (required, one object per composite stratification)
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
    grp_cnt_col: Optional[str] = None

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

        # Check that required columns exist
        self._check_columns()

        # Preprocess the DataFrame
        object.__setattr__(self, "df_part", self._preprocess())

        # Perform validation
        self.validate()

    def _check_columns(self) -> None:
        """
        Check that required columns exist in the DataFrame.
        Creates a list of required columns (self._required_cols) to be used in _preprocess().

        Raises
        ------
        KeyError
            If required columns are missing from the DataFrame.
        """
        # Step 2: Check that required columns exist BEFORE trying to access them
        age_column = self.age_col if self.age_col else self.age_grp_col

        # Check ID column exists
        if self.id_col not in self.df_part.columns:
            raise KeyError(
                f"Missing participant ID column '{self.id_col}' in DataFrame.\n"
                f"Available columns: {list(self.df_part.columns)}"
            )

        # Check age column exists
        if age_column not in self.df_part.columns:
            col_type = "age" if self.age_col else "age group"
            raise KeyError(
                f"Missing participant {col_type} column '{age_column}' in DataFrame.\n"
                f"Available columns: {list(self.df_part.columns)}"
            )

        # Check stratification variables exist (if specified)
        if self.strat_var_cols:
            missing_vars = [
                var for var in self.strat_var_cols if var not in self.df_part.columns
            ]
            if missing_vars:
                raise KeyError(
                    f"strat_var_cols '{self.strat_var_cols}' is specified but missing '{missing_vars}' in DataFrame.\n"
                    f"Available columns: {list(self.df_part.columns)}"
                )

        # Check repeat column exists (if specified)
        if self.repeat_col and self.repeat_col not in self.df_part.columns:
            raise KeyError(
                f"repeat_col '{self.repeat_col}' is specified but missing in DataFrame.\n"
                f"Available columns: {list(self.df_part.columns)}"
            )

        # Check group contact count column exist (if specified)
        if self.grp_cnt_col and self.grp_cnt_col not in self.df_part.columns:
            raise KeyError(
                f"grp_cnt_col '{self.grp_cnt_col}' is specified but missing in DataFrame.\n"
                f"Available columns: {list(self.df_part.columns)}"
            )

        # Step 3: Identify required columns and check for missing values
        self._required_cols = [self.id_col, age_column]
        if self.strat_var_cols:
            self._required_cols.extend(self.strat_var_cols)
        if self.repeat_col:  # If repeat column is specified
            self._required_cols.append(self.repeat_col)

    def _preprocess(self) -> pd.DataFrame:
        """
        Preprocess participant DataFrame for modeling.

        Performs the following preprocessing steps:
        1. Creates a copy to avoid modifying the original DataFrame
        2. Drops rows with missing values in required columns
        3. Converts object-type columns to categorical for efficiency
        4. Renames columns to standardized names:
           - id_col → 'id' (only if id_col != 'id')
           - age_col → 'age_part'
           - age_grp_col → 'age_grp_part'
           - Each strat_var_cols → '{var}_part'
           - repeat_col → 'repeat_part' (if specified)
        5. Adds 'z' column (group contact count) if not present

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

        # Check for missing values before dropping
        n_rows_before = len(df)
        missing_counts = df[self._required_cols].isnull().sum()
        has_missing = missing_counts.sum() > 0

        # Drop rows with missing values in required columns
        df = df[self._required_cols].dropna().copy()

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
                f"Required columns: {self._required_cols}\n"
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
                    f"Converting '{col}' to categorical dtype.",
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
        if self.grp_cnt_col is None:
            self.grp_cnt_col = "z"
            df[self.grp_cnt_col] = 0

        return df

    def validate(self) -> None:
        """
        Perform comprehensive validation of participant data.

        Validation checks:
        1. Participant IDs are unique (no duplicates)
        2. (If age_col specified) Age column contains valid non-negative numeric values
        3. (If age_grp_col specified) Age group column contains valid pd.IntervalIndex or categorical values
        4. (If repeat_col specified) Repeat interview values are non-negative integers
        5. (If grp_cnt_col specified) Group contact count values are non-negative integers

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
        duplicate_ids = self.df_part["id"].duplicated()
        if duplicate_ids.any():
            duplicate_examples = self.df_part[duplicate_ids]["id"].head(5).tolist()
            n_duplicates = duplicate_ids.sum()
            raise ValueError(
                f"Found {n_duplicates} duplicate participant ID(s) in 'id' column.\n"
                f"Participant IDs must be unique. Examples of duplicates: {duplicate_examples}\n"
                f"Hint: Each row should represent a unique participant."
            )

        # Check 2: Validate age values if using exact ages
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

        # Check 3: Validate age group values if using age groups
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

        # Check 4: Validate repeat interview values if specified
        if self.repeat_col:
            repeats = self.df_part["repeat_part"]

            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(repeats):
                raise ValueError(
                    f"Repeat interview column 'repeat_part' must contain numeric values.\n"
                    f"Current dtype: {repeats.dtype}\n"
                    f"Hint: Convert repeat interview to integer type."
                )

            # Check for negative repeat values
            if (repeats < 0).any():
                negative_indices = self.df_part[repeats < 0].index[:5].tolist()
                raise ValueError(
                    f"Repeat interview column 'repeat_part' contains negative values.\n"
                    f"Values must be non-negative. Rows with negative values: {negative_indices}\n"
                    f"Values: {repeats[repeats < 0].head().tolist()}"
                )

        # Check 5: Validate group contact count values if specified
        if self.grp_cnt_col:
            grp_counts = self.df_part[self.grp_cnt_col]

            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(grp_counts):
                raise ValueError(
                    f"Group contact count column '{self.grp_cnt_col}' must contain numeric values.\n"
                    f"Current dtype: {grp_counts.dtype}\n"
                    f"Hint: Convert group contact count to integer type."
                )

            # Check for negative group contact counts
            if (grp_counts < 0).any():
                negative_indices = self.df_part[grp_counts < 0].index[:5].tolist()
                raise ValueError(
                    f"Group contact count column '{self.grp_cnt_col}' contains negative values.\n"
                    f"Values must be non-negative. Rows with negative values: {negative_indices}\n"
                    f"Values: {grp_counts[grp_counts < 0].head().tolist()}"
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
    def strat_vars(self) -> List[str]:
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

    def get_sample_sizes(self, stratify=False) -> pd.Series:
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
                self.df_part.groupby(group_cols, observed=True)
                .agg(N=("id", "count"))
                .reset_index()
            )

        return (
            self.df_part.groupby(age_column, observed=False)
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
