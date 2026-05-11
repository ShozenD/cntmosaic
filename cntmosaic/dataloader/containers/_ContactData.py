import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


@dataclass
class ContactData:
    """
    Validated contact data container for social contact surveys.

    This class provides a type-safe, validated wrapper around contact DataFrames,
    ensuring data integrity before use in contact matrix estimation models. It performs
    comprehensive validation of required columns, data types, and value ranges, along
    with automatic preprocessing for downstream modeling.

    Attributes
    ----------
    df_cnt : pd.DataFrame
        DataFrame containing contact information. Each row represents one contact reported
        by a participant. Must contain columns specified by id_col, age_col (or age_grp_col),
        and strat_var_cols.
        Note: The DataFrame is automatically preprocessed (copied, cleaned, type-converted).
    id_col : str
        Name of the column containing participant identifiers (links to participant data).
        Used to associate contacts with their reporting participants.
    age_col : Optional[str], default=None
        Name of the column containing contact ages as integers.
        Use this OR age_grp_col, not both. Ages should be non-negative integers.
    age_grp_col : Optional[str], default=None
        Name of the column containing contact age groups.
        Use this OR age_col, not both. Should be pd.IntervalIndex or categorical age groups.
    strat_var_cols : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) for contacts.
        Can be a single string or list of strings. Examples: 'setting', ['setting', 'duration'].
    cnt_col : str, default='y'
        Name of the column for contact counts/indicators. If not present in df_cnt,
        it will be automatically created and initialized to 1 (one contact per row).

    Properties
    ----------
    data : pd.DataFrame
        Returns the validated and preprocessed contact DataFrame.
    n : int
        Returns the number of contact records in the dataset (after preprocessing).
    n_cnt : int
        Returns the number of unique participants who reported contacts.
    age_range : Tuple[float, float]
        Returns (min_age, max_age) of contacts if age_col is specified.
    stratification_vars : List[str]
        Returns list of stratification variable names (empty list if none).

    Methods
    -------
    validate()
        Performs comprehensive validation of the contact data.
    get_contacts_per_participant()
        Returns Series with number of contacts reported by each participant.
    summary()
        Returns a dictionary with summary statistics about the contact data.

    Raises
    ------
    ValueError
        If neither age_col nor age_grp_col is provided.
        If both age_col and age_grp_col are provided simultaneously.
        If age values contain negative values or non-numeric types.
        If DataFrame becomes empty after removing missing values.
    KeyError
        If required columns (id_col, age_col, age_grp_col, strat_var_cols) are missing.
    TypeError
        If df_cnt is not a pandas DataFrame.

    Examples
    --------
    >>> # Basic usage with individual contact ages
    >>> df = pd.DataFrame({
    ...     'participant_id': [1, 1, 2, 3],
    ...     'contact_age': [30, 45, 25, 60],
    ...     'setting': ['home', 'work', 'home', 'other']
    ... })
    >>> cnt_data = ContactData(
    ...     df_cnt=df,
    ...     id_col='participant_id',
    ...     age_col='contact_age',
    ...     strat_var_cols='setting'
    ... )
    >>> cnt_data.n
    4
    >>> cnt_data.n_cnt
    3
    >>> # Note: 'y' column is automatically added with value 1
    >>> 'y' in cnt_data.data.columns
    True
    >>>
    >>> # With age groups and multiple stratification variables
    >>> df = pd.DataFrame({
    ...     'pid': [1, 1, 2],
    ...     'age_group': pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
    ...     'setting': ['home', 'school', 'work'],
    ...     'duration': ['short', 'long', 'medium']
    ... })
    >>> cnt_data = ContactData(
    ...     df_cnt=df,
    ...     id_col='pid',
    ...     age_grp_col='age_group',
    ...     strat_var_cols=['setting', 'duration']
    ... )
    >>> cnt_data.stratification_vars
    ['setting', 'duration']
    >>> # Object columns are automatically converted to categorical
    >>> cnt_data.data['setting'].dtype.name
    'category'

    Notes
    -----
    Preprocessing Steps (Automatic):
    - Creates a copy of the input DataFrame to avoid side effects
    - Drops rows with missing values in required columns
    - Converts object-type columns (except ID) to categorical for efficiency
    - Renames columns to standardized names:
      * id_col → 'id' (standardized participant identifier)
      * age_col → 'age_cnt' (contact age with _cnt suffix)
      * age_grp_col → 'age_grp_cnt' (contact age group with _cnt suffix)
      * Each strat_var → '{strat_var}_cnt' (stratification vars with _cnt suffix)
    - Adds 'y' column (contact count indicator) initialized to 1 if not present

    Processed DataFrame Structure:
    After preprocessing, the DataFrame will contain:
    - 'id': Standardized participant identifier (renamed from id_col)
    - 'age_cnt' OR 'age_grp_cnt': Contact age with _cnt suffix
    - '{var}_cnt': Each stratification variable with _cnt suffix (if specified)
    - 'y': Contact count indicator (always present)

    Validation Checks:
    - Exactly one of age_col or age_grp_col must be specified
    - Participant IDs can appear multiple times (each row is one contact)
    - Age values (if using age_col) must be non-negative integers or floats
    - No missing values in required columns (removed during preprocessing)
    - Stratification variables are optional but commonly include: setting (home/work/school),
      duration (short/long), physical contact (yes/no), etc.

    Warnings:
    - UserWarning if rows are dropped due to missing values
    - UserWarning if object columns are converted to categorical

    Examples of Column Naming:
    Original columns: ['participant_id', 'contact_age', 'setting']
    Processed columns: ['id', 'age_cnt', 'setting_cnt', 'y']

    See Also
    --------
    ParticipantData : Validated container for participant data
    """

    df_cnt: pd.DataFrame
    id_col: str
    age_col: Optional[str] = None
    age_grp_col: Optional[str] = None
    strat_var_cols: Optional[Union[List[str], str]] = None
    cnt_col: str = "y"

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
            If df_cnt is not a pandas DataFrame.
        ValueError
            If both or neither of age_col and age_grp_col are specified.
            If DataFrame becomes empty after preprocessing.
        """
        # Type validation
        if not isinstance(self.df_cnt, pd.DataFrame):
            raise TypeError(
                f"df_cnt must be a pandas DataFrame, got {type(self.df_cnt).__name__}"
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
                "Use 'age_col' for exact integer contact ages (e.g., 25, 34, 45),\n"
                "or 'age_grp_col' for age groups (e.g., pd.IntervalIndex or categorical)."
            )

        if self.age_col is not None and self.age_grp_col is not None:
            raise ValueError(
                "Cannot specify both 'age_col' and 'age_grp_col' simultaneously.\n"
                f"Currently: age_col='{self.age_col}', age_grp_col='{self.age_grp_col}'\n"
                "Please specify only one age representation."
            )

        # Preprocess the DataFrame
        object.__setattr__(self, "df_cnt", self._preprocess())

        # Perform comprehensive validation
        self.validate()

    def _preprocess(self) -> pd.DataFrame:
        """
        Preprocess contact DataFrame for modeling.

        Performs the following preprocessing steps:
        1. Creates a copy to avoid modifying the original DataFrame
        2. Validates that required columns exist
        3. Drops rows with missing values in required columns
        4. Converts object-type columns to categorical for efficiency
        5. Renames columns to standardized names:
           - id_col → 'id' (only if id_col != 'id')
           - age_col → 'age_cnt'
           - age_grp_col → 'age_grp_cnt'
           - Each strat_var → '{strat_var}_cnt'
        6. Adds 'y' column (contact count indicator) if not present

        Returns
        -------
        pd.DataFrame
            Preprocessed contact DataFrame ready for validation and modeling.

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
        and DataLoader._validate_cnt() to ensure consistency with the package's
        data handling conventions.
        """
        # Step 1: Create a copy to avoid side effects
        df = self.df_cnt.copy()

        # Step 2: Check that required columns exist BEFORE trying to access them
        age_column = self.age_col if self.age_col else self.age_grp_col

        # Check ID column exists
        if self.id_col not in df.columns:
            raise KeyError(
                f"Missing participant ID column '{self.id_col}' in contacts DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Check age column exists
        if age_column not in df.columns:
            col_type = "contact age" if self.age_col else "contact age group"
            raise KeyError(
                f"Missing {col_type} column '{age_column}' in contacts DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Check stratification variables exist
        if self.strat_var_cols:
            missing_vars = [var for var in self.strat_var_cols if var not in df.columns]
            if missing_vars:
                raise KeyError(
                    f"Missing contact stratification variable(s) {missing_vars} in DataFrame.\n"
                    f"Available columns: {list(df.columns)}"
                )

        # Step 3: Identify required columns and check for missing values
        required_cols = [self.id_col, age_column]
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
                f"Dropped {n_dropped} contact record(s) with missing values in required columns.\n"
                f"Missing value counts by column: {cols_with_missing.to_dict()}\n"
                f"Remaining contacts: {n_rows_after}",
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

        # Step 4: Convert object columns to categorical.
        # (excluding the ID column which should remain as-is for merging)
        object_cols = df.select_dtypes(include="object").columns.tolist()
        if self.id_col in object_cols:
            object_cols.remove(self.id_col)

        for col in object_cols:
            if not isinstance(df[col].dtype, pd.CategoricalDtype):
                warnings.warn(
                    f"Converting '{col}' to categorical dtype.",
                    UserWarning,
                )
                df[col] = df[col].astype("category")

        # Step 5: Rename columns to standardized names
        rename_map = {}

        # Rename id_col to 'id' (only if not already 'id')
        if self.id_col != "id":
            rename_map[self.id_col] = "id"

        # Rename age columns with _cnt suffix (only if not already present)
        if self.age_col:
            new_name = "age_cnt" if not self.age_col.endswith("_cnt") else self.age_col
            if self.age_col != new_name:
                rename_map[self.age_col] = new_name
        if self.age_grp_col:
            new_name = (
                "age_grp_cnt"
                if not self.age_grp_col.endswith("_cnt")
                else self.age_grp_col
            )
            if self.age_grp_col != new_name:
                rename_map[self.age_grp_col] = new_name

        # Rename stratification variables with _cnt suffix (only if not already present)
        if self.strat_var_cols:
            for var in self.strat_var_cols:
                new_name = f"{var}_cnt" if not var.endswith("_cnt") else var
                if var != new_name:
                    rename_map[var] = new_name

        df = df.rename(columns=rename_map)

        # Step 6: Add 'y' column (contact count indicator) if not present
        if self.cnt_col not in df.columns:
            df[self.cnt_col] = 1

        return df

    def validate(self) -> None:
        """
        Perform comprehensive validation of contact data.

        Validation checks:
        1. All required columns exist in the DataFrame (post-preprocessing)
        2. Participant ID column exists (contacts must link to participants)
        3. Age column contains valid non-negative numeric values (if age_col specified)
        4. Age group column contains valid pd.IntervalIndex or categorical values (if age_grp_col specified)

        Note: Missing value checks are handled during preprocessing in _preprocess(),
        so this method assumes clean data without NaNs. Unlike ParticipantData,
        duplicate participant IDs are allowed since each contact is a separate row.

        Raises
        ------
        KeyError
            If required columns are missing from the DataFrame.
        ValueError
            If age values are invalid (negative or non-numeric).

        Examples
        --------
        >>> cnt_data = ContactData(df, 'id', age_col='contact_age')
        >>> # Validation happens automatically, but can be called explicitly:
        >>> cnt_data.validate()
        """
        # Check 1: Validate standardized ID column exists
        if "id" not in self.df_cnt.columns:
            raise KeyError(
                f"Missing standardized 'id' column in contacts DataFrame.\n"
                f"Available columns: {list(self.df_cnt.columns)}\n"
                f"Note: Original column '{self.id_col}' should have been renamed to 'id'."
            )

        # Check 2: Validate standardized age column exists
        if self.age_col:
            if "age_cnt" not in self.df_cnt.columns:
                raise KeyError(
                    f"Missing standardized 'age_cnt' column in contacts DataFrame.\n"
                    f"Available columns: {list(self.df_cnt.columns)}\n"
                    f"Note: Original column '{self.age_col}' should have been renamed to 'age_cnt'."
                )
        else:
            if "age_grp_cnt" not in self.df_cnt.columns:
                raise KeyError(
                    f"Missing standardized 'age_grp_cnt' column in contacts DataFrame.\n"
                    f"Available columns: {list(self.df_cnt.columns)}\n"
                    f"Note: Original column '{self.age_grp_col}' should have been renamed to 'age_grp_cnt'."
                )

        # Check 3: Validate standardized stratification variables exist
        if self.strat_var_cols:
            # Calculate expected column names (add _cnt only if not already present)
            expected_vars = [
                f"{var}_cnt" if not var.endswith("_cnt") else var
                for var in self.strat_var_cols
            ]
            missing_vars = [
                var for var in expected_vars if var not in self.df_cnt.columns
            ]
            if missing_vars:
                raise KeyError(
                    f"Missing standardized contact stratification variable(s) {missing_vars} in DataFrame.\n"
                    f"Available columns: {list(self.df_cnt.columns)}\n"
                    f"Note: Original columns {self.strat_var_cols} should have been renamed with '_cnt' suffix."
                )

        # Check 4: Validate age values if using exact ages
        if self.age_col:
            ages = self.df_cnt["age_cnt"]

            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(ages):
                raise ValueError(
                    f"Contact age column 'age_cnt' must contain numeric values.\n"
                    f"Current dtype: {ages.dtype}\n"
                    f"Hint: Convert contact age to integer or float type."
                )

            # Check for negative ages
            if (ages < 0).any():
                negative_indices = self.df_cnt[ages < 0].index[:5].tolist()
                raise ValueError(
                    f"Contact age column 'age_cnt' contains negative values.\n"
                    f"Contact ages must be non-negative. Rows with negative ages: {negative_indices}\n"
                    f"Values: {ages[ages < 0].head().tolist()}"
                )

        # Check 5: Validate age group values if using age groups
        if self.age_grp_col:
            is_categorical = isinstance(
                self.df_cnt["age_grp_cnt"].dtype, pd.CategoricalDtype
            )
            if is_categorical:
                are_intervals = isinstance(
                    self.df_cnt["age_grp_cnt"].cat.categories, pd.IntervalIndex
                )
                if not are_intervals:
                    raise TypeError(
                        f"Column '{"age_grp_cnt"}' must have pd.IntervalIndex categories, "
                        f"got {type(self.df_cnt["age_grp_cnt"].cat.categories)}"
                    )
            else:
                raise TypeError(
                    f"Column '{"age_grp_cnt"}' must be categorical with interval categories. "
                    f"Current type: {self.df_cnt[self.age_grp_col].dtype}"
                )

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the validated contact DataFrame.

        Returns
        -------
        pd.DataFrame
            The validated contact data.

        Examples
        --------
        >>> cnt_data = ContactData(df, 'id', age_col='contact_age')
        >>> validated_df = cnt_data.data
        """
        return self.df_cnt

    @property
    def n(self) -> int:
        """
        Return the number of contact records in the dataset.

        Returns
        -------
        int
            Total number of contacts (rows in the DataFrame).

        Examples
        --------
        >>> cnt_data = ContactData(df, 'id', age_col='contact_age')
        >>> cnt_data.n
        150
        """
        return len(self.df_cnt)

    @property
    def n_part(self) -> int:
        """
        Return the number of unique participants who reported contacts.

        Returns
        -------
        int
            Number of unique participant IDs in the contact data.

        Examples
        --------
        >>> cnt_data = ContactData(df, 'id', age_col='contact_age')
        >>> cnt_data.n_part
        45
        """
        return self.df_cnt["id"].nunique()

    @property
    def age_range(self) -> Tuple[float, float]:
        """
        Return the age range (min, max) of contacts.

        Only available when using age_col (exact ages), not age_grp_col.

        Returns
        -------
        Tuple[float, float]
            Tuple of (minimum_age, maximum_age) of contacts.

        Raises
        ------
        ValueError
            If age_grp_col is used instead of age_col.

        Examples
        --------
        >>> cnt_data = ContactData(df, 'id', age_col='contact_age')
        >>> cnt_data.age_range
        (0, 85)
        """
        if self.age_col is None:
            raise ValueError(
                "age_range is only available when using 'age_col' (exact ages).\n"
                f"Currently using 'age_grp_col': {self.age_grp_col}"
            )

        ages = self.df_cnt["age_cnt"]
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
        >>> cnt_data = ContactData(df, 'id', age_col='contact_age', strat_var_cols=['setting', 'duration'])
        >>> cnt_data.stratification_vars
        ['setting', 'duration']
        """
        return self.strat_var_cols if self.strat_var_cols else []

    def get_strat_vars(self, suffix: bool = False) -> List[str]:
        """
        Return list of stratification variable names, optionally with _cnt suffix.

        Parameters
        ----------
        suffix : bool, default=True
            If True, return stratification variable names with '_cnt' suffix.
            If False, return original stratification variable names.

        Returns
        -------
        List[str]
            List of stratification variable column names.

        Examples
        --------
        >>> cnt_data = ContactData(df, 'id', age_col='contact_age', strat_var_cols=['setting', 'duration'])
        >>> cnt_data.get_strat_vars(suffix=True)
        ['setting_cnt', 'duration_cnt']
        >>> cnt_data.get_strat_vars(suffix=False)
        ['setting', 'duration']
        """
        if not self.strat_var_cols:
            return []

        if suffix:
            return [
                f"{var}_cnt" if not var.endswith("_cnt") else var
                for var in self.strat_var_cols
            ]
        else:
            return [var.removesuffix("_cnt") for var in self.strat_var_cols]

    def get_strat_var_schema(self) -> Dict[str, Dict[str, List[str | int]]]:
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
            var_cnt = f"{var}_cnt" if not var.endswith("_cnt") else var
            if var_cnt in self.df_cnt.columns:
                categories = self.df_cnt[var_cnt].cat.categories.tolist()
                codes = sorted(self.df_cnt[var_cnt].cat.codes.unique().tolist())

                var = (
                    var.removesuffix("_cnt") if var.endswith("_cnt") else var
                )  # Remove suffix
                schema[var] = {"categories": categories, "codes": codes}

        return schema
