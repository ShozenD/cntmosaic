from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ._contact_preprocessing import preprocess_contact_data
from ._contact_validation import validate_contact_data


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
    data : pd.DataFrame
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
        Name of the column for contact counts/indicators. If not present in data,
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
        If data is not a pandas DataFrame.

    Examples
    --------
    >>> # Basic usage with individual contact ages
    >>> df = pd.DataFrame({
    ...     'participant_id': [1, 1, 2, 3],
    ...     'contact_age': [30, 45, 25, 60],
    ...     'setting': ['home', 'work', 'home', 'other']
    ... })
    >>> cnt_data = ContactData(
    ...     data=df,
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
    ...     data=df,
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

    data: pd.DataFrame
    id_col: str
    age_col: Optional[str] = None
    age_min_col: Optional[str] = None
    age_max_col: Optional[str] = None
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
            If data is not a pandas DataFrame.
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

        # Validate mutual exclusivity of age specifications.
        # Exactly one of the three forms must be used:
        #   (a) age_col only
        #   (b) age_grp_col only
        #   (c) age_min_col + age_max_col (both required together)
        _has_exact = self.age_col is not None
        _has_grp = self.age_grp_col is not None
        _has_range = self.age_min_col is not None or self.age_max_col is not None

        _n_forms = sum([_has_exact, _has_grp, _has_range])

        if _n_forms == 0:
            raise ValueError(
                "Must specify exactly one age representation:\n"
                "  'age_col' for exact integer contact ages (e.g., 25, 34, 45),\n"
                "  'age_grp_col' for age groups (e.g., pd.IntervalIndex or categorical),\n"
                "  or both 'age_min_col' and 'age_max_col' for age ranges."
            )
        if _n_forms > 1:
            raise ValueError(
                "Age specification forms are mutually exclusive — provide exactly one:\n"
                "  'age_col', 'age_grp_col', or 'age_min_col'/'age_max_col'.\n"
                f"  Got: age_col={self.age_col!r}, age_grp_col={self.age_grp_col!r}, "
                f"age_min_col={self.age_min_col!r}, age_max_col={self.age_max_col!r}"
            )
        if _has_range and (self.age_min_col is None or self.age_max_col is None):
            raise ValueError(
                "Both 'age_min_col' and 'age_max_col' must be specified together.\n"
                f"  Got: age_min_col={self.age_min_col!r}, age_max_col={self.age_max_col!r}"
            )

        # Delegate column validation, NaN removal, dtype coercion, and renaming
        object.__setattr__(
            self,
            "data",
            preprocess_contact_data(
                self.data,
                self.id_col,
                self.age_col,
                self.age_min_col,
                self.age_max_col,
                self.age_grp_col,
                self.strat_var_cols,
                self.cnt_col,
            ),
        )

        # Perform domain validation on the cleaned data
        self.validate()

    def validate(self) -> None:
        """
        Perform comprehensive validation of contact data.

        Validation checks:
        1. Age column contains valid non-negative numeric values (if age_col specified)
        2. Age min/max columns contain valid non-negative numeric values (if age_min_col/age_max_col specified)
        3. Age group column contains valid pd.IntervalIndex or categorical values (if age_grp_col specified)

        Note: Missing value checks are handled during preprocessing in preprocess_contact_data(),
        so this method assumes clean data without NaNs. Unlike ParticipantData,
        duplicate participant IDs are allowed since each contact is a separate row.

        Raises
        ------
        ValueError
            If age values are invalid (negative or non-numeric).
        TypeError
            If age-group column is not categorical with pd.IntervalIndex categories.

        Examples
        --------
        >>> cnt_data = ContactData(df, 'id', age_col='contact_age')
        >>> # Validation happens automatically, but can be called explicitly:
        >>> cnt_data.validate()
        """
        validate_contact_data(
            self.data,
            self.age_col,
            self.age_min_col,
            self.age_max_col,
            self.age_grp_col,
            self.strat_var_cols,
        )

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
        return len(self.data)

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
        return self.data["id"].nunique()

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

        ages = self.data["age_cnt"]
        return (float(ages.min()), float(ages.max()))

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
        if self.strat_var_cols is None:
            return schema

        else:
            for var in self.strat_var_cols:
                var_cnt = f"{var}_cnt" if not var.endswith("_cnt") else var
                if var_cnt in self.data.columns:
                    categories = self.data[var_cnt].cat.categories.tolist()
                    codes = sorted(self.data[var_cnt].cat.codes.unique().tolist())

                    var = (
                        var.removesuffix("_cnt") if var.endswith("_cnt") else var
                    )  # Remove suffix
                    schema[var] = {"categories": categories, "codes": codes}

            return schema
