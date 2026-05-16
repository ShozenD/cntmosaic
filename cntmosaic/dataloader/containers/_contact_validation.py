"""
Domain validation for contact survey data.

This module provides the validation logic for ContactData: checking
numeric constraints on ages and categorical structure of age groups.
Keeping this logic here separates the validation concern from the
data-container and query API defined in ContactData.
"""

from typing import List, Optional

import pandas as pd


def validate_contact_data(
    data: pd.DataFrame,
    age_col: Optional[str],
    age_min_col: Optional[str],
    age_max_col: Optional[str],
    age_grp_col: Optional[str],
    strat_var_cols: List[str],
) -> None:
    """
    Run all domain validation checks on a preprocessed contact DataFrame.

    Assumes the DataFrame has already been cleaned (no NaN rows) and that
    column names have been standardised by preprocess_contact_data.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed contact DataFrame with standardised column names.
    age_col : Optional[str]
        Original age column name; truthy when exact ages are in use.
    age_min_col : Optional[str]
        Original age_min column name; truthy when age ranges are in use.
    age_max_col : Optional[str]
        Original age_max column name; truthy when age ranges are in use.
    age_grp_col : Optional[str]
        Original age-group column name; truthy when age groups are in use.
    strat_var_cols : List[str]
        Original stratification variable column names (already normalised to
        a list); used only to verify expected renamed columns are present.

    Raises
    ------
    ValueError
        If age values are invalid (negative or non-numeric).
    TypeError
        If age-group column is not categorical with pd.IntervalIndex categories.
    """
    if age_col:
        _validate_ages(data)

    if age_min_col and age_max_col:
        _validate_age_min_max(data)

    if age_grp_col:
        _validate_age_groups(data, age_grp_col)


# ---------------------------------------------------------------------------
# Internal helpers — one responsibility each
# ---------------------------------------------------------------------------


def _validate_ages(data: pd.DataFrame) -> None:
    """Raise ValueError if 'age_cnt' is non-numeric or contains negative values."""
    ages = data["age_cnt"]

    if not pd.api.types.is_numeric_dtype(ages):
        raise ValueError(
            f"Contact age column 'age_cnt' must contain numeric values.\n"
            f"  Current dtype: {ages.dtype}\n"
            f"  Hint: convert contact age to integer or float type."
        )

    if (ages < 0).any():
        negative_indices = data[ages < 0].index[:5].tolist()
        raise ValueError(
            f"Contact age column 'age_cnt' contains negative values.\n"
            f"  Contact ages must be non-negative.\n"
            f"  Rows with negative ages: {negative_indices}\n"
            f"  Values: {ages[ages < 0].head().tolist()}"
        )


def _validate_age_min_max(data: pd.DataFrame) -> None:
    """Raise ValueError if 'age_min_cnt'/'age_max_cnt' are non-numeric or contain negative values."""
    for col in ("age_min_cnt", "age_max_cnt"):
        vals = data[col]
        if not pd.api.types.is_numeric_dtype(vals):
            raise ValueError(
                f"Contact age column '{col}' must contain numeric values.\n"
                f"  Current dtype: {vals.dtype}\n"
                f"  Hint: convert contact age to integer or float type."
            )
        if (vals < 0).any():
            negative_indices = data[vals < 0].index[:5].tolist()
            raise ValueError(
                f"Contact age column '{col}' contains negative values.\n"
                f"  Contact ages must be non-negative.\n"
                f"  Rows with negative values: {negative_indices}\n"
                f"  Values: {vals[vals < 0].head().tolist()}"
            )


def _validate_age_groups(data: pd.DataFrame, age_grp_col: str) -> None:
    """Raise TypeError if 'age_grp_cnt' is not categorical with IntervalIndex categories."""
    col = data["age_grp_cnt"]

    if not isinstance(col.dtype, pd.CategoricalDtype):
        raise TypeError(
            f"Column '{age_grp_col}' must be categorical with interval categories.\n"
            f"  Current type: {col.dtype}"
        )

    if not isinstance(col.cat.categories, pd.IntervalIndex):
        raise TypeError(
            f"Column '{age_grp_col}' must have pd.IntervalIndex categories.\n"
            f"  Got: {type(col.cat.categories)}"
        )
