"""
Domain validation for stratification population data.

This module provides the validation logic for StratificationData: checking
that required columns are present, that proportion values are within [0, 1],
and that proportions sum to 1.0 within each age group.
Keeping this logic here separates the validation concern from the
data-container and query API defined in StratificationData.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def validate_stratification_data(
    data: pd.DataFrame,
    age_col: Optional[str],
    strat_var_cols: List[str],
    prop_col: str,
    age_min_col: Optional[str] = None,
    age_max_col: Optional[str] = None,
) -> None:
    """
    Run all domain validation checks on a preprocessed stratification DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed stratification DataFrame.
    age_col : Optional[str]
        Column containing age values. Mutually exclusive with age_min_col/age_max_col.
    strat_var_cols : List[str]
        Stratification variable column names.
    prop_col : str
        Column containing population proportions.
    age_min_col : Optional[str]
        Column containing minimum ages (for age range representation).
    age_max_col : Optional[str]
        Column containing maximum ages (for age range representation).

    Raises
    ------
    ValueError
        If required columns are missing.
        If proportion values are outside [0, 1].
        If proportions do not sum to 1.0 within each age group (tolerance 1e-6).
    """
    _validate_required_columns(
        data, age_col, strat_var_cols, prop_col, age_min_col, age_max_col
    )
    _validate_prop_range(data, prop_col)
    _validate_prop_sums(data, age_col, prop_col, age_min_col, age_max_col)


# ---------------------------------------------------------------------------
# Internal helpers — one responsibility each
# ---------------------------------------------------------------------------


def _validate_required_columns(
    data: pd.DataFrame,
    age_col: Optional[str],
    strat_var_cols: List[str],
    prop_col: str,
    age_min_col: Optional[str],
    age_max_col: Optional[str],
) -> None:
    """Raise ValueError if any required column is absent from data."""
    age_cols = []
    if age_col:
        age_cols = [age_col]
    elif age_min_col and age_max_col:
        age_cols = [age_min_col, age_max_col]

    required_cols = age_cols + strat_var_cols + [prop_col]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in population proportion data: {missing}\n"
            f"  Required: {required_cols}\n"
            f"  Available: {list(data.columns)}"
        )


def _validate_prop_range(
    data: pd.DataFrame,
    prop_col: str,
) -> None:
    """Raise ValueError if any proportion value is outside [0, 1]."""
    props = data[prop_col]
    if (props < 0).any() or (props > 1).any():
        invalid_indices = data[(props < 0) | (props > 1)].index
        raise ValueError(
            f"Population proportions must be in range [0, 1]. "
            f"Found invalid values at indices: {list(invalid_indices)}\n"
            f"  Invalid values: {props[invalid_indices].to_dict()}"
        )


def _validate_prop_sums(
    data: pd.DataFrame,
    age_col: Optional[str],
    prop_col: str,
    age_min_col: Optional[str],
    age_max_col: Optional[str],
) -> None:
    """Raise ValueError if proportions do not sum to 1.0 within each age group."""
    if age_col:
        group_sums = data.groupby(age_col, observed=False)[prop_col].sum()
        bad_groups = group_sums[np.abs(group_sums - 1.0) > 1e-6]
        if not bad_groups.empty:
            raise ValueError(
                f"Population proportions must sum to 1.0 within each age group (tolerance: 1e-6).\n"
                f"  Ages with invalid sums: {list(bad_groups.index)}\n"
                f"  Actual sums: {bad_groups.to_dict()}\n"
                f"  Hint: Use StratificationData.from_counts() to automatically compute proportions from counts."
            )
    elif age_min_col and age_max_col:
        group_sums = data.groupby([age_min_col, age_max_col], observed=False)[prop_col].sum()
        bad_groups = group_sums[np.abs(group_sums - 1.0) > 1e-6]
        if not bad_groups.empty:
            raise ValueError(
                f"Population proportions must sum to 1.0 within each age range (tolerance: 1e-6).\n"
                f"  Age ranges with invalid sums: {list(bad_groups.index)}\n"
                f"  Actual sums: {bad_groups.to_dict()}\n"
                f"  Hint: Use StratificationData.from_counts() to automatically compute proportions from counts."
            )
