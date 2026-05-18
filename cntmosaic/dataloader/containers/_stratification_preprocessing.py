"""
Preprocessing pipeline for stratification population data.

This module provides the transformation logic for StratificationData: column
validation and dtype coercion for stratification variable columns.
Keeping this logic here separates the transformation concern from the
data-container and query API defined in StratificationData.
"""

import warnings
from typing import List, Optional

import pandas as pd


def preprocess_stratification_data(
    data: pd.DataFrame,
    age_col: Optional[str],
    strat_var_cols: List[str],
    prop_col: str,
    age_min_col: Optional[str] = None,
    age_max_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate and clean a raw stratification DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Raw stratification DataFrame (not yet validated or converted).
    age_col : Optional[str]
        Column containing age values. Mutually exclusive with age_min_col/age_max_col.
    strat_var_cols : List[str]
        Stratification variable column names (already normalised to a list).
    prop_col : str
        Column containing population proportions.
    age_min_col : Optional[str]
        Column containing minimum ages (for age range representation).
    age_max_col : Optional[str]
        Column containing maximum ages (for age range representation).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with stratification columns coerced to categorical.

    Raises
    ------
    ValueError
        If any required stratification column is absent from data.
    """
    _check_columns(data, age_col, strat_var_cols, age_min_col, age_max_col)
    return _preprocess(data, strat_var_cols)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_columns(
    df: pd.DataFrame,
    age_col: Optional[str],
    strat_var_cols: List[str],
    age_min_col: Optional[str],
    age_max_col: Optional[str],
) -> None:
    """
    Assert required columns exist in df.

    Raises ValueError if any required column is missing.
    """
    if age_col and age_col not in df.columns:
        raise ValueError(
            f"Age column '{age_col}' not found in data.\n"
            f"  Available columns: {list(df.columns)}"
        )

    for col in [c for c in [age_min_col, age_max_col] if c]:
        if col not in df.columns:
            raise ValueError(
                f"Age column '{col}' not found in data.\n"
                f"  Available columns: {list(df.columns)}"
            )

    missing = [col for col in strat_var_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Stratification columns not found in data: {missing}\n"
            f"  Available columns: {list(df.columns)}"
        )


def _preprocess(
    data: pd.DataFrame,
    strat_var_cols: List[str],
) -> pd.DataFrame:
    """
    Copy data and coerce stratification columns to categorical dtype.
    """
    df = data.copy()

    for col in strat_var_cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            warnings.warn(
                f"Converting '{col}' to categorical dtype.",
                UserWarning,
                stacklevel=5,
            )
            df[col] = df[col].astype("category")

    return df
