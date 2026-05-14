"""
Preprocessing pipeline for stratification population data.

This module provides the transformation logic for StratificationData: column
validation and dtype coercion for stratification variable columns.
Keeping this logic here separates the transformation concern from the
data-container and query API defined in StratificationData.
"""

import warnings
from typing import List

import pandas as pd


def preprocess_stratification_data(
    data: pd.DataFrame,
    age_col: str,
    strat_var_cols: List[str],
    prop_col: str,
) -> pd.DataFrame:
    """
    Validate and clean a raw stratification DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Raw stratification DataFrame (not yet validated or converted).
    age_col : str
        Column containing age values.
    strat_var_cols : List[str]
        Stratification variable column names (already normalised to a list).
    prop_col : str
        Column containing population proportions.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with stratification columns coerced to categorical.

    Raises
    ------
    ValueError
        If any required stratification column is absent from data.
    """
    _check_columns(data, strat_var_cols)
    return _preprocess(data, strat_var_cols)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_columns(
    df: pd.DataFrame,
    strat_var_cols: List[str],
) -> None:
    """
    Assert required stratification columns exist in df.

    Raises ValueError if any column in strat_var_cols is missing.
    """
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
