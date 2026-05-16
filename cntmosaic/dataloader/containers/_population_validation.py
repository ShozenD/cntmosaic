"""
Domain validation for population data.

This module provides the validation logic for PopulationData: checking that
required standardised columns exist, that age values are non-negative numeric,
that population sizes are non-negative numeric, and warning about zero sizes.
Keeping this logic here separates the validation concern from the
data-container and query API defined in PopulationData.
"""

import warnings
from typing import List, Optional

import pandas as pd


def validate_population_data(
    data: pd.DataFrame,
    age_col: Optional[str],
    size_col: str,
    strat_var_cols: List[str],
    age_min_col: Optional[str] = None,
    age_max_col: Optional[str] = None,
) -> None:
    """
    Run all domain validation checks on a preprocessed population DataFrame.

    Assumes the DataFrame has already been cleaned (no NaN rows), aggregated,
    and that column names have been standardised by preprocess_population_data.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed population DataFrame with standardised column names
        ("age" or "age_min"/"age_max", "P", and optionally "age_grp_pop" and strat_var_cols).
    age_col : Optional[str]
        Original age column name; truthy when exact ages are in use.
    size_col : str
        Original size column name (used only for error context).
    strat_var_cols : List[str]
        Stratification variable column names (already normalised to a list).
    age_min_col : Optional[str]
        Original age_min column name; truthy when age ranges are in use.
    age_max_col : Optional[str]
        Original age_max column name; truthy when age ranges are in use.

    Raises
    ------
    KeyError
        If standard columns "age" or "P" (or "age_min"/"age_max") are missing.
        If any stratification variable is missing from data.
    ValueError
        If the DataFrame is empty.
        If age or population-size values are invalid (negative or non-numeric).
    """
    _validate_required_columns(data, strat_var_cols, age_col, age_min_col, age_max_col)
    _validate_not_empty(data)
    if age_col:
        _validate_ages(data)
    if age_min_col and age_max_col:
        _validate_age_min_max(data)
    _validate_sizes(data)
    _validate_zero_sizes(data, age_col, age_min_col)


# ---------------------------------------------------------------------------
# Internal helpers — one responsibility each
# ---------------------------------------------------------------------------


def _validate_required_columns(
    data: pd.DataFrame,
    strat_var_cols: List[str],
    age_col: Optional[str],
    age_min_col: Optional[str],
    age_max_col: Optional[str],
) -> None:
    """Raise KeyError if standard age/P columns (or strat vars) are missing."""
    if age_col and "age" not in data.columns:
        raise KeyError(
            "Missing 'age' column in processed population DataFrame.\n"
            f"  Available columns: {list(data.columns)}\n"
            "  This should not happen - please report this bug."
        )

    if age_min_col:
        for col in ("age_min", "age_max"):
            if col not in data.columns:
                raise KeyError(
                    f"Missing '{col}' column in processed population DataFrame.\n"
                    f"  Available columns: {list(data.columns)}\n"
                    "  This should not happen - please report this bug."
                )

    if "P" not in data.columns:
        raise KeyError(
            "Missing 'P' (population size) column in processed population DataFrame.\n"
            f"  Available columns: {list(data.columns)}\n"
            "  This should not happen - please report this bug."
        )

    if strat_var_cols:
        missing_vars = [var for var in strat_var_cols if var not in data.columns]
        if missing_vars:
            raise KeyError(
                f"Missing population stratification variable(s) {missing_vars} in processed DataFrame.\n"
                f"  Available columns: {list(data.columns)}"
            )


def _validate_not_empty(data: pd.DataFrame) -> None:
    """Raise ValueError if the DataFrame has no rows."""
    if len(data) == 0:
        raise ValueError(
            "Population DataFrame is empty after preprocessing.\n"
            "  At least one age group with population data is required."
        )


def _validate_ages(data: pd.DataFrame) -> None:
    """Raise ValueError if 'age' is non-numeric or contains negative values."""
    ages = data["age"]

    if not pd.api.types.is_numeric_dtype(ages):
        raise ValueError(
            f"Population age column must contain numeric values.\n"
            f"  Current dtype: {ages.dtype}"
        )

    if (ages < 0).any():
        raise ValueError(
            "Population age column contains negative values after preprocessing.\n"
            "  This should not happen - please report this bug."
        )


def _validate_age_min_max(data: pd.DataFrame) -> None:
    """Raise ValueError if 'age_min'/'age_max' are non-numeric, negative, or inverted."""
    for col in ("age_min", "age_max"):
        vals = data[col]
        if not pd.api.types.is_numeric_dtype(vals):
            raise ValueError(
                f"Population age column '{col}' must contain numeric values.\n"
                f"  Current dtype: {vals.dtype}"
            )
        if (vals < 0).any():
            raise ValueError(
                f"Population age column '{col}' contains negative values after preprocessing.\n"
                "  This should not happen - please report this bug."
            )

    if (data["age_min"] > data["age_max"]).any():
        raise ValueError(
            "Population age_min must be <= age_max for all rows.\n"
            "  Found rows where age_min > age_max."
        )


def _validate_sizes(data: pd.DataFrame) -> None:
    """Raise ValueError if 'P' is non-numeric or contains negative values."""
    sizes = data["P"]

    if not pd.api.types.is_numeric_dtype(sizes):
        raise ValueError(
            f"Population size column must contain numeric values.\n"
            f"  Current dtype: {sizes.dtype}"
        )

    if (sizes < 0).any():
        raise ValueError(
            "Population size column contains negative values after preprocessing.\n"
            "  This should not happen - please report this bug."
        )


def _validate_zero_sizes(
    data: pd.DataFrame,
    age_col: Optional[str],
    age_min_col: Optional[str],
) -> None:
    """Warn if any age groups have zero population size."""
    sizes = data["P"]
    if (sizes == 0).any():
        n_zero = int((sizes == 0).sum())
        if age_col:
            zero_ages = data[sizes == 0]["age"].head(5).tolist()
        elif age_min_col:
            zero_ages = data[sizes == 0][["age_min", "age_max"]].head(5).values.tolist()
        else:
            zero_ages = []
        warnings.warn(
            f"Found {n_zero} age group(s) with zero population size.\n"
            f"  Ages with zero population: {zero_ages}\n"
            f"  This may cause issues in contact matrix estimation.",
            UserWarning,
            stacklevel=5,
        )
