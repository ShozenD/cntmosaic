"""
Preprocessing pipeline for population data.

This module provides the transformation logic for PopulationData: column
validation, NaN removal, dtype coercion, aggregation, column standardisation,
and sorting. Keeping this logic here separates the transformation concern from
the data-container and query API defined in PopulationData.
"""

import warnings
from typing import List, Optional

import pandas as pd


def preprocess_population_data(
    df_pop: pd.DataFrame,
    age_col: Optional[str],
    size_col: str,
    age_grp_col: Optional[str],
    strat_var_cols: List[str],
    age_min_col: Optional[str] = None,
    age_max_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate, clean, aggregate, and standardise a raw population DataFrame.

    Parameters
    ----------
    df_pop : pd.DataFrame
        Raw population DataFrame (not yet validated or renamed).
    age_col : Optional[str]
        Column containing population ages as integers. Mutually exclusive with
        age_min_col/age_max_col.
    size_col : str
        Column containing population sizes (counts or proportions).
    age_grp_col : Optional[str]
        Column containing population age groups. If None, no age-group column
        is preserved in the output.
    strat_var_cols : List[str]
        Stratification variable column names (already normalised to a list).
    age_min_col : Optional[str]
        Column containing minimum ages (for age range representation).
    age_max_col : Optional[str]
        Column containing maximum ages (for age range representation).

    Returns
    -------
    pd.DataFrame
        Cleaned and aggregated DataFrame with standardised column names:
        - age_col → "age", size_col → "P", age_grp_col → "age_grp_pop"
        - age_min_col → "age_min", age_max_col → "age_max"

    Raises
    ------
    KeyError
        If any required column is absent from df_pop.
    ValueError
        If df_pop is empty after NaN removal.
        If age or size values are negative or non-numeric.
    """
    required_cols = _check_columns(
        df_pop, age_col, size_col, age_grp_col, strat_var_cols, age_min_col, age_max_col
    )
    return _preprocess(
        df_pop,
        age_col,
        size_col,
        age_grp_col,
        strat_var_cols,
        required_cols,
        age_min_col,
        age_max_col,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_columns(
    df: pd.DataFrame,
    age_col: Optional[str],
    size_col: str,
    age_grp_col: Optional[str],
    strat_var_cols: List[str],
    age_min_col: Optional[str],
    age_max_col: Optional[str],
) -> List[str]:
    """
    Assert required columns exist. Return the list of required column names.
    """
    if age_col and age_col not in df.columns:
        raise KeyError(
            f"Missing population age column '{age_col}' in DataFrame.\n"
            f"  Available columns: {list(df.columns)}"
        )

    if age_min_col or age_max_col:
        for col in [c for c in [age_min_col, age_max_col] if c]:
            if col not in df.columns:
                raise KeyError(
                    f"Missing population age column '{col}' in DataFrame.\n"
                    f"  Available columns: {list(df.columns)}"
                )

    if size_col not in df.columns:
        raise KeyError(
            f"Missing population size column '{size_col}' in DataFrame.\n"
            f"  Available columns: {list(df.columns)}"
        )

    if age_grp_col and age_grp_col not in df.columns:
        raise KeyError(
            f"Missing population age group column '{age_grp_col}' in DataFrame.\n"
            f"  Available columns: {list(df.columns)}"
        )

    if strat_var_cols:
        missing_vars = [var for var in strat_var_cols if var not in df.columns]
        if missing_vars:
            raise KeyError(
                f"Missing population stratification variable(s) {missing_vars} in DataFrame.\n"
                f"  Available columns: {list(df.columns)}"
            )

    required_cols = []
    if age_col:
        required_cols.append(age_col)
    if age_min_col:
        required_cols.append(age_min_col)
    if age_max_col:
        required_cols.append(age_max_col)
    required_cols.append(size_col)
    if age_grp_col:
        required_cols.append(age_grp_col)
    if strat_var_cols:
        required_cols.extend(strat_var_cols)
    return required_cols


def _preprocess(
    df_pop: pd.DataFrame,
    age_col: Optional[str],
    size_col: str,
    age_grp_col: Optional[str],
    strat_var_cols: List[str],
    required_cols: List[str],
    age_min_col: Optional[str],
    age_max_col: Optional[str],
) -> pd.DataFrame:
    """
    Copy, coerce dtypes, clean, validate, aggregate, rename, and sort.

    Assumes required columns have already been verified by _check_columns.
    """
    df = df_pop.copy()

    # --- dtype coercion: object → category -----------------------------------
    object_cols = df.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            warnings.warn(
                f"Converting '{col}' to categorical dtype.",
                UserWarning,
                stacklevel=5,
            )
            df[col] = df[col].astype("category")

    # --- NaN removal ---------------------------------------------------------
    n_rows_before = len(df)
    missing_counts = df[required_cols].isnull().sum()
    has_missing = missing_counts.sum() > 0

    df = df[required_cols].dropna().copy()

    if has_missing:
        n_rows_after = len(df)
        n_dropped = n_rows_before - n_rows_after
        cols_with_missing = missing_counts[missing_counts > 0]
        warnings.warn(
            f"Dropped {n_dropped} row(s) with missing values in required columns.\n"
            f"  Missing value counts by column: {cols_with_missing.to_dict()}\n"
            f"  Remaining population records: {n_rows_after}",
            UserWarning,
            stacklevel=5,
        )

    if df.empty:
        raise ValueError(
            "DataFrame is empty after removing rows with missing values.\n"
            f"  Required columns: {required_cols}\n"
            f"  Missing value counts: {missing_counts.to_dict()}"
        )

    # --- validate age and size values are non-negative -----------------------
    if age_col:
        ages = df[age_col]
        if not pd.api.types.is_numeric_dtype(ages):
            raise ValueError(
                f"Population age column '{age_col}' must contain numeric values.\n"
                f"  Current dtype: {ages.dtype}\n"
                f"  Hint: Convert age to integer or float type."
            )
        if (ages < 0).any():
            negative_indices = df[ages < 0].index[:5].tolist()
            raise ValueError(
                f"Population age column '{age_col}' contains negative values.\n"
                f"  Ages must be non-negative. Rows with negative ages: {negative_indices}\n"
                f"  Values: {ages[ages < 0].head().tolist()}"
            )

    if age_min_col and age_max_col:
        for col in (age_min_col, age_max_col):
            vals = df[col]
            if not pd.api.types.is_numeric_dtype(vals):
                raise ValueError(
                    f"Population age column '{col}' must contain numeric values.\n"
                    f"  Current dtype: {vals.dtype}\n"
                    f"  Hint: Convert age to integer or float type."
                )
            if (vals < 0).any():
                negative_indices = df[vals < 0].index[:5].tolist()
                raise ValueError(
                    f"Population age column '{col}' contains negative values.\n"
                    f"  Ages must be non-negative. Rows with negative ages: {negative_indices}\n"
                    f"  Values: {vals[vals < 0].head().tolist()}"
                )
        if (df[age_min_col] > df[age_max_col]).any():
            raise ValueError(
                f"Population age_min_col '{age_min_col}' must be <= age_max_col '{age_max_col}'.\n"
                "  Found rows where age_min > age_max."
            )

    sizes = df[size_col]
    if not pd.api.types.is_numeric_dtype(sizes):
        raise ValueError(
            f"Population size column '{size_col}' must contain numeric values.\n"
            f"  Current dtype: {sizes.dtype}\n"
            f"  Hint: Convert size to numeric type."
        )

    if (sizes < 0).any():
        negative_indices = df[sizes < 0].index[:5].tolist()
        raise ValueError(
            f"Population size column '{size_col}' contains negative values.\n"
            f"  Population sizes must be non-negative. Rows with negative sizes: {negative_indices}\n"
            f"  Values: {sizes[sizes < 0].head().tolist()}"
        )

    # --- aggregate population sizes by age (and strat vars) ------------------
    if age_col:
        groupby_cols = [age_col]
        if age_grp_col:
            groupby_cols.append(age_grp_col)
    elif age_grp_col:
        groupby_cols = [age_grp_col]
    else:
        groupby_cols = [age_min_col, age_max_col]

    if strat_var_cols:
        groupby_cols.extend(strat_var_cols)

    n_unique_groups = df.groupby(groupby_cols, observed=False).ngroups
    if n_unique_groups < len(df):
        n_aggregated = len(df) - n_unique_groups
        warnings.warn(
            f"Aggregating population data: {len(df)} rows → {n_unique_groups} unique age groups.\n"
            f"  Population sizes are summed within each age"
            f"{' and stratification' if strat_var_cols else ''} group.\n"
            f"  Number of rows aggregated: {n_aggregated}",
            UserWarning,
            stacklevel=5,
        )

    df = df.groupby(groupby_cols, as_index=False, observed=False)[size_col].sum()

    # --- column renaming -----------------------------------------------------
    rename_map = {size_col: "P"}
    if age_col:
        rename_map[age_col] = "age"
    if age_min_col:
        rename_map[age_min_col] = "age_min"
    if age_max_col:
        rename_map[age_max_col] = "age_max"
    if age_grp_col:
        rename_map[age_grp_col] = "age_grp_pop"

    df = df.rename(columns=rename_map)

    # --- sort by age (and strat vars) ----------------------------------------
    if age_col:
        sort_cols = ["age"]
    elif age_grp_col:
        sort_cols = ["age_grp_pop"]
    else:
        sort_cols = ["age_min", "age_max"]

    if strat_var_cols:
        sort_cols.extend(strat_var_cols)

    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df
