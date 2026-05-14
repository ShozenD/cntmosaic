"""
Preprocessing pipeline for participant survey data.

This module provides the transformation logic for ParticipantData: column
validation, NaN removal, dtype coercion, and column standardisation.
Keeping this logic here separates the transformation concern from the
data-container and query API defined in ParticipantData.
"""

import warnings
from typing import List, Optional

import pandas as pd


def preprocess_participant_data(
    df_part: pd.DataFrame,
    id_col: str,
    age_col: Optional[str],
    age_grp_col: Optional[str],
    strat_var_cols: List[str],
    repeat_col: Optional[str],
    amb_cnt_col: Optional[str],
) -> pd.DataFrame:
    """
    Validate, clean, and standardise a raw participant DataFrame.

    Parameters
    ----------
    df_part : pd.DataFrame
        Raw participant DataFrame (not yet validated or renamed).
    id_col : str
        Column containing unique participant identifiers.
    age_col : Optional[str]
        Column containing exact participant ages.
    age_grp_col : Optional[str]
        Column containing participant age groups (IntervalIndex).
    strat_var_cols : List[str]
        Stratification variable column names (already normalised to a list).
    repeat_col : Optional[str]
        Column indicating repeat interviews.
    amb_cnt_col : Optional[str]
        Column containing ambiguous contact counts. If None, no column is
        created — the field stays None on the parent ParticipantData.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised column names.

    Raises
    ------
    KeyError
        If any required column is absent from df_part.
    ValueError
        If df_part is empty after NaN removal.
    """
    required_cols = _check_columns(
        df_part, id_col, age_col, age_grp_col, strat_var_cols, repeat_col, amb_cnt_col
    )
    return _preprocess(
        df_part, id_col, age_col, age_grp_col, strat_var_cols, repeat_col, required_cols
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_columns(
    df: pd.DataFrame,
    id_col: str,
    age_col: Optional[str],
    age_grp_col: Optional[str],
    strat_var_cols: List[str],
    repeat_col: Optional[str],
    amb_cnt_col: Optional[str],
) -> List[str]:
    """
    Assert required columns exist. Return the list of required column names.

    Returning the list (rather than storing it as instance state) makes the
    dependency on this check explicit and testable.
    """

    def _cols_display(columns):
        cols = list(columns)
        return cols[:8] + (["..."] if len(cols) > 8 else [])

    age_column = age_col if age_col else age_grp_col

    if id_col not in df.columns:
        raise KeyError(
            f"Missing participant ID column '{id_col}' in DataFrame.\n"
            f"  Available columns: {_cols_display(df.columns)}"
        )

    if age_column not in df.columns:
        col_type = "age" if age_col else "age group"
        raise KeyError(
            f"Missing participant {col_type} column '{age_column}' in DataFrame.\n"
            f"  Available columns: {_cols_display(df.columns)}"
        )

    if strat_var_cols:
        missing_vars = [v for v in strat_var_cols if v not in df.columns]
        if missing_vars:
            raise KeyError(
                f"strat_var_cols '{strat_var_cols}' is specified but missing '{missing_vars}' in DataFrame.\n"
                f"  Available columns: {_cols_display(df.columns)}"
            )

    if repeat_col and repeat_col not in df.columns:
        raise KeyError(
            f"repeat_col '{repeat_col}' is specified but missing in DataFrame.\n"
            f"  Available columns: {_cols_display(df.columns)}"
        )

    if amb_cnt_col and amb_cnt_col not in df.columns:
        raise KeyError(
            f"amb_cnt_col '{amb_cnt_col}' is specified but missing in DataFrame.\n"
            f"  Available columns: {_cols_display(df.columns)}"
        )

    required_cols = [id_col, age_column]
    if strat_var_cols:
        required_cols.extend(strat_var_cols)
    if repeat_col:
        required_cols.append(repeat_col)
    if amb_cnt_col:
        required_cols.append(amb_cnt_col)
    return required_cols


def _preprocess(
    df_part: pd.DataFrame,
    id_col: str,
    age_col: Optional[str],
    age_grp_col: Optional[str],
    strat_var_cols: List[str],
    repeat_col: Optional[str],
    required_cols: List[str],
) -> pd.DataFrame:
    """
    Copy, clean, coerce dtypes, and rename columns.

    No default 'z' column is added here. If amb_cnt_col was provided it will
    already be included in required_cols and renamed alongside the other columns.
    """
    df = df_part.copy()

    # --- NaN removal ---------------------------------------------------------
    n_rows_before = len(df)
    missing_counts = df[required_cols].isnull().sum()
    has_missing = missing_counts.sum() > 0

    df = df[required_cols].dropna().copy()

    if has_missing:
        n_dropped = n_rows_before - len(df)
        cols_with_missing = missing_counts[missing_counts > 0]
        warnings.warn(
            f"Dropped {n_dropped} row(s) with missing values in required columns.\n"
            f"  Columns affected: {cols_with_missing.to_dict()}\n"
            f"  Remaining participants: {len(df)}",
            UserWarning,
            stacklevel=5,
        )

    if df.empty:
        raise ValueError(
            "DataFrame is empty after removing rows with missing values.\n"
            f"  Required columns: {required_cols}\n"
            f"  Missing value counts: {missing_counts.to_dict()}"
        )

    # --- dtype coercion: object → category (excluding id column) ------------
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if id_col in object_cols:
        object_cols.remove(id_col)

    for col in object_cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            warnings.warn(
                f"Converting '{col}' to categorical dtype.",
                UserWarning,
                stacklevel=5,
            )
            df[col] = df[col].astype("category")

    # --- column renaming -----------------------------------------------------
    rename_map = {}

    if id_col != "id":
        rename_map[id_col] = "id"

    if age_col and not age_col.endswith("_part"):
        rename_map[age_col] = "age_part"

    if age_grp_col and not age_grp_col.endswith("_part"):
        rename_map[age_grp_col] = "age_grp_part"

    for var in strat_var_cols:
        if not var.endswith("_part"):
            rename_map[var] = f"{var}_part"

    if repeat_col and not repeat_col.endswith("_part"):
        rename_map[repeat_col] = "repeat_part"

    return df.rename(columns=rename_map)
