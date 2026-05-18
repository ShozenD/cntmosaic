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
    age_min_col: Optional[str],
    age_max_col: Optional[str],
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
    age_min_col : Optional[str]
        Column containing minimum age of participants (for age ranges).
    age_max_col : Optional[str]
        Column containing maximum age of participants (for age ranges).
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
        df_part,
        id_col,
        age_col,
        age_min_col,
        age_max_col,
        age_grp_col,
        strat_var_cols,
        repeat_col,
        amb_cnt_col,
    )
    return _preprocess(
        df_part,
        id_col,
        age_col,
        age_min_col,
        age_max_col,
        age_grp_col,
        strat_var_cols,
        repeat_col,
        required_cols,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_columns(
    df: pd.DataFrame,
    id_col: str,
    age_col: Optional[str],
    age_min_col: Optional[str],
    age_max_col: Optional[str],
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

    age_column = list[str]()
    if age_col:
        age_column = [age_col]
    elif age_min_col and age_max_col:
        age_column = [age_min_col, age_max_col]
    elif age_grp_col:
        age_column = [age_grp_col]

    if id_col not in df.columns:
        raise KeyError(
            f"Missing participant ID column '{id_col}' in DataFrame.\n"
            f"  Available columns: {_cols_display(df.columns)}"
        )

    if isinstance(age_column, list):
        missing_age_cols = [col for col in age_column if col not in df.columns]
        if missing_age_cols:
            raise KeyError(
                f"Missing participant age column(s) '{', '.join(missing_age_cols)}' in DataFrame.\n"
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

    required_cols = [id_col]
    required_cols.extend(age_column)

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
    age_min_col: Optional[str],
    age_max_col: Optional[str],
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

    if age_min_col and not age_min_col.endswith("_part"):
        rename_map[age_min_col] = "age_min_part"

    if age_max_col and not age_max_col.endswith("_part"):
        rename_map[age_max_col] = "age_max_part"

    if age_grp_col and not age_grp_col.endswith("_part"):
        rename_map[age_grp_col] = "age_grp_part"

    for var in strat_var_cols:
        if not var.endswith("_part"):
            rename_map[var] = f"{var}_part"

    if repeat_col and not repeat_col.endswith("_part"):
        rename_map[repeat_col] = "repeat_part"

    df = df.rename(columns=rename_map)

    # --- synthesise age_grp_part from age_min_part / age_max_part -----------
    if age_min_col and age_max_col:
        df["age_grp_part"] = _build_age_grp_from_min_max(df, "age_min_part", "age_max_part")
        _warn_if_overlapping(df["age_grp_part"].cat.categories)

    return df


# ---------------------------------------------------------------------------
# Age-group helpers
# ---------------------------------------------------------------------------


def _build_age_grp_from_min_max(
    df: pd.DataFrame, min_col: str, max_col: str
) -> "pd.Categorical":
    """Return an ordered Categorical of left-closed intervals [min, max)."""
    unique = (
        df[[min_col, max_col]]
        .drop_duplicates()
        .sort_values([min_col, max_col])
    )
    cats = pd.IntervalIndex.from_arrays(
        unique[min_col], unique[max_col], closed="left"
    )
    raw = pd.arrays.IntervalArray.from_arrays(
        df[min_col], df[max_col], closed="left"
    )
    return pd.Categorical(raw, categories=cats, ordered=True)


def _warn_if_overlapping(categories: "pd.IntervalIndex") -> None:
    """Warn if any two intervals in an IntervalIndex overlap."""
    pairs = sorted(categories, key=lambda iv: (iv.left, iv.right))
    for i in range(1, len(pairs)):
        if pairs[i].left < pairs[i - 1].right:
            warnings.warn(
                f"Age intervals overlap (e.g. {pairs[i - 1]} and {pairs[i]}). "
                "Current models do not handle overlapping age ranges — "
                "consider rebinning to non-overlapping intervals.",
                UserWarning,
                stacklevel=6,
            )
            break
