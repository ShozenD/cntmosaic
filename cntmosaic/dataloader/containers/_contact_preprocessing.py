"""
Preprocessing pipeline for contact survey data.

This module provides the transformation logic for ContactData: column
validation, NaN removal, dtype coercion, column standardisation, and
addition of a contact-count indicator column.
Keeping this logic here separates the transformation concern from the
data-container and query API defined in ContactData.
"""

import warnings
from typing import List, Optional

import pandas as pd


def preprocess_contact_data(
    df_cnt: pd.DataFrame,
    id_col: str,
    age_col: Optional[str],
    age_min_col: Optional[str],
    age_max_col: Optional[str],
    age_grp_col: Optional[str],
    strat_var_cols: List[str],
    cnt_col: str,
) -> pd.DataFrame:
    """
    Validate, clean, and standardise a raw contact DataFrame.

    Parameters
    ----------
    df_cnt : pd.DataFrame
        Raw contact DataFrame (not yet validated or renamed).
    id_col : str
        Column containing participant identifiers.
    age_col : Optional[str]
        Column containing exact contact ages.
    age_min_col : Optional[str]
        Column containing minimum contact ages (for age ranges).
    age_max_col : Optional[str]
        Column containing maximum contact ages (for age ranges).
    age_grp_col : Optional[str]
        Column containing contact age groups (IntervalIndex).
    strat_var_cols : List[str]
        Stratification variable column names (already normalised to a list).
    cnt_col : str
        Column name for contact counts/indicators.  If absent from df_cnt a
        column of ones is added under this name.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised column names and a contact-count
        indicator column.

    Raises
    ------
    KeyError
        If any required column is absent from df_cnt.
    ValueError
        If df_cnt is empty after NaN removal.
    """
    required_cols = _check_columns(
        df_cnt, id_col, age_col, age_min_col, age_max_col, age_grp_col, strat_var_cols
    )
    return _preprocess(
        df_cnt,
        id_col,
        age_col,
        age_min_col,
        age_max_col,
        age_grp_col,
        strat_var_cols,
        cnt_col,
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
) -> List[str]:
    """
    Assert required columns exist.  Return the list of required column names.
    """

    def _cols_display(columns):
        cols = list(columns)
        return cols[:8] + (["..."] if len(cols) > 8 else [])

    if id_col not in df.columns:
        raise KeyError(
            f"Missing participant ID column '{id_col}' in contacts DataFrame.\n"
            f"  Available columns: {_cols_display(df.columns)}"
        )

    if age_col:
        if age_col not in df.columns:
            raise KeyError(
                f"Missing contact age column '{age_col}' in contacts DataFrame.\n"
                f"  Available columns: {_cols_display(df.columns)}"
            )
    elif age_min_col and age_max_col:
        missing_age_cols = [c for c in [age_min_col, age_max_col] if c not in df.columns]
        if missing_age_cols:
            raise KeyError(
                f"Missing contact age column(s) '{', '.join(missing_age_cols)}' in contacts DataFrame.\n"
                f"  Available columns: {_cols_display(df.columns)}"
            )
    elif age_grp_col:
        if age_grp_col not in df.columns:
            raise KeyError(
                f"Missing contact age group column '{age_grp_col}' in contacts DataFrame.\n"
                f"  Available columns: {_cols_display(df.columns)}"
            )

    if strat_var_cols:
        missing_vars = [v for v in strat_var_cols if v not in df.columns]
        if missing_vars:
            raise KeyError(
                f"Missing contact stratification variable(s) {missing_vars} in DataFrame.\n"
                f"  Available columns: {_cols_display(df.columns)}"
            )

    required_cols = [id_col]
    if age_col:
        required_cols.append(age_col)
    elif age_min_col and age_max_col:
        required_cols.extend([age_min_col, age_max_col])
    elif age_grp_col:
        required_cols.append(age_grp_col)

    if strat_var_cols:
        required_cols.extend(strat_var_cols)
    return required_cols


def _preprocess(
    df_cnt: pd.DataFrame,
    id_col: str,
    age_col: Optional[str],
    age_min_col: Optional[str],
    age_max_col: Optional[str],
    age_grp_col: Optional[str],
    strat_var_cols: List[str],
    cnt_col: str,
    required_cols: List[str],
) -> pd.DataFrame:
    """
    Copy, clean, coerce dtypes, rename columns, and add contact-count column.
    """
    df = df_cnt.copy()

    # --- NaN removal ---------------------------------------------------------
    n_rows_before = len(df)
    missing_counts = df[required_cols].isnull().sum()
    has_missing = missing_counts.sum() > 0

    df = df[required_cols].dropna().copy()

    if has_missing:
        n_dropped = n_rows_before - len(df)
        cols_with_missing = missing_counts[missing_counts > 0]
        warnings.warn(
            f"Dropped {n_dropped} contact record(s) with missing values in required columns.\n"
            f"  Missing value counts by column: {cols_with_missing.to_dict()}\n"
            f"  Remaining contacts: {len(df)}",
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

    if age_col and not age_col.endswith("_cnt"):
        rename_map[age_col] = "age_cnt"

    if age_min_col and not age_min_col.endswith("_cnt"):
        rename_map[age_min_col] = "age_min_cnt"

    if age_max_col and not age_max_col.endswith("_cnt"):
        rename_map[age_max_col] = "age_max_cnt"

    if age_grp_col and not age_grp_col.endswith("_cnt"):
        rename_map[age_grp_col] = "age_grp_cnt"

    for var in strat_var_cols:
        if not var.endswith("_cnt"):
            rename_map[var] = f"{var}_cnt"

    df = df.rename(columns=rename_map)

    # --- synthesise age_grp_cnt from age_min_cnt / age_max_cnt --------------
    if age_min_col and age_max_col:
        df["age_grp_cnt"] = _build_age_grp_from_min_max(df, "age_min_cnt", "age_max_cnt")
        _warn_if_overlapping(df["age_grp_cnt"].cat.categories)

    # --- contact-count indicator column --------------------------------------
    if cnt_col not in df.columns:
        df[cnt_col] = 1

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
