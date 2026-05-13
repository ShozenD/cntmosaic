"""
Free functions for building the observation grid and related arrays.

These functions were extracted from BaseLoader to make each concern
independently testable and to support the ContactSurveyLoader pipeline.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._CoordToColumns import CoordToColumns
from ._utils import gaussian_smooth_by_group


def build_participant_counts(data: pd.DataFrame, col_spec: CoordToColumns) -> pd.DataFrame:
    """Aggregate participant counts (N) by age and stratification variables."""
    df_n = (
        data.groupby(col_spec.strat_vars_n, observed=False)
        .agg(N=(col_spec.id_col, "nunique"))
        .reset_index()
    )
    return df_n


def build_contact_offsets(
    data: pd.DataFrame, col_spec: CoordToColumns, smooth: bool
) -> pd.DataFrame:
    """
    Compute ambiguous contact offsets (V) by age and stratification variables.

    Returns a DataFrame with column 'V' where V = 1 when no ambiguous contacts
    are present, or V = 1 - z/(z+y) otherwise.
    """
    # No ambiguous contacts: V = 1.0 everywhere
    if col_spec.z is None:
        df_V = build_participant_counts(data, col_spec).drop(columns=["N"])
        df_V["V"] = 1.0
        return df_V

    df_z = (
        data[[col_spec.id_col] + col_spec.strat_vars_n + [col_spec.z]]
        .drop_duplicates()
        .groupby(col_spec.strat_vars_n, observed=False)[col_spec.z]
        .sum()
        .reset_index()
    )
    df_yz = (
        data[[col_spec.id_col] + col_spec.strat_vars_n + [col_spec.y]]
        .drop_duplicates()
        .groupby(col_spec.strat_vars_n, observed=False)[col_spec.y]
        .sum()
        .reset_index()
    )
    df_V = df_yz.merge(df_z, on=col_spec.strat_vars_n, how="left")
    df_V["V"] = 1 - df_V[col_spec.z] / (df_V[col_spec.z] + df_V[col_spec.y])
    df_V["V"] = np.where(
        df_V["V"] == 0, 1.0 / (df_V[col_spec.z] + 1.0), df_V["V"]
    )
    df_V.fillna({"V": 1.0}, inplace=True)
    df_V = df_V.drop(columns=[col_spec.z, col_spec.y])

    if smooth:
        smooth_group_vars = [
            var for var in col_spec.strat_vars_n if var != col_spec.age_part
        ]
        if len(smooth_group_vars) > 0:
            df_V = gaussian_smooth_by_group(
                df_V, group_by=smooth_group_vars, target="V", cv=True, sort_by=col_spec.age_part
            )
        else:
            df_V = gaussian_smooth_by_group(
                df_V, group_by=None, target="V", cv=True, sort_by=col_spec.age_part
            )
        df_V["V"] = df_V["V_gs"]

    return df_V


def build_contact_counts(data: pd.DataFrame, col_spec: CoordToColumns) -> pd.DataFrame:
    """Aggregate contact counts (y) by age and stratification variables."""
    df_y = (
        data.groupby(col_spec.strat_vars_y, observed=False)
        .agg({col_spec.y: "sum"})
        .reset_index()
    )
    return df_y


def build_observation_grid(
    data: pd.DataFrame,
    col_spec: CoordToColumns,
    age_min: int,
    age_max: int,
    df_n: pd.DataFrame,
    df_y: pd.DataFrame,
    df_V: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full Cartesian observation grid by merging participant counts,
    contact counts, and offsets, then zero-filling missing cells.

    Parameters
    ----------
    data : pd.DataFrame
        Merged participant-contact DataFrame (used for category information).
    col_spec : CoordToColumns
        Column mapping specification.
    age_min, age_max : int
        Aligned age range bounds.
    df_n, df_y, df_V : pd.DataFrame
        Pre-computed participant counts, contact counts, and offsets.
    """
    unique_coords = {
        var: data[var].unique() for var in col_spec.strat_vars_y
    }
    unique_coords[col_spec.age_part] = np.arange(age_min, age_max + 1, dtype=int)

    if col_spec.age_cnt:
        unique_coords[col_spec.age_cnt] = np.arange(age_min, age_max + 1, dtype=int)
    elif col_spec.age_grp_cnt:
        unique_coords[col_spec.age_grp_cnt] = data[col_spec.age_grp_cnt].cat.categories

    index = pd.MultiIndex.from_product(
        unique_coords.values(), names=unique_coords.keys()
    )
    df_full = pd.DataFrame(list(index), columns=unique_coords.keys())
    df_full = pd.merge(df_full, df_y, on=col_spec.strat_vars_y, how="left")
    df_full = pd.merge(df_full, df_n, on=col_spec.strat_vars_n, how="left")
    df_full = pd.merge(df_full, df_V, on=col_spec.strat_vars_n, how="left")

    # Restore categorical dtypes after merges
    if col_spec.age_grp_cnt:
        df_full[col_spec.age_grp_cnt] = pd.Categorical(
            df_full[col_spec.age_grp_cnt],
            categories=data[col_spec.age_grp_cnt].cat.categories,
            ordered=True,
        )
    if col_spec.strat_vars_part:
        for var in col_spec.strat_vars_part:
            categories = data[var].cat.categories
            df_full[var] = pd.Categorical(df_full[var], categories=categories, ordered=True)
    if col_spec.strat_vars_cnt:
        for var in col_spec.strat_vars_cnt:
            categories = data[var].cat.categories
            df_full[var] = pd.Categorical(df_full[var], categories=categories, ordered=True)

    df_full = df_full.dropna(subset=["N"])
    df_full = df_full[df_full["N"] > 0]
    df_full["V"] = df_full["V"].fillna(1.0)
    df_full["log_V"] = np.where(df_full["V"] > 0, np.log(df_full["V"]), 0.0)
    df_full["y"] = df_full["y"].fillna(0)

    return df_full


def construct_log_P(pop_df: pd.DataFrame, col_spec: CoordToColumns) -> NDArray:
    """
    Construct log population proportions (log_P) from population data.

    Returns shape (1, A) for unstratified or (K, A) for stratified cases.
    """
    if col_spec.strat_vars_pop:
        P = (
            pop_df.pivot(
                index=col_spec.strat_vars_pop,
                columns=col_spec.age_pop,
                values=col_spec.P,
            )
            .fillna(1)
            .to_numpy()
        )  # shape (K, A)
    else:
        P = pop_df[col_spec.P].to_numpy()[np.newaxis, :]  # shape (1, A)

    return np.log(P)


def align_age_range(
    data: pd.DataFrame,
    pop_df: pd.DataFrame,
    col_spec: CoordToColumns,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Align the participant/contact age range to the population age range.

    Returns a (possibly filtered) copy of data plus the aligned min and max ages.
    Emits UserWarning when the sample age range differs from the population age range.
    """
    part_min_age = int(data[col_spec.age_part].min())
    part_max_age = int(data[col_spec.age_part].max())

    if col_spec.age_cnt:
        cnt_min_age = int(data[col_spec.age_cnt].min())
        cnt_max_age = int(data[col_spec.age_cnt].max())
    else:
        cnt_min_age = int(data[col_spec.age_grp_cnt].min().left)
        cnt_max_age = int(data[col_spec.age_grp_cnt].max().right - 1)

    pop_min_age = int(pop_df[col_spec.age_pop].min())
    pop_max_age = int(pop_df[col_spec.age_pop].max())

    sample_min_age = min(part_min_age, cnt_min_age)
    sample_max_age = max(part_max_age, cnt_max_age)

    if sample_min_age != pop_min_age:
        warnings.warn(
            f"Sample minimum age ({sample_min_age}) differs from population minimum age "
            f"({pop_min_age}). Filtering sample data to match population (age >= {pop_min_age}).",
            UserWarning,
            stacklevel=3,
        )
        data = data[data[col_spec.age_part] >= pop_min_age].copy()
        if data.empty:
            raise ValueError(
                f"After filtering to age >= {pop_min_age}, no data remains. "
                "Check age range compatibility."
            )
        min_age = pop_min_age
    else:
        min_age = sample_min_age

    if sample_max_age != pop_max_age:
        warnings.warn(
            f"Sample maximum age ({sample_max_age}) differs from population maximum age "
            f"({pop_max_age}). Using population maximum age ({pop_max_age}) for analysis.",
            UserWarning,
            stacklevel=3,
        )
        max_age = pop_max_age
    else:
        max_age = sample_max_age

    return data, min_age, max_age
