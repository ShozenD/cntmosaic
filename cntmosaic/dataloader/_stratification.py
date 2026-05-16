"""
Free functions for stratification inference.

These functions were extracted from BaseLoader to make each concern
independently testable and to support the ContactSurveyLoader pipeline.
"""

from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._types import StratMode
from ._ColumnSpec import ColumnSpec
from .containers._StratificationData import StratificationData


def infer_strat_modes(col_spec: ColumnSpec) -> Dict[str, StratMode]:
    """
    Infer stratification modes (PARTIAL vs FULL) for each stratification variable.

    A variable is FULL when it appears in both participant and contact columns;
    PARTIAL when it only appears on the participant side.
    """
    strat_modes: Dict[str, StratMode] = {}

    strat_vars_part = (
        [var.replace("_part", "") for var in col_spec.strat_vars_part]
        if col_spec.strat_vars_part
        else []
    )
    strat_vars_cnt = (
        [var.replace("_cnt", "") for var in col_spec.strat_vars_cnt]
        if col_spec.strat_vars_cnt
        else []
    )

    for var in strat_vars_part:
        if strat_vars_cnt and var in strat_vars_cnt:
            strat_modes[var] = StratMode.FULL
        else:
            strat_modes[var] = StratMode.PARTIAL

    return strat_modes


def infer_strat_dims(
    df_full: pd.DataFrame, strat_modes: Dict[str, StratMode]
) -> Dict[str, int]:
    """Infer number of category combinations for each stratification variable."""
    strat_dims: Dict[str, int] = {}

    for var, mode in strat_modes.items():
        categories = df_full[var + "_part"].cat.categories
        if mode == StratMode.PARTIAL:
            strat_dims[var] = len(categories)
        elif mode == StratMode.FULL:
            strat_dims[var] = len(categories) ** 2

    return strat_dims


def infer_strat_labels(
    df_full: pd.DataFrame, strat_modes: Dict[str, StratMode]
) -> Dict[str, List[str]]:
    """Infer labels for each stratification variable based on its mode."""
    strat_labels: Dict[str, List[str]] = {}

    for var, mode in strat_modes.items():
        categories = df_full[var + "_part"].cat.categories

        if mode == StratMode.PARTIAL:
            labels = [f"{cat}->All" for cat in categories]
        elif mode == StratMode.FULL:
            labels = [f"{cat1}->{cat2}" for cat1 in categories for cat2 in categories]

        strat_labels[var] = labels

    return strat_labels


def infer_full_strat_labels(
    strat_dims: Dict[str, int], strat_labels: Dict[str, List[str]]
) -> List[str]:
    """
    Infer full stratification labels combining all possible category combinations.

    Labels are generated in row-major order (rightmost variable varies fastest)
    to match the ordering produced by make_flat_ix.
    """
    full_labels: List[str] = []
    strat_vars = list(strat_dims.keys())
    dim_ranges = [range(strat_dims[var]) for var in strat_vars]

    for cat_codes in product(*dim_ranges):
        parts = [strat_labels[strat_vars[i]][code] for i, code in enumerate(cat_codes)]
        sources = [p.split("->")[0] for p in parts]
        targets = [p.split("->")[1] for p in parts]

        source_label = "_".join(sources)
        target_parts = [t for t in targets if t != "All"]
        target_label = "_".join(target_parts) if target_parts else "All"

        full_labels.append(f"{source_label}->{target_label}")

    return full_labels


def infer_strat_ixs(
    df_full: pd.DataFrame, strat_modes: Dict[str, StratMode]
) -> Dict[str, NDArray]:
    """Infer stratification variable indices for each observation."""
    strat_ixs: Dict[str, NDArray] = {}

    for var, mode in strat_modes.items():
        if mode == StratMode.PARTIAL:
            strat_ixs[var] = df_full[var + "_part"].cat.codes.to_numpy()
        elif mode == StratMode.FULL:
            part_codes = df_full[var + "_part"].cat.codes.to_numpy()
            cnt_codes = df_full[var + "_cnt"].cat.codes.to_numpy()
            n_categories = len(df_full[var + "_part"].cat.categories)
            strat_ixs[var] = part_codes * n_categories + cnt_codes

    return strat_ixs


def infer_strat_pixs(
    df_full: pd.DataFrame,
    strat_modes: Dict[str, StratMode],
    strat_dims: Dict[str, int],
) -> NDArray:
    """Infer population stratum flat indices for each observation."""
    strat_pixs: Dict[str, NDArray] = {}

    for var, mode in strat_modes.items():
        if mode == StratMode.FULL:
            strat_pixs[var] = df_full[var + "_cnt"].cat.codes.to_numpy()
        else:
            strat_pixs[var] = np.zeros(len(df_full), dtype=int)

    n_obs = len(next(iter(strat_pixs.values())))
    flat_pixs = np.zeros(n_obs, dtype=int)
    multiplier = 1
    for var, mode in reversed(strat_modes.items()):
        dim = strat_dims[var] if mode == StratMode.FULL else 1
        flat_pixs += strat_pixs[var] * multiplier
        multiplier *= dim

    return flat_pixs


def make_flat_ix(strat_ixs: Dict[str, NDArray], strat_dims: Dict[str, int]) -> NDArray:
    """Create flat index combining all stratification variable indices."""
    n_obs = len(next(iter(strat_ixs.values())))
    flat_ix = np.zeros(n_obs, dtype=int)

    multiplier = 1
    for var, dim in reversed(strat_dims.items()):
        flat_ix += strat_ixs[var] * multiplier
        multiplier *= dim

    return flat_ix


def assemble_strat_kwargs(
    df_full: pd.DataFrame,
    col_spec: ColumnSpec,
    strat_data: Optional[StratificationData],
) -> Dict:
    """
    Assemble stratification keyword arguments for ModelData construction.

    Returns a dict suitable for unpacking as **strat_kwargs into ModelData(...).
    """
    modes = infer_strat_modes(col_spec)
    dims = infer_strat_dims(df_full, modes)
    labels = infer_strat_labels(df_full, modes)
    ixs = infer_strat_ixs(df_full, modes)
    flat_pixs = infer_strat_pixs(df_full, modes, dims)
    flat_ix = make_flat_ix(ixs, dims)
    full_labels = infer_full_strat_labels(dims, labels)

    kwargs: Dict = dict(
        strat_modes=modes,
        strat_dims=dims,
        strat_labels=labels,
        flat_pixs=flat_pixs,
        flat_ix=flat_ix,
        full_labels=full_labels,
    )

    if strat_data is not None:
        kwargs["marginal_multipliers"] = strat_data.compute_marginal_demopty(
            modes, labels
        )
        kwargs["multipliers"] = strat_data.compute_demopty(modes, full_labels)

    return kwargs
