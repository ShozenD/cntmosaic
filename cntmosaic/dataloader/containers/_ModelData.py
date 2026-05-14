"""
Unified data container for contact survey model inputs.

Internal API — not exported from ``cntmosaic.dataloader``. Produced by
``DataLoader.load()`` and consumed by all model classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from numpy.typing import NDArray

from ..._types import StratMode


@dataclass
class ModelData:
    """
    Flat, type-safe container for contact survey model inputs.

    All fields are direct attributes — no nested dicts. Optional fields default
    to ``None``; use ``field is not None`` to check presence.

    Required Fields
    ---------------
    y : NDArray
        Contact counts, shape (n_obs,).
    aid : NDArray
        Participant age indices, shape (n_obs,).
    log_N : NDArray
        Log sample sizes, shape (n_obs,).
    log_P : NDArray
        Log population distribution, shape (1, A) or (K, A) for stratified.
    age_min : int
        Minimum age in the aligned age range.
    age_max : int
        Maximum age in the aligned age range.

    Optional Observation Fields
    ---------------------------
    log_V : NDArray or None
        Log ambiguous-contact offsets, shape (n_obs,).
    rid : NDArray or None
        Repeat interview indicators, shape (n_obs,).
    bid : NDArray or None
        Contact age indices (fine resolution), shape (n_obs,).
    aid_exp : NDArray or None
        Expanded participant age indices (coarse contact ages), shape (n_obs, B).
    bid_pad : NDArray or None
        Padded contact age indices (coarse contact ages), shape (n_obs, B).
    log_S : NDArray or None
        Log setting offsets (vdKassteele model), shape (n_obs,).

    Optional Stratification Fields
    --------------------------------
    strat_modes : Dict[str, StratMode] or None
        Stratification mode per variable (PARTIAL or FULL).
    strat_dims : Dict[str, int] or None
        Number of category combinations per stratification variable.
    strat_labels : Dict[str, List[str]] or None
        Category labels per variable in ``"source->target"`` format.
    flat_ix : NDArray or None
        Combined flat stratum indices for each observation.
    flat_pixs : NDArray or None
        Combined flat population-stratum indices for each observation.
    flat_ix_exp : NDArray or None
        Expanded flat indices for HiBRCrefine (coarse contact ages).
    full_labels : List[str] or None
        Labels for all possible stratum combinations.
    marginal_multipliers : Dict[str, NDArray] or None
        Marginal demographic opportunity weights per variable.
    multipliers : NDArray or None
        Joint demographic opportunity weights.
    """

    # Required observation fields
    y: NDArray
    aid: NDArray
    log_N: NDArray
    log_P: NDArray
    age_min: int
    age_max: int

    # Optional observation fields
    log_V: Optional[NDArray] = None
    rid: Optional[NDArray] = None
    bid: Optional[NDArray] = None
    aid_exp: Optional[NDArray] = None
    bid_pad: Optional[NDArray] = None
    log_S: Optional[NDArray] = None

    # Optional stratification fields
    strat_modes: Optional[Dict[str, StratMode]] = None
    strat_dims: Optional[Dict[str, int]] = None
    strat_labels: Optional[Dict[str, List[str]]] = None
    flat_ix: Optional[NDArray] = None
    flat_pixs: Optional[NDArray] = None
    flat_ix_exp: Optional[NDArray] = None
    full_labels: Optional[List[str]] = None
    marginal_multipliers: Optional[Dict[str, NDArray]] = None
    multipliers: Optional[NDArray] = None

    def __post_init__(self) -> None:
        n_obs = len(self.y)
        for field in ["aid", "log_N"]:
            arr = getattr(self, field)
            if len(arr) != n_obs:
                raise ValueError(
                    f"Shape mismatch: {field} has length {len(arr)}, expected {n_obs}"
                )

    @property
    def age_range(self) -> Tuple[int, int]:
        """Return ``(age_min, age_max)``."""
        return (self.age_min, self.age_max)

    @property
    def is_stratified(self) -> bool:
        """True when stratification metadata is present."""
        return self.strat_modes is not None
