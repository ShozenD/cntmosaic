"""
This module defines data container classes for BRC models.
These containers hold processed data that are then passed to the models as inputs.
They make sure that the data is of the correct format and type required by the models.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, TypedDict, Union

import jax.numpy as jnp
from numpy.typing import NDArray

from ..._types import StratMode

if TYPE_CHECKING:
    from ._StratificationData import StratificationData


class ModelBaseData(TypedDict, total=False):
    """
    Type-safe container for contact survey data.

    Required Fields
    ---------------
    y : NDArray
        Contact counts, shape (n_obs,)
    aid : NDArray
        Participant age indices, shape (n_obs,)
    log_N : NDArray
        Log sample sizes, shape (n_obs,)
    log_P : NDArray
        Log population distribution, shape (A,)

    Optional Fields
    ---------------
    log_S : NDArray
        Log offsets (e.g., for settings), shape (n_obs,)
    rid : NDArray
        Repeat interview indicators, shape (n_obs,)
    aid_exp : NDArray
        Expanded age indices (for BRCrefine), shape (n_obs,)
    bid : NDArray
        Contact age indices, shape (n_obs,)
    bid_pad : NDArray
        Padded age indices (for BRCrefine), shape (n_obs,)
    """

    # Required fields
    y: NDArray
    aid: NDArray
    log_N: NDArray
    log_P: NDArray
    age_min: int
    age_max: int

    # Optional fields (total=False allows these to be missing)
    log_V: NDArray
    rid: NDArray
    aid_exp: NDArray
    bid: NDArray
    bid_pad: NDArray


class ModelStratData(TypedDict):
    """Metadata for stratification (HiBRC models)."""

    vars: Dict[str, List[str]]  # {var_name: [category_names]}
    modes: Dict[str, StratMode]  # {var_name: StratMode.PARTIAL | StratMode.FULL}
    dims: Dict[str, int]  # {var_name: [number of dimensions]}
    labels: Dict[str, str]  # {var_name: label}
    ixs: Dict[str, NDArray]  # {var_name: categorical_codes}
    flat_pixs: NDArray  # Combined flat population category indices
    flat_ix: NDArray  # Combined flat indices
    flat_ix_exp: NDArray  # Expanded flat indices for HiBRCrefine model
    full_labels: List[str]
    marginal_multipliers: Optional[Dict[str, NDArray]] = None  # {var_name: NDArray}
    multipliers: Optional[NDArray] = None


class ModelData:
    """
    Lightweight, type-safe container for model input data.
    """

    def __init__(
        self,
        base_data: ModelBaseData,
        strat_data: Optional[ModelStratData] = None,
    ) -> None:
        self.base_data = base_data
        self.strat_data = strat_data or {}
        self._validate()

    def _validate(self) -> None:
        """Validate required fields and array shapes."""
        required = ["y", "aid", "log_N", "log_P"]
        for field in required:
            if field not in self.base_data:
                raise ValueError(f"Missing required field: {field}")

        # Shape validation
        n_obs = len(self.base_data["y"])
        for field in ["aid", "log_N"]:
            if len(self.base_data[field]) != n_obs:
                raise ValueError(
                    f"Shape mismatch: {field} has length {len(self.base_data[field])}, "
                    f"expected {n_obs}"
                )

    def get(self, key: str, default=None):
        """Get an array by key, with optional default."""
        return self.base_data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if an array exists in the container."""
        return key in self.base_data

    @property
    def age_range(self) -> tuple[int, int]:
        """Return the age range as (min_age, max_age)."""
        return (self.base_data["age_min"], self.base_data["age_max"])
