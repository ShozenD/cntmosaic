"""
Base class for loading and preprocessing contact survey data.

Internal API — not exported from ``cntmosaic.dataloader``. Provides the
``BaseLoader`` abstract base class with core functionality for validating,
merging, and transforming contact survey data into formats suitable for
statistical modeling. Concrete users should subclass ``DataLoader`` instead.
"""

import warnings
from abc import ABC
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._types import StratMode
from ._CoordToColumns import CoordToColumns
from ._observation import (
    align_age_range,
    build_contact_counts,
    build_contact_offsets,
    build_observation_grid,
    build_participant_counts,
    construct_log_P,
)
from ._stratification import (
    assemble_strat_kwargs,
    infer_full_strat_labels,
    infer_strat_dims,
    infer_strat_ixs,
    infer_strat_labels,
    infer_strat_modes,
    infer_strat_pixs,
    make_flat_ix,
)
from ._utils import expand_ix_array, gaussian_smooth_by_group, make_idarrs_for_intervals
from .containers._ModelData import ModelData
from .containers._StratificationData import StratificationData


class BaseLoader(ABC):
    """
    Abstract base class for loading and preprocessing contact survey data.

    This class provides the core functionality for validating, merging, and transforming
    contact survey data into formats suitable for statistical modeling. It handles:
    - Data validation and quality checks
    - Age range alignment between sample and population data
    - Calculation of sample sizes and contact frequencies
    - Group contact offset corrections
    - Creation of xarray datasets for model input

    Subclasses must implement specific validation methods for their data sources.

    Parameters
    ----------
    data : pd.DataFrame
        Combined dataframe containing both participant and contact information.
        Must include columns specified in col_map.
    pop : pd.DataFrame
        Population dataframe with age distribution and population sizes.
        Used for normalization and rate calculations.
    col_map : CoordToColumns
        Column mapping object specifying how to interpret dataframe columns.
    strat_data : Optional[StratificationData], default=None
        Optional stratification data for stratified models.
    smooth_amb_cnt_offsets : bool, default=True
        Whether to apply Gaussian smoothing to the ambiguous contact offsets (V)
        before they are used as log-offsets in the model. Smoothing is performed
        separately within each stratum across participant ages; the bandwidth is
        selected automatically by leave-one-out cross-validation. Set to ``False``
        to use the raw, unsmoothed offsets.

    Attributes
    ----------
    data : pd.DataFrame
        Validated and cleaned input data.
    col_map : CoordToColumns
        Column mapping configuration.
    pop : pd.DataFrame
        Population dataframe.
    min_age : int
        Minimum age in the aligned age range.
    max_age : int
        Maximum age in the aligned age range.

    Raises
    ------
    KeyError
        If required columns are missing from input dataframes.
    TypeError
        If age group column is not properly formatted as pd.IntervalIndex,
        or if pop_prop contains non-StratPropData objects.
    ValueError
        If age ranges are inconsistent or population proportions don't sum to 1.

    Notes
    -----
    - Age ranges are automatically aligned between sample and population data
    - Missing values in critical columns will raise errors
    - Object-type columns are automatically converted to categorical
    - Group contacts are handled with special offset corrections

    See Also
    --------
    DataLoader : Concrete implementation for separate participant/contact dataframes
    CoordToColumns : Column mapping specification
    StratPropData : Stratified population proportion specification
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pop_data: pd.DataFrame,
        col_map: CoordToColumns,
        strat_data: Optional[StratificationData] = None,
        smooth_amb_cnt_offsets: bool = True,
    ) -> None:
        """
        Initialize base loader with data validation.

        Parameters
        ----------
        data : pd.DataFrame
            Combined input dataframe.
        pop_data : pd.DataFrame
            Population dataframe.
        col_map : CoordToColumns
            Column mapping specification.
        strat_data : Optional[StratificationData], default=None
            Optional stratification data for statified models.
        smooth_amb_cnt_offsets : bool, optional
            Whether to apply Gaussian smoothing to the ambiguous contact offsets (V). By default, True.
            The scale of smoothing is determined by leave-one-out cross-validation across all stratification groups.
        """
        self.data = data
        self.col_map = col_map
        self.pop_data = pop_data

        # Auto-create the ambiguous contact count column if missing
        # (documented behaviour: if not present, created with value 0)
        if col_map.z is not None and col_map.z not in self.data.columns:
            self.data = self.data.copy()
            self.data[col_map.z] = 0
        self.strat_data = strat_data
        self.smooth_amb_cnt_offsets = smooth_amb_cnt_offsets
        self._align_age_range()

        # Initialize empty ModelData
        self.model_data: Optional[ModelData] = None

    def _align_age_range(self) -> None:
        """Align age ranges between sample and population data."""
        self.data, min_age, max_age = align_age_range(
            self.data, self.pop_data, self.col_map
        )
        self.min_age: int = min_age
        self.max_age: int = max_age

    @property
    def df_participant_counts(self) -> pd.DataFrame:
        """Construct dataframe of participant counts (N) stratified by age and grouping variables."""
        return build_participant_counts(self.data, self.col_map)

    @property
    def df_contact_offsets(self) -> pd.DataFrame:
        """Construct dataframe of ambiguous contact offsets (V) stratified by age and grouping variables."""
        return build_contact_offsets(
            self.data, self.col_map, self.smooth_amb_cnt_offsets
        )

    @property
    def df_contact_counts(self) -> pd.DataFrame:
        """Construct dataframe of contact counts (y) stratified by age and grouping variables."""
        return build_contact_counts(self.data, self.col_map)

    @property
    def df_full(self) -> pd.DataFrame:
        """Construct full dataframe with all combinations of stratification variables.

        Note: After load() is called, this returns the cached snapshot to ensure
        consistent row ordering across all accesses.
        """
        if hasattr(self, "_df_full_cache") and self._df_full_cache is not None:
            return self._df_full_cache
        return self._build_df_full()

    def _build_df_full(self) -> pd.DataFrame:
        """Build the full dataframe from scratch."""
        df_n = self.df_participant_counts
        df_V = self.df_contact_offsets
        df_y = self.df_contact_counts
        return build_observation_grid(
            self.data, self.col_map, self.min_age, self.max_age, df_n, df_y, df_V
        )

    def construct_log_P(self) -> NDArray:
        """
        Construct log population proportions (log_P) stratified by age and grouping variables.

        Returns
        -------
        NDArray
            Log population distribution. If no stratification, shape (1, A). If stratified,
            shape (K, A) where K is the possible strata combinations.
        """
        return construct_log_P(self.pop_data, self.col_map)

    def infer_strat_modes(self) -> Dict[str, StratMode]:
        """Infer stratification modes (PARTIAL vs FULL) for each stratification variable."""
        return infer_strat_modes(self.col_map)

    def infer_strat_dims(self, strat_modes: Dict[str, StratMode]) -> Dict[str, int]:
        """Infer number of category combinations for each stratification variable."""
        return infer_strat_dims(self.df_full, strat_modes)

    def infer_strat_labels(
        self, strat_modes: Dict[str, StratMode]
    ) -> Dict[str, List[str]]:
        """Infer labels for each stratification variable based on its mode."""
        return infer_strat_labels(self.df_full, strat_modes)

    def infer_full_strat_labels(
        self, strat_dims: Dict[str, int], strat_labels: Dict[str, List[str]]
    ) -> List[str]:
        """Infer full stratification labels for all possible category combinations."""
        return infer_full_strat_labels(strat_dims, strat_labels)

    def infer_strat_ixs(self, strat_modes: Dict[str, StratMode]) -> Dict[str, NDArray]:
        """Infer stratification variable indices for each observation."""
        return infer_strat_ixs(self.df_full, strat_modes)

    def infer_strat_pixs(
        self, strat_modes: Dict[str, StratMode], strat_dims: Dict[str, int]
    ) -> NDArray:
        """Infer population stratum flat indices for each observation."""
        return infer_strat_pixs(self.df_full, strat_modes, strat_dims)

    def make_flat_ix(
        self, strat_ixs: Dict[str, NDArray], strat_dims: Dict[str, int]
    ) -> NDArray:
        """Create flat index combining all stratification variable indices."""
        return make_flat_ix(strat_ixs, strat_dims)

    def make_strat_data(self) -> Dict:
        """Return stratification fields as a dict of ModelData keyword arguments."""
        return assemble_strat_kwargs(self.df_full, self.col_map, self.strat_data)

    def load(self) -> ModelData:
        """
        Load and transform data into a ModelData for model fitting.

        This is the main method that transforms raw contact survey data into a
        structured container with two components:

        1. **ModelBaseData**: Contains the numerical arrays needed for inference
           - Contact counts (y), age indices (aid, bid)
           - Sample sizes (log_N), population distribution (log_P)
           - Optional: offsets (log_V), repeat indicators (rid)

        2. **ModelStratData**: Contains hierarchical stratification information
           - strat_vars: Variable names and their categories
           - strat_modes: PARTIAL vs FULL stratification for each variable
           - strat_vars_full: Participant and contact categories for FULL mode
           - strat_ix: Categorical codes for indexing

        The method performs several key transformations:
        1. Aggregates participant counts by age and grouping variables
        2. Computes group contact offsets to adjust for household contacts
        3. Creates contact count matrix across all age combinations
        4. Builds full Cartesian product of all stratification variables
        5. Merges aggregated data with full grid (zero-filling missing cells)
        6. Constructs ModelBaseData and ModelStratData containers

        Returns
        -------
        ModelData
            Type-safe container with arrays and optional stratification metadata.
            Contains:
            - arrays: ModelBaseData (required fields: y, aid, bid, log_N, log_P, age_min, age_max)
            - strat_metadata: ModelStratData or None (stratification configuration)

        Notes
        -----
        **Group Contact Offset (S)**:
        The offset S adjusts for group contacts (e.g., "household members")
        that inflate contact counts. Different adjustments are applied:
        - Children (5-18): Assumes group contacts are with other children
        - Adults (18+): Assumes group contacts are random across population
        - S = 1 - z/(z+y) for children
        - S = 1 - z/(z+y)/(max_age - min_age + 1) for adults

        Where:
        - z = number of group contacts
        - y = number of individual contacts

        **Stratification Modes**:
        - PARTIAL: Variable recorded for participants only (e.g., setting)
        - FULL: Variable recorded for both participants AND contacts (e.g., gender)

        The method automatically:
        - Zero-fills for unobserved age combinations
        - Maintains categorical variable consistency
        - Expands age group intervals for coarse age data
        - Detects PARTIAL vs FULL stratification modes

        See Also
        --------
        ModelData : Type-safe container for model inputs
        ModelStratData : TypedDict for stratification configuration
        docs/full_stratification_integration_guide.md : Detailed integration diagrams

        Examples
        --------
        >>> # Basic usage
        >>> loader = BaseLoader(data, pop, col_map)
        >>> container = loader.load()
        >>> container.arrays['y'].shape
        (12345,)  # Number of unique strata
        >>>
        >>> # Access stratification metadata
        >>> if container.strat_metadata:
        ...     print(container.strat_metadata['strat_modes'])
        {'gender': 'full', 'setting': 'partial'}
        >>>
        >>> # Check for specific fields
        >>> container.has('log_V')  # Check if offset exists
        True
        >>> container.age_range
        (0, 75)
        """
        # ========================
        # Cache df_full snapshot
        # ========================
        # CRITICAL: Cache df_full so all downstream calls (make_strat_data,
        # infer_strat_ixs, etc.) and post-load access use the same row ordering.
        self._df_full_cache = self._build_df_full()

        # ========================
        # Build required fields
        # ========================
        aid = self.df_full[self.col_map.age_part].to_numpy()

        bid = None
        aid_exp = None
        bid_pad = None
        if self.col_map.age_cnt:
            bid = self.df_full[self.col_map.age_cnt].to_numpy()
        elif self.col_map.age_grp_cnt:
            aid_exp, bid_pad = make_idarrs_for_intervals(
                self.df_full, self.col_map.age_grp_cnt, aid
            )

        rid = None
        if self.col_map.repeat_part is not None:
            rid = self.df_full[self.col_map.repeat_part].astype(int).to_numpy()

        # ============================
        # Build stratification kwargs
        # ============================
        strat_kwargs: Dict = {}
        if len(self.col_map.strat_vars_part) > 0:  # type: ignore
            strat_kwargs = self.make_strat_data()
            if self.col_map.age_grp_cnt:
                strat_kwargs["flat_ix_exp"] = expand_ix_array(
                    strat_kwargs["flat_ix"], bid_pad.shape[1]  # type: ignore
                )

        # ============================
        # Construct ModelData
        # ============================
        self.model_data = ModelData(
            y=self.df_full["y"].to_numpy(),
            aid=aid,
            log_N=np.log(self.df_full["N"].to_numpy()),
            log_V=self.df_full["log_V"].to_numpy(),
            log_P=self.construct_log_P(),
            age_min=self.min_age,
            age_max=self.max_age,
            bid=bid,
            aid_exp=aid_exp,
            bid_pad=bid_pad,
            rid=rid,
            **strat_kwargs,
        )

        return self.model_data
