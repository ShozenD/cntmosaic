"""
Base class for loading and preprocessing contact survey data.

This module provides the BaseLoader abstract base class with core functionality
for validating, merging, and transforming contact survey data into formats suitable
for statistical modeling.
"""

import warnings
from abc import ABC
from itertools import product
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._types import StratMode
from ._CoordToColumns import CoordToColumns
from ._utils import expand_ix_array, make_idarrs_for_intervals
from .containers._ModelData import ModelBaseData, ModelData, ModelStratData
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
        """
        self.data = data
        self.col_map = col_map
        self.pop_data = pop_data
        self.strat_data = strat_data
        self._align_age_range()

        # Initialize empty ModelData
        self.model_data: Optional[ModelData] = None

    def _align_age_range(self) -> None:
        """
        Align age ranges between sample and population data.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe with contact and participant information.
        pop : pd.DataFrame
            Population dataframe with age distribution.
        col_map : CoordToColumns
            Column mapping specification.

        Warnings
        --------
        UserWarning
            - If age ranges between sample and population don't match
        """
        # Determine age ranges and ensure consistency
        part_min_age = int(self.data[self.col_map.age_part].min())
        part_max_age = int(self.data[self.col_map.age_part].max())

        if self.col_map.age_cnt:
            cnt_min_age = int(self.data[self.col_map.age_cnt].min())
            cnt_max_age = int(self.data[self.col_map.age_cnt].max())
        else:  # age_grp_cnt is specified
            cnt_min_age = int(self.data[self.col_map.age_grp_cnt].min().left)
            cnt_max_age = int(self.data[self.col_map.age_grp_cnt].max().right - 1)

        pop_min_age = int(self.pop_data[self.col_map.age_pop].min())
        pop_max_age = int(self.pop_data[self.col_map.age_pop].max())

        # Determine overall sample age range
        sample_min_age = min(part_min_age, cnt_min_age)
        sample_max_age = max(part_max_age, cnt_max_age)

        # Align minimum age with population data
        if sample_min_age != pop_min_age:
            warnings.warn(
                f"Sample minimum age ({sample_min_age}) differs from population minimum age ({pop_min_age}). "
                f"Filtering sample data to match population (age >= {pop_min_age}).",
                UserWarning,
                stacklevel=2,
            )
            self.data = self.data[
                self.data[self.col_map.age_part] >= pop_min_age
            ].copy()
            if self.data.empty:
                raise ValueError(
                    f"After filtering to age >= {pop_min_age}, no data remains. "
                    "Check age range compatibility."
                )
            min_age = pop_min_age
        else:
            min_age = sample_min_age

        # Align maximum age with population data
        if sample_max_age != pop_max_age:
            warnings.warn(
                f"Sample maximum age ({sample_max_age}) differs from population maximum age ({pop_max_age}). "
                f"Using population maximum age ({pop_max_age}) for analysis.",
                UserWarning,
                stacklevel=2,
            )
            max_age = pop_max_age
        else:
            max_age = sample_max_age

        # Store age bounds as instance attributes
        self.min_age: int = min_age
        self.max_age: int = max_age

    @property
    def df_n(self) -> pd.DataFrame:
        """Construct dataframe of participant counts (N) stratified by age and grouping variables."""
        df_n = (
            self.data.groupby(self.col_map.strat_vars_n, observed=False)
            .agg(N=(self.col_map.id_col, "nunique"))
            .reset_index()
        )
        return df_n

    @property
    def df_V(self) -> pd.DataFrame:
        """Construct dataframe of group contact offsets (S) stratified by age and grouping variables."""
        df_z = (
            self.data[
                [self.col_map.id_col] + self.col_map.strat_vars_n + [self.col_map.z]
            ]
            .drop_duplicates()
            .groupby(self.col_map.strat_vars_n, observed=False)[self.col_map.z]
            .sum()
            .reset_index()
        )
        df_yz = (
            self.data[
                [self.col_map.id_col] + self.col_map.strat_vars_n + [self.col_map.y]
            ]
            .drop_duplicates()
            .groupby(self.col_map.strat_vars_n, observed=False)[self.col_map.y]
            .sum()
            .reset_index()
        )
        df_V = df_yz.merge(df_z, on=self.col_map.strat_vars_n, how="left")
        df_V["V"] = 1 - df_V[self.col_map.z] / (
            df_V[self.col_map.z] + df_V[self.col_map.y]
        )
        # Little bit arbitrary - to avoid zero offsets
        df_V["V"] = np.where(
            df_V["V"] == 0, 1.0 / (df_V[self.col_map.z] + 1.0), df_V["V"]
        )
        df_V.fillna({"V": 1.0}, inplace=True)
        df_V = df_V.drop(columns=[self.col_map.z, self.col_map.y])

        return df_V

    @property
    def df_y(self) -> pd.DataFrame:
        """Construct dataframe of contact counts (y) stratified by age and grouping variables."""
        df_y = (
            self.data.groupby(self.col_map.strat_vars_y, observed=False)
            .agg({self.col_map.y: "sum"})
            .reset_index()
        )
        return df_y

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
        df_n = self.df_n
        df_V = self.df_V
        df_y = self.df_y

        # Create a full Cartesian product of all stratification variable levels
        unique_coords = {
            var: self.data[var].unique() for var in self.col_map.strat_vars_y
        }
        unique_coords[self.col_map.age_part] = np.arange(
            self.min_age, self.max_age + 1, dtype=int
        )

        if self.col_map.age_cnt:
            unique_coords[self.col_map.age_cnt] = np.arange(
                self.min_age, self.max_age + 1, dtype=int
            )
        elif self.col_map.age_grp_cnt:
            unique_coords[self.col_map.age_grp_cnt] = self.data[
                self.col_map.age_grp_cnt
            ].cat.categories

        index = pd.MultiIndex.from_product(
            unique_coords.values(), names=unique_coords.keys()
        )

        df_full = pd.DataFrame(list(index), columns=unique_coords.keys())
        df_full = pd.merge(df_full, df_y, on=self.col_map.strat_vars_y, how="left")
        df_full = pd.merge(df_full, df_n, on=self.col_map.strat_vars_n, how="left")
        df_full = pd.merge(df_full, df_V, on=self.col_map.strat_vars_n, how="left")

        # Restore all category information for categorical columns
        if self.col_map.age_grp_cnt:
            df_full[self.col_map.age_grp_cnt] = pd.Categorical(
                df_full[self.col_map.age_grp_cnt],
                categories=self.data[self.col_map.age_grp_cnt].cat.categories,
                ordered=True,
            )
        if self.col_map.strat_vars_part:
            for var in self.col_map.strat_vars_part:
                categories = self.data[var].cat.categories
                df_full[var] = pd.Categorical(
                    df_full[var],
                    categories=categories,
                    ordered=True,
                )
        if self.col_map.strat_vars_cnt:
            for var in self.col_map.strat_vars_cnt:
                categories = self.data[var].cat.categories
                df_full[var] = pd.Categorical(
                    df_full[var],
                    categories=categories,
                    ordered=True,
                )

        # [Do] Finalise the data
        df_full = df_full.dropna(subset=["N"])
        df_full = df_full[df_full["N"] > 0]
        df_full["V"] = df_full["V"].fillna(1.0)
        df_full["log_V"] = np.where(df_full["V"] > 0, np.log(df_full["V"]), 0.0)
        df_full["y"] = df_full["y"].fillna(0)

        return df_full

    def construct_log_P(self) -> NDArray:
        """
        Construct log population proportions (log_P) stratified by age and grouping variables.

        Returns
        -------
        NDArray
            Log population distribution. If no stratification, shape (1, A). If stratified,
            shape (1, K*A) where K is the possible strata combinations.
        """
        if self.col_map.strat_vars_pop:
            P = (
                self.pop_data.pivot(
                    index=self.col_map.strat_vars_pop,
                    columns=self.col_map.age_pop,
                    values=self.col_map.P,
                )
                .fillna(1)
                .to_numpy()
            )  # shape (K, A)

        else:
            P = self.pop_data[self.col_map.P].to_numpy()[np.newaxis, :]  # shape (1, A)

        return np.log(P)

    def infer_strat_modes(self) -> Dict[str, StratMode]:
        """
        Infer stratification modes (PARTIAL vs FULL) for each stratification variable.

        Returns
        -------
        Dict[str, StratMode]
            Dictionary mapping variable name to StratMode ('partial' or 'full').
        """
        strat_modes: Dict[str, StratMode] = {}

        strat_vars_part = (
            [var.replace("_part", "") for var in self.col_map.strat_vars_part]
            if self.col_map.strat_vars_part
            else []
        )

        strat_vars_cnt = (
            [var.replace("_cnt", "") for var in self.col_map.strat_vars_cnt]
            if self.col_map.strat_vars_cnt
            else []
        )

        if len(strat_vars_part) > 0:
            for var in strat_vars_part:
                if len(strat_vars_cnt) > 0 and var in strat_vars_cnt:
                    strat_modes[var] = StratMode.FULL
                else:
                    strat_modes[var] = StratMode.PARTIAL

        return strat_modes

    def infer_strat_dims(self, strat_modes: Dict[str, StratMode]) -> Dict[str, int]:
        """Infer number of categories (dimensions) for each stratification variable."""
        strat_dims: Dict[str, int] = {}

        for var, mode in strat_modes.items():
            categories = self.data[var + "_part"].cat.categories

            if mode == StratMode.PARTIAL:
                strat_dims[var] = len(categories)
            elif mode == StratMode.FULL:
                strat_dims[var] = len(categories) ** 2

        return strat_dims

    def infer_strat_labels(
        self, strat_modes: Dict[str, StratMode]
    ) -> Dict[str, List[str]]:
        """Infer labels for each stratification variable based on its mode."""
        strat_labels: Dict[str, List[str]] = {}

        for var, mode in strat_modes.items():
            categories = self.data[var + "_part"].cat.categories

            if mode == StratMode.PARTIAL:
                labels = [f"{cat}->All" for cat in categories]
            elif mode == StratMode.FULL:
                labels = [
                    f"{cat1}->{cat2}" for cat1 in categories for cat2 in categories
                ]

            strat_labels[var] = labels

        return strat_labels

    def infer_full_strat_labels(
        self, strat_dims: Dict[str, int], strat_labels: Dict[str, List[str]]
    ) -> List[str]:
        """
        Infer full stratification labels combining participant and contact categories.

        Generates labels for ALL possible category combinations based on dims,
        not just observed combinations. This is important for cross-validation
        scenarios where some category combinations may be missing from the data.

        Parameters
        ----------
        strat_dims : Dict[str, int]
            Dictionary mapping stratification variable names to their dimensions
            (already accounting for FULL mode squaring).
        strat_labels : Dict[str, List[str]]
            Dictionary mapping variable names to their category labels
            (in "source->target" format).

        Returns
        -------
        List[str]
            Full stratification labels for all possible combinations.

        Notes
        -----
        Labels are generated in row-major order to match make_flat_ix():
        rightmost variable varies fastest.
        """
        full_labels: List[str] = []
        strat_vars = list(strat_dims.keys())

        # CRITICAL: Reverse the variable order to match make_flat_ix()
        # This ensures rightmost variable varies fastest, consistent with how flat_ix is constructed
        dim_ranges = [range(strat_dims[var]) for var in strat_vars]

        for cat_codes in product(*dim_ranges):
            # Build composite label by concatenating source and target parts
            parts = []
            for i, code in enumerate(cat_codes):
                var = strat_vars[i]
                parts.append(strat_labels[var][code])

            # Split each "source->target" and combine
            sources = [p.split("->")[0] for p in parts]
            targets = [p.split("->")[1] for p in parts]

            # Join with underscores
            source_label = "_".join(sources)
            # Filter out "All" targets and join remaining
            target_parts = [t for t in targets if t != "All"]
            target_label = "_".join(target_parts) if target_parts else "All"

            full_label = f"{source_label}->{target_label}"
            full_labels.append(full_label)

        return full_labels

    def infer_strat_ixs(self, strat_modes: Dict[str, StratMode]) -> Dict[str, NDArray]:
        """Infer stratification variable indices for each observation."""
        strat_ixs: Dict[str, NDArray] = {}

        for var, mode in strat_modes.items():
            if mode == StratMode.PARTIAL:
                strat_ixs[var] = self.df_full[var + "_part"].cat.codes.to_numpy()
            elif mode == StratMode.FULL:
                part_codes = self.df_full[var + "_part"].cat.codes.to_numpy()
                cnt_codes = self.df_full[var + "_cnt"].cat.codes.to_numpy()
                n_categories = len(self.df_full[var + "_part"].cat.categories)
                strat_ixs[var] = part_codes * n_categories + cnt_codes

        return strat_ixs

    def infer_strat_pixs(
        self, strat_modes: Dict[str, StratMode], strat_dims: Dict[str, int]
    ) -> Dict[str, NDArray]:
        """Infer population stratification variable indices for each observation."""
        strat_pixs: Dict[str, NDArray] = {}

        for var, mode in strat_modes.items():
            # Only relevanat for FULL mode
            if mode == StratMode.FULL:
                cnt_codes = self.df_full[var + "_cnt"].cat.codes.to_numpy()
                strat_pixs[var] = cnt_codes
            else:
                strat_pixs[var] = np.zeros(len(self.df_full), dtype=int)

        # Create flat indices for population stratification
        n_obs = len(next(iter(strat_pixs.values())))
        flat_pixs = np.zeros(n_obs, dtype=int)
        multiplier = 1
        for var, mode in reversed(strat_modes.items()):
            dim = strat_dims[var] if mode == StratMode.FULL else 1
            flat_pixs += strat_pixs[var] * multiplier
            multiplier *= dim

        return flat_pixs

    def make_flat_ix(
        self, strat_ixs: Dict[str, NDArray], strat_dims: Dict[str, int]
    ) -> NDArray:
        """Create flat index combining all stratification variable indices."""
        n_obs = len(next(iter(strat_ixs.values())))
        flat_ix = np.zeros(n_obs, dtype=int)

        multiplier = 1
        for var, dim in reversed(strat_dims.items()):
            flat_ix += strat_ixs[var] * multiplier
            multiplier *= dim

        return flat_ix

    def make_strat_data(self) -> ModelStratData:
        modes = self.infer_strat_modes()
        dims = self.infer_strat_dims(modes)
        labels = self.infer_strat_labels(modes)
        ixs = self.infer_strat_ixs(modes)
        flat_pixs = self.infer_strat_pixs(modes, dims)
        flat_ix = self.make_flat_ix(ixs, dims)
        full_labels = self.infer_full_strat_labels(dims, labels)

        if self.strat_data is not None:
            marginal_demopty = self.strat_data.compute_marginal_demopty(modes)

            # Demographic Opportunity
            demopty = self.strat_data.compute_demopty(modes, full_labels)

            return ModelStratData(
                modes=modes,
                dims=dims,
                labels=labels,
                ixs=ixs,
                flat_pixs=flat_pixs,
                flat_ix=flat_ix,
                full_labels=full_labels,
                marginal_multipliers=marginal_demopty,
                multipliers=demopty,
            )
        else:
            # Multipliers are not needed for vdKassteele models
            return ModelStratData(
                modes=modes,
                dims=dims,
                labels=labels,
                ixs=ixs,
                flat_pixs=flat_pixs,
                flat_ix=flat_ix,
                full_labels=full_labels,
            )

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
        # Construct ModelBaseData
        # ========================
        base_data = ModelBaseData(
            y=self.df_full["y"].to_numpy(),
            aid=self.df_full[self.col_map.age_part].to_numpy(),
            log_N=np.log(self.df_full["N"].to_numpy()),
            log_V=self.df_full["log_V"].to_numpy(),
            log_P=self.construct_log_P(),
            age_min=self.min_age,
            age_max=self.max_age,
        )

        if self.col_map.age_cnt:
            base_data["bid"] = self.df_full[self.col_map.age_cnt].to_numpy()
        elif self.col_map.age_grp_cnt:
            # [Do] Create indices for age aggregation
            aid_exp, bid_pad = make_idarrs_for_intervals(
                self.df_full, self.col_map.age_grp_cnt, base_data["aid"]
            )
            base_data["aid_exp"] = aid_exp
            base_data["bid_pad"] = bid_pad

        # If repeat effects are specified
        if self.col_map.repeat_part is not None:
            base_data["rid"] = (
                self.df_full[self.col_map.repeat_part].astype(int).to_numpy()
            )

        # ============================
        # Construct stratification data
        # ============================
        if len(self.col_map.strat_vars_part) > 0:
            strat_data = self.make_strat_data()
            if self.col_map.age_grp_cnt:
                strat_data["flat_ix_exp"] = expand_ix_array(
                    strat_data["flat_ix"], base_data["bid_pad"].shape[1]
                )
        else:
            strat_data = {}

        # ============================
        # Construct ModelData
        # ============================
        self.model_data = ModelData(base_data, strat_data)

        return self.model_data
