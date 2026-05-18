"""
Data Loading Utilities for SocialMix

This module provides helper classes for loading and processing stratified contact data.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...dataloader import ContactData, ParticipantData, PopulationData
from ...utils import AgeGroupSpecs

if TYPE_CHECKING:
    from ._SocialMix import SocialMix


class SocialMixDataLoader:
    """
    Helper class for loading and processing stratified contact data.

    This class handles the conversion of contact survey data into the
    numerical arrays needed for matrix estimation, with full support for
    stratification.
    """

    def __init__(self, socialmix_instance: "SocialMix"):
        """
        Initialize data processor.

        Parameters
        ----------
        socialmix_instance : SocialMix
            Reference to parent SocialMix instance for accessing data and state
        """
        self.sm = socialmix_instance

    def load_data(self) -> None:
        """
        Orchestrate the data loading process.

        Sets the following attributes on the parent SocialMix instance:
        - Y: Contact count matrix/tensor
        - N: Participant count matrix/vector
        - P: Population size matrix/vector (if applicable)
        - C, D, K_part, K_cnt: Dimensions
        """
        # 1. Calculate dimensions
        self._calculate_dimensions()

        # 2. Compute N: participant counts
        self._compute_N()

        # 3. Compute Y: contact counts
        self._compute_Y()

        # 4. Compute P: population sizes (if pop_data provided)
        if self.sm.pop_data is not None:
            self._assign_age_groups_to_population()
            self._compute_P()
        else:
            self.sm.P = None

    def _calculate_dimensions(self) -> None:
        """
        Calculate and store stratification dimensions.
        """
        # Get age group dimensions
        self.sm.C = len(self.sm.part_data.data["age_grp_part"].cat.categories)
        self.sm.D = len(self.sm.cnt_data.data["age_grp_cnt"].cat.categories)

        # Calculate K_part: number of participant strata
        if self.sm.strat_vars_part:
            self.sm.K_part = 1
            for var in self.sm.strat_vars_part:
                self.sm.K_part *= self.sm.strat_dims_part[var]
        else:
            self.sm.K_part = 1

        # Calculate K_cnt: number of contact strata
        if self.sm.strat_vars_cnt:
            self.sm.K_cnt = 1
            for var in self.sm.strat_vars_cnt:
                self.sm.K_cnt *= self.sm.strat_dims_cnt[var]
        else:
            self.sm.K_cnt = 1

    def _compute_N(self) -> None:
        """
        Compute participant counts per age group and stratum.

        Sets
        ----
        self.sm.N : NDArray
            - Shape (C,) if K_part == 1 (no stratification)
            - Shape (K_part, C) if K_part > 1 (stratified)
        """
        df_part = self.sm.part_data.data
        age_grps_part = df_part["age_grp_part"].cat.categories

        if self.sm.K_part == 1:
            # No stratification: simple count by age group
            counts = (
                df_part.groupby("age_grp_part", observed=False)
                .size()
                .reindex(pd.Index(age_grps_part), fill_value=0)
            )
            self.sm.N = counts.values.astype(np.int64)
        else:
            # Stratified: group by strat vars + age
            group_cols = [f"{var}_part" for var in self.sm.strat_vars_part] + [
                "age_grp_part"
            ]
            counts = df_part.groupby(group_cols, observed=False).size()

            # Reshape to (K_part, C) matrix
            self.sm.N = self._series_to_stratified_array(
                counts,
                shape=(self.sm.K_part, self.sm.C),
                strat_vars=self.sm.strat_vars_part,
                suffix="part",
            )

    def _compute_Y(self) -> None:
        """
        Compute contact count matrix with stratification.

        Sets
        ----
        self.sm.Y : NDArray
            - Shape (C, D) if K_part == 1 and K_cnt == 1 (no stratification)
            - Shape (K_part, C, D) if K_part > 1 and K_cnt == 1 (partial)
            - Shape (K_part, K_cnt, C, D) if both K_part > 1 and K_cnt > 1 (full/mixed)
        """
        df_part = self.sm.part_data.data
        df_cnt = self.sm.cnt_data.data
        age_grps_part = df_part["age_grp_part"].cat.categories
        age_grps_cnt = df_cnt["age_grp_cnt"].cat.categories

        # Merge participants with contacts
        part_cols = ["id", "age_grp_part"] + [
            f"{v}_part" for v in self.sm.strat_vars_part
        ]
        merged = df_cnt.merge(df_part[part_cols], on="id", how="inner")

        if len(merged) == 0:
            raise ValueError(
                "No valid contacts after merging with participants. "
                "Check that contact IDs match participant IDs."
            )

        if self.sm.K_part == 1 and self.sm.K_cnt == 1:
            # No stratification: simple (C, D) matrix
            Y_agg = (
                merged.groupby(["age_grp_part", "age_grp_cnt"], observed=False)["y"]
                .sum()
                .unstack(fill_value=0)
            )
            Y_df = Y_agg.reindex(
                index=pd.Index(age_grps_part),
                columns=pd.Index(age_grps_cnt),
                fill_value=0,
            )
            self.sm.Y = Y_df.values.astype(np.float64)

        elif self.sm.K_part > 1 and self.sm.K_cnt == 1:
            # Partial stratification: (K_part, C, D) tensor
            group_cols = [f"{v}_part" for v in self.sm.strat_vars_part] + [
                "age_grp_part",
                "age_grp_cnt",
            ]
            Y_agg = merged.groupby(group_cols, observed=False)["y"].sum()

            self.sm.Y = self._series_to_stratified_array(
                Y_agg,
                shape=(self.sm.K_part, self.sm.C, self.sm.D),
                strat_vars=self.sm.strat_vars_part,
                suffix="part",
                has_contact_age=True,
            )

        else:
            # Full/mixed stratification: (K_part, K_cnt, C, D) tensor
            group_cols = (
                [f"{v}_part" for v in self.sm.strat_vars_part]
                + ["age_grp_part"]
                + [f"{v}_cnt" for v in self.sm.strat_vars_cnt]
                + ["age_grp_cnt"]
            )
            Y_agg = merged.groupby(group_cols, observed=False)["y"].sum()

            self.sm.Y = self._series_to_stratified_array(
                Y_agg,
                shape=(self.sm.K_part, self.sm.K_cnt, self.sm.C, self.sm.D),
                strat_vars_part=self.sm.strat_vars_part,
                strat_vars_cnt=self.sm.strat_vars_cnt,
                is_full=True,
            )

    def _compute_P(self) -> None:
        """
        Compute population sizes per age group and stratum.

        Sets
        ----
        self.sm.P : NDArray
            - Shape (D,) if K_cnt == 1 (no contact stratification)
            - Shape (K_cnt, D) if K_cnt > 1 (contact stratified)
        """
        df_pop = self.sm.pop_data.data
        age_grps = df_pop["age_grp"].cat.categories

        if self.sm.K_cnt == 1:
            # No stratification: simple sum by age group
            pop_sizes = (
                df_pop.groupby("age_grp", observed=False)["P"]
                .sum()
                .reindex(pd.Index(age_grps), fill_value=0)
            )

            if (pop_sizes == 0).any():
                zero_groups = pop_sizes[pop_sizes == 0].index.tolist()
                raise ValueError(
                    f"Age groups have zero population: {zero_groups}. "
                    f"Check that population data covers all age bins."
                )

            self.sm.P = pop_sizes.values.astype(np.int64)
        else:
            # Stratified: group by strat vars + age
            group_cols = self.sm.strat_vars_pop + ["age_grp"]
            pop_sizes = df_pop.groupby(group_cols, observed=False)["P"].sum()

            # Reshape to (K_cnt, D) matrix
            self.sm.P = self._series_to_stratified_array(
                pop_sizes,
                shape=(self.sm.K_cnt, self.sm.D),
                strat_vars=self.sm.strat_vars_pop,
                suffix=None,  # Population data doesn't have _part/_cnt suffix
            )

    def _assign_age_groups_to_population(self) -> None:
        """
        Assign age groups to population data using same bins as contacts.
        """
        if "age_grp" in self.sm.pop_data.data.columns:
            return  # Already assigned

        # Use same bins as participant/contact data
        bin_edges = self.sm.age_group_specs.left + [self.sm.age_group_specs.right[-1] + 1]
        intervals = [
            pd.Interval(left=l, right=r + 1, closed="left")
            for l, r in zip(self.sm.age_group_specs.left, self.sm.age_group_specs.right)
        ]

        ages = self.sm.pop_data.data["age"]
        age_grps = pd.cut(ages, bins=bin_edges, right=False, labels=intervals)
        self.sm.pop_data.data["age_grp"] = age_grps

    def _series_to_stratified_array(
        self,
        series: pd.Series,
        shape: Tuple[int, ...],
        strat_vars: Optional[List[str]] = None,
        suffix: Optional[str] = None,
        has_contact_age: bool = False,
        strat_vars_part: Optional[List[str]] = None,
        strat_vars_cnt: Optional[List[str]] = None,
        is_full: bool = False,
    ) -> NDArray:
        """
        Convert multi-indexed Series to numpy array with specified shape.

        Parameters
        ----------
        series : pd.Series
            Multi-indexed series from groupby operation
        shape : tuple
            Target array shape
        strat_vars : list, optional
            Stratification variables (for partial mode)
        suffix : str, optional
            Column suffix ('part', 'cnt', or None for population)
        has_contact_age : bool
            Whether series includes contact age dimension
        strat_vars_part : list, optional
            Participant strat vars (for full mode)
        strat_vars_cnt : list, optional
            Contact strat vars (for full mode)
        is_full : bool
            Whether this is full stratification mode

        Returns
        -------
        NDArray
            Array with specified shape filled from series data
        """
        arr = np.zeros(shape, dtype=np.float64)

        # Reset index to get all coordinates as columns
        df = series.reset_index()
        df.columns = [*df.columns[:-1], "value"]

        if is_full:
            # Full mode: (K_part, K_cnt, C, D)
            # Get categorical codes for all dimensions
            for var in strat_vars_part:
                df[f"{var}_part_code"] = df[f"{var}_part"].cat.codes
            for var in strat_vars_cnt:
                df[f"{var}_cnt_code"] = df[f"{var}_cnt"].cat.codes
            df["age_grp_part_code"] = df["age_grp_part"].cat.codes
            df["age_grp_cnt_code"] = df["age_grp_cnt"].cat.codes

            # Create composite stratum codes
            df["k_part"] = self._create_composite_index(
                df,
                [f"{v}_part_code" for v in strat_vars_part],
                [self.sm.strat_dims_part[v] for v in strat_vars_part],
            )
            df["k_cnt"] = self._create_composite_index(
                df,
                [f"{v}_cnt_code" for v in strat_vars_cnt],
                [self.sm.strat_dims_cnt[v] for v in strat_vars_cnt],
            )

            # Fill array
            for _, row in df.iterrows():
                arr[
                    int(row["k_part"]),
                    int(row["k_cnt"]),
                    int(row["age_grp_part_code"]),
                    int(row["age_grp_cnt_code"]),
                ] = row["value"]

        elif has_contact_age:
            # Partial mode: (K_part, C, D)
            for var in strat_vars:
                col_name = f"{var}_{suffix}" if suffix else var
                df[f"{var}_code"] = df[col_name].cat.codes
            df["age_grp_part_code"] = df["age_grp_part"].cat.codes
            df["age_grp_cnt_code"] = df["age_grp_cnt"].cat.codes

            # Create composite stratum code
            df["k"] = self._create_composite_index(
                df,
                [f"{v}_code" for v in strat_vars],
                [self.sm.strat_dims_part[v] for v in strat_vars],
            )

            # Fill array
            for _, row in df.iterrows():
                arr[
                    int(row["k"]),
                    int(row["age_grp_part_code"]),
                    int(row["age_grp_cnt_code"]),
                ] = row["value"]

        else:
            # Simple stratified: (K, C) or (K, D)
            for var in strat_vars:
                col_name = f"{var}_{suffix}" if suffix else var
                df[f"{var}_code"] = df[col_name].cat.codes

            # Determine age column name
            if "age_grp_part" in df.columns:
                age_col = "age_grp_part"
            elif "age_grp" in df.columns:
                age_col = "age_grp"
            else:
                raise ValueError("No age group column found")

            df["age_code"] = df[age_col].cat.codes

            # Create composite stratum code
            if strat_vars:
                strat_dims = [
                    self.sm.strat_dims_part.get(v) or self.sm.strat_dims_cnt.get(v)
                    for v in strat_vars
                ]
                df["k"] = self._create_composite_index(
                    df, [f"{v}_code" for v in strat_vars], strat_dims
                )
            else:
                df["k"] = 0

            # Fill array
            for _, row in df.iterrows():
                arr[int(row["k"]), int(row["age_code"])] = row["value"]

        return arr

    @staticmethod
    def _create_composite_index(
        df: pd.DataFrame, code_cols: List[str], dims: List[int]
    ) -> pd.Series:
        """
        Create composite index from multiple categorical codes.

        Similar to np.ravel_multi_index but for pandas.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with code columns
        code_cols : list
            Names of columns containing categorical codes
        dims : list
            Dimensions for each categorical variable

        Returns
        -------
        pd.Series
            Composite integer index
        """
        if not code_cols:
            return pd.Series(0, index=df.index)

        composite = df[code_cols[0]].astype(int)
        multiplier = 1

        for i in range(1, len(code_cols)):
            multiplier *= dims[i - 1]
            composite = composite + multiplier * df[code_cols[i]].astype(int)

        return composite
