"""
Prem2 Validation Module

Handles validation logic for Prem2 preprocessing, including:
- Shared stratification variable validation
- Empty age group detection
"""

from typing import Dict, List

import pandas as pd

from ...dataloader import ContactData, ParticipantData
from ...utils import AgeGroupSpecs
from ._socialmix_age_processing import AgeBinProcessor
from ._socialmix_helpers import infer_strat_mode


class Prem2Validator:
    """
    Validates Prem2 inputs.

    This class encapsulates validation logic for the Prem2 preprocessing
    pipeline: stratification consistency checks and empty age group detection.
    Unlike :class:`SocialMixValidator`, there is no population data, reciprocity,
    adaptive merging, or bootstrap-stability validation.

    Parameters
    ----------
    part_data : ParticipantData
        Participant data container
    cnt_data : ContactData
        Contact data container
    age_group_specs : AgeGroupSpecs
        Age stratification bins

    Attributes
    ----------
    age_processor : AgeBinProcessor
        Helper for age group operations
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_group_specs: AgeGroupSpecs,
    ):
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_group_specs = age_group_specs

        # Initialize age processor
        self.age_processor = AgeBinProcessor(age_group_specs)

        # Stratification variables
        self.strat_vars_part: List[str] = []
        self.strat_vars_cnt: List[str] = []
        self.strat_vars_shared: List[str] = []
        self.strat_mode: str = None

    def validate_all(self) -> Dict:
        """
        Run all validation checks and return updated components.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'part_data': ParticipantData (unchanged)
            - 'cnt_data': ContactData (unchanged)
            - 'age_group_specs': AgeGroupSpecs (unchanged)

        Notes
        -----
        This method orchestrates validation in the correct order:
        1. Extract stratification variables
        2. Validate shared stratification variables
        3. Validate estimation requirements (no empty age groups)
        """
        # Extract stratification variables
        self._extract_strat_vars()

        # Validate shared stratification variables (if any)
        if self.strat_vars_shared:
            self._validate_shared_strat_vars()

        # Validate estimation requirements (no empty age groups)
        self._validate_estimation_requirements()

        return {
            "part_data": self.part_data,
            "cnt_data": self.cnt_data,
            "age_group_specs": self.age_group_specs,
        }

    def _extract_strat_vars(self) -> None:
        """Extract stratification variables from data containers."""
        self.strat_vars_part = self.part_data.get_strat_vars(prefix=False)
        self.strat_vars_cnt = self.cnt_data.get_strat_vars(prefix=False)

        # Identify shared variables
        self.strat_vars_shared = sorted(
            list(set(self.strat_vars_part) & set(self.strat_vars_cnt))
        )

        # Infer stratification mode
        self._infer_strat_mode()

    def _infer_strat_mode(self) -> None:
        """Infer and cache the stratification mode."""
        self.strat_mode = infer_strat_mode(self.strat_vars_part, self.strat_vars_cnt)

    def _validate_shared_strat_vars(self) -> None:
        """
        Validate and align shared stratification variables.

        For variables that appear in both participant and contact data, this
        ensures:
        1. Both sides have the same unique categories
        2. Categories are encoded in the same order (using participant order as
           reference)

        If categories match but order differs, contact data is automatically
        reordered to match participant data encoding.

        Raises
        ------
        ValueError
            If categories don't match (different sets of values).
        """
        for var in self.strat_vars_shared:
            # Get column names
            col_part = f"part_{var}"
            col_cnt = f"cnt_{var}"

            # Get categories from both sides
            part_col = self.part_data.data[col_part]
            cnt_col = self.cnt_data.data[col_cnt]

            # Convert to categorical if not already
            if not hasattr(part_col, "cat"):
                part_col = part_col.astype("category")
                self.part_data.data[col_part] = part_col
            if not hasattr(cnt_col, "cat"):
                cnt_col = cnt_col.astype("category")
                self.cnt_data.data[col_cnt] = cnt_col

            # Get categories in encoding order
            part_cats = list(part_col.cat.categories)
            cnt_cats = list(cnt_col.cat.categories)

            # Check if categories match as sets
            part_set = set(part_cats)
            cnt_set = set(cnt_cats)

            if part_set != cnt_set:
                # Different categories - this is an error
                only_part = part_set - cnt_set
                only_cnt = cnt_set - part_set
                raise ValueError(
                    f"Shared stratification variable '{var}' has different categories:\n"
                    f"  Participant side: {part_cats}\n"
                    f"  Contact side: {cnt_cats}\n"
                    f"  Only in participants: {sorted(only_part) if only_part else 'None'}\n"
                    f"  Only in contacts: {sorted(only_cnt) if only_cnt else 'None'}\n"
                    f"For shared variables, both sides must have identical categories."
                )

            # Same categories but possibly different order
            # Use participant ordering as reference and reorder contact data
            if part_cats != cnt_cats:
                self.cnt_data.data[col_cnt] = self.cnt_data.data[
                    col_cnt
                ].cat.reorder_categories(part_cats, ordered=False)

    def _validate_estimation_requirements(self) -> None:
        """
        Validate that contact intensity matrix can be estimated.

        Ensures that there are no empty participant age groups, which would
        cause division by zero when computing contact intensities.

        Raises
        ------
        ValueError
            If there are empty age groups.
        """
        # Verify that the number of bins in age_group_specs matches the actual
        # number of age-group categories in the data.  A mismatch typically means
        # the data was pre-binned with pd.cut(..., bins=range(0, N, step)) whose
        # N is an *exclusive* upper edge, while AgeGroupSpecs(0, N, step) treats
        # N as an *inclusive* maximum age, producing one extra bin.
        n_bins = len(self.age_group_specs.left)
        n_part_cats = len(self.part_data.data["part_age_grp"].cat.categories)
        n_cnt_cats = len(self.cnt_data.data["cnt_age_grp"].cat.categories)
        if n_part_cats != n_bins or n_cnt_cats != n_bins:
            raise ValueError(
                f"age_group_specs defines {n_bins} age bins, but the data has "
                f"{n_part_cats} participant age group(s) and {n_cnt_cats} contact "
                f"age group(s). "
                f"If you pre-binned the data with "
                f"pd.cut(..., bins=range(0, {self.age_group_specs.max + 1}, {self.age_group_specs.step or 'step'}), right=False), "
                f"note that AgeGroupSpecs treats 'max' as an inclusive upper bound. "
                f"Use AgeGroupSpecs(0, {self.age_group_specs.max - (self.age_group_specs.step or 1)}, "
                f"{self.age_group_specs.step or 'step'}) so that the last bin ends at "
                f"{self.age_group_specs.max - 1}."
            )

        # Determine grouping columns based on stratification
        group_cols = [f"part_{var}" for var in self.strat_vars_part] + ["part_age_grp"]

        # Compute counts for all (stratum, age_group) combinations
        if self.strat_vars_part:
            # Stratified case: group by strat vars + age
            strata_age_counts = self.part_data.data.groupby(
                group_cols, observed=False
            ).size()

            # For each age group, find minimum count across all strata
            age_min_counts = strata_age_counts.groupby(
                "part_age_grp", observed=False
            ).min()

            # Reindex to ensure all age groups present
            age_min_counts = age_min_counts.reindex(
                pd.Index(self.part_data.data["part_age_grp"].cat.categories),
                fill_value=0,
            )

            empty_groups = age_min_counts[age_min_counts == 0].index.tolist()
        else:
            # No stratification: simple age group counts
            age_grp_counts = (
                self.part_data.data.groupby("part_age_grp", observed=False)
                .size()
                .reindex(
                    pd.Index(self.part_data.data["part_age_grp"].cat.categories),
                    fill_value=0,
                )
            )

            empty_groups = age_grp_counts[age_grp_counts == 0].index.tolist()

        # Handle empty age groups
        if empty_groups:
            strata_info = ""
            if self.strat_vars_part:
                strata_info = f" across {len(self.strat_vars_part)} stratification variable(s)"

            raise ValueError(
                f"\nCannot estimate contact intensity matrix: found empty participant age group(s){strata_info}: {empty_groups}. "
                f"\nEmpty age groups cause division by zero when computing contact intensities. "
                f"\nPlease either:\n"
                f"  1. Use coarser age bins that avoid empty groups, or\n"
                f"  2. Collect more participant data to fill all age groups."
            )
