"""Validation for contact survey data containers.

This module provides cross-dataset validation for contact matrix estimation,
ensuring consistency of stratification variables and categorical encodings
across participant, contact, population, and stratification proportion data.
"""

import warnings
from typing import Optional, Set, Tuple

import pandas as pd

from .containers import ContactData, ParticipantData, PopulationData, StratificationData


class DataValidator:
    """
    Validates the different data sources for integrity and consistency before preprocessing.

    Performs cross-dataset validation to ensure:
    - Stratification variables are consistently defined across all data containers
    - Categorical variables have aligned categories and ordering
    - Data containers are properly typed and structured

    This validator is critical for models requiring demographic stratification,
    as inconsistent category definitions can lead to silent modeling errors.

    Parameters
    ----------
    part_data : ParticipantData
        Participant survey data container
    cnt_data : ContactData
        Contact observation data container
    pop_data : PopulationData
        Population age distribution data container
    strat_data : StratificationData or None, default=None
        Stratified population proportions for demographic adjustment

    Attributes
    ----------
    part_vars : Set[str]
        Stratification variable names from participant data (with '_part' suffix)
    cnt_vars : Set[str]
        Stratification variable names from contact data (with '_cnt' suffix)
    pop_vars : Set[str]
        Stratification variable names from population data (with '_pop' suffix)
    strat_vars : Set[str]
        Stratification variable names from proportion data

    Examples
    --------
    >>> validator = DataValidator(part_data, cnt_data, pop_data, strat_data)
    >>> validated = validator.validate()
    >>> part_data, cnt_data, pop_data, strat_data = validated

    Notes
    -----
    The validator modifies categorical columns in-place to ensure consistency.
    After validation, all categorical columns across datasets will share:
    - Identical category sets
    - Identical category ordering
    - The 'ordered=True' flag

    This is essential for proper broadcasting and alignment in NumPyro models.
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        pop_data: PopulationData,
        strat_data: Optional[StratificationData] = None,
    ):
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.pop_data = pop_data
        self.strat_data = strat_data

        # Cache for stratification variable sets (computed on demand)
        self.part_vars: Optional[Set[str]] = None
        self.cnt_vars: Optional[Set[str]] = None
        self.pop_vars: Optional[Set[str]] = None
        self.strat_vars: Optional[Set[str]] = None

    def _get_strat_vars(self) -> None:
        """
        Extract and cache stratification variable names from all containers.

        Computes variable name sets once and caches them for reuse. This method
        populates the part_vars, cnt_vars, pop_vars, and strat_vars attributes.
        """
        self.part_vars = set(self.part_data.get_strat_vars(suffix=False))
        self.cnt_vars = set(self.cnt_data.get_strat_vars(suffix=False))
        self.pop_vars = set(self.pop_data.get_strat_vars(suffix=False))
        if self.strat_data:
            self.strat_vars = set(self.strat_data.get_strat_vars())
        else:
            self.strat_vars = set()

        self.part_vars_list = self.part_data.get_strat_vars(suffix=False)
        self.cnt_vars_list = self.cnt_data.get_strat_vars(suffix=False)
        self.pop_vars_list = self.pop_data.get_strat_vars(suffix=False)

        if self.strat_data:
            self.strat_vars_list = self.strat_data.get_strat_vars()
        else:
            self.strat_vars_list = []

    def _check_strat_var_consistency(self) -> None:
        """
        Check if stratification variables are consistently specified across datasets
        for the required stratification mode (PARTIAL v.s. FULL).

        1. Partially stratified scenario (PARTIAL):
            - If part_data has strat_vars, strat_data must be provided
            - The stratification variables in part_data must also be present in strat_data
            - The categories and codes for these variables must be consistent between part_data and strat_data

        2. Fully stratified scenario (FULL):
            - Contact strat_vars (base names) must match participant strat_vars
            - Population strat_vars must be defined and match participant strat_vars
            - strat_vars must match part_vars

        Notes
        -----
        This method must be called before _check_stratum_categories() since
        category validation depends on knowing which variables to check.
        """
        # Extract stratification variables from all containers
        self._get_strat_vars()

        # =======================================
        # No stratification
        # =======================================
        if not self.part_vars:
            return  # No validation required

        # =======================================
        # Partially stratified scenario (PARTIAL)
        # =======================================
        if not self.strat_data:
            warnings.warn(
                "strat_data is not provided. Continuing without stratified population proportions. "
                "Some models (e.g., BRC) may not be available.",
                UserWarning,
                stacklevel=3,
            )
        else:
            # StratificationData variables must match participant variables exactly
            if self.strat_vars != self.part_vars:
                raise ValueError(
                    "Stratification variables in StratificationData must match ParticipantData.\n"
                    f"Stratification variables in ParticipantData (no suffix): {sorted(self.part_vars)}\n"
                    f"Stratification variables in StratificationData (no suffix): {sorted(self.strat_vars)}\n"
                    "These must be identical."
                )

            # The order in which the stratification variables are specified must also match to ensure consistent category codes
            # Raise warning if the order is different, and reorder strat_data to match participant data
            if self.part_vars_list != self.strat_vars_list:
                warnings.warn(
                    "The order in which the stratification variables are specified in StratificationData does not match ParticipantData.\n"
                    f"ParticipantData strat_var_cols (no suffix): {self.part_vars_list}\n"
                    f"StratificationData strat_var_cols (no suffix): {self.strat_vars_list}\n"
                    "Reordering strat_var_cols in StratificationData to match ParticipantData variable order.",
                    UserWarning,
                    stacklevel=3,
                )
                # Reorder strat_data to match participant data variable order
                self.strat_data.strat_var_cols = [var for var in self.part_vars_list]

        if not self.cnt_vars:
            return  # Only participant-level stratification (PARTIAL mode)

        # =======================================
        # Fully stratified scenario (FULL)
        # =======================================
        # Contact variables (base names) must match participant variables
        if self.cnt_vars != self.part_vars:
            raise ValueError(
                "Stratification variables must match between ParticipantData and ContactData.\n"
                f"Stratification variables in ParticipantData (no suffix): {sorted(self.part_vars)}\n"
                f"Stratification variables in ContactData (no suffix): {sorted(self.cnt_vars)}\n"
                "For FULL stratification mode, both must have identical variables."
            )

        if self.cnt_vars_list != self.part_vars_list:
            warnings.warn(
                "The order in which the stratification variables are specified in ContactData does not match ParticipantData.\n"
                f"ParticipantData strat_var_cols (no suffix): {self.part_vars_list}\n"
                f"ContactData strat_var_cols (no suffix): {self.cnt_vars_list}\n"
                "Reordering strat_var_cols in ContactData to match ParticipantData variable order.",
                UserWarning,
                stacklevel=3,
            )
            # Reorder contact data to match participant data variable order
            self.cnt_data.strat_var_cols = [f"{var}_cnt" for var in self.part_vars_list]

        # Population must have stratification in FULL mode
        if not self.pop_vars:
            raise ValueError(
                "PopulationData must have stratification variables when using FULL stratification.\n"
                f"Stratification variables in ContactData (no suffix): {sorted(self.cnt_vars)}\n"
                "Please provide PopulationData with corresponding stratification columns."
            )

        # Population variables must match participant variables
        if self.pop_vars != self.part_vars:
            raise ValueError(
                "Stratification variables must match between ParticipantData and PopulationData.\n"
                f"ParticipantData strat_var_cols (no suffix): {sorted(self.part_vars)}\n"
                f"PopulationData strat_var_cols (no suffix): {sorted(self.pop_vars)}\n"
                "For FULL stratification mode, both must have identical variables."
            )

        if self.pop_vars_list != self.part_vars_list:
            warnings.warn(
                "The order in which the stratification variables are specified in PopulationData does not match ParticipantData.\n"
                f"ParticipantData strat_var_cols (no suffix): {self.part_vars_list}\n"
                f"PopulationData strat_var_cols (no suffix): {self.pop_vars_list}\n"
                "Reordering strat_var_cols in PopulationData to match ParticipantData variable order.",
                UserWarning,
                stacklevel=3,
            )
            # Reorder population data to match participant data variable order
            self.pop_data.strat_var_cols = [var for var in self.part_vars_list]

    def _consolidate_schema_for_variable(
        self, var: str, reference_schema: list
    ) -> None:
        """Consolidate categorical encodings for a single stratification variable.

        Ensures that all containers using a stratification variable share the same
        category set and ordering. The reference categories (from participant data)
        are propagated to other containers, validating that no unexpected categories
        exist in downstream data.

        Parameters
        ----------
        base_var : str
            Base variable name (without suffix), e.g., 'gender', 'region'
        reference_schema : list
            Ordered list of valid categories from participant data

        Raises
        ------
        ValueError
            If any container has categories not present in reference_categories

        Notes
        -----
        - Participant data categories are treated as the authoritative reference
        - Other containers may have subsets of these categories
        - Categories are ordered consistently across all containers
        - All categorical columns are set to ordered=True
        - This ensures consistent categorical codes: e.g., if participant data
          has ['M', 'F'] (M=0, F=1), all other datasets will use the same
          category order and codes, preventing silent indexing errors
        """
        # Keep ordered list for category assignment (preserves order and codes)
        reference_cats = reference_schema["categories"]
        # Use set for subset checking (order-independent)
        reference_cats_set = set(reference_cats)

        # Validate and consolidate strat_data categories
        if self.strat_data:
            current_schema = self.strat_data.get_strat_var_schema()
            current_cats = current_schema[var]["categories"]

            if not set(current_cats).issubset(reference_cats_set):
                raise ValueError(
                    f"strat_data variable '{var}' contains unexpected categories.\n"
                    f"Reference categories (ParticipantData): {reference_cats}\n"
                    f"Found in strat_data: {current_cats}\n"
                    f"Extra categories: {set(current_cats) - reference_cats_set}\n"
                    "Please ensure all categories are defined in ParticipantData first."
                )

            # Reorder categories to match reference
            self.strat_data.data[var] = pd.Categorical(
                self.strat_data.data[var],
                categories=reference_cats,
                ordered=True,
            )

        # Validate and consolidate ContactData categories (FULL mode)
        if self.cnt_vars:
            current_schema = self.cnt_data.get_strat_var_schema()
            current_cats = current_schema[var]["categories"]

            if not set(current_cats).issubset(reference_cats_set):
                raise ValueError(
                    f"ContactData variable '{var}' contains unexpected categories.\n"
                    f"Reference categories (ParticipantData): {reference_cats}\n"
                    f"Found in ContactData: {current_cats}\n"
                    f"Extra categories: {set(current_cats) - reference_cats_set}\n"
                    "Please ensure all categories are defined in ParticipantData first."
                )

            # Reorder categories to match reference
            self.cnt_data.df_cnt[f"{var}_cnt"] = pd.Categorical(
                self.cnt_data.df_cnt[f"{var}_cnt"],
                categories=reference_cats,
                ordered=True,
            )

        # Validate and consolidate PopulationData categories (FULL mode)
        if self.pop_vars:
            current_schema = self.pop_data.get_strat_var_schema()
            current_cats = current_schema[var]["categories"]

            if not set(current_cats).issubset(reference_cats_set):
                raise ValueError(
                    f"PopulationData variable '{var}' contains unexpected categories.\n"
                    f"Reference categories (ParticipantData): {reference_cats}\n"
                    f"Found in PopulationData: {current_cats}\n"
                    f"Extra categories: {set(current_cats) - reference_cats_set}\n"
                    "Please ensure all categories are defined in ParticipantData first."
                )

            # Reorder categories to match reference
            self.pop_data.df_pop[var] = pd.Categorical(
                self.pop_data.df_pop[var],
                categories=reference_cats,
                ordered=True,
            )

    def _consolidate_strat_var(self) -> None:
        """Validate and consolidate categorical encodings across all datasets.

        This method ensures that all stratification variables use consistent
        categorical encodings across ParticipantData, ContactData, PopulationData,
        and strat_data. Consistency is critical for:
        - Proper array broadcasting in NumPyro models
        - Correct alignment of population weights
        - Avoiding silent indexing errors

        Process:
        1. For each stratification variable in ParticipantData:
           - Extract the reference category ordering
           - Validate that all other containers have subset of these categories
           - Reorder categories in all containers to match reference
           - Set all categoricals to ordered=True

        The ParticipantData categories are treated as the authoritative reference
        because participants are the primary sampling unit in contact surveys.

        Raises
        ------
        ValueError
            If any container has categories not present in ParticipantData

        Notes
        -----
        - Must be called after _check_strat_var_consistency() and _ensure_categorical_dtype()
        - Modifies categorical columns in-place
        - Handles both PARTIAL (participant-only) and FULL (participant+contact) modes
        - Missing categories in downstream data are allowed (will be added)

        Examples
        --------
        If ParticipantData has gender categories ['M', 'F', 'Other']:
        - strat_data must use a subset: ['M', 'F'] is valid
        - ContactData with ['F', 'M', 'Unknown'] will raise an error ('Unknown' not in reference)
        """
        # Skip if no stratification variables
        if not self.part_data.get_strat_vars():
            return

        # Process each stratification variable
        reference_schemas = self.part_data.get_strat_var_schema()
        for var in self.part_vars:
            reference_schema = reference_schemas[var]

            # Consolidate categories across all containers
            self._consolidate_schema_for_variable(var, reference_schema)

    def validate(
        self,
    ) -> Tuple[
        ParticipantData, ContactData, PopulationData, Optional[StratificationData]
    ]:
        """Execute all validation checks and return validated data containers.

        Performs comprehensive validation of contact survey data:
        1. Validates stratification variable consistency across datasets
        2. Consolidates categorical encodings for aligned indexing

        Returns
        -------
        tuple of (ParticipantData, ContactData, PopulationData, strat_data or None)
            Validated data containers with consistent categorical encodings.
            Categorical columns are modified in-place during validation.

        Raises
        ------
        ValueError
            If stratification variables are inconsistent or categories don't align

        Examples
        --------
        >>> validator = DataValidator(part_data, cnt_data, pop_data, strat_data)
        >>> part, cnt, pop, strat = validator.validate()
        >>> # All categorical columns now have consistent categories and ordering

        Notes
        -----
        This method should be called before passing data to BRC model classes.
        The validation process modifies categorical columns in-place to ensure
        consistency, which is essential for proper array broadcasting in JAX/NumPyro.
        """
        self._check_strat_var_consistency()
        self._consolidate_strat_var()

        return (
            self.part_data,
            self.cnt_data,
            self.pop_data,
            self.strat_data,
        )
