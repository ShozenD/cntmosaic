"""Validation for contact survey data containers.

This module provides cross-dataset validation for contact matrix estimation,
ensuring consistency of stratification variables and categorical encodings
across participant, contact, population, and stratification proportion data.
"""

import warnings
from typing import Optional, Set, Tuple

import pandas as pd

from .containers import ContactData, ParticipantData, PopulationData, StratPropData


class DataValidator:
    """Validator for contact survey data consistency and integrity.

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
    strat_prop_data : StratPropData or None, default=None
        Stratified population proportions for demographic adjustment

    Attributes
    ----------
    part_vars : Set[str]
        Stratification variable names from participant data (with '_part' suffix)
    cnt_vars : Set[str]
        Stratification variable names from contact data (with '_cnt' suffix)
    pop_vars : Set[str]
        Stratification variable names from population data (with '_pop' suffix)
    strat_prop_vars : Set[str]
        Stratification variable names from proportion data

    Examples
    --------
    >>> validator = DataValidator(part_data, cnt_data, pop_data, strat_prop_data)
    >>> validated = validator.validate()
    >>> part_data, cnt_data, pop_data, strat_prop_data = validated

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
        strat_prop_data: Optional[StratPropData] = None,
    ):
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.pop_data = pop_data
        self.strat_prop_data = strat_prop_data

        # Cache for stratification variable sets (computed on demand)
        self.part_vars: Optional[Set[str]] = None
        self.cnt_vars: Optional[Set[str]] = None
        self.pop_vars: Optional[Set[str]] = None
        self.strat_prop_vars: Optional[Set[str]] = None

    def _get_strat_vars(self) -> None:
        """Extract and cache stratification variable names from all containers.

        Computes variable name sets once and caches them for reuse. This method
        populates the part_vars, cnt_vars, pop_vars, and strat_prop_vars attributes.
        """
        self.part_vars = set(self.part_data.get_strat_vars(suffix=False) or [])
        self.cnt_vars = set(self.cnt_data.get_strat_vars(suffix=False) or [])
        self.pop_vars = set(self.pop_data.strat_var_cols or [])
        self.strat_prop_vars = (
            set(self.strat_prop_data.strat_var_cols or [])
            if self.strat_prop_data
            else set()
        )

    @staticmethod
    def _strip_suffix(var_names: Set[str], suffix: str) -> Set[str]:
        return {v.replace(suffix, "") for v in var_names}

    def _check_strat_var_consistency(self) -> None:
        """Validate stratification variable consistency across all datasets.

        Ensures that stratification variables are properly defined and aligned:

        1. **Participant-level stratification** (PARTIAL mode):
           - If part_data has strat_vars, strat_prop_data must be provided
           - strat_prop_vars must exactly match part_vars

        2. **Full stratification** (FULL mode - both participant and contact):
           - Contact strat_vars (base names) must match participant strat_vars
           - Population strat_vars must be defined and match participant strat_vars
           - strat_prop_vars must match part_vars

        Variable naming convention:
        - Participant variables: 'gender_part', 'region_part'
        - Contact variables: 'setting_cnt', 'duration_cnt'
        - Population variables: 'gender_pop', 'region_pop'
        - StratProp variables: 'gender', 'region'

        Raises
        ------
        ValueError
            If stratification variables are inconsistent or improperly defined

        Notes
        -----
        This method must be called before _check_stratum_categories() since
        category validation depends on knowing which variables to check.
        """
        # Extract stratification variables from all containers
        self._get_strat_vars()

        # No validation needed if no stratification
        if not self.part_vars:
            return

        # Get base variable names (without suffixes)
        part_vars_base = self._strip_suffix(self.part_vars, "_part")

        # Validate PARTIAL mode: Participant stratification requires StratPropData
        if not self.strat_prop_data:
            warnings.warn(
                "StratPropData is not provided. Continuing without stratified population proportions. "
                "Some models (e.g., BRC) may not be available.",
                UserWarning,
                stacklevel=3,
            )
        else:
            # StratPropData variables must match participant variables exactly
            if self.strat_prop_vars != self.part_vars:
                raise ValueError(
                    "Stratification variables in StratPropData must match ParticipantData.\n"
                    f"ParticipantData strat_var_cols: {sorted(self.part_vars)}\n"
                    f"StratPropData strat_var_cols: {sorted(self.strat_prop_vars)}\n"
                    "These must be identical for proper demographic weighting."
                )

        # Validate FULL mode: Contact-level stratification
        if not self.cnt_vars:
            return  # Only participant-level stratification (PARTIAL mode)

        # Contact variables (base names) must match participant variables
        cnt_vars_base = self._strip_suffix(self.cnt_vars, "_cnt")
        if cnt_vars_base != part_vars_base:
            raise ValueError(
                "Stratification variables must match between ParticipantData and ContactData.\n"
                f"ParticipantData strat_var_cols (base): {sorted(part_vars_base)}\n"
                f"ContactData strat_var_cols (base): {sorted(cnt_vars_base)}\n"
                "For FULL stratification mode, both must have identical variables."
            )

        # Population must have stratification in FULL mode
        if not self.pop_vars:
            raise ValueError(
                "PopulationData must have stratification variables when using FULL stratification.\n"
                f"ContactData defines strat_var_cols: {sorted(self.cnt_vars)}\n"
                "Please provide PopulationData with corresponding stratification columns."
            )

        # Population variables (base names) must match participant variables
        pop_vars_base = self._strip_suffix(self.pop_vars, "_pop")
        if pop_vars_base != part_vars_base:
            raise ValueError(
                "Stratification variables must match between ParticipantData and PopulationData.\n"
                f"ParticipantData strat_var_cols (base): {sorted(part_vars_base)}\n"
                f"PopulationData strat_var_cols (base): {sorted(pop_vars_base)}\n"
                "For FULL stratification mode, both must have identical variables."
            )

    def _consolidate_categories_for_variable(
        self, base_var: str, reference_categories: list
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
        reference_categories : list
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
        """
        # Validate and consolidate StratPropData categories
        if self.strat_prop_data:
            prop_col = f"{base_var}"
            current_categories = self.strat_prop_data.data[
                prop_col
            ].cat.categories.tolist()

            if not set(current_categories).issubset(set(reference_categories)):
                raise ValueError(
                    f"StratPropData variable '{base_var}' contains unexpected categories.\n"
                    f"Reference categories (ParticipantData): {reference_categories}\n"
                    f"Found in StratPropData: {current_categories}\n"
                    f"Extra categories: {set(current_categories) - set(reference_categories)}\n"
                    "Please ensure all categories are defined in ParticipantData first."
                )

            # Reorder categories to match reference
            self.strat_prop_data.data[prop_col] = pd.Categorical(
                self.strat_prop_data.data[prop_col],
                categories=reference_categories,
                ordered=True,
            )

        # Validate and consolidate ContactData categories (FULL mode)
        if self.cnt_vars:
            cnt_col = f"{base_var}_cnt"
            if cnt_col in self.cnt_data.df_cnt.columns:
                current_categories = self.cnt_data.df_cnt[
                    cnt_col
                ].cat.categories.tolist()

                if not set(current_categories).issubset(set(reference_categories)):
                    raise ValueError(
                        f"ContactData variable '{base_var}' contains unexpected categories.\n"
                        f"Reference categories (ParticipantData): {reference_categories}\n"
                        f"Found in ContactData: {current_categories}\n"
                        f"Extra categories: {set(current_categories) - set(reference_categories)}\n"
                        "Please ensure all categories are defined in ParticipantData first."
                    )

                # Reorder categories to match reference
                self.cnt_data.df_cnt[cnt_col] = pd.Categorical(
                    self.cnt_data.df_cnt[cnt_col],
                    categories=reference_categories,
                    ordered=True,
                )

        # Validate and consolidate PopulationData categories (FULL mode)
        if self.pop_vars:
            pop_col = f"{base_var}_pop"
            if pop_col in self.pop_data.df_pop.columns:
                current_categories = self.pop_data.df_pop[
                    pop_col
                ].cat.categories.tolist()

                if not set(current_categories).issubset(set(reference_categories)):
                    raise ValueError(
                        f"PopulationData variable '{base_var}' contains unexpected categories.\n"
                        f"Reference categories (ParticipantData): {reference_categories}\n"
                        f"Found in PopulationData: {current_categories}\n"
                        f"Extra categories: {set(current_categories) - set(reference_categories)}\n"
                        "Please ensure all categories are defined in ParticipantData first."
                    )

                # Reorder categories to match reference
                self.pop_data.df_pop[pop_col] = pd.Categorical(
                    self.pop_data.df_pop[pop_col],
                    categories=reference_categories,
                    ordered=True,
                )

    def _ensure_categorical_dtype(self) -> None:
        """Ensure all stratification variables have categorical dtype.

        Automatically converts stratification variables to categorical dtype if they
        are not already categorical. Issues a warning to inform users about the
        conversion and that the ordering is determined by sorted unique values.

        Notes
        -----
        - Modifies dataframes in-place
        - Uses sorted unique values to determine category ordering
        - Should be called before _check_stratum_categories()
        """
        # Skip if no stratification variables
        if not self.part_data.strat_var_cols:
            return

        converted_vars = []

        # Check and convert ParticipantData
        for col in self.part_vars:
            if col in self.part_data.df_part.columns:
                if (
                    isinstance(self.part_data.df_part[col].dtype, pd.CategoricalDtype)
                    is False
                ):
                    self.part_data.df_part[col] = pd.Categorical(
                        self.part_data.df_part[col],
                        categories=sorted(self.part_data.df_part[col].unique()),
                        ordered=True,
                    )
                    converted_vars.append(f"ParticipantData.{col}")

        # Check and convert ContactData
        for col in self.cnt_vars:
            if col in self.cnt_data.df_cnt.columns:
                if (
                    isinstance(self.cnt_data.df_cnt[col].dtype, pd.CategoricalDtype)
                    is False
                ):
                    self.cnt_data.df_cnt[col] = pd.Categorical(
                        self.cnt_data.df_cnt[col],
                        categories=sorted(self.cnt_data.df_cnt[col].unique()),
                        ordered=True,
                    )
                    converted_vars.append(f"ContactData.{col}")

        # Check and convert PopulationData
        for col in self.pop_vars:
            if col in self.pop_data.df_pop.columns:
                if (
                    isinstance(self.pop_data.df_pop[col].dtype, pd.CategoricalDtype)
                    is False
                ):
                    self.pop_data.df_pop[col] = pd.Categorical(
                        self.pop_data.df_pop[col],
                        categories=sorted(self.pop_data.df_pop[col].unique()),
                        ordered=True,
                    )
                    converted_vars.append(f"PopulationData.{col}")

        # Check and convert StratPropData
        if self.strat_prop_data:
            for col in self.strat_prop_vars:
                # StratPropData uses base variable names without suffix
                base_col = col.replace("_part", "")
                if base_col in self.strat_prop_data.data.columns:
                    if (
                        isinstance(
                            self.strat_prop_data.data[base_col].dtype,
                            pd.CategoricalDtype,
                        )
                        is False
                    ):
                        self.strat_prop_data.data[base_col] = pd.Categorical(
                            self.strat_prop_data.data[base_col],
                            categories=sorted(
                                self.strat_prop_data.data[base_col].unique()
                            ),
                            ordered=True,
                        )
                        converted_vars.append(f"StratPropData.{base_col}")

        # Warn user if any variables were converted
        if converted_vars:
            warnings.warn(
                f"Automatically converted the following stratification variables to categorical dtype: "
                f"{', '.join(converted_vars)}. "
                f"Categories are ordered alphabetically/numerically by sorted unique values. "
                f"If you need a specific ordering, please convert to pd.Categorical with explicit "
                f"category ordering before creating the DataLoader.",
                UserWarning,
                stacklevel=4,
            )

    def _check_stratum_categories(self) -> None:
        """Validate and consolidate categorical encodings across all datasets.

        This method ensures that all stratification variables use consistent
        categorical encodings across ParticipantData, ContactData, PopulationData,
        and StratPropData. Consistency is critical for:
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
        - StratPropData must use a subset: ['M', 'F'] is valid
        - ContactData with ['F', 'M', 'Unknown'] will raise an error ('Unknown' not in reference)
        """
        # Skip if no stratification variables
        if not self.part_data.strat_var_cols:
            return

        # Process each stratification variable
        part_vars_base = self._strip_suffix(self.part_vars, "_part")
        for base_var in part_vars_base:
            part_col = f"{base_var}_part"
            reference_categories = self.part_data.df_part[
                part_col
            ].cat.categories.tolist()

            # Consolidate categories across all containers
            self._consolidate_categories_for_variable(base_var, reference_categories)

    def validate(
        self,
    ) -> Tuple[ParticipantData, ContactData, PopulationData, Optional[StratPropData]]:
        """Execute all validation checks and return validated data containers.

        Performs comprehensive validation of contact survey data:
        1. Validates stratification variable consistency across datasets
        2. Consolidates categorical encodings for aligned indexing

        Returns
        -------
        tuple of (ParticipantData, ContactData, PopulationData, StratPropData or None)
            Validated data containers with consistent categorical encodings.
            Categorical columns are modified in-place during validation.

        Raises
        ------
        ValueError
            If stratification variables are inconsistent or categories don't align

        Examples
        --------
        >>> validator = DataValidator(part_data, cnt_data, pop_data, strat_prop_data)
        >>> part, cnt, pop, strat_prop = validator.validate()
        >>> # All categorical columns now have consistent categories and ordering

        Notes
        -----
        This method should be called before passing data to BRC model classes.
        The validation process modifies categorical columns in-place to ensure
        consistency, which is essential for proper array broadcasting in JAX/NumPyro.
        """
        self._check_strat_var_consistency()
        self._ensure_categorical_dtype()
        self._check_stratum_categories()

        return (
            self.part_data,
            self.cnt_data,
            self.pop_data,
            self.strat_prop_data,
        )
