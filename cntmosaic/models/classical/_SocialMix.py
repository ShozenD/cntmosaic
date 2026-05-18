"""
Social Contact Matrix Estimation

This module implements the socialmixr algorithm for estimating age-structured
social contact matrices from survey data. Based on Funk et al. (2024).

Key Features:
- Contact intensity and rate matrix estimation
- Optional reciprocity (symmetry) adjustment
- Adaptive merging of zero-sample age groups
- Bootstrap uncertainty quantification
- Comprehensive input validation
"""

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...dataloader import ContactData, ParticipantData, PopulationData
from ...utils import AgeGroupSpecs
from ._base import DeterministicContactModel
from ._socialmix_age_processing import AgeBinProcessor
from ._socialmix_bootstrap import BootstrapResults, SocialMixBootstrap
from ._socialmix_utils import SocialMixDataLoader
from ._socialmix_validation import SocialMixValidator

# ============================================================================
# Main Class
# ============================================================================


class SocialMix(DeterministicContactModel):
    """
    Estimate age-structured social contact matrices from survey data.

    Implements the socialmixr algorithm (Funk et al. 2024) for computing
    contact intensity and contact rate matrices from participant and contact
    data, with optional reciprocity adjustment and bootstrap uncertainty.

    Parameters
    ----------
    df_part : pd.DataFrame
        Participant data with columns:
        - 'id': unique participant identifier
        - 'age_part': participant age (numeric)
        - 'age_grp_part': participant age group (pd.Interval, categorical)
    df_cnt : pd.DataFrame
        Contact data with columns:
        - 'id': participant identifier (links to df_part)
        - 'age_cnt': contact age (numeric)
        - 'age_grp_cnt': contact age group (pd.Interval, categorical)
        - 'y': number of contacts (numeric, >= 0)
    df_age_dist : pd.DataFrame
        Population age distribution with columns:
        - 'age': age value (numeric)
        - 'P': population size at that age (numeric, > 0)
    age_bins : AgeGroupSpecs
        Age stratification bins defining age groups
    symmetric : bool, default False
        Apply reciprocity adjustment to ensure M[c,d]*P[c] = M[d,c]*P[d]
    adaptive_merge : bool, default False
        Automatically merge age groups with insufficient participants
    validate_for_bootstrap : bool, default False
        Validate data for bootstrap stability. If True, performs more aggressive
        age group merging to ensure bootstrap resampling will succeed. If False,
        only merges age groups necessary for contact intensity estimation.
        Set to True if you plan to use run_bootstrap().

    Attributes
    ----------
    Y : NDArray
        Contact count matrix, shape (B, B)
    N : NDArray
        Sample sizes per age group, shape (B,)
    P : NDArray
        Population sizes per age group, shape (B,)
    effective_age_bins : AgeGroupSpecs
        Age bins after any adaptive merging

    Methods
    -------
    compute_cint(recover_bins=False)
        Compute contact intensity matrix M
    compute_rate(recover_bins=False)
        Compute contact rate matrix ω
    run_bootstrap(n_boot=1000, random_state=None, progress=True)
        Estimate uncertainty via bootstrap

    Examples
    --------
    >>> # Create SocialMix instance
    >>> sm = SocialMix(df_part, df_cnt, df_age_dist, age_bins)
    >>>
    >>> # Get contact intensity matrix
    >>> M = sm.compute_cint()
    >>>
    >>> # Get contact rate matrix
    >>> omega = sm.compute_rate()
    >>>
    >>> # Bootstrap uncertainty
    >>> boot_results = sm.run_bootstrap(n_boot=1000, random_state=42)
    >>> M_std, omega_std = boot_results.std()
    >>> M_ci, omega_ci = boot_results.quantiles([0.025, 0.975])

    Notes
    -----
    Contact intensity M[c,d] represents the average number of contacts that
    individuals in age group c have with individuals in age group d.

    Contact rate ω[c,d] = M[c,d] / P[d] represents the per-capita rate at
    which individuals in age group c contact individuals in age group d.

    The reciprocity adjustment (symmetric=True) ensures that the total number
    of contacts from c to d equals the total from d to c: M[c,d]*P[c] = M[d,c]*P[d].
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_group_specs: AgeGroupSpecs,
        pop_data: Optional[PopulationData] = None,
        apply_reciprocity: bool = True,
        adaptive_merge: bool = False,
        validate_for_bootstrap: bool = False,
    ):
        # Store parameters
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_group_specs = age_group_specs
        self.pop_data = pop_data
        self.apply_reciprocity = apply_reciprocity
        self.adaptive_merge = adaptive_merge
        self.validate_for_bootstrap = validate_for_bootstrap

        # Stratification attributes (initialized in _preprocess)
        self.strat_vars_part: List[str] = []
        self.strat_vars_cnt: List[str] = []
        self.strat_vars_pop: List[str] = []
        self.strat_vars_shared: List[str] = []
        self.strat_vars_part_only: List[str] = []
        self.strat_vars_cnt_only: List[str] = []
        self.strat_mode: str = None
        self.strat_dims_part: Dict[str, int] = {}
        self.strat_dims_cnt: Dict[str, int] = {}
        self.K: int = 1  # Total number of strata

        # Initialize helper classes
        self.age_processor = AgeBinProcessor(age_group_specs)

        # Computed attributes (initialized in pipeline)
        self.effective_age_group_specs: Optional[AgeGroupSpecs] = None
        self._cint: Optional[NDArray] = None
        self._rate: Optional[NDArray] = None
        self._boot: Optional[BootstrapResults] = None

        # Data arrays (initialized in _load)
        self.Y: Optional[NDArray] = None  # Contact counts
        self.N: Optional[NDArray] = None  # Participant counts
        self.P: Optional[NDArray] = None  # Population sizes
        self.C: int = 0  # Number of participant age groups
        self.D: int = 0  # Number of contact age groups
        self.K_part: int = 1  # Number of participant strata
        self.K_cnt: int = 1  # Number of contact strata

        # Run processing pipeline
        self.fit()

    # ------------------------------------------------------------------
    # DeterministicContactModel interface
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """
        Prepare and load data for matrix estimation.

        Runs the full preprocessing and loading pipeline:
        1. Assigns age groups to participants / contacts
        2. Validates stratification, reciprocity requirements, and age bins
        3. Aggregates contact counts into Y, N, P arrays
        """
        self._preprocess()
        self._load()

    def predict(self) -> Dict[str, NDArray]:
        """
        Compute and return the contact intensity matrix.

        This is a convenience wrapper around :meth:`cint`.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping stratum labels to contact intensity matrices.
        """
        return self.cint()

    # ------------------------------------------------------------------
    # Internal preprocessing / loading
    # ------------------------------------------------------------------

    def _preprocess(self) -> None:
        """
        Extract and validate stratification information from data containers.

        This method:
        1. Extracts stratification variables from both containers
        2. Identifies shared, participant-only, and contact-only variables
        3. Computes stratification dimensions and expected number of strata
        4. Assigns age groups to raw ages if needed

        Notes
        -----
        Stratification variables do NOT need to match between participants and contacts.
        Four modes are supported:
        - No stratification: Both empty → K=1
        - Partial: Only participant vars → K=product of participant categories
        - Full: Same vars on both sides → K=product of squares
        - Mixed: Some overlap → K=complex product
        """
        # Assign age groups first (needed for validation)
        self._assign_age_groups()

        # Use validator to handle validation logic
        validator = SocialMixValidator(
            self.part_data,
            self.cnt_data,
            self.age_group_specs,
            self.pop_data,
            self.apply_reciprocity,
            self.adaptive_merge,
            self.validate_for_bootstrap,
        )

        # Run all validations and get updated components
        validated = validator.validate_all()

        # Update instance with validated components
        self.part_data = validated["part_data"]
        self.cnt_data = validated["cnt_data"]
        self.age_group_specs = validated["age_group_specs"]
        self.apply_reciprocity = validated["apply_reciprocity"]

        # Extract stratification variables from validated data
        self.strat_vars_part = self.part_data.get_strat_vars(suffix=False)
        self.strat_vars_cnt = self.cnt_data.get_strat_vars(suffix=False)
        self.strat_vars_pop = (
            self.pop_data.get_strat_vars(suffix=False)
            if self.pop_data is not None
            else []
        )

        # Identify shared and unique variables
        self.strat_vars_shared = sorted(
            list(set(self.strat_vars_part) & set(self.strat_vars_cnt))
        )
        self.strat_vars_part_only = sorted(
            list(set(self.strat_vars_part) - set(self.strat_vars_cnt))
        )
        self.strat_vars_cnt_only = sorted(
            list(set(self.strat_vars_cnt) - set(self.strat_vars_part))
        )

        # Calculate stratification dimensions for participant variables
        self.strat_dims_part = {}
        if self.strat_vars_part:
            for var in self.strat_vars_part:
                col_name = f"{var}_part"
                self.strat_dims_part[var] = self.part_data.data[col_name].nunique()

        # Calculate stratification dimensions for contact variables
        self.strat_dims_cnt = {}
        if self.strat_vars_cnt:
            for var in self.strat_vars_cnt:
                col_name = f"{var}_cnt"
                self.strat_dims_cnt[var] = self.cnt_data.data[col_name].nunique()

        # Calculate expected number of strata
        self.K = self._calculate_K()

    def _infer_strat_mode(self):
        """
        Infer the stratification mode based on the stratification variables.
        """
        if len(self.strat_vars_part) == 0 and len(self.strat_vars_cnt) == 0:
            self.strat_mode = "single"
        elif len(self.strat_vars_cnt) == 0:
            self.strat_mode = "partial"
        elif set(self.strat_vars_part) == set(self.strat_vars_cnt):
            self.strat_mode = "full"
        else:
            self.strat_mode = "mixed"

    def _calculate_K(self) -> int:
        """
        Calculate the number of strata based on stratification mode.

        Returns
        -------
        int
            Expected number of unique strata:
            - Case 1 (no stratification): 1
            - Case 2 (partial): product of participant categories
            - Case 3 (mixed): product of (part-only x shared^2 x cnt-only)
            - Case 4 (full): product of squares of categories
        """
        if self._infer_strat_mode() == "single":
            # Case 1: No stratification
            return 1

        K = 1

        # Participant-only variables (partial mode)
        for var in self.strat_vars_part_only:
            K *= self.strat_dims_part[var]

        # Contact-only variables
        for var in self.strat_vars_cnt_only:
            K *= self.strat_dims_cnt[var]

        # Shared variables (full mode for these variables)
        for var in self.strat_vars_shared:
            # For shared vars, we get all combinations: n_categories × n_categories
            K *= self.strat_dims_part[var] ** 2

        return int(K)

    def _assign_age_groups(self) -> None:
        """
        Assign age groups to participants and contacts if not already provided.

        Uses the age_bins provided during initialization to categorize
        raw ages into age groups.
        """
        bin_edges = self.age_group_specs.left + [self.age_group_specs.right[-1] + 1]

        # Interval labels: right bound is exclusive (right[i] + 1)
        intervals = [
            pd.Interval(left=l, right=r + 1, closed="left")
            for l, r in zip(self.age_group_specs.left, self.age_group_specs.right)
        ]

        # Assign age groups to participants if not present
        if "age_grp_part" not in self.part_data.data.columns:
            if "age_part" in self.part_data.data.columns:
                # Create age groups from raw ages
                ages = self.part_data.data["age_part"]
                age_grps = pd.cut(
                    ages,
                    bins=bin_edges,
                    right=False,
                    labels=intervals,
                )
                self.part_data.data["age_grp_part"] = age_grps
            else:
                raise ValueError(
                    "ParticipantData must have either 'age_part' or 'age_grp_part' column."
                )

        # Assign age groups to contacts if not present
        if "age_grp_cnt" not in self.cnt_data.data.columns:
            if "age_cnt" in self.cnt_data.data.columns:
                # Create age groups from raw ages
                ages = self.cnt_data.data["age_cnt"]
                age_grps = pd.cut(
                    ages,
                    bins=bin_edges,
                    right=False,
                    labels=intervals,
                )
                self.cnt_data.data["age_grp_cnt"] = age_grps
            else:
                raise ValueError(
                    "ContactData must have either 'age_cnt' or 'age_grp_cnt' column."
                )

    def _load(self) -> None:
        """
        Prepare contact count matrices for estimation.

        Delegates to SocialMixDataProcessor for loading stratified data arrays.

        Computes stratified contact count arrays based on stratification mode:

        - Single mode: Y (C, D), N (C,), P (D,)
        - Partial mode: Y (K_part, C, D), N (K_part, C), P (D,)
        - Full/Mixed mode: Y (K_part, K_cnt, C, D), N (K_part, C), P (K_cnt, D)

        Where:
            - C: number of participant age groups
            - D: number of contact age groups
            - K_part: number of participant strata
            - K_cnt: number of contact strata

        Sets
        ----
        self.Y : NDArray
            Contact count matrix/tensor
        self.N : NDArray
            Participant count matrix/vector
        self.P : NDArray or None
            Population size matrix/vector (if pop_data available)
        self.C : int
            Number of participant age groups
        self.D : int
            Number of contact age groups
        self.K_part : int
            Number of participant strata
        self.K_cnt : int
            Number of contact strata
        """
        # Delegate to data processor
        loader = SocialMixDataLoader(self)
        loader.load_data()

    def _create_stratum_labels(self) -> List[str]:
        """
        Create stratum labels following 'source->target' naming convention.

        Returns
        -------
        List[str]
            List of stratum labels in the order corresponding to array indices.
            Examples: ["All->All"], ["M->All", "F->All"], ["M->M", "M->F", "F->M", "F->F"]
        """
        if self.K_part == 1 and self.K_cnt == 1:
            return ["All->All"]

        # Get categorical values for participant and contact strata
        part_categories = []
        if self.strat_vars_part:
            # Build list of category combinations for participant side
            for var in self.strat_vars_part:
                col_name = f"{var}_part"
                cats = self.part_data.data[col_name].cat.categories.tolist()
                part_categories.append(cats)
        else:
            part_categories = [["All"]]

        cnt_categories = []
        if self.strat_vars_cnt:
            # Build list of category combinations for contact side
            for var in self.strat_vars_cnt:
                col_name = f"{var}_cnt"
                cats = self.cnt_data.data[col_name].cat.categories.tolist()
                cnt_categories.append(cats)
        else:
            cnt_categories = [["All"]]

        # Generate all combinations
        import itertools

        part_combos = list(itertools.product(*part_categories))
        cnt_combos = list(itertools.product(*cnt_categories))

        # Create labels in format "part->cnt"
        labels = []
        for part in part_combos:
            for cnt in cnt_combos:
                # Join multiple variables with underscore
                part_str = "_".join(str(p) for p in part)
                cnt_str = "_".join(str(c) for c in cnt)
                labels.append(f"{part_str}->{cnt_str}")

        return labels

    def _apply_reciprocity(
        self, cint_dict: Dict[str, NDArray[np.float64]], labels: List[str]
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Apply reciprocity adjustment to contact intensity matrices.

        For single/full stratification modes, this ensures that contact patterns
        are reciprocal according to population-weighted constraints.

        Uses vectorized operations for efficiency.

        Parameters
        ----------
        cint_dict : Dict[str, NDArray]
            Raw contact intensity matrices before reciprocity adjustment
        labels : List[str]
            Stratum labels in order

        Returns
        -------
        Dict[str, NDArray]
            Reciprocity-adjusted contact intensity matrices

        Notes
        -----
        For within-stratum contacts (s=s):
            m^{s,s†}_{c,d} = (m^{s,s}_{c,d} * P^s_c + m^{s,s}_{d,c} * P^s_d) / (2 * P^s_c)

        For between-stratum contacts (s≠t):
            m^{s,t†}_{c,d} = (m^{s,t}_{c,d} + m^{t,s}_{d,c} * P^t_d / P^s_c) / 2
        """
        result = {}

        if self.strat_mode == "single":
            # Single stratum: apply within-stratum formula (vectorized)
            # m†_{c,d} = (m_{c,d} * P_c + m_{d,c} * P_d) / (2 * P_c)
            m = cint_dict["All->All"]
            P = self.P  # Shape (D,)

            # Vectorized computation:
            # m * P[:, None] broadcasts P along columns
            # m.T * P[None, :] transposes m and broadcasts P along rows
            # Division by (2 * P[:, None]) normalizes by source population
            m_adj = (m * P[:, np.newaxis] + m.T * P[np.newaxis, :]) / (
                2 * P[:, np.newaxis]
            )

            result["All->All"] = m_adj

        elif self.strat_mode == "full":
            # Full stratification: apply both within and between-stratum formulas
            # P has shape (K_cnt, D)

            # Parse labels to identify strata
            # Labels are like "M->M", "M->F", "F->M", "F->F"
            for idx, label in enumerate(labels):
                parts = label.split("->")
                s_str = parts[0]  # Source stratum
                t_str = parts[1]  # Target stratum

                # Get matrix indices
                k_s = idx // self.K_cnt  # Participant stratum index
                k_t = idx % self.K_cnt  # Contact stratum index

                m_st = cint_dict[label]
                P_s = self.P[k_s, :]  # Population of source stratum
                P_t = self.P[k_t, :]  # Population of target stratum

                if s_str == t_str:
                    # Within-stratum (vectorized)
                    # m^{s,s†}_{c,d} = (m^{s,s}_{c,d} * P^s_c + m^{s,s}_{d,c} * P^s_d) / (2 * P^s_c)
                    m_adj = (
                        m_st * P_s[:, np.newaxis] + m_st.T * P_s[np.newaxis, :]
                    ) / (2 * P_s[:, np.newaxis])
                    result[label] = m_adj
                else:
                    # Between-stratum (vectorized)
                    # m^{s,t†}_{c,d} = (m^{s,t}_{c,d} + m^{t,s}_{d,c} * P^t_d / P^s_c) / 2
                    reverse_label = f"{t_str}->{s_str}"
                    m_ts = cint_dict[reverse_label]

                    # Vectorized computation with safe division
                    # Use np.where to handle potential division by zero
                    # If P_s[c] == 0, keep original m_st[c, d]
                    P_s_safe = np.where(P_s > 0, P_s, 1)  # Avoid division by zero
                    m_adj = (
                        m_st + m_ts.T * P_t[np.newaxis, :] / P_s_safe[:, np.newaxis]
                    ) / 2

                    # Where P_s was zero, restore original values
                    m_adj = np.where(P_s[:, np.newaxis] > 0, m_adj, m_st)

                    result[label] = m_adj

        else:
            # Partial or mixed: no reciprocity adjustment
            result = cint_dict

        return result

    def cint(self) -> Dict[str, NDArray[np.float64]]:
        """
        Compute contact intensity matrix.

        M[c,d] represents the average number of contacts that individuals
        in age group c have with individuals in age group d.

        If apply_reciprocity=True, applies post-hoc reciprocity adjustment
        to ensure contact symmetry (for single/full stratification modes).

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping stratum labels to contact intensity matrices.
            Keys follow "source->target" format:
            - "All->All" for unstratified data
            - "M->All", "F->All" for participant-only stratification
            - "M->M", "M->F", "F->M", "F->F" for full stratification
            Each matrix has shape (C, D) where C is participant age groups
            and D is contact age groups.
        """
        if self._cint is None:
            # Get stratum labels
            labels = self._create_stratum_labels()

            # Compute raw intensity matrices based on data structure
            result = {}

            if self.K_part == 1 and self.K_cnt == 1:
                # Single stratum: Y is (C, D), N is (C,)
                result["All->All"] = self.Y / self.N[:, np.newaxis]

            elif self.K_part > 1 and self.K_cnt == 1:
                # Partial stratification: Y is (K_part, C, D), N is (K_part, C)
                for k in range(self.K_part):
                    result[labels[k]] = self.Y[k] / self.N[k, :, np.newaxis]

            else:
                # Full/mixed stratification: Y is (K_part, K_cnt, C, D), N is (K_part, C)
                idx = 0
                for k_part in range(self.K_part):
                    for k_cnt in range(self.K_cnt):
                        result[labels[idx]] = (
                            self.Y[k_part, k_cnt] / self.N[k_part, :, np.newaxis]
                        )
                        idx += 1

            # Apply reciprocity adjustment if requested
            if self.apply_reciprocity and self.P is not None:
                result = self._apply_reciprocity(result, labels)

            self._cint = result

        return self._cint

    def rate(self) -> Dict[str, NDArray[np.float64]]:
        """
        Compute contact rate matrix.

        ω[c,d] represents the per-capita rate at which individuals in
        age group c contact individuals in age group d.

        The rate is computed from the contact intensity matrix by dividing
        by the population size of the target age group:
            ω[c,d] = M[c,d] / P[d]

        Returns
        -------
        Dict[str, NDArray[np.float64]]
            Dictionary of contact rate matrices, one per stratum.
            Keys follow the format "source->target" (e.g., "All->All", "M->F").
            Each matrix has shape (C, D) where C is the number of participant
            age groups and D is the number of contact age groups.

        Raises
        ------
        ValueError
            If population data was not provided during initialization.

        Examples
        --------
        >>> # Single stratum (no stratification)
        >>> omega_dict = sm.rate()
        >>> omega = omega_dict["All->All"]  # shape (C, D)
        >>>
        >>> # Partial stratification (participant only)
        >>> omega_dict = sm.rate()
        >>> omega_M = omega_dict["M->All"]  # Males' contact rates
        >>> omega_F = omega_dict["F->All"]  # Females' contact rates
        >>>
        >>> # Full stratification (both sides)
        >>> omega_dict = sm.rate()
        >>> omega_MF = omega_dict["M->F"]  # Males contacting females
        """
        if self._rate is None:
            # Check that population data is available
            if self.P is None:
                raise ValueError(
                    "Cannot compute contact rates without population data. "
                    "Please provide 'pop_data' when initializing SocialMix."
                )

            # Get contact intensity matrices
            cint_dict = self.cint()

            # Compute rate for each stratum: rate = cint / P
            result = {}
            for key, M in cint_dict.items():
                # M has shape (C, D), P has shape (D,) for single/partial
                # or P has shape (K_cnt, D) for full stratification
                if self.K_cnt == 1:
                    # Single or partial stratification: divide by population vector
                    result[key] = M / self.P[np.newaxis, :]
                else:
                    # Full stratification: need to extract correct P slice
                    # Parse stratum key to get contact stratum index
                    labels = self._create_stratum_labels()
                    stratum_idx = labels.index(key)
                    # For full stratification with K_part participant strata and K_cnt contact strata:
                    # stratum_idx = k_part * K_cnt + k_cnt
                    k_cnt = stratum_idx % self.K_cnt
                    result[key] = M / self.P[k_cnt, :][np.newaxis, :]

            self._rate = result

        return self._rate

    def run_inference_bootstrap(
        self,
        n_boot: int = 1000,
        random_state: Optional[int] = None,
        progress: bool = True,
        min_success_rate: float = 0.5,
    ) -> None:
        """
        Estimate uncertainty via bootstrap resampling.

        This method runs bootstrap resampling to quantify uncertainty in contact
        intensity and rate estimates. Results are stored in `self._boot` for later
        access by analysis tools.

        Parameters
        ----------
        n_boot : int, default=1000
            Number of bootstrap resamples to generate
        random_state : int, optional
            Random seed for reproducibility. If None, results will vary between runs.
        progress : bool, default=True
            Whether to display a progress bar during bootstrap iterations
        min_success_rate : float, default=0.5
            Minimum fraction of successful bootstrap iterations required.
            If the success rate falls below this threshold, a ValueError is raised.

        Returns
        -------
        BootstrapResults
            Container with bootstrap samples and methods for computing statistics:
            - `mean(statistic='cint')`: Mean contact intensity across bootstrap samples
            - `std(statistic='rate')`: Standard deviation of contact rates
            - `quantiles(q=[0.025, 0.975])`: Confidence intervals

        Raises
        ------
        ValueError
            If the success rate is below `min_success_rate`, indicating insufficient
            sample sizes in some age groups or strata.

        Examples
        --------
        >>> model = SocialMix(part_data, cnt_data, age_bins, pop_data)
        >>> results = model.run_inference_bootstrap(n_boot=1000, random_state=42)
        >>> cint_mean = results.mean(statistic='cint')
        >>> cint_std = results.std(statistic='cint')
        >>> ci_95 = results.quantiles(q=[0.025, 0.975], statistic='cint')

        Notes
        -----
        - Bootstrap results are automatically stored in `self._boot`
        - If `apply_reciprocity=True` was set during initialization, reciprocity
          adjustment is applied within each bootstrap iteration
        - For stratified models, results include separate matrices for each stratum
        """
        # Create bootstrap estimator
        bootstrap = SocialMixBootstrap(
            part_data=self.part_data,
            cnt_data=self.cnt_data,
            age_group_specs=self.age_group_specs,
            pop_data=self.pop_data,
            apply_reciprocity=self.apply_reciprocity,
            n_boot=n_boot,
            random_state=random_state,
        )

        # Run bootstrap and store results
        self._boot = bootstrap.run(
            progress=progress,
            min_success_rate=min_success_rate,
        )
