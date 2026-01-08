import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from jax.random import PRNGKey
from numpy.typing import NDArray

from ...dataloader.containers import PopulationData
from ...models import Prem
from ...models._socialmix_age_processing import AgeBinProcessor
from ...utils import AgeBins, depixilate, pixilate


def validate_alpha(alpha: float) -> None:
    """Validate alpha parameter."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def get_probs_from_alpha(alpha: float) -> Tuple[float, float, float]:
    """Convert alpha to (lower, median, upper) probabilities."""
    return (alpha / 2, 0.5, 1 - alpha / 2)


def compute_quantiles(
    samples: NDArray, probs: Tuple[float, ...], axis: int = 0
) -> NDArray:
    """
    Compute quantiles with validation and R-compatible method.

    Parameters
    ----------
    samples : NDArray
        Input data, shape (n_samples, ...)
    probs : tuple of float
        Quantile probabilities in [0, 1]
    axis : int, default=0
        Axis along which to compute quantiles

    Returns
    -------
    quantiles : NDArray
        Shape (len(probs), ...) with quantiles along axis 0
    """
    # Validate probabilities
    if not all(0 <= p <= 1 for p in probs):
        raise ValueError(f"All probabilities must be in [0, 1], got {probs}")

    # Sort probabilities to ensure correct ordering in output
    if list(probs) != sorted(probs):
        warnings.warn(
            "Probabilities are not sorted. Output will follow input order.",
            UserWarning,
        )

    result = np.quantile(samples, probs, axis=axis)

    return result


class ModelSummariserPrem:
    """
    Statistical summariser for Prem model inference results.

    Computes quantiles and credible intervals for contact matrices from MCMC or SVI
    posterior samples, with proper handling of reciprocity and depixilation.

    Parameters
    ----------
    prem : Prem
        Fitted Prem model with MCMC or SVI results.
    pop_data : PopulationData, optional
        Population data container with fine-grained (1-year) age distribution.
        Required for reciprocity adjustment and depixilation operations.
    num_samples : int, default=3000
        Number of posterior samples to draw if using SVI.

    Attributes
    ----------
    prem : Prem
        Reference to the Prem model
    age_bins : AgeBins
        Age bins used in the model
    pop_data : PopulationData or None
        Population data container
    age_dist : NDArray, optional
        Fine-grained (1-year) population distribution extracted from pop_data
    age_grp_dist : NDArray, optional
        Coarse-grained (age group) population distribution
    post_samples : Dict[str, NDArray]
        Posterior samples from MCMC or SVI
    post_cint_samples : NDArray or Dict[str, NDArray]
        Posterior contact intensity samples (exponential of log_cint)

    Examples
    --------
    >>> # Basic usage
    >>> prem = Prem(part_data, cnt_data, age_bins)
    >>> prem.run_inference_mcmc(rng_key, num_samples=1000)
    >>>
    >>> pop_data = PopulationData(df_age_dist, age_col='age', size_col='P')
    >>> summariser = ModelSummariserPrem(prem, pop_data=pop_data)
    >>>
    >>> # Get 95% credible intervals for contact intensity
    >>> summary = summariser.summarise_cint(alpha=0.05)
    >>> lower, median, upper = summary[0], summary[1], summary[2]
    >>>
    >>> # Get reciprocity-adjusted and depixilated results
    >>> summary_full = summariser.summarise_cint(
    ...     alpha=0.05,
    ...     apply_reciprocity=True,
    ...     return_depixilated=True
    ... )
    """

    def __init__(
        self,
        prem: Prem,
        pop_data: Optional[PopulationData] = None,
        num_samples: int = 3000,
    ) -> None:
        """
        Initialize summariser with a Prem model.

        Parameters
        ----------
        prem : Prem
            Prem model with completed MCMC or SVI inference.
        pop_data : PopulationData, optional
            Population data container for reciprocity and depixilation operations.
            Must be provided for depixilation (fine-grained age distribution) and
            reciprocity adjustment (population weighting).
        num_samples : int, default=3000
            Number of posterior samples to draw if using SVI.

        Raises
        ------
        ValueError
            If neither MCMC nor SVI has been run on the model.
            If model has not been properly initialized.
        """
        # Validate that either MCMC or SVI has been run
        has_mcmc = prem._mcmc_result is not None
        has_svi = prem._svi_result is not None
        if not (has_mcmc or has_svi):
            raise ValueError(
                "Either MCMC or SVI must have been run on the model. "
                "Call prem.run_inference_mcmc() or prem.run_inference_svi() first."
            )

        # Validate model data is loaded
        if prem.data is None:
            raise ValueError("Prem model data not initialized")

        # Store reference to model
        self.prem = prem

        # Reference key attributes
        self.age_bins = prem.age_bins
        self.pop_data = pop_data
        self.num_samples = num_samples

        # Detect stratification
        self.K = prem.K if hasattr(prem, "K") else 1
        self.strat_mode = self._detect_stratification_mode()
        self.strata_labels = self._create_stratum_labels()

        # Initialize helper classes
        self.age_processor = AgeBinProcessor(self.age_bins)

        # Computed attributes (initialized in pipeline)
        self.age_dist: Optional[NDArray] = None
        self.age_grp_dist: Optional[NDArray] = None
        self.post_samples: Optional[Dict[str, NDArray]] = None
        self.post_cint_samples: Optional[NDArray] = None

        # Simple cache: {cache_key: result_dict}
        self._cache: Dict[str, Dict[str, NDArray]] = {}

        # Run processing pipeline
        self._validate()
        self._load()

        # Derive age_grp_dist from age_dist if not provided
        if (
            self.age_grp_dist is None
            and self.age_bins is not None
            and self.age_dist is not None
        ):
            self.age_grp_dist = self._compute_age_grp_dist()

    def _detect_stratification_mode(self) -> str:
        """
        Detect stratification mode from model attributes.

        Returns
        -------
        str
            One of "none", "partial", "full", or "mixed"
        """
        if self.K == 1:
            return "none"

        # For stratified models, check if we have stratification info
        # This is a simplified version - actual implementation would need
        # to check prem.strat_vars_part and prem.strat_vars_cnt attributes
        if hasattr(self.prem, "strat_vars_part") and hasattr(
            self.prem, "strat_vars_cnt"
        ):
            strat_part = self.prem.strat_vars_part or []
            strat_cnt = self.prem.strat_vars_cnt or []

            if len(strat_part) > 0 and len(strat_cnt) == 0:
                return "partial"
            elif len(strat_part) == 0 and len(strat_cnt) > 0:
                return "partial"
            elif len(strat_part) > 0 and len(strat_cnt) > 0:
                # Check if same variables
                if set(strat_part) == set(strat_cnt):
                    return "full"
                else:
                    return "mixed"

        # Default to partial if we can't determine
        return "partial"

    def _create_stratum_labels(self) -> list:
        """
        Create ordered stratum labels from model data.

        Returns
        -------
        list of str
            Ordered list of stratum labels (e.g., ["M->M", "M->F", "F->M", "F->F"])
            Order matches the factorized encoding in prem.six

        Notes
        -----
        - For K=1: Returns ["All->All"]
        - For K>1: Extracts from prem.data["stratum"] column
        - Format: "source->target" where source = participant stratum, target = contact stratum
        - Maintains consistent ordering with prem.six encoding
        """
        if self.K == 1:
            return ["All->All"]

        # For K>1, extract from data if available
        if hasattr(self.prem, "data") and "stratum" in self.prem.data.columns:
            # Get categories in alphabetical order (matches prem.six encoding with categorical)
            if hasattr(self.prem.data["stratum"], "cat"):
                return list(self.prem.data["stratum"].cat.categories)
            else:
                # If not categorical, sort unique values for deterministic ordering
                return sorted(self.prem.data["stratum"].unique())

        # Fallback: generate generic labels
        return [f"Stratum_{i}" for i in range(self.K)]

    def _validate(self) -> None:
        """Validate population data and stratification consistency."""
        # If no population data provided, nothing to validate
        if self.pop_data is None:
            return

        # If model has stratification, validate against participant data
        if self.K > 1 and self.pop_data.strat_vars:
            self._validate_population_stratification()

    def _validate_population_stratification(self) -> None:
        """
        Validate that population stratification variables match participant data.

        Ensures:
        1. Same stratification variables exist in both
        2. Same categories for each variable
        3. Categories are encoded in the same order (using participant order as reference)

        If categories match but order differs, population data is automatically reordered.

        Raises
        ------
        ValueError
            If stratification variables don't match or categories differ.
        """
        # Get participant stratification variables from the Prem model
        part_strat_vars = self.prem.strat_vars_part
        pop_strat_vars = self.pop_data.strat_vars

        # Check if variables match
        if set(part_strat_vars) != set(pop_strat_vars):
            raise ValueError(
                f"Population stratification variables {pop_strat_vars} don't match "
                f"participant stratification variables {part_strat_vars}. "
                f"For stratified models, PopulationData must have the same stratification variables."
            )

        # For each shared variable, validate categories and ordering
        for var in part_strat_vars:
            # Get column names
            col_part = f"{var}_part"
            col_pop = var  # PopulationData uses base variable names

            # Get categories from both sides
            part_col = self.prem.part_data.data[col_part]
            pop_col = self.pop_data.data[col_pop]

            # Convert to categorical if not already
            if not hasattr(part_col, "cat"):
                part_col = part_col.astype("category")
            if not hasattr(pop_col, "cat"):
                pop_col = pop_col.astype("category")
                self.pop_data.data[col_pop] = pop_col

            # Get categories in encoding order
            part_cats = list(part_col.cat.categories)
            pop_cats = list(pop_col.cat.categories)

            # Check if categories match as sets
            part_set = set(part_cats)
            pop_set = set(pop_cats)

            if part_set != pop_set:
                # Different categories - this is an error
                only_part = part_set - pop_set
                only_pop = pop_set - part_set
                raise ValueError(
                    f"Population stratification variable '{var}' has different categories:\n"
                    f"  Participant side: {part_cats}\n"
                    f"  Population side: {pop_cats}\n"
                    f"  Only in participants: {sorted(only_part) if only_part else 'None'}\n"
                    f"  Only in population: {sorted(only_pop) if only_pop else 'None'}\n"
                    f"For stratified models, population must have the same categories as participants."
                )

            # Same categories but possibly different order
            # Use participant ordering as reference and reorder population data
            if part_cats != pop_cats:
                self.pop_data.data[col_pop] = self.pop_data.data[
                    col_pop
                ].cat.reorder_categories(part_cats, ordered=False)

    def _load(self) -> None:
        """Load age distributions and posterior samples."""
        # Extract age distribution from PopulationData if provided
        if self.pop_data is not None:
            # Get fine-grained age distribution (unstratified)
            age_dist_df = self.pop_data.get_age_distribution(by_group=False)
            self.age_dist = age_dist_df.values
        else:
            warnings.warn(
                "PopulationData not provided. "
                "Reciprocity adjustment and depixilation will not be possible.",
                UserWarning,
            )

        # Load posterior samples
        if self.prem._mcmc_result is not None:
            self.post_samples = self.prem._mcmc_result.get_samples()
        elif self.prem._svi_result is not None:
            # For SVI, use posterior_predictive to get deterministic variables (log_cint)
            # This samples from guide and runs model forward to compute log_cint
            self.post_samples = self.prem.posterior_predictive_svi(
                PRNGKey(0), num_samples=self.num_samples
            )

        # Compute contact intensity samples
        self.post_cint_samples = self._compute_contact_intensities()

    def _compute_age_grp_dist(self) -> NDArray:
        """Compute age group distribution from fine-grained age distribution."""
        age_grp_dist = []
        age_edges = self.age_bins.left + [self.age_bins.max + 1]

        for i in range(len(age_edges) - 1):
            start_age = int(age_edges[i])
            end_age = int(age_edges[i + 1])
            age_grp_dist.append(self.age_dist[start_age:end_age].sum())

        return np.array(age_grp_dist)

    def _compute_contact_intensities(self):
        """
        Compute contact intensity samples from posterior.

        Returns
        -------
        NDArray or Dict[str, NDArray]
            For K=1: NDArray of shape (n_samples, D, C)
            For K>1: Dict mapping stratum labels to NDArray of shape (n_samples, D, C)
        """
        if self.post_samples is None:
            raise ValueError("Posterior samples not loaded")

        # log_cint should now always be available as a deterministic variable
        if "log_cint" not in self.post_samples:
            available_fields = list(self.post_samples.keys())
            raise ValueError(
                f"Posterior samples must contain 'log_cint' field. "
                f"Available fields: {available_fields}."
            )

        log_cint = self.post_samples["log_cint"]

        if self.K == 1:
            # Unstratified: log_cint shape is (n_samples, D, C)
            return np.exp(log_cint)

        # Stratified: log_cint shape is (n_samples, K, D, C)
        # Split by stratum and return dict
        cint_samples = {}
        _, K = log_cint.shape[0], log_cint.shape[1]

        for k, label in enumerate(self.strata_labels[:K]):
            cint_samples[label] = np.exp(log_cint[:, k, :, :])

        return cint_samples

    @staticmethod
    def _aggregate_population_to_bins(
        pop_data: PopulationData, age_bins, strat_mode: str = "none"
    ):
        """
        Aggregate fine-grained population to match model age bins.

        Parameters
        ----------
        pop_data : PopulationData
            Fine-grained population data
        age_bins : AgeBins
            Age bins from the model
        strat_mode : str
            Stratification mode

        Returns
        -------
        PopulationData
            Aggregated population matching age bins
        """
        from cntmosaic.utils import AgeBins

        # Get the population DataFrame
        df_pop = pop_data.df_pop.copy()

        # Create age group bins
        age_edges = list(age_bins.left) + [age_bins.max + 1]
        age_labels = age_bins.left

        # Bin the ages
        df_pop["age_grp"] = pd.cut(
            df_pop["age"],
            bins=age_edges,
            labels=age_labels,
            right=False,
            include_lowest=True,
        )

        # Aggregate by age group (and stratification variables if present)
        group_cols = ["age_grp"]
        if pop_data.strat_var_cols:
            group_cols.extend(pop_data.strat_var_cols)

        df_pop_agg = df_pop.groupby(group_cols, observed=True)["P"].sum().reset_index()
        df_pop_agg = df_pop_agg.rename(columns={"age_grp": "age"})

        # Convert age back to int (it's categorical from pd.cut)
        df_pop_agg["age"] = df_pop_agg["age"].astype(int)

        # Create new PopulationData with aggregated data
        pop_data_agg = PopulationData(
            df_pop_agg,
            age_col="age",
            size_col="P",
            strat_var_cols=pop_data.strat_var_cols,
        )

        return pop_data_agg

    @staticmethod
    def apply_reciprocity(
        cint_samples,
        pop_data: Optional[PopulationData],
        strat_mode: str = "none",
        strata_labels: Optional[list] = None,
        age_bins=None,
    ):
        """
        Apply reciprocity adjustment to contact intensity matrices.

        For unstratified (K=1) and full stratification modes, applies reciprocity
        to ensure balanced contact flows weighted by population. For partial and
        mixed stratification, reciprocity is not applicable and returns unchanged.

        Parameters
        ----------
        cint_samples : NDArray or Dict[str, NDArray]
            Contact intensity samples.
            - For K=1: NDArray of shape (n_samples, B, B)
            - For K>1: Dict mapping stratum labels to NDArray of shape (n_samples, B, B)
        pop_data : PopulationData, optional
            Population data container. Required for reciprocity adjustment.
            - For K=1: Should contain unstratified population
            - For full stratification: Should contain stratified population with matching strat_var_cols
            - Can have finer age resolution than contact matrices (will auto-aggregate using age_bins)
        strat_mode : str, default="none"
            Stratification mode: "none", "partial", "full", or "mixed"
        strata_labels : list of str, optional
            Stratum labels for full stratification mode. Required if strat_mode="full".
        age_bins : AgeBins, optional
            Age bin definition for automatic population aggregation. If pop_data has finer
            age resolution than contact matrices, population will be automatically aggregated
            to match. Required if population resolution doesn't match contact matrix resolution.

        Returns
        -------
        NDArray or Dict[str, NDArray]
            Adjusted contact intensity samples, same structure as input.

        Raises
        ------
        ValueError
            If reciprocity requested for partial/mixed modes.
            If pop_data not provided when required.
            If population dimensions don't match contact matrix dimensions and age_bins not provided.

        Notes
        -----
        Reciprocity formulas:
        - Unstratified: M ← 0.5 * (M + P^{-1} @ M.T @ P)
        - Within-stratum (s=s): M^{s,s} ← 0.5 * (M^{s,s} + P_s^{-1} @ (M^{s,s}).T @ P_s)
        - Between-stratum (s≠t): M^{s,t} ← 0.5 * (M^{s,t} + (M^{t,s}).T @ P_s^{-1} @ P_t)

        Automatic Aggregation:
        If pop_data contains fine-grained age distribution (e.g., 1-year ages) but contact
        matrices use coarser age groups (e.g., 5-year bins), population will be automatically
        aggregated using the provided age_bins. This is especially useful when using detailed
        census data with age-grouped contact models.
        """
        # Check if reciprocity applies to this stratification mode
        if strat_mode in ["partial", "mixed"]:
            warnings.warn(
                f"Reciprocity not applied for {strat_mode} stratification mode. "
                "Contact rates M^{{s,t}} and M^{{t,s}} have no inherent symmetry when "
                "only one side is stratified or different variables are used.",
                UserWarning,
            )
            return cint_samples

        if pop_data is None:
            raise ValueError(
                "PopulationData required for reciprocity adjustment. "
                f"Provide pop_data parameter for {strat_mode} stratification mode."
            )

        # Unstratified (K=1)
        if strat_mode == "none":
            if isinstance(cint_samples, dict):
                raise ValueError(
                    "For unstratified mode, cint_samples should be NDArray, not Dict"
                )

            # Get population distribution
            age_grp_dist = pop_data.get_age_distribution(by_group=False).values

            # Check if aggregation is needed
            n_age_groups = cint_samples.shape[1]
            if len(age_grp_dist) != n_age_groups:
                # Auto-aggregate population to match model age bins
                if age_bins is None:
                    raise ValueError(
                        f"Population size ({len(age_grp_dist)}) doesn't match "
                        f"number of age groups ({n_age_groups}). "
                        "Provide age_bins parameter for automatic aggregation."
                    )
                pop_data = ModelSummariserPrem._aggregate_population_to_bins(
                    pop_data, age_bins, strat_mode
                )
                age_grp_dist = pop_data.get_age_distribution(by_group=False).values

                # Validate after aggregation
                if len(age_grp_dist) != n_age_groups:
                    raise ValueError(
                        f"After aggregation, population size ({len(age_grp_dist)}) still doesn't match "
                        f"number of age groups ({n_age_groups})"
                    )

            if np.any(age_grp_dist <= 0):
                raise ValueError("Population must contain positive values")

            # Apply reciprocity: M ← 0.5 * (M + P^{-1} @ M.T @ P)
            M = cint_samples
            P = np.diag(age_grp_dist)[np.newaxis, ...]
            P_inv = np.diag(1 / age_grp_dist)[np.newaxis, ...]

            return 0.5 * (M + P_inv @ np.transpose(M, (0, 2, 1)) @ P)

        # Full stratification
        if strat_mode == "full":
            if not isinstance(cint_samples, dict):
                raise ValueError("For full stratification, cint_samples should be Dict")

            if strata_labels is None:
                raise ValueError(
                    "strata_labels required for full stratification reciprocity"
                )

            # Extract stratified populations
            pop_by_group = pop_data.get_age_distribution(by_group=True)

            # Determine stratification variable name
            if not pop_data.strat_var_cols:
                raise ValueError(
                    "PopulationData must have stratification variables for full mode"
                )

            if pop_data.n_strat_vars > 1:
                # Build composite strata string
                pop_by_group["strata"] = pop_by_group[pop_data.strat_var_cols[0]].astype(
                    str
                )
                for col in pop_data.strat_var_cols[1:]:
                    pop_by_group["strata"] = (
                        pop_by_group["strata"] + "_" + pop_by_group[col].astype(str)
                    )

                # Set as index and keep only age and population columns
                composite_col = "strata"
                pop_by_group_indexed = pop_by_group[["strata", "age", "P"]].set_index(
                    ["strata", "age"]
                )["P"]
            else:
                # Set as index and keep only age and population columns
                composite_col = pop_data.strat_var_cols[0]
                pop_by_group_indexed = pop_by_group[[composite_col, "age", "P"]].set_index(
                    [composite_col, "age"]
                )["P"]

            # Extract unique strata from labels
            strata = set()
            for label in strata_labels:
                if "->" in label:
                    source, target = label.split("->")
                    strata.add(source)
                    strata.add(target)
            strata = sorted(list(strata))

            # Check if aggregation is needed (check first matrix to determine expected size)
            first_key = next(iter(cint_samples.keys()))
            n_age_groups = cint_samples[first_key].shape[1]

            # Get unique ages in population from the MultiIndex
            n_pop_ages = len(pop_by_group_indexed.index.get_level_values(1).unique())

            if n_pop_ages != n_age_groups:
                if age_bins is None:
                    raise ValueError(
                        f"Population has {n_pop_ages} age groups but model has {n_age_groups}. "
                        "Provide age_bins parameter for automatic aggregation."
                    )
                pop_data = ModelSummariserPrem._aggregate_population_to_bins(
                    pop_data, age_bins, strat_mode
                )
                pop_by_group = pop_data.get_age_distribution(by_group=True)
                
                # Re-index after aggregation
                if pop_data.n_strat_vars > 1:
                    pop_by_group["strata"] = pop_by_group[pop_data.strat_var_cols[0]].astype(str)
                    for col in pop_data.strat_var_cols[1:]:
                        pop_by_group["strata"] = pop_by_group["strata"] + "_" + pop_by_group[col].astype(str)
                    pop_by_group_indexed = pop_by_group[["strata", "age", "P"]].set_index(["strata", "age"])["P"]
                else:
                    pop_by_group_indexed = pop_by_group[[composite_col, "age", "P"]].set_index([composite_col, "age"])["P"]

            # Build population dict
            pop_dict = {}
            for stratum in strata:
                if stratum in pop_by_group_indexed.index.get_level_values(0):
                    pop_dict[stratum] = pop_by_group_indexed.xs(stratum, level=0).values
                else:
                    # Try composite stratification
                    # For now, this is simplified - full implementation would parse composite labels
                    warnings.warn(
                        f"Could not extract population for stratum '{stratum}'. "
                        "Composite stratification handling is simplified.",
                        UserWarning,
                    )
                    continue

            adjusted = {}

            # Within-stratum pairs
            for s in strata:
                key = f"{s}->{s}"
                if key in cint_samples and s in pop_dict:
                    M_ss = cint_samples[key]
                    P_s = np.diag(pop_dict[s])[np.newaxis, ...]
                    P_s_inv = np.diag(1.0 / pop_dict[s])[np.newaxis, ...]

                    # M^{s,s} ← 0.5 * (M^{s,s} + P_s^{-1} @ M^{s,s}.T @ P_s)
                    adjusted[key] = 0.5 * (
                        M_ss + P_s_inv @ np.transpose(M_ss, (0, 2, 1)) @ P_s
                    )

            # Between-stratum pairs
            for i, s in enumerate(strata):
                for t in strata[i + 1 :]:
                    key_st = f"{s}->{t}"
                    key_ts = f"{t}->{s}"

                    if key_st in cint_samples and key_ts in cint_samples:
                        if s in pop_dict and t in pop_dict:
                            M_st = cint_samples[key_st]
                            M_ts = cint_samples[key_ts]

                            P_s = np.diag(pop_dict[s])[np.newaxis, ...]
                            P_t = np.diag(pop_dict[t])[np.newaxis, ...]
                            P_s_inv = np.diag(1.0 / pop_dict[s])[np.newaxis, ...]
                            P_t_inv = np.diag(1.0 / pop_dict[t])[np.newaxis, ...]

                            # M^{s,t} ← 0.5 * (M^{s,t} + M^{t,s}.T @ P_s^{-1} @ P_t)
                            adjusted[key_st] = 0.5 * (
                                M_st + np.transpose(M_ts, (0, 2, 1)) @ P_s_inv @ P_t
                            )

                            # M^{t,s} ← 0.5 * (M^{t,s} + M^{s,t}.T @ P_t^{-1} @ P_s)
                            adjusted[key_ts] = 0.5 * (
                                M_ts + np.transpose(M_st, (0, 2, 1)) @ P_t_inv @ P_s
                            )

            # Include any keys that weren't adjusted
            for key in cint_samples:
                if key not in adjusted:
                    adjusted[key] = cint_samples[key]

            return adjusted

        # Should not reach here
        raise ValueError(f"Unsupported stratification mode: {strat_mode}")

    def _depixilate_samples(
        self,
        samples,
        pop_data: Optional[PopulationData],
        strat_mode: str = "none",
        strata_labels: Optional[list] = None,
        age_bins: Optional[AgeBins] = None,
    ):
        """
        Depixilate posterior samples to 1-year age resolution.

        CRITICAL: This must be done BEFORE computing quantiles, because
        depixilation is a nonlinear transformation that doesn't commute
        with quantile operations.

        Parameters
        ----------
        samples : NDArray or Dict[str, NDArray]
            Posterior samples at age group resolution.
            - For K=1: NDArray of shape (n_samples, B, B)
            - For K>1: Dict mapping stratum labels to NDArray of shape (n_samples, B, B)
        pop_data : PopulationData, optional
            Population data with fine-grained (1-year) age distribution.
            Required for depixilation with population weighting.
        strat_mode : str, default="none"
            Stratification mode: "none", "partial", "full", or "mixed"
        strata_labels : list of str, optional
            Stratum labels for stratified models.
        age_bins : AgeBins, optional
            Age bin definition. If not provided, uses self.age_bins.

        Returns
        -------
        NDArray or Dict[str, NDArray]
            Depixilated samples at 1-year age resolution, same structure as input.
            - For K=1: NDArray of shape (n_samples, A, A)
            - For K>1: Dict mapping stratum labels to NDArray of shape (n_samples, A, A)

        Raises
        ------
        ValueError
            If age_bins or age_dist not available when required.
            If PopulationData stratification doesn't match model stratification.

        Notes
        -----
        Depixilation formula:
        - Unstratified: m̄_{a,b} = (P_c / P_a) * (w_{c,d} / |c||d|)
        - Stratified: m̄^{s,t}_{a,b} = (P^s_c / P^s_a) * (w^{s,t}_{c,d} / |c||d|)

        Uses source stratum population (P^s) for disaggregation weights.
        """
        if self.age_bins is None and age_bins is None:
            raise ValueError("age_bins must be provided for depixilation")

        # Validate population has fine-grained age data
        if pop_data.n_ages < self.age_bins.range:
            raise ValueError(
                f"PopulationData has only {pop_data.n_ages} ages, "
                f"but need {self.age_bins.range} for fine-grained depixilation"
            )

        A = self.age_bins.range

        # Unstratified (K=1)
        if strat_mode == "none":
            if isinstance(samples, dict):
                raise ValueError(
                    "For unstratified mode, samples should be NDArray, not Dict"
                )

            age_dist = pop_data.get_age_distribution(by_group=False)["P"].values
            n_samples = samples.shape[0]

            # Preallocate output
            depix_samples = np.empty((n_samples, A, A), dtype=np.float64)

            # Depixilate each sample
            for i in range(n_samples):
                depix_samples[i] = depixilate(samples[i], self.age_bins, age_dist)

            return depix_samples

        # Stratified (K>1)
        if not isinstance(samples, dict):
            raise ValueError("For stratified mode, samples should be Dict")

        # Extract stratified populations
        pop_by_group = pop_data.get_age_distribution(by_group=True)

        if not pop_data.strat_var_cols:
            raise ValueError(
                "PopulationData must have stratification variables for stratified depixilation"
            )

        if pop_data.n_strat_vars > 1:
            # Build composite strata string
            pop_by_group["strata"] = pop_by_group[pop_data.strat_var_cols[0]].astype(
                str
            )
            for col in pop_data.strat_var_cols[1:]:
                pop_by_group["strata"] = (
                    pop_by_group["strata"] + "_" + pop_by_group[col].astype(str)
                )

            # Set as index and keep only age and population columns
            composite_var = "strata"
            pop_by_group = pop_by_group[["strata", "age", "P"]].set_index(
                ["strata", "age"]
            )["P"]
        else:
            # Set as index and keep only age and population columns
            composite_var = pop_data.strat_var_cols[0]
            pop_by_group = pop_by_group[[composite_var, "age", "P"]].set_index(
                [composite_var, "age"]
            )["P"]

        # Extract unique source strata from labels
        source_strata = set()
        for label in strata_labels or samples.keys():
            if "->" in label:
                source, _ = label.split("->")
                source_strata.add(source)

        # Build source population dict
        source_pop_dict = {}
        for stratum in source_strata:
            if stratum in pop_by_group.index.get_level_values(0):
                source_pop_dict[stratum] = pop_by_group.xs(stratum, level=0).values
            else:
                warnings.warn(
                    f"Could not extract population for source stratum '{stratum}'. "
                    "Skipping depixilation for this stratum.",
                    UserWarning,
                )

        # Depixilate each stratum pair
        depix_samples = {}
        for label, sample in samples.items():
            if "->" in label:
                source, target = label.split("->")
                if source in source_pop_dict:
                    n_samples = sample.shape[0]
                    depix_label = np.empty((n_samples, A, A), dtype=np.float64)

                    # Use source stratum population for depixilation
                    source_age_dist = source_pop_dict[source]

                    for i in range(n_samples):
                        depix_label[i] = depixilate(
                            sample[i], self.age_bins, source_age_dist
                        )

                    depix_samples[label] = depix_label
                else:
                    # Keep original if we can't depixilate
                    depix_samples[label] = sample
            else:
                depix_samples[label] = sample

        return depix_samples

    def summarise_cint(
        self,
        alpha: float = 0.05,
        apply_reciprocity: bool = False,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for contact intensity matrix.

        Contact intensity M[c,d] represents the average number of contacts
        that individuals in age group c have with individuals in age group d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals (e.g., 0.05 for 95% CI).
        apply_reciprocity : bool, default=False
            If True, apply reciprocity adjustment to enforce demographic symmetry.
            Only applied for "none" and "full" stratification modes.
            Requires pop_data to be provided.
        return_depixilated : bool, default=False
            If True, return results at 1-year age resolution instead of age groups.
            Requires age_bins and pop_data (or age_dist) to be available.
        force_recompute : bool, default=False
            Force recomputation even if cached.

        Returns
        -------
        Dict[str, NDArray]
            Dict mapping stratum labels to NDArray of shape (3, A, A)

        Raises
        ------
        ValueError
            If alpha not in (0, 1), or required data not available for
            reciprocity/depixilation.

        Examples
        --------
        >>> # Unstratified (K=1)
        >>> summary = summariser.summarise_cint(alpha=0.05)
        >>> lower, median, upper = summary[0], summary[1], summary[2]
        >>>
        >>> # Stratified (K>1)
        >>> summary = summariser.summarise_cint(alpha=0.05)
        >>> median_M_to_F = summary["M->F"][1]  # [1] for median
        >>>
        >>> # With reciprocity and depixilation (full stratification)
        >>> pop_data = PopulationData(df, age_col='age', size_col='pop', strat_var_cols=['gender'])
        >>> summariser = ModelSummariserPrem(prem, pop_data=pop_data)
        >>> summary = summariser.summarise_cint(
        ...     alpha=0.05,
        ...     apply_reciprocity=True,
        ...     return_depixilated=True
        ... )

        Notes
        -----
        Order of operations:
        1. Reciprocity adjustment (if requested and applicable)
        2. Depixilation (if requested)
        3. Quantile computation

        This order is critical because depixilation and quantiles don't commute.

        Output format changed in v2.0:
        - Old: Dict with keys ['lower', 'median', 'upper']
        - New K=1: NDArray of shape (3, A, A)
        - New K>1: Dict[str, NDArray] with stratum labels as keys
        """
        validate_alpha(alpha)
        probs = get_probs_from_alpha(alpha)

        # Check cache
        cache_key = f"cint_alpha{alpha}_recip{apply_reciprocity}_depix{return_depixilated}_mode{self.strat_mode}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Start with posterior samples
        if isinstance(self.post_cint_samples, dict):
            samples = {k: v.copy() for k, v in self.post_cint_samples.items()}
        else:
            samples = self.post_cint_samples.copy()

        # Apply reciprocity if requested
        if apply_reciprocity:
            samples = self.apply_reciprocity(
                samples,
                self.pop_data,
                self.strat_mode,
                self.strata_labels,
                age_bins=self.age_bins,
            )

        # Apply depixilation if requested
        if return_depixilated:
            if self.pop_data is not None:
                samples = self._depixilate_samples(
                    samples,
                    self.pop_data,
                    self.strat_mode,
                    self.strata_labels,
                    age_bins=self.age_bins,
                )
            else:
                raise ValueError("pop_data must be provided for depixilation.")

        # Compute quantiles and format output
        if self.K == 1:
            # Unstratified: return NDArray of shape (3, A, A)
            if isinstance(samples, dict):
                # Extract single sample if mistakenly wrapped in dict
                samples = samples[self.strata_labels[0]]

            quantiles = compute_quantiles(samples, probs, axis=0)
            result = {"All->All": quantiles}

        else:
            # Stratified: return Dict[str, NDArray]
            if not isinstance(samples, dict):
                raise ValueError("Stratified samples should be Dict")

            result = {}
            for label, sample in samples.items():
                quantiles = compute_quantiles(sample, probs, axis=0)
                result[label] = quantiles  # Shape: (3, A, A)

        # Cache and return
        self._cache[cache_key] = result

        return result

    def summarise_rate(
        self,
        alpha: float = 0.05,
        return_symmetrized: bool = False,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for contact rate matrix.

        Contact rate R[c,d] represents the per-capita rate at which
        individuals in age group c contact individuals in age group d.
        Computed as: R[c,d] = M[c,d] / P[d]

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals.
        return_symmetrized : bool, default=False
            If True, symmetrize before computing rates.
        return_depixilated : bool, default=False
            If True, return at 1-year age resolution.
        force_recompute : bool, default=False
            Force recomputation even if cached.

        Returns
        -------
        summary : Dict[str, NDArray]
            Dictionary containing:
            - 'lower': Lower credible bound
            - 'median': Median estimate
            - 'upper': Upper credible bound
            - 'alpha': Significance level used

        Raises
        ------
        ValueError
            If age_grp_dist not available (required for rate computation).

        Returns
        -------
        NDArray
            Array of shape (3, A, A) containing quantiles:
            - [0, :, :]: Lower credible bound (2.5th percentile if alpha=0.05)
            - [1, :, :]: Median estimate (50th percentile)
            - [2, :, :]: Upper credible bound (97.5th percentile if alpha=0.05)
            where A is number of age groups (B) or fine-grained ages if depixilated

        Examples
        --------
        >>> summary = summariser.summarise_rate(alpha=0.05)
        >>> lower = summary[0]  # Lower bound
        >>> median = summary[1]  # Median
        >>> upper = summary[2]  # Upper bound

        Notes
        -----
        Rates are computed by dividing intensity by contacted population.
        This transformation is applied AFTER symmetrization/depixilation
        to ensure proper handling of population weights.
        """
        validate_alpha(alpha)
        probs = get_probs_from_alpha(alpha)

        # Check cache
        cache_key = (
            f"rate_alpha{alpha}_sym{return_symmetrized}_depix{return_depixilated}"
        )
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Validate age_grp_dist is available
        if self.age_grp_dist is None:
            raise ValueError(
                "Age group distribution required for rate computation. "
                "Provide df_age_dist or df_age_grp_dist to constructor."
            )

        # Validate requirements for symmetrization
        if return_symmetrized and self.age_grp_dist is None:
            raise ValueError("Age group distribution required for symmetrization")

        # Validate requirements for depixilation
        if return_depixilated:
            if self.age_bins is None:
                raise ValueError("age_bins required for depixilation")
            if self.age_dist is None:
                raise ValueError(
                    "Fine-grained age distribution required for depixilation"
                )

        # Start with posterior intensity samples
        samples = self.post_cint_samples.copy()

        # Apply symmetrization if requested (before converting to rate)
        if return_symmetrized:
            samples = self.symmetrize_cint_samples(samples, self.age_grp_dist)

        # Apply depixilation if requested (before converting to rate)
        if return_depixilated:
            samples = self._depixilate_samples(
                samples,
                self.pop_data,
                self.strat_mode,
                self.strata_labels,
                age_bins=self.age_bins,
            )
            # Use fine-grained age distribution for rate computation
            pop_dist = self.age_dist
        else:
            # Use age group distribution
            pop_dist = self.age_grp_dist

        # Convert intensity to rate: R[c,d] = M[c,d] / P[d]
        if isinstance(samples, dict):
            # Stratified: compute rate for each stratum
            rate_dict = {}
            for label, cint_samples in samples.items():
                # Broadcasting: cint_samples is (n_samples, B, B), pop_dist is (B,)
                rate_samples = cint_samples / pop_dist[np.newaxis, np.newaxis, :]
                quantiles = compute_quantiles(rate_samples, probs, axis=0)
                rate_dict[label] = np.stack(
                    [quantiles[0], quantiles[1], quantiles[2]], axis=0
                )
            result = rate_dict
        else:
            # Unstratified: compute rate directly
            # Broadcasting: samples is (n_samples, B, B), pop_dist is (B,)
            rate_samples = samples / pop_dist[np.newaxis, np.newaxis, :]
            quantiles = compute_quantiles(rate_samples, probs, axis=0)
            result = np.stack([quantiles[0], quantiles[1], quantiles[2]], axis=0)

        # Cache and return
        self._cache[cache_key] = result
        return result

    def summarise_mcint(
        self,
        alpha: float = 0.05,
        return_symmetrized: bool = False,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for marginal contact intensity.

        Marginal contact intensity m[c] = Σ_d M[c,d] represents the total
        average number of contacts made by individuals in age group c
        across all age groups.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals.
        return_symmetrized : bool, default=False
            If True, symmetrize before computing marginals.
        return_depixilated : bool, default=False
            If True, return at 1-year age resolution.
        force_recompute : bool, default=False
            Force recomputation even if cached.

        Returns
        -------
        NDArray
            Array of shape (3, A) containing quantiles:
            - [0, :]: Lower credible bound (2.5th percentile if alpha=0.05)
            - [1, :]: Median estimate (50th percentile)
            - [2, :]: Upper credible bound (97.5th percentile if alpha=0.05)
            where A is number of age groups (B) or fine-grained ages if depixilated

        Examples
        --------
        >>> summary = summariser.summarise_mcint(alpha=0.05)
        >>> lower = summary[0]  # Lower bound
        >>> median = summary[1]  # Median
        >>> upper = summary[2]  # Upper bound

        Notes
        -----
        Marginal intensity is computed by summing the intensity matrix
        over the contact age dimension. When depixilation is requested:
        1. Depixilate each full intensity matrix sample
        2. Compute marginals from depixilated matrices
        3. Then compute quantiles

        This ordering is critical because marginals and depixilation
        don't commute in general.
        """
        validate_alpha(alpha)
        probs = get_probs_from_alpha(alpha)

        # Check cache
        cache_key = (
            f"mcint_alpha{alpha}_sym{return_symmetrized}_depix{return_depixilated}"
        )
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Validate requirements
        if return_symmetrized and self.age_grp_dist is None:
            raise ValueError("Age group distribution required for symmetrization")

        if return_depixilated:
            if self.age_bins is None:
                raise ValueError("age_bins required for depixilation")
            if self.age_dist is None:
                raise ValueError(
                    "Fine-grained age distribution required for depixilation"
                )

        # Start with posterior samples
        samples = self.post_cint_samples.copy()

        # Apply depixilation if requested
        if return_depixilated:
            if self.pop_data is None:
                raise ValueError("pop_data must be provided for depixilation.")
            else:
                samples = self._depixilate_samples(
                    samples,
                    self.pop_data,
                    self.strat_mode,
                    self.strata_labels,
                    age_bins=self.age_bins,
                )

        # Compute marginals by summing over contact age (last axis)
        mcint_dict = {}
        if isinstance(samples, dict):
            # Stratified: compute marginals for each stratum
            for label, cint_samples in samples.items():
                mcint_samples = cint_samples.sum(
                    axis=-1
                )  # Shape: (n_samples, B) or (n_samples, A)
                quantiles = compute_quantiles(mcint_samples, probs, axis=0)
                mcint_dict[label] = np.stack(
                    [quantiles[0], quantiles[1], quantiles[2]], axis=0
                )
        else:
            # Unstratified: compute marginals directly
            mcint_samples = samples.sum(
                axis=-1
            )  # Shape: (n_samples, B) or (n_samples, A)
            quantiles = compute_quantiles(mcint_samples, probs, axis=0)
            mcint_dict["All->All"] = np.stack(
                [quantiles[0], quantiles[1], quantiles[2]], axis=0
            )

        # Cache and return
        self._cache[cache_key] = mcint_dict
        return mcint_dict

    def get_point_estimates(
        self,
        return_symmetrized: bool = False,
        return_depixilated: bool = False,
    ) -> Dict[str, Dict[str, NDArray]]:
        """
        Get point estimates (mean and std) for all statistics.

        Parameters
        ----------
        return_symmetrized : bool, default=False
            Whether to apply symmetrization.
        return_depixilated : bool, default=False
            Whether to return depixilated results.

        Returns
        -------
        estimates : Dict[str, Dict[str, NDArray]]
            Nested dictionary with structure:
            {
                'cint': {'mean': array, 'std': array},
                'rate': {'mean': array, 'std': array},
                'mcint': {'mean': array, 'std': array}
            }

        Examples
        --------
        >>> estimates = summariser.get_point_estimates()
        >>> cint_mean = estimates['cint']['mean']
        >>> cint_std = estimates['cint']['std']
        """
        # Get samples (potentially symmetrized/depixilated)
        samples = self.post_cint_samples.copy()

        if return_symmetrized:
            if self.age_grp_dist is None:
                raise ValueError("Age group distribution required for symmetrization")
            samples = self.symmetrize_cint_samples(samples, self.age_grp_dist)

        if return_depixilated:
            if self.age_bins is None or self.age_dist is None:
                raise ValueError("age_bins and age_dist required for depixilation")
            cint_samples = self._depixilate_samples(
                samples,
                self.pop_data,
                self.strat_mode,
                self.strata_labels,
                age_bins=self.age_bins,
            )
            pop_dist = self.age_dist
        else:
            cint_samples = samples
            pop_dist = self.age_grp_dist

        # Compute rate samples
        if pop_dist is not None:
            rate_samples = cint_samples / pop_dist[np.newaxis, np.newaxis, :]
        else:
            rate_samples = None

        # Compute marginals
        mcint_samples = cint_samples.sum(axis=2)

        # Prepare results
        result = {
            "cint": {
                "mean": cint_samples.mean(axis=0),
                "std": cint_samples.std(axis=0, ddof=1),
            },
        }

        if rate_samples is not None:
            result["rate"] = {
                "mean": rate_samples.mean(axis=0),
                "std": rate_samples.std(axis=0, ddof=1),
            }

        result["mcint"] = {
            "mean": mcint_samples.mean(axis=0),
            "std": mcint_samples.std(axis=0, ddof=1),
        }

        return result

    def clear_cache(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached results.

        Returns
        -------
        info : Dict[str, Any]
            Dictionary with cache statistics including number of cached items
            and their keys.
        """
        return {
            "n_cached": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }
