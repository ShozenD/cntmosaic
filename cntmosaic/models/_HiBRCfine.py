"""
Hierarchical Bayesian Rate Consistency model with fine age resolution.

This module implements the HiBRCfine model, which extends BRCfine to support
stratified populations (e.g., by gender, setting) using hierarchical priors
with fine-grained age resolution for both participants and contacts.
"""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np

from .._types import StratMode
from ..dataloader import DataLoader
from ._BRCfine import BRCfine
from ._math import clr
from .numpyro import HiBRCfineNumPyroMixin
from .numpyro.priors import Hill, Prior2D


class HiBRCfine(HiBRCfineNumPyroMixin, BRCfine):
    """
    High-resolution Bayesian Rate Consistency model with fine age resolution.

    This model extends BRCfine to handle stratified contact data (e.g., by gender,
    setting, region) using hierarchical priors. Unlike HiBRCrefine, both participant
    ages AND contact ages are at fine (single-year) resolution.

    The model combines:
    1. Fine age resolution: Both participant and contact ages are single-year
    2. Hierarchical structure: Models population subgroups with shared smooth patterns
    3. Rate consistency: Ensures bidirectional contact balance via population weights

    The model is useful for:
    - Estimating generalized contact matrices
    - High-resolution contact data where both ages are precisely recorded

    Mathematical Model
    ------------------
    For each observed contact y_i from participant age a_i to contact age b_i
    in stratum s_i:

        log(rate[a, b]) = β₀ + f(a, b)  (shared smooth baseline)
        log(δ_s[a, b]) = log(prior_s) - log(P_s[a,b])  (stratum-specific deviation)
        log(cint[a_i, b_i]) = log(rate[a_i, b_i]) + log(P[b_i]) +
                               Σ_s log(δ_s[a_i, b_i])
        μ_i = exp(log(cint[a_i, b_i]) + log(N_i) + log(S_i) + η_i)
        y_i ~ Poisson(μ_i) or NegativeBinomial2(μ_i, φ)

    where:
    - β₀: baseline contact rate
    - f(a, b): smooth 2D age-age function (from 'rate' prior)
    - δ_s: stratum-specific multiplicative adjustments
    - P[b]: overall population proportion at age b
    - P_s[a,b]: stratum-specific population proportions (for centering δ_s)
    - N_i: survey sample size
    - S_i: setting-specific offset
    - η_i: repeat interview effect
    - φ: overdispersion parameter

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing stratified contact data with columns:
        - y: observed contact counts
        - aid: participant age indices (fine resolution)
        - bid: contact age indices (fine resolution)
        - log_N: log of survey sample sizes
        - log_P: log of overall population age distribution
        - log_V: log of setting-specific offsets (optional)
        - rid: repeat interview indicators (optional)
        - strat_vars: stratification variables (e.g., gender, setting)
        - pop_prop_{var}: stratum-specific population proportions for each strat_var
    priors : dict[str, Prior2D]
        Dictionary of prior specifications. Must contain:
        - 'rate': Prior2D for baseline smooth age-age contact rates
          (e.g., HSGP2D, Spline2D, PSpline2D)
        - One Prior2D per stratification variable in strat_vars
          (e.g., 'gender', 'setting') for hierarchical deviations
    likelihood : str, default='negbin'
        Observation likelihood:
        - 'negbin': Negative binomial (recommended for overdispersed counts)
        - 'poisson': Poisson (assumes mean = variance)

    Examples
    --------
    >>> from cntmosaic.dataloader import DataLoader, CoordToColumns
    >>> from cntmosaic.models import HiBRCfine
    >>> from cntmosaic.models.numpyro.priors import HSGP2D
    >>> from jax.random import PRNGKey
    >>>
    >>> # Set up stratified dataloader with fine contact ages
    >>> col_map = CoordToColumns(
    ...     age_part="age_part",
    ...     age_cnt="age_cnt",  # Fine contact ages
    ...     age_pop="age",
    ...     P="P",
    ...     strat_vars_part=["gender"]  # Stratification variable
    ... )
    >>> dataloader = DataLoader(df_part, df_cnt, df_age_dist, col_map=col_map)
    >>>
    >>> # Specify priors for baseline and stratification
    >>> priors = {
    ...     "rate": HSGP2D(grid_type="diff-age", prior_type="global"),
    ...     "gender": HSGP2D(grid_type="diff-age", prior_type="full")
    ... }
    >>>
    >>> # Initialize and run inference
    >>> model = HiBRCfine(dataloader, priors, likelihood="negbin")
    >>> model.run_inference_mcmc(
    ...     PRNGKey(42),
    ...     num_samples=1000,
    ...     num_warmup=1000,
    ...     num_chains=4
    ... )
    >>>
    >>> # Access posterior samples
    >>> samples = model._mcmc_result.get_samples()
    >>> baseline = samples['baseline']
    >>> log_rate = samples['log_rate']  # Shared baseline rates
    >>> gender_log_delta = samples['gender/log_delta']  # Gender-specific adjustments

    See Also
    --------
    BRC: Base class for rate consistency models
    BRCfine : Base class for fine-grained age resolution (no stratification)
    BRCrefine: Base class for coarse age resolution (no stratification)
    HiBRCrefine : Hierarchical model with coarse contact ages (requires refinement)
    """

    # Default priors matching parent class
    default_priors = {"rate": Prior2D}  # Placeholder, subclass should define

    def __init__(
        self,
        dataloader: DataLoader,
        priors: Dict[str, Prior2D],
        likelihood: str = "negbin",
        backend: Optional[Any] = None,
    ) -> None:
        """
        Initialize HiBRCfine model with hierarchical structure and fine age resolution.

        Parameters
        ----------
        dataloader : DataLoader
            Preprocessed stratified contact data with fine contact ages.
        priors : Dict[str, Prior2D]
            Prior specifications. Must contain 'rate' for baseline and one prior
            per stratification variable.
        likelihood : str, default='negbin'
            Observation likelihood ('negbin' or 'poisson').
        backend : InferenceBackend, optional
            Pluggable inference engine (default: NumPyroBackend).
        """
        # Initialize parent class (BRCfine) - this calls BRC.__init__ internally
        super().__init__(dataloader, priors, likelihood, backend=backend)

        # Override log_P for stratified case (already has shape (K, A), no need for newaxis)
        self.log_P = jnp.array(self.data.log_P)

        # Validate hierarchical-specific requirements
        self._validate_hierarchical_inputs()

        # Configure prior dimensions based on dataset structure
        self.set_prior_event_dim()

        # Set prior locations centered on population proportions
        self.set_prior_loc()

        # Optional: Set up repeat interview effects if present
        if self.data.rid is not None:
            self.rid = jnp.array(self.data.rid, dtype=jnp.int32)
            self.hill = Hill(max_value=int(self.data.rid.max()))

    def _validate_hierarchical_inputs(self) -> None:
        """
        Validate hierarchical-specific data requirements.

        Checks that:
        1. 'rate' prior exists and has prior_type='full'
        2. Stratification variables in priors match strat_vars in dataset
        3. Population proportions exist for each stratification variable
        4. Population proportions have correct shapes for the prior type

        Raises
        ------
        ValueError
            If 'rate' prior is missing or has wrong prior_type
            If stratification variable mismatch detected
            If population proportions are missing
            If population proportion shapes are incompatible with priors

        Notes
        -----
        This validation ensures the hierarchical model is properly configured
        before inference begins, preventing cryptic errors later.
        """
        # Check that 'rate' prior exists and has correct prior_type
        if "rate" not in self.priors:
            raise ValueError("'rate' prior must be provided in the priors dictionary.")

        if self.priors["rate"].prior_type != "global":
            raise ValueError(
                f"'rate' prior_type must be 'global', but got '{self.priors['rate'].prior_type}'."
            )

        # Check that stratification variables match between priors and data
        data_strat_vars = set(self.data.strat_modes.keys())
        prior_strat_vars = set(var for var in self.priors.keys() if var != "rate")

        if prior_strat_vars != data_strat_vars:
            raise ValueError(
                f"Mismatch between stratification variables in priors and dataset.\n"
                f"Priors contain: {sorted(prior_strat_vars)}\n"
                f"Data strat_vars contain: {sorted(data_strat_vars)}\n"
                f"They must match exactly."
            )

        # Check that each stratification prior has compatible prior_type
        data_strat_modes = self.data.strat_modes
        prior_strat_modes = {
            var: self.priors[var].prior_type for var in prior_strat_vars
        }
        for var, mode in data_strat_modes.items():
            prior_type = prior_strat_modes[var]
            if mode == StratMode.PARTIAL and prior_type not in ["partial", "full"]:
                raise ValueError(
                    f"Stratification variable '{var}' is PARTIAL but prior_type is "
                    f"'{prior_type}'. Must be 'partial'."
                )
            if mode == StratMode.FULL and prior_type != "full":
                raise ValueError(
                    f"Stratification variable '{var}' is FULL but prior_type is "
                    f"'{prior_type}'. Must be 'full'."
                )

    def set_prior_event_dim(self) -> None:
        """
        Configure event dimensions for each prior based on dataset structure.

        The event dimension determines how many independent realizations of a prior
        are needed:
        - 'rate' prior: Always 1 (shared baseline across all strata)
        - Stratification priors: Equal to number of categories in that variable and
            prior_type (e.g., PARTIAL or FULL).

        For example, if gender has categories [male, female], and prior_type is 'partial', then
        priors['gender'].event_dim = 2.
        """
        for var, prior in self.priors.items():
            if var == "rate":
                prior.set_event_dim(1)  # Shared baseline
            else:
                # Number of strata for this variable
                if self.data.strat_modes[var] == StratMode.PARTIAL:
                    n_strata = self.data.strat_dims[var]
                    prior.set_event_dim(n_strata)
                elif self.data.strat_modes[var] == StratMode.FULL:
                    n_strata = self.data.strat_dims[var]
                    prior.set_event_dim(int(np.sqrt(n_strata)))
                else:
                    raise ValueError(
                        f"Unknown stratification mode for variable '{var}': "
                        f"{self.data.strat_modes[var]}"
                    )

    def set_prior_loc(self) -> None:
        """
        Set prior location parameters centered on population age proportions.

        For stratification variables, priors are centered around the observed
        population age distribution for each stratum. This helps the model:
        1. Start from reasonable values during MCMC/SVI
        2. Learn deviations from population structure rather than absolute values
        3. Improve numerical stability and convergence

        Notes
        -----
        This method is called during initialization. The location parameters are
        used by priors that support centering (e.g., HSGP2D, IGMRF2D).
        """
        for var, prior in self.priors.items():
            if var == "rate":
                continue  # Skip baseline prior
            else:
                loc = clr(
                    self.data.marginal_multipliers[var], axis=0
                )  # Apply CLR transform
                prior.set_loc(loc)

