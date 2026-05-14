"""
Hierarchical Bayesian Rate Consistency model with coarse-to-fine age refinement.

This module implements the HiBRCrefine model, which extends BRCrefine to support
stratified populations (e.g., by gender, setting) using hierarchical priors.
"""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np

from .._types import StratMode
from ..dataloader import DataLoader
from ._BRCrefine import BRCrefine
from ._math import clr
from .numpyro import HiBRCrefineNumPyroMixin
from .numpyro.priors import Hill, Prior2D, PSpline2D


class HiBRCrefine(HiBRCrefineNumPyroMixin, BRCrefine):
    """
    Hierarchical Bayesian Rate Consistency model with coarse-to-fine age refinement.

    This model extends BRCrefine to handle stratified contact data (e.g., by gender,
    setting, region) using hierarchical priors. It combines:
    1. Age refinement: Estimates fine-age contact rates from coarse contact age data
    2. Hierarchical structure: Models population subgroups with shared smooth patterns
    3. Rate consistency: Ensures bidirectional contact balance via population weights

    The model is particularly useful for:
    - Estimating gender-specific or setting-specific contact matrices
    - Analyzing contact patterns across multiple demographic strata
    - Handling survey data with coarse contact age groups but stratified participants

    Mathematical Model
    ------------------
    For each observed contact y_i from participant age a_i to contact age group [b_l, b_u)
    in stratum s_i:

        log(rate[a, b]) = β₀ + f(a, b)  (shared smooth baseline)
        log(δ_s[a, b]) = log(prior_s) - log(P_s[a,b])  (stratum-specific deviation)
        log(cint_base[a, b]) = log(rate[a, b]) + log(P[b])
        log(cint_i) = logsumexp_{b ∈ [b_l,b_u)} (log(cint_base[a_i,b]) + log(δ_s[a_i,b]))
        μ_i = exp(log(cint_i) + log(N_i) + log(S_i) + η_i)
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
    - logsumexp: age aggregation over coarse groups

    Key Differences from BRCrefine
    -------------------------------
    - Adds hierarchical priors for stratification variables (gender, setting, etc.)
    - Requires population age distributions for each stratum
    - Uses stratum-specific deviations centered on population proportions
    - Aggregates contributions from multiple strata during age refinement

    Key Differences from HiBRCfine
    -------------------------------
    - Uses coarse contact age data (age groups) instead of fine ages
    - Employs age aggregation mechanism (index_mask_logsumexp)
    - Requires aid_exp and bid_pad arrays for refinement

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing stratified contact data with columns:
        - y: observed contact counts
        - aid: participant age indices (fine resolution)
        - aid_exp: expanded participant ages for age aggregation
        - bid_pad: padded contact age indices for coarse groups
        - log_N: log of survey sample sizes
        - log_P: log of overall population age distribution
        - log_V: log of setting-specific offsets (optional)
        - rid: repeat interview indicators (optional)
        - strat_vars: stratification variables (e.g., gender, setting)
        - pop_prop_{var}: stratum-specific population proportions for each strat_var
    priors : dict
        Dictionary of prior specifications. Must contain:
        - 'rate': Prior2D for baseline smooth age-age contact rates
          (e.g., HSGP2D, PSpline2D)
        - One Prior2D per stratification variable in strat_vars
          (e.g., 'gender', 'setting') for hierarchical deviations
    likelihood : str, default='negbin'
        Observation likelihood:
        - 'negbin': Negative binomial (recommended for overdispersed counts)
        - 'poisson': Poisson (assumes mean = variance)

    Attributes
    ----------
    X_vars : list[str]
        Names of stratification variables (keys in priors except 'rate')
    strat_ix : dict[str, NDArray]
        Categorical codes for each stratification variable, shape (n_obs,)
    strat_ix_exp : dict[str, NDArray]
        Expanded categorical codes for age aggregation, shape (n_obs, max_int_length)
    log_age_dist_props : dict[str, jax.Array]
        Log population proportions for each stratum, used to center deviations

    Raises
    ------
    ValueError
        If priors don't match strat_vars in dataset
        If population proportions have incorrect shapes
        If required data fields are missing

    Notes
    -----
    Data Preparation Requirements:
    1. DataLoader must include stratification via strat_vars_part in CoordToColumns
    2. Must use age_grp_cnt (not age_cnt) for coarse contact ages
    3. Population data must include proportions for each stratum combination

    The model uses compositional transformations (ALR/CLR/ILR) specified in each
    Prior2D to handle the simplex constraint on age-specific contact proportions.

    References
    ----------
    Shozen Dan et al., "Estimating fine age structure and time trends in
    human contact patterns from coarse contact data: The Bayesian rate consistency model",
    PLoS Computational Biology. 2023

    Examples
    --------
    >>> from cntmosaic.dataloader import DataLoader, CoordToColumns
    >>> from cntmosaic.models import HiBRCrefine
    >>> from cntmosaic.models.priors import HSGP2D
    >>> from jax.random import PRNGKey
    >>>
    >>> # Set up stratified dataloader with coarse contact ages
    >>> col_map = CoordToColumns(
    ...     age_part="age_part",
    ...     age_grp_cnt="age_grp_cnt",  # Coarse contact age groups
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
    >>> model = HiBRCrefine(dataloader, priors, likelihood="negbin")
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
    BRCrefine : Base class for coarse-to-fine age refinement
    HiBRCfine : Hierarchical model with fine contact ages (no refinement)
    HSGP2D : Hilbert Space Gaussian Process prior for smooth functions
    DataLoader : Data preprocessing with stratification support
    """

    # Default priors matching parent class
    default_priors = {"rate": PSpline2D(grid_type="diff-age", prior_type="global")}

    def __init__(
        self,
        dataloader: DataLoader,
        priors: Dict[str, Prior2D],
        likelihood: str = "negbin",
        backend: Optional[Any] = None,
    ) -> None:
        """
        Initialize HiBRCrefine model with hierarchical structure and age refinement.

        Parameters
        ----------
        dataloader : DataLoader
            Preprocessed stratified contact data with coarse contact age groups.
        priors : Dict[str, Prior2D]
            Prior specifications. Must contain 'rate' for baseline and one prior
            per stratification variable.
        likelihood : str, default='negbin'
            Observation likelihood ('negbin' or 'poisson').
        backend : InferenceBackend, optional
            Pluggable inference engine (default: NumPyroBackend).
        """
        # Merge user priors with defaults (user priors take precedence)
        effective_priors = self.default_priors.copy()
        effective_priors.update(priors)

        # Initialize parent class (BRCrefine) - this calls BRC.__init__ internally
        super().__init__(dataloader, effective_priors, likelihood, backend=backend)

        # Override log_P for stratified case
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
        1. Population proportions exist for each stratification variable
        2. Population proportions have correct shapes for the prior type

        Raises
        ------
        ValueError
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

        For the 'rate' prior, no location is set (defaults to zero/uninformative).

        Notes
        -----
        This method is called during initialization. The location parameters are
        used by priors that support centering (e.g., HSGP2D, IGMRF2D).
        Not all priors use the location parameter - it's prior-specific.
        """
        for var, prior in self.priors.items():
            if var != "rate":
                loc = clr(
                    self.data.marginal_multipliers[var], axis=0
                )  # Apply CLR transform
                prior.set_loc(loc)

