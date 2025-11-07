"""
Hierarchical Bayesian Rate Consistency model with fine age resolution.

This module implements the HiBRCfine model, which extends BRCfine to support
stratified populations (e.g., by gender, setting) using hierarchical priors
with fine-grained age resolution for both participants and contacts.
"""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax.typing import ArrayLike
from numpy.typing import NDArray
from numpyro import distributions as dist
from numpyro.handlers import plate, scope

from ..dataloader import DataLoader
from ._BRCfine import BRCfine
from .priors import Hill, Prior2D


class HiBRCfine(BRCfine):
    """
    Hierarchical Bayesian Rate Consistency model with fine age resolution.

    This model extends BRCfine to handle stratified contact data (e.g., by gender,
    setting, region) using hierarchical priors. Unlike HiBRCrefine, both participant
    ages AND contact ages are at fine (single-year) resolution.

    The model combines:
    1. Fine age resolution: Both participant and contact ages are single-year
    2. Hierarchical structure: Models population subgroups with shared smooth patterns
    3. Rate consistency: Ensures bidirectional contact balance via population weights

    The model is particularly useful for:
    - Estimating gender-specific or setting-specific contact matrices
    - Analyzing contact patterns across multiple demographic strata
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

    Key Differences from BRCfine
    -----------------------------
    - Adds hierarchical priors for stratification variables (gender, setting, etc.)
    - Requires population age distributions for each stratum
    - Uses stratum-specific deviations centered on population proportions
    - Accumulates contributions from multiple strata additively in log space

    Key Differences from HiBRCrefine
    ---------------------------------
    - Uses fine contact ages (single-year) instead of coarse groups
    - No age aggregation mechanism needed (simpler indexing)
    - Direct indexing via aid[i] and bid[i] instead of aid_exp/bid_pad

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing stratified contact data with columns:
        - y: observed contact counts
        - aid: participant age indices (fine resolution)
        - bid: contact age indices (fine resolution)
        - log_N: log of survey sample sizes
        - log_P: log of overall population age distribution
        - log_S: log of setting-specific offsets (optional)
        - rid: repeat interview indicators (optional)
        - grp_vars: stratification variables (e.g., gender, setting)
        - pop_prop_{var}: stratum-specific population proportions for each grp_var
    priors : Dict[str, Prior2D]
        Dictionary of prior specifications. Must contain:
        - 'rate': Prior2D for baseline smooth age-age contact rates
          (e.g., HSGP2D, Spline2D, PSpline2D)
        - One Prior2D per stratification variable in grp_vars
          (e.g., 'gender', 'setting') for hierarchical deviations
    likelihood : str, default='negbin'
        Observation likelihood:
        - 'negbin': Negative binomial (recommended for overdispersed counts)
        - 'poisson': Poisson (assumes mean = variance)

    Attributes
    ----------
    X_vars : list[str]
        Names of stratification variables (keys in priors except 'rate')
    X_ids : dict[str, NDArray]
        Categorical codes for each stratification variable, shape (n_obs,)
    log_age_dist_props : dict[str, jax.Array]
        Log population proportions for each stratum, used to center deviations

    Raises
    ------
    ValueError
        If priors don't match grp_vars in dataset
        If population proportions have incorrect shapes
        If required data fields are missing

    Notes
    -----
    Data Preparation Requirements:
    1. DataLoader must include stratification via grp_vars_part in CoordToColumns
    2. Must use age_cnt (not age_grp_cnt) for fine contact ages
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
    >>> from cntmosaic.models import HiBRCfine
    >>> from cntmosaic.models.priors import HSGP2D
    >>> from jax.random import PRNGKey
    >>>
    >>> # Set up stratified dataloader with fine contact ages
    >>> col_map = CoordToColumns(
    ...     age_part="age_part",
    ...     age_cnt="age_cnt",  # Fine contact ages
    ...     age_pop="age",
    ...     size_pop="P",
    ...     grp_vars_part=["gender"]  # Stratification variable
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
    BRCfine : Base class for fine-grained age resolution (no stratification)
    HiBRCrefine : Hierarchical model with coarse contact ages (requires refinement)
    HSGP2D : Hilbert Space Gaussian Process prior for smooth functions
    DataLoader : Data preprocessing with stratification support
    """

    # Default priors matching parent class
    default_priors = {"rate": Prior2D}  # Placeholder, subclass should define

    def __init__(
        self,
        dataloader: DataLoader,
        priors: Dict[str, Prior2D],
        likelihood: str = "negbin",
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

        Notes
        -----
        Initialization steps:
        1. Calls parent BRCfine.__init__ for base setup
        2. Validates hierarchical-specific requirements
        3. Extracts and encodes stratification variables
        4. Configures priors with correct event dimensions
        5. Sets up log population proportions for centering
        6. Initializes repeat interview effects if present
        """
        # Initialize parent class (BRCfine) - this calls BRC.__init__ internally
        super().__init__(dataloader, priors, likelihood)

        # Extract stratification variable names BEFORE validation
        # (all priors except 'rate')
        self.X_vars = [key for key in priors.keys() if key != "rate"]

        # Validate hierarchical-specific requirements
        self._validate_hierarchical_inputs()

        # Encode categorical stratification variables as integer codes
        self.X_ids = {
            var: pd.Categorical(
                self.ds[var].values, categories=sorted(set(self.ds[var].values))
            ).codes
            for var in self.X_vars
        }

        # Configure prior dimensions based on dataset structure
        self.set_prior_event_dim()

        # Set prior locations centered on population proportions
        self.set_prior_loc()

        # Compute log population proportions for each stratum
        self.set_log_age_dist_props()

        # Optional: Set up repeat interview effects if present
        if hasattr(self.ds, "rid"):
            self.rid = jnp.array(self.ds.rid.values, dtype=jnp.int32)
            self.hill = Hill(max_value=int(self.ds.rid.max()))

    def _validate_hierarchical_inputs(self) -> None:
        """
        Validate hierarchical-specific data requirements.

        Checks that:
        1. 'rate' prior exists and has prior_type='full'
        2. Stratification variables in priors match grp_vars in dataset
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
            raise ValueError(
                "'rate' prior must be provided in the priors dictionary.\n"
                "This prior defines the baseline age-age contact pattern."
            )

        if self.priors["rate"].prior_type != "global":
            raise ValueError(
                f"'rate' prior_type must be 'global', but got '{self.priors['rate'].prior_type}'.\n"
                "The baseline contact pattern requires a global prior."
            )

        # Check that stratification variables match between priors and data
        dataset_grp_vars = set(self.ds.attrs.get("grp_vars", {}).keys())
        prior_grp_vars = set(var for var in self.priors.keys() if var != "rate")

        if prior_grp_vars != dataset_grp_vars:
            raise ValueError(
                f"Mismatch between stratification variables in priors and dataset.\n"
                f"Priors contain: {sorted(prior_grp_vars)}\n"
                f"Dataset grp_vars contain: {sorted(dataset_grp_vars)}\n"
                f"They must match exactly."
            )

        # Check that each stratification prior has compatible prior_type
        for var in prior_grp_vars:
            if self.priors[var].prior_type not in ["partial", "full"]:
                raise ValueError(
                    f"Prior for stratification variable '{var}' must have prior_type "
                    f"'partial' or 'full', but got '{self.priors[var].prior_type}'."
                )

        # Check population proportions exist for each stratification variable
        for var in self.X_vars:
            pop_prop_key = f"pop_prop_{var}"
            if not hasattr(self.ds, pop_prop_key):
                raise ValueError(
                    f"Missing population proportions for stratification variable '{var}'.\n"
                    f"Expected dataset to have '{pop_prop_key}' attribute.\n"
                    f"Ensure your DataLoader includes population data for this variable."
                )

    def set_log_age_dist_props(self) -> None:
        """
        Compute log-transformed population age proportions for each stratum.

        These proportions are used to center the stratum-specific deviations (δ_s)
        around the population structure, ensuring that the model doesn't simply
        learn the population age distribution.

        The shape depends on the prior_type:
        - prior_type='partial': (n_strata, A) - row/column-specific
        - prior_type='full': (n_strata, A, A) - full matrix for each stratum

        Raises
        ------
        ValueError
            If population proportion shapes don't match expected dimensions

        Notes
        -----
        The computed log proportions are stored in self.log_age_dist_props[var]
        and are subtracted from the prior samples in sample_log_delta() to create
        centered deviations.
        """
        self.log_age_dist_props = {}
        grp_vars = self.ds.attrs.get("grp_vars", {})
        for var in grp_vars.keys():
            pop_prop = self.ds[f"pop_prop_{var}"].to_numpy()
            expected_event_dim = self.priors[var].event_dim

            # Check shape and apply appropriate transformation
            if pop_prop.shape == (expected_event_dim, self.A):
                # Partial prior: add trailing dimension for broadcasting
                self.log_age_dist_props[var] = jnp.log(pop_prop)[:, :, jnp.newaxis]
            elif pop_prop.shape == (expected_event_dim, self.A, self.A):
                # Full prior: use as-is
                self.log_age_dist_props[var] = jnp.log(pop_prop)
            else:
                raise ValueError(
                    f"Invalid shape for population proportions of '{var}'.\n"
                    f"Expected shape: ({expected_event_dim}, {self.A}) or "
                    f"({expected_event_dim}, {self.A}, {self.A})\n"
                    f"Got shape: {pop_prop.shape}\n"
                    f"This mismatch may be due to incorrect prior_type or "
                    f"malformed population data."
                )

    def set_prior_event_dim(self) -> None:
        """
        Configure event dimensions for each prior based on dataset structure.

        The event dimension determines how many independent realizations of a prior
        are needed:
        - 'rate' prior: Always 1 (shared baseline across all strata)
        - Stratification priors: Equal to number of categories in that variable

        For example, if gender has categories [male, female], then
        priors['gender'].event_dim = 2.

        Notes
        -----
        This method is called during initialization and should not be called manually.
        The event_dim affects how samples are drawn from the prior and how they're
        indexed during the model evaluation.
        """
        for var, prior in self.priors.items():
            if var == "rate":
                prior.set_event_dim(1)  # Shared baseline
            else:
                # Number of strata for this variable
                n_strata = len(self.ds.grp_vars[var])
                prior.set_event_dim(n_strata)

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
                pop_prop_data = self.ds[f"pop_prop_{var}"].to_numpy()
                prior.set_loc(pop_prop_data)

    def sample_log_delta(self, var: str) -> ArrayLike:
        """
        Sample stratum-specific log-scale deviations from population baseline.

        This method generates hierarchical adjustments for a given stratification
        variable by:
        1. Sampling from the specified prior
        2. Taking the log (if prior outputs are in probability space)
        3. Centering around population proportions by subtracting log(P_s)

        The result represents multiplicative deviations: δ_s = sample / P_s

        Parameters
        ----------
        var : str
            Name of the stratification variable (must be in self.X_vars)

        Returns
        -------
        ArrayLike
            Log-scale deviations with shape depending on prior_type:
            - prior_type='partial': (n_strata, A, 1)
            - prior_type='full': (n_strata, A, A)

        Notes
        -----
        The returned values are registered as a deterministic site in NumPyro
        with name 'log_delta', allowing posterior tracking.
        """
        log_delta = numpyro.deterministic(
            "log_delta",
            jnp.log(self.priors[var].sample()) - self.log_age_dist_props[var],
        )
        return log_delta

    def model(self, y: Optional[ArrayLike] = None) -> None:
        """
        NumPyro generative model for hierarchical Bayesian contact matrix estimation.

        This model specifies the complete generative process:
        1. Sample baseline log-intensity (β₀) from Normal prior
        2. Sample rate pattern f from 'rate' prior (shared across strata)
        3. Compute log rate: log(λ) = β₀ + f
        4. Add population age adjustment and stratification effects
        5. Convert to expected contact counts and sample observations

        **Model Structure**:

        log(E[y_ijk]) = log(λ_ab) + log(P_b) + log(N_i) + log(S_i) + h_r + Σ_s δ_s,ij,ab

        where:
        - λ_ab: Baseline contact intensity (age a → age b)
        - P_b: Population age proportion
        - N_i: Participant survey duration
        - S_i: Participant seasonal weight
        - h_r: Repeat interview effect (if applicable)
        - δ_s: Stratum-specific multiplicative adjustment

        **Likelihood Options**:
        - Poisson: E[y] = μ
        - Negative Binomial: E[y] = μ, Var[y] = μ + μ²/ϕ (with ϕ ~ Exponential(1))

        Parameters
        ----------
        y : ArrayLike, optional
            Observed contact counts. Shape: (n_observations,)
            If None, samples from the prior (useful for prior predictive checks)

        Notes
        -----
        This method is intended for use with NumPyro's inference algorithms
        (MCMC, SVI). Do not call directly - use run_inference_mcmc/svi instead.

        The model uses hierarchical scoping (`numpyro.scope`) to organize parameters:
        - 'rate' scope: Baseline contact pattern
        - Variable-specific scopes: Stratification adjustments (e.g., 'gender', 'setting')

        Examples
        --------
        Run MCMC inference:

        >>> model = HiBRCfine(dataloader, priors={...})
        >>> model.set_age_dims(0, 85)
        >>> model.run_inference_mcmc(rng_key, num_samples=1000)

        Prior predictive sampling:

        >>> predictive = numpyro.infer.Predictive(model.model, num_samples=100)
        >>> prior_samples = predictive(rng_key)

        See Also
        --------
        run_inference_mcmc : Run NUTS sampling
        run_inference_svi : Run stochastic variational inference
        """
        # Sample baseline log-intensity
        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))

        # Sample shared rate pattern
        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        # Compute baseline log rate
        log_rate = numpyro.deterministic("log_rate", beta0 + f)

        # Initialize log contact intensity with population adjustment
        log_cint = (log_rate + self.log_P)[self.aid, self.bid]

        # Add stratification effects
        for var in self.X_vars:
            with scope(prefix=var):
                log_cint += self.sample_log_delta(var)[
                    self.X_ids[var], self.aid, self.bid
                ]

        # Add repeat interview effect if present
        repeat_effect = self.hill.sample()[self.rid] if hasattr(self, "rid") else 0.0

        # Compute expected counts
        mu = jnp.exp(log_cint + self.log_N + self.log_S + repeat_effect)

        # Likelihood
        if self.likelihood == "poisson":
            with plate("data", len(self.y)):
                numpyro.sample("obs", dist.Poisson(rate=mu), obs=y)

        if self.likelihood == "negbin":
            inv_disp = numpyro.sample("inv_disp", dist.Exponential(1.0))
            with plate("data", len(self.y)):
                numpyro.sample(
                    "obs",
                    dist.NegativeBinomial2(mean=mu, concentration=1.0 / inv_disp),
                    obs=y,
                )
