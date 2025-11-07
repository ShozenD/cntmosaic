"""
Hierarchical Bayesian Rate Consistency model with coarse-to-fine age refinement.

This module implements the HiBRCrefine model, which extends BRCrefine to support
stratified populations (e.g., by gender, setting) using hierarchical priors.
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
from ._BRCrefine import BRCrefine
from ._utils import index_mask_logsumexp
from .priors import Hill, Prior2D, PSpline2D


def _expand_id_array(id_array: NDArray, length: int) -> NDArray:
    """
    Expand a 1D ID array by repeating each element along a new axis.

    This helper function is used to broadcast participant-level categorical IDs
    across the maximum coarse age group width for age aggregation.

    Parameters
    ----------
    id_array : NDArray
        1D array of categorical IDs, shape (n_obs,)
    length : int
        Number of repetitions along the new axis (typically max coarse age group width)

    Returns
    -------
    NDArray
        2D array with shape (n_obs, length) where each row contains repeated IDs

    Examples
    --------
    >>> ids = np.array([0, 1, 0, 2])
    >>> _expand_id_array(ids, 3)
    array([[0, 0, 0],
           [1, 1, 1],
           [0, 0, 0],
           [2, 2, 2]])
    """
    return np.repeat(id_array[:, np.newaxis], length, axis=1)


class HiBRCrefine(BRCrefine):
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
        - log_S: log of setting-specific offsets (optional)
        - rid: repeat interview indicators (optional)
        - grp_vars: stratification variables (e.g., gender, setting)
        - pop_prop_{var}: stratum-specific population proportions for each grp_var
    priors : dict
        Dictionary of prior specifications. Must contain:
        - 'rate': Prior2D for baseline smooth age-age contact rates
          (e.g., HSGP2D, PSpline2D)
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
    X_ids_exp : dict[str, NDArray]
        Expanded categorical codes for age aggregation, shape (n_obs, max_int_length)
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

        Notes
        -----
        Initialization steps:
        1. Merges user priors with default_priors
        2. Calls parent BRCrefine.__init__ for base setup
        3. Validates hierarchical-specific requirements
        4. Extracts and encodes stratification variables
        5. Expands categorical IDs for age aggregation
        6. Configures priors with correct event dimensions
        7. Sets up log population proportions for centering
        8. Initializes repeat interview effects if present
        """
        # Merge user priors with defaults (user priors take precedence)
        effective_priors = self.default_priors.copy()
        effective_priors.update(priors)

        # Initialize parent class (BRCrefine) - this calls BRC.__init__ internally
        super().__init__(dataloader, effective_priors, likelihood)

        # Extract stratification variable names BEFORE validation
        # (all priors except 'rate')
        self.X_vars = [key for key in self.priors.keys() if key != "rate"]

        # Validate hierarchical-specific requirements
        self._validate_hierarchical_inputs()

        # Encode categorical stratification variables as integer codes
        self.X_ids = {
            var: pd.Categorical(
                self.ds[var].values, categories=sorted(set(self.ds[var].values))
            ).codes
            for var in self.X_vars
        }

        # Expand categorical IDs for age aggregation (repeat across max coarse age width)
        self.X_ids_exp = {
            var: _expand_id_array(self.X_ids[var], self.bid_pad.shape[1])
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
        for var in self.ds.attrs["grp_vars"].keys():
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
        Define the hierarchical generative model with age refinement.

        This model combines:
        1. Shared smooth baseline contact rates across all strata
        2. Stratum-specific multiplicative adjustments
        3. Age aggregation over coarse contact age groups
        4. Rate consistency through population weighting

        Model Structure
        ---------------
        1. **Baseline**: β₀ ~ Normal(0, 10²)
           Global intercept for contact rates

        2. **Smooth baseline function**: f(a,b) from priors['rate']
           Shared across all strata, captures common age patterns

        3. **Log contact rate**: log(rate) = β₀ + f(a,b)
           Fine-age resolution baseline rates (A × A matrix)

        4. **Contact intensity baseline**: log(cint_base) = log(rate) + log(P)
           Adds population structure for rate consistency

        5. **Stratum-specific adjustments**:
           For each stratification variable s:
               log(δ_s[a,b]) = log(prior_s) - log(P_s[a,b])
               contribution_s = logsumexp(log(cint_base) + log(δ_s))

           The age aggregation happens here using index_mask_logsumexp:
           - Sums over fine ages b ∈ [b_l, b_u) within each coarse group
           - Uses aid_exp and bid_pad for indexing
           - Applies stratum-specific adjustment (X_ids_exp)

        6. **Total log contact intensity**:
           log(cint) = Σ_s contribution_s (sum over stratification variables)

        7. **Repeat interview effect**: η ~ Hill(max_rid) if repeat data present

        8. **Expected contacts**:
           μ = exp(log(cint) + log(N) + log(S) + η)

        9. **Observation likelihood**:
           - Poisson: y ~ Poisson(μ)
           - Negative Binomial: y ~ NegativeBinomial2(μ, concentration=1/φ)
             where φ ~ Exponential(1)

        Parameters
        ----------
        y : ArrayLike, optional
            Observed contact counts. If None, samples from prior predictive.
            During inference, set to actual observations.

        Notes
        -----
        - All computations in log space for numerical stability
        - index_mask_logsumexp handles age aggregation with masking (-1 → -inf)
        - Stratum contributions are additive in log space (multiplicative in counts)
        - Rate consistency enforced via log(P) term in baseline

        Examples
        --------
        >>> # Inspect model structure
        >>> model.print_model_shape()
        >>>
        >>> # Sample from prior predictive
        >>> from numpyro.infer import Predictive
        >>> prior_pred = Predictive(model.model, num_samples=100)
        >>> prior_samples = prior_pred(PRNGKey(0))

        See Also
        --------
        BRCrefine.model : Base class model without hierarchical structure
        HiBRCfine.model : Hierarchical model without age aggregation
        index_mask_logsumexp : Age aggregation function
        """
        # 1. Global baseline intercept
        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))

        # 2. Shared smooth baseline function
        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        # 3. Log contact rate (fine-age resolution)
        log_rate = numpyro.deterministic("log_rate", beta0 + f)

        # 4. Baseline contact intensity with population structure
        log_cint_base = log_rate + self.log_P

        # 5. Initialize total contact intensity (will accumulate stratum contributions)
        log_cint = jnp.zeros(self.y.shape[0])

        # 6. Add stratum-specific contributions with age aggregation
        for var in self.X_vars:
            with scope(prefix=var):
                # Sample stratum-specific deviations centered on population
                log_delta = self.sample_log_delta(var)

                # Age aggregation: sum over coarse contact age groups with stratum indexing
                # - aid_exp: expanded participant ages (n_obs, max_int_length)
                # - bid_pad: padded contact ages (n_obs, max_int_length)
                # - X_ids_exp[var]: expanded stratum IDs (n_obs, max_int_length)
                # - log_cint_base: baseline intensity (1, A)
                # - log_delta: stratum adjustments (n_strata, A, 1 or A, A)
                #
                # index_mask_logsumexp computes:
                #   logsumexp(log_cint_base[aid_exp, bid_pad] +
                #             log_delta[X_ids_exp, aid_exp, bid_pad])
                # where invalid entries (bid_pad == -1) are masked with -inf
                contribution = index_mask_logsumexp(
                    log_cint_base + log_delta,
                    self.aid_exp,
                    self.bid_pad,
                    self.X_ids_exp[var],
                )

                # Accumulate contribution (additive in log space)
                log_cint += contribution

        # 7. Optional repeat interview effect
        repeat_effect = self.hill.sample()[self.rid] if hasattr(self, "rid") else 0.0

        # 8. Expected number of contacts
        mu = jnp.exp(
            log_cint  # Aggregated contact intensity
            + self.log_N  # Sample size adjustment
            + self.log_S  # Setting offset
            + repeat_effect  # Repeat interview correction
        )

        # 9. Observation likelihood
        if self.likelihood == "poisson":
            with plate("data", len(self.y)):
                numpyro.sample("obs", dist.Poisson(rate=mu), obs=y)

        if self.likelihood == "negbin":
            # Overdispersion parameter (smaller = more overdispersion)
            inv_disp = numpyro.sample("inv_disp", dist.Exponential(1.0))
            with plate("data", len(self.y)):
                numpyro.sample(
                    "obs",
                    dist.NegativeBinomial2(mean=mu, concentration=1.0 / inv_disp),
                    obs=y,
                )
