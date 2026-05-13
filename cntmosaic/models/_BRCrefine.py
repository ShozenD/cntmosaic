from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro import distributions as dist
from numpyro.handlers import plate, scope

from ..dataloader import DataLoader
from ._BRC import BRC
from ._utils import index_mask_logsumexp
from .priors import Hill, PSpline2D


class BRCrefine(BRC):
    """
    Bayesian Rate Consistency model for coarse-age contact data.

    This model estimates contact matrices at single-year age resolution from contact
    survey data where participant ages are fine-grained (single-year) but contact ages
    are reported in coarse age groups (e.g., 0-4, 5-9, etc.). The model uses smooth
    priors and an age aggregation mechanism to "refine" the coarse contact age data
    back to single-year resolution.

    The model assumes:
    1. Contact rates are smooth functions of participant and contact ages
    2. Rate consistency: forward and reciprocal contact rates are balanced by population
    3. Coarse age groups are aggregations of underlying fine-age contacts
    4. Observation model: Contacts follow Poisson or Negative Binomial distribution

    Mathematical Model
    ------------------
    For each observed contact y_i from participant age a_i to contact age group [b_l, b_u):

        log(rate[a, b]) = β₀ + f(a, b)  for all fine ages b
        log(contact_intensity[a, b]) = log(rate[a, b]) + log(P[b])
        log(μ_i) = logsumexp_{b ∈ [b_l, b_u)} log(cint[a_i, b]) + log(N_i) + log(S_i) + η_i
        y_i ~ Poisson(μ_i) or NegativeBinomial2(μ_i, φ)

    where:
    - β₀: baseline contact rate
    - f(a, b): smooth 2D age-age function (from prior)
    - P[b]: population proportion in age group b
    - N_i: survey sample size for observation i
    - S_i: additional offset (e.g., for different settings)
    - η_i: repeat interview effect (if applicable)
    - φ: overdispersion parameter (negative binomial only)
    - logsumexp: numerically stable log-sum-exp over age group

    The key difference from BRCfine is the age aggregation step: instead of using
    a single contact age index bid, we sum over all fine ages within the coarse
    age group using the index_mask_logsumexp function with aid_exp and bid_pad arrays.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing processed contact data with columns:
        - y: observed contact counts
        - aid: participant age indices (fine resolution)
        - aid_exp: expanded participant age indices for aggregation
        - bid_pad: padded contact age indices for coarse age groups
        - cid: coarse contact age group codes (categorical)
        - log_N: log of survey sample sizes
        - log_P: log of population age distribution
        - log_V: log of setting-specific offsets (optional)
        - rid: repeat interview indicators (optional)
    priors : dict, optional
        Dictionary of prior specifications. If None, uses default_priors.
        Must contain:
        - 'rate': Prior2D object for the age-age contact rate function
          (e.g., HSGP2D, Spline2D, PSpline2D, IGMRF2D)
    likelihood : str, default='negbin'
        Observation likelihood:
        - 'negbin': Negative binomial (recommended for overdispersed count data)
        - 'poisson': Poisson (assumes mean = variance)

    Attributes
    ----------
    default_priors : dict
        Class-level default prior specifications using HSGP2D with difference-age grid.
    y : jax.Array
        Observed contact counts, shape (n_obs,)
    aid : jax.Array
        Participant age indices (fine resolution), shape (n_obs,)
    aid_exp : jax.Array
        Expanded participant age indices for aggregation, shape (n_obs, max_int_length)
    bid_pad : jax.Array
        Padded contact age indices for coarse groups, shape (n_obs, max_int_length)
    log_N : jax.Array
        Log of sample sizes, shape (n_obs,)
    log_P : jax.Array
        Log of population age distribution, shape (1, A)
    log_V : jax.Array
        Log of offsets, shape (n_obs,)
    rid : jax.Array, optional
        Repeat interview indicators, shape (n_obs,)
    hill : Hill, optional
        Hill prior for repeat interview effects

    Raises
    ------
    ValueError
        If required data columns (aid_exp, bid_pad, log_N, log_P) are missing.

    Notes
    -----
    The age aggregation mechanism:
    - aid_exp: Repeats participant age indices to match the maximum coarse age group width
    - bid_pad: Contains all fine ages within each coarse group, padded with -1
    - index_mask_logsumexp: Sums over valid ages (non-negative) in log space
    - This allows the model to estimate fine-age contact rates while handling coarse data

    References
    ----------
    Shozen Dan et al., "Estimating fine age structure and time trends in
    human contact patterns from coarse contact data: The Bayesian rate consistency model",
    PLoS Computational Biology. (2023)

    Examples
    --------
    >>> from cntmosaic.dataloader import DataLoader, CoordToColumns
    >>> from cntmosaic.models import BRCrefine
    >>> from cntmosaic.models.priors import HSGP2D
    >>> from jax.random import PRNGKey
    >>> import pandas as pd
    >>>
    >>> # Set up dataloader with coarse contact ages
    >>> col_map = CoordToColumns(
    ...     age_part="age_part",
    ...     age_grp_cnt="age_grp_cnt",  # Note: coarse age groups
    ...     age_pop="age",
    ...     P="P"
    ... )
    >>> dataloader = DataLoader(df_part, df_cnt, df_age_dist, col_map=col_map)
    >>>
    >>> # Use default priors or specify custom ones
    >>> priors = {
    ...     "rate": HSGP2D(
    ...         grid_type="diff-age",
    ...         prior_type="global",
    ...         ell_alpha=2.0,
    ...         ell_beta=0.5
    ...     )
    ... }
    >>>
    >>> # Initialize model
    >>> model = BRCrefine(dataloader, priors, likelihood="negbin")
    >>>
    >>> # Run MCMC inference
    >>> model.run_inference_mcmc(
    ...     PRNGKey(42),
    ...     num_samples=1000,
    ...     num_warmup=1000,
    ...     num_chains=4
    ... )
    >>>
    >>> # Access posterior samples
    >>> samples = model._mcmc_result.get_samples()
    >>> baseline_posterior = samples['baseline']
    >>> log_rate_posterior = samples['log_rate']  # shape (n_samples, A, A)

    See Also
    --------
    BRCfine : Fine-grained age resolution model (both participant and contact ages fine)
    HiBRCrefine : Hierarchical BRC for multiple populations with coarse contact ages
    DataLoader : Data preprocessing utilities
    HSGP2D : Hilbert Space Gaussian Process prior (default for BRCrefine)
    """

    # Default priors
    default_priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global", M=15)
    }

    def __init__(
        self,
        dataloader: DataLoader,
        priors: Optional[Dict[str, Any]] = None,
        likelihood: str = "negbin",
    ) -> None:
        """
        Initialize BRCrefine model.

        Parameters
        ----------
        dataloader : DataLoader
            Preprocessed contact data with coarse contact age groups.
        priors : Optional[Dict[str, Any]], default=None
            Prior specifications. If None, uses default_priors.
            Must contain 'rate' key with a Prior2D object.
        likelihood : str, default='negbin'
            Observation likelihood ('negbin' or 'poisson').

        Notes
        -----
        The initialization process:
        1. Merges user-provided priors with default_priors
        2. Calls parent BRC.__init__ for common setup
        3. Validates BRCrefine-specific requirements
        4. Converts dataset arrays to JAX format for efficient computation
        5. Sets up optional repeat interview effects if present
        """
        # Merge user priors with defaults (user priors take precedence)
        effective_priors = self.default_priors.copy()
        if priors is not None:
            effective_priors.update(priors)

        # Initialize parent class with merged priors
        super().__init__(dataloader, effective_priors, likelihood)

        # Convert data to JAX arrays for efficient computation
        self.y = jnp.array(self.data.y)
        self.log_N = jnp.array(self.data.log_N)
        self.log_P = jnp.array(self.data.log_P)

        # Optional offset for different settings (e.g., home, work, school)
        self.log_V = (
            jnp.array(self.data.log_V)
            if self.data.log_V is not None
            else jnp.zeros_like(self.y)
        )

        # Age aggregation indices for coarse-to-fine refinement
        self.aid = jnp.array(self.data.aid, dtype=jnp.int32)
        self.aid_exp = jnp.array(self.data.aid_exp, dtype=jnp.int32)
        self.bid_pad = jnp.array(self.data.bid_pad, dtype=jnp.int32)

        # Optional repeat interview effect
        if self.data.rid is not None:
            self.rid = jnp.array(self.data.rid, dtype=jnp.int32)
            self.hill = Hill(max_value=int(self.data.rid.max()))

    def model(
        self,
        aid: Optional[ArrayLike] = None,
        aid_exp: Optional[ArrayLike] = None,
        bid_pad: Optional[ArrayLike] = None,
        rid: Optional[ArrayLike] = None,
        log_N: Optional[ArrayLike] = None,
        log_V: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
    ) -> None:
        """
        Define the generative model for contact matrix estimation with coarse contact ages.

        This method specifies the complete Bayesian model including priors,
        transformations, age aggregation, and likelihood. The key feature is the
        age aggregation step that refines coarse contact age groups to fine resolution.

        Model Structure
        ---------------
        1. **Baseline effect**: β₀ ~ Normal(0, 10²)
           - Global intercept for contact rates

        2. **Smooth age-age function**: f(a,b) from specified prior
           - Captures age-structured contact patterns
           - Prior type (HSGP2D, Spline2D, etc.) specified in self.priors['rate']

        3. **Log contact rate**: log(rate) = β₀ + f(a,b)
           - Deterministic transformation for fine-age contact rates

        4. **Log contact intensity**: log(cint) = log(rate) + log(P)
           - Incorporates population age distribution for rate consistency

        5. **Age aggregation**: Aggregate fine-age intensities over coarse groups
           - Uses index_mask_logsumexp to sum over contact ages within each coarse group
           - aid_exp: Expanded participant age indices (repeated to match group width)
           - bid_pad: Padded contact age indices (all ages in group, padded with -1)
           - This step "refines" coarse age data by estimating underlying fine contacts

        6. **Expected contacts**: μ = exp(aggregated_log_cint + log(N) + log(V) + η)
           - log(N): log sample size
           - log(V): log offset (ambiguous contact effects)
           - η: repeat interview effect (if applicable)

        7. **Observation likelihood**:
           - Poisson: y ~ Poisson(μ)
           - Negative Binomial: y ~ NegativeBinomial2(μ, concentration=1/φ)
             where φ ~ Exponential(1)

        Parameters
        ----------
        y : ArrayLike, optional
            Observed contact counts. If None, samples from the prior predictive.
            During inference, this should be set to the actual observations.

        Notes
        -----
        - All transformations use log space for numerical stability
        - The rate consistency constraint is enforced through the log(P) term
        - Age aggregation via index_mask_logsumexp handles variable-width age groups
        - Invalid ages (marked as -1 in bid_pad) are masked with -inf before logsumexp
        - For negative binomial, smaller φ (inv_disp) means more overdispersion

        Examples
        --------
        >>> # Trace model structure
        >>> model.print_model_shape()
        >>>
        >>> # Sample from prior predictive
        >>> from numpyro.infer import Predictive
        >>> from jax.random import PRNGKey
        >>> prior_pred = Predictive(model.model, num_samples=100)
        >>> prior_samples = prior_pred(PRNGKey(0))

        See Also
        --------
        index_mask_logsumexp : Core function for age aggregation
        BRCfine.model : Fine-age model without aggregation step
        """
        aid = self.aid if aid is None else aid
        aid_exp = self.aid_exp if aid_exp is None else aid_exp
        bid_pad = self.bid_pad if bid_pad is None else bid_pad
        log_N = self.log_N if log_N is None else log_N
        log_V = self.log_V if log_V is None else log_V
        rid = getattr(self, "rid", None) if rid is None else rid
        len_y = len(self.y) if y is None else len(y)

        # Baseline contact rate (global intercept)
        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))

        # Smooth age-age contact rate function from prior
        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        # Log contact rate: baseline + smooth function (fine-age resolution)
        log_rate = numpyro.deterministic("log_rate", beta0 + f)

        # Log contact intensity: rate + population effect (rate consistency)
        log_cint = numpyro.deterministic("log_cint", log_rate + self.log_P)

        # Repeat interview effect (if repeat interviews in data)
        repeat_effect = self.hill.sample()[rid] if hasattr(self, "rid") else 0.0

        # Age aggregation: Sum fine-age contact intensities over coarse age groups
        # This is the key step that "refines" coarse contact age data
        # - aid_exp contains repeated participant ages (one per fine age in contact group)
        # - bid_pad contains all fine ages within each coarse contact group (padded with -1)
        # - index_mask_logsumexp sums over valid ages in log space (masking -1 with -inf)
        aggregated_log_cint = index_mask_logsumexp(log_cint, aid_exp, bid_pad)

        # Expected number of contacts (combining all effects)
        mu = jnp.exp(
            aggregated_log_cint  # Aggregated contact intensity
            + repeat_effect  # Repeat interview correction
            + log_N  # Sample size adjustment
            + log_V  # Setting-specific offset
        )

        # Observation likelihood
        if self.likelihood == "poisson":
            with plate("data", len_y):
                numpyro.sample("obs", dist.Poisson(rate=mu), obs=y)

        if self.likelihood == "negbin":
            # Overdispersion parameter: smaller values = more overdispersion
            inv_disp = numpyro.sample("inv_disp", dist.Exponential(1.0))
            with plate("data", len_y):
                numpyro.sample(
                    "obs",
                    dist.NegativeBinomial2(mean=mu, concentration=1.0 / inv_disp),
                    obs=y,
                )
