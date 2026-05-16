from typing import Any, Dict, Optional

import jax.numpy as jnp

from ..dataloader import ContactSurveyLoader
from ._GenMix import GenMix
from .numpyro import AgeMixFFNumPyroMixin
from .numpyro.priors import Hill


class AgeMixFF(AgeMixFFNumPyroMixin, GenMix):
    """
    Age-only mixing model with fine-age resolution for both participant and contact.

    AgeMixFF (Age Mixing, Fine-Fine) estimates social contact matrices at
    single-year age resolution using contact survey data where both participant
    and contact ages are recorded at 1-year resolution. It uses smooth priors
    (e.g., B-splines, Gaussian processes) to regularize the high-dimensional
    contact rate estimation problem.

    The model assumes:
    1. Contact rates are smooth functions of participant and contact ages
    2. Rate consistency: forward and reciprocal contact rates are balanced by population
    3. Observation model: Contacts follow Poisson or Negative Binomial distribution

    Mathematical Model
    ------------------
    For each observed contact y_i between age a_i and age b_i:

        log(rate[a, b]) = β₀ + f(a, b)
        log(contact_intensity[a, b]) = log(rate[a, b]) + log(P[b])
        μ_i = exp(log(contact_intensity[a_i, b_i]) + log(N_i) + log(S_i) + η_i)
        y_i ~ Poisson(μ_i) or NegativeBinomial2(μ_i, φ)

    where:
    - β₀: baseline contact rate
    - f(a, b): smooth 2D age-age function (from prior)
    - P[b]: population proportion in age group b
    - N_i: survey sample size for observation i
    - S_i: additional offset (e.g., for different settings)
    - η_i: repeat interview effect (if applicable)
    - φ: overdispersion parameter (negative binomial only)

    Parameters
    ----------
    dataloader : ContactSurveyLoader
        ContactSurveyLoader object containing processed contact data with columns:
        - y: observed contact counts
        - aid: participant age indices (1-year resolution)
        - bid: contact age indices (1-year resolution)
        - log_N: log of survey sample sizes
        - log_P: log of population age distribution
        - log_V: log of setting-specific offsets (optional)
        - rid: repeat interview indicators (optional)
    priors : dict
        Dictionary of prior specifications. Must contain:
        - 'rate': Prior2D object for the age-age contact rate function
          (e.g., Spline2D, PSpline2D, HSGP2D, IGMRF2D)
    likelihood : str, default='negbin'
        Observation likelihood:
        - 'negbin': Negative binomial (recommended for overdispersed count data)
        - 'poisson': Poisson (assumes mean = variance)
    inv_odist : float, default=1.0
        Prior mean for inverse overdispersion parameter (negative binomial only).
        Smaller values allow more overdispersion. Actual value is sampled during inference.

    Attributes
    ----------
    y : jax.Array
        Observed contact counts, shape (n_obs,)
    aid : jax.Array
        Participant age indices (1-year resolution), shape (n_obs,)
    bid : jax.Array
        Contact age indices (1-year resolution), shape (n_obs,)
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
        If required data columns (aid, bid, log_N, log_P) are missing.

    References
    ----------
    Shozen Dan et al., "Estimating fine age structure and time trends in
    human contact patterns from coarse contact data: The Bayesian rate consistency model",
    PLoS Computational Biology. 2023

    Examples
    --------
    >>> from cntmosaic.dataloader import ContactSurveyLoader, CoordToColumns
    >>> from cntmosaic.models import AgeMixFF
    >>> from cntmosaic.models.numpyro.priors import Spline2D
    >>> from jax.random import PRNGKey
    >>>
    >>> # Set up dataloader
    >>> col_map = CoordToColumns(
    ...     age_part="age_part",
    ...     age_cnt="age_cnt",
    ...     age_pop="age",
    ...     P="P"
    ... )
    >>> dataloader = ContactSurveyLoader(df_part, df_cnt, df_age_dist, col_map=col_map)
    >>>
    >>> # Specify smooth prior for contact rates
    >>> priors = {
    ...     "rate": Spline2D(
    ...         prior_type="global",
    ...         M=30,  # 30 basis functions per dimension
    ...         degree=3  # cubic splines
    ...     )
    ... }
    >>>
    >>> # Initialize model
    >>> model = AgeMixFF(dataloader, priors, likelihood="negbin")
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
    AgeMixFC : Age-only mixing model with coarse contact age resolution
    GenMixFF : Generalised mixing model with fine-age resolution for both ages
    ContactSurveyLoader : Data preprocessing utilities
    Spline2D : B-spline prior for smooth contact rates
    """

    def __init__(
        self,
        dataloader: ContactSurveyLoader,
        priors: Dict[str, Any],
        likelihood: str = "negbin",
        inv_odist: float = 1.0,
        backend: Optional[Any] = None,
    ) -> None:
        """
        Initialize AgeMixFF model with fine-age resolution for both participant and contact.

        Parameters
        ----------
        dataloader : ContactSurveyLoader
            Preprocessed contact data with 1-year resolution for both ages.
        priors : Dict[str, Any]
            Prior specifications (must include 'rate').
        likelihood : str, default='negbin'
            Observation likelihood ('negbin' or 'poisson').
        inv_odist : float, default=1.0
            Prior mean for inverse overdispersion (negbin only).
        backend : InferenceBackend, optional
            Pluggable inference engine (default: NumPyroBackend).
        """
        self.inv_odist = inv_odist
        super().__init__(dataloader, priors, likelihood, backend=backend)

        # Convert to JAX arrays
        self.y = jnp.array(self.data.y)
        self.aid = jnp.array(self.data.aid, dtype=jnp.int32)
        self.bid = jnp.array(self.data.bid, dtype=jnp.int32)
        self.log_N = jnp.array(self.data.log_N)
        self.log_P = jnp.array(self.data.log_P)

        # Optional offset for different settings (e.g., home, work, school)
        self.log_V = (
            jnp.array(self.data.log_V)
            if self.data.log_V is not None
            else jnp.zeros_like(self.y)
        )

        # Optional repeat interview effect
        if self.data.rid is not None:
            self.rid = jnp.array(self.data.rid, dtype=jnp.int32)
            self.hill = Hill(max_value=int(self.data.rid.max()))

