from typing import Any, Dict, Optional

import jax.numpy as jnp

from ..dataloader import ContactSurveyLoader
from ..utils import AgeGroupSpecs
from ._GenMix import GenMix
from .numpyro import AgeMixCCNumPyroMixin
from .numpyro.priors import Hill, vdKassteele2D


class AgeMixCC(AgeMixCCNumPyroMixin, GenMix):
    """
    Age-only mixing model with coarse-age resolution for both participant and contact.

    AgeMixCC (Age Mixing, Coarse-Coarse) estimates social contact matrices at
    coarse age resolution using contact survey data where both participant
    and contact ages are recorded in coarse age groups. It uses GMRF priors to regularize
    the high-dimensional contact rate estimation problem.

    The model assumes:
    1. Neighboring contact rate cells are correlated (GMRF prior)
    2. Rate consistency: contact rates are symmetric in the age-age space, adjusted for population distribution
    3. Observation model: Contacts follow Poisson or Negative Binomial distribution

    Mathematical Model
    ------------------
    For each observed contact y_i between age c_i and age d_i:

        log(rate[c, d]) = β₀ + f(c, d)
        log(contact_intensity[c, d]) = log(rate[c, d]) + log(P[d])
        μ_i = exp(log(contact_intensity[c_i, d_i]) + log(N_i) + log(S_i) + η_i)
        y_i ~ Poisson(μ_i) or NegativeBinomial2(μ_i, φ)

    where:
    - β₀: baseline contact rate
    - f(c, d): 2D age-age function (from prior)
    - P[d]: population proportion in age group d
    - N_i: survey sample size for observation i
    - S_i: additional offset (e.g., for different settings)
    - η_i: repeat interview effect (if applicable)
    - φ: overdispersion parameter (negative binomial only)

    Parameters
    ----------
    dataloader : ContactSurveyLoader
        ContactSurveyLoader object containing processed contact data with columns:
        - y: observed contact counts
        - cid: participant age indices (coarse age groups)
        - did: contact age indices (coarse age groups)
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
    cid : jax.Array
        Participant age indices (coarse age groups), shape (n_obs,)
    did : jax.Array
        Contact age indices (coarse age groups), shape (n_obs,)
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
        If required data columns (cid, did, log_N, log_P) are missing.

    References
    ----------
    Shozen Dan et al., "Estimating fine age structure and time trends in
    human contact patterns from coarse contact data: The Bayesian rate consistency model",
    PLoS Computational Biology. 2023

    Examples
    --------
    >>> from cntmosaic.dataloader import ContactSurveyLoader, CoordToColumns
    >>> from cntmosaic.models import AgeMixCC
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
    >>> model = AgeMixCC(dataloader, priors, likelihood="negbin")
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
    >>> log_rate_posterior = samples['log_rate']  # shape (n_samples, C, D)

    See Also
    --------
    AgeMixFF : Age-only contact matrix model with fine contact age resolution
    AgeMixFC : Age-only contact matrix model with coarse contact age resolution
    GenMixFF : Generalised contact matrix model with fine-age resolution for both ages
    ContactSurveyLoader : Data preprocessing utilities
    vdKassteele : IGMRF based prior
    """

    # Default priors
    default_priors = {"rate": vdKassteele2D(prior_type="global")}

    def __init__(
        self,
        dataloader: ContactSurveyLoader,
        priors: Optional[Dict[str, Any]] = None,
        likelihood: str = "negbin",
        inv_odist: float = 1.0,
        backend: Optional[Any] = None,
        age_group_specs: Optional[AgeGroupSpecs] = None,
    ) -> None:
        """
        Initialize AgeMixCC model with coarse-age resolution for both participant and contact.

        Parameters
        ----------
        dataloader : ContactSurveyLoader
            Preprocessed contact data with coarse-age resolution for both ages.
        priors : Optional[Dict[str, Any]], default=None
            Prior specifications. If None, uses default_priors (vdKassteele2D global).
            Must contain 'rate' key with a Prior2D object.
        likelihood : str, default='negbin'
            Observation likelihood ('negbin' or 'poisson').
        inv_odist : float, default=1.0
            Prior mean for inverse overdispersion (negbin only).
        backend : InferenceBackend, optional
            Pluggable inference engine (default: NumPyroBackend).
        age_group_specs : AgeGroupSpecs, optional
            Age group specification object encoding bin boundaries. When provided,
            it is propagated into ContactSummary so that plot_mosaic_pixilated can
            be called directly on a summary without a separate AgeGroupSpecs argument.
        """
        # Merge user priors with defaults (user priors take precedence)
        effective_priors = self.default_priors.copy()
        if priors is not None:
            effective_priors.update(priors)

        self.inv_odist = inv_odist
        self.age_group_specs = age_group_specs
        super().__init__(dataloader, effective_priors, likelihood, backend=backend)

        # Convert to JAX arrays
        self.y = jnp.array(self.data.y)
        self.cid = jnp.array(self.data.cid, dtype=jnp.int32)
        self.did = jnp.array(self.data.did, dtype=jnp.int32)
        self.log_N = jnp.array(self.data.log_N)
        self.log_P = jnp.array(self.data.log_P)

        # B: number of coarse age groups (participant, contact, and population)
        # The parent sets A from the fine-age range; re-configure priors for the
        # coarse grid so that sampled f has shape (B, B) rather than (A, A).
        self.B = int(self.log_P.shape[-1])
        for prior in self.priors.values():
            prior.set_age_bounds(0, self.B - 1)

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
