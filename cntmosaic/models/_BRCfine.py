from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpyro
import pandas as pd
from jax.typing import ArrayLike
from numpy.typing import NDArray
from numpyro import distributions as dist
from numpyro.handlers import scope

from ..dataloader import DataLoader
from ._BRC import BRC
from .priors import Hill


class BRCfine(BRC):
    """
    Bayesian Rate Consistency model with fine-grained age resolution.

    This model estimates contact matrices at single-year age resolution using
    contact survey data. It uses smooth priors (e.g., B-splines, Gaussian processes)
    to regularize the high-dimensional contact rate estimation problem.

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
    dataloader : DataLoader
        DataLoader object containing processed contact data with columns:
        - y: observed contact counts
        - aid: participant age indices
        - bid: contact age indices
        - log_N: log of survey sample sizes
        - log_P: log of population age distribution
        - log_S: log of setting-specific offsets (optional)
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
        Participant age indices, shape (n_obs,)
    bid : jax.Array
        Contact age indices, shape (n_obs,)
    log_N : jax.Array
        Log of sample sizes, shape (n_obs,)
    log_P : jax.Array
        Log of population age distribution, shape (1, A)
    log_S : jax.Array
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
    >>> from cntmosaic.dataloader import DataLoader, CoordToColumns
    >>> from cntmosaic.models import BRCfine
    >>> from cntmosaic.models.priors import Spline2D
    >>> from jax.random import PRNGKey
    >>>
    >>> # Set up dataloader
    >>> col_map = CoordToColumns(
    ...     age_part="age_part",
    ...     age_cnt="age_cnt",
    ...     age_pop="age",
    ...     P="P"
    ... )
    >>> dataloader = DataLoader(df_part, df_cnt, df_age_dist, col_map=col_map)
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
    >>> model = BRCfine(dataloader, priors, likelihood="negbin")
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
    BRCrefine : Coarse-to-fine age refinement model
    HiBRCfine : Hierarchical BRC for multiple populations
    DataLoader : Data preprocessing utilities
    Spline2D : B-spline prior for smooth contact rates
    """

    def __init__(
        self,
        dataloader: DataLoader,
        priors: Dict[str, Any],
        likelihood: str = "negbin",
        inv_odist: float = 1.0,
    ) -> None:
        """
        Initialize BRCfine model with fine-grained age resolution.

        Parameters
        ----------
        dataloader : DataLoader
            Preprocessed contact data.
        priors : Dict[str, Any]
            Prior specifications (must include 'rate').
        likelihood : str, default='negbin'
            Observation likelihood ('negbin' or 'poisson').
        inv_odist : float, default=1.0
            Prior mean for inverse overdispersion (negbin only).
        """
        self.inv_odist = inv_odist
        super().__init__(dataloader, priors, likelihood)

        # Convert to JAX arrays
        self.y = jnp.array(self.data.base_data["y"])
        self.aid = jnp.array(self.data.base_data["aid"], dtype=jnp.int32)
        self.bid = jnp.array(self.data.base_data["bid"], dtype=jnp.int32)
        self.log_N = jnp.array(self.data.base_data["log_N"])
        self.log_P = jnp.array(self.data.base_data["log_P"][jnp.newaxis, :])

        # Optional offset for different settings (e.g., home, work, school)
        self.log_S = (
            jnp.array(self.data.base_data["log_S"])
            if "log_S" in self.data.base_data
            else jnp.zeros_like(self.y)
        )

        # Optional repeat interview effect
        if "rid" in self.data.base_data:
            self.rid = jnp.array(self.data.base_data["rid"], dtype=jnp.int32)
            self.hill = Hill(max_value=int(self.data.base_data["rid"].max()))

    def model(self, y: Optional[ArrayLike] = None) -> None:
        """
        Define the generative model for contact matrix estimation.

        This method specifies the complete Bayesian model including priors,
        transformations, and likelihood. It follows NumPyro's model specification
        conventions and can be used with both MCMC and SVI inference.

        Model Structure
        ---------------
        1. **Baseline effect**: β₀ ~ Normal(0, 2.5²)
           - Global intercept for contact rates

        2. **Smooth age-age function**: f(a,b) from specified prior
           - Captures age-structured contact patterns
           - Prior type (Spline2D, HSGP2D, etc.) specified in self.priors['rate']

        3. **Log contact rate**: log(rate) = β₀ + f(a,b)
           - Deterministic transformation

        4. **Log contact intensity**: log(cint) = log(rate) + log(P)
           - Incorporates population age distribution

        5. **Expected contacts**: μ = exp(log(cint[aid, bid]) + log(N) + log(S) + η)
           - aid, bid: age indices for each observation
           - log(N): log sample size
           - log(S): log offset (setting effects)
           - η: repeat interview effect (if applicable)

        6. **Observation likelihood**:
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
        - For negative binomial, smaller φ (inv_disp) means more overdispersion
        - Uses numpyro.deterministic for quantities we want to track in posterior

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
        """
        # Baseline contact rate (global intercept)
        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))

        # Smooth age-age contact rate function from prior
        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        # Log contact rate: baseline + smooth function
        log_rate = numpyro.deterministic("log_rate", beta0 + f)

        # Log contact intensity: rate + population effect
        log_cint = numpyro.deterministic("log_cint", log_rate + self.log_P)

        # Repeat interview effect (if repeat interviews in data)
        repeat_effect = self.hill.sample()[self.rid] if hasattr(self, "rid") else 0.0

        # Expected number of contacts
        mu = numpyro.deterministic(
            "mu",
            jnp.exp(
                log_cint[self.aid, self.bid] + self.log_N + self.log_S + repeat_effect
            ),
        )

        # Observation likelihood
        if self.likelihood == "poisson":
            with numpyro.plate("data", len(self.y)):
                numpyro.sample("obs", dist.Poisson(rate=mu), obs=y)

        elif self.likelihood == "negbin":
            # Overdispersion parameter: smaller values = more overdispersion
            inv_disp = numpyro.sample("inv_disp", dist.Exponential(1.0))
            with numpyro.plate("data", len(self.y)):
                numpyro.sample(
                    "obs",
                    dist.NegativeBinomial2(mean=mu, concentration=1.0 / inv_disp),
                    obs=y,
                )
