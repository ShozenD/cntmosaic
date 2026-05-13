import warnings
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from jax.typing import ArrayLike
from numpyro.handlers import plate, seed, trace
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value

from .._types import StratMode
from ..dataloader import DataLoader
from ._base import ContactModel
from ._numpyro import (
    posterior_predictive_mcmc,
    posterior_predictive_svi,
    run_inference_mcmc,
    run_inference_svi,
)
from .priors import Hill, vdKassteele2D


class vdKassteele(ContactModel):
    """
    van de Kassteele model for estimating social contact matrices.

    This class implement the van de Kassteele model which is used for inferring
    social contact matrices based on age-specific contact data.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        likelihood: str,
        order: int = 2,
        tau_shape: float = 2.0,
        tau_rate: float = 0.1,
        prior: Optional[vdKassteele2D] = None,
    ) -> None:
        """
        Initialise the van de Kassteele model.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader object containing the processed contact data.
        likelihood : str
            Observation likelihood.  Either ``'poisson'`` or ``'negbin'``.
        order : int, default=2
            P-spline order used by the vdKassteele2D prior.
        tau_shape : float, default=2.0
            Shape parameter of the Gamma prior on the smoothing precision ``tau``.
        tau_rate : float, default=0.1
            Rate parameter of the Gamma prior on the smoothing precision ``tau``.
        prior : vdKassteele2D, optional
            Pre-constructed ``vdKassteele2D`` prior object.  When supplied the
            model uses it directly and skips automatic prior construction from
            ``order``, ``tau_shape``, and ``tau_rate``.  This mirrors the
            ``priors`` argument of ``BRC`` and allows external prior sharing or
            customisation without sub-classing.  Backward compatibility is
            fully preserved: omitting this argument keeps the original behaviour.
        """
        self.data = dataloader.load()
        self.likelihood = likelihood

        # ================
        # Prior parameters
        # ================
        self.order = order
        self.tau_shape = tau_shape
        self.tau_rate = tau_rate

        self.age_min = self.data.age_min
        self.age_max = self.data.age_max
        self.A = int(self.age_max - self.age_min + 1)
        self.aid = jnp.array(self.data.aid, dtype=jnp.int8)
        self.bid = jnp.array(self.data.bid, dtype=jnp.int8)
        self.y = jnp.array(self.data.y)
        self.log_N = jnp.array(self.data.log_N)
        # Handle log_P shape: add newaxis only if 1D (unstratified case)
        log_P_raw = self.data.log_P
        if log_P_raw.ndim == 1:
            self.log_P = jnp.array(log_P_raw[jnp.newaxis, :])
        else:
            self.log_P = jnp.array(log_P_raw)

        # log_S is an optional field; default to zeros if not provided by the loader
        self.log_S = (
            jnp.array(self.data.log_S)
            if self.data.log_S is not None
            else jnp.zeros_like(self.y)
        )

        # Initialize optional attributes
        self.rid: Optional[ArrayLike] = None
        self.hill: Optional[Hill] = None

        # Optional repeat interview effect
        if self.data.rid is not None:
            self.rid = jnp.array(self.data.rid, dtype=jnp.int8)
            self.hill = Hill(max_value=int(self.data.rid.max()))

        self.prior_type: str = None
        self.prior: vdKassteele2D = None

        self._mcmc_result: Optional[numpyro.infer.MCMC] = None
        self._svi_result: Optional[numpyro.infer.SVI] = None
        self._guide: Optional[Callable] = None

        self._set_prior(prior)

    def _infer_prior_type(self) -> None:
        """
        Infers which type of prior to use amongst global, partial, and full

        Note: This method is used within _set_prior
        """
        if not self.data.is_stratified:
            prior_type = "global"
        else:
            modes = list(self.data.strat_modes.values())

            # If mixed type stratification
            if StratMode.PARTIAL in modes and StratMode.FULL in modes:
                # vdKassteele can only handle this pattern as partial
                prior_type = "partial"
            elif StratMode.FULL in modes:
                prior_type = "full"
            else:
                prior_type = "partial"

        self.prior_type = prior_type

    def _set_prior(self, prior: Optional[vdKassteele2D] = None) -> None:
        """
        Configure the vdKassteele2D prior for this model instance.

        If a pre-constructed *prior* object is supplied it is used directly
        (bypassing automatic construction).  Otherwise the prior is built
        from the ``order``, ``tau_shape``, and ``tau_rate`` parameters that
        were passed to ``__init__``, using the stratification mode inferred
        from the loaded dataset.

        Parameters
        ----------
        prior : vdKassteele2D, optional
            A fully-configured ``vdKassteele2D`` prior.  When provided, all
            automatic construction logic is skipped.
        """
        if prior is not None:
            self.prior = prior
            # Infer prior_type for use in model() (needed even when prior is external)
            self._infer_prior_type()
            return

        self._infer_prior_type()

        if self.prior_type == "global":
            self.prior = vdKassteele2D(
                prior_type="global",
                order=self.order,
                tau_shape=self.tau_shape,
                tau_rate=self.tau_rate,
            )
            self.prior.set_age_bounds(self.age_min, self.age_max)
            self.prior.set_event_dim(1)
            self.prior.set_loc(0.0)
        elif self.prior_type != "full":
            modes = list(self.data.strat_modes.values())
            dims = list(self.data.strat_dims.values())

            # Calculate total number of strata (product of all dimensions)
            # For example: gender (2) × setting (4) = 8 strata
            total_dims = 1
            for mode, dim in zip(modes, dims):
                if mode == StratMode.PARTIAL:
                    total_dims *= dim
                else:
                    # For FULL mode, dim is already the squared count (e.g., 4 for 2x2)
                    total_dims *= int(np.sqrt(dim))

            self.prior = vdKassteele2D(
                prior_type="partial",
                order=self.order,
                tau_shape=self.tau_shape,
                tau_rate=self.tau_rate,
            )
            self.prior.set_age_bounds(self.age_min, self.age_max)
            self.prior.set_event_dim(int(total_dims))
            self.prior.set_loc(total_dims)
        else:
            # For FULL mode with multiple variables, compute product of sqrt(dims)
            dims = np.asarray(list(self.data.strat_dims.values()))
            total_dims = int(np.prod(np.sqrt(dims)))
            self.prior = vdKassteele2D(
                prior_type="full",
                order=self.order,
                tau_shape=self.tau_shape,
                tau_rate=self.tau_rate,
            )
            self.prior.set_age_bounds(self.age_min, self.age_max)
            self.prior.set_event_dim(total_dims)
            self.prior.set_loc(0.0)

    def model(self, y: Optional[ArrayLike] = None) -> None:
        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))
        f = self.prior.sample()
        log_rate = numpyro.deterministic("log_rate", beta0 + f)

        if self.prior_type == "global":
            log_cint = numpyro.deterministic("log_cint", log_rate + self.log_P)[
                self.aid, self.bid
            ]
        else:
            # Initialize log contact intensity with population adjustment
            log_cint = (
                log_rate[self.data.flat_ix, self.aid, self.bid]
                + self.log_P[self.data.flat_pixs, self.bid]
            )

        # Add repeat interview effect if present
        repeat_effect = self.hill.sample()[self.rid] if self.rid is not None else 0.0

        # Calculate Poisson and Negative Binomial mean
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

    def print_model_shape(self) -> None:
        """
        Print the shapes of model parameters and sample sites for debugging.

        This utility method traces through the model with a dummy random seed and
        displays the shapes of all parameter sites, sample sites, and plates. This
        is useful for:
        - Debugging model specification issues
        - Verifying that dimensions align correctly
        - Understanding the model structure before running inference

        Notes
        -----
        The model is traced with a fixed seed (PRNGKey(0)) and dummy inputs,
        so the actual values are not meaningful - only the shapes matter.

        Examples
        --------
        >>> model = BRCfine(dataloader, priors)
        >>> model.print_model_shape()
        Trace Shapes:
         Param Sites:
        Sample Sites:
        baseline dist     |
                  value   |
        rate/spline_coefs dist 900 |
                      value 900 |
        ...
        """
        tr = trace(seed(self.model, PRNGKey(0))).get_trace()
        print(numpyro.util.format_shapes(tr))

    def _log_mcmc_diagnostics(self) -> None:
        """Log MCMC diagnostics information."""
        if self._mcmc_result is None:
            return

        try:
            extra_fields = self._mcmc_result.get_extra_fields()
            # Access the diverging field correctly - it's a dict key, not tuple indexing
            if "diverging" in extra_fields:
                diverging = extra_fields["diverging"]
                # Sum across all chains (diverging is typically shape (num_chains, num_samples))
                n_divergent = int(jnp.sum(diverging))
                print(f"Number of divergent transitions: {n_divergent}")

                if n_divergent > 0:
                    warnings.warn(
                        f"Found {n_divergent} divergent transitions. "
                        "Consider increasing target_accept_prob or max_tree_depth.",
                        stacklevel=2,
                    )
            else:
                # Divergence tracking might not be available for all samplers
                print("Divergence information not available for this sampler.")

        except Exception as e:
            warnings.warn(f"Failed to compute MCMC diagnostics: {e}", stacklevel=2)

    def run_inference_mcmc(
        self,
        prng_key: PRNGKey,
        num_samples: int = 500,
        num_warmup: int = 500,
        num_chains: int = 2,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Run full Bayesian inference using Hamiltonian Monte Carlo with the NUTS sampler.

        This method performs Markov Chain Monte Carlo (MCMC) sampling to obtain samples
        from the posterior distribution of model parameters. It uses the No-U-Turn Sampler
        (NUTS), an adaptive variant of Hamiltonian Monte Carlo that automatically tunes
        step sizes and trajectory lengths.

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            JAX random number generator key for reproducibility. Generate with
            `jax.random.PRNGKey(seed)`.
        num_samples : int, default=500
            Number of posterior samples to draw after warmup for each chain.
        num_warmup : int, default=500
            Number of warmup (burn-in) steps for adaptation and tuning.
            These samples are discarded.
        num_chains : int, default=2
            Number of independent Markov chains to run in parallel.
            Multiple chains help assess convergence (e.g., via R-hat statistics).
        target_accept_prob : float, default=0.8
            Target acceptance probability for the NUTS sampler. Higher values
            (e.g., 0.9-0.95) lead to smaller step sizes and fewer divergences
            but slower sampling. Typical range: [0.6, 0.99].
        max_tree_depth : int, default=10
            Maximum tree depth for NUTS trajectory. Controls the maximum number
            of leapfrog steps (2^max_tree_depth). Increase if you see many
            "maximum tree depth" warnings.
        **kwargs : Any
            Additional keyword arguments passed to `numpyro.infer.MCMC`.
            Common options include:
            - `progress_bar`: bool, show progress during sampling
            - `chain_method`: str, parallelization method ('parallel', 'sequential', 'vectorized')

        Raises
        ------
        RuntimeError
            If MCMC inference fails due to numerical issues or other errors.
        AttributeError
            If model.y has not been set (observation data missing).

        Notes
        -----
        - After successful inference, results are stored in `self._mcmc_result`
        - Access samples via `self._mcmc_result.get_samples()`
        - Diagnostics are automatically logged, including divergent transitions
        - Divergent transitions indicate potential sampling problems; consider:
          * Increasing target_accept_prob
          * Increasing max_tree_depth
          * Reparameterizing the model
          * Using more informative priors

        Examples
        --------
        >>> from jax.random import PRNGKey
        >>> model = BRCfine(dataloader, priors, likelihood="negbin")
        >>>
        >>> # Basic usage
        >>> model.run_inference_mcmc(PRNGKey(42), num_samples=1000, num_warmup=1000)
        >>>
        >>> # More conservative sampling to reduce divergences
        >>> model.run_inference_mcmc(
        ...     PRNGKey(42),
        ...     num_samples=1000,
        ...     num_warmup=1000,
        ...     num_chains=4,
        ...     target_accept_prob=0.95,
        ...     max_tree_depth=12
        ... )
        >>>
        >>> # Access samples
        >>> samples = model._mcmc_result.get_samples()
        >>> print(samples.keys())
        >>> print(samples['baseline'].shape)  # (num_chains * num_samples,)

        See Also
        --------
        run_inference_svi : Faster variational inference alternative
        posterior_predictive_mcmc : Generate predictive samples from MCMC results
        """
        if self.y is None:
            raise AttributeError(
                "Observation data (self.y) has not been set. "
                "Ensure the model is properly initialized."
            )

        try:
            self._mcmc_result = run_inference_mcmc(
                prng_key,
                self.model,
                num_samples=num_samples,
                num_warmup=num_warmup,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                y=self.y,
                **kwargs,
            )

            # Log diagnostics
            self._log_mcmc_diagnostics()

        except Exception as e:
            raise RuntimeError(f"MCMC inference failed: {e}") from e

    def run_inference_svi(
        self,
        prng_key: PRNGKey,
        guide: Optional[Callable] = None,
        num_steps: int = 5_000,
        peak_lr: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """
        Run stochastic variational inference for fast approximate posterior estimation.

        Stochastic Variational Inference (SVI) provides a scalable alternative to MCMC
        by optimizing a variational approximation to the posterior distribution. This
        method is faster than MCMC but provides an approximation rather than exact samples.

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            JAX random number generator key for reproducibility.
        guide : Callable
            The guide (variational family) function that defines the approximating
            distribution. Common choices:
            - `AutoNormal`: Multivariate normal with diagonal covariance
            - `AutoMultivariateNormal`: Full covariance multivariate normal
            - `AutoLowRankMultivariateNormal`: Low-rank approximation
            - Custom guide functions for more control
        num_steps : int, default=5_000
            Number of optimization steps. More steps generally improve approximation
            but take longer. Monitor ELBO convergence to determine sufficiency.
        peak_lr : float, default=0.01
            Peak learning rate for the optimizer. Uses a cosine annealing schedule
            that starts and ends at peak_lr/10, reaching peak_lr at the midpoint.
            Typical range: [0.001, 0.1].
        **kwargs : Any
            Additional keyword arguments passed to `run_inference_svi`.
            Options include:
            - Additional model arguments

        Raises
        ------
        RuntimeError
            If SVI inference fails due to numerical issues or other errors.
        AttributeError
            If model.y has not been set (observation data missing).

        Notes
        -----
        - After successful inference, results are stored in `self._svi_result`
        - Access optimized parameters via `self._svi_result.params`
        - The Evidence Lower BOund (ELBO) loss history is in `self._svi_result`
        - SVI is generally 10-100x faster than MCMC but less accurate
        - For complex posteriors, MCMC may be more reliable

        Examples
        --------
        >>> from jax.random import PRNGKey
        >>> from numpyro.infer.autoguide import AutoNormal
        >>> model = BRCfine(dataloader, priors, likelihood="poisson")
        >>>
        >>> # Basic usage with AutoNormal guide
        >>> guide = AutoNormal(model.model)
        >>> model.run_inference_svi(PRNGKey(42), guide, num_steps=10_000)
        >>>
        >>> # Adjust learning rate for better convergence
        >>> model.run_inference_svi(
        ...     PRNGKey(42),
        ...     guide,
        ...     num_steps=20_000,
        ...     peak_lr=0.05
        ... )
        >>>
        >>> # Generate posterior predictive samples
        >>> pred_samples = model.posterior_predictive_svi(
        ...     PRNGKey(123),
        ...     guide,
        ...     num_samples=1000
        ... )

        See Also
        --------
        run_inference_mcmc : Exact MCMC inference (slower but more accurate)
        posterior_predictive_svi : Generate predictions from SVI results
        """
        if self.y is None:
            raise AttributeError(
                "Observation data (self.y) has not been set. "
                "Ensure the model is properly initialized."
            )

        if guide is None:
            init_value = {"baseline": -self.log_P.mean()}
            self._guide = AutoNormal(
                self.model, init_loc_fn=init_to_value(values=init_value)
            )
        else:
            self._guide = guide

        try:
            self._svi_result = run_inference_svi(
                prng_key,
                self.model,
                guide,
                num_steps=num_steps,
                peak_lr=peak_lr,
                y=self.y,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"SVI inference failed: {e}") from e

    def posterior_predictive_svi(
        self,
        prng_key: PRNGKey,
        guide: Callable,
        num_samples: int = 1000,
    ) -> Dict[str, ArrayLike]:
        """
        Generate posterior predictive samples using SVI results.

        Generates samples from the posterior predictive distribution by:
        1. Sampling parameters from the variational approximation (guide)
        2. Running the forward model to generate predicted observations

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            Random number generator key for reproducibility.
        guide : Callable
            The guide function used during SVI inference. Should be the same
            guide passed to `run_inference_svi`.
        num_samples : int, default=1000
            Number of posterior predictive samples to generate.

        Returns
        -------
        Dict[str, jax.Array]
            Dictionary containing posterior predictive samples for all model
            variables, including:
            - 'obs': Predicted observations with shape (num_samples, len(y))
            - Other latent variables and deterministic quantities

        Raises
        ------
        AttributeError
            If `run_inference_svi` has not been called first.

        Notes
        -----
        - Posterior predictive checks assess model fit by comparing observed
          data to predictions from the fitted model
        - Use these samples to:
          * Check if the model can reproduce key features of the data
          * Identify systematic discrepancies
          * Assess predictive performance

        Examples
        --------
        >>> from jax.random import PRNGKey
        >>> from numpyro.infer.autoguide import AutoNormal
        >>>
        >>> # After running SVI inference
        >>> guide = AutoNormal(model.model)
        >>> model.run_inference_svi(PRNGKey(42), guide, num_steps=5000)
        >>>
        >>> # Generate posterior predictive samples
        >>> pred_samples = model.posterior_predictive_svi(
        ...     PRNGKey(123),
        ...     guide,
        ...     num_samples=1000
        ... )
        >>>
        >>> # Posterior predictive checks
        >>> import numpy as np
        >>> y_pred_mean = np.mean(pred_samples['obs'], axis=0)
        >>> y_observed = model.y
        >>> # Compare predicted vs observed

        See Also
        --------
        run_inference_svi : Run SVI inference first
        posterior_predictive_mcmc : Generate predictions from MCMC results
        """
        if self._svi_result is None:
            raise AttributeError(
                "SVI inference has not been run. Call run_inference_svi() first."
            )

        return posterior_predictive_svi(
            prng_key,
            self.model,
            guide,
            self._svi_result.params,
            num_samples=num_samples,
        )

    def posterior_predictive_mcmc(
        self,
        prng_key: PRNGKey,
        num_samples: int = 5_000,
    ) -> Dict[str, ArrayLike]:
        """
        Generate posterior predictive samples using MCMC results.

        Generates samples from the posterior predictive distribution by:
        1. Using parameter samples from MCMC posterior
        2. Running the forward model to generate predicted observations

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            Random number generator key for reproducibility.
        num_samples : int, default=5_000
            Number of posterior predictive samples to generate. If this exceeds
            the number of MCMC posterior samples, samples will be reused.

        Returns
        -------
        Dict[str, jax.Array]
            Dictionary containing posterior predictive samples for all model
            variables, including:
            - 'obs': Predicted observations with shape (num_samples, len(y))
            - Other latent variables and deterministic quantities

        Raises
        ------
        AttributeError
            If `run_inference_mcmc` has not been called first.

        Notes
        -----
        - Posterior predictive samples incorporate full posterior uncertainty
        - Each predicted observation uses a different parameter draw from the posterior
        - Use these samples for:
          * Posterior predictive checks (comparing data to model predictions)
          * Out-of-sample prediction
          * Model comparison via predictive performance

        Examples
        --------
        >>> from jax.random import PRNGKey
        >>>
        >>> # After running MCMC inference
        >>> model.run_inference_mcmc(PRNGKey(42), num_samples=1000)
        >>>
        >>> # Generate posterior predictive samples
        >>> pred_samples = model.posterior_predictive_mcmc(
        ...     PRNGKey(123),
        ...     num_samples=2000
        ... )
        >>>
        >>> # Posterior predictive checks
        >>> import numpy as np
        >>> y_pred = pred_samples['obs']
        >>> print(y_pred.shape)  # (2000, len(model.y))
        >>>
        >>> # Compare quantiles of predictions vs observations
        >>> y_pred_quantiles = np.quantile(y_pred, [0.025, 0.5, 0.975], axis=0)
        >>> # Check if observations fall within credible intervals

        See Also
        --------
        run_inference_mcmc : Run MCMC inference first
        posterior_predictive_svi : Generate predictions from SVI results
        """
        if self._mcmc_result is None:
            raise AttributeError(
                "MCMC inference has not been run. Call run_inference_mcmc() first."
            )

        return posterior_predictive_mcmc(
            prng_key,
            self.model,
            self._mcmc_result,  # Pass MCMC object, not samples dict
            num_samples=num_samples,
        )
