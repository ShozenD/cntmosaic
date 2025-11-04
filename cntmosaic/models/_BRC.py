import warnings
from typing import Optional, Dict, Callable, Any

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

import jax
import jax.numpy as jnp
from jax import random
from jax.typing import ArrayLike
from jax.random import PRNGKey

import numpyro
from numpyro.handlers import seed, trace

from ..dataloader import DataLoader
from .priors import Prior2D
from ._numpyro import (
    run_inference_mcmc,
    run_inference_svi,
    posterior_predictive_mcmc,
    posterior_predictive_svi,
)


class BRC(ABC):
    """
    Base class for the Bayesian Rate Consistency model.

    The BRC model provides a Bayesian framework for estimating social contact matrices
    from contact survey data. It ensures consistency between forward and reciprocal
    contact rates by incorporating population age distribution constraints.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing the processed participant and contact data.
        Must be properly initialized with participant data, contact data, and
        population age distribution. See `DataLoader` for more details.
    priors : dict
        Dictionary containing prior specifications for model parameters.
        Must contain at least a 'rate' key specifying the prior for contact rates.
        Additional priors depend on the specific model subclass.
    likelihood : str, default='negbin'
        Likelihood function to use for the observation model.
        Options are:
            - 'negbin': Negative binomial distribution (handles overdispersion)
            - 'poisson': Poisson distribution (assumes equidispersion)

    Attributes
    ----------
    ds : xarray.Dataset
        Loaded dataset from the dataloader containing processed contact data.
    priors : dict
        Dictionary of prior specifications.
    likelihood : str
        Selected likelihood function.
    age_min : int
        Minimum age in the population (set via `set_age_dims`).
    age_max : int
        Maximum age in the population (set via `set_age_dims`).
    A : int
        Number of age groups (age_max - age_min + 1).
    y : ArrayLike
        Observed contact counts (set by subclasses).
    age_dist : NDArray, optional
        Population age distribution (set via `set_age_dist`).

    Raises
    ------
    ValueError
        If likelihood is not one of the allowed values.
        If priors is not a dictionary.
        If priors does not contain 'rate' specification.

    References
    ----------
    Shozen Dan et al., "Estimating fine age structure and time trends in
    human contact patterns from coarse contact data: The Bayesian rate consistency model",
    PLoS Computational Biology. 2023

    Examples
    --------
    >>> from cntmosaic.dataloader import DataLoader
    >>> from cntmosaic.models.priors import Spline2D
    >>> from cntmosaic.models import BRCfine
    >>>
    >>> # Create dataloader with contact data
    >>> dataloader = DataLoader(df_part, df_cnt, df_age_dist)
    >>>
    >>> # Specify priors
    >>> priors = {"rate": Spline2D(prior_type="global", M=30, degree=3)}
    >>>
    >>> # Initialize model
    >>> model = BRCfine(dataloader, priors, likelihood="negbin")
    >>>
    >>> # Run inference
    >>> from jax.random import PRNGKey
    >>> model.run_inference_mcmc(PRNGKey(0), num_samples=1000)

    See Also
    --------
    BRCfine : Fine-grained age resolution BRC model
    BRCrefine : Coarse-to-fine age refinement BRC model
    DataLoader : Data preprocessing and loading utilities
    """

    ALLOWED_LIKELIHOODS = ["negbin", "poisson"]

    def __init__(
        self, dataloader: DataLoader, priors: Dict[str, Any], likelihood: str = "negbin"
    ) -> None:
        self._validate_common_inputs(dataloader, priors, likelihood)

        self.ds: DataLoader = dataloader.load()
        self.priors: Dict[str, Prior2D] = priors
        self.likelihood: str = likelihood

        # Computed attributes
        self.age_dist: Optional[NDArray] = None
        self.age_min: Optional[int] = None
        self.age_max: Optional[int] = None
        self.A: Optional[int] = None
        self.y: Optional[ArrayLike] = None
        self._mcmc_result: Optional[numpyro.infer.MCMC] = None
        self._svi_result: Optional[numpyro.infer.SVI] = None
        self._guide: Optional[Callable] = None

        self.set_age_dims(int(self.ds.age.values.min()), int(self.ds.age.values.max()))

    def _validate_common_inputs(
        self, dataloader: DataLoader, priors: Dict[str, Any], likelihood: str
    ) -> None:
        """
        Validate inputs common to all BRC model variants.

        This method performs essential validation checks on the core model inputs:
        1. Ensures the likelihood function is supported
        2. Verifies priors is a dictionary
        3. Confirms that rate prior specification is present

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader object to validate.
        priors : dict
            Dictionary of prior specifications to validate.
        likelihood : str
            Likelihood function name to validate.

        Raises
        ------
        ValueError
            If likelihood is not in ALLOWED_LIKELIHOODS.
            If priors is not a dictionary.
            If 'rate' key is missing from priors.

        Notes
        -----
        Subclasses may implement additional validation via their own
        `_validate_inputs()` method for model-specific requirements.
        """
        if likelihood not in self.ALLOWED_LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of: {self.ALLOWED_LIKELIHOODS}, "
                f"got '{likelihood}'"
            )

        if not isinstance(priors, dict):
            raise ValueError("priors must be a dictionary")

        if "rate" not in priors.keys():
            raise ValueError("priors must contain the specifications for 'rate'")

    def set_age_dims(self, age_min: int, age_max: int) -> None:
        """
        Set the age dimensions of the model and configure priors accordingly.

        This method establishes the age range for the contact matrix and communicates
        these bounds to all prior objects. It computes the number of age groups (A)
        and ensures all priors are properly initialized for the specified age range.

        Parameters
        ----------
        age_min : int
            Minimum age in years (inclusive). Must be non-negative.
        age_max : int
            Maximum age in years (inclusive). Must be greater than age_min.

        Raises
        ------
        ValueError
            If age_min is negative.
            If age_max <= age_min.

        Notes
        -----
        The number of age groups is computed as A = age_max - age_min + 1.
        For example, if age_min=0 and age_max=80, then A=81 age groups.

        This method should be called automatically during initialization, but can
        be manually invoked to reconfigure the model for a different age range.

        Examples
        --------
        >>> model = BRCfine(dataloader, priors)
        >>> model.set_age_dims(0, 80)  # 81 age groups (0-80)
        >>> print(model.A)
        81
        """
        if age_min < 0:
            raise ValueError(f"age_min must be non-negative, got {age_min}")
        if age_max <= age_min:
            raise ValueError(
                f"age_max must be greater than age_min, got age_max={age_max}, "
                f"age_min={age_min}"
            )

        self.age_min = age_min
        self.age_max = age_max
        self.A = age_max - age_min + 1

        for prior in self.priors.values():
            prior.set_age_bounds(age_min, age_max)

    def set_age_dist(self, age_dist: NDArray) -> None:
        """
        Set the population age distribution for the model.

        The population age distribution is used to ensure rate consistency in the
        contact matrix estimation. It represents the proportion of the population
        in each age group and is typically obtained from census or demographic data.

        Parameters
        ----------
        age_dist : NDArray
            Population age distribution as a 1D array of length A, where A is the
            number of age groups. Should contain non-negative values that sum to 1.
            Element i represents the proportion of the population in age group i.

        Raises
        ------
        ValueError
            If age_dist length doesn't match the number of age groups (A).
            If age_dist contains negative values.
            If age_dist doesn't sum to approximately 1.0.

        Notes
        -----
        The age distribution is used in the model to weight contact rates and ensure
        demographic consistency in the estimated contact matrix. It's essential that
        the age_dist aligns with the age grouping defined by age_min and age_max.

        Examples
        --------
        >>> import numpy as np
        >>> # Uniform age distribution for 81 age groups
        >>> age_dist = np.ones(81) / 81
        >>> model.set_age_dist(age_dist)
        >>>
        >>> # Or load from demographic data
        >>> from cntmosaic.datasets import load_age_distribution
        >>> df_age = load_age_distribution("United_States")
        >>> model.set_age_dist(df_age.P.values)
        """
        if not hasattr(self, "A"):
            raise AttributeError("Age dimensions not set. Call set_age_dims() first.")

        age_dist = np.asarray(age_dist)

        if age_dist.ndim != 1:
            raise ValueError(
                f"age_dist must be 1-dimensional, got shape {age_dist.shape}"
            )

        if len(age_dist) != self.A:
            raise ValueError(
                f"age_dist length ({len(age_dist)}) must match number of age "
                f"groups (A={self.A})"
            )

        if np.any(age_dist < 0):
            raise ValueError("age_dist must contain only non-negative values")

        age_dist_sum = np.sum(age_dist)
        if not np.isclose(age_dist_sum, 1.0, rtol=1e-3):
            warnings.warn(
                f"age_dist sums to {age_dist_sum:.6f}, not 1.0. "
                "Consider normalizing the distribution.",
                stacklevel=2,
            )

        self.age_dist = age_dist

    @abstractmethod
    def model(self) -> None:
        """
        Define the probabilistic model for contact matrix estimation.

        This abstract method must be implemented by all subclasses to specify the
        generative model structure, including priors, likelihood, and any deterministic
        transformations. The model should be compatible with NumPyro's primitives.

        Raises
        ------
        NotImplementedError
            If a subclass doesn't implement this method.

        Notes
        -----
        Subclass implementations should:
        1. Sample from priors for all latent variables
        2. Apply necessary transformations to compute contact intensities
        3. Define the observation likelihood given the model parameters
        4. Use numpyro.sample() and numpyro.deterministic() appropriately

        Examples
        --------
        >>> def model(self, y=None):
        ...     beta0 = numpyro.sample("baseline", dist.Normal(0.0, 2.5))
        ...     with scope(prefix="rate"):
        ...         f = self.priors["rate"].sample()
        ...     # ... rest of model specification
        """
        raise NotImplementedError

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
        tr = trace(seed(self.model, random.PRNGKey(0))).get_trace()
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
        guide: Callable,
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
    ) -> Dict[str, jax.Array]:
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
    ) -> Dict[str, jax.Array]:
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

    def prior_sampler(
        self, para_name: str, num_samples: int = 1, seed: int = 0
    ) -> jax.Array:
        """
        Sample from the prior distribution of a specified parameter.

        This method is useful for:
        - Prior predictive checks: Assessing if priors are reasonable before seeing data
        - Visualization: Understanding the prior distribution's shape and range
        - Sensitivity analysis: Testing how prior choices affect inference
        - Debugging: Verifying prior specifications are correct

        Parameters
        ----------
        para_name : str
            Name of the parameter to sample from. Must be a key in `self.priors`.
            Common parameter names include 'rate', 'baseline', 'inv_disp', etc.
        num_samples : int, default=1
            Number of samples to draw from the prior distribution.
        seed : int, default=0
            Seed for random number generation to ensure reproducibility.

        Returns
        -------
        jax.Array
            Array of shape `(num_samples, ...)` containing the sampled values.
            The trailing dimensions depend on the parameter's shape in the model.

        Raises
        ------
        KeyError
            If `para_name` is not found in `self.priors`.
        ValueError
            If `num_samples` is not positive.

        Notes
        -----
        - This samples directly from the prior, ignoring any observed data
        - For compositional priors (e.g., with ALR/CLR/ILR transforms),
          samples are returned in the transformed space
        - The prior must have a `sample()` method compatible with JAX

        Examples
        --------
        >>> from cntmosaic.models.priors import Spline2D
        >>> priors = {"rate": Spline2D(prior_type="global", M=30, degree=3)}
        >>> model = BRCfine(dataloader, priors)
        >>>
        >>> # Sample from the rate prior
        >>> rate_samples = model.prior_sampler("rate", num_samples=100, seed=42)
        >>> print(rate_samples.shape)  # (100, A, A) where A is number of age groups
        >>>
        >>> # Visualize prior samples
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(rate_samples[0])  # Show first sample
        >>> plt.colorbar()
        >>> plt.title("Prior sample for contact rate matrix")

        See Also
        --------
        print_model_shape : Display model structure and parameter shapes
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        if para_name not in self.priors:
            raise KeyError(
                f"Parameter '{para_name}' not found in priors. "
                f"Available parameters: {list(self.priors.keys())}"
            )

        prng_key = jax.random.PRNGKey(seed)
        _, subkey = jax.random.split(prng_key)

        samples = self.priors[para_name].sample(subkey, sample_shape=(num_samples,))

        return samples
