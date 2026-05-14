"""
Abstract base class for all Bayesian contact models in cntmosaic.

This module defines `ContactModel`, the common interface that every
Bayesian model must satisfy.  Concrete implementations live in the
sibling modules (_BRC, _Prem, _vdKassteele, …) and must not be
imported here to avoid circular imports.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Optional

from jax.random import PRNGKey


class ContactModel(ABC):
    """
    Abstract base class for Bayesian social contact models.

    All Bayesian contact models in cntmosaic inherit from this class.
    It declares the minimal interface for probabilistic model definition
    and inference that every concrete model must implement.

    The inheritance hierarchy for the built-in models is::

        ContactModel          ← this class (ABC)
        └── BRC (ABC)         ← Bayesian Rate Consistency base
            ├── BRCfine
            ├── BRCrefine
            ├── HiBRCfine
            └── HiBRCrefine
        └── Prem              ← Prem et al. (2017) model
        └── vdKassteele       ← van de Kassteele model

    Parameters
    ----------
    backend : InferenceBackend, optional
        Pluggable inference engine.  When ``None`` (default), a
        :class:`~cntmosaic.models.numpyro.NumPyroBackend` is constructed
        on first use via a lazy import so that ``import cntmosaic.models``
        does not require NumPyro to be installed.
    """

    def __init__(self, backend: Optional[Any] = None) -> None:
        self._backend: Optional[Any] = backend
        self._mcmc_result: Optional[Any] = None
        self._svi_result: Optional[Any] = None
        self._guide: Optional[Callable] = None
        self.y: Optional[Any] = None  # overridden by concrete subclasses

    # ------------------------------------------------------------------
    # Backend plumbing
    # ------------------------------------------------------------------

    def _get_backend(self) -> Any:
        """Return the inference backend, lazily creating a NumPyroBackend."""
        if self._backend is None:
            from .numpyro._backend import NumPyroBackend

            self._backend = NumPyroBackend()
        return self._backend

    def _make_guide(self, guide: Optional[Callable]) -> Callable:
        """Return the guide to use for SVI.

        Override in subclasses that need custom guide construction (e.g. BRC
        which requires an explicit guide and raises if one is not provided).
        """
        if guide is None:
            return self._get_backend()._build_default_guide(self.model)
        return guide

    # ------------------------------------------------------------------
    # Inference state helpers
    # ------------------------------------------------------------------

    @property
    def inference_method(self) -> Optional[Literal["mcmc", "svi"]]:
        """Return ``"mcmc"``, ``"svi"``, or ``None`` depending on which inference was run."""
        if self._mcmc_result is not None:
            return "mcmc"
        if self._svi_result is not None:
            return "svi"
        return None

    def draw_posterior_samples(
        self, prng_key: PRNGKey, num_samples: int = 1_000
    ) -> Dict[str, Any]:
        """Extract raw posterior sample dict, routing through the active backend.

        For MCMC delegates to ``backend.get_mcmc_samples``; for SVI delegates
        to ``posterior_predictive_svi`` (which in turn calls the backend).

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            Random key (used only for the SVI path).
        num_samples : int, default=1_000
            Number of samples to draw (used only for the SVI path).

        Returns
        -------
        Dict[str, Any]
            Raw posterior sample dict keyed by sample-site name.

        Raises
        ------
        ValueError
            If no inference has been run yet.
        """
        method = self.inference_method
        if method == "mcmc":
            return self._get_backend().get_mcmc_samples(self._mcmc_result)
        if method == "svi":
            return self.posterior_predictive_svi(prng_key, num_samples=num_samples)
        raise ValueError(
            "No inference has been run. "
            "Call run_inference_mcmc() or run_inference_svi() first."
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def model(self, y: Optional[Any] = None) -> None:
        """
        Define the NumPyro probabilistic model.

        Subclasses must implement this method to specify the full generative
        model, including prior sampling, deterministic transformations, and
        the observation likelihood.

        Parameters
        ----------
        y : array-like, optional
            Observed contact counts.  When ``None`` the model samples from
            the prior (prior predictive mode).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete inference methods (delegating to backend)
    # ------------------------------------------------------------------

    def print_model_shape(self) -> None:
        """Print all sample-site shapes via the backend's NumPyro trace."""
        self._get_backend().print_model_shape(self.model)

    def _log_mcmc_diagnostics(self) -> None:
        """Log divergence count from the most recent MCMC run."""
        if self._mcmc_result is None:
            return
        try:
            extra_fields = self._get_backend().get_mcmc_extra_fields(
                self._mcmc_result
            )
            if "diverging" in extra_fields:
                import jax.numpy as jnp

                n_divergent = int(jnp.sum(extra_fields["diverging"]))
                print(f"Number of divergent transitions: {n_divergent}")
                if n_divergent > 0:
                    warnings.warn(
                        f"Found {n_divergent} divergent transitions. "
                        "Consider increasing target_accept_prob or max_tree_depth.",
                        stacklevel=2,
                    )
            else:
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
        Run full Bayesian inference using HMC with the NUTS sampler.

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            JAX random number generator key for reproducibility.
        num_samples : int, default=500
            Number of posterior samples per chain after warmup.
        num_warmup : int, default=500
            Number of warmup / adaptation steps (discarded).
        num_chains : int, default=2
            Number of independent Markov chains.
        target_accept_prob : float, default=0.8
            Target acceptance probability for NUTS.
        max_tree_depth : int, default=10
            Maximum tree depth for NUTS trajectory.
        **kwargs : Any
            Additional keyword arguments forwarded to the MCMC runner.
        """
        if self.y is None:
            raise AttributeError(
                "Observation data (self.y) has not been set. "
                "Ensure the model is properly initialized."
            )
        try:
            self._mcmc_result = self._get_backend().run_mcmc(
                self.model,
                prng_key,
                num_samples=num_samples,
                num_warmup=num_warmup,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                y=self.y,
                **kwargs,
            )
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
        Run stochastic variational inference.

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            JAX random number generator key for reproducibility.
        guide : Callable, optional
            Variational guide.  When ``None``, ``_make_guide()`` constructs
            a default ``AutoNormal`` guide via the backend.  BRC-family models
            override ``_make_guide`` to raise if no guide is supplied.
        num_steps : int, default=5_000
            Number of SVI optimisation steps.
        peak_lr : float, default=0.01
            Peak learning rate for the cosine-annealing schedule.
        **kwargs : Any
            Additional keyword arguments forwarded to the SVI runner.
        """
        if self.y is None:
            raise AttributeError(
                "Observation data (self.y) has not been set. "
                "Ensure the model is properly initialized."
            )
        self._guide = self._make_guide(guide)
        try:
            self._svi_result = self._get_backend().run_svi(
                self.model,
                self._guide,
                prng_key,
                num_steps=num_steps,
                peak_lr=peak_lr,
                y=self.y,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"SVI inference failed: {e}") from e

    def posterior_predictive_mcmc(
        self,
        prng_key: PRNGKey,
        num_samples: int = 5_000,
    ) -> Dict[str, Any]:
        """
        Generate posterior predictive samples from MCMC results.

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            Random number generator key.
        num_samples : int, default=5_000
            Number of posterior predictive samples to draw.

        Returns
        -------
        Dict[str, jax.Array]
            Posterior predictive samples for all model variables.
        """
        if self._mcmc_result is None:
            raise AttributeError(
                "MCMC inference has not been run. Call run_inference_mcmc() first."
            )
        return self._get_backend().posterior_predictive_mcmc(
            prng_key, self.model, self._mcmc_result, num_samples=num_samples
        )

    def posterior_predictive_svi(
        self,
        prng_key: PRNGKey,
        guide: Optional[Callable] = None,
        num_samples: int = 1_000,
    ) -> Dict[str, Any]:
        """
        Generate posterior predictive samples from SVI results.

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            Random number generator key.
        guide : Callable, optional
            Guide used during inference.  Falls back to ``self._guide`` when
            ``None`` (the stored guide from the most recent SVI run).
        num_samples : int, default=1_000
            Number of posterior predictive samples to draw.

        Returns
        -------
        Dict[str, jax.Array]
            Posterior predictive samples for all model variables.
        """
        if self._svi_result is None:
            raise AttributeError(
                "SVI inference has not been run. Call run_inference_svi() first."
            )
        _guide = guide if guide is not None else self._guide
        return self._get_backend().posterior_predictive_svi(
            prng_key, self.model, _guide, self._svi_result, num_samples=num_samples
        )
