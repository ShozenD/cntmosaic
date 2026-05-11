"""
Abstract base class for all Bayesian contact models in cntmosaic.

This module defines `ContactModel`, the common interface that every
Bayesian model must satisfy.  Concrete implementations live in the
sibling modules (_BRC, _Prem, _vdKassteele, …) and must not be
imported here to avoid circular imports.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

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

    Methods
    -------
    model(y=None)
        Define the NumPyro probabilistic model.
    run_inference_mcmc(prng_key, ...)
        Run full Bayesian inference via HMC/NUTS.
    run_inference_svi(prng_key, guide, ...)
        Run stochastic variational inference.
    posterior_predictive_mcmc(prng_key, ...)
        Generate posterior predictive samples from MCMC results.
    posterior_predictive_svi(prng_key, guide, ...)
        Generate posterior predictive samples from SVI results.
    """

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

        Raises
        ------
        NotImplementedError
            Raised by default if a subclass does not override this method.
        """
        raise NotImplementedError

    @abstractmethod
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

        Raises
        ------
        NotImplementedError
            Raised by default if a subclass does not override this method.
        """
        raise NotImplementedError

    @abstractmethod
    def run_inference_svi(
        self,
        prng_key: PRNGKey,
        guide: Callable,
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
        guide : Callable
            Variational guide (e.g. ``AutoNormal``, ``AutoLowRankMultivariateNormal``).
        num_steps : int, default=5_000
            Number of SVI optimisation steps.
        peak_lr : float, default=0.01
            Peak learning rate for the cosine-annealing schedule.
        **kwargs : Any
            Additional keyword arguments forwarded to the SVI runner.

        Raises
        ------
        NotImplementedError
            Raised by default if a subclass does not override this method.
        """
        raise NotImplementedError

    @abstractmethod
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

        Raises
        ------
        NotImplementedError
            Raised by default if a subclass does not override this method.
        AttributeError
            If ``run_inference_mcmc`` has not been called first.
        """
        raise NotImplementedError

    @abstractmethod
    def posterior_predictive_svi(
        self,
        prng_key: PRNGKey,
        guide: Callable,
        num_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Generate posterior predictive samples from SVI results.

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            Random number generator key.
        guide : Callable
            The guide function used during SVI inference.
        num_samples : int, default=1000
            Number of posterior predictive samples to draw.

        Returns
        -------
        Dict[str, jax.Array]
            Posterior predictive samples for all model variables.

        Raises
        ------
        NotImplementedError
            Raised by default if a subclass does not override this method.
        AttributeError
            If ``run_inference_svi`` has not been called first.
        """
        raise NotImplementedError
