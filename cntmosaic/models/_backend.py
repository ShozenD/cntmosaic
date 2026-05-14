"""
InferenceBackend protocol for cntmosaic.

Defines the structural interface that any inference backend must satisfy.
The NumPyro implementation lives in ``cntmosaic.models.numpyro``.

Internal API — not exported from ``cntmosaic.models``. Consumed by
``ContactModel`` (item 1.7 Stage 4) and by ``NumPyroBackend``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """
    Structural protocol for a pluggable inference engine.

    Any class that implements all these methods can serve as a backend for
    ``ContactModel`` subclasses. The NumPyro implementation is
    ``cntmosaic.models.numpyro.NumPyroBackend``.

    To add a new backend (e.g., PyMC):
    1. Create ``cntmosaic/models/pymc/__init__.py``
    2. Implement a class satisfying this protocol — no import of this file
       is required; structural compatibility is checked at runtime via
       ``isinstance(backend, InferenceBackend)``.
    3. Pass an instance to any model constructor:
       ``BRCfine(loader, priors, backend=PyMCBackend())``

    Notes
    -----
    - ``format_model_shapes`` / ``print_model_shape`` is intentionally
      excluded: it is a NumPyro-specific diagnostic utility, not an
      inference operation.
    - The ``model`` argument in ``run_mcmc`` / ``run_svi`` is the bound
      ``model.model`` callable (a NumPyro-style generative function).
      A PyMC backend would wrap it with its own context machinery.
    """

    def run_mcmc(
        self,
        model: Callable,
        prng_key: Any,
        *,
        num_samples: int,
        num_warmup: int,
        num_chains: int,
        target_accept_prob: float,
        max_tree_depth: int,
        **model_kwargs: Any,
    ) -> Any:
        """Run MCMC and return the raw result object."""
        ...

    def run_svi(
        self,
        model: Callable,
        guide: Callable,
        prng_key: Any,
        *,
        num_steps: int,
        peak_lr: float,
        **model_kwargs: Any,
    ) -> Any:
        """Run SVI and return the raw result object."""
        ...

    def get_mcmc_samples(self, mcmc_result: Any) -> Dict[str, Any]:
        """Extract posterior sample dict from an MCMC result object."""
        ...

    def get_mcmc_extra_fields(self, mcmc_result: Any) -> Dict[str, Any]:
        """Extract extra fields (e.g. divergences) from an MCMC result object."""
        ...

    def get_svi_params(self, svi_result: Any) -> Dict[str, Any]:
        """Extract optimised variational parameters from an SVI result object."""
        ...

    def get_svi_samples(
        self,
        prng_key: Any,
        guide: Callable,
        svi_result: Any,
        num_samples: int,
        **guide_kwargs: Any,
    ) -> Dict[str, Any]:
        """Sample latent parameters from the variational posterior (guide)."""
        ...

    def posterior_predictive_mcmc(
        self,
        prng_key: Any,
        model: Callable,
        mcmc_result: Any,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate posterior predictive samples using MCMC posterior."""
        ...

    def posterior_predictive_svi(
        self,
        prng_key: Any,
        model: Callable,
        guide: Callable,
        svi_result: Any,
        num_samples: int,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate posterior predictive samples using the variational posterior."""
        ...

    def backend_name(self) -> str:
        """Identifier string for this backend (e.g. ``"numpyro"``)."""
        ...
