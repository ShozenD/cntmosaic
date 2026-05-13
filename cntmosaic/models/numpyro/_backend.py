"""
NumPyro implementation of InferenceBackend.

All methods delegate to the free functions in ``cntmosaic.models._numpyro``
without duplicating any logic. Stages 3 and 4 of item 1.7 will thread this
class through the model constructors and delete the duplicated inference
boilerplate from ``_BRC.py``, ``_Prem.py``, and ``_vdKassteele.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class NumPyroBackend:
    """
    NumPyro implementation of the ``InferenceBackend`` protocol.

    Wraps the free functions in ``cntmosaic.models._numpyro`` as instance
    methods so that model classes can delegate inference through a
    swappable backend object.

    Extra methods not on the Protocol
    ----------------------------------
    print_model_shape(model_fn)
        NumPyro-specific diagnostic; calls ``numpyro.util.format_shapes``.
    _build_default_guide(model_fn)
        Returns ``AutoNormal(model_fn)`` — used by ``Prem`` and
        ``vdKassteele`` when ``guide=None`` is passed to ``run_svi``.
    """

    # ------------------------------------------------------------------
    # InferenceBackend Protocol methods
    # ------------------------------------------------------------------

    def run_mcmc(
        self,
        model: Callable,
        prng_key: Any,
        *,
        num_samples: int = 500,
        num_warmup: int = 500,
        num_chains: int = 2,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        **model_kwargs: Any,
    ) -> Any:
        """Run HMC/NUTS and return the NumPyro MCMC object."""
        from .._numpyro import run_inference_mcmc

        return run_inference_mcmc(
            prng_key,
            model,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            **model_kwargs,
        )

    def run_svi(
        self,
        model: Callable,
        guide: Callable,
        prng_key: Any,
        *,
        num_steps: int = 5_000,
        peak_lr: float = 0.01,
        **model_kwargs: Any,
    ) -> Any:
        """Run SVI and return the NumPyro SVIRunResult."""
        from .._numpyro import run_inference_svi

        return run_inference_svi(
            prng_key,
            model,
            guide,
            num_steps=num_steps,
            peak_lr=peak_lr,
            **model_kwargs,
        )

    def get_mcmc_samples(self, mcmc_result: Any) -> Dict[str, Any]:
        """Return posterior sample dict from the MCMC object."""
        return mcmc_result.get_samples()

    def get_mcmc_extra_fields(self, mcmc_result: Any) -> Dict[str, Any]:
        """Return extra diagnostic fields from the MCMC object."""
        return mcmc_result.get_extra_fields()

    def get_svi_params(self, svi_result: Any) -> Dict[str, Any]:
        """Return optimised variational parameters from the SVI result."""
        return svi_result.params

    def get_svi_samples(
        self,
        prng_key: Any,
        guide: Callable,
        svi_result: Any,
        num_samples: int = 2000,
        **guide_kwargs: Any,
    ) -> Dict[str, Any]:
        """Sample latent parameters from the variational posterior."""
        from .._numpyro import get_samples_svi

        return get_samples_svi(
            prng_key,
            guide,
            svi_result.params,
            num_samples=num_samples,
            **guide_kwargs,
        )

    def posterior_predictive_mcmc(
        self,
        prng_key: Any,
        model: Callable,
        mcmc_result: Any,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate posterior predictive samples using MCMC posterior."""
        from .._numpyro import posterior_predictive_mcmc

        return posterior_predictive_mcmc(prng_key, model, mcmc_result, **model_kwargs)

    def posterior_predictive_svi(
        self,
        prng_key: Any,
        model: Callable,
        guide: Callable,
        svi_result: Any,
        num_samples: int = 1000,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate posterior predictive samples from the variational posterior."""
        from .._numpyro import posterior_predictive_svi

        return posterior_predictive_svi(
            prng_key,
            model,
            guide,
            svi_result.params,
            num_samples=num_samples,
            **model_kwargs,
        )

    # ------------------------------------------------------------------
    # Extra NumPyro-specific methods (not on the Protocol)
    # ------------------------------------------------------------------

    def print_model_shape(self, model_fn: Callable) -> None:
        """
        Print all sample-site shapes for *model_fn* using NumPyro trace.

        Parameters
        ----------
        model_fn : Callable
            The bound ``model.model`` callable to trace.
        """
        import numpyro
        from jax import random
        from numpyro.handlers import seed, trace

        tr = trace(seed(model_fn, random.PRNGKey(0))).get_trace()
        print(numpyro.util.format_shapes(tr))

    def _build_default_guide(self, model_fn: Callable) -> Callable:
        """
        Return ``AutoNormal(model_fn)`` as the default variational guide.

        Used by ``Prem`` and ``vdKassteele`` when ``guide=None`` is passed
        to ``run_inference_svi``. BRC-family models require an explicit guide
        and do not call this method.
        """
        from numpyro.infer.autoguide import AutoNormal

        return AutoNormal(model_fn)
