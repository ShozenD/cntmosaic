import os

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from arviz.data.base import dict_to_dataset
from arviz.data.inference_data import InferenceData
from jax import random
from numpyro.handlers import substitute
from numpyro.infer import MCMC, NUTS, SVI, Predictive
from numpyro.infer.elbo import Trace_ELBO
from numpyro.infer.initialization import init_to_median, init_to_uniform
from numpyro.infer.util import _predictive, log_likelihood
from numpyro.optim import Adam
from optax import linear_onecycle_schedule


def run_inference_mcmc(
    prng_key,
    model: callable,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 4,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    init_strategy: callable = init_to_median,
    **model_kwargs,
):
    kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(prng_key, **model_kwargs)

    extra_fields = mcmc.get_extra_fields()
    print(f"Number of divergences: {jnp.sum(extra_fields['diverging'])}")

    return mcmc


def run_inference_svi(
    prng_key,
    model: callable,
    guide: callable,
    num_steps: int = 5_000,
    peak_lr: float = 0.01,
    **model_kwargs,
):
    lr_scheduler = linear_onecycle_schedule(num_steps, peak_lr)
    svi = SVI(model, guide, Adam(lr_scheduler), Trace_ELBO())
    return svi.run(prng_key, num_steps, progress_bar=True, **model_kwargs)


def posterior_predictive_mcmc(
    prng_key,
    model: callable,
    mcmc: MCMC,
    **model_kwargs,
) -> dict[str, jax.Array]:
    samples = mcmc.get_samples()
    predictive = Predictive(model, samples, parallel=True)
    return predictive(prng_key, **model_kwargs)


def get_samples_svi(
    prng_key,
    guide: callable,
    params: dict,
    num_samples: int = 2000,
    **guide_kwargs,
) -> dict[str, jax.Array]:
    """Sample parameters from the variational posterior (guide).

    This is the SVI equivalent of MCMC.get_samples() - it samples the latent
    parameters (e.g., beta0, beta_cd) from the learned variational distribution.

    Parameters
    ----------
    prng_key : PRNGKey
        Random number generator key.
    guide : callable
        The variational guide function.
    params : dict
        SVI parameters from svi_result.params.
    num_samples : int, default=2000
        Number of posterior samples to draw.
    **guide_kwargs
        Additional keyword arguments for the guide.

    Returns
    -------
    dict[str, jax.Array]
        Dictionary of parameter samples, excluding auto-guide internal variables.
    """
    # Substitute guide parameters and sample from it
    sub_guide = substitute(guide, params)
    posterior_samples = {}
    batch_size = (num_samples,)

    # Sample from the guide to get parameter posterior
    samples = _predictive(
        prng_key,
        sub_guide,
        posterior_samples,
        batch_size,
        return_sites="",
        parallel=True,
        model_args=(),
        model_kwargs=guide_kwargs,
        exclude_deterministic=True,
    )

    # Filter out auto-guide internal variables (contain '_auto_')
    return {k: v for k, v in samples.items() if "_auto_" not in k}


def posterior_predictive_svi(
    prng_key,
    model: callable,
    guide: callable,
    params: dict,
    num_samples: int = 2000,
    **model_kwargs,
) -> dict[str, jax.Array]:
    predictive = Predictive(model, guide=guide, params=params, num_samples=num_samples)
    return predictive(prng_key, **model_kwargs)


class NumPyroSVIConverter:
    def __init__(self, model: callable, guide: callable, svi: SVI, **model_kwargs):
        """Convert NumPyro SVI data into an InferenceData object.

        Parameters
        ----------
        model: callable
                The generative model.
        guide: callable
                The variational guide.
        svi: SVI
                The SVI object.
        num_samples: int
                The number of posterior samples to draw.
        **model_kwargs
                Additional keyword arguments for the model.
        """
        self.model = model
        self.svi = svi
        self.guide = guide
        self.num_samples = int(svi.state.optim_state[0])
        self.model_kwargs = model_kwargs
        self.inference_dict = {}

        posterior_samples = {}
        batch_size = (self.num_samples,)
        sub_guide = substitute(self.guide, self.svi.params)
        self.posterior = _predictive(
            random.PRNGKey(0),
            sub_guide,
            posterior_samples,
            batch_size,
            return_sites="",
            parallel=True,
            model_args=(),
            model_kwargs=model_kwargs,
            exclude_deterministic=True,
        )

        sub_model = substitute(self.model, self.svi.params)
        self.posterior_predictive = _predictive(
            random.PRNGKey(0),
            sub_model,
            self.posterior,
            (self.num_samples,),
            return_sites="",
            parallel=True,
            model_args=(),
            model_kwargs=model_kwargs,
        )

    def posterior_to_xarray(self):
        # Remove items that contain '_auto_' from data
        data = {k: v for k, v in self.posterior.items() if "_auto_" not in k}
        for key, values in data.items():
            data[key] = values[None, ...]

        return dict_to_dataset(data, library=numpyro)

    def log_likelihood_to_xarray(self):
        log_likelihood_dict = log_likelihood(
            self.model, self.posterior, **self.model_kwargs
        )

        data = {}
        for obs_name, log_like in log_likelihood_dict.items():
            shape = (1, log_like.shape[0]) + log_like.shape[1:]
            data[obs_name] = np.reshape(np.asarray(log_like), shape)

        return dict_to_dataset(data, library=numpyro)

    def to_inference_data(self):
        """Convert all available data to an Inference object."""
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "log_likelihood": self.log_likelihood_to_xarray(),
            }
        )


def to_inference_data(model: callable, guide: callable, svi: SVI, **model_kwargs):
    """
    Convert NumPyro SVI data to an InferenceData object.

    Parameters
    ----------
    model: callable
            The generative model.
    guide: callable
            The variational guide.
    svi: SVI
            The SVI object.
    **model_kwargs
            Additional keyword arguments for the model.
    """
    converter = NumPyroSVIConverter(model=model, guide=guide, svi=svi, **model_kwargs)

    return converter.to_inference_data()
