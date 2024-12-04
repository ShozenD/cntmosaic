import os

import jax
from jax import random
import jax.numpy as jnp
from optax import linear_onecycle_schedule

from numpyro.infer import MCMC, NUTS, SVI, Predictive
from numpyro.infer.elbo import Trace_ELBO
from numpyro.infer.initialization import init_to_median, init_to_uniform
from numpyro.optim import Adam

def run_inference_mcmc(
    prng_key,
    model: callable,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 4,
    target_accept_prob: float = 0.8,
    init_strategy: callable = init_to_median,
    **model_kwargs,
):
  kernel = NUTS(
    model,
    target_accept_prob=target_accept_prob,
    init_strategy=init_strategy
  )
  mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    progress_bar=False if 'NUMPYRO_SPHINXBUILD' in os.environ else True,
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