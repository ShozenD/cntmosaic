import jax
from jax import random
from optax import linear_onecycle_schedule

from numpyro.infer import MCMC, NUTS, SVI, Predictive
from numpyro.infer.elbo import Trace_ELBO
from numpyro.infer.initialization import init_to_median, init_to_uniform
from numpyro.optim import Adam

def fit_mcmc(
    seed: int,
    model: callable,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 4,
    target_accept_prob: float = 0.8,
    init_strategy: callable = init_to_median,
    **model_kwargs,
):
  rng_key = random.PRNGKey(seed)
  kernel = NUTS(
    model, target_accept_prob=target_accept_prob, init_strategy=init_strategy
  )
  mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    progress_bar=True,
  )
  mcmc.run(rng_key, **model_kwargs)
  return mcmc

def fit_svi(
    seed: int,
    model: callable,
    guide: callable,
    num_steps: int = 5_000,
    peak_lr: float = 0.01,
    **model_kwargs,
):
  lr_scheduler = linear_onecycle_schedule(num_steps, peak_lr)
  svi = SVI(model, guide, Adam(lr_scheduler), Trace_ELBO())
  return svi.run(random.PRNGKey(seed), num_steps, progress_bar=True, **model_kwargs)

def posterior_predictive_mcmc(
    seed: int,
    model: callable,
    mcmc: MCMC,
    **model_kwargs,
) -> dict[str, jax.Array]:
    samples = mcmc.get_samples()
    predictive = Predictive(model, samples, parallel=True)
    return predictive(random.PRNGKey(seed), **model_kwargs)

def posterior_predictive_svi(
    seed: int,
    model: callable,
    guide: callable,
    params: dict,
    num_samples: int = 2000,
    **model_kwargs,
) -> dict[str, jax.Array]:
    predictive = Predictive(model, guide=guide, params=params, num_samples=num_samples)
    return predictive(random.PRNGKey(seed), **model_kwargs)