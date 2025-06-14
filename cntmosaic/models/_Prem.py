import pandas as pd

import jax
import jax.numpy as jnp
import jaxlib

import numpyro
from numpyro import distributions as dist
from numpyro.handlers import seed, trace
from ._inference import (
  run_inference_mcmc,
  run_inference_svi,
  posterior_predictive_mcmc,
  posterior_predictive_svi
)

from .funcs import gmrf2d_operators, gmrf

class Prem:
  def __init__(self,
               part: pd.DataFrame,
               cnt: pd.DataFrame,
               random_effects: bool = False):
    self._validate_inputs(part, cnt)
    self.part = part.copy()
    self.cnt = cnt.copy()
    self.random_effects = random_effects
    self._load()
    
  def _validate_inputs(self, part: pd.DataFrame, cnt: pd.DataFrame):
    part_cols = part.columns
    cnt_cols = cnt.columns
    
    if "id" not in part_cols:
      raise ValueError("participant DataFrame must contain 'id' column")
    if "id" not in cnt_cols:
      raise ValueError("contact DataFrame must containt 'id' column")
    
    if "age_grp_part" not in part_cols:
      raise ValueError("participant DataFrame must contain 'age_grp_part' column")
    if "age_grp_cnt" not in cnt_cols:
      raise ValueError("contact DataFrame must contain 'age_grp_cnt' column")
    
    if "y" not in cnt_cols:
      raise ValueError("Missing column 'y' in contact DataFrame")
    
  def _load(self):
    # [Do] Create full contact dataframe
    coords = {
      "id": self.cnt["id"].unique(),
      "age_grp_cnt": self.cnt["age_grp_cnt"].cat.categories
    }
    index = pd.MultiIndex.from_product(coords.values(), names=coords.keys())
    df_cnt_full = pd.DataFrame(list(index), columns=coords.keys())
    df_cnt_full = pd.merge(df_cnt_full, self.cnt, on=["id", "age_grp_cnt"], how="left")
    df_cnt_full["y"] = df_cnt_full["y"].fillna(0).astype(int)
		# [Do] Restore the original information of the age group column
    df_cnt_full["age_grp_cnt"] = pd.Categorical(
			df_cnt_full["age_grp_cnt"],
			categories=self.cnt["age_grp_cnt"].cat.categories,
			ordered=True
		)
    
    # [Do] Merge contact data and participant data
    self.data = pd.merge(df_cnt_full, self.part, on="id", how="left")
    self.data = (
      self.data
      .groupby(["id", "age_grp_part", "age_grp_cnt"], observed=True)['y']
      .sum()
      .reset_index()
    )
    
    self.data["iid"] = pd.factorize(self.data["id"])[0]
    
    self.N = self.data["id"].nunique()
    self.y = jnp.array(self.data["y"].values)
    
    self.iid = jnp.array(self.data["iid"].values)
    self.C = self.data["age_grp_cnt"].cat.categories.size
    self.D = self.data["age_grp_part"].cat.categories.size
    self.cid = jnp.array(self.data["age_grp_part"].cat.codes)
    self.did = jnp.array(self.data["age_grp_cnt"].cat.codes)
    
    self.L = gmrf2d_operators((self.C, self.D), (1, 1), cov_struct="additive")
    
  def model(self):
    beta0 = numpyro.sample("baseline", dist.Exponential(0.001))
    z = numpyro.sample("z", dist.Normal(0., 1.), sample_shape=(self.C * self.D,))
    log_cint = numpyro.deterministic(
      "log_cint",
      jnp.log(beta0)
      + gmrf(z, self.L, scale=1).reshape(self.C, self.D, order="F")
    )[self.cid, self.did]
    
    if self.random_effects:
      theta = numpyro.sample("theta", dist.Exponential(0.0001))
      with numpyro.plate('random_effects', self.N):
        sigma = numpyro.sample("sigma", dist.Gamma(theta, theta))
      
      mu = jnp.exp(log_cint) * sigma[self.iid]
    else:
      mu = jnp.exp(log_cint)
    
    with numpyro.plate("data", len(self.y)):
      numpyro.sample("obs", dist.Poisson(rate=mu), obs=self.y)
      
  def print_model_shape(self):
    """Print the shapes of the model parameters."""
    tr = trace(seed(self.model, jax.random.PRNGKey(0))).get_trace()
    print(numpyro.util.format_shapes(tr))
    
  def run_inference_mcmc(
    self,
    rng_key,
    num_samples: int = 500,
    num_warmup: int = 500,
    num_chains: int = 2,
    **kwargs):
    """Run full Bayesian inference using Hamiltonian Monte Carlo and NUT Sampler.

    Parameters
    ----------
    rng_key:
      Random number generator key.
    num_samples: int, default=1000
      Number of samples to draw from the posterior.
    num_warmup: int, default=1000
      Number of warmup steps.
    num_chains: int, default=1
      Number of chains to run.
    **kwargs
      Additional keyword arguments to pass to the MCMC
    """
    if not isinstance(rng_key, jaxlib.xla_extension.ArrayImpl):
      rng_key = jax.random.PRNGKey(int(rng_key))
    self.mcmc = run_inference_mcmc(
      rng_key,
      self.model,
      num_samples=num_samples,
      num_warmup=num_warmup,
      num_chains=num_chains,
      **kwargs
    )
    
  def run_inference_svi(
    self,
    prng_key,
    guide: callable,
    num_steps: int = 5_000,
    peak_lr: float = 0.01,
    **model_kwargs,
  ):
    """Run stochastic variational inference.

    Parameters
    ----------
    prng_key:
      Random number generator key.
    guide: callable
      The guide function.
    num_steps: int, default=5_000
      Number of steps to run.
    peak_lr: float, default=0.01
      Peak learning rate.
    **model_kwargs
      Additional keyword arguments to pass to the SVI
      """
    self.guide = guide
    self.svi = run_inference_svi(
      prng_key,
      self.model,
      guide,
      num_steps=num_steps,
      peak_lr=peak_lr,
      **model_kwargs
    )
    
  def posterior_predictive_svi(
    self,
    prng_key,
    guide: callable,
    num_samples: int = 5_000,
    **model_kwargs,
  ) -> dict[str, jax.Array]:
    """Generate posterior predictive samples using SVI.

    Parameters
    ----------
    prng_key:
      Random number generator key.
    guide: callable
      The guide function.
    num_samples: int, default=2000
      Number of samples to draw.
    **model_kwargs
      Additional keyword arguments to pass to the Predictive
    """
    if hasattr(self, 'svi') is False:
      raise AttributeError('run_inferece_svi must be run first.')

    return posterior_predictive_svi(
      prng_key,
      self.model,
      guide,
      self.svi.params,
      num_samples=num_samples,
      **model_kwargs
    )


class Prem2(Prem):
  def __init__(self,
               part: pd.DataFrame,
               cnt: pd.DataFrame,
               random_effects: bool = False):
    super().__init__(part, cnt, random_effects)
    
  def model(self):
    beta0 = numpyro.sample("baseline", dist.Normal(0, 3))
    
    z = numpyro.sample("z", dist.Normal(0., 1.), sample_shape=(self.C * self.D,))
    log_cint = numpyro.deterministic(
      "log_cint",
      beta0
      + gmrf(z, self.L, scale=1).reshape(self.C, self.D, order="F")
    )[self.cid, self.did]
    
    if self.random_effects:
      tau = numpyro.sample("tau", dist.HalfCauchy(1))
      with numpyro.plate('random_effects', self.N):
        sigma = numpyro.sample("sigma", dist.Normal(0, tau))
        
      mu = jnp.exp(log_cint + sigma[self.iid]) 
    else:
      mu = jnp.exp(log_cint) 
    
    with numpyro.plate("data", len(self.y)):
      numpyro.sample("obs", dist.Poisson(rate=mu), obs=self.y)