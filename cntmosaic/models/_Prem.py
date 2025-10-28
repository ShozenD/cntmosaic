import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp
import jaxlib
from jax import random

import numpyro
from numpyro import distributions as dist
from numpyro.handlers import seed, trace
from ._numpyro import (
  run_inference_mcmc,
  run_inference_svi,
  posterior_predictive_mcmc,
  posterior_predictive_svi
)

from ..distributions import IGMRF2D

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
    
    self.data["iix"] = pd.factorize(self.data["id"])[0]
    
    self.N = self.data["id"].nunique()
    self.y = np.array(self.data["y"].values)
    
    self.iix = np.array(self.data["iix"].values)
    self.C = self.data["age_grp_cnt"].cat.categories.size
    self.D = self.data["age_grp_part"].cat.categories.size
    self.cix = np.array(self.data["age_grp_part"].cat.codes)
    self.dix = np.array(self.data["age_grp_cnt"].cat.codes)
    
  def model(self, y=None):
    beta0 = numpyro.sample("beta0", dist.Normal(0., 10.))
    
    tau = numpyro.sample("tau", dist.Gamma(1.0, 0.01))
    beta_cd = numpyro.sample("beta_cd", IGMRF2D(
      num_nodes=(self.D, self.C),
      order=(1, 1),
      cond_prec1=tau,
      cond_prec2=tau,
    )).reshape((self.D, self.C))
    
    log_cint = numpyro.deterministic('log_cint', beta0 + beta_cd)
    
    if self.random_effects:
      mu_re = numpyro.sample("mu_re", dist.Normal(0.0, 1.0))
      tau_re = numpyro.sample("tau_re", dist.HalfNormal(1.0))
      with numpyro.plate("random_effects", self.N):
        sigma_re = numpyro.sample("sigma_re", dist.Normal(mu_re, tau_re))
      lam = jnp.exp(log_cint[self.cix, self.dix] + sigma_re[self.iix])
    else:
      lam = jnp.exp(log_cint[self.cix, self.dix])

    with numpyro.plate("data", len(self.y)):
      numpyro.sample("obs", dist.Poisson(lam), obs=y)
      
  def print_model_shape(self):
    """Print the shapes of the model parameters."""
    tr = trace(seed(self.model, random.PRNGKey(0))).get_trace()
    print(numpyro.util.format_shapes(tr))

  def run_inference_mcmc(
    self,
    rng_key,
    num_samples: int = 500,
    num_warmup: int = 500,
    num_chains: int = 2):
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
    self.mcmc = run_inference_mcmc(
      rng_key,
      self.model,
      num_samples=num_samples,
      num_warmup=num_warmup,
      num_chains=num_chains,
      y=self.y
    )

  def run_inference_svi(
    self,
    prng_key,
    guide: callable,
    num_steps: int = 5_000,
    peak_lr: float = 0.01
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
      y=self.y
    )

  def posterior_predictive_svi(
    self,
    prng_key,
    guide: callable,
    num_samples: int = 5_000,
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
      num_samples=num_samples
    )
