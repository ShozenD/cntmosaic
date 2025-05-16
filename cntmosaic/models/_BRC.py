from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import jax
import jaxlib
from jax import random
import numpyro
from numpyro.handlers import seed, trace

from ..dataloader import DataLoader
from ._inference import (
  run_inference_mcmc,
  run_inference_svi,
  posterior_predictive_mcmc,
  posterior_predictive_svi
)

class BRC(ABC):
  # consider changing the base model? 
  """Base class for the Bayesian Rate Consistency model.
  Parameters
  ----------
  dataloader: DataLoader
    DataLoader object containing the processed participant and contact data.
    See `DataLoader` for more details.
  priors: dict
    A dictionary containing the priors for the model.
  likelihood: str, default='negbin'
      Likelihood function to use.

  References
  ----------
  Shozen Dan et al., "Estimating fine age structure and time trends in
  human contact patterns from coarse contact data: The Bayesian rate consistency model",
  PLoS Computational Biology. 2023
  """
  ALLOWED_LIKELIHOODS = ['negbin', 'poisson']
  
  def __init__(self, dataloader: DataLoader, priors: dict, likelihood: str='negbin'):
    self._validate_common_inputs(dataloader, priors, likelihood)
    
    self.ds = dataloader.load()
    self.priors = priors
    self.likelihood = likelihood
  
    self.set_age_dims(self.ds.age.values.min(), self.ds.age.values.max())
    
  def _validate_common_inputs(self, dataloader: DataLoader, priors: dict, likelihood: str):
    """
    This method validates the inputs which are common to all models. Specifically, it checks if the 
    likelihood is one of allowed values and if the priors is a dictionary.
    It also checks if the priors contain the specifications for 'rate'.
    """
    if likelihood not in self.ALLOWED_LIKELIHOODS:
      raise ValueError(f"likelihood must be one of: {self.ALLOWED_LIKELIHOODS}")
    
    if not isinstance(priors, dict):
      raise ValueError("priors must be a dictionary")
    
    if 'rate' not in priors.keys():
      raise ValueError("priors must contain the specifications for 'rate'")
  
  def set_age_dims(self, age_min: int, age_max: int):
    """
    Set the age dimensions of the model.
    Also communicates with the priors to set the age bounds.

    Parameters
    ----------
    min: int
      Minimum age.
    max: int
      Maximum age.
    """
    self.age_min = age_min
    self.age_max = age_max
    self.A = age_max - age_min + 1
    for prior in self.priors.values():
      prior.set_age_bounds(age_min, age_max)

  def set_age_dist(self, age_dist: NDArray):
    """Set the population age distribution.

    Parameters
    ----------
    age_dist: NDArray
      Population age distribution.
    """
    self.age_dist = age_dist

  @abstractmethod
  def model(self):
    raise NotImplementedError

  def print_model_shape(self):
    """Print the shapes of the model parameters."""
    tr = trace(seed(self.model, random.PRNGKey(0))).get_trace()
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
  
  def prior_sampler(self, para_name, num_samples=1, seed=0):
    '''
    Sample from the prior distribution of a given parameter.

    Parameters
    ----------
    para_name : str
        Name of the parameter to sample from. Must be a key in `self.params.prior`.
    num_samples : int, optional
        Number of samples to draw from the prior distribution. Default is 1.
    seed : int, optional
        Seed for random number generation to ensure reproducibility. Default is 0.

    Returns
    -------
    Array
        A NumPy or JAX array of shape `(num_samples, ...)` containing the sampled values.

    Raises
    ------
    AssertionError
        If `para_name` is not found in `self.params.prior`.
    '''
    assert(para_name in self.params.prior)
    _, subkey = jrd.split(jrd.PRNGKey(seed))
    samples = self.params.prior[para_name].sample(
    subkey, sample_shape=(num_samples,))
    return samples
