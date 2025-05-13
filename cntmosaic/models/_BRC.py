from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

import jax
import jaxlib
from jax import random
import pandas as pd
import numpyro
from numpyro.handlers import seed, trace

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
  data: DataFrame
      DataFrame containing the contact data. Must contain the columns 'y', 'age_part', and 'age_cnt.
      'y' is the number of contacts between 'age_part' and 'age_cnt'.
      'age_part' is the age of the contactor.
      'age_cnt' is the age of the contacted.
  age_dist: NDArray
      The population age distribution.
  likelihood: str, default='negbin'
      Likelihood function to use.

  References
  ----------
  Shozen Dan et al., "Estimating fine age structure and time trends in
  human contact patterns from coarse contact data: The Bayesian rate consistency model",
  PLoS Computational Biology. 2023
  """
  ALLOWED_LIKELIHOODS = ['negbin', 'poisson']
  
  def __init__(self,
               data: pd.DataFrame,
               age_dist: NDArray,
               priors: dict,
               likelihood: str='negbin'):
    
    self.data = data.copy()
    self.age_dist = age_dist
    self.priors = priors
    self.likelihood = likelihood    
    self._validate_params()

    if 'age_grp_cnt' in self.data.columns:
      age_part_min = self.data['age_part'].min()
      age_cnt_min = self.data['age_grp_cnt'].apply(lambda x: x.left).astype(int).min()
      
      age_part_max = self.data['age_part'].max()
      age_cnt_max = self.data['age_grp_cnt'].apply(lambda x: x.right - 1).astype(int).max()
      
      age_min = np.min([age_part_min, age_cnt_min])
      age_max = np.max([age_part_max, age_cnt_max])
    else:
      age_min = np.min([self.data['age_part'].min(), self.data['age_cnt'].min()])
      age_max = np.max([self.data['age_part'].max(), self.data['age_cnt'].max()])
    
    self.set_age_bounds(age_min, age_max)
    
  def _validate_params(self):
    if not isinstance(self.data, pd.DataFrame):
      raise ValueError("data must be a pandas DataFrame")
    
    if not isinstance(self.age_dist, np.ndarray):
      raise ValueError("age_dist must be a numpy array")
    
    if not isinstance(self.priors, dict):
      raise ValueError("priors must be a dictionary")
    
    if self.likelihood not in self.ALLOWED_LIKELIHOODS:
      raise ValueError(f"likelihood must be one of: {self.ALLOWED_LIKELIHOODS}")
    
    if 'age_part' not in self.data.columns:
      raise ValueError("data must contain the column 'age_part'")
    
    if 'age_cnt' not in self.data.columns and 'age_grp_cnt' not in self.data.columns:
      raise ValueError("data must contain the column 'age_cnt' or 'age_grp_cnt'")
    
    if 'rate' not in self.priors.keys():
      raise ValueError("priors must contain the specifications for 'rate'")
  
  def set_age_bounds(self, age_min: int, age_max: int):
    """Set the minimum and maximum age.

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
    for _, prior in self.priors.items():
      prior.set_age_bounds(age_min, age_max)

  def set_age_dist(self, age_dist: NDArray):
    """Set the population age distribution.

    Parameters
    ----------
    age_dist: NDArray
      Population age distribution.
    """
    self.age_dist = age_dist

  def _compute_indices(self):
    """Precompute the indices for symmetrizing and transposing the contact matrix."""
    self.sym_tri_idx = symmetrize_from_lower_tri(self.A)
    self.tran_vec_idx = transpose_vector_indices(self.A, self.A)

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
