from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

import jax
from jax import random
import pandas as pd
import numpyro
from numpyro.handlers import seed, trace

from ..utils.model import (
  non_nuisance_grid,
  lower_tri_indices,
  symmetrize_from_lower_tri,
  transpose_vector_indices
)

from ..utils.inference import (
	run_inference_mcmc,
	run_inference_svi,
	posterior_predictive_mcmc,
 	posterior_predictive_svi
)

class BRC(ABC):
	"""
 	Base class for the Bayesian Rate Consistency model.
	
	Parameters
	----------
	data: DataFrame
		DataFrame containing the contact data. Must contain the columns 'y', 'age_part', and 'age_cnt.
		'y' is the number of contacts between 'age_part' and 'age_cnt'.
		'age_part' is the age of the contactor.
		'age_cnt' is the age of the contacted.
		
	likelihood: str, default='negbin'
		Likelihood function to use.
		
	References
	----------
	
	Shozen Dan et al., "Estimating fine age structure and time trends in 
	human contact patterns from coarse contact data: The Bayesian rate consistency model",
	PLoS Computational Biology. 2023
	"""
  
	def __init__(self,
				 data: pd.DataFrame,
				 age_dist: NDArray,
				 likelihood: str='negbin'):
	  
		self.data = data.copy()
		self.age_dist = age_dist
		self.likelihood = likelihood
		
		self.set_hsgp_params()
		self.A = self.data['age_part'].max() + 1
		self._precompute_indices()
		self._precompute_fine_age_grid()
	
	def set_hsgp_params(self,
						M: int | list[int]=None,
						C: float | list[float]=None):
		"""Set the hyperparameters for the Hilbert space approximate Gaussian process prior.
		
		Parameters
		----------
		M: int | list[int], default=None
			Number of eigenfunctions to use.
			If int, the same number of eigenfunctions will be used for each dimension.
			If list, the number of eigenfunctions to use for each dimension.
		C: float | list[float], default=None
			Scaling factor for the length scale.
			If float, the same scaling factor will be used for each dimension.
			If list, the scaling factor for each dimension
		"""
		self.M = M if M is not None else [30, 30]
		self.C = C if C is not None else [1.5, 1.5]
		
	def set_age_dist(self, age_dist: NDArray):
		"""Set the population age distribution.
		
		Parameters
		----------
		age_dist: NDArray
			Population age distribution.
		"""
		self.age_dist = age_dist
		
	def _precompute_fine_age_grid(self):
		"""Precompute the age grid.
		
		The default age grid is the difference in age by age parameterisation.
		
		References
		----------
		Shozen Dan et al., "Estimating fine age structure and time trends in
		human contact patterns from coarse contact data: The Bayesian rate consistency model",
		PLoS Computational Biology. 2023
		
		Vendendijck et al., "Cohort-based smoothing methods for age-specific contact rates",
		BioRxiv. 2022
		"""
		X = non_nuisance_grid(self.A)
		ltri_idx = lower_tri_indices(self.A)
		Xn = (X - X.mean(axis=0)) / X.std(axis=0)
		self.L = list(np.abs(Xn).max(axis=0) * self.C)
		self.X = Xn[ltri_idx]
	
	def _precompute_indices(self):
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
		
	def run_inference_mcmc(self,
						   rng_key,
						   num_samples: int=500,
						   num_warmup: int=500,
						   num_chains: int=2,
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
		num_samples: int = 2000,
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