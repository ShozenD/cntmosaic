from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap

import numpyro
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.util import is_prng_key
from numpyro.distributions.util import validate_sample, promote_shapes

from .._IGMRF import diff_matrix, laplacian
from ..models._utils import symmetrize_from_lower_tri

from ..models._math import (
	inverse_alr,
	inverse_clr,
	inverse_ilr,
)

class IGMRF2D(Distribution):
	"""Class for sampling from a 2 dimensional intrinsic Gaussian Markov random field (IGMRF).
	
	Parameters
	----------
	grid_type: str, default='age-age'
			The type of grid to use. Options are 'age-age' and 'diff-age'.
	loc: float, default=0
			The mean or mean function of the HSGP.
	event_dim: int, default=1
			The number of dimensions of the event.
	transform: str | None, default=None
			The transformation to apply to the HSGP. Options are 'alr', 'clr', and 'ilr'.
	"""
	def __init__(
		self,
		num_nodes: tuple,
		order: tuple,
		loc: ArrayLike = 0.0,
		grid_type: str='age-age',
		transform: str | None=None,
		prior_type: str='global'
  ):
		self.scale = scale
		self.order = order
		super().__init__(grid_type, transform, prior_type)
								
	def set_age_bounds(self, min_age: int, max_age: int):
			self.min_age = min_age
			self.max_age = max_age
			self.A = max_age - min_age + 1
			
			self._set_grid()

	def _set_grid(self):
		self.sym_idx = symmetrize_from_lower_tri(self.A)
		num_nodes = (self.A, self.A)
		order = (2, 2)
   
		if self.prior_type == 'global':
			self.operator_sym = make_sym_igmrf2d_operator(num_nodes, order)
		elif self.prior_type == 'partial':
			self.operator = make_igmrf2d_operator(num_nodes, order)
		elif self.prior_type == 'full':
			self.operator_sym = make_sym_igmrf2d_operator(num_nodes, order)
			self.operator = make_igmrf2d_operator(num_nodes, order)

	def sample(self):
		if self.prior_type == 'global':
			N = self.A * (self.A + 1) // 2
			z = numpyro.sample('z', dist.Normal(0, 1).expand((N,)).to_event(1))
			tau = numpyro.sample('tau', dist.Gamma(2, 0.1))
			f = self.operator_sym(z, tau)

			return f

		elif self.prior_type == 'partial':
			def map_igmrf(x): return self.operator(x, tau)
   
			z = numpyro.sample(
     		'z',
     		dist.Normal(0, 1),
       	sample_shape=(self.event_dim_eff, self.A**2,)
     	)
			tau = numpyro.sample(
     		'tau',
      	dist.Gamma(2, 0.1),
      	sample_shape=(self.event_dim_eff,)
      )

			f = self.trans_loc + vmap(map_igmrf, in_axes=(0, 0))(z, tau)

		elif self.prior_type == 'full':
			def map_igmrf_sym(x, tau): return self.operator_sym(x, tau)
			def map_igmrf(x, tau): return self.operator(x, tau)

			z_diag = numpyro.sample(
				'z_diag',
				dist.Normal(0, 1),
				sample_shape=(self.event_dim_diag, self.A*(self.A+1)//2,)
			)
			tau_diag = numpyro.sample(
				'tau_diag',
				dist.Gamma(2, 0.1),
				sample_shape=(self.event_dim_diag,)
			)
  
			z_non_diag = numpyro.sample(
				'z_non_diag',
				dist.Normal(0, 1),
				sample_shape=(self.event_dim_non_diag, self.A**2,)
			)
			tau_non_diag = numpyro.sample(
				'tau_non_diag',
				dist.Gamma(2, 0.1),
				sample_shape=(self.event_dim_non_diag,)
			)

			f_diag = vmap(map_igmrf_sym)(z_diag, tau_diag) 				 # shape: (event_dim_diag, A, A)
			f_non_diag = vmap(map_igmrf)(z_non_diag, tau_non_diag) # shape: (event_dim_non_diag, A, A)

			# Preallocate flat output: (event_dim_eff, A, A)
			f = jnp.zeros((self.event_dim_eff, self.A, self.A))

			diag_idx = jnp.array([i * self.A + i for i in range(self.A)])
			all_idx = jnp.arange(self.A * self.A)
			non_diag_idx = jnp.setdiff1d(all_idx, diag_idx)

			# Insert values
			f = f.at[diag_idx,:,:].set(f_diag)
			f = f.at[non_diag_idx,:,:].set(f_non_diag)
			f = self.trans_loc + f
		else:
			raise ValueError(f"Unknown type '{self.prior_type}'. Must be one of 'global', 'partial', or 'full'.")

		# Optional transformations
		if self.transform == 'alr':
			return inverse_alr(f, axis=0)
		elif self.transform == 'clr':
			return inverse_clr(f, axis=0)
		elif self.transform == 'ilr':
			return inverse_ilr(f, axis=0)
		else:
			return f