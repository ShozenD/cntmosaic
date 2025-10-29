import numpy as np
import jax.numpy as jnp
from jax import vmap
import numpyro
from numpyro import distributions as dist

from ..funcs import (
  gmrf2d_operators,
  gmrf2d_sym_operators,
  gmrf,
  gmrf_sym
)
from .._utils import symmetrize_from_lower_tri

from ._Prior2D import Prior2D

from .._math import (
		inverse_alr,
		inverse_clr,
		inverse_ilr,
)

class vdKassteele(Prior2D):
		"""Class for sampling from a 2 dimensional intrinsic Gaussian Markov random field (gmrf).
		
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
		def __init__(self,
								 transform: str | None=None,
								 prior_type: str='global'):
				super().__init__('age-age', transform, prior_type)
								
		def set_age_bounds(self, min_age: int, max_age: int):
				self.min_age = min_age
				self.max_age = max_age
				self.A = int(max_age - min_age + 1)
				
				self._set_grid()
		
		def _set_grid(self):
				self.sym_idx = symmetrize_from_lower_tri(self.A)

				if self.type == 'global':
					self.L, self.sym_idx = gmrf2d_sym_operators(self.A, (2, 2), cov_struct='additive')
				elif self.type == 'partial':
					self.L = gmrf2d_operators((self.A, self.A), (2, 2), cov_struct='additive')
				elif self.type == 'full':
					self.L_diag, self.sym_idx = gmrf2d_sym_operators((self.A, self.A), (2, 2), cov_struct='additive')
					self.L_non_diag = gmrf2d_operators(self.A, (2, 2), cov_struct='additive')
	
		def sample(self):
				"""Sample from the HSGP prior."""
				if self.type == 'global':
						N = self.A * (self.A + 1) // 2
						tau = numpyro.sample('tau', dist.InverseGamma(1, 0.0001))
						z = numpyro.sample('z', dist.Normal(0, 1).expand((N,)).to_event(1))
						f = gmrf_sym(z, self.L, self.sym_idx, tau)
      
						return f.reshape((self.A, self.A), order='F')
				
				elif self.type == 'partial':
						tau = numpyro.sample('tau', dist.InverseGamma(1, 0.0001))
						def map_gmrf(x): return gmrf(x, self.L, tau)
						
						plate_event = numpyro.plate('event', self.event_dim_eff, dim=-2)
						with plate_event:
								z = numpyro.sample('z', dist.Normal(0, 1), sample_shape=(self.A**2,))
						f = vmap(map_gmrf)(z)
						f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A), order='F')
				
				elif self.type == 'full':
						tau = numpyro.sample('tau', dist.InverseGamma(1, 0.0001))
      
						def map_gmrf_sym(x): return gmrf_sym(x, self.L_diag, self.sym_idx, tau)
						def map_gmrf(x): return gmrf(x, self.L_non_diag, tau)
  
						plate_diag = numpyro.plate('diag', self.event_dim_diag, dim=-2)
						plate_non_diag = numpyro.plate('non_diag', self.event_dim_non_diag, dim=-2)

						with plate_diag:
								N = self.A * (self.A - 1) // 2
								z_diag = numpyro.sample('z_diag', dist.Normal(0, 1).expand((N,)).to_event(1))

						with plate_non_diag:
								z_non_diag = numpyro.sample(
          					'z_non_diag',
               			dist.Normal(0, 1).expand((self.A**2,)).to_event(1)
                )

						f_diag = vmap(map_gmrf_sym)(z_diag)
						f_non_diag = vmap(map_gmrf)(z_non_diag)

						# Preallocate flat output: (event_dim_eff, A*A)
						f = jnp.zeros((self.event_dim_eff, self.A * self.A))

						diag_idx = jnp.array([i * self.A + i for i in range(self.A)])
						all_idx = jnp.arange(self.A * self.A)
						non_diag_idx = jnp.setdiff1d(all_idx, diag_idx)

						# Insert values
						f = f.at[:, diag_idx].set(f_diag)
						f = f.at[:, non_diag_idx].set(f_non_diag)
						f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A), order='F')
				else:
						raise ValueError(f"Unknown type '{self.type}'. Must be one of 'global', 'partial', or 'full'.")

				# Optional transformations
				if self.transform == 'alr':
						return inverse_alr(f, axis=0)
				elif self.transform == 'clr':
						return inverse_clr(f, axis=0)
				elif self.transform == 'ilr':
						return inverse_ilr(f, axis=0)
				else:
						return f