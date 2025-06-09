import numpy as np
import jax.numpy as jnp
from jax import vmap
import numpyro
from numpyro import distributions as dist

from ..funcs import (
  igmrf2d_operators,
  igmrf2d_sym_operators,
  igmrf,
  igmrf_sym
)
from .._utils import symmetrize_from_lower_tri

from ._Prior2D import Prior2D

from .._math import (
		inverse_alr,
		inverse_clr,
		inverse_ilr,
)

class IGMRF2D(Prior2D):
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
		def __init__(self,
								 grid_type: str='age-age',
								 loc: float=0,
								 scale: float=1,
								 event_dim: int=1,
								 transform: str | None=None,
								 type: str='global'):
				self.scale = scale
				super().__init__(grid_type, loc, event_dim, transform, type)
								
		def set_age_bounds(self, min_age: int, max_age: int):
				self.min_age = min_age
				self.max_age = max_age
				self.A = max_age - min_age + 1
				
				self._set_grid()
				self._set_loc()
		
		def _set_grid(self):
				self.sym_idx = symmetrize_from_lower_tri(self.A)

				if self.type == 'global':
					self.Q, self.L, self.sym_idx = igmrf2d_sym_operators(self.A, (2, 2), cov_struct='additive')
				elif self.type == 'partial':
					self.Q, self.L = igmrf2d_operators((self.A, self.A), (2, 2), cov_struct='additive')
				elif self.type == 'full':
					self.Q_diag, self.L_diag, self.sym_idx = igmrf2d_sym_operators((self.A, self.A), (2, 2), cov_struct='additive')
					self.Q_non_diag, self.L_non_diag = igmrf2d_operators(self.A, (2, 2), cov_struct='additive')
	
		def sample(self):
				"""Sample from the HSGP prior."""
				def map_igmrf(x, Q, L): return igmrf(x, Q, L, self.scale)
				def map_igmrf_sym(x, Q, L, sym_idx): return igmrf_sym(x, Q, L, sym_idx, self.scale)

				if self.type == 'global':
						N = self.A * (self.A + 1) // 2
						z = numpyro.sample('z', dist.Normal(0, 1).expand((N,)).to_event(1))
						f = igmrf_sym(z, self.Q, self.L, self.sym_idx, self.scale)
      
						return self.loc + f.reshape((self.A, self.A), order='F')
				
				elif self.type == 'partial':
						plate_event = numpyro.plate('event', self.event_dim_eff, dim=-2)
						with plate_event:
								z = numpyro.sample('z', dist.Normal(0, 1), sample_shape=(self.A**2,))
						f = vmap(igmrf)(z, self.Q, self.L)
						f = self.loc + f.reshape((self.event_dim_eff, self.A, self.A), order='F')
				
				elif self.type == 'full':
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

						f_diag = vmap(map_igmrf_sym)(z_diag, self.Q_diag, self.L_diag, self.sym_idx)
						f_non_diag = vmap(map_igmrf)(z_non_diag, self.Q_non_diag, self.L_non_diag)

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