import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern

from ._Prior2D import Prior2D

from .._utils import (
    age_age_grid,
    diff_age_age_grid,
    lower_tri_indices,
    symmetrize_from_lower_tri
)

from .._math import (
    inverse_alr,
    inverse_clr,
    inverse_ilr,
)

class HSGP2D(Prior2D):
    """Class for sampling from a 2 dimensional Hilbert space approximate Gaussian process (HSGP).
    
    Parameters
    ----------
    C: int | list[float], default=[1.5, 1.5]
		The boundary condition of the approximation.
    M: int | list[int], default=[30, 30]
		The number of eigenfunctions to use.
    grid_type: str, default='age-age'
        The type of grid to use. Options are 'age-age' and 'diff-age'.
    loc: float, default=0
        The mean or mean function of the HSGP.
    event_dim: int, default=1
        The number of dimensions of the event.
    transform: str | None, default=None
        The transformation to apply to the HSGP. Options are 'alr', 'clr', and 'ilr'.
    symmetric: bool, default=False
		Whether the sampled matrix should be symmetric.
    """
    def __init__(self,
                 C: float | list[float] = [1.5, 1.5],
                 M: int | list[int] = [30, 30],
                 grid_type: str='age-age',
                 loc: float=0,
                 event_dim: int=1,
                 transform: str | None=None,
                 symmetric: bool=False):
        super().__init__(grid_type, loc, event_dim, transform)
        self.C = C
        self.M = M
        self.symmetric = symmetric

    def set_age_bounds(self, min_age: int, max_age: int):
        self.min_age = min_age
        self.max_age = max_age
        self.A = max_age - min_age + 1
        
        self._set_grid()
        self._make_eigenfunctions()
        self._set_loc()
    
    def _set_grid(self):
        if self.grid_type == 'age-age':
            X = age_age_grid(self.A)
        elif self.grid_type == 'diff-age':
            X = diff_age_age_grid(self.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")
        
        Xn = (X - X.mean(axis=0)) / X.std(axis=0)
        self.L = list(np.abs(Xn).max(axis=0) * self.C)
        
        if self.symmetric:
            ltri_idx = lower_tri_indices(self.A)
            self.X = Xn[ltri_idx]
            self.sym_tri_idx = symmetrize_from_lower_tri(self.A)
        else:
            self.X = Xn
    
    def _make_eigenfunctions(self):
        self.eig_func = eigenfunctions(x=self.X, ell=self.L, m=self.M)
        self.eig_func_transpose = self.eig_func.T
  
    def sample(self, alpha, rho):
        """Sample from the HSGP."""
        if self.event_dim == 1:
            sigma = numpyro.sample('gp_scale', alpha)
            lenscale = numpyro.sample('gp_lenscale', rho)
            
            # Compute the spectral density
            diag_spd = diag_spectral_density_matern(
                nu=5/2,
                alpha=sigma,
                length=lenscale,
                ell=self.L,
                m=self.M,
                dim=2
            )
            
            with numpyro.plate('coef', self.eig_func.shape[-1]):
                beta = numpyro.sample('gp_beta', dist.Normal(0, 1))
            
            f = self.eig_func @ (diag_spd * beta)
            f = f[self.sym_tri_idx] if self.symmetric else f
            
            return self.loc.reshape(self.A, self.A) + f.reshape((self.A, self.A), order='F')
            
        else:
            plate_event = numpyro.plate('event', self.event_dim, dim=-2)
            plate_coef = numpyro.plate('coef', self.eig_func.shape[-1], dim=-1)
            
            with plate_event:
                sigma = numpyro.sample('gp_scale', alpha)
                lenscale = numpyro.sample('gp_lenscale', rho)

            # Compute the spectral density
            diag_spd = jnp.vstack([
                diag_spectral_density_matern(
                    nu=5/2,
                    alpha=s,
                    length=l,
                    ell=self.L,
                    m=self.M,
                    dim=2
                )[np.newaxis] for s, l in zip(sigma, lenscale)
            ])
            diag_spd = jnp.squeeze(diag_spd, axis=-1)
            
            with plate_coef:
                beta = numpyro.sample('gp_beta', dist.Normal(0, 1))
            
            f = (diag_spd * beta) @ self.eig_func_transpose
            f = f[self.sym_tri_idx] if self.symmetric else f
            f = self.loc + f.reshape((self.event_dim_eff, self.A, self.A), order='F')
            
            if self.transform == 'alr':
                return inverse_alr(f, axis=0)
            elif self.transform == 'clr':
                return inverse_clr(f, axis=0)
            elif self.transform == 'ilr':
                return inverse_ilr(f, axis=0)
            else:
                return f
