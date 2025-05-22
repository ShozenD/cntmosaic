from numpy.typing import NDArray
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from .._utils import gmrf_adjacency_matrix
from ._TensorSpline2D import TensorSpline2D

from .._math import (
    inverse_alr,
    inverse_clr,
    inverse_ilr
)

class PenalisedTensorSpline2D(TensorSpline2D):
    """Sample from a 2 dimensional penalised tensor product spline.
    
    Parameters
    ----------
    M: int | list[int], default=30
        The number of basis functions to use in each dimension.
    degree: int | list[int], default=3
        The degree of the B-splines.
    neighborhood: int, default=4
        The number of neighbors to consider in the Gaussian Markov random field.
    grid_type: str, default='age-age'
        The type of grid to use. Options are 'age-age' and 'diff-age'.
    loc: float | NDArray, default=0
        The prior mean of the tensor spline.
    coef_scale: float, default=1
        The prior scale of parameters in the GMRF.
    event_dim: int, default=1
        The size of the leading dimension of the output tensor. If 1 the output is a matrix, if >1 the output is a tensor.
    transform: str, default=None
        The transformation to apply to the output tensor. Options are 'alr', 'clr', and 'ilr'.
    symmetric: bool, default=False
        Whether to symmetrize the output matrix/tensor.
        
    Examples
    --------
    
    >>> priors = {'rate': PenalisedTensorSpline2D()}
    """
    
    ALLOWED_NEIGHBORHOODS = [4, 8]
    pytree_aux_fields = ("self.PHI", "order",)
    
    def __init__(self,
                 M: int | list[int]=30,
                 degree: int | list[int]=3,
                 neighborhood: int=4,
                 grid_type: str='age-age',
                 loc: float | NDArray=0,
                 coef_scale: float | NDArray=1,
                 event_dim: int=1,
                 transform: str | None=None,
                 type: str='global',
                 symmetric: bool=False):
        
        super().__init__(M, degree, grid_type, loc, coef_scale, event_dim, transform, type, symmetric)
        
        if neighborhood not in self.ALLOWED_NEIGHBORHOODS:
            raise ValueError(f"neighborhood must be one of: {self.ALLOWED_NEIGHBORHOODS}")
        self.adj_matrix = gmrf_adjacency_matrix(M, M, neighborhood)
        
    def set_age_bounds(self, min_age, max_age):
        return super().set_age_bounds(min_age, max_age)
    
    def _set_grid(self):
        return super()._set_grid()
    
    def sample(self):
        """Sample from the penalised tensor spline prior."""
        
        if self.type == 'global':
            beta = numpyro.sample('spline_coef', dist.CAR(0, 0.999, 1/self.coef_scale, self.adj_matrix, is_sparse=True))
            
            f = self.PHI @ beta
            f = f[self.sym_tri_idx] if self.symmetric else f
            return f.reshape((self.A, self.A), order='F')
        
        elif self.type == 'partial':
            with numpyro.plate('event', self.event_dim_eff):
                beta = numpyro.sample('spline_coef', dist.CAR(0, 0.999, 1/self.coef_scale, self.adj_matrix, is_sparse=True))
            
            f = beta @ self.PHI_T
            f = f[:,self.sym_tri_idx] if self.symmetric else f
            f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A), order='F')
        
        elif self.type == 'full':
            plate_diag = numpyro.plate('diag', self.event_dim_diag)
            plate_non_diag = numpyro.plate('non_diag', self.event_dim_non_diag)
            
            with plate_diag:
                beta_diag = numpyro.sample('spline_coef_diag', dist.CAR(0, 0.999, 1/self.coef_scale, self.adj_matrix, is_sparse=True))
            
            with plate_non_diag:
                beta_non_diag = numpyro.sample('spline_coef_non_diag', dist.CAR(0, 0.999, 1/self.coef_scale, self.adj_matrix, is_sparse=True))
            
            f_diag = beta_diag @ self.PHI_diag_T
            f_diag = f_diag[:,self.sym_tri_idx] # Must be symmetric
            
            f = beta_non_diag @ self.PHI_non_diag_T
            
            for i in range(self.event_dim_diag):
                f = jnp.insert(f, (i+1)**2 - 1, f_diag[i,:], axis=0)
                
            f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A), order='F')
        else:
            raise ValueError("Unknown prior type")
        
        if self.transform == 'alr':
            return inverse_alr(f, axis=0)
        elif self.transform == 'clr':
            return inverse_clr(f, axis=0)
        elif self.transform == 'ilr':
            return inverse_ilr(f, axis=0)
        else:
            return f