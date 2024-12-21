import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

from scipy.interpolate import BSpline

import numpyro
from numpyro import distributions as dist

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

class TensorSpline2D(Prior2D):
    """Sample from a 2 dimensional tensor product of B-splines.
    
    Parameters
    ----------
    M: int | list[int], default=30
        The number of basis functions to use in each dimension.
    degree: int | list[int], default=3
        The degree of the B-splines for each dimension.
    grid_type: str, default='age-age'
        The type of grid to use. Either 'age-age' or 'diff-age'.
    loc: float | NDArray, default=0
        The prior mean of the tensor spline.
    coef_scale: float | NDArray, default=1
        The scale of the spline coefficients.
    event_dim: int, optional
        The size of the leading dimension of the output tensor. Default is 1.
    transform: str, optional
        The transformation to apply to the output tensor. Either 'alr', 'clr', or 'ilr'.
    symmetric: bool, default=False
        Whether to enforce symmetry in the tensor spline.
    
    Attributes
    ----------
    basis: NDArray
        The tensor product of B-splines.
    
    Examples
    --------
    >>> priors = {'rate' = TensorSpline2D(symmetric=True)}
    """
    def __init__(self,
                 M: int | list[int]=30,
                 degree: int | list[int]=3,
                 grid_type: str='age-age',
                 loc: float | NDArray=0,
                 coef_scale: float | NDArray=1,
                 event_dim: int=1,
                 transform: str | None=None,
                 symmetric: bool=False):
        
        super().__init__(grid_type, loc, event_dim, transform, symmetric)
        
        self.coef_scale = coef_scale
        self.M = np.array([M]*2) if isinstance(M, int) else np.array(M)
        self.degree = np.array([degree]*2) if isinstance(degree, int) else np.array(degree)
        self.n_knots_inner = self.M + self.degree + 1
        self.n_knots_outer = 2 * (self.degree + 1)
        
    def set_age_bounds(self, min_age: int, max_age: int):
        self.min_age = min_age
        self.max_age = max_age
        self.A = max_age - min_age + 1
        
        self._make_loc()
        self._make_grid()
        self._make_basis()
        
    def _make_grid(self):
        if self.grid_type == 'age-age':
            X = age_age_grid(self.A)
        elif self.grid_type == 'diff-age':
            X = diff_age_age_grid(self.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")
        
        if self.symmetric:
            self.sym_tri_idx = symmetrize_from_lower_tri(self.A)
            
        self.x = np.sort(np.unique(X[:, 0]))
        self.y = np.sort(np.unique(X[:, 1]))
        
    def _define_knots(self, x: NDArray, n_knots: int, degree: int) -> NDArray:
        boundary_extension = (x.max() - x.min()) * 0.05
        x_quantiles = np.quantile(x, np.linspace(0, 1, n_knots))
        return np.hstack([
            [x.min() - boundary_extension] * (degree + 1),
            x_quantiles,
            [x.max() + boundary_extension] * (degree + 1)
        ])
    
    def tensor_spline_basis(self,
                      x: NDArray,
                      y: NDArray,
                      n_knots: list[int],
                      degree: list[int]) -> NDArray:
        
        x_knots = self._define_knots(x, n_knots[0], degree[0])
        y_knots = self._define_knots(y, n_knots[1], degree[1])
        
        PHI1 = BSpline(x_knots, np.eye(len(x_knots) - degree[0] - 1), degree[0])(x)[:, 1:]
        PHI2 = BSpline(y_knots, np.eye(len(y_knots) - degree[1] - 1), degree[1])(y)[:, 1:]
        
        return np.kron(PHI1, PHI2)
    
    def _make_basis(self):  
        self.basis = self.tensor_spline_basis(
            x=self.x,
            y=self.y,
            n_knots=self.n_knots_inner - self.n_knots_outer + 1, # Control the degree of freedom
            degree=self.degree
        )
        if self.symmetric:
            self.ltri_idx = lower_tri_indices(self.A)
            self.basis = self.basis[self.ltri_idx]
        
        self.basis_transpose = self.basis.T
        
    def sample(self):
        if self.event_dim == 1:
            with numpyro.plate('coef', self.basis.shape[-1]):
                beta = numpyro.sample('spline_coef', dist.Normal(0, self.coef_scale))
            
            f = beta @ self.basis_transpose
            f = f[self.sym_tri_idx] if self.symmetric else f
            
            return (self.loc + f).reshape((self.A, self.A), order='F')
        else:
            plate_event = numpyro.plate('event', self.event_dim_eff, dim=-2)
            plate_coef = numpyro.plate('coef', self.basis.shape[-1], dim=-1)
            with plate_event, plate_coef:
                beta = numpyro.sample('coef', dist.Normal(0, self.coef_scale))
            
            f = beta @ self.basis_transpose
            f = f[self.sym_tri_idx] if self.symmetric else f
            f = (self.loc + beta @ self.basis_transpose).reshape((self.event_dim_eff, self.A, self.A), order='F')
            
            if self.transform == 'alr':
                return inverse_alr(f, axis=1)
            elif self.transform == 'clr':
                return inverse_clr(f, axis=1)
            elif self.transform == 'ilr':
                return inverse_ilr(f, axis=1)
            
            
            