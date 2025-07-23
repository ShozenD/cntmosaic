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
    diff_age_age_index,
    tril_indices_row,
    symm_from_tril_indices_row
)

from .._math import (
    inverse_alr,
    inverse_clr,
    inverse_ilr,
)

def validate_init_params(M: int | list[int], degree: int | list[int], coef_scale: float | NDArray):
    """Validates additional input parameters for the TensorSpline2D class."""
    
    if isinstance(M, int) and M <= 0:
        raise ValueError("M must be greater than 0")
    elif isinstance(M, list) and (len(M) == 1 or len(M) > 2):
        raise ValueError("M must be scalar or a list of length 2")
    
    if isinstance(degree, int) and degree <= 0:
        raise ValueError("degree must be greater than 0")
    elif isinstance(degree, list) and len(degree) > 2:
        raise ValueError("degree must be a list of length 2")
    
    if coef_scale <= 0:
        raise ValueError("coef_scale must be greater than 0")

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
    coef_scale: float | NDArray, default=1
        The scale of the spline coefficients.
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
    >>> priors = {'rate' = TensorSpline2D('global', symmetric=True)}
    """
    pytree_aux_fields = ("self.PHI", "self.PHI_T", "self.tril_idx", "self.symm_tril_idx")
    
    def __init__(self,
                 M: int | list[int]=30,
                 degree: int | list[int]=3,
                 grid_type: str='age-age',
                 coef_scale: float | NDArray=1,
                 transform: str | None=None,
                 prior_type: str='global',
                 symmetric: bool=False):
        
        validate_init_params(M, degree, coef_scale)
        super().__init__(grid_type, transform, prior_type)
        
        self.coef_scale = coef_scale
        self.M = np.array([M]*2) if isinstance(M, int) else np.array(M)
        self.degree = np.array([degree]*2) if isinstance(degree, int) else np.array(degree)
        self.n_knots_inner = self.M + self.degree + 1
        self.n_knots_outer = 2 * (self.degree + 1)
        self.symmetric = symmetric
        
    def set_age_bounds(self, min_age: int, max_age: int):
        self.min_age = min_age
        self.max_age = max_age
        self.A = max_age - min_age + 1
        
        self._set_grid()
        self._set_basis()
        
    def _set_grid(self):
        if self.grid_type == 'age-age':
            X = age_age_grid(self.A)
        elif self.grid_type == 'diff-age':
            X = diff_age_age_grid(self.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")

        x = np.sort(np.unique(X[:, 0]))
        y = np.sort(np.unique(X[:, 1]))
        
        # Scale x and y to [0, 1]
        self.x = (x - self.min_age) / (self.max_age - self.min_age)
        self.y = (y - self.min_age) / (self.max_age - self.min_age)
        
        self.symm_tril_idx = symm_from_tril_indices_row(self.A) if self.symmetric else None
        
    def _define_knots(self, x: NDArray, n_knots: int, degree: int) -> NDArray:
        boundary_extension = (x.max() - x.min()) * 0.05
        x_quantiles = np.quantile(x, np.linspace(0, 1, n_knots))
        return np.hstack([
            [x.min() - boundary_extension] * (degree + 1),
            x_quantiles,
            [x.max() + boundary_extension] * (degree + 1)
        ])
    
    def tensor_spline_basis(self,
                            x: np.ndarray,
                            y: np.ndarray,
                            n_knots: list[int],
                            degree: list[int]) -> np.ndarray:
        x_knots = self._define_knots(x, n_knots[0], degree[0])
        y_knots = self._define_knots(y, n_knots[1], degree[1])
        
        PHI1 = BSpline(x_knots, np.eye(len(x_knots) - degree[0] - 1), degree[0])(x)[:, 1:]
        PHI2 = BSpline(y_knots, np.eye(len(y_knots) - degree[1] - 1), degree[1])(y)[:, 1:]
        
        if self.grid_type == 'age-age':
            return np.kron(PHI1, PHI2)
        elif self.grid_type == 'diff-age':
            diff_age_idx = diff_age_age_index(self.A)
            return np.kron(PHI1, PHI2)[diff_age_idx]
    
    def _set_basis(self):
        """Sets the basis matrices based on the prior type."""
        n_basis_functions = self.n_knots_inner - self.n_knots_outer + 1 # Clear variable name
        self.PHI = self.tensor_spline_basis(self.x, self.y, n_basis_functions, self.degree)
        
        if self.type == 'global':
            self.tril_idx = tril_indices_row(self.A)
            self.symm_tril_idx = symm_from_tril_indices_row(self.A)
            self.PHI = self.PHI[self.tril_idx]
            self.PHI_T = self.PHI.T
        elif self.type == 'full': # Full case
            self.tril_idx = tril_indices_row(self.A)
            self.symm_tril_idx = symm_from_tril_indices_row(self.A)
            self.PHI_diag = self.PHI[self.tril_idx]
            self.PHI_non_diag = self.PHI
            self.PHI_diag_T = self.PHI_diag.T
            self.PHI_non_diag_T = self.PHI_non_diag.T
        else: # partial case
            self.PHI_T = self.PHI.T
        
    def sample(self):
        """Samples from the tensor spline prior."""
        
        if self.type == 'global':
            with numpyro.plate('coef', self.PHI.shape[-1]):
                beta = numpyro.sample('spline_coef', dist.Normal(0, self.coef_scale))
            
            f = beta @ self.PHI_T
            f = f[self.symm_tril_idx] if self.symmetric else f
            return f.reshape((self.A, self.A))
        
        elif self.type == 'partial':
            plate_event = numpyro.plate('event', self.event_dim_eff, dim=-2)
            plate_coef = numpyro.plate('coef', self.PHI.shape[-1], dim=-1)
            with plate_event, plate_coef:
                beta = numpyro.sample('coef', dist.Normal(0, self.coef_scale))

            f = beta @ self.PHI_T
            f = f[:,self.symm_tril_idx] if self.symmetric else f
            f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A))
                
        elif self.type == 'full':
            plate_diag = numpyro.plate('diag', self.event_dim_diag, dim=-2)
            plate_non_diag = numpyro.plate('non_diag', self.event_dim_non_diag, dim=-2)
            plate_coef = numpyro.plate('coef', self.PHI_diag.shape[-1], dim=-1)
            
            with plate_diag, plate_coef:
                beta_diag = numpyro.sample('coef_diag', dist.Normal(0, self.coef_scale))
                
            with plate_non_diag, plate_coef:
                beta_non_diag = numpyro.sample('coef_non_diag', dist.Normal(0, self.coef_scale))
            
            f_diag = beta_diag @ self.PHI_diag_T
            f_diag = f_diag[:,self.symm_tril_idx] # Must be symmetric
            
            f = beta_non_diag @ self.PHI_non_diag_T
            
            # Insert the diagonal elements into the non-diagonal elements
            # to complete the tensor.
            for i in range(self.event_dim_diag):
                f = jnp.insert(f, (i+1)**2 - 1, f_diag[i,:], axis=0)
                
            f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A))
        else:
            raise ValueError("Unknown prior type")
                
        # Apply real-to-simplex transformation
        if self.transform == 'alr':
            return inverse_alr(f, axis=0)
        elif self.transform == 'clr':
            return inverse_clr(f, axis=0)
        elif self.transform == 'ilr':
            return inverse_ilr(f, axis=0)
        else:
            return f
            
                
                
            
            