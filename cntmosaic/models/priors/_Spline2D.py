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

def validate_init_params(M: int | list[int], degree: int | list[int]):
    """Validates additional input parameters for the Spline2D class."""
    
    if isinstance(M, int) and M <= 0:
        raise ValueError("M must be greater than 0")
    elif isinstance(M, list) and (len(M) == 1 or len(M) > 2):
        raise ValueError("M must be scalar or a list of length 2")
    
    if isinstance(degree, int) and degree <= 0:
        raise ValueError("degree must be greater than 0")
    elif isinstance(degree, list) and len(degree) > 2:
        raise ValueError("degree must be a list of length 2")

class Spline2D(Prior2D):
    """Sample from a 2 dimensional tensor product of B-splines.
    
    Parameters
    ----------
    prior_type: str
        The type of prior to use. Either 'global', 'partial', or 'full'.
    M: int | list[int], default=30
        The number of basis functions to use in each dimension.
    degree: int | list[int], default=3
        The degree of the B-splines for each dimension.
    grid_type: str, default='age-age'
        The type of grid to use. Either 'age-age' or 'diff-age'.
    transform: str, optional
        The transformation to apply to the output tensor. Either 'alr', 'clr', or 'ilr'.
    
    Examples
    --------
    >>> priors = {'rate' = Spline2D('global')}
    """
    pytree_aux_fields = ("self.PHI", "self.PHI_T", "self.tril_idx", "self.symm_tril_idx")
    
    def __init__(
        self,
        prior_type: str,
        M: int = 30,
        degree: int = 3,
        grid_type: str='age-age',
        transform: str='ilr'
    ):

        validate_init_params(M, degree)
        super().__init__(grid_type, transform, prior_type)
        self.M = M             # Number of basis functions (same for both dimensions)
        self.degree = degree   # Degree of B-splines (same for both dimensions)
        self.n_knots_inner = M + degree + 1
        self.n_knots_outer = 2 * (degree + 1)

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
        
        self.symm_tril_idx = symm_from_tril_indices_row(self.A)
        
    def _define_knots(self, x: NDArray, n_knots: int, degree: int) -> NDArray:
        boundary_extension = (x.max() - x.min()) * 0.05
        x_quantiles = np.quantile(x, np.linspace(0, 1, n_knots))
        return np.hstack([
            [x.min() - boundary_extension] * (degree + 1),
            x_quantiles,
            [x.max() + boundary_extension] * (degree + 1)
        ])
    
    def tensor_spline_basis(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_knots: list[int],
        degree: list[int]
    ) -> np.ndarray:
        
        x_knots = self._define_knots(x, n_knots, degree)
        y_knots = x_knots # Crucial that we use the same knots for both dimensions for anisotropic smoothing

        PHI1 = BSpline(x_knots, np.eye(len(x_knots) - degree - 1), degree)(x)[:, 1:] # shape: (len(x), M)
        PHI2 = BSpline(y_knots, np.eye(len(y_knots) - degree - 1), degree)(y)[:, 1:] # shape: (len(y), M)

        if self.grid_type == 'age-age':
            return np.kron(PHI1, PHI2)
        elif self.grid_type == 'diff-age':
            diff_age_idx = diff_age_age_index(self.A)
            return np.kron(PHI1, PHI2)[diff_age_idx]
    
    def _set_basis(self):
        """Sets the basis matrices based on the prior type."""
        n_basis_functions = self.n_knots_inner - self.n_knots_outer + 1 # Clear variable name
        self.PHI = self.tensor_spline_basis(self.x, self.y, n_basis_functions, self.degree)
        
        if self.prior_type == 'global':
            self.symm_tril_idx = symm_from_tril_indices_row(self.A)
            self.PHI = self.PHI[tril_indices_row(self.A)]
            
        elif self.prior_type == 'full': # Full case
            self.symm_tril_idx = symm_from_tril_indices_row(self.A)
            self.PHI_diag = self.PHI[tril_indices_row(self.A)]
            self.PHI_non_diag = self.PHI
        
    def sample(self):
        """Samples from the tensor spline prior."""
        
        if self.prior_type == 'global':
            beta = numpyro.sample(
                'spline_coef',
                dist.Normal(0,1),
                sample_shape=(self.PHI.shape[-1],)
            )
            f = (self.PHI @ beta)[self.symm_tril_idx].reshape((self.A, self.A))
            return f
        
        elif self.prior_type == 'partial':
            beta = numpyro.sample(
                'spline_coef',
                dist.Normal(0,1),
                sample_shape=(self.PHI.shape[-1], self.event_dim_eff)
            )
            f = self.PHI @ beta
            f = f.swapaxes(0, 1)
            f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A))
                
        elif self.prior_type == 'full':
            beta_diag = numpyro.sample(
                'spline_coef_diag',
                dist.Normal(0, 1),
                sample_shape=(self.PHI_diag.shape[-1], self.event_dim_diag)
            )
            beta_non_diag = numpyro.sample(
                'spline_coef_non_diag',
                dist.Normal(0, 1),
                sample_shape=(self.PHI_non_diag.shape[-1], self.event_dim_non_diag)
            )

            f_diag = self.PHI_diag @ beta_diag
            f_diag = f_diag[self.symm_tril_idx, :].swapaxes(0, 1)           # shape: (event_dim_diag, A**2)
            f_non_diag = (self.PHI_non_diag @ beta_non_diag).swapaxes(0, 1) # shape: (event_dim_non_diag, A**2)

            # Preallocate the output tensor
            f = jnp.zeros((self.event_dim_eff, self.A**2))

            # Allocate diagonal elements
            sqrt_event_dim = jnp.sqrt(self.event_dim).astype(int)
            diag_idx = jnp.array([(i * sqrt_event_dim + i) for i in range(sqrt_event_dim)])  # Flat index of (i,i) in row-major order
            all_idx = jnp.arange(self.event_dim)
            non_diag_idx = jnp.setdiff1d(all_idx, diag_idx)

            # Allocate elements
            if self.transform in ['alr', 'ilr']:
                f = f.at[diag_idx[:-1], :].set(f_diag) # Last element left out for log-ratio transformation
            else:
                f = f.at[diag_idx, :].set(f_diag)
            f = f.at[non_diag_idx, :].set(f_non_diag)

            # Reshape
            f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A))
                
        # Apply real-to-simplex transformation
        if self.transform == 'alr':
            return inverse_alr(f, axis=0)
        elif self.transform == 'clr':
            return inverse_clr(f, axis=0)
        elif self.transform == 'ilr':
            return inverse_ilr(f, axis=0)
        else:
            return f
