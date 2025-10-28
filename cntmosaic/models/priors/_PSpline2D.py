from numpy.typing import NDArray
import numpy as np
import jax.numpy as jnp
from jax import vmap
import numpyro
from numpyro import distributions as dist

from ..funcs._IGMRF import make_igmrf2d_operator
from ._Spline2D import Spline2D

from .._utils import (
    age_age_grid,
    diff_age_age_grid,
    symm_from_tril_ix_row
)

from .._math import (
    inverse_alr,
    inverse_clr,
    inverse_ilr
)

class PSpline2D(Spline2D):
    """Sample from a 2 dimensional penalised spline.
    
    Parameters
    ----------
    M: int | list[int], default=30
        The number of basis functions to use in each dimension.
    degree: int | list[int], default=3
        The degree of the B-splines.
    order: int, default=1
        The number of neighborhood orders to consider in the Gaussian Markov random field.
    grid_type: str, default='age-age'
        The type of grid to use. Options are 'age-age' and 'diff-age'.
    transform: str, default=None
        The transformation to apply to the output tensor. Options are 'alr', 'clr', and 'ilr'.
    symmetric: bool, default=False
        Whether to symmetrize the output matrix/tensor.
        
    Examples
    --------

    >>> priors = {'rate': PSpline2D()}
    """
    
    pytree_aux_fields = ("self.PHI", "order",)
    
    def __init__(
        self,
        prior_type: str,
        M: int | list[int]=30,
        degree: int | list[int]=3,
        order: int=1,
        tau_shape: float=1.0,
        tau_rate: float=0.01,
        tau_ratio: float=1.0,
        grid_type: str='age-age',
        transform: str='ilr',
        tau_init: float=1.0  # Add initialization value for tau
    ):
        super().__init__(prior_type, M, degree, grid_type, transform)
        self.order = order
        self.tau_shape = tau_shape
        self.tau_rate = tau_rate
        self.tau_ratio = tau_ratio
        self.tau_init = tau_init  # Store initialization value
        
        self.igmrf_operator = make_igmrf2d_operator(
            (self.M, self.M), (self.order, self.order)
        )
        
    def set_age_bounds(self, min_age, max_age):
        return super().set_age_bounds(min_age, max_age)
    
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

        self.symm_tril_idx = symm_from_tril_ix_row(self.A)

    def sample(self):
        """Sample from the penalised tensor spline prior."""
        
        if self.prior_type == 'global':
            Z = numpyro.sample(
                'spline_coef',
                dist.Normal(0., 1./self.order),
                sample_shape=(self.M, self.M)
            )
            tau = numpyro.sample('spline_tau', dist.Gamma(self.tau_shape, self.tau_rate))
            beta = self.igmrf_operator(Z, tau, tau * self.tau_ratio).flatten()  # (M*M,)
            f = (self.PHI @ beta)[self.symm_tril_idx].reshape((self.A, self.A))
            return f

        elif self.prior_type == 'partial':
            # with numpyro.plate('event', self.event_dim_eff):
            #     beta = numpyro.sample('spline_coef', dist.CAR(0, 0.999, 1/self.coef_scale, self.adj_matrix, is_sparse=True))
            Z = numpyro.sample(
                'spline_coef',
                dist.Normal(0, 1),
                sample_shape=(self.event_dim_eff, self.M, self.M)
            )
            tau = numpyro.sample(
                'spline_tau', dist.Gamma(self.tau_shape, self.tau_rate),
                sample_shape=(self.event_dim_eff,)
            )
            tau = jnp.maximum(tau, 1e-3)  # Ensure tau is not too small
            beta = vmap(
                self.igmrf_operator,
                in_axes=(0, 0, 0),
                out_axes=2
            )(Z, tau, tau * self.tau_ratio) # (M, M, event_dim_eff)
            beta = beta.reshape((self.M * self.M, self.event_dim_eff))  # (M*M, event_dim_eff)

            f = self.PHI @ beta  # (A*A, event_dim_eff)
            f = f.swapaxes(0, 1)
            f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A))
        
        elif self.prior_type == 'full':
            # Diagonal matrices
            Z_diag = numpyro.sample(
                'spline_coef_diag',
                dist.Normal(0, 1),
                sample_shape=(self.event_dim_diag, self.M, self.M)
            )
            tau_diag = numpyro.sample(
                'spline_tau_diag', dist.Gamma(self.tau_shape, self.tau_rate),
                sample_shape=(self.event_dim_diag,)
            )
            tau_diag = jnp.maximum(tau_diag, 1e-3)  # Ensure tau is not too small
            
            beta_diag = vmap(
                self.igmrf_operator,
                in_axes=(0, 0, 0),
                out_axes=2
            )(Z_diag, tau_diag, tau_diag * self.tau_ratio) # shape: (M, M, event_dim_diag)
            beta_diag = beta_diag.reshape((self.M * self.M, self.event_dim_diag))  # shape: (M*M, event_dim_diag)
            f_diag = self.PHI_diag @ beta_diag
            f_diag = f_diag[self.symm_tril_idx, :].swapaxes(0, 1) # Must be symmetric
            
            Z_non_diag = numpyro.sample(
                'spline_coef_non_diag',
                dist.Normal(0, 1),
                sample_shape=(self.event_dim_non_diag, self.M, self.M)
            )
            tau_non_diag = numpyro.sample(
                'spline_tau_non_diag', dist.Gamma(self.tau_shape, self.tau_rate),
                sample_shape=(self.event_dim_non_diag,)
            )
            tau_non_diag = jnp.maximum(tau_non_diag, 1e-3)  # Ensure tau is not too small

            beta_non_diag = vmap(
                self.igmrf_operator,
                in_axes=(0, 0, 0),
                out_axes=2
            )(Z_non_diag, tau_non_diag, tau_non_diag * self.tau_ratio) # shape: (M, M, event_dim_non_diag)
            beta_non_diag = beta_non_diag.reshape((self.M * self.M, self.event_dim_non_diag))  # shape: (M*M, event_dim_non_diag)
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

            # Reshape to (event_dim_eff, A, A)
            f = self.trans_loc + f.reshape((self.event_dim_eff, self.A, self.A))
        
        if self.transform == 'alr':
            return inverse_alr(f, axis=0)
        elif self.transform == 'clr':
            return inverse_clr(f, axis=0)
        elif self.transform == 'ilr':
            return inverse_ilr(f, axis=0)
        else:
            return f