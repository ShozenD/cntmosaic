import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

from statsmodels.gam.smooth_basis import BSplines

import numpyro
from numpyro import distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern

class HSGP:
    """Sample from a Hilbert space approximate Gaussian process (HSGP).
    
    Parameters
    ----------
    x: jnp.ndarray
		    The input data.
    alpha: float
		    The smoothness parameter of the Matern kernel.
    rho: float
		    The length scale of the Matern kernel.
    L: int
		    The boundary condition of the approximation.
    M: int
		    The number of eigenfunctions to use.
    sym_tri_idx: NDArray, optional
		    The indices to symmetrize a lower triangular matrix to a symmetric matrix.
    """
    def __init__(self,
                 x: NDArray,
                 L: float | list[float],
                 M: int | list[int],
                 sym_tri_idx: NDArray=None):
        self.x = x
        self.L = L
        self.M = M
        self.sym_tri_idx = sym_tri_idx
        self.compute_eigenfunctions()
    
    def compute_eigenfunctions(self):
        self.eigenfunctions = eigenfunctions(x=self.x, ell=self.L, m=self.M)
  
    def sample(self, alpha: float, rho: float):
        """Sample from the HSGP.
        
        Parameters
        ----------
        alpha: float
			    The marginal variance of the Matern kernel.
		    rho: float
			    The length scale of the Matern kernel.
		    sym_tri_idx: NDArray, optional
			    The indices to symmetrize a lower triangular matrix to a symmetric matrix.
        """
        spd = jnp.sqrt(
          diag_spectral_density_matern(
            nu=5/2, alpha=alpha, length=rho, ell=self.L, m=self.M, dim=2
          )
        )
        
        with numpyro.plate('hsgp_basis_coef', self.eigenfunctions.shape[-1]):
            beta = numpyro.sample('beta', dist.Normal(0, 1))
        
        f = self.eigenfunctions @ (spd * beta)
        
        # Handle symmetric matrices if necessary
        if hasattr(self, 'sym_tri_idx'):
            return f[self.sym_tri_idx]
        else:
            return f
  
class TensorSplines:
    def __init__(self,
                 x: NDArray,
                 df: int=30,
                 degree: int=3):
        self.x = x
        self.A = len(x)
        self.df = df
        self.degree = degree
        self.compute_basis()
    
    def tspline_basis(self,
                      x: NDArray,
                      df: int=30,
                      degree: int=3,
                      include_intercept: bool=False):
        bs = BSplines(x, df=df, degree=degree, include_intercept=include_intercept)
        
        return np.kron(bs.basis, bs.basis)
    
    def compute_basis(self):
        self.basis = self.tspline_basis(self.x, df=self.df, degree=self.degree).T
        
    def sample(self, K: int):
        plate_x = numpyro.plate('x', K, dim=-2)
        plate_tspline = numpyro.plate('tspline', self.basis.shape[0], dim=-1)
        with plate_x:
            alpha = numpyro.sample('tspline_intercept', dist.Normal(0, 0.1))
        with plate_x, plate_tspline:
            beta = numpyro.sample('tspline_coef', dist.Normal(0, 0.1))
        
        return (alpha + beta @ self.basis).reshape((K, self.A, self.A))