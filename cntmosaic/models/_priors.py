import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

from scipy.interpolate import BSpline

import numpyro
from numpyro import distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern

from ._utils import gmrf_adjacency_matrix

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
    """Sample from a tensor product of B-splines.
    
    Parameters
    ----------
    x: NDArray
        The input data.
    df: int
        The degrees of freedom. The number of basis functions is df - 1.
    degree: int
        The degree of the B-splines.
    
    Attributes
    ----------
    basis: NDArray
        The tensor product of B-splines.
    
    Examples
    --------
    >>> x = np.linspace(0, 1, 100)
    >>> ts = TensorSplines(x, df=30, degree=3)
    """
    def __init__(self,
                 x: NDArray,
                 df: int=30,
                 degree: int=3):
        self.x = x
        self.A = len(x)
        self.df = df
        self.degree = degree
        self.n_knots_inner = df + degree + 1
        self.n_knots_outer = 2 * (degree + 1)
        self.compute_basis()
    
    def tspline_basis(self,
                      x: NDArray,
                      n_knots: int,
                      degree: int) -> NDArray:
        
        boundary_extension = (x.max() - x.min()) * 0.05
        x_quantiles = np.quantile(x, np.linspace(0, 1, n_knots))
        padded_knots = np.hstack([
            [x.min() - boundary_extension] * (degree + 1),
            x_quantiles,
            [x.max() + boundary_extension] * (degree + 1)
        ])
        basis = BSpline(padded_knots, np.eye(len(padded_knots) - degree - 1), degree)(x)[:, 1:]
        
        return np.kron(basis, basis)
    
    def compute_basis(self):
        self.basis = self.tspline_basis(self.x,
                                        self.n_knots_inner - self.n_knots_outer + 1, # Control the degree of freedom
                                        self.degree)
        self.basis_transpose = self.basis.T
        
    def sample(self, K: int):
        plate_x = numpyro.plate('x', K, dim=-2)
        plate_tspline = numpyro.plate('tspline', self.basis.shape[0], dim=-1)
        with plate_x:
            alpha = numpyro.sample('tspline_intercept', dist.Normal(0, 0.1))
        with plate_x, plate_tspline:
            beta = numpyro.sample('tspline_coef', dist.Normal(0, 0.1))
        
        return (alpha + beta @ self.basis_transpose).reshape((K, self.A, self.A))
    
class TensorPSplines(TensorSplines):
    """Sample from a tensor product of P-splines.
    
    Parameters
    ----------
    x: NDArray
        The input data.
    df: int
        The degrees of freedom. The number of basis functions is df - 1.
    degree: int
        The degree of the B-splines.
    neighborhood: int
        The number of neighbors to consider in the GMRF. Default is 4.
    cond_prec: float
        The conditional precision of the GMRF. Default is 0.1.    
        
    Examples
    --------
    
    >>> x = np.linspace(0, 1, 100)
    >>> tps = TensorPSplines(x, df=30, degree=3)
    """
    def __init__(self,
                 x: NDArray,
                 df: int=30,
                 degree: int=3,
                 neighborhood: int=4,
                 cond_prec: float=10):
        super().__init__(x, df, degree)
        self.cond_prec = cond_prec
        self.adj_matrix = gmrf_adjacency_matrix(df, df, neighborhood)
        
    def sample(self, K: int):
        with numpyro.plate('x_alpha', K, dim=-2):
            alpha = numpyro.sample('tpspline_intercept', dist.Normal(0, 0.1))
        with numpyro.plate('x_beta', K, dim=-1):
            beta = numpyro.sample('tpspline_coef', dist.CAR(0, 0.999, self.cond_prec, self.adj_matrix, is_sparse=True))
        
        return (alpha + beta @ self.basis_transpose).reshape((K, self.A, self.A))
    