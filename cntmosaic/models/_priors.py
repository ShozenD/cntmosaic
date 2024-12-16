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
  
class TensorSplines2D:
    """Sample from a 2 dimensional tensor product of B-splines.
    
    Parameters
    ----------
    x: NDArray
        The input points in the first dimension
    y: NDArray, optional
        The input points in the second dimension. Default is the same as x.
    loc: float | NDArray, optional
        The prior mean of the tensor spline. Default is 0.
    M: int | list[int], optional
        The number of basis functions to use in each dimension. 
    degree: int | list[int], optional
        The degree of the B-splines for each dimension.
    
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
                 y: NDArray=None,
                 M: int | list[int]=30,
                 degree: int | list[int]=3):
        
        # Implement a function to check the inputs
        
        self.x = x
        self.y = y if y is not None else x
        self.Nx = len(x)
        self.Ny = len(self.y)
        
        self.M = np.array([M]*2) if isinstance(M, int) else np.array(M)
        self.degree = np.array([degree]*2) if isinstance(degree, int) else np.array(degree)
        self.n_knots_inner = self.M + self.degree + 1
        self.n_knots_outer = 2 * (self.degree + 1)
        self.compute_basis()
        
    def define_knots(self, x: NDArray, n_knots: int, degree: int) -> NDArray:
        boundary_extension = (x.max() - x.min()) * 0.05
        x_quantiles = np.quantile(x, np.linspace(0, 1, n_knots))
        return np.hstack([
            [x.min() - boundary_extension] * (degree + 1),
            x_quantiles,
            [x.max() + boundary_extension] * (degree + 1)
        ])
    
    def tspline_basis(self,
                      x: NDArray,
                      y: NDArray,
                      n_knots: list[int],
                      degree: list[int]) -> NDArray:
        
        x_knots = self.define_knots(x, n_knots[0], degree[0])
        y_knots = self.define_knots(y, n_knots[1], degree[1])
        PHI1 = BSpline(x_knots, np.eye(len(x_knots) - degree[0] - 1), degree[0])(x)[:, 1:]
        PHI2 = BSpline(y_knots, np.eye(len(y_knots) - degree[1] - 1), degree[1])(y)[:, 1:]
        
        return np.kron(PHI1, PHI2)
    
    def compute_basis(self):
        self.basis = self.tspline_basis(
            x=self.x,
            y=self.y,
            n_knots=self.n_knots_inner - self.n_knots_outer + 1, # Control the degree of freedom
            degree=self.degree
        )
        self.basis_transpose = self.basis.T
        
    def sample(self,
               loc: int | float | NDArray,
               coef_scale: float | NDArray=1,
               intercept_scale: float=1,
               event_dim: int=1,
               intercept: bool=False):
        
        plate_f = numpyro.plate('f', self.basis.shape[0], dim=-1)
        if intercept:
            if isinstance(loc, (int, float)):
                plate_intcpt = numpyro.plate('x', event_dim, dim=-2)
                with plate_intcpt:
                    alpha = numpyro.sample('tsp_intcpt', dist.Normal(loc, intercept_scale))
            else:
                loc = loc[jnp.newaxis,:]
                alpha = numpyro.sample('tsp_intcpt', dist.Normal(loc, intercept_scale))
                
            with plate_intcpt, plate_f:
                beta = numpyro.sample('tsp_coef', dist.Normal(0, coef_scale))
        
            return (alpha + beta @ self.basis_transpose).reshape((event_dim, self.Nx, self.Ny))
        else:
            if isinstance(loc, (int, float)):
                loc = jnp.array([loc])
            
            with plate_f:
                beta = numpyro.sample('tsp_coef', dist.Normal(0, coef_scale))
            return (loc[jnp.newaxis,:] + beta @ self.basis_transpose).reshape((event_dim, self.Nx, self.Ny))
    
class PenalisedTensorSplines2D(TensorSplines2D):
    """Sample from a penalised tensor product spline.
    
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
                 y: NDArray=None,
                 M: int | list[int]=30,
                 degree: int | list[int]=3,
                 neighborhood: int=4):
        super().__init__(x, y, M, degree)
        self.adj_matrix = gmrf_adjacency_matrix(M, M, neighborhood)
    
    def sample(self,
               loc: int | float | NDArray,
               coef_scale: float | NDArray=1,
               intercept_scale: float=1,
               event_dim: int=1,
               intercept: bool=False):
        
        plate_f = numpyro.plate('f', event_dim, dim=-1)
        if intercept:
            if isinstance(loc, (int, float)):
                plate_intcpt = numpyro.plate('x', event_dim, dim=-2)
                with plate_intcpt:
                    alpha = numpyro.sample('tsp_intcpt', dist.Normal(loc, intercept_scale))
            else:
                loc = loc[jnp.newaxis,:]
                alpha = numpyro.sample('tsp_intcpt', dist.Normal(loc, intercept_scale))
                
            with plate_f:
                beta = numpyro.sample('tsp_coef', dist.CAR(0, 0.999, 1/coef_scale, self.adj_matrix, is_sparse=True))
        
            return (alpha + beta @ self.basis_transpose).reshape((event_dim, self.Nx, self.Ny))
        else:
            if isinstance(loc, (int, float)):
                loc = jnp.array([loc])
            
            with plate_f:
                beta = numpyro.sample('tsp_coef', dist.CAR(0, 0.999, 1/coef_scale, self.adj_matrix, is_sparse=True))
            return (loc[jnp.newaxis,:] + beta @ self.basis_transpose).reshape((event_dim, self.Nx, self.Ny))
    