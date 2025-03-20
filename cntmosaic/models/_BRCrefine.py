from numpy.typing import NDArray 
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from ._BRC import BRC
from ._priors import HSGP
from ._utils import (
  age_age_grid,
  diff_age_age_grid,
  lower_tri_indices,
  fine_coarse_matrix
)


class BRCrefine(BRC):
  """Bayesian Rate Consistency model with fine participant age but coarse contacted age.

  Parameters
  ----------
  data: DataFrame
      DataFrame containing the contact data. Must contain the columns 'y', 'age_part', and 'age_grp_cnt.
      'y' is the number of contacts between 'age_part' and 'age_grp_cnt'.
      'age_part' is the age of the contactor.
      'age_grp_cnt' is the age group of the contacted.
  age_dist: NDArray
      The population age distribution.
  likelihood: str, default='negbin'
      Likelihood function to use.
      
  References
  ----------
  Shozen Dan et al., "Estimating fine age structure and time trends in 
  human contact patterns from coarse contact data: The Bayesian rate consistency model",
  PLoS Computational Biology. 2023
  """
  def __init__(self,
               data: pd.DataFrame,
               age_dist: NDArray,
               likelihood: str='negbin'):
    super().__init__(data, age_dist, likelihood)
    
    self.age_dist = age_dist
    self.set_hsgp_params()
        
    # Setup
    self.aid = self.data['age_part'].values
    self.cid = self.data['age_grp_cnt'].cat.codes.values
    self.fine_coarse_matrix = fine_coarse_matrix(self.data['age_grp_cnt'])
    
    self.y = self.data['y'].values
    self.N = self.data['N'].values
    self.S = self.data['S'].values if 'S' in self.data.columns else np.ones_like(self.y)
    self.log_P = jnp.log(self.age_dist)[jnp.newaxis,:]
    
  def set_age_dim(self, A):
    self.A = A
    self._compute_indices()
    self.set_hsgp_params()
    
  def set_hsgp_params(self,
                      M: list[int]=[30, 30],
                      C: list[float]=[1.5, 1.5],
                      grid_type: str='diff-age'):
    """Set the hyperparameters for the Hilbert space approximate Gaussian process prior.
    
    Parameters
    ----------
    M: int | list[int], default=[30, 30]
        Number of eigenfunctions to use.
        If int, the same number of eigenfunctions will be used for each dimension.
        If list, the number of eigenfunctions to use for each dimension.
    C: float | list[float], default=[1.5, 1.5]
        Scaling factor for the length scale.
        If float, the same scaling factor will be used for each dimension.
        If list, the scaling factor for each dimension
    grid_type: str, default='age-age'
        The type of grid to use for the input data.
        'age-age': age-age grid.
        'diff-age': difference-in-age by age grid.
        
    References
    ----------
    Shozen Dan et al., "Estimating fine age structure and time trends in
    human contact patterns from coarse contact data: The Bayesian rate consistency model",
    PLoS Computational Biology. 2023
    """
    self.M = M
    self.C = C
    
    if grid_type == 'age-age':
        self.grid_type = 'age-age'
        X = age_age_grid(self.A)
    elif grid_type == 'diff-age':
        self.grid_type = 'diff-age'
        X = diff_age_age_grid(self.A)
    else:
        raise ValueError("grid_type must be 'age-age' or 'diff-age'")
  
    ltri_idx = lower_tri_indices(self.A)
    Xn = (X - X.mean(axis=0)) / X.std(axis=0)
    self.L = list(np.abs(Xn).max(axis=0) * self.C)
    self.X = Xn[ltri_idx]
    
    self.hsgp = HSGP(self.X, self.L, M, self.sym_tri_idx)
    
  def model(self):
    beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
    alpha = numpyro.sample('hsgp_scale', dist.InverseGamma(5, 5))
    rho = numpyro.sample('hsgp_lenscale', dist.InverseGamma(5, 5).expand([2]))

    f = self.hsgp.sample(alpha, rho).reshape((self.A, self.A), order='F')
    log_rate = numpyro.deterministic('log_rate', beta0 + f)
    log_cint = numpyro.deterministic('log_cint', log_rate + self.log_P)
      
    mu = (jnp.exp(log_cint) @ self.fine_coarse_matrix)[self.aid, self.cid] * self.N * self.S
    with numpyro.plate('data', len(self.y)):
      if self.likelihood == 'poisson':
        numpyro.sample('obs', dist.Poisson(rate=mu), obs=self.y)
        
      if self.likelihood == 'negbin':
        inv_disp = numpyro.sample('inv_disp', dist.Exponential(1))
        numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                     concentration=inv_disp),
                       obs=self.y)