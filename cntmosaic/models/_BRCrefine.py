from numpy.typing import NDArray 
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from ._BRC import BRC
from ._utils import fine_coarse_matrix

class BRCrefine(BRC):
  def __init__(self,
               data: pd.DataFrame,
               age_dist: NDArray,
               priors: dist,
               likelihood: str='negbin'):
    super().__init__(data, age_dist, likelihood)
    
    self.age_dist = age_dist
    self.priors = priors
        
    # Setup
    self.aid = self.data['age_part'].values
    self.cid = self.data['age_grp_cnt'].cat.codes.values
    self.fine_coarse_matrix = fine_coarse_matrix(self.data['age_grp_cnt'])
    
    self.y = self.data['y'].values
    self.N = self.data['N'].values
    self.S = self.data['S'].values if 'S' in self.data.columns else np.ones_like(self.y)
    self.log_P = jnp.log(self.age_dist)[jnp.newaxis,:]
    
  def set_age_dim(self, A):
    pass
    
  def model(self):
    beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))

    f = self.priors['rate'].sample()
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