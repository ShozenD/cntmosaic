from numpy.typing import NDArray 
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from ._BRC import BRC
from ._utils import index_mask_logsumexp
from .priors import HSGP2D

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
  
  # Default priors
  default_priors = {
    'rate': HSGP2D(
      grid_type='diff-age',
      prior_type='global',
      symmetric=True
    )
  }
  
  def __init__(self,
               dataloader: pd.DataFrame,
               priors: dict=None,
               likelihood: str='negbin'):
    
    # [Do] Update defualt priors if users provide their own
    if priors is not None:
      for key in priors.keys():
          self.default_priors[key] = priors[key]
    
    super().__init__(dataloader, self.default_priors, likelihood)
      
    self.y = jnp.array(self.ds.y.values)
    self.log_N = jnp.array(self.ds.log_N.values)
    self.log_P = jnp.array(self.ds.log_P.values)[jnp.newaxis,:]
    self.log_S = jnp.array(self.ds.log_S) if hasattr(self.ds, 'log_S') else np.ones_like(self.y)
    
    self.aid = jnp.array(self.ds.aid.values)
    self.aid_exp = jnp.array(self.ds.aid_exp.values)
    self.bid_pad = jnp.array(self.ds.bid_pad.values)
    
  def _validate_inputs(self):
    """
    This method validates the inputs which are specific to the BRCrefine model.
    Specifically, it checks if the data contains all the required ingredients:
    aid_exp, bid_pad, log_N, and log_P
    """
    if not hasattr(self.ds, 'aid_exp'):
      raise ValueError("Expanded participant age indexes (aid_exp) are missing.")
    if not hasattr(self.ds, 'bid_pad'):
      raise ValueError("Padded contact age indexes (bid_pad) are missing.")
    if not hasattr(self.ds, 'log_N'):
      raise ValueError("Log of sample size (log_N) is missing.")
    if not hasattr(self.ds, 'log_P'):
      raise ValueError("Log of population age distribution (log_P) is missing.")
    
  def model(self):
    beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
    f = self.priors['rate'].sample()
    log_rate = numpyro.deterministic('log_rate', beta0 + f)
    log_cint = numpyro.deterministic('log_cint', log_rate + self.log_P)
    log_cint = index_mask_logsumexp(log_cint, self.aid_exp, self.bid_pad) # [Do] sum across contact age groups
    
    mu = jnp.exp(log_cint + self.log_N + self.log_S)
    with numpyro.plate('data', len(self.y)):
      if self.likelihood == 'poisson':
        numpyro.sample('obs', dist.Poisson(rate=mu), obs=self.y)
        
      if self.likelihood == 'negbin':
        inv_disp = numpyro.sample('inv_disp', dist.Exponential(1))
        numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                     concentration=1/inv_disp),
                       obs=self.y)