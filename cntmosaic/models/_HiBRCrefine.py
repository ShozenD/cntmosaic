import numpy as np
from numpy.typing import NDArray
import pandas as pd
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.handlers import plate, scope

from ._BRCrefine import BRCrefine
from ._utils import index_mask_logsumexp
  
def expand_idarr(id, length):
    return np.repeat(id[:,np.newaxis], length, axis=1)

class HiBRCrefine(BRCrefine):
    """High-resolution Bayesian Rate Consistency model with fine age inputs.
    
    Parameters
    ----------
    data: DataFrame
        DataFrame containing the contact data.
        Must contain the columns 'y', 'age_part', 'age_cnt', and additional stratification variables.
        'y' is the number of contacts between 'age_part' and 'age_cnt'.
        'age_part' is the age of the contacting individual.
        'age_cnt' is the age of the contacted individual.
    age_dist: NDArray
        The population level age distribution.
    age_dist_props: dict
        Dictionary of NDArrays which containing the ratios of the population age distribution for each stratification variable.
        The NDArrays must be of shape (event_dim, A) or (event_dim, A, A).
    priors: dict, optional
        Dictionary containing the priors for the components of the model. If None, default priors are used.
    offset: NDArray, optional
        Additional offset to be multiplied to the contact intensity.
    likelihood: str, default='negbin'
        Likelihood function to use. Options are 'poisson' and 'negbin'.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 age_dist: NDArray,
                 age_dist_props: dict,
                 priors: dict,
                 likelihood: str='negbin'):
        
        super().__init__(data, age_dist, priors, likelihood)
            
        self.log_N = jnp.log(self.N)
        self.log_S = jnp.log(self.S)
        self.X_vars = [key for key in priors.keys() if key != 'rate']
        self.set_log_age_dist_props(age_dist_props)
        
        # --- Setup indices ---        
        self.aid_exp, self.bid_pad = make_idarrs_for_intervals(self.data, 'age_grp_cnt', self.aid)
        self.X_ids_exp = {
          c: expand_idarr(self.data[c].cat.codes.values, self.bid_pad.shape[1])
          for c in self.X_vars
        }
        
    # Compute the log of the age distribution proportions
    def set_log_age_dist_props(self, age_dist_props):
        self.log_age_dist_props = {}
        for k, v in age_dist_props.items():
            if age_dist_props[k].shape == (self.priors[k].event_dim, self.A):
                self.log_age_dist_props[k] = jnp.log(v)[:,:,jnp.newaxis]
            elif age_dist_props[k].shape == (self.priors[k].event_dim, self.A, self.A):
                self.log_age_dist_props[k] = jnp.log(v)
            else:
                raise ValueError(f"Invalid shape for age_dist_props[{k}].")
    
    def sample_log_delta(self, var):
        log_delta = numpyro.deterministic(
            'log_delta',
            jnp.log(self.priors[var].sample()) - self.log_age_dist_props[var]
        )
        return log_delta

    def model(self):
        beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
        with scope(prefix='rate'):
            f = self.priors['rate'].sample()
        log_rate = numpyro.deterministic('log_rate', beta0 + f)
        
        log_cint_base = log_rate + self.log_P # Precompute the base contribution
        log_cint = jnp.zeros(self.y.shape[0]) # Initialize the contact intensity
        for var in self.X_vars:
            with scope(prefix=var):
                contribution = index_mask_logsumexp(
                    log_cint_base + self.sample_log_delta(var),
                    self.aid_exp,
                    self.bid_pad,
                    self.X_ids_exp[var]
                )
                log_cint += contribution
        
        mu = jnp.exp(log_cint + self.log_N + self.log_S)
        with plate('data', len(self.y)):
            if self.likelihood == 'poisson':
                numpyro.sample('obs', dist.Poisson(rate=mu), obs=self.y) 
                
            if self.likelihood == 'negbin':
                inv_disp = numpyro.sample('inv_disp', dist.Exponential(1))
                numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                             concentration=1/inv_disp), 
                                obs=self.y)