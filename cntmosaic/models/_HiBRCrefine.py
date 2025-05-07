import numpy as np
from numpy.typing import NDArray
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.handlers import plate, scope

from ._BRCrefine import BRCrefine
from ._math import (
    alr,
    ilr,
    log_inverse_alr,
    log_inverse_ilr,
)
from ._utils import make_idarrs_for_intervals, index_mask_logsumexp
  
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
        'age_part' is the age of the contactor.
        'age_cnt' is the age of the contacted.
    age_dist: NDArray
        The population level age distribution.
    age_dist_props: dict
        Dictionary containing the ratios of the population age distribution for each stratification variable.
    smoother_type: dict, optional
        Dictionary containing the type of smoother to use for each stratification variable.
    offset: NDArray, optional
        Additional offset to be multiplied to the contact intensity.
    likelihood: str, default='negbin'
        Likelihood function to use.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 age_dist: NDArray,
                 age_dist_props: dict,
                 smoothers: dict,
                 likelihood: str='negbin'):
        
        super().__init__(data, age_dist, likelihood)
            
        self.y = self.data['y'].values
        self.log_N = jnp.log(self.data['N'].values)
        self.log_S = jnp.log(self.data['S'].values) if 'S' in self.data.columns else jnp.zeros_like(self.y)
        self.log_P = jnp.log(self.age_dist)[jnp.newaxis,:]
        self.X_vars = self.data.drop(columns='age_grp_cnt').select_dtypes(include='category').columns
        self.X_dims = {x: len(self.data[x].cat.categories) for x in self.X_vars}
        
        # Compute the log of the age distribution proportions
        self.log_age_dist_props = {k: np.log(v).T for k, v in age_dist_props.items()}
        
        self.smoothers = smoothers
        
        # --- Setup indices ---        
        self.aid = self.data['age_part'].values
        self.aid_exp, self.bid_pad = make_idarrs_for_intervals(self.data, 'age_grp_cnt', self.aid)
        self.X_ids_exp = {
          c: expand_idarr(self.data[c].cat.codes.values, self.bid_pad.shape[1])
          for c in self.X_vars
        }
        
    def set_age_dim(self, A):
        self.A = A
        self._compute_indices()
        self.set_hsgp_params()
    
    def sample_log_delta(self, var):
        omega = self.smoothers[var].sample()
        log_delta = numpyro.deterministic(
            'log_delta',
            log_inverse_ilr(omega) - self.log_age_dist_props[var][:,:,None]
        )
        return log_delta

    def model(self):
        beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
        alpha = numpyro.sample('gp_scale', dist.HalfNormal(1.))
        rho = numpyro.sample('gp_lenscale', dist.InverseGamma(5., 5.))

        f = self.hsgp.sample(alpha, rho).reshape((self.A, self.A), order='F')
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
                                                            concentration=inv_disp), 
                                obs=self.y)