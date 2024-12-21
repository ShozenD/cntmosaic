import numpy as np
from numpy.typing import NDArray
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.handlers import plate, scope

from ._BRCfine import BRCfine
from ._math import (
    alr,
    ilr,
    log_inverse_alr,
    log_inverse_ilr,
)

class HiBRCfine(BRCfine):
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
                 priors: dict,
                 likelihood: str='negbin'):
        
        super().__init__(data, age_dist, priors, likelihood)
        
        self.X_vars = self.data.select_dtypes(include='category').columns
        self.X_ids = {c: self.data[c].cat.codes.values for c in self.X_vars}        
        self.log_age_dist_props = {k: np.log(v).T for k, v in age_dist_props.items()}
    
    def sample_log_delta(self, var):
        log_delta = numpyro.deterministic(
            'log_delta',
            jnp.log(self.priors[var].sample()) - self.log_age_dist_props[var][:,:,None]
        )
        return log_delta

    def model(self):
        beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
        with scope(prefix='rate'):
            f = self.priors['rate'].sample()
        log_rate = numpyro.deterministic('log_rate', beta0 + f)
        log_cint = (log_rate + self.log_P)[self.aid, self.bid]

        for var in self.X_vars:
            with scope(prefix=var):
                log_cint += self.sample_log_delta(var)[self.X_ids[var], self.aid, self.bid]
        
        mu = jnp.exp(log_cint + self.log_N + self.log_S)
        with plate('data', len(self.y)):
            if self.likelihood == 'poisson':
                numpyro.sample('obs', dist.Poisson(rate=mu), obs=self.y) 
                
            if self.likelihood == 'negbin':
                inv_disp = numpyro.sample('inv_disp', dist.Exponential(1))
                numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                            concentration=inv_disp), 
                                obs=self.y)