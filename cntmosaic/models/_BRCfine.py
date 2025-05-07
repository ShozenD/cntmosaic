from numpy.typing import NDArray 
import pandas as pd
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.handlers import scope

from ._BRC import BRC

class BRCfine(BRC):
    """Bayesian Rate Consistency model with fine age inputs.

    Parameters
    ----------
    data: DataFrame
        DataFrame containing the contact data. Must contain the columns 'y', 'age_part', and 'age_cnt.
        'y' is the number of contacts between 'age_part' and 'age_cnt'.
        'age_part' is the age of the contactor.
        'age_cnt' is the age of the contacted.
    age_dist: NDArray
        The population age distribution.
    offset: NDArray, optional
        Additional offset to be multiplied to the contact intensity.
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
                 priors: dict, 
                 likelihood: str='negbin'):
        super().__init__(data, age_dist, priors, likelihood)
        
        # Setup
        self.aid = self.data['age_part'].values
        self.bid = self.data['age_cnt'].values
        self.y = self.data['y'].values
        self.log_N = jnp.log(self.data['N'].values)
        self.log_P = jnp.log(self.age_dist)[jnp.newaxis,:]
        self.log_S = jnp.log(self.data['S'].values) if 'S' in self.data.columns else jnp.zeros_like(self.y)
        
    def model(self):
        beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
        with scope(prefix='rate'):
            f = self.priors['rate'].sample()
        log_rate = numpyro.deterministic('log_rate', beta0 + f)
        log_cint = numpyro.deterministic('log_cint', log_rate + self.log_P)
        
        mu = jnp.exp(log_cint[self.aid, self.bid] + self.log_N + self.log_S)        
        with numpyro.plate('data', len(self.y)):
            if self.likelihood == 'poisson':
                numpyro.sample('obs', dist.Poisson(rate=mu), obs=self.y)
            elif self.likelihood == 'negbin':
                inv_disp = numpyro.sample('inv_disp', dist.Exponential(1))
                numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                            concentration=inv_disp),
                               obs=self.y)