from numpy.typing import NDArray 
import pandas as pd
import jax.numpy as jnp
from dataclasses import dataclass

import numpyro
from numpyro import distributions as dist
from numpyro.handlers import scope

from ._BRC import BRC
from ..dataloader import DataLoader

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
                 dataloader: DataLoader,
                 priors: dict, 
                 likelihood: str='negbin'):
        
        super().__init__(dataloader, priors, likelihood)
        
        self._validate_inputs()
        self.y = jnp.array(self.ds.y.values)
        self.aid = jnp.array(self.ds.aid.values)
        self.bid = jnp.array(self.ds.bid.values)
        self.log_N = jnp.array(self.ds.log_N.values)
        self.log_P = jnp.array(self.ds.log_P.values)[jnp.newaxis,:]
        self.log_S = jnp.array(self.ds.log_S.values) if hasattr(self.ds, 'log_S') else jnp.zeros_like(self.y)
        
    def _validate_inputs(self):
        """
        This method validates the inputs which are specific to the BRCfine model.
        Specifically, it checks if the data contains all the required ingredients:
        aid, bid, log_N, and log_P
        """
        if not hasattr(self.ds, 'aid'):
            raise ValueError("Participant age indexes (aid) are missing.")
        if not hasattr(self.ds, 'bid'):
            raise ValueError("Contact age indexes (bid) are missing.")
        if not hasattr(self.ds, 'log_N'):
            raise ValueError("Log of sample size (log_N) is missing.")
        if not hasattr(self.ds, 'log_P'):
            raise ValueError("Log of population age distribution (log_P) is missing.")
        
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
                                                             concentration=1/inv_disp),
                               obs=self.y)