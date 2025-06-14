import pandas as pd
import jax.numpy as jnp
import xarray

import numpyro
from numpyro import distributions as dist
from numpyro.handlers import plate, scope
from ._BRCfine import BRCfine
from .priors import Prior2D

class HiBRCfine(BRCfine):
    """High-resolution Bayesian Rate Consistency model with fine age inputs.
    
    Parameters
    ----------
    ds: xarray dataset containing necessary input
    priors: dict, optional
        Dictionary containing the priors for the components of the model.
    likelihood: str, default='negbin'
        Likelihood function to use. Options are 'poisson' and 'negbin'.
    """
    def __init__(self,
                 ds: xarray.Dataset,
                 priors: dict[Prior2D],
                 likelihood: str='negbin'):
        
        super().__init__(ds, priors, likelihood)
        self.X_vars = [key for key in priors.keys() if key != 'rate']
        self.X_ids = {
            c: pd.Categorical(self.ds[c].values, categories=sorted(set(self.ds[c].values))).codes
            for c in self.X_vars
        }
        self.set_prior_event_dim()
        self.set_prior_loc()
        self.set_log_age_dist_props()
        
    def set_log_age_dist_props(self):
        self.log_age_dist_props = {}
        for k in self.ds.attrs['grp_vars'].keys():
            v = self.ds['pop_prop_' + k].to_numpy()
            if v.shape == (self.priors[k].event_dim, self.A):
                self.log_age_dist_props[k] = jnp.log(v)[:,:,jnp.newaxis]
            elif v.shape == (self.priors[k].event_dim, self.A, self.A):
                self.log_age_dist_props[k] = jnp.log(v)
            else:
                raise ValueError(f"Invalid shape for age_dist_props[{k}].")
            
    def set_prior_event_dim(self):
        """
        Sets the event dimension for each prior based on the dataset.
        """
        for var, prior in self.priors.items():
            if var == 'rate':
                prior.set_event_dim(1)
            else:
                prior.set_event_dim(len(self.ds.grp_vars[var]))
    
    def set_prior_loc(self):
        """
        Sets the location of the priors around the transformed population age proportions.
        """
        for var, prior in self.priors.items():
            if var != 'rate':
                prior.set_loc(self.ds['pop_prop_' + var].to_numpy())
            
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
                                                             concentration=1/inv_disp), 
                                obs=self.y)