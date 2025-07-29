import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.handlers import plate, scope

from ..dataloader import DataLoader
from ._BRCrefine import BRCrefine
from ._utils import index_mask_logsumexp
from .priors import HSGP2D, Hill
  
def expand_idarr(id, length):
    return np.repeat(id[:,np.newaxis], length, axis=1)

class HiBRCrefine(BRCrefine):
    """High-resolution Bayesian Rate Consistency model with fine participant age but coarse contacted age.
    
    Parameters
    ----------
    data: DataLoader
        DataLoader object containing the contact data.
    priors: dict, optional
        Dictionary containing the priors for the components of the model.
        If None, default priors are used.
    likelihood: str, default='negbin'
        Likelihood function to use. Options are 'poisson' and 'negbin'.
    """
    
    def __init__(self,
                 dataloader: DataLoader,
                 priors: dict,
                 likelihood: str='negbin'):
        
        super().__init__(dataloader, priors, likelihood)
        
        self.y = jnp.array(self.ds.y.values)
        self.log_N = jnp.array(self.ds.log_N.values)
        self.log_P = jnp.array(self.ds.log_P.values)[jnp.newaxis,:]
        self.log_S = jnp.array(self.ds.log_S.values) if hasattr(self.ds, 'log_S') else jnp.zeros_like(self.y)
            
        self.aid = jnp.array(self.ds.aid.values)
        self.aid_exp = jnp.array(self.ds.aid_exp.values)
        self.bid_pad = jnp.array(self.ds.bid_pad.values)
        self.X_vars = [key for key in priors.keys() if key != 'rate']
        self.X_ids = {
            c: pd.Categorical(self.ds[c].values, categories=sorted(set(self.ds[c].values))).codes
            for c in self.X_vars
        }
        self.X_ids_exp = {
          c: expand_idarr(self.X_ids[c], self.bid_pad.shape[1])
          for c in self.X_vars
        }
        self.set_prior_event_dim()
        self.set_prior_loc()
        self.set_log_age_dist_props()
        
        if hasattr(self.ds, 'rid'):
            self.rid = jnp.array(self.ds.rid.values)
            self.hill = Hill(max_value=self.ds.rid.max())
        
    # Compute the log of the age distribution proportions
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
        
        repeat_effect = self.hill.sample()[self.rid] if hasattr(self, 'rid') else 0.0

        mu = jnp.exp(log_cint + self.log_N + self.log_S + repeat_effect)
        if self.likelihood == 'poisson':
            with plate('data', len(self.y)):
                numpyro.sample('obs', dist.Poisson(rate=mu), obs=self.y) 
            
        if self.likelihood == 'negbin':
            inv_disp = numpyro.sample('inv_disp', dist.Exponential(1))
            with plate('data', len(self.y)):
                numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                            concentration=1/inv_disp), 
                                obs=self.y)