from numpy.typing import NDArray 
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as jrd
import numpyro
from numpyro import distributions as dist
from ..dataloader import DataLoader
from ..dataloader.restru_loaders import HyperParams
from dataclasses import dataclass

from ._BRC import BRC
from ._utils import age_age_grid, diff_age_age_grid, lower_tri_indices, symmetrize_from_lower_tri
from .priors import HSGP2D



class restr_BRCfine(BRC):
    """Bayesian Rate Consistency model with fine age inputs.

    Parameters
    ----------
    data: DataFrame
        DataFrame containing the contact data. Must contain the columns 'y', 'age_part', and 'age_cnt.
        'y' is the number of contacts between 'age_part' and 'age_cnt'.
        'age_part' is the age of the contactor.
        'age_cnt' is the age of the contacted.

    likelihood: str, default='negbin'
        Likelihood function to use.
        
    References
	----------
	Shozen Dan et al., "Estimating fine age structure and time trends in 
	human contact patterns from coarse contact data: The Bayesian rate consistency model",
	PLoS Computational Biology. 2023
    """
    def __init__(self, data: DataLoader):
        self.params = HyperParams()
        self.set_default_params()
        self._precompute = data.precomputes
        print('new model instantiated, please check default hyperparameters')
                
    def prior_sampler(self, para_name, num_samples=1, seed=0):
        '''
        prior sampling
        para_name: parameter name to sample from
        num_samples: number of samples
        '''

        assert(para_name in self.params.prior)
        _, subkey = jrd.split(jrd.PRNGKey(seed))
        samples = self.params.prior[para_name].sample(subkey, sample_shape=(num_samples,))
        return samples
        
    def set_default_params(self):
        self.params.M = [30, 30]
        self.params.C = [1.5, 1.5]
        self.params.grid_type = 'age-age'
        self.params.likelihood = 'negbin'
        self.params.prior['beta0'] = dist.Normal(0., 10.)
        self.params.prior['alpha'] = dist.InverseGamma(5, 5)
        self.params.prior['rho'] = dist.InverseGamma(5, 5).expand([2])
        self.params.offset = None

    def set_hsgp(self):
        if self.params.grid_type == 'age-age':
            
            X = age_age_grid(self._precompute.A)
        elif self.params.grid_type == 'diff-age':
            X = diff_age_age_grid(self._precompute.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")

        ltri_idx = lower_tri_indices(self._precompute.A)
        Xn = (X - X.mean(axis=0)) / X.std(axis=0)
        L = list(np.abs(Xn).max(axis=0) * self.params.C)
        X = Xn[ltri_idx]
        sym_tri_idx = symmetrize_from_lower_tri(self._precompute.A)
        return HSGP2D(X, L, self.params.M, sym_tri_idx)

    def model(self):
        if not self._precompute.prior:
            raise NotImplementedError('DataLoader.precompute() failed, please check DataLoader object')
        hsgp = self.set_hsgp()
        alpha = numpyro.sample('baseline', self.params.prior['alpha'])
        rho = numpyro.sample('rho', self.params.prior['rho'])
        beta0 = numpyro.sample('beta0', self.params.prior['beta0'])
        
        f = hsgp.sample(alpha, rho).reshape((self._precompute.A, self._precompute.A), order='F')
        log_rate = numpyro.deterministic('log_rate', beta0 + f)
        log_cint = numpyro.deterministic('log_cint', log_rate + self._precompute.log_P)
        
        if self.params.offset is not None:
            log_cint += jnp.log(self.params.offset)
        
        with numpyro.plate('data', len(self._precompute.y)):
            if self.params.likelihood == 'poisson':
                lam = jnp.exp(log_cint[self._precompute.aid, self._precompute.bid] + self._precompute.log_N)
                numpyro.sample('obs', dist.Poisson(rate=lam), obs=self._precompute.y)
            elif self.params.likelihood == 'negbin':
                inv_disp = numpyro.sample('inv_disp', dist.Exponential(1))
                mu = jnp.exp(log_cint[self._precompute.aid, self._precompute.bid] + self._precompute.log_N)
                numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                            concentration=inv_disp),
                               obs=self._precompute.y)
            else:
                raise NotImplementedError('Available likelihood are negbin and poisson')