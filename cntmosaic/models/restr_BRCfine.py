from numpy.typing import NDArray 
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as jrd
import numpyro
from numpyro import distributions as dist
from dataclasses import dataclass

from ._BRC import BRC
from ._utils import age_age_grid, diff_age_age_grid, lower_tri_indices, symmetrize_from_lower_tri
from ._priors import HSGP

def get_params(distr):
    common_distribution_params = [
        # Central tendency & spread
        "loc",
        "scale",
        "mean",
        "variance",
        
        # Rate & shape
        "rate",
        "concentration",
        "concentration0",
        "concentration1",
        "scale_tril",
        "precision_matrix",
        "covariance_matrix",
        
        # Discrete/multivariate
        "total_count",
        "probs",
        "logits",
        "low",
        "high",
        "df",  # degrees of freedom (StudentT)
        
        # Meta/shape
        "batch_shape",
        "event_shape",
        "support"
    ]
    return {
            attr: getattr(distr, attr)
            for attr in dir(distr)
            if (not attr.startswith("_") and not callable(getattr(distr, attr) )) and (attr in common_distribution_params)
        }

@dataclass
class HyperParams:
    def __init__(self):
        self.prior = {}

    def __str__(self):
        lines = []
        for k, v in self.__dict__.items():
            if k != 'prior':
                lines.append(f"{k}: {v}")
        lines.append("prior:")
        for k, v in self.prior.items():
            lines.append(f'{k}:{v}')
            d = get_params(v)
            for k1, v1 in d.items():
                lines.append(f'{k1}:{v1}')
        return '\n'.join(lines)


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
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.params = HyperParams()
        self.set_default_params()
        self._precompute = HyperParams()
        print('new model instantiated, please check default hyperparameters')

    def compile(self):
        # checks
        self._precompute.prior = True
        self._precompute.aid = self.data['age_part'].values
        self._precompute.bid = self.data['age_cnt'].values
        self._precompute.y = self.data['y'].values
        self._precompute.log_N = jnp.log(self.data['N'].values)
        self._precompute.log_P = jnp.log(self.params.age_dist)[jnp.newaxis,:]
        self._precompute.hsgp = self.set_hsgp()
        print('model compiled, ready for sampling')
        
    def prior_sampler(self, para, n=1):
        '''
        prior sampling
        para: parameter to sample from
        n: number of samples
        '''

        assert(para in self.params.prior)
        _, subkey = jrd.split(jrd.PRNGKey(0))
        samples = self.params.prior[para].sample(subkey, sample_shape=(n,))
        return samples
        
    def set_default_params(self):
        self.params.M = [30, 30]
        self.params.C = [1.5, 1.5]
        self.params.grid_type = 'age-age'
        self.params.likelihood = 'negbin'
        self.params.A = len(set(self.data['age_cnt']).union(self.data['age_part']))
        self.params.prior['beta0'] = dist.Normal(0., 10.)
        self.params.prior['alpha'] = dist.InverseGamma(5, 5)
        self.params.prior['rho'] = dist.InverseGamma(5, 5).expand([2])
        self.params.offset = None
        self.params.age_dist = 1 / self.params.A * np.ones(self.params.A)

    def set_hsgp(self):
        if self.params.grid_type == 'age-age':
            
            X = age_age_grid(self.params.A)
        elif self.params.grid_type == 'diff-age':
            X = diff_age_age_grid(self.params.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")

        ltri_idx = lower_tri_indices(self.params.A)
        Xn = (X - X.mean(axis=0)) / X.std(axis=0)
        L = list(np.abs(Xn).max(axis=0) * self.params.C)
        X = Xn[ltri_idx]
        sym_tri_idx = symmetrize_from_lower_tri(self.params.A)
        return HSGP(X, L, self.params.M, sym_tri_idx)

    def model(self):
        if not self._precompute.prior:
            raise NotImplementedError('Please compile the model first')
        
        alpha = numpyro.sample('baseline', self.params.prior['alpha'])
        rho = numpyro.sample('rho', self.params.prior['rho'])
        beta0 = numpyro.sample('beta0', self.params.prior['beta0'])

        f = self._precompute.hsgp.sample(alpha, rho).reshape((self.params.A, self.params.A), order='F')
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