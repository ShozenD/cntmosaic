from numpy.typing import NDArray 
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from ._BRC import BRC
from ..utils.prior import HSGP

@jax.tree_util.register_pytree_node_class
class BRCFineAge(BRC):
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
    """
    def __init__(self,
                 data: pd.DataFrame,
                 age_dist: NDArray,
                 offset: NDArray=None, 
                 likelihood: str='negbin'):
        super().__init__(data, likelihood)
        
        self.age_dist = age_dist
        self.offset = offset
        self.hsgp = HSGP(self.X, self.L, self.M, self.sym_tri_idx)
        self._preprocess_input()
        
    def _preprocess_input(self):
        self.aid = self.data['age_part'].values
        self.bid = self.data['age_cnt'].values
        self.y = self.data['y'].values
        self.log_P = jnp.log(self.age_dist)[jnp.newaxis, :]

    def model(self):
        beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
        alpha = numpyro.sample('hsgp_scale', dist.InverseGamma(5, 5))
        rho = numpyro.sample('hsgp_lenscale', dist.InverseGamma(5, 5).expand([2]))

        f = self.hsgp.sample(alpha, rho).reshape((self.A, self.A), order='F')
        log_rate = numpyro.deterministic('log_rate', beta0 + f)
        log_cint = numpyro.deterministic('log_cint', log_rate + self.log_P)
        
        if hasattr(self, 'offset'):
            log_cint += jnp.log(self.offset)
        
        if self.likelihood == 'poisson':
            lam = jnp.exp(log_cint[self.aid, self.bid])
            numpyro.sample('obs', dist.Poisson(rate=lam, obs=self.y))
        elif self.likelihood == 'negbin':
            inv_varphi = numpyro.sample('inv_dispersion', dist.Exponential(1))
            mu = jnp.exp(log_cint[self.aid, self.bid])
            numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                         concentration=1/inv_varphi),
                           obs=self.y)