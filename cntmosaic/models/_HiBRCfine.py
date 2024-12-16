import numpy as np
from numpy.typing import NDArray
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.handlers import plate, scope

from ._BRCfine import BRCfine
from ._priors import TensorSpline2D, PenalisedTensorSpline2D
from ._math import (
    alr,
    ilr,
    log_inverse_alr,
    log_inverse_ilr,
)

def set_default_smoother_types(X_vars: list, smoother_types: dict | None):
    if smoother_types is None:
        smoother_types = {k: 'pts' for k in X_vars}
    else:
        for x in X_vars:
            if x not in smoother_types.keys():
                smoother_types[x] = 'pts'
                
    return smoother_types

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
                 smoother_types: dict=None,
                 offset: NDArray=None,
                 likelihood: str='negbin'):
        
        super().__init__(data, age_dist, offset, likelihood)
            
        self.y = self.data['y'].values
        self.log_N = jnp.log(self.data['N'].values)
        self.log_P = jnp.log(self.age_dist)[jnp.newaxis,:]
        self.log_S = jnp.log(offset) if offset is not None else jnp.zeros_like(self.y)
        self.X_vars = self.data.select_dtypes(include='category').columns        
        self.X_dims = {x: len(self.data[x].cat.categories) for x in self.X_vars}
        
        # Compute the log of the age distribution proportions
        self.age_dist_props = age_dist_props
        self.log_age_dist_props = {k: np.log(v).T for k, v in age_dist_props.items()}
        
        # Set default smoother types
        self.smoother_types = set_default_smoother_types(self.X_vars, smoother_types)
        if 'ts' in self.smoother_types.values():
            self.ts = TensorSpline2D(np.arange(self.A), M=30, degree=3)
        if 'pts' in self.smoother_types.values():
            self.pts = PenalisedTensorSpline2D(np.arange(self.A), M=30, degree=3, neighborhood=8)
        
        # Setup indices
        self.aid = self.data['age_part'].values
        self.bid = self.data['age_cnt'].values
        self.X_ids = {c: self.data[c].cat.codes.values for c in self.X_vars}
        
    def set_age_dim(self, A):
        self.A = A
        self._compute_indices()
        self.set_hsgp_params()
        if 'ts' in self.smoother_types.values():
            self.ts = TensorSpline2D(np.arange(self.A), M=30, degree=3)
        if 'pts' in self.smoother_types.values():
            self.pts = TensorSpline2D(np.arange(self.A), M=30, degree=3)
        
    def set_spline_params(self, n_knots: int=27, degree: int=3):
        """Set the parameters for the splines.
        
        Parameters
        ----------
        n_knots: int, default=27
            The number of knots to use.
        degree: int, default=3
            The degree of the B-splines.
        """
        self.n_knots = n_knots
        self.degree = degree
        
        #TODO: Implement for multiple variables
        
    def sample_omega(self, var):
        if self.smoother_types[var] == 'ts':
            omega = self.ts.sample(
                loc=np.repeat(ilr(self.age_dist_props[var], axis=1), self.A),
                coef_scale=0.5,
                event_dim=self.X_dims[var]-1
            )
            return omega
        elif self.smoother_types[var] == 'pts':
            omega = self.pts.sample(
                loc=np.repeat(ilr(self.age_dist_props[var], axis=1), self.A),
                coef_scale=0.5,
                event_dim=self.X_dims[var]-1
            )
            return omega
    
    def sample_log_delta(self, var):
        omega = self.sample_omega(var)
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