import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.handlers import plate, scope

from ._BRC import BRC
from ..utils.model import bspline_basis
from ..utils.math import log_inverse_ilr
from ..utils.prior import HSGP

class BRCStratified(BRC):
    def __init__(self,
                 data: pd.DataFrame,
                 pratio: dict,
                 smooth_type: dict=None,
                 likelihood: str='negbin'):
        
        self.pratio = pratio
        super().__init__(data, likelihood)

        # Numpyro plates
        self.plate_a = numpyro.plate('age_part', self.A, dim=-2)
        self.plate_b = numpyro.plate('age_cnt', self.A, dim=-1)
        self.plates_X = {c: numpyro.plate(c, self.K_dim[c]-1, dim=-3) for c in self.X_cols}

        if smooth_type is None:
            self.smooth_type = {col: 'random' for col in self.X_cols}
        else:
            self.smooth_type = smooth_type
            x = np.linspace(start=0, stop=1, num=self.A)
            if 'bspline' in self.smooth_type.values():
                self.PHI = bspline_basis(x)
            if 'regularised_bspline' in self.smooth_type.values():
                self.PHI = bspline_basis(x)
        
    def _preprocess_input(self):
        self.y = self.data['y'].values
        self.log_n = jnp.log(self.data['n'].values)
        self.log_p = jnp.log(self.data['p'].values)

        self.a_id = self.data['age_part'].values
        self.b_id = self.data['age_cnt'].values
        self.ab_id = self.A * self.a_id + self.b_id
        self.X_cols = self.data.columns[self.data.columns.str.startswith('X_')]
        self.K_ids = {col: self.data[col].astype(int).values for col in self.X_cols}
        self.K_dim = {col: self.data[col].nunique() for col in self.X_cols}
        self.log_pratio = {k: np.log(v) for k, v in self.pratio.items()}
    
    def sample_omega(self, key):
        # TODO refactor code
        if self.smooth_type[key] == 'random':
            with self.plates_X[key], self.plate_a, self.plate_b:
                omega = numpyro.sample('omega', dist.Normal(0, 1)) # omega.dim = (Kx, A, A)
                
        
        elif self.smooth_type[key] == 'plate':
            with self.plates_X[key]:
                a = numpyro.sample('omega_a', dist.Normal(0, 1)) # a: (Kx, 1, 1)
            b = numpyro.sample('omega_b', dist.Normal(0, 1)) # b: (1,)

            x = jnp.arange(self.A)
            X, Y = jnp.ix_(x, x)
            omega = numpyro.deterministic('omega', a + b*(X + Y)) # omega.dim = (Kx, A, A)
        
        elif self.smooth_type[key] == 'bspline':
            with self.plates_X[key]:
                alpha = numpyro.sample('spline_intercept', dist.Normal(0, 1)) # alpha: (Kx-1, 1, 1)
            with numpyro.plate('bspline', self.PHI.shape[1]):
                beta = numpyro.sample('spline_coef', dist.Normal(0, 1)) # beta: (900,)
            
            g = (self.PHI @ beta).reshape(self.A, self.A) # g: (A, A)
            omega = numpyro.deterministic('omega', alpha + g) # omega.dim = (Kx-1, A, A)
            
        return omega
  
    def sample_log_delta(self, key):
        omega = self.sample_omega(key)
        log_delta = numpyro.deterministic(
            'log_delta',
            log_inverse_ilr(omega, axis=0) - self.log_pratio[key][:,:,None]
        )
        return log_delta

    def model(self):
        beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
        alpha = numpyro.sample('gp_scale', dist.HalfNormal(1.))
        rho = numpyro.sample('gp_lenscale', dist.InverseGamma(5., 5.))

        f = numpyro.deterministic('f', self.sample_latent_field(self.X, alpha, rho, symmetric=True))
        log_rate = (beta0 + f)[self.ab_id]

        for col in self.X_cols:
            with scope(prefix=col):
                log_rate_x = self.sample_log_delta(col)
                log_rate += log_rate_x[self.K_ids[col], self.a_id, self.b_id]

        log_cint = numpyro.deterministic('log_cint', log_rate + self.log_p)
        log_mu = log_cint + self.log_n
        
        if self.likelihood == 'poisson':
            numpyro.sample('obs', dist.Poisson(rate=jnp.exp(log_mu)), obs=self.y)
        elif self.likelihood == 'negbin':
            inv_varphi = numpyro.sample('inv_dispersion', dist.Exponential(1))
            numpyro.sample('obs', dist.NegativeBinomial2(mean=jnp.exp(log_mu),
                                                         concentration=1/inv_varphi), 
                           obs=self.y)