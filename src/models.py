import pandas as pd
import numpy as np
from numpy.typing import NDArray

import jax
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern

import model_utils

class BRC:
    def __init__(self,
                 data: pd.DataFrame,
                 M: int | list[int],
                 C: float | list[float]=[1.5, 1.5]):
        self.data = data.copy()

        self.M = M
        self.C = C

        self.A = self.data['age_part'].unique().size
        self._precompute_indices()
        self._precompute_grid()
        self._preprocess_input()
    
    def _precompute_grid(self):
        X = model_utils.non_nuisance_grid(self.A)
        ltri_idx = model_utils.lower_tri_indices(self.A)
        Xn = (X - X.mean(axis=0)) / X.std(axis=0)
        self.L = list(np.abs(Xn).max(axis=0) * self.C)
        self.X = Xn[ltri_idx]
    
    def _precompute_indices(self):
        self.sym_tri_idx = model_utils.symmetrize_from_lower_tri(self.A)
        self.tran_vec_idx = model_utils.transpose_vector_indices(self.A, self.A)

    def _preprocess_input(self):
        self.data = self.data.sort_values(by=['age_cnt', 'age_part'])
        self.data.reset_index(names='id', inplace=True)
        
        # Extract non-missing values and their corresponding index
        self.y = self.data['y'][~self.data['y'].isna()].values
        self.yid = self.data['id'][~self.data['y'].isna()].values

        # Compute offsets
        self.data['n'] = np.where(self.data['n'].isna(), 1.0, self.data['n'])
        self.log_n = jnp.log(self.data['n'].values)
        self.log_p = jnp.log(self.data['p'].values)

    def sample_latent_field(self, x, alpha, rho):
        phi = eigenfunctions(x=x, ell=self.L, m=self.M)
        spd = jnp.sqrt(
            diag_spectral_density_matern(
                nu=5/2, alpha=alpha, length=rho, ell=self.L, m=self.M, dim=2
            )
        )
        with numpyro.plate('hsgp_basis_coef', phi.shape[-1]):
            beta = numpyro.sample('beta', dist.Normal(0, 1))

        return phi @ (spd * beta)
    
    def tree_flatten(self):
        children = ()
        aux_data = (
            self.L,
            self.M
        )
        return (children, aux_data)
  
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class BRCBasic(BRC):
    def __init__(self,
                 data: pd.DataFrame,
                 M: int | list[int],
                 C: float | list[float]=[1.5, 1.5]):
        super().__init__(data, M, C)

    def model(self):
        beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
        alpha = numpyro.sample('gp_scale', dist.InverseGamma(5, 5))
        rho = numpyro.sample('gp_lenscale', dist.InverseGamma(5, 5).expand([2]))

        f = self.sample_latent_field(self.X, alpha, rho)[self.sym_tri_idx]
        log_rate = numpyro.deterministic('log_rate', beta0 + f)
        log_cint = numpyro.deterministic('log_cint', log_rate + self.log_p)
        log_lam = log_cint + self.log_n

        numpyro.sample('obs', dist.Poisson(rate=jnp.exp(log_lam[self.yid])), obs=self.y)

# ==========
# Statified Bayesian Rate Consistency Model
# ==========
@jax.tree_util.register_pytree_node_class
class BRCStratified(BRC):
    def __init__(self,
                 data: pd.DataFrame,
                 pratio: dict,
                 M: int | list[int],
                 smooth_type: dict=None,
                 C: float | list[float]=[1.5, 1.5]):
        
        self.pratio = pratio
        super().__init__(data, M, C)

        if smooth_type is None:
            self.smooth_type = {col: 'plate' for col in self.X_cols}
        else:
            self.smooth_type = smooth_type
        
    def _preprocess_input(self):
        self.y = self.data['y'].values
        self.log_n = jnp.log(self.data['n'].values)
        self.log_p = jnp.log(self.data['p'].values)

        self.aid = self.A*self.data['age_part'].values + self.data['age_cnt'].values
        self.X_cols = self.data.columns[self.data.columns.str.startswith('X_')]
        self.K_ids = {col: self.data[col].astype(int).values for col in self.X_cols}
        self.D = {col: pd.get_dummies(self.data[col], dtype=float).values for col in self.X_cols}
        self.K = {col: self.data[col].nunique() for col in self.X_cols}
        self.log_pratio = {k: np.log(v.T) for k, v in self.pratio.items()}
    
    def sample_omega(self, key, type): # TODO: implementations for different regularised priors
        if type == 'random':
            with numpyro.plate(f'plate_omega_{key}', self.K[key]):
                x = numpyro.sample(f'omega_{key}', dist.Normal(0, 1).expand([self.A, self.A]).to_event(2))

        elif type == 'plate':
            a = numpyro.sample(f'omega_a_{key}', dist.Normal(0, 1).expand([self.K[key]]))
            b = numpyro.sample(f'omega_b_{key}', dist.Normal(0, 1))
            c = numpyro.sample(f'omega_c_{key}', dist.Normal(0, 1))

            x = jnp.arange(self.A)
            X, Y = jnp.ix_(x, x)

            x = a[:, None, None] + b * X + c * Y
            
        return x
  
    def sample_log_delta(self, key, type):
        omega = self.sample_omega(key, type)
        x = jax.nn.log_softmax(omega, axis=0) - self.log_pratio[key][:,:,None]
        dims = (self.K[key], self.A**2)
        return numpyro.deterministic(f'log_delta_{key}', x.reshape(dims))

    def model(self):
        beta0 = numpyro.sample('baseline', dist.Normal(0., 10.))
        alpha = numpyro.sample('gp_scale', dist.HalfNormal(1.))
        rho = numpyro.sample('gp_lenscale', dist.InverseGamma(5., 5.))

        f = numpyro.deterministic('f', self.sample_latent_field(self.X, alpha, rho)[self.sym_tri_idx])
        log_rate = (beta0 + f)[self.aid]

        for col in self.X_cols:
            log_rate += self.sample_log_delta(col, 'plate')[self.K_ids[col], self.aid]

        log_cint = numpyro.deterministic('log_cint', log_rate + self.log_p)
        log_mu = log_cint + self.log_n

        numpyro.sample('y', dist.Poisson(rate=jnp.exp(log_mu)), obs=self.y)