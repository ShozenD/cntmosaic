import pandas as pd
import numpy as np
from numpy.typing import NDArray
from statsmodels.gam.smooth_basis import BSplines

import jax
import jax.numpy as jnp
from jax import random

import numpyro
from numpyro import distributions as dist
from numpyro.handlers import seed, scope, trace
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

    def sample_latent_field(self, x, alpha, rho, symmetric=False):
        phi = eigenfunctions(x=x, ell=self.L, m=self.M)
        spd = jnp.sqrt(
            diag_spectral_density_matern(
                nu=5/2, alpha=alpha, length=rho, ell=self.L, m=self.M, dim=2
            )
        )
        with numpyro.plate('hsgp_basis_coef', phi.shape[-1]):
            beta = numpyro.sample('beta', dist.Normal(0, 1))

        f = phi @ (spd * beta)
        if symmetric:
            return f[self.sym_tri_idx]
        else:
            return f
    
    def print_model_shape(self):
        tr = trace(seed(self.model, random.PRNGKey(0))).get_trace()
        print(numpyro.util.format_shapes(tr))

    def model(self):
        # Empty model for subclassing
        pass
    
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

        f = self.sample_latent_field(self.X, alpha, rho, symmetric=True)
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

        # Numpyro plates
        self.plate_a = numpyro.plate('age_part', self.A, dim=-2)
        self.plate_b = numpyro.plate('age_cnt', self.A, dim=-1)
        self.plates_X = {c: numpyro.plate(c, self.K_dim[c], dim=-3) for c in self.X_cols}

        if smooth_type is None:
            self.smooth_type = {col: 'random' for col in self.X_cols}
        else:
            self.smooth_type = smooth_type
            if 'bspline' in self.smooth_type.values():
                self._make_bspline_bases()
            if 'regularised_bspline' in self.smooth_type.values():
                self._make_bspline_bases()
        
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

    def _make_bspline_bases(self):
        x = jnp.arange(self.A)
        self.bspline = BSplines(x, df=30, degree=3, include_intercept=False) # TODO: avoid hardcoding df
        self.bspline_basis = self.bspline.basis
        self.PHI = jnp.kron(self.bspline_basis, self.bspline_basis)
    
    def sample_omega(self, key):
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
                alpha = numpyro.sample('spline_intercept', dist.Normal(0, 1)) # alpha: (Kx, 1, 1)
            with numpyro.plate('bspline', self.PHI.shape[1]):
                beta = numpyro.sample('spline_coef', dist.Normal(0, 1)) # beta: (900,)
            
            g = (self.PHI @ beta).reshape(self.A, self.A) # g: (A, A)
            omega = numpyro.deterministic('omega', alpha + g) # omega.dim = (Kx, A, A)
        
        elif self.smooth_type[key] == 'regularised_bspline':
            global_shrink = numpyro.sample('omega_global_shrink', dist.HalfCauchy(1))
            with self.plates_X[key]:
                alpha = numpyro.sample('spline_intercept', dist.Normal(0, 1))
            with numpyro.plate('bspline', self.PHI.shape[1]):
                loc_shrink = numpyro.sample('omega_loc_shrink', dist.HalfCauchy(1))
                beta = numpyro.sample('spline_coef', dist.Normal(0, global_shrink * loc_shrink))
            
            g = (self.PHI @ beta).reshape(self.A, self.A)
            omega = numpyro.deterministic('omega', alpha + g)
            
        return omega
  
    def sample_log_delta(self, key):
        omega = self.sample_omega(key)
        log_delta = numpyro.deterministic(
            'log_delta',
            jax.nn.log_softmax(omega, axis=0) - self.log_pratio[key][:,:,None]
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

        numpyro.sample('y', dist.Poisson(rate=jnp.exp(log_mu)), obs=self.y)