# write a function, output a model class
from ..preprocess._utils import check_required_columns
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from numpy.typing import NDArray
import pandas as pd
import numpy as np
import re
from ._BRC import BRC
from ._utils import age_age_grid, diff_age_age_grid, lower_tri_indices
from ._priors import HSGP


class BRCcustom(BRC):
    """Bayesian Rate Consistency model with fine age inputs and custom rules.

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
                 data: pd.DataFrame,
                 age_dist: NDArray,
                 offset: NDArray = None, 
                 likelihood: str = 'negbin'):
        super().__init__(data, age_dist, likelihood)
        
        self.age_dist = age_dist
        self.offset = offset
        self.set_hsgp_params()
        
        # Setup
        self.aid = self.data['age_part'].values
        self.bid = self.data['age_cnt'].values
        self.y = self.data['y'].values
        self.log_N = jnp.log(self.data['N'].values)
        self.log_P = jnp.log(self.age_dist)[jnp.newaxis, :]
        
        self.prior = {}
        self.intermediate = {}
        self.y_related = {}

    def set_age_dim(self, A):
        self.A = A
        self._compute_indices()
        self.set_hsgp_params()
        
    def set_hsgp_params(self, M: list[int]=[30, 30], C: list[float]=[1.5, 1.5], grid_type: str='age-age'):
        """Set the hyperparameters for the Hilbert space approximate Gaussian process prior.
    
        Parameters
        ----------
        M: int | list[int], default=[30, 30]
            Number of eigenfunctions to use.
            If int, the same number of eigenfunctions will be used for each dimension.
            If list, the number of eigenfunctions to use for each dimension.
        C: float | list[float], default=[1.5, 1.5]
            Scaling factor for the length scale.
            If float, the same scaling factor will be used for each dimension.
            If list, the scaling factor for each dimension
        grid_type: str, default='age-age'
            The type of grid to use for the input data.
            'age-age': age-age grid.
            'diff-age': difference-in-age by age grid.
            
        References
        ----------
        Shozen Dan et al., "Estimating fine age structure and time trends in
        human contact patterns from coarse contact data: The Bayesian rate consistency model",
        PLoS Computational Biology. 2023
        """
        self.M = M
        self.C = C
        
        if grid_type == 'age-age':
            self.grid_type = 'age-age'
            X = age_age_grid(self.A)
        elif grid_type == 'diff-age':
            self.grid_type = 'diff-age'
            X = diff_age_age_grid(self.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")

        ltri_idx = lower_tri_indices(self.A)
        Xn = (X - X.mean(axis=0)) / X.std(axis=0)
        self.L = list(np.abs(Xn).max(axis=0) * self.C)
        self.X = Xn[ltri_idx]
        
        self.hsgp = HSGP(self.X, self.L, M, self.sym_tri_idx)

   
    def add_prior(self, name, distribution):
        if not isinstance(distribution, dist.Distribution):
            raise ValueError('Please use a numpyro distribution')
        
        self.prior[name] = distribution
    

    def add_HSGP(self, magnitude, lengthscale):
        if magnitude not in self.prior:
            raise ValueError('HSGP magnitude not set in prior')
        if lengthscale not in self.prior:
            raise ValueError('HSGP lengthscale not set in prior')
        self.hsgp_magnitude = magnitude
        self.hsgp_lengthscale = lengthscale
    
    def add_intermediate(self, instruction):
        '''it should have the form log_A=prior1*prior2 '''
        lhs, rhs = instruction.replace(' ', '').split('=')
        self.intermediate[lhs] = rhs

    def model(self):
        priors = {}
        for name, distri in self.prior.items():
            priors[name] = numpyro.sample(name, distri)

        priors['f'] = self.hsgp.sample(priors[self.hsgp_magnitude], priors[self.hsgp_lengthscale]).reshape((self.A, self.A), order='F')

        math_func = {'log': jnp.log, 'exp': jnp.exp}
        for name, instruction in self.intermediate.items():
            operands = re.findall(r'\b[a-zA-Z_]\w*\b', instruction)
            for operand in operands:
                if hasattr(self, operand):
                    instruction = re.sub(rf'\b{operand}\b', f'self.{operand}', instruction)
                elif operand in priors:
                    instruction = re.sub(rf'\b{operand}\b', f'priors["{operand}"]', instruction)
                else:
                    raise ValueError('Unknown parameter ' + operand)
            eval_context = {**math_func, 'self': self, 'priors': priors}
            priors[name] = numpyro.deterministic(name, eval(instruction, {}, eval_context))

        with numpyro.plate('data', len(self.y)):
            if self.likelihood == 'poisson':
                lam = jnp.exp(priors['log_cint'][self.aid, self.bid] + self.log_N)
                numpyro.sample('obs', dist.Poisson(rate=lam), obs=self.y)
            elif self.likelihood == 'negbin':
                inv_disp = numpyro.sample('inv_disp', dist.Exponential(1))
                mu = jnp.exp(priors['log_cint'][self.aid, self.bid] + self.log_N)
                numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                             concentration=inv_disp),
                               obs=self.y)

    def compile(self, jupyter=True):
        if jupyter:
            from IPython.display import display, Math
            for prior, d in self.prior.items():
                distri = str(d)
                distri = distri[:distri.find(' ')].split('.')[-1]
                attrs = [f'{k}={v}' for k, v in d.__dict__.items()]
                try:
                    formula = f'{prior}\sim {distri}({",".join(attrs[:2]).replace("_", " ")})'
                except IndexError:
                    formula = f'{prior}\sim {distri}({attrs[0].replace("_", " ")})'
                display(Math(formula))
            display(Math(f'f\sim HSGP(magnitude={self.hsgp_magnitude}, lengthscale={self.hsgp_lengthscale})'))
            for inter, recipe in self.intermediate.items():
                formula = f'{inter}={recipe.replace("_", " ")}'
                display(Math(formula))
            if self.likelihood == 'poisson':
                display(Math('lambda = exp(log\_cint) + log N'))
                display(Math('y\sim Poisson(rate=lambda)'))
            elif self.likelihood == 'negbin':
                display(Math('inv\_disp \sim Exp(1)'))
                display(Math('mu = exp(log\_cint) + log N'))
                display(Math('y\sim NegBin(mean=mu, concentration=inv\_disp)'))