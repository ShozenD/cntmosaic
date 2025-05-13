from numpy.typing import NDArray 
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as jrd
import numpyro
from numpyro import distributions as dist
from cntmosaic.dataloader.restru_loaders import GeneralLoader, HyperParams
from dataclasses import dataclass
from ._utils import fine_coarse_matrix

from ._BRC import BRC
from .priors import HSGP2D

    
class restr_BRCrefine(BRC):
    """Bayesian Rate Consistency model with fine participant age, coarse contact age inputs.

    Parameter
    ----------
    data: GeneralLoader
        Loaded GeneralLoader object
    References
	----------
	Shozen Dan et al., "Estimating fine age structure and time trends in 
	human contact patterns from coarse contact data: The Bayesian rate consistency model",
	PLoS Computational Biology. 2023
    """
    def __init__(self, data: GeneralLoader):
        '''
        Get precomputes from data.precomputes
        Calculate fine_coarse_matrix and cid for coarse age input
        '''
        self.params = HyperParams()
        self._set_default_params()
        self._precompute = data.precomputes
        assert(data.col_map.age_grp_cnt)
        self._precompute.cid = self.interval_code_map(self._precompute.bid, data.min_age)
        self._precompute.fine_coarse_matrix = fine_coarse_matrix(
            pd.Series(self._precompute.bid), data.categories[data.col_map.age_grp_cnt])
        print('new model instantiated, please check default hyperparameters')

    @staticmethod
    def interval_code_map(interval, min_age):
        """
        Maps intervals to integer codes based on their position relative to `min_age`.

        Parameters
        ----------
        interval : list of intervals
            A list of interval objects (e.g., from `pandas.Interval`) with `left` and `right` attributes.
        min_age : int or float
            The minimum age used as a reference point for calculating interval codes.

        Returns
        -------
        jnp.ndarray
            An array of integers where each value represents the index code of an interval,
            calculated as the number of interval lengths from `min_age`.

        Notes
        -----
        The function assumes all intervals are of equal length and uses the first interval
        to determine the interval size.
    """
        
        interval_len = interval[0].right - interval[0].left
        return jnp.array([(i.left-min_age) // interval_len for i in interval], dtype=int)

    def _set_default_params(self):
        """
        Initialize default model parameters for the contact intensity prior.

        This method sets reasonable default values for hyperparameters used in the 
        low-rank Gaussian Process (HSGP) approximation and the prior distributions 
        for model parameters. These values are chosen to provide a good starting 
        point for most use cases.

        Default Parameters
        ------------------
        M : list[int]
            Number of basis functions used for the HSGP approximation along each axis.
            Default is [30, 30].
        C : list[float]
            Boundary factors defining the domain limits of the Gaussian Process.
            Default is [1.5, 1.5].
        grid_type : str
            Type of grid used to model the contact dynamics. Can be 'age-age' or 'diff-age'.
            Use 'diff-age' if the dynamics are believed to depend primarily on the age difference.
            Default is 'age-age'.
        likelihood : str
            Likelihood model for observed contact counts. Default is 'negbin' (Negative Binomial).
            Available likelihood includes 'negbin', 'poisson'
        offset : array-like or None
            Optional additive offset to contact intensity. Should match the shape of the age grid.
            Default is None.

        Priors
        ------
        beta0 : Normal
            Prior for the baseline log-intensity parameter. Default is Normal(0, 10).
        alpha : InverseGamma
            Prior for the magnitude of the Gaussian Process. Default is InverseGamma(5, 5).
        rho : InverseGamma (expanded to 2D)
            Prior for the length scales of the Gaussian Process in each dimension.
            Default is InverseGamma(5, 5) expanded to shape [2].

        HSGP2D
        ------
        hsgp : HSGP2D
            A low-rank GP approximation initialized using the specified `M`, `C`, and `grid_type`.

        Notes
        -----
        These defaults are recommended for general use. However, all parameters and priors 
        can be modified after initialization as needed.
        """
        self.params.M = [30, 30]
        self.params.C = [1.5, 1.5]
        self.params.grid_type = 'age-age'
        self.params.likelihood = 'negbin'
        self.params.prior['beta0'] = dist.Normal(0., 10.)
        self.params.prior['alpha'] = dist.InverseGamma(5, 5)
        self.params.prior['rho'] = dist.InverseGamma(5, 5).expand([2])
        self.params.offset = None
        self.params.hsgp = HSGP2D(C=self.params.C, 
                                  M=self.params.M, 
                                  grid_type=self.params.grid_type)
        
    def model(self):
        """
        Define the probabilistic model for contact intensity estimation using a 
        low-rank Gaussian Process approximation (HSGP) within a Bayesian framework.

        This method is intended to be used with NumPyro for probabilistic inference. 
        It models contact counts using either a Poisson or Negative Binomial likelihood, 
        depending on the specified setting in `self.params.likelihood`.

        The latent log contact intensity is composed of:
        - A global intercept term `beta0`
        - A structured latent function `f` sampled from the HSGP prior
        - A log-contact matrix `log_P` representing expected structure in contact dynamics
        - An optional offset term applied multiplicatively on the rate scale

        Raises
        ------
        NotImplementedError
            If the `DataLoader.precompute()` step has not been completed or if 
            an unsupported likelihood is specified.

        Sampling Parameters
        -------------------
        beta0 : float
            Global intercept sampled from the prior specified in `params.prior['beta0']`.
        f : array
            Structured latent effect sampled from HSGP prior, depending on priors for `alpha` and `rho`.
        inv_disp : float, optional
            Inverse dispersion parameter for Negative Binomial likelihood. Sampled from Exponential(1).

        Deterministic Outputs
        ---------------------
        log_rate : array
            Sum of beta0 and latent GP term.
        log_cint : array
            Log contact intensity, adjusted with log_P and optional offset.

        Likelihood
        ----------
        Poisson:
            If `params.likelihood == 'poisson'`, observed contact counts `y` are modeled as:
                obs ~ Poisson(exp(log_cint + log_N))

        Negative Binomial:
            If `params.likelihood == 'negbin'`, the model uses:
                obs ~ NegativeBinomial2(mean=mu, concentration=inv_disp)

        Notes
        -----
        The precomputed data (`_precompute`) must include:
        - `y` : observed counts
        - `aid`, `bid` : indices into the contact matrix
        - `log_P`, `log_N` : known covariates or exposure matrices

        This model must be run within a NumPyro inference context (e.g., MCMC or SVI).
        """
        if not hasattr(self.params.hsgp, 'L'):
            self.params.hsgp.set_age_bounds(0, self._precompute.A-1)
        if not self._precompute.prior:
            raise NotImplementedError(
                'DataLoader.precompute() failed, please check DataLoader object')
        beta0 = numpyro.sample('beta0', self.params.prior['beta0'])
        
        f = self.params.hsgp.sample(self.params.prior['alpha'],
                        self.params.prior['rho'])
        log_rate = numpyro.deterministic('log_rate', beta0 + f)
        log_cint = numpyro.deterministic('log_cint', log_rate + self._precompute.log_P)
        
        if self.params.offset is not None:
            log_cint += jnp.log(self.params.offset)
        
        with numpyro.plate('data', len(self._precompute.y)):
            if self.params.likelihood == 'poisson':
                lam = jnp.exp( (log_cint  @ self._precompute.fine_coarse_matrix)[self._precompute.aid, self._precompute.cid] + self._precompute.log_N)
                numpyro.sample('obs', dist.Poisson(rate=lam), obs=self._precompute.y)
            elif self.params.likelihood == 'negbin':
                inv_disp = numpyro.sample('inv_disp', dist.Exponential(1))
                mu = jnp.exp( (log_cint @ self._precompute.fine_coarse_matrix)[self._precompute.aid, self._precompute.cid]  + self._precompute.log_N) 
                numpyro.sample('obs', dist.NegativeBinomial2(mean=mu,
                                                            concentration=inv_disp),
                               obs=self._precompute.y)
            else:
                raise NotImplementedError('Available likelihood are negbin and poisson')