from numpy.typing import NDArray

from jax import random
import jax.numpy as jnp
import pandas as pd
import numpyro
from numpyro import distributions as dist
from numpyro.handlers import seed, trace
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern

from ..model_utils import (
  non_nuisance_grid,
  lower_tri_indices,
  symmetrize_from_lower_tri,
  transpose_vector_indices
)

class BRCBase:
    """Base class for the Bayesian Rate Consistency model.
    
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
  
    def __init__(self,
                 data: pd.DataFrame,
                 age_dist: NDArray,
                 likelihood: str='negbin'):
      
        self.data = data.copy()
        self.age_dist = age_dist
        self.likelihood = likelihood
        
        self.set_hsgp_params()
        self.A = self.data['age_part'].max()
    
    def set_hsgp_params(self,
                        M: int | list[int]=None,
                        C: float | list[float]=None):
        """Set the hyperparameters for the Hilbert space approximate Gaussian process prior.
        
        Parameters
        ----------
        M: int | list[int], default=None
            Number of eigenfunctions to use.
            If int, the same number of eigenfunctions will be used for each dimension.
            If list, the number of eigenfunctions to use for each dimension.
        C: float | list[float], default=None
            Scaling factor for the length scale.
            If float, the same scaling factor will be used for each dimension.
            If list, the scaling factor for each dimension
        """
        self.M = M if M is not None else [30, 30]
        self.C = C if C is not None else [1.5, 1.5]
        
    def set_age_dist(self, age_dist: NDArray):
        """Set the population age distribution.
        
        Parameters
        ----------
        age_dist: NDArray
            Population age distribution.
        """
        self.age_dist = age_dist
    
    def print_model_shape(self):
        """Print the shapes of the model parameters."""
        tr = trace(seed(self.model, random.PRNGKey(0))).get_trace()
        print(numpyro.util.format_shapes(tr))