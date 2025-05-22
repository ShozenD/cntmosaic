from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
import warnings
import jax.numpy as jnp
from .._math import (
  alr,
  clr,
  ilr,
  is_square
)

class Prior2D(ABC):
  ALLOWED_GRID_TYPES = ['age-age', 'diff-age']
  ALLOWED_TRANSFORMS = [None, 'alr', 'clr', 'ilr']
  ALLOWED_TYPES = ['global', 'partial', 'full']
  
  def __init__(self,
               grid_type: str='age-age',
               loc: float | NDArray=0,
               event_dim: int=1,
               transform: str | None=None,
               type: str='global'):
    self.grid_type = grid_type
    self.loc = loc
    self.event_dim = event_dim
    self.transform = transform
    self.type = type
    self._validate_params()
    self._set_event_dim_eff()
  
  def _validate_params(self):
    if self.grid_type not in self.ALLOWED_GRID_TYPES:
      raise ValueError(f"grid_type must be one of: {self.ALLOWED_GRID_TYPES}")

    if self.event_dim <= 0:
      raise ValueError("event_dim must be greater than 0")

    if self.transform not in self.ALLOWED_TRANSFORMS:
      raise ValueError(f"transform must be one of: {self.ALLOWED_TRANSFORMS}")
            
    if self.type not in self.ALLOWED_TYPES:
      raise ValueError(f"type must be one of: {self.ALLOWED_TYPES}")

    if self.type == "global" and self.event_dim != 1:
      raise ValueError("event_dim must be 1 for global priors")

    if self.type == "full":
      if self.event_dim <= 1:
        raise ValueError("event_dim must be greater than 1 for full priors")
      if not is_square(self.event_dim):
        raise ValueError("event_dim must be a perfect square for full priors")
      
  def _set_loc(self):
    """Sets the location parameter (prior mean), handling different input shapes and transformations."""
    
    if isinstance(self.loc, (int, float)): # If loc is a scalar
      self.trans_loc = jnp.full((self.event_dim, self.A, self.A), self.loc)
    
    # If loc is a tensor that is the same shape as the prior samples then we are good
    elif self.loc.shape == (self.event_dim, self.A, self.A):
      if self.transform: # Check if transform is not None
        transform_func = {'alr': alr, 'clr': clr, 'ilr': ilr}.get(self.transform)
        if transform_func:
          self.trans_loc = transform_func(self.loc)
        else: # Should never happen as we have already validated the transform
          raise ValueError(f"Unknown transform: {self.transform}")
    
    elif self.loc.shape == (self.event_dim, self.A):
      if self.transform:
        transform_func = {'alr': alr, 'clr': clr, 'ilr': ilr}.get(self.transform)
        if transform_func:
          transformed_loc = transform_func(self.loc)
          self.trans_loc = jnp.repeat(transformed_loc[:, :, None], self.A, axis=2)
        else:
          raise ValueError(f"Unknown transform: {self.transform}")
    else:
      warnings.warn(
        f"loc must be a scalar, or have shape ({self.event_dim}, {self.A}, {self.A}) or ({self.event_dim}, {self.A}) but got {self.loc.shape}. Please check the input shape or manually set the age bounds."
      )
  
  def _set_event_dim_eff(self):
    """Sets the effective event dimension based on the transformation and prior type."""
    
    if self.type == 'full':
      self.event_dim_diag = int(np.sqrt(self.event_dim))
      if self.transform in ['alr', 'ilr']:
        self.event_dim_eff = self.event_dim - 1
        self.event_dim_diag -= 1
        self.event_dim_non_diag = self.event_dim_eff - self.event_dim_diag
      else: # clr or None
        self.event_dim_eff = self.event_dim
        self.event_dim_non_diag = self.event_dim - self.event_dim_diag
        
    elif self.type == 'partial':
      if self.transform in ['alr', 'ilr']:
        self.event_dim_eff = self.event_dim - 1
      else:
        self.event_dim_eff = self.event_dim
        
    else: # global
      self.event_dim_eff = 1
    
  @abstractmethod
  def set_age_bounds(self, min_age: int, max_age: int):
    pass
  
  @abstractmethod
  def _set_grid(self):
    pass
  
  @abstractmethod
  def sample(self):
    pass
  