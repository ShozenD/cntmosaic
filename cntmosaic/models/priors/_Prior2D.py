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
               transform: str | None=None,
               prior_type: str='global'):
    self.validate_params(grid_type, transform, prior_type)
    self.grid_type = grid_type
    self.transform = transform
    self.type = prior_type
    
  def validate_params(self,
                      grid_type: str,
                      transform: str | None,
                      prior_type: str):
    
    if grid_type not in self.ALLOWED_GRID_TYPES:
      raise ValueError(f"grid_type must be one of: {self.ALLOWED_GRID_TYPES}")

    if transform not in self.ALLOWED_TRANSFORMS:
      raise ValueError(f"transform must be one of: {self.ALLOWED_TRANSFORMS}")
            
    if prior_type not in self.ALLOWED_TYPES:
      raise ValueError(f"type must be one of: {self.ALLOWED_TYPES}")
      
  def set_loc(self, loc: int | float | np.ndarray):
    """
    Sets the location parameter (prior mean),
    handling different input shapes and transformations.
    """
    
    if isinstance(loc, (int, float)): # If loc is a scalar
      self.trans_loc = jnp.full((self.event_dim, self.A, self.A), loc)
    
    valid_shapes = {
      (self.event_dim, self.A, self.A): lambda x: x,
      (self.event_dim, self.A): lambda x: jnp.repeat(x[:, :, None], self.A, axis=2)
    }
    data = next((fn(loc) for shape, fn in valid_shapes.items() if loc.shape == shape), None)

    if data is None:
      warnings.warn(
        "loc must be a scalar, or have shape "
        f"({self.event_dim}, {self.A}, {self.A}) or "
        f"({self.event_dim}, {self.A}) but got {loc.shape}. "
        "Please check the input shape or manually set the age bounds."
      )
    elif self.transform:
      transform_func = {'alr': alr, 'clr': clr, 'ilr': ilr}.get(self.transform)
      if transform_func:
        self.trans_loc = transform_func(data)
      else:
        raise ValueError(f"Unknown transform: {self.transform}")
  
  def set_event_dim(self, event_dim: int):
    """Sets the effective event dimension based on the transformation and prior type."""
    assert isinstance(event_dim, int) and event_dim > 0, "event_dim must be a positive integer"
    self.event_dim = event_dim
    
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
  