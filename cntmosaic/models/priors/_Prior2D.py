from abc import ABC, abstractmethod
from numpy.typing import NDArray
import jax.numpy as jnp
from .._math import (
  alr,
  clr,
  ilr
)

class Prior2D(ABC):
  def __init__(self,
               grid_type: str='age-age',
               loc: float | NDArray=0,
               event_dim: int=1,
               transform: str | None=None,
               symmetric: bool=False):
    self.grid_type = grid_type
    self.loc = loc
    self.event_dim = event_dim
    self.transform = transform
    self.symmetric = symmetric
    self.validate_params()
  
  def validate_params(self):
    assert self.grid_type in ['age-age', 'diff-age'], "grid_type must be 'age-age' or 'diff-age'"
    assert self.event_dim > 0, "event_dim must be greater than 0"
    assert self.transform in [None, 'alr', 'clr', 'ilr'], "transform must be None, 'alr', 'clr', or 'ilr'"
    
  @abstractmethod
  def set_age_bounds(self, min_age: int, max_age: int):
    pass
  
  @abstractmethod
  def _make_grid(self):
    pass
  
  @abstractmethod
  def sample(self):
    pass
  
  def _make_loc(self):
    if not isinstance(self.loc, (int, float)):
      assert self.loc.shape[-1] == self.event_dim, "loc must have the same number of columns as the event dimension."
      assert self.A == self.loc.shape[0], "loc must have the same number of rows as the age dimension."

      if self.transform == 'alr':
        self.loc = jnp.repeat(alr(self.loc, axis=1), self.A, axis=0).T
        self.event_dim_eff = self.event_dim - 1
      elif self.transform == 'clr':
        self.loc = jnp.repeat(clr(self.loc, axis=1), self.A, axis=0).T
        self.event_dim_eff = self.event_dim
      elif self.transform == 'ilr':
        self.loc = jnp.repeat(ilr(self.loc, axis=1), self.A, axis=0).T
        self.event_dim_eff = self.event_dim - 1
    else:
      self.loc = jnp.array([self.loc])
      self.event_dim_eff = self.event_dim