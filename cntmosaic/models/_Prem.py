import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from .dists import IGMRF1D
from ._BRC import BRC
from ..dataloader import DataLoader

class Prem:
  def __init__(self,
               part: pd.DataFrame,
               cnt: pd.DataFrame):
    self._validate_inputs(part, cnt)
    self.part = part.copy()
    self.cnt = cnt.copy()
    
  def _validate_inputs(self, part: pd.DataFrame, cnt: pd.DataFrame):
    part_cols = part.columns
    cnt_cols = cnt.columns
    
    if "id" not in part_cols:
      raise ValueError("participant DataFrame must contain 'id' column")
    if "id" not in cnt_cols:
      raise ValueError("contact DataFrame must containt 'id' column")
    
    if "age_grp_part" not in part_cols:
      raise ValueError("participant DataFrame must contain 'age_grp_part' column")
    if "age_grp_cnt" not in cnt_cols:
      raise ValueError("contact DataFrame must contain 'age_grp_cnt' column")
    
    if "y" not in cnt_cols:
      raise ValueError("Missing column 'y' in contact DataFrame")
    
  def _load(self):
    # [Do] Merge contact data and participant data
    self.merged = pd.merge(self.cnt, self.part, on="id", how="left")
    