import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp
import jaxlib

import numpyro
from numpyro import distributions as dist
from numpyro.handlers import seed, trace
from ._numpyro import (
  run_inference_mcmc,
  run_inference_svi,
  posterior_predictive_mcmc,
  posterior_predictive_svi
)

from .funcs import gmrf2d_operators, gmrf

class Prem:
  def __init__(self,
               part: pd.DataFrame,
               cnt: pd.DataFrame,
               random_effects: bool = False):
    self._validate_inputs(part, cnt)
    self.part = part.copy()
    self.cnt = cnt.copy()
    self.random_effects = random_effects
    self._load()
    
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
    # [Do] Create full contact dataframe
    coords = {
      "id": self.cnt["id"].unique(),
      "age_grp_cnt": self.cnt["age_grp_cnt"].cat.categories
    }
    index = pd.MultiIndex.from_product(coords.values(), names=coords.keys())
    df_cnt_full = pd.DataFrame(list(index), columns=coords.keys())
    df_cnt_full = pd.merge(df_cnt_full, self.cnt, on=["id", "age_grp_cnt"], how="left")
    df_cnt_full["y"] = df_cnt_full["y"].fillna(0).astype(int)
		# [Do] Restore the original information of the age group column
    df_cnt_full["age_grp_cnt"] = pd.Categorical(
			df_cnt_full["age_grp_cnt"],
			categories=self.cnt["age_grp_cnt"].cat.categories,
			ordered=True
		)
    
    # [Do] Merge contact data and participant data
    self.data = pd.merge(df_cnt_full, self.part, on="id", how="left")
    self.data = (
      self.data
      .groupby(["id", "age_grp_part", "age_grp_cnt"], observed=True)['y']
      .sum()
      .reset_index()
    )
    
    self.data["iix"] = pd.factorize(self.data["id"])[0]
    
    self.N = self.data["id"].nunique()
    self.y = np.array(self.data["y"].values)
    
    self.iix = np.array(self.data["iix"].values)
    self.C = self.data["age_grp_cnt"].cat.categories.size
    self.D = self.data["age_grp_part"].cat.categories.size
    self.cix = np.array(self.data["age_grp_part"].cat.codes)
    self.dix = np.array(self.data["age_grp_cnt"].cat.codes)
