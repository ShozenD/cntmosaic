import pytest
import numpy as np
import pandas as pd
from .._HiBRCfine import HiBRCfine
from ...preprocess._utils import expand_grid
from ...datasets import load_age_distribution
from ..priors import (
  TensorSpline2D
)

df_full = expand_grid({
  'age_part': np.arange(0, 85),
  'age_cnt': np.arange(0, 85),
  'sex_part': np.array(['M', 'F']),
  'sex_cnt': np.array(['M', 'F'])
})

df_full['y'] = np.random.randint(0, 100, df_full.shape[0])
df_full['N'] = np.random.randint(1, 20, df_full.shape[0])
df_full['sex_part_cnt'] = df_full['sex_part'] + '/' + df_full['sex_cnt']

age_dist = load_age_distribution('United_States')
age_dist_props = {
  'sex_part_cnt': np.column_stack([
    age_dist.P.values * 0.4,
    age_dist.P.values * 0.2,
    age_dist.P.values * 0.2,
    age_dist.P.values * 0.4
  ])
}

def test_full_prior():
  priors = {
    'rate': TensorSpline2D(type='global', symmetric=True, grid_type='diff-age'),
    'sex_part_cnt': TensorSpline2D(type='full', event_dim=4, transform='ilr')
  }
  
  model = HiBRCfine(
    df_full,
    age_dist=age_dist.P.values,
    age_dist_props=age_dist_props,
    priors=priors
  )
  
  assert model.A == 85
  assert model.priors['rate'].sym_tri_idx.size == 85 * 85
  assert list(model.X_ids.keys()) == ['sex_part_cnt']
  
  model.print_model_shape()
  