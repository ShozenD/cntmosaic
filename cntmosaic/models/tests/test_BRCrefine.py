import pytest
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ...dataloader import CoordToColumns, DataLoader
from .._BRCrefine import BRCrefine

def test_initialisation():
    df_part = pd.DataFrame({
        'id': [1, 2, 3],
        'y': [1, 2, 3],
        'N': [10, 20, 30],
        'age_part': [0, 1, 2],
        'age_grp_cnt': [
            pd.Interval(0, 1, closed='left'),
            pd.Interval(1, 5, closed='left'),
            pd.Interval(5, 10, closed='left')
        ],
    })
    
    df_cnt = pd.DataFrame({
        'id': [1, 2, 3],
        'age_grp_cnt': [
            pd.Interval(0, 1, closed='left'),
            pd.Interval(1, 5, closed='left'),
            pd.Interval(5, 10, closed='left')
        ]
    })
    df_cnt['age_grp_cnt'] = pd.Categorical(df_cnt['age_grp_cnt'], ordered=True)

    df_age_dist = pd.DataFrame({
        'age': np.arange(0, 10),
        'P': np.abs(np.random.rand(10))
    })
    
    col_map = CoordToColumns(
        id_var='id',
        age_part='age_part',
        age_grp_cnt='age_grp_cnt',
        age_pop='age',
        size_pop='P'
    )

    dataloader = DataLoader(df_part, df_cnt, df_age_dist, col_map)
    model = BRCrefine(dataloader)
    
    # Offsets
    assert jnp.array_equal(model.log_P, jnp.log(df_age_dist['P'].values)[jnp.newaxis,:])
    
    # Likelihood
    assert model.likelihood == 'negbin'
    