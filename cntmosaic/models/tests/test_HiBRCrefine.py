import pytest
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .._HiBRCrefine import HiBRCrefine
from ..priors import HSGP2D, PenalisedTensorSpline2D

def test_initialisation():
    data = pd.DataFrame({
        'y': [1, 2, 3],
        'N': [10, 20, 30],
        'age_part': [0, 1, 2],
        'age_grp_cnt': [
            pd.Interval(0, 1, closed='left'),
            pd.Interval(1, 5, closed='left'),
            pd.Interval(5, 10, closed='left')
        ],
        'sex_part': ['M', 'F', 'M'],
    })
    data['age_grp_cnt'] = pd.Categorical(data['age_grp_cnt'])
    data['sex_part'] = pd.Categorical(data['sex_part'], categories=['M', 'F'])
    age_dist = np.array([0.2, 0.3, 0.5, 0.1, 0.4, 0.5, 0.2, 0.3, 0.5, 0.6])
    age_dist_props = {
      'sex_part': np.array([[0.1, 0.5, 0.3, 0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.8],
                            [0.9, 0.5, 0.7, 0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.2]]),
    }
    priors = {
        'rate': HSGP2D(grid_type='diff-age', type='global'),
        'sex_part': PenalisedTensorSpline2D(
            grid_type='age-age',
            transform='ilr',
            type='partial',
            event_dim=2
        )
    }
    
    model = HiBRCrefine(data, age_dist, age_dist_props, priors)
    
    # Check data
    assert model.data.equals(data)
    
    # Offsets
    assert jnp.array_equal(model.log_P, jnp.log(age_dist)[jnp.newaxis,:])
    assert np.array_equal(model.N, data['N'].values)
    assert np.array_equal(model.S, np.ones(3))
    
    # Dimensions and indices
    assert model.A == 10
    assert np.array_equal(model.aid, data['age_part'].values)
    assert np.array_equal(model.cid, data['age_grp_cnt'].cat.codes.values)
    
    # Likelihood
    assert model.likelihood == 'negbin'