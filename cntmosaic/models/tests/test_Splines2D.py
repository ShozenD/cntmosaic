import pytest

import numpy as np
import jax.random as random
import numpyro
from ..priors._priors import TensorSpline2D

prng_key = random.PRNGKey(0)

def test_TensorSpline2D_initialisation():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    M = 3
    degree = 2
    tps = TensorSpline2D(x, y, M, degree)
    
    assert tps.basis.shape == (25, 9)