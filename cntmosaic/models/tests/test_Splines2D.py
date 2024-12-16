import pytest

import numpy as np
import jax.random as random
import numpyro
from .._priors import TensorSplines2D

prng_key = random.PRNGKey(0)

def test_TensorSplines2D_initialisation():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    M = 3
    degree = 2
    tps = TensorSplines2D(x, y, M, degree)
    
    assert tps.basis.shape == (25, 9)