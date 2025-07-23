from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.distributions import constraints, transforms
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
  validate_sample,
  promote_shapes
)
from numpyro.util import is_prng_key

class QuasiPoisson(Distribution):
  arg_constraints = {'mu': constraints.positive}
  support = constraints.nonnegative_integer
  
  def __init__(
    self,
    mu: ArrayLike,
    psi: float = 1.0,
    validate_args: Optional[bool] = None
  ):
    self.mu = mu
    self.psi = psi
    super(QuasiPoisson, self).__init__(jnp.shape(mu), validate_args=validate_args)
    
  def sample(self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
    assert is_prng_key(key)
    # TODO: Figure out how to sample from a QuasiPoisson
    
  @validate_sample
  def log_prob(self, value: ArrayLike) -> ArrayLike:
    if self._validate_args:
      value = validate_sample(value)
    
    return 1/self.psi * (value * jnp.log(self.mu) - self.mu)