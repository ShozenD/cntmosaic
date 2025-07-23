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

class QuasiNegBin(Distribution):
  arg_constraints = {'mu': constraints.positive}
  support = constraints.nonnegative_integer
  
  def __init__(
    self,
    mu: ArrayLike,
    inv_odist: float = 1.0,
    psi: float = 1.0,
    validate_args: Optional[bool] = None
  ):
    self.mu = mu
    self.inv_odist = inv_odist
    self.psi = psi
    super(QuasiNegBin, self).__init__(jnp.shape(mu), validate_args=validate_args)
    
  def sample(self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()) -> ArrayLike:
    assert is_prng_key(key)
    # TODO: Figure out how to sample from a QuasiNegBin
    
  @validate_sample
  def log_prob(self, value: ArrayLike) -> ArrayLike:
    if self._validate_args:
      value = validate_sample(value)
    
    return 1/self.psi * (
      value * jnp.log(self.mu/(self.inv_odist + self.mu))
      + self.inv_odist * jnp.log(1/(self.inv_odist + self.mu))
    )