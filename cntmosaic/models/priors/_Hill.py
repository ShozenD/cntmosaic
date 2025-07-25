import numpy as np
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

class Hill:
  """
  Decreasing hill function
  
  Parameters
  ----------
  max_value: int
      The maximum value of the hill function.
  scale_scale: float, optional
      Scale parameter for the scale of the hill function.
  shape1_loc: float, optional
      Location parameter for the first shape parameter.
  shape1_scale: float, optional
      Scale parameter for the first shape parameter.
  shape2_rate: float, optional
      Rate parameter for the second shape parameter.
  """
  def __init__(self,
               max_value: int,
               scale_scale: float = 1.0,
               shape1_loc: float = 0.0,
               shape1_scale: float = 1.0,
               shape2_rate: float = 1.0):
    
    self.max_value = max_value
    self.scale_scale = scale_scale
    self.shape1_loc = shape1_loc
    self.shape1_scale = shape1_scale
    self.shape2_rate = shape2_rate
    
    self.x = np.arange(0, max_value + 1) # Don't use jnp here to avoid JAX tracing issues

  def sample(self):
    
    scale = numpyro.sample('hill_scale', dist.HalfNormal(self.scale_scale))
    shape1 = numpyro.sample('hill_shape1', dist.Normal(self.shape1_loc, self.shape1_scale))
    shape2 = numpyro.sample('hill_shape2', dist.Exponential(self.shape2_rate))

    return (
      - scale * 
      jnp.exp(shape1) * jnp.power(self.x, shape2) /
      (1 + jnp.exp(shape1) * jnp.power(self.x, shape2))
    )