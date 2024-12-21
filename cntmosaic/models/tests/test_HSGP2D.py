import jax.numpy as jnp
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern

diag_spd = jnp.column_stack(
  [diag_spectral_density_matern(5/2, 1, 1, 1.5, 30, 2) for _ in range(2)]
)

print(diag_spd.shape)