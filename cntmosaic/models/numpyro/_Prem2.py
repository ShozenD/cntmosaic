"""NumPyro model mixin for Prem2."""

from typing import Optional

import jax.numpy as jnp
from jax.typing import ArrayLike


class Prem2NumPyroMixin:
    """Carries the NumPyro ``model()`` for Prem2.

    Reformulates :class:`~cntmosaic.models.classical.Prem2`'s Poisson/NegBin
    likelihood over *aggregated* contact-count cells ``Y`` (shape ``(C, D)``,
    ``(K_part, C, D)``, or ``(K_part, K_cnt, C, D)`` depending on
    stratification — see :class:`SocialMixDataLoader`), rather than Prem's
    individual-level long-format rows. Each stratum gets an independent
    IGMRF2D-smoothed intensity surface with a penalized-complexity (PC) prior
    on its precision, following :class:`~cntmosaic.models.classical.Prem`.
    Unlike Prem, there are no participant-level random effects (they don't
    make sense once counts are aggregated), and unlike
    ``AgeMixCC``/``GenMixCC`` there is no ``log_P`` population-size offset
    (Prem2 has no population data).
    """

    def model(self, y: Optional[ArrayLike] = None) -> None:
        import numpyro
        from numpyro import distributions as dist
        from numpyro.handlers import plate

        from .distributions import IGMRF2D, Gumbel2

        y = self.Y if y is None else y
        log_N = jnp.log(self.N)

        if self.K_part == 1 and self.K_cnt == 1:
            # Single stratum: Y (C, D), N (C,)
            beta0 = numpyro.sample(
                "baseline", dist.Normal(jnp.log(self.Y.mean()), jnp.log(10))
            )
            tau = numpyro.sample("tau", Gumbel2(concentration=0.5, rate=2.86))
            f = numpyro.sample(
                "f",
                IGMRF2D(
                    num_nodes=(self.C, self.D),
                    order=1,
                    cond_prec=tau,
                    rescale_marginal_variance=True,
                ),
            ).reshape((self.C, self.D))

            log_cint = numpyro.deterministic("log_cint", beta0 + f)
            mu = jnp.exp(log_cint + log_N[:, jnp.newaxis])

        elif self.K_part > 1 and self.K_cnt == 1:
            # Partial stratification: Y (K_part, C, D), N (K_part, C)
            with plate("strata", self.K_part):
                beta0 = numpyro.sample(
                    "baseline", dist.Normal(jnp.log(self.Y.mean()), jnp.log(10))
                )
                tau = numpyro.sample("tau", Gumbel2(concentration=0.5, rate=2.86))

            f = numpyro.sample(
                "f",
                IGMRF2D(
                    num_nodes=(self.C, self.D),
                    order=1,
                    cond_prec=tau,
                    rescale_marginal_variance=True,
                )
                .expand([self.K_part])
                .to_event(1),
            ).reshape((self.K_part, self.C, self.D))

            log_cint = numpyro.deterministic(
                "log_cint", beta0[:, jnp.newaxis, jnp.newaxis] + f
            )
            mu = jnp.exp(log_cint + log_N[:, :, jnp.newaxis])

        else:
            # Full/mixed stratification: Y (K_part, K_cnt, C, D), N (K_part, C)
            K = self.K_part * self.K_cnt
            with plate("strata", K):
                beta0 = numpyro.sample(
                    "baseline", dist.Normal(jnp.log(self.Y.mean()), jnp.log(10))
                )
                tau = numpyro.sample("tau", Gumbel2(concentration=0.5, rate=2.86))

            f = numpyro.sample(
                "f",
                IGMRF2D(
                    num_nodes=(self.C, self.D),
                    order=1,
                    cond_prec=tau,
                    rescale_marginal_variance=True,
                )
                .expand([K])
                .to_event(1),
            ).reshape((self.K_part, self.K_cnt, self.C, self.D))

            log_cint = numpyro.deterministic(
                "log_cint",
                beta0.reshape((self.K_part, self.K_cnt))[:, :, jnp.newaxis, jnp.newaxis]
                + f,
            )
            mu = jnp.exp(log_cint + log_N[:, jnp.newaxis, :, jnp.newaxis])

        if self.likelihood == "poisson":
            numpyro.sample("obs", dist.Poisson(rate=mu), obs=y)

        if self.likelihood == "negbin":
            inv_disp = numpyro.sample("inv_disp", dist.Exponential(1.0))
            numpyro.sample(
                "obs",
                dist.NegativeBinomial2(mean=mu, concentration=1.0 / inv_disp),
                obs=y,
            )
