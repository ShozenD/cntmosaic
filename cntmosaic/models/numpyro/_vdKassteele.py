"""NumPyro model mixin for vdKassteele."""
from typing import Optional

import jax.numpy as jnp
from jax.typing import ArrayLike


class vdKassteeleNumPyroMixin:
    """Carries the NumPyro ``model()`` for vdKassteele."""

    def model(self, y: Optional[ArrayLike] = None) -> None:
        import numpyro
        from numpyro import distributions as dist
        from numpyro.handlers import plate

        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))
        f = self.prior.sample()
        log_rate = numpyro.deterministic("log_rate", beta0 + f)

        if self.prior_type == "global":
            log_cint = numpyro.deterministic("log_cint", log_rate + self.log_P)[
                self.aid, self.bid
            ]
        else:
            log_cint = (
                log_rate[self.data.flat_ix, self.aid, self.bid]
                + self.log_P[self.data.flat_pixs, self.bid]
            )

        repeat_effect = self.hill.sample()[self.rid] if self.rid is not None else 0.0

        mu = jnp.exp(log_cint + self.log_N + self.log_S + repeat_effect)

        if self.likelihood == "poisson":
            with plate("data", len(self.y)):
                numpyro.sample("obs", dist.Poisson(rate=mu), obs=y)

        if self.likelihood == "negbin":
            inv_disp = numpyro.sample("inv_disp", dist.Exponential(1.0))
            with plate("data", len(self.y)):
                numpyro.sample(
                    "obs",
                    dist.NegativeBinomial2(mean=mu, concentration=1.0 / inv_disp),
                    obs=y,
                )
