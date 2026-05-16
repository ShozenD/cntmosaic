"""NumPyro model mixin for AgeMixCC."""

from typing import Optional

import jax.numpy as jnp
from jax.typing import ArrayLike


class AgeMixCCNumPyroMixin:
    """Carries the NumPyro ``model()`` for AgeMixCC."""

    def model(
        self,
        cid: Optional[ArrayLike] = None,
        did: Optional[ArrayLike] = None,
        rid: Optional[ArrayLike] = None,
        log_N: Optional[ArrayLike] = None,
        log_V: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
    ) -> None:
        import numpyro
        from numpyro import distributions as dist
        from numpyro.handlers import plate, scope

        cid = self.cid if cid is None else cid
        did = self.did if did is None else did
        log_N = self.log_N if log_N is None else log_N
        log_V = self.log_V if log_V is None else log_V
        rid = getattr(self, "rid", None) if rid is None else rid
        len_y = len(self.y) if y is None else len(y)

        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))

        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        log_rate = numpyro.deterministic("log_rate", beta0 + f)
        log_cint = numpyro.deterministic("log_cint", log_rate + self.log_P)

        repeat_effect = self.hill.sample()[rid] if hasattr(self, "rid") else 0.0

        aggregated_log_cint = log_cint[cid, did]

        mu = jnp.exp(aggregated_log_cint + repeat_effect + log_N + log_V)

        if self.likelihood == "poisson":
            with plate("data", len_y):
                numpyro.sample("obs", dist.Poisson(rate=mu), obs=y)

        if self.likelihood == "negbin":
            inv_disp = numpyro.sample("inv_disp", dist.Exponential(1.0))
            with plate("data", len_y):
                numpyro.sample(
                    "obs",
                    dist.NegativeBinomial2(mean=mu, concentration=1.0 / inv_disp),
                    obs=y,
                )
