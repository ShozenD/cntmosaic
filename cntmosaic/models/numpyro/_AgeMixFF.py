"""NumPyro model mixin for AgeMixFF."""
from typing import Optional

import jax.numpy as jnp
from jax.typing import ArrayLike


class AgeMixFFNumPyroMixin:
    """Carries the NumPyro ``model()`` for AgeMixFF.

    Mixed in before ``GenMix`` in the MRO so that the abstract ``model()``
    on ``ContactModel`` / ``GenMix`` is satisfied without touching those classes.
    """

    def model(
        self,
        y: Optional[ArrayLike] = None,
        aid: Optional[ArrayLike] = None,
        bid: Optional[ArrayLike] = None,
        rid: Optional[ArrayLike] = None,
        log_N: Optional[ArrayLike] = None,
        log_V: Optional[ArrayLike] = None,
    ) -> None:
        import numpyro
        from numpyro import distributions as dist
        from numpyro.handlers import scope

        aid = self.aid if aid is None else aid
        bid = self.bid if bid is None else bid
        log_N = self.log_N if log_N is None else log_N
        log_V = self.log_V if log_V is None else log_V
        rid = getattr(self, "rid", None) if rid is None else rid
        len_y = len(self.y) if y is None else len(y)

        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))

        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        log_rate = numpyro.deterministic("log_rate", beta0 + f)
        log_cint = numpyro.deterministic("log_cint", log_rate + self.log_P)

        repeat_effect = self.hill.sample()[rid] if rid is not None else 0.0

        mu = numpyro.deterministic(
            "mu",
            jnp.exp(log_cint[aid, bid] + log_N + log_V + repeat_effect),
        )

        if self.likelihood == "poisson":
            with numpyro.plate("data", len_y):
                numpyro.sample("obs", dist.Poisson(rate=mu), obs=y)

        elif self.likelihood == "negbin":
            inv_disp = numpyro.sample("inv_disp", dist.Exponential(1.0))
            with numpyro.plate("data", len_y):
                numpyro.sample(
                    "obs",
                    dist.NegativeBinomial2(mean=mu, concentration=1.0 / inv_disp),
                    obs=y,
                )
