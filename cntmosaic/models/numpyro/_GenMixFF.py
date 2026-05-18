"""NumPyro model mixin for GenMixFF."""
from typing import Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from .._math import inverse_clr, kron_sum_mode_1


class GenMixFFNumPyroMixin:
    """Carries ``sample_log_delta()`` and ``model()`` for GenMixFF."""

    def sample_log_delta(self) -> ArrayLike:
        import numpyro
        from numpyro.handlers import scope

        Omega = None
        for var, prior in self.priors.items():
            if var == "rate":
                continue
            with scope(prefix=var):
                if Omega is None:
                    Omega = prior.sample()
                else:
                    Omega = kron_sum_mode_1(Omega, prior.sample())

        delta = inverse_clr(Omega)
        return numpyro.deterministic(
            "log_delta", jnp.log(delta) - jnp.log(self.data.multipliers)
        )

    def model(
        self,
        y: Optional[ArrayLike] = None,
        aid: Optional[ArrayLike] = None,
        bid: Optional[ArrayLike] = None,
        rid: Optional[ArrayLike] = None,
        flat_ix: Optional[ArrayLike] = None,
        flat_pixs: Optional[ArrayLike] = None,
        log_N: Optional[ArrayLike] = None,
        log_V: Optional[ArrayLike] = None,
    ) -> None:
        import numpyro
        from numpyro import distributions as dist
        from numpyro.handlers import plate, scope

        aid = self.aid if aid is None else aid
        bid = self.bid if bid is None else bid
        rid = getattr(self, "rid", None) if rid is None else rid
        flat_ix = self.data.flat_ix if flat_ix is None else flat_ix
        flat_pixs = self.data.flat_pixs if flat_pixs is None else flat_pixs
        log_N = self.log_N if log_N is None else log_N
        log_V = self.log_V if log_V is None else log_V
        len_y = len(self.y) if y is None else len(y)

        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))

        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        log_rate = numpyro.deterministic("log_rate", beta0 + f)

        log_cint = log_rate[aid, bid]
        log_cint += self.sample_log_delta()[flat_ix, aid, bid]
        log_cint += self.log_P[flat_pixs, bid]

        repeat_effect = self.hill.sample()[rid] if hasattr(self, "rid") else 0.0

        mu = jnp.exp(log_cint + log_N + log_V + repeat_effect)

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
