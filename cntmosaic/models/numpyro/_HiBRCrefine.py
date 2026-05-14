"""NumPyro model mixin for HiBRCrefine."""
from typing import Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from .._math import inverse_clr, kron_sum_mode_1
from .._utils import index_mask_logsumexp


class HiBRCrefineNumPyroMixin:
    """Carries ``sample_log_delta()`` and ``model()`` for HiBRCrefine."""

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
        aid_exp: Optional[ArrayLike] = None,
        bid_pad: Optional[ArrayLike] = None,
        flat_ix_exp: Optional[ArrayLike] = None,
        log_N: Optional[ArrayLike] = None,
        log_V: Optional[ArrayLike] = None,
        rid: Optional[ArrayLike] = None,
    ) -> None:
        import numpyro
        from numpyro import distributions as dist
        from numpyro.handlers import plate, scope

        len_y = len(self.y) if y is None else len(y)
        aid_exp = self.data.aid_exp if aid_exp is None else aid_exp
        bid_pad = self.data.bid_pad if bid_pad is None else bid_pad
        flat_ix_exp = self.data.flat_ix_exp if flat_ix_exp is None else flat_ix_exp
        log_N = self.log_N if log_N is None else log_N
        log_V = self.log_V if log_V is None else log_V
        rid = self.rid if hasattr(self, "rid") and rid is None else rid

        beta0 = numpyro.sample("baseline", dist.Normal(-self.log_P.mean(), 2.5))

        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        log_rate = numpyro.deterministic("log_rate", beta0 + f)

        log_delta = self.sample_log_delta()
        log_cint_tensor = (
            log_rate[jnp.newaxis, :, :]
            + self.log_P[:, jnp.newaxis, :]
            + log_delta
        )

        log_cint = index_mask_logsumexp(log_cint_tensor, aid_exp, bid_pad, flat_ix_exp)

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
