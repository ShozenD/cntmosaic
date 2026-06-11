"""Quasi-likelihood "distributions" for quasi-posterior inference in NumPyro.

Implements Wedderburn (1974) log-quasi-likelihoods as NumPyro
:class:`~numpyro.distributions.Distribution` classes, for use as the likelihood
term of a quasi-posterior in the sense of Agnoletto, Rigon & Dunson (2025,
Biometrika 112(2), asaf022):

- :class:`QuasiPoisson`: linear variance, V(mu) = mu.
- :class:`QuasiGamma`: quadratic variance, V(mu) = mu**2.
- :class:`QuasiNegativeBinomial`: NB-type variance, V(mu) = mu + mu**2/k.

Per observation, with mean ``mu > 0``, dispersion (loss-scale) ``psi > 0`` and
response ``y >= 0``:

.. math::

    \\ell_Q(\\mu; y, \\psi)
      = \\frac{1}{\\psi} \\int_a^{\\mu} \\frac{y - t}{t^2} \\, dt
      = \\frac{1}{\\psi} \\left( -\\frac{y}{\\mu} - \\log \\mu \\right)
      + \\text{const}(y),

matching Table S1 of the paper's Supplementary Material. Terms depending only
on ``y`` (and on the arbitrary anchor ``a``) are dropped: they cancel in the
quasi-posterior, which is defined only up to a normalizing constant in ``beta``.

This object is *not* a probability distribution: ``exp(log_prob)`` does not
integrate to one in ``y``. It coincides (up to y-only constants) with a Gamma
density with shape ``1/psi`` and mean ``mu`` when the data really are gamma,
but it remains a valid generalized-Bayes loss whenever only the first two
moments ``E(Y) = mu``, ``var(Y) = psi * mu**2`` are correctly specified.
Consequently ``sample()`` is intentionally not implemented, and quantities that
require a normalized likelihood (prior/posterior predictive draws, marginal
likelihood) are not meaningful with this class.
"""

import jax.numpy as jnp
from jax import lax
from jax.scipy.special import xlogy
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes, validate_sample


class QuasiGamma(Distribution):
    r"""Quasi-likelihood with quadratic variance function :math:`V(\mu) = \mu^2`.

    Second-order assumptions: :math:`E(Y) = \mu`, :math:`\mathrm{var}(Y) = \psi \mu^2`.

    .. math::

        \log q(y \mid \mu, \psi)
          = \frac{1}{\psi}\left(-\frac{y}{\mu} - \log\mu\right)

    The quasi-score is :math:`\partial_\mu \log q = (y - \mu) / (\psi \mu^2)`,
    i.e. :math:`(y - \mu)/\{\psi V(\mu)\}` with :math:`V(\mu) = \mu^2`.

    :param mu: mean parameter, :math:`\mu > 0`.
    :param dispersion: loss-scale / dispersion parameter :math:`\psi > 0`.
        In a quasi-posterior this is held fixed at a calibrated value
        (e.g. the method-of-moments / Pearson estimator), not sampled.
    """

    arg_constraints = {
        "mu": constraints.positive,
        "dispersion": constraints.positive,
    }
    support = constraints.nonnegative

    def __init__(self, mu, dispersion=1.0, *, validate_args=None):
        self.mu, self.dispersion = promote_shapes(mu, dispersion)
        batch_shape = lax.broadcast_shapes(jnp.shape(mu), jnp.shape(dispersion))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value):
        ftype = jnp.result_type(float)
        value = jnp.astype(value, ftype)
        mu = jnp.astype(self.mu, ftype)
        return -(value / mu + jnp.log(mu)) / self.dispersion

    def sample(self, key, sample_shape=()):
        raise NotImplementedError(
            "QuasiGamma is a quasi-likelihood, not a probability distribution; "
            "it has no sampler. Use it only as an observed likelihood "
            "(numpyro.sample(..., obs=y))."
        )

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.dispersion * jnp.square(self.mu)
