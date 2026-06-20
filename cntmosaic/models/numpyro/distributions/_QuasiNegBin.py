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


class QuasiNegBin(Distribution):
    r"""Quasi-likelihood with negative-binomial-type variance function
    :math:`V(\mu) = \mu + \mu^2 / k`.

    Second-order assumptions:
    :math:`E(Y) = \mu`, :math:`\mathrm{var}(Y) = \psi (\mu + \mu^2 / k)`.

    From Table S1 of Agnoletto, Rigon & Dunson (2025, Biometrika), the
    log-quasi-likelihood is

    .. math::

        \log q(y \mid \mu, k, \psi)
          = \frac{1}{\psi} \left\{ y \log\frac{\mu}{k + \mu}
            + k \log\frac{k}{k + \mu} \right\},
        \qquad \mu > 0, \; y \ge 0,

    obtained from :math:`\int_a^\mu (y - t) / V(t)\, dt` by partial fractions
    (:math:`k / \{t (k + t)\} = 1/t - 1/(k + t)`), dropping :math:`\beta`-free
    terms. The quasi-score is

    .. math::

        \partial_\mu \log q
          = \frac{k (y - \mu)}{\psi\, \mu (k + \mu)}
          = \frac{y - \mu}{\psi V(\mu)}.

    With :math:`\psi = 1` this equals the NegativeBinomial2 (mean,
    concentration) log-pmf up to ``y``-only constants, but it is a valid
    generalized-Bayes loss for *any* data satisfying the two moment
    conditions, including non-integer nonnegative responses. As with
    :class:`QuasiGamma`, ``exp(log_prob)`` is not a normalized density in
    ``y``, so there is no ``sample()``.

    Parameter roles differ: :math:`k` shapes the mean-variance relationship
    and *is* part of the loss (all :math:`k`-dependent terms are kept, so it
    may in principle be estimated, though fixing it from a preliminary fit is
    the more standard and better-behaved choice); :math:`\psi` is the
    loss-scale, to be held fixed at a calibrated value (e.g. the Pearson
    estimator), never sampled.

    :param mu: mean parameter, :math:`\mu > 0`.
    :param concentration: shape parameter :math:`k > 0` of the variance
        function; smaller :math:`k` means more overdispersion. As
        :math:`k \to \infty` the loss approaches the Poisson quasi-likelihood
        :math:`(y \log \mu - \mu)/\psi`.
    :param dispersion: loss-scale / dispersion parameter :math:`\psi > 0`.
    """

    arg_constraints = {
        "mu": constraints.positive,
        "concentration": constraints.positive,
        "dispersion": constraints.positive,
    }
    support = constraints.nonnegative

    def __init__(self, mu, concentration, dispersion=1.0, *, validate_args=None):
        self.mu, self.concentration, self.dispersion = promote_shapes(
            mu, concentration, dispersion
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(mu), jnp.shape(concentration), jnp.shape(dispersion)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @validate_sample
    def log_prob(self, value):
        ftype = jnp.result_type(float)
        value = jnp.astype(value, ftype)
        mu = jnp.astype(self.mu, ftype)
        k = jnp.astype(self.concentration, ftype)
        # y log mu - (y + k) log(k + mu) + k log k ; xlogy handles y = 0.
        return (
            xlogy(value, mu) - (value + k) * jnp.log(k + mu) + xlogy(k, k)
        ) / self.dispersion

    def sample(self, key, sample_shape=()):
        raise NotImplementedError(
            "QuasiNegativeBinomial is a quasi-likelihood, not a probability "
            "distribution; it has no sampler. Use it only as an observed "
            "likelihood (numpyro.sample(..., obs=y))."
        )

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.dispersion * (self.mu + jnp.square(self.mu) / self.concentration)
