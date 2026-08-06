"""Type-2 Gumbel (Fréchet / inverse-Weibull) distribution.

The Type-2 Gumbel distribution, also known as the Fréchet distribution, the
inverse-Weibull distribution, or the Type-II extreme value distribution, provides
the penalized-complexity (PC) prior for the precision parameter of a Gaussian
random effect (Simpson, Rue, Riebler, Sørbye & Fuglstad, 2017, "Penalising Model
Component Complexity: A Principled, Practical Approach to Constructing Priors,"
*Statistical Science*, 32(1):1-28, Section 3.3).

For a Gaussian random effect with precision :math:`\\tau` and standard deviation
:math:`\\sigma = \\tau^{-1/2}`, Simpson et al. (2017) derive a PC prior on
:math:`\\sigma` that is exponential, :math:`\\sigma \\sim \\mathrm{Exponential}(\\lambda)`.
The density this induces on :math:`\\tau` is exactly
``Gumbel2(concentration=0.5, rate=lambda)``, equivalent to R-INLA's ``pc.prec``.
At ``concentration=0.5`` both ``mean`` and ``variance`` are ``nan`` (undefined) --
see below -- which is the expected, mathematically correct behaviour for this
prior, not a bug.
"""

import jax.numpy as jnp
from jax import lax, random
from jax.scipy.special import gammaln
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key


class Gumbel2(Distribution):
    r"""Type-2 Gumbel (Fréchet / inverse-Weibull) distribution.

    .. math::

        f(x \mid a, b) = a\,b\,x^{-a-1}\,\exp(-b\,x^{-a}), \quad x > 0

        F(x \mid a, b) = \exp(-b\,x^{-a})

    Mean (finite only for :math:`a > 1`; ``nan`` otherwise):

    .. math::

        E[X] = b^{1/a}\,\Gamma(1 - 1/a)

    Variance (finite only for :math:`a > 2`; ``nan`` otherwise):

    .. math::

        \mathrm{Var}(X) = b^{2/a}\left[\Gamma(1 - 2/a) - \Gamma(1 - 1/a)^2\right]

    where :math:`a` is ``concentration`` and :math:`b` is ``rate``. Note that
    :math:`b` is not itself the scale of :math:`X` -- the corresponding Fréchet
    scale is :math:`b^{1/a}`.

    See the module docstring for the penalized-complexity-prior interpretation and
    citation.

    :param concentration: shape parameter :math:`a > 0`.
    :param rate: rate parameter :math:`b > 0`. Default 1.0.
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["concentration", "rate"]

    def __init__(self, concentration, rate=1.0, *, validate_args=None):
        self.concentration, self.rate = promote_shapes(concentration, rate)
        batch_shape = lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(rate))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        shape = sample_shape + self.batch_shape + self.event_shape
        standard_exp = random.exponential(key, shape=shape)
        return jnp.power(self.rate, 1.0 / self.concentration) * jnp.power(
            standard_exp, -1.0 / self.concentration
        )

    @validate_sample
    def log_prob(self, value):
        a, b = self.concentration, self.rate
        log_value = jnp.log(value)
        return (
            jnp.log(a)
            + jnp.log(b)
            - (a + 1.0) * log_value
            - b * jnp.exp(-a * log_value)
        )

    def cdf(self, value):
        a, b = self.concentration, self.rate
        return jnp.exp(-b * jnp.exp(-a * jnp.log(value)))

    def icdf(self, q):
        a, b = self.concentration, self.rate
        return jnp.power(b / (-jnp.log(q)), 1.0 / a)

    @property
    def mean(self):
        a, b = self.concentration, self.rate
        value = jnp.exp(jnp.log(b) / a + gammaln(1.0 - 1.0 / a))
        return jnp.where(a > 1.0, value, jnp.nan)

    @property
    def variance(self):
        a, b = self.concentration, self.rate
        term1 = jnp.exp(2.0 * jnp.log(b) / a + gammaln(1.0 - 2.0 / a))
        term2 = jnp.exp(2.0 * (jnp.log(b) / a + gammaln(1.0 - 1.0 / a)))
        value = term1 - term2
        return jnp.where(a > 2.0, value, jnp.nan)
