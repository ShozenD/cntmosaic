import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

from .. import Gumbel2

# (concentration, rate) pairs covering the moment-finiteness regimes:
# a<=1: mean & variance both nan (this is the PC-prior-for-precision case)
# 1<a<=2: mean finite, variance nan
# a>2: both mean and variance finite
PARAMS = [
    (0.5, 1.0),
    (1.5, 2.0),
    (3.0, 1.0),
    (5.0, 0.5),
]


def _scipy_dist(concentration, rate):
    scale = rate ** (1.0 / concentration)
    return stats.invweibull(c=concentration, scale=scale)


@pytest.mark.parametrize("concentration, rate", PARAMS)
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("sample_shape", [(), (5,), (4, 2)])
def test_sample_shape(concentration, rate, batch_shape, sample_shape):
    a = jnp.full(batch_shape, concentration)
    b = jnp.full(batch_shape, rate)
    d = Gumbel2(a, b)
    samples = d.sample(random.PRNGKey(0), sample_shape=sample_shape)
    assert samples.shape == sample_shape + d.batch_shape
    assert jnp.all(samples > 0)


@pytest.mark.parametrize("concentration, rate", PARAMS)
def test_log_prob_matches_scipy(concentration, rate):
    # x starts at 0.5 (rather than near 0) because scipy's invweibull.logpdf is not
    # implemented in log-space and underflows to -inf deep in the left tail, while
    # our log-space implementation stays finite there.
    x = jnp.linspace(0.5, 10.0, 50)
    d = Gumbel2(concentration, rate)
    actual = d.log_prob(x)
    expected = _scipy_dist(concentration, rate).logpdf(np.asarray(x))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("concentration, rate", PARAMS)
def test_cdf_matches_scipy(concentration, rate):
    x = jnp.linspace(0.05, 10.0, 50)
    d = Gumbel2(concentration, rate)
    actual = d.cdf(x)
    expected = _scipy_dist(concentration, rate).cdf(np.asarray(x))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("concentration, rate", PARAMS)
def test_icdf_matches_scipy_and_is_cdf_inverse(concentration, rate):
    q = jnp.linspace(0.01, 0.99, 20)
    d = Gumbel2(concentration, rate)
    x = d.icdf(q)
    expected = _scipy_dist(concentration, rate).ppf(np.asarray(q))
    np.testing.assert_allclose(np.asarray(x), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(d.cdf(x)), np.asarray(q), rtol=1e-4, atol=1e-5
    )


def test_mean_variance_matches_scipy_and_monte_carlo():
    concentration, rate = 5.0, 0.5  # a > 2: both mean and variance finite
    d = Gumbel2(concentration, rate)

    scipy_dist = _scipy_dist(concentration, rate)
    np.testing.assert_allclose(np.asarray(d.mean), scipy_dist.mean(), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(d.variance), scipy_dist.var(), rtol=1e-4)

    samples = d.sample(random.PRNGKey(42), sample_shape=(200_000,))
    np.testing.assert_allclose(
        np.asarray(d.mean), np.asarray(jnp.mean(samples)), rtol=0.05
    )
    np.testing.assert_allclose(
        np.asarray(d.variance), np.asarray(jnp.var(samples)), rtol=0.1
    )


@pytest.mark.parametrize("concentration", [0.5, 1.0])
def test_mean_is_nan_for_concentration_leq_1(concentration):
    d = Gumbel2(concentration, 1.0)
    assert jnp.isnan(d.mean)


@pytest.mark.parametrize("concentration", [0.5, 1.0, 1.5, 2.0])
def test_variance_is_nan_for_concentration_leq_2(concentration):
    d = Gumbel2(concentration, 1.0)
    assert jnp.isnan(d.variance)


def test_mean_finite_for_concentration_gt_1_but_variance_still_nan():
    d = Gumbel2(1.5, 1.0)
    assert jnp.isfinite(d.mean)
    assert jnp.isnan(d.variance)


def test_log_prob_grad():
    def fn(a, b, x):
        return jnp.sum(Gumbel2(a, b).log_prob(x))

    a, b, x = 3.0, 1.5, jnp.array([0.5, 1.0, 2.0, 3.0])
    grad_a, grad_b = jax.grad(fn, argnums=(0, 1))(a, b, x)
    assert jnp.isfinite(grad_a)
    assert jnp.isfinite(grad_b)

    # eps=1e-4 is too small relative to float32 precision (JAX's default dtype) for
    # an accurate central difference here; 1e-2 balances truncation vs. rounding error.
    eps = 1e-2
    fd_grad_a = (fn(a + eps, b, x) - fn(a - eps, b, x)) / (2 * eps)
    fd_grad_b = (fn(a, b + eps, x) - fn(a, b - eps, x)) / (2 * eps)
    np.testing.assert_allclose(
        np.asarray(grad_a), np.asarray(fd_grad_a), rtol=1e-2, atol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(grad_b), np.asarray(fd_grad_b), rtol=1e-2, atol=1e-3
    )


def test_log_prob_jit():
    def fn(a, b, x):
        return Gumbel2(a, b).log_prob(x)

    x = jnp.array([0.5, 1.0, 2.0])
    result_eager = fn(3.0, 1.5, x)
    result_jit = jax.jit(fn)(3.0, 1.5, x)
    np.testing.assert_allclose(
        np.asarray(result_eager), np.asarray(result_jit), rtol=1e-6
    )


def test_sample_jit():
    def fn(a, b, key):
        return Gumbel2(a, b).sample(key, sample_shape=(10,))

    key = random.PRNGKey(0)
    result_eager = fn(3.0, 1.5, key)
    result_jit = jax.jit(fn)(3.0, 1.5, key)
    np.testing.assert_allclose(
        np.asarray(result_eager), np.asarray(result_jit), rtol=1e-6
    )
