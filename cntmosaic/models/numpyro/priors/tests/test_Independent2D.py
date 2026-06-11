import jax.numpy as jnp
import numpyro
import pytest

from .._Independent2D import Independent2D


class TestInitialization:

    def test_default_scale(self):
        prior = Independent2D(prior_type="global")
        assert prior.scale == 1.0

    def test_custom_scale(self):
        prior = Independent2D(prior_type="partial", scale=0.5)
        assert prior.scale == 0.5

    def test_prior_type_stored(self):
        for pt in ("global", "partial", "full"):
            prior = Independent2D(prior_type=pt)
            assert prior.prior_type == pt

    def test_invalid_prior_type(self):
        with pytest.raises(ValueError, match="prior_type must be one of"):
            Independent2D(prior_type="invalid")


class TestAgeBounds:

    def test_valid_age_bounds(self):
        prior = Independent2D(prior_type="global")
        prior.set_age_bounds(0, 15)
        assert prior.min_age == 0
        assert prior.max_age == 15
        assert prior.A == 16
        assert hasattr(prior, "symm_tril_ix")


class TestSampling:

    A = 10

    def test_sample_global_shape(self):
        prior = Independent2D(prior_type="global")
        prior.set_age_bounds(0, self.A - 1)
        with numpyro.handlers.seed(rng_seed=0):
            f = prior.sample_global()
        assert f.shape == (self.A, self.A)

    def test_sample_global_symmetric(self):
        prior = Independent2D(prior_type="global")
        prior.set_age_bounds(0, self.A - 1)
        with numpyro.handlers.seed(rng_seed=0):
            f = prior.sample_global()
        assert jnp.allclose(f, f.T)

    def test_sample_partial_shape(self):
        K = 3
        prior = Independent2D(prior_type="partial")
        prior.set_age_bounds(0, self.A - 1)
        prior.set_event_dim(K)
        prior.loc = 0.0
        with numpyro.handlers.seed(rng_seed=0):
            f = prior.sample_partial()
        assert f.shape == (K, self.A, self.A)

    def test_sample_full_shape(self):
        K = 2
        prior = Independent2D(prior_type="full")
        prior.set_age_bounds(0, self.A - 1)
        prior.set_event_dim(K)
        prior.loc = 0.0
        with numpyro.handlers.seed(rng_seed=0):
            f = prior.sample_full()
        assert f.shape == (K * K, self.A, self.A)
