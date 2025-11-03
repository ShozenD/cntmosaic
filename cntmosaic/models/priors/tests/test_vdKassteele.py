"""
Unit tests for vdKassteele IGMRF prior class.

Tests cover initialization, configuration, sampling methods, and edge cases
for all three prior types: global, partial, and full.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro

from cntmosaic.models.priors import vdKassteele


class TestInitialization:
    """Test vdKassteele initialization and validation."""

    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        prior = vdKassteele(prior_type="global")

        assert prior.prior_type == "global"
        assert prior.order == 2
        assert prior.tau_shape == 2.0
        assert prior.tau_rate == 0.01
        assert prior.transform is None

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        prior = vdKassteele(
            prior_type="partial",
            order=1,
            tau_shape=3.0,
            tau_rate=0.05,
        )

        assert prior.prior_type == "partial"
        assert prior.order == 1
        assert prior.tau_shape == 3.0
        assert prior.tau_rate == 0.05

    def test_invalid_order(self):
        """Test that invalid order raises ValueError."""
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            vdKassteele(order=3)


class TestAgeBounds:
    """Test age bounds configuration."""

    def test_valid_age_bounds(self):
        """Test setting valid age bounds."""
        prior = vdKassteele(prior_type="global")
        prior.set_age_bounds(0, 80)

        assert prior.min_age == 0
        assert prior.max_age == 80
        assert prior.A == 81

    def test_negative_min_age(self):
        """Test that negative min_age raises ValueError."""
        prior = vdKassteele(prior_type="global")

        with pytest.raises(ValueError, match="min_age must be non-negative"):
            prior.set_age_bounds(-5, 80)

    def test_invalid_age_range(self):
        """Test that max_age <= min_age raises ValueError."""
        prior = vdKassteele(prior_type="global")

        with pytest.raises(ValueError, match="max_age must be greater than min_age"):
            prior.set_age_bounds(50, 50)

        with pytest.raises(ValueError, match="max_age must be greater than min_age"):
            prior.set_age_bounds(60, 50)

    def test_grid_initialization(self):
        """Test that grid structures are initialized."""
        prior = vdKassteele(prior_type="global")
        prior.set_age_bounds(0, 20)

        # Check that symmetrization indices are created
        assert hasattr(prior, "sym_idx")
        assert prior.sym_idx is not None


class TestSampleSingle:
    """Test symmetric (global) sampling."""

    def test_sample_single_shape(self):
        """Test that sample_single returns correct shape."""
        prior = vdKassteele(prior_type="global", order=2)
        prior.set_age_bounds(0, 40)

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample_single()

        assert f.shape == (prior.A, prior.A)

    def test_sample_single_symmetric(self):
        """Test that sample_single produces symmetric matrix."""
        prior = vdKassteele(prior_type="global", order=2)
        prior.set_age_bounds(0, 30)

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample_single()

        # Check symmetry
        assert jnp.allclose(f, f.T, atol=1e-6)

    def test_sample_single_first_order(self):
        """Test sample_single with first-order penalty."""
        prior = vdKassteele(prior_type="global", order=1)
        prior.set_age_bounds(0, 25)

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample_single()

        assert f.shape == (prior.A, prior.A)
        assert jnp.allclose(f, f.T, atol=1e-6)


class TestSamplePartial:
    """Test partial (asymmetric) sampling."""

    def test_sample_partial_shape(self):
        """Test that sample_partial returns correct shape."""
        prior = vdKassteele(prior_type="partial", order=2)
        prior.set_age_bounds(0, 40)
        prior.set_event_dim(4)

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample_partial()

        # Note: Returns event_dim, not event_dim_eff
        assert f.shape == (4, prior.A, prior.A)

    def test_sample_partial_not_symmetric(self):
        """Test that sample_partial can produce asymmetric matrices."""
        prior = vdKassteele(prior_type="partial", order=2)
        prior.set_age_bounds(0, 30)
        prior.set_event_dim(3)

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample_partial()

        # Check that at least one matrix is not symmetric
        # (with high probability, IGMRF samples won't be symmetric)
        non_symmetric = False
        for k in range(f.shape[0]):
            if not jnp.allclose(f[k], f[k].T, atol=1e-6):
                non_symmetric = True
                break

        assert non_symmetric, "Expected at least one non-symmetric matrix"

    def test_sample_partial_multiple_realizations(self):
        """Test sample_partial with different event dimensions."""
        for event_dim in [2, 4, 9]:
            prior = vdKassteele(prior_type="partial", order=2)
            prior.set_age_bounds(0, 30)
            prior.set_event_dim(event_dim)

            with numpyro.handlers.seed(rng_seed=42):
                f = prior.sample_partial()

            assert f.shape == (event_dim, prior.A, prior.A)


class TestSampleFull:
    """Test full (diagonal/off-diagonal) sampling."""

    def test_sample_full_shape(self):
        """Test that sample_full returns correct shape."""
        prior = vdKassteele(prior_type="full", order=2)
        prior.set_age_bounds(0, 40)
        prior.set_event_dim(9)  # 3x3 stratification

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample_full()

        assert f.shape == (9, prior.A, prior.A)

    def test_sample_full_diagonal_symmetric(self):
        """Test that diagonal blocks in sample_full are symmetric."""
        prior = vdKassteele(prior_type="full", order=2)
        prior.set_age_bounds(0, 30)
        prior.set_event_dim(9)  # 3x3

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample_full()

        # Check diagonal blocks (indices 0, 4, 8 for 3x3)
        for k in [0, 4, 8]:
            assert jnp.allclose(f[k], f[k].T, atol=1e-6)

    def test_sample_full_off_diagonal_asymmetric(self):
        """Test that off-diagonal blocks can be asymmetric."""
        prior = vdKassteele(prior_type="full", order=2)
        prior.set_age_bounds(0, 25)
        prior.set_event_dim(9)  # 3x3

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample_full()

        # Check at least one off-diagonal block is not symmetric
        non_symmetric = False
        for k in [1, 2, 3, 5, 6, 7]:  # Off-diagonal indices for 3x3
            if not jnp.allclose(f[k], f[k].T, atol=1e-6):
                non_symmetric = True
                break

        assert non_symmetric, "Expected at least one non-symmetric off-diagonal block"

    def test_sample_full_different_stratifications(self):
        """Test sample_full with different stratification sizes."""
        for event_dim in [4, 9, 16]:  # 2x2, 3x3, 4x4
            prior = vdKassteele(prior_type="full", order=2)
            prior.set_age_bounds(0, 30)
            prior.set_event_dim(event_dim)

            with numpyro.handlers.seed(rng_seed=42):
                f = prior.sample_full()

            assert f.shape == (event_dim, prior.A, prior.A)


class TestSampleDispatch:
    """Test main sample() method dispatching."""

    def test_sample_global(self):
        """Test that sample() dispatches to sample_single for global."""
        prior = vdKassteele(prior_type="global")
        prior.set_age_bounds(0, 35)

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample()

        assert f.shape == (prior.A, prior.A)
        assert jnp.allclose(f, f.T)

    def test_sample_partial(self):
        """Test that sample() dispatches to sample_partial."""
        prior = vdKassteele(prior_type="partial")
        prior.set_age_bounds(0, 35)
        prior.set_event_dim(4)

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample()

        assert f.shape == (4, prior.A, prior.A)

    def test_sample_full(self):
        """Test that sample() dispatches to sample_full."""
        prior = vdKassteele(prior_type="full")
        prior.set_age_bounds(0, 35)
        prior.set_event_dim(9)

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample()

        assert f.shape == (9, prior.A, prior.A)

    def test_sample_invalid_type(self):
        """Test that invalid prior_type raises ValueError."""
        prior = vdKassteele(prior_type="global")
        prior.prior_type = "invalid"  # Manually set invalid type
        prior.set_age_bounds(0, 30)

        with pytest.raises(ValueError, match="Unknown prior_type"):
            with numpyro.handlers.seed(rng_seed=42):
                prior.sample()


class TestNumericalProperties:
    """Test numerical properties of samples."""

    def test_sample_values_finite(self):
        """Test that samples contain finite values."""
        prior = vdKassteele(prior_type="global", order=2)
        prior.set_age_bounds(0, 30)

        with numpyro.handlers.seed(rng_seed=42):
            f = prior.sample()

        assert jnp.all(jnp.isfinite(f))

    def test_sample_reproducibility(self):
        """Test that samples are reproducible with same seed."""
        prior1 = vdKassteele(prior_type="partial", order=2)
        prior1.set_age_bounds(0, 30)
        prior1.set_event_dim(4)

        prior2 = vdKassteele(prior_type="partial", order=2)
        prior2.set_age_bounds(0, 30)
        prior2.set_event_dim(4)

        with numpyro.handlers.seed(rng_seed=42):
            f1 = prior1.sample()

        with numpyro.handlers.seed(rng_seed=42):
            f2 = prior2.sample()

        assert jnp.allclose(f1, f2)

    def test_different_seeds_different_samples(self):
        """Test that different seeds produce different samples."""
        prior = vdKassteele(prior_type="global", order=2)
        prior.set_age_bounds(0, 30)

        with numpyro.handlers.seed(rng_seed=42):
            f1 = prior.sample()

        with numpyro.handlers.seed(rng_seed=43):
            f2 = prior.sample()

        assert not jnp.allclose(f1, f2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
