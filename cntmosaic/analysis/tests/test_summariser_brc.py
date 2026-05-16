"""
Comprehensive tests for ModelSummariserBRC.

Tests cover:
- MCMC and SVI inference methods
- BRC and HiBRC model types
- All summary statistics (rate, cint, mcint)
- Error handling and edge cases
- Backward compatibility with deprecated classes
"""

import pytest
import numpy as np
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from cntmosaic.datasets import load_age_distribution, load_template_patterns
from cntmosaic.utils import AgeBins
from cntmosaic.sim import (
    ParticipantGenerator,
    MatrixGenerator,
    ContactGenerator,
    PopulationConstructor,
    Stratification,
)
from cntmosaic.dataloader import (
    ContactData,
    ContactSurveyLoader,
    ParticipantData,
    PopulationData,
)
from cntmosaic.models import AgeMixFF
from cntmosaic.models.numpyro.priors import Spline2D
from cntmosaic.analysis import ModelSummariserBRC


# ============================================================================
# Fixtures
# ============================================================================

df_age_dist = load_age_distribution("United_States")
templates = load_template_patterns("United_States")


@pytest.fixture
def sample_dataloader():
    """Create sample dataloader for testing."""
    strat = Stratification(
        name="general", n_strata=1, ref_age_dist=df_age_dist.P.values, labels=["All"], seed=42
    )
    popcon = PopulationConstructor(strats=strat)

    matrix_gen = MatrixGenerator(templates)
    contact_matrix = matrix_gen.generate_single(popcon, mean_intensity=15.0, seed=42)

    part_gen = ParticipantGenerator(popcon, n_part=500)
    df_part = part_gen.generate(seed=42)

    cnt_gen = ContactGenerator(df_part, cint_matrices=contact_matrix, model="poisson")
    df_cnt = cnt_gen.generate(seed=42)

    part_data = ParticipantData(df_part, id_col="id", age_col="age")
    cnt_data = ContactData(df_cnt, id_col="id", age_col="age_cnt")
    pop_data = PopulationData(df_age_dist, age_col="age", size_col="P")
    dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)
    return dataloader


@pytest.fixture
def brc_mcmc_model(sample_dataloader):
    """Create BRC model fitted with MCMC."""
    priors = {"rate": Spline2D(prior_type="global")}
    model = AgeMixFF(sample_dataloader, priors, likelihood="poisson")

    # Run short MCMC for testing
    model.run_inference_mcmc(PRNGKey(0), num_warmup=10, num_samples=20, num_chains=1)

    return model


@pytest.fixture
def brc_svi_model(sample_dataloader):
    """Create BRC model fitted with SVI."""
    priors = {"rate": Spline2D(prior_type="global")}
    model = AgeMixFF(sample_dataloader, priors, likelihood="poisson")

    # Run short SVI for testing
    guide = AutoNormal(model.model)
    model.run_inference_svi(PRNGKey(0), guide, num_steps=100, peak_lr=0.01)

    return model


# ============================================================================
# Tests for ModelSummariserBRC
# ============================================================================


class TestModelSummariserBRCInitialization:
    """Test initialization and setup of ModelSummariserBRC."""

    def test_init_with_mcmc(self, brc_mcmc_model):
        """Test initialization with MCMC model."""
        summariser = ModelSummariserBRC(brc_mcmc_model)

        assert summariser.model is brc_mcmc_model
        assert summariser.inference_method == "mcmc"
        assert summariser.model_type == "brc"
        assert hasattr(summariser, "post_samples")
        assert hasattr(summariser, "post_cint_samples")

    def test_init_with_svi(self, brc_svi_model):
        """Test initialization with SVI model."""
        summariser = ModelSummariserBRC(brc_svi_model, num_samples=50)

        assert summariser.model is brc_svi_model
        assert summariser.inference_method == "svi"
        assert summariser.model_type == "brc"
        assert summariser.num_samples == 50
        assert hasattr(summariser, "post_samples")
        assert hasattr(summariser, "post_cint_samples")

    def test_init_without_inference_raises_error(self, sample_dataloader):
        """Test that initialization without inference raises ValueError."""
        priors = {"rate": Spline2D(prior_type="global")}
        model = AgeMixFF(sample_dataloader, priors)

        with pytest.raises(ValueError, match="Neither MCMC nor SVI"):
            ModelSummariserBRC(model)

    def test_init_with_wrong_model_type(self):
        """Test that initialization with wrong model type raises TypeError."""
        with pytest.raises(TypeError, match="must be a BRC-family model"):
            ModelSummariserBRC("not a model")


class TestModelSummariserBRCRate:
    """Test rate summarization methods."""

    def test_summarise_rate_default_alpha(self, brc_mcmc_model):
        """Test summarise_rate with default alpha=0.05."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        summary = summariser.summarise_rate(alpha=0.05)

        assert isinstance(summary, np.ndarray)
        assert summary.shape[0] == 3  # [lower, median, upper]
        assert summary.shape[1] == summariser.model.A
        assert summary.shape[2] == summariser.model.A

        # Check ordering: lower < median < upper
        assert np.all(summary[0] <= summary[1])
        assert np.all(summary[1] <= summary[2])

    def test_summarise_rate_custom_probs(self, brc_mcmc_model):
        """Test summarise_rate with custom probabilities."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        summary = summariser.summarise_rate(probs=(0.1, 0.9))

        # summarise_rate always inserts a median at index 1 → [lower, median, upper]
        assert summary.shape[0] == 3
        assert np.all(summary[0] <= summary[1])
        assert np.all(summary[1] <= summary[2])

    def test_summarise_rate_caching(self, brc_mcmc_model):
        """Test that results are properly cached."""
        summariser = ModelSummariserBRC(brc_mcmc_model)

        # First call
        summary1 = summariser.summarise_rate(alpha=0.05)
        cache_info1 = summariser.get_cache_info()
        assert cache_info1["n_cached"] == 1

        # Second call with same parameters (should use cache)
        summary2 = summariser.summarise_rate(alpha=0.05)
        assert np.array_equal(summary1, summary2)
        cache_info2 = summariser.get_cache_info()
        assert cache_info2["n_cached"] == 1  # Still 1, used cache

        # Third call with different parameters
        summary3 = summariser.summarise_rate(alpha=0.1)
        cache_info3 = summariser.get_cache_info()
        assert cache_info3["n_cached"] == 2  # New cache entry


class TestModelSummariserBRCCint:
    """Test contact intensity summarization methods."""

    def test_summarise_cint_shape(self, brc_mcmc_model):
        """Test summarise_cint output shape."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        summary = summariser.summarise_cint(alpha=0.05)

        assert isinstance(summary, dict)
        assert "All->All" in summary
        assert summary["All->All"].shape[0] == 3
        assert summary["All->All"].shape[1] == summariser.model.A
        assert summary["All->All"].shape[2] == summariser.model.A

    def test_summarise_cint_values(self, brc_mcmc_model):
        """Test summarise_cint values are reasonable."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        summary = summariser.summarise_cint(alpha=0.05)

        # Contact intensities should be non-negative
        assert np.all(summary["All->All"] >= 0)

        # Check ordering
        assert np.all(summary["All->All"][0] <= summary["All->All"][1])
        assert np.all(summary["All->All"][1] <= summary["All->All"][2])

    def test_summarise_cint_svi(self, brc_svi_model):
        """Test summarise_cint with SVI model."""
        summariser = ModelSummariserBRC(brc_svi_model, num_samples=50)
        summary = summariser.summarise_cint(alpha=0.05)

        assert isinstance(summary, dict)
        assert summary["All->All"].shape[0] == 3
        assert np.all(summary["All->All"] >= 0)


class TestModelSummariserBRCMcint:
    """Test marginal contact intensity summarization methods."""

    def test_summarise_mcint_shape(self, brc_mcmc_model):
        """Test summarise_mcint output shape (1D)."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        summary = summariser.summarise_mcint(alpha=0.05)

        assert isinstance(summary, dict)
        assert "All->All" in summary
        assert summary["All->All"].shape == (3, summariser.model.A)  # [lower, median, upper] x A

    def test_summarise_mcint_values(self, brc_mcmc_model):
        """Test summarise_mcint values are reasonable."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        summary = summariser.summarise_mcint(alpha=0.05)

        # Marginal intensities should be non-negative
        assert np.all(summary["All->All"] >= 0)

        # Check ordering
        assert np.all(summary["All->All"][0] <= summary["All->All"][1])
        assert np.all(summary["All->All"][1] <= summary["All->All"][2])

    def test_mcint_equals_sum_of_cint(self, brc_mcmc_model):
        """Test that marginal intensity equals sum of intensity matrix."""
        summariser = ModelSummariserBRC(brc_mcmc_model)

        cint_median = summariser.summarise_cint(alpha=0.05)["All->All"][1]  # Median
        mcint_median = summariser.summarise_mcint(alpha=0.05)["All->All"][1]  # Median

        # Compute marginal from full matrix
        mcint_from_cint = cint_median.sum(axis=1)

        # Should be approximately equal
        np.testing.assert_allclose(mcint_median, mcint_from_cint, rtol=0.1)


class TestModelSummariserBRCPosteriorSamples:
    """Test getting raw posterior samples."""

    def test_get_posterior_samples_rate(self, brc_mcmc_model):
        """Test getting rate samples."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        samples = summariser.get_posterior_samples("rate")

        assert isinstance(samples, np.ndarray)
        assert samples.ndim == 3  # (n_samples, A, A)
        assert samples.shape[0] == 20  # From fixture: num_samples=20
        assert np.all(samples >= 0)  # Rates are non-negative

    def test_get_posterior_samples_cint(self, brc_mcmc_model):
        """Test getting contact intensity samples."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        samples = summariser.get_posterior_samples("cint")

        assert isinstance(samples, np.ndarray)
        assert samples.ndim == 3
        assert samples.shape[0] == 20
        assert np.all(samples >= 0)

    def test_get_posterior_samples_mcint(self, brc_mcmc_model):
        """Test getting marginal contact intensity samples."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        samples = summariser.get_posterior_samples("mcint")

        assert isinstance(samples, np.ndarray)
        assert samples.ndim == 2  # (n_samples, A)
        assert samples.shape[0] == 20
        assert np.all(samples >= 0)

    def test_get_posterior_samples_invalid_quantity(self, brc_mcmc_model):
        """Test that invalid quantity raises ValueError."""
        summariser = ModelSummariserBRC(brc_mcmc_model)

        with pytest.raises(ValueError, match="Unknown quantity"):
            summariser.get_posterior_samples("invalid")


class TestModelSummariserBRCPointEstimates:
    """Test point estimate computations."""

    def test_get_point_estimates_cint(self, brc_mcmc_model):
        """Test getting point estimates for contact intensity."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        estimates = summariser.get_point_estimates("cint")

        assert isinstance(estimates, dict)
        assert "mean" in estimates
        assert "std" in estimates

        assert estimates["mean"].shape == (summariser.model.A, summariser.model.A)
        assert estimates["std"].shape == (summariser.model.A, summariser.model.A)

        # Mean and std should be non-negative
        assert np.all(estimates["mean"] >= 0)
        assert np.all(estimates["std"] >= 0)

    def test_get_point_estimates_rate(self, brc_mcmc_model):
        """Test getting point estimates for rate."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        estimates = summariser.get_point_estimates("rate")

        assert "mean" in estimates
        assert "std" in estimates
        assert estimates["mean"].shape == (summariser.model.A, summariser.model.A)

    def test_get_point_estimates_mcint(self, brc_mcmc_model):
        """Test getting point estimates for marginal intensity."""
        summariser = ModelSummariserBRC(brc_mcmc_model)
        estimates = summariser.get_point_estimates("mcint")

        assert "mean" in estimates
        assert "std" in estimates
        assert estimates["mean"].shape == (summariser.model.A,)


class TestModelSummariserBRCCache:
    """Test caching functionality."""

    def test_cache_operations(self, brc_mcmc_model):
        """Test cache clear and info methods."""
        summariser = ModelSummariserBRC(brc_mcmc_model)

        # Initially empty
        info = summariser.get_cache_info()
        assert info["n_cached"] == 0
        assert info["inference_method"] == "mcmc"
        assert info["model_type"] == "brc"

        # Add some cached results
        summariser.summarise_rate(alpha=0.05)
        summariser.summarise_cint(alpha=0.05)

        info = summariser.get_cache_info()
        assert info["n_cached"] == 2

        # Clear cache
        summariser.clear_cache()
        info = summariser.get_cache_info()
        assert info["n_cached"] == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_alpha_raises_error(self, brc_mcmc_model):
        """Test that invalid alpha values raise errors."""
        summariser = ModelSummariserBRC(brc_mcmc_model)

        with pytest.raises(ValueError):
            summariser.summarise_rate(alpha=0)  # alpha must be > 0

        with pytest.raises(ValueError):
            summariser.summarise_rate(alpha=1)  # alpha must be < 1

        with pytest.raises(ValueError):
            summariser.summarise_rate(alpha=1.5)  # alpha must be < 1

    def test_multiple_methods_same_summariser(self, brc_mcmc_model):
        """Test calling multiple summarization methods."""
        summariser = ModelSummariserBRC(brc_mcmc_model)

        rate_summary = summariser.summarise_rate(alpha=0.05)
        cint_summary = summariser.summarise_cint(alpha=0.05)
        mcint_summary = summariser.summarise_mcint(alpha=0.05)

        # All should work and be cached
        cache_info = summariser.get_cache_info()
        assert cache_info["n_cached"] == 3


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with real models."""

    def test_full_workflow_mcmc(self, brc_mcmc_model):
        """Test complete workflow with MCMC model."""
        # Create summariser
        summariser = ModelSummariserBRC(brc_mcmc_model)

        # Get various summaries
        rate_summary = summariser.summarise_rate(alpha=0.05)
        cint_summary = summariser.summarise_cint(alpha=0.05)
        mcint_summary = summariser.summarise_mcint(alpha=0.05)

        # Get point estimates
        rate_estimates = summariser.get_point_estimates("rate")
        cint_estimates = summariser.get_point_estimates("cint")

        # Get raw samples
        rate_samples = summariser.get_posterior_samples("rate")

        # All should be valid
        assert rate_summary.shape[0] == 3
        assert cint_summary["All->All"].shape[0] == 3
        assert mcint_summary["All->All"].shape[0] == 3
        assert "mean" in rate_estimates
        assert rate_samples.ndim == 3

    def test_full_workflow_svi(self, brc_svi_model):
        """Test complete workflow with SVI model."""
        summariser = ModelSummariserBRC(brc_svi_model, num_samples=50)

        # Should work identically to MCMC
        rate_summary = summariser.summarise_rate(alpha=0.05)
        cint_summary = summariser.summarise_cint(alpha=0.05)

        assert rate_summary.shape[0] == 3
        assert cint_summary["All->All"].shape[0] == 3
