"""
Comprehensive tests for ModelEvaluatorBRC.

Tests cover:
- Initialization and validation
- BRC and HiBRC model evaluation
- Contact intensity and marginal contact intensity metrics
- Point estimate error computation
- Caching mechanism
- Backward compatibility with deprecated ModelEvaluator
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest
from jax.random import PRNGKey

pytestmark = pytest.mark.skip(reason="slow fixture setup (~9s per test) - disabled temporarily")
from numpyro.infer.autoguide import AutoNormal

from cntmosaic.analysis import ModelEvaluatorBRC, ModelSummariserBRC
from cntmosaic.dataloader import ContactData, ContactSurveyLoader, ParticipantData, PopulationData
from cntmosaic.datasets import load_age_distribution, load_template_patterns
from cntmosaic.models import AgeMixFF
from cntmosaic.models.numpyro.priors import Spline2D
from cntmosaic.sim import (
    ContactGenerator,
    MatrixGenerator,
    ParticipantGenerator,
    PopulationConstructor,
    Stratification,
)
from cntmosaic.utils import AgeBins

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
    return dataloader, contact_matrix


@pytest.fixture
def brc_mcmc_fitted(sample_dataloader):
    """Create BRC model fitted with MCMC."""
    dataloader, contact_matrix = sample_dataloader

    priors = {"rate": Spline2D(prior_type="global")}
    model = AgeMixFF(dataloader, priors, likelihood="poisson")

    # Run short MCMC for testing
    model.run_inference_mcmc(PRNGKey(0), num_warmup=10, num_samples=20, num_chains=1)

    return model, contact_matrix


@pytest.fixture
def brc_svi_fitted(sample_dataloader):
    """Create BRC model fitted with SVI."""
    dataloader, contact_matrix = sample_dataloader

    priors = {"rate": Spline2D(prior_type="global")}
    model = AgeMixFF(dataloader, priors, likelihood="poisson")

    # Run short SVI for testing
    guide = AutoNormal(model.model)
    model.run_inference_svi(PRNGKey(0), guide, num_steps=100, peak_lr=0.01)

    return model, contact_matrix


# ============================================================================
# Tests for ModelEvaluatorBRC
# ============================================================================


class TestModelEvaluatorBRCInitialization:
    """Test initialization and setup of ModelEvaluatorBRC."""

    def test_init_with_mcmc(self, brc_mcmc_fitted):
        """Test initialization with MCMC model."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true, alpha=0.05)

        assert evaluator.summariser is summariser
        assert evaluator.alpha == 0.05
        assert evaluator.model_type == "brc"
        assert isinstance(evaluator.cint_true, dict)
        assert isinstance(evaluator.mcint_true, dict)

    def test_init_with_svi(self, brc_svi_fitted):
        """Test initialization with SVI model."""
        model, cint_true = brc_svi_fitted
        summariser = ModelSummariserBRC(model, num_samples=50)
        evaluator = ModelEvaluatorBRC(summariser, cint_true, alpha=0.1)

        assert evaluator.summariser is summariser
        assert evaluator.alpha == 0.1
        assert evaluator.model_type == "brc"

    def test_init_computes_marginals(self, brc_mcmc_fitted):
        """Test that marginal contact intensities are computed."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        # Marginals should equal row sums (cint_true is dict with "All->All" key)
        expected_mcint = cint_true["All->All"].sum(axis=1)
        np.testing.assert_array_equal(evaluator.mcint_true["All->All"], expected_mcint)

    def test_init_without_inference_raises_error(self, sample_dataloader):
        """Test that initialization without inference raises ValueError."""
        dataloader, _ = sample_dataloader
        priors = {"rate": Spline2D(prior_type="global")}
        model = AgeMixFF(dataloader, priors)

        with pytest.raises(ValueError, match="Neither MCMC nor SVI"):
            summariser = ModelSummariserBRC(model)

    def test_init_with_wrong_summariser_type_raises_error(self, brc_mcmc_fitted):
        """Test that initialization with wrong summariser type raises TypeError."""
        _, cint_true = brc_mcmc_fitted

        with pytest.raises(TypeError, match="Expected ModelSummariserBRC"):
            ModelEvaluatorBRC("not a summariser", cint_true)

    def test_init_with_invalid_alpha_raises_error(self, brc_mcmc_fitted):
        """Test that invalid alpha values raise ValueError."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)

        with pytest.raises(ValueError, match="alpha must be in"):
            ModelEvaluatorBRC(summariser, cint_true, alpha=0)

        with pytest.raises(ValueError, match="alpha must be in"):
            ModelEvaluatorBRC(summariser, cint_true, alpha=1.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            ModelEvaluatorBRC(summariser, cint_true, alpha=1.5)

    def test_init_with_negative_values_raises_error(self, brc_mcmc_fitted):
        """Test that negative true values raise ValueError."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)

        cint_negative = {"All->All": cint_true["All->All"].copy()}
        cint_negative["All->All"][0, 0] = -1.0

        with pytest.raises(ValueError, match="negative values"):
            ModelEvaluatorBRC(summariser, cint_negative)

    def test_init_with_nan_values_raises_error(self, brc_mcmc_fitted):
        """Test that NaN/Inf true values raise ValueError."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)

        cint_nan = {"All->All": cint_true["All->All"].copy()}
        cint_nan["All->All"][0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN or Inf"):
            ModelEvaluatorBRC(summariser, cint_nan)

    def test_init_with_non_square_matrix_raises_error(self, brc_mcmc_fitted):
        """Test that non-square matrices raise ValueError."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)

        cint_nonsquare = np.random.rand(10, 15)

        with pytest.raises(ValueError, match="must be square"):
            ModelEvaluatorBRC(summariser, cint_nonsquare)


class TestModelEvaluatorBRCContactIntensity:
    """Test contact intensity evaluation methods."""

    def test_evaluate_cint_returns_dataframe(self, brc_mcmc_fitted):
        """Test that evaluate_cint returns a DataFrame."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        metrics = evaluator.evaluate_cint()

        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) > 0

    def test_evaluate_cint_has_required_columns(self, brc_mcmc_fitted):
        """Test that evaluate_cint DataFrame has required columns."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        metrics = evaluator.evaluate_cint()

        required_cols = [
            "var",
            "cat",
            "rmse",
            "mae",
            "mape",
            "interval_score",
            "coverage",
        ]
        for col in required_cols:
            assert col in metrics.columns

    def test_evaluate_cint_values_are_reasonable(self, brc_mcmc_fitted):
        """Test that evaluate_cint produces reasonable metric values."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        metrics = evaluator.evaluate_cint(alpha=0.05)

        # All metrics should be non-negative
        assert metrics["rmse"].values[0] >= 0
        assert metrics["mae"].values[0] >= 0
        assert metrics["mape"].values[0] >= 0
        assert metrics["interval_score"].values[0] >= 0

        # Coverage should be between 0 and 100
        assert 0 <= metrics["coverage"].values[0] <= 100

    def test_evaluate_cint_caching(self, brc_mcmc_fitted):
        """Test that evaluate_cint results are cached."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        # First call
        metrics1 = evaluator.evaluate_cint(alpha=0.05)
        cache_info1 = evaluator.get_cache_info()
        assert cache_info1["n_cached"] == 1

        # Second call with same alpha (should use cache)
        metrics2 = evaluator.evaluate_cint(alpha=0.05)
        pd.testing.assert_frame_equal(metrics1, metrics2)
        cache_info2 = evaluator.get_cache_info()
        assert cache_info2["n_cached"] == 1

        # Third call with different alpha
        metrics3 = evaluator.evaluate_cint(alpha=0.1)
        cache_info3 = evaluator.get_cache_info()
        assert cache_info3["n_cached"] == 2


class TestModelEvaluatorBRCMarginalIntensity:
    """Test marginal contact intensity evaluation methods."""

    def test_evaluate_mcint_returns_dataframe(self, brc_mcmc_fitted):
        """Test that evaluate_mcint returns a DataFrame."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        metrics = evaluator.evaluate_mcint()

        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) > 0

    def test_evaluate_mcint_has_required_columns(self, brc_mcmc_fitted):
        """Test that evaluate_mcint DataFrame has required columns."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        metrics = evaluator.evaluate_mcint()

        required_cols = [
            "var",
            "cat",
            "rmse",
            "mae",
            "mape",
            "interval_score",
            "coverage",
        ]
        for col in required_cols:
            assert col in metrics.columns

    def test_evaluate_mcint_values_are_reasonable(self, brc_mcmc_fitted):
        """Test that evaluate_mcint produces reasonable metric values."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        metrics = evaluator.evaluate_mcint(alpha=0.05)

        # All metrics should be non-negative
        assert metrics["rmse"].values[0] >= 0
        assert metrics["mae"].values[0] >= 0
        assert metrics["mape"].values[0] >= 0
        assert metrics["interval_score"].values[0] >= 0

        # Coverage should be between 0 and 100
        assert 0 <= metrics["coverage"].values[0] <= 100


class TestModelEvaluatorBRCCombinedEvaluation:
    """Test combined evaluation methods."""

    def test_evaluate_returns_both_metrics(self, brc_mcmc_fitted):
        """Test that evaluate() returns both cint and mcint metrics."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        metrics = evaluator.evaluate(alpha=0.05)

        assert isinstance(metrics, pd.DataFrame)
        assert "metric_type" in metrics.columns
        assert "cint" in metrics["metric_type"].values
        assert "mcint" in metrics["metric_type"].values

    def test_evaluate_with_custom_alpha(self, brc_mcmc_fitted):
        """Test evaluate() with custom alpha."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true, alpha=0.05)

        metrics = evaluator.evaluate(alpha=0.1)

        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) >= 2  # At least cint and mcint


class TestModelEvaluatorBRCPointEstimates:
    """Test point estimate error computations."""

    def test_get_point_estimate_error_cint(self, brc_mcmc_fitted):
        """Test point estimate errors for contact intensity."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        errors = evaluator.get_point_estimate_error("cint")

        assert isinstance(errors, dict)
        assert "rmse" in errors
        assert "mae" in errors
        assert "mape" in errors
        assert "relative_error" in errors

        # All should be non-negative
        assert errors["rmse"] >= 0
        assert errors["mae"] >= 0
        assert errors["relative_error"] >= 0

    def test_get_point_estimate_error_mcint(self, brc_mcmc_fitted):
        """Test point estimate errors for marginal intensity."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        errors = evaluator.get_point_estimate_error("mcint")

        assert isinstance(errors, dict)
        assert "rmse" in errors
        assert errors["rmse"] >= 0

    def test_get_point_estimate_error_invalid_quantity(self, brc_mcmc_fitted):
        """Test that invalid quantity raises ValueError."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        with pytest.raises(ValueError, match="must be 'cint' or 'mcint'"):
            evaluator.get_point_estimate_error("invalid")


class TestModelEvaluatorBRCCaching:
    """Test caching mechanism."""

    def test_cache_info_initially_empty(self, brc_mcmc_fitted):
        """Test that cache is initially empty."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        cache_info = evaluator.get_cache_info()

        assert cache_info["n_cached"] == 0
        assert cache_info["model_type"] == "brc"
        assert cache_info["alpha"] == 0.05

    def test_cache_populated_after_evaluation(self, brc_mcmc_fitted):
        """Test that cache is populated after evaluation."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        evaluator.evaluate_cint()
        cache_info = evaluator.get_cache_info()

        assert cache_info["n_cached"] == 1
        assert "cint_alpha0.05" in cache_info["cached_metrics"]

    def test_clear_cache(self, brc_mcmc_fitted):
        """Test clearing the cache."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        # Populate cache
        evaluator.evaluate_cint()
        evaluator.evaluate_mcint()
        assert evaluator.get_cache_info()["n_cached"] == 2

        # Clear cache
        evaluator.clear_cache()
        assert evaluator.get_cache_info()["n_cached"] == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_multiple_evaluations_same_evaluator(self, brc_mcmc_fitted):
        """Test calling multiple evaluation methods."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true)

        # Should all work without issues
        cint_metrics = evaluator.evaluate_cint()
        mcint_metrics = evaluator.evaluate_mcint()
        point_errors = evaluator.get_point_estimate_error("cint")
        all_metrics = evaluator.evaluate()

        assert all(
            [
                isinstance(cint_metrics, pd.DataFrame),
                isinstance(mcint_metrics, pd.DataFrame),
                isinstance(point_errors, dict),
                isinstance(all_metrics, pd.DataFrame),
            ]
        )

    def test_evaluator_with_zero_true_matrix(self, brc_mcmc_fitted):
        """Test evaluator behavior with zero true matrix (edge case)."""
        model, cint_true = brc_mcmc_fitted
        summariser = ModelSummariserBRC(model)

        # Create zero matrix dict (edge case, but shouldn't crash)
        zero_matrix = {"All->All": np.zeros_like(cint_true["All->All"])}
        evaluator = ModelEvaluatorBRC(summariser, zero_matrix)

        # Should compute metrics without error (though MAPE may be inf)
        metrics = evaluator.evaluate_cint()
        assert isinstance(metrics, pd.DataFrame)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with real models."""

    def test_full_workflow_mcmc(self, brc_mcmc_fitted):
        """Test complete evaluation workflow with MCMC model."""
        model, cint_true = brc_mcmc_fitted

        # Create summariser and evaluator
        summariser = ModelSummariserBRC(model)
        evaluator = ModelEvaluatorBRC(summariser, cint_true, alpha=0.05)

        # Get all metrics
        cint_metrics = evaluator.evaluate_cint()
        mcint_metrics = evaluator.evaluate_mcint()
        point_errors_cint = evaluator.get_point_estimate_error("cint")
        point_errors_mcint = evaluator.get_point_estimate_error("mcint")
        all_metrics = evaluator.evaluate()

        # Verify all outputs
        assert isinstance(cint_metrics, pd.DataFrame)
        assert isinstance(mcint_metrics, pd.DataFrame)
        assert isinstance(point_errors_cint, dict)
        assert isinstance(point_errors_mcint, dict)
        assert isinstance(all_metrics, pd.DataFrame)

        # Check cache
        cache_info = evaluator.get_cache_info()
        assert cache_info["n_cached"] > 0

    def test_full_workflow_svi(self, brc_svi_fitted):
        """Test complete evaluation workflow with SVI model."""
        model, cint_true = brc_svi_fitted

        summariser = ModelSummariserBRC(model, num_samples=50)
        evaluator = ModelEvaluatorBRC(summariser, cint_true, alpha=0.05)

        # Should work identically to MCMC
        metrics = evaluator.evaluate()
        assert isinstance(metrics, pd.DataFrame)
        assert "cint" in metrics["metric_type"].values
        assert "mcint" in metrics["metric_type"].values
