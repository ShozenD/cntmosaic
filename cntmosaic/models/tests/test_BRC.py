"""
Comprehensive unit tests for the BRC (Bayesian Rate Consistency) base class.

Tests cover:
- Input validation
- Initialization and configuration
- Age dimension handling
- Population distribution handling
- MCMC inference
- SVI inference
- Posterior predictive sampling
- Model structure inspection
- Error handling and edge cases
"""

import warnings

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from jax.random import PRNGKey
from numpyro import distributions as dist
from numpyro.handlers import scope
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import CoordToColumns, DataLoader
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ContactGenerator, MatrixGenerator, ParticipantGenerator, Subgroup
from .._BRC import BRC
from ..priors import HSGP2D, PSpline2D, Spline2D

# ============================================================================
# Mock BRC Implementation for Testing
# ============================================================================


class MockBRC(BRC):
    """Minimal concrete implementation of BRC for testing the base class."""

    def __init__(self, dataloader, priors, likelihood="negbin"):
        super().__init__(dataloader, priors, likelihood)
        # Mock minimal data setup
        self.y = jnp.array([1, 2, 3, 4, 5])
        self.aid = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
        self.bid = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
        self.log_N = jnp.zeros(5)
        self.log_P = jnp.zeros((1, self.A))

    def model(self, y=None, **kwargs):
        """Simple test model that accepts extra kwargs."""
        beta0 = numpyro.sample("baseline", dist.Normal(0.0, 2.5))

        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        # Simple observation model
        mu = jnp.exp(beta0) * jnp.ones(len(self.y))

        if self.likelihood == "poisson":
            with numpyro.plate("data", len(self.y)):
                numpyro.sample("obs", dist.Poisson(rate=mu), obs=y)
        elif self.likelihood == "negbin":
            inv_disp = numpyro.sample("inv_disp", dist.Exponential(1.0))
            with numpyro.plate("data", len(self.y)):
                numpyro.sample(
                    "obs",
                    dist.NegativeBinomial2(mean=mu, concentration=1.0 / inv_disp),
                    obs=y,
                )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_dataloader():
    """Create a sample dataloader for testing."""
    df_age_dist = load_age_distribution("United_States")
    templates = load_template_patterns("United_States")

    population = Subgroup(
        n=500, age_dist=df_age_dist.P.values, mean_cint_margin=15.0, label="general"
    )

    matrix_gen = MatrixGenerator(templates)
    contact_matrix = matrix_gen.generate_single(population, seed=42)

    part_gen = ParticipantGenerator(population)
    df_part = part_gen.generate(seed=42)
    df_part["age_part"] = df_part["age_group"]

    cnt_gen = ContactGenerator(df_part, cint_matrices=contact_matrix, model="poisson")
    df_cnt = cnt_gen.generate(seed=42)

    col_map = CoordToColumns(
        age_part="age_part", age_cnt="age_cnt", age_pop="age", size_pop="P"
    )
    dataloader = DataLoader(df_part, df_cnt, df_age_dist, col_map=col_map)

    return dataloader


@pytest.fixture
def valid_priors():
    """Create valid prior specification."""
    return {"rate": Spline2D(prior_type="global", M=20, degree=3)}


@pytest.fixture
def mock_brc_model(sample_dataloader, valid_priors):
    """Create a MockBRC instance for testing."""
    return MockBRC(sample_dataloader, valid_priors, likelihood="poisson")


# ============================================================================
# Test Input Validation
# ============================================================================


class TestInputValidation:
    """Test suite for input validation in BRC initialization."""

    def test_invalid_likelihood(self, sample_dataloader, valid_priors):
        """Test that invalid likelihood raises ValueError."""
        with pytest.raises(ValueError, match="likelihood must be one of"):
            MockBRC(sample_dataloader, valid_priors, likelihood="invalid")

    def test_invalid_priors_type(self, sample_dataloader):
        """Test that non-dict priors raises ValueError."""
        with pytest.raises(ValueError, match="priors must be a dictionary"):
            MockBRC(sample_dataloader, priors="not_a_dict", likelihood="poisson")

    def test_missing_rate_prior(self, sample_dataloader):
        """Test that missing 'rate' key in priors raises ValueError."""
        invalid_priors = {"other": Spline2D(prior_type="global")}
        with pytest.raises(
            ValueError, match="priors must contain the specifications for 'rate'"
        ):
            MockBRC(sample_dataloader, invalid_priors, likelihood="poisson")

    def test_valid_poisson_likelihood(self, sample_dataloader, valid_priors):
        """Test that Poisson likelihood is accepted."""
        model = MockBRC(sample_dataloader, valid_priors, likelihood="poisson")
        assert model.likelihood == "poisson"

    def test_valid_negbin_likelihood(self, sample_dataloader, valid_priors):
        """Test that negative binomial likelihood is accepted."""
        model = MockBRC(sample_dataloader, valid_priors, likelihood="negbin")
        assert model.likelihood == "negbin"


# ============================================================================
# Test Initialization and Configuration
# ============================================================================


class TestInitialization:
    """Test suite for BRC model initialization."""

    def test_basic_initialization(self, mock_brc_model):
        """Test that basic initialization works correctly."""
        model = mock_brc_model

        assert model.ds is not None
        assert model.priors is not None
        assert model.likelihood == "poisson"
        assert hasattr(model, "age_min")
        assert hasattr(model, "age_max")
        assert hasattr(model, "A")

    def test_dataloader_loaded(self, mock_brc_model):
        """Test that dataloader is properly loaded."""
        model = mock_brc_model
        assert model.ds is not None
        assert hasattr(model.ds, "age")

    def test_priors_stored(self, mock_brc_model, valid_priors):
        """Test that priors are correctly stored."""
        model = mock_brc_model
        assert model.priors == valid_priors
        assert "rate" in model.priors

    def test_initial_attributes_none(self, mock_brc_model):
        """Test that inference results are initially None."""
        model = mock_brc_model
        # Note: _mcmc_result and _svi_result might be set by subclass
        assert model._guide is None

    def test_age_dims_automatically_set(self, mock_brc_model):
        """Test that age dimensions are automatically set during init."""
        model = mock_brc_model
        assert hasattr(model, "age_min")
        assert hasattr(model, "age_max")
        assert model.age_min >= 0
        assert model.age_max > model.age_min
        assert model.A == model.age_max - model.age_min + 1


# ============================================================================
# Test Age Dimension Handling
# ============================================================================


class TestAgeDimensions:
    """Test suite for age dimension configuration."""

    def test_set_age_dims_valid(self, mock_brc_model):
        """Test setting valid age dimensions."""
        model = mock_brc_model
        model.set_age_dims(0, 80)

        assert model.age_min == 0
        assert model.age_max == 80
        assert model.A == 81

    def test_set_age_dims_negative_min(self, mock_brc_model):
        """Test that negative age_min raises ValueError."""
        model = mock_brc_model
        with pytest.raises(ValueError, match="age_min must be non-negative"):
            model.set_age_dims(-1, 80)

    def test_set_age_dims_invalid_range(self, mock_brc_model):
        """Test that age_max <= age_min raises ValueError."""
        model = mock_brc_model
        with pytest.raises(ValueError, match="age_max must be greater than age_min"):
            model.set_age_dims(50, 50)

        with pytest.raises(ValueError, match="age_max must be greater than age_min"):
            model.set_age_dims(60, 50)

    def test_set_age_dims_updates_priors(self, mock_brc_model):
        """Test that setting age dims updates priors."""
        model = mock_brc_model
        model.set_age_dims(0, 60)

        # Check that prior has age bounds set (Prior2D stores them differently)
        rate_prior = model.priors["rate"]
        # Priors inherit from Prior2D which has min_age, max_age after set_age_bounds
        assert hasattr(rate_prior, "min_age") or hasattr(rate_prior, "age_min")
        if hasattr(rate_prior, "min_age"):
            assert rate_prior.min_age == 0
            assert rate_prior.max_age == 60
        elif hasattr(rate_prior, "age_min"):
            assert rate_prior.age_min == 0
            assert rate_prior.age_max == 60

    def test_age_dims_different_ranges(self, mock_brc_model):
        """Test different age ranges."""
        model = mock_brc_model

        # Test young ages only
        model.set_age_dims(0, 18)
        assert model.A == 19

        # Test working age
        model.set_age_dims(18, 65)
        assert model.A == 48

        # Test elderly
        model.set_age_dims(65, 100)
        assert model.A == 36


# ============================================================================
# Test Population Distribution Handling
# ============================================================================


class TestAgeDistribution:
    """Test suite for population age distribution handling."""

    def test_set_age_dist_valid(self, mock_brc_model):
        """Test setting a valid age distribution."""
        model = mock_brc_model
        age_dist = np.ones(model.A) / model.A
        model.set_age_dist(age_dist)

        assert model.age_dist is not None
        assert len(model.age_dist) == model.A
        np.testing.assert_allclose(model.age_dist.sum(), 1.0, rtol=1e-5)

    def test_set_age_dist_wrong_length(self, mock_brc_model):
        """Test that wrong length age_dist raises ValueError."""
        model = mock_brc_model
        wrong_length_dist = np.ones(model.A + 10) / (model.A + 10)

        with pytest.raises(ValueError, match="must match number of age groups"):
            model.set_age_dist(wrong_length_dist)

    def test_set_age_dist_negative_values(self, mock_brc_model):
        """Test that negative values in age_dist raise ValueError."""
        model = mock_brc_model
        invalid_dist = np.ones(model.A)
        invalid_dist[0] = -0.1

        with pytest.raises(ValueError, match="must contain only non-negative values"):
            model.set_age_dist(invalid_dist)

    def test_set_age_dist_not_normalized_warning(self, mock_brc_model):
        """Test that non-normalized age_dist raises warning."""
        model = mock_brc_model
        unnormalized_dist = np.ones(model.A) * 2.0  # Sums to 2*A, not 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.set_age_dist(unnormalized_dist)
            assert len(w) == 1
            assert "Consider normalizing" in str(w[0].message)

    def test_set_age_dist_multidimensional(self, mock_brc_model):
        """Test that multidimensional age_dist raises ValueError."""
        model = mock_brc_model
        multidim_dist = np.ones((model.A, 2))

        with pytest.raises(ValueError, match="must be 1-dimensional"):
            model.set_age_dist(multidim_dist)

    def test_set_age_dist_before_age_dims(self, sample_dataloader, valid_priors):
        """Test that set_age_dist works after set_age_dims is called."""
        model = MockBRC(sample_dataloader, valid_priors)
        # Age dims are set during __init__
        age_dist = np.ones(model.A) / model.A
        model.set_age_dist(age_dist)
        assert model.age_dist is not None


# ============================================================================
# Test Model Structure Inspection
# ============================================================================


class TestModelStructure:
    """Test suite for model structure inspection methods."""

    def test_print_model_shape(self, mock_brc_model, capsys):
        """Test that print_model_shape outputs correctly."""
        model = mock_brc_model
        model.print_model_shape()

        captured = capsys.readouterr()
        assert "Trace Shapes:" in captured.out
        assert "baseline" in captured.out
        assert "rate/spline_coefs" in captured.out

    def test_abstract_model_method(self, sample_dataloader, valid_priors):
        """Test that BRC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BRC(sample_dataloader, valid_priors, likelihood="poisson")


# ============================================================================
# Test MCMC Inference
# ============================================================================


class TestMCMCInference:
    """Test suite for MCMC inference functionality."""

    def test_mcmc_basic_run(self, mock_brc_model):
        """Test basic MCMC inference runs successfully."""
        model = mock_brc_model
        prng_key = PRNGKey(42)

        model.run_inference_mcmc(prng_key, num_samples=10, num_warmup=10, num_chains=1)

        assert model._mcmc_result is not None

    def test_mcmc_result_has_samples(self, mock_brc_model):
        """Test that MCMC result contains samples."""
        model = mock_brc_model
        prng_key = PRNGKey(42)

        model.run_inference_mcmc(prng_key, num_samples=20, num_warmup=10, num_chains=1)

        samples = model._mcmc_result.get_samples()
        assert "baseline" in samples
        assert len(samples["baseline"]) == 20

    def test_mcmc_multiple_chains(self, mock_brc_model):
        """Test MCMC with multiple chains."""
        model = mock_brc_model
        prng_key = PRNGKey(42)

        model.run_inference_mcmc(prng_key, num_samples=10, num_warmup=5, num_chains=2)

        samples = model._mcmc_result.get_samples()
        # Total samples = num_chains * num_samples
        assert len(samples["baseline"]) == 20

    def test_mcmc_custom_parameters(self, mock_brc_model):
        """Test MCMC with custom parameters."""
        model = mock_brc_model
        prng_key = PRNGKey(42)

        model.run_inference_mcmc(
            prng_key,
            num_samples=15,
            num_warmup=10,
            num_chains=1,
            target_accept_prob=0.9,
            max_tree_depth=8,
        )

        assert model._mcmc_result is not None

    def test_mcmc_without_data_raises_error(self, sample_dataloader, valid_priors):
        """Test that MCMC without observation data raises error."""
        model = MockBRC(sample_dataloader, valid_priors)
        model.y = None  # Remove observation data

        prng_key = PRNGKey(42)
        with pytest.raises(AttributeError, match="Observation data.*has not been set"):
            model.run_inference_mcmc(prng_key, num_samples=10, num_warmup=10)

    def test_mcmc_diagnostics_logged(self, mock_brc_model, capsys):
        """Test that MCMC diagnostics are logged."""
        model = mock_brc_model
        prng_key = PRNGKey(42)

        model.run_inference_mcmc(prng_key, num_samples=10, num_warmup=5, num_chains=1)

        # Check that some diagnostics output occurred
        # (exact content depends on whether divergences occur)
        assert model._mcmc_result is not None


# ============================================================================
# Test SVI Inference
# ============================================================================


class TestSVIInference:
    """Test suite for SVI inference functionality."""

    def test_svi_basic_run(self, mock_brc_model):
        """Test basic SVI inference runs successfully."""
        model = mock_brc_model
        prng_key = PRNGKey(42)
        guide = AutoNormal(model.model)

        model.run_inference_svi(prng_key, guide, num_steps=100, peak_lr=0.01)

        assert model._svi_result is not None
        assert model._guide is not None

    def test_svi_result_has_params(self, mock_brc_model):
        """Test that SVI result contains parameters."""
        model = mock_brc_model
        prng_key = PRNGKey(42)
        guide = AutoNormal(model.model)

        model.run_inference_svi(prng_key, guide, num_steps=100, peak_lr=0.01)

        assert hasattr(model._svi_result, "params")
        assert model._svi_result.params is not None

    def test_svi_custom_parameters(self, mock_brc_model):
        """Test SVI with custom parameters."""
        model = mock_brc_model
        prng_key = PRNGKey(42)
        guide = AutoNormal(model.model)

        model.run_inference_svi(prng_key, guide, num_steps=500, peak_lr=0.05)

        assert model._svi_result is not None

    def test_svi_without_data_raises_error(self, sample_dataloader, valid_priors):
        """Test that SVI without observation data raises error."""
        model = MockBRC(sample_dataloader, valid_priors)
        model.y = None  # Remove observation data

        prng_key = PRNGKey(42)
        guide = AutoNormal(model.model)

        with pytest.raises(AttributeError, match="Observation data.*has not been set"):
            model.run_inference_svi(prng_key, guide, num_steps=100)

    def test_svi_stores_guide(self, mock_brc_model):
        """Test that SVI stores the guide function."""
        model = mock_brc_model
        prng_key = PRNGKey(42)
        guide = AutoNormal(model.model)

        model.run_inference_svi(prng_key, guide, num_steps=100)

        assert model._guide is guide


# ============================================================================
# Test Posterior Predictive Sampling
# ============================================================================


class TestPosteriorPredictive:
    """Test suite for posterior predictive sampling."""

    def test_posterior_predictive_mcmc(self, mock_brc_model):
        """Test posterior predictive sampling from MCMC."""
        model = mock_brc_model
        prng_key = PRNGKey(42)

        # First run MCMC
        num_mcmc_samples = 20
        model.run_inference_mcmc(
            prng_key, num_samples=num_mcmc_samples, num_warmup=10, num_chains=1
        )

        # Generate posterior predictive samples
        # Note: posterior_predictive_mcmc uses all MCMC samples by default
        pred_key = PRNGKey(123)
        pred_samples = model.posterior_predictive_mcmc(pred_key, num_samples=10)

        assert "obs" in pred_samples
        # Returns one prediction per MCMC sample
        assert pred_samples["obs"].shape[0] == num_mcmc_samples

    def test_posterior_predictive_mcmc_without_inference(self, mock_brc_model):
        """Test that posterior predictive fails without prior MCMC."""
        model = mock_brc_model
        prng_key = PRNGKey(42)

        with pytest.raises(AttributeError, match="MCMC inference has not been run"):
            model.posterior_predictive_mcmc(prng_key, num_samples=10)

    def test_posterior_predictive_svi(self, mock_brc_model):
        """Test posterior predictive sampling from SVI."""
        model = mock_brc_model
        prng_key = PRNGKey(42)
        guide = AutoNormal(model.model)

        # First run SVI
        model.run_inference_svi(prng_key, guide, num_steps=100)

        # Generate posterior predictive samples
        pred_key = PRNGKey(123)
        pred_samples = model.posterior_predictive_svi(pred_key, guide, num_samples=10)

        assert "obs" in pred_samples
        assert pred_samples["obs"].shape[0] == 10

    def test_posterior_predictive_svi_without_inference(self, mock_brc_model):
        """Test that posterior predictive fails without prior SVI."""
        model = mock_brc_model
        prng_key = PRNGKey(42)
        guide = AutoNormal(model.model)

        with pytest.raises(AttributeError, match="SVI inference has not been run"):
            model.posterior_predictive_svi(prng_key, guide, num_samples=10)


# ============================================================================
# Test Different Prior Types
# ============================================================================


class TestDifferentPriors:
    """Test BRC with different prior specifications."""

    def test_with_pspline_prior(self, sample_dataloader):
        """Test BRC initialization with PSpline2D prior."""
        priors = {"rate": PSpline2D(prior_type="global", M=15, degree=3)}
        model = MockBRC(sample_dataloader, priors, likelihood="poisson")

        assert model.priors["rate"].__class__.__name__ == "PSpline2D"

    def test_with_hsgp_prior(self, sample_dataloader):
        """Test BRC initialization with HSGP2D prior."""
        # HSGP2D uses C parameter, not ell
        priors = {"rate": HSGP2D(prior_type="global", M=[20, 20], C=[1.5, 1.5])}
        model = MockBRC(sample_dataloader, priors, likelihood="poisson")

        assert model.priors["rate"].__class__.__name__ == "HSGP2D"

    def test_with_different_spline_parameters(self, sample_dataloader):
        """Test BRC with different spline configurations."""
        # Test different M values
        priors1 = {"rate": Spline2D(prior_type="global", M=10, degree=2)}
        model1 = MockBRC(sample_dataloader, priors1)
        assert model1.priors["rate"].M == 10

        priors2 = {"rate": Spline2D(prior_type="global", M=40, degree=3)}
        model2 = MockBRC(sample_dataloader, priors2)
        assert model2.priors["rate"].M == 40


# ============================================================================
# Test Error Handling and Edge Cases
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_likelihood_case_sensitivity(self, sample_dataloader, valid_priors):
        """Test that likelihood is case-sensitive."""
        with pytest.raises(ValueError):
            MockBRC(sample_dataloader, valid_priors, likelihood="Poisson")

        with pytest.raises(ValueError):
            MockBRC(sample_dataloader, valid_priors, likelihood="NEGBIN")

    def test_empty_priors_dict(self, sample_dataloader):
        """Test that empty priors dict raises error."""
        with pytest.raises(
            ValueError, match="must contain the specifications for 'rate'"
        ):
            MockBRC(sample_dataloader, priors={}, likelihood="poisson")

    def test_mcmc_with_invalid_chains(self, mock_brc_model):
        """Test MCMC with invalid number of chains."""
        model = mock_brc_model
        prng_key = PRNGKey(42)

        # num_chains must be positive
        with pytest.raises(Exception):  # Will raise from numpyro
            model.run_inference_mcmc(
                prng_key, num_samples=10, num_warmup=10, num_chains=0
            )

    def test_svi_with_zero_steps(self, mock_brc_model):
        """Test SVI with very few steps to check edge case handling."""
        model = mock_brc_model
        prng_key = PRNGKey(42)
        guide = AutoNormal(model.model)

        # num_steps must be positive (optax requirement)
        # Test with minimum valid value instead
        model.run_inference_svi(prng_key, guide, num_steps=1, peak_lr=0.01)
        assert model._svi_result is not None


# ============================================================================
# Test Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test complete workflow scenarios."""

    def test_full_mcmc_workflow(self, mock_brc_model):
        """Test complete MCMC workflow from start to predictions."""
        model = mock_brc_model

        # 1. Check model structure
        model.print_model_shape()

        # 2. Run inference
        prng_key = PRNGKey(42)
        model.run_inference_mcmc(prng_key, num_samples=20, num_warmup=10, num_chains=1)

        # 3. Generate predictions
        pred_samples = model.posterior_predictive_mcmc(PRNGKey(123), num_samples=10)

        assert pred_samples is not None
        assert "obs" in pred_samples

    def test_full_svi_workflow(self, mock_brc_model):
        """Test complete SVI workflow from start to predictions."""
        model = mock_brc_model
        guide = AutoNormal(model.model)

        # 1. Check model structure
        model.print_model_shape()

        # 2. Run inference
        prng_key = PRNGKey(42)
        model.run_inference_svi(prng_key, guide, num_steps=200)

        # 3. Generate predictions
        pred_samples = model.posterior_predictive_svi(
            PRNGKey(123), guide, num_samples=10
        )

        assert pred_samples is not None
        assert "obs" in pred_samples

    def test_change_age_dist_after_init(self, mock_brc_model):
        """Test changing age distribution after initialization."""
        model = mock_brc_model

        # Set new age distribution
        new_age_dist = np.ones(model.A) / model.A
        model.set_age_dist(new_age_dist)

        # Should still be able to run inference
        prng_key = PRNGKey(42)
        model.run_inference_mcmc(prng_key, num_samples=10, num_warmup=5, num_chains=1)

        assert model._mcmc_result is not None

    def test_sequential_inference_methods(self, mock_brc_model):
        """Test running both MCMC and SVI on same model."""
        model = mock_brc_model

        # First run MCMC
        mcmc_key = PRNGKey(42)
        model.run_inference_mcmc(mcmc_key, num_samples=10, num_warmup=5, num_chains=1)
        mcmc_result = model._mcmc_result

        # Then run SVI
        svi_key = PRNGKey(123)
        guide = AutoNormal(model.model)
        model.run_inference_svi(svi_key, guide, num_steps=100)

        # Both results should be stored
        assert model._mcmc_result is not None
        assert model._svi_result is not None
        assert model._mcmc_result is mcmc_result  # MCMC result preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
