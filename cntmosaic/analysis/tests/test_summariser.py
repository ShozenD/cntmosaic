"""
Tests for ModelSummariser (AgeMix / GenMix family).

Covers:
- MCMC and SVI inference methods
- agemix and genmix model types
- All summary statistics (rate, cint, mcint)
- ContactSummary dataclass output
- Lazy loading behaviour
- Error handling and edge cases
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
from cntmosaic.analysis import ModelSummariser, ContactSummary


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
def agemix_mcmc_model(sample_dataloader):
    """AgeMix model fitted with MCMC."""
    priors = {"rate": Spline2D(prior_type="global")}
    model = AgeMixFF(sample_dataloader, priors, likelihood="poisson")
    model.run_inference_mcmc(PRNGKey(0), num_warmup=10, num_samples=20, num_chains=1)
    return model


@pytest.fixture
def agemix_svi_model(sample_dataloader):
    """AgeMix model fitted with SVI."""
    priors = {"rate": Spline2D(prior_type="global")}
    model = AgeMixFF(sample_dataloader, priors, likelihood="poisson")
    guide = AutoNormal(model.model)
    model.run_inference_svi(PRNGKey(0), guide, num_steps=100, peak_lr=0.01)
    return model


# ============================================================================
# Initialization tests
# ============================================================================


class TestModelSummariserInit:
    def test_init_with_mcmc(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)

        assert summariser.model is agemix_mcmc_model
        assert summariser.inference_method == "mcmc"
        assert summariser.model_type == "agemix"
        assert summariser.num_samples == 3000
        # Samples not yet loaded (lazy)
        assert summariser._post_samples is None

    def test_init_with_svi(self, agemix_svi_model):
        summariser = ModelSummariser(agemix_svi_model, num_samples=50)

        assert summariser.model is agemix_svi_model
        assert summariser.inference_method == "svi"
        assert summariser.model_type == "agemix"
        assert summariser.num_samples == 50
        assert summariser._post_samples is None

    def test_init_custom_rng_key(self, agemix_mcmc_model):
        key = PRNGKey(42)
        summariser = ModelSummariser(agemix_mcmc_model, rng_key=key)
        assert summariser._rng_key is key

    def test_init_without_inference_raises(self, sample_dataloader):
        priors = {"rate": Spline2D(prior_type="global")}
        model = AgeMixFF(sample_dataloader, priors)
        with pytest.raises(ValueError, match="Neither MCMC nor SVI"):
            ModelSummariser(model)

    def test_init_wrong_type_raises(self):
        with pytest.raises(TypeError, match="must be one of"):
            ModelSummariser("not a model")

    def test_lazy_loading_triggered_on_access(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        assert summariser._post_samples is None
        # Access a summarise method to trigger loading
        _ = summariser.summarise_cint()
        assert summariser._post_samples is not None

    def test_repr(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        r = repr(summariser)
        assert "ModelSummariser" in r
        assert "agemix" in r
        assert "mcmc" in r


# ============================================================================
# ContactSummary output format
# ============================================================================


class TestContactSummaryOutput:
    def test_summarise_cint_returns_contact_summary(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        result = summariser.summarise_cint(alpha=0.05)

        assert isinstance(result, dict)
        assert "All->All" in result
        cs = result["All->All"]
        assert isinstance(cs, ContactSummary)

    def test_contact_summary_fields(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cs = summariser.summarise_cint(alpha=0.05)["All->All"]
        A = summariser.model.A

        assert cs.lower.shape == (A, A)
        assert cs.central.shape == (A, A)
        assert cs.upper.shape == (A, A)
        assert cs.alpha == 0.05
        assert cs.measure == "median"

    def test_contact_summary_ordering(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cs = summariser.summarise_cint(alpha=0.05)["All->All"]

        assert np.all(cs.lower <= cs.central)
        assert np.all(cs.central <= cs.upper)

    def test_contact_summary_non_negative(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cs = summariser.summarise_cint(alpha=0.05)["All->All"]

        assert np.all(cs.lower >= 0)
        assert np.all(cs.central >= 0)
        assert np.all(cs.upper >= 0)

    def test_to_array(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cs = summariser.summarise_cint(alpha=0.05)["All->All"]
        arr = cs.to_array()

        assert arr.shape == (3, summariser.model.A, summariser.model.A)
        np.testing.assert_array_equal(arr[0], cs.lower)
        np.testing.assert_array_equal(arr[1], cs.central)
        np.testing.assert_array_equal(arr[2], cs.upper)

    def test_to_dataframe(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cs = summariser.summarise_cint(alpha=0.05)["All->All"]
        df = cs.to_dataframe()

        A = summariser.model.A
        assert len(df) == A * A
        assert set(df.columns) >= {"age_part", "age_cnt", "lower", "central", "upper"}


# ============================================================================
# Rate summarisation
# ============================================================================


class TestSummariseRate:
    def test_returns_dict_with_all_key(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        result = summariser.summarise_rate(alpha=0.05)
        assert isinstance(result, dict)
        assert "All->All" in result
        assert isinstance(result["All->All"], ContactSummary)

    def test_shape(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cs = summariser.summarise_rate(alpha=0.05)["All->All"]
        A = summariser.model.A
        assert cs.lower.shape == (A, A)
        assert cs.central.shape == (A, A)

    def test_ordering(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cs = summariser.summarise_rate(alpha=0.05)["All->All"]
        assert np.all(cs.lower <= cs.central)
        assert np.all(cs.central <= cs.upper)

    def test_caching(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        r1 = summariser.summarise_rate(alpha=0.05)
        n1 = summariser.get_cache_info()["n_cached"]
        r2 = summariser.summarise_rate(alpha=0.05)
        n2 = summariser.get_cache_info()["n_cached"]

        assert n2 == n1  # no new entry
        np.testing.assert_array_equal(r1["All->All"].central, r2["All->All"].central)

    def test_different_alpha_adds_cache_entry(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        summariser.summarise_rate(alpha=0.05)
        summariser.summarise_rate(alpha=0.10)
        assert summariser.get_cache_info()["n_cached"] == 2

    def test_measure_mean(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cs = summariser.summarise_rate(alpha=0.05, measure="mean")["All->All"]
        assert cs.measure == "mean"

    def test_invalid_measure_raises(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        with pytest.raises(ValueError, match="measure must be"):
            summariser.summarise_rate(measure="mode")


# ============================================================================
# Contact intensity (cint)
# ============================================================================


class TestSummariseCint:
    def test_shape_and_keys(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        result = summariser.summarise_cint(alpha=0.05)
        assert list(result.keys()) == ["All->All"]

    def test_svi_model(self, agemix_svi_model):
        summariser = ModelSummariser(agemix_svi_model, num_samples=50)
        result = summariser.summarise_cint(alpha=0.05)
        assert "All->All" in result
        assert isinstance(result["All->All"], ContactSummary)


# ============================================================================
# Marginal contact intensity (mcint)
# ============================================================================


class TestSummariseMcint:
    def test_shape(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        result = summariser.summarise_mcint(alpha=0.05)
        A = summariser.model.A

        assert "All->All" in result
        cs = result["All->All"]
        assert cs.lower.shape == (A,)
        assert cs.central.shape == (A,)
        assert cs.upper.shape == (A,)

    def test_ordering(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cs = summariser.summarise_mcint(alpha=0.05)["All->All"]
        assert np.all(cs.lower <= cs.central)
        assert np.all(cs.central <= cs.upper)

    def test_mcint_equals_sum_of_cint(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        cint_central = summariser.summarise_cint(alpha=0.05)["All->All"].central
        mcint_central = summariser.summarise_mcint(alpha=0.05)["All->All"].central
        np.testing.assert_allclose(mcint_central, cint_central.sum(axis=1), rtol=0.1)


# ============================================================================
# Posterior samples
# ============================================================================


class TestGetPosteriorSamples:
    def test_rate_samples(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        samples = summariser.get_posterior_samples("rate")
        assert isinstance(samples, np.ndarray)
        assert samples.ndim == 3
        assert samples.shape[0] == 20  # num_samples from fixture
        assert np.all(samples >= 0)

    def test_cint_samples(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        samples = summariser.get_posterior_samples("cint")
        assert isinstance(samples, np.ndarray)
        assert samples.ndim == 3
        assert np.all(samples >= 0)

    def test_mcint_samples(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        samples = summariser.get_posterior_samples("mcint")
        assert isinstance(samples, np.ndarray)
        assert samples.ndim == 2
        assert np.all(samples >= 0)

    def test_delta_raises_for_agemix(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        with pytest.raises(ValueError, match="only available for genmix"):
            summariser.get_posterior_samples("delta")

    def test_unknown_quantity_raises(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        with pytest.raises(ValueError, match="Unknown quantity"):
            summariser.get_posterior_samples("unknown")


# ============================================================================
# Point estimates
# ============================================================================


class TestGetPointEstimates:
    def test_cint_estimates(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        est = summariser.get_point_estimates("cint")
        A = summariser.model.A

        assert isinstance(est, dict)
        assert "mean" in est and "std" in est
        assert est["mean"].shape == (A, A)
        assert est["std"].shape == (A, A)
        assert np.all(est["mean"] >= 0)
        assert np.all(est["std"] >= 0)

    def test_rate_estimates(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        est = summariser.get_point_estimates("rate")
        assert "mean" in est and "std" in est
        assert est["mean"].shape == (summariser.model.A, summariser.model.A)

    def test_mcint_estimates(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        est = summariser.get_point_estimates("mcint")
        assert "mean" in est
        assert est["mean"].shape == (summariser.model.A,)


# ============================================================================
# Strata property
# ============================================================================


class TestStrataProperty:
    def test_agemix_strata(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        assert summariser.strata == ["All->All"]


# ============================================================================
# Cache management
# ============================================================================


class TestCache:
    def test_cache_initially_empty(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        info = summariser.get_cache_info()
        assert info["n_cached"] == 0
        assert info["inference_method"] == "mcmc"
        assert info["model_type"] == "agemix"

    def test_cache_populated_after_summarise(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        summariser.summarise_rate(alpha=0.05)
        summariser.summarise_cint(alpha=0.05)
        assert summariser.get_cache_info()["n_cached"] == 2

    def test_clear_cache(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        summariser.summarise_cint(alpha=0.05)
        summariser.clear_cache()
        assert summariser.get_cache_info()["n_cached"] == 0


# ============================================================================
# Memory management
# ============================================================================


class TestMemoryManagement:
    def test_release_raw_samples(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        summariser.summarise_cint()
        assert summariser._post_samples is not None
        summariser.release_raw_samples()
        assert summariser._post_samples is None


# ============================================================================
# Error handling
# ============================================================================


class TestEdgeCases:
    def test_invalid_alpha(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        for bad_alpha in (0, 1, 1.5, -0.1):
            with pytest.raises(ValueError):
                summariser.summarise_cint(alpha=bad_alpha)

    def test_multiple_methods_cached_independently(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)
        summariser.summarise_rate(alpha=0.05)
        summariser.summarise_cint(alpha=0.05)
        summariser.summarise_mcint(alpha=0.05)
        assert summariser.get_cache_info()["n_cached"] == 3


# ============================================================================
# Integration
# ============================================================================


class TestIntegration:
    def test_full_workflow_mcmc(self, agemix_mcmc_model):
        summariser = ModelSummariser(agemix_mcmc_model)

        rate_summary = summariser.summarise_rate(alpha=0.05)
        cint_summary = summariser.summarise_cint(alpha=0.05)
        mcint_summary = summariser.summarise_mcint(alpha=0.05)
        rate_est = summariser.get_point_estimates("rate")
        rate_samples = summariser.get_posterior_samples("rate")

        assert isinstance(rate_summary["All->All"], ContactSummary)
        assert isinstance(cint_summary["All->All"], ContactSummary)
        assert isinstance(mcint_summary["All->All"], ContactSummary)
        assert "mean" in rate_est
        assert rate_samples.ndim == 3

    def test_full_workflow_svi(self, agemix_svi_model):
        summariser = ModelSummariser(agemix_svi_model, num_samples=50)
        cint_summary = summariser.summarise_cint(alpha=0.05)
        assert isinstance(cint_summary["All->All"], ContactSummary)
