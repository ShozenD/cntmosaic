"""
Tests for ModelSummariserPrem2.

Tests are organised by concern:
  TestReshapeCintSamples        – log_cint -> Dict[str, NDArray] reshape branches
  TestApplyReciprocity          – batched reciprocity staticmethod (pure arrays)
  TestRealSmallModelIntegration – real Prem2 + a few SVI steps, end-to-end
  TestSummariseAPI              – alpha validation, caching, missing-pop_data errors
  TestInitValidation            – inference_method / population-stratification checks
"""

import numpy as np
import pytest
from jax.random import PRNGKey

from ...dataloader.containers import PopulationData
from ...models.classical._Prem2 import Prem2
from ...models.classical._socialmix_helpers import apply_reciprocity as _sm_apply_reciprocity
from ...models.tests.fixtures import (
    full_large_sample,
    partial_large_sample,
    single_large_sample,
)
from ...utils import AgeGroupSpecs
from .. import ModelSummariserPrem2

AGE_BINS = AgeGroupSpecs(0, 80, 5)  # 16 age groups


def _new_summariser() -> ModelSummariserPrem2:
    """Bare instance, bypassing __init__, for testing pure methods in isolation."""
    return object.__new__(ModelSummariserPrem2)


def _build_summariser(fixture, num_steps=5, num_samples=20) -> ModelSummariserPrem2:
    part_data, cnt_data, pop_data = fixture
    model = Prem2(part_data, cnt_data, AGE_BINS)
    model.run_inference_svi(PRNGKey(0), num_steps=num_steps)
    return ModelSummariserPrem2(model, pop_data=pop_data, num_samples=num_samples)


# ============================================================================
# _reshape_cint_samples
# ============================================================================


class TestReshapeCintSamples:
    """log_cint posterior samples -> Dict[str, NDArray], one entry per stratum."""

    def test_single(self):
        summ = _new_summariser()
        summ.K_part, summ.K_cnt, summ.C, summ.D = 1, 1, 4, 5
        summ.strata_labels = ["All->All"]

        log_cint = np.random.normal(size=(10, 4, 5))
        result = summ._reshape_cint_samples(log_cint)

        assert set(result.keys()) == {"All->All"}
        assert result["All->All"].shape == (10, 4, 5)
        assert np.allclose(result["All->All"], np.exp(log_cint))

    def test_partial(self):
        summ = _new_summariser()
        summ.K_part, summ.K_cnt, summ.C, summ.D = 2, 1, 4, 5
        summ.strata_labels = ["M->All", "F->All"]

        log_cint = np.random.normal(size=(10, 2, 4, 5))
        result = summ._reshape_cint_samples(log_cint)

        assert set(result.keys()) == {"M->All", "F->All"}
        assert result["M->All"].shape == (10, 4, 5)
        assert np.allclose(result["M->All"], np.exp(log_cint[:, 0]))
        assert np.allclose(result["F->All"], np.exp(log_cint[:, 1]))

    def test_full(self):
        summ = _new_summariser()
        summ.K_part, summ.K_cnt, summ.C, summ.D = 2, 2, 4, 5
        summ.strata_labels = ["F->F", "F->M", "M->F", "M->M"]

        log_cint = np.random.normal(size=(10, 2, 2, 4, 5))
        result = summ._reshape_cint_samples(log_cint)

        assert set(result.keys()) == set(summ.strata_labels)
        for k_part in range(2):
            for k_cnt in range(2):
                idx = k_part * 2 + k_cnt
                label = summ.strata_labels[idx]
                assert np.allclose(result[label], np.exp(log_cint[:, k_part, k_cnt]))


# ============================================================================
# apply_reciprocity (staticmethod, batched over posterior draws)
# ============================================================================


class TestApplyReciprocity:
    """Batched reciprocity wraps _socialmix_helpers.apply_reciprocity per draw."""

    def test_matches_per_draw_helper_single(self):
        n, C, D = 8, 5, 5
        samples = {"All->All": np.random.uniform(1, 10, size=(n, C, D))}
        P = np.random.uniform(100, 1000, size=D)

        result = ModelSummariserPrem2.apply_reciprocity(samples, P, "single", K_cnt=1)

        for i in range(n):
            expected = _sm_apply_reciprocity(
                {"All->All": samples["All->All"][i]}, "single", 1, P
            )
            assert np.allclose(result["All->All"][i], expected["All->All"])

    def test_matches_per_draw_helper_full(self):
        n, C, D = 6, 5, 5
        labels = ["M->M", "M->F", "F->M", "F->F"]
        samples = {label: np.random.uniform(1, 10, size=(n, C, D)) for label in labels}
        P = np.random.uniform(100, 1000, size=(2, D))

        result = ModelSummariserPrem2.apply_reciprocity(samples, P, "full", K_cnt=2)

        for i in range(n):
            draw = {label: samples[label][i] for label in labels}
            expected = _sm_apply_reciprocity(draw, "full", 2, P)
            for label in labels:
                assert np.allclose(result[label][i], expected[label])

    @pytest.mark.parametrize("strat_mode", ["partial", "mixed"])
    def test_noop_with_warning(self, strat_mode):
        samples = {"M->All": np.random.rand(5, 4, 4)}

        with pytest.warns(UserWarning, match="Reciprocity not applied"):
            result = ModelSummariserPrem2.apply_reciprocity(
                samples, P=None, strat_mode=strat_mode, K_cnt=1
            )

        assert np.array_equal(result["M->All"], samples["M->All"])

    def test_missing_P_raises(self):
        samples = {"All->All": np.random.rand(5, 4, 4)}
        with pytest.raises(ValueError, match="P .* required"):
            ModelSummariserPrem2.apply_reciprocity(samples, P=None, strat_mode="single", K_cnt=1)


# ============================================================================
# Real small-model integration
# ============================================================================


class TestRealSmallModelIntegration:
    """Build a real (small) Prem2, run a few SVI steps, summarise end-to-end."""

    def test_single(self, single_large_sample):
        summ = _build_summariser(single_large_sample)

        assert summ.strata_labels == ["All->All"]

        cint = summ.summarise_cint(alpha=0.05)
        rate = summ.summarise_rate(alpha=0.05)
        mcint = summ.summarise_mcint(alpha=0.05)

        assert set(cint.keys()) == {"All->All"}
        c = cint["All->All"]
        assert c.central.shape == (16, 16)
        assert np.all(c.lower <= c.central)
        assert np.all(c.central <= c.upper)

        assert rate["All->All"].central.shape == (16, 16)
        assert mcint["All->All"].central.shape == (16,)

    def test_partial(self, partial_large_sample):
        summ = _build_summariser(partial_large_sample)

        assert set(summ.strata_labels) == {"M->All", "F->All"}

        cint = summ.summarise_cint(alpha=0.05)
        assert set(cint.keys()) == {"M->All", "F->All"}
        for label, c in cint.items():
            assert c.central.shape == (16, 16)
            assert np.all(c.lower <= c.central)
            assert np.all(c.central <= c.upper)

        # Reciprocity is a no-op (with warning) for partial stratification.
        with pytest.warns(UserWarning, match="Reciprocity not applied"):
            recip = summ.summarise_cint(alpha=0.05, apply_reciprocity=True, force_recompute=True)
        for label in cint:
            assert np.allclose(recip[label].central, cint[label].central)

    def test_full_reciprocity_and_depixilation(self, full_large_sample):
        summ = _build_summariser(full_large_sample)

        assert set(summ.strata_labels) == {"M->M", "M->F", "F->M", "F->F"}

        # Combined reciprocity + depixilation — a combination ModelSummariserPrem's
        # own tests skip (differing population resolutions needed for each op).
        result = summ.summarise_cint(alpha=0.05, apply_reciprocity=True, return_depixilated=True)

        assert set(result.keys()) == set(summ.strata_labels)
        for label, c in result.items():
            assert c.central.shape == (81, 81)
            assert np.all(c.lower <= c.central + 1e-9)
            assert np.all(c.central <= c.upper + 1e-9)

        rate = summ.summarise_rate(alpha=0.05, apply_reciprocity=True)
        for label, c in rate.items():
            assert c.central.shape == (16, 16)

        pe = summ.get_point_estimates(apply_reciprocity=True)
        assert set(pe.keys()) == set(summ.strata_labels)
        for label, entry in pe.items():
            assert set(entry.keys()) >= {"cint", "mcint"}
            assert entry["cint"]["mean"].shape == (16, 16)


# ============================================================================
# Public API: alpha validation, caching, error handling
# ============================================================================


class TestSummariseAPI:
    """Alpha validation, caching behaviour, and error paths."""

    @pytest.fixture
    def summ(self, single_large_sample):
        return _build_summariser(single_large_sample)

    def test_invalid_alpha_raises(self, summ):
        with pytest.raises(ValueError, match="alpha must be in"):
            summ.summarise_cint(alpha=1.5)

    def test_caching(self, summ):
        summ.clear_cache()
        assert summ.get_cache_info()["n_cached"] == 0

        summ.summarise_cint(alpha=0.1)
        info = summ.get_cache_info()
        assert info["n_cached"] == 1

        # Repeated call with same args hits cache (no new key).
        summ.summarise_cint(alpha=0.1)
        assert summ.get_cache_info()["n_cached"] == 1

        # force_recompute doesn't grow the cache, just refreshes the entry.
        summ.summarise_cint(alpha=0.1, force_recompute=True)
        assert summ.get_cache_info()["n_cached"] == 1

        summ.summarise_cint(alpha=0.2)
        assert summ.get_cache_info()["n_cached"] == 2

        summ.clear_cache()
        assert summ.get_cache_info()["n_cached"] == 0

    def test_depixilation_without_pop_data_raises(self, single_large_sample):
        part_data, cnt_data, _pop_data = single_large_sample
        model = Prem2(part_data, cnt_data, AGE_BINS)
        model.run_inference_svi(PRNGKey(0), num_steps=5)

        with pytest.warns(UserWarning, match="PopulationData not provided"):
            summ = ModelSummariserPrem2(model, pop_data=None, num_samples=10)

        with pytest.raises(ValueError, match="pop_data must be provided"):
            summ.summarise_cint(return_depixilated=True)

        with pytest.raises(ValueError, match="Population data required"):
            summ.summarise_rate()


# ============================================================================
# Constructor validation
# ============================================================================


class TestInitValidation:
    """Constructor-time validation."""

    def test_no_inference_raises(self):
        prem2 = type("Prem2", (), {})()
        prem2.inference_method = None

        with pytest.raises(ValueError, match="Either MCMC or SVI must have been run"):
            ModelSummariserPrem2(prem2)

    def test_mismatched_population_categories_raises(self, partial_large_sample):
        part_data, cnt_data, _pop_data = partial_large_sample
        model = Prem2(part_data, cnt_data, AGE_BINS)
        model.run_inference_svi(PRNGKey(0), num_steps=5)

        import pandas as pd

        bad_pop_df = pd.DataFrame(
            {
                "age": list(range(80)) * 2,
                "sex": ["X"] * 80 + ["Y"] * 80,
                "P": np.random.uniform(100, 1000, 160),
            }
        )
        bad_pop_data = PopulationData(
            bad_pop_df, age_col="age", size_col="P", strat_var_cols=["sex"]
        )

        with pytest.raises(ValueError, match="different categories"):
            ModelSummariserPrem2(model, pop_data=bad_pop_data)
