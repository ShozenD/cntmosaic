"""
Unit tests for Prem2 classical data preparation and NumPyro model.

Tests are organised by concern:
  TestInit            – stratification bookkeeping (strat_vars, strat_dims, K)
  TestDataLoading     – aggregated arrays (Y, N, C, D, K_part, K_cnt, P)
  TestEmptyAgeGroups  – no adaptive_merge, so empty groups always raise
  TestModel           – NumPyro model() is callable and produces correct shapes
  TestInference       – run_inference_svi actually runs (catches self.y bugs)
"""

import numpy as np
import pytest
from jax.random import PRNGKey

from ...utils import AgeGroupSpecs
from ..classical._Prem2 import Prem2
from .fixtures import (
    full_large_sample,
    full_multi_strat_large_sample,
    partial_large_sample,
    partial_multi_strat_large_sample,
    partial_small_sample,
    single_large_sample,
    single_small_sample,
)

AGE_BINS = AgeGroupSpecs(0, 80, 5)  # 16 age groups


def _from_containers(fixture):
    """Build Prem2 with default 5-year age bins, ignoring pop_data."""
    part_data, cnt_data, _pop_data = fixture
    return Prem2(part_data, cnt_data, AGE_BINS)


class TestInit:
    """Test Prem2 model initialization."""

    def test_single(self, single_large_sample):
        """Test initialization without stratification."""
        model = _from_containers(single_large_sample)

        assert model.strat_vars_part == []
        assert model.strat_dims_part == {}
        assert model.strat_vars_cnt == []
        assert model.strat_dims_cnt == {}
        assert model.strat_vars_shared == []
        assert model.strat_mode == "single"
        assert model.K == 1

    def test_partial(self, partial_large_sample):
        """Test initialization with participant-only stratification."""
        model = _from_containers(partial_large_sample)

        assert model.strat_vars_part == ["sex"]
        assert model.strat_dims_part == {"sex": 2}
        assert model.strat_vars_cnt == []
        assert model.strat_dims_cnt == {}
        assert model.strat_vars_shared == []
        assert model.strat_mode == "partial"
        assert model.K == 2

    def test_partial_multi_strat(self, partial_multi_strat_large_sample):
        """Test initialization with participant-only multi-variable stratification."""
        model = _from_containers(partial_multi_strat_large_sample)

        assert sorted(model.strat_vars_part) == ["ses", "sex"]
        assert model.strat_dims_part == {"sex": 2, "ses": 2}
        assert model.strat_vars_cnt == []
        assert model.strat_dims_cnt == {}
        assert model.strat_vars_shared == []
        assert model.K == 4  # 2 (sex) * 2 (ses)

    def test_full(self, full_large_sample):
        """Test initialization with full stratification."""
        model = _from_containers(full_large_sample)

        assert model.strat_vars_part == ["sex"]
        assert model.strat_dims_part == {"sex": 2}
        assert model.strat_vars_cnt == ["sex"]
        assert model.strat_dims_cnt == {"sex": 2}
        assert model.strat_vars_shared == ["sex"]
        assert model.strat_mode == "full"
        assert model.K == 4

    def test_full_multi_strat(self, full_multi_strat_large_sample):
        """Test initialization with full multi-variable stratification."""
        model = _from_containers(full_multi_strat_large_sample)

        assert sorted(model.strat_vars_part) == ["ses", "sex"]
        assert model.strat_dims_part == {"sex": 2, "ses": 2}
        assert sorted(model.strat_vars_cnt) == ["ses", "sex"]
        assert model.strat_dims_cnt == {"sex": 2, "ses": 2}
        assert sorted(model.strat_vars_shared) == ["ses", "sex"]
        assert model.K == 16  # 2 (sex) * 2 (ses) * 2 (sex) * 2 (ses)


class TestDataLoading:
    """Test that _load populates the aggregated count arrays."""

    def test_single(self, single_large_sample):
        model = _from_containers(single_large_sample)

        assert model.Y is not None
        assert model.N is not None
        assert model.P is None
        assert isinstance(model.Y, np.ndarray)
        assert isinstance(model.N, np.ndarray)
        assert model.K_part == 1
        assert model.K_cnt == 1
        assert model.C == 16
        assert model.D == 16
        assert model.Y.shape == (model.C, model.D)
        assert model.N.shape == (model.C,)

    def test_partial(self, partial_large_sample):
        model = _from_containers(partial_large_sample)

        assert model.Y is not None
        assert model.N is not None
        assert model.P is None
        assert model.K_part == 2
        assert model.K_cnt == 1
        assert model.Y.shape == (model.K_part, model.C, model.D)
        assert model.N.shape == (model.K_part, model.C)

    def test_full(self, full_large_sample):
        model = _from_containers(full_large_sample)

        assert model.Y is not None
        assert model.N is not None
        assert model.P is None
        assert model.K_part == 2
        assert model.K_cnt == 2
        assert model.Y.shape == (model.K_part, model.K_cnt, model.C, model.D)
        assert model.N.shape == (model.K_part, model.C)

    def test_partial_multi_strat(self, partial_multi_strat_large_sample):
        model = _from_containers(partial_multi_strat_large_sample)

        assert model.K_part == 4  # 2 (sex) * 2 (ses)
        assert model.K_cnt == 1
        assert model.Y.shape == (model.K_part, model.C, model.D)

    def test_full_multi_strat(self, full_multi_strat_large_sample):
        model = _from_containers(full_multi_strat_large_sample)

        assert model.K_part == 4  # 2 (sex) * 2 (ses)
        assert model.K_cnt == 4
        assert model.Y.shape == (model.K_part, model.K_cnt, model.C, model.D)


class TestEmptyAgeGroups:
    """Prem2 has no adaptive_merge — empty participant age groups always raise.

    Unlike SocialMix (which supports adaptive_merge=True), Prem2Validator
    always raises ValueError on empty age groups, matching the "no
    adaptive_merge argument" requirement.
    """

    def test_single_small_raises(self, single_small_sample):
        part_data, cnt_data, _pop_data = single_small_sample
        with pytest.raises(ValueError):
            Prem2(part_data, cnt_data, AGE_BINS)

    def test_partial_small_raises(self, partial_small_sample):
        part_data, cnt_data, _pop_data = partial_small_sample
        with pytest.raises(ValueError):
            Prem2(part_data, cnt_data, AGE_BINS)


class TestModel:
    """Test the NumPyro model specification over aggregated counts."""

    def test_model_callable_single(self, single_large_sample):
        """Test that model is callable for the unstratified case."""
        from numpyro.handlers import seed

        model = _from_containers(single_large_sample)

        try:
            with seed(rng_seed=0):
                model.model(y=model.Y)
        except Exception as e:
            pytest.fail(f"Model call raised exception: {e}")

    def test_model_callable_partial(self, partial_large_sample):
        """Test that model is callable for participant-only stratification."""
        from numpyro.handlers import seed

        model = _from_containers(partial_large_sample)

        try:
            with seed(rng_seed=0):
                model.model(y=model.Y)
        except Exception as e:
            pytest.fail(f"Model call raised exception: {e}")

    def test_model_callable_full(self, full_large_sample):
        """Test that model is callable for full stratification."""
        from numpyro.handlers import seed

        model = _from_containers(full_large_sample)

        try:
            with seed(rng_seed=0):
                model.model(y=model.Y)
        except Exception as e:
            pytest.fail(f"Model call raised exception: {e}")

    def test_model_callable_poisson(self, single_large_sample):
        """Test that the poisson likelihood branch is callable."""
        from numpyro.handlers import seed

        part_data, cnt_data, _pop_data = single_large_sample
        model = Prem2(part_data, cnt_data, AGE_BINS, likelihood="poisson")

        try:
            with seed(rng_seed=0):
                model.model(y=model.Y)
        except Exception as e:
            pytest.fail(f"Model call raised exception: {e}")

    def test_model_trace_shapes_single(self, single_large_sample):
        """Sample sites have the expected (unstratified) shapes."""
        from numpyro.handlers import seed, trace

        model = _from_containers(single_large_sample)

        with seed(rng_seed=0):
            tr = trace(model.model).get_trace(y=model.Y)

        assert tr["baseline"]["value"].shape == ()
        assert tr["tau"]["value"].shape == ()
        assert tr["f"]["value"].shape == (model.C * model.D,)
        assert tr["log_cint"]["value"].shape == (model.C, model.D)
        assert tr["obs"]["value"].shape == (model.C, model.D)

    def test_model_trace_shapes_full(self, full_large_sample):
        """Sample sites have the expected (fully stratified) shapes."""
        from numpyro.handlers import seed, trace

        model = _from_containers(full_large_sample)
        K = model.K_part * model.K_cnt

        with seed(rng_seed=0):
            tr = trace(model.model).get_trace(y=model.Y)

        assert tr["baseline"]["value"].shape == (K,)
        assert tr["tau"]["value"].shape == (K,)
        assert tr["log_cint"]["value"].shape == (
            model.K_part,
            model.K_cnt,
            model.C,
            model.D,
        )
        assert tr["obs"]["value"].shape == (
            model.K_part,
            model.K_cnt,
            model.C,
            model.D,
        )


class TestInference:
    """Test that inference actually runs end-to-end.

    Unlike shallow "method exists" checks, these run a few SVI steps so that
    bugs like a missing/mismatched ``self.y`` (which ``ContactModel.
    run_inference_svi``/``run_inference_mcmc`` read observation data from) are
    caught rather than silently passing.
    """

    def test_svi_runs_single(self, single_large_sample):
        model = _from_containers(single_large_sample)
        model.run_inference_svi(PRNGKey(0), num_steps=5)

        assert model._svi_result is not None
        samples = model.draw_posterior_samples(PRNGKey(1), num_samples=5)
        assert samples["log_cint"].shape == (5, model.C, model.D)

    def test_svi_runs_full(self, full_large_sample):
        model = _from_containers(full_large_sample)
        model.run_inference_svi(PRNGKey(0), num_steps=5)

        assert model._svi_result is not None
        samples = model.draw_posterior_samples(PRNGKey(1), num_samples=5)
        assert samples["log_cint"].shape == (
            5,
            model.K_part,
            model.K_cnt,
            model.C,
            model.D,
        )

    def test_svi_runs_poisson_likelihood(self, single_large_sample):
        part_data, cnt_data, _pop_data = single_large_sample
        model = Prem2(part_data, cnt_data, AGE_BINS, likelihood="poisson")
        model.run_inference_svi(PRNGKey(0), num_steps=5)

        assert model._svi_result is not None
