import numpy as np
import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import DataLoader
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ContactGenerator, MatrixGenerator, ParticipantGenerator
from .._vdKassteele import vdKassteele
from ..priors import Spline2D
from .fixtures import (
    full_large_sample,
    full_multi_strat_large_sample,
    partial_large_sample,
    partial_multi_strat_large_sample,
    single_large_sample,
    single_large_sample_with_repeats,
    single_small_sample,
)


class TestInit:

    def test_single(self, single_large_sample):
        part_data, cnt_data, pop_data = single_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")
        model.prior_type == "global"
        assert model.y.shape == model.log_N.shape
        assert model.log_P.shape[1] == model.A

    def test_partial(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")
        assert model.prior_type == "partial"

    def test_full(self, full_large_sample):
        part_data, cnt_data, pop_data = full_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")
        assert model.prior_type == "full"

    def test_multi_strat_partial(self, partial_multi_strat_large_sample):
        part_data, cnt_data, pop_data = partial_multi_strat_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")
        assert model.prior_type == "partial"

    def test_multi_strat_full(self, full_multi_strat_large_sample):
        part_data, cnt_data, pop_data = full_multi_strat_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")
        assert model.prior_type == "full"

    def test_rid(self, single_large_sample_with_repeats):
        part_data, cnt_data, pop_data = single_large_sample_with_repeats
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")
        assert hasattr(model, "rid")
        assert len(model.rid) == len(model.y)


class TestModel:

    def test_model_callable(self, single_small_sample):
        """Test that model is callable"""
        from numpyro.handlers import seed

        part_data, cnt_data, pop_data = single_small_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")

        try:
            with seed(rng_seed=0):
                model.model(y=model.y)
        except Exception as e:
            pytest.fail(f"Model callable test failed with exception: {e}")

    def test_print_model_shape(self, single_small_sample):
        """Test print_model_shape method."""
        part_data, cnt_data, pop_data = single_small_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")

        try:
            model.print_model_shape()
        except Exception as e:
            pytest.fail(f"print_model_shape test failed with exception: {e}")


class TestInference:
    SEED = 0

    def test_single_svi(self, single_large_sample):
        part_data, cnt_data, pop_data = single_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")
        guide = AutoNormal(model.model)
        model.run_inference_svi(prng_key=PRNGKey(self.SEED), guide=guide, num_steps=5)

        assert model._svi_result is not None

    def test_partial_svi(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")
        guide = AutoNormal(model.model)

        model.run_inference_svi(prng_key=PRNGKey(self.SEED), guide=guide, num_steps=5)

        assert model._svi_result is not None

    def test_full_svi(self, full_large_sample):
        part_data, cnt_data, pop_data = full_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")
        guide = AutoNormal(model.model)

        model.run_inference_svi(prng_key=PRNGKey(self.SEED), guide=guide, num_steps=5)

        assert model._svi_result is not None

    def test_single_mcmc_init(self, single_large_sample):
        part_data, cnt_data, pop_data = single_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")

        assert hasattr(model, "run_inference_mcmc")
        assert callable(model.run_inference_mcmc)

    def test_partial_mcmc_init(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")

        assert hasattr(model, "run_inference_mcmc")
        assert callable(model.run_inference_mcmc)

    def test_full_mcmc_init(self, full_large_sample):
        part_data, cnt_data, pop_data = full_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model = vdKassteele(dataloader, "poisson")

        assert hasattr(model, "run_inference_mcmc")
        assert callable(model.run_inference_mcmc)
