import numpy as np
import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import DataLoader
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ContactGenerator, MatrixGenerator, ParticipantGenerator
from .._BRCfine import BRCfine
from ..priors import Spline2D
from .fixtures import (
    single_large_sample,
    single_large_sample_with_repeats,
    single_small_sample,
)

# Language: python


# Test initialization
class TestInit:

    def test_single(self, single_large_sample):
        part_data, cnt_data, pop_data = single_large_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        assert len(model.y) > 0
        assert len(model.aid) == len(model.y)
        assert len(model.bid) == len(model.y)
        assert len(model.log_N) > 0
        assert model.log_P.shape[1] == model.A
        assert model.log_S.shape[0] == len(model.y)

        model = BRCfine(dataloader, priors, likelihood="negbin", inv_odist=2.0)

    def test_single_small(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        assert len(model.y) > 0
        assert len(model.aid) == len(model.y)
        assert len(model.bid) == len(model.y)
        assert len(model.log_N) > 0
        assert model.log_P.shape[1] == model.A
        assert model.log_S.shape[0] == len(model.y)

    def test_init_with_rid(self, single_large_sample_with_repeats):
        part_data, cnt_data, pop_data = single_large_sample_with_repeats
        dataloader = DataLoader(part_data, cnt_data, pop_data)
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        assert hasattr(model, "rid")
        assert len(model.rid) == len(model.y)
        assert model.hill.max_value == 4


class testModel:

    def test_model_callable(self, single_small_sample):
        """Test that model is callable"""
        from numpyro.handlers import seed

        part_data, cnt_data, pop_data = single_small_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        try:
            with seed(rng_seed=0):
                model.model(y=model.y)
        except Exception as e:
            pytest.fail(f"Model call raised exception: {e}")

    def test_print_model_shape(self, single_small_sample):
        """Test print_model_shape method."""
        part_data, cnt_data, pop_data = single_small_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        try:
            model.print_model_shape()
        except Exception as e:
            pytest.fail(f"print_model_shape raised exception: {e}")


# Test SVI inference
class TestInference:

    def test_svi_inference(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        prng_key = PRNGKey(0)
        guide = AutoNormal(model.model)
        model.run_inference_svi(prng_key, guide, num_steps=1000, peak_lr=0.01)

        assert model._svi_result is not None

    # Test MCMC inference
    def test_mcmc_inference(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        dataloader = DataLoader(part_data, cnt_data, pop_data)
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        prng_key = PRNGKey(1)
        model.run_inference_mcmc(prng_key, num_warmup=10, num_samples=10, num_chains=1)

        assert model._mcmc_result is not None
