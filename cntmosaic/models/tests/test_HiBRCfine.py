import numpy as np
import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import DataLoader, StratificationData
from ...datasets import load_age_distribution, load_template_patterns
from .._HiBRCfine import HiBRCfine
from ..priors import PSpline2D
from .fixtures import (
    full_large_sample,
    full_multi_strat_large_sample,
    partial_large_sample,
    partial_multi_strat_large_sample,
)


class TestInit:

    def test_partial(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_data,
        )

        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="partial", M=10)

        # Test model initialization
        model = HiBRCfine(dataloader, priors, "poisson")

    def test_full(self, full_large_sample):
        part_data, cnt_data, pop_data = full_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_data,
        )

        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="full", M=10)

        # Test model initialization
        model = HiBRCfine(dataloader, priors, "poisson")

    def test_multi_strat_partial(self, partial_multi_strat_large_sample):
        part_data, cnt_data, pop_data = partial_multi_strat_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_data,
        )

        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="partial", M=10)

        # Test model initialization
        model = HiBRCfine(dataloader, priors, "poisson")

    def test_multi_strat_full(self, full_multi_strat_large_sample):
        part_data, cnt_data, pop_data = full_multi_strat_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_data,
        )

        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="full", M=10)

        # Test model initialization
        model = HiBRCfine(dataloader, priors, "poisson")


class TestModel:

    def test_model_callable(self, partial_large_sample):
        """Test that model is callable"""
        from numpyro.handlers import seed

        part_data, cnt_data, pop_data = partial_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data=strat_data)
        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="partial", M=10)
        model = HiBRCfine(dataloader, priors, likelihood="poisson")

        try:
            with seed(rng_seed=0):
                model.model(y=model.y)
        except Exception as e:
            pytest.fail(f"Model callable test failed with error: {e}")

    def test_print_model_shape(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data=strat_data)
        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="partial", M=10)
        model = HiBRCfine(dataloader, priors, likelihood="poisson")

        try:
            model.print_model_shape()
        except Exception as e:
            pytest.fail(f"Print model shape test failed with error: {e}")


class TestInference:
    SEED = 0

    def test_partial_svi(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_data,
        )

        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="partial", M=10)

        model = HiBRCfine(dataloader, priors, "poisson")
        guide = AutoNormal(model.model)

        try:
            model.run_inference_svi(PRNGKey(self.SEED), guide, num_steps=5)
            assert model._svi_result is not None
        except Exception as e:
            pytest.fail(f"SVI inference test failed with error: {e}")

    def test_full_svi(self, full_large_sample):
        part_data, cnt_data, pop_data = full_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_data,
        )

        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="full", M=10)

        model = HiBRCfine(dataloader, priors, "poisson")
        guide = AutoNormal(model.model)

        try:
            model.run_inference_svi(PRNGKey(self.SEED), guide, num_steps=5)
            assert model._svi_result is not None
        except Exception as e:
            pytest.fail(f"SVI inference test failed with error: {e}")

    def test_partial_mcmc_init(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_data,
        )

        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="partial", M=10)

        model = HiBRCfine(dataloader, priors, "poisson")

        assert hasattr(model, "run_inference_mcmc")
        assert callable(model.run_inference_mcmc)

    def test_full_mcmc_init(self, full_large_sample):
        part_data, cnt_data, pop_data = full_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_data,
        )

        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="full", M=10)

        model = HiBRCfine(dataloader, priors, "poisson")

        assert hasattr(model, "run_inference_mcmc")
        assert callable(model.run_inference_mcmc)
