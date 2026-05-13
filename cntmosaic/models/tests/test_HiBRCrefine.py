import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import DataLoader, StratificationData
from .._HiBRCrefine import HiBRCrefine
from ..priors import PSpline2D
from .fixtures import (
    full_coarse_large_sample,
    full_coarse_multi_strat_large_sample,
    partial_coarse_large_sample,
    partial_coarse_multi_strat_large_sample,
)


class TestInit:

    def test_partial(self, partial_coarse_large_sample):
        part_data, cnt_data, pop_data = partial_coarse_large_sample
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
        model = HiBRCrefine(dataloader, priors, "poisson")

    def test_full(self, full_coarse_large_sample):
        part_data, cnt_data, pop_data = full_coarse_large_sample
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
        model = HiBRCrefine(dataloader, priors, "poisson")

    def test_multi_strat_partial(self, partial_coarse_multi_strat_large_sample):
        part_data, cnt_data, pop_data = partial_coarse_multi_strat_large_sample
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
        model = HiBRCrefine(dataloader, priors, "poisson")

    def test_multi_strat_full(self, full_coarse_multi_strat_large_sample):
        part_data, cnt_data, pop_data = full_coarse_multi_strat_large_sample
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
        model = HiBRCrefine(dataloader, priors, "poisson")


class TestModel:

    def test_model_callable(self, partial_coarse_large_sample):
        """Test that model is callable"""
        from numpyro.handlers import seed

        part_data, cnt_data, pop_data = partial_coarse_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_data)
        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="partial", M=10)
        model = HiBRCrefine(dataloader, priors, likelihood="poisson")

        try:
            with seed(rng_seed=0):
                model.model(y=model.y)
        except Exception as e:
            pytest.fail(f"Model callable test failed with error: {e}")

    def test_print_model_shape(self, partial_coarse_large_sample):
        part_data, cnt_data, pop_data = partial_coarse_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_data)
        priors = {"rate": PSpline2D(prior_type="global", M=10)}
        for var in strat_vars:
            priors[var] = PSpline2D(prior_type="partial", M=10)
        model = HiBRCrefine(dataloader, priors, likelihood="poisson")

        try:
            model.print_model_shape()
        except Exception as e:
            pytest.fail(f"Print model shape test failed with error: {e}")


# Test inference
class TestInference:

    def test_svi_inference(self, partial_coarse_large_sample):
        part_data, cnt_data, pop_data = partial_coarse_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )
        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_data)
        priors = {
            "rate": PSpline2D(prior_type="global", M=5),
            "sex": PSpline2D(prior_type="partial", M=5),
        }
        model = HiBRCrefine(dataloader, priors, likelihood="negbin")

        prng_key = PRNGKey(0)
        guide = AutoNormal(model.model)
        model.run_inference_svi(prng_key, guide, num_steps=1000, peak_lr=0.01)

        assert model._svi_result is not None

    # Test MCMC inference
    def test_mcmc_inference(self, partial_coarse_large_sample):
        part_data, cnt_data, pop_data = partial_coarse_large_sample
        df_pop = pop_data.data
        strat_vars = pop_data.get_strat_vars()
        strat_data = StratificationData.from_counts(
            df_pop, age_col="age", strat_var_cols=strat_vars, count_col="P"
        )
        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_data)
        priors = {
            "rate": PSpline2D(prior_type="global", M=5),
            "sex": PSpline2D(prior_type="partial", M=5),
        }
        model = HiBRCrefine(dataloader, priors, likelihood="negbin")

        prng_key = PRNGKey(1)
        model.run_inference_mcmc(prng_key, num_warmup=10, num_samples=10, num_chains=1)

        assert model._mcmc_result is not None
