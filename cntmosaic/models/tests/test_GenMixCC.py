import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import ContactSurveyLoader, StratificationData
from .._GenMixCC import GenMixCC
from ..numpyro.priors import vdKassteele2D
from .fixtures import full_coarse_coarse_large_sample, partial_coarse_coarse_large_sample


def _build_model(fixture, prior_type):
    part_data, cnt_data, pop_data = fixture
    strat_vars = pop_data.get_strat_vars()

    # Extract age_min/age_max from the interval column so StratificationData can
    # handle the coarse age groups.
    df_strat = pop_data.data.copy()
    df_strat["age_min"] = df_strat["age_grp_pop"].apply(lambda x: x.left).astype(int)
    df_strat["age_max"] = df_strat["age_grp_pop"].apply(lambda x: x.right).astype(int)
    strat_data = StratificationData.from_counts(
        df_strat,
        age_min_col="age_min",
        age_max_col="age_max",
        strat_var_cols=strat_vars,
        count_col="P",
    )

    dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data, strat_data)
    priors = {"rate": vdKassteele2D(prior_type="global")}
    for var in strat_vars:
        priors[var] = vdKassteele2D(prior_type=prior_type)
    return GenMixCC(dataloader, priors, "poisson")


class TestNumPyroModel:

    def test_model_callable(self, partial_coarse_coarse_large_sample):
        from numpyro.handlers import seed

        model = _build_model(partial_coarse_coarse_large_sample, "partial")
        with seed(rng_seed=0):
            model.model(y=model.y)

    def test_print_model_shape(self, partial_coarse_coarse_large_sample):
        model = _build_model(partial_coarse_coarse_large_sample, "partial")
        model.print_model_shape()


class TestNumPyroInference:

    def test_svi_partial(self, partial_coarse_coarse_large_sample):
        model = _build_model(partial_coarse_coarse_large_sample, "partial")
        guide = AutoNormal(model.model)
        model.run_inference_svi(PRNGKey(0), guide, num_steps=50, peak_lr=0.01)

        assert model._svi_result is not None
        assert model._svi_result.params is not None

    def test_svi_full(self, full_coarse_coarse_large_sample):
        model = _build_model(full_coarse_coarse_large_sample, "full")
        guide = AutoNormal(model.model)
        model.run_inference_svi(PRNGKey(0), guide, num_steps=50, peak_lr=0.01)

        assert model._svi_result is not None
        assert model._svi_result.params is not None

    def test_mcmc_partial(self, partial_coarse_coarse_large_sample):
        model = _build_model(partial_coarse_coarse_large_sample, "partial")
        model.run_inference_mcmc(PRNGKey(1), num_warmup=10, num_samples=10, num_chains=1)

        assert model._mcmc_result is not None
        samples = model._mcmc_result.get_samples()
        assert "baseline" in samples
        assert samples["baseline"].shape[0] == 10

    def test_mcmc_full(self, full_coarse_coarse_large_sample):
        model = _build_model(full_coarse_coarse_large_sample, "full")
        model.run_inference_mcmc(PRNGKey(1), num_warmup=10, num_samples=10, num_chains=1)

        assert model._mcmc_result is not None
        samples = model._mcmc_result.get_samples()
        assert "baseline" in samples
        assert samples["baseline"].shape[0] == 10
