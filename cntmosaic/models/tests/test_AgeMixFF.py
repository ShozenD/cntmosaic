import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import ContactSurveyLoader
from .._AgeMixFF import AgeMixFF
from ..numpyro.priors import Spline2D
from .fixtures import single_small_sample


class TestNumPyroModel:

    def test_model_callable(self, single_small_sample):
        from numpyro.handlers import seed

        part_data, cnt_data, pop_data = single_small_sample
        dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)
        model = AgeMixFF(dataloader, {"rate": Spline2D(prior_type="global")}, likelihood="poisson")

        with seed(rng_seed=0):
            model.model(y=model.y)

    def test_print_model_shape(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)
        model = AgeMixFF(dataloader, {"rate": Spline2D(prior_type="global")}, likelihood="poisson")

        model.print_model_shape()


class TestNumPyroInference:

    def test_svi(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)
        model = AgeMixFF(dataloader, {"rate": Spline2D(prior_type="global")}, likelihood="poisson")

        guide = AutoNormal(model.model)
        model.run_inference_svi(PRNGKey(0), guide, num_steps=50, peak_lr=0.01)

        assert model._svi_result is not None
        assert model._svi_result.params is not None

    def test_mcmc(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)
        model = AgeMixFF(dataloader, {"rate": Spline2D(prior_type="global")}, likelihood="poisson")

        model.run_inference_mcmc(PRNGKey(1), num_warmup=10, num_samples=10, num_chains=1)

        assert model._mcmc_result is not None
        samples = model._mcmc_result.get_samples()
        assert "baseline" in samples
        assert samples["baseline"].shape[0] == 10
