import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import ContactSurveyLoader
from .._AgeMixCC import AgeMixCC
from ..numpyro.priors import vdKassteele2D
from .fixtures import single_coarse_coarse_small


class TestNumPyroModel:

    def test_model_callable(self, single_coarse_coarse_small):
        from numpyro.handlers import seed

        part_data, cnt_data, pop_data = single_coarse_coarse_small
        dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)
        model = AgeMixCC(
            dataloader,
            {"rate": vdKassteele2D(prior_type="global")},
            likelihood="poisson",
        )

        with seed(rng_seed=0):
            model.model(y=model.y)

    def test_print_model_shape(self, single_coarse_coarse_small):
        part_data, cnt_data, pop_data = single_coarse_coarse_small
        dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)
        model = AgeMixCC(
            dataloader,
            {"rate": vdKassteele2D(prior_type="global")},
            likelihood="poisson",
        )

        model.print_model_shape()


class TestNumPyroInference:

    def test_svi(self, single_coarse_coarse_small):
        part_data, cnt_data, pop_data = single_coarse_coarse_small
        dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)
        model = AgeMixCC(
            dataloader,
            {"rate": vdKassteele2D(prior_type="global")},
            likelihood="poisson",
        )

        guide = AutoNormal(model.model)
        model.run_inference_svi(PRNGKey(0), guide, num_steps=50, peak_lr=0.01)

        assert model._svi_result is not None
        assert model._svi_result.params is not None

    def test_mcmc(self, single_coarse_coarse_small):
        part_data, cnt_data, pop_data = single_coarse_coarse_small
        dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)
        model = AgeMixCC(
            dataloader,
            {"rate": vdKassteele2D(prior_type="global")},
            likelihood="poisson",
        )

        model.run_inference_mcmc(
            PRNGKey(1), num_warmup=10, num_samples=10, num_chains=1
        )

        assert model._mcmc_result is not None
        samples = model._mcmc_result.get_samples()
        assert "baseline" in samples
        assert samples["baseline"].shape[0] == 10
