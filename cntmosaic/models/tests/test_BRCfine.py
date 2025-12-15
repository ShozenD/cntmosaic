import numpy as np
import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import ContactData, DataLoader, ParticipantData, PopulationData
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ContactGenerator, MatrixGenerator, ParticipantGenerator, Subgroup
from .._BRCfine import BRCfine
from ..priors import Spline2D

# Language: python

# Constants
df_age_dist = load_age_distribution("United_States")
templates = load_template_patterns("United_States")

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def generate_data():
    population = Subgroup(
        n=1500, age_dist=df_age_dist.P.values, mean_cint_margin=15.0, label="general"
    )

    matrix_gen = MatrixGenerator(templates)
    contact_matrix = matrix_gen.generate_single(population, seed=42)

    part_gen = ParticipantGenerator(population)
    df_part = part_gen.generate(seed=42)

    cnt_gen = ContactGenerator(df_part, cint_matrices=contact_matrix, model="poisson")
    df_cnt = cnt_gen.generate(seed=42)

    part_data = ParticipantData(df_part, id_col="id", age_col="age_group")
    cnt_data = ContactData(df_cnt, id_col="id", age_col="age_cnt")
    pop_data = PopulationData(df_age_dist, age_col="age", size_col="P")

    dataloader = DataLoader(part_data, cnt_data, pop_data)

    return dataloader


@pytest.fixture
def generate_data_with_repeats():
    population = Subgroup(
        n=1500, age_dist=df_age_dist.P.values, mean_cint_margin=15.0, label="general"
    )

    matrix_gen = MatrixGenerator(templates)
    contact_matrix = matrix_gen.generate_single(population, seed=42)

    part_gen = ParticipantGenerator(population)
    df_part = part_gen.generate(seed=42)
    df_part["repeat"] = np.random.randint(0, 3, size=df_part.shape[0])  # 3 repeat max

    cnt_gen = ContactGenerator(df_part, cint_matrices=contact_matrix, model="poisson")
    df_cnt = cnt_gen.generate(seed=42)

    part_data = ParticipantData(
        df_part, id_col="id", age_col="age_group", repeat_col="repeat"
    )
    cnt_data = ContactData(df_cnt, id_col="id", age_col="age_cnt")
    pop_data = PopulationData(df_age_dist, age_col="age", size_col="P")

    dataloader = DataLoader(part_data, cnt_data, pop_data)

    return dataloader


# Test initialization
class TestInit:

    def test_basic_init(self, generate_data):
        dataloader = generate_data
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        assert len(model.y) > 0
        assert len(model.aid) == len(model.y)
        assert len(model.bid) == len(model.y)
        assert len(model.log_N) > 0
        assert model.log_P.shape[1] == model.A
        assert model.log_S.shape[0] == len(model.y)

        model = BRCfine(dataloader, priors, likelihood="negbin", inv_odist=2.0)
        assert model.inv_odist == 2.0
        assert model.likelihood == "negbin"

    def test_init_with_rid(self, generate_data_with_repeats):
        dataloader = generate_data_with_repeats

        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        assert hasattr(model, "rid")
        assert len(model.rid) == len(model.y)
        assert model.hill.max_value == 2


# Test SVI inference
class TestInference:

    def test_svi_inference(self, generate_data):
        dataloader = generate_data
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        prng_key = PRNGKey(0)
        guide = AutoNormal(model.model)
        model.run_inference_svi(prng_key, guide, num_steps=1000, peak_lr=0.01)

        assert model._svi_result is not None

    # Test MCMC inference
    def test_mcmc_inference(self, generate_data):
        dataloader = generate_data
        priors = {"rate": Spline2D(prior_type="global")}
        model = BRCfine(dataloader, priors, likelihood="poisson")

        prng_key = PRNGKey(1)
        model.run_inference_mcmc(prng_key, num_warmup=10, num_samples=10, num_chains=1)

        assert model._mcmc_result is not None
