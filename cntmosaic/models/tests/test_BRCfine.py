import pytest
import numpy as np

from jax.random import PRNGKey
import numpyro
from numpyro.infer.autoguide import AutoNormal

from ...datasets import load_age_distribution, load_template_patterns
from ...utils import AgeBins
from ...sim import ParticipantGenerator, MatrixGenerator, ContactGenerator, Subgroup

from ...dataloader import DataLoader, CoordToColumns
from .._BRCfine import BRCfine
from ..priors import Spline2D, PSpline2D

# Language: python

# Constants
df_age_dist = load_age_distribution("United_States")
templates = load_template_patterns("United_States")

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def generate_contact_data():
    population = Subgroup(
        n=1500, age_dist=df_age_dist.P.values, mean_cint_margin=15.0, label="general"
    )

    matrix_gen = MatrixGenerator(templates)
    contact_matrix = matrix_gen.generate_single(population, seed=42)

    part_gen = ParticipantGenerator(population)
    df_part = part_gen.generate(seed=42)
    df_part["age_part"] = df_part["age_group"]

    cnt_gen = ContactGenerator(df_part, cint_matrices=contact_matrix, model="poisson")
    df_cnt = cnt_gen.generate(seed=42)

    col_map = CoordToColumns(
        age_part="age_part", age_cnt="age_cnt", age_pop="age", size_pop="P"
    )
    dataloader = DataLoader(df_part, df_cnt, df_age_dist, col_map=col_map)

    return dataloader


# Test initialization
def test_initialization(generate_contact_data):
    dataloader = generate_contact_data
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


# Test SVI inference
def test_svi_inference(generate_contact_data):
    dataloader = generate_contact_data
    priors = {"rate": Spline2D(prior_type="global")}
    model = BRCfine(dataloader, priors, likelihood="poisson")

    guide = AutoNormal(model.model)
    prng_key = PRNGKey(0)
    model.run_inference_svi(prng_key, guide, num_steps=1000, peak_lr=0.01)

    assert model._svi_result is not None


# Test MCMC inference
def test_mcmc_inference(generate_contact_data):
    dataloader = generate_contact_data
    priors = {"rate": Spline2D(prior_type="global")}
    model = BRCfine(dataloader, priors, likelihood="poisson")

    prng_key = PRNGKey(1)
    model.run_inference_mcmc(prng_key, num_warmup=10, num_samples=10, num_chains=1)

    assert model._mcmc_result is not None
