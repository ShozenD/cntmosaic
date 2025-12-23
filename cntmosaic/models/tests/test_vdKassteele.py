from typing import Dict

import pandas as pd
import pytest
from jax.random import PRNGKey
from numpy.typing import NDArray
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import ContactData, DataLoader, ParticipantData, PopulationData
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ContactGenerator, MatrixGenerator, ParticipantGenerator, Subgroup
from .._vdKassteele import vdKassteele

SEED = 42

df_age_dist: pd.DataFrame = load_age_distribution("United_States")
templates: Dict[str, NDArray] = load_template_patterns("United_States")

# ============================
# Fixtures for vdKassteele tests
# ============================


@pytest.fixture
def generate_data_partial():
    """
    Generate contact data for the partial case (multiple subgroups, incomplete contact information)
    """

    # Define subgroups
    subgroups = [
        Subgroup(
            n=300,
            age_dist=df_age_dist.P.values,
            mean_cint_margin=8,
            label="A",
        ),
        Subgroup(
            n=400,
            age_dist=df_age_dist.P.values,
            mean_cint_margin=12,
            label="B",
        ),
    ]

    # Generate participants
    part_gen = ParticipantGenerator(subgroups)
    df_part = part_gen.generate(SEED)
    df_part["subgroup"] = pd.Categorical(
        df_part["subgroup"], categories=["A", "B"], ordered=True
    )

    # Generate contact matrix
    matrix_gen = MatrixGenerator(templates)
    contact_matrices = matrix_gen.generate_partial(subgroups, SEED)

    # Generate contacts
    cnt_gen = ContactGenerator(df_part, contact_matrices)
    df_cnt = cnt_gen.generate(SEED)

    # Population size offsets
    part_data = ParticipantData(df_part, "id", "age_group", strat_var_cols="subgroup")
    cnt_data = ContactData(df_cnt, "id", "age_cnt")
    pop_data = PopulationData(df_age_dist, "age", "P")
    dataloader = DataLoader(part_data, cnt_data, pop_data)

    return dataloader


@pytest.fixture
def generate_data_full():
    """
    Generate contact data for the full case (multiple subgroups, complete contact information)
    """

    # Define subgroups
    subgroups = [
        Subgroup(
            n=300,
            age_dist=df_age_dist.P.values,
            mean_cint_margin=8,
            label="A",
        ),
        Subgroup(
            n=400,
            age_dist=df_age_dist.P.values,
            mean_cint_margin=12,
            label="B",
        ),
    ]

    # Generate participants
    part_gen = ParticipantGenerator(subgroups)
    df_part = part_gen.generate(SEED)
    df_part["subgroup"] = pd.Categorical(
        df_part["subgroup"], categories=["A", "B"], ordered=True
    )

    # Generate contact matrix
    matrix_gen = MatrixGenerator(templates)
    contact_matrices = matrix_gen.generate_full(subgroups, SEED)

    # Generate contacts
    cnt_gen = ContactGenerator(df_part, contact_matrices)
    df_cnt = cnt_gen.generate(SEED)
    df_cnt["subgroup_cnt"] = pd.Categorical(
        df_cnt["subgroup_cnt"], categories=["A", "B"], ordered=True
    )

    # Population size offsets
    df_strat_prop = pd.concat(
        [
            pd.DataFrame(
                {
                    "age": df_age_dist["age"],
                    "P": df_age_dist["P"] * 0.6,
                    "subgroup": "A",
                }
            ),
            pd.DataFrame(
                {
                    "age": df_age_dist["age"],
                    "P": df_age_dist["P"] * 0.4,
                    "subgroup": "B",
                }
            ),
        ]
    )
    df_pop_total = df_strat_prop.groupby("age")["P"].sum().reset_index()
    df_strat_prop = df_strat_prop.merge(df_pop_total, on="age", suffixes=("", "_total"))
    df_strat_prop["prop"] = df_strat_prop["P"] / df_strat_prop["P_total"]
    df_strat_prop["subgroup"] = pd.Categorical(
        df_strat_prop["subgroup"], categories=["A", "B"], ordered=True
    )

    part_data = ParticipantData(df_part, "id", "age_group", strat_var_cols="subgroup")
    cnt_data = ContactData(df_cnt, "id", "age_cnt", strat_var_cols="subgroup_cnt")
    pop_data = PopulationData(df_strat_prop, "age", "P", strat_var_cols="subgroup")
    dataloader = DataLoader(part_data, cnt_data, pop_data)

    return dataloader


class TestInit:

    def test_init_partial(self, generate_data_partial):
        dataloader = generate_data_partial

        model = vdKassteele(dataloader, "poisson")
        assert model.A == df_age_dist.shape[0]
        assert model.likelihood == "poisson"
        assert model.prior_type == "partial"

    def test_init_full(self, generate_data_full):
        dataloader = generate_data_full

        model = vdKassteele(dataloader, "poisson")
        assert model.A == df_age_dist.shape[0]
        assert model.likelihood == "poisson"
        assert model.prior_type == "full"


class TestInference:

    def test_inference_partial_svi(self, generate_data_partial):
        dataloader = generate_data_partial

        model = vdKassteele(dataloader, "poisson")
        guide = AutoNormal(model.model)

        model.run_inference_svi(prng_key=PRNGKey(SEED), guide=guide, num_steps=10)

        assert model._svi_result is not None
        assert hasattr(model._svi_result, "params")
        assert hasattr(model._svi_result, "losses")
        assert len(model._svi_result.losses) == 10

    def test_inference_full_svi(self, generate_data_full):
        dataloader = generate_data_full

        model = vdKassteele(dataloader, "poisson")
        guide = AutoNormal(model.model)

        model.run_inference_svi(prng_key=PRNGKey(SEED), guide=guide, num_steps=10)

        assert model._svi_result is not None
        assert hasattr(model._svi_result, "params")
        assert hasattr(model._svi_result, "losses")
        assert len(model._svi_result.losses) == 10
