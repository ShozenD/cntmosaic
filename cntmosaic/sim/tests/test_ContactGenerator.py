import pytest

from ...datasets._base import load_age_distribution, load_template_patterns
from .._ContactGenerator import ContactGenerator
from .._MatrixGenerator import MatrixGenerator
from .._ParticipantGenerator import ParticipantGenerator, Subgroup

df_age_dist = load_age_distribution("United_States", max_age=10)
patterns = load_template_patterns("United_States", max_age=10)

# ============================
# Fixtures for ContactGenerator tests
# ============================


@pytest.fixture
def generate_single():
    subgroup = Subgroup(n=1000, age_dist=df_age_dist["P"].values, mean_cint_margin=15.0)
    df_part = ParticipantGenerator(subgroup).generate(seed=0)
    cint_matrix = MatrixGenerator(patterns).generate_single(subgroup, seed=0)

    return df_part, cint_matrix


@pytest.fixture
def generate_partial():
    subgroups = [
        Subgroup(
            n=1000, age_dist=df_age_dist["P"].values, mean_cint_margin=15.0, label="0"
        ),
        Subgroup(
            n=2000, age_dist=df_age_dist["P"].values, mean_cint_margin=20.0, label="1"
        ),
    ]
    df_part = ParticipantGenerator(subgroups).generate(seed=0)
    cint_matrices = MatrixGenerator(patterns).generate_partial(subgroups, seed=0)

    return df_part, cint_matrices


@pytest.fixture
def generate_full():
    subgroups = [
        Subgroup(
            n=1000, age_dist=df_age_dist["P"].values, mean_cint_margin=15.0, label="0"
        ),
        Subgroup(
            n=2000, age_dist=df_age_dist["P"].values, mean_cint_margin=20.0, label="1"
        ),
    ]
    df_part = ParticipantGenerator(subgroups).generate(seed=0)
    cint_matrices = MatrixGenerator(patterns).generate_full(subgroups, seed=0)

    return df_part, cint_matrices


# ============================
# Tests for ContactGenerator
# ============================
def test_single(generate_single):
    df_part, cint_matrix = generate_single

    cnt_gen = ContactGenerator(df_part, cint_matrix)
    df_cnt = cnt_gen.generate(seed=0)

    assert df_cnt.shape[0] > 0, "No contacts generated in single subgroup case"
    assert df_cnt.columns.tolist() == ["id", "age_cnt", "y"]
    assert (
        df_cnt["id"].nunique() <= df_part["id"].nunique()
    ), "More unique IDs in contacts than participants"


def test_partial(generate_partial):
    df_part, cint_matrices = generate_partial

    cnt_gen = ContactGenerator(df_part, cint_matrices)
    df_cnt = cnt_gen.generate(seed=0)

    assert df_cnt.shape[0] > 0, "No contacts generated in partial subgroup case"
    assert df_cnt.columns.tolist() == ["id", "age_cnt", "y"]
    assert (
        df_cnt["id"].nunique() <= df_part["id"].nunique()
    ), "More unique IDs in contacts than participants"


def test_full(generate_full):
    df_part, cint_matrices = generate_full

    cnt_gen = ContactGenerator(df_part, cint_matrices)
    df_cnt = cnt_gen.generate(seed=0)

    assert df_cnt.shape[0] > 0, "No contacts generated in full subgroup case"
    assert df_cnt.columns.tolist() == ["id", "age_cnt", "subgroup_cnt", "y"]
    assert (
        df_cnt["id"].nunique() <= df_part["id"].nunique()
    ), "More unique IDs in contacts than participants"
