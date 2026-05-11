import numpy as np
import pytest

from ...datasets._base import load_template_patterns
from .._ContactGenerator import ContactGenerator
from .._MatrixGenerator import MatrixGenerator
from .._ParticipantGenerator import ParticipantGenerator
from .._PopulationConstructor import PopulationConstructor
from .._Stratification import Stratification

patterns = load_template_patterns("United_States", max_age=50)
n_ages = patterns["household"].shape[0]

# ============================
# Fixtures for ContactGenerator tests
# ============================


@pytest.fixture
def generate_single():
    """Single population case with new API."""
    ref_age_dist = np.random.rand(n_ages) * 1000
    strat = Stratification("group", 1, ref_age_dist, labels=["All"], seed=42)
    pop = PopulationConstructor(strat)

    df_part = ParticipantGenerator(pop, n_participants=1000).generate(seed=0)
    cint_matrices = MatrixGenerator(patterns).generate_single(
        pop, mean_intensity=15.0, seed=0
    )

    return df_part, cint_matrices


@pytest.fixture
def generate_partial():
    """Partial case with two strata."""
    ref_age_dist = np.random.rand(n_ages) * 1000
    region_strat = Stratification(
        "region", 2, ref_age_dist, labels=["Urban", "Rural"], seed=42
    )
    pop = PopulationConstructor(region_strat)

    df_part = ParticipantGenerator(pop, n_participants=1500).generate(seed=0)
    cint_matrices = MatrixGenerator(patterns).generate_partial(
        pop, mean_intensity=15.0, seed=0
    )

    return df_part, cint_matrices


@pytest.fixture
def generate_full():
    """Full case with two strata."""
    ref_age_dist = np.random.rand(n_ages) * 1000
    region_strat = Stratification(
        "region", 2, ref_age_dist, labels=["Urban", "Rural"], seed=42
    )
    pop = PopulationConstructor(region_strat)

    df_part = ParticipantGenerator(pop, n_participants=1500).generate(seed=0)
    cint_matrices = MatrixGenerator(patterns).generate_full(
        pop, mean_intensity=15.0, seed=0
    )

    return df_part, cint_matrices


# ============================
# Tests for ContactGenerator
# ============================
def test_single(generate_single):
    """Test single population case."""
    df_part, cint_matrices = generate_single

    cnt_gen = ContactGenerator(df_part, cint_matrices)
    df_cnt = cnt_gen.generate(seed=0)

    assert df_cnt.shape[0] > 0, "No contacts generated in single population case"
    assert df_cnt.columns.tolist() == ["id", "age_cnt", "y"]
    assert (
        df_cnt["id"].nunique() <= df_part["id"].nunique()
    ), "More unique IDs in contacts than participants"


def test_partial(generate_partial):
    """Test partial case with stratified participants."""
    df_part, cint_matrices = generate_partial

    cnt_gen = ContactGenerator(df_part, cint_matrices)
    df_cnt = cnt_gen.generate(seed=0)

    assert df_cnt.shape[0] > 0, "No contacts generated in partial case"
    assert df_cnt.columns.tolist() == ["id", "age_cnt", "y"]
    assert (
        df_cnt["id"].nunique() <= df_part["id"].nunique()
    ), "More unique IDs in contacts than participants"


def test_full(generate_full):
    """Test full case with all stratum pair interactions."""
    df_part, cint_matrices = generate_full

    cnt_gen = ContactGenerator(df_part, cint_matrices)
    df_cnt = cnt_gen.generate(seed=0)

    assert df_cnt.shape[0] > 0, "No contacts generated in full case"
    assert df_cnt.columns.tolist() == ["id", "age_cnt", "region_cnt", "y"]
    assert (
        df_cnt["id"].nunique() <= df_part["id"].nunique()
    ), "More unique IDs in contacts than participants"

    # Check that contacts include both strata
    assert set(df_cnt["region_cnt"].unique()) == {"Urban", "Rural"}
