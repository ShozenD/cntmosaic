"""
This modules contains common test fixture for testing various models.
"""

import numpy as np
import pandas as pd
import pytest

from ...dataloader import ContactData, ParticipantData, PopulationData
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import (
    ContactGenerator,
    MatrixGenerator,
    ParticipantGenerator,
    PopulationConstructor,
    Stratification,
)

df_age_dist = load_age_distribution("United_States")
templates = load_template_patterns("United_States")


@pytest.fixture
def single_large_sample():
    """Generate a large sample with single stratification (no stratification)."""
    # Define stratification
    strat = Stratification(
        name="general", n_strata=1, labels=["All"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P

    # Generate contact matrix
    cnt_matrix = MatrixGenerator(templates).generate_single(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrix, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(df_part, id_col="id", part_age_col="age")
    cnt_data = ContactData(df_cnt, id_col="id", cnt_age_col="cnt_age")
    pop_data = PopulationData(df_pop, age_col="age", size_col="P")

    return part_data, cnt_data, pop_data


@pytest.fixture
def single_coarse_large_sample():
    """Generate a large sample with single stratification (no stratification)."""
    # Define stratification
    strat = Stratification(
        name="general", n_strata=1, labels=["All"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P
    df_pop.drop(columns="general", inplace=True)

    # Generate contact matrix
    cnt_matrix = MatrixGenerator(templates).generate_single(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrix, model="poisson"
    ).generate(seed=42)

    # Simulate coarse age group reporting
    df_cnt["cnt_age_grp"] = pd.cut(
        df_cnt["cnt_age"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80],
        right=False,
    )
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    part_data = ParticipantData(df_part, id_col="id", part_age_col="age")
    cnt_data = ContactData(df_cnt, id_col="id", cnt_age_grp_col="cnt_age_grp")
    pop_data = PopulationData(df_pop, age_col="age", size_col="P")

    return part_data, cnt_data, pop_data


@pytest.fixture
def single_small_sample():
    """Generate a small sample with single stratification (no stratification)."""
    # Define stratification
    strat = Stratification(
        name="general", n_strata=1, labels=["All"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P

    # Generate contact matrix
    cnt_matrix = MatrixGenerator(templates).generate_single(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=50).generate(seed=42)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrix, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(df_part, id_col="id", part_age_col="age")
    cnt_data = ContactData(df_cnt, id_col="id", cnt_age_col="cnt_age")
    pop_data = PopulationData(df_pop, age_col="age", size_col="P")

    return part_data, cnt_data, pop_data


@pytest.fixture
def single_coarse_coarse_small():
    """Generate a small sample with single stratification (no stratification)."""
    # Define stratification
    strat = Stratification(
        name="general", n_strata=1, labels=["All"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P
    df_pop.drop(columns="general", inplace=True)
    df_pop["pop_age_grp"] = pd.cut(
        df_pop["age"],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
        right=False,
    )
    df_pop.dropna(subset=["pop_age_grp"], inplace=True)
    df_pop = df_pop.groupby("pop_age_grp", observed=False)["P"].sum().reset_index()

    # Generate contact matrix
    cnt_matrix = MatrixGenerator(templates).generate_single(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=50).generate(seed=42)

    df_part["part_age_grp"] = pd.cut(
        df_part["age"],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
        right=False,
    )
    df_part.dropna(subset=["part_age_grp"], inplace=True)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrix, model="poisson"
    ).generate(seed=42)

    # Simulate coarse age group reporting
    df_cnt["cnt_age_grp"] = pd.cut(
        df_cnt["cnt_age"],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
        right=False,
    )
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    part_data = ParticipantData(df_part, id_col="id", part_age_grp_col="part_age_grp")
    cnt_data = ContactData(df_cnt, id_col="id", cnt_age_grp_col="cnt_age_grp")
    pop_data = PopulationData(df_pop, pop_age_grp_col="pop_age_grp", size_col="P")

    return part_data, cnt_data, pop_data


@pytest.fixture
def single_coarse_large_sample_with_repeats():
    """Generate a large sample with single stratification (no stratification)."""
    # Define stratification
    strat = Stratification(
        name="general", n_strata=1, labels=["All"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P
    df_pop.drop(columns="general", inplace=True)

    # Generate contact matrix
    cnt_matrix = MatrixGenerator(templates).generate_single(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["rid"] = np.random.choice(5, size=len(df_part))

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrix, model="poisson"
    ).generate(seed=42)

    # Simulate coarse age group reporting
    df_cnt["cnt_age_grp"] = pd.cut(
        df_cnt["cnt_age"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80],
        right=False,
    )
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    part_data = ParticipantData(df_part, id_col="id", part_age_col="age", part_repeat_col="rid")
    cnt_data = ContactData(df_cnt, id_col="id", cnt_age_grp_col="cnt_age_grp")
    pop_data = PopulationData(df_pop, age_col="age", size_col="P")

    return part_data, cnt_data, pop_data


@pytest.fixture
def single_large_sample_with_repeats():
    """Generate a large sample with single stratification (no stratification)."""
    # Define stratification
    strat = Stratification(
        name="general", n_strata=1, labels=["All"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P

    # Generate contact matrix
    cnt_matrix = MatrixGenerator(templates).generate_single(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["rid"] = np.random.choice(5, size=len(df_part))

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrix, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(df_part, id_col="id", part_age_col="age", part_repeat_col="rid")
    cnt_data = ContactData(df_cnt, id_col="id", cnt_age_col="cnt_age")
    pop_data = PopulationData(df_pop, age_col="age", size_col="P")

    return part_data, cnt_data, pop_data


@pytest.fixture
def partial_large_sample():
    """Generate a large sample with partial stratification."""

    # Define stratification
    strat = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P
    df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["M", "F"], ordered=True)

    # Generate contact matrix
    cnt_matrices = MatrixGenerator(templates).generate_partial(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrices, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_col="cnt_age",
    )
    pop_data = PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols="sex")

    return part_data, cnt_data, pop_data


@pytest.fixture
def partial_coarse_large_sample():
    """Generate a large sample with partial stratification."""

    # Define stratification
    strat = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P
    df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["M", "F"], ordered=True)

    # Generate contact matrix
    cnt_matrices = MatrixGenerator(templates).generate_partial(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrices, model="poisson"
    ).generate(seed=42)
    df_cnt["cnt_age_grp"] = pd.cut(
        df_cnt["cnt_age"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80],
        right=False,
    )
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_grp_col="cnt_age_grp",
    )
    pop_data = PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols="sex")

    return part_data, cnt_data, pop_data


@pytest.fixture
def partial_small_sample():
    """Generate a small sample with partial stratification."""

    # Define stratification
    strat = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P

    # Generate contact matrix
    cnt_matrices = MatrixGenerator(templates).generate_partial(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=200).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrices, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_col="cnt_age",
    )
    pop_data = PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols="sex")

    return part_data, cnt_data, pop_data


@pytest.fixture
def partial_coarse_small_sample():
    """Generate a small sample with partial stratification."""

    # Define stratification
    strat = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P

    # Generate contact matrix
    cnt_matrices = MatrixGenerator(templates).generate_partial(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=200).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrices, model="poisson"
    ).generate(seed=42)
    df_cnt["cnt_age_grp"] = pd.cut(
        df_cnt["cnt_age"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80],
        right=False,
    )
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_grp_col="cnt_age_grp",
    )
    pop_data = PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols="sex")

    return part_data, cnt_data, pop_data


@pytest.fixture
def partial_multi_strat_large_sample():
    """Generate a large sample with partial stratification on multiple variables."""

    # Define stratifications
    strat_sex = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )
    strat_ses = Stratification(
        name="ses",
        n_strata=2,
        labels=["Low", "High"],
        ref_age_dist=df_age_dist.P.values,
    )

    # Construct population
    popcon = PopulationConstructor([strat_sex, strat_ses])
    df_pop = popcon.df_P

    # Generate contact matrices
    cint_matrices = MatrixGenerator(templates).generate_partial(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)
    df_part["ses"] = pd.Categorical(
        df_part["ses"], categories=["Low", "High"], ordered=True
    )

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cint_matrices).generate(seed=42)

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols=["sex", "ses"]
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_col="cnt_age",
    )
    pop_data = PopulationData(
        df_pop, age_col="age", size_col="P", strat_var_cols=["sex", "ses"]
    )

    return part_data, cnt_data, pop_data


@pytest.fixture
def partial_coarse_multi_strat_large_sample():
    """Generate a large sample with partial stratification on multiple variables."""

    # Define stratifications
    strat_sex = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )
    strat_ses = Stratification(
        name="ses",
        n_strata=2,
        labels=["Low", "High"],
        ref_age_dist=df_age_dist.P.values,
    )

    # Construct population
    popcon = PopulationConstructor([strat_sex, strat_ses])
    df_pop = popcon.df_P

    # Generate contact matrices
    cint_matrices = MatrixGenerator(templates).generate_partial(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)
    df_part["ses"] = pd.Categorical(
        df_part["ses"], categories=["Low", "High"], ordered=True
    )

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cint_matrices).generate(seed=42)
    df_cnt["cnt_age_grp"] = pd.cut(
        df_cnt["cnt_age"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80],
        right=False,
    )
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols=["sex", "ses"]
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_grp_col="cnt_age_grp",
    )
    pop_data = PopulationData(
        df_pop, age_col="age", size_col="P", strat_var_cols=["sex", "ses"]
    )

    return part_data, cnt_data, pop_data


@pytest.fixture
def full_large_sample():
    """Generate a large sample with full stratification."""

    # Define stratification
    strat = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P

    # Generate contact matrices
    cnt_matrices = MatrixGenerator(templates).generate_full(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["F", "M"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices).generate(seed=42)
    df_cnt["cnt_sex"] = pd.Categorical(
        df_cnt["cnt_sex"], categories=["F", "M"], ordered=True
    )

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_col="cnt_age",
        cnt_strat_var_cols="cnt_sex",
    )
    pop_data = PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols="sex")

    return part_data, cnt_data, pop_data


@pytest.fixture
def full_coarse_large_sample():
    """Generate a large sample with full stratification."""

    # Define stratification
    strat = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P

    # Generate contact matrices
    cnt_matrices = MatrixGenerator(templates).generate_full(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["F", "M"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices).generate(seed=42)
    df_cnt["cnt_sex"] = pd.Categorical(
        df_cnt["cnt_sex"], categories=["F", "M"], ordered=True
    )
    df_cnt["cnt_age_grp"] = pd.cut(
        df_cnt["cnt_age"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80],
        right=False,
    )
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_grp_col="cnt_age_grp",
        cnt_strat_var_cols="cnt_sex",
    )
    pop_data = PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols="sex")

    return part_data, cnt_data, pop_data


@pytest.fixture
def full_small_sample():
    """Generate a small sample with full stratification."""
    N = 200

    # Define stratification
    strat = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )

    # Construct population
    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P

    # Generate contact matrices
    cnt_matrices = MatrixGenerator(templates).generate_full(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=N).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["F", "M"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices).generate(seed=42)
    df_cnt["cnt_sex"] = pd.Categorical(
        df_cnt["cnt_sex"], categories=["F", "M"], ordered=True
    )

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_col="cnt_age",
        cnt_strat_var_cols="cnt_sex",
    )
    pop_data = PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols="sex")

    return part_data, cnt_data, pop_data


@pytest.fixture
def full_multi_strat_large_sample():
    """Generate a large sample with full stratification on multiple variables."""

    # Define stratifications
    strat_sex = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )
    strat_ses = Stratification(
        name="ses",
        n_strata=2,
        labels=["Low", "High"],
        ref_age_dist=df_age_dist.P.values,
    )

    # Construct population
    popcon = PopulationConstructor([strat_sex, strat_ses])
    df_pop = popcon.df_P

    # Generate contact matrices
    cnt_matrices = MatrixGenerator(templates).generate_full(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["F", "M"], ordered=True)
    df_part["ses"] = pd.Categorical(
        df_part["ses"], categories=["High", "Low"], ordered=True
    )

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices).generate(seed=42)
    df_cnt["cnt_sex"] = pd.Categorical(
        df_cnt["cnt_sex"], categories=["F", "M"], ordered=True
    )
    df_cnt["cnt_ses"] = pd.Categorical(
        df_cnt["cnt_ses"], categories=["High", "Low"], ordered=True
    )

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols=["sex", "ses"]
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_col="cnt_age",
        cnt_strat_var_cols=["cnt_sex", "cnt_ses"],
    )
    pop_data = PopulationData(
        df_pop, age_col="age", size_col="P", strat_var_cols=["sex", "ses"]
    )

    return part_data, cnt_data, pop_data


# ---------------------------------------------------------------------------
# Coarse-coarse fixtures with stratification (for GenMixCC)
# Both participant and contact ages are recorded as coarse age groups.
# ---------------------------------------------------------------------------

_CC_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80]


@pytest.fixture
def partial_coarse_coarse_large_sample():
    """Generate a large sample with partial stratification, coarse ages for both participant and contact."""

    strat = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )

    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P
    df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["M", "F"], ordered=True)

    cnt_matrices = MatrixGenerator(templates).generate_partial(popcon, seed=42)

    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)
    df_part["part_age_grp"] = pd.cut(df_part["age"], bins=_CC_BINS, right=False)
    df_part.dropna(subset=["part_age_grp"], inplace=True)

    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices, model="poisson").generate(seed=42)
    df_cnt["cnt_age_grp"] = pd.cut(df_cnt["cnt_age"], bins=_CC_BINS, right=False)
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    df_pop["pop_age_grp"] = pd.cut(df_pop["age"], bins=_CC_BINS, right=False)
    df_pop.dropna(subset=["pop_age_grp"], inplace=True)
    df_pop_coarse = (
        df_pop.groupby(["pop_age_grp", "sex"], observed=False)["P"].sum().reset_index()
    )

    part_data = ParticipantData(
        df_part, id_col="id", part_age_grp_col="part_age_grp", part_strat_var_cols="sex"
    )
    cnt_data = ContactData(df_cnt, id_col="id", cnt_age_grp_col="cnt_age_grp")
    pop_data = PopulationData(
        df_pop_coarse, pop_age_grp_col="pop_age_grp", size_col="P", strat_var_cols="sex"
    )

    return part_data, cnt_data, pop_data


@pytest.fixture
def full_coarse_coarse_large_sample():
    """Generate a large sample with full stratification, coarse ages for both participant and contact."""

    strat = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )

    popcon = PopulationConstructor(strat)
    df_pop = popcon.df_P
    df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["F", "M"], ordered=True)

    cnt_matrices = MatrixGenerator(templates).generate_full(popcon, seed=42)

    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["F", "M"], ordered=True)
    df_part["part_age_grp"] = pd.cut(df_part["age"], bins=_CC_BINS, right=False)
    df_part.dropna(subset=["part_age_grp"], inplace=True)

    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices).generate(seed=42)
    df_cnt["cnt_sex"] = pd.Categorical(
        df_cnt["cnt_sex"], categories=["F", "M"], ordered=True
    )
    df_cnt["cnt_age_grp"] = pd.cut(df_cnt["cnt_age"], bins=_CC_BINS, right=False)
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    df_pop["pop_age_grp"] = pd.cut(df_pop["age"], bins=_CC_BINS, right=False)
    df_pop.dropna(subset=["pop_age_grp"], inplace=True)
    df_pop_coarse = (
        df_pop.groupby(["pop_age_grp", "sex"], observed=False)["P"].sum().reset_index()
    )

    part_data = ParticipantData(
        df_part, id_col="id", part_age_grp_col="part_age_grp", part_strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_grp_col="cnt_age_grp",
        cnt_strat_var_cols="cnt_sex",
    )
    pop_data = PopulationData(
        df_pop_coarse, pop_age_grp_col="pop_age_grp", size_col="P", strat_var_cols="sex"
    )

    return part_data, cnt_data, pop_data


@pytest.fixture
def full_coarse_multi_strat_large_sample():
    """Generate a large sample with full stratification on multiple variables."""

    # Define stratifications
    strat_sex = Stratification(
        name="sex", n_strata=2, labels=["M", "F"], ref_age_dist=df_age_dist.P.values
    )
    strat_ses = Stratification(
        name="ses",
        n_strata=2,
        labels=["Low", "High"],
        ref_age_dist=df_age_dist.P.values,
    )

    # Construct population
    popcon = PopulationConstructor([strat_sex, strat_ses])
    df_pop = popcon.df_P

    # Generate contact matrices
    cnt_matrices = MatrixGenerator(templates).generate_full(popcon, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(popcon, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["F", "M"], ordered=True)
    df_part["ses"] = pd.Categorical(
        df_part["ses"], categories=["High", "Low"], ordered=True
    )

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices).generate(seed=42)
    df_cnt["cnt_sex"] = pd.Categorical(
        df_cnt["cnt_sex"], categories=["F", "M"], ordered=True
    )
    df_cnt["cnt_ses"] = pd.Categorical(
        df_cnt["cnt_ses"], categories=["High", "Low"], ordered=True
    )
    df_cnt["cnt_age_grp"] = pd.cut(
        df_cnt["cnt_age"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80],
        right=False,
    )
    df_cnt.drop(columns="cnt_age", inplace=True)
    df_cnt.dropna(subset=["cnt_age_grp"], inplace=True)

    part_data = ParticipantData(
        df_part, id_col="id", part_age_col="age", part_strat_var_cols=["sex", "ses"]
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        cnt_age_grp_col="cnt_age_grp",
        cnt_strat_var_cols=["cnt_sex", "cnt_ses"],
    )
    pop_data = PopulationData(
        df_pop, age_col="age", size_col="P", strat_var_cols=["sex", "ses"]
    )

    return part_data, cnt_data, pop_data
