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
    pop_constructor = PopulationConstructor(strat)
    df_pop = pop_constructor.df_P

    # Generate contact matrix
    cnt_matrix = MatrixGenerator(templates).generate_single(pop_constructor, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(pop_constructor, n_part=1500).generate(seed=42)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrix, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(df_part, id_col="id", age_col="age")
    cnt_data = ContactData(df_cnt, id_col="id", age_col="age_cnt")
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
    pop_constructor = PopulationConstructor(strat)
    df_pop = pop_constructor.df_P

    # Generate contact matrix
    cnt_matrix = MatrixGenerator(templates).generate_single(pop_constructor, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(pop_constructor, n_part=50).generate(seed=42)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrix, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(df_part, id_col="id", age_col="age")
    cnt_data = ContactData(df_cnt, id_col="id", age_col="age_cnt")
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
    pop_constructor = PopulationConstructor(strat)
    df_pop = pop_constructor.df_P

    # Generate contact matrix
    cnt_matrix = MatrixGenerator(templates).generate_single(pop_constructor, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(pop_constructor, n_part=1500).generate(seed=42)
    df_part["rid"] = np.random.choice(5, size=len(df_part))

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrix, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(df_part, id_col="id", age_col="age", repeat_col="rid")
    cnt_data = ContactData(df_cnt, id_col="id", age_col="age_cnt")
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
    pop_constructor = PopulationConstructor(strat)
    df_pop = pop_constructor.df_P
    df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["M", "F"], ordered=True)

    # Generate contact matrix
    cnt_matrices = MatrixGenerator(templates).generate_partial(pop_constructor, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(pop_constructor, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrices, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(
        df_part, id_col="id", age_col="age", strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        age_col="age_cnt",
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
    pop_constructor = PopulationConstructor(strat)
    df_pop = pop_constructor.df_P

    # Generate contact matrix
    cnt_matrices = MatrixGenerator(templates).generate_partial(pop_constructor, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(pop_constructor, n_part=200).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(
        df_part, cint_matrices=cnt_matrices, model="poisson"
    ).generate(seed=42)

    part_data = ParticipantData(
        df_part, id_col="id", age_col="age", strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        age_col="age_cnt",
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
    pop_constructor = PopulationConstructor([strat_sex, strat_ses])
    df_pop = pop_constructor.df_P

    # Generate contact matrices
    cint_matrices = MatrixGenerator(templates).generate_partial(
        pop_constructor, seed=42
    )

    # Generate participants
    df_part = ParticipantGenerator(pop_constructor, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)
    df_part["ses"] = pd.Categorical(
        df_part["ses"], categories=["Low", "High"], ordered=True
    )

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cint_matrices).generate(seed=42)

    part_data = ParticipantData(
        df_part, id_col="id", age_col="age", strat_var_cols=["sex", "ses"]
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        age_col="age_cnt",
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
    pop_constructor = PopulationConstructor(strat)
    df_pop = pop_constructor.df_P

    # Generate contact matrices
    cnt_matrices = MatrixGenerator(templates).generate_full(pop_constructor, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(pop_constructor, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["F", "M"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices).generate(seed=42)
    df_cnt["sex_cnt"] = pd.Categorical(
        df_cnt["sex_cnt"], categories=["F", "M"], ordered=True
    )

    part_data = ParticipantData(
        df_part, id_col="id", age_col="age", strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        age_col="age_cnt",
        strat_var_cols="sex_cnt",
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
    pop_constructor = PopulationConstructor(strat)
    df_pop = pop_constructor.df_P

    # Generate contact matrices
    cnt_matrices = MatrixGenerator(templates).generate_full(pop_constructor, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(pop_constructor, n_part=N).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["F", "M"], ordered=True)

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices).generate(seed=42)
    df_cnt["sex_cnt"] = pd.Categorical(
        df_cnt["sex_cnt"], categories=["F", "M"], ordered=True
    )

    part_data = ParticipantData(
        df_part, id_col="id", age_col="age", strat_var_cols="sex"
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        age_col="age_cnt",
        strat_var_cols="sex_cnt",
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
    pop_constructor = PopulationConstructor([strat_sex, strat_ses])
    df_pop = pop_constructor.df_P

    # Generate contact matrices
    cnt_matrices = MatrixGenerator(templates).generate_full(pop_constructor, seed=42)

    # Generate participants
    df_part = ParticipantGenerator(pop_constructor, n_part=1500).generate(seed=42)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["F", "M"], ordered=True)
    df_part["ses"] = pd.Categorical(
        df_part["ses"], categories=["High", "Low"], ordered=True
    )

    # Generate contacts
    df_cnt = ContactGenerator(df_part, cint_matrices=cnt_matrices).generate(seed=42)
    df_cnt["sex_cnt"] = pd.Categorical(
        df_cnt["sex_cnt"], categories=["F", "M"], ordered=True
    )
    df_cnt["ses_cnt"] = pd.Categorical(
        df_cnt["ses_cnt"], categories=["High", "Low"], ordered=True
    )

    part_data = ParticipantData(
        df_part, id_col="id", age_col="age", strat_var_cols=["sex", "ses"]
    )
    cnt_data = ContactData(
        df_cnt,
        id_col="id",
        age_col="age_cnt",
        strat_var_cols=["sex_cnt", "ses_cnt"],
    )
    pop_data = PopulationData(
        df_pop, age_col="age", size_col="P", strat_var_cols=["sex", "ses"]
    )

    return part_data, cnt_data, pop_data
