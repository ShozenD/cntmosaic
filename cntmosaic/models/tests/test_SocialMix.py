import pytest
import numpy as np
from ...datasets import load_age_distribution, load_template_patterns
from ...utils import AgeBins
from ...sim import ParticipantGenerator, MatrixGenerator, ContactGenerator, Subgroup
from .._SocialMix import SocialMix, BootstrapResults

# Language: python

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

    return df_part, df_cnt


@pytest.fixture
def generate_small_sample_data():
    population = Subgroup(
        n=50, age_dist=df_age_dist.P.values, mean_cint_margin=3.0, label="small"
    )

    matrix_gen = MatrixGenerator(templates)
    contact_matrix = matrix_gen.generate_single(population, seed=24)

    part_gen = ParticipantGenerator(population)
    df_part = part_gen.generate(seed=24)
    df_part["age_part"] = df_part["age_group"]

    cnt_gen = ContactGenerator(df_part, cint_matrices=contact_matrix, model="poisson")
    df_cnt = cnt_gen.generate(seed=24)

    return df_part, df_cnt


def test_basic_functionality(generate_contact_data):
    df_part, df_cnt = generate_contact_data

    age_bins = AgeBins(0, 80, 5)
    sm = SocialMix(df_part, df_cnt, df_age_dist, age_bins)

    # Compute contact intensity matrix
    cint_matrix = sm.compute_cint()
    assert cint_matrix.shape == (16, 16)

    # Compute contact rate matrix
    rate_matrix = sm.compute_rate()
    assert rate_matrix.shape == (16, 16)


def test_symmetrization(generate_contact_data):
    df_part, df_cnt = generate_contact_data

    age_bins = AgeBins(0, 80, 5)
    sm = SocialMix(df_part, df_cnt, df_age_dist, age_bins, symmetric=True)

    # Compute contact rate matrix
    rate_matrix = sm.compute_rate()
    assert np.allclose(rate_matrix, rate_matrix.T)


def test_bootstrap(generate_contact_data):
    df_part, df_cnt = generate_contact_data

    age_bins = AgeBins(0, 80, 5)
    sm = SocialMix(df_part, df_cnt, df_age_dist, age_bins)

    # Compute bootstrap samples
    boot_results = sm.run_bootstrap(n_boot=10)
    assert len(boot_results.intensity_samples) == 10
    for mat in boot_results.intensity_samples:
        assert mat.shape == (16, 16)

    # Compute quantiles
    cint_q, rate_q = boot_results.quantiles([0.025, 0.5, 0.975])
    assert cint_q.shape == (3, 16, 16)
    assert rate_q.shape == (3, 16, 16)

    # Compute standard deviations
    cint_std, rate_std = boot_results.std()
    assert cint_std.shape == (16, 16)
    assert rate_std.shape == (16, 16)

    # Compute means
    cint_mean, rate_mean = boot_results.mean()
    assert cint_mean.shape == (16, 16)
    assert rate_mean.shape == (16, 16)


def test_adaptive_merge(generate_small_sample_data):
    df_part, df_cnt = generate_small_sample_data

    age_bins = AgeBins(0, 80, 5)
    sm = SocialMix(
        df_part, df_cnt, df_age_dist, age_bins, adaptive_merge=True, verbose=False
    )

    # Compute contact intensity matrix
    cint_matrix = sm.compute_cint()
    assert cint_matrix.shape[0] <= 16  # Some bins should be merged

    # Compute contact rate matrix
    rate_matrix = sm.compute_rate()
    assert rate_matrix.shape[0] <= 16  # Some bins should be merged
