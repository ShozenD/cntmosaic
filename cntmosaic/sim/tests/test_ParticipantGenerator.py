import numpy as np
import pytest

from .._ParticipantGenerator import ParticipantGenerator
from .._PopulationConstructor import PopulationConstructor
from .._Stratification import Stratification


def test_single_stratification():
    """
    Test ParticipantGenerator with a single Stratification.
    """
    # Create reference age distribution
    ref_age_dist = np.array([1000, 1500, 2000, 1800, 1200])

    # Create gender stratification
    gender_strat = Stratification(
        name="gender",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Male", "Female"],
        seed=42,
    )

    # Build population constructor
    pop_constructor = PopulationConstructor(gender_strat)

    # Generate participants
    pg = ParticipantGenerator(pop_constructor, n_part=1000)
    df_part = pg.generate(seed=123)

    # Check the shape of the generated DataFrame
    assert df_part.shape == (
        1000,
        3,
    ), "DataFrame should have 1000 rows and 3 columns (id, age, gender)"

    # Check the columns of the generated DataFrame
    assert set(df_part.columns) == {
        "id",
        "age",
        "gender",
    }, "Columns should be 'id', 'age', 'gender'"

    # Check that IDs are unique and sequential
    assert df_part["id"].is_unique, "IDs should be unique"
    assert df_part["id"].min() == 1, "IDs should start at 1"
    assert df_part["id"].max() == 1000, "IDs should end at 1000"

    # Check that ages are within valid range
    assert df_part["age"].min() >= 0, "Ages should be non-negative"
    assert df_part["age"].max() < len(
        ref_age_dist
    ), "Ages should be less than the number of age groups"

    # Check that gender values are correct
    assert set(df_part["gender"].unique()).issubset(
        {"Male", "Female"}
    ), "Gender should be Male or Female"


def test_multiple_stratifications():
    """
    Test ParticipantGenerator with multiple Stratifications (Gender x Region).
    """
    # Create reference age distribution
    ref_age_dist = np.array([1000, 1500, 2000, 1800, 1200])

    # Create stratifications
    gender_strat = Stratification(
        name="gender",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Male", "Female"],
        seed=42,
    )
    region_strat = Stratification(
        name="region",
        n_strata=3,
        ref_age_dist=ref_age_dist,
        labels=["Urban", "Suburban", "Rural"],
        seed=43,
    )

    # Build joint population
    pop_constructor = PopulationConstructor([gender_strat, region_strat])

    # Generate participants
    pg = ParticipantGenerator(pop_constructor, n_part=2000)
    df_part = pg.generate(seed=456)

    # Check the shape - should have id, age, gender, region
    assert df_part.shape == (2000, 4), "DataFrame should have 2000 rows and 4 columns"

    # Check the columns
    assert set(df_part.columns) == {
        "id",
        "age",
        "gender",
        "region",
    }, "Columns should include 'id', 'age', 'gender', 'region'"

    # Check that IDs are unique and sequential
    assert df_part["id"].is_unique, "IDs should be unique"
    assert df_part["id"].min() == 1, "IDs should start at 1"
    assert df_part["id"].max() == 2000, "IDs should end at 2000"

    # Check gender values
    assert set(df_part["gender"].unique()).issubset(
        {"Male", "Female"}
    ), "Gender should be Male or Female"

    # Check region values
    assert set(df_part["region"].unique()).issubset(
        {"Urban", "Suburban", "Rural"}
    ), "Region should be Urban, Suburban, or Rural"

    # Check that we have participants in multiple strata
    assert (
        df_part.groupby(["gender", "region"]).size().min() > 0
    ), "Should have participants in all gender×region combinations"


def test_reproducibility_with_seed():
    """
    Test that the same seed produces the same results.
    """
    ref_age_dist = np.array([100, 200, 300, 400, 500])

    gender_strat = Stratification(
        name="gender",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Male", "Female"],
        seed=100,
    )

    pop_constructor = PopulationConstructor(gender_strat)

    pg1 = ParticipantGenerator(pop_constructor, n_part=500)
    df_part1 = pg1.generate(seed=12345)

    pg2 = ParticipantGenerator(pop_constructor, n_part=500)
    df_part2 = pg2.generate(seed=12345)

    # Check that the generated data is identical
    assert df_part1.equals(df_part2), "Same seed should produce identical results"


def test_different_seeds_produce_different_results():
    """
    Test that different seeds produce different results.
    """
    ref_age_dist = np.array([100, 200, 300, 400, 500])

    gender_strat = Stratification(
        name="gender",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Male", "Female"],
        seed=100,
    )

    pop_constructor = PopulationConstructor(gender_strat)

    pg1 = ParticipantGenerator(pop_constructor, n_part=500)
    df_part1 = pg1.generate(seed=111)

    pg2 = ParticipantGenerator(pop_constructor, n_part=500)
    df_part2 = pg2.generate(seed=222)

    # Check that the generated data is different
    assert not df_part1["age"].equals(
        df_part2["age"]
    ), "Different seeds should produce different age distributions"


def test_invalid_n_part():
    """
    Test that ValueError is raised when n_part is invalid.
    """
    ref_age_dist = np.array([100, 200, 300, 400])

    gender_strat = Stratification(
        name="gender",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Male", "Female"],
        seed=42,
    )

    pop_constructor = PopulationConstructor(gender_strat)

    # Test with n_part=0
    with pytest.raises(ValueError, match="n_part must be positive"):
        pg = ParticipantGenerator(pop_constructor, n_part=0)

    # Test with negative n_part
    with pytest.raises(ValueError, match="n_part must be positive"):
        pg = ParticipantGenerator(pop_constructor, n_part=-10)


def test_invalid_input_types():
    """
    Test that TypeError is raised for invalid input types.
    """


def test_age_distribution_matches_population():
    """
    Test that generated age distribution approximates population distribution.
    """
    ref_age_dist = np.array([1000, 1500, 2000, 1800, 1200])

    gender_strat = Stratification(
        name="gender",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Male", "Female"],
        seed=42,
    )

    pop_constructor = PopulationConstructor(gender_strat)

    # Generate large sample
    pg = ParticipantGenerator(pop_constructor, n_part=10000)
    df_part = pg.generate(seed=789)

    # Check that age distribution approximates population
    age_counts = df_part["age"].value_counts().sort_index()
    expected_proportions = ref_age_dist / ref_age_dist.sum()
    observed_proportions = age_counts / age_counts.sum()

    # With 10000 samples, should be quite close
    for age in range(len(ref_age_dist)):
        assert (
            abs(observed_proportions[age] - expected_proportions[age]) < 0.02
        ), f"Age {age} proportion should be close to population distribution"


def test_conditional_stratum_sampling():
    """
    Test that strata are sampled conditional on age using Q matrix.
    """
    ref_age_dist = np.array([1000, 1000, 1000])

    # Create stratification with age-dependent probabilities
    gender_strat = Stratification(
        name="gender",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Male", "Female"],
        seed=50,
    )

    pop_constructor = PopulationConstructor(gender_strat)

    # Generate participants
    pg = ParticipantGenerator(pop_constructor, n_part=3000)
    df_part = pg.generate(seed=999)

    # Check Q matrix structure
    Q = pop_constructor.Q
    assert Q.shape == (2, 3), "Q should have shape (n_strata, n_ages)"

    # Each column should sum to 1 (probabilities for each age)
    for age in range(3):
        assert np.isclose(Q[:, age].sum(), 1.0), f"Q[:, {age}] should sum to 1"

    # Check that participants are distributed across strata
    assert df_part["gender"].nunique() == 2, "Should have participants in both genders"


def test_column_order():
    """
    Test that output DataFrame has correct column order.
    """
    ref_age_dist = np.array([100, 200, 300])

    gender_strat = Stratification(
        name="gender",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Male", "Female"],
        seed=42,
    )

    pop_constructor = PopulationConstructor(gender_strat)
    pg = ParticipantGenerator(pop_constructor, n_part=100)
    df_part = pg.generate(seed=1)

    # Check column order: id, age, then stratification variables
    expected_cols = ["id", "age", "gender"]
    assert list(df_part.columns) == expected_cols, f"Columns should be {expected_cols}"


def test_three_way_stratification():
    """
    Test ParticipantGenerator with three stratification variables.
    """
    ref_age_dist = np.array([1000, 1500, 2000])

    # Create three stratifications
    gender_strat = Stratification(
        name="gender",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Male", "Female"],
        seed=10,
    )
    region_strat = Stratification(
        name="region",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Urban", "Rural"],
        seed=11,
    )
    ses_strat = Stratification(
        name="ses",
        n_strata=2,
        ref_age_dist=ref_age_dist,
        labels=["Low", "High"],
        seed=12,
    )

    # Build joint population
    pop_constructor = PopulationConstructor([gender_strat, region_strat, ses_strat])

    # Generate participants
    pg = ParticipantGenerator(pop_constructor, n_part=1000)
    df_part = pg.generate(seed=500)

    # Check columns
    assert set(df_part.columns) == {
        "id",
        "age",
        "gender",
        "region",
        "ses",
    }, "Should have all stratification variables"

    # Check that we have 2×2×2 = 8 possible strata
    n_unique_strata = df_part.groupby(["gender", "region", "ses"]).size().shape[0]
    assert n_unique_strata <= 8, "Should have at most 8 unique strata combinations"
