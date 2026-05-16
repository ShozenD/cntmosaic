import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df_part_one_year() -> pd.DataFrame:
    """
    Participant data with one-year age groups.
    """

    df_part = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 35, 45, 55, 65],
            "sex": ["M", "F", "M", "F", "M"],
            "hhsize": ["1", "2", "3", "4", "5+"],
            "repeat": [0, 1, 0, 1, 2],
            "amb_cnt": [1, 2, 3, 4, 5],
        }
    )

    return df_part


@pytest.fixture
def df_part_age_min_max() -> pd.DataFrame:
    """
    Participant data with age_min and age_max columns.
    """

    df_part = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age_min": [20, 30, 40, 50, 60],
            "age_max": [29, 39, 49, 59, 69],
        }
    )
    return df_part


@pytest.fixture
def df_part_age_grps() -> pd.DataFrame:
    """
    Participant data with age groups.
    """

    df_part = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age_grp": pd.IntervalIndex.from_tuples(
                [(0, 5), (5, 10), (10, 15), (15, 20), (20, 35)]
            ),
            "sex": ["M", "F", "M", "F", "M"],
            "hhsize": ["1", "2", "3", "4", "5+"],
            "repeat": [0, 1, 0, 1, 2],
            "amb_cnt": [1, 2, 3, 4, 5],
        }
    )
    df_part["age_grp"] = df_part["age_grp"].astype("category")

    return df_part


@pytest.fixture
def df_cnt_one_year() -> pd.DataFrame:
    """
    Contact data with one-year age groups.
    """

    df_cnt = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 4, 5],
            "age_cnt": [30, 40, 50, 60, 70, 80],
            "sex_cnt": ["M", "F", "M", "F", "M", "F"],
            "hhsize_cnt": ["1", "2", "3", "4", "5+", "1"],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )

    return df_cnt


@pytest.fixture
def df_cnt_age_min_max() -> pd.DataFrame:
    """
    Contact data with age_min and age_max columns.
    """
    df_cnt = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 4, 5],
            "age_min": [20, 30, 40, 50, 60, 70],
            "age_max": [29, 39, 49, 59, 69, 79],
        }
    )
    return df_cnt


@pytest.fixture
def df_cnt_age_grps() -> pd.DataFrame:
    """
    Contact data with age groups.
    """

    df_cnt = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 4, 5],
            "age_grp_cnt": pd.IntervalIndex.from_tuples(
                [(0, 5), (5, 10), (10, 15), (15, 20), (20, 35), (35, 50)]
            ),
            "sex": ["M", "F", "M", "F", "M", "F"],
            "hhsize": ["1", "2", "3", "4", "5+", "1"],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )
    df_cnt["age_grp_cnt"] = df_cnt["age_grp_cnt"].astype("category")

    return df_cnt


@pytest.fixture
def df_strat_age_min_max() -> pd.DataFrame:
    """
    Stratification data using age_min/age_max columns instead of age_col.
    """
    df = pd.DataFrame(
        {
            "age_min": [0, 0, 5, 5, 10, 10],
            "age_max": [4, 4, 9, 9, 14, 14],
            "stratum": ["A", "B", "A", "B", "A", "B"],
            "prop": [0.3, 0.7, 0.4, 0.6, 0.2, 0.8],
        }
    )
    df["stratum"] = pd.Categorical(df["stratum"], categories=["A", "B"])
    return df


@pytest.fixture
def df_strat_single_var() -> pd.DataFrame:
    """
    Stratification data with a single stratification variable.
    """
    df = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],  # Three age groups: 0, 1, 2
            "stratum": ["A", "B", "A", "B", "A", "B"],  # Two strata: A, B
            "prop": [0.3, 0.7, 0.4, 0.6, 0.2, 0.8],  # Proportions
        }
    )
    df["stratum"] = pd.Categorical(df["stratum"], categories=["A", "B"])

    return df


@pytest.fixture
def df_strat_multi_var():
    # Create full cartesian product of all categories
    ages = [0, 1, 2]
    stratum1_cats = ["A", "B"]
    stratum2_cats = ["X", "Y", "Z"]

    df = pd.DataFrame(
        {
            "age": np.repeat(ages, len(stratum1_cats) * len(stratum2_cats)),
            "stratum1": np.tile(
                np.repeat(stratum1_cats, len(stratum2_cats)), len(ages)
            ),
            "stratum2": np.tile(stratum2_cats, len(ages) * len(stratum1_cats)),
            "prop": np.array(
                [
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],  # (s1=A, s2=X)
                    [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],  # (s1=A, s2=Y)
                    [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],  # (s1=A, s2=Z)
                ]
            ).flatten(),  # Random proportions for testing
        }
    )
    # Normalize proportions to sum to 1 within each age group
    df["prop"] = df.groupby("age")["prop"].transform(lambda x: x / x.sum())
    df["stratum1"] = pd.Categorical(df["stratum1"], categories=stratum1_cats)
    df["stratum2"] = pd.Categorical(df["stratum2"], categories=stratum2_cats)

    return df


@pytest.fixture
def df_strat_count():
    df = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],  # Three age groups: 0, 1, 2
            "stratum": ["A", "B", "A", "B", "A", "B"],  # Two strata: A, B
            "count": [30, 70, 40, 60, 20, 80],  # Counts
        }
    )
    return df


@pytest.fixture
def df_pop_age_min_max() -> pd.DataFrame:
    """
    Population data with age_min and age_max columns.
    """
    df = pd.DataFrame(
        {
            "age_min": [0, 5, 10, 15],
            "age_max": [4, 9, 14, 19],
            "population": [5000, 4800, 4600, 4400],
        }
    )
    return df


@pytest.fixture
def df_pop_basic():
    df = pd.DataFrame({"age": [0, 1, 2, 3], "P": [1000, 1100, 1200, 1150]})
    return df


@pytest.fixture
def df_pop_single_var():
    df = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "P": [510, 490, 530, 520, 550, 550],
        }
    )
    df["sex"] = pd.Categorical(df["sex"], categories=["F", "M"])

    return df


@pytest.fixture
def df_pop_multi_var():
    df = pd.DataFrame(
        {
            "age": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "sex": ["M", "M", "F", "F", "M", "M", "F", "F", "M", "M", "F", "F"],
            "hhsize": ["1", "2", "1", "2", "1", "2", "1", "2", "1", "2", "1", "2"],
            "P": [250, 240, 260, 250, 270, 230, 280, 220, 290, 210, 300, 200],
        }
    )
    df["sex"] = pd.Categorical(df["sex"], categories=["F", "M"])
    df["hhsize"] = pd.Categorical(df["hhsize"], categories=["1", "2"])

    return df


@pytest.fixture
def df_pop_age_grps():
    df = pd.DataFrame(
        {
            "age": [0, 5, 10],
            "age_grp": pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
            "P": [5000, 4800, 4600],
        }
    )
    df["age_grp"] = df["age_grp"].astype("category")

    return df
