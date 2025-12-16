import numpy as np
import pandas as pd
import pytest

from ..._types import StratMode
from ..containers._StratPropData import StratPropData


@pytest.fixture
def test_data_single_var():
    data = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],  # Three age groups: 0, 1, 2
            "stratum": ["A", "B", "A", "B", "A", "B"],  # Two strata: A, B
            "prop": [0.3, 0.7, 0.4, 0.6, 0.2, 0.8],  # Proportions
        }
    )
    data["stratum"] = pd.Categorical(data["stratum"], categories=["A", "B"])

    return data


@pytest.fixture
def test_data_multi_var():
    # Create full cartesian product of all categories
    ages = [0, 1, 2]
    stratum1_cats = ["A", "B"]
    stratum2_cats = ["X", "Y", "Z"]

    data = pd.DataFrame(
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
    data["prop"] = data.groupby("age")["prop"].transform(lambda x: x / x.sum())
    data["stratum1"] = pd.Categorical(data["stratum1"], categories=stratum1_cats)
    data["stratum2"] = pd.Categorical(data["stratum2"], categories=stratum2_cats)

    return data


@pytest.fixture
def test_data_count():
    data = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],  # Three age groups: 0, 1, 2
            "stratum": ["A", "B", "A", "B", "A", "B"],  # Two strata: A, B
            "count": [30, 70, 40, 60, 20, 80],  # Counts
        }
    )
    return data


class TestStratPropData:

    def test_partial_prop_single(self, test_data_single_var):
        strat_prop = StratPropData(
            data=test_data_single_var,
            age_col="age",
            strat_var_cols="stratum",
            prop_col="prop",
        )

        strat_modes = {"stratum": StratMode.PARTIAL}

        props = strat_prop.compute_multipliers(strat_modes=strat_modes)
        assert props.shape == (2, 3)
        np.testing.assert_allclose(
            props,
            np.array([[0.3, 0.4, 0.2], [0.7, 0.6, 0.8]]),
        )

    def test_partial_prop_multi(self, test_data_multi_var):
        strat_prop = StratPropData(
            data=test_data_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )

        strat_modes = {"stratum1": StratMode.PARTIAL, "stratum2": StratMode.PARTIAL}

        props = strat_prop.compute_multipliers(strat_modes=strat_modes)
        assert props.shape == (6, 3)
        np.testing.assert_allclose(
            props,
            np.array(
                [
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],  # (s1=A, s2=X)
                    [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],  # (s1=A, s2=Y)
                    [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],  # (s1=A, s2=Z)
                ]
            ).T,
        )

    def test_full_prop_single(self, test_data_single_var):
        strat_prop = StratPropData(
            data=test_data_single_var,
            age_col="age",
            strat_var_cols="stratum",
            prop_col="prop",
        )
        strat_modes = {"stratum": StratMode.FULL}
        props = strat_prop.compute_multipliers(strat_modes=strat_modes)
        assert props.shape == (4, 3, 3)

        expected = np.array([0.3, 0.7, 0.4, 0.6, 0.2, 0.8]).reshape((2, 3), order="F")
        expected = (expected[:, None, :, None] * expected[None, :, None, :]).reshape(
            4, 3, 3
        )
        np.testing.assert_allclose(props, expected)

    def test_full_prop_multi(self, test_data_multi_var):
        strat_prop = StratPropData(
            data=test_data_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )
        strat_modes = {"stratum1": StratMode.FULL, "stratum2": StratMode.FULL}
        props = strat_prop.compute_multipliers(strat_modes=strat_modes)
        assert props.shape == (36, 3, 3)

        prop_array = np.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],  # (age=0, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
                [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],  # (age=1, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
                [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],  # (age=2, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
            ]
        ).T  # shape (3, 6)

        expected = prop_array[:, None, :, None] * prop_array[None, :, None, :]
        expected = expected.reshape(36, 3, 3)

        np.testing.assert_allclose(props, expected)

    def test_mixed_prop_multi(self, test_data_multi_var):
        strat_prop = StratPropData(
            data=test_data_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )
        strat_modes = {"stratum1": StratMode.FULL, "stratum2": StratMode.PARTIAL}
        props = strat_prop.compute_multipliers(strat_modes=strat_modes)
        assert props.shape == (12, 3, 3)

        prop_s1 = np.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],  # (age=0, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
                [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],  # (age=1, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
                [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],  # (age=2, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
            ]
        ).T  # shape (6, 3)

        prop_s2 = np.array(
            [
                [0.3, 0.7],  # (age=0, s1=A, B)
                [0.6, 0.4],  # (age=1, s1=A, B)
                [0.7, 0.3],  # (age=2, s1=A, B)
            ]
        ).T  # shape (2, 3)

        expected = prop_s1[:, None, :, None] * prop_s2[None, :, None, :]
        expected = expected.reshape(12, 3, 3)

        np.testing.assert_allclose(props, expected)
