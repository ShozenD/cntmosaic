import numpy as np
import pandas as pd
import pytest

from ..._types import StratMode
from ..containers._StratificationData import StratificationData
from .fixtures import df_strat_count, df_strat_multi_var, df_strat_single_var


class TestMethods:

    def get_strat_vars(self, df_strat_multi_var):
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )

        strat_vars = strat_data.get_strat_vars()
        assert strat_vars == ["stratum1", "stratum2"]

    def test_get_strat_var_schema(self, df_strat_multi_var):
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )

        schema = strat_data.get_strat_var_schema()
        expected_schema = {
            "stratum1": {"categories": ["A", "B"], "codes": [0, 1]},
            "stratum2": {"categories": ["X", "Y", "Z"], "codes": [0, 1, 2]},
        }
        assert schema == expected_schema


class TestDemopty:

    def test_partial_prop_single(self, df_strat_single_var):
        strat_data = StratificationData(
            data=df_strat_single_var,
            age_col="age",
            strat_var_cols="stratum",
            prop_col="prop",
        )

        strat_modes = {"stratum": StratMode.PARTIAL}

        props = strat_data.compute_demopty(strat_modes=strat_modes)
        assert props.shape == (2, 3, 1)
        expected = np.array([[0.3, 0.4, 0.2], [0.7, 0.6, 0.8]])[:, :, np.newaxis]
        np.testing.assert_allclose(
            props,
            expected,
        )

    def test_partial_prop_multi(self, df_strat_multi_var):
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )

        strat_modes = {"stratum1": StratMode.PARTIAL, "stratum2": StratMode.PARTIAL}

        props = strat_data.compute_demopty(strat_modes=strat_modes)
        assert props.shape == (6, 3, 1)

        expected = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.1],
                [0.1, 0.2, 0.1],
                [0.1, 0.1, 0.1],
                [0.5, 0.1, 0.1],
            ]
        )[
            :, :, np.newaxis
        ]  # shape (6, 3, 1)

        np.testing.assert_allclose(props, expected)

    def test_full_prop_single(self, df_strat_single_var):
        strat_data = StratificationData(
            data=df_strat_single_var,
            age_col="age",
            strat_var_cols="stratum",
            prop_col="prop",
        )
        strat_modes = {"stratum": StratMode.FULL}
        props = strat_data.compute_demopty(strat_modes=strat_modes)
        assert props.shape == (4, 3, 3)

        expected = np.array([0.3, 0.7, 0.4, 0.6, 0.2, 0.8]).reshape((2, 3), order="F")
        expected = (expected[:, None, :, None] * expected[None, :, None, :]).reshape(
            4, 3, 3
        )
        np.testing.assert_allclose(props, expected)

    def test_full_prop_multi(self, df_strat_multi_var):
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )
        strat_modes = {"stratum1": StratMode.FULL, "stratum2": StratMode.FULL}
        props = strat_data.compute_demopty(strat_modes=strat_modes)
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

    def test_mixed_prop_multi(self, df_strat_multi_var):
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )
        strat_modes = {"stratum1": StratMode.FULL, "stratum2": StratMode.PARTIAL}
        props = strat_data.compute_demopty(strat_modes=strat_modes)
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
