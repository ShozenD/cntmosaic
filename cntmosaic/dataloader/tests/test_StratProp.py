import numpy as np
import pandas as pd
import pytest

from ..._types import StratMode
from ..containers._StratPropData import StratPropData


@pytest.fixture
def test_data_prop():
    data = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],  # Three age groups: 0, 1, 2
            "stratum": ["A", "B", "A", "B", "A", "B"],  # Two strata: A, B
            "prop": [0.3, 0.7, 0.4, 0.6, 0.2, 0.8],  # Proportions
        }
    )
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

    def test_partial_prop(self, test_data_prop):
        strat_prop = StratPropData(
            data=test_data_prop,
            age_col="age",
            strat_col="stratum",
            prop_col="prop",
        )
        props = strat_prop.compute_props(mode=StratMode.PARTIAL)
        assert props.shape == (2, 3)
        np.testing.assert_allclose(
            props,
            np.array([[0.3, 0.4, 0.2], [0.7, 0.6, 0.8]]),
        )

    def test_partial_count(self, test_data_count):
        strat_prop = StratPropData.from_counts(
            data=test_data_count,
            age_col="age",
            strat_col="stratum",
            count_col="count",
        )
        props = strat_prop.compute_props(mode=StratMode.PARTIAL)
        assert props.shape == (2, 3)
        np.testing.assert_allclose(
            props,
            np.array([[0.3, 0.4, 0.2], [0.7, 0.6, 0.8]]),
        )

    def test_full_prop(self, test_data_prop):
        strat_prop = StratPropData(
            data=test_data_prop,
            age_col="age",
            strat_col="stratum",
            prop_col="prop",
        )
        props = strat_prop.compute_props(mode=StratMode.FULL)
        assert props.shape == (2, 2, 3, 3)
        np.testing.assert_allclose(
            props,
            np.array(
                [
                    [
                        [
                            [0.3 * 0.3, 0.3 * 0.4, 0.3 * 0.2],  # (s=1, t=1, a=1, ...)
                            [0.4 * 0.3, 0.4 * 0.4, 0.4 * 0.2],  # (s=1, t=1, a=2, ...)
                            [0.2 * 0.3, 0.2 * 0.4, 0.2 * 0.2],  # (s=1, t=1, a=3, ...)
                        ],
                        [
                            [0.3 * 0.7, 0.3 * 0.6, 0.3 * 0.8],  # (s=1, t=2, a=1, ...)
                            [0.4 * 0.7, 0.4 * 0.6, 0.4 * 0.8],  # (s=1, t=2, a=2, ...)
                            [0.2 * 0.7, 0.2 * 0.6, 0.2 * 0.8],  # (s=1, t=2, a=3, ...)
                        ],
                    ],
                    [
                        [
                            [0.7 * 0.3, 0.7 * 0.4, 0.7 * 0.2],  # (s=2, t=1, a=1, ...)
                            [0.6 * 0.3, 0.6 * 0.4, 0.6 * 0.2],  # (s=2, t=1, a=2, ...)
                            [0.8 * 0.3, 0.8 * 0.4, 0.8 * 0.2],  # (s=2, t=1, a=3, ...)
                        ],
                        [
                            [0.7 * 0.7, 0.7 * 0.6, 0.7 * 0.8],  # (s=2, t=2, a=1, ...)
                            [0.6 * 0.7, 0.6 * 0.6, 0.6 * 0.8],  # (s=2, t=2, a=2, ...)
                            [0.8 * 0.7, 0.8 * 0.6, 0.8 * 0.8],  # (s=2, t=2, a=3, ...)
                        ],
                    ],
                ],
            ),
        )

    def test_full_count(self, test_data_count):
        strat_prop = StratPropData.from_counts(
            data=test_data_count,
            age_col="age",
            strat_col="stratum",
            count_col="count",
        )
        props = strat_prop.compute_props(mode=StratMode.FULL)
        assert props.shape == (2, 2, 3, 3)
        np.testing.assert_allclose(
            props,
            np.array(
                [
                    [
                        [
                            [0.3 * 0.3, 0.3 * 0.4, 0.3 * 0.2],  # (s=1, t=1, a=1, ...)
                            [0.4 * 0.3, 0.4 * 0.4, 0.4 * 0.2],  # (s=1, t=1, a=2, ...)
                            [0.2 * 0.3, 0.2 * 0.4, 0.2 * 0.2],  # (s=1, t=1, a=3, ...)
                        ],
                        [
                            [0.3 * 0.7, 0.3 * 0.6, 0.3 * 0.8],  # (s=1, t=2, a=1, ...)
                            [0.4 * 0.7, 0.4 * 0.6, 0.4 * 0.8],  # (s=1, t=2, a=2, ...)
                            [0.2 * 0.7, 0.2 * 0.6, 0.2 * 0.8],  # (s=1, t=2, a=3, ...)
                        ],
                    ],
                    [
                        [
                            [0.7 * 0.3, 0.7 * 0.4, 0.7 * 0.2],  # (s=2, t=1, a=1, ...)
                            [0.6 * 0.3, 0.6 * 0.4, 0.6 * 0.2],  # (s=2, t=1, a=2, ...)
                            [0.8 * 0.3, 0.8 * 0.4, 0.8 * 0.2],  # (s=2, t=1, a=3, ...)
                        ],
                        [
                            [0.7 * 0.7, 0.7 * 0.6, 0.7 * 0.8],  # (s=2, t=2, a=1, ...)
                            [0.6 * 0.7, 0.6 * 0.6, 0.6 * 0.8],  # (s=2, t=2, a=2, ...)
                            [0.8 * 0.7, 0.8 * 0.6, 0.8 * 0.8],  # (s=2, t=2, a=3, ...)
                        ],
                    ],
                ],
            ),
        )
