"""
Tests for PopulationData dataclass.

Test organisation
-----------------
1. TestCoreContracts  — column renaming/standardisation, automatic aggregation,
                        original-df immutability.
2. TestInputValidation — TypeError/ValueError/KeyError on bad inputs.
3. TestAccessorMethods — properties and methods that expose processed data.
4. TestEdgeCases       — genuine boundary conditions.
"""

import warnings as warnings_module

import numpy as np
import pandas as pd
import pytest

from ..containers._PopulationData import PopulationData
from .fixtures import df_pop_age_grps, df_pop_age_min_max, df_pop_basic, df_pop_multi_var, df_pop_single_var


# =====================
# 1. Core Contracts
# =====================


class TestCoreContracts:
    """Test the fundamental data-pipeline guarantees of PopulationData."""

    def test_basic_column_renaming(self, df_pop_basic):
        """size_col is renamed to 'P'; age column is preserved as 'age'."""
        pop_data = PopulationData(data=df_pop_basic, age_col="age", size_col="P")

        assert pop_data.n_ages == 4
        assert "age" in pop_data.data.columns
        assert "P" in pop_data.data.columns

    def test_single_strat_var(self, df_pop_single_var):
        """Single stratification column is preserved; row count matches age × strata."""
        pop_data = PopulationData(
            data=df_pop_single_var,
            age_col="age",
            size_col="P",
            strat_var_cols="sex",
        )

        assert pop_data.n_ages == 3
        assert "sex" in pop_data.data.columns
        assert len(pop_data.data) == 6  # 3 ages × 2 sexes

    def test_multiple_strat_vars(self, df_pop_multi_var):
        """Multiple stratification columns are all preserved and total is correct."""
        pop_data = PopulationData(
            data=df_pop_multi_var,
            age_col="age",
            size_col="P",
            strat_var_cols=["sex", "hhsize"],
        )

        assert pop_data.n_ages == 3
        assert "sex" in pop_data.data.columns
        assert "hhsize" in pop_data.data.columns
        assert pop_data.total == 3000

    def test_duplicate_age_rows_are_summed(self):
        """Multiple rows per age are aggregated (summed) with a UserWarning."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1, 2],
                "population": [500, 500, 550, 550, 1200],
            }
        )

        with pytest.warns(UserWarning, match="Aggregating population data"):
            pop_data = PopulationData(data=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 3
        assert pop_data.data.loc[pop_data.data["age"] == 0, "P"].iloc[0] == 1000
        assert pop_data.data.loc[pop_data.data["age"] == 1, "P"].iloc[0] == 1100

    def test_aggregation_with_stratification(self):
        """Duplicate rows within (age, strat) groups are summed correctly."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 0, 0],
                "sex": ["M", "M", "F", "F"],
                "P": [250, 260, 240, 250],
            }
        )

        with pytest.warns(UserWarning, match="Aggregating population data"):
            pop_data = PopulationData(
                data=df, age_col="age", size_col="P", strat_var_cols="sex"
            )

        assert len(pop_data.data) == 2  # M and F
        male_pop = pop_data.data.loc[pop_data.data["sex"] == "M", "P"].iloc[0]
        female_pop = pop_data.data.loc[pop_data.data["sex"] == "F", "P"].iloc[0]
        assert male_pop == 510  # 250 + 260
        assert female_pop == 490  # 240 + 250

    def test_no_aggregation_warning_when_not_needed(self):
        """No aggregation warning is raised when each age already has one row."""
        df = pd.DataFrame({"age": [0, 1, 2], "P": [1000, 1100, 1200]})

        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            PopulationData(data=df, age_col="age", size_col="P")

        agg_warnings = [
            w for w in warning_list if "Aggregating population" in str(w.message)
        ]
        assert len(agg_warnings) == 0

    def test_original_dataframe_not_mutated(self, df_pop_basic):
        """PopulationData works on a copy; the caller's DataFrame is unchanged."""
        original_cols = df_pop_basic.columns.tolist()
        original_shape = df_pop_basic.shape
        PopulationData(data=df_pop_basic, age_col="age", size_col="P")
        assert df_pop_basic.columns.tolist() == original_cols
        assert df_pop_basic.shape == original_shape


# =====================
# 2. Input Validation
# =====================


class TestInputValidation:
    """Test TypeError/ValueError/KeyError on bad inputs."""

    def test_missing_age_specification(self):
        """Neither age_col nor age_min/max raises ValueError."""
        df = pd.DataFrame({"population": [1000, 1100, 1200]})
        with pytest.raises(ValueError, match="Must specify an age representation"):
            PopulationData(data=df, size_col="population")

    def test_age_col_and_age_min_max_raises(self):
        """Providing age_col and age_min/max raises ValueError."""
        df = pd.DataFrame({"age": [0, 1], "age_min": [0, 5], "age_max": [4, 9], "P": [1000, 1100]})
        with pytest.raises(ValueError, match="Age specification forms are mutually exclusive"):
            PopulationData(data=df, size_col="P", age_col="age", age_min_col="age_min", age_max_col="age_max")

    def test_age_min_without_age_max_raises(self):
        """age_min_col without age_max_col raises ValueError."""
        df = pd.DataFrame({"age_min": [0, 5], "P": [1000, 1100]})
        with pytest.raises(ValueError, match="Both 'age_min_col' and 'age_max_col' must be specified"):
            PopulationData(data=df, size_col="P", age_min_col="age_min")

    def test_missing_age_column(self):
        """Missing age column raises KeyError."""
        df = pd.DataFrame({"not_age": [0, 1, 2], "population": [1000, 1100, 1200]})
        with pytest.raises(KeyError, match="Missing population age column"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_missing_size_column(self):
        """Missing size column raises KeyError."""
        df = pd.DataFrame({"age": [0, 1, 2], "not_population": [1000, 1100, 1200]})
        with pytest.raises(KeyError, match="Missing population size column"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_missing_stratification_column(self):
        """Missing stratification variable raises KeyError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})
        with pytest.raises(KeyError, match="Missing population stratification variable"):
            PopulationData(
                data=df,
                age_col="age",
                size_col="population",
                strat_var_cols=["gender"],
            )

    def test_missing_age_grp_column(self):
        """Missing age_grp_col raises KeyError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})
        with pytest.raises(KeyError, match="Missing population age group column"):
            PopulationData(
                data=df,
                age_col="age",
                size_col="population",
                age_grp_col="age_group",
            )

    def test_negative_ages(self):
        """Negative ages raise ValueError."""
        df = pd.DataFrame({"age": [-1, 0, 1], "population": [1000, 1100, 1200]})
        with pytest.raises(ValueError, match="contains negative values"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_negative_population_sizes(self):
        """Negative population sizes raise ValueError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, -100, 1200]})
        with pytest.raises(ValueError, match="contains negative values"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_non_numeric_ages(self):
        """Non-numeric ages raise ValueError."""
        df = pd.DataFrame({"age": ["zero", "one", "two"], "population": [1000, 1100, 1200]})
        with pytest.raises(ValueError, match="must contain numeric values"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_non_numeric_population_sizes(self):
        """Non-numeric population sizes raise ValueError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": ["1000", "1100", "1200"]})
        with pytest.raises(ValueError, match="must contain numeric values"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_missing_values_dropped_with_warning(self):
        """Missing values trigger a UserWarning and the row is dropped."""
        df = pd.DataFrame(
            {
                "age": [0, 1, 2, 3],
                "population": [1000, np.nan, 1200, 1150],
            }
        )

        with pytest.warns(UserWarning, match="Dropped 1 row"):
            pop_data = PopulationData(data=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 3
        assert pop_data.total == 3350

    def test_empty_dataframe_after_dropping_missing(self):
        """Empty DataFrame after dropping missing raises ValueError."""
        df = pd.DataFrame({"age": [np.nan, np.nan], "population": [1000, 1100]})
        with pytest.raises(ValueError, match="DataFrame is empty after removing"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_zero_population_sizes_warning(self):
        """Zero population sizes trigger a UserWarning."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 0, 1200]})
        with pytest.warns(UserWarning, match="zero population size"):
            pop_data = PopulationData(data=df, age_col="age", size_col="population")
        assert pop_data.n_ages == 3


# =====================
# 3. Accessor Methods
# =====================


class TestAccessorMethods:
    """Test properties and methods that expose processed data."""

    def test_total(self):
        """total returns the sum of all population sizes."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})
        pop_data = PopulationData(data=df, age_col="age", size_col="population")
        assert pop_data.total == 3300

    def test_age_range(self):
        """age_range returns the (min, max) tuple of age values."""
        df = pd.DataFrame(
            {"age": [5, 10, 15, 20], "population": [1000, 1100, 1200, 1150]}
        )
        pop_data = PopulationData(data=df, age_col="age", size_col="population")
        assert pop_data.age_range == (5, 20)

    def test_get_strat_var_schema(self, df_pop_multi_var):
        """get_strat_var_schema returns correct categories and integer codes."""
        pop_data = PopulationData(
            data=df_pop_multi_var,
            age_col="age",
            size_col="P",
            strat_var_cols=["sex", "hhsize"],
        )
        schema = pop_data.get_strat_var_schema()
        assert schema == {
            "sex": {"categories": ["F", "M"], "codes": [0, 1]},
            "hhsize": {"categories": ["1", "2"], "codes": [0, 1]},
        }

    def test_summary_method(self):
        """summary() returns all expected keys with correct values."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1],
                "gender": ["M", "F", "M", "F"],
                "population": [510, 490, 530, 520],
            }
        )
        pop_data = PopulationData(
            data=df, age_col="age", size_col="population", strat_var_cols="gender"
        )
        summary = pop_data.summary()

        assert summary["n_ages"] == 2
        assert summary["age_range"] == (0, 1)
        assert summary["total"] == 2050
        assert summary["strat_vars"] == ["gender"]
        assert summary["n_strat_vars"] == 1
        assert summary["is_stratified"] is True


# =====================
# 4. Edge Cases
# =====================


class TestEdgeCases:
    """Test genuine boundary conditions."""

    def test_age_group_col_preserved(self, df_pop_age_grps):
        """age_grp_col is preserved as 'age_grp_pop' in the processed data."""
        pop_data = PopulationData(
            data=df_pop_age_grps,
            age_col="age",
            size_col="P",
            age_grp_col="age_grp",
        )
        assert "age_grp_pop" in pop_data.data.columns
        assert pop_data.n_ages == 3

    def test_age_min_max_form(self, df_pop_age_min_max):
        """age_min_col/age_max_col form works; 'age' is absent; P is present."""
        pop_data = PopulationData(
            data=df_pop_age_min_max,
            size_col="population",
            age_min_col="age_min",
            age_max_col="age_max",
        )
        assert "age_min" in pop_data.data.columns
        assert "age_max" in pop_data.data.columns
        assert "P" in pop_data.data.columns
        assert "age" not in pop_data.data.columns
        assert pop_data.n_ages == 4
        assert pop_data.age_range == (0, 19)
        assert pop_data.total == 18800

    def test_single_age(self):
        """Single-age DataFrame is valid; age_range is a degenerate tuple."""
        df = pd.DataFrame({"age": [0], "P": [1000]})
        pop_data = PopulationData(data=df, age_col="age", size_col="P")
        assert pop_data.n_ages == 1
        assert pop_data.age_range == (0, 0)
        assert pop_data.total == 1000

    def test_float_population_sizes(self):
        """Float population sizes (e.g. proportions) are accepted."""
        df = pd.DataFrame({"age": [0, 1, 2], "P": [0.3, 0.4, 0.3]})
        pop_data = PopulationData(data=df, age_col="age", size_col="P")
        assert np.isclose(pop_data.total, 1.0)
