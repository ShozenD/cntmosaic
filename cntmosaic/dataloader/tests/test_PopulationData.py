"""
Tests for PopulationData dataclass.

This test suite comprehensively validates the PopulationData class functionality,
including initialization, validation, preprocessing, aggregation, and all helper methods.
"""

import numpy as np
import pandas as pd
import pytest

from ..containers._PopulationData import PopulationData
from .fixtures import df_pop_age_grps, df_pop_basic, df_pop_multi_var, df_pop_single_var


class TestInit:
    """Test PopulationData initialization with various configurations."""

    def test_basic(self, df_pop_basic):
        """Test basic initialization with required parameters."""
        df = df_pop_basic
        pop_data = PopulationData(data=df, age_col="age", size_col="P")

        assert pop_data.n_ages == 4
        assert "age" in pop_data.data.columns
        assert "P" in pop_data.data.columns

    def test_single_var(self, df_pop_single_var):
        """Test initialization with a single stratification variable."""

        pop_data = PopulationData(
            data=df_pop_single_var,
            age_col="age",
            size_col="P",
            strat_var_cols="sex",
        )

        assert pop_data.n_ages == 3
        assert "sex" in pop_data.data.columns
        assert len(pop_data.data) == 6  # 3 ages × 2 sexes

    def test_multiple_vars(self, df_pop_multi_var):
        """Test initialization with multiple stratification variables."""
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

    def test_age_group(self, df_pop_age_grps):
        """Test initialization with age groups."""

        pop_data = PopulationData(
            data=df_pop_age_grps,
            age_col="age",
            size_col="P",
            age_grp_col="age_grp",
        )

        assert "age_grp_pop" in pop_data.data.columns
        assert pop_data.n_ages == 3


class TestPopulationDataValidation:
    """Test PopulationData validation methods."""

    def test_missing_age_column(self):
        """Test that missing age column raises KeyError."""
        df = pd.DataFrame({"not_age": [0, 1, 2], "population": [1000, 1100, 1200]})

        with pytest.raises(KeyError, match="Missing population age column"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_missing_size_column(self):
        """Test that missing size column raises KeyError."""
        df = pd.DataFrame({"age": [0, 1, 2], "not_population": [1000, 1100, 1200]})

        with pytest.raises(KeyError, match="Missing population size column"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_missing_stratification_column(self):
        """Test that missing stratification variable raises KeyError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        with pytest.raises(
            KeyError, match="Missing population stratification variable"
        ):
            PopulationData(
                data=df,
                age_col="age",
                size_col="population",
                strat_var_cols=["gender"],  # 'gender' doesn't exist
            )

    def test_missing_age_grp_column(self):
        """Test that missing age_grp_col raises KeyError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        with pytest.raises(KeyError, match="Missing population age group column"):
            PopulationData(
                data=df,
                age_col="age",
                size_col="population",
                age_grp_col="age_group",  # doesn't exist
            )

    def test_negative_ages(self):
        """Test that negative ages raise ValueError."""
        df = pd.DataFrame({"age": [-1, 0, 1], "population": [1000, 1100, 1200]})

        with pytest.raises(ValueError, match="contains negative values"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_negative_population_sizes(self):
        """Test that negative population sizes raise ValueError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, -100, 1200]})

        with pytest.raises(ValueError, match="contains negative values"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_non_numeric_ages(self):
        """Test that non-numeric ages raise ValueError."""
        df = pd.DataFrame(
            {"age": ["zero", "one", "two"], "population": [1000, 1100, 1200]}
        )

        with pytest.raises(ValueError, match="must contain numeric values"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_non_numeric_population_sizes(self):
        """Test that non-numeric population sizes raise ValueError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": ["1000", "1100", "1200"]})

        with pytest.raises(ValueError, match="must contain numeric values"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_missing_values_dropped_with_warning(self):
        """Test that missing values trigger warning and are dropped."""
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
        """Test that empty DataFrame after dropping missing raises ValueError."""
        df = pd.DataFrame({"age": [np.nan, np.nan], "population": [1000, 1100]})

        with pytest.raises(ValueError, match="DataFrame is empty after removing"):
            PopulationData(data=df, age_col="age", size_col="population")

    def test_zero_population_sizes_warning(self):
        """Test that zero population sizes trigger warning."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 0, 1200]})

        with pytest.warns(UserWarning, match="zero population size"):
            pop_data = PopulationData(data=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 3


class TestProperties:
    """Test PopulationData properties."""

    def test_data_property(self):
        """Test data property returns processed DataFrame."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        pop_data = PopulationData(data=df, age_col="age", size_col="population")

        assert isinstance(pop_data.data, pd.DataFrame)
        assert "age" in pop_data.data.columns
        assert "P" in pop_data.data.columns

    def test_n_ages_property(self):
        """Test n_ages property."""
        df = pd.DataFrame(
            {"age": [0, 1, 2, 3, 4], "population": [1000, 1100, 1200, 1150, 1180]}
        )

        pop_data = PopulationData(data=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 5

    def test_total(self):
        """Test total property."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        pop_data = PopulationData(data=df, age_col="age", size_col="population")

        assert pop_data.total == 3300

    def test_age_range(self):
        """Test age_range property."""
        df = pd.DataFrame(
            {"age": [5, 10, 15, 20], "population": [1000, 1100, 1200, 1150]}
        )

        pop_data = PopulationData(data=df, age_col="age", size_col="population")

        assert pop_data.age_range == (5, 20)

    def test_strat_vars(self, df_pop_multi_var):
        """Test strat_vars with multiple variables."""
        pop_data = PopulationData(
            data=df_pop_multi_var,
            age_col="age",
            size_col="P",
            strat_var_cols=["sex", "hhsize"],
        )

        assert pop_data.strat_vars == ["sex", "hhsize"]


class TestMethods:
    """Test PopulationData methods."""

    def test_get_strat_var_schema(self, df_pop_multi_var):
        """Test get_strat_var_schema method."""

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
        """Test summary method."""
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


class TestAggregation:
    """Test PopulationData aggregation behavior."""

    def test_duplicate_age(self):
        """Test that multiple rows per age are aggregated."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1, 2],  # Duplicate ages
                "population": [500, 500, 550, 550, 1200],
            }
        )

        with pytest.warns(UserWarning, match="Aggregating population data"):
            pop_data = PopulationData(data=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 3
        assert pop_data.data.loc[pop_data.data["age"] == 0, "P"].iloc[0] == 1000
        assert pop_data.data.loc[pop_data.data["age"] == 1, "P"].iloc[0] == 1100

    def test_aggregation_with_stratification(self):
        """Test aggregation with stratification variables."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 0, 0],
                "sex": ["M", "M", "F", "F"],  # Duplicates within gender
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
        """Test that no aggregation warning is raised when data is already unique."""
        df = pd.DataFrame({"age": [0, 1, 2], "P": [1000, 1100, 1200]})

        # Should NOT raise aggregation warning
        import warnings as warnings_module

        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            pop_data = PopulationData(data=df, age_col="age", size_col="P")

        # Filter for aggregation warnings only
        agg_warnings = [
            w for w in warning_list if "Aggregating population" in str(w.message)
        ]
        assert len(agg_warnings) == 0


class TestPopulationDataEdgeCases:
    """Test PopulationData with edge cases."""

    def test_single_age(self):
        """Test with a single age."""
        df = pd.DataFrame({"age": [0], "P": [1000]})

        pop_data = PopulationData(data=df, age_col="age", size_col="P")

        assert pop_data.n_ages == 1
        assert pop_data.age_range == (0, 0)
        assert pop_data.total == 1000

    def test_float_population_sizes(self):
        """Test with float population sizes (proportions)."""
        df = pd.DataFrame({"age": [0, 1, 2], "P": [0.3, 0.4, 0.3]})

        pop_data = PopulationData(data=df, age_col="age", size_col="P")

        assert np.isclose(pop_data.total, 1.0)
