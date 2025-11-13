"""
Tests for PopulationData dataclass.

This test suite comprehensively validates the PopulationData class functionality,
including initialization, validation, preprocessing, aggregation, and all helper methods.
"""

import numpy as np
import pandas as pd
import pytest

from ..containers._PopulationData import PopulationData


class TestPopulationDataInitialization:
    """Test PopulationData initialization with various configurations."""

    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        df = pd.DataFrame({"age": [0, 1, 2, 3], "population": [1000, 1100, 1200, 1150]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 4
        assert "age" in pop_data.data.columns
        assert "P" in pop_data.data.columns

    def test_initialization_with_stratification_var(self):
        """Test initialization with a single stratification variable."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1, 2, 2],
                "gender": ["M", "F", "M", "F", "M", "F"],
                "population": [510, 490, 530, 520, 550, 550],
            }
        )

        pop_data = PopulationData(
            df_pop=df, age_col="age", size_col="population", strat_vars="gender"
        )

        assert pop_data.n_ages == 3
        assert "gender" in pop_data.data.columns
        assert len(pop_data.data) == 6  # 3 ages × 2 genders

    def test_initialization_with_multiple_stratification_vars(self):
        """Test initialization with multiple stratification variables."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 0, 0],
                "gender": ["M", "F", "M", "F"],
                "region": ["North", "North", "South", "South"],
                "population": [250, 240, 260, 250],
            }
        )

        pop_data = PopulationData(
            df_pop=df,
            age_col="age",
            size_col="population",
            strat_vars=["gender", "region"],
        )

        assert pop_data.n_ages == 1
        assert "gender" in pop_data.data.columns
        assert "region" in pop_data.data.columns
        assert pop_data.total_population == 1000

    def test_initialization_with_age_group(self):
        """Test initialization with age groups."""
        df = pd.DataFrame(
            {
                "age": [0, 5, 10],
                "age_group": pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
                "population": [5000, 4800, 4600],
            }
        )

        pop_data = PopulationData(
            df_pop=df,
            age_col="age",
            size_col="population",
            age_grp_col="age_group",
        )

        assert "age_grp_pop" in pop_data.data.columns
        assert pop_data.n_ages == 3
        assert isinstance(pop_data.data["age_grp_pop"].dtype, pd.IntervalDtype)

    def test_invalid_type_for_df_pop(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="df_pop must be a pandas DataFrame"):
            PopulationData(df_pop=[1, 2, 3], age_col="age", size_col="population")

    def test_missing_age_col_raises_error(self):
        """Test that missing age_col raises ValueError."""
        df = pd.DataFrame({"population": [1000, 1100]})

        with pytest.raises(ValueError, match="Must specify 'age_col'"):
            PopulationData(df_pop=df, age_col=None, size_col="population")


class TestPopulationDataValidation:
    """Test PopulationData validation methods."""

    def test_missing_age_column(self):
        """Test that missing age column raises KeyError."""
        df = pd.DataFrame({"not_age": [0, 1, 2], "population": [1000, 1100, 1200]})

        with pytest.raises(KeyError, match="Missing population age column"):
            PopulationData(df_pop=df, age_col="age", size_col="population")

    def test_missing_size_column(self):
        """Test that missing size column raises KeyError."""
        df = pd.DataFrame({"age": [0, 1, 2], "not_population": [1000, 1100, 1200]})

        with pytest.raises(KeyError, match="Missing population size column"):
            PopulationData(df_pop=df, age_col="age", size_col="population")

    def test_missing_stratification_column(self):
        """Test that missing stratification variable raises KeyError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        with pytest.raises(
            KeyError, match="Missing population stratification variable"
        ):
            PopulationData(
                df_pop=df,
                age_col="age",
                size_col="population",
                strat_vars=["gender"],  # 'gender' doesn't exist
            )

    def test_missing_age_grp_column(self):
        """Test that missing age_grp_col raises KeyError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        with pytest.raises(KeyError, match="Missing population age group column"):
            PopulationData(
                df_pop=df,
                age_col="age",
                size_col="population",
                age_grp_col="age_group",  # doesn't exist
            )

    def test_negative_ages(self):
        """Test that negative ages raise ValueError."""
        df = pd.DataFrame({"age": [-1, 0, 1], "population": [1000, 1100, 1200]})

        with pytest.raises(ValueError, match="contains negative values"):
            PopulationData(df_pop=df, age_col="age", size_col="population")

    def test_negative_population_sizes(self):
        """Test that negative population sizes raise ValueError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, -100, 1200]})

        with pytest.raises(ValueError, match="contains negative values"):
            PopulationData(df_pop=df, age_col="age", size_col="population")

    def test_non_numeric_ages(self):
        """Test that non-numeric ages raise ValueError."""
        df = pd.DataFrame(
            {"age": ["zero", "one", "two"], "population": [1000, 1100, 1200]}
        )

        with pytest.raises(ValueError, match="must contain numeric values"):
            PopulationData(df_pop=df, age_col="age", size_col="population")

    def test_non_numeric_population_sizes(self):
        """Test that non-numeric population sizes raise ValueError."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": ["1000", "1100", "1200"]})

        with pytest.raises(ValueError, match="must contain numeric values"):
            PopulationData(df_pop=df, age_col="age", size_col="population")

    def test_missing_values_dropped_with_warning(self):
        """Test that missing values trigger warning and are dropped."""
        df = pd.DataFrame(
            {
                "age": [0, 1, 2, 3],
                "population": [1000, np.nan, 1200, 1150],
            }
        )

        with pytest.warns(UserWarning, match="Dropped 1 row"):
            pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 3
        assert pop_data.total_population == 3350

    def test_empty_dataframe_after_dropping_missing(self):
        """Test that empty DataFrame after dropping missing raises ValueError."""
        df = pd.DataFrame({"age": [np.nan, np.nan], "population": [1000, 1100]})

        with pytest.raises(ValueError, match="DataFrame is empty after removing"):
            PopulationData(df_pop=df, age_col="age", size_col="population")

    def test_zero_population_sizes_warning(self):
        """Test that zero population sizes trigger warning."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 0, 1200]})

        with pytest.warns(UserWarning, match="zero population size"):
            pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 3


class TestPopulationDataProperties:
    """Test PopulationData properties."""

    def test_data_property(self):
        """Test data property returns processed DataFrame."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert isinstance(pop_data.data, pd.DataFrame)
        assert "age" in pop_data.data.columns
        assert "P" in pop_data.data.columns

    def test_n_ages_property(self):
        """Test n_ages property."""
        df = pd.DataFrame(
            {"age": [0, 1, 2, 3, 4], "population": [1000, 1100, 1200, 1150, 1180]}
        )

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 5

    def test_total_population_property(self):
        """Test total_population property."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.total_population == 3300

    def test_age_range_property(self):
        """Test age_range property."""
        df = pd.DataFrame(
            {"age": [5, 10, 15, 20], "population": [1000, 1100, 1200, 1150]}
        )

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.age_range == (5, 20)

    def test_stratification_vars_property_empty(self):
        """Test stratification_vars with no variables."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.stratification_vars == []

    def test_stratification_vars_property_with_vars(self):
        """Test stratification_vars with multiple variables."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 0, 0],
                "gender": ["M", "F", "M", "F"],
                "region": ["A", "A", "B", "B"],
                "population": [250, 250, 250, 250],
            }
        )

        pop_data = PopulationData(
            df_pop=df,
            age_col="age",
            size_col="population",
            strat_vars=["gender", "region"],
        )

        assert pop_data.stratification_vars == ["gender", "region"]

    def test_as_proportions_property(self):
        """Test as_proportions property."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 2000, 3000]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")
        pop_prop = pop_data.as_proportions

        assert np.isclose(pop_prop["P"].sum(), 1.0)
        assert np.isclose(pop_prop["P"].iloc[0], 1000 / 6000)
        assert np.isclose(pop_prop["P"].iloc[1], 2000 / 6000)
        # Original unchanged
        assert pop_data.total_population == 6000


class TestPopulationDataMethods:
    """Test PopulationData methods."""

    def test_get_age_distribution_simple(self):
        """Test get_age_distribution for non-stratified data."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")
        age_dist = pop_data.get_age_distribution()

        assert len(age_dist) == 3
        assert age_dist[0] == 1000
        assert age_dist[1] == 1100
        assert age_dist[2] == 1200

    def test_get_age_distribution_stratified_marginal(self):
        """Test get_age_distribution with stratification (marginal)."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1],
                "gender": ["M", "F", "M", "F"],
                "population": [510, 490, 530, 520],
            }
        )

        pop_data = PopulationData(
            df_pop=df, age_col="age", size_col="population", strat_vars="gender"
        )
        age_dist = pop_data.get_age_distribution(by_group=False)

        assert len(age_dist) == 2
        assert age_dist[0] == 1000  # 510 + 490
        assert age_dist[1] == 1050  # 530 + 520

    def test_get_age_distribution_stratified_by_group(self):
        """Test get_age_distribution with stratification (by group)."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1],
                "gender": ["M", "F", "M", "F"],
                "population": [510, 490, 530, 520],
            }
        )

        pop_data = PopulationData(
            df_pop=df, age_col="age", size_col="population", strat_vars="gender"
        )
        age_dist = pop_data.get_age_distribution(by_group=True)

        assert len(age_dist) == 4  # 2 ages × 2 genders
        assert age_dist[(0, "M")] == 510
        assert age_dist[(0, "F")] == 490

    def test_normalize_method(self):
        """Test normalize method."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 2000, 3000]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")
        pop_normalized = pop_data.normalize()

        assert np.isclose(pop_normalized.total_population, 1.0)
        assert np.isclose(pop_normalized.data["P"].iloc[0], 1000 / 6000)
        # Original unchanged
        assert pop_data.total_population == 6000

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
            df_pop=df, age_col="age", size_col="population", strat_vars="gender"
        )
        summary = pop_data.summary()

        assert summary["n_ages"] == 2
        assert summary["age_range"] == (0, 1)
        assert summary["total_population"] == 2050
        assert summary["stratification_vars"] == ["gender"]
        assert summary["n_stratification_vars"] == 1
        assert summary["is_stratified"] is True


class TestPopulationDataAggregation:
    """Test PopulationData aggregation behavior."""

    def test_aggregation_without_stratification(self):
        """Test that multiple rows per age are aggregated."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1, 2],  # Duplicate ages
                "population": [500, 500, 550, 550, 1200],
            }
        )

        with pytest.warns(UserWarning, match="Aggregating population data"):
            pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 3
        assert pop_data.data.loc[pop_data.data["age"] == 0, "P"].iloc[0] == 1000
        assert pop_data.data.loc[pop_data.data["age"] == 1, "P"].iloc[0] == 1100

    def test_aggregation_with_stratification(self):
        """Test aggregation with stratification variables."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 0, 0],
                "gender": ["M", "M", "F", "F"],  # Duplicates within gender
                "population": [250, 260, 240, 250],
            }
        )

        with pytest.warns(UserWarning, match="Aggregating population data"):
            pop_data = PopulationData(
                df_pop=df, age_col="age", size_col="population", strat_vars="gender"
            )

        assert len(pop_data.data) == 2  # M and F
        male_pop = pop_data.data.loc[pop_data.data["gender"] == "M", "P"].iloc[0]
        female_pop = pop_data.data.loc[pop_data.data["gender"] == "F", "P"].iloc[0]
        assert male_pop == 510  # 250 + 260
        assert female_pop == 490  # 240 + 250

    def test_no_aggregation_warning_when_not_needed(self):
        """Test that no aggregation warning is raised when data is already unique."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 1100, 1200]})

        # Should NOT raise aggregation warning
        import warnings as warnings_module

        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        # Filter for aggregation warnings only
        agg_warnings = [
            w for w in warning_list if "Aggregating population" in str(w.message)
        ]
        assert len(agg_warnings) == 0


class TestPopulationDataEdgeCases:
    """Test PopulationData with edge cases."""

    def test_single_age(self):
        """Test with a single age."""
        df = pd.DataFrame({"age": [0], "population": [1000]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 1
        assert pop_data.age_range == (0, 0)
        assert pop_data.total_population == 1000

    def test_large_age_range(self):
        """Test with large age range."""
        ages = list(range(0, 101))
        populations = [1000 + i * 10 for i in range(101)]
        df = pd.DataFrame({"age": ages, "population": populations})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 101
        assert pop_data.age_range == (0, 100)

    def test_float_ages(self):
        """Test with float ages (should work)."""
        df = pd.DataFrame({"age": [0.5, 1.5, 2.5], "population": [1000, 1100, 1200]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 3

    def test_float_population_sizes(self):
        """Test with float population sizes (proportions)."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [0.3, 0.4, 0.3]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert np.isclose(pop_data.total_population, 1.0)

    def test_very_small_populations(self):
        """Test with very small population sizes."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1e-6, 2e-6, 3e-6]})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.total_population > 0


class TestPopulationDataIntegration:
    """Integration tests for PopulationData with realistic scenarios."""

    def test_realistic_population_data(self):
        """Test with realistic population distribution."""
        np.random.seed(42)
        ages = list(range(0, 86))
        populations = [
            np.random.randint(800000, 1200000) for _ in range(86)
        ]  # Realistic US-like distribution

        df = pd.DataFrame({"age": ages, "population": populations})

        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")

        assert pop_data.n_ages == 86
        assert pop_data.age_range == (0, 85)
        assert pop_data.total_population > 0

        summary = pop_data.summary()
        assert summary["n_ages"] == 86
        assert summary["is_stratified"] is False

    def test_stratified_population_data(self):
        """Test with stratified population (gender and region)."""
        ages = list(range(0, 10))
        data_rows = []

        for age in ages:
            for gender in ["M", "F"]:
                for region in ["North", "South"]:
                    pop = np.random.randint(5000, 10000)
                    data_rows.append(
                        {
                            "age": age,
                            "gender": gender,
                            "region": region,
                            "population": pop,
                        }
                    )

        df = pd.DataFrame(data_rows)

        pop_data = PopulationData(
            df_pop=df,
            age_col="age",
            size_col="population",
            strat_vars=["gender", "region"],
        )

        assert pop_data.n_ages == 10
        assert len(pop_data.data) == 40  # 10 ages × 2 genders × 2 regions
        assert pop_data.stratification_vars == ["gender", "region"]

        # Test marginal distribution
        age_dist = pop_data.get_age_distribution(by_group=False)
        assert len(age_dist) == 10

        # Test stratified distribution
        age_dist_by_group = pop_data.get_age_distribution(by_group=True)
        assert len(age_dist_by_group) == 40

    def test_conversion_between_counts_and_proportions(self):
        """Test converting between counts and proportions."""
        df = pd.DataFrame({"age": [0, 1, 2], "population": [1000, 2000, 3000]})

        # Start with counts
        pop_data = PopulationData(df_pop=df, age_col="age", size_col="population")
        assert pop_data.total_population == 6000

        # Convert to proportions
        pop_prop = pop_data.as_proportions
        assert np.isclose(pop_prop["P"].sum(), 1.0)

        # Normalize (creates new instance)
        pop_normalized = pop_data.normalize()
        assert np.isclose(pop_normalized.total_population, 1.0)

        # Original unchanged
        assert pop_data.total_population == 6000
