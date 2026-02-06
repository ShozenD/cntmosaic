"""
Tests for ParticipantData dataclass.

This module tests the validation, properties, and methods of the ParticipantData class,
ensuring robust handling of participant survey data.
"""

import numpy as np
import pandas as pd
import pytest

from ..containers._ParticipantData import ParticipantData
from .fixtures import df_part_age_grps, df_part_one_year


class TestInit:
    """Test initialization and validation of ParticipantData."""

    def test_init_one_year(self, df_part_one_year):
        """Test basic initialization with exact ages."""
        df_part = df_part_one_year

        part_data = ParticipantData(df_part, id_col="id", age_col="age")

        assert part_data.n == 5
        assert part_data.age_col == "age"
        assert part_data.age_grp_col is None
        assert part_data.data.columns.tolist() == ["id", "age_part", "z"]

    def test_init_age_groups(self, df_part_age_grps):
        """Test initialization with age groups (IntervalIndex)."""
        df_part = df_part_age_grps
        part_data = ParticipantData(df_part=df_part, id_col="id", age_grp_col="age_grp")

        assert part_data.n == 5
        assert part_data.age_grp_col == "age_grp"
        assert part_data.age_col is None
        assert part_data.data.columns.tolist() == ["id", "age_grp_part", "z"]

    def test_init_single_strat(self, df_part_one_year):
        """Test initialization with a single stratification variable as string."""
        part_data = ParticipantData(
            df_part=df_part_one_year,
            id_col="id",
            age_col="age",
            strat_var_cols="sex",
        )

        # Should be converted to list internally
        assert part_data.strat_vars == ["sex"]
        assert part_data.data.columns.tolist() == ["id", "age_part", "sex_part", "z"]

    def test_init_multiple_strat(self, df_part_one_year):
        """Test initialization with multiple stratification variables."""
        df_part = df_part_one_year

        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )

        assert part_data.strat_vars == ["sex", "hhsize"]
        assert part_data.data.columns.tolist() == [
            "id",
            "age_part",
            "sex_part",
            "hhsize_part",
            "z",
        ]

    def test_init_with_repeat(self, df_part_one_year):
        """Test initialization with repeat interview variable."""
        df_part = df_part_one_year
        part_data = ParticipantData(
            df_part=df_part, id_col="id", age_col="age", repeat_col="repeat"
        )

        assert part_data.data.columns.tolist() == ["id", "age_part", "repeat_part", "z"]


class TestValidation:
    """Test validation logic and error handling."""

    def test_invalid_dataframe_type(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="df_part must be a pandas DataFrame"):
            ParticipantData(
                df_part=[1, 2, 3], id_col="id", age_col="age"  # Not a DataFrame
            )

    def test_missing_age_specification(self):
        """Test that neither age_col nor age_grp_col raises ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [25, 34, 45]})

        with pytest.raises(ValueError, match="Must specify exactly one"):
            ParticipantData(
                df_part=df,
                id_col="id",
                # Neither age_col nor age_grp_col specified
            )

    def test_both_age_specifications(self):
        """Test that both age_col and age_grp_col raises ValueError."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age": [25, 34, 45],
                "age_group": pd.IntervalIndex.from_tuples(
                    [(20, 30), (30, 40), (40, 50)]
                ),
            }
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            ParticipantData(
                df_part=df,
                id_col="id",
                age_col="age",
                age_grp_col="age_group",  # Both specified - invalid
            )

    def test_missing_id_column(self):
        """Test that missing ID column raises KeyError."""
        df = pd.DataFrame({"participant_id": [1, 2, 3], "age": [25, 34, 45]})

        with pytest.raises(KeyError, match="Missing participant ID column 'id'"):
            ParticipantData(
                df_part=df, id_col="id", age_col="age"  # Column doesn't exist
            )

    def test_missing_age_column(self):
        """Test that missing age column raises KeyError."""
        df = pd.DataFrame({"id": [1, 2, 3], "participant_age": [25, 34, 45]})

        with pytest.raises(KeyError, match="Missing participant age column 'age'"):
            ParticipantData(
                df_part=df, id_col="id", age_col="age"  # Column doesn't exist
            )

    def test_missing_strat_col(self):
        """Test that missing stratification variable raises KeyError."""
        df = pd.DataFrame(
            {"id": [1, 2, 3], "age": [25, 34, 45], "sex": ["M", "F", "M"]}
        )

        with pytest.raises(
            KeyError, match="strat_var_cols '\\['sex', 'hhsize'\\]' is specified"
        ):
            ParticipantData(
                df_part=df,
                id_col="id",
                age_col="age",
                strat_var_cols=["sex", "hhsize"],  # 'hhsize' doesn't exist
            )

    def test_duplicate_participant_ids(self):
        """Test that duplicate IDs raise ValueError."""
        df = pd.DataFrame(
            {"id": [1, 2, 2, 3], "age": [25, 34, 45, 52]}  # ID 2 is duplicated
        )

        with pytest.raises(ValueError, match="duplicate participant ID"):
            ParticipantData(df_part=df, id_col="id", age_col="age")

    def test_missing_values_in_id_column(self):
        """Test that missing values in ID column trigger warning and are dropped."""
        df = pd.DataFrame({"id": [1, 2, np.nan, 4], "age": [25, 34, 45, 52]})

        with pytest.warns(UserWarning, match="Dropped 1 row"):
            part_data = ParticipantData(df_part=df, id_col="id", age_col="age")
        # Check that row was dropped
        assert part_data.n == 3

    def test_missing_values_in_age_column(self):
        """Test that missing values in age column trigger warning and are dropped."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age": [25, np.nan, 45, 52]})

        with pytest.warns(UserWarning, match="Dropped 1 row"):
            part_data = ParticipantData(df_part=df, id_col="id", age_col="age")
        # Check that row was dropped
        assert part_data.n == 3

    def test_missing_values_in_strat_var(self):
        """Test that missing values in stratification variables trigger warning and are dropped."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "age": [25, 34, 45, 52],
                "sex": ["M", np.nan, "M", "F"],
            }
        )

        with pytest.warns(UserWarning, match="Dropped 1 row"):
            part_data = ParticipantData(
                df_part=df, id_col="id", age_col="age", strat_var_cols="sex"
            )
        # Check that row was dropped
        assert part_data.n == 3

    def test_negative_ages(self):
        """Test that negative ages raise ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age": [25, -5, 45, 52]})  # Negative age

        with pytest.raises(ValueError, match="negative values"):
            ParticipantData(df_part=df, id_col="id", age_col="age")

    def test_non_numeric_ages(self):
        """Test that non-numeric ages raise ValueError."""
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "age": ["25", "34", "45", "52"]}  # String ages
        )

        with pytest.raises(ValueError, match="must contain numeric values"):
            ParticipantData(df_part=df, id_col="id", age_col="age")


class TestProperties:
    """Test properties and accessor methods."""

    def test_data_property(self):
        """Test that data property returns the preprocessed DataFrame with 'z' column."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [25, 34, 45]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        returned_df = part_data.data
        assert isinstance(returned_df, pd.DataFrame)
        # Check that 'z' column was added during preprocessing
        assert "z" in returned_df.columns
        assert len(returned_df) == 3
        # Check that all 'z' values are 0
        assert (returned_df["z"] == 0).all()

    def test_n_property(self):
        """Test that n returns correct count."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "age": [25, 34, 45, 52, 61]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        assert part_data.n == 5

    def test_age_range_property(self):
        """Test that age_range returns correct min and max."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age": [18, 25, 65, 42]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        assert part_data.age_range == (18, 65)

    def test_age_range_with_age_groups_raises_error(self):
        """Test that age_range raises error when using age groups."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age_group": pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
            }
        )
        df["age_group"] = df["age_group"].astype("category")

        part_data = ParticipantData(df_part=df, id_col="id", age_grp_col="age_group")

        with pytest.raises(ValueError, match="only available when using 'age_col'"):
            _ = part_data.age_range

    def test_strat_vars_property_empty(self):
        """Test strat_vars when no variables specified."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [25, 34, 45]})
        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        assert part_data.strat_vars == []

    def test_strat_vars_property_with_vars(self):
        """Test strat_vars with multiple variables."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age": [25, 34, 45],
                "gender": ["M", "F", "M"],
                "region": ["A", "B", "C"],
            }
        )

        part_data = ParticipantData(
            df_part=df, id_col="id", age_col="age", strat_var_cols=["gender", "region"]
        )

        assert part_data.strat_vars == ["gender", "region"]


class TestMethods:
    """Test methods for data analysis and summarization."""

    def test_get_sample_sizes(self):
        """Test age distribution computation."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "age": [25, 25, 34, 34, 34, 45]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        sample_sizes = part_data.get_sample_sizes()

        assert isinstance(sample_sizes, pd.DataFrame)
        assert np.array_equal(sample_sizes["age_part"].values, np.array([25, 34, 45]))
        assert np.array_equal(sample_sizes["N"].values, np.array([2, 3, 1]))

    def test_get_sample_sizes_with_age_groups(self):
        """Test age distribution with age groups."""
        intervals = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (5, 10), (10, 15)])
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age_group": intervals})
        df["age_group"] = df["age_group"].astype("category")

        part_data = ParticipantData(df_part=df, id_col="id", age_grp_col="age_group")

        sample_sizes = part_data.get_sample_sizes()

        assert isinstance(sample_sizes, pd.DataFrame)
        assert len(sample_sizes) == 3  # Three unique intervals

    def test_summary_method_with_age_col(self):
        """Test summary method with exact ages."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "age": [18, 25, 45, 65],
                "gender": ["M", "F", "M", "F"],
            }
        )

        part_data = ParticipantData(
            df_part=df, id_col="id", age_col="age", strat_var_cols="gender"
        )

        summary = part_data.summary()

        assert isinstance(summary, dict)
        assert summary["n"] == 4
        assert summary["id_col"] == "id"
        assert summary["age_col"] == "age"
        assert summary["age_grp_col"] is None
        assert summary["age_range"] == (18, 65)
        assert summary["strat_vars"] == ["gender"]

    def test_summary_method_with_age_groups(self):
        """Test summary method with age groups."""
        df = pd.DataFrame(
            {
                "pid": [1, 2, 3],
                "age_group": pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
            }
        )
        df["age_group"] = df["age_group"].astype("category")

        part_data = ParticipantData(df_part=df, id_col="pid", age_grp_col="age_group")

        summary = part_data.summary()

        assert summary["n"] == 3
        assert summary["id_col"] == "pid"
        assert summary["age_col"] is None
        assert summary["age_grp_col"] == "age_group"
        assert "age_range" not in summary  # Not available for age groups
        assert summary["strat_vars"] == []

    def test_summary_without_stratification(self):
        """Test summary with no stratification variables."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [25, 34, 45]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        summary = part_data.summary()

        assert summary["strat_vars"] == []

    def test_get_strat_var_schema(self, df_part_one_year):
        part_data = ParticipantData(
            df_part_one_year,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )

        schema = part_data.get_strat_var_schema()
        assert schema["sex"] == {"categories": ["F", "M"], "codes": [0, 1]}
        assert schema["hhsize"] == {
            "categories": ["1", "2", "3", "4", "5+"],
            "codes": [0, 1, 2, 3, 4],
        }


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_participant(self):
        """Test with a single participant."""
        df = pd.DataFrame({"id": [1], "age": [25]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        assert part_data.n == 1
        assert part_data.age_range == (25, 25)

    def test_age_zero(self):
        """Test that age 0 is valid."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [0, 5, 10]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        assert part_data.age_range == (0, 10)

    def test_float_ages(self):
        """Test that float ages are valid."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [25.5, 34.2, 45.8]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        assert part_data.n == 3
        assert part_data.age_range == (25.5, 45.8)

    def test_non_sequential_ids(self):
        """Test that non-sequential IDs work correctly."""
        df = pd.DataFrame({"id": [100, 205, 37, 999], "age": [25, 34, 45, 52]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        assert part_data.n == 4

    def test_string_ids(self):
        """Test that string IDs work correctly."""
        df = pd.DataFrame({"id": ["A001", "B002", "C003"], "age": [25, 34, 45]})

        part_data = ParticipantData(df_part=df, id_col="id", age_col="age")

        assert part_data.n == 3
