"""
Tests for ContactData dataclass.

This module tests the validation, properties, and methods of the ContactData class,
ensuring robust handling of contact survey data.
"""

import numpy as np
import pandas as pd
import pytest

from ..containers._ContactData import ContactData
from .fixtures import df_cnt_age_grps, df_cnt_one_year

# =====================
# Test Basic Behaviour
# =====================


class TestInit:
    """Test initialization and validation of ContactData."""

    def test_basic(self, df_cnt_one_year):
        """Test basic initialization with exact contact ages."""

        cnt_data = ContactData(data=df_cnt_one_year, id_col="id", age_col="age_cnt")

        assert cnt_data.n == 6
        assert cnt_data.n_part == 5
        assert cnt_data.age_col == "age_cnt"
        assert cnt_data.age_grp_col is None

    def test_basic_age_grp(self, df_cnt_age_grps):
        """Test initialization with age groups (IntervalIndex)."""
        df_cnt = df_cnt_age_grps
        cnt_data = ContactData(data=df_cnt, id_col="id", age_grp_col="age_grp_cnt")

        assert cnt_data.n == 6
        assert cnt_data.n_part == 5
        assert cnt_data.age_grp_col == "age_grp_cnt"
        assert cnt_data.age_col is None

    def test_single_strat_var(self, df_cnt_one_year):
        """Test initialization with a single stratification variable as string."""

        cnt_data = ContactData(
            data=df_cnt_one_year,
            id_col="id",
            age_col="age_cnt",
            strat_var_cols="sex_cnt",
        )

        assert cnt_data.get_strat_vars(suffix=False) == ["sex"]
        assert cnt_data.get_strat_vars(suffix=True) == ["sex_cnt"]

    def test_multiple_strat_vars(self, df_cnt_one_year):
        """Test initialization with multiple stratification variables."""
        df_cnt = df_cnt_one_year

        cnt_data = ContactData(
            data=df_cnt,
            id_col="id",
            age_col="age_cnt",
            strat_var_cols=["sex_cnt", "hhsize_cnt"],
        )

        assert cnt_data.get_strat_vars(suffix=False) == ["sex", "hhsize"]
        assert cnt_data.get_strat_vars(suffix=True) == ["sex_cnt", "hhsize_cnt"]


# =====================
# Test Validation
# =====================


class TestValidation:
    """Test validation logic and error handling."""

    def test_invalid_input(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            ContactData(
                data=[1, 2, 3],  # type: ignore
                id_col="id",
                age_col="contact_age",  # Not a DataFrame
            )

    def test_missing_age_specification(self, df_cnt_one_year):
        """Test that neither age_col nor age_grp_col raises ValueError."""

        with pytest.raises(ValueError, match="Must specify exactly one"):
            ContactData(
                data=df_cnt_one_year,
                id_col="id",
                # Neither age_col nor age_grp_col specified
            )

    def test_both_age_specifications(self):
        """Test that both age_col and age_grp_col raises ValueError."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age_cnt": [25, 34, 45],
                "age_grp_cnt": pd.IntervalIndex.from_tuples(
                    [(20, 30), (30, 40), (40, 50)]
                ),
            }
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            ContactData(
                data=df,
                id_col="id",
                age_col="age_cnt",
                age_grp_col="age_grp_cnt",  # Both specified - invalid
            )

    def test_missing_id_column(self):
        """Test that missing ID column raises KeyError."""
        df = pd.DataFrame({"pid": [1, 2, 3], "age_cnt": [25, 34, 45]})

        with pytest.raises(KeyError, match="Missing participant ID column 'id'"):
            ContactData(data=df, id_col="id", age_col="age_cnt")  # Column doesn't exist

    def test_missing_age_column(self):
        """Test that missing age column raises KeyError."""
        df = pd.DataFrame({"id": [1, 2, 3], "age_of_contact": [25, 34, 45]})

        with pytest.raises(KeyError, match="Missing contact age column 'age_cnt'"):
            ContactData(data=df, id_col="id", age_col="age_cnt")  # Column doesn't exist

    def test_missing_strat_col(self, df_cnt_one_year):
        """Test that missing stratification variable raises KeyError."""
        with pytest.raises(KeyError, match="Missing contact stratification variable"):
            ContactData(
                data=df_cnt_one_year,
                id_col="id",
                age_col="age_cnt",
                strat_var_cols=["sex_cnt", "workstat_cnt"],  # 'workstat' doesn't exist
            )

    def test_missing_values_in_id_column(self):
        """Test that missing values in ID column trigger warning and are dropped."""
        df = pd.DataFrame({"id": [1, 2, np.nan, 4], "age_cnt": [25, 34, 45, 52]})

        with pytest.warns(UserWarning, match="Dropped 1 contact record"):
            cnt_data = ContactData(data=df, id_col="id", age_col="age_cnt")
        # Check that row was dropped
        assert cnt_data.n == 3

    def test_missing_values_in_age_column(self):
        """Test that missing values in age column trigger warning and are dropped."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age_cnt": [25, np.nan, 45, 52]})

        with pytest.warns(UserWarning, match="Dropped 1 contact record"):
            cnt_data = ContactData(data=df, id_col="id", age_col="age_cnt")
        # Check that row was dropped
        assert cnt_data.n == 3

    def test_missing_values_strat_var(self):
        """Test that missing values in stratification variables trigger warning and are dropped."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "age_cnt": [25, 34, 45, 52],
                "setting": ["home", np.nan, "work", "school"],
            }
        )

        with pytest.warns(UserWarning, match="Dropped 1 contact record"):
            cnt_data = ContactData(
                data=df, id_col="id", age_col="age_cnt", strat_var_cols="setting"
            )
        # Check that row was dropped
        assert cnt_data.n == 3

    def test_negative_ages(self):
        """Test that negative ages raise ValueError."""
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "contact_age": [25, -5, 45, 52]}  # Negative age
        )

        with pytest.raises(ValueError, match="negative values"):
            ContactData(data=df, id_col="id", age_col="contact_age")

    def test_non_numeric_ages(self):
        """Test that non-numeric ages raise ValueError."""
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "contact_age": ["25", "34", "45", "52"]}  # String ages
        )

        with pytest.raises(ValueError, match="must contain numeric values"):
            ContactData(data=df, id_col="id", age_col="contact_age")


class TestProperties:
    """Test properties and accessor methods."""

    def test_data_property(self):
        """Test that data property returns the preprocessed DataFrame with 'y' column."""
        df = pd.DataFrame({"id": [1, 2, 3], "age_cnt": [25, 34, 45]})

        cnt_data = ContactData(data=df, id_col="id", age_col="age_cnt")

        returned_df = cnt_data.data
        assert isinstance(returned_df, pd.DataFrame)
        # Check that 'y' column was added during preprocessing
        assert "y" in returned_df.columns
        assert len(returned_df) == 3
        # Check that all 'y' values are 1
        assert (returned_df["y"] == 1).all()

    def test_age_range(self, df_cnt_one_year):
        """Test that age_range returns correct min and max."""
        df_cnt = df_cnt_one_year
        cnt_data = ContactData(df_cnt, id_col="id", age_col="age_cnt")

        assert cnt_data.age_range == (30, 80)

    def test_strat_vars_empty(self, df_cnt_one_year):
        """Test get_strat_vars() when no variables specified."""
        df_cnt = df_cnt_one_year
        cnt_data = ContactData(data=df_cnt, id_col="id", age_col="age_cnt")
        assert cnt_data.get_strat_vars() == []

    def test_strat_vars_with_vars(self, df_cnt_one_year):
        """Test get_strat_vars() with multiple variables."""
        df_cnt = df_cnt_one_year

        cnt_data = ContactData(
            data=df_cnt,
            id_col="id",
            age_col="age_cnt",
            strat_var_cols=["sex_cnt", "hhsize_cnt"],
        )

        assert cnt_data.get_strat_vars(suffix=True) == ["sex_cnt", "hhsize_cnt"]


class TestMethods:
    """Test methods for data analysis and summarization."""

    def test_get_strat_vars(self, df_cnt_one_year):
        df_cnt = df_cnt_one_year

        cnt_data = ContactData(
            df_cnt,
            id_col="id",
            age_col="age_cnt",
            strat_var_cols=["sex_cnt", "hhsize_cnt"],
        )
        assert cnt_data.get_strat_vars(suffix=False) == ["sex", "hhsize"]
        assert cnt_data.get_strat_vars(suffix=True) == ["sex_cnt", "hhsize_cnt"]

    def test_get_strat_var_schema(self, df_cnt_one_year):
        """Test get_strat_var_schema returns correct categories and codes."""
        df_cnt = df_cnt_one_year

        cnt_data = ContactData(
            df_cnt,
            id_col="id",
            age_col="age_cnt",
            strat_var_cols=["sex_cnt", "hhsize_cnt"],
        )
        schema = cnt_data.get_strat_var_schema()
        assert schema["sex"] == {"categories": ["F", "M"], "codes": [0, 1]}
        assert schema["hhsize"] == {
            "categories": ["1", "2", "3", "4", "5+"],
            "codes": [0, 1, 2, 3, 4],
        }


class TestCEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_contact(self):
        """Test with a single contact."""
        df = pd.DataFrame({"id": [1], "contact_age": [25]})

        cnt_data = ContactData(data=df, id_col="id", age_col="contact_age")

        assert cnt_data.n == 1
        assert cnt_data.n_part == 1
        assert cnt_data.age_range == (25, 25)

    def test_age_zero(self):
        """Test that age 0 is valid."""
        df = pd.DataFrame({"id": [1, 2, 3], "contact_age": [0, 5, 10]})

        cnt_data = ContactData(data=df, id_col="id", age_col="contact_age")

        assert cnt_data.age_range == (0, 10)

    def test_large_dataset(self):
        """Test with a larger dataset for performance."""
        n = 10000
        df = pd.DataFrame(
            {
                "id": np.random.randint(1, 1000, n),  # 1000 participants
                "age_cnt": np.random.randint(0, 100, n),
                "setting": np.random.choice(["home", "work", "school", "other"], n),
            }
        )

        cnt_data = ContactData(
            data=df, id_col="id", age_col="age_cnt", strat_var_cols="setting"
        )

        assert cnt_data.n == n
        assert cnt_data.n_part <= 1000
        assert len(cnt_data.get_strat_vars()) == 1
