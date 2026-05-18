"""
Tests for ContactData dataclass.

This module tests the validation, properties, and methods of the ContactData class,
ensuring robust handling of contact survey data.

Test organisation
-----------------
1. TestCoreContracts  — column renaming/standardisation, categorical conversion,
                        NaN dropping, 'y' column auto-creation, original-df immutability.
2. TestInputValidation — TypeError/ValueError/KeyError on bad inputs.
3. TestAccessorMethods — properties and methods that expose processed data.
4. TestEdgeCases       — genuine boundary conditions (age=0, single contact).
"""

import numpy as np
import pandas as pd
import pytest

from ..containers._ContactData import ContactData
from .fixtures import df_cnt_age_grps, df_cnt_age_min_max, df_cnt_one_year


# =====================
# 1. Core Contracts
# =====================


class TestCoreContracts:
    """Test the fundamental data-pipeline guarantees of ContactData."""

    def test_column_renaming_with_age_col(self):
        """Columns are renamed: id_col→'id', age_col→'age_cnt'."""
        df = pd.DataFrame({"pid": [1, 2, 3], "contact_age": [25, 34, 45]})
        cnt_data = ContactData(data=df, id_col="pid", age_col="contact_age")
        assert "id" in cnt_data.data.columns
        assert "age_cnt" in cnt_data.data.columns
        assert "pid" not in cnt_data.data.columns
        assert "contact_age" not in cnt_data.data.columns

    def test_column_renaming_with_age_grp_col(self, df_cnt_age_grps):
        """age_grp_col is renamed to 'age_grp_cnt'."""
        cnt_data = ContactData(data=df_cnt_age_grps, id_col="id", age_grp_col="age_grp_cnt")
        assert "age_grp_cnt" in cnt_data.data.columns

    def test_column_renaming_with_age_min_max_cols(self, df_cnt_age_min_max):
        """age_min_col/age_max_col are renamed to 'age_min_cnt'/'age_max_cnt'."""
        cnt_data = ContactData(
            data=df_cnt_age_min_max,
            id_col="id",
            age_min_col="age_min",
            age_max_col="age_max",
        )
        assert "age_min_cnt" in cnt_data.data.columns
        assert "age_max_cnt" in cnt_data.data.columns
        assert "age_min" not in cnt_data.data.columns
        assert "age_max" not in cnt_data.data.columns
        assert "y" in cnt_data.data.columns

    def test_column_renaming_strat_vars(self):
        """Stratification variables are renamed with '_cnt' suffix."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "contact_age": [25, 34, 45],
                "setting": ["home", "work", "school"],
            }
        )
        cnt_data = ContactData(
            data=df, id_col="id", age_col="contact_age", strat_var_cols="setting"
        )
        assert "setting_cnt" in cnt_data.data.columns
        assert "setting" not in cnt_data.data.columns

    def test_y_column_added_automatically(self):
        """'y' column is added with value 1 when not present in input."""
        df = pd.DataFrame({"id": [1, 2, 3], "age_cnt": [25, 34, 45]})
        cnt_data = ContactData(data=df, id_col="id", age_col="age_cnt")
        assert "y" in cnt_data.data.columns
        assert (cnt_data.data["y"] == 1).all()

    def test_categorical_conversion(self):
        """Object-type stratification columns are converted to categorical dtype."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age_cnt": [25, 34, 45],
                "setting": ["home", "work", "school"],
            }
        )
        cnt_data = ContactData(
            data=df, id_col="id", age_col="age_cnt", strat_var_cols="setting"
        )
        assert cnt_data.data["setting_cnt"].dtype.name == "category"

    @pytest.mark.parametrize(
        "nan_col, df_factory",
        [
            (
                "id",
                lambda: pd.DataFrame({"id": [1, 2, np.nan, 4], "age_cnt": [25, 34, 45, 52]}),
            ),
            (
                "age_cnt",
                lambda: pd.DataFrame({"id": [1, 2, 3, 4], "age_cnt": [25, np.nan, 45, 52]}),
            ),
            (
                "setting",
                lambda: pd.DataFrame(
                    {
                        "id": [1, 2, 3, 4],
                        "age_cnt": [25, 34, 45, 52],
                        "setting": ["home", np.nan, "work", "school"],
                    }
                ),
            ),
        ],
    )
    def test_nan_rows_dropped_with_warning(self, nan_col, df_factory):
        """Rows with NaN in any required column are dropped with a UserWarning."""
        df = df_factory()
        kwargs = dict(id_col="id", age_col="age_cnt")
        if nan_col == "setting":
            kwargs["strat_var_cols"] = "setting"

        with pytest.warns(UserWarning, match="Dropped 1 contact record"):
            cnt_data = ContactData(data=df, **kwargs)
        assert cnt_data.n == 3

    def test_original_dataframe_not_mutated(self, df_cnt_one_year):
        """ContactData works on a copy; the caller's DataFrame is unchanged."""
        original_cols = df_cnt_one_year.columns.tolist()
        original_shape = df_cnt_one_year.shape
        ContactData(
            data=df_cnt_one_year,
            id_col="id",
            age_col="age_cnt",
            strat_var_cols=["sex_cnt", "hhsize_cnt"],
        )
        assert df_cnt_one_year.columns.tolist() == original_cols
        assert df_cnt_one_year.shape == original_shape


# =====================
# 2. Input Validation
# =====================


class TestInputValidation:
    """Test TypeError/ValueError/KeyError on bad inputs."""

    def test_invalid_dataframe_type(self):
        """Non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            ContactData(data=[1, 2, 3], id_col="id", age_col="contact_age")  # type: ignore

    def test_missing_age_specification(self, df_cnt_one_year):
        """Neither age_col, age_grp_col, nor age_min/max raises ValueError."""
        with pytest.raises(ValueError, match="Must specify exactly one"):
            ContactData(data=df_cnt_one_year, id_col="id")

    def test_both_age_specifications(self):
        """Both age_col and age_grp_col raises ValueError."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age_cnt": [25, 34, 45],
                "age_grp_cnt": pd.IntervalIndex.from_tuples(
                    [(20, 30), (30, 40), (40, 50)]
                ),
            }
        )
        with pytest.raises(ValueError, match="Age specification forms are mutually exclusive"):
            ContactData(data=df, id_col="id", age_col="age_cnt", age_grp_col="age_grp_cnt")

    def test_age_min_without_age_max(self):
        """age_min_col without age_max_col raises ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3], "age_min": [20, 30, 40]})
        with pytest.raises(ValueError, match="Both 'age_min_col' and 'age_max_col' must be specified"):
            ContactData(data=df, id_col="id", age_min_col="age_min")

    def test_age_min_max_and_age_col_raises(self):
        """Providing age_col and age_min_col together raises ValueError."""
        df = pd.DataFrame({"id": [1, 2], "age": [25, 34], "age_min": [20, 30], "age_max": [29, 39]})
        with pytest.raises(ValueError, match="Age specification forms are mutually exclusive"):
            ContactData(data=df, id_col="id", age_col="age", age_min_col="age_min", age_max_col="age_max")

    def test_missing_id_column(self):
        """Missing ID column raises KeyError with informative message."""
        df = pd.DataFrame({"pid": [1, 2, 3], "age_cnt": [25, 34, 45]})
        with pytest.raises(KeyError, match="Missing participant ID column 'id'"):
            ContactData(data=df, id_col="id", age_col="age_cnt")

    def test_missing_age_column(self):
        """Missing age column raises KeyError with informative message."""
        df = pd.DataFrame({"id": [1, 2, 3], "age_of_contact": [25, 34, 45]})
        with pytest.raises(KeyError, match="Missing contact age column 'age_cnt'"):
            ContactData(data=df, id_col="id", age_col="age_cnt")

    def test_missing_strat_col(self, df_cnt_one_year):
        """Missing stratification column raises KeyError with informative message."""
        with pytest.raises(KeyError, match="Missing contact stratification variable"):
            ContactData(
                data=df_cnt_one_year,
                id_col="id",
                age_col="age_cnt",
                strat_var_cols=["sex_cnt", "workstat_cnt"],  # 'workstat' doesn't exist
            )

    def test_negative_ages(self):
        """Negative contact ages raise ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "contact_age": [25, -5, 45, 52]})
        with pytest.raises(ValueError, match="negative values"):
            ContactData(data=df, id_col="id", age_col="contact_age")

    def test_non_numeric_ages(self):
        """Non-numeric contact ages raise ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "contact_age": ["25", "34", "45", "52"]})
        with pytest.raises(ValueError, match="must contain numeric values"):
            ContactData(data=df, id_col="id", age_col="contact_age")


# =====================
# 3. Accessor Methods
# =====================


class TestAccessorMethods:
    """Test properties and methods that expose processed data in specific formats."""

    def test_basic_counts(self, df_cnt_one_year):
        """n and n_part return total contacts and unique participants respectively."""
        cnt_data = ContactData(data=df_cnt_one_year, id_col="id", age_col="age_cnt")
        assert cnt_data.n == 6
        assert cnt_data.n_part == 5

    def test_age_range(self, df_cnt_one_year):
        """age_range returns correct (min, max) tuple."""
        cnt_data = ContactData(df_cnt_one_year, id_col="id", age_col="age_cnt")
        assert cnt_data.age_range == (30, 80)

    def test_age_range_raises_with_age_groups(self, df_cnt_age_grps):
        """age_range raises ValueError when age_grp_col is used."""
        cnt_data = ContactData(data=df_cnt_age_grps, id_col="id", age_grp_col="age_grp_cnt")
        with pytest.raises(ValueError, match="only available when using 'age_col'"):
            _ = cnt_data.age_range

    def test_get_strat_vars_empty(self, df_cnt_one_year):
        """get_strat_vars() returns empty list when no stratification variables specified."""
        cnt_data = ContactData(data=df_cnt_one_year, id_col="id", age_col="age_cnt")
        assert cnt_data.get_strat_vars() == []

    def test_get_strat_vars_with_suffix(self, df_cnt_one_year):
        """get_strat_vars() returns names with and without '_cnt' suffix correctly."""
        cnt_data = ContactData(
            data=df_cnt_one_year,
            id_col="id",
            age_col="age_cnt",
            strat_var_cols=["sex_cnt", "hhsize_cnt"],
        )
        assert cnt_data.get_strat_vars(suffix=False) == ["sex", "hhsize"]
        assert cnt_data.get_strat_vars(suffix=True) == ["sex_cnt", "hhsize_cnt"]

    def test_get_strat_var_schema(self, df_cnt_one_year):
        """get_strat_var_schema() returns correct categories and integer codes."""
        cnt_data = ContactData(
            df_cnt_one_year,
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


# =====================
# 4. Edge Cases
# =====================


class TestEdgeCases:
    """Test genuine boundary conditions."""

    def test_single_contact(self):
        """Single-row DataFrame is valid; age_range is a degenerate tuple."""
        df = pd.DataFrame({"id": [1], "contact_age": [25]})
        cnt_data = ContactData(data=df, id_col="id", age_col="contact_age")
        assert cnt_data.n == 1
        assert cnt_data.n_part == 1
        assert cnt_data.age_range == (25, 25)

    def test_age_zero(self):
        """Contact age 0 is a valid non-negative age."""
        df = pd.DataFrame({"id": [1, 2, 3], "contact_age": [0, 5, 10]})
        cnt_data = ContactData(data=df, id_col="id", age_col="contact_age")
        assert cnt_data.age_range == (0, 10)
