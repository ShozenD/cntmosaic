"""
Tests for ContactData dataclass.

This module tests the validation, properties, and methods of the ContactData class,
ensuring robust handling of contact survey data.
"""

import numpy as np
import pandas as pd
import pytest

from ..containers._ContactData import ContactData


class TestContactDataInitialization:
    """Test initialization and validation of ContactData."""

    def test_basic_initialization_with_age_col(self):
        """Test basic initialization with exact contact ages."""
        df = pd.DataFrame(
            {
                "participant_id": [1, 1, 2, 3],
                "contact_age": [30, 45, 25, 60],
                "setting": ["home", "work", "home", "other"],
            }
        )

        cnt_data = ContactData(
            df_cnt=df, id_col="participant_id", age_col="contact_age"
        )

        assert cnt_data.n_contacts == 4
        assert cnt_data.n_unique_participants == 3
        assert cnt_data.age_col == "contact_age"
        assert cnt_data.age_grp_col is None

    def test_initialization_with_age_groups(self):
        """Test initialization with age groups (IntervalIndex)."""
        df = pd.DataFrame(
            {
                "pid": [1, 1, 2],
                "age_group": pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
                "setting": ["home", "school", "work"],
            }
        )
        df["age_group"] = df["age_group"].astype("category")

        cnt_data = ContactData(df_cnt=df, id_col="pid", age_grp_col="age_group")

        assert cnt_data.n_contacts == 3
        assert cnt_data.age_grp_col == "age_group"
        assert cnt_data.age_col is None

    def test_initialization_with_single_stratification_var(self):
        """Test initialization with a single stratification variable as string."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "contact_age": [25, 34, 45],
                "setting": ["home", "work", "school"],
            }
        )

        cnt_data = ContactData(
            df_cnt=df, id_col="id", age_col="contact_age", strat_vars="setting"
        )

        # Should be converted to list internally
        assert cnt_data.stratification_vars == ["setting"]

    def test_initialization_with_multiple_stratification_vars(self):
        """Test initialization with multiple stratification variables."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "contact_age": [25, 34, 45],
                "setting": ["home", "work", "school"],
                "duration": ["short", "long", "medium"],
                "physical": ["yes", "no", "yes"],
            }
        )

        cnt_data = ContactData(
            df_cnt=df,
            id_col="id",
            age_col="contact_age",
            strat_vars=["setting", "duration", "physical"],
        )

        assert cnt_data.stratification_vars == ["setting", "duration", "physical"]

    def test_initialization_without_stratification_vars(self):
        """Test initialization without stratification variables."""
        df = pd.DataFrame({"id": [1, 2, 3], "contact_age": [25, 34, 45]})

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.stratification_vars == []

    def test_duplicate_participant_ids_allowed(self):
        """Test that duplicate participant IDs are allowed (multiple contacts per person)."""
        df = pd.DataFrame(
            {
                "id": [
                    1,
                    1,
                    1,
                    2,
                    2,
                ],  # Participant 1 reports 3 contacts, participant 2 reports 2
                "contact_age": [25, 30, 35, 40, 45],
            }
        )

        # Should not raise an error
        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.n_contacts == 5
        assert cnt_data.n_unique_participants == 2


class TestContactDataValidation:
    """Test validation logic and error handling."""

    def test_invalid_dataframe_type(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="df_cnt must be a pandas DataFrame"):
            ContactData(
                df_cnt=[1, 2, 3], id_col="id", age_col="contact_age"  # Not a DataFrame
            )

    def test_missing_age_specification(self):
        """Test that neither age_col nor age_grp_col raises ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3], "contact_age": [25, 34, 45]})

        with pytest.raises(ValueError, match="Must specify exactly one"):
            ContactData(
                df_cnt=df,
                id_col="id",
                # Neither age_col nor age_grp_col specified
            )

    def test_both_age_specifications(self):
        """Test that both age_col and age_grp_col raises ValueError."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "contact_age": [25, 34, 45],
                "age_group": pd.IntervalIndex.from_tuples(
                    [(20, 30), (30, 40), (40, 50)]
                ),
            }
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            ContactData(
                df_cnt=df,
                id_col="id",
                age_col="contact_age",
                age_grp_col="age_group",  # Both specified - invalid
            )

    def test_missing_id_column(self):
        """Test that missing ID column raises KeyError."""
        df = pd.DataFrame({"participant_id": [1, 2, 3], "contact_age": [25, 34, 45]})

        with pytest.raises(KeyError, match="Missing participant ID column 'id'"):
            ContactData(
                df_cnt=df, id_col="id", age_col="contact_age"  # Column doesn't exist
            )

    def test_missing_age_column(self):
        """Test that missing age column raises KeyError."""
        df = pd.DataFrame({"id": [1, 2, 3], "age_of_contact": [25, 34, 45]})

        with pytest.raises(KeyError, match="Missing contact age column 'contact_age'"):
            ContactData(
                df_cnt=df, id_col="id", age_col="contact_age"  # Column doesn't exist
            )

    def test_missing_stratification_column(self):
        """Test that missing stratification variable raises KeyError."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "contact_age": [25, 34, 45],
                "setting": ["home", "work", "school"],
            }
        )

        with pytest.raises(KeyError, match="Missing contact stratification variable"):
            ContactData(
                df_cnt=df,
                id_col="id",
                age_col="contact_age",
                strat_vars=["setting", "duration"],  # 'duration' doesn't exist
            )

    def test_missing_values_in_id_column(self):
        """Test that missing values in ID column trigger warning and are dropped."""
        df = pd.DataFrame({"id": [1, 2, np.nan, 4], "contact_age": [25, 34, 45, 52]})

        with pytest.warns(UserWarning, match="Dropped 1 contact record"):
            cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")
        # Check that row was dropped
        assert cnt_data.n_contacts == 3

    def test_missing_values_in_age_column(self):
        """Test that missing values in age column trigger warning and are dropped."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "contact_age": [25, np.nan, 45, 52]})

        with pytest.warns(UserWarning, match="Dropped 1 contact record"):
            cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")
        # Check that row was dropped
        assert cnt_data.n_contacts == 3

    def test_missing_values_in_stratification_var(self):
        """Test that missing values in stratification variables trigger warning and are dropped."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "contact_age": [25, 34, 45, 52],
                "setting": ["home", np.nan, "work", "school"],
            }
        )

        with pytest.warns(UserWarning, match="Dropped 1 contact record"):
            cnt_data = ContactData(
                df_cnt=df, id_col="id", age_col="contact_age", strat_vars="setting"
            )
        # Check that row was dropped
        assert cnt_data.n_contacts == 3

    def test_negative_ages(self):
        """Test that negative ages raise ValueError."""
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "contact_age": [25, -5, 45, 52]}  # Negative age
        )

        with pytest.raises(ValueError, match="negative values"):
            ContactData(df_cnt=df, id_col="id", age_col="contact_age")

    def test_non_numeric_ages(self):
        """Test that non-numeric ages raise ValueError."""
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "contact_age": ["25", "34", "45", "52"]}  # String ages
        )

        with pytest.raises(ValueError, match="must contain numeric values"):
            ContactData(df_cnt=df, id_col="id", age_col="contact_age")


class TestContactDataProperties:
    """Test properties and accessor methods."""

    def test_data_property(self):
        """Test that data property returns the preprocessed DataFrame with 'y' column."""
        df = pd.DataFrame({"id": [1, 2, 3], "contact_age": [25, 34, 45]})

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        returned_df = cnt_data.data
        assert isinstance(returned_df, pd.DataFrame)
        # Check that 'y' column was added during preprocessing
        assert "y" in returned_df.columns
        assert len(returned_df) == 3
        # Check that all 'y' values are 1
        assert (returned_df["y"] == 1).all()

    def test_n_contacts_property(self):
        """Test that n_contacts returns correct count."""
        df = pd.DataFrame({"id": [1, 1, 2, 2, 3], "contact_age": [25, 30, 35, 40, 45]})

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.n_contacts == 5

    def test_n_unique_participants_property(self):
        """Test that n_unique_participants returns correct count."""
        df = pd.DataFrame(
            {"id": [1, 1, 1, 2, 2, 3], "contact_age": [25, 30, 35, 40, 45, 50]}
        )

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.n_unique_participants == 3

    def test_age_range_property(self):
        """Test that age_range returns correct min and max."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "contact_age": [18, 25, 65, 42]})

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.age_range == (18, 65)

    def test_age_range_with_age_groups_raises_error(self):
        """Test that age_range raises error when using age groups."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age_group": pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
            }
        )
        df["age_group"] = df["age_group"].astype("category")

        cnt_data = ContactData(df_cnt=df, id_col="id", age_grp_col="age_group")

        with pytest.raises(ValueError, match="only available when using 'age_col'"):
            _ = cnt_data.age_range

    def test_stratification_vars_property_empty(self):
        """Test stratification_vars when no variables specified."""
        df = pd.DataFrame({"id": [1, 2, 3], "contact_age": [25, 34, 45]})

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.stratification_vars == []

    def test_stratification_vars_property_with_vars(self):
        """Test stratification_vars with multiple variables."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "contact_age": [25, 34, 45],
                "setting": ["home", "work", "school"],
                "duration": ["short", "long", "medium"],
            }
        )

        cnt_data = ContactData(
            df_cnt=df,
            id_col="id",
            age_col="contact_age",
            strat_vars=["setting", "duration"],
        )

        assert cnt_data.stratification_vars == ["setting", "duration"]


class TestContactDataMethods:
    """Test methods for data analysis and summarization."""

    def test_get_age_distribution(self):
        """Test age distribution computation."""
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4, 5, 6], "contact_age": [25, 25, 34, 34, 34, 45]}
        )

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        age_dist = cnt_data.get_age_distribution()

        assert isinstance(age_dist, pd.Series)
        assert age_dist[25] == 2
        assert age_dist[34] == 3
        assert age_dist[45] == 1

    def test_get_age_distribution_with_age_groups(self):
        """Test age distribution with age groups."""
        intervals = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (5, 10), (10, 15)])
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age_group": intervals})
        df["age_group"] = df["age_group"].astype("category")

        cnt_data = ContactData(df_cnt=df, id_col="id", age_grp_col="age_group")

        age_dist = cnt_data.get_age_distribution()

        assert isinstance(age_dist, pd.Series)
        assert len(age_dist) == 3  # Three unique intervals

    def test_get_contacts_per_participant(self):
        """Test contacts per participant computation."""
        df = pd.DataFrame(
            {"id": [1, 1, 1, 2, 2, 3], "contact_age": [25, 30, 35, 40, 45, 50]}
        )

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        contacts_per_part = cnt_data.get_contacts_per_participant()

        assert isinstance(contacts_per_part, pd.Series)
        assert contacts_per_part[1] == 3
        assert contacts_per_part[2] == 2
        assert contacts_per_part[3] == 1

    def test_summary_method_with_age_col(self):
        """Test summary method with exact ages."""
        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 3],
                "contact_age": [18, 25, 45, 65],
                "setting": ["home", "work", "home", "other"],
            }
        )

        cnt_data = ContactData(
            df_cnt=df, id_col="id", age_col="contact_age", strat_vars="setting"
        )

        summary = cnt_data.summary()

        assert isinstance(summary, dict)
        assert summary["n_contacts"] == 4
        assert summary["n_unique_participants"] == 3
        assert summary["mean_contacts_per_participant"] == pytest.approx(4 / 3)
        assert summary["id_col"] == "id"
        assert summary["age_col"] == "contact_age"
        assert summary["age_grp_col"] is None
        assert summary["age_range"] == (18, 65)
        assert summary["stratification_vars"] == ["setting"]
        assert summary["n_stratification_vars"] == 1

    def test_summary_method_with_age_groups(self):
        """Test summary method with age groups."""
        df = pd.DataFrame(
            {
                "pid": [1, 2, 3],
                "age_group": pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
            }
        )
        df["age_group"] = df["age_group"].astype("category")

        cnt_data = ContactData(df_cnt=df, id_col="pid", age_grp_col="age_group")

        summary = cnt_data.summary()

        assert summary["n_contacts"] == 3
        assert summary["n_unique_participants"] == 3
        assert summary["id_col"] == "pid"
        assert summary["age_col"] is None
        assert summary["age_grp_col"] == "age_group"
        assert "age_range" not in summary  # Not available for age groups
        assert summary["stratification_vars"] == []
        assert summary["n_stratification_vars"] == 0

    def test_summary_without_stratification(self):
        """Test summary with no stratification variables."""
        df = pd.DataFrame({"id": [1, 2, 3], "contact_age": [25, 34, 45]})

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        summary = cnt_data.summary()

        assert summary["stratification_vars"] == []
        assert summary["n_stratification_vars"] == 0


class TestContactDataEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_contact(self):
        """Test with a single contact."""
        df = pd.DataFrame({"id": [1], "contact_age": [25]})

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.n_contacts == 1
        assert cnt_data.n_unique_participants == 1
        assert cnt_data.age_range == (25, 25)

    def test_age_zero(self):
        """Test that age 0 is valid."""
        df = pd.DataFrame({"id": [1, 2, 3], "contact_age": [0, 5, 10]})

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.age_range == (0, 10)

    def test_float_ages(self):
        """Test that float ages are valid."""
        df = pd.DataFrame({"id": [1, 2, 3], "contact_age": [25.5, 34.2, 45.8]})

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.n_contacts == 3
        assert cnt_data.age_range == (25.5, 45.8)

    def test_large_dataset(self):
        """Test with a larger dataset for performance."""
        n = 10000
        df = pd.DataFrame(
            {
                "id": np.random.randint(1, 1000, n),  # 1000 participants
                "contact_age": np.random.randint(0, 100, n),
                "setting": np.random.choice(["home", "work", "school", "other"], n),
            }
        )

        cnt_data = ContactData(
            df_cnt=df, id_col="id", age_col="contact_age", strat_vars="setting"
        )

        assert cnt_data.n_contacts == n
        assert cnt_data.n_unique_participants <= 1000
        assert len(cnt_data.stratification_vars) == 1

    def test_many_contacts_per_participant(self):
        """Test participant with many contacts."""
        df = pd.DataFrame(
            {
                "id": [1] * 50
                + [2] * 3,  # Participant 1 has 50 contacts, participant 2 has 3
                "contact_age": np.random.randint(0, 85, 53),
            }
        )

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.n_contacts == 53
        assert cnt_data.n_unique_participants == 2

        contacts_per_part = cnt_data.get_contacts_per_participant()
        assert contacts_per_part[1] == 50
        assert contacts_per_part[2] == 3

    def test_non_sequential_ids(self):
        """Test that non-sequential IDs work correctly."""
        df = pd.DataFrame(
            {"id": [100, 100, 205, 37, 999], "contact_age": [25, 30, 34, 45, 52]}
        )

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.n_contacts == 5
        assert cnt_data.n_unique_participants == 4

    def test_string_ids(self):
        """Test that string IDs work correctly."""
        df = pd.DataFrame(
            {"id": ["A001", "A001", "B002", "C003"], "contact_age": [25, 30, 34, 45]}
        )

        cnt_data = ContactData(df_cnt=df, id_col="id", age_col="contact_age")

        assert cnt_data.n_contacts == 4
        assert cnt_data.n_unique_participants == 3


class TestContactDataIntegration:
    """Integration tests with realistic data scenarios."""

    def test_realistic_contact_data(self):
        """Test with realistic social contact survey structure."""
        # Simulate a small contact survey
        # 50 participants reporting 150 contacts
        participant_ids = np.repeat(range(1, 51), np.random.randint(1, 8, 50))

        df = pd.DataFrame(
            {
                "participant_id": participant_ids,
                "contact_age": np.random.randint(0, 85, len(participant_ids)),
                "setting": np.random.choice(
                    ["home", "work", "school", "transport", "other"],
                    len(participant_ids),
                ),
                "duration": np.random.choice(
                    ["<5min", "5-15min", "15-60min", ">1hr"], len(participant_ids)
                ),
            }
        )

        cnt_data = ContactData(
            df_cnt=df,
            id_col="participant_id",
            age_col="contact_age",
            strat_vars=["setting", "duration"],
        )

        summary = cnt_data.summary()

        assert summary["n_unique_participants"] <= 50
        assert summary["n_stratification_vars"] == 2
        assert "age_range" in summary

        # Check contacts per participant
        contacts_per_part = cnt_data.get_contacts_per_participant()
        assert len(contacts_per_part) == summary["n_unique_participants"]
        assert contacts_per_part.sum() == summary["n_contacts"]

    def test_compatibility_with_dataloader_pattern(self):
        """Test that ContactData works with DataLoader patterns."""
        # This mimics how data is prepared for DataLoader
        np.random.seed(42)
        contacts_per_person = np.random.randint(1, 6, 20)
        n_contacts = sum(contacts_per_person)

        df = pd.DataFrame(
            {
                "id": np.repeat(range(1, 21), contacts_per_person),
                "age_group": pd.cut(
                    np.random.randint(0, 85, n_contacts),
                    bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 85],
                    right=False,
                ),
                "setting": np.random.choice(["home", "work", "other"], n_contacts),
            }
        )

        cnt_data = ContactData(
            df_cnt=df, id_col="id", age_grp_col="age_group", strat_vars="setting"
        )

        # Verify it's compatible with expected structure
        assert hasattr(cnt_data, "data")
        assert hasattr(cnt_data, "n_contacts")
        # Check for renamed columns with _cnt suffix
        assert "age_grp_cnt" in cnt_data.data.columns
        assert "setting_cnt" in cnt_data.data.columns
        assert "id" in cnt_data.data.columns
