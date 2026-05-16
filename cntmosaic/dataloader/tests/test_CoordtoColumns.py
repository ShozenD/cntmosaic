import pytest

from .._ColumnSpec import ColumnSpec

# language: python

# =================================
# Fixtures
# =================================


class TestSingle:

    def test_basic(self):
        """Test ColumnSpec with single subgroup data."""
        colmap = ColumnSpec(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            P="P",
        )

        # Basic assertions
        assert colmap.age_part == "age_group"
        assert colmap.age_cnt == "age_cnt"
        assert colmap.age_pop == "age"
        assert colmap.P == "P"
        assert colmap.age_vars == ["age_cnt", "age_group"]

    def test_with_repeat(self):
        """Test ColumnSpec with repeat column."""
        colmap = ColumnSpec(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            P="P",
            repeat_part="repeat_id",
        )

        # Basic assertions
        assert colmap.age_part == "age_group"
        assert colmap.age_cnt == "age_cnt"
        assert colmap.age_pop == "age"
        assert colmap.P == "P"
        assert colmap.repeat_part == "repeat_id"
        assert colmap.age_vars == ["age_cnt", "age_group"]


class TestPartial:

    def test_single_strat(self):
        """Test ColumnSpec with partial subgroup data."""
        colmap = ColumnSpec(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            P="P",
            strat_vars_part="subgroup_part",
        )

        # Basic assertions
        assert colmap.age_part == "age_group"
        assert colmap.age_cnt == "age_cnt"
        assert colmap.age_pop == "age"
        assert colmap.P == "P"
        assert colmap.strat_vars_part == ["subgroup_part"]
        assert colmap.age_vars == ["age_cnt", "age_group"]

    def test_multiple_strat(self):
        """Test ColumnSpec with multiple subgroup data."""
        colmap = ColumnSpec(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            P="P",
            strat_vars_part=["subgroup1_part", "subgroup2_part"],
        )

        assert colmap.strat_vars_part == ["subgroup1_part", "subgroup2_part"]

    def test_single_strat_with_repeat(self):
        """Test ColumnSpec with partial subgroup data and repeat column."""
        colmap = ColumnSpec(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            P="P",
            strat_vars_part="subgroup_part",
            repeat_part="repeat_id",
        )

        # Basic assertions
        assert colmap.age_part == "age_group"
        assert colmap.age_cnt == "age_cnt"
        assert colmap.age_pop == "age"
        assert colmap.P == "P"
        assert colmap.strat_vars_part == ["subgroup_part"]
        assert colmap.repeat_part == "repeat_id"
        assert colmap.age_vars == ["age_cnt", "age_group"]


def test_coord_to_columns_full():
    """Test ColumnSpec with full subgroup data."""
    colmap = ColumnSpec(
        age_part="age_group",
        age_cnt="age_cnt",
        age_pop="age",
        P="P",
        strat_vars_part="subgroup_part",
        strat_vars_cnt="subgroup_cnt",
        strat_vars_pop="subgroup",  # Original name (without _cnt suffix)
    )

    # Basic assertions
    assert colmap.age_part == "age_group"
    assert colmap.age_cnt == "age_cnt"
    assert colmap.age_pop == "age"
    assert colmap.P == "P"
    assert colmap.strat_vars_part == ["subgroup_part"]
    assert colmap.strat_vars_cnt == ["subgroup_cnt"]
    assert colmap.strat_vars_pop == ["subgroup"]  # Original name
    assert colmap.age_vars == ["age_cnt", "age_group"]


def test_coord_to_columns_empty_strat_vars_match():
    """Test that empty contact and population grouping variables are considered matching."""
    colmap = ColumnSpec(
        age_part="age_group",
        age_cnt="age_cnt",
        age_pop="age",
        P="P",
        strat_vars_cnt=None,
        strat_vars_pop=None,
    )
    assert colmap.strat_vars_cnt == []
    assert colmap.strat_vars_pop == []


def test_coord_to_columns_strat_vars_different_order():
    """Test that strat_vars_cnt and strat_vars_pop can be in different order (set comparison)."""
    colmap = ColumnSpec(
        age_part="age_group",
        age_cnt="age_cnt",
        age_pop="age",
        P="P",
        strat_vars_cnt=["gender_cnt", "setting_cnt"],  # With _cnt suffix
        strat_vars_pop=["setting", "gender"],  # Different order, original names
    )
    # After stripping _cnt suffix, they should match
    strat_vars_cnt_original = [
        var.removesuffix("_cnt") for var in colmap.strat_vars_cnt
    ]
    assert set(strat_vars_cnt_original) == set(colmap.strat_vars_pop)
    assert colmap.strat_vars_cnt == ["gender_cnt", "setting_cnt"]
    assert colmap.strat_vars_pop == ["setting", "gender"]


def test_coord_to_columns_cnt_suffix_validation():
    """Test that _cnt suffix is automatically stripped for validation."""
    # This should work: strat_vars_cnt has _cnt suffix, strat_vars_pop doesn't
    colmap = ColumnSpec(
        age_part="age_group",
        age_cnt="age_cnt",
        age_pop="age",
        P="P",
        strat_vars_cnt=["gender_cnt"],  # With suffix
        strat_vars_pop=["gender"],  # Without suffix - should match
    )
    assert colmap.strat_vars_cnt == ["gender_cnt"]
    assert colmap.strat_vars_pop == ["gender"]
