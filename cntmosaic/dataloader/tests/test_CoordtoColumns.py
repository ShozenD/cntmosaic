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
            part_age="age_group",
            cnt_age="cnt_age",
            pop_age="age",
            P="P",
        )

        # Basic assertions
        assert colmap.part_age == "age_group"
        assert colmap.cnt_age == "cnt_age"
        assert colmap.pop_age == "age"
        assert colmap.P == "P"
        assert colmap.age_vars == ["cnt_age", "age_group"]

    def test_with_repeat(self):
        """Test ColumnSpec with repeat column."""
        colmap = ColumnSpec(
            part_age="age_group",
            cnt_age="cnt_age",
            pop_age="age",
            P="P",
            part_repeat="repeat_id",
        )

        # Basic assertions
        assert colmap.part_age == "age_group"
        assert colmap.cnt_age == "cnt_age"
        assert colmap.pop_age == "age"
        assert colmap.P == "P"
        assert colmap.part_repeat == "repeat_id"
        assert colmap.age_vars == ["cnt_age", "age_group"]


class TestPartial:

    def test_single_strat(self):
        """Test ColumnSpec with partial subgroup data."""
        colmap = ColumnSpec(
            part_age="age_group",
            cnt_age="cnt_age",
            pop_age="age",
            P="P",
            part_strat_vars="subgroup_part",
        )

        # Basic assertions
        assert colmap.part_age == "age_group"
        assert colmap.cnt_age == "cnt_age"
        assert colmap.pop_age == "age"
        assert colmap.P == "P"
        assert colmap.part_strat_vars == ["subgroup_part"]
        assert colmap.age_vars == ["cnt_age", "age_group"]

    def test_multiple_strat(self):
        """Test ColumnSpec with multiple subgroup data."""
        colmap = ColumnSpec(
            part_age="age_group",
            cnt_age="cnt_age",
            pop_age="age",
            P="P",
            part_strat_vars=["subgroup1_part", "subgroup2_part"],
        )

        assert colmap.part_strat_vars == ["subgroup1_part", "subgroup2_part"]

    def test_single_strat_with_repeat(self):
        """Test ColumnSpec with partial subgroup data and repeat column."""
        colmap = ColumnSpec(
            part_age="age_group",
            cnt_age="cnt_age",
            pop_age="age",
            P="P",
            part_strat_vars="subgroup_part",
            part_repeat="repeat_id",
        )

        # Basic assertions
        assert colmap.part_age == "age_group"
        assert colmap.cnt_age == "cnt_age"
        assert colmap.pop_age == "age"
        assert colmap.P == "P"
        assert colmap.part_strat_vars == ["subgroup_part"]
        assert colmap.part_repeat == "repeat_id"
        assert colmap.age_vars == ["cnt_age", "age_group"]


def test_coord_to_columns_full():
    """Test ColumnSpec with full subgroup data."""
    colmap = ColumnSpec(
        part_age="age_group",
        cnt_age="cnt_age",
        pop_age="age",
        P="P",
        part_strat_vars="subgroup_part",
        cnt_strat_vars="subgroup_cnt",
        pop_strat_vars="subgroup",  # Original name (without _cnt suffix)
    )

    # Basic assertions
    assert colmap.part_age == "age_group"
    assert colmap.cnt_age == "cnt_age"
    assert colmap.pop_age == "age"
    assert colmap.P == "P"
    assert colmap.part_strat_vars == ["subgroup_part"]
    assert colmap.cnt_strat_vars == ["subgroup_cnt"]
    assert colmap.pop_strat_vars == ["subgroup"]  # Original name
    assert colmap.age_vars == ["cnt_age", "age_group"]


def test_coord_to_columns_empty_strat_vars_match():
    """Test that empty contact and population grouping variables are considered matching."""
    colmap = ColumnSpec(
        part_age="age_group",
        cnt_age="cnt_age",
        pop_age="age",
        P="P",
        cnt_strat_vars=None,
        pop_strat_vars=None,
    )
    assert colmap.cnt_strat_vars == []
    assert colmap.pop_strat_vars == []


def test_coord_to_columns_strat_vars_different_order():
    """Test that cnt_strat_vars and pop_strat_vars can be in different order (set comparison)."""
    colmap = ColumnSpec(
        part_age="age_group",
        cnt_age="cnt_age",
        pop_age="age",
        P="P",
        cnt_strat_vars=["gender_cnt", "setting_cnt"],  # With _cnt suffix
        pop_strat_vars=["setting", "gender"],  # Different order, original names
    )
    # After stripping _cnt suffix, they should match
    cnt_strat_vars_original = [
        var.removesuffix("_cnt") for var in colmap.cnt_strat_vars
    ]
    assert set(cnt_strat_vars_original) == set(colmap.pop_strat_vars)
    assert colmap.cnt_strat_vars == ["gender_cnt", "setting_cnt"]
    assert colmap.pop_strat_vars == ["setting", "gender"]


def test_coord_to_columns_cnt_suffix_validation():
    """Test that _cnt suffix is automatically stripped for validation."""
    # This should work: cnt_strat_vars has _cnt suffix, pop_strat_vars doesn't
    colmap = ColumnSpec(
        part_age="age_group",
        cnt_age="cnt_age",
        pop_age="age",
        P="P",
        cnt_strat_vars=["gender_cnt"],  # With suffix
        pop_strat_vars=["gender"],  # Without suffix - should match
    )
    assert colmap.cnt_strat_vars == ["gender_cnt"]
    assert colmap.pop_strat_vars == ["gender"]
