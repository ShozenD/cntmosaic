import pytest

from .._dataloader import CoordToColumns

# language: python

# =================================
# Fixtures
# =================================


def test_coord_to_columns_single():
    """Test CoordToColumns with single subgroup data."""
    colmap = CoordToColumns(
        age_part="age_group",
        age_cnt="age_cnt",
        age_pop="age",
        size_pop="P",
    )

    # Basic assertions
    assert colmap.age_part == "age_group"
    assert colmap.age_cnt == "age_cnt"
    assert colmap.age_pop == "age"
    assert colmap.size_pop == "P"
    assert colmap.age_vars() == ["age_cnt", "age_group"]


def test_coord_to_columns_partial():
    """Test CoordToColumns with partial subgroup data."""
    colmap = CoordToColumns(
        age_part="age_group",
        age_cnt="age_cnt",
        age_pop="age",
        size_pop="P",
        grp_vars_part="subgroup",
    )

    # Basic assertions
    assert colmap.age_part == "age_group"
    assert colmap.age_cnt == "age_cnt"
    assert colmap.age_pop == "age"
    assert colmap.size_pop == "P"
    assert colmap.grp_vars_part == ["subgroup"]
    assert colmap.age_vars() == ["age_cnt", "age_group"]


def test_coord_to_columns_full():
    """Test CoordToColumns with full subgroup data."""
    colmap = CoordToColumns(
        age_part="age_group",
        age_cnt="age_cnt",
        age_pop="age",
        size_pop="P",
        grp_vars_part="subgroup",
        grp_vars_cnt="subgroup_cnt",
    )

    # Basic assertions
    assert colmap.age_part == "age_group"
    assert colmap.age_cnt == "age_cnt"
    assert colmap.age_pop == "age"
    assert colmap.size_pop == "P"
    assert colmap.grp_vars_part == ["subgroup"]
    assert colmap.grp_vars_cnt == ["subgroup_cnt"]
    assert colmap.age_vars() == ["age_cnt", "age_group"]
