import pytest

from .._base import load_polymod_germany, load_polymod_uk


def test_load_polymod_germany():
    """Tests that the function works."""

    data = load_polymod_germany()

    assert "contacts" in data
    assert "participants" in data
    assert "population" in data
    assert data["contacts"].shape[0] > 0
    assert data["participants"].shape[0] > 0
    assert data["population"].shape[0] > 0


def test_load_polymod_uk():
    """Tests that the function works."""

    data = load_polymod_uk()

    assert "contacts" in data
    assert "participants" in data
    assert "population" in data
    assert data["contacts"].shape[0] > 0
    assert data["participants"].shape[0] > 0
    assert data["population"].shape[0] > 0
