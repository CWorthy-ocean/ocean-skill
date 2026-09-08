"""Tests for the coastline resolution ladder shared by both renderers.

Pure logic — no figure is drawn and no shapefile is fetched, so unlike the renderer
tests that exercise this module (``tests/test_renderers.py``, ``tests/test_movie.py``)
these are fast and unmarked, run every time.
"""

from __future__ import annotations

import pytest

from ocean_skill.plot.coastline import (
    DEFAULT_COASTLINE_RESOLUTION,
    auto_ne_resolution,
    is_gshhs,
    nearest_ne_resolution,
    normalize_coastline_resolution,
)


def test_default_is_auto():
    assert DEFAULT_COASTLINE_RESOLUTION == "auto"


@pytest.mark.parametrize(
    "value", ["auto", "110m", "50m", "10m", "coarse", "low", "intermediate", "high", "full"]
)
def test_normalize_accepts_every_documented_value_unchanged(value):
    assert normalize_coastline_resolution(value) == value


@pytest.mark.parametrize(
    "alias, expected",
    [("c", "coarse"), ("l", "low"), ("i", "intermediate"), ("h", "high"), ("f", "full")],
)
def test_normalize_expands_gshhs_single_letter_aliases(alias, expected):
    assert normalize_coastline_resolution(alias) == expected


def test_normalize_rejects_an_unknown_value_by_name():
    with pytest.raises(ValueError, match="coastline_resolution"):
        normalize_coastline_resolution("extra-crispy")


@pytest.mark.parametrize(
    "value, expected", [("auto", False), ("10m", False), ("full", True), ("coarse", True)]
)
def test_is_gshhs(value, expected):
    assert is_gshhs(value) is expected


@pytest.mark.parametrize(
    "extent, expected",
    [
        (None, "110m"),  # no extent to measure -- assume the coarsest, safest scale
        ((-180, 180, -90, 90), "110m"),  # near-global
        ((-100, -60, 10, 40), "50m"),  # ~40 degrees: between the two thresholds
        ((-98, -80, 18, 31), "10m"),  # ~13 degrees: a GOM-scale domain, zoomed in
    ],
)
def test_auto_ne_resolution_matches_cartopys_adaptive_scaler(extent, expected):
    """Delegates to cartopy's own ``AdaptiveScaler`` -- this just checks it stayed so."""
    assert auto_ne_resolution(extent) == expected


def test_nearest_ne_resolution_passes_a_natural_earth_value_through():
    assert nearest_ne_resolution("10m") == "10m"
    assert nearest_ne_resolution("auto", extent=(-98, -80, 18, 31)) == "10m"
    assert nearest_ne_resolution("auto", extent=None) == "110m"


@pytest.mark.parametrize(
    "gshhs_scale, expected_ne",
    [("coarse", "110m"), ("low", "50m"), ("intermediate", "10m"), ("high", "10m"), ("full", "10m")],
)
def test_nearest_ne_resolution_maps_every_gshhs_scale_down_and_warns(gshhs_scale, expected_ne):
    """GSHHS has no interactive-renderer counterpart, so this both degrades and says so."""
    with pytest.warns(UserWarning, match="GSHHS"):
        assert nearest_ne_resolution(gshhs_scale) == expected_ne
