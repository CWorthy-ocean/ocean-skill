"""Tests for :mod:`ocean_skill.cf` -- cf-xarray-based axis detection.

``find_coord`` is the gridded counterpart of :func:`ocean_skill.tabular.coord_column`:
both must refuse a "bottom" name for the vertical axis the same way, sharing one
definition (:data:`ocean_skill.vocabulary.COORD_VOCABULARY`) rather than drifting apart.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ocean_skill import cf


def test_find_coord_ignores_a_bottom_depth_coordinate():
    """A lone ``depth_bottom`` coordinate must not be picked as the vertical axis.

    cf-xarray's own stock ``Z`` criteria would otherwise fullmatch a lowercase
    ``depth_bottom`` name by attrs/regex alone.
    """
    ds = xr.Dataset(
        {"temp": (("depth_bottom",), np.array([1.0, 2.0]))},
        coords={"depth_bottom": [10.0, 20.0]},
    )
    assert cf.find_coord(ds, "vertical") is None


def test_find_coord_prefers_depth_over_a_bottom_depth_coordinate():
    ds = xr.Dataset(
        {"temp": (("depth",), np.array([1.0, 2.0]))},
        coords={"depth": [10.0, 20.0], "depth_bottom": 500.0},
    )
    found = cf.find_coord(ds, "vertical")
    assert found is not None
    assert found.name == "depth"


def test_find_coord_still_finds_a_plain_depth_coordinate():
    ds = xr.Dataset(
        {"temp": (("depth",), np.array([1.0, 2.0]))}, coords={"depth": [10.0, 20.0]}
    )
    found = cf.find_coord(ds, "vertical")
    assert found is not None and found.name == "depth"


def test_find_coord_falls_back_to_roms_style_names():
    """ROMS writes ``s_rho`` with no CF-recognizable attrs -- the name fallback path."""
    ds = xr.Dataset(
        {"temp": (("s_rho",), np.array([1.0, 2.0]))}, coords={"s_rho": [-0.9, -0.1]}
    )
    found = cf.find_coord(ds, "vertical")
    assert found is not None and found.name == "s_rho"


def test_find_coord_returns_none_when_nothing_plausible_is_present():
    ds = xr.Dataset({"temp": (("x",), np.array([1.0, 2.0]))}, coords={"x": [0, 1]})
    assert cf.find_coord(ds, "vertical") is None
