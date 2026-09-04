"""Tests for the neutral vertical default: a bare select leaves depth alone.

``osk.field(source, variable)`` with no ``select`` must not default to any
particular time, depth, lon, or lat -- time/lon/lat already had no default,
but an absent vertical key used to mean "the model's own top level" (see
``ocean_skill.comparison._prepare``'s old ``surface = is_surface_request(depth)``).
It no longer does: only the *compare* lane still defaults to the surface, by
writing it in explicitly (``Comparison._prepare_lane``) before ``_prepare`` ever
sees the select. A bare ``Field`` keeps every native level standing instead --
the same shape ``select={"depth": "column"}`` gives a comparison, for a model
source, or every reported level for an observational one.

Complements ``tests/test_comparison.py`` (the core semantic, on a ROMS fixture)
and ``tests/test_column_request.py`` (the COLUMN machinery this reuses) with the
``Field``-level consequences: labels, and the new label-less-native-axis guard
on ``.plot()``/``.movie()``/``._series_items()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.comparison import COLUMN, _prepare

NITRATE = "nitrate"


# -- _prepare itself: bare select vs. the explicit COLUMN/obs equivalents ------------


def _stretch(s, theta_s=5.0, theta_b=2.0):
    c = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
    return (np.exp(theta_b * c) - 1) / (1 - np.exp(-theta_b))


@pytest.fixture
def roms_grid():
    """Build a standardize()-shaped grid, matching test_column_request.py's own."""
    from ocean_skill import roms

    n = 12
    h = np.array([[50.0, 500.0]])
    sigma_r = (np.arange(1, n + 1) - n - 0.5) / n
    sigma_w = np.linspace(-1, 0, n + 1)
    ds = xr.Dataset(
        {
            "temp": (
                ("s_rho", "eta_rho", "xi_rho"),
                np.linspace(5.0, 20.0, n)[:, None, None] * np.ones((n, 1, 2)),
                {"units": "degC"},
            )
        },
        coords={
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((1, 2))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
            "lon": (("eta_rho", "xi_rho"), np.array([[-95.0, -94.0]])),
            "lat": (("eta_rho", "xi_rho"), np.array([[25.0, 25.0]])),
        },
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": 250.0}}
    return roms.add_depth_coord(ds, meta), meta


def test_a_bare_select_at_a_roms_point_matches_the_column_request(roms_grid):
    """No depth key at all keeps every native level standing.

    Exactly like ``select={"depth": "column"}`` -- the field lane never needs
    that spelling.
    """
    ds, meta = roms_grid
    bare, bare_depth = _prepare(ds, meta, "temp", {"eta_rho": 0, "xi_rho": 0})
    column, column_depth = _prepare(
        ds, meta, "temp", {"depth": COLUMN, "eta_rho": 0, "xi_rho": 0}
    )
    xr.testing.assert_identical(bare, column)
    assert bare_depth is column_depth is None
    assert "z_rho" in bare.coords
    assert "dz" in bare.coords


def test_a_bare_select_at_a_roms_point_does_not_isel_the_top_level(roms_grid):
    ds, meta = roms_grid
    da, _ = _prepare(ds, meta, "temp", {"eta_rho": 0, "xi_rho": 0})
    assert da.sizes["s_rho"] == ds.sizes["s_rho"]


def test_a_bare_select_on_an_observational_profile_keeps_every_level():
    """The non-ROMS branch of ``_prepare``.

    No isel, no ``actual_depth``, unlike the old surface-by-default behavior.
    """
    ds = xr.Dataset(
        {"v": (("depth",), np.array([1.0, 3.0, 100.0]), {"units": "mg/m^3"})},
        coords={"depth": [0.0, 10.0, 50.0]},
    )
    da, actual = _prepare(ds, {}, "v", {})
    assert list(da["depth"].values) == [0.0, 10.0, 50.0]
    assert actual is None
    assert "actual_depth" not in da.attrs


def test_an_explicit_surface_select_still_isels_the_top_level(roms_grid):
    """The escape hatch: naming ``"surface"`` explicitly reproduces the old default."""
    ds, meta = roms_grid
    select = {"depth": "surface", "eta_rho": 0, "xi_rho": 0}
    da, _ = _prepare(ds, meta, "temp", select)
    assert "s_rho" not in da.dims


# -- Field-level consequences: labels and the label-less-native-axis guard ----------


@pytest.fixture
def stub(monkeypatch):
    """Swap ``comparison.prepare_source`` for one hand-built field.

    Like tests/test_field_series.py's own fixture of the same name.
    """
    from ocean_skill import comparison

    def use(field_da, actual_depth=None):
        monkeypatch.setattr(
            comparison, "prepare_source", lambda *a, **k: (field_da, actual_depth)
        )

    return use


def _make(**kwargs):
    from ocean_skill.field import field as make_field

    return make_field("stub", NITRATE, **kwargs)


def test_as_item_reports_no_depth_for_a_bare_select(stub):
    """A bare Field's label says nothing about depth when nothing was selected.

    Unlike a Comparison, whose own default is the surface -- a bare Field did
    not reduce anything.
    """
    da = xr.DataArray(
        np.arange(4.0),
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=4, freq="MS")},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    ).assign_coords(lon=-144.0, lat=50.0)
    stub(da)
    assert _make().as_item()["depth"] is None


def test_as_item_still_reports_an_explicit_depth(stub):
    da = xr.DataArray(
        np.arange(4.0),
        dims="time",
        coords={"time": pd.date_range("2020-01-01", periods=4, freq="MS")},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    ).assign_coords(lon=-144.0, lat=50.0)
    stub(da)
    assert _make(select={"depth": 50}).as_item()["depth"] == "50 m"


def _bare_native_facet(nt=2, ns=3, ny=2, nx=2):
    """Build a model field with a bare, coordinate-less native s_rho axis standing.

    The shape a bare field() call now leaves a ROMS source in when it is not
    narrowed to a point. z_rho on a facet this shape genuinely varies over
    eta/xi too (a stretched coordinate on a curvilinear grid), so it is never a
    single per-row label -- see test_a_labeled_obs_depth_axis_is_not_labelless
    for the case that *is* exempt.
    """
    return xr.DataArray(
        np.random.default_rng(0).normal(15.0, 1.0, (nt, ns, ny, nx)),
        dims=("time", "s_rho", "eta_rho", "xi_rho"),
        coords={
            "time": pd.date_range("2020-01-01", periods=nt, freq="D"),
            "lon": (
                ("eta_rho", "xi_rho"),
                np.linspace(-95, -94, nx)[None, :] * np.ones((ny, 1)),
            ),
            "lat": (
                ("eta_rho", "xi_rho"),
                np.linspace(20, 21, ny)[:, None] * np.ones((1, nx)),
            ),
        },
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


def test_a_bare_native_facet_axis_refuses_to_plot(stub):
    stub(_bare_native_facet())
    with pytest.raises(ValueError, match="native vertical axis"):
        _make().plot()


def test_a_bare_native_facet_axis_refuses_to_play_as_a_movie(stub):
    stub(_bare_native_facet())
    with pytest.raises(ValueError, match="native vertical axis"):
        _make().movie()


def test_a_labeled_obs_depth_axis_is_not_labelless():
    """A gridded product's own reported ``depth`` coordinate is a real label.

    Facet-ing over it is untouched by the new guard, whatever the facet size.
    """
    from ocean_skill.field import _labelless_vertical

    da = xr.DataArray(
        np.zeros((3, 2, 2)),
        dims=("depth", "lat", "lon"),
        coords={"depth": [0.0, 10.0, 50.0], "lat": [1.0, 2.0], "lon": [1.0, 2.0]},
        name=NITRATE,
    )
    assert _labelless_vertical(da) is None


def _bare_native_point(nt=3, ns=4, with_z_rho=False):
    """Build a model point with a bare native s_rho axis still standing.

    A series of levels with no coordinate of its own to fan a legend by.
    """
    da = xr.DataArray(
        np.random.default_rng(1).normal(15.0, 1.0, (nt, ns)),
        dims=("time", "s_rho"),
        coords={"time": pd.date_range("2020-01-01", periods=nt, freq="D")},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    ).assign_coords(lon=-144.0, lat=50.0)
    if with_z_rho:
        da = da.assign_coords(z_rho=(("s_rho",), -np.linspace(5.0, 100.0, ns)))
    return da


def test_a_bare_native_point_series_refuses_without_z_rho(stub):
    stub(_bare_native_point())
    with pytest.raises(ValueError, match="native vertical axis"):
        _make()._series_items()


def test_a_bare_native_point_series_fans_by_z_rho_when_present(stub):
    stub(_bare_native_point(ns=4, with_z_rho=True))
    items = _make()._series_items()
    assert len(items) == 4
    depths = [item["aligned"].attrs["actual_depth"] for item in items]
    assert depths == sorted(depths)
    assert all(d > 0 for d in depths)  # positive-down, like every other depth label
