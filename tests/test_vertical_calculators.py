"""Water-column calculators (ocean_skill/vertical.py) and the ``depth`` aggregate key.

Small synthetic columns with hand-computed answers: a ROMS-like ``(s_rho, eta, xi)``
dataset carrying ``z_rho``/``z_w`` (levels run bottom -> top, as ROMS stores them) and a
WOA-like ``(depth, lat, lon)`` dataset on standard levels.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms, vertical
from ocean_skill.operators import CALCULATORS, aggregate, resolve_variable, spec_names

NAN = np.nan

# -- ROMS-like fixture ----------------------------------------------------------------
# Four levels, bottom -> top. Interfaces z_w = -100, -60, -30, -10, 0 (thickness 40, 30,
# 20, 10 from the bottom up) and centres halfway: -80, -45, -20, -5.
Z_W = np.array([-100.0, -60.0, -30.0, -10.0, 0.0])
Z_RHO = np.array([-80.0, -45.0, -20.0, -5.0])
DZ = np.array([40.0, 30.0, 20.0, 10.0])


def _roms(columns, *, z_w=True):
    """Build a (s_rho, eta_rho, xi_rho=1) dataset from per-column values (bottom -> top).

    ``columns`` is a list of 4-value columns, one per ``eta_rho`` row, in the units of
    ``oxygen`` (mmol m-3).
    """
    vals = np.array(columns, dtype=float).T[:, :, None]  # (s, eta, xi)
    n_eta = vals.shape[1]
    zr = np.broadcast_to(Z_RHO[:, None, None], vals.shape).copy()
    coords = {"z_rho": (("s_rho", "eta_rho", "xi_rho"), zr)}
    if z_w:
        zw = np.broadcast_to(Z_W[:, None, None], (5, n_eta, 1)).copy()
        coords["z_w"] = (("s_w", "eta_rho", "xi_rho"), zw)
    return xr.Dataset(
        {
            "oxygen": (
                ("s_rho", "eta_rho", "xi_rho"),
                vals,
                {"units": "mmol m-3", "long_name": "dissolved oxygen"},
            )
        },
        coords=coords,
    )


# -- WOA-like fixture -----------------------------------------------------------------
# Levels 0, 10, 30, 50 m. Interfaces: 0 (clipped from -5), 5, 20, 40, 60 -> thickness
# 5, 15, 20, 20.
WOA_DEPTH = np.array([0.0, 10.0, 30.0, 50.0])
WOA_DZ = np.array([5.0, 15.0, 20.0, 20.0])


def _woa(columns, *, depth=WOA_DEPTH, attrs=None):
    vals = np.array(columns, dtype=float).T[:, :, None]  # (depth, lat, lon)
    return xr.Dataset(
        {
            "oxygen": (
                ("depth", "lat", "lon"),
                vals,
                {"units": "mmol m-3", "long_name": "dissolved oxygen"},
            )
        },
        coords={
            "depth": ("depth", np.asarray(depth, float), attrs or {}),
            "lat": np.arange(vals.shape[1], dtype=float),
            "lon": [0.0],
        },
    )


# -- depth_of -------------------------------------------------------------------------


def test_depth_of_roms_uses_the_columns_own_depths():
    """Minimum at index 1 (bottom -> top) of the first column, so z_rho = -45 -> 45 m;
    the second column's minimum is its top level, 5 m. Per-column, not a shared axis."""
    ds = _roms([[90, 10, 40, 60], [50, 40, 30, 20]])
    ds["z_rho"] = ds["z_rho"].copy()
    ds["z_rho"].values[:, 1, 0] = [-200, -150, -100, -50]  # a deeper column
    out = vertical.calculate_depth_of(ds, variable="oxygen", extremum="min")
    assert out.dims == ("eta_rho", "xi_rho")
    assert out.values[:, 0] == pytest.approx([45.0, 50.0])
    assert out.attrs["units"] == "m"
    assert out.name == "depth_of_minimum_oxygen"
    assert "minimum" in out.attrs["long_name"]


def test_depth_of_max_roms():
    ds = _roms([[90, 10, 40, 60]])
    out = vertical.calculate_depth_of(ds, variable="oxygen", extremum="max")
    assert out.values[0, 0] == pytest.approx(80.0)  # the bottom level, z_rho = -80
    assert out.name == "depth_of_maximum_oxygen"


def test_depth_of_woa():
    ds = _woa([[200, 120, 8, 30], [5, 20, 40, 60]])
    out = vertical.calculate_depth_of(ds, variable="oxygen", extremum="min")
    assert out.dims == ("lat", "lon")
    assert out.values[:, 0] == pytest.approx([30.0, 0.0])


def test_depth_of_tie_takes_the_shallowest():
    ds = _woa([[100, 5, 5, 50]])
    out = vertical.calculate_depth_of(ds, variable="oxygen", extremum="min")
    assert out.values[0, 0] == pytest.approx(10.0)


def test_depth_of_all_nan_column_is_nan_and_nan_levels_are_skipped():
    ds = _woa([[NAN, NAN, NAN, NAN], [NAN, 30, 10, NAN]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no all-NaN RuntimeWarning leaks out
        out = vertical.calculate_depth_of(ds, variable="oxygen", extremum="min")
    assert np.isnan(out.values[0, 0])
    assert out.values[1, 0] == pytest.approx(30.0)


def test_depth_of_land_column_roms_is_nan():
    ds = _roms([[NAN] * 4, [10, 20, 30, 40]])
    out = vertical.calculate_depth_of(ds, variable="oxygen", extremum="min")
    assert np.isnan(out.values[0, 0])
    assert out.values[1, 0] == pytest.approx(80.0)


def test_depth_of_descending_and_negative_up_axes():
    """The level order does not matter, and a ``positive: up`` axis is flipped."""
    ds = _woa([[200, 120, 8, 30]], depth=WOA_DEPTH)
    flipped = ds.isel(depth=slice(None, None, -1))
    out = vertical.calculate_depth_of(flipped, variable="oxygen", extremum="min")
    assert out.values[0, 0] == pytest.approx(30.0)

    up = _woa([[200, 120, 8, 30]], depth=-WOA_DEPTH, attrs={"positive": "up"})
    out = vertical.calculate_depth_of(up, variable="oxygen", extremum="min")
    assert out.values[0, 0] == pytest.approx(30.0)


def test_depth_of_needs_an_extremum_and_a_vertical_axis():
    ds = _woa([[1, 2, 3, 4]])
    with pytest.raises(KeyError, match="extremum"):
        vertical.calculate_depth_of(ds, variable="oxygen")
    with pytest.raises(KeyError, match="not in this dataset"):
        vertical.calculate_depth_of(ds, variable="nitrate", extremum="min")
    flat = ds.isel(depth=0, drop=True)
    with pytest.raises(ValueError, match="no vertical axis"):
        vertical.calculate_depth_of(flat, variable="oxygen", extremum="min")


# -- thickness ------------------------------------------------------------------------


def test_thickness_roms_uses_real_cell_thickness():
    """Oxygen below 50: levels 0 (40 m) and 1 (30 m) qualify -> 70 m. Not 2 levels."""
    ds = _roms([[10, 20, 60, 70]])
    out = vertical.calculate_thickness(ds, variable="oxygen", below=50)
    assert out.values[0, 0] == pytest.approx(70.0)
    assert out.attrs["units"] == "m"
    assert "50 mmol m-3" in out.attrs["long_name"]


def test_thickness_adds_separate_layers():
    ds = _roms([[10, 60, 20, 70]])  # 40 m + 20 m, not the 70 m span between them
    out = vertical.calculate_thickness(ds, variable="oxygen", below=50)
    assert out.values[0, 0] == pytest.approx(60.0)


def test_thickness_above():
    ds = _roms([[10, 60, 20, 70]])
    out = vertical.calculate_thickness(ds, variable="oxygen", above=50)
    assert out.values[0, 0] == pytest.approx(40.0)  # 30 m + 10 m


def test_thickness_threshold_is_strict():
    ds = _roms([[50, 50, 50, 50]])
    assert vertical.calculate_thickness(ds, variable="oxygen", below=50).values[0, 0] == 0.0
    assert vertical.calculate_thickness(ds, variable="oxygen", above=50).values[0, 0] == 0.0
    assert vertical.calculate_thickness(ds, variable="oxygen", below=50.5).values[0, 0] == pytest.approx(100.0)


def test_thickness_none_qualifying_is_zero_and_land_is_nan():
    ds = _roms([[60, 70, 80, 90], [NAN] * 4, [10, NAN, NAN, NAN]])
    out = vertical.calculate_thickness(ds, variable="oxygen", below=50)
    assert out.values[0, 0] == 0.0
    assert np.isnan(out.values[1, 0])
    assert out.values[2, 0] == pytest.approx(40.0)  # NaN levels add nothing


def test_thickness_woa_midpoints_and_surface_clip():
    """Levels 0/10/30/50 m have thicknesses 5/15/20/20 (see the fixture comment)."""
    ds = _woa([[10, 10, 90, 90], [90, 90, 90, 10], [10, 90, 90, 90], [90, 90, 90, 90]])
    out = vertical.calculate_thickness(ds, variable="oxygen", below=50)
    assert out.values[:, 0] == pytest.approx([20.0, 20.0, 5.0, 0.0])


def test_thickness_woa_deep_levels_bounded_by_half_a_spacing():
    """WOA's real shape: a column valid to 200 m, on 100 m spacing below 100, is bounded
    at 250 m (the last valid level plus half a spacing), not at the list's own bottom."""
    depth = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    col = [[10, 10, 10, NAN, NAN]]
    out = vertical.calculate_thickness(_woa(col, depth=depth), variable="oxygen", below=50)
    # interfaces 0, 50, 150, 250, 350, 450 -> thickness 50, 100, 100 (NaN levels add 0)
    assert out.values[0, 0] == pytest.approx(250.0)


def test_thickness_woa_unsorted_levels():
    ds = _woa([[10, 10, 90, 90]])
    flipped = ds.isel(depth=slice(None, None, -1))
    out = vertical.calculate_thickness(flipped, variable="oxygen", below=50)
    assert out.values[0, 0] == pytest.approx(20.0)


def test_thickness_roms_builds_interfaces_from_sigma():
    """A real Vtransform-2 transform, hc=0, Cs=sigma: z_w = (zeta + h) * sigma + zeta,
    so with h=100, zeta=2 every layer is 102/4 = 25.5 m thick."""
    sigma_w = np.linspace(-1.0, 0.0, 5)
    sigma_r = (sigma_w[1:] + sigma_w[:-1]) / 2
    ds = xr.Dataset(
        {
            "oxygen": (("s_rho", "eta_rho", "xi_rho"), np.array([10, 60, 20, 70.0]).reshape(4, 1, 1), {"units": "mmol m-3"}),
            "zeta": (("eta_rho", "xi_rho"), [[2.0]]),
        },
        coords={
            "h": (("eta_rho", "xi_rho"), [[100.0]]),
            "sigma_r": ("s_rho", sigma_r),
            "Cs_r": ("s_rho", sigma_r),
            "sigma_w": ("s_w", sigma_w),
            "Cs_w": ("s_w", sigma_w),
            "hc": ((), 0.0),
            "Vtransform": ((), 2),
        },
    )
    meta = {"vertical": {"hc": 0.0, "Vtransform": 2}}
    ds = roms.add_depth_coord(ds, meta)
    out = vertical.calculate_thickness(ds, variable="oxygen", below=50)
    assert out.values[0, 0] == pytest.approx(2 * 25.5)


def test_thickness_roms_without_interfaces_estimates_with_a_warning():
    ds = _roms([[10, 20, 60, 70]], z_w=False)
    with pytest.warns(UserWarning, match="estimated from the level centres"):
        out = vertical.calculate_thickness(ds, variable="oxygen", below=50)
    # centres -80, -45, -20, -5 -> interfaces -97.5(-> extrapolated), -62.5, -32.5, -12.5,
    # 2.5: thickness 35, 30, 20, 15; levels 0 and 1 qualify.
    assert out.values[0, 0] == pytest.approx(65.0)


def test_thickness_needs_exactly_one_threshold():
    ds = _woa([[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="exactly one"):
        vertical.calculate_thickness(ds, variable="oxygen")
    with pytest.raises(ValueError, match="exactly one"):
        vertical.calculate_thickness(ds, variable="oxygen", below=1, above=2)


def test_time_axis_is_kept():
    """z_rho with a time axis (a moving free surface): one answer per time step."""
    ds = xr.concat([_roms([[10, 20, 60, 70]]), _roms([[60, 20, 10, 70]])], dim="time")
    out = vertical.calculate_depth_of(ds, variable="oxygen", extremum="min")
    assert out.dims == ("time", "eta_rho", "xi_rho")
    assert out.values[:, 0, 0] == pytest.approx([80.0, 20.0])
    out = vertical.calculate_thickness(ds, variable="oxygen", below=50)
    assert out.values[:, 0, 0] == pytest.approx([70.0, 30.0 + 20.0])


# -- registration and the request grammar ---------------------------------------------


def test_registered_and_resolvable_as_variable_specs():
    assert CALCULATORS["depth_of"] is vertical.calculate_depth_of
    assert CALCULATORS["thickness"] is vertical.calculate_thickness
    ds = _woa([[200, 120, 8, 30]])
    got = resolve_variable(ds, {"calculate": "depth_of", "variable": "oxygen", "extremum": "min"})
    assert got.values[0, 0] == pytest.approx(30.0)
    got = resolve_variable(ds, {"calculate": "thickness", "variable": "oxygen", "below": 50})
    assert got.values[0, 0] == pytest.approx(40.0)  # levels 30 m and 50 m: 20 + 20


def test_spec_names_reports_the_underlying_variable():
    assert spec_names({"calculate": "depth_of", "variable": "oxygen", "extremum": "min"}) == [["oxygen"]]
    assert spec_names({"calculate": "thickness", "variable": {"sum": ["a", "b"]}, "below": 1}) == [["a", "b"]]


# -- aggregate: the `depth` key on native ROMS levels ----------------------------------


def _roms_field(values):
    return xr.DataArray(
        np.asarray(values, float).reshape(-1, 1, 1),
        dims=("s_rho", "eta_rho", "xi_rho"),
        name="oxygen",
        attrs={"units": "mmol m-3"},
    )


def test_depth_key_min_max_reduce_bare_native_levels():
    da = _roms_field([30, 10, 20, 40])
    assert float(aggregate(da, {"depth": "min"}).squeeze()) == 10.0
    assert float(aggregate(da, {"depth": "max"}).squeeze()) == 40.0
    # and Z still does what it always did
    assert float(aggregate(da, {"Z": "min"}).squeeze()) == 10.0


def test_depth_key_on_woa_is_unchanged():
    ds = _woa([[200, 120, 8, 30]])
    assert float(aggregate(ds["oxygen"], {"depth": "min"}).squeeze()) == 8.0


def test_depth_key_refuses_an_unweighted_mean_over_native_levels():
    da = _roms_field([30, 10, 20, 40])
    with pytest.raises(ValueError, match="not a depth mean"):
        aggregate(da, {"depth": "mean"})
    with pytest.raises(ValueError, match="not a depth mean"):
        aggregate(da, {"depth": {"reduce": "mean"}})
    # Z: mean keeps its old (unweighted) behavior
    assert float(aggregate(da, {"Z": "mean"}).squeeze()) == 25.0


def test_depth_key_mean_uses_cell_weights_when_a_band_attached_them():
    da = _roms_field([30, 10, 20, 40]).assign_coords(
        dz=("s_rho", np.array([1.0, 1.0, 2.0, 0.0]))
    )
    # (30*1 + 10*1 + 20*2 + 40*0) / 4 = 20
    assert float(aggregate(da, {"depth": "mean"}).squeeze()) == pytest.approx(20.0)


def test_depth_key_leaves_a_variable_without_a_vertical_axis_alone():
    da = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"), name="x")
    assert aggregate(da, {"depth": "min"}).dims == ("lat", "lon")


# -- through the real prepare_source, and lazily --------------------------------------


def _mock_source(monkeypatch, ds, meta):
    from unittest import mock

    monkeypatch.setattr("ocean_skill.catalog.resolve", lambda name: mock.Mock(metadata=meta))
    monkeypatch.setattr("ocean_skill.read", lambda name, **kw: ds)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"calculate": "depth_of", "variable": "oxygen", "extremum": "min"}, 30.0),
        ({"calculate": "thickness", "variable": "oxygen", "below": 50}, 40.0),
    ],
)
def test_prepare_source_on_a_woa_like_grid(monkeypatch, spec, expected):
    from ocean_skill.comparison import prepare_source

    _mock_source(monkeypatch, _woa([[200, 120, 8, 30]]), {"featureType": "grid"})
    da, _ = prepare_source("fake_woa", spec, {}, None, use_cache=False)
    assert float(da.squeeze()) == pytest.approx(expected)


def test_prepare_source_on_a_roms_like_grid(monkeypatch):
    from ocean_skill.comparison import prepare_source

    spec = {"calculate": "thickness", "variable": "oxygen", "below": 50}
    _mock_source(monkeypatch, _roms([[10, 20, 60, 70]]), {"model": "roms", "featureType": "grid"})
    da, _ = prepare_source("fake_roms", spec, {"depth": "surface"}, None, use_cache=False)
    assert float(da.squeeze()) == pytest.approx(70.0)


def test_calculators_stay_lazy_on_dask_data():
    ds = _roms([[10, 20, 60, 70], [90, 40, 30, 60]]).chunk({"eta_rho": 1})
    depth_of = vertical.calculate_depth_of(ds, variable="oxygen", extremum="min")
    thick = vertical.calculate_thickness(ds, variable="oxygen", below=50)
    assert depth_of.chunks is not None
    assert thick.chunks is not None
    assert depth_of.values[:, 0] == pytest.approx([80.0, 20.0])
    assert thick.values[:, 0] == pytest.approx([70.0, 50.0])


def test_depth_key_refuses_an_unweighted_mean_when_s_rho_is_also_a_coordinate():
    # the usual layout: s_rho is a dimension *and* a (sigma) coordinate, z_rho the
    # per-column depths -- the refusal fires on what the axis is, not on a coord
    # being absent
    da = _roms_field([30, 10, 20, 40]).assign_coords(
        s_rho=("s_rho", np.array([-0.9, -0.6, -0.3, -0.1])),
        z_rho=(
            ("s_rho", "eta_rho", "xi_rho"),
            -np.array([90, 60, 30, 10.0]).reshape(4, 1, 1),
        ),
    )
    assert "s_rho" in da.coords
    with pytest.raises(ValueError, match="not a depth mean"):
        aggregate(da, {"depth": "mean"})
    # min/max and Z are unchanged
    assert float(aggregate(da, {"depth": "max"}).squeeze()) == 40.0
    assert float(aggregate(da, {"Z": "mean"}).squeeze()) == 25.0
    # and the band-attached weights still make the mean legitimate
    weighted = da.assign_coords(dz=("s_rho", np.array([1.0, 1.0, 2.0, 0.0])))
    got = aggregate(weighted, {"depth": "mean"})
    assert float(got.squeeze()) == pytest.approx(20.0)
