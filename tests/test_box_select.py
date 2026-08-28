"""Tests for a lon/lat *box* select — the plural of a point.

``select={"lon": {"min": ..., "max": ...}, "lat": {"min": ..., "max": ...}}`` already
parsed on a rectilinear grid before this (as two independent range selects), but had
no seam handling of its own and silently did nothing on a curvilinear (ROMS) grid,
where ``lon``/``lat`` are 2-D and neither names a dimension a per-axis ``.sel`` can
use. This mirrors ``tests/test_operators.py``'s point-select tests in structure: a
lon+lat *range* pair is a box exactly the way a lon+lat *scalar* pair is a point, and
is routed the same way — through :func:`ocean_skill.align.subset_to_box` rather than
two independent per-axis slices.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as A
from ocean_skill.operators import _box_selectable, box_in_spec, select


def _gridded(lat, lon=(-98.0, -96.0, -94.0, -92.0)):
    """A rectilinear field on the given latitude order, letting a test pick direction."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return xr.DataArray(
        np.arange(lat.size * lon.size, dtype=float).reshape(lat.size, lon.size),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )


def _curvilinear(nx: int = 11, ny: int = 10, lon0: float = -150.5, lat0: float = 45.5):
    """A ROMS-shaped field: 2-D lon_rho/lat_rho riding on (eta_rho, xi_rho)."""
    lon_1d = np.arange(lon0, lon0 + nx)
    lat_1d = np.arange(lat0, lat0 + ny)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    return xr.DataArray(
        np.arange(ny * nx, dtype=float).reshape(ny, nx),
        dims=("eta_rho", "xi_rho"),
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon_2d),
            "lat_rho": (("eta_rho", "xi_rho"), lat_2d),
        },
        attrs={"units": "degC"},
    )


def _rotated_grid(ny=20, nx=30, lon0=150.0, lat0=10.0, degrees=30.0):
    """A regular grid rotated about its own origin -- a stand-in for a ROMS domain
    whose rows/columns are not lines of constant lon/lat, so a box's corners land
    partway through a cell rather than on a clean index boundary."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    theta = np.radians(degrees)
    x = i * np.cos(theta) - j * np.sin(theta)
    y = i * np.sin(theta) + j * np.cos(theta)
    lon, lat = lon0 + x, lat0 + y
    return xr.DataArray(
        np.ones((ny, nx)),
        dims=("eta_rho", "xi_rho"),
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
        },
    )


# --------------------------------------------------------------- box_in_spec itself


def test_two_ranges_are_a_box():
    hit = box_in_spec({"lon": {"min": -100, "max": -90}, "lat": {"min": 20, "max": 30}})
    assert hit == ("lon", "lat", (-100.0, -90.0), (20.0, 30.0))


def test_slices_are_also_ranges():
    hit = box_in_spec({"lon": slice(-100, -90), "lat": slice(20, 30)})
    assert hit == ("lon", "lat", (-100.0, -90.0), (20.0, 30.0))


def test_a_scalar_pair_is_a_point_not_a_box():
    assert box_in_spec({"lon": -95.5, "lat": 21.0}) is None


def test_a_lone_range_is_a_band_not_a_box():
    assert box_in_spec({"lon": {"min": -100, "max": -90}}) is None
    assert box_in_spec({"lat": {"min": 20, "max": 30}}) is None


def test_empty_or_missing_spec_is_not_a_box():
    assert box_in_spec({}) is None
    assert box_in_spec(None) is None


def test_a_one_sided_latitude_defaults_to_a_pole():
    hit = box_in_spec({"lon": {"min": -100, "max": -90}, "lat": {"min": 60}})
    assert hit == ("lon", "lat", (-100.0, -90.0), (60.0, 90.0))
    hit = box_in_spec({"lon": {"min": -100, "max": -90}, "lat": {"max": -60}})
    assert hit == ("lon", "lat", (-100.0, -90.0), (-90.0, -60.0))


def test_a_one_sided_longitude_is_not_a_box():
    """An open bound on a circle names no band -- there is no sensible default."""
    assert box_in_spec({"lon": {"min": -100}, "lat": {"min": 20, "max": 30}}) is None
    assert box_in_spec({"lon": {"max": -90}, "lat": {"min": 20, "max": 30}}) is None


def test_backwards_latitude_is_swapped():
    hit = box_in_spec({"lon": {"min": -100, "max": -90}, "lat": {"min": 30, "max": 20}})
    assert hit[3] == (20.0, 30.0)


def test_a_seam_straddling_box_is_recognized_and_wrapped_to_0_360():
    """The pac_dt_ramp-style stress case: 170 to -170 spans the dateline."""
    hit = box_in_spec({"lon": {"min": 170, "max": -170}, "lat": {"min": -5, "max": 5}})
    assert hit == ("lon", "lat", (170.0, 190.0), (-5.0, 5.0))


def test_a_plain_backwards_longitude_typo_is_swapped_not_wrapped():
    """30 before 20 does not resolve as a seam-straddling box -- it's just backwards."""
    hit = box_in_spec({"lon": {"min": 30, "max": 20}, "lat": {"min": -5, "max": 5}})
    assert hit == ("lon", "lat", (20.0, 30.0), (-5.0, 5.0))


def test_box_selectable_requires_a_real_lon_lat_coordinate():
    traj = xr.DataArray(
        np.arange(5.0),
        dims="time",
        coords={
            "time": np.arange(5),
            "lon": ("time", np.array([-98.0, -97.0, -96.0, -95.0, -94.0])),
            "lat": ("time", np.array([10.0, 11.0, 12.0, 13.0, 14.0])),
        },
    )
    spec = {"lon": {"min": -98, "max": -95}, "lat": {"min": 10, "max": 14}}
    assert _box_selectable(traj, spec) is None


# ------------------------------------------------------------------ rectilinear box


def test_a_rectilinear_box_matches_two_independent_slices():
    grid = _gridded([10.0, 20.0, 30.0, 40.0])
    boxed = select(grid, {"lon": {"min": -96, "max": -93}, "lat": {"min": 15, "max": 35}})
    direct = grid.sel(lon=slice(-96, -93), lat=slice(15, 35))
    xr.testing.assert_equal(boxed, direct)
    assert boxed.attrs["region"] == [-96.0, 15.0, -93.0, 35.0]


def test_a_rectilinear_box_works_on_a_descending_axis():
    """MODIS-style latitude, stored north-to-south."""
    grid = _gridded([40.0, 30.0, 20.0, 10.0])
    boxed = select(grid, {"lon": {"min": -96, "max": -93}, "lat": {"min": 15, "max": 35}})
    assert sorted(boxed["lat"].values.tolist()) == [20.0, 30.0]


def test_a_seam_straddling_box_crops_a_180_grid_to_one_contiguous_band():
    lon = np.arange(-180.0, 180.0, 10.0)
    lat = np.array([0.0, 10.0])
    grid = xr.DataArray(
        np.arange(lat.size * lon.size, dtype=float).reshape(lat.size, lon.size),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    boxed = select(grid, {"lon": {"min": 170, "max": 190}, "lat": {"min": 0, "max": 10}})
    # 170 to 190 (0-360) is 170..180 union -180..-170 in this grid's own +/-180
    # storage -- a real band, not the whole globe and not empty.
    assert boxed.sizes["lon"] > 0
    assert boxed.sizes["lon"] < lon.size
    assert (boxed["lon"].values >= 170).all() or (boxed["lon"].values <= -170).all() \
        or set(np.sign(boxed["lon"].values)) <= {1.0, -1.0}


def test_an_empty_box_raises_with_selects_nothing_not_no_overlap():
    grid = _gridded([10.0, 20.0, 30.0])
    with pytest.raises(ValueError, match="selects nothing"):
        select(grid, {"lon": {"min": 40, "max": 50}, "lat": {"min": 10, "max": 30}})


def test_a_dataset_box_crops_too():
    """Unlike the point gate, a box does not require a single array to NaN-check."""
    grid = _gridded([10.0, 20.0, 30.0])
    ds = xr.Dataset({"sst": grid})
    out = select(ds, {"lon": {"min": -96, "max": -93}, "lat": {"min": 15, "max": 25}})
    assert out.sizes["lon"] > 0 and out.sizes["lat"] > 0


# ------------------------------------------------------------------- curvilinear box


def test_a_curvilinear_box_crops_the_index_window():
    grid = _curvilinear()
    boxed = select(
        grid, {"lon_rho": {"min": -148, "max": -146}, "lat_rho": {"min": 47, "max": 49}}
    )
    assert boxed.sizes["xi_rho"] < grid.sizes["xi_rho"]
    assert boxed.sizes["eta_rho"] < grid.sizes["eta_rho"]
    assert float(boxed["lon_rho"].min()) <= -146
    assert float(boxed["lon_rho"].max()) >= -148


def test_a_curvilinear_box_masks_cells_outside_it_but_inside_the_window():
    """A rotated grid's index window is a superset of the box; corners are masked."""
    grid = _rotated_grid()
    boxed = select(
        grid, {"lon_rho": {"min": 155, "max": 165}, "lat_rho": {"min": 15, "max": 25}}
    )
    lon, lat = boxed["lon_rho"].values, boxed["lat_rho"].values
    inside = (lon >= 155) & (lon <= 165) & (lat >= 15) & (lat <= 25)
    assert inside.any(), "the box must keep something"
    assert not inside.all(), "a rotated grid's window must be a strict superset"
    assert np.isnan(boxed.values[~inside]).all()
    assert not np.isnan(boxed.values[inside]).any()


def test_a_curvilinear_box_carries_no_matched_no_axis_warning():
    """resolve_dim alone can't see 2-D lon_rho/lat_rho; select's box branch can."""
    grid = _curvilinear()
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        select(
            grid,
            {"lon_rho": {"min": -148, "max": -146}, "lat_rho": {"min": 47, "max": 49}},
        )
    assert not log


def test_a_curvilinear_box_stamps_the_requested_region_in_attrs():
    grid = _curvilinear()
    boxed = select(
        grid, {"lon_rho": {"min": -148, "max": -146}, "lat_rho": {"min": 47, "max": 49}}
    )
    assert boxed.attrs["region"] == [-148.0, 47.0, -146.0, 49.0]


def test_a_curvilinear_box_with_no_overlap_raises():
    grid = _curvilinear()
    with pytest.raises(ValueError, match="selects nothing"):
        select(
            grid, {"lon_rho": {"min": 100, "max": 105}, "lat_rho": {"min": 47, "max": 49}}
        )


# ---------------------------------------------------------- align.subset_to_bbox too


def test_subset_to_bbox_now_windows_a_curvilinear_object():
    """The internal bbox crop (align/prepare's cost optimization) used to silently
    return a curvilinear object uncropped; it now windows it, without masking."""
    grid = _curvilinear()
    out = A.subset_to_bbox(grid, (-148.0, 47.0, -146.0, 49.0), pad=0.0)
    assert out.sizes["xi_rho"] < grid.sizes["xi_rho"]
    assert out.sizes["eta_rho"] < grid.sizes["eta_rho"]
    # a window, not a select -- nothing inside the kept index range is masked
    assert np.isfinite(out.values).all()


def test_subset_to_bbox_keeps_the_no_overlap_wording_on_a_curvilinear_miss():
    """Comparison.align's point-route retry matches on this literal phrase."""
    grid = _curvilinear()
    with pytest.raises(ValueError, match="no overlap"):
        A.subset_to_bbox(grid, (100.0, 47.0, 105.0, 49.0), pad=0.0)


# ------------------------------------------- a degenerate (point) bbox windows cells


def test_is_point_bbox_requires_exact_equality():
    """A near-degenerate box (a real, if tiny, region) still takes the padded path."""
    assert A._is_point_bbox((-148.0, 47.0, -148.0, 47.0))
    assert not A._is_point_bbox((-148.0, 47.0, -147.999, 47.0))
    assert not A._is_point_bbox(None)


def test_a_degenerate_bbox_windows_a_curvilinear_grid_to_cells_not_degrees():
    grid = _curvilinear()
    lon, lat = -148.3, 47.2
    out = A.subset_to_bbox(grid, (lon, lat, lon, lat))
    assert out.sizes["xi_rho"] <= 2 * A.POINT_WINDOW_CELLS + 1
    assert out.sizes["eta_rho"] <= 2 * A.POINT_WINDOW_CELLS + 1
    # centred on the true global-nearest cell, not just some window containing it
    iy, ix = A._nearest_indices(
        np.asarray(grid["lon_rho"]), np.asarray(grid["lat_rho"]), lon, lat
    )
    nearest_lon = float(grid["lon_rho"].values[iy, ix])
    nearest_lat = float(grid["lat_rho"].values[iy, ix])
    match = (out["lon_rho"].values == nearest_lon) & (
        out["lat_rho"].values == nearest_lat
    )
    assert match.any()
    # pad is ignored for a point -- a huge pad changes nothing
    huge_pad = A.subset_to_bbox(grid, (lon, lat, lon, lat), pad=50.0)
    xr.testing.assert_equal(out, huge_pad)


def test_a_degenerate_bbox_windows_a_rotated_curvilinear_grid_too():
    """A rotated grid's rows/columns are not lines of constant lon/lat -- same
    nearest-cell-plus-margin window applies regardless."""
    grid = _rotated_grid()
    lon = float(grid["lon_rho"].values[10, 15])
    lat = float(grid["lat_rho"].values[10, 15])
    out = A.subset_to_bbox(grid, (lon, lat, lon, lat))
    assert out.sizes["xi_rho"] <= 2 * A.POINT_WINDOW_CELLS + 1
    assert out.sizes["eta_rho"] <= 2 * A.POINT_WINDOW_CELLS + 1
    match = (out["lon_rho"].values == lon) & (out["lat_rho"].values == lat)
    assert match.any()


def test_a_degenerate_bbox_windows_a_rectilinear_grid():
    grid = _gridded(lat=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0))
    lon, lat = -95.5, 42.0
    out = A.subset_to_bbox(grid, (lon, lat, lon, lat))
    assert out.sizes["lon"] <= 2 * A.POINT_WINDOW_CELLS + 1
    assert out.sizes["lat"] <= 2 * A.POINT_WINDOW_CELLS + 1


def test_a_degenerate_bbox_windows_a_descending_latitude_axis_too():
    grid = _gridded(lat=(80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0))
    lon, lat = -95.5, 42.0
    out = A.subset_to_bbox(grid, (lon, lat, lon, lat))
    assert out.sizes["lon"] <= 2 * A.POINT_WINDOW_CELLS + 1
    assert out.sizes["lat"] <= 2 * A.POINT_WINDOW_CELLS + 1
    # the nearest value (40.0) is still in the kept window
    assert 40.0 in out["lat"].values


def test_a_point_far_outside_the_grid_never_raises():
    """Unlike a region bbox, a point always has *some* nearest cell."""
    grid = _curvilinear()
    out = A.subset_to_bbox(grid, (40.0, -60.0, 40.0, -60.0))
    assert out.sizes["xi_rho"] > 0 and out.sizes["eta_rho"] > 0


def test_a_point_windowed_bbox_samples_identically_to_the_full_grid():
    grid = _curvilinear()
    lon, lat = -148.3, 47.2
    windowed = A.subset_to_bbox(grid, (lon, lat, lon, lat))
    full_sample = A.sample_at(grid, lon, lat)
    windowed_sample = A.sample_at(windowed, lon, lat)
    assert float(windowed_sample) == float(full_sample)
    assert float(windowed_sample["lon_rho"]) == float(full_sample["lon_rho"])
    assert float(windowed_sample["lat_rho"]) == float(full_sample["lat_rho"])
