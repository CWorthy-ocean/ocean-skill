"""Tests for arbitrary-path vertical slices: waypoints, points, and lon/lat lines.

The grid-aligned pathway (``select={"transect": {"<dim>": <index>}}``, a pure
``isel``) is ``tests/test_transect.py``'s subject. This file covers the other one:
a list of lon/lat waypoints, a resolved ``points`` list (sampled exactly as given,
no densification), or a fixed longitude/latitude line -- densified to roughly the
model's own resolution (:func:`ocean_skill.transect.densify_waypoints`) and read
off the grid by nearest-neighbour (default) or bilinear interpolation
(:func:`ocean_skill.transect.sample_along`). Both pathways finish through the same
:func:`ocean_skill.transect._attach_along_coord`, so ``align.path_of`` and the
``section`` plot family read either one identically -- that shared contract is
``tests/test_transect.py``'s to pin, not this file's.

Three layers, the house convention: the pure grammar/densify functions (checked
against independent computation, not the code under test); ``sample_along`` itself,
nearest checked against a hand-rolled per-point loop and bilinear checked against an
analytically linear field; and ``apply_transect``'s dispatch, end to end on a small
synthetic ROMS grid. Bilinear-on-a-curvilinear-grid tests need xesmf and
``importorskip`` it (skips locally where it is not installed; runs in CI/conda) --
bilinear-on-a-*rectilinear* grid needs only ``xarray.Dataset.interp`` and runs
everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms
from ocean_skill.align import ALONG_DIM, _haversine_km, _nearest_indices
from ocean_skill.transect import (
    apply_transect,
    as_transect,
    densify_waypoints,
    sample_along,
)

N = 12
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


@pytest.fixture
def roms_grid():
    """A 6x4 ROMS-shaped grid, big enough for a real multi-cell path across it."""
    ny, nx = 6, 4
    h = np.linspace(30.0, 2500.0, ny * nx).reshape(ny, nx)
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    lon_1d = np.linspace(-96.0, -93.0, nx)
    lat_1d = np.linspace(23.0, 28.0, ny)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    ds = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((ny, nx))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
        },
        coords={
            "lon": (("eta_rho", "xi_rho"), lon_2d),
            "lat": (("eta_rho", "xi_rho"), lat_2d),
        },
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    ds = roms.add_depth_coord(ds, meta)
    # linear in lon/lat so a bilinear sample can be checked against a plane, not
    # just against itself
    chl = 5.0 + 0.5 * (ds["lon"] + 95.0) + 0.3 * (ds["lat"] - 25.0)
    ds = ds.assign(chl=chl.broadcast_like(ds["z_rho"]))
    return ds, meta


def _rectilinear_grid(descending_lat: bool = False):
    lon = np.linspace(-96.0, -93.0, 7)
    lat = np.linspace(23.0, 28.0, 6)
    if descending_lat:
        lat = lat[::-1]
    lon2d, lat2d = np.meshgrid(lon, lat)
    values = 5.0 + 0.5 * (lon2d + 95.0) + 0.3 * (lat2d - 25.0)
    return xr.DataArray(
        values, dims=("lat", "lon"), coords={"lon": lon, "lat": lat}, name="chl"
    ).to_dataset()


# -- as_transect: the arbitrary-path grammar ----------------------------------------


def test_as_transect_reads_waypoints():
    parsed = as_transect({"waypoints": [[-95.0, 24.0], [-94.0, 25.0], [-93.5, 26.0]]})
    assert parsed == {
        "kind": "waypoints",
        "waypoints": [[-95.0, 24.0], [-94.0, 25.0], [-93.5, 26.0]],
        "spacing_km": None,
        "method": "nearest",
    }


def test_as_transect_tuple_and_list_waypoints_normalize_identically():
    a = as_transect({"waypoints": ((-95.0, 24.0), (-94.0, 25.0))})
    b = as_transect({"waypoints": [[-95.0, 24.0], [-94.0, 25.0]]})
    assert a == b


def test_as_transect_reads_points_with_no_densification():
    parsed = as_transect({"points": [[-95.0, 24.0], [-94.0, 25.0]], "method": "bilinear"})
    assert parsed == {
        "kind": "points",
        "points": [[-95.0, 24.0], [-94.0, 25.0]],
        "method": "bilinear",
    }


def test_as_transect_reads_a_lon_line_with_open_bounds():
    parsed = as_transect({"lon": -94.0})
    assert parsed == {
        "kind": "lon_line",
        "lon": -94.0,
        "lat_bounds": [None, None],
        "spacing_km": None,
        "method": "nearest",
    }


def test_as_transect_reads_a_bounded_lon_line():
    parsed = as_transect({"lon": -94.0, "lat": {"min": 24.0, "max": 26.0}})
    assert parsed["kind"] == "lon_line"
    assert parsed["lat_bounds"] == [24.0, 26.0]


def test_as_transect_reads_a_lat_line():
    parsed = as_transect({"lat": 25.0, "spacing_km": 10.0})
    assert parsed == {
        "kind": "lat_line",
        "lat": 25.0,
        "lon_bounds": [None, None],
        "spacing_km": 10.0,
        "method": "nearest",
    }


def test_as_transect_normalizes_linear_to_bilinear():
    assert as_transect({"lat": 25.0, "method": "linear"})["method"] == "bilinear"
    assert as_transect({"lat": 25.0, "method": "LINEAR"})["method"] == "bilinear"


def test_as_transect_refuses_a_single_waypoint():
    with pytest.raises(ValueError, match="at least 2 points"):
        as_transect({"waypoints": [[-95.0, 24.0]]})


def test_as_transect_refuses_a_swapped_lon_lat_pair():
    with pytest.raises(ValueError, match="did you swap them"):
        as_transect({"waypoints": [[24.0, -95.0], [25.0, -94.0]]})


def test_as_transect_refuses_both_lon_and_lat_scalar():
    with pytest.raises(ValueError, match="that names one place"):
        as_transect({"lon": -94.0, "lat": 25.0})


def test_as_transect_refuses_both_lon_and_lat_range():
    with pytest.raises(ValueError, match="that names a box"):
        as_transect(
            {"lon": {"min": -95.0, "max": -93.0}, "lat": {"min": 24.0, "max": 26.0}}
        )


def test_as_transect_refuses_a_lone_range_with_no_fixed_axis():
    with pytest.raises(ValueError, match="fixes one axis"):
        as_transect({"lat": {"min": 24.0, "max": 26.0}})


def test_as_transect_refuses_points_with_spacing_km():
    with pytest.raises(ValueError, match="sampled exactly where given"):
        as_transect({"points": [[-95.0, 24.0], [-94.0, 25.0]], "spacing_km": 5.0})


def test_as_transect_refuses_more_than_one_path_form():
    with pytest.raises(ValueError, match="more than one path form"):
        as_transect({"waypoints": [[-95.0, 24.0], [-94.0, 25.0]], "lon": -94.0})


def test_as_transect_refuses_a_conservative_method():
    with pytest.raises(ValueError, match="no \\*area\\*"):
        as_transect({"lat": 25.0, "method": "conservative_normed"})


def test_as_transect_refuses_a_non_numeric_spacing_km():
    with pytest.raises(ValueError, match="positive"):
        as_transect({"lat": 25.0, "spacing_km": "far"})
    with pytest.raises(ValueError, match="positive"):
        as_transect({"lat": 25.0, "spacing_km": -5.0})


# -- densify_waypoints ---------------------------------------------------------------


def test_densify_waypoints_honors_spacing():
    lons, lats = densify_waypoints([[-150.0, 45.0], [-148.0, 46.5]], spacing_km=20.0)
    seg_km = _haversine_km(lons[:-1], lats[:-1], lons[1:], lats[1:])
    assert np.all(seg_km <= 20.0 * 1.01)  # a hair of float slack


def test_densify_waypoints_keeps_a_vertex_at_each_waypoint():
    waypoints = [[-150.0, 45.0], [-148.0, 46.5], [-145.0, 47.0]]
    lons, lats = densify_waypoints(waypoints, spacing_km=500.0)  # coarse: few points
    for lon, lat in waypoints:
        assert np.any(np.isclose(lons, lon) & np.isclose(lats, lat))


def test_densify_waypoints_crosses_the_antimeridian_the_short_way():
    lons, lats = densify_waypoints([[170.0, 10.0], [-170.0, 10.0]], spacing_km=200.0)
    # unwrapped, the short 20-degree hop runs 170 -> 190, never doubling back
    assert lons.min() >= 169.0
    assert lons.max() <= 191.0
    assert np.all(np.diff(lons) > 0)


def test_densify_waypoints_dedupes_coincident_points():
    lons, lats = densify_waypoints(
        [[-150.0, 45.0], [-150.0, 45.0], [-148.0, 46.5]], spacing_km=500.0
    )
    assert lons.size >= 2  # collapsed, not tripled


def test_densify_waypoints_refuses_a_single_distinct_point():
    with pytest.raises(ValueError, match="at least 2 distinct points"):
        densify_waypoints([[-150.0, 45.0], [-150.0, 45.0]], spacing_km=10.0)


# -- sample_along: nearest, curvilinear -----------------------------------------------


def test_sample_along_nearest_matches_a_per_point_loop(roms_grid):
    ds, _ = roms_grid
    lons = np.array([-95.5, -95.0, -94.5, -94.0])
    lats = np.array([23.5, 24.5, 25.5, 26.5])
    out = sample_along(ds, lons, lats, method="nearest", subject="test")

    lon2d, lat2d = np.asarray(ds["lon"]), np.asarray(ds["lat"])
    expected = [_nearest_indices(lon2d, lat2d, lo, la) for lo, la in zip(lons, lats)]
    expected_chl = np.array(
        [float(ds["chl"].isel(eta_rho=iy, xi_rho=ix, s_rho=-1)) for iy, ix in expected]
    )
    np.testing.assert_allclose(np.asarray(out["chl"].isel(s_rho=-1)), expected_chl)


def test_sample_along_output_matches_grid_slice_contract(roms_grid):
    ds, _ = roms_grid
    out = sample_along(
        ds, [-95.5, -94.5, -93.5], [24.0, 25.0, 26.0], method="nearest", subject="t"
    )
    assert ALONG_DIM in out.dims
    assert np.all(np.diff(np.asarray(out[ALONG_DIM])) > 0)
    assert out[ALONG_DIM].attrs["path_method"] == "nearest"
    assert out["lon"].dims == (ALONG_DIM,)
    assert out["lat"].dims == (ALONG_DIM,)


def test_sample_along_drops_out_of_domain_points_with_one_warning(roms_grid):
    ds, _ = roms_grid
    # two points inside the domain, one far outside it
    lons = [-95.5, -94.5, 40.0]
    lats = [24.0, 25.0, 89.0]
    with pytest.warns(UserWarning, match="dropped"):
        out = sample_along(ds, lons, lats, method="nearest", subject="test")
    assert out.sizes[ALONG_DIM] == 2


def test_sample_along_raises_when_the_whole_path_misses_the_domain(roms_grid):
    ds, _ = roms_grid
    with pytest.raises(ValueError, match="does not cross"):
        sample_along(ds, [40.0, 41.0], [89.0, 88.0], method="nearest", subject="test")


def test_sample_along_collapses_consecutive_duplicate_cells(roms_grid):
    ds, _ = roms_grid
    # two clusters of points, each crammed inside one cell's worth of space, but
    # far enough apart from each other to land on two different cells -- should
    # collapse to 2 along-path entries, not the 6 requested
    lons = np.array([-95.95, -95.9, -95.85, -94.15, -94.1, -94.05])
    lats = np.full(6, 24.0)
    out = sample_along(ds, lons, lats, method="nearest", subject="test")
    assert out.sizes[ALONG_DIM] == 2
    assert np.all(np.diff(np.asarray(out[ALONG_DIM])) > 0)


# -- sample_along: nearest, rectilinear -----------------------------------------------


def test_sample_along_rectilinear_nearest_matches_independent_lookup():
    ds = _rectilinear_grid()
    lons = np.array([-95.4, -94.6, -93.9])
    lats = np.array([23.4, 25.1, 27.6])
    out = sample_along(ds, lons, lats, method="nearest", subject="test")

    lon_vals, lat_vals = np.asarray(ds["lon"]), np.asarray(ds["lat"])
    ix = [int(np.abs(lon_vals - lo).argmin()) for lo in lons]
    iy = [int(np.abs(lat_vals - la).argmin()) for la in lats]
    expected = np.array([float(ds["chl"].isel(lon=x, lat=y)) for x, y in zip(ix, iy)])
    np.testing.assert_allclose(np.asarray(out["chl"]), expected)


def test_sample_along_rectilinear_handles_descending_latitude():
    ds = _rectilinear_grid(descending_lat=True)
    out = sample_along(
        ds, [-95.5, -94.5, -93.5], [24.0, 25.0, 26.0], method="nearest", subject="t"
    )
    assert out.sizes[ALONG_DIM] >= 2
    assert np.all(np.isfinite(np.asarray(out["chl"])))


# -- sample_along: bilinear, rectilinear (no xesmf needed) -----------------------------


def test_sample_along_rectilinear_bilinear_recovers_the_exact_plane():
    ds = _rectilinear_grid()
    lons = np.array([-95.3, -94.7, -94.0])
    lats = np.array([23.7, 25.2, 27.1])
    out = sample_along(ds, lons, lats, method="bilinear", subject="test")
    expected = 5.0 + 0.5 * (lons + 95.0) + 0.3 * (lats - 25.0)
    np.testing.assert_allclose(np.asarray(out["chl"]), expected, atol=1e-8)


# -- sample_along: bilinear, curvilinear (needs xesmf) ----------------------------------


def test_sample_along_curvilinear_bilinear_recovers_the_exact_plane(roms_grid):
    pytest.importorskip("xesmf")
    ds, _ = roms_grid
    lons = np.array([-95.3, -94.7, -94.0])
    lats = np.array([23.7, 25.2, 27.1])
    out = sample_along(ds, lons, lats, method="bilinear", subject="test")
    expected = 5.0 + 0.5 * (lons + 95.0) + 0.3 * (lats - 25.0)
    np.testing.assert_allclose(
        np.asarray(out["chl"].isel(s_rho=-1)), expected, rtol=1e-3
    )


def test_bilinear_curvilinear_drops_z_rho_and_keeps_h(roms_grid):
    pytest.importorskip("xesmf")
    ds, _ = roms_grid
    out = sample_along(ds, [-95.3, -94.0], [23.7, 27.1], method="bilinear", subject="t")
    assert "z_rho" not in out.coords
    assert "h" in out.variables


def test_bilinear_curvilinear_drops_a_staggered_variable_with_a_warning(roms_grid):
    pytest.importorskip("xesmf")
    ds, _ = roms_grid
    ds = ds.assign(
        u=(("s_rho", "eta_rho", "xi_u"), np.zeros((N, ds.sizes["eta_rho"], 2)))
    )
    with pytest.warns(UserWarning, match="cannot interpolate"):
        out = sample_along(ds, [-95.3, -94.0], [23.7, 27.1], method="bilinear", subject="t")
    assert "u" not in out.data_vars
    assert "chl" in out.data_vars


def test_bilinear_dataset_classification_without_needing_xesmf_installed(
    monkeypatch, roms_grid
):
    """Stub the one actual xesmf call to test ``_bilinear_dataset``'s own logic in
    isolation -- which variables it sends to regrid, which it passes through
    untouched, which it drops, and how it restores h/mask_rho to coordinates --
    all logic this module wrote itself, independent of whether xesmf is
    installed. The interpolation math xesmf performs is covered separately (see
    the tests above, which need the real package and importorskip it).
    """
    import ocean_skill.align as align_mod
    from ocean_skill.transect import _bilinear_dataset

    ds, _ = roms_grid
    ds = ds.assign(
        u=(("s_rho", "eta_rho", "xi_u"), np.zeros((N, ds.sizes["eta_rho"], 2))),
        spherical=xr.DataArray("T"),
    )
    # roms.standardize() attaches h/mask_rho as *coordinates* in the real
    # pipeline (this fixture, like the rest of the test suite, builds them as
    # plain data variables for simplicity) -- set that up explicitly here so
    # the promote-then-restore path this test exists to check actually runs.
    ds = ds.set_coords(["h", "mask_rho"])

    def _fake_interp_locstream(obj, lons, lats):
        spatial_dims = obj["lon"].dims
        n = len(lons)
        out = obj.isel({d: 0 for d in spatial_dims}).expand_dims({ALONG_DIM: n})
        return out.assign_coords(
            {
                "lon": (ALONG_DIM, np.asarray(lons, dtype="float64")),
                "lat": (ALONG_DIM, np.asarray(lats, dtype="float64")),
            }
        )

    monkeypatch.setattr(align_mod, "_interp_locstream", _fake_interp_locstream)

    lons, lats = np.array([-95.0, -94.0]), np.array([24.0, 26.0])
    with pytest.warns(UserWarning, match="cannot interpolate"):
        out = _bilinear_dataset(ds, "lon", "lat", lons, lats)

    assert "u" not in out.data_vars  # staggered onto a different grid point
    assert "chl" in out.data_vars  # rho-point tracer -- regridded
    assert "h" in out.coords  # promoted for the regrid, restored to a coordinate
    assert "mask_rho" in out.coords
    assert "z_rho" not in out.variables  # dropped -- re-derived downstream instead
    assert "Cs_r" in out.variables  # no horizontal dim at all -- passed through
    assert "spherical" in out.variables  # scalar flag -- passed through
    assert out.sizes[ALONG_DIM] == 2


# -- apply_transect: end-to-end dispatch ----------------------------------------------


def test_apply_transect_waypoints(roms_grid):
    ds, _ = roms_grid
    out = apply_transect(
        ds, {"waypoints": [[-95.5, 24.0], [-94.0, 26.0]]}, subject="test"
    )
    assert ALONG_DIM in out.dims
    assert out.sizes[ALONG_DIM] > 2  # densified, not just the two waypoints


def test_apply_transect_points_samples_exactly_two(roms_grid):
    ds, _ = roms_grid
    out = apply_transect(
        ds, {"points": [[-95.5, 24.0], [-94.0, 26.0]]}, subject="test"
    )
    assert out.sizes[ALONG_DIM] == 2  # no densification


def test_apply_transect_lon_line_fills_bounds_from_the_domain(roms_grid):
    ds, _ = roms_grid
    out = apply_transect(ds, {"lon": -94.5}, subject="test")
    lat = np.asarray(out["lat"])
    assert lat.min() < 24.0 and lat.max() > 27.0  # spans close to the domain's own


def test_apply_transect_bilinear_dispatch_without_needing_xesmf_installed(
    monkeypatch, roms_grid
):
    """The full grammar -> densify -> sample_along -> _bilinear_dataset chain,
    stubbing only the actual xesmf call (see the classification test above) --
    checks the pieces are wired together correctly, not just each in isolation.
    """
    import ocean_skill.align as align_mod

    ds, _ = roms_grid
    ds = ds.set_coords(["h", "mask_rho"])

    def _fake_interp_locstream(obj, lons, lats):
        spatial_dims = obj["lon"].dims
        n = len(lons)
        out = obj.isel({d: 0 for d in spatial_dims}).expand_dims({ALONG_DIM: n})
        return out.assign_coords(
            {
                "lon": (ALONG_DIM, np.asarray(lons, dtype="float64")),
                "lat": (ALONG_DIM, np.asarray(lats, dtype="float64")),
            }
        )

    monkeypatch.setattr(align_mod, "_interp_locstream", _fake_interp_locstream)

    out = apply_transect(
        ds,
        {"waypoints": [[-95.5, 24.0], [-94.0, 26.0]], "method": "bilinear"},
        subject="test",
    )
    assert ALONG_DIM in out.dims
    assert out[ALONG_DIM].attrs["path_method"] == "bilinear"
    assert out.sizes[ALONG_DIM] > 2  # still densified before sampling


def test_apply_transect_lat_line_respects_explicit_bounds(roms_grid):
    ds, _ = roms_grid
    out = apply_transect(
        ds, {"lat": 25.0, "lon": {"min": -95.0, "max": -94.0}}, subject="test"
    )
    lon = np.asarray(out["lon"])
    assert lon.min() >= -95.3
    assert lon.max() <= -93.7
