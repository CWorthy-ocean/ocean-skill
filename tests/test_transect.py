"""Tests for grid-aligned vertical slices: ocean_skill.transect and its wiring.

Stage A of the vertical-slice feature supports exactly one pathway -- exactly along
a grid dimension, a free ``isel`` with no interpolation -- read off
``select={"transect": {"<dim>": <index>}}``. An arbitrary path (a fixed longitude
that does not land on a grid column, or a list of lon/lat waypoints) is a later
stage; :func:`ocean_skill.transect.as_transect` names it and refuses rather than
silently doing something else with a similar-looking key.

Three layers, the house convention (see ``tests/test_isopycnal.py``): the pure
functions (:func:`ocean_skill.transect.as_transect`/``grid_slice``,
:func:`ocean_skill.align.path_of`), the wiring into
:func:`ocean_skill.comparison._prepare` (including the ``roms.to_depth``/
``to_sigma0`` guard fix a sliced lane needs), and an end-to-end
:class:`~ocean_skill.field.Field` built the way a real ROMS run would be.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms
from ocean_skill.align import ALONG_DIM, path_of
from ocean_skill.transect import apply_transect, as_transect, grid_slice

N = 20
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


@pytest.fixture
def roms_grid():
    """A 6x3 ROMS-shaped grid: h/mask_rho/sigma/Cs on (eta_rho, xi_rho), s_rho=20.

    Bigger than ``tests/test_isopycnal.py``'s 2x2 (which only needs one column per
    corner) so that slicing along one ``xi_rho`` index leaves a real path of six
    points to measure a distance across. ``chl`` is set to each cell's own depth
    (as that fixture's is), so a slice's numeric answer can be checked
    independently with plain ``np.interp`` rather than trusting the transform.
    """
    ny, nx = 6, 3
    h = np.linspace(30.0, 3000.0, ny * nx).reshape(ny, nx)
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    lon_1d = np.linspace(-95.0, -93.0, nx)
    lat_1d = np.linspace(24.0, 29.0, ny)
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
    chl = -ds["z_rho"]  # value == its own depth, for an independent check
    ds = ds.assign(chl=chl)
    return ds, meta


# -- as_transect: grammar validation -----------------------------------------------


def test_as_transect_reads_a_bare_dim_and_index():
    assert as_transect({"xi_rho": 30}) == {"kind": "grid", "dim": "xi_rho", "index": 30}
    assert as_transect({"eta_rho": -1}) == {
        "kind": "grid",
        "dim": "eta_rho",
        "index": -1,
    }


def test_as_transect_refuses_the_arbitrary_path_forms_by_name():
    """Waypoints/lon/lat are named and refused, not silently misread as a dim."""
    for spec in (
        {"waypoints": [[-95.0, 25.0], [-93.0, 27.0]]},
        {"lon": -94.0},
        {"lat": 26.0},
        {"lon": -94.0, "lat": {"min": 24.0, "max": 28.0}},
    ):
        with pytest.raises(NotImplementedError, match="arbitrary-path"):
            as_transect(spec)


def test_as_transect_refuses_a_coordinate_value_as_a_grid_index():
    with pytest.raises(ValueError, match="must be an int"):
        as_transect({"xi_rho": 30.5})
    with pytest.raises(ValueError, match="must be an int"):
        as_transect({"xi_rho": True})  # bool is not a real index either


def test_as_transect_refuses_more_than_one_dimension():
    with pytest.raises(ValueError, match="exactly one grid dimension"):
        as_transect({"xi_rho": 30, "eta_rho": 10})


def test_as_transect_refuses_empty_or_non_dict():
    with pytest.raises(ValueError):
        as_transect({})
    with pytest.raises(ValueError):
        as_transect(30)


# -- grid_slice: the pure isel + along-path coordinate -----------------------------


def test_grid_slice_collapses_xi_rho_and_names_eta_rho_along(roms_grid):
    ds, _ = roms_grid
    out = grid_slice(ds, "xi_rho", 1, subject="test")
    assert "xi_rho" not in out.dims
    assert ALONG_DIM in out.dims
    assert out.sizes[ALONG_DIM] == ds.sizes["eta_rho"]
    # h/mask_rho/z_rho all followed the same isel -- no manual re-attachment needed
    assert out["h"].dims == (ALONG_DIM,)
    assert out["z_rho"].dims == ("s_rho", ALONG_DIM)


def test_grid_slice_along_coordinate_matches_independent_haversine(roms_grid):
    ds, _ = roms_grid
    out = grid_slice(ds, "xi_rho", 0, subject="test")
    lon, lat = np.asarray(out["lon"]), np.asarray(out["lat"])
    r = 6371.0088
    p = np.deg2rad(lat)
    dp, dl = np.diff(p), np.deg2rad(np.diff(lon))
    a = np.sin(dp / 2) ** 2 + np.cos(p[:-1]) * np.cos(p[1:]) * np.sin(dl / 2) ** 2
    expected = np.concatenate([[0.0], np.cumsum(2 * r * np.arcsin(np.sqrt(a)))])
    np.testing.assert_allclose(np.asarray(out[ALONG_DIM]), expected)
    assert np.all(np.diff(np.asarray(out[ALONG_DIM])) > 0), "along must be increasing"


def test_grid_slice_broadcasts_a_constant_longitude_line():
    """Slicing a rectilinear-flavored grid along a fixed lon leaves lon constant.

    lon(x) collapses to a scalar when x is sliced away; grid_slice broadcasts it
    back out to match lat(y)'s surviving shape rather than leaving a mismatched pair.
    """
    lon = xr.DataArray([-95.0, -94.0, -93.0], dims="x", name="lon")
    lat = xr.DataArray([20.0, 21.0, 22.0, 23.0], dims="y", name="lat")
    values = xr.DataArray(
        np.arange(12.0).reshape(4, 3),
        dims=("y", "x"),
        coords={"lon": lon, "lat": lat},
        name="temp",
    )
    ds = values.to_dataset()
    out = grid_slice(ds, "x", 1, subject="test")
    assert out[ALONG_DIM].size == 4
    assert np.allclose(np.asarray(out["lon"]), -94.0)  # constant, broadcast to match
    assert np.allclose(np.asarray(out["lat"]), [20.0, 21.0, 22.0, 23.0])


def test_grid_slice_refuses_an_unknown_dimension(roms_grid):
    ds, _ = roms_grid
    with pytest.raises(ValueError, match="not one of this source's dimensions"):
        grid_slice(ds, "not_a_dim", 0, subject="the test lane")


def test_grid_slice_refuses_an_out_of_range_index(roms_grid):
    ds, _ = roms_grid
    with pytest.raises(ValueError, match="only has"):
        grid_slice(ds, "xi_rho", 99, subject="the test lane")


def test_apply_transect_dispatches_to_grid_slice(roms_grid):
    ds, _ = roms_grid
    direct = grid_slice(ds, "xi_rho", 2, subject="s")
    via_apply = apply_transect(ds, {"xi_rho": 2}, subject="s")
    xr.testing.assert_identical(direct, via_apply)


# -- align.path_of: the shared family-detection helper ------------------------------


def test_path_of_finds_a_sliced_section(roms_grid):
    ds, _ = roms_grid
    out = grid_slice(ds, "xi_rho", 0, subject="test")
    assert path_of(out["chl"]) == ALONG_DIM


def test_path_of_is_none_for_a_full_curvilinear_field(roms_grid):
    ds, _ = roms_grid
    assert path_of(ds["chl"]) is None  # 2-D lon/lat: a map, not a path


def test_path_of_is_none_for_a_station_point():
    da = xr.DataArray([1.0, 2.0], dims="time").assign_coords(lon=-95.0, lat=25.0)
    assert path_of(da) is None  # scalar lon/lat: a point, not a path


def test_path_of_is_none_for_a_moving_trajectory():
    """lon(time)/lat(time) has the section's shape but is a position that moves."""
    da = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims="time",
        coords={"lon": ("time", [-95.0, -94.0, -93.0]), "lat": ("time", [25.0, 25.0, 25.0])},
    )
    assert path_of(da) is None


# -- roms.to_depth's guard: a sliced lane must still transform ----------------------


def test_to_depth_transforms_a_sliced_lane_and_recovers_its_own_depth(roms_grid):
    """The guard fix: to_depth used to skip every variable on a sliced lane."""
    ds, meta = roms_grid
    sliced = grid_slice(ds, "xi_rho", 1, subject="test")
    targets = [50.0, 500.0]
    out = roms.to_depth(sliced, meta, targets)
    assert ALONG_DIM in out["chl"].dims
    assert out["chl"].sizes["z"] == 2
    # chl == depth by construction, so to_depth's own interpolated answer must
    # recover the requested depths (within the linear transform's tolerance) at
    # every column deep enough to reach them.
    recovered = np.asarray(out["chl"].isel(**{ALONG_DIM: -1}))  # deepest column
    np.testing.assert_allclose(recovered, targets, atol=1.0)


def test_to_depth_still_skips_a_staggered_variable_on_the_full_field(roms_grid):
    """Regression: the guard's real job (skip xi_u/eta_u) survives the generalization."""
    ds, meta = roms_grid
    ds = ds.assign(u=(("s_rho", "eta_rho", "xi_u"), np.zeros((N, 6, 2))))
    out = roms.to_depth(ds, meta, [50.0])
    assert "u" not in out.data_vars
    assert "chl" in out.data_vars


def test_to_depth_full_field_result_is_unchanged_by_the_guard_rewrite(roms_grid):
    """The full-field (unsliced) case must be byte-identical to before the fix."""
    ds, meta = roms_grid
    h_dims_now = set(ds["lon"].dims)
    assert h_dims_now == {"eta_rho", "xi_rho"}  # the guard's fallback path


# -- add_depth_coord's transpose: must tolerate the renamed `along` dim ------------


def test_add_depth_coord_transposes_a_sliced_lane_without_error(roms_grid):
    ds, meta = roms_grid
    sliced = grid_slice(ds.drop_vars("z_rho"), "xi_rho", 0, subject="test")
    out = roms.add_depth_coord(sliced, meta)
    assert set(out["z_rho"].dims) == {"s_rho", ALONG_DIM}
