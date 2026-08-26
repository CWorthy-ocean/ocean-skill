"""Tests for the joint, area-weighted spatial mean.

``aggregate={"lat": "mean", "lon": "mean"}`` used to be two sequential unweighted
means — a silent no-op on a curvilinear (ROMS) grid, and on a rectilinear one, wrong
whenever latitude varies enough for a degree of longitude to mean a different area at
different rows, or whenever the wet-cell mask is ragged (see
``test_a_joint_mean_differs_from_mean_of_means_under_ragged_nan`` below for why a mean
of means is not a mean). This is now one joint, area-weighted reduction over both
horizontal axes together: true cell area on a ROMS grid that carries ``pm``/``pn``,
``cos(latitude)`` otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms
from ocean_skill.operators import aggregate, spatial_mean_in_spec

# --------------------------------------------------------------- spatial_mean_in_spec


def test_a_plain_mean_pair_is_recognized():
    assert spatial_mean_in_spec({"lat": "mean", "lon": "mean"}) == ("lon", "lat")


def test_the_dict_reduce_mean_spelling_is_recognized_too():
    hit = spatial_mean_in_spec({"lat": {"reduce": "mean"}, "lon": {"reduce": "mean"}})
    assert hit == ("lon", "lat")


def test_a_lone_zonal_mean_is_not_the_joint_path():
    assert spatial_mean_in_spec({"lat": "mean"}) is None


def test_a_mismatched_reduction_is_not_the_joint_path():
    assert spatial_mean_in_spec({"lat": "mean", "lon": "max"}) is None


def test_a_groupby_or_resample_on_either_axis_is_not_the_joint_path():
    assert spatial_mean_in_spec({"lat": "mean", "lon": {"groupby": "x", "reduce": "mean"}}) is None
    assert spatial_mean_in_spec({"lat": "mean", "lon": {"resample": "1D", "reduce": "mean"}}) is None


def test_empty_or_missing_spec_is_not_the_joint_path():
    assert spatial_mean_in_spec({}) is None
    assert spatial_mean_in_spec(None) is None


# --------------------------------------------------------------------- rectilinear


def test_a_uniform_field_area_means_to_itself():
    lat, lon = np.array([-10.0, 0.0, 50.0]), np.array([0.0, 10.0, 20.0])
    da = xr.DataArray(np.full((3, 3), 7.5), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    out = aggregate(da, {"lat": "mean", "lon": "mean"})
    assert out.dims == ()
    assert float(out) == pytest.approx(7.5)


def test_a_rectilinear_mean_is_weighted_by_cos_latitude():
    lat, lon = np.array([0.0, 60.0]), np.array([10.0, 20.0])
    da = xr.DataArray(
        np.array([[10.0, 10.0], [20.0, 20.0]]),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "degC"},
    )
    out = aggregate(da, {"lat": "mean", "lon": "mean"})
    w0, w1 = np.cos(np.deg2rad(0.0)), np.cos(np.deg2rad(60.0))
    expected = (10 * 2 * w0 + 20 * 2 * w1) / (2 * w0 + 2 * w1)
    assert float(out) == pytest.approx(expected)
    assert float(out) != pytest.approx(float(da.mean()))
    assert out.attrs["spatial_mean"] == "area-weighted mean (cos latitude)"
    assert out.attrs["units"] == "degC"  # a reduction drops attrs; units must survive


def test_a_mixed_reduction_keeps_the_ordinary_unweighted_per_axis_behavior():
    lat, lon = np.array([0.0, 60.0]), np.array([10.0, 20.0])
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    )
    out = aggregate(da, {"lat": "mean", "lon": "max"})
    # sequential: max over lon per lat row -> [2, 4], then a plain unweighted mean -> 3
    assert float(out) == pytest.approx(3.0)
    assert "spatial_mean" not in out.attrs


def test_X_Y_spelling_is_recognized_the_same_way():
    lat, lon = np.array([0.0, 10.0]), np.array([0.0, 10.0])
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    )
    out = aggregate(da, {"X": "mean", "Y": "mean"})
    assert out.dims == ()


def test_a_joint_mean_differs_from_mean_of_means_under_ragged_nan():
    """The reason for a joint reduction rather than two sequential per-axis means."""
    lon, lat = np.meshgrid([10.0, 11.0, 12.0], [40.0, 41.0])
    value = np.array([[1.0, 2.0, 3.0], [10.0, np.nan, np.nan]])
    da = xr.DataArray(
        value,
        dims=("eta_rho", "xi_rho"),
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "cell_area": (("eta_rho", "xi_rho"), np.ones((2, 3))),
        },
    )
    out = aggregate(da, {"lon_rho": "mean", "lat_rho": "mean"})
    assert float(out) == pytest.approx(4.0)  # true (uniform-weight) mean of the 4 valid cells
    sequential = float(da.mean("xi_rho").mean("eta_rho"))  # the wrong number, for contrast
    assert sequential == pytest.approx(6.0)
    assert float(out) != pytest.approx(sequential)


# --------------------------------------------------------------------- curvilinear


def test_a_curvilinear_mean_uses_cell_area_not_cos_latitude_when_present():
    lon, lat = np.meshgrid([10.0, 11.0], [40.0, 60.0])
    value = np.array([[10.0, 20.0], [30.0, 40.0]])
    area = np.array([[1.0, 1.0], [3.0, 3.0]])  # the lat=60 row weighted 3x the lat=40 row
    da = xr.DataArray(
        value,
        dims=("eta_rho", "xi_rho"),
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "cell_area": (("eta_rho", "xi_rho"), area),
        },
        attrs={"units": "degC"},
    )
    out = aggregate(da, {"lon_rho": "mean", "lat_rho": "mean"})
    # (10*1 + 20*1 + 30*3 + 40*3) / (1+1+3+3) = 240/8 = 30 -- cos(lat) would give a
    # materially different number (~22.9) since 40 deg and 60 deg have different
    # cosines, which is exactly what this test would catch if cell_area were ignored.
    assert float(out) == pytest.approx(30.0)
    assert out.attrs["spatial_mean"] == "area-weighted mean (cell_area)"


def test_a_curvilinear_mean_falls_back_to_cos_latitude_without_cell_area():
    lon, lat = np.meshgrid([10.0, 11.0], [0.0, 60.0])
    da = xr.DataArray(
        np.array([[10.0, 10.0], [20.0, 20.0]]),
        dims=("eta_rho", "xi_rho"),
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
        },
    )
    out = aggregate(da, {"lon_rho": "mean", "lat_rho": "mean"})
    assert out.attrs["spatial_mean"] == "area-weighted mean (cos latitude)"


# ------------------------------------------------------------- the box's own midpoint


def test_the_mean_carries_the_requested_boxs_midpoint_when_region_is_present():
    lat, lon = np.array([40.0, 42.0]), np.array([-150.0, -148.0])
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    )
    da.attrs["region"] = [-151.0, 39.0, -147.0, 43.0]  # wider than the data itself
    out = aggregate(da, {"lat": "mean", "lon": "mean"})
    assert float(out["lon"]) == pytest.approx(-149.0)
    assert float(out["lat"]) == pytest.approx(41.0)
    assert out.attrs["region"] == [-151.0, 39.0, -147.0, 43.0]


def test_the_mean_falls_back_to_the_datas_own_extent_with_no_requested_box():
    lat, lon = np.array([40.0, 42.0]), np.array([-150.0, -148.0])
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
    )
    out = aggregate(da, {"lat": "mean", "lon": "mean"})
    assert float(out["lon"]) == pytest.approx(-149.0)
    assert float(out["lat"]) == pytest.approx(41.0)
    assert "region" not in out.attrs


# --------------------------------------------------------- roms.standardize attaches it


@pytest.fixture
def raw_roms_grid():
    """A minimal pre-standardize ROMS Dataset: rho-point field plus its own grid."""
    ny, nx = 3, 4
    lon, lat = np.meshgrid(np.arange(nx, dtype=float) + 10.0, np.arange(ny, dtype=float) + 40.0)

    def build(with_pm_pn: bool):
        data = {
            "temp": (
                ("eta_rho", "xi_rho"),
                np.arange(ny * nx, dtype=float).reshape(ny, nx),
                {"units": "degC"},
            ),
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "h": (("eta_rho", "xi_rho"), np.full((ny, nx), 100.0)),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((ny, nx))),
            "Cs_r": (("s_rho",), np.array([-0.5])),
            "sigma_r": (("s_rho",), np.array([-0.5])),
        }
        if with_pm_pn:
            data["pm"] = (("eta_rho", "xi_rho"), np.full((ny, nx), 1.0 / 1000.0))
            data["pn"] = (("eta_rho", "xi_rho"), np.full((ny, nx), 1.0 / 2000.0))
        ds = xr.Dataset(data)
        meta = {
            "model": "roms",
            "self_contained_grid": True,
            "standard_names": {"temp": "sea_water_potential_temperature"},
            "vertical": {"s_dim": "s_rho", "hc": 250.0},
        }
        return ds, meta

    return build


def test_standardize_attaches_cell_area_from_pm_pn(raw_roms_grid):
    ds, meta = raw_roms_grid(with_pm_pn=True)
    out = roms.standardize(ds, meta)
    assert roms.AREA_COORD in out.coords
    # 1 / (pm * pn) = 1 / ((1/1000) * (1/2000)) = 2,000,000 m^2
    assert float(out[roms.AREA_COORD].isel(eta_rho=0, xi_rho=0)) == pytest.approx(2_000_000.0)


def test_standardize_omits_cell_area_without_pm_pn(raw_roms_grid):
    ds, meta = raw_roms_grid(with_pm_pn=False)
    out = roms.standardize(ds, meta)
    assert roms.AREA_COORD not in out.coords
