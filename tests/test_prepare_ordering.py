"""The surface selection must run before the non-vertical reduction, not after.

The reported failure: a ROMS store chunked one time record per chunk carries every
s_rho level in that one chunk (a per-timestep kerchunk reference, say). Doing
``aggregate={"time": "mean"}`` before the surface ``isel`` -- the general ordering
:func:`ocean_skill.comparison._prepare` otherwise wants, since an expensive vertical
*transform* should see as little as the reduction leaves it -- means the mean runs over
every level a chunk happens to store, and only then throws all but the top one away. A
plain surface request is a free ``isel``, unlike ``to_depth``/``depth_band``/
``to_sigma0``, and taking it first means the reduction only ever sees the top level.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from ocean_skill import operators
from ocean_skill.comparison import _prepare

META = {"model": "roms", "vertical": {"s_dim": "s_rho"}}


def _roms_like(nt=4, ns=6, ny=3, nx=4) -> xr.Dataset:
    """A tracer on (time, s_rho, eta_rho, xi_rho), one dask chunk per time step.

    Mirrors the real failure shape: the whole water column rides in a single chunk
    (as a per-timestep kerchunk reference does), and only the horizontal is split
    further -- so nothing here depends on ``s_rho`` ever being chunked on its own.
    """
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "temp": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                rng.normal(15.0, 2.0, (nt, ns, ny, nx)),
            ),
        },
        coords={"time": np.arange(nt)},
    )
    return ds.chunk({"time": 1, "s_rho": -1, "eta_rho": -1, "xi_rho": -1})


def test_surface_isel_runs_before_the_time_mean():
    """The non-vertical aggregate's input must already have s_rho collapsed."""
    pytest.importorskip("dask")
    ds = _roms_like()
    captured: dict[str, tuple[str, ...]] = {}
    real_aggregate = operators.aggregate

    def spy(da, spec):
        captured.setdefault("dims", da.dims)
        return real_aggregate(da, spec)

    with mock.patch("ocean_skill.operators.aggregate", side_effect=spy):
        _prepare(ds, META, "temp", {"depth": "surface"}, {"time": "mean"})

    assert "s_rho" not in captured["dims"], (
        "the time mean must see a field with the vertical axis already gone, "
        f"not {captured['dims']!r}"
    )


def test_surface_isel_runs_before_the_time_mean_with_depth_unset():
    """Unset depth means surface too (see is_surface_request) -- same hoist applies."""
    pytest.importorskip("dask")
    ds = _roms_like()
    captured: dict[str, tuple[str, ...]] = {}
    real_aggregate = operators.aggregate

    def spy(da, spec):
        captured.setdefault("dims", da.dims)
        return real_aggregate(da, spec)

    with mock.patch("ocean_skill.operators.aggregate", side_effect=spy):
        _prepare(ds, META, "temp", {}, {"time": "mean"})

    assert "s_rho" not in captured["dims"]


def test_the_hoist_does_not_change_the_result():
    """A hoisted isel changes nothing that a reduction over other dims touches."""
    pytest.importorskip("dask")
    ds = _roms_like()

    hoisted, _ = _prepare(ds, META, "temp", {"depth": "surface"}, {"time": "mean"})
    old_order = ds["temp"].isel(s_rho=-1).mean("time")

    xr.testing.assert_allclose(hoisted, old_order.drop_vars("s_rho", errors="ignore"))


def _roms_column_with_time(nt=3, ns=6, ny=2, nx=2) -> tuple[xr.Dataset, dict]:
    """A grid-carrying fixture -- ``depth_band``/``to_depth`` need it, surface not."""
    theta_s, theta_b = 5.0, 2.0

    def stretch(s):
        c = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
        return (np.exp(theta_b * c) - 1) / (1 - np.exp(-theta_b))

    h = np.array([[50.0, 100.0], [200.0, 500.0]])[:ny, :nx]
    sigma_r = (np.arange(1, ns + 1) - ns - 0.5) / ns
    sigma_w = np.linspace(-1, 0, ns + 1)
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "temp": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                rng.normal(15.0, 2.0, (nt, ns, ny, nx)),
            ),
            "h": (("eta_rho", "xi_rho"), h),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), stretch(sigma_w)),
        },
        coords={"time": np.arange(nt)},
    )
    ds = ds.chunk({"time": 1, "s_rho": -1, "eta_rho": -1, "xi_rho": -1})
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": 250.0}}
    return ds, meta


def test_a_depth_band_is_not_hoisted():
    """A band still needs the full column -- only the plain surface case is hoisted."""
    pytest.importorskip("dask")
    ds, meta = _roms_column_with_time()
    captured: dict[str, tuple[str, ...]] = {}
    real_aggregate = operators.aggregate

    def spy(da, spec):
        captured.setdefault("dims", da.dims)
        return real_aggregate(da, spec)

    with mock.patch("ocean_skill.operators.aggregate", side_effect=spy):
        _prepare(ds, meta, "temp", {"depth": {"min": 0, "max": 10}}, {"time": "mean"})

    assert "s_rho" in captured["dims"], "a band needs the whole column, not the top"


# ---------------------------------------- the horizontal spatial mean runs dead last


def _roms_column_with_lon_lat(nt=3, ns=6, ny=2, nx=2) -> tuple[xr.Dataset, dict]:
    """``_roms_column_with_time`` plus 2-D lon/lat, for a box select/spatial mean."""
    ds, meta = _roms_column_with_time(nt=nt, ns=ns, ny=ny, nx=nx)
    lon, lat = np.meshgrid(
        np.arange(nx, dtype=float) - 150.0, np.arange(ny, dtype=float) + 45.0
    )
    ds = ds.assign_coords(
        lon=(("eta_rho", "xi_rho"), lon), lat=(("eta_rho", "xi_rho"), lat)
    )
    return ds, meta


def test_the_spatial_mean_runs_after_the_vertical_ladder_not_with_the_early_aggregate():
    """The early (non-vertical) aggregate call must never see the lat/lon mean keys,
    and by the time a call *does* carry them, the vertical axis must already be gone
    (see the ordering note in ``ocean_skill.comparison._prepare``: to_depth/to_sigma0
    interpolate per column against horizontally-varying z_rho, and a depth-band mean
    must collapse each column's own thickness weights before columns are
    area-averaged together -- either breaks if the spatial mean runs first)."""
    pytest.importorskip("dask")
    ds, meta = _roms_column_with_lon_lat()
    calls: list[tuple[dict, tuple[str, ...]]] = []
    real_aggregate = operators.aggregate

    def spy(da, spec):
        calls.append((dict(spec or {}), da.dims))
        return real_aggregate(da, spec)

    with mock.patch("ocean_skill.operators.aggregate", side_effect=spy):
        _prepare(
            ds,
            meta,
            "temp",
            {
                "depth": {"min": 0, "max": 300},
                "lon": {"min": -150.5, "max": 0.5},
                "lat": {"min": 44.5, "max": 47.0},
            },
            {"time": "mean", "Z": "mean", "lat": "mean", "lon": "mean"},
        )

    spatial_calls = [c for c in calls if "lat" in c[0] and "lon" in c[0]]
    assert len(spatial_calls) == 1, "the spatial mean must run in exactly one call"
    spatial_spec, spatial_dims = spatial_calls[0]

    # every call before the spatial one carries no lat/lon key at all
    before = calls[: calls.index(spatial_calls[0])]
    assert all("lat" not in spec and "lon" not in spec for spec, _ in before)
    # and by the time the spatial call runs, the vertical axis is already gone --
    # the band's own dz weights already collapsed it, one column at a time
    assert "s_rho" not in spatial_dims
    assert {"eta_rho", "xi_rho"} <= set(spatial_dims)


def test_the_composition_matches_a_hand_computed_per_column_then_area_mean():
    """depth-band + {"Z": "mean"} + a box mean == weight each column by its own dz,
    collapse it, *then* area-average the four columns -- not the other order."""
    pytest.importorskip("dask")
    ds, meta = _roms_column_with_lon_lat()
    ds = ds.assign(temp=xr.ones_like(ds["temp"]))  # uniform: any correct order gives 1
    out, _ = _prepare(
        ds,
        meta,
        "temp",
        {"depth": {"min": 0, "max": 300}},
        {"time": "mean", "Z": "mean", "lat": "mean", "lon": "mean"},
    )
    assert out.dims == ()
    assert float(out) == pytest.approx(1.0)
    assert out.attrs["spatial_mean"].startswith("area-weighted mean")
