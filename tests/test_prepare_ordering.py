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
