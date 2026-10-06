"""Tests for ocean_skill/internal_tides.py: the baroclinic pressure flux calculator."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import internal_tides
from ocean_skill.internal_tides import X_FLUX, Y_FLUX, baroclinic_pressure_flux
from ocean_skill.operators import resolve_variable

RHO0 = 1027.0
# Time-mean up on xi_u (3 pts) is 1.5*[1, 2, 3]; vp on eta_v (2 pts) is 1.5*[10, 20].
# Edge-extend / 2-point average to rho (4 xi pts, 3 eta pts):
UP_RHO = np.array([1.5, 2.25, 3.75, 4.5])  # along xi_rho
VP_RHO = np.array([15.0, 22.5, 30.0])  # along eta_rho


def _ds(angle=0.0, up_units="m4 s-3", vp_units="m4 s-3"):
    t = np.arange(2)
    up = (t + 1)[:, None, None] * np.arange(1, 4)[None, None, :] * np.ones((1, 3, 1))
    vp = (
        (t + 1)[:, None, None]
        * 10
        * np.arange(1, 3)[None, :, None]
        * np.ones((1, 1, 4))
    )
    up_attrs = {} if up_units is None else {"units": up_units}
    vp_attrs = {} if vp_units is None else {"units": vp_units}
    return xr.Dataset(
        {
            X_FLUX: (("time", "eta_rho", "xi_u"), up, up_attrs),
            Y_FLUX: (("time", "eta_v", "xi_rho"), vp, vp_attrs),
        },
        coords={
            "time": t,
            "angle": (("eta_rho", "xi_rho"), np.full((3, 4), angle)),
            "lon": (("eta_rho", "xi_rho"), np.tile(np.arange(4.0), (3, 1))),
            "lat": (("eta_rho", "xi_rho"), np.tile(np.arange(3.0)[:, None], (1, 4))),
        },
    )


def test_unrotated_east_is_rho_averaged_time_mean_times_rho0():
    east = baroclinic_pressure_flux(_ds(), component="eastward")
    assert east.dims == ("eta_rho", "xi_rho")
    np.testing.assert_allclose(east.values, np.tile(UP_RHO * RHO0, (3, 1)))
    assert east.attrs["units"] == "W m-1"
    assert east.attrs["standard_name"] == "eastward_baroclinic_pressure_flux"
    assert "lon" in east.coords and "lat" in east.coords
    north = baroclinic_pressure_flux(_ds(), component="northward")
    np.testing.assert_allclose(north.values, np.tile(VP_RHO[:, None] * RHO0, (1, 4)))
    assert north.attrs["standard_name"] == "northward_baroclinic_pressure_flux"


def test_rotated_quarter_turn_swaps_components():
    ds = _ds(angle=np.pi / 2)
    east = baroclinic_pressure_flux(ds, component="eastward")
    north = baroclinic_pressure_flux(ds, component="northward")
    np.testing.assert_allclose(
        east.values, -np.tile(VP_RHO[:, None], (1, 4)) * RHO0, atol=1e-9
    )
    np.testing.assert_allclose(north.values, np.tile(UP_RHO, (3, 1)) * RHO0, atol=1e-9)


def test_units_conversion_and_already_watts():
    kw = baroclinic_pressure_flux(_ds(), component="eastward", units="kW m-1")
    np.testing.assert_allclose(kw.values[0], UP_RHO * RHO0 * 1e-3)
    assert kw.attrs["units"] == "kW m-1"
    w = baroclinic_pressure_flux(
        _ds(up_units="W m-1", vp_units="W m-1"), component="eastward"
    )
    np.testing.assert_allclose(w.values[0], UP_RHO)  # not multiplied by rho0


def test_missing_units_warns_and_assumes_m4_s3():
    with pytest.warns(UserWarning, match="assuming m4 s-3"):
        east = baroclinic_pressure_flux(
            _ds(up_units=None, vp_units=None), component="eastward"
        )
    np.testing.assert_allclose(east.values[0], UP_RHO * RHO0)


def test_missing_flux_or_angle_raises():
    with pytest.raises(KeyError, match="calc_pflx"):
        baroclinic_pressure_flux(_ds().drop_vars(Y_FLUX), component="eastward")
    with pytest.raises(KeyError, match="angle"):
        baroclinic_pressure_flux(_ds().drop_vars("angle"), component="eastward")


def test_resolves_through_operators():
    assert internal_tides  # imported explicitly: registration happens at import
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = resolve_variable(
            _ds(), {"calculate": "baroclinic_pressure_flux", "component": "eastward"}
        )
    np.testing.assert_allclose(out.values[0], UP_RHO * RHO0)
