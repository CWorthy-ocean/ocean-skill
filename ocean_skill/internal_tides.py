"""Internal-tide diagnostics: the depth-integrated baroclinic pressure flux.

UCLA ROMS computes the flux online (``src/calc_pflx_mod.F90``): ``up = sum_k
u_fast * p_fast * Hz`` on u-points and ``vp`` on v-points, grid-relative, where ``p``
there is pressure / rho0 -- so ``up``/``vp`` are in m4 s-3 and ``rho0 * up`` is W m-1.
:func:`baroclinic_pressure_flux` time-means them, averages to rho points and rotates
to true east/north, the same recipe as the velocity in
:func:`ocean_skill.roms._add_geographic_velocity`.

The model must have written ``up``/``vp`` (``calc_pflx`` on). Recomputing the flux
offline from 3-D u/v/T/S needs the tidal-band (fast) split of the fields and is not
supported.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

__all__ = ["baroclinic_pressure_flux"]

X_FLUX = "x_baroclinic_pressure_flux"  # ROMS up: staggered, grid-relative, m4 s-3
Y_FLUX = "y_baroclinic_pressure_flux"  # ROMS vp
EAST_FLUX = "eastward_baroclinic_pressure_flux"  # W m-1
NORTH_FLUX = "northward_baroclinic_pressure_flux"

_PER_RHO0 = "m4 s-3"  # what calc_pflx writes (pressure / rho0)
_POWER = "W m-1"


def _in_watts_per_m(da: xr.DataArray, rho0: float) -> xr.DataArray:
    """Return ``da`` in W m-1: times ``rho0`` if m4 s-3, as-is if already W m-1."""
    from ocean_skill.units import compatible

    src = da.attrs.get("units")
    if src is None:
        warnings.warn(
            f"{da.name!r} has no units attribute; assuming {_PER_RHO0} "
            "(pressure/rho0, as UCLA ROMS calc_pflx writes it).",
            stacklevel=3,
        )
        factor = rho0
    elif compatible(src, _PER_RHO0):
        factor = rho0
    elif compatible(src, _POWER):
        factor = 1.0
    else:
        raise ValueError(
            f"{da.name!r} has units {src!r}; expected {_PER_RHO0} or {_POWER}."
        )
    return (da * factor).assign_attrs(units=_POWER)


def baroclinic_pressure_flux(
    ds: xr.Dataset, *, component: str, rho0: float = 1027.0, units: str = _POWER
) -> xr.DataArray:
    """Time-mean depth-integrated baroclinic pressure flux, east or north.

    Needs the model's own ``up``/``vp`` (see the module docstring) and the grid
    ``angle`` coordinate. ``rho0`` converts UCLA ROMS's pressure/rho0 convention
    (m4 s-3) to W m-1; a field already in W m-1 is left alone. The result is 2-D on
    rho points, converted to ``units``.
    """
    from ocean_skill.operators import time_axis_dim
    from ocean_skill.roms import _average_to_rho
    from ocean_skill.units import find_variable, to_units

    names = {"eastward": EAST_FLUX, "northward": NORTH_FLUX}
    if component not in names:
        raise KeyError(f"component must be one of {sorted(names)}, got {component!r}.")
    up, vp = find_variable(ds, X_FLUX), find_variable(ds, Y_FLUX)
    if up is None or vp is None:
        raise KeyError(
            "baroclinic_pressure_flux needs the model's own depth-integrated pressure "
            "flux: write `up`/`vp` (UCLA ROMS `calc_pflx`). Computing it offline from "
            "3-D u/v/T/S is not supported."
        )
    if "angle" not in ds.coords:
        raise KeyError(
            "baroclinic_pressure_flux needs the grid `angle` coordinate to rotate."
        )

    flux = []
    for da, stagger, rho_dim in ((up, "xi_u", "xi_rho"), (vp, "eta_v", "eta_rho")):
        da = _in_watts_per_m(da, rho0)
        if (tdim := time_axis_dim(da)) is not None:
            da = da.mean(tdim, keep_attrs=True)
        flux.append(
            _average_to_rho(da, stagger, rho_dim) if stagger in da.dims else da.variable
        )
    u_rho, v_rho = flux
    # Same rotation as roms._add_geographic_velocity (angle: true east -> grid xi, CCW).
    angle = ds["angle"].variable
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated = (
        u_rho * cos_a - v_rho * sin_a
        if component == "eastward"
        else u_rho * sin_a + v_rho * cos_a
    )
    coords = {k: c for k, c in ds.coords.items() if set(c.dims) <= set(rotated.dims)}
    out = xr.DataArray(rotated, coords=coords, name=names[component])
    out.attrs = {
        "standard_name": names[component],
        "long_name": f"depth-integrated {component} baroclinic pressure flux",
        "units": _POWER,
    }
    return to_units(out, units)


def _register() -> None:
    from ocean_skill.operators import register_calculator

    register_calculator(
        "baroclinic_pressure_flux", inputs=lambda spec: [[X_FLUX, Y_FLUX]]
    )(baroclinic_pressure_flux)


_register()
