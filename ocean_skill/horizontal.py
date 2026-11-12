"""Horizontal calculators: how fast a 2-D field changes in space.

One registered calculator, which reads a variable's horizontal layer and returns the
magnitude of its gradient (a map, per time step and per level)::

    {"calculate": "horizontal_gradient", "variable": "oxygen"}

It is a ``layer`` calculator: the pipeline first picks the depth layer (an interpolated
300 m on ROMS, a standard level on WOA), takes the time mean and converts units, and
then hands the resulting *DataArray* here -- so the vertical and time choices live in
the spec's ``select``/``aggregate``, not in this function.

The gradient is a centered difference along each of the grid's two horizontal axes,
divided by the great-circle distance (km) between the two neighboring cells. The two
components are combined as ``sqrt(g1**2 + g2**2)``, which does not depend on how the
grid is rotated, so a rectilinear lat/lon grid and a curvilinear one (ROMS
``eta``/``xi``) are handled the same way and neither needs the grid angle. The haversine
distance is periodic in longitude, so a grid crossing the dateline, in either the
-180..180 or the 0..360 convention, has the right spacing across the seam.

Cells on the domain edge, NaN (land) cells, and cells next to a NaN neighbor are NaN:
the centered difference needs both neighbors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

__all__ = ["calculate_horizontal_gradient"]


def _horizontal_axes(da: xr.DataArray) -> tuple[tuple[str, str], Any, Any]:
    """Return ``(dims, lon2d, lat2d)``: the two horizontal dims and 2-D coordinates.

    ``lon2d``/``lat2d`` are bare (coordinate-free) arrays over those two dims, so they
    shift and broadcast against the data by dimension name alone.
    """
    from ocean_skill.cf import find_coord

    lon, lat = find_coord(da, "longitude"), find_coord(da, "latitude")
    if lon is None or lat is None:
        missing = "longitude" if lon is None else "latitude"
        raise ValueError(
            f"a horizontal gradient needs longitude and latitude coordinates, and "
            f"{da.name!r} (dimensions {list(da.dims)}) has no {missing}."
        )
    lon, lat = xr.DataArray(lon.variable), xr.DataArray(lat.variable)
    if lon.ndim == 1 and lat.ndim == 1 and lon.dims != lat.dims:
        dims = (lat.dims[0], lon.dims[0])  # rectilinear: each axis on its own dim
        lon, lat = xr.broadcast(lon, lat)
    elif lon.ndim == 2 and lat.ndim == 2 and set(lon.dims) == set(lat.dims):
        dims = lon.dims  # curvilinear: both 2-D on the same pair of dims
    else:
        raise ValueError(
            "a horizontal gradient needs a 2-D horizontal field (a lon/lat grid or a "
            f"curvilinear grid); {da.name!r} has longitude on {list(lon.dims)} and "
            f"latitude on {list(lat.dims)}, which is a transect or a point."
        )
    if not set(dims) <= set(da.dims):
        raise ValueError(
            f"the horizontal dimensions {list(dims)} are not both in {da.name!r} "
            f"(dimensions {list(da.dims)}); a horizontal gradient needs the 2-D field."
        )
    return dims, lon.transpose(*dims), lat.transpose(*dims)


def _label(variable: Any, da) -> str:
    """Return the short name a result is named after: the spec's, else the field's."""
    if isinstance(variable, str):
        return variable
    return str(da.name or "field")


def _per_km(units: str) -> str:
    """Return ``units`` per kilometer; a compound spelling is parenthesized first."""
    if not units:
        return "1/km"
    return f"({units})/km" if any(c in units for c in " .") else f"{units}/km"


def calculate_horizontal_gradient(da: xr.DataArray, *, variable=None) -> xr.DataArray:
    """Return the magnitude of the horizontal gradient of the 2-D field ``da``.

    ``da`` carries longitude and latitude as either two 1-D axes or two 2-D
    coordinates; extra dimensions (time, depth) are broadcast through. The result is in
    the field's units per km (per km, not per m: ``mmol/m^3/m`` reads as ``mmol/m^4``).
    ``variable`` only names the result. Lazy for dask input.
    """
    from ocean_skill.align import _haversine_km

    dims, lon, lat = _horizontal_axes(da)
    parts = []
    for d in dims:
        ahead, behind = {d: -1}, {d: 1}
        dist = _haversine_km(
            lon.shift(ahead), lat.shift(ahead), lon.shift(behind), lat.shift(behind)
        )
        parts.append(((da.shift(ahead) - da.shift(behind)) / dist.where(dist > 0)) ** 2)
    out = np.sqrt(parts[0] + parts[1]).where(da.notnull())  # land stays land

    label = _label(variable, da)
    units = str(da.attrs.get("units", "")).strip()
    describe = da.attrs.get("long_name") or da.name or "field"
    out.name = f"horizontal_gradient_of_{label}"
    out.attrs = {
        "units": _per_km(units),
        "long_name": f"horizontal gradient magnitude of {describe}",
        "horizontal_gradient_variable": label,
    }
    return out


def _inputs(spec: dict[str, Any]) -> list[list[str]]:
    """Return what the calculator needs, for :func:`ocean_skill.operators.spec_names`.

    That is the variable the spec names (itself possibly a combination or other spec).
    """
    from ocean_skill.operators import spec_names

    variable = spec.get("variable")
    return spec_names(variable) if variable is not None else []


def _register() -> None:
    from ocean_skill.operators import register_calculator

    register_calculator("horizontal_gradient", inputs=_inputs, layer=True)(
        calculate_horizontal_gradient
    )


_register()
