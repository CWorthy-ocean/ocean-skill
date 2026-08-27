"""Shared geometry for vertical-section plots.

One place both renderers read, so a section cannot look different statically than
interactively — the same reason :mod:`ocean_skill.plot.series` exists for the line
family. :func:`prepare_section` decides every axis convention a section needs once:
which coordinate is "depth" and which sign it reads positive, how the along-path
axis is labelled, and what a title calls the path itself. A renderer's own drawing
function calls this first and then only draws — it makes no convention decisions
of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

__all__ = ["SectionGeometry", "prepare_section"]


@dataclass
class SectionGeometry:
    """What a section's axes mean, decided once and read by both renderers.

    ``x_name``/``y_name`` name the coordinates :func:`prepare_section` attaches to
    its returned field -- ``"distance"``/``"depth"`` -- both broadcast to the
    field's own 2-D shape regardless of whether the vertical axis is native
    s-levels (where depth genuinely varies along the path, so it is honestly 2-D)
    or a fixed-depth list (where it is 1-D values repeated across the other axis):
    one shape of coordinate for a renderer to read either way, no special case for
    which kind of vertical axis it is drawing.
    """

    x_name: str
    y_name: str
    x_label: str
    y_label: str
    native_s: bool
    path_note: str


def _path_note(da, lon_name: str | None, lat_name: str | None) -> str:
    """``"29.0°N, 94.5°W → 27.5°N, 90.0°W"``, or ``"along 94.5°W"`` if lon is fixed.

    Read off the along-path lon/lat coordinates' own endpoints -- the same
    positive-down-style degree formatting :func:`ocean_skill.plot.series._place_of`
    uses for a station, extended to a pair of points rather than one. Empty when
    the field carries no lon/lat coordinates to read (should not happen for a
    real section, but a title with nothing to say is better than one that raises).
    """
    if lon_name is None or lat_name is None:
        return ""
    lon = np.asarray(da[lon_name], dtype="float64")
    lat = np.asarray(da[lat_name], dtype="float64")
    if lon.size == 0:
        return ""

    def _fmt(lon_v: float, lat_v: float) -> str:
        return (
            f"{abs(lat_v):.1f}°{'N' if lat_v >= 0 else 'S'}, "
            f"{abs(lon_v):.1f}°{'E' if lon_v >= 0 else 'W'}"
        )

    lon_fixed = np.ptp(lon) < 1e-6
    lat_fixed = np.ptp(lat) < 1e-6
    if lon_fixed and not lat_fixed:
        return f"along {abs(lon[0]):.1f}°{'E' if lon[0] >= 0 else 'W'}"
    if lat_fixed and not lon_fixed:
        return f"along {abs(lat[0]):.1f}°{'N' if lat[0] >= 0 else 'S'}"
    return f"{_fmt(lon[0], lat[0])} → {_fmt(lon[-1], lat[-1])}"


def prepare_section(da: xr.DataArray) -> tuple[xr.DataArray, SectionGeometry]:
    """Return ``(field, geometry)``: ``da`` with ``depth``/``distance`` coordinates.

    ``da`` must be exactly two-dimensional: :data:`ocean_skill.align.ALONG_DIM` and
    one vertical axis — either ``z`` (fixed depths, from
    :func:`ocean_skill.roms.to_depth`) or the model's native ``s_rho``/``s_w``,
    carrying a 2-D ``z_rho``/``z_w`` coordinate for its true depth. Both read as
    negative-down (the model's own convention); the returned ``depth`` coordinate
    is flipped to positive-down, which is what a reader — and every other depth
    label in this package (see ``facet_labels``' own ``abs()``) — already expects.
    The two renderers then each invert their y-axis once, so 0 m draws at the top
    and the seafloor at the bottom.

    The returned field carries two new coordinates, ``depth`` and ``distance``,
    both broadcast to its own two dimensions -- the along-path coordinate is
    renamed from :data:`~ocean_skill.align.ALONG_DIM` (rather than reused under
    that name) because a *dimension* coordinate must be one-dimensional, and this
    one is not once the vertical axis is native s-levels. A renderer draws
    ``x=geometry.x_name, y=geometry.y_name`` against the returned field and never
    needs to know which kind of vertical axis it got.
    """
    from ocean_skill.align import ALONG_DIM, _lat_name, _lon_name

    if ALONG_DIM not in da.dims:
        raise ValueError(
            f"prepare_section expects a field with an {ALONG_DIM!r} dimension "
            f"(see ocean_skill.align.path_of) -- got dims {sorted(da.dims)}."
        )
    extra = [d for d in da.dims if d != ALONG_DIM]
    if len(extra) != 1:
        raise ValueError(
            "prepare_section expects exactly one vertical axis beside "
            f"{ALONG_DIM!r} -- got {sorted(da.dims)}."
        )
    vertical = extra[0]

    native_s = vertical != "z" and "z_rho" in da.coords
    depth_source = da["z_rho"] if native_s else da[vertical]
    depth = (-depth_source).rename("depth")
    depth.attrs["units"] = "m"

    distance = da[ALONG_DIM].rename("distance")
    distance.attrs["units"] = da[ALONG_DIM].attrs.get("units", "km")

    depth2d, distance2d, values2d = xr.broadcast(depth, distance, da)
    order = tuple(values2d.dims)
    depth2d = depth2d.transpose(*order)
    distance2d = distance2d.transpose(*order)
    result = values2d.assign_coords(depth=depth2d, distance=distance2d)

    lon_name, lat_name = _lon_name(da), _lat_name(da)
    geometry = SectionGeometry(
        x_name="distance",
        y_name="depth",
        x_label="distance along transect (km)",
        y_label="depth (m)",
        native_s=native_s,
        path_note=_path_note(da, lon_name, lat_name),
    )
    return result, geometry
