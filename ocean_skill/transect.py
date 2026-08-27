"""Vertical slices (transects/sections) through model output.

A :func:`~ocean_skill.field.field` normally reduces to a map or, once both horizontal
axes are pinned to one place, a line through time (:attr:`~ocean_skill.field.Field.is_series`).
This module adds a third shape: a cut through *space* rather than time, read off
``select={"transect": ...}`` and left standing as one new dimension
(:data:`ocean_skill.align.ALONG_DIM`) alongside whatever vertical axis the request
leaves — a (depth × distance) section, the model-output counterpart of a ship's CTD
transect.

This build supports exactly one pathway: **grid-aligned**, ``{"<dim>": <index>}``,
a plain ``isel`` along a named grid dimension (e.g. ``xi_rho``) — free, no
interpolation, exact. An arbitrary path (a fixed longitude that does not land on a
grid column, or a list of lon/lat waypoints) is a different operation entirely —
interpolating a curvilinear grid at points of the caller's choosing — and is not yet
built; :func:`as_transect` names it and says so, rather than silently doing something
else with a key that looks similar.

Applied *before* variable resolution and the vertical ladder in
:func:`ocean_skill.comparison._prepare`, on the whole source ``Dataset`` rather than
one resolved variable: a ROMS vertical transform (:func:`ocean_skill.roms.to_depth`)
needs ``h``/``mask_rho``/``Cs_r`` sliced exactly the way the tracer field was, and
slicing the Dataset once, up front, keeps every one of those consistent for free —
the alternative (slicing the variable alone and re-attaching the grid separately)
would have to redo that consistency by hand for every vertical operation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

__all__ = ["as_transect", "apply_transect", "grid_slice"]

#: Keys :func:`as_transect` reserves for the arbitrary-path pathway (not yet built —
#: see the module docstring) and for the resolved internal form written when the two
#: lanes of a comparison are sampled at one shared set of points. Any *other* single
#: key in a transect spec is read as a raw grid dimension name instead: ROMS ships no
#: coordinate variables for ``eta_rho``/``xi_rho``, so those names can never collide
#: with these.
_ARBITRARY_PATH_KEYS = frozenset({"waypoints", "lon", "lat", "points"})
_OPTION_KEYS = frozenset({"spacing_km", "method"})


def as_transect(spec: Any) -> dict[str, Any]:
    """Validate and normalize a ``select={"transect": ...}`` value.

    Returns an internal, normalized form: ``{"kind": "grid", "dim": <name>, "index":
    <int>}`` for the one pathway this build supports. Raises a clear error for
    anything else, including the arbitrary-path spellings (``waypoints``, a fixed
    ``lon``/``lat``) that a later build will add — named here, rather than left to
    fall through to "unknown grid dimension", so the grammar reads as stable and the
    caller learns what is missing rather than what is misspelled.
    """
    if not isinstance(spec, dict) or not spec:
        raise ValueError(
            f"select={{'transect': {spec!r}}} is not a valid transect request -- "
            "name a grid dimension and its index, e.g. "
            "select={'transect': {'xi_rho': 30}}."
        )
    if _ARBITRARY_PATH_KEYS & set(spec):
        raise NotImplementedError(
            f"select={{'transect': {spec!r}}} asks for an arbitrary-path transect "
            "(waypoints, or a fixed longitude/latitude line) -- this build only "
            "slices exactly along a grid dimension, e.g. "
            "select={'transect': {'xi_rho': 30}}. Arbitrary paths are coming in a "
            "follow-up."
        )
    keys = set(spec) - _OPTION_KEYS
    if len(keys) != 1:
        raise ValueError(
            f"select={{'transect': {spec!r}}}: a grid-aligned transect names "
            "exactly one grid dimension and its index, e.g. {'xi_rho': 30}."
        )
    dim = next(iter(keys))
    index = spec[dim]
    if isinstance(index, bool) or not isinstance(index, int | np.integer):
        raise ValueError(
            f"select={{'transect': {{{dim!r}: {index!r}}}}}: the grid index must "
            "be an int (e.g. 30) naming a position along the dimension -- this is "
            "the grid-aligned pathway. A coordinate value (a longitude, a "
            "latitude) belongs to the arbitrary-path pathway, not yet built."
        )
    return {"kind": "grid", "dim": str(dim), "index": int(index)}


def grid_slice(obj, dim: str, index: int, *, subject: str = "the source"):
    """Slice ``obj`` to one index along ``dim``: the grid-aligned transect pathway.

    A plain ``obj.isel({dim: index})`` — no interpolation, no grid attached, exact.
    ``dim`` collapses entirely; whichever horizontal dimension the longitude/latitude
    coordinates still vary along afterward is renamed to
    :data:`ocean_skill.align.ALONG_DIM` and given a cumulative great-circle-distance
    coordinate (km, from :func:`ocean_skill.align._haversine_km` over the now-1-D
    lon/lat), so both the grid-aligned pathway and a future interpolated one produce
    the same shape for :func:`ocean_skill.align.path_of` and the renderer to read.

    Runs on the whole ``Dataset`` (see the module docstring for why), and leaves the
    result lazy -- the caller (:func:`ocean_skill.comparison._prepare`) computes it
    together with the rest of the reduction, not here.
    """
    from ocean_skill.align import ALONG_DIM, _haversine_km, _lat_name, _lon_name

    if dim not in obj.dims:
        raise ValueError(
            f"{subject}: select={{'transect': {{{dim!r}: {index!r}}}}} names "
            f"{dim!r}, which is not one of this source's dimensions "
            f"({sorted(obj.dims)}). A grid-aligned transect names a real grid "
            "dimension -- for a ROMS run, typically 'xi_rho' or 'eta_rho'."
        )
    size = obj.sizes[dim]
    if not -size <= index < size:
        raise ValueError(
            f"{subject}: select={{'transect': {{{dim!r}: {index!r}}}}}: {dim!r} "
            f"only has {size} indices (0..{size - 1}, or negative to count from "
            "the end) -- this is a raw grid index, not a coordinate value."
        )
    sliced = obj.isel({dim: index})

    lon_name, lat_name = _lon_name(sliced), _lat_name(sliced)
    if lon_name is None or lat_name is None:
        raise ValueError(
            f"{subject}: cannot build a transect along {dim!r} -- no "
            "longitude/latitude coordinate survives the slice, so there is "
            "nothing to measure an along-path distance against."
        )
    # On a curvilinear grid, both lon and lat still vary along the one surviving
    # dim -- the ordinary case. On a rectilinear-flavored one (a grid-aligned line
    # of constant longitude, say), one of the two collapsed to a 0-d scalar in the
    # isel above; it is broadcast back out onto the surviving dim below so
    # lon(along)/lat(along) is always a full pair of 1-D coordinates, whether or
    # not the value in it actually varies -- one shape for every caller downstream
    # to read, rather than a scalar-or-array special case at each of them.
    surviving = {
        d
        for d in (*sliced[lon_name].dims, *sliced[lat_name].dims)
        if sliced.sizes.get(d, 1) > 1
    }
    if len(surviving) != 1:
        raise ValueError(
            f"{subject}: cannot build a transect along {dim!r} -- longitude and "
            f"latitude leave {sorted(surviving) or 'nothing'} varying after the "
            "slice, not exactly one axis, so there is no single along-path axis "
            "left to measure a distance against."
        )
    along_dim = next(iter(surviving))
    lon_bc, lat_bc = xr.broadcast(sliced[lon_name], sliced[lat_name])
    sliced = sliced.assign_coords({lon_name: lon_bc, lat_name: lat_bc})
    sliced = sliced.rename({along_dim: ALONG_DIM})

    lon_vals = np.asarray(sliced[lon_name], dtype="float64")
    lat_vals = np.asarray(sliced[lat_name], dtype="float64")
    distance = np.zeros(lon_vals.size)
    if lon_vals.size > 1:
        seg_km = _haversine_km(
            lon_vals[:-1], lat_vals[:-1], lon_vals[1:], lat_vals[1:]
        )
        distance[1:] = np.cumsum(seg_km)
    sliced = sliced.assign_coords({ALONG_DIM: (ALONG_DIM, distance)})
    sliced[ALONG_DIM].attrs.update(
        units="km", long_name="distance along transect", path_method="grid"
    )
    sliced[lon_name].attrs.setdefault("units", "degrees_east")
    sliced[lat_name].attrs.setdefault("units", "degrees_north")
    return sliced


def apply_transect(obj, spec: dict[str, Any], *, subject: str = "the source"):
    """Validate ``spec`` and apply it to ``obj``: the one entry point ``_prepare`` calls.

    Dispatches on :func:`as_transect`'s normalized ``kind`` -- today always
    ``"grid"``, so always :func:`grid_slice`, but kept as a dispatch (rather than
    calling ``grid_slice`` directly) so a later pathway slots in here without
    changing the caller.
    """
    parsed = as_transect(spec)
    if parsed["kind"] == "grid":
        return grid_slice(obj, parsed["dim"], parsed["index"], subject=subject)
    raise NotImplementedError(  # pragma: no cover -- as_transect already gates this
        f"transect kind {parsed['kind']!r} is not implemented"
    )
