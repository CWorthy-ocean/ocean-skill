"""Vertical slices (transects/sections) through model output.

A :func:`~ocean_skill.field.field` normally reduces to a map or, once both horizontal
axes are pinned to one place, a line through time (:attr:`~ocean_skill.field.Field.is_series`).
This module adds a third shape: a cut through *space* rather than time, read off
``select={"transect": ...}`` and left standing as one new dimension
(:data:`ocean_skill.align.ALONG_DIM`) alongside whatever vertical axis the request
leaves — a (depth × distance) section, the model-output counterpart of a ship's CTD
transect.

Two pathways. **Grid-aligned** (``{"<dim>": <index>}``) is a plain ``isel`` along a
named grid dimension (e.g. ``xi_rho``) — free, no interpolation, exact
(:func:`grid_slice`). **Arbitrary-path** is everything else — a list of lon/lat
``waypoints``, a fixed ``lon``/``lat`` line, or a resolved ``points`` list sampled
exactly as given — read off the model's curvilinear grid by nearest-neighbour
(default) or bilinear interpolation (:func:`sample_along`), after densifying
waypoints/lines to roughly the model's own resolution
(:func:`densify_waypoints`). Both pathways produce the same output shape (see
:func:`_attach_along_coord`), so everything downstream — :func:`
ocean_skill.align.path_of`, the ``section`` plot family — reads either one
identically.

Applied *before* variable resolution and the vertical ladder in
:func:`ocean_skill.comparison._prepare`, on the whole source ``Dataset`` rather than
one resolved variable: a ROMS vertical transform (:func:`ocean_skill.roms.to_depth`)
needs ``h``/``mask_rho``/``Cs_r`` sampled exactly the way the tracer field was, and
slicing/sampling the Dataset once, up front, keeps every one of those consistent for
free — the alternative (sampling the variable alone and re-attaching the grid
separately) would have to redo that consistency by hand for every vertical
operation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "as_transect",
    "apply_transect",
    "densify_waypoints",
    "grid_slice",
    "sample_along",
]

#: Keys :func:`as_transect` reads as an arbitrary-path request rather than a raw grid
#: dimension name. ROMS ships no coordinate variables for ``eta_rho``/``xi_rho``, so
#: those names can never collide with these -- a transect spec is unambiguous by key
#: alone, with no need to inspect the source to tell "a dimension" from "a longitude".
_ARBITRARY_PATH_KEYS = frozenset({"waypoints", "lon", "lat", "points"})
_OPTION_KEYS = frozenset({"spacing_km", "method"})
_METHODS = frozenset({"nearest", "bilinear"})


def as_transect(spec: Any) -> dict[str, Any]:
    """Validate and normalize a ``select={"transect": ...}`` value.

    Returns an internal, normalized form. Grid-aligned: ``{"kind": "grid", "dim":
    <name>, "index": <int>}``. Arbitrary-path (see :func:`_as_path_transect`):
    ``{"kind": "waypoints"|"points"|"lon_line"|"lat_line", ...}``, each carrying
    plain Python floats/lists/strs -- normalized here for deterministic internal
    equality and error text, not for the cache key (that hashes the caller's raw
    ``select`` in :func:`ocean_skill.cache.key_for_prepared`, before this ever
    runs; a tuple and a list of the same waypoints already serialize identically
    through ``json.dumps``, and a numpy-scalar spelling merely earns its own cache
    entry rather than colliding with or corrupting another one).
    """
    if not isinstance(spec, dict) or not spec:
        raise ValueError(
            f"select={{'transect': {spec!r}}} is not a valid transect request -- "
            "name a grid dimension and its index (select={'transect': {'xi_rho': "
            "30}}), a list of lon/lat waypoints, or a fixed lon/lat line."
        )
    if _ARBITRARY_PATH_KEYS & set(spec):
        return _as_path_transect(spec)
    if _OPTION_KEYS & set(spec):
        raise ValueError(
            f"select={{'transect': {spec!r}}}: 'spacing_km'/'method' only apply "
            "to an arbitrary-path transect (waypoints, points, or a lon/lat "
            "line) -- a grid-aligned slice is exact, with nothing to space out "
            "or interpolate."
        )
    keys = set(spec)
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
            "latitude) belongs to the arbitrary-path pathway: select={'transect': "
            "{'lon': ...}} or {'waypoints': [...]}."
        )
    return {"kind": "grid", "dim": str(dim), "index": int(index)}


def _normalize_options(spec: dict[str, Any]) -> tuple[float | None, str]:
    """Return ``(spacing_km, method)`` from a transect spec's option keys.

    ``spacing_km=None`` means "use the source's own cell size" (resolved at apply
    time, once the source is known -- see :func:`apply_transect`). ``method``
    defaults to ``"nearest"``, matching :func:`ocean_skill.align.sample_at`'s own
    default; ``"linear"`` is accepted as a synonym for ``"bilinear"`` (they are one
    branch downstream). A conservative method is refused with the same reasoning
    :func:`~ocean_skill.align.sample_at` refuses one for a single point: a path has
    no area to conservatively regrid onto either.
    """
    spacing_km = spec.get("spacing_km")
    if spacing_km is not None:
        if isinstance(spacing_km, bool) or not isinstance(spacing_km, int | float):
            raise ValueError(
                f"select={{'transect': ...}}: spacing_km must be a positive "
                f"number of kilometres, got {spacing_km!r}."
            )
        spacing_km = float(spacing_km)
        if not (np.isfinite(spacing_km) and spacing_km > 0):
            raise ValueError(
                f"select={{'transect': ...}}: spacing_km must be a positive "
                f"number of kilometres, got {spacing_km!r}."
            )

    method = spec.get("method", "nearest")
    if not isinstance(method, str):
        raise ValueError(
            f"select={{'transect': ...}}: method must be 'nearest' or "
            f"'bilinear', got {method!r}."
        )
    normalized = method.strip().lower()
    if normalized == "linear":
        normalized = "bilinear"
    if normalized not in _METHODS:
        raise ValueError(
            f"select={{'transect': ...}}: method={method!r} -- a transect "
            "samples the grid at a set of points, which has no *area* to "
            "conservatively regrid onto (the same reason "
            "ocean_skill.align.sample_at refuses one for a single point); use "
            "'nearest' or 'bilinear'."
        )
    return spacing_km, normalized


def _normalize_pairs(value: Any, *, key: str) -> list[list[float]]:
    """Return ``value`` as a validated ``[[lon, lat], ...]`` list of ≥2 pairs.

    Lon-first, matching :func:`ocean_skill.operators.point_in_spec` and
    :func:`ocean_skill.align.bbox_of`'s order. A single pair is refused rather
    than treated as a degenerate one-point path -- that request already has a
    grammar (a top-level scalar ``select={'lon': ..., 'lat': ...}``), and serving
    it here too would just be a second, worse-supported way to ask for it.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise ValueError(
            f"select={{'transect': {{{key!r}: {value!r}}}}}: {key} must be a "
            "list of [lon, lat] pairs, e.g. [[-150.0, 45.0], [-148.0, 46.5]]."
        )
    pairs: list[list[float]] = []
    for i, pair in enumerate(value):
        if isinstance(pair, np.ndarray):
            pair = pair.tolist()
        if (
            not isinstance(pair, list | tuple)
            or len(pair) != 2
            or any(isinstance(v, bool) for v in pair)
        ):
            raise ValueError(
                f"select={{'transect': {{{key!r}: ...}}}}: entry {i} ({pair!r}) "
                "is not a [lon, lat] pair."
            )
        try:
            lon, lat = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            raise ValueError(
                f"select={{'transect': {{{key!r}: ...}}}}: entry {i} ({pair!r}) "
                "is not a pair of numbers."
            ) from None
        if not (np.isfinite(lon) and np.isfinite(lat)):
            raise ValueError(
                f"select={{'transect': {{{key!r}: ...}}}}: entry {i} ({pair!r}) "
                "is not finite."
            )
        if abs(lat) > 90:
            raise ValueError(
                f"select={{'transect': {{{key!r}: ...}}}}: entry {i} ({pair!r}) "
                "has |lat| > 90 -- pairs are [lon, lat]; did you swap them?"
            )
        pairs.append([lon, lat])
    if len(pairs) < 2:
        raise ValueError(
            f"select={{'transect': {{{key!r}: {value!r}}}}}: a path needs at "
            "least 2 points -- for one place, use a top-level select={'lon': "
            "..., 'lat': ...} instead."
        )
    return pairs


def _as_path_transect(spec: dict[str, Any]) -> dict[str, Any]:
    """The arbitrary-path branch of :func:`as_transect`: waypoints/points/a line."""
    spacing_km, method = _normalize_options(spec)
    forms = [
        name
        for name, present in (
            ("waypoints", "waypoints" in spec),
            ("points", "points" in spec),
            ("a lon/lat line", "lon" in spec or "lat" in spec),
        )
        if present
    ]
    if len(forms) > 1:
        raise ValueError(
            f"select={{'transect': {spec!r}}} names more than one path form "
            f"({', '.join(forms)}) -- give exactly one: 'waypoints', 'points', "
            "or a fixed 'lon'/'lat' line."
        )

    if "waypoints" in spec:
        pairs = _normalize_pairs(spec["waypoints"], key="waypoints")
        return {
            "kind": "waypoints",
            "waypoints": pairs,
            "spacing_km": spacing_km,
            "method": method,
        }

    if "points" in spec:
        if spacing_km is not None:
            raise ValueError(
                f"select={{'transect': {spec!r}}}: 'points' are sampled exactly "
                "where given -- 'spacing_km' densifies waypoints/lines and does "
                "not apply here."
            )
        pairs = _normalize_pairs(spec["points"], key="points")
        return {"kind": "points", "points": pairs, "method": method}

    return _as_line_transect(spec, spacing_km, method)


def _as_line_transect(
    spec: dict[str, Any], spacing_km: float | None, method: str
) -> dict[str, Any]:
    """The fixed-``lon``/``lat``-line branch of :func:`_as_path_transect`."""
    from ocean_skill.operators import _point_scalar, _range_bounds

    has_lon, has_lat = "lon" in spec, "lat" in spec
    lon_scalar = _point_scalar(spec.get("lon")) if has_lon else None
    lat_scalar = _point_scalar(spec.get("lat")) if has_lat else None
    lon_range = _range_bounds(spec.get("lon")) if has_lon else None
    lat_range = _range_bounds(spec.get("lat")) if has_lat else None

    if lon_scalar is not None and lat_scalar is not None:
        raise ValueError(
            f"select={{'transect': {spec!r}}}: both lon and lat are single "
            "values -- that names one place, a point, not a line. For a point, "
            "use a top-level select={'lon': ..., 'lat': ...} (outside "
            "'transect')."
        )
    if lon_range is not None and lat_range is not None:
        raise ValueError(
            f"select={{'transect': {spec!r}}}: both lon and lat are ranges -- "
            "that names a box, not a line. For a box, use a top-level "
            "select={'lon': {'min': ..., 'max': ...}, 'lat': {'min': ..., "
            "'max': ...}} (outside 'transect')."
        )
    if lon_scalar is not None:
        bounds = list(lat_range) if lat_range is not None else [None, None]
        return {
            "kind": "lon_line",
            "lon": lon_scalar,
            "lat_bounds": bounds,
            "spacing_km": spacing_km,
            "method": method,
        }
    if lat_scalar is not None:
        bounds = list(lon_range) if lon_range is not None else [None, None]
        return {
            "kind": "lat_line",
            "lat": lat_scalar,
            "lon_bounds": bounds,
            "spacing_km": spacing_km,
            "method": method,
        }
    raise ValueError(
        f"select={{'transect': {spec!r}}}: a line fixes one axis to a single "
        "value and (optionally) bounds the other -- name a scalar 'lon' (a "
        "meridional line) or a scalar 'lat' (a zonal line)."
    )


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
    return _attach_along_coord(sliced, lon_name, lat_name, path_method="grid")


def _attach_along_coord(sliced, lon_name: str, lat_name: str, *, path_method: str):
    """Attach the cumulative along-path distance coordinate (km) and its units.

    The one place that decides what "along" means, shared by :func:`grid_slice`
    and :func:`sample_along` so a grid-aligned slice and an interpolated/sampled
    path produce byte-identical output shapes for
    :func:`ocean_skill.align.path_of` and :func:`ocean_skill.plot.section.
    prepare_section` to read. ``sliced`` must already have its along-path
    dimension named :data:`ocean_skill.align.ALONG_DIM`, with 1-D
    ``lon_name``/``lat_name`` coordinates riding on it.
    """
    from ocean_skill.align import ALONG_DIM, _haversine_km

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
        units="km", long_name="distance along transect", path_method=path_method
    )
    sliced[lon_name].attrs.setdefault("units", "degrees_east")
    sliced[lat_name].attrs.setdefault("units", "degrees_north")
    return sliced


def densify_waypoints(
    waypoints: list[list[float]], spacing_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lons, lats)``: ``waypoints`` linearly resampled to ~``spacing_km``.

    Consecutive coincident waypoints are dropped first (comparing longitude modulo
    360, so a stray 180/-180 duplicate still counts as one point); at least 2
    distinct points must remain. Longitudes are then :func:`numpy.unwrap` --
    **before** anything else — so a leg crossing the antimeridian (170 -> -170)
    densifies through 180, the short way, rather than the long way around the
    globe. :func:`ocean_skill.align._haversine_km`, used for each segment's
    length, is periodic and would give the same answer either way, but a *linear
    interpolation* in raw, not-unwrapped degrees would not be -- it would walk
    the long way round the world instead of the 20-degree hop the two waypoints
    actually describe.

    Returned in unwrapped degrees; the caller (:func:`apply_transect`, via
    :func:`sample_along`) wraps them into the target grid's own convention.
    """
    from ocean_skill.align import _haversine_km

    arr = np.asarray(waypoints, dtype="float64")
    lons_in, lats_in = arr[:, 0], arr[:, 1]

    keep = np.ones(lons_in.size, dtype=bool)
    if lons_in.size > 1:
        dlon = ((np.diff(lons_in) + 180.0) % 360.0) - 180.0
        dlat = np.diff(lats_in)
        coincident = (np.abs(dlon) < 1e-9) & (np.abs(dlat) < 1e-9)
        keep[1:] = ~coincident
    lons_in, lats_in = lons_in[keep], lats_in[keep]
    if lons_in.size < 2:
        raise ValueError(
            "densify_waypoints needs at least 2 distinct points -- for one "
            "place, use a top-level select={'lon': ..., 'lat': ...} instead."
        )

    lons_in = np.unwrap(lons_in, period=360.0)

    lon_parts = [lons_in[:1]]
    lat_parts = [lats_in[:1]]
    for i in range(lons_in.size - 1):
        lon0, lat0 = lons_in[i], lats_in[i]
        lon1, lat1 = lons_in[i + 1], lats_in[i + 1]
        km = float(_haversine_km(lon0, lat0, lon1, lat1))
        n = max(1, int(np.ceil(km / spacing_km)))
        t = np.arange(1, n + 1, dtype="float64") / n  # excludes 0 -- no doubled
        lon_parts.append(lon0 + t * (lon1 - lon0))  # vertex at each waypoint
        lat_parts.append(lat0 + t * (lat1 - lat0))
    return np.concatenate(lon_parts), np.concatenate(lat_parts)


def sample_along(
    obj,
    lons,
    lats,
    *,
    method: str = "nearest",
    convention: str = "-180-180",
    subject: str = "the source",
):
    """Sample ``obj`` along an arbitrary lon/lat path: the N-point sibling of
    :func:`ocean_skill.align.sample_at`.

    ``method="nearest"`` (default) takes each point's containing cell.
    ``method="bilinear"`` interpolates instead, via xesmf's locstream mode
    (:func:`ocean_skill.align._interp_locstream` on a curvilinear grid,
    ``.interp`` on a rectilinear one).

    Both methods share one determination of *which* requested points are close
    enough to the domain to sample at all: a curvilinear grid is first
    pre-cropped to the path's own bounding box (a pure cost optimization -- the
    per-point nearest search then scans a window sized to the path, not the
    whole model domain), then every point's nearest-cell offset is measured; a
    point more than one cell from its nearest neighbour is outside the domain
    and dropped, with one warning naming how many. This keeps a fixed-lon line's
    trim to the domain's true footprint (and a waypoint path's out-of-domain
    tail) identical between the two methods, rather than nearest silently
    covering less of the path than bilinear (or vice versa) for no reason a
    caller could see.

    Consecutive requested points that snap to the same cell are then collapsed,
    so the along-path coordinate is always strictly increasing -- see
    :func:`_attach_along_coord`, which both methods finish through, giving the
    same output shape :func:`grid_slice` does.
    """
    import warnings

    from ocean_skill import _stacklevel
    from ocean_skill.align import (
        ALONG_DIM,
        DEFAULT_PAD,
        _cell_km,
        _haversine_km,
        _lat_name,
        _lon_name,
        _nearest_indices,
        _wrap_lon,
        harmonize_longitude,
        subset_to_bbox,
    )

    lon_name, lat_name = _lon_name(obj), _lat_name(obj)
    if lon_name is None or lat_name is None:
        raise ValueError(
            f"{subject} has no longitude/latitude coordinate, so a transect "
            "cannot be sampled along it."
        )

    obj = harmonize_longitude(obj, convention)
    req_lons = np.asarray(
        [_wrap_lon(float(v), convention) for v in lons], dtype="float64"
    )
    req_lats = np.asarray(lats, dtype="float64")

    # Pre-crop to the path's own bbox, padded generously -- correctness never
    # depends on this (the offset test below still runs against whatever survives),
    # only speed, so a folded-looking span (>180 degrees wrapped -- part of the
    # path is then outside the domain regardless) skips it rather than risk
    # cropping to the wrong side of a seam.
    cropped = obj
    if req_lons.size and float(np.ptp(req_lons)) <= 180.0:
        cell_km_full = _cell_km(obj, lon_name, lat_name)
        pad = max(DEFAULT_PAD, 3.0 * cell_km_full / 111.0)
        bbox = (
            float(req_lons.min()),
            float(req_lats.min()),
            float(req_lons.max()),
            float(req_lats.max()),
        )
        try:
            cropped = subset_to_bbox(obj, bbox, pad)
        except ValueError:
            pass  # the shared offset test below raises its own clearer error

    lon_vals = np.asarray(cropped[lon_name])
    lat_vals = np.asarray(cropped[lat_name])
    curvilinear = lon_vals.ndim == 2
    cell_km = _cell_km(cropped, lon_name, lat_name)

    if curvilinear:
        snapped = [
            _nearest_indices(lon_vals, lat_vals, lo, la)
            for lo, la in zip(req_lons, req_lats, strict=True)
        ]
        iys = np.array([s[0] for s in snapped])
        ixs = np.array([s[1] for s in snapped])
        snapped_lon, snapped_lat = lon_vals[iys, ixs], lat_vals[iys, ixs]
    else:
        ixs = np.array(
            [np.abs(((lon_vals - lo + 180.0) % 360.0) - 180.0).argmin() for lo in req_lons]
        )
        iys = np.array([np.abs(lat_vals - la).argmin() for la in req_lats])
        snapped_lon, snapped_lat = lon_vals[ixs], lat_vals[iys]

    offsets = _haversine_km(snapped_lon, snapped_lat, req_lons, req_lats)
    inside = offsets <= max(cell_km, 1e-9)
    n_dropped = int((~inside).sum())
    if int(inside.sum()) < 2:
        raise ValueError(
            f"{subject}: the transect path does not cross this source's domain."
        )
    if n_dropped:
        warnings.warn(
            f"{subject}: {n_dropped} of {inside.size} transect points fall "
            f"more than one cell (~{cell_km:.1f} km) outside the domain and "
            "were dropped; the section is trimmed to the domain.",
            stacklevel=_stacklevel.find(),
        )
    iys, ixs = iys[inside], ixs[inside]
    req_lons, req_lats = req_lons[inside], req_lats[inside]

    # Consecutive requests landing on the same cell would otherwise leave the
    # along coordinate flat (or, after a nearest isel, duplicated) rather than
    # strictly increasing.
    dup = np.zeros(iys.size, dtype=bool)
    if iys.size > 1:
        dup[1:] = (np.diff(iys) == 0) & (np.diff(ixs) == 0)
    keep = ~dup
    iys, ixs = iys[keep], ixs[keep]
    req_lons, req_lats = req_lons[keep], req_lats[keep]
    if iys.size < 2:
        raise ValueError(
            f"{subject}: the transect path collapses to a single grid cell -- "
            "widen it, or use a top-level select={'lon': ..., 'lat': ...} for "
            "one place."
        )

    if method == "nearest":
        if curvilinear:
            dim0, dim1 = cropped[lon_name].dims
            sampled = cropped.isel(
                {
                    dim0: xr.DataArray(iys, dims=ALONG_DIM),
                    dim1: xr.DataArray(ixs, dims=ALONG_DIM),
                }
            )
        else:
            lon_dim, lat_dim = cropped[lon_name].dims[0], cropped[lat_name].dims[0]
            sampled = cropped.isel(
                {
                    lon_dim: xr.DataArray(ixs, dims=ALONG_DIM),
                    lat_dim: xr.DataArray(iys, dims=ALONG_DIM),
                }
            )
    elif curvilinear:
        sampled = _bilinear_dataset(cropped, lon_name, lat_name, req_lons, req_lats)
        lon_name, lat_name = "lon", "lat"  # _as_xesmf's canonical output names
    else:
        lon_dim, lat_dim = cropped[lon_name].dims[0], cropped[lat_name].dims[0]
        sampled = cropped.interp(
            {
                lon_dim: xr.DataArray(req_lons, dims=ALONG_DIM),
                lat_dim: xr.DataArray(req_lats, dims=ALONG_DIM),
            }
        )

    if "mask_rho" in sampled.variables:
        land_frac = float((np.asarray(sampled["mask_rho"]) < 1).mean())
        if land_frac > 0:
            warnings.warn(
                f"{subject}: {land_frac:.0%} of the transect passes within one cell "
                "of land (mask_rho < 1 at at least one bilinear neighbor); those "
                "columns are renormalized over whichever neighbors are wet, and are "
                "NaN only where every neighbor is land.",
                stacklevel=_stacklevel.find(),
            )

    return _attach_along_coord(sampled, lon_name, lat_name, path_method=method)


def _bilinear_dataset(cropped, lon_name: str, lat_name: str, lons, lats):
    """Bilinearly sample every regriddable variable of ``cropped`` along a path.

    Classifies each variable before handing anything to xesmf, rather than
    trusting its own Dataset-wide dispatch on a grid as irregular as ROMS's: a
    variable whose dims are a *superset* of the grid's own two horizontal dims
    (and numeric) is regridded; one with *none* of them (``Cs_r``, ``sigma_r``,
    the ``spherical`` flag) rides through untouched; one with *some* but not all
    of them (a staggered ``eta_u``/``xi_u`` momentum variable) cannot be
    regridded against this rho-point target and is dropped, with a warning --
    interpolating it onto rho positions first is deferred, the same way
    :func:`ocean_skill.roms.to_depth`'s own guard defers it.

    ``h``/``mask_rho`` are promoted from coordinates to data variables across the
    regrid (so they are sampled along with everything else) and restored to
    coordinates in the result; ``z_rho``/``z_w`` are dropped outright rather than
    (wrongly) linearly regridded -- the existing ``_prepare``/``to_depth`` call
    sites already re-derive them from the sampled ``h`` whenever they are absent.
    """
    import warnings

    import xarray as xr

    from ocean_skill import _stacklevel
    from ocean_skill.align import ALONG_DIM, _interp_locstream

    spatial = set(cropped[lon_name].dims)
    promote = [v for v in ("h", "mask_rho") if v in cropped.coords]
    src = cropped.reset_coords(promote) if promote else cropped
    src = src.drop_vars(
        [v for v in ("z_rho", "z_w") if v in src.variables], errors="ignore"
    )

    regrid_vars, dropped = [], []
    for name, var in src.variables.items():
        if name in (lon_name, lat_name):
            continue
        dims = set(var.dims)
        if not (dims & spatial):
            continue  # no horizontal dim at all -- carried through untouched
        numeric = np.issubdtype(var.dtype, np.number) or var.dtype == bool
        if spatial <= dims and numeric:
            regrid_vars.append(name)
        else:
            dropped.append(name)

    to_regrid = src[[v for v in regrid_vars if v in src.data_vars]]
    regridded = _interp_locstream(to_regrid, lons, lats)
    passthrough = src.drop_vars(
        [*regrid_vars, *dropped, lon_name, lat_name], errors="ignore"
    )
    out = xr.merge([regridded, passthrough], combine_attrs="override")
    out = out.set_coords([v for v in promote if v in out.variables])
    if dropped:
        warnings.warn(
            f"a bilinear transect cannot interpolate {sorted(dropped)} "
            "(staggered onto a different grid point than the rho-point "
            "field being sliced) -- dropped from the sampled result.",
            stacklevel=_stacklevel.find(),
        )
    return out


def _line_endpoints(obj, parsed: dict[str, Any]) -> list[list[float]]:
    """Return the two endpoint waypoints for a fixed-``lon``/``lat``-line transect.

    Missing bounds are filled from ``obj``'s own bounding box
    (:func:`ocean_skill.align.bbox_of`) -- ``obj`` must already be harmonized to
    its own natural longitude convention (:func:`apply_transect` does this once,
    before dispatch), or a seam-straddling domain would report a whole-globe
    span here instead of its own true extent.
    """
    from ocean_skill.align import bbox_of

    lon_min, lat_min, lon_max, lat_max = bbox_of(obj)
    if parsed["kind"] == "lon_line":
        lon = parsed["lon"]
        lo, hi = parsed["lat_bounds"]
        lo = lat_min if lo is None else float(lo)
        hi = lat_max if hi is None else float(hi)
        if lo > hi:
            lo, hi = hi, lo
        return [[lon, lo], [lon, hi]]

    lat = parsed["lat"]
    lo, hi = parsed["lon_bounds"]
    lo = lon_min if lo is None else float(lo)
    hi = lon_max if hi is None else float(hi)
    if lo > hi:
        # a seam-straddling 0-360 band first (box_in_spec's own convention),
        # falling back to a plain swap for a genuine backwards typo that does
        # not resolve that way
        wrapped_lo, wrapped_hi = lo % 360.0, hi % 360.0
        lo, hi = (wrapped_lo, wrapped_hi) if wrapped_lo <= wrapped_hi else (hi, lo)
    return [[lo, lat], [hi, lat]]


def apply_transect(obj, spec: dict[str, Any], *, subject: str = "the source"):
    """Validate ``spec`` and apply it to ``obj``: the one entry point ``_prepare`` calls.

    Dispatches on :func:`as_transect`'s normalized ``kind``. A grid index is a
    pure ``isel`` (:func:`grid_slice`). Everything else is a path sampled with
    :func:`sample_along`: ``waypoints`` and a fixed ``lon``/``lat`` line are
    densified first, to roughly the source's own grid resolution unless
    ``spacing_km`` says otherwise (:func:`densify_waypoints`); ``points`` are
    sampled exactly as given, with no densification.
    """
    parsed = as_transect(spec)
    if parsed["kind"] == "grid":
        return grid_slice(obj, parsed["dim"], parsed["index"], subject=subject)

    from ocean_skill.align import (
        _cell_km,
        _lat_name,
        _lon_name,
        harmonize_longitude,
        natural_convention,
    )

    convention = natural_convention(obj)
    obj_h = harmonize_longitude(obj, convention)

    if parsed["kind"] == "points":
        lons, lats = zip(*parsed["points"], strict=True)
        lons = np.asarray(lons, dtype="float64")
        lats = np.asarray(lats, dtype="float64")
    else:
        waypoints = (
            parsed["waypoints"]
            if parsed["kind"] == "waypoints"
            else _line_endpoints(obj_h, parsed)
        )
        spacing_km = parsed["spacing_km"]
        if spacing_km is None:
            lon_name, lat_name = _lon_name(obj_h), _lat_name(obj_h)
            if lon_name is None or lat_name is None:
                raise ValueError(
                    f"{subject}: cannot build a transect -- no longitude/"
                    "latitude coordinate found to measure a default spacing "
                    "against."
                )
            spacing_km = _cell_km(obj_h, lon_name, lat_name) or 1.0
        lons, lats = densify_waypoints(waypoints, spacing_km)

    return sample_along(
        obj_h,
        lons,
        lats,
        method=parsed["method"],
        convention=convention,
        subject=subject,
    )
