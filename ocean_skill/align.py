"""Alignment: bring the two lanes onto one grid/coordinate frame.

Roles are set by name (not argument order): ``test`` vs ``reference``. In both time and
space the pair lands on the **coarser** of the two — the reference's when it is
comparable or coarser, the test's when it is materially finer (a 15-minute mooring
reference against weekly output, a fine satellite product against a coarse model) —
so the finer lane is always the one averaged down, in time (see
:func:`resolve_match_method`) as in space (see :func:`_regrid_target`). The shared
axis keeps the reference's *name* in both directions; which lane actually moved is
recorded (``match_target``/``regrid_target``) rather than assumed. Longitude
conventions are harmonized first — a 0-360 model vs a ±180 observational grid
otherwise produces silently empty overlap. The reference is subset to the test's
bounding box so we regrid over the overlap rather than a whole globe. The result is
**always xarray**, so one metrics engine serves both gridded and point comparisons.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import OrderedDict
from typing import Any, Literal

import numpy as np
import xarray as xr

from ocean_skill import _stacklevel
from ocean_skill.cf import find_coord

__all__ = [
    "ALONG_DIM",
    "align",
    "axis_edges",
    "clear_regridder_memo",
    "grid_of",
    "harmonize_longitude",
    "is_composite",
    "match_axis",
    "natural_convention",
    "path_of",
    "perimeter_of",
    "point_of",
    "resolve_match_method",
    "sample_at",
    "subset_to_box",
    "subset_to_bbox",
]

#: Sampling methods :func:`sample_at` understands, and what each means at a point.
#: ``nearest`` takes the containing cell's own value; the interpolating spellings weight
#: the surrounding cells. Anything else (a conservative regrid) has no meaning against a
#: zero-area target — see :func:`sample_at`.
NEAREST = "nearest"
_INTERPOLATING = ("bilinear", "linear")

#: Tolerance (degrees) for :func:`natural_convention`'s span comparison and its
#: +180 seam handling. Comfortably above float64 rounding noise on longitude
#: values up to ~720° (~1e-13), comfortably below any real grid spacing.
_CONVENTION_TOL = 1e-6


def _lon_name(obj) -> str | None:
    """Name of the longitude coordinate, preferring canonical then ROMS names."""
    for nm in ("lon", "longitude", "lon_rho"):
        if nm in obj.coords or nm in getattr(obj, "variables", {}):
            return nm
    return None


def _time_name(obj) -> str | None:
    """Name of the time coordinate, via cf-xarray with a plain-name fallback."""
    coord = find_coord(obj, "time")
    if coord is not None:
        return str(coord.name)
    return next((nm for nm in ("time", "ocean_time") if nm in obj.coords), None)


def _lat_name(obj) -> str | None:
    """Name of the latitude coordinate, preferring canonical then ROMS names."""
    for nm in ("lat", "latitude", "lat_rho"):
        if nm in obj.coords or nm in getattr(obj, "variables", {}):
            return nm
    return None


def harmonize_longitude(obj, convention: Literal["0-360", "-180-180"] = "-180-180"):
    """Return ``obj`` with longitudes in ``convention``.

    Mismatched conventions (a 0-360 model against a ±180 climatology) are a silent
    correctness hazard: regridding "succeeds" and yields all-NaN or garbage. A
    longitude *dimension* coordinate is re-sorted so it stays monotonic.
    """
    lon = _lon_name(obj)
    if lon is None:
        return obj
    vals = obj[lon]
    if convention == "-180-180":
        new = ((vals + 180) % 360) - 180
    else:
        new = vals % 360
    out = obj.assign_coords({lon: new})
    if lon in out.dims:  # 1-D dimension coordinate must stay sorted
        out = out.sortby(lon)
    return out


def natural_convention(obj) -> Literal["0-360", "-180-180"]:
    """Return the longitude convention in which ``obj`` is one contiguous span.

    A domain straddling the antimeridian (a Pacific model running 77°E to 316°E)
    is contiguous in 0-360 and split in ±180 — forced into ±180 its bounding box
    reads as the whole globe, so the reference it is compared against never gets
    cropped, and cell corners derived across the fold average out to ~0°, painting
    conservative regrids across half the planet. Measured rather than assumed:
    whichever convention gives the smaller longitude span is the one the domain is
    contiguous in. Ties (a truly global field, or two spans equal within float
    tolerance) keep ±180, the maps' usual frame. A value that lands on +180 itself
    (a domain that reaches, but does not cross, the dateline) is treated as the
    seam's own coordinate rather than wrapped to -180, which would otherwise
    inflate the ±180 span to the whole globe for a domain that never left it.
    """
    lon = _lon_name(obj)
    if lon is None:
        return "-180-180"
    vals = np.asarray(obj[lon], dtype="float64").ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "-180-180"
    wrapped_180 = ((vals + 180.0) % 360.0) - 180.0
    wrapped_180 = np.where(np.abs(vals - 180.0) <= _CONVENTION_TOL, 180.0, wrapped_180)
    span_180 = np.ptp(wrapped_180)
    span_360 = np.ptp(vals % 360.0)
    return "0-360" if span_360 < span_180 - _CONVENTION_TOL else "-180-180"


#: Degrees of margin kept around the test's own extent when the reference is cropped to
#: it. Named rather than repeated because two places crop with it and they must agree: a
#: lane pre-cropped with a *smaller* pad than :func:`align` uses would silently hand the
#: comparison a narrower reference than the same call without the pre-crop.
DEFAULT_PAD = 1.0


def _bbox_lon_in_convention(values, lon_min: float, lon_max: float):
    """Re-express a bbox's longitudes in whichever convention ``values`` uses.

    A bbox is a pair of bare numbers, so it carries no convention of its own — and
    the two sources in a comparison need not agree. Cropping a ±180 reference to a
    0-360 test's box asks for longitudes 190 to 250 on an axis that stops at 180 and
    gets an empty slice, which reads as "these do not overlap" when in fact one of
    them is global. Both MUR (±180) and a North Pacific model (0-360) are ordinary
    choices, so the mismatch is the common case rather than the exotic one.

    Returns the pair unchanged when both already agree, or when the box lies in
    0-180 where the two conventions coincide.
    """
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return lon_min, lon_max
    obj_is_0360 = float(np.nanmax(finite)) > 180.0
    box_is_0360 = max(lon_min, lon_max) > 180.0
    if obj_is_0360 == box_is_0360:
        return lon_min, lon_max
    if obj_is_0360:
        return lon_min % 360, lon_max % 360
    return ((lon_min + 180) % 360) - 180, ((lon_max + 180) % 360) - 180


def _curvilinear_window(obj, lon_name: str, lat_name: str, bbox, pad: float):
    """Return ``(obj windowed to bbox's index bounds, its inside-mask)`` for a 2-D grid.

    The mask is built elementwise over the 2-D lon/lat, which sidesteps the
    mid-array longitude fold a curvilinear grid can have crossing its own seam:
    unlike a corners/perimeter derivation (:func:`perimeter_of`, :func:`grid_of`),
    which differences *neighbouring* cells and so needs :func:`numpy.unwrap` first,
    a per-cell predicate has no neighbour to compare against and needs none.

    Returns the object windowed to the smallest index rectangle containing every
    inside cell, plus the mask over that same window as an
    :class:`xarray.DataArray` on the grid's own dimensions (so a caller's
    ``.where(mask)`` broadcasts correctly over any extra axis — depth, time — the
    object also carries). When no cell is inside, ``obj`` is returned unwindowed
    and the mask is all ``False`` over its full extent; the caller decides what an
    empty result means.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lon2d, lat2d = np.asarray(obj[lon_name]), np.asarray(obj[lat_name])
    box_lo, box_hi = _bbox_lon_in_convention(lon2d, lon_min, lon_max)
    if box_lo > box_hi:
        # Box is contiguous only in its own convention: wrap the grid's lon
        # *values* (not the coordinate itself — data stays in its native
        # convention) into that convention, for the comparison below only.
        conv = "0-360" if max(lon_min, lon_max) > 180.0 else "-180-180"
        lon_vals = _wrap_lon(lon2d, conv)
        box_lo, box_hi = lon_min, lon_max
    else:
        lon_vals = lon2d
    inside = (
        (lon_vals >= box_lo - pad)
        & (lon_vals <= box_hi + pad)
        & (lat2d >= lat_min - pad)
        & (lat2d <= lat_max + pad)
    )
    dims = obj[lon_name].dims
    if inside.any():
        rows = np.where(inside.any(axis=1))[0]
        cols = np.where(inside.any(axis=0))[0]
        i0, i1 = int(rows.min()), int(rows.max())
        j0, j1 = int(cols.min()), int(cols.max())
        window = {dims[0]: slice(i0, i1 + 1), dims[1]: slice(j0, j1 + 1)}
        obj = obj.isel(window)
        inside = inside[i0 : i1 + 1, j0 : j1 + 1]
    mask = xr.DataArray(inside, dims=dims)
    return obj, mask


def subset_to_box(obj, bbox, *, subject: str = "the source"):
    """Subset ``obj`` to ``bbox`` (lon_min, lat_min, lon_max, lat_max), exactly.

    The ``select=`` counterpart of :func:`subset_to_bbox`: no padding, and an empty
    result is refused outright rather than silently handed back unchanged, since a
    caller who named a box wants that box, not a hint that something went missing.

    Rectilinear lon/lat (each 1-D, each its own dimension) narrows exactly like an
    ordinary :func:`ocean_skill.operators.oriented_slice` pair, with the same seam
    handling :func:`subset_to_bbox` has: a box straddling the object's own
    longitude convention (a 170-to-190 request against a ±180 grid) re-expresses
    the *object* in the box's convention rather than silently returning nothing.

    Curvilinear lon/lat (each 2-D, e.g. ROMS's ``lon_rho``/``lat_rho``) has no
    dimension to slice along, so the crop is a boolean predicate over the whole
    2-D grid instead (:func:`_curvilinear_window`): the object is windowed to the
    smallest index rectangle containing every cell inside the box, and cells
    inside that rectangle but outside the box itself (a rotated grid's corners)
    are masked to NaN with ``.where`` — a map drawn from the result shows exactly
    the requested box, not the enclosing parallelogram of grid indices.

    The result carries ``attrs["region"] = [lon_min, lat_min, lon_max, lat_max]``,
    the box as asked (not padded, not re-expressed), for renderers and labels to
    read back.
    """
    from ocean_skill.operators import oriented_slice

    lon, lat = _lon_name(obj), _lat_name(obj)
    if lon is None or lat is None:
        return obj
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_values = np.asarray(obj[lon])
    rectilinear = lon_values.ndim == 1 and lon in obj.dims and lat in obj.dims

    if rectilinear:
        box_lo, box_hi = _bbox_lon_in_convention(obj[lon], lon_min, lon_max)
        target = obj
        if box_lo > box_hi:
            target = harmonize_longitude(
                target, "0-360" if max(lon_min, lon_max) > 180.0 else "-180-180"
            )
            box_lo, box_hi = lon_min, lon_max
        out = target.sel(
            {
                lon: oriented_slice(target, lon, slice(box_lo, box_hi)),
                lat: oriented_slice(target, lat, slice(lat_min, lat_max)),
            }
        )
        empty = out.sizes.get(lon, 1) == 0 or out.sizes.get(lat, 1) == 0
    else:
        out, inside = _curvilinear_window(obj, lon, lat, bbox, pad=0.0)
        empty = not bool(inside.any())
        if not empty:
            out = out.where(inside)
    if empty:
        raise ValueError(
            f"select lon/lat box ({lon_min:g}..{lon_max:g}, {lat_min:g}..{lat_max:g}) "
            f"selects nothing from {subject}: check the box against the source's "
            "extent and longitude convention (0-360 vs +/-180)."
        )
    out.attrs["region"] = [lon_min, lat_min, lon_max, lat_max]
    return out


def subset_to_bbox(obj, bbox, pad: float = DEFAULT_PAD):
    """Subset ``obj`` to ``bbox`` (lon_min, lat_min, lon_max, lat_max) plus ``pad``.

    Honours each axis's stored direction (see
    :func:`ocean_skill.operators.oriented_slice` — the same rule ``select`` applies to a
    caller's own range, and deliberately the same code, since a bbox and a ``select``
    band are the same question asked twice), and refuses to return an empty result: no
    overlap at all means the two sources do not cover the same region, which is worth
    saying plainly rather than failing later.

    A box that straddles ``obj``'s longitude seam (a Pacific domain's 77°E-316°E
    against a ±180 reference) comes back cropped in the *box's* convention — the
    reference's longitudes are re-expressed and re-sorted so the band stays one
    contiguous slice rather than being split at the seam or half-dropped.

    A curvilinear ``obj`` (2-D lon/lat, e.g. ROMS's ``lon_rho``/``lat_rho``) has no
    dimension to slice along, so it is windowed to the smallest index rectangle
    containing the padded box instead (:func:`_curvilinear_window`) — unlike
    :func:`subset_to_box`, no ``.where`` mask is applied: this crop is a cost
    optimization ahead of a regrid, not a selection, and windowing to a superset
    of the box is exactly what the rectilinear branch below does too.
    """
    from ocean_skill.operators import oriented_slice

    lon, lat = _lon_name(obj), _lat_name(obj)
    if lon is None or lat is None:
        return obj
    lon_values = np.asarray(obj[lon])
    if lon not in obj.dims and lon_values.ndim == 2:
        out, inside = _curvilinear_window(obj, lon, lat, bbox, pad=pad)
        if not bool(inside.any()):
            lon_min, lat_min, lon_max, lat_max = bbox
            raise ValueError(
                f"no overlap: the reference does not cover the test's extent "
                f"(lon {lon_min:g} to {lon_max:g}, lat {lat_min:g} to "
                f"{lat_max:g}). Check the two sources really overlap in space."
            )
        return out

    lon_min, lat_min, lon_max, lat_max = bbox
    box_lo, box_hi = _bbox_lon_in_convention(obj[lon], lon_min, lon_max)
    if box_lo > box_hi and lon in obj.dims:
        # Contiguous in the box's own convention, split in the reference's: the box
        # straddles the reference's seam (0/360, or the antimeridian — the Pacific
        # case). Slicing either half would silently drop the other, so re-express the
        # *reference* in the box's convention instead: a global axis stays global,
        # re-sorted, and the slice below keeps the one contiguous band the box names.
        obj = harmonize_longitude(
            obj, "0-360" if max(lon_min, lon_max) > 180.0 else "-180-180"
        )
        box_lo, box_hi = lon_min, lon_max
    lon_min, lon_max = box_lo, box_hi
    sel = {}
    if lon in obj.dims:
        sel[lon] = oriented_slice(obj, lon, slice(lon_min - pad, lon_max + pad))
    if lat in obj.dims:
        sel[lat] = oriented_slice(obj, lat, slice(lat_min - pad, lat_max + pad))
    if not sel:
        return obj
    out = obj.sel(**sel)
    empty = [d for d in sel if out.sizes.get(d, 1) == 0]
    if empty:
        raise ValueError(
            f"no overlap along {empty}: the reference does not cover the test's "
            f"extent (lon {lon_min:g} to {lon_max:g}, lat {lat_min:g} to "
            f"{lat_max:g}). Check the two sources really overlap in space."
        )
    return out


def time_span_of(obj, pad_steps: float = 1.0):
    """``(start, stop)`` of ``obj``'s time axis, or ``None`` if it has none.

    The temporal counterpart of :func:`bbox_of`, and wanted for the same reason: a
    reference cropped to the test's *region* but not to its *window* is still the
    whole record. MUR against one year of a regional model is 5462x6251 per step —
    a manageable map, and 2.2 TB once all 8838 daily steps come with it.

    Padded by one of the test's own steps on each side so nearest-neighbour matching
    at the first and last times still has a candidate to reach for; a single-time
    test gets no pad, having no step to measure.
    """
    name = _time_name(obj)
    if name is None or obj.sizes.get(name, 0) == 0:
        return None
    values = np.asarray(obj[name])
    lo, hi = values.min(), values.max()
    if values.size > 1 and pad_steps:
        step = np.median(np.diff(np.sort(values)))
        lo, hi = lo - pad_steps * step, hi + pad_steps * step
    return lo, hi


def subset_to_time(obj, window):
    """Crop ``obj`` to ``window`` along its time axis, if it has one.

    Unlike :func:`subset_to_bbox` an empty result is *not* an error here: a
    climatology or a static field legitimately shares no calendar span with the
    test, and the comparison's own time handling reports that far more precisely
    than a crop can.
    """
    from ocean_skill.operators import oriented_slice

    name = _time_name(obj)
    if name is None or window is None or name not in obj.dims:
        return obj
    out = obj.sel({name: oriented_slice(obj, name, slice(window[0], window[1]))})
    return obj if out.sizes.get(name, 0) == 0 else out


def bbox_of(obj) -> tuple[float, float, float, float]:
    """Return ``(lon_min, lat_min, lon_max, lat_max)`` of ``obj``."""
    lon, lat = _lon_name(obj), _lat_name(obj)
    lo, la = np.asarray(obj[lon]), np.asarray(obj[lat])
    return (
        float(np.nanmin(lo)),
        float(np.nanmin(la)),
        float(np.nanmax(lo)),
        float(np.nanmax(la)),
    )


def _thin_ring(n: int, max_points: int) -> np.ndarray:
    """Return indices thinning a length-``n`` open edge to at most ``max_points``.

    Always keeps index 0 and ``n - 1`` (the edge's own endpoints, i.e. the
    quadrilateral's corners), so thinning can shrink a long edge without ever
    rounding a corner off the traced shape.
    """
    if n <= max_points:
        return np.arange(n)
    step = int(np.ceil((n - 1) / (max_points - 1)))
    return np.r_[np.arange(0, n - 1, step), n - 1]


def perimeter_of(lon, lat, *, max_points: int = 400) -> np.ndarray | None:
    """Return the closed grid-edge ring of ``lon``/``lat`` as an ``(N, 2)`` array.

    For a 2-D (curvilinear) grid this traces the true boundary — the four edge rows/
    columns of the array, walked corner to corner — which is exact for a rotated grid
    like a ROMS domain, not an approximation fit to a point cloud. For a 1-D
    (rectilinear) grid the perimeter *is* the bounding-box rectangle, so that is what
    comes back. Returns ``None`` for anything else (0-D, mismatched ndim, empty).

    Each edge is thinned independently with :func:`_thin_ring` before assembly, so no
    corner of a rotated quadrilateral is ever rounded off by thinning a long edge.
    Longitude is unwrapped first (period 360) so a ring crossing the antimeridian
    stays contiguous — the same idiom :func:`grid_of` uses for the same reason; the
    unwrapped values are equivalent degrees on the sphere, not out-of-range ones.
    """
    lon, lat = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    if lon.ndim != lat.ndim or lon.ndim not in (1, 2) or lon.size == 0 or lat.size == 0:
        return None
    if lon.ndim == 1:
        # 1-D axes are independent (lon has nx values, lat has ny) — unlike the 2-D
        # case they need not share a shape.
        lo0, la0, lo1, la1 = (
            float(np.nanmin(lon)),
            float(np.nanmin(lat)),
            float(np.nanmax(lon)),
            float(np.nanmax(lat)),
        )
        return np.array(
            [[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]]
        )
    if lon.shape != lat.shape:
        return None

    lon = np.unwrap(np.unwrap(lon, axis=1, period=360.0), axis=0, period=360.0)
    ny, nx = lon.shape
    if ny < 2 or nx < 2:
        return None

    # Walk the four edges corner to corner — top left→right, right top→bottom,
    # bottom right→left, left bottom→top — each edge dropping the point it shares
    # with the edge before it, so the ring has no repeated vertex except the closing
    # one. Per-edge thinning budget so a long, thin domain (nx >> ny, say) doesn't
    # starve its short edges to bare corners while the long ones stay dense.
    budget = max(2, max_points // 4)
    top_i = _thin_ring(nx, budget)
    side_i = _thin_ring(ny, budget)
    bottom_i = _thin_ring(nx, budget)

    def _ring(arr: np.ndarray) -> np.ndarray:
        top = arr[0, :][top_i]
        right = arr[:, -1][side_i][1:]
        bottom = arr[-1, ::-1][bottom_i][1:]
        left = arr[::-1, 0][side_i][1:-1]  # both ends already the top/bottom corners
        return np.concatenate([top, right, bottom, left, top[:1]])

    return np.column_stack([_ring(lon), _ring(lat)])


def _as_xesmf(obj):
    """Return ``obj`` with coords named ``lon``/``lat`` as xesmf expects."""
    lon, lat = _lon_name(obj), _lat_name(obj)
    ren = {}
    if lon and lon != "lon":
        ren[lon] = "lon"
    if lat and lat != "lat":
        ren[lat] = "lat"
    return obj.rename(ren) if ren else obj


def _corners_1d(c: np.ndarray) -> np.ndarray:
    """Cell edges for 1-D centres: midpoints, with the two ends extrapolated."""
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _corners_2d(a: np.ndarray) -> np.ndarray:
    """Cell corners ``(ny+1, nx+1)`` for 2-D curvilinear centres ``(ny, nx)``.

    Pads the centres by one ring using linear extrapolation, then averages each 2x2
    block. ROMS grids ship rho/u/v points but no corner (psi) array covering the full
    boundary, so corners have to be derived like this for conservative regridding.
    """
    ny, nx = a.shape
    p = np.empty((ny + 2, nx + 2), dtype=float)
    p[1:-1, 1:-1] = a
    p[0, 1:-1] = 2 * a[0, :] - a[1, :]
    p[-1, 1:-1] = 2 * a[-1, :] - a[-2, :]
    p[:, 0] = 2 * p[:, 1] - p[:, 2]
    p[:, -1] = 2 * p[:, -2] - p[:, -3]
    return 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])


def grid_of(obj, bounds: bool = False) -> xr.Dataset:
    """Return the grid xesmf needs: ``lon``/``lat`` (+ corners when ``bounds``).

    A separate Dataset is required because cell corners live on their own dimensions,
    which a DataArray cannot carry. Handles 1-D (regular) and 2-D (curvilinear)
    coordinates; ROMS ships no full corner array, so corners are derived.
    """
    lon_da, lat_da = obj["lon"], obj["lat"]
    grid = xr.Dataset({"lon": lon_da, "lat": lat_da})
    if not bounds:
        return grid
    lon, lat = np.asarray(lon_da), np.asarray(lat_da)
    if lon.ndim == 1 and lat.ndim == 1:
        grid = grid.assign_coords(
            lon_b=("lon_b", _corners_1d(lon)), lat_b=("lat_b", _corners_1d(lat))
        )
    elif lon.ndim == 2 and lat.ndim == 2:
        # A curvilinear grid straddling its convention's seam folds mid-array
        # (…179.9, -179.9…), and corners averaged across the fold land near 0° —
        # cells spanning half the globe. Unwrap to a continuous field first; ESMF
        # reads degrees on the sphere, so values past ±180 are equivalent, not wrong.
        lon = np.unwrap(np.unwrap(lon, axis=1, period=360.0), axis=0, period=360.0)
        grid = grid.assign_coords(
            lon_b=(("y_b", "x_b"), _corners_2d(lon)),
            lat_b=(("y_b", "x_b"), _corners_2d(lat)),
        )
    else:
        raise ValueError(f"cannot derive bounds for lon/lat with ndim {lon.ndim}")
    return grid


def _single_value(obj, name: str, tol: float = 1e-6) -> float | None:
    """Return the one value ``obj[name]`` takes, or ``None`` if it varies."""
    values = np.asarray(obj[name], dtype="float64").ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if finite.size > 1 and float(np.ptp(finite)) > tol:
        return None
    return float(finite[0])


def point_of(obj) -> tuple[float, float] | None:
    """Return ``(lon, lat)`` when ``obj`` sits at one place, else ``None``.

    Both "is this a point?" and "where?", because every caller needs them together: a
    station lane is recognized *by* having one position, and the position is then what
    the other lane gets sampled at. A field whose lon/lat vary returns ``None``.
    """
    lon_name, lat_name = _lon_name(obj), _lat_name(obj)
    if lon_name is None or lat_name is None:
        return None
    lon, lat = _single_value(obj, lon_name), _single_value(obj, lat_name)
    return None if lon is None or lat is None else (lon, lat)


#: The along-path dimension a vertical section is carried on -- see :func:`path_of`
#: and :func:`ocean_skill.transect.grid_slice`. One name shared by both modules so a
#: section is recognized the same way everywhere, the way :data:`NEAREST` is one
#: spelling shared by every caller of :func:`sample_at`.
ALONG_DIM = "along"


def path_of(obj) -> str | None:
    """Return the along-path dimension name when ``obj`` is a vertical-section lane.

    A :func:`~ocean_skill.transect.grid_slice` result carries 1-D ``lon``/``lat``
    coordinates riding on one dedicated dimension (:data:`ALONG_DIM`) — distinct
    from a rectilinear map, where longitude *is* its own dimension, and distinct
    from a curvilinear one, where it is 2-D. Deliberately keyed on the dimension's
    *name* rather than inferred from shape alone: a moving trajectory's
    ``lon(time)``/``lat(time)`` has the same 1-D-coordinate-on-a-shared-dim shape
    as a section, but is a position that moves in *time*, not a cut through
    *space* — reading it as a section here would be wrong the same way
    :func:`ocean_skill.operators._point_selectable` deliberately leaves a
    trajectory to its own recipe rather than sampling it like a curvilinear grid.
    """
    lon_name, lat_name = _lon_name(obj), _lat_name(obj)
    if lon_name is None or lat_name is None:
        return None
    if ALONG_DIM not in obj.dims:
        return None
    if obj[lon_name].dims != (ALONG_DIM,) or obj[lat_name].dims != (ALONG_DIM,):
        return None
    return ALONG_DIM


#: Vertical-axis dimension names a section lane may carry: ``"z"`` from
#: :func:`ocean_skill.roms.to_depth` (negative-down), or an observational
#: product's own level axis (positive-down) before :func:`_align_along_path`
#: renames it onto ``"z"``. One tuple so the two speak the same vocabulary --
#: :func:`ocean_skill.comparison._prepare`'s own observational ``zname`` lookup
#: is this list's tail (``SECTION_VERTICAL_DIMS[1:]``).
SECTION_VERTICAL_DIMS: tuple[str, ...] = ("z", "depth", "depth_surface", "lev")


def _wrap_lon(lon: float, convention: str) -> float:
    """Put a single longitude in ``convention``, as :func:`harmonize_longitude` does."""
    return lon % 360 if convention == "0-360" else ((lon + 180) % 360) - 180


def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km, elementwise.

    A plain Euclidean distance in *degrees* is not a distance: a degree of longitude is
    64 km at Station Papa's latitude and 111 km at the equator, so the cell it picks can
    be tens of km further away than the nearest one. Verified at 50 N: degrees pick a
    cell 100 km away where this picks one at 71.5 km.
    """
    r = 6371.0088
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _cell_km(obj, lon_name: str, lat_name: str) -> float:
    """Return a representative cell diagonal in km, for reporting an offset against.

    Read off the coordinates themselves rather than trusted from metadata, so it is
    right for a subset grid and for a curvilinear one. A median over the whole grid is
    enough: this only ever scales a warning threshold.
    """
    lon, lat = np.asarray(obj[lon_name]), np.asarray(obj[lat_name])
    mid = float(np.nanmedian(lat))
    if lon.ndim == 1 and lat.ndim == 1:
        dx = float(np.nanmedian(np.abs(np.diff(lon)))) if lon.size > 1 else 0.0
        dy = float(np.nanmedian(np.abs(np.diff(lat)))) if lat.size > 1 else 0.0
    else:
        dx = (
            float(np.nanmedian(np.abs(np.diff(lon, axis=-1))))
            if lon.shape[-1] > 1
            else 0.0
        )
        dy = (
            float(np.nanmedian(np.abs(np.diff(lat, axis=0))))
            if lat.shape[0] > 1
            else 0.0
        )
    km_x = dx * 111.32 * float(np.cos(np.deg2rad(mid)))
    km_y = dy * 110.57
    return float(np.hypot(km_x, km_y))


def _nearest_indices(lon_2d, lat_2d, lon: float, lat: float) -> tuple[int, ...]:
    """Return indices of the closest cell centre, by great-circle distance."""
    distance = _haversine_km(lon_2d, lat_2d, lon, lat)
    return np.unravel_index(int(np.nanargmin(distance)), distance.shape)


def sample_at(
    obj,
    lon: float,
    lat: float,
    *,
    method: str = NEAREST,
    convention: Literal["0-360", "-180-180"] = "-180-180",
    subject: str = "the test lane",
):
    """Return ``obj`` at one location: the nearest cell, or interpolated to the point.

    Both are supported because which one is right depends on the question.
    ``method="nearest"`` invents nothing and never mixes a masked neighbour into the
    answer, so it is the default; ``method="bilinear"`` (or ``"linear"``) removes the
    grid-offset step that makes a coarse product look biased at a point, at the cost of
    a value no cell actually holds.

    Longitudes are harmonized *and* the target is wrapped to match, since a −144.2
    station against a 0-360 grid is the silent-empty-overlap case this module exists to
    prevent. The offset between the requested position and the grid — ``0`` when
    interpolating — is recorded as ``nearest_distance_km`` in the result's attrs
    alongside ``cell_km``, and warned about when it exceeds one cell.

    Missing data is reported rather than routed around: an all-missing result raises
    instead of quietly relocating to the closest wet cell, which for a station near a
    coast or a mask edge is a different body of water.
    """
    lon_name, lat_name = _lon_name(obj), _lat_name(obj)
    if lon_name is None or lat_name is None:
        raise ValueError(
            f"{subject} has no longitude/latitude coordinate, so it cannot be sampled "
            "at a station. Check the source's coordinate names."
        )
    obj = harmonize_longitude(obj, convention)
    lon = _wrap_lon(lon, convention)

    lon_values, lat_values = np.asarray(obj[lon_name]), np.asarray(obj[lat_name])
    rectilinear = (
        lon_values.ndim == 1
        and lat_values.ndim == 1
        and lon_name in obj.dims
        and lat_name in obj.dims
    )
    cell_km = _cell_km(obj, lon_name, lat_name)

    if method == NEAREST:
        if rectilinear:
            out = obj.sel({lon_name: lon, lat_name: lat}, method="nearest")
        else:
            iy, ix = _nearest_indices(lon_values, lat_values, lon, lat)
            dims = obj[lon_name].dims
            out = obj.isel({str(dims[0]): int(iy), str(dims[1]): int(ix)})
        offset = float(
            _haversine_km(float(out[lon_name]), float(out[lat_name]), lon, lat)
        )
    elif method in _INTERPOLATING:
        if rectilinear:
            out = obj.interp({lon_name: lon, lat_name: lat})
        else:
            out = _interp_curvilinear(obj, lon_name, lat_name, lon, lat)
        offset = 0.0
    else:
        raise ValueError(
            f"{method!r} cannot sample at a point: a conservative regrid area-averages "
            "onto a destination cell, and a station has no area. Use "
            'method="nearest" (the containing cell) or method="bilinear" '
            "(interpolated to the position)."
        )

    if not bool(np.isfinite(out).any()):
        remedy = (
            'Use method="nearest" if a neighbouring cell being masked is what did it.'
            if method in _INTERPOLATING
            else "The station may sit in a masked cell; check the source covers it."
        )
        raise ValueError(
            f"{subject} has no valid data at ({lon:g}, {lat:g}) — the "
            f"{'interpolated' if method in _INTERPOLATING else 'nearest'} value is "
            f"missing everywhere (offset {offset:.1f} km, cell ~{cell_km:.1f} km). "
            + remedy
        )
    if offset > cell_km > 0:
        warnings.warn(
            f"{subject}'s nearest cell is {offset:.1f} km from ({lon:g}, {lat:g}), "
            f"more than one cell away (~{cell_km:.1f} km) — the station may be outside "
            "the source's coverage, or in a hole in it.",
            stacklevel=_stacklevel.find(),
        )
    out.attrs["nearest_distance_km"] = offset
    out.attrs["cell_km"] = cell_km
    out.attrs["point_method"] = method
    return out


def _interp_curvilinear(obj, lon_name: str, lat_name: str, lon: float, lat: float):
    """Bilinearly interpolate a curvilinear grid to one point, via xesmf.

    The same library the map path regrids with (``locstream_out`` is its point mode), so
    a point sample and a regridded map agree by construction rather than by two
    implementations happening to match. ROMS is why this exists: ``lon_rho``/``lat_rho``
    are 2-D, so xarray's own ``.interp`` has no orthogonal axes to work along.
    """
    import xesmf as xe

    src = _as_xesmf(obj)
    target = xr.Dataset({"lon": ("point", [lon]), "lat": ("point", [lat])})
    regridder = xe.Regridder(
        grid_of(src), target, "bilinear", locstream_out=True, unmapped_to_nan=True
    )
    out = regridder(src, keep_attrs=True).isel(point=0, drop=True)
    return out.assign_coords({"lon": lon, "lat": lat})


def _interp_locstream(obj, lons, lats):
    """Bilinearly interpolate a curvilinear grid to N points, via xesmf.

    :func:`_interp_curvilinear` generalized from one point to a whole path: the
    locstream target's own dim is named :data:`ALONG_DIM` directly, rather than
    xesmf's own ``"point"`` (as the single-point form uses, then discards via
    ``isel(point=0, drop=True)``) -- so the regridded result already carries the
    along-path dimension every caller downstream expects, with no rename step.

    ``obj`` should already carry only variables safe to hand to one xesmf call --
    see :func:`ocean_skill.transect._bilinear_dataset`, which builds that Dataset
    and is this function's only caller.
    """
    import xesmf as xe

    src = _as_xesmf(obj)
    lons = np.asarray(lons, dtype="float64")
    lats = np.asarray(lats, dtype="float64")
    target = xr.Dataset({"lon": (ALONG_DIM, lons), "lat": (ALONG_DIM, lats)})
    regridder = xe.Regridder(
        grid_of(src), target, "bilinear", locstream_out=True, unmapped_to_nan=True
    )
    out = regridder(src, keep_attrs=True)
    return out.assign_coords({"lon": (ALONG_DIM, lons), "lat": (ALONG_DIM, lats)})


def _check_units(test, reference):
    """Return ``test`` in the reference's units, refusing an impossible difference.

    Subtracting umol/kg from mmol/m3 used to yield a difference of 0.0, labelled with
    the reference's units and no warning at all — plausible, and wrong by the density
    factor. Harmonize first, and refuse outright when the two are not the same physical
    quantity, since no conversion can rescue that.

    Shared by both alignment paths: a mooring against a model has exactly the same
    hazard as a climatology against one, and only the *joining* differs.
    """
    from ocean_skill import units as _units

    same = _units.compatible(test.attrs.get("units"), reference.attrs.get("units"))
    if same is False:
        raise ValueError(
            f"cannot difference {test.attrs.get('units')!r} against "
            f"{reference.attrs.get('units')!r}: not the same physical quantity. "
            "Convert one first, or check the variables really do match."
        )
    if same:
        return _units.to_units(test, reference.attrs.get("units"))
    warnings.warn(
        f"cannot verify units {test.attrs.get('units')!r} vs "
        f"{reference.attrs.get('units')!r}; differencing them unchecked.",
        stacklevel=_stacklevel.find(),
    )
    return test


def _require_2d(da, role: str, *, keep: tuple[str, ...] = ()) -> None:
    """Raise a useful error if ``da`` still carries a dimension beyond lat/lon.

    A leftover axis means the selection or aggregation did not collapse it — most
    often ``aggregate={"time": {"groupby": "month", ...}}``, which produces twelve
    fields where a single reference field is expected, or the ``resample`` spelling
    of the same idea. Without this, xesmf fails much later with a shape mismatch
    that says nothing about the cause.

    ``keep`` names the axes a caller has *asked* to survive — the axis a comparison is
    scoring ``over``, which it reduces itself once the pair is aligned. Anything beyond
    those is still refused: a pointwise metric reduces over the axes it was given, and
    there is nothing it could do with a further one.
    """
    # The horizontal dims are whatever lon/lat are defined *on* — not necessarily
    # named lat/lon: a curvilinear ROMS field is (eta_rho, xi_rho) with 2-D lon/lat
    # coordinates riding on those dims.
    lon, lat = _lon_name(da), _lat_name(da)
    if lon is None or lat is None:
        return  # nothing to measure against; let the regridder complain instead
    spatial = set(da[lon].dims) | set(da[lat].dims) | set(keep)
    extra = [str(d) for d in da.dims if d not in spatial]
    if not extra:
        return
    if keep:
        raise ValueError(
            f"the {role} field has {extra} beyond its horizontal axes and the "
            f"{list(keep)} it is being scored over, so a pointwise metric has nothing "
            "to do with it: it reduces over the axis it was given and draws the rest "
            "as a map. Collapse it with aggregate= (e.g. "
            f'{{"{extra[0]}": "mean"}}), narrow it with select=, or score over it too '
            "by naming it in over=."
        )
    raise ValueError(
        f"the {role} field still has {extra} beyond its horizontal axes, so it "
        "is not a single map. Collapse it with aggregate= (e.g. "
        '{"time": "mean"}) or narrow it with select= (e.g. {"time": "2012-01"}); '
        'a groupby such as {"groupby": "month"} -- or a resample such as '
        '{"resample": "1MS"} -- deliberately keeps a dimension and cannot be '
        "compared against a single field. To plot those panels as they are, "
        "use a model-only field() rather than a comparison, or score against the "
        'axis pointwise with compare(..., over="time").'
    )


def _sample_test_at_instant(test, reference, over):
    """Sample a time-varying test lane at a single-instant reference's own time.

    A ``profile`` reference is one CTD cast: a depth column at a single instant, its
    time a *scalar* coordinate rather than an axis (see
    :func:`ocean_skill.tabular._profile_dataset`). Scored down depth (``over="Z"``),
    the model lane it is compared against still carries its whole time axis, and a
    pointwise metric has nothing to reduce it with -- the vertical score keeps depth,
    not time, so :func:`_require_2d` would refuse the leftover axis. The pairing a
    profile comparison actually means is the model snapshot nearest the cast, so that
    is what this selects, collapsing the test's time to that one instant.

    Deliberately narrow -- returns the test unchanged unless every part holds:

    * the reference is a **point/profile** (:func:`point_of`). A gridded reference's
      scalar time is nominal (an annual climatology stamped ``2000-01-01``), not an
      instant to snap a model to, and its own leftover-axis error still stands.
    * the reference's time is a **scalar** (0-D) coordinate, not a time axis it is
      being matched on.
    * the test's own time axis is **not** the one being scored ``over`` -- that is the
      mooring recipe, where :func:`match_axis` already owns the time pairing.

    The chosen snapshot's stamp and its distance from the cast are recorded on the
    result's attrs, the way :func:`sample_at` records its spatial offset: a large gap
    (the model run barely covers the cast date) is the caller's to judge, not this to
    refuse.
    """
    if point_of(reference) is None:
        return test
    rtime = find_coord(reference, "time")
    if rtime is None or rtime.ndim != 0:
        return test
    from ocean_skill.operators import resolve_dim

    tdim = resolve_dim(test, "T")
    if tdim is None or tdim not in test.dims:
        return test
    if over is not None and tdim == over:
        return test  # scored over time -- match_axis owns this pairing
    picked = test.sel({tdim: rtime.values}, method="nearest")
    chosen = picked[tdim].values if tdim in picked.coords else None
    if chosen is not None:
        picked.attrs["test_time"] = str(chosen)
        offset = (np.datetime64(chosen) - np.datetime64(rtime.values)) / np.timedelta64(
            1, "h"
        )
        picked.attrs["time_to_reference_hours"] = float(np.abs(offset))
    return picked


# ------------------------------------------------------------------- axis alignment

#: How much finer one lane must be before its steps are *averaged into* the other's bins
#: rather than paired with them one-for-one. Below this the two are the same cadence
#: stamped differently — a ROMS daily average at 12:00 against an L3 daily composite
#: stamped 00:00 — and rebinning would only relabel what pairing already matches.
COARSER_BY = 1.5

#: Fraction of the coarser cadence a nearest-match reaches across when the caller names
#: no tolerance. Half a bin, so at most one candidate can claim each stamp and the
#: pairing is unambiguous rather than merely closest.
NEAREST_TOLERANCE_FRACTION = 0.5

#: Matched steps below which a pointwise metric is worth warning about: a correlation
#: over five time steps is noise wearing a number's clothes.
MIN_OVERLAP = 10

#: Words a catalog ``period`` uses for an instantaneous product rather than a composite.
_SNAPSHOT_PERIODS = ("snapshot", "instantaneous", "instant", "point")


def _axis_floats(da, axis: str, role: str) -> np.ndarray:
    """Return the axis' coordinate as float64 seconds (times), or its own units.

    Seconds rather than nanoseconds deliberately: ns since 1970 is ~1.3e18, well past
    float64's exactly-representable range, so a round trip through it jitters stamps by
    hundreds of nanoseconds. Seconds are exact and nothing here needs finer.
    """
    if axis not in da.coords:
        raise ValueError(
            f"the {role} lane's {axis!r} is a bare dimension with no coordinate, so "
            "there is nothing to match against: pairing by position would line up step "
            "0 with step 0 for no reason at all. For a ROMS run this is usually a "
            "dataset opened with decode_times=False — give the catalog entry a "
            "reference_date (or time_coord) so the axis carries real dates."
        )
    arr = np.asarray(da[axis].values)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[s]").astype("float64")
    if arr.dtype.kind in "iuf":
        return arr.astype("float64")
    raise ValueError(
        f"the {role} lane's {axis!r} is a {arr.dtype} axis (cftime, most likely: a "
        "model run on a 360-day or noleap calendar). It cannot be matched against real "
        "calendar dates — the overlap would come out empty with nothing to say why. "
        'Convert it first with .convert_calendar("standard").'
    )


def _cadence(values: np.ndarray) -> float | None:
    """Median absolute spacing of an axis, or ``None`` if it cannot be measured.

    The median rather than the mean, for the reason :func:`ocean_skill.build._spacing`
    gives: composite products are not evenly spaced — MODIS 8-day bins restart every
    1 January, so one bin a year is short — and the mean quietly launders that.
    """
    diffs = np.abs(np.diff(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else None


def axis_edges(values: np.ndarray, *, anchor: str = "center") -> np.ndarray:
    """Bin edges for an axis whose product does not declare its own.

    ``anchor="center"`` treats each value as the middle of its bin, so the edges are the
    midpoints between neighbours with the two ends extrapolated — which is
    :func:`_corners_1d`, the same operation applied to longitude and latitude, and
    deliberately the same code so that "cell edges" has one definition here.

    ``anchor="start"`` treats each value as the *beginning* of its bin instead, which is
    how many composite products stamp themselves: a daily L3 file labelled with midnight
    covers that whole day, not the twelve hours either side of midnight. The distinction
    is not cosmetic — getting it wrong misassigns half of every bin's steps — so
    :func:`infer_bin_anchor` reads it off the stamps rather than assuming one.
    """
    values = np.asarray(values, dtype="float64")
    if values.size < 2:
        raise ValueError(
            "cannot derive bin edges from a single step: there is no spacing to "
            "measure. Name a tolerance= and the steps will be paired instead of binned."
        )
    if anchor == "center":
        return _corners_1d(values)
    step = float(np.median(np.diff(values)))
    if anchor == "start":
        return np.concatenate([values, [values[-1] + step]])
    if anchor == "end":
        return np.concatenate([[values[0] - step], values])
    raise ValueError(f"unknown anchor {anchor!r}; expected 'center', 'start' or 'end'")


def infer_bin_anchor(values) -> str:
    """Whether an axis stamps the *start* of each bin or its *middle*.

    Read off the stamps, which say more than they look like they do. A product labelling
    a bin with its first instant lands on a period boundary — midnight for a daily or
    8-day composite, the first of the month for a monthly one — while a product that
    labels the middle deliberately does not: WOA and OceanSODA stamp the 15th, a ROMS
    daily average noon. So "is every stamp on its period's boundary?" answers it, and
    answers it correctly for both spellings of the same product.

    This matters most where it is least visible. Hourly model output averaged into
    centre-anchored bins around a midnight-stamped daily satellite composite would put
    noon-to-noon in a box labelled with the date — half of every day's data in the wrong
    bin, with nothing in the result to show it. Falls back to ``"center"`` for a numeric
    (non-calendar) axis and for sub-daily bins, where the offset is below anything the
    comparison resolves.
    """
    arr = np.asarray(values)
    if arr.dtype.kind != "M":
        return "center"
    import pandas as pd

    stamps = pd.DatetimeIndex(arr)
    cadence = _cadence(arr.astype("datetime64[s]").astype("float64"))
    if cadence is None:
        return "center"
    days = cadence / 86400.0
    if days >= 27:  # monthly or longer: the boundary is the first of the month
        return "start" if bool(stamps.is_month_start.all()) else "center"
    if days >= 0.9:  # daily or a multi-day composite: the boundary is midnight
        return "start" if bool((stamps.normalize() == stamps).all()) else "center"
    return "center"


def is_composite(da, axis: str, metadata: dict | None = None) -> bool | None:
    """Whether the values on ``axis`` are averages over a period or instants.

    ``None`` means nothing says. Read in order from the CF ``cell_methods`` attribute
    (``"time: mean"`` is a composite, ``"time: point"`` an instant) and then from the
    catalog entry's ``period``, which the MODIS builder writes.
    """
    methods = str(da.attrs.get("cell_methods", ""))
    for token in (f"{axis}:", "time:", "T:"):
        if token in methods:
            tail = methods.split(token, 1)[1].strip().split()[:1]
            word = tail[0].rstrip(",;") if tail else ""
            if word == "point":
                return False
            if word in ("mean", "sum", "average", "maximum", "minimum", "median"):
                return True
    period = (metadata or {}).get("period")
    if period:
        return str(period).lower() not in _SNAPSHOT_PERIODS
    return None


def resolve_match_method(
    test_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    composite: bool | None,
    test_composite: bool | None = None,
    tolerance: float | None = None,
    calendar: bool = True,
) -> tuple[str, str, float | None, str]:
    """Decide how the two axes should be brought onto one another.

    The temporal counterpart of choosing a regrid method, and settled the same way:
    the pair lands on whichever lane is **coarser**, and the finer lane moves. A
    materially finer test is **averaged into** the reference's bins — the analogue of
    ``conservative_normed``, and the reason it is a default rather than something to
    ask for: nobody has to request area-averaging in space either. Against an
    instantaneous reference the test is **sampled** at the nearest step instead, the
    analogue of ``bilinear``. A materially finer *reference* is handled the same way,
    mirrored — averaged into the test's bins, or sampled at the test's instants if the
    test declares itself a series of snapshots — with a warning rather than silence,
    since coarsening the reference does change the thing being scored against and
    that is worth saying out loud even though it is not refused.

    Returns ``(method, reason, tolerance, target)``; ``target`` is ``"test"`` or
    ``"reference"`` and names the lane whose axis is the frame — the temporal twin of
    a regrid's destination grid. The reason is recorded in the aligned result's attrs
    so the choice is on paper rather than in someone's memory.
    """
    ct, cr = _cadence(test_values), _cadence(reference_values)
    if ct is None or cr is None:
        # one lane is a single step: there is nothing to bin, only something to pair
        known = ct or cr
        if known is None:
            return "exact", "neither lane has a measurable cadence", None, "reference"
        return (
            "nearest",
            "one lane has a single step, so its counterpart is sampled at it",
            tolerance if tolerance is not None else NEAREST_TOLERANCE_FRACTION * known,
            "reference",
        )
    if cr * COARSER_BY <= ct:
        disclosure = (
            ", and nothing on the test says whether its steps are period averages or "
            "instants, so they are taken to be averages"
            if test_composite is None
            else ""
        )
        hatch = (
            "To coarsen the reference deliberately (or differently), spell it out — "
            'aggregate={"reference": {"time": {"resample": "'
            + _pandas_freq(ct)
            + '", "reduce": "mean"}}, "test": {}} — or swap the roles so the finer '
            "product is the test."
        )
        if test_composite is False:
            warnings.warn(
                f"the reference steps every {_duration(cr, calendar)} and the test "
                f"every {_duration(ct, calendar)}, so the reference is the finer of "
                "the two and is sampled at the test's instants (the test declares "
                "instantaneous steps). That subsets the thing being scored against. "
                f"{hatch}",
                stacklevel=_stacklevel.find(),
            )
            return (
                "nearest",
                f"the test is an instantaneous product every "
                f"{_duration(ct, calendar)}; the reference steps every "
                f"{_duration(cr, calendar)} and is sampled at those instants",
                tolerance if tolerance is not None else NEAREST_TOLERANCE_FRACTION * ct,
                "test",
            )
        warnings.warn(
            f"the reference steps every {_duration(cr, calendar)} and the test every "
            f"{_duration(ct, calendar)}, so the reference is the finer of the two and "
            "is averaged into the test's bins — the finer lane lands on the coarser "
            "one's axis, exactly as a regrid does in space. That coarsens the thing "
            f"being scored against{disclosure}. {hatch}",
            stacklevel=_stacklevel.find(),
        )
        return (
            "mean",
            f"the reference steps every {_duration(cr, calendar)} and the test every "
            f"{_duration(ct, calendar)}, so the reference is averaged into the "
            "test's bins",
            tolerance,
            "test",
        )
    if ct * COARSER_BY <= cr:
        if composite is False:
            return (
                "nearest",
                f"the reference is an instantaneous product every "
                f"{_duration(cr, calendar)}; the test steps every "
                f"{_duration(ct, calendar)} and is sampled at those instants",
                tolerance if tolerance is not None else NEAREST_TOLERANCE_FRACTION * cr,
                "reference",
            )
        return (
            "mean",
            f"the test steps every {_duration(ct, calendar)} and the reference every "
            f"{_duration(cr, calendar)}, so the test is averaged into its bins",
            tolerance,
            "reference",
        )
    return (
        "nearest",
        f"both lanes step about every {_duration(cr, calendar)}, so their steps are "
        "paired rather than rebinned",
        tolerance
        if tolerance is not None
        else NEAREST_TOLERANCE_FRACTION * max(ct, cr),
        "reference",
    )


def _duration(seconds: float | None, calendar: bool = True) -> str:
    """Spell a step along the axis the way a person would say it.

    ``calendar=False`` for an axis that is not time — a depth in metres — where the
    value is its own unit and calling it seconds would be a plain lie.
    """
    if seconds is None:
        return "unknown"
    if not calendar:
        return f"{seconds:g}"
    days = seconds / 86400.0
    if 27 <= days <= 32:
        return "month"
    if 355 <= days <= 375:
        return "year"
    if days >= 1:
        return f"{days:.3g} day{'s' if round(days, 3) != 1 else ''}"
    hours = seconds / 3600.0
    if hours >= 1:
        return f"{hours:.3g} hour{'s' if round(hours, 3) != 1 else ''}"
    minutes = seconds / 60.0
    if minutes >= 1:
        return f"{minutes:.3g} minute{'s' if round(minutes, 3) != 1 else ''}"
    return f"{seconds:.3g} s"


def _pandas_freq(seconds: float) -> str:
    """Return a pandas resample alias for a cadence, for use in an error message."""
    days = seconds / 86400.0
    if 27 <= days <= 32:
        return "1MS"
    if days >= 1:
        return f"{max(round(days), 1)}D"
    return f"{max(round(seconds / 3600.0), 1)}h"


def _sorted_on(da, axis: str):
    """Return ``da`` with ``axis`` ascending: every step below assumes that order."""
    values = np.asarray(da[axis].values)
    if values.size > 1 and values[0] > values[-1]:
        return da.sortby(axis)
    return da


def _match_vertical(test, reference, tdim: str, rdim: str):
    """Bring the test lane onto the reference's own vertical levels, by interpolation.

    The vertical counterpart of the ``{mean, nearest, exact}`` choice
    :func:`match_axis` makes for time -- but settled once here, not chosen: a water
    column has no "composite vs instantaneous" question to answer the way a time
    axis does (there is no such thing as a depth level that is itself an *average*
    over a range of depths), so the test lane is always linearly interpolated onto
    the reference's own levels rather than binned or nearest-matched. A level
    outside the test's own vertical range comes back NaN (no extrapolation), the
    same convention :func:`ocean_skill.roms.to_depth` uses for exactly the same
    reason.

    Sign conventions are reconciled before interpolating -- ROMS's own ``z``/
    ``z_rho`` read negative-down, an observational product's own axis usually
    already reads positive-down -- and the *reference's* convention is what
    survives: the shared axis keeps the reference's own literal values, exactly as
    :func:`match_axis` keeps the reference's own stamps for a time match. Always
    lands on the reference (never the test), unlike time's coarser-wins rule:
    a station reference has one water column and nothing coarser to defer to.
    """
    for role, dim, lane in (("test", tdim, test), ("reference", rdim, reference)):
        if dim not in lane.coords:
            raise ValueError(
                f"the {role} lane's {dim!r} axis has no coordinate of its own -- "
                "still in native s-coordinates (ROMS ships no coordinate for a "
                "bare s_rho index), which varies by grid column and so cannot be "
                "matched against the reference's fixed levels before the test is "
                'sampled at a point. Use select={"depth": [...]} (a list of '
                "metres) to interpolate the test onto fixed levels first, rather "
                'than select={"depth": "column"} or a band.'
            )
    ref_vals = np.asarray(reference[rdim].values, dtype="float64")
    test_vals = np.asarray(test[tdim].values, dtype="float64")
    ref_pos = np.abs(ref_vals)
    test_pos = np.abs(test_vals)
    order = np.argsort(test_pos)
    test_sorted = test.isel({tdim: order}).assign_coords({tdim: test_pos[order]})
    interpolated = test_sorted.interp({tdim: ref_pos}, method="linear")
    if tdim != rdim:
        interpolated = interpolated.rename({tdim: rdim})
    interpolated = interpolated.assign_coords({rdim: (rdim, ref_vals)})

    if not bool(np.isfinite(interpolated).any()):
        warnings.warn(
            f"interpolating the test lane onto the reference's {rdim!r} levels "
            f"({np.nanmin(ref_vals):g} to {np.nanmax(ref_vals):g}) leaves nothing "
            f"finite -- the test's own vertical range is "
            f"{np.nanmin(test_vals):g} to {np.nanmax(test_vals):g}. The two "
            "columns may not overlap at all.",
            stacklevel=_stacklevel.find(),
        )

    report = {
        "match_method": "interp",
        "match_reason": (
            "a vertical axis is matched by linear interpolation of the test lane "
            "onto the reference's own levels, not binned or nearest-matched the "
            "way time is"
        ),
        "match_target": "reference",
        "axis": rdim,
        "n_matched": int(reference.sizes[rdim]),
    }
    return interpolated, reference, report


def match_axis(
    test,
    reference,
    *,
    over: str,
    method: str = "auto",
    tolerance: float | None = None,
    min_overlap: int = MIN_OVERLAP,
    metadata: dict | None = None,
    test_metadata: dict | None = None,
    bin_anchor: str = "auto",
):
    """Bring the two lanes onto one ``over`` axis, and report what it took.

    Direction follows the same rule alignment uses everywhere here: the pair lands on
    whichever lane is **coarser**, named as the reference names it either way. Nothing
    is done to the frame lane except that steps it ends up with no counterpart data for
    are dropped — an all-NaN step contributes nothing to a metric and would make the
    difference field say something untrue about it.

    Returns ``(test, reference, report)`` with both lanes on one axis, named as the
    reference names it; ``report["match_target"]`` says which lane's *stamps* the axis
    actually carries (see :func:`resolve_match_method`). Everything measured for the
    report is measured on the *coordinate*, never by walking the data: the counts are
    index arithmetic and stay free even when the lanes are a year of daily maps.
    """
    from ocean_skill.operators import _CF_AXES, resolve_dim

    tdim, rdim = resolve_dim(test, over), resolve_dim(reference, over)
    for role, dim, lane in (("test", tdim, test), ("reference", rdim, reference)):
        if dim is None or dim not in lane.dims:
            raise ValueError(
                f"the {role} lane has no {over!r} axis to score over (its dimensions "
                f"are {list(lane.dims)}). For a comparison of single maps, leave over= "
                "unset."
            )

    if _CF_AXES.get(over) == "vertical":
        # A water column has no "composite vs instantaneous" question the way a
        # time axis does, so there is nothing here to choose the way `method`
        # chooses for time -- see _match_vertical for what happens instead.
        return _match_vertical(test, reference, tdim, rdim)

    test, reference = _sorted_on(test, tdim), _sorted_on(reference, rdim)
    tf = _axis_floats(test, tdim, "test")
    rf = _axis_floats(reference, rdim, "reference")
    # captured before matching: a failure to match empties the lanes, and the spans are
    # exactly what the message about it has to say
    spans = (_span(test, tdim), _span(reference, rdim))

    reason = f"method={method!r} as asked"
    target = "reference"
    if method == "auto":
        ref_composite = is_composite(reference, rdim, metadata)
        test_composite = is_composite(test, tdim, test_metadata)
        method, reason, tolerance, target = resolve_match_method(
            tf,
            rf,
            composite=ref_composite,
            test_composite=test_composite,
            tolerance=tolerance,
        )
        if target == "reference" and ref_composite is None and method == "mean":
            warnings.warn(
                "nothing on the reference says whether its steps are period averages "
                "or instantaneous — no CF cell_methods on the variable, no 'period' in "
                "its catalog entry — so it is taken to be a composite and the test is "
                f"averaged into its bins ({reason}). Pass time_method='nearest' if the "
                "reference is really a series of snapshots, or give the catalog entry "
                "a period (or the variable a cell_methods) to settle it for good.",
                stacklevel=_stacklevel.find(),
            )

    report: dict[str, Any] = {
        "match_method": method,
        "match_reason": reason,
        "match_target": target,
    }
    if tolerance is not None:
        report["match_tolerance"] = float(tolerance)

    if method == "mean":
        test, reference, extra = _match_by_mean(
            test, reference, tdim, rdim, tf, rf, bin_anchor, target=target
        )
        report.update(extra)
    elif method == "nearest":
        test, reference, extra = _match_by_nearest(
            test, reference, tdim, rdim, tf, rf, tolerance, target=target
        )
        report.update(extra)
    elif method == "exact":
        test, reference, extra = _match_exactly(test, reference, tdim, rdim)
        report.update(extra)
    else:
        raise ValueError(
            f"unknown time_method {method!r}; expected 'auto', 'mean', 'nearest' or "
            "'exact'"
        )

    matched = int(reference.sizes[rdim])
    # both lanes now name the axis as the reference names it, whichever lane moved
    report["axis"] = rdim
    report["n_matched"] = matched
    if matched == 0:
        raise ValueError(
            f"no overlap along {over!r}: the test spans {spans[0]} and the reference "
            f"{spans[1]}, matched with {method!r}. Check the two really cover the same "
            "period; a tolerance= widens a nearest match, and select= narrows either "
            "side to a period they share."
        )
    if matched < min_overlap:
        warnings.warn(
            f"only {matched} steps matched along {over!r}. A pointwise metric is a "
            "statistic over exactly those steps, so a correlation from this few is "
            "noise. Widen the period with select=, or reduce it to a single map and "
            "use a plain comparison.",
            stacklevel=_stacklevel.find(),
        )
    return test, reference, report


def _span(da, dim: str) -> str:
    """``first to last`` along ``dim``, for an error message."""
    if dim not in da.coords or da.sizes.get(dim, 0) == 0:
        return "an unlabelled axis"
    values = np.asarray(da[dim].values)
    return f"{values[0]} to {values[-1]}"


def _match_by_mean(test, reference, tdim, rdim, tf, rf, bin_anchor, target="reference"):
    """Average the finer lane's steps into the coarser lane's (the frame's) bins.

    ``target`` names the frame — ``"reference"`` for the historical direction (the
    test is binned into the reference's bins), ``"test"`` for the mirror (a finer
    reference is binned into the test's). Either way the *shared* axis keeps the
    reference's dimension name (``rdim``); only which lane's stamps survive differs.
    """
    if target == "reference":
        frame, fdim, fvals, frame_role = reference, rdim, rf, "reference"
        binned, bdim, bvals, binned_role = test, tdim, tf, "test"
    else:
        frame, fdim, fvals, frame_role = test, tdim, tf, "test"
        binned, bdim, bvals, binned_role = reference, rdim, rf, "reference"

    if bin_anchor == "auto":
        bin_anchor = infer_bin_anchor(frame[fdim].values)
    edges = axis_edges(fvals, anchor=bin_anchor)

    which = np.searchsorted(edges, bvals, side="right") - 1
    inside = (which >= 0) & (which < fvals.size)
    stamps = np.asarray(frame[fdim].values)
    if not inside.any():
        # nothing to group: hand back empty lanes, and match_axis raises with the spans
        empty_binned = binned.isel({bdim: []})
        if bdim != rdim:
            empty_binned = empty_binned.rename({bdim: rdim})
        empty_frame = frame.isel({fdim: []})
        if fdim != rdim:
            empty_frame = empty_frame.rename({fdim: rdim})
        extra = {"bin_anchor": bin_anchor, "steps_outside_bins": int(bvals.size)}
        if target == "reference":
            return empty_binned, empty_frame, extra
        return empty_frame, empty_binned, extra
    label = xr.DataArray(stamps[which[inside]], dims=(bdim,), name="_osk_bin")
    attrs = dict(binned.attrs)
    grouped = binned.isel({bdim: inside}).groupby(label).mean(bdim)
    # a reduction drops attrs, and `units` has to survive: align() checks it next
    grouped.attrs = attrs
    grouped = grouped.rename({"_osk_bin": rdim})
    filled = np.asarray(grouped[rdim].values)
    frame_out = frame.sel({fdim: filled})
    if fdim != rdim:
        frame_out = frame_out.rename({fdim: rdim})

    counts = np.bincount(which[inside], minlength=fvals.size)
    typical = float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0
    from ocean_skill.operators import SHORT_BIN_FRACTION

    short = int(((counts > 0) & (counts < SHORT_BIN_FRACTION * typical)).sum())
    empty = int((counts == 0).sum())
    if empty:
        warnings.warn(
            f"{empty} of the {frame_role}'s {fvals.size} steps had no {binned_role} "
            "data in their bin and were dropped. An all-NaN step scores nothing and "
            "would make the difference field claim otherwise.",
            stacklevel=_stacklevel.find(),
        )
    if short:
        warnings.warn(
            f"{short} of the {frame_role}'s bins caught fewer than "
            f"{SHORT_BIN_FRACTION:.0%} of the usual {typical:g} {binned_role} steps, "
            "so those steps are averages over part of a period labelled like a whole "
            "one — usually the first and last bin of the selection. Narrow select= to "
            "whole periods to drop them.",
            stacklevel=_stacklevel.find(),
        )
    extra = {
        "bin_anchor": bin_anchor,
        "steps_outside_bins": int((~inside).sum()),
        "bins_empty": empty,
        "bins_short": short,
        "steps_per_bin": typical,
    }
    if target == "reference":
        return grouped, frame_out, extra
    return frame_out, grouped, extra


def _match_by_nearest(
    test, reference, tdim, rdim, tf, rf, tolerance, calendar=True, target="reference"
):
    """Pair each frame step with the nearest step in the lane that moves onto it.

    ``target`` names the frame, as in :func:`_match_by_mean`: ``"reference"`` for the
    historical direction (the test is sampled at the reference's instants), ``"test"``
    for the mirror (a finer reference is sampled at the test's).
    """
    import pandas as pd

    if target == "reference":
        frame_dim, frame_vals = rdim, rf
        mover_dim, mover_vals = tdim, tf
        frame_role, mover_role = "reference", "test"
    else:
        frame_dim, frame_vals = tdim, tf
        mover_dim, mover_vals = rdim, rf
        frame_role, mover_role = "test", "reference"

    index = pd.Index(mover_vals)
    pos = index.get_indexer(
        frame_vals, method="nearest", **({"tolerance": tolerance} if tolerance else {})
    )
    keep = pos >= 0

    frame_da = reference if target == "reference" else test
    mover_da = test if target == "reference" else reference
    frame_da = frame_da.isel({frame_dim: keep})
    mover_da = mover_da.isel({mover_dim: pos[keep]})
    offsets = (
        np.abs(mover_vals[pos[keep]] - frame_vals[keep]) if keep.any() else np.empty(0)
    )

    # the frame's own stamps become the shared axis, whichever lane is the frame
    attrs = dict(mover_da.attrs)
    frame_stamps = np.asarray(frame_da[frame_dim].values)
    mover_da = mover_da.assign_coords({mover_dim: frame_stamps})
    if mover_dim != rdim:
        mover_da = mover_da.rename({mover_dim: rdim})
    mover_da.attrs = attrs
    if frame_dim != rdim:
        frame_da = frame_da.rename({frame_dim: rdim})

    test, reference = (
        (mover_da, frame_da) if target == "reference" else (frame_da, mover_da)
    )

    if offsets.size and float(offsets.max()) > 0:
        warnings.warn(
            f"paired {int(keep.sum())} steps by nearest match, shifting each by up to "
            f"{_duration(float(offsets.max()), calendar)} (typically "
            f"{_duration(float(np.median(offsets)), calendar)}). The two products "
            "stamp the same period differently; the pairing is in the result's attrs.",
            stacklevel=_stacklevel.find(),
        )
    unmatched = int((~keep).sum())
    if unmatched:
        warnings.warn(
            f"{unmatched} {frame_role} steps had no {mover_role} step within "
            f"{_duration(tolerance, calendar)} and were dropped.",
            stacklevel=_stacklevel.find(),
        )
    return (
        test,
        reference,
        {
            "steps_unmatched": unmatched,
            "offset_max": float(offsets.max()) if offsets.size else 0.0,
            "offset_median": float(np.median(offsets)) if offsets.size else 0.0,
        },
    )


def _match_exactly(test, reference, tdim, rdim):
    """Inner-join the two lanes on identical axis values.

    ``exclude`` is load-bearing: without it xarray also inner-joins the horizontal
    coordinates, and two lanes on different grids — which is the whole reason a regrid
    follows — come back empty. That is the silent-empty-overlap failure this module
    exists to prevent, arriving by a different door.
    """
    if tdim != rdim:
        test = test.rename({tdim: rdim})
    exclude = (set(test.dims) | set(reference.dims)) - {rdim}
    test, reference = xr.align(test, reference, join="inner", exclude=exclude)
    return test, reference, {}


def _coverage(regridder, src, over: str | None, role: str = "test"):
    """Regrid a field of ones over the source's valid cells: the fraction covered.

    ``src`` is whichever lane the regrid reads from — the finer one — so the mask
    always describes how much valid *source* data landed in each coarser target cell;
    ``role`` names that lane for the warning below.

    With a surviving axis this is one extra regrid *per step*, which is worth avoiding
    when it buys nothing. A model's missing cells are its land mask, which does not
    move, so the mask is checked for invariance along ``over`` — exactly, with
    ``any == all``, not assumed — and collapsed to one step when it holds. The 2-D
    result broadcasts back over every step, which is what a static mask means. A mask
    that genuinely does move — a satellite reference's clouds — pays for the per-step
    version and says so.
    """
    finite = np.isfinite(src)
    if over is not None and over in finite.dims:
        if src.chunks is None and bool((finite.any(over) == finite.all(over)).all()):
            one = finite.isel({over: 0}, drop=True)
            return regridder(xr.ones_like(src.isel({over: 0}, drop=True)).where(one))
        warnings.warn(
            f"the {role}'s valid cells change along {over!r}, so coverage is computed "
            f"for every one of its {src.sizes[over]} steps rather than once. Expected "
            "of a field with a moving mask; surprising for a model land mask.",
            stacklevel=_stacklevel.find(),
        )
    return regridder(xr.ones_like(src).where(finite))


#: How many distinct (method, source grid, target grid) weight sets to keep alive at
#: once. Weight matrices are large, so this is small on purpose -- big enough that a
#: fan over one pair of grids (a monthly `times=` fan against a fixed reference, most
#: obviously) reuses one entry every step rather than evicting itself, too small to
#: let an unrelated run's weights linger once that fan is done.
_REGRIDDER_MEMO_SIZE = 4

#: ``{(method, src_token, tgt_token): regridder}``, most-recently-used last. Keyed on
#: grid *content*, not object identity, since every fanned comparison builds its own
#: fresh (but numerically identical) lane objects -- see :func:`_grid_token`.
_REGRIDDER_MEMO: OrderedDict[tuple, Any] = OrderedDict()


def _grid_token(grid: xr.Dataset) -> tuple:
    """Return a hashable fingerprint of a :func:`grid_of` result: its shape and values.

    Built from the centre coordinates (``lon``/``lat``) only, not the corner arrays
    (``lon_b``/``lat_b``) a conservative method also carries: corners are a pure
    function of the centres (:func:`_corners_1d`/:func:`_corners_2d`), so two grids
    with equal centres always have equal corners too, and hashing the smaller array
    is cheaper for no loss of precision. ``need_bounds`` itself is a pure function of
    ``method``, which the memo key already carries alongside this token.
    """
    lon, lat = np.asarray(grid["lon"]), np.asarray(grid["lat"])
    return (
        lon.shape,
        lat.shape,
        hashlib.sha256(np.ascontiguousarray(lon).tobytes()).hexdigest(),
        hashlib.sha256(np.ascontiguousarray(lat).tobytes()).hexdigest(),
    )


def _regridder_for(src_grid: xr.Dataset, tgt_grid: xr.Dataset, method: str):
    """Return a weight-built ``xe.Regridder`` for this (grid, grid, method), memoized.

    A :func:`compare` fan over time builds one :class:`Comparison` per bin, each
    reading its own fresh lane objects -- but against one fixed reference (or two
    runs of one model, before :func:`_shared_grid` even gets here) every bin regrids
    the *same two grids*, and the weights xesmf computes for them do not depend on
    the data, only the geometry. Recomputing that per bin would pay the pipeline's
    most expensive step N times for a result that is identical N times over; this
    keys on the grids' own content (:func:`_grid_token`) so fresh-but-equal lane
    objects hit, evicting least-recently-used past :data:`_REGRIDDER_MEMO_SIZE`.

    Direction is already resolved by the time this is called (the caller passes
    ``src``/``tgt`` in the order :func:`_regrid_target` decided), so it is part of
    the key only in the sense that swapping the two arguments changes the token
    order -- there is no separate "direction" component to track.
    """
    import xesmf as xe

    key = (method, _grid_token(src_grid), _grid_token(tgt_grid))
    cached = _REGRIDDER_MEMO.get(key)
    if cached is not None:
        _REGRIDDER_MEMO.move_to_end(key)
        return cached
    regridder = xe.Regridder(src_grid, tgt_grid, method, unmapped_to_nan=True)
    _REGRIDDER_MEMO[key] = regridder
    if len(_REGRIDDER_MEMO) > _REGRIDDER_MEMO_SIZE:
        _REGRIDDER_MEMO.popitem(last=False)
    return regridder


def clear_regridder_memo() -> None:
    """Drop every memoized regridder, freeing the weights it held.

    Exposed for tests (each gets a clean memo, see ``tests/conftest.py``) and for a
    long-lived process that has moved on to different grids and wants the memory
    back sooner than LRU eviction would give it up.
    """
    _REGRIDDER_MEMO.clear()


def _shared_grid(test, reference) -> bool:
    """Report whether ``test``/``reference`` sit on exactly the same horizontal grid.

    Checked on the coordinate *values* (not names or attrs) after
    :func:`harmonize_longitude` has already reconciled conventions, so two lanes of
    the same model run — a physics run and a biogeochemistry rerun sharing one
    ROMS domain, most obviously — are recognized however each renamed its own
    lon/lat. Exact equality on purpose: a near-equal grid is a genuinely different
    grid, and :func:`_regrid_target`'s hysteresis already covers "close enough to
    not flip-flop" — this is only for "there is nothing to regrid at all".
    """
    tlon, tlat = _lon_name(test), _lat_name(test)
    rlon, rlat = _lon_name(reference), _lat_name(reference)
    if tlon is None or tlat is None or rlon is None or rlat is None:
        return False
    tlon_da, tlat_da = test[tlon], test[tlat]
    rlon_da, rlat_da = reference[rlon], reference[rlat]
    if tlon_da.dims != rlon_da.dims or tlat_da.dims != rlat_da.dims:
        # Same physical grid but different dimension names would still make
        # xarray's own name-based alignment misbehave on `test - reference` below,
        # so it is treated as not shared rather than risking a silent broadcast.
        return False
    return bool(
        np.array_equal(np.asarray(tlon_da), np.asarray(rlon_da))
    ) and bool(np.array_equal(np.asarray(tlat_da), np.asarray(rlat_da)))


def _regrid_target(
    test,
    reference,
    target: Literal["auto", "test", "reference"] = "auto",
) -> tuple[str, str]:
    """Decide which lane's grid the pair lands on: the **coarser** of the two.

    The spatial counterpart of :func:`resolve_match_method`, settled by the same
    measurement: cell size read off the coordinates themselves (:func:`_cell_km`,
    which handles regular and curvilinear grids alike), with :data:`COARSER_BY` as
    the same hysteresis, so two grids of essentially one resolution never flip-flop
    between directions. Regridding always moves the finer lane onto the coarser
    because that is what ``conservative_normed`` *means* — area-averaging source
    cells into bigger target cells; aimed the other way it would invent subgrid
    structure the coarse lane never had. A 4 km satellite reference against a
    coarser model is therefore regridded onto the model's grid, while the ordinary
    fine-model-vs-coarse-climatology case keeps the reference as the frame, exactly
    as before. Near-equal grids also keep the reference, the frame nothing has to
    justify.

    Returns ``(target, reason)``; both are recorded in the result's attrs so the
    choice is on paper rather than in someone's memory.
    """
    if target != "auto":
        if target not in ("test", "reference"):
            raise ValueError(
                f"unknown target {target!r}; expected 'auto', 'test' or 'reference'"
            )
        return target, f"target={target!r} as asked"
    test_km = _cell_km(test, _lon_name(test), _lat_name(test))
    reference_km = _cell_km(reference, _lon_name(reference), _lat_name(reference))
    if (
        not np.isfinite(test_km)
        or not np.isfinite(reference_km)
        or test_km <= 0
        or reference_km <= 0
    ):
        return (
            "reference",
            "a lane's cell size could not be measured, so the reference stays the "
            "frame",
        )
    if test_km >= COARSER_BY * reference_km:
        return (
            "test",
            f"the reference's cells (~{reference_km:.3g} km) are materially finer "
            f"than the test's (~{test_km:.3g} km), so the reference is regridded "
            "onto the test's coarser grid",
        )
    return (
        "reference",
        f"the reference's cells (~{reference_km:.3g} km) are not materially finer "
        f"than the test's (~{test_km:.3g} km), so the test is regridded onto the "
        "reference's grid",
    )


def align(
    test,
    reference,
    *,
    method: str = "bilinear",
    convention: Literal["auto", "0-360", "-180-180"] = "auto",
    target: Literal["auto", "test", "reference"] = "auto",
    pad: float = DEFAULT_PAD,
    min_coverage: float = 0.5,
    test_name: str = "test",
    reference_name: str = "reference",
    over: str | None = None,
    time_method: str = "auto",
    tolerance: float | None = None,
    min_overlap: int = MIN_OVERLAP,
    metadata: dict | None = None,
    test_metadata: dict | None = None,
    bin_anchor: str = "auto",
) -> xr.Dataset:
    """Regrid the pair onto one grid — the coarser one — plus their difference.

    Both inputs should be 2-D (lat/lon) DataArrays — select time/depth beforehand.
    Returns a Dataset with ``test``, ``reference`` and ``difference`` (test −
    reference, whichever lane moved) on the **coarser** lane's grid: the reference's
    when the grids are comparable or the reference is coarser (the historical
    behavior), the test's when the reference is materially finer — a 4 km satellite
    product against a coarse model (see :func:`_regrid_target`; the choice and its
    reason are recorded as ``regrid_target``/``regrid_reason`` in the result's
    attrs). ``target="test"`` or ``"reference"`` forces a direction instead, the way
    ``convention=`` forces a longitude convention. When the two lanes already sit on
    exactly the same grid — two runs of the same model, most commonly — nothing is
    regridded at all: no weights are built, ``regrid_target`` reads ``"none"``, and
    the difference is taken directly (see :func:`_shared_grid`).

    ``over`` names one axis that is *allowed to survive* — the axis a caller is going to
    score the pair over, cell by cell (see
    :func:`ocean_skill.metrics.evaluate`). The two lanes are then matched along it first
    (:func:`match_axis`) and regridded after, which is both the correct order and much
    the cheaper one: averaging hourly output into daily bins before the regrid turns
    8760 regridded fields into 365. The match itself lands on whichever lane is
    coarser — the reference's cadence when it is comparable or coarser (the historical
    behavior), the test's when the reference is materially finer, with a warning (see
    :func:`resolve_match_method`). ``time_method``/``tolerance``/``bin_anchor``/
    ``metadata``/``test_metadata`` are its arguments; what it decided, including which
    lane's stamps survived (``match_target``), is recorded in the result's attrs.

    A **station reference** — one position rather than a grid, as a mooring is — has no
    cells to regrid onto, so the test lane is *sampled* at its position instead
    (:func:`_align_at_point`, via :func:`sample_at`). Everything before that point is
    the same for both: the axis matching, the dimensionality check, the units. The
    result has the same three variables on the matched axis, 1-D not 2-D.

    ``method="conservative_normed"`` (or ``"conservative"``) **area-averages** the test
    onto the reference cells, which is the right operator when the test is much finer
    than the reference (a km-scale model against a 1-degree climatology): bilinear would
    *sample* the fine field and discard subgrid structure. Cell corners are derived by
    :func:`grid_of` when absent. Against a station it names how the lane travels to the
    position instead — nearest cell or interpolated.

    ``min_coverage`` drops reference cells that the test only partly covers. Plain
    ``"conservative"`` divides by the *whole* destination cell area, so a half-covered
    coastal or edge cell reads about half its true value — a large, purely artificial
    difference. ``"conservative_normed"`` renormalizes by the covered fraction, and the
    coverage mask then removes cells too sparsely covered to be meaningful.

    ``convention="auto"`` (the default) expresses both lanes in whichever longitude
    convention keeps the *test* contiguous (:func:`natural_convention`): ±180 for
    most domains, 0-360 for one that straddles the antimeridian, as a Pacific model
    does. The resolved choice is recorded as ``lon_convention`` in the result's
    attrs; pass ``"0-360"`` or ``"-180-180"`` to force one instead.
    """
    # Matched *before* the regrid: the binning is what decides how many fields there are
    # to regrid, so doing it after would pay for every step of the finer lane and then
    # throw most of them away.
    report: dict[str, Any] = {}

    # A vertical section has no cells to regrid onto either -- both lanes are already
    # reduced to columns along one shared path (see Comparison's transect route, which
    # prepares the reference at the test lane's own snapped points) -- so this is
    # detected early, the same way a station reference is, and for the same reason: a
    # mismatch or an incompatible over= has to be refused before match_axis/_require_2d
    # apply rules that assume a real grid.
    section_test, section_reference = path_of(test), path_of(reference)
    if (section_test is None) != (section_reference is None):
        raise ValueError(
            "one lane is a vertical section (a select={'transect': ...} result) "
            "and the other is not -- both must sample the same transect path. "
            "This is Comparison's own job to arrange (it prepares the reference "
            "at the test lane's own snapped points); a direct align() call needs "
            "to pass two section lanes or two ordinary ones, not a mix."
        )
    is_section = section_test is not None
    if is_section and over is not None:
        raise ValueError(
            "over= scores a pair cell by cell along one further axis, but a "
            "vertical section already stands on two axes (depth and along-path "
            "distance) -- scoring a third axis per section cell is a follow-up, "
            "not yet built. Drop over= for a section comparison."
        )

    if over is not None:
        test, reference, report = match_axis(
            test,
            reference,
            over=over,
            method=time_method,
            tolerance=tolerance,
            min_overlap=min_overlap,
            metadata=metadata,
            test_metadata=test_metadata,
            bin_anchor=bin_anchor,
        )
        over = str(report.pop("axis", over))

    keep = () if over is None else (over,)
    if not is_section:
        # A profile reference is one cast at one instant, so a time-varying test lane
        # is sampled at that instant before the dimensionality check -- otherwise its
        # leftover time axis (which the vertical score keeps depth, not time, so
        # nothing has reduced) is refused below. A no-op for every other shape (see
        # _sample_test_at_instant).
        test = _sample_test_at_instant(test, reference, over)
        # A section's own shape (vertical + along-path) is validated with its own,
        # more specific messages inside _align_along_path instead -- _require_2d's
        # "beyond its horizontal axes" reading does not apply to it.
        _require_2d(test, "test", keep=keep)
        _require_2d(reference, "reference", keep=keep)

    test = _check_units(test, reference)

    # "auto" follows the *test* lane: a domain straddling the antimeridian forced
    # into ±180 has a bounding box the width of the globe, so the reference below
    # never gets cropped — and its derived cell corners fold, so a conservative
    # regrid paints the test across oceans it never covered (see
    # :func:`natural_convention`). The reference follows the test so both lanes,
    # the bbox and the crop all speak one convention. Safe for a section lane too:
    # harmonize_longitude only re-sorts a longitude that is itself a *dimension*
    # coordinate, and a section's lon rides on `along`, not on its own dimension --
    # the path's order survives untouched.
    if convention == "auto":
        convention = natural_convention(test)
    test = harmonize_longitude(test, convention)
    reference = harmonize_longitude(reference, convention)

    if is_section:
        return _align_along_path(
            test,
            reference,
            convention=convention,
            test_name=test_name,
            reference_name=reference_name,
        )

    # A point reference has no grid to regrid onto: the test lane is *sampled* at the
    # station instead (see _align_at_point). Everything up to here -- the axis matching,
    # the dimensionality check, the units -- is the same question either way, which is
    # why the branch sits here rather than in a second align function.
    station = point_of(reference)
    if station is not None:
        return _align_at_point(
            test,
            reference,
            station,
            method=method,
            convention=convention,
            test_name=test_name,
            reference_name=reference_name,
            over=over,
            report=report,
        )

    # regrid over the overlap, not the reference's full (often global) grid — the
    # reference is the possibly-global lane whichever direction the regrid runs
    reference = subset_to_bbox(reference, bbox_of(test), pad=pad)

    if _shared_grid(test, reference):
        # Two lanes already on one grid -- two runs of the same model, most
        # commonly -- so there is nothing to regrid: building weights to map a
        # field onto the grid it is already on would only interpolate it onto
        # itself, at the cost of the whole regridder machinery. Difference the
        # two fields directly instead -- still renamed to lon/lat like a regridded
        # result, since every downstream reader (the renderers, most visibly)
        # expects those canonical names on the aligned output.
        test_out, reference_out = _as_xesmf(test), _as_xesmf(reference)
        coverage = None
        target, target_reason = (
            "none",
            "the two lanes share one grid (identical lon/lat coordinates), so "
            "nothing was regridded",
        )
    else:
        # the pair lands on the coarser grid, so the finer lane is always the
        # source the regrid area-averages down — see _regrid_target
        target, target_reason = _regrid_target(test, reference, target)
        if target == "test":
            src, tgt = _as_xesmf(reference), _as_xesmf(test)
            src_role, tgt_role = "reference", "test"
        else:
            src, tgt = _as_xesmf(test), _as_xesmf(reference)
            src_role, tgt_role = "test", "reference"
        need_bounds = method.startswith("conservative")
        # memoized on the two grids' own content -- see _regridder_for -- so a fan
        # of comparisons against one fixed reference (compare()'s times= fan, most
        # concretely) builds these weights once rather than once per fanned step
        regridder = _regridder_for(
            grid_of(src, need_bounds), grid_of(tgt, need_bounds), method
        )
        regridded = regridder(src, keep_attrs=True)

        # Coverage = the same regrid applied to a field of ones over the valid
        # source cells: however the direction resolved, it is the finer lane's
        # coverage of the coarser target cells.
        coverage = None
        if min_coverage:
            coverage = _coverage(regridder, src, over, role=src_role)
            regridded = regridded.where(coverage >= min_coverage)

        # the lanes keep their identities whichever one moved: difference stays
        # test − reference, never target − source
        test_out, reference_out = (
            (tgt, regridded) if target == "test" else (regridded, tgt)
        )
    out = xr.Dataset(
        {
            test_name: test_out,
            reference_name: reference_out,
            "difference": test_out - reference_out,
        }
    )
    out["difference"].attrs = {
        "long_name": f"{test_name} − {reference_name}",
        "units": reference_out.attrs.get("units", ""),
    }
    if coverage is not None:
        out["coverage"] = coverage
        out["coverage"].attrs = {
            "long_name": (
                f"fraction of the {tgt_role} cell covered by valid {src_role} data"
            )
        }
    out.attrs["regrid_method"] = method
    out.attrs["regrid_target"] = target
    out.attrs["regrid_reason"] = target_reason
    out.attrs["lon_convention"] = convention
    out.attrs["min_coverage"] = min_coverage
    if over is not None:
        # what the matching decided, and what it cost, recorded beside how the regrid
        # was done -- both are choices a reader of the numbers is entitled to see
        out.attrs["scored_over"] = over
        out.attrs["coverage_time_invariant"] = over not in getattr(coverage, "dims", ())
        out.attrs.update(report)
    return out


def _align_at_point(
    test,
    reference,
    station: tuple[float, float],
    *,
    method: str,
    convention: str,
    test_name: str,
    reference_name: str,
    over: str | None,
    report: dict[str, Any],
) -> xr.Dataset:
    """Pair a gridded test lane with a station reference, on the already-matched axis.

    The point counterpart of the regrid in :func:`align`, and the only step that
    differs: a station has no cells to remap onto, so the test lane is **sampled** at it
    (:func:`sample_at`, ``method="nearest"`` or ``"bilinear"``). Everything else a
    comparison needs has already happened in :func:`align` — the axis matching, the
    dimensionality check, the units — which is why this is a branch there rather than a
    second alignment function with its own copy of them.

    ``method`` is read the same way it is for a grid: it says how the test lane travels
    to the reference. The package default (``"conservative_normed"``) becomes
    ``"nearest"``, since area-averaging onto a station's zero area is not a thing; an
    explicitly conservative method says so rather than being quietly reinterpreted.
    """
    if method.startswith("conservative"):
        # Not a warning when it is the package default -- Comparison passes it on every
        # call without the caller having chosen it, so warning would be noise on the
        # common path. sample_at raises for a conservative method asked for explicitly.
        method = NEAREST
    if point_of(test) is None:
        test = sample_at(
            test,
            *station,
            method=method,
            convention=convention,
            subject="the test lane",
        )
    else:
        # Two point lanes -- a hand-narrowed test, or two moorings. Nothing to sample,
        # but the positions are worth comparing: a select= that picked the wrong cell
        # would otherwise be invisible.
        offset = float(_haversine_km(*point_of(test), *station))
        test.attrs.setdefault("nearest_distance_km", offset)
        if offset > 1.0:
            warnings.warn(
                f"the two lanes are {offset:.1f} km apart: the test lane sits at "
                f"{point_of(test)} and the reference at {station}. They are being "
                "compared as if co-located.",
                stacklevel=_stacklevel.find(),
            )
    _warn_if_depths_differ(test, reference)

    # The two lanes carry different positions -- the station's, and the cell the test
    # came from. Merging them under one name is a MergeError, and dropping the test's
    # loses the offset the metrics report, so the test's are renamed. The station keeps
    # the plain names, the comparison being at the station.
    test = _rename_position(test, test_name)
    out = xr.Dataset(
        {
            test_name: test,
            reference_name: reference,
            "difference": test - reference,
        }
    )
    out["difference"].attrs = {
        "long_name": f"{test_name} − {reference_name}",
        "units": reference.attrs.get("units", ""),
    }
    out.attrs["station_lon"], out.attrs["station_lat"] = station
    out.attrs["point_method"] = test.attrs.get("point_method", method)
    out.attrs["lon_convention"] = convention
    for key in ("nearest_distance_km", "cell_km", "test_time", "time_to_reference_hours"):
        if key in test.attrs:
            out.attrs[key] = test.attrs[key]
    if over is not None:
        out.attrs["scored_over"] = over
        out.attrs.update(report)
    return out


def _along_spacing_km(da) -> float:
    """Return the median along-path column spacing (km), or NaN if unmeasurable."""
    values = np.asarray(da[ALONG_DIM], dtype="float64")
    if values.size < 2:
        return float("nan")
    return float(np.median(np.diff(values)))


def _section_target(test, reference) -> tuple[str, str]:
    """Decide which lane's along-path columns the pair lands on: the coarser one.

    The along-path counterpart of :func:`_regrid_target`, with the same measurement
    and the same hysteresis (:data:`COARSER_BY`): whichever lane's columns are
    spaced farther apart becomes the frame, since averaging the finer lane's
    columns into the coarser one's is what a coarser resolution *means* — aimed
    the other way it would invent along-path structure the coarse lane never had.
    Ties (comparable resolutions — the ordinary model-vs-model case) keep the
    reference, the same default direction :func:`_regrid_target` uses for the same
    reason: it is the frame nothing has to justify.
    """
    test_km, reference_km = _along_spacing_km(test), _along_spacing_km(reference)
    if (
        not np.isfinite(test_km)
        or not np.isfinite(reference_km)
        or test_km <= 0
        or reference_km <= 0
    ):
        return (
            "reference",
            "a lane's column spacing could not be measured, so the reference "
            "stays the frame",
        )
    if test_km >= COARSER_BY * reference_km:
        return (
            "test",
            f"the reference's columns (~{reference_km:.3g} km apart) are "
            f"materially finer than the test's (~{test_km:.3g} km), so the "
            "reference is binned onto the test's coarser columns",
        )
    return (
        "reference",
        f"the reference's columns (~{reference_km:.3g} km apart) are not "
        f"materially finer than the test's (~{test_km:.3g} km), so the test is "
        "binned onto the reference's columns",
    )


def _bin_into_frame(frame, moving, *, frame_lon: str, frame_lat: str):
    """Return ``(frame_trimmed, moving_binned, offsets_km)``.

    ``moving``'s along-path columns are grouped onto whichever of ``frame``'s
    columns they are great-circle-nearest to (:func:`_haversine_km`, periodic in
    longitude, so a path crossing the antimeridian needs no special-casing here),
    then mean-reduced per group (``skipna=True`` — one masked column should not
    blank out a whole bin). A ``frame`` column no ``moving`` column lands near
    simply has no group and is dropped from ``frame_trimmed`` along with it —
    the section shrinks to the columns both lanes actually speak to, rather than
    padding the gap with fabricated data.

    ``offsets_km`` is each ``moving`` column's distance to the ``frame`` column it
    was grouped into, for the caller's own coverage check: a large offset means
    ``moving``'s path wandered away from anything ``frame`` covers, not that
    resolutions merely differ.
    """
    moving_lon_name, moving_lat_name = _lon_name(moving), _lat_name(moving)
    frame_lon_vals = np.asarray(frame[frame_lon], dtype="float64")
    frame_lat_vals = np.asarray(frame[frame_lat], dtype="float64")
    moving_lon_vals = np.asarray(moving[moving_lon_name], dtype="float64")
    moving_lat_vals = np.asarray(moving[moving_lat_name], dtype="float64")

    j = np.array(
        [
            int(np.argmin(_haversine_km(frame_lon_vals, frame_lat_vals, lo, la)))
            for lo, la in zip(moving_lon_vals, moving_lat_vals, strict=True)
        ]
    )
    offsets_km = _haversine_km(frame_lon_vals[j], frame_lat_vals[j], moving_lon_vals, moving_lat_vals)

    label = xr.DataArray(j, dims=ALONG_DIM, name="_osk_section_bin")
    # Coordinates riding on the along dim (this lane's own lon/lat) do not survive
    # a groupby reduction -- xarray drops them rather than guess how to combine
    # them, and averaging lon/lat directly would be dubious near the antimeridian
    # anyway. The frame's own positions are reattached below instead, which is
    # the right answer regardless: once several `moving` columns are averaged
    # into one, they no longer have a single position of their own -- they are
    # now data attached to the frame's column.
    attrs = dict(moving.attrs)
    binned = moving.groupby(label).mean(ALONG_DIM, skipna=True)
    binned.attrs = attrs
    binned = binned.rename({"_osk_section_bin": ALONG_DIM})

    kept = np.asarray(binned[ALONG_DIM].values)
    frame_trimmed = frame.isel({ALONG_DIM: kept})
    binned = binned.assign_coords(
        {
            ALONG_DIM: frame_trimmed[ALONG_DIM].values,
            frame_lon: (ALONG_DIM, np.asarray(frame_trimmed[frame_lon].values)),
            frame_lat: (ALONG_DIM, np.asarray(frame_trimmed[frame_lat].values)),
        }
    )
    return frame_trimmed, binned, offsets_km


def _align_along_path(
    test, reference, *, convention: str, test_name: str, reference_name: str
) -> xr.Dataset:
    """Pair a model section with a reference sampled along the same path.

    The section counterpart of the regrid in :func:`align`: both lanes are
    already reduced to columns along one shared transect (test's own path;
    reference sampled at the test's snapped points — see
    :meth:`ocean_skill.comparison.Comparison.align`'s transect route), so there
    is no 2-D grid to regrid onto and nothing here calls xesmf. What is left is
    two housekeeping steps a real grid regrid does not need: putting both lanes'
    depth lists on one shared ``z`` coordinate, and reconciling their along-path
    columns, which differ even when both were asked for the same points — a
    coarser lane's sampler collapses points that land in the same cell and
    reports its own cells' positions, not the request's (see
    :func:`ocean_skill.transect.sample_along`).
    """
    test_extra = [d for d in test.dims if d not in (ALONG_DIM, "z")]
    if "z" not in test.dims:
        raise ValueError(
            f"the test lane still has its native vertical axis ({sorted(test.dims)}) "
            "-- a section comparison needs fixed depths on both lanes: "
            "select={'transect': ..., 'depth': [50, 200, ...]}."
        )
    if test_extra:
        raise ValueError(
            f"the test lane still has {test_extra} beyond depth and the "
            "along-path axis -- collapse it with aggregate= or narrow it with "
            "select= (most often a surviving time axis)."
        )
    ref_vdim = next((d for d in SECTION_VERTICAL_DIMS if d in reference.dims), None)
    if ref_vdim is None:
        raise ValueError(
            "the reference has no vertical axis along this section -- a "
            "surface-only product cannot be compared against a section, which "
            "needs levels on both sides. Compare at the surface as a map "
            "instead (drop select={'transect': ...})."
        )
    ref_extra = [d for d in reference.dims if d not in (ALONG_DIM, ref_vdim)]
    if ref_extra:
        raise ValueError(
            f"the reference lane still has {ref_extra} beyond depth and the "
            "along-path axis -- collapse it with aggregate= or narrow it with "
            "select=."
        )

    if reference.sizes[ref_vdim] != test.sizes["z"]:
        raise ValueError(
            f"the test lane has {test.sizes['z']} depth(s) and the reference "
            f"has {reference.sizes[ref_vdim]} -- both lanes need the same depth "
            "list, in the same order (select={'depth': [...]})."
        )
    reference_levels = (
        [-float(v) for v in np.asarray(reference["z"])]
        if ref_vdim == "z"
        else [float(v) for v in np.asarray(reference[ref_vdim])]
    )
    if ref_vdim != "z":
        reference = reference.rename({ref_vdim: "z"})
    reference = reference.assign_coords(z=test["z"])

    # Captured before binning replaces each lane's own along coordinate (and so
    # loses whatever attrs rode on it) with the frame's.
    test_path_method = test[ALONG_DIM].attrs.get("path_method", "")
    reference_path_method = reference[ALONG_DIM].attrs.get("path_method", "")

    target, reason = _section_target(test, reference)
    if target == "reference":
        lon_r, lat_r = _lon_name(reference), _lat_name(reference)
        reference, test, offsets_km = _bin_into_frame(
            reference, test, frame_lon=lon_r, frame_lat=lat_r
        )
    else:
        lon_t, lat_t = _lon_name(test), _lat_name(test)
        test, reference, offsets_km = _bin_into_frame(
            test, reference, frame_lon=lon_t, frame_lat=lat_t
        )

    frame = reference if target == "reference" else test
    frame_spacing = _along_spacing_km(frame)
    threshold = max(
        1.5 * (frame_spacing if np.isfinite(frame_spacing) and frame_spacing > 0 else 1.0),
        1.0,
    )
    uncovered = offsets_km > threshold
    if uncovered.any():
        moving_role = test_name if target == "reference" else reference_name
        frame_role = reference_name if target == "reference" else test_name
        along_km = np.asarray(frame[ALONG_DIM])
        lo_km, hi_km = float(along_km.min()), float(along_km.max())
        raise ValueError(
            f"the {frame_role} lane does not cover {int(uncovered.sum())} of the "
            f"{moving_role} lane's columns (roughly km {lo_km:.0f}-{hi_km:.0f} "
            "along the path) -- trim the transect to the region both sources "
            "span, or compare against a reference that covers it."
        )

    # No renaming needed here, unlike the station branch: once several of the
    # moving lane's columns have been averaged into one, they no longer have a
    # position of their own to disambiguate from the frame's -- both lanes
    # already carry the frame's own lon/lat/along (see _bin_into_frame), so the
    # Dataset below merges them as the same coordinate, not a conflicting one.
    out = xr.Dataset(
        {test_name: test, reference_name: reference, "difference": test - reference}
    )
    out["difference"].attrs = {
        "long_name": f"{test_name} − {reference_name}",
        "units": reference.attrs.get("units", ""),
    }
    out.attrs["section_length_km"] = float(np.asarray(frame[ALONG_DIM]).max())
    out.attrs["n_points"] = int(frame.sizes[ALONG_DIM])
    out.attrs["path_method"] = test_path_method
    out.attrs["reference_path_method"] = reference_path_method
    out.attrs["lon_convention"] = convention
    out.attrs["reference_levels"] = reference_levels
    out.attrs["max_column_offset_km"] = float(np.max(offsets_km)) if offsets_km.size else 0.0
    out.attrs["section_target"] = target

    coverage = np.isfinite(test).any("z") & np.isfinite(reference).any("z")
    frac = float(coverage.mean()) if coverage.size else 0.0
    if frac < 0.5:
        warnings.warn(
            f"only {frac:.0%} of the section's columns have any valid data on "
            "both lanes -- likely land gaps in one or both sources, or a "
            "reference that only partly covers the path.",
            stacklevel=_stacklevel.find(),
        )
    return out


def _rename_position(da, prefix: str):
    """Rename a sampled lane's own lon/lat (and cast-instant time) beside the station's.

    lon/lat always: the sampled cell sits at its own position, not the station's, and
    the two must coexist for the metrics to report the offset. Time only when it is a
    *scalar* coordinate -- a profile test sampled at the cast instant
    (:func:`_sample_test_at_instant`) carries a snapshot stamp that differs from the
    reference's cast stamp and would collide under one ``time`` name. A shared time
    *dimension* (the mooring recipe, both lanes on one matched axis) is left untouched:
    renaming it would split the very axis the pair was matched onto.
    """
    renames = {
        name: f"{prefix}_{axis}"
        for name, axis in ((_lon_name(da), "lon"), (_lat_name(da), "lat"))
        if name is not None
    }
    tname = _time_name(da)
    if tname is not None and tname in da.coords and tname not in da.dims:
        renames[tname] = f"{prefix}_time"
    return da.rename(renames) if renames else da


def _warn_if_depths_differ(test, reference) -> None:
    """Say so when a surface-only test lane is being compared against a deep reference.

    This is the step that creates the situation, so this is where it is said: the
    reference lane knows its instrument depth and the test lane knows whether it has a
    vertical axis at all, but only here are both in view. It also closes a silent path —
    ``compare(depths=("surface",))`` reaches a station lane, finds no vertical dimension
    to select from, and correctly does nothing, so asking for the surface and receiving
    30 m used to pass without comment.
    """
    # The coordinate first (via find_coord directly, not resolve_dim -- a station's
    # instrument depth is ordinarily a *scalar* coordinate, which resolve_dim's own
    # contract deliberately excludes, being about indexable dimensions rather than
    # every coordinate an object carries; find_coord has no such restriction, and
    # recognizes an observational product's own spelling -- WHOTS' "DEPTH", say --
    # case-insensitively the same way {"Z": "mean"} finds a real vertical axis).
    # Then the variable's own attrs: a depth riding along `time` does not survive a
    # reduction (resampling a mooring to monthly means drops it), so the attrs are
    # what is left by the time a comparison gets here -- and this caveat matters
    # most for exactly those records, whose instrument moved.
    from ocean_skill.cf import find_coord

    depth = find_coord(reference, "vertical")
    values = None
    if depth is not None:
        values = np.asarray(depth.values, dtype="float64")
    elif reference.attrs.get("depth_m") is not None:
        values = np.asarray([reference.attrs["depth_m"]], dtype="float64")
    if values is None or not np.isfinite(values).any():
        return
    value = float(np.nanmedian(values))
    low, high = reference.attrs.get("depth_range_m", (None, None))
    if low is None and values.size > 1:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    spread = (
        f" (varying {low:g}-{high:g} m)"
        if low is not None and high is not None and high - low > 1.0
        else ""
    )
    if abs(value) <= 5.0:
        return
    if find_coord(test, "vertical") is not None:
        return
    source = reference.attrs.get("depth_source") or "its own metadata"
    warnings.warn(
        f"the reference is at {value:g} m{spread} (from {source}) while the test lane "
        "has no vertical axis, so this compares a subsurface record against a surface "
        "field. "
        "Expect a depth-related bias — use a test source with a vertical axis, or "
        "state the comparison as surface-versus-depth.",
        stacklevel=_stacklevel.find(),
    )
