"""Alignment: bring the ``test`` entity onto the ``reference`` grid/coordinates.

Direction is set by role (not argument order): test → reference (⇒ model→data by
default), with an ``onto`` override. Longitude conventions are harmonized first —
a 0-360 model vs a ±180 observational grid otherwise produces silently empty overlap.
The reference is subset to the test's bounding box so we regrid onto the overlap
rather than a whole globe. The result is **always xarray**, so one metrics engine
serves both gridded and point comparisons.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import xarray as xr

from ocean_skill import _stacklevel
from ocean_skill.cf import find_coord

__all__ = [
    "align",
    "axis_edges",
    "grid_of",
    "harmonize_longitude",
    "is_composite",
    "match_axis",
    "natural_convention",
    "perimeter_of",
    "point_of",
    "resolve_match_method",
    "sample_at",
    "subset_to_bbox",
]

#: Sampling methods :func:`sample_at` understands, and what each means at a point.
#: ``nearest`` takes the containing cell's own value; the interpolating spellings weight
#: the surrounding cells. Anything else (a conservative regrid) has no meaning against a
#: zero-area target — see :func:`sample_at`.
NEAREST = "nearest"
_INTERPOLATING = ("bilinear", "linear")


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
    contiguous in. Ties (a truly global field) keep ±180, the maps' usual frame.
    """
    lon = _lon_name(obj)
    if lon is None:
        return "-180-180"
    vals = np.asarray(obj[lon], dtype="float64").ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "-180-180"
    span_180 = np.ptp(((vals + 180.0) % 360.0) - 180.0)
    span_360 = np.ptp(vals % 360.0)
    return "0-360" if span_360 < span_180 else "-180-180"


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
    """
    from ocean_skill.operators import oriented_slice

    lon, lat = _lon_name(obj), _lat_name(obj)
    if lon is None or lat is None:
        return obj
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
    tolerance: float | None = None,
    calendar: bool = True,
) -> tuple[str, str, float | None]:
    """Decide how the test's axis should be brought onto the reference's.

    The temporal counterpart of choosing a regrid method, and settled the same way: the
    reference is the frame, the test is what moves, and how it moves depends on which is
    coarser. A materially finer test is **averaged into** the reference's bins — the
    analogue of ``conservative_normed``, and the reason it is a default rather than
    something to ask for: nobody has to request area-averaging in space either. Against
    an instantaneous reference the test is **sampled** at the nearest step instead, the
    analogue of ``bilinear``.

    A reference finer than the test is refused rather than resolved. Coarsening the
    reference would alter the thing being scored against, and this package never touches
    the reference silently.

    Returns ``(method, reason, tolerance)``; the reason is recorded in the aligned
    result's attrs so the choice is on paper rather than in someone's memory.
    """
    ct, cr = _cadence(test_values), _cadence(reference_values)
    if ct is None or cr is None:
        # one lane is a single step: there is nothing to bin, only something to pair
        known = ct or cr
        if known is None:
            return "exact", "neither lane has a measurable cadence", None
        return (
            "nearest",
            "one lane has a single step, so its counterpart is sampled at it",
            tolerance if tolerance is not None else NEAREST_TOLERANCE_FRACTION * known,
        )
    if cr * COARSER_BY <= ct:
        raise ValueError(
            f"the reference steps every {_duration(cr, calendar)} and the test every "
            f"{_duration(ct, calendar)}, so the reference is the finer of the two. "
            "Averaging it down to the test's cadence would change what is being "
            "scored against, which is not something this will do on its own. Coarsen "
            'reference deliberately (aggregate={"time": {"resample": "'
            f'{_pandas_freq(ct)}", "reduce": "mean"}}) or swap the roles so the finer '
            "product is the test."
        )
    if ct * COARSER_BY <= cr:
        if composite is False:
            return (
                "nearest",
                f"the reference is an instantaneous product every "
                f"{_duration(cr, calendar)}; the test steps every "
                f"{_duration(ct, calendar)} and is sampled at those instants",
                tolerance if tolerance is not None else NEAREST_TOLERANCE_FRACTION * cr,
            )
        return (
            "mean",
            f"the test steps every {_duration(ct, calendar)} and the reference every "
            f"{_duration(cr, calendar)}, so the test is averaged into its bins",
            tolerance,
        )
    return (
        "nearest",
        f"both lanes step about every {_duration(cr, calendar)}, so their steps are "
        "paired rather than rebinned",
        tolerance
        if tolerance is not None
        else NEAREST_TOLERANCE_FRACTION * max(ct, cr),
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


def match_axis(
    test,
    reference,
    *,
    over: str,
    method: str = "auto",
    tolerance: float | None = None,
    min_overlap: int = MIN_OVERLAP,
    metadata: dict | None = None,
    bin_anchor: str = "auto",
):
    """Bring the test's ``over`` axis onto the reference's, and report what it took.

    Direction is the same rule alignment follows everywhere here: **test → reference**.
    The reference's axis is the frame and nothing is done to it, except that steps it
    ends up with no test data for are dropped — an all-NaN step contributes nothing to a
    metric and would make the difference field say something untrue about it.

    Returns ``(test, reference, report)`` with both lanes on one axis, named as the
    reference names it. Everything measured for the report is measured on the
    *coordinate*, never by walking the data: the counts are index arithmetic and stay
    free even when the lanes are a year of daily maps.
    """
    from ocean_skill.operators import resolve_dim

    tdim, rdim = resolve_dim(test, over), resolve_dim(reference, over)
    for role, dim, lane in (("test", tdim, test), ("reference", rdim, reference)):
        if dim is None or dim not in lane.dims:
            raise ValueError(
                f"the {role} lane has no {over!r} axis to score over (its dimensions "
                f"are {list(lane.dims)}). For a comparison of single maps, leave over= "
                "unset."
            )
    test, reference = _sorted_on(test, tdim), _sorted_on(reference, rdim)
    tf = _axis_floats(test, tdim, "test")
    rf = _axis_floats(reference, rdim, "reference")
    # captured before matching: a failure to match empties the lanes, and the spans are
    # exactly what the message about it has to say
    spans = (_span(test, tdim), _span(reference, rdim))

    reason = f"method={method!r} as asked"
    if method == "auto":
        method, reason, tolerance = resolve_match_method(
            tf,
            rf,
            composite=is_composite(reference, rdim, metadata),
            tolerance=tolerance,
        )
        if is_composite(reference, rdim, metadata) is None and method == "mean":
            warnings.warn(
                "nothing on the reference says whether its steps are period averages "
                "or instantaneous — no CF cell_methods on the variable, no 'period' in "
                "its catalog entry — so it is taken to be a composite and the test is "
                f"averaged into its bins ({reason}). Pass time_method='nearest' if the "
                "reference is really a series of snapshots, or give the catalog entry "
                "a period (or the variable a cell_methods) to settle it for good.",
                stacklevel=_stacklevel.find(),
            )

    report: dict[str, Any] = {"match_method": method, "match_reason": reason}
    if tolerance is not None:
        report["match_tolerance"] = float(tolerance)

    if method == "mean":
        test, reference, extra = _match_by_mean(
            test, reference, tdim, rdim, tf, rf, bin_anchor
        )
        report.update(extra)
    elif method == "nearest":
        test, reference, extra = _match_by_nearest(
            test, reference, tdim, rdim, tf, rf, tolerance
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
    # both lanes now name the axis as the reference names it -- test -> reference again
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


def _match_by_mean(test, reference, tdim, rdim, tf, rf, bin_anchor):
    """Average the test's steps into the reference's bins."""
    if bin_anchor == "auto":
        bin_anchor = infer_bin_anchor(reference[rdim].values)
    edges = axis_edges(rf, anchor=bin_anchor)

    which = np.searchsorted(edges, tf, side="right") - 1
    inside = (which >= 0) & (which < rf.size)
    stamps = np.asarray(reference[rdim].values)
    if not inside.any():
        # nothing to group: hand back empty lanes, and match_axis raises with the spans
        empty = test.isel({tdim: []})
        if tdim != rdim:
            empty = empty.rename({tdim: rdim})
        return (
            empty,
            reference.isel({rdim: []}),
            {"bin_anchor": bin_anchor, "steps_outside_bins": int(tf.size)},
        )
    label = xr.DataArray(stamps[which[inside]], dims=(tdim,), name="_osk_bin")
    attrs = dict(test.attrs)
    grouped = test.isel({tdim: inside}).groupby(label).mean(tdim)
    # a reduction drops attrs, and `units` has to survive: align() checks it next
    grouped.attrs = attrs
    grouped = grouped.rename({"_osk_bin": rdim})
    filled = np.asarray(grouped[rdim].values)
    reference = reference.sel({rdim: filled})

    counts = np.bincount(which[inside], minlength=rf.size)
    typical = float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0
    from ocean_skill.operators import SHORT_BIN_FRACTION

    short = int(((counts > 0) & (counts < SHORT_BIN_FRACTION * typical)).sum())
    empty = int((counts == 0).sum())
    if empty:
        warnings.warn(
            f"{empty} of the reference's {rf.size} steps had no test data in their bin "
            "and were dropped. An all-NaN step scores nothing and would make the "
            "difference field claim otherwise.",
            stacklevel=_stacklevel.find(),
        )
    if short:
        warnings.warn(
            f"{short} of the reference's bins caught fewer than "
            f"{SHORT_BIN_FRACTION:.0%} of the usual {typical:g} test steps, so those "
            "steps are averages over part of a period labelled like a whole one — "
            "usually the first and last bin of the selection. Narrow select= to whole "
            "periods to drop them.",
            stacklevel=_stacklevel.find(),
        )
    return (
        grouped,
        reference,
        {
            "bin_anchor": bin_anchor,
            "steps_outside_bins": int((~inside).sum()),
            "bins_empty": empty,
            "bins_short": short,
            "steps_per_bin": typical,
        },
    )


def _match_by_nearest(test, reference, tdim, rdim, tf, rf, tolerance, calendar=True):
    """Pair each reference step with the nearest test step within ``tolerance``."""
    import pandas as pd

    index = pd.Index(tf)
    pos = index.get_indexer(
        rf, method="nearest", **({"tolerance": tolerance} if tolerance else {})
    )
    keep = pos >= 0
    test = test.isel({tdim: pos[keep]})
    reference = reference.isel({rdim: keep})
    offsets = np.abs(tf[pos[keep]] - rf[keep]) if keep.any() else np.empty(0)
    # the reference's own stamps become the shared axis: test -> reference, here too
    attrs = dict(test.attrs)
    test = test.assign_coords({tdim: np.asarray(reference[rdim].values)})
    if tdim != rdim:
        test = test.rename({tdim: rdim})
    test.attrs = attrs
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
            f"{unmatched} reference steps had no test step within "
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


def _coverage(regridder, src, over: str | None):
    """Regrid a field of ones over the test's valid cells: the fraction covered.

    With a surviving axis this is one extra regrid *per step*, which is worth avoiding
    when it buys nothing. A model's missing cells are its land mask, which does not
    move, so the mask is checked for invariance along ``over`` — exactly, with
    ``any == all``, not assumed — and collapsed to one step when it holds. The 2-D
    result broadcasts back over every step, which is what a static mask means. A mask
    that genuinely does move pays for the per-step version and says so.
    """
    finite = np.isfinite(src)
    if over is not None and over in finite.dims:
        if src.chunks is None and bool((finite.any(over) == finite.all(over)).all()):
            one = finite.isel({over: 0}, drop=True)
            return regridder(xr.ones_like(src.isel({over: 0}, drop=True)).where(one))
        warnings.warn(
            f"the test's valid cells change along {over!r}, so coverage is computed "
            f"for every one of its {src.sizes[over]} steps rather than once. Expected "
            "of a field with a moving mask; surprising for a model land mask.",
            stacklevel=_stacklevel.find(),
        )
    return regridder(xr.ones_like(src).where(finite))


def align(
    test,
    reference,
    *,
    method: str = "bilinear",
    convention: Literal["auto", "0-360", "-180-180"] = "auto",
    pad: float = DEFAULT_PAD,
    min_coverage: float = 0.5,
    test_name: str = "test",
    reference_name: str = "reference",
    over: str | None = None,
    time_method: str = "auto",
    tolerance: float | None = None,
    min_overlap: int = MIN_OVERLAP,
    metadata: dict | None = None,
    bin_anchor: str = "auto",
) -> xr.Dataset:
    """Regrid ``test`` onto ``reference``'s grid; return both plus their difference.

    Both inputs should be 2-D (lat/lon) DataArrays — select time/depth beforehand.
    Returns a Dataset with ``test``, ``reference`` and ``difference`` (test − reference)
    on the reference grid.

    ``over`` names one axis that is *allowed to survive* — the axis a caller is going to
    score the pair over, cell by cell (see
    :func:`ocean_skill.metrics.evaluate`). The two lanes are then matched along it first
    (:func:`match_axis`) and regridded after, which is both the correct order and much
    the cheaper one: averaging hourly output into daily bins before the regrid turns
    8760 regridded fields into 365. ``time_method``/``tolerance``/``bin_anchor``/
    ``metadata`` are its arguments; what it decided is recorded in the result's attrs.

    A **station reference** — one position rather than a grid, as a mooring is — has no
    cells to regrid onto, so the test lane is *sampled* at its position instead
    (:func:`_align_at_point`, via :func:`sample_at`). Everything before that point is
    the same for both: the axis matching, the dimensionality check, the units. The
    result has the same three variables on the reference's own axis, 1-D not 2-D.

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
    import xesmf as xe

    # Matched *before* the regrid: the binning is what decides how many fields there are
    # to regrid, so doing it after would pay for every step of the finer lane and then
    # throw most of them away.
    report: dict[str, Any] = {}
    if over is not None:
        test, reference, report = match_axis(
            test,
            reference,
            over=over,
            method=time_method,
            tolerance=tolerance,
            min_overlap=min_overlap,
            metadata=metadata,
            bin_anchor=bin_anchor,
        )
        over = str(report.pop("axis", over))

    keep = () if over is None else (over,)
    _require_2d(test, "test", keep=keep)
    _require_2d(reference, "reference", keep=keep)

    test = _check_units(test, reference)

    # "auto" follows the *test* lane: a domain straddling the antimeridian forced
    # into ±180 has a bounding box the width of the globe, so the reference below
    # never gets cropped — and its derived cell corners fold, so a conservative
    # regrid paints the test across oceans it never covered (see
    # :func:`natural_convention`). The reference follows the test so both lanes,
    # the bbox and the crop all speak one convention.
    if convention == "auto":
        convention = natural_convention(test)
    test = harmonize_longitude(test, convention)
    reference = harmonize_longitude(reference, convention)

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

    # regrid onto the overlap, not the reference's full (often global) grid
    reference = subset_to_bbox(reference, bbox_of(test), pad=pad)

    src, tgt = _as_xesmf(test), _as_xesmf(reference)
    need_bounds = method.startswith("conservative")
    regridder = xe.Regridder(
        grid_of(src, need_bounds),
        grid_of(tgt, need_bounds),
        method,
        unmapped_to_nan=True,
    )
    regridded = regridder(src, keep_attrs=True)

    # Coverage = the same regrid applied to a field of ones over the valid test cells.
    coverage = None
    if min_coverage:
        coverage = _coverage(regridder, src, over)
        regridded = regridded.where(coverage >= min_coverage)

    out = xr.Dataset(
        {
            test_name: regridded,
            reference_name: tgt,
            "difference": regridded - tgt,
        }
    )
    out["difference"].attrs = {
        "long_name": f"{test_name} − {reference_name}",
        "units": tgt.attrs.get("units", ""),
    }
    if coverage is not None:
        out["coverage"] = coverage
        out["coverage"].attrs = {
            "long_name": "fraction of the reference cell covered by valid test data"
        }
    out.attrs["regrid_method"] = method
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
    """Pair a gridded test lane with a station reference, on the reference's own axis.

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
    for key in ("nearest_distance_km", "cell_km"):
        if key in test.attrs:
            out.attrs[key] = test.attrs[key]
    if over is not None:
        out.attrs["scored_over"] = over
        out.attrs.update(report)
    return out


def _rename_position(da, prefix: str):
    """Rename a sampled lane's own lon/lat so they can sit beside the station's."""
    renames = {
        name: f"{prefix}_{axis}"
        for name, axis in ((_lon_name(da), "lon"), (_lat_name(da), "lat"))
        if name is not None
    }
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
    # The coordinate first, then the variable's own attrs: a depth riding along `time`
    # does not survive a reduction (resampling a mooring to monthly means drops it), so
    # the attrs are what is left by the time a comparison gets here -- and this caveat
    # matters most for exactly those records, whose instrument moved.
    depth = reference.coords.get("depth")
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
    if any(name in test.coords for name in ("depth", "z", "z_rho", "lev")):
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
