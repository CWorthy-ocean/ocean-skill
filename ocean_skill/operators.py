"""Combining variables and reducing dimensions, as data rather than code.

Two composable layers, both declarative so a YAML suite expresses exactly what the
Python API does:

**Variable resolution** — how the field is obtained. A plain name, a *combination* of
several (MARBL carries chlorophyll as ``spChl``/``diatChl``/``diazChl``; the total is
their sum), or a registered derived diagnostic.

**Selection** — which part of an axis to keep (:func:`select`), e.g.
``{"time": "2012-01"}``. Narrows a dimension; does not remove it.

**Aggregation** — how dimensions collapse. ``{"time": "mean"}``, a climatology
``{"time": {"groupby": "month", "reduce": "mean"}}``, consecutive periods
``{"time": {"resample": "1MS", "reduce": "mean"}}``, a percentile
``{"time": {"reduce": "quantile", "q": 0.9}}``.

``groupby`` and ``resample`` are the two ways to keep an axis standing rather than
collapse it, and they are not the same axis: ``groupby`` bins by *label*, giving a
climatology (every January of the record averaged into one field), while ``resample``
bins by *interval*, giving consecutive periods (January 2012, February 2012, ...).
On a single-year run they coincide, which is exactly why they are spelled
differently — and why the result carries a different dimension either way, so
nothing downstream has to be told which was used.

The design goal is that adding an operator costs no code. Two things make that work:

1. **Reductions are already xarray methods.** :func:`aggregate` ends in
   ``getattr(da, name)(dim, **kwargs)``, so ``mean``/``sum``/``std``/``min``/``max``/
   ``median``/``var``/``quantile``/``count``/``integrate`` all work with no entry
   anywhere. :data:`REDUCERS` exists only for the rare reduction that is not a method.
2. **Combination is arithmetic.** :data:`COMBINERS` is four entries from the stdlib
   ``operator`` module, not four functions.

Only a genuinely custom diagnostic (MLD, EKE, geostrophic speed) needs a registered
function, because there a real formula is the irreducible content.
"""

from __future__ import annotations

import functools
import numbers
import operator
import warnings
from typing import Any

__all__ = [
    "CALCULATORS",
    "CALCULATOR_INPUTS",
    "COMBINERS",
    "DERIVED",
    "REDUCERS",
    "TIME_GROUPBY_ATTR",
    "aggregate",
    "box_in_spec",
    "combine",
    "oriented_slice",
    "point_in_spec",
    "register_calculator",
    "register_derived",
    "register_reducer",
    "resolve_dim",
    "resolve_variable",
    "select",
    "spatial_mean_in_spec",
    "spec_names",
    "time_axis_dim",
]

#: How a list of variables becomes one field. Straight from the stdlib — a new
#: combiner is a dict entry, not a function.
COMBINERS: dict[str, Any] = {
    "sum": operator.add,
    "product": operator.mul,
    "difference": operator.sub,
    "ratio": operator.truediv,
}

#: Named variable specs, so a recurring combination is written once and then used by
#: name (``variable="total_chlorophyll"``). Data, not code: adding one is a line.
DERIVED: dict[str, dict[str, Any]] = {
    # MARBL splits chlorophyll across phytoplankton functional types; CF's
    # mass_concentration_of_chlorophyll_a_in_sea_water means the total, which is why
    # aliasing any single component to it (as a vocabulary alias) would be wrong.
    "total_chlorophyll": {
        "sum": ["spChl", "diatChl", "diazChl"],
        "standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
    },
}

#: Reductions that are *not* xarray methods. Deliberately near-empty — anything
#: xarray can already do needs no entry, and putting one here would only add a name
#: to keep in sync. Signature: ``fn(da, dim, **kwargs) -> DataArray``.
REDUCERS: dict[str, Any] = {}


def register_reducer(name: str):
    """Register a reduction xarray has no method for. Use as a decorator."""

    def decorate(fn):
        REDUCERS[name] = fn
        return fn

    return decorate


#: Registered derived diagnostics -- the ``{"calculate": name}`` half of a variable
#: spec (see :func:`resolve_variable`). Empty until something registers into it
#: (:mod:`ocean_skill.mld` does, at import time), which is why ``{"calculate": "mld"}``
#: still raises "register a calculator first" if that module is never imported.
#: Signature: ``fn(ds, **kwargs) -> DataArray``.
CALCULATORS: dict[str, Any] = {}

#: A calculator's own report of which standard_names it needs, as a function of its
#: spec (minus ``calculate``/``standard_name``) -- e.g. MLD needs temperature *and*
#: salinity for a density criterion but temperature alone for a temperature one, which
#: :func:`spec_names` cannot know without asking. Optional: a calculator that skips
#: this is simply invisible to catalog pre-filtering, exactly as before it registered.
#: Signature: ``fn(spec: dict) -> list[list[str]]``, the same shape :func:`spec_names`
#: returns for everything else.
CALCULATOR_INPUTS: dict[str, Any] = {}


def register_calculator(name: str, *, inputs: Any = None):
    """Register a derived diagnostic -- a real formula, not an operator dispatch.

    Unlike :data:`COMBINERS`/:data:`REDUCERS`, a calculator needs the whole dataset
    (not just the arrays a spec names), because a genuine formula like MLD reads
    variables the spec never mentions (temperature *and* salinity for a density
    criterion) and needs coordinates :func:`resolve_variable` does not have (the
    depth axis). Use as a decorator; pass ``inputs=`` to register into
    :data:`CALCULATOR_INPUTS` at the same time.

    This is public runtime API, not an internal detail: any function can be plugged
    in this way from a notebook, with no codebase change --

    >>> @register_calculator("eke")
    ... def eddy_kinetic_energy(ds, **kwargs):
    ...     ...
    ...     return da
    >>> osk.field("GOM_bgc", {"calculate": "eke"})

    A spec stays data even so (a name plus keyword arguments, not the function
    itself), which is what lets it serialize into a cache key and survive a round
    trip through YAML -- see :mod:`ocean_skill.mld` for a complete example.
    """

    def decorate(fn):
        CALCULATORS[name] = fn
        if inputs is not None:
            CALCULATOR_INPUTS[name] = inputs
        return fn

    return decorate


def register_derived(name: str, spec: dict[str, Any]) -> None:
    """Register a named variable spec, e.g. a combination used across many runs."""
    DERIVED[name] = dict(spec)


def _units_of(da) -> str:
    return str(da.attrs.get("units", "")).strip()


def combine(arrays: list, how: str, *, name: str | None = None):
    """Combine ``arrays`` into one field with the :data:`COMBINERS` operator ``how``.

    Refuses to combine fields whose ``units`` disagree. xarray will happily add a
    mg/m3 field to a mmol/m3 one and label the result with neither; for a sum of
    phytoplankton components a silent unit mismatch is exactly the kind of wrong
    number that looks right on a map.
    """
    if how not in COMBINERS:
        raise KeyError(f"unknown combiner {how!r}; known: {sorted(COMBINERS)}")
    if not arrays:
        raise ValueError(f"nothing to {how}")

    units = {_units_of(a) for a in arrays}
    if len(units) > 1:
        raise ValueError(
            f"cannot {how} fields with different units {sorted(units)}: "
            f"{[str(a.name) for a in arrays]}. Convert them first."
        )

    out = functools.reduce(COMBINERS[how], arrays)
    out.attrs = dict(arrays[0].attrs)  # arithmetic drops attrs; units are the point
    if name:
        out = out.rename(name)
    return out


def resolve_variable(ds, spec: Any):
    """Return the field ``spec`` describes, or ``None`` if it is not in ``ds``.

    ``spec`` is one of:

    - a name — anything :func:`ocean_skill.units.find_variable` accepts (short
      vocabulary key, canonical standard_name, alias, any case), including a key of
      :data:`DERIVED`, which expands to its stored spec;
    - ``{"<combiner>": [names]}`` — e.g. ``{"sum": ["spChl", "diatChl", "diazChl"]}``,
      optionally with ``standard_name`` naming the result;
    - ``{"calculate": name}`` — a registered derived diagnostic (:data:`CALCULATORS`;
      e.g. ``{"calculate": "mld", "method": "density_threshold"}``, see
      :mod:`ocean_skill.mld`), given the whole dataset rather than named arrays,
      because a genuine formula needs variables and coordinates the spec never
      mentions.

    **A combination falls back to its ``standard_name``.** The two sides of a
    comparison rarely store a quantity the same way: MARBL splits chlorophyll into
    ``spChl``/``diatChl``/``diazChl``, while MODIS ships the total under one CF name.
    One spec has to serve both lanes, so when a source has *none* of the components
    it is taken to use the other convention and the ``standard_name`` is looked up
    directly.

    Having *some* components but not all is different — that is a typo or truncated
    output, not a convention — so it warns loudly rather than quietly returning
    either a partial sum (three-quarters of a total is not a smaller total, it is a
    wrong one) or the total from a source that clearly meant to provide parts.
    """
    from ocean_skill.units import find_variable

    if isinstance(spec, str):
        if spec in DERIVED:
            return resolve_variable(ds, DERIVED[spec])
        return find_variable(ds, spec)

    if not isinstance(spec, dict):
        raise TypeError(f"variable spec must be a name or a dict, got {type(spec)}")

    spec = dict(spec)
    standard_name = spec.pop("standard_name", None)
    if "calculate" in spec:
        spec = dict(spec)
        name = spec.pop("calculate")
        if name not in CALCULATORS:
            raise NotImplementedError(
                f"derived diagnostic {name!r}: register a calculator first"
            )
        out = CALCULATORS[name](ds, **spec)
        if standard_name:
            out = out.rename(standard_name)
            out.attrs["standard_name"] = standard_name
        return out

    how, names = next(iter(spec.items()))
    # Recurse, so a component may itself be a combination -- ``{"ratio":
    # [{"sum": ["a", "b"]}, "c"]}`` is ``(a + b) / c``. Without this, anything
    # beyond one flat operation would need a registered function, which is a lot of
    # ceremony for ordinary arithmetic.
    arrays = [resolve_variable(ds, n) for n in names]
    missing = [str(n) for n, a in zip(names, arrays, strict=True) if a is None]

    if missing:
        if len(missing) < len(names):
            # `is not None`, not truthiness: bool() on a DataArray is ambiguous.
            present = sorted(
                str(n) for n, a in zip(names, arrays, strict=True) if a is not None
            )
            detail = (
                f"falling back to {standard_name!r}"
                if standard_name
                else "and no standard_name to fall back to"
            )
            warnings.warn(
                f"{sorted(missing)} missing but {present} present, so this source "
                f"cannot supply the {how} as written -- check the component names; "
                f"{detail}.",
                stacklevel=2,
            )
        return find_variable(ds, standard_name) if standard_name else None

    out = combine(arrays, how, name=standard_name)
    if standard_name:
        out.attrs["standard_name"] = standard_name
    return out


#: cf-xarray's canonical axis/coordinate names -> the ``find_coord`` kind that
#: resolves them against whatever a given dataset actually calls that axis. Accepting
#: both spellings ("Z" and "vertical") because cf-xarray itself does.
_CF_AXES: dict[str, str] = {
    "X": "longitude",
    "longitude": "longitude",
    "Y": "latitude",
    "latitude": "latitude",
    "Z": "vertical",
    "vertical": "vertical",
    "T": "time",
    "time": "time",
}


def resolve_dim(obj, name: str) -> str | None:
    """Return the dimension of ``obj`` that ``name`` refers to, or ``None``.

    ``name`` may be a cf-xarray axis (``"Z"``, ``"T"``, ``"X"``, ``"Y"``) or
    coordinate (``"vertical"``, ``"time"``, ...), which resolves through
    :func:`ocean_skill.cf.find_coord` — cf-xarray first, then a name-fallback
    list. That matters because the same axis is called different things by every
    product: after a ROMS vertical transform it is ``z``, WOA calls it ``depth``,
    GLODAP ``depth_surface``. A raw dimension name is returned as-is, so nothing
    that worked before stops working.

    ``None`` means "this object has no such axis", which callers treat as "skip",
    since one spec is routinely shared across variables with different axes.
    """
    if name in obj.dims:
        return name
    kind = _CF_AXES.get(name)
    if kind is None:
        return name if name in getattr(obj, "coords", ()) else None
    from ocean_skill.cf import find_coord
    from ocean_skill.vocabulary import COORD_FALLBACKS

    found = find_coord(obj, kind)
    # The coordinate may be multi-dimensional (curvilinear lon/lat); only a
    # 1-D coordinate names a dimension that can be selected or reduced.
    if found is not None and found.name in obj.dims:
        return str(found.name)
    # ROMS output carries no coordinate variables at all, so its axes are bare
    # dimensions that find_coord cannot see -- `s_rho` is a dimension with nothing
    # attached. Match the fallback names against the dimensions themselves, or
    # `{"Z": "mean"}` silently does nothing on exactly the model this exists for.
    # Case-insensitively: a source's own spelling (WHOTS' `DEPTH`, say) is one a
    # builder chose, not one this list can enumerate every capitalization of.
    dims_by_lower = {str(d).lower(): str(d) for d in obj.dims}
    for candidate in COORD_FALLBACKS[kind]:
        if candidate in dims_by_lower:
            return dims_by_lower[candidate]
    return None


def time_axis_dim(obj) -> str | None:
    """Return the dimension that plays time's role, including a time groupby's.

    :func:`resolve_dim(obj, "T") <resolve_dim>` widened one step: a
    ``{"time": {"groupby": "month", "reduce": "mean"}}`` climatology renames the
    time dimension to ``month`` (or ``year``, ``hour``, ...), a bare integer index
    with no coordinate attributes of its own -- nothing cf-xarray or a name
    fallback can recognize as time. :func:`~ocean_skill.operators._reduce_dim`
    marks exactly that case with :data:`TIME_GROUPBY_ATTR` on the new coordinate,
    and this is where the mark is read back, so a plot can keep treating "time
    standing" as one shape rather than two.

    Deliberately scoped to the plot path (:mod:`ocean_skill.field`,
    :mod:`ocean_skill.plot.time_depth`) rather than folded into :func:`resolve_dim`
    itself: :func:`aggregate` and a deferred post-aggregate :func:`select` both
    resolve ``"time"``/``"T"`` against the *original* axis on purpose (a second
    ``"T"`` key in the same spec must not silently collapse a climatology this
    function would otherwise make it match), and neither a comparison's own time
    bookkeeping nor :func:`~ocean_skill.extrema` need or want an integer groupby
    dim answering to "time".

    A season groupby is deliberately never marked -- see
    :mod:`ocean_skill.plot.profile`'s season fan -- so this never returns a
    ``"season"`` dim.
    """
    dim = resolve_dim(obj, "T")
    if dim is not None and dim in obj.dims:
        return dim
    for d in obj.dims:
        coord = obj.coords.get(d)
        if coord is not None and TIME_GROUPBY_ATTR in coord.attrs:
            return str(d)
    return None


#: Spec key spellings that name the horizontal axes -- the union of
#: :data:`_CF_AXES`'s "X"/"Y" and every name :func:`ocean_skill.cf.find_coord`'s
#: fallback list resolves, so a request is recognized under whichever spelling a
#: source or a caller happens to use. Shared by :func:`point_in_spec` (both scalar
#: -- a place), :func:`box_in_spec` (both a range -- a box), and
#: :func:`spatial_mean_in_spec` (both a plain "mean" in an ``aggregate`` -- one
#: area-weighted reduction over both axes together).
_POINT_LON_KEYS = frozenset({"X", "lon", "longitude", "lon_rho", "nav_lon"})
_POINT_LAT_KEYS = frozenset({"Y", "lat", "latitude", "lat_rho", "nav_lat"})


def _point_scalar(value: Any) -> float | None:
    """Return ``value`` as a float when it is a real scalar, else ``None``.

    Bools are excluded even though ``bool`` is a subclass of ``int`` -- a
    ``lat=True`` typo should not silently pass as a genuine coordinate value.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def point_in_spec(spec: dict[str, Any] | None) -> tuple[str, str, float, float] | None:
    """Return ``(lon_key, lat_key, lon, lat)`` when ``spec`` pins one position.

    Both horizontal axes have to be named *and* scalar for this to be a point: a
    slice, a list, a range dict, or a string narrows an axis without collapsing it
    to one place, and a lone ``lon`` with no ``lat`` (or vice versa) is a
    meridional/zonal slice, not a point. Used both by :func:`select` to route a
    point request through :func:`ocean_skill.align.sample_at`, and by
    :func:`ocean_skill.comparison.Comparison` to infer ``over="time"`` when a
    select already narrows the reference to a place.
    """
    if not spec:
        return None
    lon_key = next((k for k in _POINT_LON_KEYS if k in spec), None)
    lat_key = next((k for k in _POINT_LAT_KEYS if k in spec), None)
    if lon_key is None or lat_key is None:
        return None
    lon, lat = _point_scalar(spec[lon_key]), _point_scalar(spec[lat_key])
    if lon is None or lat is None:
        return None
    return lon_key, lat_key, lon, lat


def _point_selectable(
    obj, spec: dict[str, Any] | None
) -> tuple[str, str, float, float] | None:
    """Return :func:`point_in_spec`'s hit when ``obj`` itself can be sampled at it.

    Narrower than ``point_in_spec`` alone: a Dataset has no one array to sample
    (:func:`ocean_skill.comparison._select_horizontal_then_aggregate` calls this
    per-variable, after :func:`resolve_variable` has already picked one out), and a
    spec's lon/lat keys mean nothing unless ``obj`` actually carries a
    longitude/latitude coordinate that is either rectilinear (each 1-D and its own
    dimension) or curvilinear (each 2-D, e.g. ROMS's ``lon_rho``/``lat_rho``). A
    trajectory's ``lon(time)``/``lat(time)`` is neither, and is left to the
    ordinary per-axis loop in :func:`select` -- :func:`ocean_skill.align.sample_at`'s
    curvilinear branch searches a 2-D grid, not a 1-D path through one.
    """
    hit = point_in_spec(spec)
    if hit is None or hasattr(obj, "data_vars"):
        return None
    from ocean_skill.cf import find_coord

    lon_coord, lat_coord = find_coord(obj, "longitude"), find_coord(obj, "latitude")
    if lon_coord is None or lat_coord is None:
        return None
    rectilinear = (
        lon_coord.ndim == 1
        and lat_coord.ndim == 1
        and lon_coord.name in obj.dims
        and lat_coord.name in obj.dims
    )
    curvilinear = lon_coord.ndim == 2 and lat_coord.ndim == 2
    return hit if (rectilinear or curvilinear) else None


def _range_bounds(value: Any) -> tuple[Any, Any] | None:
    """Return ``(lo, hi)`` when ``value`` spells a range, else ``None``.

    Mirrors the two range spellings :func:`select` accepts: a ``slice``, or the
    YAML-friendly ``{"min": ..., "max": ...}`` dict. A scalar, list, or string is
    not a range and returns ``None`` — the caller decides what that means (a
    point, an explicit value set, a date).
    """
    if isinstance(value, slice):
        return value.start, value.stop
    if isinstance(value, dict) and ("min" in value or "max" in value):
        return value.get("min"), value.get("max")
    return None


def box_in_spec(
    spec: dict[str, Any] | None,
) -> tuple[str, str, tuple[float, float], tuple[float, float]] | None:
    """Return ``(lon_key, lat_key, (lon_min, lon_max), (lat_min, lat_max))`` for a box.

    A lon/lat pair both spelled as *ranges* (a ``slice`` or a ``{"min", "max"}``
    dict — see :func:`select`) is a box, the plural of :func:`point_in_spec`'s
    scalar pair: the same key sets serve both, since one value is either a scalar
    (a point) or a range (a box), never both at once. A scalar lon+lat stays a
    point; a lone lon-range or lat-range (a meridional/zonal band) is neither and
    is left to the ordinary per-axis loop in :func:`select`.

    Latitude may be one-sided (open toward a pole, defaulting to ±90); longitude
    may not — an open bound on a circle names no band, so a lone-sided longitude
    falls through to ``None`` (not a box) rather than guessing a hemisphere.
    Backwards bounds (``max`` < ``min``) are accepted on both axes: latitude is
    simply swapped, and longitude is first tried as a seam-straddling box in
    0-360 — ``{"min": 170, "max": -170}`` becomes ``170..190``, the
    ``pac_dt_ramp``-style stress case — before falling back to a swap for a
    plain backwards typo that does not resolve that way (e.g. ``30`` before
    ``20``).
    """
    if not spec:
        return None
    lon_key = next((k for k in _POINT_LON_KEYS if k in spec), None)
    lat_key = next((k for k in _POINT_LAT_KEYS if k in spec), None)
    if lon_key is None or lat_key is None:
        return None
    lon_range, lat_range = _range_bounds(spec[lon_key]), _range_bounds(spec[lat_key])
    if lon_range is None or lat_range is None:
        return None
    lon_lo, lon_hi = lon_range
    if lon_lo is None or lon_hi is None:
        return None  # an open longitude bound names no band on a circle
    lat_lo, lat_hi = lat_range
    lat_lo = -90.0 if lat_lo is None else float(lat_lo)
    lat_hi = 90.0 if lat_hi is None else float(lat_hi)
    if lat_lo > lat_hi:
        lat_lo, lat_hi = lat_hi, lat_lo
    lon_lo, lon_hi = float(lon_lo), float(lon_hi)
    if lon_lo > lon_hi:
        wrapped_lo, wrapped_hi = lon_lo % 360, lon_hi % 360
        lon_lo, lon_hi = (
            (wrapped_lo, wrapped_hi) if wrapped_lo <= wrapped_hi else (lon_hi, lon_lo)
        )
    return lon_key, lat_key, (lon_lo, lon_hi), (lat_lo, lat_hi)


def _box_selectable(
    obj, spec: dict[str, Any] | None
) -> tuple[str, str, tuple[float, float], tuple[float, float]] | None:
    """Return :func:`box_in_spec`'s hit when ``obj`` itself can be cropped to it.

    Narrower than ``box_in_spec`` alone, the same way :func:`_point_selectable` is
    narrower than :func:`point_in_spec`: a box's keys mean nothing unless ``obj``
    actually carries a longitude/latitude coordinate that is either rectilinear
    (each 1-D and its own dimension) or curvilinear (each 2-D, e.g. ROMS's
    ``lon_rho``/``lat_rho``). Unlike the point gate, a **Dataset** is not excluded
    here: :func:`ocean_skill.align.sample_at` needs one array to NaN-check for a
    point, but a crop has no such constraint —
    :func:`ocean_skill.align.subset_to_bbox` already crops whole Datasets, and
    :func:`ocean_skill.align.subset_to_box` does the same.
    """
    hit = box_in_spec(spec)
    if hit is None:
        return None
    from ocean_skill.cf import find_coord

    lon_coord, lat_coord = find_coord(obj, "longitude"), find_coord(obj, "latitude")
    if lon_coord is None or lat_coord is None:
        return None
    rectilinear = (
        lon_coord.ndim == 1
        and lat_coord.ndim == 1
        and lon_coord.name in obj.dims
        and lat_coord.name in obj.dims
    )
    curvilinear = lon_coord.ndim == 2 and lat_coord.ndim == 2
    return hit if (rectilinear or curvilinear) else None


def _refuse_curvilinear_line(
    obj, spec: dict[str, Any] | None, subject: str = "the source"
) -> None:
    """Raise when a lone scalar lon/lat in ``spec`` would silently no-op on a 2-D grid.

    A curvilinear grid's longitude/latitude are 2-D fields, not dimensions --
    ``select={"lon": -144.0}`` has no axis to narrow, so the ordinary per-axis
    loop in :func:`select` silently skips it (the key resolves to no dimension,
    indistinguishable from a genuinely absent one). The identical spec against a
    *rectilinear* grid is a perfectly good meridional slice (see
    :func:`point_in_spec`'s own note on the reading); this only fires where that
    reading is impossible.

    Left alone when ``spec`` is a genuine point (both axes scalar --
    :func:`point_in_spec` already routes that through
    :func:`~ocean_skill.align.sample_at`) or has already had its point/box keys
    consumed by the time this runs (see the two call sites: here in
    :func:`select`, after the point/box branches; and in
    :func:`ocean_skill.comparison._select_horizontal_then_aggregate`, on the
    *unconsumed* spec, before either branch has had a chance to run at all).
    Fires on a lone scalar *or* a scalar paired with a range (the bounded-line
    spelling, ``{'lon': -144.0, 'lat': {'min': ..., 'max': ...}}``), since both
    leave one axis with nothing to narrow.
    """
    if not spec or point_in_spec(spec) is not None:
        return
    from ocean_skill.cf import find_coord

    lon_coord, lat_coord = find_coord(obj, "longitude"), find_coord(obj, "latitude")
    if (
        lon_coord is None
        or lat_coord is None
        or lon_coord.ndim != 2
        or lat_coord.ndim != 2
    ):
        return
    for key in _POINT_LON_KEYS | _POINT_LAT_KEYS:
        if key not in spec or _point_scalar(spec[key]) is None:
            continue
        word = "longitude" if key in _POINT_LON_KEYS else "latitude"
        raise ValueError(
            f"{subject}: select={{{key!r}: {spec[key]!r}}} names a fixed "
            f"{word}, but this source's grid is curvilinear -- {word} is a "
            "2-D field with no grid column to follow, so this key would "
            "silently match nothing. For a line, use select={'transect': "
            f"{{{key!r}: {spec[key]!r}}}}} (bound the other axis with e.g. "
            "'lat': {'min': ..., 'max': ...}); for one place, give both lon "
            "and lat together as scalars (a point)."
        )


def _descending(obj, dim: str) -> bool:
    """Whether ``dim``'s coordinate is stored high-to-low.

    ``False`` for a bare dimension: with no coordinate there is no order to read, and
    ``.sel`` with a slice would not work on it anyway.
    """
    import numpy as np

    if dim not in obj.coords:
        return False
    values = np.asarray(obj[dim].values)
    return bool(values.size > 1 and values[0] > values[-1])


def oriented_slice(obj, dim: str, value: slice) -> slice:
    """Return ``value`` as a *range*, ordered to match ``dim``'s own stored direction.

    ``.sel`` with a slice follows the coordinate's stored order, so ``slice(20, 30)``
    against a **descending** axis selects nothing at all — silently, an empty array
    rather than an error. Satellite L3 products are stored north-to-south (MODIS
    latitude runs 89.979 to -89.979), so this is the common case, not an exotic one:
    ``select={"lat": {"min": 20, "max": 30}}`` against any of them came back empty.

    Both spellings of a range are accepted and mean the same thing, on either kind of
    axis: the bounds are normalized low-to-high first, then flipped if the axis
    descends. That matters for compatibility as much as convenience — anyone who had
    already worked around this by writing the bounds backwards keeps working, rather
    than being broken by the fix.

    One-sided bounds flip too: ``{"min": 20}`` on a north-to-south axis becomes
    ``slice(None, 20)``, which is still "latitude 20 and above".

    :func:`ocean_skill.align.subset_to_bbox` calls this too, for the bbox it crops a
    reference to: a bounding box and a ``select`` band are the same question asked
    twice, so they are one function rather than two kept in step by hand.
    """
    lo, hi = value.start, value.stop
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo  # written high-to-low; a range is a range either way
    return (
        slice(hi, lo, value.step)
        if _descending(obj, dim)
        else slice(lo, hi, value.step)
    )


def _require_coordinate(obj, dim: str, asked: str) -> None:
    """Refuse a *range* against a dimension that carries no coordinate values.

    ``.sel`` on a dimension with no coordinate falls back to positional indexing, so
    ``{"depth": {"min": 0, "max": 50}}`` against a bare axis quietly returns the first
    *index* rather than the first fifty metres — a wrong answer wearing a right one's
    clothes, and one that gets worse the more the axis' values differ from its indices.
    ROMS is the source that makes this reachable: it ships no coordinate variables at
    all, so ``s_rho`` and an undecoded ``time`` are bare dimensions.

    A scalar is deliberately left alone. There the fallback is at least *arguably* what
    was meant (``{"s_rho": 0}`` reading as "the first level" is how xarray behaves
    everywhere), and narrowing that is a separate decision from refusing a range.
    """
    if dim in obj.coords:
        return
    raise ValueError(
        f"cannot select a range along {asked!r}: {dim!r} is a dimension of size "
        f"{obj.sizes[dim]} with no coordinate values, so there is nothing for a "
        "min/max to be measured against — xarray would fall back to positional "
        "indexing and hand back the first few *indices* instead. Give the axis a "
        "coordinate first (for a ROMS run, a catalog reference_date decodes time, and "
        "the vertical is reached with select={'depth': ...} rather than by s-level), "
        "or index by position yourself with .isel() on the data."
    )


def select(obj, spec: dict[str, Any] | None, *, subject: str = "the source"):
    """Subset ``obj`` along each dimension in ``spec``. Returns it unchanged if empty.

    Selection is *not* aggregation, and keeping them apart is the point: this
    narrows an axis (60 days -> the 31 in January) and leaves it standing, while
    :func:`aggregate` collapses it (31 days -> one field). Abigale Wyatt's
    chlorophyll comparison needs both in that order — select the month, then mean
    it — and merging the two would make "January's mean" indistinguishable from
    "each day in January".

    Values may be:

    - a string — including a partial date, so ``{"time": "2012-01"}`` is all of
      January (xarray's own partial-datetime indexing);
    - a ``slice`` — ``{"time": slice("2012-01", "2012-03")}``;
    - a dict — ``{"lat": {"min": 20, "max": 30}}``, the YAML-friendly spelling of a
      slice, since YAML has no slice literal;
    - a list — an explicit set of values;
    - a scalar — exact if present, otherwise the nearest value, because a float
      coordinate almost never matches exactly and failing on that is unhelpful.

    **Both horizontal axes as scalars is a point**, not two independent nearest
    selections: ``{"lon": -144.25, "lat": 49.98}`` is routed through
    :func:`ocean_skill.align.sample_at` (see :func:`point_in_spec`) rather than two
    plain ``.sel(method="nearest")`` calls, which buys three things a per-axis
    nearest search cannot: it works on a curvilinear grid, where ``lon``/``lat``
    are 2-D and neither names a dimension a per-axis ``.sel`` could use; it wraps
    the request into whatever longitude convention the grid is actually stored in
    (``natural_convention``), so a −144° request against a 0-360 Pacific grid does
    not silently snap to the wrong edge column; and it raises rather than returning
    all-NaN when the nearest cell is masked. A lone ``lon`` or ``lat`` (a
    meridional/zonal slice) is unaffected and keeps the per-axis behavior below.
    ``subject`` names the source in the errors/warnings this can raise.

    **A lon range and a lat range together are a box** (see :func:`box_in_spec`),
    routed through :func:`ocean_skill.align.subset_to_box` rather than two
    independent per-axis slices — the box counterpart of the point routing above,
    with the same three benefits: it works on a curvilinear grid (a boolean
    predicate over the 2-D lon/lat, rather than a slice along a dimension neither
    coordinate names), it wraps the box into whichever longitude convention the
    grid actually uses, and an empty result raises rather than silently returning
    a size-zero array. A lone lon-range or lat-range is unaffected.

    A date string that names a single **instant** — ``"2013-01-30T02:00:00"``, or
    any date no coarser than the axis's own resolution — behaves like a scalar:
    exact if present, otherwise the nearest step, because model output is stamped
    at whatever offsets the run chose and the caller cannot be expected to know
    them. A **period** (``"2012-01"``) is never snapped: it either contains data
    or it is an error, because the nearest neighbour of an empty January is data
    from a month the caller did not name. See :func:`_string_instant` for where
    the line sits.

    A date string only ever addresses a *datetime* axis. Some sources carry a
    ``time`` that was deliberately left undecoded — a climatology's "months since
    ..." units describe no fixed-length calendar, so :func:`ocean_skill.build`
    leaves the axis as its raw numeric position rather than guessing a date range
    for it. Asking such an axis for ``"2010-01"`` is refused, with a pointer at
    selecting the axis's own numeric values instead, or giving each side of a
    comparison its own ``select`` (see :func:`_explain_undecoded_axis`) — not at
    the nearest-step fallback above, which presumes a calendar exists to be
    nearest along.

    There is no ``method`` key — nearest is automatic, and every key here is an
    axis name. Since xarray's own ``KeyError`` suggests exactly that keyword, a
    ``method``/``tolerance`` key that matches no axis draws a warning instead of
    the silent skip other absent names get.

    A range — either spelling — is honoured on a **descending** axis as well as an
    ascending one; see :func:`oriented_slice` for why that needs saying, and why a
    satellite product is where it bites.

    Dimensions absent from ``obj`` are skipped, so one spec can be shared across
    variables that do not all carry the same axes. A dimension that is *present* but
    carries no coordinate is a different case and a range against it is refused — see
    :func:`_require_coordinate`.

    A key naming an axis that only exists *after* :func:`aggregate` runs — a groupby
    renames its dim to the grouping key, ``time`` becoming ``month`` — is skipped here
    too, the same as any other absent axis. :func:`ocean_skill.comparison._prepare`
    gives such a key a second try once the aggregate has run, via
    :func:`ocean_skill.comparison._select_horizontal_then_aggregate`; this function
    itself only ever sees one axis's-worth of dimensions at a time.
    """
    spec = dict(spec or {})
    hit = _point_selectable(obj, spec)
    if hit is not None:
        lon_key, lat_key, lon, lat = hit
        from ocean_skill.align import NEAREST, natural_convention, sample_at

        del spec[lon_key], spec[lat_key]
        obj = sample_at(
            obj,
            lon,
            lat,
            method=NEAREST,
            convention=natural_convention(obj),
            subject=subject,
        )
    box = _box_selectable(obj, spec)
    if box is not None:
        lon_key, lat_key, (lon_lo, lon_hi), (lat_lo, lat_hi) = box
        from ocean_skill.align import subset_to_box

        del spec[lon_key], spec[lat_key]
        obj = subset_to_box(obj, (lon_lo, lat_lo, lon_hi, lat_hi), subject=subject)
    _refuse_curvilinear_line(obj, spec, subject)
    for name, value in spec.items():
        dim = resolve_dim(obj, name)
        if dim is None or dim not in obj.dims:
            if name in ("method", "tolerance"):
                warnings.warn(
                    f"select ignores {name!r}: spec keys are axis names, not "
                    "xarray .sel() keywords. Nearest matching is automatic — "
                    "an inexact scalar or a full timestamp snaps to the "
                    "nearest step on its own.",
                    stacklevel=2,
                )
            continue
        if isinstance(value, dict):
            value = slice(value.get("min"), value.get("max"))
        if isinstance(value, slice):
            _require_coordinate(obj, dim, name)
            value = oriented_slice(obj, dim, value)
        if isinstance(value, slice | list | tuple):
            obj = obj.sel({dim: value})
            continue
        try:
            obj = obj.sel({dim: value})
        except (KeyError, IndexError) as err:
            # A value that is not an exact coordinate match: nearest is what the
            # caller meant. A string gets that treatment only when it names a
            # single instant; a period ("2013-01") failing means the period is
            # *empty*, and its nearest neighbour would be data from a time the
            # caller did not name — that stays an error, restated in select's
            # own terms because xarray's hint to try method='nearest' points at
            # a keyword this spec deliberately does not have.
            if isinstance(value, str):
                instant = _string_instant(obj, dim, value)
                if instant is None:
                    _explain_empty_period(obj, dim, name, value, err)
                    raise
                value = instant
            obj = obj.sel({dim: value}, method="nearest")
        except ValueError as err:
            # A numeric axis fed a date string: pandas raises "could not convert
            # string to float", naming neither the axis nor the cure. Restated
            # only when this can be sure that is what happened -- see
            # _explain_undecoded_axis, whose guards keep a typo or a genuinely
            # non-numeric axis's own error intact.
            _explain_undecoded_axis(obj, dim, name, value, err, subject)
            raise
    return obj


def _string_instant(obj, dim: str, value: str):
    """The moment ``value`` names, as a ``Timestamp`` — ``None`` if it names a span.

    ``"2013-01-30T02:00:00"`` is an instant; ``"2013-01"`` is a span (all of
    January); ``"2013-01-30"`` is either, depending on the axis — one step of
    daily output, twenty-four of hourly. The dividing line is the axis's own
    resolution: a string no coarser than the axis pins down a single step's worth
    of time, so *the nearest step* means something, exactly as it does for a
    scalar. Anything coarser is a period, whose emptiness is the caller's to hear
    about (see :func:`select`).

    ``None`` also covers everything this cannot judge: a string that is not a
    date, and an axis that is not a ``DatetimeIndex`` — station labels, or a
    cftime axis, where xarray's own string handling is the only safe reading.
    """
    import pandas as pd

    index = obj.indexes.get(dim)
    if not isinstance(index, pd.DatetimeIndex):
        return None
    try:
        period = pd.Period(value)
    except (TypeError, ValueError):
        return None
    # pandas spells the axis resolution as a word; the matching span is how much
    # time a string at that resolution covers. Coarser entries are unreachable —
    # midnight-stamped monthly data already resolves as "day".
    step = {
        "day": pd.Timedelta(days=1),
        "hour": pd.Timedelta(hours=1),
        "minute": pd.Timedelta(minutes=1),
        "second": pd.Timedelta(seconds=1),
        "millisecond": pd.Timedelta(milliseconds=1),
        "microsecond": pd.Timedelta(microseconds=1),
        "nanosecond": pd.Timedelta(nanoseconds=1),
    }.get(index.resolution)
    if step is None or period.end_time - period.start_time > step:
        return None
    return pd.Timestamp(period.start_time)


def _explain_empty_period(obj, dim: str, name: str, value: str, err) -> None:
    """Raise the empty-period ``KeyError`` in select's vocabulary, when it can.

    Only a date string against a datetime axis earns the restatement — there the
    axis extent says at a glance whether the period missed the record entirely or
    fell in a gap. Anything else that failed (a station name against a label
    index, a typo that is no date at all) keeps xarray's error: returning without
    raising lets :func:`select` re-raise it.
    """
    import pandas as pd

    index = obj.indexes.get(dim)
    if not isinstance(index, pd.DatetimeIndex) or not len(index):
        return
    try:
        pd.Period(value)
    except (TypeError, ValueError):
        return
    raise KeyError(
        f"{name!r} has no data within {value!r}; the axis runs {index[0]} to "
        f"{index[-1]}. A period must contain data — name a single instant "
        "(a full timestamp) to snap to the nearest step instead."
    ) from err


def _explain_undecoded_axis(
    obj, dim: str, name: str, value: str, err, subject: str
) -> None:
    """Raise the numeric-axis ``ValueError`` in select's vocabulary, when it can.

    Only a *string* value against a numeric, non-datetime index earns the
    restatement — a climatology's undecoded "months since ..." time is exactly
    such an axis (see :func:`ocean_skill.build._decode_times`), and pandas'
    ``could not convert string to float`` names neither the axis nor the cure.
    A non-string value, a datetime/cftime axis (whatever else went wrong there),
    or a string that is not even date-shaped keeps xarray's own error: returning
    without raising lets :func:`select` re-raise it.
    """
    import pandas as pd

    index = obj.indexes.get(dim)
    if index is None or not pd.api.types.is_numeric_dtype(index):
        return
    try:
        pd.Period(value)
    except (TypeError, ValueError):
        return
    units = obj[dim].attrs.get("units")
    tell = f" ({units})" if units else ""
    first, last = float(index[0]), float(index[-1])
    raise ValueError(
        f"{name!r} on {subject} is a numeric axis{tell}, not a calendar — a date "
        f"string like {value!r} cannot address it. Its own values run {first!r} to "
        f"{last!r}; select one of those directly (e.g. {{{name!r}: {first!r}}}), "
        "or, in a comparison, give this side its own select instead of sharing the "
        f'date string across both: select={{"test": {{{name!r}: {value!r}}}, '
        '"reference": {}}.'
    ) from err


def spec_names(spec: Any) -> list[list[str]]:
    """Return the alternative name-sets that would satisfy ``spec``.

    Used by :func:`ocean_skill.comparison.compare` to decide, from catalog metadata
    alone, whether a source can supply a variable. A combination has two ways to be
    satisfied — the components, or the ``standard_name`` it falls back to (see
    :func:`resolve_variable`) — so this returns a list of sets, satisfied if *any*
    one of them is fully present.
    """
    if isinstance(spec, str):
        return [[spec]] if spec not in DERIVED else spec_names(DERIVED[spec])
    if not isinstance(spec, dict):
        return []
    spec = dict(spec)
    standard_name = spec.pop("standard_name", None)
    if "calculate" in spec:
        # The remaining keys are the calculator's own kwargs (`method`, `threshold`,
        # ...), not name-lists -- treating them as such the way a combination's
        # components are treated would hand `_leaf_names` a string to iterate
        # character-by-character or a float it cannot iterate at all. A calculator
        # that registered an `inputs` function is asked instead; one that didn't is
        # simply invisible to catalog pre-filtering, same as before it registered.
        name = spec.pop("calculate")
        inputs_fn = CALCULATOR_INPUTS.get(name)
        options = list(inputs_fn(spec)) if inputs_fn is not None else []
    else:
        options = [_leaf_names(names) for names in spec.values()]
    if standard_name:
        options.append([standard_name])
    return options


def _leaf_names(names) -> list[str]:
    """Flatten a component list to the plain variable names at its leaves.

    A nested combination's components are themselves specs, but a catalog can only
    be checked against real variable names, so the tree collapses to its leaves.
    """
    out: list[str] = []
    for name in names:
        if isinstance(name, dict):
            for option in spec_names(name):
                out.extend(option)
        else:
            out.append(name)
    return out


def spatial_mean_in_spec(agg: dict[str, Any] | None) -> tuple[str, str] | None:
    """Return ``(lon_key, lat_key)`` when ``agg`` asks for one joint area-weighted mean.

    Both horizontal axes reduced by a plain ``"mean"`` (or ``{"reduce": "mean"}``,
    with no ``groupby``/``resample``/other kwargs) together — not sequentially, the
    way two ordinary ``aggregate`` entries would be applied one after another (this
    function's own per-key loop) — because with a ragged NaN mask the mean of the
    row-means is not the mean: a joint reduction is the only one that weighs every
    wet cell in the box equally, whichever row or column it sits in. See
    :func:`_horizontal_mean`, the reduction this routes to.

    Anything else — a lone ``{"lat": "mean"}`` zonal mean, ``{"lat": "mean", "lon":
    "max"}``, a groupby/resample on either axis — is not this joint path and is
    left to the ordinary per-axis loop, unweighted as it has always been.
    """
    if not agg:
        return None
    lon_key = next((k for k in _POINT_LON_KEYS if k in agg), None)
    lat_key = next((k for k in _POINT_LAT_KEYS if k in agg), None)
    if lon_key is None or lat_key is None:
        return None

    def _is_plain_mean(value: Any) -> bool:
        if value == "mean":
            return True
        if isinstance(value, dict):
            return value.get("reduce") == "mean" and set(value) <= {"reduce"}
        return False

    if _is_plain_mean(agg[lon_key]) and _is_plain_mean(agg[lat_key]):
        return lon_key, lat_key
    return None


def _horizontal_reducible(da) -> bool:
    """Whether ``da`` has a lon/lat coordinate :func:`_horizontal_mean` can reduce.

    The same rectilinear-or-curvilinear gate :func:`_box_selectable` uses for a
    crop: each 1-D and its own dimension, or each 2-D and sharing dimensions. A
    Dataset (no single array's dims to check) or a trajectory's ``lon(time)``
    fails this and is left to the ordinary per-axis loop below, unweighted.
    """
    from ocean_skill.cf import find_coord

    if hasattr(da, "data_vars"):
        return False
    lon_coord, lat_coord = find_coord(da, "longitude"), find_coord(da, "latitude")
    if lon_coord is None or lat_coord is None:
        return False
    rectilinear = (
        lon_coord.ndim == 1
        and lat_coord.ndim == 1
        and lon_coord.name in da.dims
        and lat_coord.name in da.dims
    )
    curvilinear = lon_coord.ndim == 2 and lat_coord.ndim == 2
    return rectilinear or curvilinear


def _area_weights_for(da) -> tuple[Any, str]:
    """Return ``(weights, description)`` for an area-weighted spatial mean of ``da``.

    True cell area (:data:`ocean_skill.roms.AREA_COORD`, ``1/(pm*pn)``) when a ROMS
    grid carries it — the same "weights ride on the data" pattern
    :func:`_weights_for` reads for a depth band — else ``cos(latitude)``, exact for
    a rectilinear lat/lon grid and the same approximation
    :func:`ocean_skill.metrics.area_weights` already uses for skill metrics. NaN
    weights (dry cells, and cos(lat) wherever latitude itself is missing) are
    filled to zero rather than left to poison the reduction —
    :meth:`xarray.DataArray.weighted` refuses NaN weights outright.
    """
    import numpy as np

    from ocean_skill.cf import find_coord
    from ocean_skill.roms import AREA_COORD

    area = da.coords.get(AREA_COORD)
    if area is not None and set(area.dims) <= set(da.dims):
        return area.fillna(0.0), "area-weighted mean (cell_area)"
    lat_coord = find_coord(da, "latitude")
    if lat_coord is None:
        raise ValueError(
            "cannot area-weight a spatial mean: no latitude coordinate found to "
            "compute cos(latitude) from, and no cell_area coordinate is present."
        )
    weights = np.cos(np.deg2rad(lat_coord))
    return weights.fillna(0.0), "area-weighted mean (cos latitude)"


def _horizontal_mean(da):
    """Return ``da`` reduced by one area-weighted mean over both horizontal axes.

    The reduction :func:`spatial_mean_in_spec` routes to: a single
    ``da.weighted(weights).mean(...)`` over the union of the lon and lat
    coordinates' own dimensions (one dim each on a rectilinear grid, two shared
    dims — ``eta_rho``/``xi_rho`` — on a curvilinear one), which is why it must be
    one call rather than two sequential per-axis means (see
    :func:`spatial_mean_in_spec`).

    The result's lon/lat coordinates become the box's own midpoint, read from
    ``da.attrs["region"]`` when a ``select`` box drove this call — set by
    :func:`ocean_skill.align.subset_to_box`, and carried forward through
    whatever vertical transform ran in between, the same way ``units`` already
    is — or from the field's own extent (:func:`ocean_skill.align.bbox_of`) for a
    whole-domain mean with no preceding box. Reading the *requested* box back
    this way, rather than the reduced data's own (possibly ragged) extent, is
    what lets two lanes of one comparison sharing one box land on exactly the
    same position: this is what drops a box-mean into the same "one place, one
    time axis" recipe a station reference already has
    (:func:`ocean_skill.align.point_of`).
    """
    from ocean_skill.align import _wrap_lon, bbox_of
    from ocean_skill.cf import find_coord

    lon_coord, lat_coord = find_coord(da, "longitude"), find_coord(da, "latitude")
    if lon_coord is None or lat_coord is None:
        raise ValueError(
            "cannot take a spatial mean: no longitude/latitude coordinate found "
            "to reduce."
        )
    dims = sorted(set(lon_coord.dims) | set(lat_coord.dims))
    weights, description = _area_weights_for(da)
    attrs = dict(da.attrs)
    region = da.attrs.get("region")
    out = da.weighted(weights).mean(dim=dims)
    out.attrs = {**attrs, **out.attrs}  # a reduction drops attrs; units must survive
    lon_min, lat_min, lon_max, lat_max = region if region is not None else bbox_of(da)
    mid_lon = _wrap_lon(0.5 * (float(lon_min) + float(lon_max)), "-180-180")
    mid_lat = 0.5 * (float(lat_min) + float(lat_max))
    out = out.assign_coords({lon_coord.name: mid_lon, lat_coord.name: mid_lat})
    out.attrs["spatial_mean"] = description
    if region is not None:
        out.attrs["region"] = list(region)
    return out


def aggregate(da, spec: dict[str, Any] | None):
    """Reduce ``da`` over each dimension in ``spec``. Returns ``da`` unchanged if empty.

    Each value is either a reduction name (``"mean"``) or a dict with ``reduce`` plus
    optional ``groupby`` *or* ``resample``, and any keyword the reduction takes::

        {"time": "mean"}
        {"time": {"groupby": "month", "reduce": "mean"}}     # climatology: 12 fields
        {"time": {"groupby": "season", "reduce": "mean"}}  # DJF/MAM/JJA/SON, in order
        {"time": {"groupby": "season", "seasons": ["JFMA", "MJJA", "SOND"],
                   "reduce": "mean"}}
        {"time": {"resample": "1MS", "reduce": "mean"}}      # consecutive months
        {"time": {"reduce": "quantile", "q": 0.9}}
        {"depth": {"reduce": "integrate"}}
        {"time": {"groupby": "season", "reduce": "mean", "spread": "std"}}  # mean+std

    ``groupby`` and ``resample`` both *keep* an axis rather than collapsing it, and
    they keep different ones — see this module's docstring. Setting both is an error
    rather than a precedence rule, since either alone is a complete answer and
    silently honouring one would give a plausible figure of the wrong thing.

    ``groupby: "season"`` defaults to the standard DJF/MAM/JJA/SON, in that
    (calendar, not alphabetical) order. A ``seasons`` list gives a custom
    definition instead — any number of month-initial strings, e.g. ``["JFMA",
    "MJJA", "SOND"]`` for three, wrapping December to January is fine (``"NDJ"``).
    A requested season with no data in ``da`` is dropped with a warning (all of
    them missing raises instead); a season present but missing part of its months
    warns as well. ``spread`` names a second reduction (usually ``"std"``) computed
    over the same groups as ``reduce`` and attached as a same-shaped ``"spread"``
    coordinate on the result, riding alongside the value through any later
    ``select``/``sel`` or vertical interpolation.

    A time ``groupby`` result stays plottable: :meth:`ocean_skill.field.Field.plot`
    draws the surviving ``month``/``year``/... dimension as the x (or time_depth)
    axis in place of time, with ``month`` spelled ``Jan``..``Dec``; ``season``
    keeps drawing as the profile family's one-line-per-season fan instead.

    **Both horizontal axes reduced by a plain "mean" together** is one joint
    area-weighted spatial mean rather than two sequential unweighted ones — see
    :func:`spatial_mean_in_spec` and :func:`_horizontal_mean`. This is the plural
    of the point routing :func:`select` gives a scalar lon+lat pair: the same key
    spellings, reduced together rather than sliced together, because a mean of
    means is not a mean once the wet-cell mask is ragged.

    Dimensions absent from ``da`` are skipped rather than raising: a selection may
    already have collapsed one, and a spec shared across variables should not fail on
    the one that has no depth axis.
    """
    spec = dict(spec or {})
    pair = spatial_mean_in_spec(spec)
    if pair is not None and _horizontal_reducible(da):
        lon_key, lat_key = pair
        del spec[lon_key], spec[lat_key]
        da = _horizontal_mean(da)
    for name, how in spec.items():
        dim = resolve_dim(da, name)
        if dim is None or dim not in da.dims:
            continue
        da = _reduce_dim(da, dim, how)
    return da


def _weights_for(da, dim: str):
    """Return per-cell weights riding on ``da`` for ``dim``, if any.

    A depth *band* selection attaches cell overlaps (see
    :func:`ocean_skill.roms.depth_band`) so that a later ``{"Z": "mean"}`` is a
    proper thickness-weighted average without ``aggregate`` needing to know anything
    about vertical coordinates. Weights travel with the data; the reduction just
    uses them when they are there.
    """
    from ocean_skill.roms import WEIGHT_COORD

    weights = da.coords.get(WEIGHT_COORD)
    if weights is None or dim not in weights.dims:
        return None
    return weights


def _dim_kwarg(fn) -> str:
    """Return the keyword ``fn`` takes the dimension under: ``dim`` or ``coord``.

    Passing the dimension positionally looks tidier and is wrong: xarray's reductions
    are mostly ``(dim=None, ...)``, but ``quantile`` is ``(q, dim=None, ...)`` — so a
    positional argument silently lands in ``q`` — and ``integrate`` names it ``coord``
    instead. Reading the signature keeps every method working without a per-method
    table to maintain, which is the whole point of dispatching to xarray.
    """
    import inspect

    params = inspect.signature(fn).parameters
    return "coord" if "coord" in params and "dim" not in params else "dim"


#: A resample bin holding less than this share of the median bin's sample count is
#: reported by :func:`_warn_short_bins`. The threshold has to clear the *legitimate*
#: variation between bins — a 28-day February against a 31-day median is 0.90 — while
#: still catching a selection that starts or ends mid-period, which halves a bin to
#: ~0.5. 0.8 sits between the two with room on either side.
SHORT_BIN_FRACTION = 0.8


def _bin_counts(coord, freq: str):
    """Return the number of samples per resample bin, read off ``coord`` alone.

    Deliberately not ``da.resample(...).count()``: that walks the *data*, which for a
    lazy multi-file model run is the entire read, and this is a check that has to be
    cheap enough to run unconditionally. A time coordinate is a small in-memory index
    and already carries everything the count needs.
    """
    import xarray as xr

    dim = str(coord.dims[0])
    return xr.ones_like(coord, dtype=float).resample({dim: freq}).sum()


def _bin_label(value) -> str:
    """Return a resample bin's start as ``YYYY-MM-DD``, for numpy or cftime datetimes.

    Slicing the string representation rather than calling ``strftime`` because the two
    datetime families do not share one: a ROMS run on a 360-day calendar carries cftime
    objects, and both spell their first ten characters the same way.
    """
    return str(value)[:10]


def _warn_short_bins(coord, freq: str, dim: str) -> None:
    """Warn about resample bins the selection only partly covers.

    Selecting a time range that does not land on period boundaries — 15 February to
    15 May, resampled monthly — silently yields a first and last bin averaged over
    half as many samples as the rest. The panels are still labelled ``Feb 2012`` and
    ``May 2012``, and a half-month mean sitting beside full ones under one shared
    colour scale is the kind of wrong number that looks right on a map.

    Compared against the *median* bin rather than the longest: month lengths genuinely
    differ, and flagging February every time would train the warning away.
    """
    counts = _bin_counts(coord, freq)
    values = [float(v) for v in counts.values]
    if len(values) < 2:
        return  # one bin has nothing to be short against
    median = sorted(values)[len(values) // 2]
    if median <= 0:
        return
    short = [
        (_bin_label(label), int(n))
        for label, n in zip(counts[dim].values, values, strict=True)
        if n < SHORT_BIN_FRACTION * median
    ]
    if not short:
        return
    detail = ", ".join(f"{label} has {n}" for label, n in short)
    warnings.warn(
        f"resampling {dim!r} to {freq!r} leaves {len(short)} of {len(values)} bins "
        f"with fewer samples than the usual {median:g}: {detail}. Those means are "
        "over part of a period but are labelled like whole ones — narrow the "
        "selection to whole periods, or drop the short bins, if that is not what "
        "you want.",
        stacklevel=2,
    )


#: Default season definition, in calendar order. ``groupby: "season"`` uses these
#: unless a spec sets ``seasons`` explicitly. Routing through
#: :class:`~xarray.groupers.SeasonGrouper` (rather than xarray's own
#: ``groupby("time.season")``) is what pins this order: xarray's own accessor
#: sorts the result alphabetically (DJF, JJA, MAM, SON), which reads out of
#: calendar order in a panel title or a legend.
DEFAULT_SEASONS: tuple[str, ...] = ("DJF", "MAM", "JJA", "SON")

#: Name of the non-dimension coordinate a ``spread`` reduction rides on. Not an
#: attribute -- an attribute cannot hold an array and would not survive a zarr
#: round trip -- and not a second data variable, either: a coordinate slices and
#: interpolates *with* the value it describes, for free, which is what lets a
#: spread survive ``.sel(season=...)`` and vertical interpolation onto a
#: reference's own levels with no extra code anywhere downstream.
SPREAD_COORD = "spread"

#: Attribute name marking a groupby-renamed dimension as time's stand-in, stamped
#: on the new coordinate by :func:`_reduce_dim` -- the value is the *original*
#: time dimension's name. An attribute, not a name convention, because the new
#: dimension is named by the caller's own groupby key (``month``, ``year``,
#: ``hour``, ...) with no fixed spelling to test for, and it has to survive
#: exactly the same ``.sel``/``.isel``/zarr round trip :data:`SPREAD_COORD` does
#: -- read back by :func:`time_axis_dim`, so a plot can still find "time" once a
#: climatology has renamed it. Never stamped for a ``groupby: "season"`` result:
#: that dimension is a string, not an integer index, and stays the profile
#: family's fan (see :mod:`ocean_skill.plot.profile`), not a plottable x axis.
TIME_GROUPBY_ATTR = "time_groupby"

#: The twelve months' initials, in calendar order. Used to validate a season
#: string as a genuine run of consecutive months (doubled below so a wraparound
#: season like ``"NDJ"`` is a plain substring search).
_MONTH_WHEEL = "JFMAMJJASOND"


def _season_months(season: str) -> tuple[int, ...]:
    """Return the 1-12 month numbers named by a season string like ``"DJF"``.

    Deliberately stricter than :class:`~xarray.groupers.SeasonGrouper` itself,
    which resolves a season from only its first two letters and its length --
    ``"DJQ"`` silently resolves to December/January/February there, the third
    letter never checked. Requiring every letter to spell out a real run of
    consecutive months (December-to-January wraparound allowed, e.g. ``"NDJ"``)
    means a typo fails here, with the season named, instead of downstream as a
    bare ``KeyError`` naming a two-letter digram -- or not failing at all.
    """
    if not isinstance(season, str) or not season:
        raise ValueError(f"season must be a non-empty string, got {season!r}")
    letters = season.upper()
    if len(letters) < 2:
        raise ValueError(
            f"season {letters!r} names only one month, which is ambiguous "
            "(several months share an initial); use groupby: 'month' for a "
            "single month, or spell out the season, e.g. 'DJF'."
        )
    if len(letters) > 12:
        raise ValueError(f"season {letters!r} names more than 12 months")
    start = (_MONTH_WHEEL * 2).find(letters)
    if start == -1:
        raise ValueError(
            f"season {letters!r} is not a run of consecutive months' initials "
            f"({_MONTH_WHEEL}, wrapping December to January is fine, e.g. "
            "'NDJ') -- check for a typo."
        )
    return tuple((start + i) % 12 + 1 for i in range(len(letters)))


def _validate_seasons(seasons: Any) -> list[str]:
    """Validate and normalize a ``seasons`` list, preserving the caller's order.

    Order matters: :class:`~xarray.groupers.SeasonGrouper` keeps whatever order
    it is given, which is how :data:`DEFAULT_SEASONS` fixes the panel/legend
    order to calendar order instead of xarray's own alphabetical
    ``groupby("time.season")`` sort.
    """
    if isinstance(seasons, str) or not hasattr(seasons, "__iter__"):
        raise ValueError(
            f"'seasons' must be a list of season strings, got {seasons!r}"
        )
    seasons = [str(s).upper() for s in seasons]
    if not seasons:
        raise ValueError("'seasons' must not be empty")
    for season in seasons:
        _season_months(season)  # raises with the season named, on a bad string
    seen: set[str] = set()
    duplicates = sorted({s for s in seasons if s in seen or seen.add(s)})
    if duplicates:
        raise ValueError(
            f"'seasons' repeats {duplicates!r} -- each season names the group "
            "the output is indexed under, so a duplicate would collide."
        )
    return seasons


def _filter_seasons(seasons: list[str], coord, dim: str) -> list[str]:
    """Drop requested seasons with no data, warning about what was dropped.

    :class:`~xarray.groupers.SeasonGrouper` raises a
    ``CoordinateValidationError`` when a requested season has zero months
    present -- there is no group to reduce, but it still tries to size an axis
    entry for it. Reading the months present is coordinate-only, like
    :func:`_bin_counts`: cheap enough to run unconditionally, even against a
    lazy multi-file model run.
    """
    import calendar

    import numpy as np

    try:
        months_present = {int(m) for m in np.unique(coord.dt.month.values)}
    except (AttributeError, TypeError) as err:
        raise ValueError(
            f"{dim!r}'s time axis is not a decoded calendar axis, so a season "
            "groupby cannot read months off it. This is usually a climatology "
            "read with decode_times=False -- give it a fixed select= instead."
        ) from err
    kept, dropped, incomplete = [], [], []
    for season in seasons:
        months = set(_season_months(season))
        if not months & months_present:
            dropped.append(season)
            continue
        kept.append(season)
        missing = months - months_present
        if missing:
            names = ", ".join(calendar.month_name[m] for m in sorted(missing))
            incomplete.append(f"{season!r} is missing {names}")
    if dropped:
        warnings.warn(
            f"season(s) {sorted(dropped)!r} have no data along {dim!r} (months "
            f"present: {sorted(months_present)!r}) and were dropped.",
            stacklevel=3,
        )
    if incomplete:
        warnings.warn(
            "some requested seasons are missing part of their months, so "
            "their means are over fewer months than a full season: "
            f"{'; '.join(incomplete)}.",
            stacklevel=3,
        )
    if not kept:
        raise KeyError(
            f"none of the requested seasons {seasons!r} have any data along "
            f"{dim!r}; months present are {sorted(months_present)!r}."
        )
    return kept


def _reduce_dim(da, dim: str, how: str | dict[str, Any]):
    """Apply one reduction (optionally after a groupby or resample) along ``dim``."""
    opts = {"reduce": how} if isinstance(how, str) else dict(how)
    group = opts.pop("groupby", None)
    freq = opts.pop("resample", None)
    name = opts.pop("reduce", None)
    seasons = opts.pop("seasons", None)
    spread = opts.pop("spread", None)
    if not name:
        raise ValueError(f"aggregate spec for {dim!r} needs a 'reduce', got {how!r}")
    if group is not None and freq is not None:
        raise ValueError(
            f"aggregate spec for {dim!r} sets both 'groupby' ({group!r}) and "
            f"'resample' ({freq!r}), which are different reductions: groupby bins by "
            "label (a climatology -- every January of the record in one field), "
            "resample bins by interval (consecutive periods -- January 2012, "
            "February 2012, ...). Pick one."
        )
    if seasons is not None and group != "season":
        raise ValueError(
            f"aggregate spec for {dim!r} sets 'seasons' without "
            "{'groupby': 'season'} -- 'seasons' only names a custom season "
            "definition, so it has nothing to do without a season groupby."
        )
    spread_unknown = spread is not None and spread not in REDUCERS
    if spread_unknown and getattr(da, spread, None) is None:
        raise KeyError(
            f"unknown 'spread' reduction {spread!r} for {dim!r}: not an xarray "
            f"method and not in REDUCERS ({sorted(REDUCERS)})"
        )

    attrs = dict(da.attrs)
    target = dim
    # Read before any grouping renames or replaces `da`: whether `dim` was time's
    # own axis decides how the new groupby dimension's coordinate gets marked,
    # below.
    is_time_dim = dim == resolve_dim(da, "T")
    # Read weights before any grouping: a GroupBy object exposes no coordinates, and
    # a grouped reduction is along the grouping axis anyway, which vertical cell
    # weights never describe.
    weights = _weights_for(da, target) if name == "mean" else None
    if group is not None:
        if group == "season":
            # A season groupby is a climatology like any other -- it groups along
            # the dim, then reduces within each group -- but the group labels are a
            # spec-provided season definition, not one xarray already knows, and
            # xarray's own naive-crash-on-empty-season and alphabetical-sort
            # behaviours both need working around first.
            #
            # xarray.groupers.SeasonGrouper needs a reasonably recent xarray
            # (the new Grouper protocol it belongs to); environment.yml pins
            # nothing yet ("loose for now"), so this is the one place that
            # requirement is written down. An xarray without it fails here
            # with ModuleNotFoundError naming this submodule -- upgrade xarray.
            from xarray.groupers import SeasonGrouper

            wanted = _validate_seasons(
                seasons if seasons is not None else DEFAULT_SEASONS
            )
            kept = _filter_seasons(wanted, da[target], target)
            da = da.groupby({target: SeasonGrouper(kept)})
        else:
            # A climatology groups along the dim, then reduces *within* each group,
            # so the reduction still names the original dim.
            da = da.groupby(f"{dim}.{group}")
        weights = None
    elif freq is not None:
        # Resampling keeps the dim's *name* (unlike groupby, which renames it to the
        # grouping), so the result is still indexed by time -- consecutive period
        # starts rather than every original step.
        if target in da.coords:
            _warn_short_bins(da[target], freq, target)
        da = da.resample({target: freq})
        weights = None
    reducible = da

    def _run(obj, reduction_name: str):
        # A weighted mean is the only reduction where per-cell extent changes the
        # answer: summing a depth band without thickness weights would count a 17 m
        # cell the same as a 0.5 m one. max/std/quantile -- including a `spread`
        # reduction -- operate on the cells as a set, so they need no weighting;
        # ``weights`` is already None here whenever a groupby/resample ran.
        if weights is not None and reduction_name == "mean":
            return obj.weighted(weights).mean(target, **opts)
        if reduction_name in REDUCERS:
            return REDUCERS[reduction_name](obj, target, **opts)
        fn = getattr(obj, reduction_name, None)
        if fn is None:
            raise KeyError(
                f"unknown reduction {reduction_name!r} for {dim!r}: not an xarray "
                f"method and not in REDUCERS ({sorted(REDUCERS)})"
            )
        return fn(**{_dim_kwarg(fn): target}, **opts)

    out = _run(reducible, name)

    if spread is not None:
        # Rides as a coordinate alongside the value -- see SPREAD_COORD's docstring
        # for why -- computed off the same grouped/resampled object as the mean, so
        # a seasonal std is the spread *within* each season, not across all of them.
        spread_arr = _run(reducible, spread).rename(SPREAD_COORD)
        spread_arr.attrs = {"statistic": spread}
        if "units" in attrs:
            spread_arr.attrs["units"] = attrs["units"]
        out = out.assign_coords({SPREAD_COORD: spread_arr})

    if group is not None:
        # A groupby renames the dim to the grouping label (``month``, ``year``,
        # ``season``, ...) with a coordinate whose attrs are either empty (the
        # `.dt`-accessor route) or, for season, copied verbatim from whatever the
        # *source* time coordinate happened to carry -- an accident of
        # `SeasonGrouper`, not a decision, that would otherwise make a
        # CF-tagged source's season climatology resolve as a time axis while an
        # untagged source's does not. Normalize deliberately instead: a time
        # `.dt` groupby is marked so the plot path can still find "time" (see
        # `time_axis_dim`); a season groupby -- always the profile family's fan,
        # never a plottable axis -- gets no attrs at all, regardless of source.
        new_dim = "season" if group == "season" else str(group)
        coord = out.coords.get(new_dim)
        if coord is not None:
            fresh_attrs = (
                {TIME_GROUPBY_ATTR: str(dim)}
                if is_time_dim and group != "season"
                else {}
            )
            var = coord.variable.copy(deep=False)
            var.attrs = fresh_attrs
            out = out.assign_coords({new_dim: var})

    out.attrs = {**attrs, **out.attrs}  # reductions drop attrs; units must survive
    return out
