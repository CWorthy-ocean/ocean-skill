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
    "CALCULATOR_INPUTS",
    "CALCULATORS",
    "COMBINERS",
    "DERIVED",
    "REDUCERS",
    "aggregate",
    "combine",
    "oriented_slice",
    "point_in_spec",
    "register_calculator",
    "register_derived",
    "register_reducer",
    "resolve_dim",
    "resolve_variable",
    "select",
    "spec_names",
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
    from ocean_skill.cf import _COORD_FALLBACKS, find_coord

    found = find_coord(obj, kind)
    # The coordinate may be multi-dimensional (curvilinear lon/lat); only a
    # 1-D coordinate names a dimension that can be selected or reduced.
    if found is not None and found.name in obj.dims:
        return str(found.name)
    # ROMS output carries no coordinate variables at all, so its axes are bare
    # dimensions that find_coord cannot see -- `s_rho` is a dimension with nothing
    # attached. Match the fallback names against the dimensions themselves, or
    # `{"Z": "mean"}` silently does nothing on exactly the model this exists for.
    for candidate in _COORD_FALLBACKS[kind]:
        if candidate in obj.dims:
            return candidate
    return None


#: Spec key spellings that name the horizontal axes, for :func:`point_in_spec` --
#: the union of :data:`_CF_AXES`'s "X"/"Y" and every name
#: :func:`ocean_skill.cf.find_coord`'s fallback list resolves, so a point select is
#: recognized under whichever spelling a source or a caller happens to use.
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

    A date string that names a single **instant** — ``"2013-01-30T02:00:00"``, or
    any date no coarser than the axis's own resolution — behaves like a scalar:
    exact if present, otherwise the nearest step, because model output is stamped
    at whatever offsets the run chose and the caller cannot be expected to know
    them. A **period** (``"2012-01"``) is never snapped: it either contains data
    or it is an error, because the nearest neighbour of an empty January is data
    from a month the caller did not name. See :func:`_string_instant` for where
    the line sits.

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


def aggregate(da, spec: dict[str, Any] | None):
    """Reduce ``da`` over each dimension in ``spec``. Returns ``da`` unchanged if empty.

    Each value is either a reduction name (``"mean"``) or a dict with ``reduce`` plus
    optional ``groupby`` *or* ``resample``, and any keyword the reduction takes::

        {"time": "mean"}
        {"time": {"groupby": "month", "reduce": "mean"}}     # climatology: 12 fields
        {"time": {"resample": "1MS", "reduce": "mean"}}      # consecutive months
        {"time": {"reduce": "quantile", "q": 0.9}}
        {"depth": {"reduce": "integrate"}}

    ``groupby`` and ``resample`` both *keep* an axis rather than collapsing it, and
    they keep different ones — see this module's docstring. Setting both is an error
    rather than a precedence rule, since either alone is a complete answer and
    silently honouring one would give a plausible figure of the wrong thing.

    Dimensions absent from ``da`` are skipped rather than raising: a selection may
    already have collapsed one, and a spec shared across variables should not fail on
    the one that has no depth axis.
    """
    for name, how in (spec or {}).items():
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


def _reduce_dim(da, dim: str, how: str | dict[str, Any]):
    """Apply one reduction (optionally after a groupby or resample) along ``dim``."""
    opts = {"reduce": how} if isinstance(how, str) else dict(how)
    group = opts.pop("groupby", None)
    freq = opts.pop("resample", None)
    name = opts.pop("reduce", None)
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

    attrs = dict(da.attrs)
    target = dim
    # Read weights before any grouping: a GroupBy object exposes no coordinates, and
    # a grouped reduction is along the grouping axis anyway, which vertical cell
    # weights never describe.
    weights = _weights_for(da, target) if name == "mean" else None
    if group is not None:
        # A climatology groups along the dim, then reduces *within* each group, so
        # the reduction still names the original dim.
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

    if weights is not None:
        # A weighted mean is the only reduction where per-cell extent changes the
        # answer: summing a depth band without thickness weights would count a 17 m
        # cell the same as a 0.5 m one. max/std/quantile operate on the cells as a
        # set, so they need no weighting.
        out = da.weighted(weights).mean(target, **opts)
    elif name in REDUCERS:
        out = REDUCERS[name](da, target, **opts)
    else:
        fn = getattr(da, name, None)
        if fn is None:
            raise KeyError(
                f"unknown reduction {name!r} for {dim!r}: not an xarray method and "
                f"not in REDUCERS ({sorted(REDUCERS)})"
            )
        out = fn(**{_dim_kwarg(fn): target}, **opts)

    out.attrs = {**attrs, **out.attrs}  # reductions drop attrs; units must survive
    return out
