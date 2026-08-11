"""Combining variables and reducing dimensions, as data rather than code.

Two composable layers, both declarative so a YAML suite expresses exactly what the
Python API does:

**Variable resolution** — how the field is obtained. A plain name, a *combination* of
several (MARBL carries chlorophyll as ``spChl``/``diatChl``/``diazChl``; the total is
their sum), or a registered derived diagnostic.

**Selection** — which part of an axis to keep (:func:`select`), e.g.
``{"time": "2012-01"}``. Narrows a dimension; does not remove it.

**Aggregation** — how dimensions collapse. ``{"time": "mean"}``, a climatology
``{"time": {"groupby": "month", "reduce": "mean"}}``, a percentile
``{"time": {"reduce": "quantile", "q": 0.9}}``.

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
import operator
import warnings
from typing import Any

__all__ = [
    "COMBINERS",
    "DERIVED",
    "REDUCERS",
    "aggregate",
    "combine",
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
    - ``{"calculate": name}`` — a registered derived diagnostic (not yet populated).

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
        raise NotImplementedError(
            f"derived diagnostic {spec['calculate']!r}: register a calculator first"
        )

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


def select(obj, spec: dict[str, Any] | None):
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

    Dimensions absent from ``obj`` are skipped, so one spec can be shared across
    variables that do not all carry the same axes.
    """
    for name, value in (spec or {}).items():
        dim = resolve_dim(obj, name)
        if dim is None or dim not in obj.dims:
            continue
        if isinstance(value, dict):
            value = slice(value.get("min"), value.get("max"))
        if isinstance(value, slice | list | tuple):
            obj = obj.sel({dim: value})
            continue
        try:
            obj = obj.sel({dim: value})
        except (KeyError, IndexError):
            # A scalar that is not an exact coordinate value: nearest is what the
            # caller meant. Strings (partial dates) are exempt — xarray resolves
            # those itself, and "nearest" is undefined for them.
            if isinstance(value, str):
                raise
            obj = obj.sel({dim: value}, method="nearest")
    return obj


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
    spec.pop("calculate", None)
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
    optional ``groupby`` and any keyword the reduction takes::

        {"time": "mean"}
        {"time": {"groupby": "month", "reduce": "mean"}}     # climatology
        {"time": {"reduce": "quantile", "q": 0.9}}
        {"depth": {"reduce": "integrate"}}

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


def _reduce_dim(da, dim: str, how: str | dict[str, Any]):
    """Apply one reduction (optionally after a groupby) along ``dim``."""
    opts = {"reduce": how} if isinstance(how, str) else dict(how)
    group = opts.pop("groupby", None)
    name = opts.pop("reduce", None)
    if not name:
        raise ValueError(f"aggregate spec for {dim!r} needs a 'reduce', got {how!r}")

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
