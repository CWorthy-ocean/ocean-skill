"""Reading sources: ``osk.read`` opens an intake v2 catalog entry, standardized.

Opens the entry with intake (``cat[name].read()``), then standardizes: ROMS entries
(``metadata.model == "roms"``) route through :mod:`ocean_skill.roms`; other gridded/obs
entries get a light CF rename from the entry's ``standard_names`` map (fuller handling
lives in :mod:`ocean_skill.cf`), and a Dataset's renamed variables have their own
``standard_name`` attribute set to match -- the catalog outranks the file's attribute.
Returns a **known type** by featureType: point featureTypes →
:class:`pandas.DataFrame`; gridded/multidim → :class:`xarray.Dataset`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from ocean_skill import time_zone
from ocean_skill.catalog import SourceRef, resolve

__all__ = ["erddap_constraints", "read"]

#: In-process memo of :func:`read`'s standardized result, keyed on source identity +
#: catalog-file freshness + the call's own qc/kwargs (see ``_read_cache_key`` below).
#: Bounded (see ``_READ_CACHE_MAXSIZE``) since only lazy objects are held -- dask data
#: is never realized, and even a ROMS entry's eagerly-read coords are a few MiB.
#: Exists because a station-fan ``compare()`` (one :class:`~ocean_skill.comparison
#: .Comparison` per reference) otherwise reopens and re-standardizes the *same* test
#: source once per station -- for a large ROMS history file (tens of thousands of
#: dask chunks), that open/standardize cost dwarfs the few-column point read each
#: comparison actually needs. Cleared by :func:`ocean_skill.cache.clear` and disabled
#: alongside :func:`ocean_skill.cache.disable` -- see the ``cache.enabled()`` guard
#: below -- so the existing cache controls also govern this one.
_READ_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_READ_CACHE_MAXSIZE = 8


def _read_cache_key(ref: SourceRef, qc: Any, kwargs: dict[str, Any]) -> tuple | None:
    """The memo key for one ``read()`` call, or ``None`` to skip caching it.

    Identity is ``(path, name)``; freshness is the catalog file's own
    ``(mtime_ns, size)``, the same pair :func:`ocean_skill.catalog._catalog_fingerprint`
    already uses to notice a rebuilt catalog -- an edited/rewritten catalog file
    changes this key, so a stale open is never served past that edit. This says
    nothing about the *data* the catalog entry points at (a model rerun in place,
    say) -- exactly like the on-disk aligned-pair cache, which the module docstring
    there already documents as identity-keyed, not content-keyed; ``osk.cache.clear()``
    remains the way to force a rebuilt/rerun source to be reread, and now clears this
    memo too. ``None`` when the catalog file cannot be stat'd (a remote or otherwise
    unusual path) -- failing open (no memo) rather than caching under a freshness
    signal that cannot actually detect a change.
    """
    import os

    try:
        st = os.stat(ref.path)
    except OSError:
        return None
    qc_key = json.dumps(qc, sort_keys=True, default=str)
    kwargs_key = json.dumps(kwargs, sort_keys=True, default=str)
    return (str(ref.path), ref.name, st.st_mtime_ns, st.st_size, qc_key, kwargs_key)


def read(source: str | SourceRef, *, qc: Any = None, **kwargs: Any):
    """Open a catalog source and return a CF-standardized Dataset or DataFrame.

    Parameters
    ----------
    source
        An entry name (``"glodap"`` or ``"catalog:name"``) or a :class:`SourceRef`.
    qc
        Per-call override of the entry's own provider-QC policy (a tabular source
        only — see :mod:`ocean_skill.qc`, whose module docstring is the fuller
        spec): a dict such as ``{"keep": ["GOOD", "SUSPECT"]}`` narrows or widens
        which flag meanings survive, and the string ``"off"`` leaves provider flag
        values completely untouched (fill masking still runs). ``None`` (the
        default) uses the entry's own saved contract unchanged; an entry with no
        ``qc`` contract at all is a no-op regardless of what is passed here.
        Keyword-only so it is never mistaken for a reader keyword and passed on to
        ``entry(**kwargs)`` below.
    **kwargs
        Reader keywords, overriding the entry's own. The one that earns this is
        ``constraints=`` on an ERDDAP table: a mooring's whole record is a large
        download, and ``osk.read(entry, constraints={"time>=": "2015-01-01"})``
        subsets it *server-side*, where a later ``select={"time": ...}`` cannot.
        These used to be accepted and silently discarded.

    Repeat calls naming the same entry, the same ``qc=``/``**kwargs``, and an
    unchanged catalog file reuse one already-opened, already-standardized lazy
    result rather than reopening it (see :data:`_READ_CACHE`) -- each caller still
    gets its own shallow copy (independent ``.attrs``/coords, shared lazy data), so
    mutating one caller's result is never visible to another's. Governed by
    :mod:`ocean_skill.cache`: :func:`ocean_skill.cache.disable` also disables this
    memo, and :func:`ocean_skill.cache.clear` also empties it -- call
    ``read.cache_clear()`` directly to do just that without touching the on-disk
    cache. A bare ``cache=False`` passed to :func:`ocean_skill.compare`/``align()``
    is a per-call flag, not :func:`ocean_skill.cache.disable`, so it still benefits
    from (and populates) this memo across one fan's comparisons.
    """
    from ocean_skill import cache as _cache

    ref = source if isinstance(source, SourceRef) else resolve(source)
    meta = ref.metadata

    use_memo = _cache.enabled()
    key = _read_cache_key(ref, qc, kwargs) if use_memo else None
    if key is not None and key in _READ_CACHE:
        _READ_CACHE.move_to_end(key)
        return _READ_CACHE[key].copy(deep=False)

    obj = _read_uncached(ref, meta, qc, kwargs)

    if key is not None:
        _READ_CACHE[key] = obj
        _READ_CACHE.move_to_end(key)
        while len(_READ_CACHE) > _READ_CACHE_MAXSIZE:
            _READ_CACHE.popitem(last=False)
        return _READ_CACHE[key].copy(deep=False)
    return obj


read.cache_clear = _READ_CACHE.clear  # type: ignore[attr-defined]


def _read_uncached(ref: SourceRef, meta: dict[str, Any], qc: Any, kwargs: dict[str, Any]):
    """The actual open + standardize, uncached -- see :func:`read`'s own memo."""
    import intake

    cat = intake.from_yaml_file(str(ref.path))
    entry = cat[ref.name]
    # An intake v2 entry is called to re-parameterize it; calling it with nothing would
    # also work but reads oddly, so only when there is something to say.
    obj = entry(**kwargs).read() if kwargs else entry.read()
    subject = meta.get("datasetID") or meta.get("title") or ref.name

    if meta.get("model") == "roms" or meta.get("loader") == "ocean_skill.roms":
        from ocean_skill import roms

        # derive_velocity left at its default (False): geographic velocity is
        # derived on demand instead, only once a caller's request actually names
        # it -- see roms.standardize's own docstring and
        # ocean_skill.comparison.prepare_source/_variable_available.
        # `time` is the coordinate standardize leaves, whatever the file called it.
        return _in_utc(roms.standardize(obj, meta), meta, "time", subject=subject)

    # Point sources (e.g. ERDDAP tabledap, via add_erddap_source) come back as a
    # DataFrame rather than a Dataset — same metadata contract, different renaming and
    # time-decoding calls, since pandas has no .variables/.assign_coords.
    is_frame = hasattr(obj, "columns")

    if is_frame:
        # Applied here, before the standard_names rename just below: the saved
        # contract's flags/pairs reference the original (unrenamed) column names --
        # see ocean_skill.qc's module docstring.
        from ocean_skill.qc import apply as _apply_qc

        obj = _apply_qc(obj, meta, qc)

        # A table whose time is split over several columns (``time_columns``) gets one
        # time column here: after qc, whose contract names the original columns, and
        # before the rename below, which has no business with the pieces.
        if meta.get("time_columns"):
            from ocean_skill import tabular

            obj = tabular.apply_table_options(obj, meta, subject=subject)

    # Generic/obs: light CF rename from the entry's standard_names map (cf.standardize
    # will do axis detection + units later). Skip any rename whose target already exists
    # or is claimed twice — the mapping has to stay one-to-one for rename() to work.
    rename: dict[str, str] = {}
    existing = set(obj.columns) if is_frame else set(getattr(obj, "variables", {}))
    for src, dst in (meta.get("standard_names") or {}).items():
        if src not in existing or dst in rename.values() or dst in existing:
            continue
        rename[src] = dst
    if rename:
        obj = obj.rename(columns=rename) if is_frame else obj.rename(rename)
    if rename and not is_frame:
        # The catalog's declared standard name is the authority, so each renamed
        # variable's own ``standard_name`` attribute is made to agree with it,
        # overwriting whatever the file said. Renaming alone leaves a wrong attribute
        # standing, and downstream code reads ``attrs["standard_name"] or name`` (see
        # ocean_skill.units.find_variable, and the pair-spec mismatch check in
        # ocean_skill.comparison): a product like the Holte & Talley MLD climatology
        # carries self-named attributes (``mld_dt_mean`` says ``"mld_dt_mean"``), so the
        # variable would be renamed to its CF name and still be reported, and looked
        # up, under the old one. A DataFrame has no per-column attrs, so this is the
        # Dataset's alone.
        #
        # Functional, one variable at a time: ``assign`` builds a new Dataset, and
        # each step reads the previous one so a coordinate carried inside a later
        # variable's DataArray already has its stamp. Nothing is written into an
        # existing attrs dict, so the Dataset the reader handed back -- which it may
        # still hold -- is never touched. ``assign`` with an existing name keeps a
        # coordinate a coordinate and a data variable a data variable, so there is no
        # need to sort the renamed names into the two. Only a renamed variable whose
        # attribute is missing or disagrees is touched, which leaves an entry whose
        # attributes already match -- the usual case, since the map was derived from
        # them -- on exactly the path it took before; nothing else changes.
        for dst in rename.values():
            if obj[dst].attrs.get("standard_name") != dst:
                obj = obj.assign({dst: obj[dst].assign_attrs(standard_name=dst)})

    # A moored/fixed station read directly as xarray (rather than built through
    # ocean_skill.tabular.to_dataset, which already collapses this for a table) can
    # carry its position as size-1 X/Y *dimension* coordinates on their own separate
    # dims (SEANOE's ADCP moorings, e.g. LATITUDE/LONGITUDE each on their own length-1
    # dim) rather than the scalar lon/lat every station-shaped consumer expects
    # (ocean_skill.align.point_of, and tabular's own convention). A coordinate on a
    # dim the data variable doesn't share is silently dropped when the variable is
    # extracted (ds[var]), so the fixed position never reaches it -- point_of then
    # sees no position at all and the field is mistaken for a bare grid facet.
    # Squeezed here (kept as a scalar, not dropped) so every variable in the Dataset
    # carries it, the same shape ocean_skill.tabular.to_dataset already produces for
    # a point read as a table. A real grid is left alone -- a size-1 horizontal axis
    # there is a genuine (if degenerate) grid cell, not a station's fixed position --
    # and resolve_dim already returns None for a 2-D (curvilinear) lon/lat, so those
    # are never touched either way.
    if not is_frame and str(meta.get("featureType") or "").lower() != "grid":
        from ocean_skill.operators import resolve_dim

        singleton_horizontal = [
            d
            for d in (resolve_dim(obj, "X"), resolve_dim(obj, "Y"))
            if d is not None and d in obj.dims and obj.sizes[d] == 1
        ]
        if singleton_horizontal:
            obj = obj.squeeze(singleton_horizontal, drop=False)

    # Sources are opened with decode_times=False (ocean time units are often non-CF and
    # make xarray refuse the whole file), so decode here instead — otherwise time comes
    # back as raw integers (Dataset) or ISO8601 strings (DataFrame, e.g. ERDDAP's
    # "time (UTC)"). Undecodable units (WOA's "months since ...") return None and are
    # left alone.
    tname = (meta.get("axes") or {}).get("T")
    if is_frame and not tname and meta.get("time_columns"):
        from ocean_skill import tabular

        tname = tabular.joined_time_column(obj, meta)  # the joined column is the axis
    if is_frame and tname and tname in obj.columns:
        from ocean_skill import tabular

        # Naive timestamps are the entry's declared local time (time_zone /
        # utc_offset_h) and become UTC here, once; with nothing declared they were UTC
        # all along, and an offset-carrying one is converted as it stands.
        # decode_time_column also reads a CF "<n> since <date>" stated in the column's
        # own name, which a bare pandas parse turns into dates in 1970.
        decoded = tabular.decode_time_column(obj[tname], tname, meta, subject=subject)
        obj = obj.assign(**{tname: decoded})
    elif not is_frame and tname and tname in getattr(obj, "variables", {}):
        from ocean_skill.build import _decode_times

        decoded = _decode_times(obj, obj[tname])
        if decoded is not None:
            obj = obj.assign_coords({tname: decoded})
    if not is_frame:
        obj = _in_utc(obj, meta, tname, subject=subject)
        obj = _with_month_coordinate(obj, meta)
    return obj


def _in_utc(obj, meta: dict[str, Any], tname: str | None, *, subject: str):
    """Return ``obj`` with its time coordinate shifted from the declared zone to UTC.

    An xarray ``datetime64`` is always naive, so it cannot say what zone it is in; an
    entry that declares ``time_zone`` / ``utc_offset_h`` (see
    :mod:`ocean_skill.time_zone`) is saying its clock readings are *local*. They are
    shifted to UTC here, once, as the tabular read does for a column of timestamps, and
    the coordinate gains ``source_time_zone`` to say what the source kept. ``tname`` is
    the entry's time axis; failing that, a coordinate called ``time``. Returns ``obj``
    untouched when nothing is declared (or UTC is), which is every source that does not
    say -- no values change.

    A declared zone on a time axis that never became ``datetime64`` (undecodable units,
    cftime dates) cannot be applied; that is warned about rather than silently dropped.
    """
    label = time_zone.time_zone_label(meta)  # also validates the declaration
    if label is None:
        return obj
    variables = getattr(obj, "variables", {})
    name = next((n for n in (tname, "time") if n and n in variables), None)
    if name is None:
        return obj
    time = obj[name]
    if time.dtype.kind != "M":
        import warnings

        warnings.warn(
            f"{subject}: declares time_zone {label}, but its time axis {name!r} is not "
            f"a datetime64 (dtype {time.dtype}) -- the zone could not be applied and "
            "the times are as the file states them.",
            stacklevel=2,
        )
        return obj
    shifted = time_zone.localize_naive_datetime64(time.values, meta, subject=subject)
    return obj.assign_coords(
        {name: (time.dims, shifted, {**time.attrs, "source_time_zone": label})}
    )


def _with_month_coordinate(obj, meta: dict[str, Any]):
    """Return ``obj`` with a 1-D ``month`` coordinate (1..12) on a monthly climatology.

    The time axis of WOA's monthly climatology is left undecoded on purpose (see
    :func:`ocean_skill.build._decode_times`): ``months since 1955-01-01`` names no
    fixed span, and the twelve steps are not dates in any year. That is exactly what
    keeps ``aggregate={"time": {"groupby": "month", ...}}`` from working on them --
    there is no calendar to read a month from -- unless the months are given as a
    coordinate of their own, which :func:`ocean_skill.operators._time_group_key` then
    groups by. This supplies it.

    The month is read off the time *values*, not their order in the file: a step is
    ``floor(value)`` months after the epoch of the units, so with an epoch in month
    ``E`` it is month ``(E - 1 + floor(value)) % 12 + 1``. WOA stamps each month at
    its middle (``396.5`` against ``months since 1955-01-01`` is January, ``397.5``
    February), and flooring is what takes the half-month off. The epoch's own month
    is honoured rather than assumed to be January, since a product is free to count
    from anywhere.

    Applied to a climatology declared ``climatology_period: monthly``, and to any
    climatology whose undecoded time has exactly twelve steps. Everything else --
    a single month's entry, an annual climatology, a decoded calendar axis -- is
    returned untouched, as is an object that already has a ``month``. A declared
    monthly climatology whose time cannot be read this way (no ``months since``
    units, or twelve steps that are not twelve different months) warns rather than
    staying silent, since the aggregate it was built for will then fail on grouping.
    """
    import re
    import warnings

    import numpy as np

    from ocean_skill.operators import resolve_dim

    declared = str(meta.get("climatology_period") or "").lower() == "monthly"
    if not meta.get("climatology") or "month" in getattr(obj, "coords", {}):
        return obj
    dim = resolve_dim(obj, "T")
    if dim is None or dim not in obj.coords or obj.coords[dim].dims != (dim,):
        return obj
    time = obj.coords[dim]
    if np.issubdtype(time.dtype, np.datetime64) or not (declared or time.size == 12):
        return obj

    match = re.match(
        r"\s*months\s+since\s+(\d{4})-(\d{1,2})",
        str(time.attrs.get("units", "")),
        re.IGNORECASE,
    )
    months = None
    if match and time.size == 12:
        values = np.asarray(time.values, dtype=float)
        if np.isfinite(values).all():
            months = (int(match.group(2)) - 1 + np.floor(values).astype(int)) % 12 + 1
    if months is None or len(set(months.tolist())) != 12:
        if declared:
            warnings.warn(
                f"{meta.get('title') or 'a monthly climatology'} is declared "
                "climatology_period: monthly, but its time axis does not hold twelve "
                "different months ('months since ...' units, twelve steps), so no "
                "'month' coordinate was added and a groupby month will not work on it.",
                stacklevel=2,
            )
        return obj
    return obj.assign_coords(
        month=(dim, months, {"long_name": "calendar month of the climatology step"})
    )


#: Keys naming the time axis in a ``select``, in any accepted spelling. An entry's own
#: ``axes["T"]`` (ERDDAP spells it ``"time (UTC)"``) is added to these per call.
_TIME_KEYS = frozenset({"time", "T", "t"})


def erddap_constraints(
    meta: dict[str, Any],
    select: dict[str, Any] | None = None,
    time_window: tuple[Any, Any] | None = None,
) -> dict[str, str]:
    """Return the ERDDAP ``constraints`` narrowing a tabledap read to the time wanted.

    Tabledap hands back a finished table in one request, so a read narrowed *after* it
    returns has already been paid for in full — unlike a lazily-opened gridded source,
    where xarray simply never fetches what a later ``select`` discards. That is why
    this exists for one protocol rather than as a general feature: it turns the time
    part of a ``select``, and the window a skill map derives from its test lane, into
    ``time>=``/``time<=`` so the narrowing happens on the server.

    Both are honoured together, tightest bound winning, since they mean the same thing
    from different directions: one is the window the caller asked for, the other the
    window the pipeline worked out on its own.

    Purely an optimization. The caller applies the same ``select`` in memory either
    way — ERDDAP's inclusive string comparison is not quite xarray's slice, and the
    in-memory pass is what makes the result identical whether or not this fires. So
    every ``{}`` returned here (a gridded entry, a ``select`` naming no time, a value
    that will not read as one) costs bandwidth and never correctness.

    ``meta`` is the entry's metadata. Anything without a ``tabledap`` endpoint returns
    ``{}``: every non-ERDDAP catalog, and ERDDAP's own griddap entries, which are
    opened as OPeNDAP and are lazy already.
    """
    if not meta.get("tabledap"):
        return {}

    names = _TIME_KEYS | {(meta.get("axes") or {}).get("T")}
    lo = hi = None
    for key, value in (select or {}).items():
        if key in names:
            lo, hi = _tightest((lo, hi), _time_span(value))
    if time_window is not None:
        start, stop = time_window
        lo, hi = _tightest((lo, hi), (_stamp(start), _stamp(stop)))

    out: dict[str, str] = {}
    if lo is not None:
        out["time>="] = lo.strftime("%Y-%m-%dT%H:%M:%SZ")
    if hi is not None:
        out["time<="] = hi.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def _stamp(value: Any):
    """Return ``value`` as a tz-naive UTC :class:`pandas.Timestamp`, or ``None``.

    ``None`` for anything that will not read as a time, which is the check standing
    between a non-time ``select`` and a nonsense constraint: ``pd.Timestamp(0)`` is
    a perfectly good 1970, and ``select={"depth": 0}`` must not become one.
    """
    import pandas as pd

    if value is None or isinstance(value, bool):
        return None
    try:
        t = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if t is pd.NaT:
        return None
    return t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t


def _time_span(value: Any) -> tuple[Any, Any]:
    """Return ``(start, stop)`` covering one ``select`` value, ``(None, None)`` if none.

    Follows :func:`ocean_skill.operators.select`'s reading of the same value, so the
    server is asked for exactly what the in-memory pass will keep. In particular a
    partial date is a *span*: ``"2012-01"`` is all of January, and the ``stop`` of
    ``slice("2012-01", "2012-03")`` is the end of March, not its first instant —
    xarray's partial-datetime indexing, which a bare ``Timestamp`` would truncate to
    a single day and quietly drop the rest of the month from the download.
    """
    import pandas as pd

    if isinstance(value, dict):  # the YAML-friendly spelling of a slice
        value = slice(value.get("min"), value.get("max"))
    if isinstance(value, slice):
        return _span_of(value.start)[0], _span_of(value.stop)[1]
    if isinstance(value, list | tuple | set):
        spans = [s for v in value if (s := _span_of(v)) != (None, None)]
        if not spans:
            return None, None
        return min(s[0] for s in spans), max(s[1] for s in spans)
    if isinstance(value, str) and value in ("mean", "surface"):
        # a reduction or a depth keyword that wandered in; not a time
        return None, None
    if hasattr(value, "__array__") and not isinstance(value, pd.Timestamp):
        return _time_span(list(pd.Series(value)))
    return _span_of(value)


def _span_of(value: Any) -> tuple[Any, Any]:
    """Return ``(start, stop)`` for one scalar: a period's extent, or an instant."""
    import pandas as pd

    if isinstance(value, str):
        try:
            period = pd.Period(value)
        except (TypeError, ValueError):
            return None, None
        return _stamp(period.start_time), _stamp(period.end_time)
    stamp = _stamp(value)
    return stamp, stamp


def _tightest(a: tuple[Any, Any], b: tuple[Any, Any]) -> tuple[Any, Any]:
    """Intersect two ``(lo, hi)`` bounds, ignoring the ``None`` ends of either."""
    los = [x for x in (a[0], b[0]) if x is not None]
    his = [x for x in (a[1], b[1]) if x is not None]
    return (max(los) if los else None, min(his) if his else None)
