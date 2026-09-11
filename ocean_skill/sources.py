"""Reading sources: ``osk.read`` opens an intake v2 catalog entry, standardized.

Opens the entry with intake (``cat[name].read()``), then standardizes: ROMS entries
(``metadata.model == "roms"``) route through :mod:`ocean_skill.roms`; other gridded/obs
entries get a light CF rename from the entry's ``standard_names`` map (fuller handling
lives in :mod:`ocean_skill.cf`). Returns a **known type** by featureType: point
featureTypes → :class:`pandas.DataFrame`; gridded/multidim → :class:`xarray.Dataset`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from ocean_skill.catalog import SourceRef, resolve

__all__ = ["erddap_constraints", "read", "read_time_axis"]

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

    if meta.get("model") == "roms" or meta.get("loader") == "ocean_skill.roms":
        from ocean_skill import roms

        return roms.standardize(obj, meta)

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
    if is_frame and tname and tname in obj.columns:
        import pandas as pd

        decoded = pd.to_datetime(obj[tname], utc=True, errors="coerce")
        obj = obj.assign(**{tname: decoded})
    elif not is_frame and tname and tname in getattr(obj, "variables", {}):
        from ocean_skill.build import _decode_times

        decoded = _decode_times(obj, obj[tname])
        if decoded is not None:
            obj = obj.assign_coords({tname: decoded})
    return obj


def read_time_axis(source: str | SourceRef):
    """Open ``source`` and decode only its time axis -- for a caller (see
    :func:`ocean_skill.comparison._time_bins`) that only needs the decoded time
    coordinate to enumerate or window calendar bins, not the comparable fields
    themselves.

    The ordinary :func:`read` pays for far more than that on a ROMS entry: its
    :func:`ocean_skill.roms.standardize` unconditionally derives true geographic
    east/north velocity from the staggered grid-relative components
    (:func:`ocean_skill.roms._add_geographic_velocity`) -- a dask task graph that
    scales with the whole history file's chunk count and can cost tens of seconds to
    *build*, well before anything is computed or even the requested variable is
    known. None of that is needed just to read off ``time``, so a ROMS source is
    standardized here with ``derive_velocity=False`` instead.

    Not memoized in :data:`_READ_CACHE`: unlike :func:`read`, the object this
    returns is missing the derived-velocity variables, and must never be handed
    back in place of the genuine, fully-standardized result a later call to
    :func:`read` for the same source is entitled to.

    Falls back to the ordinary (memoized) :func:`read` for anything that is not a
    ROMS source -- the eager cost this bypasses is ROMS-specific.
    """
    ref = source if isinstance(source, SourceRef) else resolve(source)
    meta = ref.metadata
    if meta.get("model") != "roms" and meta.get("loader") != "ocean_skill.roms":
        return read(source)

    import intake

    from ocean_skill import roms

    cat = intake.from_yaml_file(str(ref.path))
    entry = cat[ref.name]
    obj = entry.read()
    return roms.standardize(obj, meta, derive_velocity=False)


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
