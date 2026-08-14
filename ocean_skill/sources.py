"""Reading sources: ``osk.read`` opens an intake v2 catalog entry, standardized.

Opens the entry with intake (``cat[name].read()``), then standardizes: ROMS entries
(``metadata.model == "roms"``) route through :mod:`ocean_skill.roms`; other gridded/obs
entries get a light CF rename from the entry's ``standard_names`` map (fuller handling
lives in :mod:`ocean_skill.cf`). Returns a **known type** by featureType: point
featureTypes → :class:`pandas.DataFrame`; gridded/multidim → :class:`xarray.Dataset`.
"""

from __future__ import annotations

from typing import Any

from ocean_skill.catalog import SourceRef, resolve

__all__ = ["erddap_constraints", "read"]


def read(source: str | SourceRef, **kwargs: Any):
    """Open a catalog source and return a CF-standardized Dataset or DataFrame.

    Parameters
    ----------
    source
        An entry name (``"glodap"`` or ``"catalog:name"``) or a :class:`SourceRef`.
    **kwargs
        Reader keywords, overriding the entry's own. The one that earns this is
        ``constraints=`` on an ERDDAP table: a mooring's whole record is a large
        download, and ``osk.read(entry, constraints={"time>=": "2015-01-01"})``
        subsets it *server-side*, where a later ``select={"time": ...}`` cannot.
        These used to be accepted and silently discarded.
    """
    import intake

    ref = source if isinstance(source, SourceRef) else resolve(source)
    meta = ref.metadata

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
