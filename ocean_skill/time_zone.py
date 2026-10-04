"""Source time zones: declare what a naive timestamp means, and turn it into UTC once.

Everything downstream of a read runs on UTC. Sources do not: a mooring logger writes
local clock times, a CSV export carries no zone at all, an ERDDAP table says ``Z``. A
timestamp with no zone or offset -- *naive* -- used to be taken as UTC, silently, so an
Alaska logger's ``12:00`` AKDT landed eight hours early with nothing to say so. This
module is how a source says what its naive timestamps are (the vocabulary), and the one
place they are turned into UTC (:func:`to_utc`, :func:`localize_naive_datetime64`).

Two declarations, alternatives -- give one:

* ``time_zone: "America/Anchorage"`` -- an IANA zone, daylight saving included: the same
  logger reads UTC-8 in July (AKDT) and UTC-9 in January (AKST). The right choice when
  the record is in local civil time.
* ``utc_offset_h: -9`` -- a fixed offset, **local clock minus UTC, in the ISO-8601
  sense**: Alaska daylight time is ``-8``, Alaska standard time ``-9``, India ``+5.5``.
  A naive stamp is converted as ``UTC = stamp - utc_offset_h``, all year round. The
  right choice for a logger that keeps local *standard* time through the summer, where
  an IANA zone would shift half the record by an hour. :func:`time_zone_label` writes
  it the ISO way too (``UTC-09:00``, ``UTC+05:30``).

Either applies only to **naive** timestamps. One that carries its own offset (``...Z``,
``...-09:00``) already says what it is, and is converted as it stands; a declared zone
that disagrees with the data's own offsets is warned about, not applied. A naive stamp
that a daylight-saving transition makes ambiguous or impossible becomes NaT, loudly:
the clocks repeat an hour going back (which pass is it?) and skip one going forward
(that reading never happened). A chronological record containing both passes through a
repeated hour is read correctly; anything else is set to NaT and counted in a warning.

Pure metadata plus the read-path helper: this imports only the standard library, numpy
and pandas -- nothing from the rest of the package -- so that it can move into a catalog
package unchanged. The key names and canonical forms of the two declarations are a
shared vocabulary (a suggester elsewhere writes values in exactly this form), so they
are kept stable.
"""

from __future__ import annotations

import difflib
import functools
import math
import re
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from numbers import Real
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

import numpy as np

__all__ = [
    "canonicalize_time_zone",
    "has_explicit_zone",
    "localize_naive_datetime64",
    "time_zone_label",
    "to_utc",
    "tzinfo_of",
]

#: Spellings of UTC accepted for ``time_zone`` (case-insensitively): ``Z`` as ISO writes
#: it, and the IANA zones that are UTC under another name. All become ``"UTC"``.
_UTC_NAMES = frozenset(
    {
        "utc",
        "z",
        "gmt",
        "uct",
        "zulu",
        "universal",
        "greenwich",
        "gmt0",
        "gmt+0",
        "gmt-0",
        "etc/utc",
        "etc/gmt",
        "etc/uct",
        "etc/zulu",
        "etc/universal",
        "etc/greenwich",
        "etc/gmt0",
        "etc/gmt+0",
        "etc/gmt-0",
    }
)

#: The widest offset ``utc_offset_h`` accepts (hours); real offsets run -12 to +14.
_MAX_OFFSET_H = 14.0

#: ``utc_offset_h`` must be a multiple of this (hours): every real offset is, down to
#: the quarter-hour (Nepal +5.75, Chatham Islands +12.75).
_OFFSET_STEP_H = 0.25

#: ``"UTC-9"``/``"GMT+5:30"``-style strings -- not a zone name, but plainly an offset,
#: so the unknown-zone message can say how to write it.
_OFFSET_SPELLING = re.compile(
    r"^(?:utc|gmt)?\s*([+-]?\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE
)

#: A zone or offset at the end of a timestamp, *after the time of day* -- the position
#: that makes it an offset rather than a day of the month (``1950-01-01`` ends in
#: ``-01`` and is not one).
_ZONE_TAIL = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|UTC|GMT|[+-]\d{1,2}(?::?\d{2})?)\s*$",
    re.IGNORECASE,
)

#: ``"<unit> since <reference>"``, with ``since`` joined to its neighbours by
#: underscores as well as spaces (the spelling ``Time[days_since_1950-01-01T00:00:00Z]``
#: column names use).
_SINCE = re.compile(r"[_\s]+since[_\s]+", re.IGNORECASE)

#: Datetime resolutions, coarsest to finest.
_UNITS = ("s", "ms", "us", "ns")


# -- the declaration -------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _zone_names() -> dict[str, str]:
    """Map every IANA zone's case-folded name to its canonical spelling."""
    return {name.casefold(): name for name in available_timezones()}


def _unknown_zone(name: str) -> ValueError:
    """Build the error for an unrecognised zone: a near miss, and how to write it."""
    zones = _zone_names()
    if not zones:
        return ValueError(
            f"time_zone: cannot check time zone {name!r}: no IANA time zone database "
            "is available on this system (install the 'tzdata' package)"
        )
    folded = name.casefold()
    # "Anchorage" is not close to "america/anchorage" by edit distance, but it is the
    # zone's own name -- offer suffix matches first, then the nearest spellings.
    near = [zones[key] for key in zones if key.endswith("/" + folded)]
    for key in difflib.get_close_matches(folded, list(zones), n=3, cutoff=0.6):
        if zones[key] not in near:
            near.append(zones[key])
    hint = f" (did you mean {', '.join(repr(z) for z in near[:3])}?)" if near else ""
    advice = (
        "give an IANA zone name such as 'America/Anchorage' (or 'UTC'), or a fixed "
        "offset as utc_offset_h = local clock minus UTC in hours, e.g. -9 for UTC-09:00"
    )
    offset = _OFFSET_SPELLING.match(name.strip())
    if offset:
        sign = -1 if offset.group(1).startswith("-") else 1
        hours = int(offset.group(1)) + sign * int(offset.group(2) or 0) / 60
        advice = f"that looks like an offset: write utc_offset_h: {hours:g} instead"
    elif re.fullmatch(r"[A-Za-z]{2,5}", name):
        advice = (
            "abbreviations such as 'AKST' or 'PDT' are ambiguous and not accepted; "
            + advice
        )
    return ValueError(f"time_zone: unknown time zone {name!r}{hint} -- {advice}")


def _canonical_zone(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(
            "time_zone: time_zone must be an IANA zone name such as "
            f"'America/Anchorage' (or 'UTC'), got {value!r}; for a fixed offset use "
            "utc_offset_h"
        )
    name = value.strip()
    folded = name.casefold()
    if folded in _UTC_NAMES:
        return "UTC"
    canonical = _zone_names().get(folded)
    if canonical is None:
        raise _unknown_zone(name)
    return canonical


def _canonical_offset(value: Any) -> float:
    shown = f"utc_offset_h {value!r}"
    if isinstance(value, bool) or not isinstance(value, (Real, str)):
        hours = math.nan
    else:
        try:
            hours = float(value)
        except ValueError:
            hours = math.nan
    if not math.isfinite(hours):
        raise ValueError(
            f"time_zone: {shown} is not a number of hours -- utc_offset_h is local "
            "clock minus UTC, e.g. -9 for UTC-09:00 or 5.5 for UTC+05:30"
        )
    if abs(hours) > _MAX_OFFSET_H:
        raise ValueError(
            f"time_zone: {shown} is outside +/-{_MAX_OFFSET_H:g} hours -- utc_offset_h "
            "is local clock minus UTC (Alaska standard time is -9, India +5.5)"
        )
    steps = hours / _OFFSET_STEP_H
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(
            f"time_zone: {shown} is not a multiple of {_OFFSET_STEP_H:g} hours -- real "
            "UTC offsets are whole quarter-hours (-9, 5.5, 5.75)"
        )
    return round(steps) * _OFFSET_STEP_H


def canonicalize_time_zone(
    time_zone: str | None = None, utc_offset_h: float | None = None
) -> dict[str, Any]:
    """Validate a source's time declaration and return it in canonical form.

    Returns ``{}`` when nothing is declared, ``{"time_zone": <IANA name>}`` or
    ``{"utc_offset_h": <float hours>}``. The two are alternatives; giving both is a
    ``ValueError``. An IANA name is checked against the system tz database
    (:mod:`zoneinfo`) and its spelling canonicalised, so ``"america/anchorage"`` becomes
    ``"America/Anchorage"``; ``UTC``, ``Etc/UTC``, ``GMT`` and ``Z`` (and the other IANA
    spellings of UTC) become ``"UTC"``. A blank ``time_zone`` is not a declaration.

    ``utc_offset_h`` is the ISO-8601 sign convention: **local clock minus UTC**, so a
    naive stamp becomes UTC by *subtracting* it -- ``-8`` for Alaska daylight time,
    ``-9`` for Alaska standard time, ``+5.5`` for India. It must be finite, within
    ``[-14, 14]`` and a multiple of 0.25 hours. Use it for a record kept in local
    standard time all year; use ``time_zone`` when the clock follows daylight saving.

    Everything wrong -- an unknown zone (with a did-you-mean), an impossible offset,
    both keys -- is a ``ValueError`` whose message starts ``"time_zone: "`` and says
    what is allowed.
    """
    zone = None if isinstance(time_zone, str) and not time_zone.strip() else time_zone
    if zone is not None and utc_offset_h is not None:
        raise ValueError(
            "time_zone: time_zone and utc_offset_h are alternatives -- give one, not "
            f"both (got time_zone={time_zone!r} and utc_offset_h={utc_offset_h!r})"
        )
    if zone is not None:
        return {"time_zone": _canonical_zone(zone)}
    if utc_offset_h is not None:
        return {"utc_offset_h": _canonical_offset(utc_offset_h)}
    return {}


def _declared(meta: Any) -> dict[str, Any]:
    """Return the canonical declaration in ``meta`` (``{}`` when there is none)."""
    if not meta:
        return {}
    return canonicalize_time_zone(meta.get("time_zone"), meta.get("utc_offset_h"))


def _tzinfo(declared: dict[str, Any]) -> tzinfo | None:
    zone = declared.get("time_zone")
    if zone is not None:
        if zone == "UTC":
            return None
        try:
            return ZoneInfo(zone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"time_zone: cannot load time zone {zone!r}: {exc}"
            ) from exc
    hours = declared.get("utc_offset_h")
    if hours is None or hours == 0:
        return None
    return timezone(timedelta(hours=hours))


def _utc_label(hours: float) -> str:
    """Write an offset the ISO way: ``UTC-09:00``, ``UTC+05:30``."""
    minutes = round(abs(hours) * 60)
    sign = "-" if hours < 0 else "+"
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def _label(declared: dict[str, Any]) -> str | None:
    zone = declared.get("time_zone")
    if zone is not None:
        return None if zone == "UTC" else zone
    hours = declared.get("utc_offset_h")
    return None if hours is None or hours == 0 else _utc_label(hours)


def tzinfo_of(meta: Any) -> tzinfo | None:
    """Return the :class:`datetime.tzinfo` ``meta`` declares for naive timestamps.

    Reads ``meta["time_zone"]`` / ``meta["utc_offset_h"]`` (validated, so an invalid
    declaration raises here rather than being trusted because it came from a catalog).
    ``None`` when nothing is declared, or when what is declared is UTC (or an offset of
    zero) -- the same as declaring nothing, so a caller can use ``tzinfo_of(meta) is
    None`` as "no conversion needed".
    """
    return _tzinfo(_declared(meta))


def time_zone_label(meta: Any) -> str | None:
    """Return a printable name for ``meta``'s declared zone, or ``None`` for UTC.

    ``"America/Anchorage"`` for an IANA zone, ``"UTC-09:00"`` / ``"UTC+05:30"`` for a
    fixed offset (the ISO sign: local clock minus UTC); ``None`` when undeclared or UTC.
    """
    return _label(_declared(meta))


def has_explicit_zone(units: str) -> bool:
    """Whether a CF ``"<unit> since <reference>"`` string's reference names a zone.

    True when the reference carries ``Z``, ``UTC``, ``GMT`` or a ``+hh[:mm]`` /
    ``-hh[:mm]`` offset *after the time of day*: ``"days since 1950-01-01T00:00:00Z"``,
    ``"hours since 2000-01-01 00:00:00 -09:00"``, ``"seconds since 1970-01-01 00:00:00
    UTC"``. A date alone (``"days since 1950-01-01"``, whose ``-01`` is a day of the
    month) or a time with no zone is not. Decoded numbers are offsets from the reference
    instant, so with a zone in the reference they are already unambiguous, and a
    declared source zone should not be applied to them a second time.
    """
    if not isinstance(units, str):
        return False
    reference = _SINCE.split(units.strip(), maxsplit=1)[-1].replace("_", " ")
    return _ZONE_TAIL.search(reference.strip()) is not None


# -- parsing ---------------------------------------------------------------------------


@dataclass
class _Parsed:
    """A time column split by what each element carried: all on one positional index.

    ``aware`` holds the UTC instants of elements that carried their own offset, and
    ``offset`` the offset each carried (only collected when a declared zone has to be
    checked against it); ``naive`` holds the clock readings of elements that carried
    none. Each is ``None`` when no element is of that kind, and NaT where one is of
    another kind (or is missing).
    """

    aware: Any = None
    offset: Any = None
    naive: Any = None


def _as_series(values: Any) -> Any:
    """Return ``values`` as a Series (a list, array or lone stamp gets a RangeIndex)."""
    import pandas as pd

    if isinstance(values, pd.Series):
        return values
    dims = np.ndim(values)
    if dims == 0:  # one timestamp / string / datetime64
        return pd.Series([values])
    if dims != 1:
        raise ValueError(
            f"time_zone: times must be one-dimensional, got shape {np.shape(values)}"
        )
    if isinstance(values, (pd.Index, np.ndarray, list, tuple)):
        return pd.Series(values)
    return pd.Series(np.asarray(values))  # any other array-like (a DataArray, ...)


def _is_blank(x: Any) -> bool:
    """Whether an element is missing or an empty string (nothing to parse)."""
    import pandas as pd

    if isinstance(x, str):
        return not x.strip()
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):  # an array-like element
        return False


def _read(series: Any, **kwargs: Any) -> Any:
    """Parse with :func:`pandas.to_datetime`, or ``None`` if it will not go in one pass.

    Never raises: pandas refuses mixed offsets outright (pandas 3), and hands back an
    object Series for mixed-awareness datetimes; both mean "go element by element".
    Unparseable values come back NaT (``errors="coerce"``), which the caller cannot tell
    from missing ones -- see :func:`_unparsed`.
    """
    import pandas as pd

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors="coerce", **kwargs)
    except (ValueError, TypeError, OverflowError):
        return None
    if not pd.api.types.is_datetime64_any_dtype(parsed.dtype):
        return None
    return parsed


def _wrap(parsed: Any, want_offsets: bool) -> _Parsed:
    """Split one datetime Series (all tz-aware or all naive) into a :class:`_Parsed`."""
    import pandas as pd

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        utc = parsed.dt.tz_convert("UTC")
        offset = None
        if want_offsets:
            offset = parsed.dt.tz_localize(None) - utc.dt.tz_localize(None)
        return _Parsed(aware=utc, offset=offset)
    return _Parsed(naive=parsed)


def _unparsed(series: Any, parsed: Any) -> Any:
    """Flag the rows that hold something but that ``parsed`` left NaT.

    pandas infers *one* format from the first element and applies it to the rest, so a
    column that mixes ``...00Z`` and ``...00.500Z``, a date-only value among date-times,
    or a naive stamp among aware ones comes back with NaT for the minority -- the same
    NaT it gives a blank cell. Blank cells are not worth another attempt; these are.
    ``parsed=None`` (nothing was parsed) flags every non-blank row.
    """
    import pandas as pd

    if parsed is None:
        missing = np.ones(len(series), dtype=bool)
    else:  # a copy: under copy-on-write the view pandas hands back is read-only
        missing = parsed.isna().to_numpy(dtype=bool, copy=True)
    missing &= ~series.isna().to_numpy()  # a null cell has nothing to parse
    rows = np.flatnonzero(missing)  # real cells that parsed to NaT: blank, or lost
    if rows.size:
        cells = series.iloc[rows].to_numpy(dtype=object)
        blank = np.fromiter((_is_blank(x) for x in cells), dtype=bool, count=rows.size)
        missing[rows[blank]] = False
    return pd.Series(missing, index=series.index)


def _scalar(x: Any) -> Any:
    """Parse one element to a datetime (aware or naive), or NaT if it will not.

    Strings are read as ISO 8601 and nothing looser: this is the rescue path for a
    column whose elements differ (fractional seconds here, an offset there), and a
    forgiving parser would turn junk into plausible dates -- ``"12:00"`` into *today*
    at noon -- where a strict one leaves a NaT. ``datetime.fromisoformat`` is also the
    fast way to do it, which is the whole cost of this path.
    """
    import pandas as pd

    if x is pd.NaT:
        return pd.NaT
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.strip())
        except ValueError:
            return pd.NaT
    if isinstance(x, (datetime, date, np.datetime64)):
        return pd.Timestamp(x)
    return pd.NaT


def _elementwise(rows: Any, want_offsets: bool) -> _Parsed:
    """Parse ``rows`` one element at a time: the slow path for what has no one format.

    Each distinct value is parsed once (a sentinel repeated a million times costs one
    parse). Aware elements are converted to UTC instants as they stand; naive ones are
    kept as clock readings for the caller to localise.
    """
    import pandas as pd

    cache: dict[Any, Any] = {}
    aware, offset, naive = [], [], []
    for x in rows.to_numpy(dtype=object):
        try:
            stamp = cache[x]
        except KeyError:
            stamp = cache[x] = _scalar(x)
        except TypeError:  # unhashable cell: parse it without the cache
            stamp = _scalar(x)
        if stamp is pd.NaT:
            aware.append(pd.NaT)
            offset.append(pd.NaT)
            naive.append(pd.NaT)
        elif stamp.tzinfo is None:
            aware.append(pd.NaT)
            offset.append(pd.NaT)
            naive.append(stamp)
        else:
            aware.append(stamp.astimezone(UTC))
            offset.append(stamp.utcoffset())
            naive.append(pd.NaT)

    def column(cells: list, **kwargs: Any) -> Any:
        return pd.to_datetime(
            pd.Series(cells, index=rows.index, dtype=object), **kwargs
        )

    out = _Parsed()
    any_aware = any(c is not pd.NaT for c in aware)
    if any_aware:
        out.aware = column(aware, utc=True)
        if want_offsets:
            out.offset = pd.to_timedelta(
                pd.Series(offset, index=rows.index, dtype=object)
            )
    if not any_aware or any(c is not pd.NaT for c in naive):
        out.naive = column(naive)
    return out


def _fill(first: Any, second: Any) -> Any:
    """Return ``first`` where it has a value, else ``second``; same kind, same index."""
    unit = max(first.dt.unit, second.dt.unit, key=_UNITS.index)
    return first.dt.as_unit(unit).fillna(second.dt.as_unit(unit))


def _combine(first: Any, second: Any, index: Any) -> Any:
    """Merge two optional parts; ``second`` may cover only some of ``index``."""
    if first is None and second is None:
        return None
    if first is None:
        return second.reindex(index)
    if second is None:
        return first
    return _fill(first, second.reindex(index))


def _merge(base: _Parsed, extra: _Parsed, index: Any) -> _Parsed:
    return _Parsed(
        aware=_combine(base.aware, extra.aware, index),
        offset=_combine(base.offset, extra.offset, index),
        naive=_combine(base.naive, extra.naive, index),
    )


def _rescue(rows: Any, want_offsets: bool) -> _Parsed:
    """Parse rows the first pass lost: ISO 8601 in any variation, then one by one."""
    iso = _read(rows, format="ISO8601")
    out = _wrap(iso, want_offsets) if iso is not None else _Parsed()
    pending = _unparsed(rows, iso)
    if pending.any():
        out = _merge(out, _elementwise(rows[pending], want_offsets), rows.index)
    return out


def _parse(series: Any, want_offsets: bool) -> _Parsed:
    """Split ``series`` (positional index) into aware instants and naive clock times."""
    import pandas as pd

    parsed = _read(series)
    pending = _unparsed(series, parsed)
    out = _wrap(parsed, want_offsets) if parsed is not None else _Parsed()
    if pending.any():
        out = _merge(out, _rescue(series[pending], want_offsets), series.index)
    if out.aware is None and out.naive is None:  # an all-blank column
        out.naive = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return out


# -- localising ------------------------------------------------------------------------


def _localize(naive: Any, tz: tzinfo | None) -> tuple[Any, int, int]:
    """Read naive clock times as ``tz`` local time; return UTC and what DST cost.

    Returns ``(utc, nonexistent, ambiguous)``: the tz-aware UTC Series (``tz=None``
    meaning the clock readings already are UTC), and how many rows a daylight-saving
    transition left as NaT -- readings in the hour the clocks skipped, and readings in
    the hour they repeated that the record cannot place. pandas can tell the two passes
    through a repeated hour apart only from a chronological run that contains both
    (``ambiguous="infer"``); when it cannot, every ambiguous reading becomes NaT.
    """
    if tz is None:
        return naive.dt.tz_localize("UTC"), 0, 0
    try:
        local = naive.dt.tz_localize(tz, ambiguous="infer", nonexistent="NaT")
    except Exception:  # pandas: ValueError (an AmbiguousTimeError in older versions)
        local = naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    utc = local.dt.tz_convert("UTC")
    lost = (naive.notna() & utc.isna()).to_numpy()
    if not lost.any():
        return utc, 0, 0
    # Tell the two losses apart by reading the lost rows as "daylight time": a reading
    # in the repeated hour then has an answer, one in the skipped hour still has none.
    probe = naive[lost].dt.tz_localize(
        tz, ambiguous=np.ones(int(lost.sum()), dtype=bool), nonexistent="NaT"
    )
    nonexistent = int(probe.isna().sum())
    return utc, nonexistent, int(lost.sum()) - nonexistent


def _dst_message(
    subject: str, label: str, nonexistent: int, ambiguous: int, total: int
) -> str:
    parts = []
    if nonexistent:
        parts.append(
            f"{nonexistent} fall in the hour the clocks skipped going onto daylight "
            "saving (that clock time never happened)"
        )
    if ambiguous:
        parts.append(
            f"{ambiguous} fall in the hour the clocks repeated going back (which pass "
            "they belong to cannot be told from the record)"
        )
    return (
        f"{subject}: {nonexistent + ambiguous} of {total} timestamp(s) cannot be "
        f"placed in time zone {label}: {'; '.join(parts)}. They are set to NaT. If "
        "this record keeps local standard time all year, declare utc_offset_h (local "
        "clock minus UTC, e.g. -9 for Alaska standard time) instead of time_zone."
    )


def _offset_conflicts(parsed: _Parsed, tz: tzinfo | None) -> list[timedelta]:
    """List the distinct own offsets that disagree with ``tz``'s at those instants."""
    if tz is None or parsed.aware is None or parsed.offset is None:
        return []
    import pandas as pd

    instants = parsed.aware
    declared = instants.dt.tz_convert(tz).dt.tz_localize(None)
    declared = declared - instants.dt.tz_convert(None)
    differs = parsed.offset.notna() & (parsed.offset != declared)
    if not differs.any():
        return []
    return sorted(
        {pd.Timedelta(v).to_pytimedelta() for v in parsed.offset[differs].unique()}
    )


def _conflict_message(subject: str, label: str, offsets: list[timedelta]) -> str:
    shown = ", ".join(_utc_label(o.total_seconds() / 3600) for o in offsets[:4])
    shown += ", ..." if len(offsets) > 4 else ""
    return (
        f"{subject}: declares time_zone {label}, but its timestamps carry their own "
        f"UTC offset(s) ({shown}) -- using the timestamps' own offsets; the declared "
        "zone only applies to naive (offset-free) timestamps."
    )


def to_utc(values: Any, meta: Any = None, *, subject: str = "this source") -> Any:
    """Return ``values`` as a tz-aware UTC pandas Series, via ``meta``'s declared zone.

    The single place a source's timestamps become UTC. ``values`` is a Series (its
    index and name are kept) or any list/array of strings, datetimes or ``datetime64``
    (it gets a RangeIndex). Strings are parsed with :func:`pandas.to_datetime`; blank,
    missing and unparseable cells are NaT. pandas reads *one* format off the first
    element and drops whatever does not match, so rows it loses are read again as ISO
    8601 in any variation -- fractional seconds, a date alone, naive stamps among
    offset-carrying ones, different offsets in one column -- instead of vanishing. Other
    formats are read only as pandas reads them: junk stays NaT rather than becoming a
    plausible date.

    What a stamp means depends on what it carries:

    * **naive** -- read as local time in ``meta``'s ``time_zone`` or ``utc_offset_h``
      (local clock minus UTC: ``UTC = stamp - utc_offset_h``, so ``-9`` turns ``12:00``
      into ``21:00`` UTC and ``+5.5`` turns it into ``06:30``), then converted to UTC.
      An IANA zone follows daylight saving: ``America/Anchorage`` is UTC-8 in July and
      UTC-9 in January. Readings that a transition makes impossible (the skipped hour)
      or ambiguous without a chronological repeat to resolve them become NaT, with one
      ``UserWarning`` naming the count and the zone. With nothing declared the stamps
      are taken as UTC, as they always were.
    * **tz-aware** (an offset in the string, or a tz-aware dtype) -- converted to UTC
      as it stands, never shifted by the declared zone. If a zone is declared and the
      data's own offsets differ from it at those instants, one ``UserWarning`` says the
      declared zone only applies to naive timestamps.

    ``subject`` names the source in those warnings. Invalid ``meta`` declarations raise
    ``ValueError`` (see :func:`canonicalize_time_zone`). The result's resolution is
    whatever pandas produced (``us`` in pandas 3, ``ns`` before), not forced to ``ns``.
    """
    decl = _declared(meta)
    tz, label = _tzinfo(decl), _label(decl)
    series = _as_series(values)
    index, name = series.index, series.name
    parsed = _parse(series.reset_index(drop=True), want_offsets=tz is not None)

    utc = parsed.aware
    nonexistent = ambiguous = total = 0
    if parsed.naive is not None:
        local, nonexistent, ambiguous = _localize(parsed.naive, tz)
        utc = local if utc is None else _fill(utc, local)
        total = int(parsed.naive.notna().sum())
    if nonexistent or ambiguous:
        warnings.warn(
            _dst_message(subject, label, nonexistent, ambiguous, total), stacklevel=2
        )
    conflicts = _offset_conflicts(parsed, tz)
    if conflicts:
        warnings.warn(_conflict_message(subject, label, conflicts), stacklevel=2)

    utc = utc.set_axis(index)
    utc.name = name
    return utc


def localize_naive_datetime64(
    values: Any, meta: Any, *, subject: str = "this source"
) -> np.ndarray:
    """Read a naive ``datetime64`` array as local time and return it as naive UTC.

    For a time coordinate that arrived as xarray ``datetime64`` -- always naive, so it
    has no way to say what zone it is in -- from a source that declares one. The values
    are read as local clock times in ``meta``'s ``time_zone`` or ``utc_offset_h`` (local
    clock minus UTC; see :func:`to_utc`, including its daylight-saving policy and the
    one ``UserWarning`` for readings it must set to NaT) and returned as naive UTC
    ``datetime64[ns]`` (another unit only if a date is outside what ``ns`` can hold).
    With nothing declared, or UTC, ``values`` come back unchanged whatever their dtype.
    NaT is kept. Non-``datetime64`` times (cftime) cannot be localised, so a declared
    zone with those raises ``ValueError`` rather than being silently skipped.
    """
    import pandas as pd

    decl = _declared(meta)
    tz, label = _tzinfo(decl), _label(decl)
    array = np.asarray(values)
    if tz is None:
        return array
    if array.dtype.kind != "M":
        raise ValueError(
            f"time_zone: cannot apply the declared time zone {label} to times of dtype "
            f"{array.dtype} -- only datetime64 times can be localised"
        )
    utc, nonexistent, ambiguous = _localize(pd.Series(array.ravel()), tz)
    if nonexistent or ambiguous:
        total = int(np.count_nonzero(~np.isnat(array)))
        warnings.warn(
            _dst_message(subject, label, nonexistent, ambiguous, total), stacklevel=2
        )
    naive = utc.dt.tz_convert(None)
    try:
        naive = naive.dt.as_unit("ns")
    except (OverflowError, ValueError):  # a date `ns` cannot hold: keep its own unit
        pass
    return naive.to_numpy().reshape(array.shape)
