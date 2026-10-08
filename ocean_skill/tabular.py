"""Tabular (point) sources: the DataFrame vocabulary, and DataFrame -> Dataset.

Point featureTypes come back from intake as a :class:`pandas.DataFrame` (see
:func:`ocean_skill.sources.read`, whose contract that is). Everything downstream of the
read — :func:`ocean_skill.operators.resolve_variable`, the operators, ``align``,
``metrics`` — speaks xarray, so a station lane converts here, once, on its way into
:func:`ocean_skill.comparison._prepare`.

The conversion is *not* done in ``sources.read``: the documented return type for a point
source is a DataFrame, and callers reading a mooring by hand want the frame. It is not
done in ``align`` either (despite what that module's docstring used to say), because a
:class:`~ocean_skill.field.Field` over one station never reaches align and wants the
same treatment — and because the conversion reads the *catalog entry's* metadata, which
align never sees.

This module also owns the column-naming vocabulary tabular sources use --
``"<name> (<units>)"``/``"<name>[<units>]"``, ``<name>_qc_agg``/``<name>_qc_tests``
(or a name containing "flag") QARTOD companions, and which columns are coordinates
(time/lon/lat/depth, recognized via :data:`ocean_skill.vocabulary.COORD_VOCABULARY`
-- see :func:`coord_column` -- not just the exact ERDDAP spelling) rather than data.
:mod:`ocean_skill.build` imports those helpers rather than keeping its own copy, so
the catalog that *describes* a table and the code that *reads* one cannot disagree
about the convention.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, NamedTuple

import numpy as np

from ocean_skill import _stacklevel, depth_convention, time_zone, vocabulary

__all__ = [
    "COORD_COLUMNS",
    "apply_table_options",
    "canonicalize_time_options",
    "coord_axis_of",
    "coord_column",
    "decode_time_column",
    "depth_of",
    "is_coordinate_column",
    "is_frame",
    "is_qc_column",
    "joined_time_column",
    "numeric_in_range",
    "split_units",
    "to_dataset",
]

#: Column names that are coordinates, not comparable data. Superseded by
#: :func:`coord_axis_of`, which also catches the ``Latitude[degrees_north]``-style
#: spellings this fixed set cannot; kept (and still exported) because it is a plain,
#: dependency-free way to ask "is this exact base a coordinate" without importing the
#: regex table below.
COORD_COLUMNS = frozenset({"time", "latitude", "longitude", "z", "depth", "altitude"})

#: Suffixes ``intake_erddap`` gives a variable's QARTOD companion columns.
_QC_SUFFIXES = ("_qc_agg", "_qc_tests")

#: A QARTOD/QC flag column by name, independent of the intake_erddap suffixes above --
#: some tables call their flag columns ``Temperature_flag`` or ``QC_Flag`` rather than
#: ``..._qc_agg``. ``flag`` as a whole token only, split on any run of non-alphanumeric
#: characters (underscore included, unlike ``\b`` -- ``\b`` does not see a boundary
#: inside ``Temperature_flag``, since ``_`` counts as a word character) so
#: ``flagellate_abundance`` is never caught by it.
_QC_NAME = re.compile(r"(?:^|[^0-9A-Za-z])flag(?:[^0-9A-Za-z]|$)", re.IGNORECASE)

#: ``"<name> (<units>)"`` — the spelling ``intake_erddap``'s readers produce. Any
#: amount of whitespace (including none) between the name and the paren is accepted
#: and discarded (``\s*``, non-greedy name so a stray extra space never leaks into
#: the returned name) — real files are not consistent about the single space the
#: convention nominally calls for.
_COLUMN_UNITS = re.compile(r"^(.+?)\s*\((.+)\)$")

#: ``"<name>[<units>]"`` — brackets instead of parens, same whitespace tolerance.
#: Seen on mooring CSVs that pack units into the column name this way (e.g.
#: ``"Salinity_qc[PSU]"``, ``"Time[days_since_1950-01-01T00:00:00Z]"``). Tried as a
#: fallback in :func:`split_units`, after the parenthesized form, so a name that
#: happens to end in a literal ``[...]`` under the ERDDAP convention (unseen so far)
#: would still prefer that reading.
_COLUMN_UNITS_BRACKET = re.compile(r"^(.+?)\s*\[(.+)\]$")

#: Depth-bearing columns, most direct first, each with the spellings accepted for it
#: (matched case-insensitively against :func:`split_units`'s base — so
#: ``"Depth[m]"``/``"DEPTH"`` reach the same tier as ``"depth"``). Pressure is last
#: because it is a *conversion*, not a reading — see :func:`depth_of` — and accepts a
#: couple of shorter spellings (``pressure``, ``pres``) beyond the ERDDAP name.
#: :data:`_DEPTH_COLUMNS`, the tuple of tier *names* (not spellings) used to report
#: ``depth_source`` and to rank ``reading`` over ``pressure``, is derived from this so
#: the two cannot drift apart.
_DEPTH_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "depth": ("depth",),
    "depth_reading": ("depth_reading",),
    "z": ("z",),
    "sea_water_pressure": ("sea_water_pressure", "pressure", "pres"),
}
_DEPTH_COLUMNS = tuple(_DEPTH_COLUMN_ALIASES)

#: Whether a column's axis is recognized by name at all is decided by
#: :data:`ocean_skill.vocabulary.COORD_VOCABULARY` (via :func:`vocabulary.
#: matches_axis`) -- one table, shared with :mod:`ocean_skill.cf`'s gridded-name
#: fallback matching, so the two matchers cannot drift apart. ``Z``'s ranking of a
#: direct depth/z reading over a pressure conversion is the vocabulary's ``direct``
#: entry, the same ranking :data:`_DEPTH_COLUMN_ALIASES` encodes for
#: :func:`depth_of`; ``Z``'s ``exclude`` entry (``"bottom"``) is what keeps
#: ``Depth_bottom``/``bottom_depth`` -- the seafloor/station depth, a data column --
#: from ever being claimed as the vertical coordinate just because "depth" appears
#: in the name.

#: ``altitude`` is excluded from ``variables`` like the other coordinate names, but
#: is not claimed as a ``Z`` axis: unlike depth/pressure it has no agreed-on sign
#: convention here (positive up vs. positive down), so guessing which one a given
#: file means would be worse than leaving it unassigned.
_ALTITUDE = "altitude"

#: Entry-metadata keys that state a depth when the data does not.
_DEPTH_ATTRS = ("nominal_depth_m", "depth", "geospatial_vertical_min")

#: How far a "fixed" instrument may range before it is really a profiler (metres).
FIXED_DEPTH_TOLERANCE = 2.0

#: How far two depth sources may disagree before saying so (metres).
DEPTH_DISAGREEMENT_TOLERANCE = 1.0

#: How far a "fixed" station may wander before it is really a trajectory (degrees).
FIXED_POSITION_TOLERANCE = 1e-4


def is_frame(obj) -> bool:
    """Whether ``obj`` is a DataFrame-like table rather than an xarray object.

    Duck-typed on ``.columns`` rather than importing pandas to run an ``isinstance``:
    the point is to route on shape, and this is the test the read path already uses.
    """
    return hasattr(obj, "columns")


def is_qc_column(name) -> bool:
    """Whether ``name`` is a variable's QARTOD companion rather than a variable.

    Two independent tells: the ``intake_erddap`` QARTOD suffixes
    (``..._qc_agg``/``..._qc_tests``), checked against the raw name since they are not
    something :func:`split_units` would ever strip as units; and the word ``flag``
    appearing anywhere in the units-stripped base (``Temperature_flag``,
    ``QC_Flag[1]``) — tables that are not ERDDAP-sourced tend to spell their QC
    companions this way instead.
    """
    raw = str(name)
    if raw.endswith(_QC_SUFFIXES):
        return True
    base, _ = split_units(raw)
    return bool(_QC_NAME.search(base))


def split_units(column) -> tuple[str, str | None]:
    """Split ``"temperature (degC)"`` or ``"Salinity_qc[PSU]"`` into name and units.

    A column with neither suffix returns ``(name, None)`` — multiple spellings occur,
    sometimes in one frame: :func:`ocean_skill.sources.read` renames the columns a
    catalog entry's ``standard_names`` map covers (leaving them bare) and leaves the
    coordinate columns alone (keeping their suffix).
    """
    match = _COLUMN_UNITS.match(str(column)) or _COLUMN_UNITS_BRACKET.match(
        str(column)
    )
    return (match.group(1), match.group(2)) if match else (str(column), None)


#: Values a coordinate axis cannot physically take, keyed by axis. Real-world CSV
#: exports (SEANOE's mooring tables among them) commonly flag "not reported" with a
#: round-number sentinel like ``9999`` rather than leaving the cell blank -- it is not
#: NaN, so :func:`pandas.to_numeric`'s ``errors="coerce"`` never catches it, and a
#: plain ``.min()``/``.max()`` (or a "does this position vary" check) takes the fill
#: value as if it were a real extreme. Longitude/latitude have hard physical bounds to
#: check against; depth/pressure has no equally sharp one, so the ceiling here is a
#: generous "past the deepest trench on Earth" rather than anything dataset-specific.
_AXIS_BOUNDS: dict[str, tuple[float, float]] = {
    "X": (-360.0, 360.0),  # longitude, either sign convention
    "Y": (-90.0, 90.0),  # latitude
    # Depth/pressure, *either sign convention*: Challenger Deep either way up. A source
    # that stores depth as a height (negative down) used to lose everything deeper than
    # 100 m to a one-sided bound; a genuine sentinel is still caught by _FILL_SENTINELS
    # and _FILL_MAGNITUDE below.
    "Z": (-11000.0, 11000.0),
}

#: Round-number "not reported" markers seen in ocean CSV/NetCDF exports, and their
#: negatives. Longitude/latitude fall outside :data:`_AXIS_BOUNDS` and are caught by the
#: range check alone, but a depth/pressure fill of ``9999`` sits *inside* the Z range (a
#: real reading could be 9999 dbar), so the sentinel set is what catches it there. Kept
#: to the widely-used round values rather than guessing: a real datum equal to one of
#: these is far rarer than the fill convention it denotes.
_FILL_SENTINELS = frozenset(
    {9999.0, 99999.0, 999999.0, -999.0, -9999.0, -99999.0, -999999.0}
)

#: NetCDF's default floating fill (``9.96920996839e+36``) and kin: any value this large
#: in magnitude is a fill, never a coordinate. A single threshold catches the family
#: (``1e20``, ``1e37``, ...) without enumerating each.
_FILL_MAGNITUDE = 1e10


def numeric_in_range(series, axis: str):
    """Return ``series`` as numeric, with non-numeric, out-of-range, and fill values NaN.

    Same job as ``pandas.to_numeric(series, errors="coerce")`` plus a fill filter: a
    value outside :data:`_AXIS_BOUNDS` for ``axis``, equal to a :data:`_FILL_SENTINELS`
    marker, or larger than :data:`_FILL_MAGNITUDE`, is a "not reported" placeholder
    rather than a real reading and is dropped the same way NaN already is. Used
    everywhere an axis's min/max/spread is computed, so a stray ``9999`` cannot
    masquerade as this mooring's northernmost latitude or its deepest reading. The
    bound for ``"Z"`` is symmetric, so depths stored negative (heights) survive too.
    """
    import pandas as pd

    values = pd.to_numeric(series, errors="coerce")
    lo, hi = _AXIS_BOUNDS[axis]
    ok = (
        values.between(lo, hi)
        & (values.abs() < _FILL_MAGNITUDE)
        & ~values.isin(_FILL_SENTINELS)
    )
    return values.where(ok)


#: ``"<n> since <date>"`` with "since" joined to its neighbors by underscores instead
#: of the CF-standard spaces -- the spelling seen on mooring CSVs whose column names
#: pack units into brackets (``"Time[days_since_1950-01-01T00:00:00Z]"``, from
#: :data:`_COLUMN_UNITS_BRACKET`). Normalized to the CF spelling before being handed to
#: xarray's own decoder.
_CF_SINCE = re.compile(r"[_\s]+since[_\s]+", re.IGNORECASE)


def _subject_of(meta, df=None) -> str:
    """Name a source for a warning: its dataset ID, else its title, else its entry name.

    The entry name is only there when whoever read the table stamped it on
    ``df.attrs["source_name"]`` (the catalog name; a table built by hand has none), so
    a catalog entry with no ``datasetID``/``title`` -- most of them -- is still
    named rather than called "this source".
    """
    meta = meta or {}
    named = getattr(df, "attrs", None) or {}
    return (
        meta.get("datasetID")
        or meta.get("title")
        or named.get("source_name")
        or "this source"
    )


def decode_time_column(series, column, meta=None, *, subject=None):
    """Decode a time column to UTC ``datetime64``: CF units, and a declared zone.

    A time column's name sometimes states its encoding explicitly, e.g.
    ``"Time[days_since_1950-01-01T00:00:00Z]"``. :func:`pandas.to_datetime` does not
    know that convention, and given the raw numbers it reads them as nanoseconds since
    the Unix epoch instead -- every timestamp lands within a heartbeat of
    1970-01-01, which is silent rather than an error. Detected here from
    :func:`split_units` and decoded with xarray's calendar-aware CF decoder
    (:func:`xarray.coding.times.decode_cf_datetime`) instead. A column with no such
    units (already timestamps, or a plain date string) is parsed by
    :func:`ocean_skill.time_zone.to_utc`.

    ``meta`` is the entry's metadata, read for its declared source time zone
    (``time_zone`` / ``utc_offset_h``; see :mod:`ocean_skill.time_zone`). A *naive*
    timestamp -- one with no offset of its own, which is every plain ``"2024-07-01
    12:00"`` -- is read as local time in that zone and converted to UTC; with nothing
    declared it is taken as UTC, as it always was. A timestamp that carries its own
    offset is never shifted. The CF branch follows the same rule: numbers counted from a
    reference that names a zone (``...since 1950-01-01T00:00:00Z``) are already
    unambiguous, and a reference with no zone is read as the declared zone's local time.

    The result is always a tz-aware UTC Series on ``series``'s own index. A column that
    is *already* tz-aware (``sources.read`` has converted the entry's time column on the
    way in, and :func:`to_dataset` decodes it again) is only converted to UTC: it states
    what it is, so the declared zone must not apply to it a second time.
    """
    import pandas as pd

    tz = time_zone.tzinfo_of(meta)  # (validates the declaration: raises on a bad one)
    if isinstance(getattr(series, "dtype", None), pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    subject = subject or _subject_of(meta)
    index = getattr(series, "index", None)
    _, units = split_units(str(column))
    if units and _CF_SINCE.search(units):
        try:
            import xarray as xr

            normalized = _CF_SINCE.sub(" since ", units)
            vals = pd.to_numeric(series, errors="coerce").to_numpy()
            decoded = xr.coding.times.decode_cf_datetime(vals, normalized)
            # A Series (not the DatetimeIndex pd.to_datetime returns for an array),
            # keyed on the original index -- callers do frame-aligned `.dt`/`.notna()`
            # on the result, exactly as they would on the generic fallback.
            decoded = pd.Series(decoded, index=index)
            if tz is None or time_zone.has_explicit_zone(normalized):
                return pd.to_datetime(decoded, utc=True)
            return time_zone.to_utc(decoded, meta, subject=subject)
        except Exception:
            pass  # not actually CF-decodable (e.g. "since" inside a longer word) --
            # fall through to generic parsing below
    return time_zone.to_utc(series, meta, subject=subject)


# -- table read options ----------------------------------------------------------------

#: The frame ``attrs`` key :func:`apply_table_options` records the name of the column it
#: joined the entry's ``time_columns`` into, so a later step finds that column by name
#: instead of guessing which of two ``time``-ish columns it is -- and so a frame that
#: has already been through it is recognised and left alone.
_JOINED_TIME_ATTR = "time_joined_column"

#: What the joined column is called, and what it is called instead when the table
#: already has a (different) ``time`` column that must not be overwritten.
_JOINED_TIME = "time"
_JOINED_TIME_FALLBACK = "time_joined"


def canonicalize_time_options(
    time_columns: Any = None, time_format: Any = None
) -> dict[str, Any]:
    """Validate a source's ``time_columns`` / ``time_format``; return them canonically.

    ``time_columns`` names two or more columns of the table that together state the
    time -- ``["Date", "Time"]`` for a logger that writes ``2024-07-01`` and ``12:00``
    in separate columns -- and ``time_format`` is the :func:`strptime
    <datetime.datetime.strptime>` format of the *joined* text (the components joined by
    a single space), for the day a format cannot be guessed (``"%d/%m/%Y %H:%M"``).
    Returns ``{}`` when neither is declared, else ``{"time_columns": [...]}`` plus
    ``"time_format"`` when given. A ``time_format`` has nothing to apply to without
    ``time_columns``, so giving it alone is an error rather than a silent no-op.

    Every problem is a ``ValueError`` whose message starts ``"time_columns: "`` or
    ``"time_format: "`` and says what is allowed.
    """
    if time_columns is None and time_format is None:
        return {}
    if time_columns is None:
        raise ValueError(
            "time_format: time_format only applies together with time_columns (the "
            "format of the joined date/time text), e.g. time_columns=['Date', 'Time'], "
            "time_format='%d/%m/%Y %H:%M'"
        )
    if isinstance(time_columns, str) or not isinstance(time_columns, (list, tuple)):
        # ValueError, not TypeError: every declaration here fails the same way
        raise ValueError(  # noqa: TRY004
            "time_columns: give a list of two or more column names that together state "
            f"the time, e.g. ['Date', 'Time'], got {time_columns!r}"
        )
    columns = [c for c in time_columns if isinstance(c, str) and c.strip()]
    if len(columns) != len(time_columns) or len(columns) < 2:
        raise ValueError(
            "time_columns: give a list of two or more column names that together state "
            f"the time, e.g. ['Date', 'Time'], got {list(time_columns)!r} (a single "
            "time column needs no joining: name it with axes={'T': ...} instead)"
        )
    out: dict[str, Any] = {"time_columns": list(columns)}
    if time_format is not None:
        if not isinstance(time_format, str) or not time_format.strip():
            raise ValueError(
                "time_format: give a strptime format string for the joined date/time "
                f"text, e.g. '%d/%m/%Y %H:%M', got {time_format!r}"
            )
        out["time_format"] = time_format
    return out


def apply_table_options(df, meta, *, subject: str = "this source"):
    """Return ``df`` with the entry's table read options applied.

    Today the one option is ``time_columns``: a table whose time is split over several
    columns (``Date`` and ``Time`` is the common pair). They are joined with one space
    after being stripped of surrounding whitespace, parsed -- with
    ``meta["time_format"]`` when the entry gives one, else as pandas reads it -- into
    one **naive** time column, and replaced by it. The joined column is named ``time``
    (or ``time_joined`` when the table already has a different ``time`` column), goes
    first, and the component columns are dropped, so they never become data variables.
    A row with an empty or missing component has no time (NaT). Naive on purpose: which
    zone those clock readings are in is the entry's ``time_zone``/``utc_offset_h``, and
    applying it is :func:`decode_time_column`'s one job, wherever the column is decoded.

    The column's name is recorded in ``df.attrs`` (:func:`joined_time_column` reads it
    back), which is also how a frame that has already been through here is recognised
    and returned as it is: ``sources.read`` applies the option, and :func:`to_dataset`
    asks again for a table handed to it directly. ``df.attrs`` is otherwise preserved.
    ``df`` is returned unchanged -- the same object -- when the entry declares no
    ``time_columns``.

    Raises ``ValueError`` for a column the table does not have (listing the ones it
    does) and for a joined text that no row parses (naming the first value and
    suggesting ``time_format=``): both mean the declaration does not describe this
    table.
    """
    import pandas as pd

    meta = meta or {}
    declared = canonicalize_time_options(
        meta.get("time_columns"), meta.get("time_format")
    )
    columns = declared.get("time_columns")
    if not columns:
        return df
    marked = df.attrs.get(_JOINED_TIME_ATTR)
    if marked is not None and marked in df.columns:
        return df  # already joined: the components are gone, so there is nothing to do
    present = {str(c): c for c in df.columns}
    missing = [c for c in columns if c not in present]
    if missing:
        raise ValueError(
            f"{subject}: time_columns names column(s) {missing} that the table does "
            f"not have. Its columns are {[str(c) for c in df.columns]}"
        )

    text = [df[present[c]].astype(str).str.strip() for c in columns]
    blank = pd.Series(False, index=df.index)
    for c, part in zip(columns, text, strict=True):
        blank |= df[present[c]].isna() | part.eq("")
    joined = text[0]
    for part in text[1:]:
        joined = joined + " " + part
    joined = joined.where(~blank)  # a row missing a piece has no time, not half of one
    time_format = declared.get("time_format")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pandas' "could not infer format" notice
        parsed = pd.to_datetime(joined, format=time_format, errors="coerce")
    if len(df) and parsed.isna().all():
        first = next((v for v in joined if isinstance(v, str)), None)
        what = (
            f"the first joined value is {first!r}"
            if first is not None
            else "every row is missing at least one of them"
        )
        raise ValueError(
            f"{subject}: time_columns {columns} joined into text that no row parses as "
            f"a time ({what}). Give time_format= the strptime format of the joined "
            "text (the columns joined by one space), e.g. '%d/%m/%Y %H:%M'."
        )

    out = df.drop(columns=[present[c] for c in columns])
    name = _JOINED_TIME if _JOINED_TIME not in out.columns else _JOINED_TIME_FALLBACK
    # .array, not the Series: a Series would be aligned on the index, which fails on the
    # duplicate labels a concatenated table can carry; the values are already in order.
    out.insert(0, name, parsed.array)
    out.attrs = {**df.attrs, _JOINED_TIME_ATTR: name}
    return out


def joined_time_column(df, meta) -> str | None:
    """Return the column :func:`apply_table_options` joined ``time_columns`` into.

    ``None`` when the entry declares no ``time_columns``, or the table has not been
    through :func:`apply_table_options` yet (its component columns are still there).
    Read from the name recorded in ``df.attrs``, and failing that from the naming rule
    itself -- ``time_joined`` if there is one, else ``time`` -- so a frame whose attrs
    were lost on the way (a concatenation, say) is still found.
    """
    components = (meta or {}).get("time_columns")
    if not components:
        return None
    named = df.attrs.get(_JOINED_TIME_ATTR)
    if named is not None and named in df.columns:
        return str(named)
    if any(c in df.columns for c in components):
        return None
    for candidate in (_JOINED_TIME_FALLBACK, _JOINED_TIME):
        if candidate in df.columns:
            return candidate
    return None


def coord_column(df, axis: str, *, exclude: frozenset = frozenset()) -> str | None:
    """Return the column naming ``axis`` (``T``/``X``/``Y``/``Z``) by its own spelling.

    Tried in column order against :data:`ocean_skill.vocabulary.COORD_VOCABULARY`
    (via :func:`vocabulary.matches_axis`). ``Z`` is tried twice — a depth/z-shaped
    name first, a pressure-shaped one only if no direct reading is named — the same
    preference :func:`depth_of` gives an actual reading over a conversion from
    pressure. A name carrying the whole token "bottom" (``Depth_bottom``,
    ``bottom_depth``) never matches ``Z`` at either rung — see
    :data:`ocean_skill.vocabulary.COORD_VOCABULARY`'s ``exclude`` entry — since it
    states the seafloor/station depth, not the vertical coordinate.

    ``T`` alone has one more rung when no column *name* matches at all: a single
    datetime64-typed column is assumed to be time (two or more, and the ambiguity is
    left unresolved rather than guessed at). No such fallback exists for X/Y/Z —
    dtype alone cannot tell a longitude column from any other float column the way it
    can tell a timestamp from one.

    ``exclude`` skips named columns entirely -- for :mod:`ocean_skill.build`'s probe,
    which passes the QC flag columns it has already detected, so a flag whose own
    name loosely matches a coordinate token (``Pressure_flag``, whose base still
    contains the word "pressure") can never be claimed as that axis in its place: its
    integer flag codes (1-9) would otherwise become a nonsense vertical extent.
    """
    if axis == "Z":
        for direct_only in (True, False):
            match = next(
                (
                    c
                    for c in df.columns
                    if str(c) not in exclude
                    and vocabulary.matches_axis(
                        split_units(str(c))[0], "Z", direct_only=direct_only
                    )
                ),
                None,
            )
            if match is not None:
                return str(match)
        return None
    match = next(
        (
            c
            for c in df.columns
            if str(c) not in exclude
            and vocabulary.matches_axis(split_units(str(c))[0], axis)
        ),
        None,
    )
    if match is not None:
        return str(match)
    if axis == "T":
        import pandas as pd

        datetime_cols = [
            c
            for c in df.columns
            if str(c) not in exclude and pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        if len(datetime_cols) == 1:
            return str(datetime_cols[0])
    return None


def coord_axis_of(column) -> str | None:
    """Return the axis (``T``/``X``/``Y``/``Z``) ``column`` names by its spelling.

    ``None`` for a data column, and also for ``altitude`` — excluded from
    ``variables`` the same as a real axis (see :func:`is_coordinate_column`) but not
    claimed as ``Z``, since unlike depth/pressure it carries no agreed-on sign here.
    Does not consult a datetime dtype the way :func:`coord_column`'s ``T`` fallback
    does: this only ever sees one column's *name*, and a column being datetime64
    says nothing about which axis a caller already believes it to be.
    """
    base = split_units(str(column))[0]
    if base.strip().casefold() == _ALTITUDE:
        return None
    for axis in ("T", "X", "Y", "Z"):
        if vocabulary.matches_axis(base, axis):
            return axis
    return None


def is_coordinate_column(column) -> bool:
    """Whether ``column`` describes *where*/*when*, not a comparable measurement.

    The exclusion :func:`ocean_skill.build._probe_dataframe` and :func:`to_dataset`
    both apply before a column can become a ``variables``/data entry — the union of
    :func:`coord_axis_of` (T/X/Y/Z by name) and ``altitude``, which
    :func:`coord_axis_of` deliberately leaves unassigned to an axis but which is
    still not a measurement to carry through as data.
    """
    base = split_units(str(column))[0].strip().casefold()
    return base == _ALTITUDE or coord_axis_of(column) is not None


def _axis_column(df, meta: dict[str, Any], axis: str) -> str | None:
    """Return the column holding ``axis`` (``T``/``X``/``Y``/``Z``), or ``None``.

    The entry's ``axes`` map is the contract and is tried first — guessing would fail on
    exactly the spellings in play (``"time (UTC)"``, ``"latitude (degrees_north)"``).
    For ``T``, an entry that joins split date/time columns (``time_columns``) names no
    column of its own: the joined one is the axis (see :func:`apply_table_options`).
    :func:`coord_column`'s regex match is the fallback for a frame from a reader that
    declared no axes.
    """
    named = (meta.get("axes") or {}).get(axis)
    if named is not None and named in df.columns:
        return str(named)
    if axis == "T" and (joined := joined_time_column(df, meta)) is not None:
        return joined
    return coord_column(df, axis)


def _units_map(df, meta: dict[str, Any]) -> dict[str, str]:
    """Return ``{column_as_it_is_now: units}``.

    Units live in two places and neither alone is enough: the column's own suffix, and
    the entry's ``units`` metadata — which is keyed by the column's *original* name, so
    a frame that ``sources.read`` has already renamed needs those keys split back to
    their base names to match what the frame now says.

    ``sources.read`` also renames columns to the entry's ``standard_names``, and a
    column that was renamed to something other than its own base name would otherwise
    lose the units declared (or suffixed) under its old name: ``units={"TA":
    "umol/kg"}`` with ``standard_names={"TA": "sea_water_alkalinity..."}`` must still
    give the renamed column its units. A rename is followed only where it really
    happened -- the new name is a column and the old one is gone, and the first
    original to claim a target wins, the rule ``sources.read`` itself applies -- so
    units are never put on a column that merely shares the target's name.
    """
    out: dict[str, str] = {}
    declared = {str(k): str(v) for k, v in (meta.get("units") or {}).items()}
    for original, unit in declared.items():
        base, _ = split_units(original)
        out[original] = unit
        out.setdefault(base, unit)
    columns = {str(c) for c in df.columns}
    claimed: set[str] = set()
    for source, target in (meta.get("standard_names") or {}).items():
        source, target = str(source), str(target)
        if source not in columns and target in columns and target not in claimed:
            claimed.add(target)
            base, suffix = split_units(source)
            unit = suffix or declared.get(source) or declared.get(base)
            if unit:
                out.setdefault(target, unit)
    for column in df.columns:
        _, unit = split_units(column)
        if unit:
            out[str(column)] = unit
    return out


def _scalar_position(df, column: str | None, axis: str) -> float | None:
    """Return a fixed lon/lat, or ``None`` when it varies (or there is no column).

    ``axis`` (``"X"``/``"Y"``) filters out-of-range fill values (see
    :func:`numeric_in_range`) before the spread check -- a handful of ``9999``
    sentinel rows would otherwise blow the range check open and report a fixed
    mooring as a drifting trajectory.
    """
    if column is None:
        return None
    values = numeric_in_range(df[column], axis).to_numpy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if float(np.ptp(finite)) > FIXED_POSITION_TOLERANCE:
        return None
    return float(np.median(finite))


class _StationDepth(NamedTuple):
    """What :func:`_find_depth` found: the depth, and what describes it.

    ``depth`` is metres, positive down, below ``convention``'s origin: a float when the
    reading is constant, an array when it varies, ``None`` when the surface was assumed.
    ``column`` is the table column it was read from (``None`` for the metadata and
    assumed-surface rungs).
    """

    depth: Any
    source: str
    approximate: bool
    convention: depth_convention.ResolvedConvention
    column: str | None


def _vertical_convention(
    meta: dict[str, Any], units: dict[str, str], column: str | None
) -> depth_convention.ResolvedConvention:
    """Resolve the depth convention a table's vertical ``column`` is stated in.

    The entry's declared (and probed) ``depth_convention`` first, then what the column's
    own name and units give away -- ``Pressure (dbar)`` is pressure, so measured below
    the free surface and in decibars -- then the defaults (see
    :func:`ocean_skill.depth_convention.resolve`). The units come from the column's
    ``(units)``/``[units]`` suffix or, for a column that has been renamed bare, the
    entry's ``units`` metadata (:func:`_units_map`).
    """
    hint: dict[str, str] = {}
    if column is not None:
        base, _ = split_units(column)
        unit = units.get(str(column)) or units.get(base)
        hint = depth_convention.infer_from_coordinate(
            base, {"units": unit} if unit else None
        )
    return depth_convention.resolve(meta, hint=hint)


def _normalize_depth(values, convention: depth_convention.ResolvedConvention):
    """Return a vertical column's values as metres, positive down.

    Called with what :func:`numeric_in_range` left, so fill values are already NaN. The
    sign is the convention's ``positive`` where somebody stated it, else the values' own
    (a column of zeros and negatives is heights, not a depth of nothing), and the units
    are decibars where the convention says so (1 dbar ~ 1 m, flagged
    ``depth_approximate`` wherever it is applied: the exact factor varies with latitude
    and water column, and gsw is not a dependency here).
    """
    return depth_convention.positive_down_values(values, None, convention=convention)


def _depth_attrs(
    convention: depth_convention.ResolvedConvention, *, approximate: bool
) -> dict[str, Any]:
    """Return the attributes a normalised depth coordinate carries (besides its units).

    ``depth_normalized`` tells every later reader the values are already metres positive
    down, so a second pass (:func:`ocean_skill.depth_convention.positive_down_values`)
    leaves them alone instead of flipping or converting them again. ``depth_origin`` and
    (for a fixed origin) ``depth_datum_z_m`` say what zero is, which is what the model
    side needs to pick the matching level; ``depth_approximate`` marks values converted
    from decibars.
    """
    attrs: dict[str, Any] = {"depth_origin": convention.origin}
    if convention.origin == "fixed":
        attrs["depth_datum_z_m"] = convention.datum_z_m
    attrs[depth_convention.NORMALIZED_ATTR] = 1
    if approximate:
        attrs["depth_approximate"] = 1
    return attrs


def _stamp_time(coord, meta) -> None:
    """Record on a UTC time coordinate that it is UTC, and what zone the source kept.

    ``time_zone`` stays ``"UTC"`` -- the values are -- and ``source_time_zone`` (only
    when the entry declared one) says what the source's own clock was, so a reader of a
    ``12:00`` that now says ``20:00`` can see why.
    """
    coord.attrs["time_zone"] = "UTC"
    if (label := time_zone.time_zone_label(meta)) is not None:
        coord.attrs["source_time_zone"] = label


def _find_depth(df, meta: dict[str, Any], *, subject: str) -> _StationDepth:
    """Run the ranked depth search behind :func:`depth_of`, keeping its context."""
    units = _units_map(df, meta)

    # The rungs, in ranking order, as (rung, label, column). An entry's own axes["Z"] is
    # the first: it names the vertical coordinate outright, where the alias search below
    # can only guess, and the guess is exactly what goes wrong when a table has both a
    # depth and a "sensor_depth" column.
    declared = (meta.get("axes") or {}).get("Z")
    declared = str(declared) if declared in df.columns else None
    ranked: list[tuple[str, str, str]] = []
    if declared is not None:
        ranked.append((f"axes:{declared}", declared, declared))
    for name, aliases in _DEPTH_COLUMN_ALIASES.items():
        column = next(
            (c for c in df.columns if split_units(c)[0].strip().casefold() in aliases),
            None,
        )
        if column is not None and str(column) != declared:
            ranked.append((name, name, str(column)))

    candidates: dict[str, tuple[Any, depth_convention.ResolvedConvention, str]] = {}
    placeholders: list[str] = []
    for rung, label, column in ranked:
        values = numeric_in_range(df[column], "Z").to_numpy()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        # A column that is *exactly* zero for a whole record is a placeholder, not a
        # reading: OOI's ERDDAP tables carry `z = 0.0` for instruments tens of metres
        # down. Ranking it first because it is "the more direct column" is how a
        # mid-water instrument comes to be compared as a surface one, so it does not
        # count as a value -- but it is reported below, since a wrongly-declared zero
        # is worth knowing about whichever rung ends up winning. That holds for a
        # declared axes["Z"] too: a catalog's axes are as often what the probe found
        # (this very column, on exactly these tables) as what somebody wrote.
        if float(np.max(np.abs(finite))) == 0.0:
            placeholders.append(label)
            continue
        convention = _vertical_convention(meta, units, column)
        candidates[rung] = (_normalize_depth(values, convention), convention, column)

    # `ranked` is already in ranking order, so the first reading with a value wins; the
    # pressure rung (always last: a conversion, not a reading) is only the fallback.
    pressure_rung = _DEPTH_COLUMNS[-1]
    reading = next(
        (r for r, _, _ in ranked if r != pressure_rung and r in candidates), None
    )
    pressure = pressure_rung if pressure_rung in candidates else None
    chosen_rung = reading if reading is not None else pressure

    # Two rungs that both have a value and disagree: report both rather than letting
    # the ranking decide silently.
    if reading is not None and pressure is not None:
        a = float(np.nanmedian(candidates[reading][0]))
        b = float(np.nanmedian(candidates[pressure][0]))
        if abs(a - b) > DEPTH_DISAGREEMENT_TOLERANCE:
            warnings.warn(
                f"{subject}: {reading} says {a:g} m but sea_water_pressure says "
                f"{b:g} m. Using {reading}, the more direct reading — pass the depth "
                "yourself if that is the wrong one.",
                stacklevel=_stacklevel.find(),
            )
    if placeholders and chosen_rung is not None:
        depth_says = float(np.nanmedian(candidates[chosen_rung][0]))
        warnings.warn(
            f"{subject}: {'/'.join(placeholders)} is 0 for the whole record, which is "
            f"a placeholder rather than a measurement — {chosen_rung} puts this "
            f"instrument at {depth_says:g} m. Using {chosen_rung}.",
            stacklevel=_stacklevel.find(),
        )

    if chosen_rung is None:
        convention = _vertical_convention(meta, units, None)
        for key in _DEPTH_ATTRS:
            value = meta.get(key)
            if value is None:
                continue
            # nominal_depth_m / depth are depths by name, in metres: only their sign is
            # ever in doubt (the heuristic reads an all-negative one as a height, as the
            # abs() this replaces did), and the entry's depth_convention -- which
            # describes the *source's* vertical values -- is not applied to them.
            # geospatial_vertical_min is a raw value off that same axis, so it is.
            raw = np.asarray([float(value)])
            from_axis = key == "geospatial_vertical_min"
            depth = depth_convention.positive_down_values(
                raw, None, convention=convention if from_axis else None
            )
            return _StationDepth(
                float(depth[0]),
                f"metadata:{key}",
                from_axis and convention.units == "dbar",
                convention,
                None,
            )
        warnings.warn(
            f"{subject}: no depth found — no depth/z/pressure column with values, and "
            f"no {'/'.join(_DEPTH_ATTRS)} in the catalog entry. Assuming the surface. "
            "Give the entry a nominal_depth_m if it is not.",
            stacklevel=_stacklevel.find(),
        )
        return _StationDepth(None, "assumed-surface", False, convention, None)

    chosen, convention, column = candidates[chosen_rung]
    approximate = convention.units == "dbar"
    finite = chosen[np.isfinite(chosen)]
    if float(np.ptp(finite)) > FIXED_DEPTH_TOLERANCE:
        return _StationDepth(chosen, chosen_rung, approximate, convention, column)
    return _StationDepth(
        float(np.median(finite)), chosen_rung, approximate, convention, column
    )


def depth_of(df, meta: dict[str, Any], *, subject: str = "this source"):
    """Return ``(depth, source, approximate)`` for a tabular source's instrument depth.

    Every dataset states depth differently — a column, a pressure reading to convert, an
    attribute on the entry, or nothing at all — so this is a ranked search that always
    reports *which* rung it landed on, rather than a rule pretending they are alike:

    1. the **declared** vertical column, ``meta["axes"]["Z"]`` (``source`` reads
       ``"axes:<column>"``), when the table has it: the entry says which column is the
       depth, so no alias or bottom-depth column can displace it;
    2. a depth **column** with finite values (``depth``, ``depth_reading``, ``z``, and
       their case-insensitive/``[units]`` variants);
    3. **pressure** (``sea_water_pressure``, ``pressure``, ``pres``), converted at
       1 dbar ~ 1 m and flagged ``approximate``;
    4. an **attribute** on the catalog entry (``nominal_depth_m``, ``depth``,
       ``geospatial_vertical_min``);
    5. **nothing** — assume the surface, and say so.

    The ranking is by how directly the source measured it, which is what the OOI Station
    Papa moorings need: ``depth_reading`` is all-NaN there, ``z`` is a flat 0.0
    placeholder, and only pressure knows the instrument is at ~8 m (its entry title
    claims 30). Two rungs that both have a value and disagree are reported, since the
    higher one silently winning is how a mid-water instrument comes to look like a
    surface one. A column that is exactly zero for the whole record is a placeholder at
    every column rung, a declared one included.

    ``depth`` is **metres, positive down**, whatever the source stored: the entry's
    ``depth_convention`` (see :mod:`ocean_skill.depth_convention`) -- declared, probed,
    or read off the column's own name and units -- says which way the values count and
    whether they are decibars, and a column that states neither but is all negative is
    read as heights. It is an array when the column varies (a profiler) and a float when
    it does not; ``None`` means the surface was assumed.
    """
    found = _find_depth(df, meta, subject=subject)
    return found.depth, found.source, found.approximate


def _convert_data_column(
    frame,
    column,
    *,
    is_flag: bool,
    units: dict[str, str],
    flag_pairs: dict[str, str],
    flag_definitions: dict,
    flag_to_qartod: dict,
    data_to_flags: dict[str, list[str]],
    qc_policy_json: str | None,
):
    """Convert one raw table column to ``(base_name, values, attrs)``.

    Shared by :func:`to_dataset`'s time build and :func:`_timeseriesprofile_dataset`
    -- everything about a column's *meaning* (is it a QC flag? what unit? what CF
    attrs?) is the same in both; only what dimension(s) the caller assigns the
    result differs, so this stops at a plain :class:`pandas.Series` rather than
    building the ``xr.DataArray`` itself.

    Returns ``(None, None, placeholder_attrs)`` for an all-NaN column that should
    become dataset-level provenance instead of a variable — see the callers.
    """
    import pandas as pd

    from ocean_skill.qc import QARTOD_FLAGS, _normalize_flag_value

    base, _ = split_units(column)
    if is_flag:
        raw = frame[column]

        def _flag_value(v):
            # A real numeric provider code (an int, or a digit stored as a string
            # like SeaDataNet's mostly-numeric column) keeps its own value --
            # flag_values below is meant to hold the *provider's* codes, not a
            # QARTOD translation of them. Only a value that cannot be read as a
            # number at all -- a letter code -- is encoded through the contract's
            # flag_to_qartod into QARTOD's own integer codes instead of being lost
            # to a blind pd.to_numeric coercion (which would otherwise make the
            # whole column look all-NaN and be dropped below like any other
            # placeholder).
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(
                    QARTOD_FLAGS.get(
                        flag_to_qartod.get(_normalize_flag_value(v)), np.nan
                    )
                )

        values = raw.map(lambda v: np.nan if pd.isna(v) else _flag_value(v))
        observed = sorted(
            (
                n
                for v in raw.dropna().unique()
                if (n := _normalize_flag_value(v)) is not None
            ),
            key=str,
        )
        attrs: dict[str, Any] = {
            "flag_values": [
                v if isinstance(v, int) else QARTOD_FLAGS.get(flag_to_qartod.get(v), -1)
                for v in observed
            ],
            "standard_name": "status_flag",
            "source_column": str(column),
            "flags_for": split_units(flag_pairs[str(column)])[0],
        }
        if flag_definitions:
            attrs["flag_meanings"] = " ".join(
                str(flag_definitions.get(v, v)).strip().replace(" ", "_")
                for v in observed
            )
        if flag_to_qartod:
            attrs["flag_qartod"] = " ".join(
                str(flag_to_qartod.get(v, "UNKNOWN")) for v in observed
            )
        return base, values, attrs

    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().all():
        # All-NaN columns are placeholders (OOI's `depth_reading`, `station`), and a
        # non-numeric column would break a bare .mean() the way ROMS' `spherical`
        # flag does -- so a single-valued one is recorded as provenance and the rest
        # are dropped rather than carried as unusable variables.
        original = frame[column].dropna().unique()
        return None, None, ({base: original[0]} if original.size == 1 else {})

    attrs = {"standard_name": base, "source_column": str(column)}
    if unit := (units.get(str(column)) or units.get(base)):
        attrs["units"] = unit
    if str(column) in data_to_flags:
        attrs["ancillary_variables"] = " ".join(
            split_units(f)[0] for f in data_to_flags[str(column)]
        )
        if qc_policy_json is not None:
            attrs["qc_policy"] = qc_policy_json
    return base, values, attrs


def to_dataset(df, meta: dict[str, Any]):
    """Return a station table as a 1-D :class:`xarray.Dataset`.

    Indexed on ``time`` for the usual mooring/timeSeries case; position and depth
    become **scalar coordinates** (``lon``/``lat``/``depth``, the names
    :mod:`ocean_skill.align` and :mod:`ocean_skill.metrics` look for), so the only
    dimension left is the one a time-series comparison is about. Units ride on each
    variable, so the alignment step's compatibility check and
    :func:`ocean_skill.units.convert_units` work on a mooring exactly as on a grid.

    An entry declared ``featureType: "profile"`` is a different shape entirely -- one
    instant, many depths -- and is built by :func:`_profile_dataset` instead: indexed
    on ``depth``, with ``time``/``lon``/``lat`` as the scalar coordinates.

    An entry declared ``featureType: "timeSeriesProfile"`` is the combination of the
    two -- a station visited repeatedly, several depths per visit -- and is built by
    :func:`_timeseriesprofile_dataset`: indexed on **both** ``time`` and ``depth``, a
    rectangle with a NaN wherever a visit did not sample a given level.

    QARTOD/flag companion columns are **kept** rather than dropped: they are real
    information, and what :func:`ocean_skill.qc.apply` reads (called by
    :func:`ocean_skill.sources.read`, before this conversion ever runs — this
    function never applies a policy itself, only reflects what already happened via
    ``meta["qc"]``/``df.attrs["qc_applied"]`` in the resulting attrs). What must not
    happen — a request for temperature being satisfied by
    ``sea_water_temperature_qc_agg`` — is refused at the matcher
    (:func:`ocean_skill.units.find_variable`), which protects gridded sources
    carrying flag variables too.

    A column the entry's ``qc`` contract names as a flag (``meta["qc"]["flags"]``)
    is never treated as a coordinate even if its name loosely matches one (a
    ``Pressure_flag`` column's base still contains the word "pressure"), and its
    values are **not** dropped by the blind ``pd.to_numeric`` coercion below when
    they are letter codes (SeaDataNet-style) rather than digits — they are encoded
    through the contract's ``flag_to_qartod`` into QARTOD's own integer codes
    instead. Its CF attrs record ``flag_values``/``flag_meanings`` (the provider's
    own codes and verbatim definitions) and ``flag_qartod`` (what each of those
    codes was mapped to); the data variable it is paired with gets
    ``ancillary_variables`` pointing back at it, plus ``qc_policy`` recording the
    policy actually applied at read time.

    What the entry declares about how the table is *read* changes the values here. A
    split time (``time_columns``) is joined first (:func:`apply_table_options`).
    ``time`` comes out as UTC: a naive timestamp is read as local time in the entry's
    ``time_zone``/``utc_offset_h`` (see :mod:`ocean_skill.time_zone`), and the
    coordinate's ``source_time_zone`` attribute says which. ``depth`` comes out as
    **metres, positive down** whatever the source stored (heights, decibars), under the
    entry's ``depth_convention`` (see :mod:`ocean_skill.depth_convention`), and carries
    ``depth_origin`` (``surface`` or ``fixed``), ``depth_datum_z_m`` (fixed only),
    ``depth_normalized`` (the values are already converted, so
    :func:`ocean_skill.depth_convention.positive_down_values` leaves them alone) and,
    for values converted from decibars, ``depth_approximate``.
    """
    import json

    import pandas as pd
    import xarray as xr

    subject = _subject_of(meta, df)
    # A table whose time is split over columns is joined here when it has not been
    # already (sources.read does it on the way in; this covers a table handed over
    # directly), so every build below finds one time column.
    df = apply_table_options(df, meta, subject=subject)
    feature_type = str(meta.get("featureType") or "").strip().casefold()
    if feature_type == "profile":
        # A CTD-style cast is indexed on depth, not time -- a different build entirely.
        return _profile_dataset(df, meta, subject=subject)
    if feature_type == "timeseriesprofile":
        # A repeat-visit station is indexed on both -- yet another build.
        return _timeseriesprofile_dataset(df, meta, subject=subject)
    time_col = _axis_column(df, meta, "T")
    if time_col is None:
        raise ValueError(
            f"{subject}: no time column, so this table cannot become a time series. "
            "Give the catalog entry an axes mapping such as "
            '`axes={"T": "time (UTC)"}`, or name the column "time".'
        )
    lon_col = _axis_column(df, meta, "X")
    lat_col = _axis_column(df, meta, "Y")

    # Timezone matters twice over, and both bite silently. A tz-aware coordinate cannot
    # be written to zarr, and cache.save only *warns* when a write fails -- so a
    # tz-aware lane would re-read the whole remote record on every run. Resampling one
    # also yields datetime.datetime labels, which no later join matches. UTC is
    # recorded in attrs rather than thrown away.
    time = decode_time_column(df[time_col], time_col, meta, subject=subject)
    frame = df.loc[time.notna()].copy()
    frame[time_col] = time[time.notna()].dt.tz_convert("UTC").dt.tz_localize(None)
    frame = frame.sort_values(time_col)

    duplicated = frame[time_col].duplicated()
    if bool(duplicated.any()):
        # Overlapping deployments repeat timestamps; a duplicated index makes the
        # later xr.align raise rather than misalign, but by then the source is out of
        # sight, so collapse them here and say how many.
        warnings.warn(
            f"{subject}: {int(duplicated.sum())} duplicate timestamps "
            f"(of {len(frame)}) — keeping the first of each. If several depths share "
            "a timestamp because this is a repeat-visit station with a cast per "
            "visit, label the entry featureType: timeSeriesProfile instead.",
            stacklevel=_stacklevel.find(),
        )
        frame = frame.loc[~duplicated]

    units = _units_map(frame, meta)
    lon = _scalar_position(frame, lon_col, "X")
    lat = _scalar_position(frame, lat_col, "Y")
    if (lon is None) != (lat is None) or (
        lon is None and lon_col is not None and lat_col is not None
    ):
        warnings.warn(
            f"{subject}: longitude/latitude vary along time, so this is a trajectory "
            "rather than a fixed station. Keeping the positions as data; a time-series "
            "comparison against it will sample one model location for a moving "
            "platform, which is not what you want.",
            stacklevel=_stacklevel.find(),
        )

    station = _find_depth(frame, meta, subject=subject)
    depth, depth_source = station.depth, station.source
    approximate = station.approximate

    # The entry's resolved qc contract (see ocean_skill.qc), if any -- flag columns
    # named in it are never treated as coordinates below (bypassing
    # is_coordinate_column even when their name loosely matches one, e.g.
    # "Pressure_flag") and their values are read specially rather than through the
    # blind pd.to_numeric coercion every other column gets. Policy actually applied
    # at read time (ocean_skill.sources.read -> ocean_skill.qc.apply) rides on
    # df.attrs["qc_applied"]; absent for an entry with no contract, or for a frame
    # built by hand (the metadata-less fallback), so this tolerates its absence.
    qc_meta = meta.get("qc") or {}
    flag_pairs = {str(k): str(v) for k, v in (qc_meta.get("flags") or {}).items()}
    flag_definitions = qc_meta.get("flag_definitions") or {}
    flag_to_qartod = qc_meta.get("flag_to_qartod") or {}
    data_to_flags: dict[str, list[str]] = {}
    for fcol, dcol in flag_pairs.items():
        data_to_flags.setdefault(dcol, []).append(fcol)
    qc_applied = frame.attrs.get("qc_applied")
    qc_policy_json = json.dumps(qc_applied, default=str) if qc_applied else None

    data: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    for column in frame.columns:
        if column in (time_col, lon_col, lat_col, station.column):
            continue  # (a declared depth column is never a variable, by any name)
        is_flag = str(column) in flag_pairs
        if is_coordinate_column(column) and not is_flag:
            # Coordinate columns (time/lon/lat/depth/pressure/altitude, by name or
            # regex-recognized variant -- see is_coordinate_column) describe where the
            # measurements were taken; depth_of has already read them, and carrying a
            # flat placeholder alongside the data as if it were a variable invites it
            # being compared. The same exclusion build.py applies to standard_names.
            # A contract flag column is exempted even if its own name loosely matches
            # a coordinate token -- see the qc contract note above.
            continue
        base, values, col_attrs = _convert_data_column(
            frame,
            column,
            is_flag=is_flag,
            units=units,
            flag_pairs=flag_pairs,
            flag_definitions=flag_definitions,
            flag_to_qartod=flag_to_qartod,
            data_to_flags=data_to_flags,
            qc_policy_json=qc_policy_json,
        )
        if base is None:
            attrs.update(col_attrs)
            continue
        variable = xr.DataArray(values.to_numpy(), dims=("time",), name=base)
        variable.attrs.update(col_attrs)
        # The depth also goes on each variable's own attrs, not only on a coordinate.
        # A coordinate along time does not survive a reduction -- resampling a mooring
        # to monthly means drops it -- and the depth is then invisible exactly where it
        # matters most, to the caveat about comparing it against a surface field.
        if depth is not None:
            variable.attrs["depth_m"] = (
                float(depth) if np.isscalar(depth) else float(np.nanmedian(depth))
            )
            if not np.isscalar(depth):
                variable.attrs["depth_range_m"] = (
                    float(np.nanmin(depth)),
                    float(np.nanmax(depth)),
                )
        data[base] = variable

    ds = xr.Dataset(data, coords={"time": frame[time_col].to_numpy()})
    _stamp_time(ds["time"], meta)
    if lon is not None:
        ds = ds.assign_coords(lon=lon, lat=lat)
        ds["lon"].attrs.update(units="degrees_east", standard_name="longitude")
        ds["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    if depth is not None:
        value = depth if np.isscalar(depth) else ("time", np.asarray(depth))
        ds = ds.assign_coords(depth=value)
        ds["depth"].attrs.update(
            units="m",
            positive="down",
            long_name="instrument depth",
            **_depth_attrs(station.convention, approximate=station.approximate),
        )
        if not np.isscalar(depth) and meta.get("nominal_depth_m") is None:
            # An entry that declares its own nominal_depth_m has already said where the
            # instrument is: the pressure-derived depth wandering by a couple of metres
            # with the tide is expected there, not news worth a warning on every read.
            # Describe the spread rather than diagnose its cause: a range can mean a
            # profiling instrument, a mooring blown down by a current, deployment and
            # recovery casts, or -- as on the Papa flanking moorings -- a record
            # spanning deployments that hung the instrument at different depths. The
            # percentiles distinguish those by eye where a min/max cannot.
            low, mid, high = np.nanpercentile(np.asarray(depth), [5, 50, 95])
            warnings.warn(
                f"{subject}: depth is not constant — median {mid:g} m, 5th-95th "
                f"percentile {low:g}-{high:g} m, full range "
                f"{float(np.nanmin(depth)):g}-{float(np.nanmax(depth)):g} m. Kept as a "
                "coordinate along time, so a comparison against a single-level source "
                "is comparing different depths at different times. Narrow it with "
                'select={"time": ...} to one deployment if that is not what you want.',
                stacklevel=_stacklevel.find(),
            )
    ds.attrs.update(attrs)
    ds.attrs["depth_source"] = depth_source
    ds.attrs["depth_approximate"] = bool(approximate)
    for key in ("featureType", "title", "institution", "datasetID"):
        if value := meta.get(key):
            # Strings only: zarr's JSON attrs cannot hold a timestamp, and the cache
            # writes this Dataset's descendants back out.
            ds.attrs[key] = str(value)
    return ds


def _profile_dataset(df, meta: dict[str, Any], *, subject: str):
    """Return a CTD-style cast as a 1-D :class:`xarray.Dataset` on ``depth``.

    The vertical twin of :func:`to_dataset`'s time-series build, for an entry
    declared ``featureType: "profile"``: one instant, many depths, so ``depth`` is the
    dimension and ``time``/``lon``/``lat`` become scalar coordinates instead. Called
    from :func:`to_dataset`, not directly -- the ``featureType`` check that picks this
    branch lives there.
    """
    import pandas as pd
    import xarray as xr

    depth_col = _axis_column(df, meta, "Z")
    if depth_col is None:
        raise ValueError(
            f"{subject}: no depth column, so this table cannot become a profile. "
            "Give the catalog entry an axes mapping such as "
            '`axes={"Z": "depth (m)"}`, or name the column "depth".'
        )

    convention = _vertical_convention(meta, _units_map(df, meta), depth_col)
    depth = numeric_in_range(df[depth_col], "Z")
    # Normalised before the sort and the duplicate check, so both run on metres positive
    # down whatever the source stored: a cast kept as heights (-1 ... -50) comes out in
    # the order a depth axis is read, and two heights that differ only by sign are not
    # mistaken for each other.
    depth = pd.Series(_normalize_depth(depth.to_numpy(), convention), index=depth.index)
    frame = df.loc[depth.notna()].copy()
    frame[depth_col] = depth[depth.notna()]
    frame = frame.sort_values(depth_col)

    duplicated = frame[depth_col].duplicated()
    if bool(duplicated.any()):
        # A station visited more than once repeats depths -- that is a
        # timeSeriesProfile, not a profile (one instant). Collapsing here rather than
        # refusing keeps a mislabeled cast readable, but says so: the featureType is
        # the fix, not this fallback.
        warnings.warn(
            f"{subject}: {int(duplicated.sum())} duplicate depths "
            f"(of {len(frame)}) — keeping the first of each. If this station was "
            'visited more than once, it is a timeSeriesProfile, not a profile.',
            stacklevel=_stacklevel.find(),
        )
        frame = frame.loc[~duplicated]

    units = _units_map(frame, meta)
    lon_col = _axis_column(frame, meta, "X")
    lat_col = _axis_column(frame, meta, "Y")
    time_col = _axis_column(frame, meta, "T")
    lon = _scalar_position(frame, lon_col, "X")
    lat = _scalar_position(frame, lat_col, "Y")
    # One cast of a transect (ocean_skill.casts) states its own position -- the ship
    # drifted while it was on station -- and its samples span minutes by construction.
    cast = meta.get("cast") if isinstance(meta.get("cast"), dict) else {}
    if np.isfinite(cast.get("lon", np.nan)) and np.isfinite(cast.get("lat", np.nan)):
        lon, lat = float(cast["lon"]), float(cast["lat"])

    time = None
    if time_col is not None:
        decoded = decode_time_column(frame[time_col], time_col, meta, subject=subject)
        decoded = decoded.dropna()
        if not decoded.empty:
            if decoded.nunique() > 1 and not cast:
                # A profile is one instant by definition -- see PROFILE_FEATURE_TYPES
                # in ocean_skill.comparison. Depths sampled seconds apart during a cast
                # commonly carry distinct timestamps; using the earliest is a label,
                # not a measurement, so this only ever warns rather than refuses.
                warnings.warn(
                    f"{subject}: time varies across the cast ({decoded.nunique()} "
                    "distinct values) -- a profile is a single instant. Using the "
                    "earliest. If this is really a repeat-visit station, label the "
                    "entry timeSeriesProfile instead.",
                    stacklevel=_stacklevel.find(),
                )
            time = decoded.min().tz_convert("UTC").tz_localize(None)

    data: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    for column in frame.columns:
        if column in (depth_col, lon_col, lat_col, time_col):
            continue
        base, _ = split_units(column)
        if is_coordinate_column(column):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().all():
            original = frame[column].dropna().unique()
            if original.size == 1:
                attrs[base] = original[0]
            continue
        variable = xr.DataArray(values.to_numpy(), dims=("depth",), name=base)
        if unit := (units.get(str(column)) or units.get(base)):
            variable.attrs["units"] = unit
        variable.attrs["standard_name"] = base
        variable.attrs["source_column"] = str(column)
        data[base] = variable

    ds = xr.Dataset(data, coords={"depth": frame[depth_col].to_numpy()})
    ds["depth"].attrs.update(
        units="m",
        positive="down",
        long_name="depth",
        **_depth_attrs(convention, approximate=convention.units == "dbar"),
    )
    if lon is not None:
        ds = ds.assign_coords(lon=lon, lat=lat)
        ds["lon"].attrs.update(units="degrees_east", standard_name="longitude")
        ds["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    if time is not None:
        ds = ds.assign_coords(time=time)
        _stamp_time(ds["time"], meta)
    ds.attrs.update(attrs)
    for key in ("featureType", "title", "institution", "datasetID"):
        if value := meta.get(key):
            ds.attrs[key] = str(value)
    return ds


def _station_position(
    frame, lon_col: str | None, lat_col: str | None, *, subject: str
) -> tuple[float | None, float | None]:
    """Return ``(lon, lat)`` for a *declared* fixed station, tolerating GPS wobble.

    Unlike :func:`_scalar_position` (used by the plain time/profile builds, where a
    spread beyond :data:`FIXED_POSITION_TOLERANCE` means "this is actually a
    trajectory, not a fixed station"), a ``timeSeriesProfile`` entry has already
    declared itself a fixed station repeat-visited over time. Ordinary GPS noise or
    re-anchoring between separate visits (seen: 100-400 m across a season of CTD
    casts at one named station) should not silently relabel it a trajectory or
    throw the position away -- the median position is used instead, with a warning
    naming the spread, whenever :func:`_scalar_position` finds it exceeds tolerance.
    """
    lon = _scalar_position(frame, lon_col, "X")
    lat = _scalar_position(frame, lat_col, "Y")
    if lon is not None or lon_col is None:
        return lon, lat

    lon_values = numeric_in_range(frame[lon_col], "X").to_numpy()
    lon_finite = lon_values[np.isfinite(lon_values)]
    if lon_finite.size == 0:
        return None, None
    lat_finite = np.array([])
    if lat_col is not None:
        lat_values = numeric_in_range(frame[lat_col], "Y").to_numpy()
        lat_finite = lat_values[np.isfinite(lat_values)]

    # Metres, not degrees times 111 km: a degree of longitude shrinks with cos(lat)
    # (about 49 km at 64N), so scaling the larger degree range by the full 111 km
    # overstated high-latitude wobble by more than a factor of two. The larger of the
    # two ranges, each in its own metres, is reported.
    lon_deg = float(np.ptp(lon_finite))
    lat_deg = float(np.ptp(lat_finite)) if lat_finite.size else 0.0
    mid_lat = float(np.median(lat_finite)) if lat_finite.size else 0.0
    spread_m = max(
        lon_deg * 111_000 * float(np.cos(np.radians(mid_lat))), lat_deg * 111_000
    )
    warnings.warn(
        f"{subject}: longitude/latitude vary by up to ~{max(lon_deg, lat_deg):.4f}"
        f"\N{DEGREE SIGN} (spread up to ~{spread_m:.0f} m, the larger of the "
        "east-west and north-south ranges) across visits — ordinary GPS/positioning "
        "wobble for a declared fixed station, not a trajectory. Using the median "
        "position for the whole record.",
        stacklevel=_stacklevel.find(),
    )
    return float(np.median(lon_finite)), (
        float(np.median(lat_finite)) if lat_finite.size else None
    )


def _timeseriesprofile_dataset(df, meta: dict[str, Any], *, subject: str):
    """Return a repeat-visit station as a 2-D :class:`xarray.Dataset` on ``(time, depth)``.

    The combined case :func:`to_dataset`'s other two builders each take a slice of:
    time varies across visits (unlike :func:`_profile_dataset`'s single instant),
    and depth varies within a visit (unlike the plain time build's one row per
    timestamp). Called from :func:`to_dataset`, not directly.

    Every ``(time, depth)`` pair read is a real sample -- a bottle at a given
    station on a given cast. A station sampled unevenly (different depths on
    different visits, as discrete bottle sampling usually is) is not "ragged" here
    so much as a rectangle with holes: unsampled (visit, depth) combinations are
    left NaN rather than interpolated or dropped, exactly the shape a WHOTS
    mooring's own ``(TIME, DEPTH)`` NetCDF already has, so every consumer downstream
    of this (:mod:`ocean_skill.comparison`, :mod:`ocean_skill.align`, both plot
    renderers) already knows what to do with it.
    """
    import json

    import pandas as pd

    time_col = _axis_column(df, meta, "T")
    if time_col is None:
        raise ValueError(
            f"{subject}: no time column, so this table cannot become a "
            "timeSeriesProfile. Give the catalog entry an axes mapping such as "
            '`axes={"T": "time (UTC)"}`, or name the column "time".'
        )
    depth_col = _axis_column(df, meta, "Z")
    if depth_col is None:
        raise ValueError(
            f"{subject}: no depth column, so this table cannot become a "
            "timeSeriesProfile. Give the catalog entry an axes mapping such as "
            '`axes={"Z": "depth (m)"}`, or name the column "depth".'
        )

    time = decode_time_column(df[time_col], time_col, meta, subject=subject)
    convention = _vertical_convention(meta, _units_map(df, meta), depth_col)
    depth = numeric_in_range(df[depth_col], "Z")
    depth = pd.Series(_normalize_depth(depth.to_numpy(), convention), index=depth.index)
    keep = time.notna() & depth.notna()
    frame = df.loc[keep].copy()
    frame[time_col] = time[keep].dt.tz_convert("UTC").dt.tz_localize(None)
    frame[depth_col] = depth[keep]
    frame = frame.sort_values([time_col, depth_col])

    duplicated = frame.duplicated(subset=[time_col, depth_col])
    if bool(duplicated.any()):
        # Bottle samples from one cast normally SHARE a timestamp -- that is the
        # whole point of this featureType, not a duplicate. Only a repeated
        # (time, depth) *pair* is: two rows claiming the same bottle.
        warnings.warn(
            f"{subject}: {int(duplicated.sum())} duplicate (time, depth) pairs "
            f"(of {len(frame)}) — keeping the first of each.",
            stacklevel=_stacklevel.find(),
        )
        frame = frame.loc[~duplicated]

    units = _units_map(frame, meta)
    lon_col = _axis_column(frame, meta, "X")
    lat_col = _axis_column(frame, meta, "Y")
    lon, lat = _station_position(frame, lon_col, lat_col, subject=subject)

    qc_meta = meta.get("qc") or {}
    flag_pairs = {str(k): str(v) for k, v in (qc_meta.get("flags") or {}).items()}
    flag_definitions = qc_meta.get("flag_definitions") or {}
    flag_to_qartod = qc_meta.get("flag_to_qartod") or {}
    data_to_flags: dict[str, list[str]] = {}
    for fcol, dcol in flag_pairs.items():
        data_to_flags.setdefault(dcol, []).append(fcol)
    qc_applied = frame.attrs.get("qc_applied")
    qc_policy_json = json.dumps(qc_applied, default=str) if qc_applied else None

    # One long (time, depth, value...) frame, pivoted in a single shot via a
    # MultiIndex -> to_xarray() -- pandas fills every (visit, level) combination no
    # row supplied with NaN and sorts each axis, which is exactly the rectangle
    # with holes this builder promises (verified: to_xarray sorts and NaN-fills
    # unlisted combinations, it does not average or drop them the way pivot_table
    # would silently do for a repeated pair -- already ruled out above).
    long = pd.DataFrame(
        {"time": frame[time_col].to_numpy(), "depth": frame[depth_col].to_numpy()}
    )
    variable_attrs: dict[str, dict[str, Any]] = {}
    attrs: dict[str, Any] = {}
    for column in frame.columns:
        if column in (time_col, depth_col, lon_col, lat_col):
            continue
        is_flag = str(column) in flag_pairs
        if is_coordinate_column(column) and not is_flag:
            continue
        base, values, col_attrs = _convert_data_column(
            frame,
            column,
            is_flag=is_flag,
            units=units,
            flag_pairs=flag_pairs,
            flag_definitions=flag_definitions,
            flag_to_qartod=flag_to_qartod,
            data_to_flags=data_to_flags,
            qc_policy_json=qc_policy_json,
        )
        if base is None:
            attrs.update(col_attrs)
            continue
        long[base] = values.to_numpy()
        variable_attrs[base] = col_attrs

    ds = long.set_index(["time", "depth"]).to_xarray()
    _stamp_time(ds["time"], meta)
    ds["depth"].attrs.update(
        units="m",
        positive="down",
        long_name="depth",
        **_depth_attrs(convention, approximate=convention.units == "dbar"),
    )
    for base, col_attrs in variable_attrs.items():
        ds[base].attrs.update(col_attrs)
    if lon is not None:
        ds = ds.assign_coords(lon=lon, lat=lat)
        ds["lon"].attrs.update(units="degrees_east", standard_name="longitude")
        ds["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    ds.attrs.update(attrs)
    for key in ("featureType", "title", "institution", "datasetID"):
        if value := meta.get(key):
            ds.attrs[key] = str(value)
    return ds
