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

This module also owns the column-naming vocabulary ERDDAP-style tables use --
``"<name> (<units>)"``, ``<name>_qc_agg``/``<name>_qc_tests`` QARTOD companions, and
which column names are coordinates rather than data. :mod:`ocean_skill.build` imports
those helpers rather than keeping its own copy, so the catalog that *describes* a table
and the code that *reads* one cannot disagree about the convention.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np

from ocean_skill import _stacklevel

__all__ = [
    "COORD_COLUMNS",
    "depth_of",
    "is_frame",
    "is_qc_column",
    "split_units",
    "to_dataset",
]

#: Column names that are coordinates, not comparable data.
COORD_COLUMNS = frozenset({"time", "latitude", "longitude", "z", "depth", "altitude"})

#: Suffixes ``intake_erddap`` gives a variable's QARTOD companion columns.
_QC_SUFFIXES = ("_qc_agg", "_qc_tests")

#: ``"<name> (<units>)"`` — the spelling ``intake_erddap``'s readers produce.
_COLUMN_UNITS = re.compile(r"^(.+) \((.+)\)$")

#: Depth-bearing columns, most direct first. Pressure is last because it is a
#: *conversion*, not a reading — see :func:`depth_of`.
_DEPTH_COLUMNS = ("depth", "depth_reading", "z", "sea_water_pressure")

#: Entry-metadata keys that state a depth when the data does not.
_DEPTH_ATTRS = ("nominal_depth_m", "depth", "geospatial_vertical_min")

#: Metres per decibar, near enough for a label: the exact factor varies with latitude
#: and water column (gsw.z_from_p), and gsw is not a dependency here. Flagged on the
#: result as ``depth_approximate`` rather than presented as a measurement.
_M_PER_DBAR = 1.0

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
    """Whether ``name`` is a variable's QARTOD companion rather than a variable."""
    return str(name).endswith(_QC_SUFFIXES)


def split_units(column) -> tuple[str, str | None]:
    """Split ``"temperature (degC)"`` into ``("temperature", "degC")``.

    A column without the parenthesized suffix returns ``(name, None)`` — both spellings
    occur, sometimes in one frame: :func:`ocean_skill.sources.read` renames the columns
    a catalog entry's ``standard_names`` map covers (leaving them bare) and leaves the
    coordinate columns alone (keeping their suffix).
    """
    match = _COLUMN_UNITS.match(str(column))
    return (match.group(1), match.group(2)) if match else (str(column), None)


def _axis_column(df, meta: dict[str, Any], axis: str, prefix: str) -> str | None:
    """Return the column holding ``axis`` (``T``/``X``/``Y``), or ``None``.

    The entry's ``axes`` map is the contract and is tried first — guessing would fail on
    exactly the spellings in play (``"time (UTC)"``, ``"latitude (degrees_north)"``).
    The prefix match is the fallback for a frame from a reader that declared no axes.
    """
    named = (meta.get("axes") or {}).get(axis)
    if named is not None and named in df.columns:
        return str(named)
    return next((str(c) for c in df.columns if str(c).startswith(prefix)), None)


def _units_map(df, meta: dict[str, Any]) -> dict[str, str]:
    """Return ``{column_as_it_is_now: units}``.

    Units live in two places and neither alone is enough: the column's own suffix, and
    the entry's ``units`` metadata — which is keyed by the column's *original* name, so
    a frame that ``sources.read`` has already renamed needs those keys split back to
    their base names to match what the frame now says.
    """
    out: dict[str, str] = {}
    for original, unit in (meta.get("units") or {}).items():
        base, _ = split_units(original)
        out[str(original)] = str(unit)
        out.setdefault(base, str(unit))
    for column in df.columns:
        _, unit = split_units(column)
        if unit:
            out[str(column)] = unit
    return out


def _scalar_position(df, column: str | None) -> float | None:
    """Return a fixed lon/lat, or ``None`` when it varies (or there is no column)."""
    if column is None:
        return None
    values = np.asarray(df[column], dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if float(np.ptp(finite)) > FIXED_POSITION_TOLERANCE:
        return None
    return float(np.median(finite))


def depth_of(df, meta: dict[str, Any], *, subject: str = "this source"):
    """Return ``(depth, source, approximate)`` for a tabular source's instrument depth.

    Every dataset states depth differently — a column, a pressure reading to convert, an
    attribute on the entry, or nothing at all — so this is a ranked search that always
    reports *which* rung it landed on, rather than a rule pretending they are alike:

    1. a depth **column** with finite values (``depth``, ``depth_reading``, ``z``);
    2. **pressure** (``sea_water_pressure``), converted at 1 dbar ~ 1 m and flagged
       ``approximate``;
    3. an **attribute** on the catalog entry (``nominal_depth_m``, ``depth``,
       ``geospatial_vertical_min``);
    4. **nothing** — assume the surface, and say so.

    The ranking is by how directly the source measured it, which is what the OOI Station
    Papa moorings need: ``depth_reading`` is all-NaN there, ``z`` is a flat 0.0
    placeholder, and only pressure knows the instrument is at ~8 m (its entry title
    claims 30). Two rungs that both have a value and disagree are reported, since the
    higher one silently winning is how a mid-water instrument comes to look like a
    surface one.

    ``depth`` is an array when the column varies (a profiler) and a float when it does
    not; ``None`` means the surface was assumed.
    """
    candidates: dict[str, np.ndarray] = {}
    placeholders: list[str] = []
    for name in _DEPTH_COLUMNS:
        column = next((c for c in df.columns if split_units(c)[0] == name), None)
        if column is None:
            continue
        values = np.asarray(df[column], dtype="float64")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        # A column that is *exactly* zero for a whole record is a placeholder, not a
        # reading: OOI's ERDDAP tables carry `z = 0.0` for instruments tens of metres
        # down. Ranking it first because it is "the more direct column" is how a
        # mid-water instrument comes to be compared as a surface one, so it does not
        # count as a value -- but it is reported below, since a wrongly-declared zero
        # is worth knowing about whichever rung ends up winning.
        if float(np.max(np.abs(finite))) == 0.0:
            placeholders.append(name)
            continue
        candidates[name] = values

    reading = next((n for n in _DEPTH_COLUMNS[:-1] if n in candidates), None)
    pressure = "sea_water_pressure" if "sea_water_pressure" in candidates else None

    chosen, source, approximate = None, None, False
    if reading is not None:
        chosen, source = candidates[reading], reading
    elif pressure is not None:
        chosen, source, approximate = (
            candidates[pressure] * _M_PER_DBAR,
            "sea_water_pressure",
            True,
        )

    # Two rungs that both have a value and disagree: report both rather than letting
    # the ranking decide silently.
    if reading is not None and pressure is not None:
        a = float(np.nanmedian(candidates[reading]))
        b = float(np.nanmedian(candidates[pressure])) * _M_PER_DBAR
        if abs(a - b) > DEPTH_DISAGREEMENT_TOLERANCE:
            warnings.warn(
                f"{subject}: {reading} says {a:g} m but sea_water_pressure says "
                f"{b:g} m. Using {reading}, the more direct reading — pass the depth "
                "yourself if that is the wrong one.",
                stacklevel=_stacklevel.find(),
            )
    if placeholders and source is not None:
        depth_says = float(np.nanmedian(chosen))
        warnings.warn(
            f"{subject}: {'/'.join(placeholders)} is 0 for the whole record, which is "
            f"a placeholder rather than a measurement — {source} puts this instrument "
            f"at {depth_says:g} m. Using {source}.",
            stacklevel=_stacklevel.find(),
        )

    if chosen is None:
        for key in _DEPTH_ATTRS:
            value = meta.get(key)
            if value is not None:
                return float(value), f"metadata:{key}", False
        warnings.warn(
            f"{subject}: no depth found — no depth/z/pressure column with values, and "
            f"no {'/'.join(_DEPTH_ATTRS)} in the catalog entry. Assuming the surface. "
            "Give the entry a nominal_depth_m if it is not.",
            stacklevel=_stacklevel.find(),
        )
        return None, "assumed-surface", False

    finite = chosen[np.isfinite(chosen)]
    if float(np.ptp(finite)) > FIXED_DEPTH_TOLERANCE:
        return chosen, source, approximate
    return float(np.median(finite)), source, approximate


def to_dataset(df, meta: dict[str, Any]):
    """Return a station table as a 1-D :class:`xarray.Dataset` on ``time``.

    Position and depth become **scalar coordinates** (``lon``/``lat``/``depth``, the
    names :mod:`ocean_skill.align` and :mod:`ocean_skill.metrics` look for), so the
    only dimension left is the one a time-series comparison is about. Units ride on each
    variable, so the alignment step's compatibility check and
    :func:`ocean_skill.units.convert_units` work on a mooring exactly as on a grid.

    QARTOD companion columns are **kept** rather than dropped: they are real information
    (and what a future ``qc()`` will read). What must not happen — a request for
    temperature being satisfied by ``sea_water_temperature_qc_agg`` — is refused at the
    matcher (:func:`ocean_skill.units.find_variable`), which protects gridded sources
    carrying flag variables too.
    """
    import pandas as pd
    import xarray as xr

    subject = meta.get("datasetID") or meta.get("title") or "this source"
    time_col = _axis_column(df, meta, "T", "time")
    if time_col is None:
        raise ValueError(
            f"{subject}: no time column, so this table cannot become a time series. "
            "Give the catalog entry an axes mapping such as "
            '`axes={"T": "time (UTC)"}`, or name the column "time".'
        )
    lon_col = _axis_column(df, meta, "X", "longitude")
    lat_col = _axis_column(df, meta, "Y", "latitude")

    # Timezone matters twice over, and both bite silently. A tz-aware coordinate cannot
    # be written to zarr, and cache.save only *warns* when a write fails -- so a
    # tz-aware lane would re-read the whole remote record on every run. Resampling one
    # also yields datetime.datetime labels, which no later join matches. UTC is
    # recorded in attrs rather than thrown away.
    time = pd.to_datetime(df[time_col], errors="coerce", utc=True)
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
            f"(of {len(frame)}) — keeping the first of each.",
            stacklevel=_stacklevel.find(),
        )
        frame = frame.loc[~duplicated]

    units = _units_map(frame, meta)
    lon = _scalar_position(frame, lon_col)
    lat = _scalar_position(frame, lat_col)
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

    depth, depth_source, approximate = depth_of(frame, meta, subject=subject)

    data: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    for column in frame.columns:
        if column in (time_col, lon_col, lat_col):
            continue
        base, _ = split_units(column)
        if base in COORD_COLUMNS:
            # Coordinate columns by name (`z`, `depth`, `altitude`) describe where the
            # measurements were taken; depth_of has already read them, and carrying a
            # flat placeholder alongside the data as if it were a variable invites it
            # being compared. The same exclusion build.py applies to standard_names.
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().all():
            # All-NaN columns are placeholders (OOI's `depth_reading`, `station`), and
            # a non-numeric column would break a bare .mean() the way ROMS' `spherical`
            # flag does -- so a single-valued one is recorded as provenance and the
            # rest are dropped rather than carried as unusable variables.
            original = frame[column].dropna().unique()
            if original.size == 1:
                attrs[base] = original[0]
            continue
        variable = xr.DataArray(values.to_numpy(), dims=("time",), name=base)
        if unit := (units.get(str(column)) or units.get(base)):
            variable.attrs["units"] = unit
        variable.attrs["standard_name"] = base
        variable.attrs["source_column"] = str(column)
        data[base] = variable

    ds = xr.Dataset(data, coords={"time": frame[time_col].to_numpy()})
    ds["time"].attrs["time_zone"] = "UTC"
    if lon is not None:
        ds = ds.assign_coords(lon=lon, lat=lat)
        ds["lon"].attrs.update(units="degrees_east", standard_name="longitude")
        ds["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    if depth is not None:
        value = depth if np.isscalar(depth) else ("time", np.asarray(depth))
        ds = ds.assign_coords(depth=value)
        ds["depth"].attrs.update(
            units="m", positive="down", long_name="instrument depth"
        )
        if not np.isscalar(depth):
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
