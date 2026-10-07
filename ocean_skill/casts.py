"""The casts of a CTD transect, each addressable as a ``profile`` source of its own.

A transect is one ship visit to a line of stations: one file holding many casts, each
at its own time and position -- CF ``featureType: trajectoryProfile``. Every cast of
such an entry is a source in its own right, named ``"<entry>[<cast id>]"`` (or
``"<catalog>:<entry>[<cast id>]"``): :func:`ocean_skill.catalog.resolve` hands back the
cast's own metadata (``featureType: "profile"``, the cast's time and position as its
extents) and :func:`ocean_skill.sources.read` returns just that cast's samples, so
everything that compares a profile compares a cast the same way.

Which samples make a cast comes from, in order:

1. the column (table) or variable (netCDF) named by the entry's ``casts: {"id": ...}``;
2. a netCDF variable carrying CF ``cf_role: profile_id``;
3. otherwise, splitting the samples (in time order) wherever two consecutive ones are
   more than ``casts["gap"]`` apart in time (default :data:`DEFAULT_GAP`) or more than
   ``casts["distance_m"]`` apart in space (default :data:`DEFAULT_DISTANCE_M`).

With an id, a cast is a run of consecutive samples (in time order) sharing one id
value; a value that comes back in a later, separate run (a station revisited) names a
second cast, ``"<id>#2"``. Without one, casts are numbered ``1``, ``2``, ... in time
order, zero-padded to the width of the count. (In a netCDF file with a profile
dimension, each profile is a cast, whatever its neighbours' ids.)

A cast's **time** is its earliest sample -- the rule a ``profile`` already follows. Its
**position** is the median of its samples' longitudes and latitudes, so a ship that
drifts during the cast still gives one place (``casts["position"]``: ``"median"``,
``"first"`` -- the earliest sample's -- or ``"mean"``).

Casts are listed in time order, the order a ship visits them, which is the order a
section built from them runs in.
"""

from __future__ import annotations

import copy
import dataclasses
import re
import warnings
from typing import Any

import numpy as np

from ocean_skill import _stacklevel

#: A gap in time longer than this between consecutive samples starts a new cast.
DEFAULT_GAP = "15min"
#: A jump in position longer than this (metres) between consecutive samples starts a
#: new cast.
DEFAULT_DISTANCE_M = 200.0
#: How a cast's one position is taken from its samples'.
DEFAULT_POSITION = "median"
POSITIONS = ("median", "first", "mean")

_CAST_NAME = re.compile(r"^(?P<base>.+)\[(?P<cast>[^\[\]]+)\]$")
#: The keys of an entry's ``casts`` declaration.
_KEYS = ("id", "gap", "distance_m", "position")
#: What :func:`ocean_skill.cf.find_coord` calls each axis.
_KINDS = {"T": "time", "X": "longitude", "Y": "latitude", "Z": "vertical"}
#: ``{source: (freshness, table)}``: :func:`table`'s memo, emptied with ``read``'s.
_TABLES: dict[str, tuple[tuple, Any]] = {}


def split_name(name: str) -> tuple[str, str | None]:
    """Return ``(base, cast id)`` for ``"<base>[<cast id>]"``, else ``(name, None)``."""
    match = _CAST_NAME.match(name) if isinstance(name, str) else None
    if match is None:
        return name, None
    return match["base"], match["cast"]


def cast_name(base: str, cast_id: str) -> str:
    """Return the source name of cast ``cast_id`` of ``base``."""
    return f"{base}[{cast_id}]"


def is_transect(source: str) -> bool:
    """Whether ``source`` is a ``trajectoryProfile`` entry, compared cast by cast.

    Read-free (catalog metadata only). ``False`` for a cast name or an unresolvable
    source.
    """
    from ocean_skill import catalog

    if split_name(source)[1] is not None:
        return False
    try:
        meta = catalog.resolve(source).metadata or {}
    except (KeyError, TypeError):
        return False
    return str(meta.get("featureType") or "") == "trajectoryProfile"


def _gap(value: Any):
    """Return ``value`` as a positive Timedelta (numbers are seconds), or ``None``."""
    import pandas as pd

    try:
        span = pd.to_timedelta(value, unit=None if isinstance(value, str) else "s")
    except (TypeError, ValueError):
        return None
    return span if isinstance(span, pd.Timedelta) and span > pd.Timedelta(0) else None


#: What each key of ``casts`` must be: a test, and how to say it.
_RULES = {
    "id": (lambda v: isinstance(v, str) and v.strip(), "a column or variable name"),
    "gap": (lambda v: _gap(v) is not None, "a positive time span such as '15min'"),
    "distance_m": (
        lambda v: (
            isinstance(v, int | float) and not isinstance(v, bool) and 0 < v < np.inf
        ),
        "a positive number of metres",
    ),
    "position": (lambda v: v in POSITIONS, f"one of {POSITIONS}"),
}


def canonicalize(spec: Any) -> dict[str, Any] | None:
    """Validate an entry's ``casts`` declaration; return it canonically.

    ``None`` or an empty mapping is no declaration (``None``). Keys: ``id`` (a column
    or variable name), ``gap`` (a time span pandas reads, e.g. ``"15min"``; a bare
    number is seconds), ``distance_m`` (a positive number) and ``position`` (one of
    :data:`POSITIONS`). Any problem is a ``ValueError`` whose message starts
    ``"casts: "``.
    """
    if spec is None or spec == {}:
        return None
    if not isinstance(spec, dict) or set(spec) - set(_KEYS):
        raise ValueError(f"casts: keys are {', '.join(_KEYS)}, got {spec!r}")
    for key in spec:
        if not _RULES[key][0](spec[key]):
            raise ValueError(f"casts: {key} is {_RULES[key][1]}, got {spec[key]!r}")
    out = {key: spec[key] for key in _KEYS if key in spec}
    if isinstance(out.get("gap"), str):
        out["gap"] = out["gap"].strip()
    if "distance_m" in out:
        out["distance_m"] = float(out["distance_m"])
    return out


def table(source: str):
    """Return ``source``'s casts as a :class:`pandas.DataFrame`, one row per cast.

    Columns ``id`` (str), ``name`` (``"<source>[<id>]"``), ``time`` (UTC, naive
    ``Timestamp``), ``lon``, ``lat`` and ``n`` (samples), in time order. Reads the
    source (:func:`ocean_skill.sources.read`); memoized, until the catalog file changes.
    """
    from ocean_skill import cache, catalog, sources

    if not is_transect(source):
        raise ValueError(f"{source!r} is not a trajectoryProfile source: no casts.")
    ref = catalog.resolve(source)
    key = sources._read_cache_key(ref, None, {}) if cache.enabled() else None
    if key is not None and _TABLES.get(source, (None,))[0] == key:
        return _TABLES[source][1].copy()
    out = _casts_of(sources.read(source), ref.metadata, source).drop(columns="selector")
    out.insert(1, "name", [cast_name(source, i) for i in out["id"]])
    if key is not None:
        _TABLES[source] = (key, out)
    return out.copy()


def names(source: str) -> list[str]:
    """Return the source names of ``source``'s casts, in time order."""
    return list(table(source)["name"])


def resolve_cast(parent, base: str, cast_id: str):
    """Return the :class:`~ocean_skill.catalog.SourceRef` of one cast of ``parent``.

    Used by :func:`ocean_skill.catalog.resolve`; ``base`` is the parent's name as it was
    written. Raises ``KeyError`` naming the known casts when there is no such cast.
    """
    if not is_transect(base):
        raise KeyError(f"{base!r} has no casts: it is not a trajectoryProfile source.")
    found = table(base)
    rows = found[found["id"] == cast_id]
    if rows.empty:
        raise KeyError(
            f"{base!r} has no cast {cast_id!r}; its casts: {list(found['id'][:10])}"
        )
    row = rows.iloc[0]
    when = row["time"].isoformat(timespec="seconds")
    lon, lat = float(row["lon"]), float(row["lat"])
    n = int(row["n"])
    info = {"id": cast_id, "of": base, "time": when, "lon": lon, "lat": lat, "n": n}
    meta = {k: copy.deepcopy(v) for k, v in parent.metadata.items() if k != "casts"}
    meta.update(featureType="profile", minTime=when, maxTime=when, cast=info)
    meta.update(dict.fromkeys(("geospatial_lon_min", "geospatial_lon_max"), lon))
    meta.update(dict.fromkeys(("geospatial_lat_min", "geospatial_lat_max"), lat))
    meta.update(dict.fromkeys(("time_coverage_start", "time_coverage_end"), when[:10]))
    return dataclasses.replace(parent, cast=cast_id, metadata=meta)


def read_cast(ref, *, qc: Any = None, **kwargs: Any):
    """Return one cast's samples: a ``DataFrame`` for a table, a ``Dataset`` otherwise.

    Used by :func:`ocean_skill.sources.read` for a ``ref`` whose ``cast`` is set. A
    netCDF cast reads like a profile file: indexed on its depths (finite, ascending, the
    first of a repeat), its time and position scalar coordinates under the file's names.
    """
    from ocean_skill import catalog, sources, tabular

    of = ref.metadata["cast"]["of"]
    parent, meta = sources.read(of, qc=qc, **kwargs), catalog.resolve(of).metadata
    found = _casts_of(parent, meta, of).set_index("id")
    if ref.cast not in found.index:
        raise KeyError(f"{of!r} has no cast {ref.cast!r} in the data it reads now.")
    cast = found.loc[ref.cast]
    if tabular.is_frame(parent):
        return parent.iloc[cast["selector"]]
    tn, xn, yn, zn = (_axis(parent, meta, axis, of) for axis in "TXYZ")
    sub = parent.isel(cast["selector"]).drop_vars([tn, xn, yn])
    dim = sub[zn].dims[0]  # the level (multidimensional) or sample (flat) dimension
    depths, first = np.unique(sub[zn].to_numpy(), return_index=True)  # NaN sorts last
    sub = sub.isel({dim: first[np.isfinite(depths)]})
    when = cast["time"].to_datetime64().astype("datetime64[ns]")  # as every obs time
    where = {tn: when, xn: cast["lon"], yn: cast["lat"]}
    scalars = {n: ((), v, parent[n].attrs) for n, v in where.items()}
    sub = sub.swap_dims({dim: zn}).assign_coords(scalars)
    return sub.assign_attrs(featureType="profile")


def _axis(obj, meta: dict[str, Any], axis: str, subject: str) -> str:
    """Return the name of ``obj``'s column or variable for ``axis`` (axes=, else CF)."""
    from ocean_skill import tabular
    from ocean_skill.cf import find_coord

    if tabular.is_frame(obj):
        name = tabular._axis_column(obj, meta, axis)
    else:
        name = (meta.get("axes") or {}).get(axis)
        if name not in obj.variables:
            name = getattr(find_coord(obj, _KINDS[axis]), "name", None)
    if name is None:
        raise ValueError(
            f"{subject}: no {_KINDS[axis]} column or variable; axes= names it."
        )
    return name


def _text(value: Any) -> str | None:
    """Return one id value as text (``3.0`` is ``"3"``), or ``None`` if missing."""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if isinstance(value, float | np.floating):
        value = None if np.isnan(value) else int(value) if value.is_integer() else value
    return None if value is None else str(value).strip() or None


def _casts_of(obj, meta: dict[str, Any], subject: str):
    """Return a source's casts: :func:`table`'s columns, plus a ``selector``."""
    opts = canonicalize(meta.get("casts")) or {}
    return _group(*_samples(obj, meta, opts, subject), opts=opts, subject=subject)


def _samples(obj, meta: dict[str, Any], opts: dict[str, Any], subject: str):
    """Return the samples of a table (a row each) or a Dataset, for :func:`_group`.

    A Dataset is flat (time and depth on one dimension: a sample per row) or
    multidimensional (data on profile, level: a sample, and a cast, per profile).
    """
    import pandas as pd

    from ocean_skill import tabular

    frame = tabular.is_frame(obj)
    tn, xn, yn = (_axis(obj, meta, axis, subject) for axis in "TXY")
    degrees = [(np.asarray(obj[n]), a) for n, a in ((xn, "X"), (yn, "Y"))]
    x, y = (
        tabular.numeric_in_range(pd.Series(v), a).to_numpy(float) for v, a in degrees
    )
    if frame:
        when = tabular.decode_time_column(obj[tn], tn, meta, subject=subject)
        t = when.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
        counts, dim, merge, idn = np.ones(len(obj), int), None, True, opts.get("id")
        have = [str(c) for c in obj.columns]
    else:
        if any("sample_dimension" in v.attrs for v in obj.variables.values()):
            raise ValueError(
                f"{subject}: its casts are a ragged array, which is not read yet; "
                "convert it to the multidimensional or the flat layout."
            )
        t, z = obj[tn], obj[_axis(obj, meta, "Z", subject)]
        if t.ndim != 1 or t.dtype.kind != "M":
            raise ValueError(f"{subject}: time {tn!r} must be 1-D datetimes.")
        dim, merge = t.dims[0], z.dims == t.dims  # flat: samples run on into casts
        levels = z if dim in z.dims else z.expand_dims({dim: t.size})
        finite = np.isfinite(levels.transpose(dim, ...).to_numpy())
        counts = finite.reshape(t.size, -1).sum(1)
        have = [str(n) for n in obj.variables if obj[n].dims == (dim,)]
        roles = [n for n in have if obj[n].attrs.get("cf_role") == "profile_id"]
        t, idn = t.to_numpy(), opts.get("id") or next(iter(roles), None)
    if idn is not None and idn not in have:
        raise ValueError(f"{subject}: casts id {idn!r} must be one of {have}.")

    def pick(rows):
        return rows if dim is None else {dim: rows if merge else int(rows[0])}

    ids = None if idn is None else [_text(v) for v in np.asarray(obj[idn])]
    return t, x, y, ids, counts, pick, merge


def _group(t, x, y, ids, counts, pick, merge, *, opts, subject):
    """Cut samples (``t``, ``x``, ``y``, ``ids``: one each) into casts.

    ``counts`` is the readings a sample stands for, ``pick`` makes a cast's selector
    (row positions of a table, an ``isel`` indexer of a Dataset) from its sample
    positions, ``merge`` whether neighbouring samples run together into a cast.
    """
    import pandas as pd

    frame = pd.DataFrame({"time": t, "lon": x, "lat": y, "n": counts, "id": ids})
    bad = frame["time"].isna() | (ids is not None and frame["id"].isna())
    if lost := int(bad.sum()):
        warnings.warn(
            f"{subject}: {lost} of {len(frame)} samples have no time (or cast id) and "
            "belong to no cast.",
            stacklevel=_stacklevel.find(),
        )
    frame = frame[~bad].sort_values("time", kind="stable")  # ties keep the file's order
    if frame.empty:
        raise ValueError(f"{subject}: no sample has a time (and a cast id).")
    if not merge:
        new = np.ones(len(frame), bool)
    elif ids is not None:
        same = frame["id"].to_numpy()
        new = np.r_[True, same[1:] != same[:-1]]
    else:
        from ocean_skill.align import _haversine_km

        lon, lat = frame["lon"].to_numpy(), frame["lat"].to_numpy()
        apart = _haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]) * 1000
        wait = frame["time"].diff().iloc[1:] > _gap(opts.get("gap", DEFAULT_GAP))
        far = apart > opts.get("distance_m", DEFAULT_DISTANCE_M)
        new = np.r_[True, wait.to_numpy() | far]

    how = opts.get("position", DEFAULT_POSITION)
    groups = frame.groupby(np.cumsum(new) - 1)
    casts = groups.agg(
        {"id": "first", "time": "first", "lon": how, "lat": how, "n": "sum"}
    )
    if ids is None:
        casts["id"] = (casts.index + 1).astype(str).str.zfill(len(str(len(casts))))
    else:  # a station visited again is "<id>#2", ...
        ident, visit = casts["id"], casts.groupby("id").cumcount() + 1
        casts["id"] = ident.where(visit == 1, ident + "#" + visit.astype(str))
    casts["selector"] = [pick(rows.to_numpy()) for rows in groups.groups.values()]
    return casts.reset_index(drop=True)
