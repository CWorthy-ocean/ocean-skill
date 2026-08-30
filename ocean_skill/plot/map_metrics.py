"""Interpolated metric maps for scattered stations — moorings, CTD casts, any point.

A :class:`~ocean_skill.comparison.Comparison` scored ``over="time"`` draws one map per
metric because it has a *grid*: bias, correlation, sigma-ratio each computed cell by
cell against a spatially continuous reference (a satellite record, a climatology). A
mooring network has no such grid — each station is a single point with a single
full-record metric value, and the points are scattered, unevenly spaced, and often
separated by real gaps (open water between survey lines, a strait, a bay nobody has
instrumented). Comparing two dozen scattered numbers on a legend-and-marker map, one
metric at a time, is how ocean-skill's own CIOFS mooring reports have had to show this.

This module is the other way there: it *fits* a smooth surface through the scattered
per-station values (one independent fit per metric, a cross-validated spline via
`verde <https://www.fatiando.org/verde/>`_), masks it by land and by distance from the
nearest station, and hands the result to the same ``skill_map`` family a scored
comparison already draws with — so a bias panel is symmetric about zero and a
correlation panel spans (-1, 1) exactly as it does there, and the panels facet the same
way. The true station values are drawn on top as dots in the same colour scale, so a
reader can always see where the surface has support and where it is only interpolating
across a gap.

**This does not route around land.** Two stations on opposite shores of a peninsula are
blended as if the water between them were open — the interpolation only knows Euclidean
distance in the map's own plane. A model grid's ocean mask (used automatically when
one is available) keeps the surface from being *drawn* on land, but nothing here stops
it from being *influenced* by a station across a barrier it cannot see. Land-locked
seas and convoluted coastlines (Cook Inlet, most obviously) deserve the same scrutiny a
kriged survey product gets in the literature: read the coastline, not just the colour.
Barrier-aware interpolation (DIVAnd) is Julia-only and out of scope here.

Every metric here is a **full-record** statistic — the whole aligned series at that
station, not a per-year or per-season split — unless the caller pools time themselves
first (see ``rows=`` on :func:`build_items`/:func:`map_metrics`, the seasonal-facet
route). See ``docs/skill_maps.md`` for why, and for the multi-year caveats that follow
from it (record lengths differing station to station, most importantly).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from ocean_skill import _stacklevel
from ocean_skill.metrics import DEFAULT_MAP_METRICS, REGISTRY

__all__ = ["build_items", "interpolate_records", "map_metrics"]

#: Positions within this many degrees (~100 m at the equator) count as one station —
#: two references sampled at "the same" mooring, not two nearby moorings.
_COLOCATION_TOL_DEG = 1e-3

#: The distance-mask radius, as a multiple of the stations' own median
#: nearest-neighbour spacing (see :func:`_default_maxdist`) — how far past the
#: densest local spacing the interpolated surface is still trusted, by default.
_DEFAULT_MAXDIST_FACTOR = 3.0

#: A record-length ratio (longest / shortest station record) at or above this is
#: worth a warning: full-record metrics interpolate stations as equals however
#: much data backs each one, and a wide spread is where that matters most.
_UNEVEN_N_RATIO = 5.0

#: Cross-validating a damping needs at least this many stations (sklearn's default
#: k-fold splitter); fewer falls back to a single fixed-damping spline.
_MIN_STATIONS_FOR_CV = 5

#: Interpolation methods :func:`interpolate_records` accepts, from most local/blocky
#: to smoothest — see its ``method`` parameter for what each one trades off.
_INTERP_METHODS = ("nearest", "knn", "linear", "cubic", "spline")


def _position_columns(columns) -> tuple[str, str]:
    """Return the ``(lon, lat)`` column names a record set actually carries.

    A :class:`~ocean_skill.comparison.ComparisonSet`'s series comparisons carry
    ``station_lon``/``station_lat`` (injected in :meth:`Comparison.metrics`); a plain
    table more often just has ``lon``/``lat`` already. Both are accepted so an
    existing CIOFS report table works without renaming a column first.
    """
    for lon_key, lat_key in (("station_lon", "station_lat"), ("lon", "lat")):
        if lon_key in columns and lat_key in columns:
            return lon_key, lat_key
    raise ValueError(
        "no station position found: records need a 'station_lon'/'station_lat' "
        "pair (what a ComparisonSet's series comparisons carry automatically) or a "
        "plain 'lon'/'lat' pair (a DataFrame's own columns)."
    )


def _records_from(data):
    """Return one metric record (row) per station, from any accepted input shape.

    Accepts a :class:`~ocean_skill.comparison.ComparisonSet` (only its *station*
    comparisons — a place through time — contribute; anything else is skipped with
    one warning naming how many), a :class:`pandas.DataFrame`, or a plain iterable
    of dicts.
    """
    import pandas as pd

    from ocean_skill.comparison import ComparisonSet

    if isinstance(data, ComparisonSet):
        series = [c for c in data.comparisons if c.is_series]
        skipped = len(data.comparisons) - len(series)
        if skipped:
            warnings.warn(
                f"{skipped} of {len(data.comparisons)} comparisons are not a place "
                "through time (no single position) and were skipped — map_metrics "
                "only maps stations.",
                stacklevel=_stacklevel.find(),
            )
        if not series:
            raise ValueError("no station comparisons to map: every one was skipped")
        return pd.DataFrame([c.metrics() for c in series])
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame([dict(r) for r in data])


def _test_name_of(data) -> str | None:
    """The model source a :class:`ComparisonSet` was tested against, or ``None``.

    ``None`` for anything else (a plain table has no test source of its own — pass
    ``test=`` to :func:`build_items` to name one), and also when the set mixes test
    sources, with a warning naming them, since there is then no single grid to pick
    without being told which.
    """
    from ocean_skill.comparison import ComparisonSet

    if not isinstance(data, ComparisonSet) or not data.comparisons:
        return None
    names = {c.test_name for c in data.comparisons}
    if len(names) > 1:
        warnings.warn(
            f"this set compares against {len(names)} different test sources "
            f"({sorted(names)}) — using {data.comparisons[0].test_name!r}'s grid. "
            "Pass grid='regular' (or test=<name>) to choose deliberately.",
            stacklevel=_stacklevel.find(),
        )
    return data.comparisons[0].test_name


def _reduce_duplicates(df, lon_key: str, lat_key: str, metric_names: Sequence[str]):
    """Collapse records at (near-)identical positions to their median, with a warning.

    Two references occasionally land within meters of each other — the same physical
    mooring reported under two catalog entries, most often. Interpolation wants one
    value per position; pooling to the median (rather than, say, keeping the first)
    means one mis-scoped duplicate cannot swing the surface on its own.
    """
    key = df[lon_key].round(3).astype(str) + "," + df[lat_key].round(3).astype(str)
    if key.nunique() == len(df):
        return df
    agg: dict[str, Any] = {lon_key: "first", lat_key: "first"}
    agg.update({name: "median" for name in metric_names if name in df.columns})
    if "reference" in df.columns:
        agg["reference"] = "first"
    reduced = df.groupby(key, sort=False).agg(agg).reset_index(drop=True)
    warnings.warn(
        f"{len(df) - len(reduced)} record(s) shared a position (within "
        f"{_COLOCATION_TOL_DEG}°) with another and were pooled to their median "
        f"before interpolating — {len(df)} records became {len(reduced)} positions.",
        stacklevel=_stacklevel.find(),
    )
    return reduced


def _warn_uneven_records(df) -> None:
    """Warn when station record lengths vary widely — see the module docstring."""
    if "n" not in df.columns:
        return
    n = df["n"].dropna().to_numpy(dtype="float64")
    n = n[n > 0]
    if n.size < 2:
        return
    lo, hi = float(n.min()), float(n.max())
    if hi / lo >= _UNEVEN_N_RATIO:
        warnings.warn(
            f"station record lengths range from {lo:g} to {hi:g} valid pairs, a "
            f"{hi / lo:.0f}x spread — every station's full-record metric is "
            "interpolated as an equal, however much data backs it. This is a "
            "property of mapping full-record metrics (see the module's docstring "
            "on ``rows=`` for a seasonal or per-era alternative), not a bug.",
            stacklevel=_stacklevel.find(),
        )


def _central_longitude(lon: np.ndarray) -> float:
    """Circular mean longitude: the antimeridian-safe centre for a local projection.

    A plain mean fails a station set straddling the dateline (180 and -179 average
    to 0.5 — the opposite side of the world); the mean of unit vectors does not,
    whichever longitude convention (0-360 or -180-180) the input happens to use.
    """
    rad = np.deg2rad(lon)
    return float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def _projector(lon: np.ndarray, lat: np.ndarray):
    """A local Mercator projection centred on the stations.

    Mercator rather than a plain degree-scaling: verde's spline assumes a roughly
    Euclidean plane, and Mercator is conformal (true local shape and distance
    ratios) at the cost of area, which an interpolated statistic never needs.
    Centred on the *data's* own circular-mean longitude (:func:`_central_longitude`)
    rather than 0° or a hemisphere convention, so a station set straddling the
    antimeridian is never split across the projection's own seam.
    """
    import pyproj

    return pyproj.Proj(
        proj="merc",
        lon_0=_central_longitude(lon),
        lat_ts=float(np.mean(lat)),
        ellps="WGS84",
    )


def _default_maxdist(easting: np.ndarray, northing: np.ndarray) -> float:
    """A distance-mask radius (metres) derived from the stations' own spacing.

    The median nearest-neighbour distance between stations, times
    :data:`_DEFAULT_MAXDIST_FACTOR` — median rather than a percentile so one
    isolated station cannot stretch everyone else's mask outward, and derived from
    the data rather than fixed so a tight cluster of moorings and a basin-scale
    survey each get a trust radius matched to their own density.
    """
    xy = np.column_stack([easting, northing])
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    return float(np.median(nearest)) * _DEFAULT_MAXDIST_FACTOR


def _fit_spline(easting, northing, values, *, name: str):
    """Fit a verde spline to one metric's scattered values.

    Cross-validated (:class:`verde.SplineCV`) when there are enough stations to
    cross-validate a damping (sklearn's default k-fold splitter needs
    :data:`_MIN_STATIONS_FOR_CV`); fewer falls back to a single fixed-damping
    :class:`verde.Spline`, with a warning, rather than raising or silently
    guessing at an untested damping.
    """
    import verde as vd

    if len(values) < _MIN_STATIONS_FOR_CV:
        warnings.warn(
            f"only {len(values)} stations carry {name!r} — too few to "
            "cross-validate a smoothing parameter, so a single fixed-damping "
            "spline is used instead. Treat this metric's map as a rougher sketch "
            "than the others.",
            stacklevel=_stacklevel.find(),
        )
        spline = vd.Spline()
    else:
        spline = vd.SplineCV(dampings=(1e-10, 1e-5, 1e-3, 1e-1, 1e0))
    spline.fit((easting, northing), np.asarray(values, dtype="float64"))
    return spline


def _make_gridder(method: str, knn_k: int):
    """Return an unfit verde gridder for one of the non-spline ``method`` choices.

    ``"spline"`` itself is not handled here — it keeps its own cross-validated path
    (:func:`_fit_spline`), which this would otherwise have to duplicate or downgrade
    to a plain, non-cross-validated :class:`verde.Spline`.
    """
    import verde as vd

    if method in ("nearest", "voronoi"):
        return vd.KNeighbors(k=1)  # each cell takes its single nearest station's value
    if method == "knn":
        return vd.KNeighbors(k=knn_k, reduction=np.mean)
    if method == "linear":
        return vd.Linear()
    if method == "cubic":
        return vd.Cubic()
    raise ValueError(f"method={method!r} — expected one of {_INTERP_METHODS}")


def _model_grid(test_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(lon2d, lat2d, ocean_mask)`` from a test source's own grid, or ``None``.

    Opens the source — unlike
    :func:`~ocean_skill.comparison._domain_of`, which stays metadata-only — because
    the grid itself, not just its bounding box, is what lets the interpolated
    surface stop at the coastline instead of painting straight over it. Returns
    ``None`` when the source cannot be read or declares no lon/lat, which sends the
    caller to the regular-grid fallback instead.
    """
    from ocean_skill.align import grid_of
    from ocean_skill.sources import read

    try:
        ds = read(test_name)
    except Exception:
        return None
    if not hasattr(ds, "coords") or "lon" not in ds.coords or "lat" not in ds.coords:
        return None
    grid = grid_of(ds)
    lon = np.asarray(grid["lon"], dtype="float64")
    lat = np.asarray(grid["lat"], dtype="float64")
    if lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    if "mask_rho" in ds.variables:
        ocean = np.asarray(ds["mask_rho"]) == 1
    else:
        warnings.warn(
            f"{test_name!r} declares no land mask ('mask_rho') — the interpolated "
            "surface will cover its whole grid, land included. The map's own "
            "coastline is still drawn on top, but the colour underneath it is not "
            "physically meaningful there.",
            stacklevel=_stacklevel.find(),
        )
        ocean = np.ones_like(lon, dtype=bool)
    return lon, lat, ocean


def interpolate_records(
    records,
    metric_names: Sequence[str] | None = None,
    *,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    spacing: float | None = None,
    maxdist: float | None = None,
    method: str = "spline",
    knn_k: int = 5,
    block_spacing: float | None = None,
) -> xr.Dataset:
    """Interpolate scattered per-station metric values onto a map.

    ``records`` is one row per station (a :class:`~pandas.DataFrame` or a list of
    dicts), each carrying a position (``station_lon``/``station_lat`` or
    ``lon``/``lat``) and one column per name in ``metric_names`` (default
    :data:`~ocean_skill.metrics.DEFAULT_MAP_METRICS`). Returns an ``xr.Dataset``
    with one 2-D variable per metric, ``lon``/``lat`` as 2-D coordinates — the same
    shape :meth:`~ocean_skill.comparison.Comparison.pointwise_metrics` returns for a
    single scored comparison, so it draws through the ``skill_map`` family unchanged.

    Each metric is fit **independently** (a station good at correlation need not be
    good at bias) with a verde spline in a local Mercator projection centred on the
    stations (see :func:`_projector`), then masked twice: by ``grid``'s ocean mask
    when a model grid was given, and always by distance from the nearest station
    (:func:`verde.distance_mask`) — a smooth surface is not evidence for a claim
    hundreds of kilometres from the nearest station. See the module docstring for
    what this does *not* do (route around land).

    ``grid``, if given, is ``(lon2d, lat2d, ocean_mask)`` (see :func:`_model_grid`,
    which builds this from a model source's own grid). ``None`` — the default for a
    plain table with no named model — interpolates onto a regular grid over the
    padded station extent instead, at ``spacing`` degrees (default: a fortieth of
    the extent's larger span). ``maxdist`` is the distance-mask radius in metres;
    ``None`` derives it from the stations' own spacing (:func:`_default_maxdist`).

    Duplicate (near-identical) positions are pooled to their median first, and
    stations with widely different record lengths raise a warning — see
    :func:`_reduce_duplicates` and :func:`_warn_uneven_records`.

    ``method`` picks the interpolator, trading smoothness for honesty about where
    the data actually says something:

    * ``"spline"`` (default) — the smooth, cross-validated fit described above.
      Good where a gradual gradient between stations is plausible, but it can
      invent one across water two stations' values say nothing about (blending a
      good station into a bad one across a strait it cannot see).
    * ``"nearest"`` — each cell takes its single nearest station's value (Voronoi
      tiles): hard-edged, disjoint, and invents nothing between stations. It also
      adapts to density for free — a dense cluster gets small tiles, an isolated
      station a large one — without any parameter to tune.
    * ``"knn"`` — the mean of the ``knn_k`` nearest stations: still local, with
      softer edges than ``"nearest"``.
    * ``"linear"`` / ``"cubic"`` — piecewise over a Delaunay triangulation of the
      stations: faceted rather than smooth, and only defined inside the data's
      convex hull (``NaN`` elsewhere — no extrapolation into empty water).

    ``block_spacing``, if given, is a distance in the same units as the
    projection (metres) over which stations are pooled to their median
    (:class:`verde.BlockMean`) before fitting — for any ``method``. This keeps a
    dense cluster of stations (a repeat CTD survey, say) from dominating a
    sparser region purely by outnumbering it, independent of the interpolator's
    own duplicate-position handling (:func:`_reduce_duplicates`, which only
    merges near-*identical* positions).
    """
    import pandas as pd
    import verde as vd

    if method not in _INTERP_METHODS:
        raise ValueError(f"method={method!r} — expected one of {_INTERP_METHODS}")
    metric_names = tuple(metric_names) if metric_names else DEFAULT_MAP_METRICS
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    if df.empty:
        raise ValueError("no station records to interpolate")
    lon_key, lat_key = _position_columns(df.columns)
    missing_pos = df[lon_key].isna() | df[lat_key].isna()
    if missing_pos.any():
        raise ValueError(
            f"{int(missing_pos.sum())} of {len(df)} records carry no position "
            f"({lon_key!r}/{lat_key!r} is null) — map_metrics needs every "
            "station's location."
        )
    missing_metrics = [m for m in metric_names if m not in df.columns]
    if missing_metrics:
        available = [c for c in df.columns if c in REGISTRY]
        raise ValueError(
            f"no {missing_metrics} column(s) in these records; this set carries "
            f"{available}. Pass metrics=(...) naming what was actually computed."
        )
    df = _reduce_duplicates(df, lon_key, lat_key, metric_names)
    _warn_uneven_records(df)

    lon = df[lon_key].to_numpy(dtype="float64")
    lat = df[lat_key].to_numpy(dtype="float64")
    proj = _projector(lon, lat)
    easting, northing = proj(lon, lat)

    if grid is not None:
        glon, glat, ocean = grid
        geast, gnorth = proj(glon, glat)
    else:
        pad_lon = max(0.1 * (lon.max() - lon.min()), 0.5)
        pad_lat = max(0.1 * (lat.max() - lat.min()), 0.5)
        lon0, lon1 = lon.min() - pad_lon, lon.max() + pad_lon
        lat0, lat1 = lat.min() - pad_lat, lat.max() + pad_lat
        if spacing is None:
            spacing = max(lon1 - lon0, lat1 - lat0) / 40.0
        glon, glat = np.meshgrid(
            np.arange(lon0, lon1 + spacing, spacing),
            np.arange(lat0, lat1 + spacing, spacing),
        )
        geast, gnorth = proj(glon, glat)
        ocean = np.ones_like(glon, dtype=bool)

    maxdist = maxdist if maxdist is not None else _default_maxdist(easting, northing)
    trusted = vd.distance_mask(
        (easting, northing), maxdist, coordinates=(geast, gnorth)
    )

    data_vars = {}
    for name in metric_names:
        if method == "spline":
            fitter = _fit_spline(easting, northing, df[name].to_numpy(), name=name)
        else:
            values = df[name].to_numpy(dtype="float64")
            good = np.isfinite(values)
            e, n, v = easting[good], northing[good], values[good]
            if block_spacing:
                (e, n), v, _ = vd.BlockMean(spacing=block_spacing).filter((e, n), v)
            fitter = _make_gridder(method, knn_k)
            fitter.fit((e, n), v)
        predicted = fitter.predict((geast, gnorth))
        predicted = np.where(ocean & trusted, predicted, np.nan)
        metric = REGISTRY.get(name)
        data_vars[name] = xr.DataArray(
            predicted,
            dims=("y", "x"),
            attrs={
                "long_name": metric.long_name if metric is not None else name,
                "units": metric.units if metric is not None else "",
            },
        )
    return xr.Dataset(
        data_vars, coords={"lon": (("y", "x"), glon), "lat": (("y", "x"), glat)}
    )


def build_items(
    data=None,
    *,
    metrics: Sequence[str] | None = None,
    test: str | None = None,
    grid: str = "model",
    spacing: float | None = None,
    maxdist: float | None = None,
    method: str = "spline",
    knn_k: int = 5,
    block_spacing: float | None = None,
    rows: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the ``skill_map`` family's items: one interpolated row per entry.

    ``data`` is a :class:`~ocean_skill.comparison.ComparisonSet` of station
    comparisons (positions come from each comparison automatically — see
    :meth:`~ocean_skill.comparison.Comparison.metrics`) or a plain table: a
    :class:`~pandas.DataFrame` or list of dicts carrying a position and metric
    columns, such as an existing CIOFS report's metrics CSV.

    ``rows={label: data, ...}`` builds one row per entry instead of one figure —
    the seasonal facet: pool each season's comparisons (or metrics table) apart
    first, one entry per season, and this draws them as :func:`Comparison.plot`'s
    scored comparisons already stack rows, metrics across and seasons down. Pass
    ``rows=`` alone, with no ``data`` — each row supplies its own.

    ``grid="model"`` (the default) interpolates onto the test source's own grid,
    which gives the surface a real ocean mask for free; ``test=`` names it
    explicitly (required for a plain table, since it has no test source of its
    own), or it is read off a :class:`ComparisonSet`. Falls back to a regular
    lon/lat grid, with a warning, when no grid is available; ``grid="regular"``
    always uses the fallback.

    ``method``/``knn_k``/``block_spacing`` are forwarded to
    :func:`interpolate_records` — see its docstring for what each interpolation
    method trades off (smooth vs. honest-about-gaps) and what ``block_spacing``
    fixes for uneven station density.
    """
    metric_names = tuple(metrics) if metrics else DEFAULT_MAP_METRICS
    if rows is None:
        if data is None:
            raise ValueError("nothing to map: pass data=, or rows={label: data, ...}")
        entries = {"": data}
    else:
        entries = dict(rows)
    if not entries:
        raise ValueError("rows={} names no data to map")

    if grid not in ("model", "regular"):
        raise ValueError(f"grid={grid!r} — expected 'model' or 'regular'")

    model_grid = None
    if grid == "model":
        name = test
        if name is None:
            names = {_test_name_of(entry) for entry in entries.values()} - {None}
            if len(names) == 1:
                (name,) = names
            elif len(names) > 1:
                # Each entry may itself be single-source (so _test_name_of raised no
                # warning of its own) while the *rows* disagree with each other — a
                # seasonal facet built from two different test runs, say.
                warnings.warn(
                    f"these rows compare against {len(names)} different test "
                    f"sources ({sorted(names)}) — falling back to a regular lon/lat "
                    "grid, since there is no single grid to pick without being told "
                    "which. Pass test=<name> to choose one, or grid='regular' to "
                    "silence this.",
                    stacklevel=_stacklevel.find(),
                )
        if name is not None:
            model_grid = _model_grid(name)
            if model_grid is None:
                warnings.warn(
                    f"{name!r} has no readable grid — falling back to a regular "
                    "lon/lat grid over the stations' own extent. Pass "
                    "grid='regular' to silence this, or test=<a readable source>.",
                    stacklevel=_stacklevel.find(),
                )

    items: list[dict[str, Any]] = []
    for label, entry in entries.items():
        df = _records_from(entry)
        lon_key, lat_key = _position_columns(df.columns)
        skill = interpolate_records(
            df, metric_names, grid=model_grid, spacing=spacing, maxdist=maxdist,
            method=method, knn_k=knn_k, block_spacing=block_spacing,
        )
        item: dict[str, Any] = {
            "skill": skill,
            "metric_names": metric_names,
            "stations": {
                "lon": df[lon_key].to_numpy(dtype="float64"),
                "lat": df[lat_key].to_numpy(dtype="float64"),
                "names": (
                    df["reference"].tolist() if "reference" in df.columns else None
                ),
                "values": {
                    name: df[name].to_numpy(dtype="float64") for name in metric_names
                },
            },
        }
        if label:
            item["row_label"] = label
        items.append(item)
    return items


def map_metrics(
    data=None,
    *,
    metrics: Sequence[str] | None = None,
    test: str | None = None,
    grid: str = "model",
    spacing: float | None = None,
    maxdist: float | None = None,
    method: str = "spline",
    knn_k: int = 5,
    block_spacing: float | None = None,
    rows: Mapping[str, Any] | None = None,
    renderer: str = "matplotlib",
    mark: str = "contourf",
    **plot_kwargs: Any,
):
    """Map a metric (or several), interpolated across many stations' comparisons.

    ::

        osk.map_metrics(mooring_set)                  # bias, crmsd, corr, sigma_ratio
        osk.map_metrics(mooring_set, metrics=("corr", "n"))
        osk.map_metrics(mooring_set, grid="regular")        # skip the model's own grid
        osk.map_metrics(metrics_df, test="ciofs3")          # a plain CIOFS report table
        osk.map_metrics(rows={"DJF": winter_set, "JJA": summer_set})  # seasonal facet
        osk.map_metrics(mooring_set, method="nearest")      # Voronoi tiles, no smoothing
        osk.map_metrics(dense_ctd_set, block_spacing=15_000)  # pool a dense cluster first

    Each panel is one metric's per-station values (see
    :meth:`~ocean_skill.comparison.Comparison.metrics`) fit to a smooth surface
    (:func:`interpolate_records`) and drawn with the same colour policy and layout
    ``compare(..., over="time")`` uses for a gridded comparison's own metric maps
    (:mod:`ocean_skill.plot.matplotlib_renderer`'s ``skill_map``) — the true
    station values are overlaid as dots in the same colour scale, so a reader can
    see where the surface has support and where it is filling a gap. See the
    module docstring for what the interpolation cannot do (route around land).

    Every argument through ``rows`` builds the figure's data (see
    :func:`build_items`, which this delegates to); everything else is a plot
    option forwarded to the renderer, exactly as for any other family
    (``docs/plot_styling_reference.md``). ``method``/``knn_k``/``block_spacing``
    choose the interpolator (see :func:`interpolate_records`).
    """
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = build_items(
        data,
        metrics=metrics,
        test=test,
        grid=grid,
        spacing=spacing,
        maxdist=maxdist,
        method=method,
        knn_k=knn_k,
        block_spacing=block_spacing,
        rows=rows,
    )
    plot_kwargs.setdefault("mark", mark)
    return render(
        PlotSpec(family="skill_map", items=items, options=plot_kwargs),
        renderer=renderer,
    )
