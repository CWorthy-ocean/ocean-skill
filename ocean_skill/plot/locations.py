"""Map where catalog datasets are, from their metadata alone.

The catalog probe records each entry's ``geospatial_*`` extent, ``featureType`` and
descriptive metadata, so a map of *where the datasets are* is a pure metadata
operation — nothing is opened, nothing is read. :func:`build_items` turns catalog
entries (all of them, a ``find()`` result, or one catalog) into the flat item dicts
the ``"locations"`` plot family draws, and :func:`map_datasets` is the public
entry point: build items, wrap them in a :class:`~ocean_skill.plot.spec.PlotSpec`,
route through the renderer registry like every other family.

Non-gridded feature types (a mooring, a profile, a track) become scatter markers at
their bounding box's midpoint; ``grid`` datasets become dashed extent rectangles.
Both renderers colour by featureType off the same constants here, so the static and
interactive maps cannot disagree about what a timeSeries looks like.

Longitudes are always normalised to the −180..180 frame, whatever convention each
catalog declared, following the same rules as :func:`ocean_skill.catalog._bbox_overlaps`
— including splitting an anti-meridian-straddling extent into two rectangles at the
seam, so one code path serves both renderers.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

from ocean_skill import _stacklevel

__all__ = [
    "FEATURE_TYPE_ORDER",
    "HOVER_FIELDS",
    "TAB10",
    "build_items",
    "map_datasets",
]

#: tab10 in matplotlib's own order. This is the palette both renderers draw the
#: locations map from (and :mod:`ocean_skill.plot.holoviews_renderer` re-exports as
#: ``_TAB10``), pinned here so a featureType keeps its colour across renderers.
TAB10 = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

#: Canonical featureTypes in a fixed order (the set from
#: :data:`ocean_skill.build._FEATURE_TYPES`, plus ``unknown`` for entries that never
#: declared one). The order *is* the style assignment: featureType → colour/marker by
#: index, in both renderers, so a legend entry means the same thing statically and
#: interactively.
FEATURE_TYPE_ORDER = (
    "grid",
    "point",
    "timeSeries",
    "profile",
    "timeSeriesProfile",
    "trajectory",
    "trajectoryProfile",
    "unknown",
)

#: What hovering a marker (or a grid's extent rectangle) shows, in order. Every field
#: is pre-formatted to a string by :func:`build_items`, so the two renderers — and the
#: bokeh tooltip — agree exactly on what a dataset's record reads.
HOVER_FIELDS = (
    "name",
    "catalog",
    "featureType",
    "variables",
    "time_coverage",
    "cadence",
    "resolution",
    "depth",
    "institution",
    "title",
)

#: How many declared variables a tooltip lists before eliding the rest.
_MAX_HOVER_VARIABLES = 6

#: Spans at (or beyond) which the default extent stops padding and snaps to the whole
#: world — a global grid plus a margin is just the globe with dead space around it.
_NEAR_GLOBAL_LON = 300.0
_NEAR_GLOBAL_LAT = 150.0

#: Minimum default-extent span per axis, so one mooring maps to a region rather than
#: a degenerate zero-area frame.
_MIN_SPAN_DEG = 10.0


def _style_index(feature_type: str) -> int:
    """The colour/marker index for a featureType (unknown types style as ``unknown``)."""
    try:
        return FEATURE_TYPE_ORDER.index(feature_type)
    except ValueError:
        return FEATURE_TYPE_ORDER.index("unknown")


def _wrap(lon: float) -> float:
    """One longitude brought into the −180..180 frame."""
    return ((lon + 180.0) % 360.0) - 180.0


def _normalized_geometry(meta: dict[str, Any]) -> dict[str, Any] | None:
    """Midpoint and −180..180 bounding box(es) from an entry's declared extent.

    ``None`` when any of the four ``geospatial_*`` bounds is missing — the caller
    skips (and reports) those. Follows :func:`ocean_skill.catalog._bbox_overlaps`
    rule for rule: a globe-spanning extent is short-circuited *before* wrapping
    (wrapping −180..180 sends both bounds to −180, collapsing the world to a
    point), bounds past 180 mean the entry declared 0–360 and are wrapped, and a
    wrapped extent whose low sits above its high straddles the anti-meridian — here
    that splits into two rectangles at the seam rather than refusing to guess,
    since a map can honestly draw both halves.
    """
    keys = (
        "geospatial_lon_min",
        "geospatial_lat_min",
        "geospatial_lon_max",
        "geospatial_lat_max",
    )
    lo, la, hi, ha = (meta.get(k) for k in keys)
    if None in (lo, la, hi, ha):
        return None
    lo, la, hi, ha = float(lo), float(la), float(hi), float(ha)
    mid_lat = (la + ha) / 2.0
    if hi - lo >= 359.0:
        return {"midpoint": (0.0, mid_lat), "bboxes": [(-180.0, la, 180.0, ha)]}
    # circular midpoint from the *raw* bounds: the naive (lo+hi)/2 of a wrapped
    # Pacific extent lands in the Atlantic
    mid_lon = _wrap(lo + ((hi - lo) % 360.0) / 2.0)
    if hi > 180.0:  # declared 0-360 (ROMS' native convention); bring into +/-180
        lo, hi = _wrap(lo), _wrap(hi)
    if lo > hi:  # straddles the anti-meridian; draw both halves
        bboxes = [(lo, la, 180.0, ha), (-180.0, la, hi, ha)]
    else:
        bboxes = [(lo, la, hi, ha)]
    return {"midpoint": (mid_lon, mid_lat), "bboxes": bboxes}


def _format_variables(variables) -> str:
    """The declared variables as one hover line, elided past a handful."""
    names = [str(v) for v in (variables or [])]
    if not names:
        return ""
    if len(names) > _MAX_HOVER_VARIABLES:
        shown = ", ".join(names[:_MAX_HOVER_VARIABLES])
        return f"{shown}, … (+{len(names) - _MAX_HOVER_VARIABLES})"
    return ", ".join(names)


def _format_time_coverage(meta: dict[str, Any]) -> str:
    """Time coverage as one line; a climatology names its period instead.

    A climatology deliberately declares no ``time_coverage_*`` (it is a calendar
    slot, not a date range — see :func:`ocean_skill.catalog.find`), so it reads
    ``"climatology: month01"`` rather than an empty field.
    """
    if meta.get("climatology"):
        period = meta.get("climatology_period")
        return f"climatology: {period}" if period else "climatology"
    start, end = meta.get("time_coverage_start"), meta.get("time_coverage_end")
    if not start or not end:
        return ""
    return f"{str(start)[:10]} → {str(end)[:10]}"


#: seconds → spoken cadence, matched within the same ±25% ``find(cadence=...)``
#: uses (:data:`ocean_skill.catalog._CADENCE_ALIASES`), because a monthly product
#: steps 28–31 days and no exact number of seconds describes it.
_CADENCE_LABELS = (
    ("hourly", 3600.0),
    ("6-hourly", 21600.0),
    ("daily", 86400.0),
    ("weekly", 604800.0),
    ("monthly", 30.4375 * 86400.0),
    ("annual", 365.25 * 86400.0),
)


def _format_cadence(seconds) -> str:
    """A time step in seconds as the word a person would use for it."""
    if not seconds:
        return ""
    s = float(seconds)
    for label, ref in _CADENCE_LABELS:
        if 0.75 * ref <= s <= 1.25 * ref:
            return label
    if s < 60.0:
        return f"{s:g} s"
    if s < 3600.0:
        return f"{s / 60.0:g} min"
    if s < 86400.0:
        return f"{s / 3600.0:g} h"
    return f"{s / 86400.0:g} d"


def _format_resolution(meta: dict[str, Any]) -> str:
    """Grid spacing as one hover line (gridded entries only declare it)."""
    deg, km = meta.get("grid_resolution_deg"), meta.get("grid_resolution_km")
    if deg is not None and km is not None:
        text = f"{float(deg):g}° (~{float(km):g} km)"
    elif km is not None:
        text = f"~{float(km):g} km"
    elif deg is not None:
        text = f"{float(deg):g}°"
    else:
        return ""
    if meta.get("grid_regular") is False:
        text += " (irregular)"
    return text


def _format_depth(meta: dict[str, Any]) -> str:
    """Vertical extent (and level count) as one hover line."""
    vmin, vmax = meta.get("geospatial_vertical_min"), meta.get(
        "geospatial_vertical_max"
    )
    levels = meta.get("vertical_levels")
    if vmin is not None and vmax is not None:
        text = f"{float(vmin):g}–{float(vmax):g} m"
        if levels:
            text += f" ({int(levels)} levels)"
        return text
    if levels:
        return f"{int(levels)} levels"
    return ""


def _default_extent(
    items: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
    """``(lon_min, lat_min, lon_max, lat_max)`` framing every item, with margin.

    The union of all point positions and extent corners, padded by 5% of the span
    (at least 2°) per side, held to a minimum span so a lone mooring maps to a
    region rather than a point, and snapped to the whole world once the union is
    effectively global anyway.
    """
    lons: list[float] = []
    lats: list[float] = []
    for item in items:
        if item["kind"] == "extent":
            for lo, la, hi, ha in item["bboxes"]:
                lons += [lo, hi]
                lats += [la, ha]
        else:
            lons.append(item["lon"])
            lats.append(item["lat"])
    lon0, lon1 = min(lons), max(lons)
    lat0, lat1 = min(lats), max(lats)
    if lon1 - lon0 >= _NEAR_GLOBAL_LON or lat1 - lat0 >= _NEAR_GLOBAL_LAT:
        return (-180.0, -90.0, 180.0, 90.0)

    pad_lon = max(0.05 * (lon1 - lon0), 2.0)
    pad_lat = max(0.05 * (lat1 - lat0), 2.0)
    lon0, lon1 = lon0 - pad_lon, lon1 + pad_lon
    lat0, lat1 = lat0 - pad_lat, lat1 + pad_lat

    def widened(a: float, b: float) -> tuple[float, float]:
        if b - a < _MIN_SPAN_DEG:
            mid = (a + b) / 2.0
            return mid - _MIN_SPAN_DEG / 2.0, mid + _MIN_SPAN_DEG / 2.0
        return a, b

    lon0, lon1 = widened(lon0, lon1)
    lat0, lat1 = widened(lat0, lat1)
    return (
        max(lon0, -180.0),
        max(lat0, -90.0),
        min(lon1, 180.0),
        min(lat1, 90.0),
    )


def build_items(
    names: str | Iterable[str] | None = None,
    *,
    catalog: str | None = None,
) -> tuple[list[dict[str, Any]], tuple[float, float, float, float]]:
    """Build the ``"locations"`` family's items from catalog metadata.

    ``names=None`` maps everything discoverable; a list of names (what
    :func:`ocean_skill.catalog.find` returns) maps just those, each resolved so
    qualified ``"catalog:source"`` spellings work; ``catalog=`` narrows either to
    matching catalogs (same substring/glob matching ``find(catalog=...)`` uses).

    Returns ``(items, default_extent)``. Entries with no declared geospatial extent
    cannot be placed and are skipped with one warning naming them — probe the
    catalog to fill extents in. Raises :class:`ValueError` when nothing mappable
    remains, rather than drawing an empty map.
    """
    from ocean_skill.build import _GRIDDED_FEATURE_TYPES
    from ocean_skill.catalog import _matches_name, discover, resolve

    if names is None:
        refs = list(discover().values())
    else:
        if isinstance(names, str):
            names = [names]
        refs = [resolve(str(n)) for n in names]
    if catalog:
        refs = [ref for ref in refs if _matches_name(ref.catalog, catalog)]

    items: list[dict[str, Any]] = []
    skipped: list[str] = []
    for ref in refs:
        meta = ref.metadata or {}
        geometry = _normalized_geometry(meta)
        if geometry is None:
            skipped.append(ref.name)
            continue
        feature_type = str(meta.get("featureType") or "unknown")
        kind = "extent" if feature_type in _GRIDDED_FEATURE_TYPES else "point"
        item: dict[str, Any] = {
            "kind": kind,
            "lon": geometry["midpoint"][0],
            "lat": geometry["midpoint"][1],
            "name": ref.name,
            "catalog": ref.catalog,
            "featureType": feature_type,
            "variables": _format_variables(meta.get("variables")),
            "time_coverage": _format_time_coverage(meta),
            "cadence": _format_cadence(meta.get("time_resolution_s")),
            "resolution": _format_resolution(meta),
            "depth": _format_depth(meta),
            "institution": str(meta.get("institution") or ""),
            "title": str(meta.get("title") or ""),
        }
        if kind == "extent":
            item["bboxes"] = geometry["bboxes"]
        items.append(item)

    if skipped:
        warnings.warn(
            f"skipping {len(skipped)} source(s) with no declared geospatial "
            f"extent: {', '.join(sorted(skipped))} — probe the catalog to fill "
            "extents in.",
            stacklevel=_stacklevel.find(),
        )
    if not items:
        raise ValueError("no datasets with a geospatial extent to map")
    return items, _default_extent(items)


def map_datasets(
    names: str | Iterable[str] | None = None,
    *,
    catalog: str | None = None,
    renderer: str = "matplotlib",
    **kwargs: Any,
):
    """Map where catalog datasets are, from metadata alone — nothing is read.

    ::

        osk.map_datasets()                                # everything discoverable
        osk.map_datasets(osk.find(variable="nitrate"))    # a query result
        osk.find(variable="nitrate").map()                # the same, as a method
        osk.map_datasets(catalog="ooi_papa")              # one catalog
        osk.map_datasets(renderer="holoviews")            # interactive, with hover

    Moorings, profiles and tracks draw as markers at their declared position;
    gridded datasets draw as dashed extent rectangles; both are coloured by
    featureType with a legend. The interactive renderer shows each dataset's
    metadata on hover (name, catalog, featureType, variables, time coverage,
    cadence, resolution, depth, institution, title) over a web basemap — pass
    ``tiles=None`` to work offline, or another ``geoviews.tile_sources`` name.

    Everything beyond ``names``/``catalog``/``renderer`` is a plot option
    (``extent=``, ``title=``, ``save=``, ``tiles=``, ``legend=``, ...), passed to
    the renderer like any other family's. The default ``extent`` frames every
    mapped dataset with a margin.
    """
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items, default_extent = build_items(names, catalog=catalog)
    kwargs.setdefault("extent", default_extent)
    spec = PlotSpec(family="locations", items=items, options=kwargs)
    return render(spec, renderer=renderer)
