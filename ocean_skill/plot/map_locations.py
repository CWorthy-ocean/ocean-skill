"""Map where anything else in this package draws from: catalog names, or a request.

:func:`~ocean_skill.plot.locations.build_items` already answers "where are these
catalog datasets" from metadata alone. This module answers the sibling question —
"where does *this* plotted thing sit" — for a
:class:`~ocean_skill.comparison.Comparison`, :class:`~ocean_skill.comparison.
ComparisonSet`, :class:`~ocean_skill.field.Field` or
:class:`~ocean_skill.field.FieldSet`: the selected point, region or slice line the
object's ``select`` asks for -- a point, a region box, a lone-lon/lat slice, or
a transect's waypoint path -- or, when the select pins nothing, each source's
own declared position, drawn over the test source's domain outline for context.

Like the catalog path, this never opens a dataset and never aligns a comparison:
everything here comes from the *request* (``select``) and catalog metadata alone, so
calling it costs the same whether the comparison has already run or not, and a
snapped-vs-requested offset — already a warning where alignment actually happens
(:func:`ocean_skill.align.sample_at`) — stays a warning there, not something this
module tries to also show on the map.

:func:`map_locations` is the one public entry point, unifying both questions: pass
catalog names (or ``None``) for the metadata-only map, an object (or several, mixed
with names in one list) for the request-based one, and both draw through the same
``"locations"`` plot family — see :mod:`ocean_skill.plot.locations` for the item
schema and both renderers.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np

from ocean_skill import _stacklevel

__all__ = ["build_map_items", "map_locations"]

#: Sentinel default for ``domain=``: outline each distinct test source
#: automatically. Distinct from ``None`` (draw no outline at all) — an ``ndarray``
#: default could not tell "not passed" from "passed and falsy" apart, which is
#: exactly the ambiguity :meth:`~ocean_skill.comparison.Comparison.plot` avoids the
#: same way for its own ``domain=``.
_AUTO = object()


def _hover(
    name: str,
    *,
    featureType: str,
    variables: str = "",
    time_coverage: str = "",
    depth: str = "",
    title: str = "",
    catalog: str = "",
) -> dict[str, str]:
    """One selection/domain item's hover record: every ``HOVER_FIELDS`` key, a string.

    Mirrors :func:`~ocean_skill.plot.locations.build_items`'s per-item record
    exactly — the same keys, all strings, an unused one left blank — so a
    selection item slots into both renderers' existing drawing path with no
    special-casing on either side.
    """
    return {
        "name": name,
        "catalog": catalog,
        "featureType": featureType,
        "variables": variables,
        "time_coverage": time_coverage,
        "cadence": "",
        "resolution": "",
        "depth": depth,
        "institution": "",
        "title": title,
    }


def _band(value: Any) -> tuple[float | None, float | None] | None:
    """``(lo, hi)`` from a region-select axis value, either bound optionally open.

    ``{"min": ..., "max": ...}`` (either key optional) or a ``slice`` — the two
    range spellings :func:`ocean_skill.operators.select` accepts for a region
    band. Anything else (a scalar, a string, a list) is not a band, and returns
    ``None``.
    """
    if isinstance(value, dict) and ({"min", "max"} & set(value)):
        lo, hi = value.get("min"), value.get("max")
        return (
            float(lo) if lo is not None else None,
            float(hi) if hi is not None else None,
        )
    if isinstance(value, slice):
        lo, hi = value.start, value.stop
        return (
            float(lo) if lo is not None else None,
            float(hi) if hi is not None else None,
        )
    return None


def _axis_bounds(value: Any) -> tuple[str, float, float] | None:
    """Classify one axis's select value: an exact point or a ranged band.

    ``("point", v, v)`` for a scalar; ``("band", lo, hi)`` for a ``{"min",
    "max"}``/``slice`` range, either bound ``None`` if left open (the caller
    clamps it). ``None`` for anything this module has no geometric reading for
    — a list of several positions, a string — treated exactly like the axis
    naming nothing at all: a wider box than a discrete selection is a safe
    default, not a misleading one.
    """
    band = _band(value)
    if band is not None:
        return ("band", band[0], band[1])
    from ocean_skill.operators import _point_scalar

    scalar = _point_scalar(value)
    if scalar is not None:
        return ("point", scalar, scalar)
    return None


def _clamp_box(
    test: str | None, reference: str | None
) -> tuple[float, float, float, float]:
    """The domain to clamp an open-ended region/slice select to.

    The test source's own declared bbox, else the reference's, else the whole
    world with a warning naming whichever source(s) declared none — matching
    :meth:`~ocean_skill.comparison.Comparison.plot`'s own ``domain=`` fallback
    (:func:`~ocean_skill.comparison._domain_of`), so a select clamped here and a
    domain ring drawn beside it agree on what "the model's extent" means.
    """
    from ocean_skill.comparison import _domain_of

    tried = [s for s in (test, reference) if s is not None]
    for source in tried:
        bbox = _domain_of(source)
        if bbox is not None:
            return bbox
    if tried:
        warnings.warn(
            f"{tried} declare no geospatial extent to clamp an open-ended "
            "region/slice select to — using the whole world instead. Probe "
            "the catalog to fill an extent in.",
            stacklevel=_stacklevel.find(),
        )
    return (-180.0, -90.0, 180.0, 90.0)


def _transect_geometry(
    spec: Any, clamp: tuple[float, float, float, float]
) -> dict[str, Any] | None:
    """Return the path a ``select={"transect": ...}`` request draws, or ``None``.

    ``waypoints``/``points`` draw as the *requested* path, unchanged -- the same
    read-free contract every other shape here keeps (the snapped cells
    :func:`ocean_skill.transect.sample_along` actually samples are never drawn,
    only warned about, exactly like a point's own snapped-vs-requested offset).
    A fixed ``lon``/``lat`` line clamps its open bound to ``clamp``, the same as
    a top-level lone-lon/lat select. A grid-aligned, ``cross``, or
    reference-derived transect names no lon/lat without opening a dataset --
    ``None`` here falls back to each source's catalog footprint, which for
    ``from_reference`` is exactly where the section runs.
    """
    from ocean_skill.transect import as_transect

    parsed = as_transect(spec)
    kind = parsed["kind"]
    if kind in ("waypoints", "points"):
        pairs = parsed[kind]
        return {
            "shape": "path",
            "lons": tuple(p[0] for p in pairs),
            "lats": tuple(p[1] for p in pairs),
        }

    clamp_lo, clamp_la, clamp_hi, clamp_ha = clamp
    if kind == "lon_line":
        lat_lo, lat_hi = parsed["lat_bounds"]
        lat_lo = clamp_la if lat_lo is None else lat_lo
        lat_hi = clamp_ha if lat_hi is None else lat_hi
        lon = parsed["lon"]
        return {
            "shape": "line",
            "lon0": lon,
            "lat0": lat_lo,
            "lon1": lon,
            "lat1": lat_hi,
        }
    if kind == "lat_line":
        lon_lo, lon_hi = parsed["lon_bounds"]
        lon_lo = clamp_lo if lon_lo is None else lon_lo
        lon_hi = clamp_hi if lon_hi is None else lon_hi
        lat = parsed["lat"]
        return {
            "shape": "line",
            "lon0": lon_lo,
            "lat0": lat,
            "lon1": lon_hi,
            "lat1": lat,
        }
    # "grid", "cross", "from_reference": no lon/lat knowable without opening a
    # dataset (a grid index, a cross-section pivot, or the reference's own
    # casts) -- the caller falls back to the source's catalog footprint.
    return None


def _fmt_pos(lon: float, lat: float) -> str:
    """Format one lon/lat pair as ``"55.0°N, 150.0°W"``.

    Matches :func:`ocean_skill.plot.section._path_note`'s own endpoint formatting.
    """
    lat_hemi = "N" if lat >= 0 else "S"
    lon_hemi = "E" if lon >= 0 else "W"
    return f"{abs(lat):.1f}°{lat_hemi}, {abs(lon):.1f}°{lon_hemi}"


def _selection_geometry(
    select: dict[str, Any], clamp: tuple[float, float, float, float]
) -> dict[str, Any] | None:
    """The drawable shape one lane's ``select`` asks for, or ``None`` for none.

    A scalar on both horizontal axes is a point
    (:func:`~ocean_skill.operators.point_in_spec`). A ``"transect"`` key draws
    the requested path (see :func:`_transect_geometry`). A ranged value on
    either axis makes a region box, its open bound(s) — and its other axis,
    when that one names nothing at all — clamped to ``clamp``. A scalar on
    exactly one axis with the other unselected or itself ranged is a slice
    line, fixed on the scalar axis and spanning the other axis's own range
    (the whole ``clamp`` box when that axis names nothing — a lone-lon/lat
    select). ``None`` when neither axis names anything this reads
    geometrically — the caller falls back to the source's catalog footprint
    then.
    """
    from ocean_skill.operators import _POINT_LAT_KEYS, _POINT_LON_KEYS, point_in_spec
    from ocean_skill.plot.locations import _wrap

    if "transect" in select:
        return _transect_geometry(select["transect"], clamp)

    hit = point_in_spec(select)
    if hit is not None:
        _, _, lon, lat = hit
        return {"shape": "point", "lon": _wrap(lon), "lat": lat}

    clamp_lo, clamp_la, clamp_hi, clamp_ha = clamp
    lon_key = next((k for k in _POINT_LON_KEYS if k in select), None)
    lat_key = next((k for k in _POINT_LAT_KEYS if k in select), None)
    lon_axis = _axis_bounds(select[lon_key]) if lon_key else None
    lat_axis = _axis_bounds(select[lat_key]) if lat_key else None
    if lon_axis is None and lat_axis is None:
        return None

    lon_kind, lo, hi = lon_axis or ("band", clamp_lo, clamp_hi)
    lat_kind, la, ha = lat_axis or ("band", clamp_la, clamp_ha)
    lo = clamp_lo if lo is None else lo
    hi = clamp_hi if hi is None else hi
    la = clamp_la if la is None else la
    ha = clamp_ha if ha is None else ha

    # exactly one of the two can be "point" here: both-scalar is caught by
    # point_in_spec above, so a scalar axis paired with a band/absent axis is a
    # slice line fixed on the scalar, spanning the other axis's own range.
    if lon_kind == "point":
        return {"shape": "line", "lon0": lo, "lat0": la, "lon1": lo, "lat1": ha}
    if lat_kind == "point":
        return {"shape": "line", "lon0": lo, "lat0": la, "lon1": hi, "lat1": la}
    return {"shape": "box", "lon_min": lo, "lat_min": la, "lon_max": hi, "lat_max": ha}


def _selection_title(geo: dict[str, Any]) -> str:
    """A short human description of a selection's geometry, for its hover title."""
    shape = geo["shape"]
    if shape == "point":
        return f"point at ({geo['lon']:.2f}, {geo['lat']:.2f})"
    if shape == "box":
        return (
            f"lat {geo['lat_min']:.2f}–{geo['lat_max']:.2f}, "
            f"lon {geo['lon_min']:.2f}–{geo['lon_max']:.2f}"
        )
    if shape == "path":
        lons, lats = geo["lons"], geo["lats"]
        start, end = _fmt_pos(lons[0], lats[0]), _fmt_pos(lons[-1], lats[-1])
        return f"transect {start} → {end} ({len(lons)} waypoints)"
    if geo["lon0"] == geo["lon1"]:
        return f"meridional slice at {geo['lon0']:.2f}°"
    return f"zonal slice at {geo['lat0']:.2f}°"


def _warn_if_outside_domain(geo: dict[str, Any], source: str) -> None:
    """Warn once when a requested point falls outside its source's own extent.

    A cheap sanity check, not a data caveat annotated on the figure — the
    package's own rule for this (a snapped-vs-requested offset is a warning,
    never drawn) applies just as well to "the model may not even have data
    here". Only a point is checked: a region/slice already draws its own
    extent, so there is nothing to compare it against. Silent when ``source``
    declares no extent to check against, or the requested point lies inside
    any piece of a seam-split one.
    """
    if geo["shape"] != "point":
        return
    from ocean_skill.comparison import _domain_of
    from ocean_skill.plot.locations import _split_bbox

    bbox = _domain_of(source)
    if bbox is None:
        return
    lon, lat = geo["lon"], geo["lat"]
    pieces = _split_bbox(*bbox)
    if any(lo <= lon <= hi and la <= lat <= ha for lo, la, hi, ha in pieces):
        return
    warnings.warn(
        f"the selected position ({lon:.2f}, {lat:.2f}) is outside {source!r}'s "
        "declared extent — the map still draws it, but it may not reflect "
        "where the model actually has data.",
        stacklevel=_stacklevel.find(),
    )


def _round_geo_value(v: Any) -> Any:
    """Round a scalar, or every element of a tuple/list, to 6 decimal places."""
    if isinstance(v, int | float):
        return round(v, 6)
    if isinstance(v, tuple | list):
        return tuple(_round_geo_value(x) for x in v)
    return v


def _geo_key(geo: dict[str, Any]) -> tuple:
    """A hashable, rounded key for deduping identical selection geometry.

    Rounded so two lanes computing the same requested position through
    different floating-point paths still collapse to one marker — a
    ``compare()`` fan of ten variables at one mooring draws one point (or, for
    a shared transect, one path), not ten.
    """
    return tuple((k, _round_geo_value(v)) for k, v in sorted(geo.items()))


def _selection_item(geo: dict[str, Any], hover: dict[str, str]) -> dict[str, Any]:
    """One ``locations``-family item for a lane's selection geometry."""
    from ocean_skill.plot.locations import _seam_split, _split_bbox

    shape = geo["shape"]
    if shape == "point":
        return {
            "kind": "point",
            "lon": geo["lon"],
            "lat": geo["lat"],
            "featureType": "selection",
            **hover,
        }
    if shape == "box":
        bboxes = _split_bbox(
            geo["lon_min"], geo["lat_min"], geo["lon_max"], geo["lat_max"]
        )
        return {"kind": "extent", "featureType": "selection", "bboxes": bboxes, **hover}
    if shape == "path":
        # A transect's requested waypoints: walked vertex to vertex, exactly
        # like a real domain perimeter, since the path -- unlike a box's
        # declared bounds -- isn't straight, so only walking it edge by edge
        # can tell where it actually crosses the seam.
        paths = _seam_split(geo["lons"], geo["lats"])
        return {"kind": "line", "featureType": "selection", "paths": paths, **hover}
    # "line": the same declared-bounds splitting a box uses, since a lone-lon/lat
    # span is a degenerate box (zero width or height) with exactly the same
    # antimeridian ambiguity — only the bounds themselves say whether it goes the
    # long way through the seam.
    pieces = _split_bbox(geo["lon0"], geo["lat0"], geo["lon1"], geo["lat1"])
    paths = [np.array([[lo, la], [hi, ha]]) for lo, la, hi, ha in pieces]
    return {"kind": "line", "featureType": "selection", "paths": paths, **hover}


def footprint_item(source: str) -> dict[str, Any] | None:
    """A source's catalog-declared position/extent, as one ``locations`` item.

    The fallback for a lane whose select pins nothing on either horizontal
    axis: reuses :func:`~ocean_skill.plot.locations.build_items`'s per-source
    logic (a mooring becomes a point, a grid an extent box), so a footprint
    item is identical to what a catalog-only :func:`map_locations` call would
    draw for the same source. ``None`` when the source is unresolvable or
    declares no geospatial extent, with one warning naming it either way.
    """
    from ocean_skill.plot.locations import (
        _NEAR_GLOBAL_LAT,
        _NEAR_GLOBAL_LON,
        build_items,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            items, _ = build_items([source])
        item = items[0]
        if item["kind"] == "extent":
            lons = [v for lo, _la, hi, _ha in item["bboxes"] for v in (lo, hi)]
            lats = [v for _lo, la, _hi, ha in item["bboxes"] for v in (la, ha)]
            if (
                max(lons) - min(lons) >= _NEAR_GLOBAL_LON
                or max(lats) - min(lats) >= _NEAR_GLOBAL_LAT
            ):
                # A footprint this wide is only ever a "nothing selected"
                # fallback's second, un-plotted-on source (see
                # _comparison_items/_field_items) -- true for it, but letting
                # it drive _default_extent would blow a regional selection's
                # own frame out to the whole world. It still draws; it just
                # doesn't get a vote on the frame.
                item["frame"] = False
        return item
    except ValueError:
        warnings.warn(
            f"{source!r} has no declared geospatial extent, so its position "
            "cannot be placed on the map — probe the catalog to fill it in.",
            stacklevel=_stacklevel.find(),
        )
        return None
    except KeyError:
        warnings.warn(
            f"{source!r} is not a known catalog source, so it cannot be "
            "placed on the map.",
            stacklevel=_stacklevel.find(),
        )
        return None


def domain_item(source: str) -> dict[str, Any] | None:
    """The source's model-domain outline, as one dashed ``kind="ring"`` item.

    The true grid-edge perimeter when the catalog declares one
    (:func:`~ocean_skill.comparison._outline_of`), else its bbox
    (:func:`~ocean_skill.comparison._domain_of`) drawn as a rectangle. Both are
    seam-split for the ``"locations"`` family's fixed centre-0 frame — the true
    perimeter edge by edge
    (:func:`~ocean_skill.plot.locations._seam_split`, correct for a real,
    possibly rotated, grid shape) and a bbox at its declared bounds
    (:func:`~ocean_skill.plot.locations._split_bbox`, correct for a box whose
    long way through the antimeridian only the bounds themselves say).
    ``None`` when the source declares neither.
    """
    from ocean_skill.comparison import _domain_of, _outline_of
    from ocean_skill.plot.locations import _seam_split, _split_bbox

    ring = _outline_of(source)
    if ring is not None:
        paths = _seam_split(ring[:, 0], ring[:, 1])
    else:
        bbox = _domain_of(source)
        if bbox is None:
            return None
        lo, la, hi, ha = bbox
        paths = [
            np.array(
                [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]
            )
            for b in _split_bbox(lo, la, hi, ha)
        ]
    return {
        "kind": "ring",
        "paths": paths,
        "featureType": "domain",
        **_hover(
            f"{source} domain", featureType="domain", title=f"model domain of {source}"
        ),
    }


def _explicit_domain_item(domain: Any) -> dict[str, Any]:
    """A ``kind="ring"`` item from a user-supplied ``domain=`` bbox/ring override."""
    from ocean_skill.plot.locations import _seam_split
    from ocean_skill.plot.matplotlib_renderer import domain_ring

    ring = domain_ring(domain)
    paths = _seam_split(ring[:, 0], ring[:, 1])
    return {
        "kind": "ring",
        "paths": paths,
        "featureType": "domain",
        **_hover("model domain", featureType="domain", title="model domain"),
    }


def _comparison_items(
    c: Any, *, auto_domain: bool, seen_geo: set, seen_domain: set
) -> list[dict[str, Any]]:
    """One :class:`~ocean_skill.comparison.Comparison`'s items.

    Its selection geometry (once, shared, unless a pair-spec select names two
    different lanes) or — when nothing is selected — each source's own
    footprint, plus the test source's domain ring.
    """
    from ocean_skill.comparison import (
        _depth_label,
        _display_depth,
        _display_time,
        _short_variable_label,
        _time_label,
        _variable_label,
        is_pair_spec,
        select_for,
    )

    items: list[dict[str, Any]] = []
    clamp = _clamp_box(c.test_name, c.reference_name)
    label = c.label or _short_variable_label(c.variable)
    variable_label = _variable_label(c.variable)
    # The test lane names the figure's depth/time when the two could disagree —
    # the same convention Comparison.as_item() and .metrics() already keep.
    depth_label = _depth_label(_display_depth(c.variable, c.select))
    time_label = _time_label(_display_time(c.select))
    paired = is_pair_spec(c.select)
    roles = ("test", "reference") if paired else ("test",)

    for role in roles:
        select = select_for(c.select, role)
        source = c.test_name if role == "test" else c.reference_name
        geo = _selection_geometry(select, clamp)
        if geo is not None:
            if role == "test":
                _warn_if_outside_domain(geo, c.test_name)
            row_label = f"{label} ({role})" if paired else label
            hover = _hover(
                row_label,
                featureType="selection",
                variables=variable_label,
                depth=depth_label,
                time_coverage=time_label,
                title=_selection_title(geo),
            )
            key = _geo_key(geo)
            if key not in seen_geo:
                seen_geo.add(key)
                items.append(_selection_item(geo, hover))
        else:
            item = footprint_item(source)
            if item is not None:
                items.append(item)
            if not paired:
                # the select is shared, so both lanes sample the same place --
                # but they are still different *sources*, each with its own
                # catalog footprint, so the reference gets its own item too.
                other = footprint_item(c.reference_name)
                if other is not None:
                    items.append(other)

    if auto_domain and c.test_name not in seen_domain:
        seen_domain.add(c.test_name)
        ditem = domain_item(c.test_name)
        if ditem is not None:
            items.append(ditem)
    return items


def _field_items(
    f: Any, *, auto_domain: bool, seen_geo: set, seen_domain: set
) -> list[dict[str, Any]]:
    """One :class:`~ocean_skill.field.Field`'s items: its selection, plus its
    source's domain ring.

    A ``Field`` has no reference, so there is no second lane and no pair-spec
    to consider.
    """
    from ocean_skill.comparison import (
        _depth_label,
        _display_depth,
        _display_time,
        _short_variable_label,
        _time_label,
        _variable_label,
    )

    items: list[dict[str, Any]] = []
    clamp = _clamp_box(f.source, None)
    geo = _selection_geometry(f.select, clamp)
    if geo is not None:
        _warn_if_outside_domain(geo, f.source)
        label = f.label or _short_variable_label(f.variable)
        # A Field does nothing vertically unless asked (unlike a Comparison, whose
        # own default is the surface) -- default=None so an unselected field's
        # hover says nothing about depth rather than claiming "surface".
        requested_depth = _display_depth(f.variable, f.select, default=None)
        hover = _hover(
            label,
            featureType="selection",
            variables=_variable_label(f.variable),
            depth=_depth_label(requested_depth) if requested_depth is not None else "",
            time_coverage=_time_label(_display_time(f.select)),
            title=_selection_title(geo),
        )
        key = _geo_key(geo)
        if key not in seen_geo:
            seen_geo.add(key)
            items.append(_selection_item(geo, hover))
    else:
        item = footprint_item(f.source)
        if item is not None:
            items.append(item)

    if auto_domain and f.source not in seen_domain:
        seen_domain.add(f.source)
        ditem = domain_item(f.source)
        if ditem is not None:
            items.append(ditem)
    return items


def build_map_items(obj: Any, *, domain: Any = _AUTO) -> list[dict[str, Any]]:
    """Build the ``"locations"`` family's items for a plotted selection.

    ``obj`` is a :class:`~ocean_skill.comparison.Comparison`,
    :class:`~ocean_skill.comparison.ComparisonSet`,
    :class:`~ocean_skill.field.Field` or :class:`~ocean_skill.field.FieldSet`.
    Every lane appears once — its requested selection geometry (a point, a
    region box, a lone-lon/lat slice line, or a transect's waypoint path) when
    its ``select`` pins one, else its declared catalog footprint — over the
    test source's domain outline. A grid-aligned, ``cross``, or
    reference-derived transect names no lon/lat without opening a dataset, so
    it falls back to its source's footprint like an unselected lane does.
    Built from the *request* and catalog metadata alone: nothing is opened, and
    no comparison is aligned, so calling this costs the same whether the
    comparison has already run or not.

    ``domain`` mirrors :meth:`~ocean_skill.comparison.Comparison.plot`'s own
    option: the default outlines each distinct test source once; ``None``
    suppresses it; a bbox or ``(N, 2)`` ring overrides it — drawn once, for
    every lane alike. Raises :class:`ValueError` when nothing at all can be
    placed.
    """
    from ocean_skill.comparison import Comparison, ComparisonSet
    from ocean_skill.field import Field, FieldSet

    if isinstance(obj, Comparison):
        comparisons, fields = [obj], []
    elif isinstance(obj, ComparisonSet):
        comparisons, fields = list(obj.comparisons), []
    elif isinstance(obj, Field):
        comparisons, fields = [], [obj]
    elif isinstance(obj, FieldSet):
        comparisons, fields = [], list(obj.fields)
    else:
        raise TypeError(
            f"build_map_items() draws a Comparison, ComparisonSet, Field or "
            f"FieldSet's selection, not {obj!r}."
        )

    items: list[dict[str, Any]] = []
    seen_geo: set[tuple] = set()
    seen_domain: set[str] = set()
    auto_domain = domain is _AUTO

    for c in comparisons:
        items += _comparison_items(
            c, auto_domain=auto_domain, seen_geo=seen_geo, seen_domain=seen_domain
        )
    for f in fields:
        items += _field_items(
            f, auto_domain=auto_domain, seen_geo=seen_geo, seen_domain=seen_domain
        )

    if domain is not _AUTO and domain is not None:
        items.append(_explicit_domain_item(domain))

    if not items:
        raise ValueError(
            "nothing to map: every lane's selection/catalog position was "
            "unplaceable — probe the catalog to fill in a missing extent."
        )
    return items


def map_locations(
    what: Any = None,
    *,
    catalog: str | None = None,
    renderer: str = "matplotlib",
    domain: Any = _AUTO,
    **kwargs: Any,
):
    """Map where something sits: catalog datasets, or a plotted selection.

    Parameters
    ----------
    what
        ``None`` (default, everything discoverable), a catalog source name
        (``str``), a :func:`~ocean_skill.catalog.find` result, a
        :class:`~ocean_skill.comparison.Comparison`,
        :class:`~ocean_skill.comparison.ComparisonSet`,
        :class:`~ocean_skill.field.Field`, :class:`~ocean_skill.field.FieldSet`,
        or a list mixing names and such objects.
    catalog
        Restrict a name-based (metadata-only) lookup to one catalog (``str``), or
        ``None`` (default) to search all discovered catalogs. Ignored for object
        input.
    renderer
        One of ``"matplotlib"``, ``"holoviews"`` (default ``"matplotlib"``).
    domain
        Object input only: the default outlines each distinct test source once;
        ``None`` suppresses the outline; a bbox or ``(N, 2)`` ring overrides it
        for every lane, the same spelling
        :meth:`~ocean_skill.comparison.Comparison.plot`'s own ``domain=`` takes.
    **kwargs
        Other plot options forwarded to the renderer -- ``extent`` (default:
        frames everything mapped, with a margin), ``title``, ``save``, ``tiles``,
        ``legend``, and the rest of the ``"locations"`` family's styling options.
        See ``docs/plot_styling_reference.md`` for the full list.

    ::

        osk.map_locations()                       # everything discoverable
        osk.map_locations(osk.find(variable="nitrate"))   # a catalog query result
        osk.find(variable="nitrate").map()         # the same, as a method
        osk.map_locations(comparison)              # where a Comparison's data sits
        osk.map_locations(comparison_set)          # a whole ComparisonSet
        osk.map_locations([comparison, "papa"])    # a mix of both

    A catalog name (or ``None``, or a whole :func:`~ocean_skill.catalog.find`
    result) draws from metadata alone, exactly as
    :func:`~ocean_skill.plot.locations.build_items` always has — nothing is
    read. A :class:`~ocean_skill.comparison.Comparison`,
    :class:`~ocean_skill.comparison.ComparisonSet`,
    :class:`~ocean_skill.field.Field` or :class:`~ocean_skill.field.FieldSet`
    (or any mix of those with names, in one list) draws its *request* instead
    — the selected point/region/slice, or, when the select pins nothing, each
    source's own declared position — over the test source's domain outline
    (see :func:`build_map_items`). Neither path opens a dataset or aligns a
    comparison.

    ``domain`` only affects object input: the default outlines each distinct
    test source once; ``None`` suppresses it; a bbox or ``(N, 2)`` ring
    overrides it for every lane, the same spelling
    :meth:`~ocean_skill.comparison.Comparison.plot`'s own ``domain=`` takes.

    Everything else beyond ``what``/``catalog``/``renderer``/``domain`` is a
    plot option (``extent=``, ``title=``, ``save=``, ``tiles=``, ``legend=``,
    ...), passed to the renderer like any other family's. The default
    ``extent`` frames everything mapped, with a margin.
    """
    from ocean_skill.comparison import Comparison, ComparisonSet
    from ocean_skill.field import Field, FieldSet
    from ocean_skill.plot.locations import _default_extent
    from ocean_skill.plot.locations import build_items as _catalog_build_items
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    object_types = (Comparison, ComparisonSet, Field, FieldSet)

    if what is None or isinstance(what, str):
        candidates: Iterable[Any] = [what]
    elif isinstance(what, object_types):
        candidates = [what]
    else:
        candidates = list(what)

    names: list[str] = []
    objects: list[Any] = []
    for item in candidates:
        if item is None:
            continue
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, object_types):
            objects.append(item)
        else:
            raise TypeError(
                f"map_locations() cannot place {item!r}: expected a catalog "
                "source name, or a Comparison/ComparisonSet/Field/FieldSet."
            )

    items: list[dict[str, Any]] = []
    if what is None or names:
        catalog_items, _ = _catalog_build_items(names or None, catalog=catalog)
        items += catalog_items
    for obj in objects:
        items += build_map_items(obj, domain=domain)

    if not items:
        raise ValueError("no datasets with a geospatial extent to map")
    kwargs.setdefault("extent", _default_extent(items))
    spec = PlotSpec(family="locations", items=items, options=kwargs)
    return render(spec, renderer=renderer)
