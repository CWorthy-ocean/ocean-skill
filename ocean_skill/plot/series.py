"""Composition for the ``series`` family: items in, a fully resolved layout out.

The two renderers must not disagree about anything except the drawing call itself, so
everything decidable is decided here: which comparisons share a panel, which line gets
which colour and dash (via :mod:`ocean_skill.plot.style`), what each panel's title and
axis labels say, what the statistics box reads and which corner it goes in. A renderer
then walks :class:`Layout` and draws.

Composition follows the bounded rule the plot taxonomy fixed: an *intrinsic* overlay
(the lanes of a comparison share one panel — that is what a line comparison is), plus at
most one user facet (``rows=`` or ``cols=``), plus at most one ``secondary_y``. The
defaults resolve the common cases without being asked:

===================  ==================================================================
one variable         one panel, every source overlaid
two variables        one panel, the second on a right-hand y axis (``secondary_y``)
three or more        one row per variable, sources overlaid within each
===================  ==================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ocean_skill import _stacklevel
from ocean_skill.plot import style as _style

__all__ = ["Layout", "Panel", "compose", "line_specs", "panel_title", "time_values"]

#: Beyond this many panels a series figure is unreadable at page width, and beyond this
#: many lines a panel is a thicket. Both warn and draw anyway — the caller may be
#: exporting at ``size="free"`` and know exactly what they are asking for.
PANEL_CAP = 6
LINE_CAP = 8

#: How many comparisons a statistics box can carry before it is a table sitting on a
#: figure. Past this the box is dropped, with a warning pointing at the metrics CSV.
METRICS_BOX_MAX_ROWS = 3

#: Corners a statistics box may occupy, in the order ties are broken.
CORNERS = ("upper left", "upper right", "lower left", "lower right")

#: Fraction of the axes a corner box occupies, for deciding which corner is emptiest.
_CORNER_W, _CORNER_H = 0.28, 0.34


@dataclass(frozen=True)
class Panel:
    """One drawn panel: its lines, its labelling, and its statistics box."""

    title: str
    ylabel: str
    lines: tuple[_style.StyledLine, ...]
    secondary: tuple[_style.StyledLine, ...] = ()
    secondary_ylabel: str | None = None
    residual: tuple[_style.StyledLine, ...] = ()
    metrics_text: str = ""
    metrics_corner: str = "upper left"
    legend_corner: str = "upper right"


@dataclass(frozen=True)
class Layout:
    """Everything a renderer needs to draw a ``series`` figure."""

    panels: tuple[Panel, ...] = ()
    nrows: int = 1
    ncols: int = 1
    legend_labels: tuple[str, ...] = ()
    shared_legend: bool = True
    residual: bool = False
    xlabel: str = "time"
    options: dict[str, Any] = field(default_factory=dict)


def time_values(da):
    """Return ``da``'s time coordinate as something both renderers can plot.

    A ROMS run on a 360-day calendar carries ``cftime`` objects, which matplotlib can
    only draw with ``nc-time-axis`` installed and bokeh cannot draw at all. Converting
    where the calendar allows keeps those runs plottable; where it does not, saying so
    beats matplotlib's own message, which names neither the calendar nor the fix.
    """
    index = da.indexes.get(str(da.dims[0])) if da.dims else None
    if index is None:
        return np.asarray(da.coords[str(da.dims[0])].values)
    if type(index).__name__ == "CFTimeIndex":
        try:
            return np.asarray(index.to_datetimeindex())
        except Exception as exc:  # a 360-day or all-leap calendar cannot be converted
            raise ValueError(
                "this time axis uses a "
                f"{getattr(index[0], 'calendar', 'non-standard')} calendar, which "
                "cannot be drawn on a real-date axis. Resample or "
                "reindex it onto real dates first, or install nc-time-axis for the "
                "static renderer (bokeh cannot draw cftime at all)."
            ) from exc
    return np.asarray(index)


def _depth_of(aligned) -> float | None:
    """Return the depth a comparison was made at, if it is a single one."""
    value = aligned.attrs.get("actual_depth")
    if value is None:
        depth = aligned.coords.get("depth")
        if depth is None or depth.dims:
            return None
        value = float(depth)
    return None if value is None else float(value)


def line_specs(item: dict[str, Any], index: int = 0) -> list[_style.LineSpec]:
    """Return the two lines one comparison item draws: its reference and its test."""
    aligned = item["aligned"]
    test_source, reference_source = item.get("labels") or ("test", "reference")
    variable = item.get("standard_name") or item.get("label")
    units = item.get("units") or aligned["reference"].attrs.get("units")
    depth = _depth_of(aligned)
    common = {
        "variable": variable,
        "depth": depth,
        "units": units,
        "item": index,
    }
    return [
        _style.LineSpec(
            role="reference",
            source=str(reference_source),
            values=aligned["reference"],
            **common,
        ),
        _style.LineSpec(
            role="test", source=str(test_source), values=aligned["test"], **common
        ),
    ]


def _group_key(item: dict[str, Any], by: str | None, index: int):
    """Return the value ``item`` is grouped by, for ``rows=``/``cols=``."""
    if by is None:
        return None
    if by in ("variable", "standard_name"):
        return item.get("standard_name") or item.get("label")
    if by in ("source", "test"):
        return (item.get("labels") or ("test", "reference"))[0]
    if by == "reference":
        return (item.get("labels") or ("test", "reference"))[1]
    if by == "depth":
        return _depth_of(item["aligned"])
    if by == "comparison":
        return index
    raise ValueError(
        f"cannot facet a series by {by!r}; expected one of variable, source, "
        "reference, depth, comparison."
    )


def _ylabel(specs) -> str:
    """``"temperature [degC]"`` — the axis label, spelled as the legend spells names."""
    from ocean_skill.plot.summary import pretty_level

    variables = {s.variable for s in specs if s.variable}
    units = {s.units for s in specs if s.units}
    name = (
        pretty_level("variable", next(iter(variables)))
        if len(variables) == 1
        else "value"
    )
    return f"{name} [{next(iter(units))}]" if len(units) == 1 else name


def panel_title(specs, *, varying) -> str:
    """Return a panel title of identity only: what, where, when.

    The statistics live in their own box, per the standing rule that a title says what
    the panel *is*. Anything the legend already distinguishes is left out, so a title
    and a legend never say the same thing twice.
    """
    from ocean_skill.plot.summary import pretty_level

    parts = []
    variables = {s.variable for s in specs if s.variable}
    if len(variables) == 1:
        parts.append(pretty_level("variable", next(iter(variables))))
    reference = next((s for s in specs if s.role == "reference"), specs[0])
    place = _place_of(reference.values)
    if place:
        parts.append(place)
    period = _period_of(reference.values)
    if period:
        parts.append(period)
    return " · ".join(parts)


def _place_of(da) -> str:
    """``"50.0°N 144.2°W"`` for a station, or ``""`` when there is no position."""
    lon, lat = da.coords.get("lon"), da.coords.get("lat")
    if lon is None or lat is None or lon.dims or lat.dims:
        return ""
    lon_v, lat_v = float(lon), float(lat)
    return (
        f"{abs(lat_v):.1f}°{'N' if lat_v >= 0 else 'S'} "
        f"{abs(lon_v):.1f}°{'E' if lon_v >= 0 else 'W'}"
    )


def _period_of(da) -> str:
    """``"2015-01 to 2015-07"`` — the span the panel covers."""
    try:
        values = time_values(da)
    except ValueError:
        return ""
    if values.size == 0:
        return ""
    return f"{str(values.min())[:7]} to {str(values.max())[:7]}"


def free_corners(lines) -> list[str]:
    """Return the panel's corners, emptiest first — measured rather than assumed.

    A map's data fills its axes, so ``field_row`` can put its box in a fixed corner. A
    line panel's data can be anywhere, so corners are ranked by counting the samples
    inside each one's box in axes coordinates. No drawing required, and deterministic
    (ties break in :data:`CORNERS` order), so it is testable without a figure.

    A list rather than one answer because a panel has *two* things to place — the
    statistics box and the legend — and they must not be given the same corner. Placing
    the legend by matplotlib's own ``loc="best"`` is what put it under the stats box.
    """
    xs, ys = [], []
    for line in lines:
        values = np.asarray(line.spec.values.values, dtype="float64")
        finite = np.isfinite(values)
        if not finite.any():
            continue
        position = np.linspace(0.0, 1.0, values.size)
        low, high = np.nanmin(values), np.nanmax(values)
        span = high - low
        scaled = (values - low) / span if span else np.full(values.size, 0.5)
        xs.append(position[finite])
        ys.append(scaled[finite])
    if not xs:
        return list(CORNERS)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    counts = {}
    for corner in CORNERS:
        vertical, horizontal = corner.split()
        in_x = x <= _CORNER_W if horizontal == "left" else x >= 1 - _CORNER_W
        in_y = y >= 1 - _CORNER_H if vertical == "upper" else y <= _CORNER_H
        counts[corner] = int(np.count_nonzero(in_x & in_y))
    return sorted(CORNERS, key=lambda c: (counts[c], CORNERS.index(c)))


def _metrics_text(items, metric_keys, *, prefix: bool) -> str:
    """Return the box's text: a line per comparison, prefixed when there are several."""
    from ocean_skill.plot.matplotlib_renderer import _metrics_text as one_row
    from ocean_skill.plot.summary import pretty_level

    lines = []
    for item in items:
        text = one_row(item.get("metrics"), metric_keys).replace("\n", " ")
        if not text:
            continue
        if prefix:
            name = item.get("standard_name") or item.get("label") or ""
            text = f"{pretty_level('variable', name)}: {text}" if name else text
        lines.append(text)
    return "\n".join(lines)


def compose(
    items,
    *,
    rows: str | None = None,
    cols: str | None = None,
    secondary_y: bool = True,
    encode: dict[str, str | None] | None = None,
    residual: bool = False,
    metric_keys=(),
    metrics_loc: str = "auto",
) -> Layout:
    """Group ``items`` into panels and resolve every line's style and labelling."""
    items = list(items)
    if not items:
        raise ValueError("a series needs at least one comparison to draw")
    if rows is not None and cols is not None:
        raise ValueError(
            f"a series takes one facet, not two: rows={rows!r} and cols={cols!r} were "
            "both given. Overlaying the lanes of each comparison is already one axis; "
            "pick rows= or cols= for the other."
        )

    all_specs = [s for i, item in enumerate(items) for s in line_specs(item, i)]
    styled = {
        (line.spec.item, line.spec.role): line
        for line in _style.resolve(all_specs, encode=encode)
    }
    varying = _style.varying_fields(all_specs)

    # Items are carried with their index throughout: the styled lines are keyed on it,
    # and two items can be equal dicts (the same comparison drawn twice), so identity
    # has to be positional rather than by value.
    indexed = list(enumerate(items))
    facet = rows or cols
    variables = []
    for _, item in indexed:
        key = _group_key(item, "variable", 0)
        if key not in variables:
            variables.append(key)
    use_secondary = facet is None and secondary_y and len(variables) == 2

    if facet is not None:
        groups: dict[Any, list[tuple[int, dict]]] = {}
        for index, item in indexed:
            groups.setdefault(_group_key(item, facet, index), []).append((index, item))
        grouped = list(groups.values())
    elif use_secondary or len(variables) <= 1:
        grouped = [indexed]
    else:
        grouped = [
            [(n, i) for n, i in indexed if _group_key(i, "variable", 0) == v]
            for v in variables
        ]

    panels = []
    for group in grouped:
        primary_items, secondary_items = group, []
        if use_secondary:
            primary_items = [
                (n, i) for n, i in group if _group_key(i, "variable", 0) == variables[0]
            ]
            secondary_items = [
                (n, i) for n, i in group if _group_key(i, "variable", 0) == variables[1]
            ]
        primary = tuple(
            styled[(n, role)]
            for n, _ in primary_items
            for role in ("reference", "test")
        )
        second = tuple(
            styled[(n, role)]
            for n, _ in secondary_items
            for role in ("reference", "test")
        )
        # Title and corner ranking see the *whole* panel, secondary axis included: a
        # title naming one variable while a second is drawn beside it is wrong, and a
        # corner judged empty by the primary lines is where the secondary ones run.
        specs = [line.spec for line in primary + second]
        box = _metrics_text([i for _, i in group], metric_keys, prefix=len(group) > 1)
        if len(group) > METRICS_BOX_MAX_ROWS and box:
            warnings.warn(
                f"{len(group)} comparisons share a panel, so their statistics box "
                "would be a table drawn on a figure; it is left off. Every number is "
                "in the metrics CSV (ComparisonSet.save) either way.",
                stacklevel=_stacklevel.find(),
            )
            box = ""
        # The box takes the emptiest corner and the legend the next emptiest, so the two
        # cannot land on top of each other however the data happens to run.
        ranked = free_corners(primary + second)
        free = ranked[0] if metrics_loc == "auto" else metrics_loc
        legend_at = next((c for c in ranked if c != free), CORNERS[1])
        residual_lines = ()
        if residual:
            residual_lines = tuple(
                _style.with_values(styled[(n, "test")], i["aligned"]["difference"])
                for n, i in primary_items
            )
        panels.append(
            Panel(
                title=panel_title(specs, varying=varying),
                ylabel=_ylabel([line.spec for line in primary]),
                lines=primary,
                secondary=second,
                secondary_ylabel=_ylabel([line.spec for line in second]) or None
                if second
                else None,
                residual=residual_lines,
                metrics_text=box,
                metrics_corner=free,
                legend_corner=legend_at,
            )
        )

    if len(panels) > PANEL_CAP:
        warnings.warn(
            f"{len(panels)} panels on one figure leaves each about "
            f"{11 / len(panels):.1f}in of page — legible only at size='free' or on a "
            "taller canvas. Drawing it anyway; split the set, or facet on something "
            "coarser, if it comes out cramped.",
            stacklevel=_stacklevel.find(),
        )
    crowded = [p for p in panels if len(p.lines) + len(p.secondary) > LINE_CAP]
    if crowded:
        warnings.warn(
            f"{max(len(p.lines) + len(p.secondary) for p in crowded)} lines in one "
            "panel is past what a reader can follow. Drawing it anyway; rows= splits "
            "them into panels.",
            stacklevel=_stacklevel.find(),
        )

    labels = []
    for panel in panels:
        for line in panel.lines + panel.secondary:
            if line.label not in labels:
                labels.append(line.label)
    shared = (
        len({tuple(line.label for line in p.lines + p.secondary) for p in panels}) == 1
    )
    ncols = len(panels) if cols else 1
    nrows = 1 if cols else len(panels)
    return Layout(
        panels=tuple(panels),
        nrows=nrows,
        ncols=ncols,
        legend_labels=tuple(labels),
        shared_legend=shared,
        residual=residual,
    )
