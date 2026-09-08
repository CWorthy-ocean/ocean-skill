"""Composition for the ``profile`` family: items in, a fully resolved layout out.

The vertical twin of :mod:`ocean_skill.plot.series` — value on x, depth on y, the
axis inverted so the surface draws at the top and the seafloor at the bottom. Reuses
:mod:`ocean_skill.plot.series`'s :class:`~ocean_skill.plot.series.Layout`/
:class:`~ocean_skill.plot.series.Panel`/:mod:`ocean_skill.plot.style` machinery by
import rather than by copy, so the two families cannot drift apart on anything but
the axes themselves: which channel a line's colour/dash comes from, how a
statistics box picks its corner, how a legend is built, all still mean what they
mean in :mod:`ocean_skill.plot.series`.

Two things a profile line knows that a series line does not: which *cast* (instant)
it was taken at -- ``depth`` is always ``None`` on a profile's own
:class:`~ocean_skill.plot.style.LineSpec` (depth is the axis every line already
draws against, not a fact to style a line by), and ``time`` takes its place as the
marker channel, so several casts overlaid in one panel still tell apart.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from ocean_skill import _stacklevel
from ocean_skill.plot import series as _series_layout
from ocean_skill.plot import style as _style

__all__ = ["compose", "depth_range", "fan_season", "panel_title", "vertical_values"]

#: Fraction of a panel's axes a corner box occupies -- the same measure
#: :data:`ocean_skill.plot.series._CORNER_W`/``_CORNER_H`` use, imported rather than
#: redefined so the two families' corner logic cannot drift apart on this constant.
_CORNER_W, _CORNER_H = _series_layout._CORNER_W, _series_layout._CORNER_H

#: The dimension name a season groupby produces (see
#: :func:`ocean_skill.operators._reduce_dim`'s ``SeasonGrouper`` route) --
#: always this literal name, whatever custom ``seasons=`` a caller passed. One
#: constant so :func:`fan_season` and every reader of the scalar coordinate it
#: leaves behind (:func:`ocean_skill.plot.series.season_of`) agree on it.
SEASON_DIM = "season"

#: The dimension name a month groupby produces
#: (``aggregate={"time": {"groupby": "month", ...}}``). Unlike :data:`SEASON_DIM`
#: this name alone does not mean "fan me" -- an ordinary ``month`` dim a caller
#: built some other way is refused like any other extra axis (see
#: :meth:`ocean_skill.field.Field._profile_items`) -- so :func:`fan_season` only
#: fans a ``month`` dim whose coordinate also carries
#: :data:`ocean_skill.operators.TIME_GROUPBY_ATTR`, the same marker
#: :func:`ocean_skill.operators.time_axis_dim` reads to route this shape here in
#: the first place (an explicit ``select={"month": [...]}`` list, see
#: :meth:`~ocean_skill.field.Field._time_axis_dim`).
MONTH_DIM = "month"


def _is_time_groupby_dim(aligned, dim: str) -> bool:
    from ocean_skill.operators import TIME_GROUPBY_ATTR

    coord = aligned.coords.get(dim)
    return coord is not None and TIME_GROUPBY_ATTR in coord.attrs


def fan_season(items: list[dict]) -> list[dict]:
    """Split a surviving season or (marked) month dim into one item per value.

    The codebase's standing idiom for "several lines in one profile panel" is
    several *items*, not one item with a surviving axis (see
    ``Field._series_items`` fanning depth levels) -- a surviving season axis
    fans the same way, chronologically; a surviving ``month`` axis (only when
    an explicit ``select={"month": [...]}`` turned off its "this is time"
    reading -- see :data:`MONTH_DIM`) fans the same way too, in the order the
    caller named the months. ``.isel`` leaves the dim as a scalar coordinate on
    each slice (the convention a profile's own ``time`` already follows -- see
    :func:`_time_of`), and slices a same-dims ``spread`` coordinate along with
    it for free. Idempotent: a fanned item carries a scalar coordinate, not a
    dimension, so calling this on already-fanned items is a no-op.
    """
    fanned = []
    for item in items:
        aligned = item["aligned"]
        dim = None
        if SEASON_DIM in aligned.dims:
            dim = SEASON_DIM
        elif MONTH_DIM in aligned.dims and _is_time_groupby_dim(aligned, MONTH_DIM):
            dim = MONTH_DIM
        if dim is None:
            fanned.append(item)
            continue
        for k in range(aligned.sizes[dim]):
            fanned.append({**item, "aligned": aligned.isel({dim: k})})
    return fanned


def _vertical_coord(da):
    """Return the coordinate ``da``'s vertical dimension actually carries its values on.

    Usually the dimension's own coordinate (``z``, ``depth``, ``sigma0``, ...), found
    tolerant of a coordinate riding under a different name than its dimension (see
    :func:`ocean_skill.operators.vertical_coord_on` -- a catalog recipe's ``depth`` on
    a dimension still spelled ``DEPTH``, say). A native s-level profile
    (``select={"depth": "column"}``) is the one exception past that: its dimension is
    a bare sigma index (ROMS ships no coordinate for ``s_rho`` itself) with the real
    depth riding on the auxiliary ``z_rho`` coordinate instead -- exactly the same
    distinction :func:`ocean_skill.plot.section.prepare_section` makes for a section's
    native-s axis, and for the identical reason.
    """
    from ocean_skill.operators import vertical_coord_on

    dim = str(da.dims[0])
    coord = vertical_coord_on(da, dim)
    if coord is not None:
        return coord
    if "z_rho" in da.coords:
        return da.coords["z_rho"]
    raise ValueError(
        f"a profile line needs a coordinate on its own dimension ({dim!r}) -- or, "
        "for native s-levels, a z_rho coordinate -- to draw against; this one "
        "carries neither."
    )


def vertical_values(da) -> np.ndarray:
    """Return ``da``'s vertical coordinate as positive-down (or positive-density) values.

    The axis every profile panel draws against, read off ``da``'s own (and only)
    dimension (see :func:`_vertical_coord` for where that coordinate actually
    lives). A depth-like axis (``z``, negative-down from
    :func:`ocean_skill.roms.to_depth`; ``z_rho``, negative-down native s-levels;
    ``depth``/``DEPTH``/``lev``, already positive-down from an observational
    product) comes back as ``abs()`` of its raw coordinate -- a no-op for an axis
    that was already positive, and exactly what turns ROMS's negative-down
    convention into the positive-down metres every other depth label in this
    package uses (see ``facet_labels``' own ``abs()`` in
    :mod:`ocean_skill.plot.matplotlib_renderer`). A ``sigma0`` axis is already
    positive (density anomaly, roughly 20-28 kg/m3), so ``abs()`` there is a no-op
    too -- there is no third case to special-case.
    """
    return np.abs(np.asarray(_vertical_coord(da).values, dtype="float64"))


def depth_range(
    lines, *, ylim: tuple[float, float] | None = None
) -> tuple[float, float]:
    """``(y_bottom, y_top)`` -- deep at the bottom, shallow at top; ``ylim`` overrides.

    One computation shared by both renderers and by every axis that draws this
    range, whether that is every panel in the figure (``sharey=True``, the
    default) or just one panel's own lines (``sharey=False``) -- so the two
    modes agree on exactly how an empty or all-``NaN`` panel falls back
    (``(1.0, 0.0)``, a full-figure axis with nothing plotted).
    """
    if ylim is not None:
        return float(ylim[1]), float(ylim[0])
    lines = list(lines)
    if lines:
        depths = np.concatenate([vertical_values(line.spec.values) for line in lines])
        finite = depths[np.isfinite(depths)]
        if finite.size:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            return (hi, lo) if hi > lo else (lo + 1.0, lo)
    return 1.0, 0.0


def _vertical_label(specs) -> str:
    """``"Depth [m]"`` or ``"σ₀ [kg/m³]"`` -- the axis every line in the panel shares.

    Read off the first spec's own values, the same "one axis, so one name" rule
    :func:`ocean_skill.plot.series._ylabel` applies to the value axis.
    """
    if str(specs[0].values.dims[0]) == "sigma0":
        return "σ₀ [kg/m³]"
    units = _vertical_coord(specs[0].values).attrs.get("units") or "m"
    return f"Depth [{units}]"


def _time_of(aligned) -> str | None:
    """The cast's own instant, pre-formatted -- or ``None`` for a multi-time item.

    A profile item carries its time as a scalar ``time`` coordinate (the contract
    :meth:`ocean_skill.field.Field._profile_items` and
    :meth:`ocean_skill.comparison.Comparison.as_item` both follow); several casts
    overlaid in one figure are several *items*, not one item with a surviving time
    axis, so ``time`` here is always scalar or absent, never an array to summarize.
    """
    time = aligned.coords.get("time")
    if time is None or time.dims:
        return None
    try:
        return str(np.datetime64(time.values, "s"))[:16].replace("T", " ")
    except (TypeError, ValueError):
        return str(time.values)


def _line_specs(item: dict[str, Any], index: int = 0) -> list[_style.LineSpec]:
    """Return the line(s) one item draws -- :func:`ocean_skill.plot.series.line_specs`
    with ``time`` (this cast's own instant) in place of ``depth`` (always ``None``
    here; see the module docstring). ``season``/``month`` -- a fanned season or
    narrowed month, if this item's aligned data was reduced to one -- and
    ``spread`` -- a mean±spread envelope, if the aggregate computed one -- are the
    same fields series carries, read the same way
    (:func:`ocean_skill.plot.series.season_of`/``month_of``/``spread_of``).
    """
    aligned = item["aligned"]
    variable = item.get("standard_name") or item.get("label")
    time = _time_of(aligned)
    season = _series_layout.season_of(aligned)
    month = _series_layout.month_of(aligned)
    if _series_layout.item_roles(item) == ("value",):
        source = str((item.get("labels") or (item.get("label") or "value",))[0])
        units = item.get("units") or aligned["value"].attrs.get("units")
        return [
            _style.LineSpec(
                role="value",
                source=source,
                variable=variable,
                time=time,
                season=season,
                month=month,
                spread=_style.spread_of(aligned, "value", aligned["value"]),
                units=units,
                values=aligned["value"],
                item=index,
            )
        ]
    test_source, reference_source = item.get("labels") or ("test", "reference")
    units = item.get("units") or aligned["reference"].attrs.get("units")
    common = {
        "variable": variable,
        "time": time,
        "season": season,
        "month": month,
        "units": units,
        "item": index,
    }
    return [
        _style.LineSpec(
            role="reference",
            source=str(reference_source),
            values=aligned["reference"],
            spread=_style.spread_of(aligned, "reference", aligned["reference"]),
            **common,
        ),
        _style.LineSpec(
            role="test",
            source=str(test_source),
            values=aligned["test"],
            spread=_style.spread_of(aligned, "test", aligned["test"]),
            **common,
        ),
    ]


def _group_key(item: dict[str, Any], by: str | None, index: int):
    """Return the value ``item`` is grouped by, for ``rows=``/``cols=``.

    Mirrors :func:`ocean_skill.plot.series._group_key`, with ``time`` (which cast)
    in place of ``depth`` -- and ``depth`` itself refused: it is the axis every
    panel already draws against, not a fact to split panels on.
    """
    if by is None:
        return None
    if by in ("variable", "standard_name"):
        return item.get("standard_name") or item.get("label")
    if by in ("source", "test"):
        return (item.get("labels") or ("test", "reference"))[0]
    if by == "reference":
        labels = item.get("labels") or ("test", "reference")
        if len(labels) < 2:
            raise ValueError(
                "cannot facet a profile by 'reference': this item has a single "
                "source with nothing compared against it, so there is no "
                "reference side to group by."
            )
        return labels[1]
    if by == "time":
        return _time_of(item["aligned"])
    if by == "season":
        return _series_layout.season_of(item["aligned"])
    if by == "month":
        return _series_layout.month_of(item["aligned"])
    if by == "depth":
        raise ValueError(
            "cannot facet a profile by 'depth': depth is the axis every panel "
            "already draws against, not a fact to split panels on. Facet on "
            "variable, source, reference, time, season, month or comparison "
            "instead."
        )
    if by == "comparison":
        return index
    raise ValueError(
        f"cannot facet a profile by {by!r}; expected one of variable, source, "
        "reference, time, season, month, comparison."
    )


def _refuse_depth_encode(encode: dict[str, str | None] | None) -> None:
    if encode and "depth" in encode.values():
        raise ValueError(
            "cannot encode a profile channel by 'depth': depth is the axis every "
            "panel already draws against, not a fact to style a line by. Encode "
            "by variable, source, role, time, season or month instead."
        )


def panel_title(specs, *, varying, source: str | None = None) -> str:
    """Return a panel title of identity only: what, where, when.

    Mirrors :func:`ocean_skill.plot.series.panel_title`, with "when" read off each
    line's own ``time`` (a cast's instant, not a period a time axis spans) rather
    than :func:`~ocean_skill.plot.series._period_of` -- and shown only when every
    line in the panel shares one, so a multi-cast overlay (whose lines already
    carry their own times in the legend) does not claim a single "when" for all of
    them. ``month`` follows the identical rule, one level up: a
    ``cols="month"``-faceted panel titles each with its own month.

    ``source=`` names the one role (usually the station) that distinguishes this
    panel from its neighbours when faceting by ``"comparison"`` -- see ``compose``'s
    own rule for when exactly one role qualifies. ``None`` (every other facet, and
    a comparison facet where zero or several roles distinguish) leaves the title
    exactly as it read before this existed.
    """
    from ocean_skill.plot.series import _place_of, month_label
    from ocean_skill.plot.summary import pretty_level

    parts = []
    variables = {s.variable for s in specs if s.variable}
    if len(variables) == 1:
        parts.append(pretty_level("variable", next(iter(variables))))
    if source is not None:
        parts.append(pretty_level("source", source))
    reference = next((s for s in specs if s.role == "reference"), specs[0])
    place = _place_of(reference.values)
    if place:
        parts.append(place)
    # Shown only when every line in the panel shares one season/month -- a
    # cols="season" (or "month") facet titles each panel with its own value; an
    # overlay of several already tells them apart by colour/legend, so no single
    # "when" is claimed.
    seasons = {s.season for s in specs if s.season}
    if len(seasons) == 1:
        parts.append(next(iter(seasons)))
    months = {s.month for s in specs if s.month is not None}
    if len(months) == 1:
        parts.append(month_label(next(iter(months))))
    times = {s.time for s in specs if s.time}
    if len(times) == 1:
        parts.append(next(iter(times)))
    return " · ".join(parts)


def _dropped_source_label(spec, *, varying, ambiguous_sources) -> str:
    """One surfaced-role line's legend entry once its station moved to the title.

    :func:`ocean_skill.plot.style.series_label` falls back to ``spec.source``
    when nothing else varies -- exactly what would silently re-print the name
    the title now already says. This drops the source unconditionally instead,
    reading whatever else varies, or nothing at all (an empty label, which both
    renderers already know to leave out of the legend) if the station was the
    only thing that did.
    """
    from ocean_skill.plot.summary import pretty_level

    parts = []
    for field in ("variable", "depth", "time", "season", "month"):
        if field in varying:
            value = spec.get(field)
            if value is not None:
                parts.append(pretty_level(field, value))
    if spec.source in ambiguous_sources:
        parts.append(f"({spec.role})")
    return " · ".join(parts)


def _free_corners(lines) -> list[str]:
    """Return the panel's corners, emptiest first -- the profile twin of
    :func:`ocean_skill.plot.series.free_corners`, with the axes swapped: x is a
    line's own value (scaled 0-1, exactly as ``free_corners`` scales its y), y is
    its own depth, scaled 0-1 and then flipped (1=shallowest, 0=deepest) to match
    how the panel actually draws once its y-axis reads surface-at-top -- axes-
    fraction y=1 is always the top of the panel, whatever the data axis reads.
    Depth's own values are used directly rather than sample order (the proxy
    ``free_corners`` needs for time): unlike time, depth is one comparable
    quantity across every line, so there is no reason to approximate it.
    """
    xs, ys = [], []
    for line in lines:
        values = np.asarray(line.spec.values.values, dtype="float64")
        depth = vertical_values(line.spec.values)
        finite = np.isfinite(values) & np.isfinite(depth)
        if not finite.any():
            continue
        vlow, vhigh = np.nanmin(values), np.nanmax(values)
        vspan = vhigh - vlow
        x = (values - vlow) / vspan if vspan else np.full(values.size, 0.5)
        dlow, dhigh = np.nanmin(depth), np.nanmax(depth)
        dspan = dhigh - dlow
        scaled = (depth - dlow) / dspan if dspan else np.full(depth.size, 0.5)
        y = 1.0 - scaled
        xs.append(x[finite])
        ys.append(y[finite])
    if not xs:
        return list(_series_layout.CORNERS)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    counts = {}
    for corner in _series_layout.CORNERS:
        vertical, horizontal = corner.split()
        in_x = x <= _CORNER_W if horizontal == "left" else x >= 1 - _CORNER_W
        in_y = y >= 1 - _CORNER_H if vertical == "upper" else y <= _CORNER_H
        counts[corner] = int(np.count_nonzero(in_x & in_y))
    return sorted(
        _series_layout.CORNERS, key=lambda c: (counts[c], _series_layout.CORNERS.index(c))
    )


def compose(
    items,
    *,
    rows: str | None = None,
    cols: str | None = None,
    secondary_x: bool = True,
    encode: dict[str, str | None] | None = None,
    metric_keys=(),
    metrics_loc: str = "auto",
    metrics_stacked: bool = False,
    colors=None,
    legend: bool | str = True,
    line_labels: Sequence[str] | None = None,
    titles: Sequence[str] | None = None,
    ncols: int | None = None,
    nrows: int | None = None,
) -> _series_layout.Layout:
    """Group ``items`` into panels and resolve every line's style and labelling.

    ``ncols=``/``nrows=`` wrap the panels into a rectangular grid instead of
    today's single row/column default; see :func:`ocean_skill.plot.series.grid_shape`.
    ``metrics_stacked=True`` keeps the statistics box narrow-and-tall (one metric
    per line) instead of the default single wide line -- a box sized for a
    portrait panel rather than a page-wide one; see
    :func:`ocean_skill.plot.series._metrics_text`. ``colors=`` pins the auto
    colour cycle to specific values instead; see
    :func:`ocean_skill.plot.style.resolve`. A band's fill follows for free.

    ``legend=``/``line_labels=`` match :func:`ocean_skill.plot.series.compose`
    exactly (:func:`~ocean_skill.plot.series._normalize_legend`,
    :func:`~ocean_skill.plot.series.remap_line_labels`,
    :func:`~ocean_skill.plot.series.corner_placement`) -- a forced placement
    (``"below"``/``"right"``/a corner name) and custom legend text work the same
    way here as they do for :mod:`series`.

    A ``cols="comparison"``/``rows="comparison"`` facet (one panel per station)
    auto-promotes the station into its panel's title and drops it from that
    line's legend entry -- see :func:`panel_title` and
    :func:`_dropped_source_label`. ``titles=`` overrides the result by hand
    afterward, one string per panel in panel order; the wrong count raises a
    copy-pasteable ``ValueError`` listing the current titles, the same way
    ``line_labels=``'s does.

    Composition follows the same bounded rule :mod:`ocean_skill.plot.series` does:
    at most one user facet (``rows=`` or ``cols=``), plus at most one
    ``secondary_x`` -- the profile twin of series' ``secondary_y``, transposed:
    a profile's value axis is x (depth is y), so the second axis it grows is a
    *top* x axis rather than a right-hand y axis.

    ===================  ==================================================================
    one variable         one panel, every source/cast overlaid
    two variables        one panel, the second on a top x axis (``secondary_x``)
    three or more         one column per variable, sources/casts overlaid within each
    ===================  ==================================================================
    """
    items = fan_season(list(items))
    if not items:
        raise ValueError("a profile needs at least one comparison to draw")
    if rows is not None and cols is not None:
        raise ValueError(
            f"a profile takes one facet, not two: rows={rows!r} and cols={cols!r} "
            "were both given. Overlaying the lanes of each comparison is already "
            "one axis; pick rows= or cols= for the other."
        )
    _refuse_depth_encode(encode)

    indexed = list(enumerate(items))
    facet = rows or cols
    variables = []
    for _, item in indexed:
        key = _group_key(item, "variable", 0)
        if key not in variables:
            variables.append(key)
    use_secondary = facet is None and secondary_x and len(variables) == 2

    all_specs = [s for i, item in enumerate(items) for s in _line_specs(item, i)]
    # marker <- time replaces series' marker <- depth: every profile spec's own
    # depth is None (depth is the axis, not a style channel here), so time takes
    # its place as the default marker key, overridable like any other channel.
    # color <- season (or, one level up, month) only when one actually varies
    # across the figure (an overlay of several fanned seasons/months, the
    # default reading of a seasonal or monthly profile) AND there is no twin
    # axis -- on a merged panel the two variables need their own colours
    # (CHANNELS' own color <- variable) so each axis label can honestly say
    # whose lines are whose; an explicit encode= still wins. The two never both
    # apply -- fan_season fans one dim per item -- so either is a safe default.
    seasons_vary = len({s.season for s in all_specs if s.season is not None}) > 1
    months_vary = len({s.month for s in all_specs if s.month is not None}) > 1
    defaults = {
        "marker": "time",
        **(
            {"color": "season"}
            if seasons_vary and not use_secondary
            else {"color": "month"}
            if months_vary and not use_secondary
            else {}
        ),
    }
    styled = {
        (line.spec.item, line.spec.role): line
        for line in _style.resolve(
            all_specs, encode={**defaults, **(encode or {})}, colors=colors
        )
    }
    varying = _style.varying_fields(all_specs)

    # A "comparison" facet puts one item/station per panel; if exactly one role's
    # source distinguishes panels (the station, usually the reference -- or
    # whichever role the caller varied instead), promote it into the title
    # (panel_title, below) and drop it from that line's own legend entry, since
    # the title already says it. Two or more distinguishing roles (a genuine
    # model-vs-model grid, where a title cannot cleanly name both) or none (a
    # single-panel comparison) leave both the title and the legend unchanged.
    surface_role = None
    if facet == "comparison":
        sources_by_role: dict[str, set] = {}
        for spec in all_specs:
            sources_by_role.setdefault(spec.role, set()).add(spec.source)
        distinguishing = [
            role for role, sources in sources_by_role.items() if len(sources) > 1
        ]
        if len(distinguishing) == 1:
            surface_role = distinguishing[0]
    if surface_role is not None:
        ambiguous = _style.ambiguous_sources(all_specs)
        styled = {
            key: (
                replace(
                    line,
                    label=_dropped_source_label(
                        line.spec, varying=varying, ambiguous_sources=ambiguous
                    ),
                )
                if line.spec.role == surface_role
                else line
            )
            for key, line in styled.items()
        }

    styled = _series_layout.remap_line_labels(styled, all_specs, line_labels)

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

    # A forced corner is handed straight to every Panel below (drawing it is then no
    # different from the "auto" per-panel case); "below"/"right"/"off" have nothing to
    # do per panel and are carried on the Layout instead, for the renderer to act on
    # once, for the whole figure.
    legend_placement = _series_layout._normalize_legend(legend)
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
            for n, it in primary_items
            for role in _series_layout.item_roles(it)
        )
        second = tuple(
            styled[(n, role)]
            for n, it in secondary_items
            for role in _series_layout.item_roles(it)
        )
        # Title and corner ranking see the *whole* panel, secondary axis included:
        # a title naming one variable while a second is drawn beside it is wrong,
        # and a corner judged empty by the primary lines is where the secondary
        # ones run.
        specs = [line.spec for line in primary + second]
        panel_source = None
        if surface_role is not None:
            panel_source = next(
                (s.source for s in specs if s.role == surface_role), None
            )
        box = _series_layout._metrics_text(
            [i for _, i in group],
            metric_keys,
            prefix=len(group) > 1,
            stacked=metrics_stacked,
        )
        # Row count, not item count: a fanned season axis puts several items in
        # one group that all share one comparison's metrics, deduped to one row
        # by _metrics_text -- counting items here would drop a box that, once
        # deduped, easily fits.
        row_count = box.count("\n") + 1 if box else 0
        if row_count > _series_layout.METRICS_BOX_MAX_ROWS:
            warnings.warn(
                f"{row_count} distinct stats rows would share one panel, which "
                "would be a table drawn on a figure; it is left off. Every number "
                "is in the metrics CSV (ComparisonSet.save) either way.",
                stacklevel=_stacklevel.find(),
            )
            box = ""
        ranked = _free_corners(primary + second)
        free, legend_at = _series_layout.corner_placement(
            legend_placement, ranked, metrics_loc
        )
        # Colour a value label like its lines only where a twin axis makes the
        # label/axis pairing ambiguous; a lone axis already says what it is via
        # its title.
        colored = bool(second)
        panels.append(
            _series_layout.Panel(
                title=panel_title(specs, varying=varying, source=panel_source),
                ylabel=_vertical_label(specs),
                lines=primary,
                xlabel=_series_layout._ylabel([line.spec for line in primary]),
                secondary=second,
                secondary_xlabel=_series_layout._ylabel([line.spec for line in second])
                or None
                if second
                else None,
                xlabel_color=_series_layout._label_color(primary) if colored else None,
                secondary_xlabel_color=(
                    _series_layout._label_color(second) if colored else None
                ),
                metrics_text=box,
                metrics_corner=free,
                legend_corner=legend_at,
            )
        )

    if titles is not None:
        titles = list(titles)
        if len(titles) != len(panels):
            current_titles = "\n".join(
                f"  {i + 1}. {panel.title!r}" for i, panel in enumerate(panels)
            )
            raise ValueError(
                f"titles needs one entry per panel -- this figure draws "
                f"{len(panels)}:\n{current_titles}\ngot {len(titles)}. Copy the "
                "list above, edit the text, and pass it back in the same order."
            )
        panels = [replace(p, title=t) for p, t in zip(panels, titles, strict=True)]

    # No explicit facet, and more than one variable that did not merge onto a
    # twin axis: the default columns-per-variable layout (see the docstring
    # table) -- the one case ncols follows the panel count without the caller
    # having asked for cols= itself.
    as_columns = cols is not None or (
        facet is None and len(variables) > 1 and not use_secondary
    )
    eff_nrows, eff_ncols = _series_layout.grid_shape(
        len(panels), as_columns=as_columns, ncols=ncols, nrows=nrows
    )
    wrapped = ncols is not None or nrows is not None
    cap_count = eff_nrows if wrapped else len(panels)
    if cap_count > _series_layout.PANEL_CAP:
        warnings.warn(
            f"{len(panels)} panels on one figure leaves each about "
            f"{11 / cap_count:.1f}in of page — legible only at size='free' or on "
            "a taller canvas. Drawing it anyway; split the set, or facet on "
            "something coarser, if it comes out cramped.",
            stacklevel=_stacklevel.find(),
        )
    crowded = [p for p in panels if len(p.lines) + len(p.secondary) > _series_layout.LINE_CAP]
    if crowded:
        warnings.warn(
            f"{max(len(p.lines) + len(p.secondary) for p in crowded)} lines in one "
            "panel is past what a reader can follow. Drawing it anyway; rows= or "
            "cols= splits them into panels.",
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
    return _series_layout.Layout(
        panels=tuple(panels),
        nrows=eff_nrows,
        ncols=eff_ncols,
        legend_labels=tuple(labels),
        shared_legend=shared,
        # A corner is already baked into every Panel above, so a renderer draws it no
        # differently from "auto" -- *except* it must not then also fall into "auto"'s
        # own shared-labels detection and combine anyway, overriding the very corner
        # the caller forced. "corner" says so explicitly; "auto" is left for when
        # nothing was forced and the renderer is free to decide for itself.
        legend_placement=(
            "corner"
            if legend_placement in _series_layout.CORNERS
            else legend_placement
        ),
        xlabel="",  # unused: every panel carries its own value label (Panel.xlabel)
    )
