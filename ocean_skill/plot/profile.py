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
from typing import Any

import numpy as np

from ocean_skill import _stacklevel
from ocean_skill.plot import series as _series_layout
from ocean_skill.plot import style as _style

__all__ = ["compose", "fan_season", "panel_title", "vertical_values"]

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


def fan_season(items: list[dict]) -> list[dict]:
    """Split a surviving season dim into one item per season, chronologically.

    The codebase's standing idiom for "several lines in one profile panel" is
    several *items*, not one item with a surviving axis (see
    ``Field._series_items`` fanning depth levels) -- a surviving season axis
    fans the same way. ``.isel`` leaves ``season`` as a scalar coordinate on
    each slice (the convention a profile's own ``time`` already follows -- see
    :func:`_time_of`), and slices a same-dims ``spread`` coordinate along with
    it for free. Idempotent: a fanned item carries a scalar coordinate, not a
    dimension, so calling this on already-fanned items is a no-op.
    """
    fanned = []
    for item in items:
        aligned = item["aligned"]
        if SEASON_DIM not in aligned.dims:
            fanned.append(item)
            continue
        for k in range(aligned.sizes[SEASON_DIM]):
            fanned.append({**item, "aligned": aligned.isel({SEASON_DIM: k})})
    return fanned


def _vertical_coord(da):
    """Return the coordinate ``da``'s vertical dimension actually carries its values on.

    Usually the dimension's own coordinate (``z``, ``depth``, ``sigma0``, ...). A
    native s-level profile (``select={"depth": "column"}``) is the one exception:
    its dimension is a bare sigma index (ROMS ships no coordinate for ``s_rho``
    itself) with the real depth riding on the auxiliary ``z_rho`` coordinate
    instead -- exactly the same distinction
    :func:`ocean_skill.plot.section.prepare_section` makes for a section's native-s
    axis, and for the identical reason.
    """
    dim = str(da.dims[0])
    if dim in da.coords:
        return da.coords[dim]
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
    here; see the module docstring). ``season`` -- a fanned season, if this item's
    aligned data was narrowed to one -- and ``spread`` -- a mean±spread envelope,
    if the aggregate computed one -- are the same fields series carries, read the
    same way (:func:`ocean_skill.plot.series.season_of`/``spread_of``).
    """
    aligned = item["aligned"]
    variable = item.get("standard_name") or item.get("label")
    time = _time_of(aligned)
    season = _series_layout.season_of(aligned)
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
    if by == "depth":
        raise ValueError(
            "cannot facet a profile by 'depth': depth is the axis every panel "
            "already draws against, not a fact to split panels on. Facet on "
            "variable, source, reference, time, season or comparison instead."
        )
    if by == "comparison":
        return index
    raise ValueError(
        f"cannot facet a profile by {by!r}; expected one of variable, source, "
        "reference, time, season, comparison."
    )


def _refuse_depth_encode(encode: dict[str, str | None] | None) -> None:
    if encode and "depth" in encode.values():
        raise ValueError(
            "cannot encode a profile channel by 'depth': depth is the axis every "
            "panel already draws against, not a fact to style a line by. Encode "
            "by variable, source, role, time or season instead."
        )


def panel_title(specs, *, varying) -> str:
    """Return a panel title of identity only: what, where, when.

    Mirrors :func:`ocean_skill.plot.series.panel_title`, with "when" read off each
    line's own ``time`` (a cast's instant, not a period a time axis spans) rather
    than :func:`~ocean_skill.plot.series._period_of` -- and shown only when every
    line in the panel shares one, so a multi-cast overlay (whose lines already
    carry their own times in the legend) does not claim a single "when" for all of
    them.
    """
    from ocean_skill.plot.series import _place_of
    from ocean_skill.plot.summary import pretty_level

    parts = []
    variables = {s.variable for s in specs if s.variable}
    if len(variables) == 1:
        parts.append(pretty_level("variable", next(iter(variables))))
    reference = next((s for s in specs if s.role == "reference"), specs[0])
    place = _place_of(reference.values)
    if place:
        parts.append(place)
    # Shown only when every line in the panel shares one season -- a cols="season"
    # facet titles each panel with its season; an overlay of several seasons
    # already tells them apart by colour/legend, so no single "when" is claimed.
    seasons = {s.season for s in specs if s.season}
    if len(seasons) == 1:
        parts.append(next(iter(seasons)))
    times = {s.time for s in specs if s.time}
    if len(times) == 1:
        parts.append(next(iter(times)))
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
    encode: dict[str, str | None] | None = None,
    metric_keys=(),
    metrics_loc: str = "auto",
) -> _series_layout.Layout:
    """Group ``items`` into panels and resolve every line's style and labelling.

    Composition follows the same bounded rule :mod:`ocean_skill.plot.series` does,
    minus the one channel a profile has no room for: at most one user facet
    (``rows=`` or ``cols=``), no ``secondary_y`` (a second value axis at the *top*
    of the panel has no interactive twin -- bokeh has no multi-x the way it has
    ``multi_y`` -- so two or more variables become their own column instead,
    sharing the one depth axis, the standard CTD layout):

    ===================  ==================================================================
    one variable         one panel, every source/cast overlaid
    two or more          one column per variable, sources/casts overlaid within each
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

    all_specs = [s for i, item in enumerate(items) for s in _line_specs(item, i)]
    # marker <- time replaces series' marker <- depth: every profile spec's own
    # depth is None (depth is the axis, not a style channel here), so time takes
    # its place as the default marker key, overridable like any other channel.
    # color <- season only when a season actually varies across the figure (an
    # overlay of several fanned seasons, the default reading of a seasonal
    # profile) -- decided here rather than in CHANNELS itself, since a series
    # line has no use for it yet, and an explicit encode= still wins either way.
    seasons_vary = len({s.season for s in all_specs if s.season is not None}) > 1
    defaults = {"marker": "time", **({"color": "season"} if seasons_vary else {})}
    styled = {
        (line.spec.item, line.spec.role): line
        for line in _style.resolve(all_specs, encode={**defaults, **(encode or {})})
    }
    varying = _style.varying_fields(all_specs)

    indexed = list(enumerate(items))
    facet = rows or cols
    variables = []
    for _, item in indexed:
        key = _group_key(item, "variable", 0)
        if key not in variables:
            variables.append(key)

    if facet is not None:
        groups: dict[Any, list[tuple[int, dict]]] = {}
        for index, item in indexed:
            groups.setdefault(_group_key(item, facet, index), []).append((index, item))
        grouped = list(groups.values())
    elif len(variables) <= 1:
        grouped = [indexed]
    else:
        grouped = [
            [(n, i) for n, i in indexed if _group_key(i, "variable", 0) == v]
            for v in variables
        ]

    panels = []
    for group in grouped:
        primary = tuple(
            styled[(n, role)]
            for n, it in group
            for role in _series_layout.item_roles(it)
        )
        specs = [line.spec for line in primary]
        box = _series_layout._metrics_text(
            [i for _, i in group], metric_keys, prefix=len(group) > 1
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
        ranked = _free_corners(primary)
        free = ranked[0] if metrics_loc == "auto" else metrics_loc
        legend_at = next((c for c in ranked if c != free), _series_layout.CORNERS[1])
        panels.append(
            _series_layout.Panel(
                title=panel_title(specs, varying=varying),
                ylabel=_vertical_label(specs),
                lines=primary,
                xlabel=_series_layout._ylabel([line.spec for line in primary]),
                metrics_text=box,
                metrics_corner=free,
                legend_corner=legend_at,
            )
        )

    if len(panels) > _series_layout.PANEL_CAP:
        warnings.warn(
            f"{len(panels)} panels on one figure leaves each about "
            f"{11 / len(panels):.1f}in of page — legible only at size='free' or on "
            "a taller canvas. Drawing it anyway; split the set, or facet on "
            "something coarser, if it comes out cramped.",
            stacklevel=_stacklevel.find(),
        )
    crowded = [p for p in panels if len(p.lines) > _series_layout.LINE_CAP]
    if crowded:
        warnings.warn(
            f"{max(len(p.lines) for p in crowded)} lines in one panel is past what "
            "a reader can follow. Drawing it anyway; rows= or cols= splits them "
            "into panels.",
            stacklevel=_stacklevel.find(),
        )

    labels = []
    for panel in panels:
        for line in panel.lines:
            if line.label not in labels:
                labels.append(line.label)
    shared = len({tuple(line.label for line in p.lines) for p in panels}) == 1
    # No explicit facet, and more than one variable: the default columns-per-
    # variable layout (see the docstring table) -- the one case ncols follows the
    # panel count without the caller having asked for cols= itself.
    as_columns = cols is not None or (facet is None and len(variables) > 1)
    ncols = len(panels) if as_columns else 1
    nrows = 1 if as_columns else len(panels)
    return _series_layout.Layout(
        panels=tuple(panels),
        nrows=nrows,
        ncols=ncols,
        legend_labels=tuple(labels),
        shared_legend=shared,
        xlabel="",  # unused: every panel carries its own value label (Panel.xlabel)
    )
