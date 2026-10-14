"""The metrics-series plot: skill metrics drawn as lines against time.

Where :func:`ocean_skill.plot.summary.target`'s ``arrows=`` and
:func:`ocean_skill.plot.portrait.portrait` show a time fan-out
(``compare(..., times={"resample": "1MS", "reduce": "mean"})``) as drift in a diagram
or as a grid of cells, a metrics series is the plain reading of it: one line per
group, x the bin's time, y one metric's value — "monthly RMSE and bias through the
simulation". It reads the same metric records the other summary families read, so it
pools, relabels and groups exactly as they do.

Everything the two renderers must agree on — which records form a line, what order
the points run in, which panel a line lands in, its colour, marker, label and the
panel's title — is decided once, here, in :func:`build_layout`. The static drawing is
:func:`metrics_series`; the interactive renderer calls :func:`build_layout` directly
and only draws what it returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["metrics_series"]

#: The metrics drawn when the caller names none: the two that say how far apart the
#: runs are (rmse) and in which direction (bias).
DEFAULT_METRICS = ("rmse", "bias")

_ROWS_CHOICES = ("metric", "variable")


@dataclass
class Line:
    """One line: every record of one (colour level, marker level) in one panel."""

    label: str
    color: str
    marker_index: int
    x: np.ndarray  # datetime64[ns], ascending
    y: np.ndarray  # float, NaN where the metric is undefined
    recs: list[dict[str, Any]]


@dataclass
class Panel:
    """One axes: a metric (for one unit group, for one row level) and its lines."""

    title: str
    metric: str
    row: int
    col: int
    lines: list[Line] = field(default_factory=list)


@dataclass
class Layout:
    """The whole figure: a grid of panels, and the legend entries they share."""

    panels: list[Panel]
    nrows: int
    ncols: int
    #: ``(label, color, marker_index)`` per (colour, marker) combination present.
    legend: list[tuple[str, str, int]]
    #: Whether the legend lists the same lines in every panel (so one key serves all).
    uniform: bool


def _stamp(text: str):
    """One date string as a :class:`pandas.Timestamp`.

    A 360-day calendar spells dates (``"2000-02-30"``) pandas rejects, so a failed parse
    falls back to the year/month/day digits with the day clamped to 28 — a point's
    position on a time axis, not a date anybody reads back.
    """
    import pandas as pd

    try:
        return pd.Timestamp(text)
    except (ValueError, TypeError):
        match = re.match(r"\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", text)
        if match is None:
            raise ValueError(f"cannot read {text!r} as a date") from None
        year, month, day = (int(g) if g else 1 for g in match.groups())
        return pd.Timestamp(year=year, month=month, day=min(day, 28))


def time_of(value) -> Any:
    """Where on the time axis a record's ``time`` value belongs, as a Timestamp.

    A ``{"min", "max"}`` window (how a ``"3MS"`` bin is spelled) and a partial date
    (``"2010-01"``, a whole month) both name a *period*; the point is drawn at the
    period's midpoint, the usual place for a monthly mean. A full timestamp is itself.
    A pooled record's ``"2010-01-01–2010-03-31"`` label reads as the window it names.
    Anything else — a season name, say — raises, since it has no place on a time axis.
    """
    import pandas as pd

    if isinstance(value, dict) and {"min", "max"} <= set(value):
        lo, hi = _stamp(str(value["min"])), _stamp(str(value["max"]))
        return lo + (hi - lo) / 2
    if isinstance(value, str) and "–" in value:
        lo, hi = value.split("–", 1)
        return time_of({"min": lo, "max": hi})
    text = str(value)
    # a full timestamp names an instant, not a period
    if re.search(r"[T:]|\d \d", text):
        try:
            return _stamp(text)
        except ValueError:
            pass
    try:
        period = pd.Period(text)
    except (ValueError, TypeError):
        try:
            return _stamp(text)
        except ValueError:
            raise ValueError(
                f"metrics_series draws metrics against time, but a comparison's time "
                f"is {value!r}, which is not a date or a period (a season name has no "
                "place on a time axis). Use compare(times={'resample': '1MS', "
                "'reduce': 'mean'}) or a list of dates."
            ) from None
    # the next period's start, not end_time (the last tick, whose resolution varies)
    return period.start_time + ((period + 1).start_time - period.start_time) / 2


def _units_of(metric: str, rec: dict[str, Any]) -> str | None:
    """Return the units of a record's ``metric``, or ``None`` for none/unknown.

    Only a metric registered as ``"same"`` (the variable's own units) takes the
    variable's; a correlation or a count is unitless, and one this package has no
    registry entry for is not guessed at.
    """
    from ocean_skill.metrics import REGISTRY

    entry = REGISTRY.get(metric)
    if entry is None or entry.units != "same":
        return None
    return rec.get("units") or None


def _check_metrics(recs, requested) -> tuple[str, ...]:
    names = (requested,) if isinstance(requested, str) else tuple(requested)
    if not names:
        raise ValueError("metrics_series needs at least one metric, got none")
    available = {k for r in recs for k in r}
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(
            f"no metric {missing} in these records — computed: {sorted(available)}. "
            "Which metrics are computed is decided by compute(), not by "
            "metrics_series()."
        )
    return names


def build_layout(
    recs: list[dict[str, Any]],
    *,
    metrics=DEFAULT_METRICS,
    color_by: str | None = "variable",
    marker_by: str | None = None,
    rows: str = "metric",
    colors=None,
    level_labels=None,
) -> Layout:
    """Group ``recs`` into panels of time-ordered lines — the shared decisions.

    A **line** is every record sharing one ``color_by`` level (and one ``marker_by``
    level); its points run in time order. Two records of one line at the same time
    raise rather than zigzag: name another field to tell them apart.

    **Panels.** ``rows="metric"`` gives one row per metric, sharing the time axis. A
    metric in the variable's own units (rmse, bias, ...) cannot share one axis between
    variables in *different* units, so such a metric splits into one panel per distinct
    unit string, laid out as columns; variables that share units stay on one axis.
    Unitless metrics (corr) never split. ``rows="variable"`` is the transpose: one row
    per variable and one column per metric, so every panel has one variable's units.
    """
    from ocean_skill.plot.summary import (
        _field_levels,
        _resolve_colors,
        pretty_level,
    )

    if not recs:
        raise ValueError("metrics_series needs at least one comparison, got none")
    if rows not in _ROWS_CHOICES:
        raise ValueError(
            f"rows={rows!r} is not one of {list(_ROWS_CHOICES)} — 'metric' stacks one "
            "panel per metric, 'variable' one row per variable with a column per metric"
        )
    names = _check_metrics(recs, metrics)
    untimed = [r for r in recs if r.get("time") is None]
    if untimed:
        raise ValueError(
            f"metrics_series needs a time on every comparison, but {len(untimed)} of "
            f"{len(recs)} have none. Build the set with a time fan-out — "
            "osk.compare(..., times={'resample': '1MS', 'reduce': 'mean'}) — so each "
            "comparison carries one period."
        )
    xs = [time_of(r["time"]) for r in recs]

    style_field = color_by or marker_by or "label"
    style_levels = _field_levels(recs, style_field)
    level_colors = _resolve_colors(colors, style_levels, style_field)
    marker_levels = _field_levels(recs, marker_by) if marker_by else []

    def _label(rec) -> str:
        parts = []
        if color_by or not marker_by:
            parts.append(pretty_level(style_field, rec.get(style_field), level_labels))
        if marker_by:
            parts.append(pretty_level(marker_by, rec.get(marker_by), level_labels))
        return " · ".join(parts)

    def _panel_lines(indices: list[int], name: str) -> list[Line]:
        lines = []
        for mi, mlevel in enumerate(marker_levels or [None]):
            for level in style_levels:
                picked = [
                    i
                    for i in indices
                    if recs[i].get(style_field) == level
                    and (mlevel is None or recs[i].get(marker_by) == mlevel)
                ]
                if not picked:
                    continue
                picked.sort(key=lambda i: xs[i])
                stamps = [xs[i] for i in picked]
                if len(set(stamps)) != len(stamps):
                    raise ValueError(
                        f"metrics_series: more than one comparison has "
                        f"{style_field}={level!r} at the same time — a line is one "
                        "value per time. Tell them apart with color_by=/marker_by= "
                        "(for example 'depth', 'test' or 'reference'), or narrow "
                        "the set with .sel()."
                    )
                lines.append(
                    Line(
                        label=_label(recs[picked[0]]),
                        color=level_colors[level],
                        marker_index=mi,
                        x=np.array(stamps, dtype="datetime64[ns]"),
                        y=np.array([_value(recs[i], name) for i in picked]),
                        recs=[recs[i] for i in picked],
                    )
                )
        return lines

    def _value(rec, name: str) -> float:
        value = rec.get(name)
        return float(value) if value is not None else np.nan

    def _title(name: str, units: str | None, prefix: str | None = None) -> str:
        text = f"{name} ({units})" if units else name
        return f"{prefix}: {text}" if prefix else text

    panels: list[Panel] = []
    if rows == "metric":
        ncols = 1
        for row, name in enumerate(names):
            groups = list(
                dict.fromkeys(_units_of(name, r) for r in recs)
            )  # distinct unit strings, first seen first
            ncols = max(ncols, len(groups))
            for col, units in enumerate(groups):
                indices = [i for i, r in enumerate(recs) if _units_of(name, r) == units]
                panels.append(
                    Panel(
                        _title(name, units),
                        name,
                        row,
                        col,
                        _panel_lines(indices, name),
                    )
                )
        nrows = len(names)
    else:
        variables = _field_levels(recs, "variable")
        for row, variable in enumerate(variables):
            indices = [i for i, r in enumerate(recs) if r.get("variable") == variable]
            for col, name in enumerate(names):
                units = _units_of(name, recs[indices[0]])
                panels.append(
                    Panel(
                        _title(
                            name,
                            units,
                            pretty_level("variable", variable, level_labels),
                        ),
                        name,
                        row,
                        col,
                        _panel_lines(indices, name),
                    )
                )
        nrows, ncols = len(variables), len(names)

    legend: list[tuple[str, str, int]] = []
    for panel in panels:
        for line in panel.lines:
            entry = (line.label, line.color, line.marker_index)
            if entry not in legend:
                legend.append(entry)
    sets = [{(ln.label, ln.color, ln.marker_index) for ln in p.lines} for p in panels]
    uniform = all(s == sets[0] for s in sets)
    return Layout(panels, nrows, ncols, legend, uniform)


def metrics_series(
    comparisons,
    *,
    metrics: str | tuple[str, ...] = DEFAULT_METRICS,
    color_by: str | None = "variable",
    marker_by: str | None = None,
    rows: str = "metric",
    colors=None,
    groups: dict[str, Any] | None = None,
    level_labels: dict[str, dict[str, str]] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    save: str | Path | None = None,
    hover: bool | None = None,
    rasterize: bool | str | None = None,
):
    """Metrics series: skill metrics as lines against time, one panel per metric.

    Built for a set from ``compare(..., times={"resample": "1MS", "reduce":
    "mean"})`` — one comparison per time bin, each metric record carrying the bin's
    ``time`` — but any set whose records carry a ``time`` works. A set with no time on
    its records raises, telling you to pass ``times=`` to ``compare()``. Each point is
    drawn at the middle of its period (a whole month at mid-month), and a series of
    monthly bias and RMSE reads straight off the figure.

    ``metrics`` names what to draw, one panel per name (default ``("rmse", "bias")``);
    any metric the records carry works, ``corr`` and ``sigma_ratio`` included. A name
    none of the records carry raises, naming what they do carry. The panels stack in
    that order and share the time axis; a ``bias`` panel gets a faint zero line.

    ``color_by`` (default ``"variable"``) names the metric-record field that splits the
    lines — ``"variable"``, ``"depth"``, ``"test"``, ``"reference"``, ``"label"``,
    ``groups=``'s ``"group"``, anything the records carry — and ``marker_by`` splits
    them further by marker shape, so ``color_by="variable", marker_by="test"`` draws
    one line per (variable, run). Two records of one line at the same time raise, since
    a line is one value per time; name another field to tell them apart. ``colors``,
    ``groups`` and ``level_labels`` mean exactly what they do in
    :func:`ocean_skill.plot.summary.taylor`.

    **Units.** A metric in the variable's own units (rmse, bias) has no meaning across
    variables in different units on one axis. When ``rows="metric"`` (the default) such
    a metric therefore gets one panel per distinct unit string, laid out as columns —
    temperature in °C and salinity in psu come out as two columns, while two variables
    sharing units share an axis — and every panel title names its units. Unitless
    metrics (``corr``) never split. ``rows="variable"`` is the transpose, one row per
    variable and one column per metric, so each panel holds a single variable's units.

    ``title`` is a figure title; ``figsize``, ``size``, ``zoom`` and ``font_scale`` size
    the figure as for the other plots (``size`` a canvas name such as ``"page"`` or
    ``"slide"``, ``zoom`` a multiplier, ``font_scale`` type only). ``save`` writes the
    figure to that path. ``hover``/``rasterize`` are accepted only so
    ``renderer="both"`` can pass one option set to each renderer and do nothing here.

    Returns the :class:`~matplotlib.figure.Figure` (the interactive renderer returns a
    holoviews object with the same panels, colours and markers, a hover on every point).
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ocean_skill.plot.matplotlib_renderer import _warn_if_interactive_only
    from ocean_skill.plot.summary import (
        _MARKERS,
        _apply_groups,
        _records,
        _resolve_level_labels,
    )
    from ocean_skill.plot.typography import (
        SERIES_ASPECT,
        SERIES_OVERHEAD,
        SERIES_PANEL_W_FRACTION,
        resolve_canvas,
        row_height,
        type_scale,
    )

    _warn_if_interactive_only(rasterize, hover)
    recs = _records(comparisons)
    if _apply_groups(recs, groups) and not color_by:
        color_by = "group"
    level_labels = _resolve_level_labels(level_labels, recs)
    layout = build_layout(
        recs,
        metrics=metrics,
        color_by=color_by,
        marker_by=marker_by,
        rows=rows,
        colors=colors,
        level_labels=level_labels,
    )

    canvas = resolve_canvas(size, zoom)
    if figsize is None:
        height = row_height(
            SERIES_ASPECT,
            nrows=layout.nrows,
            ncols=layout.ncols,
            canvas=canvas,
            font_scale=font_scale,
            panel_w_fraction=SERIES_PANEL_W_FRACTION,
            overhead=SERIES_OVERHEAD,
        )
        # a figure-level key sits under the last row; one row of entries needs ~0.4in
        figsize = (canvas.width, height * layout.nrows + 0.4)
    scale = type_scale(
        figsize, ncols=layout.ncols, nrows=layout.nrows, font_scale=font_scale
    )

    fig, axes = plt.subplots(
        layout.nrows,
        layout.ncols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    occupied = {(p.row, p.col) for p in layout.panels}
    for r in range(layout.nrows):
        for c in range(layout.ncols):
            if (r, c) not in occupied:
                axes[r, c].set_visible(False)
    for panel in layout.panels:
        ax = axes[panel.row, panel.col]
        for line in panel.lines:
            ax.plot(
                line.x,
                line.y,
                color=line.color,
                marker=_MARKERS[line.marker_index % len(_MARKERS)],
                ms=4.5,
                lw=1.4,
            )
        if panel.metric == "bias":
            ax.axhline(0, color="0.7", lw=0.7, zorder=0, gid="zero")
        ax.set_title(panel.title, fontsize=scale["title"])
        ax.tick_params(labelsize=scale["tick_label"])
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    # sharex hides every inner tick label, including under a panel whose own column has
    # a blank cell beneath it: the lowest visible panel of each column keeps them
    for c in range(layout.ncols):
        lowest = max((p.row for p in layout.panels if p.col == c), default=None)
        if lowest is not None:
            axes[lowest, c].tick_params(labelbottom=True)

    if layout.legend:
        handles = [
            Line2D(
                [],
                [],
                color=color,
                marker=_MARKERS[mi % len(_MARKERS)],
                ms=5,
                lw=1.4,
                label=label,
            )
            for label, color, mi in layout.legend
        ]
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=min(len(handles), 5),
            fontsize=scale["legend"],
            frameon=False,
        )
    if title:
        fig.suptitle(title, fontsize=scale["suptitle"])
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
