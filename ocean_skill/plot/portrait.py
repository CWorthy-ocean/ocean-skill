"""The portrait plot: a heatmap scoreboard of skill metrics.

Where :func:`ocean_skill.plot.summary.taylor`/:func:`~ocean_skill.plot.summary.target`
place one point per comparison to show the *geometric* relationship among a handful of
statistics, a portrait plot pivots the same metric records into a grid — one row per
level of one record field (``variable`` by default), one column per level of another
(``test``, i.e. which run), each cell coloured by a single metric's value. It scales
where the point diagrams cannot: dozens of variable × run combinations stay legible as
coloured cells where they would be overplotted mush as points, row/column banding
("run C is red everywhere below 200 m") jumps out at a glance, and any scalar the
metrics registry computes can fill a cell, not just the four the point diagrams are
built from.

It is the scoreboard counterpart to
:func:`ocean_skill.plot.matplotlib_renderer.skill_map` (metrics across, comparisons
down, one panel each) — a portrait cell is that panel's
single reduced number instead of the map it was reduced from — and reuses its layout and
colour-limit machinery. Colour comes from the same
:func:`ocean_skill.colormaps.metric_colors` every metric panel in the package already
calls, which is what keeps this module's grids and the interactive renderer's agreeing
without either one restating the colour policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ocean_skill.metrics import DEFAULT_MAP_METRICS

__all__ = ["portrait"]


def _resolve_metric_names(recs: list[dict[str, Any]], requested) -> tuple[str, ...]:
    """Resolve and validate the metric(s) to draw, a single name or several.

    A requested metric none of the records carry **raises**, naming what they do
    carry — mirrors :func:`ocean_skill.plot.matplotlib_renderer.metric_panels`, whose
    same reasoning applies here: which metrics exist is decided when they are
    computed, not when they are drawn, so a silently dropped panel would be invisible.
    """
    names = (requested,) if isinstance(requested, str) else tuple(requested)
    available = set(recs[0]) if recs else set()
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            f"no metric {missing} in these records — computed: {sorted(available)}. "
            "Which metrics are computed is decided by compute()/pointwise_metrics(), "
            "not by portrait(); pass metrics=(...) there to add one."
        )
    return names


def _grid(recs: list[dict[str, Any]], row_by: str, col_by: str, name: str):
    """One metric's ``(row_levels, col_levels, matrix)``.

    ``matrix`` is a masked array, masked wherever no record names that
    ``(row_by, col_by)`` combination — a run missing a variable, say — so it reads as
    "no data" rather than as a zero. A metric that came back NaN for a comparison that
    *does* exist (an undefined correlation from too few valid pairs) masks the same
    way: both are "nothing to show here", not two different states worth telling apart
    on a scoreboard.

    Two records sharing one ``(row_by, col_by)`` cell is refused rather than silently
    keeping the last one seen — a scoreboard cell is one number, and averaging or
    overwriting would hide that the request needs a third grouping field to resolve.
    """
    from ocean_skill.plot.summary import _field_levels

    row_levels = _field_levels(recs, row_by)
    col_levels = _field_levels(recs, col_by)
    row_index = {v: i for i, v in enumerate(row_levels)}
    col_index = {v: i for i, v in enumerate(col_levels)}
    matrix = np.full((len(row_levels), len(col_levels)), np.nan)
    seen = set()
    for r in recs:
        key = (r.get(row_by), r.get(col_by))
        if key in seen:
            raise ValueError(
                f"portrait: more than one comparison has {row_by}={key[0]!r}, "
                f"{col_by}={key[1]!r} — a scoreboard cell is one number. Narrow the "
                "comparisons, or choose a row_by/col_by pair that tells them apart."
            )
        seen.add(key)
        matrix[row_index[key[0]], col_index[key[1]]] = r.get(name, np.nan)
    return row_levels, col_levels, np.ma.masked_invalid(matrix)


def _shared_standard_name(recs: list[dict[str, Any]]) -> str | None:
    """The one ``variable`` every record shares, or ``None`` if there is more than one.

    Passed to :func:`~ocean_skill.colormaps.metric_colors` so a single-variable grid's
    colour follows that variable's own palette (mirrors
    :func:`ocean_skill.plot.summary._shared_units`'s reasoning); a grid spanning
    several variables has no one palette to prefer, and gets the metric's own default
    instead.
    """
    variables = {r.get("variable") for r in recs if r.get("variable")}
    return variables.pop() if len(variables) == 1 else None


def _fmt_value(name: str, val) -> str:
    """Format one cell's value for annotation — one source so both renderers agree.

    A dimensionless metric (``corr``, ``sigma_ratio``, units ``"1"``) gets two decimal
    places; a count gets a plain integer; everything else (units ``"same"`` as the
    variable, or an unregistered custom metric) gets three significant figures. A
    masked or non-finite cell has nothing to write.
    """
    from ocean_skill.metrics import REGISTRY

    if val is None or np.ma.is_masked(val) or not np.isfinite(val):
        return ""
    metric = REGISTRY.get(name)
    units = metric.units if metric is not None else None
    if units == "count":
        return f"{round(float(val))}"
    if units == "1":
        return f"{float(val):.2f}"
    return f"{float(val):.3g}"


def portrait(
    comparisons,
    *,
    row_by: str = "variable",
    col_by: str = "test",
    metric_names: str | tuple[str, ...] = DEFAULT_MAP_METRICS,
    annotate: bool = False,
    missing_color: str = "#e6e6e6",
    groups: dict[str, Any] | None = None,
    title: str | None = None,
    ncols: int | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    annot_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    save: str | Path | None = None,
    hover: bool | None = None,
    rasterize: bool | str | None = None,
):
    """Portrait plot: a heatmap scoreboard, one cell per ``(row_by, col_by)`` pair.

    ``row_by``/``col_by`` name fields of the metric records (``variable``, ``depth``,
    ``test``, ``reference``, ...) — the defaults (``variable`` down, ``test`` across)
    answer the canonical scoreboard question, "which variable does which run get
    right". Any other field works the same way; ``groups`` (a ``{reference_name:
    label}`` remap, exactly as in :mod:`ocean_skill.plot.summary`) adds one that isn't
    already in the record, for splitting on something like region.

    ``metric_names`` picks and orders the panels: a single string draws one grid, a
    tuple draws that many as small multiples (the same layout
    :func:`~ocean_skill.plot.matplotlib_renderer.skill_map` uses for its map panels) —
    the default is the four canonical statistics Taylor and Target are themselves
    built from (see :data:`ocean_skill.metrics.DEFAULT_MAP_METRICS`). A name none of
    the records carry raises, naming what they do carry.

    Each metric gets its own colour scale, from the same
    :func:`ocean_skill.colormaps.metric_colors` every other metric panel in the
    package calls — diverging about zero for ``bias``, fixed to (-1, 1) for ``corr``,
    centred on 1 for ``sigma_ratio``, sequential from zero for a magnitude — pooled
    over every cell in that metric's grid, with one colorbar per metric (they are
    different quantities, so there is no shared scale to share). A ``(row_by,
    col_by)`` pair no comparison names is drawn in ``missing_color`` — "no data", not
    a zero — and so is a cell whose metric came back non-finite (too few valid pairs
    for a correlation, say).

    ``annotate=True`` writes each cell's own value on top of its colour, formatted by
    a single shared rule so a number reads the same whichever renderer drew it (see
    :func:`_fmt_value`).

    ``hover``/``rasterize`` are accepted only so ``renderer="both"`` can pass one
    option set to each renderer — see
    :func:`ocean_skill.plot.matplotlib_renderer._warn_if_interactive_only` — and do
    nothing here; the interactive renderer's hover tooltip carries the full metric
    record already, with no extra option needed to ask for it.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.colormaps import metric_colors
    from ocean_skill.plot.matplotlib_renderer import (
        _align_colorbars,
        _draw_colorbar,
        _merged,
        _style_defaults,
        _warn_if_interactive_only,
        metric_panel_titles,
    )
    from ocean_skill.plot.summary import _records, pretty_level
    from ocean_skill.plot.typography import (
        facet_figsize,
        facet_layout,
        resolve_canvas,
        type_scale,
    )

    _warn_if_interactive_only(rasterize, hover)
    recs = _records(comparisons, groups)
    if not recs:
        raise ValueError("portrait needs at least one comparison, got none")
    names = _resolve_metric_names(recs, metric_names)
    grids = {name: _grid(recs, row_by, col_by, name) for name in names}
    standard_name = _shared_standard_name(recs) if row_by == "variable" else None
    titles = metric_panel_titles(names)

    # every grid shares one (row_by, col_by) pair, built from the same records, so one
    # cell aspect -- and therefore one panel layout -- serves every metric
    row_levels, col_levels, _ = grids[names[0]]
    aspect = len(col_levels) / max(len(row_levels), 1)
    canvas = resolve_canvas(size, zoom)
    if ncols is None:
        ncols, nrows = facet_layout(len(names), aspect, canvas=canvas)
    else:
        ncols = max(int(ncols), 1)
        nrows = -(-len(names) // ncols)
    figsize = facet_figsize(
        aspect, nrows=nrows, ncols=ncols, canvas=canvas, font_scale=font_scale
    )
    scale = type_scale(figsize, ncols=ncols, nrows=nrows, font_scale=font_scale)
    defaults = _style_defaults(scale, horizontal_colorbar=False)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True
    )
    flat_axes = axes.ravel()
    for ax in flat_axes[len(names) :]:
        ax.set_visible(False)

    merged_tick = _merged(defaults["tick_label_kwargs"], tick_label_kwargs)
    # the column labels live above the axes too (see xaxis_location below), rotated
    # 45deg, so the title needs real clearance above them or the two collide -- a
    # fixed multiple of the tick font rather than a true measured fit (skill_map's
    # _fit_text_widths pass is row-label/map-specific and does not apply to plain
    # tick labels), generous enough for a short-to-medium column name.
    title_pad = merged_tick.get("size", scale["tick_label"]) * 3.2
    merged_title = _merged({**defaults["title_kwargs"], "pad": title_pad}, title_kwargs)
    merged_annot = _merged({"fontsize": scale["metrics"]}, annot_kwargs)
    for ax, name, panel_title in zip(flat_axes, names, titles, strict=False):
        row_levels, col_levels, matrix = grids[name]
        colors = metric_colors(name, matrix.compressed(), standard_name=standard_name)
        ax.set_facecolor(missing_color)
        im = ax.imshow(matrix, cmap=colors.cmap, norm=colors.norm(), aspect="auto")
        ax.set_xticks(range(len(col_levels)))
        ax.set_xticklabels(
            [pretty_level(col_by, v) for v in col_levels], **merged_tick
        )
        ax.set_yticks(range(len(row_levels)))
        ax.set_yticklabels(
            [pretty_level(row_by, v) for v in row_levels], **merged_tick
        )
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        ax.set_title(panel_title, **merged_title)
        if annotate:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.ma.is_masked(matrix[i, j]):
                        continue
                    ax.text(
                        j,
                        i,
                        _fmt_value(name, matrix[i, j]),
                        ha="center",
                        va="center",
                        **merged_annot,
                    )
        _draw_colorbar(
            fig, im, ax, panel_title, colorbar_kwargs, defaults["colorbar_kwargs"]
        )

    _align_colorbars(fig)
    if title:
        fig.suptitle(title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs))
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
