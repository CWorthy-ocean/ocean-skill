"""Summary diagrams: one point per comparison, for a whole :class:`ComparisonSet`.

Where a field row shows *one* comparison in detail, these compress each comparison to a
single point so a suite can be read at a glance. The two are complementary and are best
read together:

**Taylor** (Taylor 2001) places correlation on the angle and standard deviation on the
radius, so distance from the reference point is the centred RMSD. It shows how well the
*pattern* matches — and is blind to bias: a model uniformly 5 degrees too warm plots
perfectly.

**Target** (Jolliff et al. 2009) plots bias against *signed* centred RMSD, both
normalized by the reference standard deviation, so distance from the origin is the
normalized total RMSD. It shows bias, which Taylor omits, and the sign of the x axis
distinguishes an under-dispersed model (left) from an over-dispersed one (right).
Inside the unit circle the model beats the observed mean as a predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ocean_skill.plot.typography import (
    MIN_PT,
    PAGE_W,
    diagram_scale_factor,
    type_scale,
)

__all__ = ["paired", "target", "taylor"]

#: Default figure sizes: both diagrams are square (a Taylor diagram's radius is a
#: standard deviation and a Target's guide rings must stay circular, so neither survives
#: a non-square canvas), and ``paired`` sets them side by side across the page.
TAYLOR_FIGSIZE = (PAGE_W * 0.62, PAGE_W * 0.62)
TARGET_FIGSIZE = (PAGE_W * 0.55, PAGE_W * 0.55)
PAIRED_FIGSIZE = (PAGE_W, PAGE_W * 0.5)


def _scale(
    figsize,
    *,
    ncols: int = 1,
    font_scale: float = 1.0,
    override: dict[str, float] | None = None,
) -> dict[str, float]:
    """Type scale for a diagram of this size — see :mod:`ocean_skill.plot.typography`.

    These diagrams carry seven text sizes of their own (title, axis labels, tick labels,
    legend, point annotations, contour labels, the ring labels), which were seven more
    constants tuned for one figure size. Same table as the field renderers use, so a
    Taylor diagram and the field row beside it in a report agree about how big a label
    is when both are drawn at the same size.

    ``override`` merges *onto* the computed scale rather than replacing it, so a caller
    can name one role and leave the rest alone. These functions have no ``*_kwargs``
    dicts, which makes this their only per-role override — it has to accept a partial
    dict to be usable at all.
    """
    scale = type_scale(figsize, ncols=ncols, nrows=1, font_scale=font_scale)
    return {**scale, **(override or {})}


def _diagram_figsize(default, *, size=None, zoom: float = 1.0):
    """Scale a diagram's default figure size onto the canvas ``size``/``zoom`` name.

    These figures have fixed proportions — Taylor and Target are square because a radius
    is a standard deviation and the rings must stay circular, and ``paired`` is the two
    side by side — so they *fit inside* a canvas rather than taking its shape. See
    :func:`~ocean_skill.plot.typography.diagram_scale_factor`; ``size="page"`` leaves
    the default untouched.
    """
    factor = diagram_scale_factor(default, size=size, zoom=zoom)
    return (default[0] * factor, default[1] * factor)


def _records(comparisons) -> list[dict[str, Any]]:
    """Metric records plus a short label for each comparison."""
    out = []
    for c in comparisons:
        rec = dict(c.metrics())
        rec["label"] = getattr(c, "label", None) or rec.get("variable", "")
        out.append(rec)
    return out


#: Label offsets (display points) cycled by index so consecutive labels point in
#: different directions. Matplotlib has no label-repel, and target points cluster near
#: the origin precisely when a model is good — when labels most need to stay legible.
_LABEL_OFFSETS = ((9, 6), (9, -12), (-9, 6), (-9, -12), (0, 12), (0, -15))


def _offset_labels(ax, xs, ys, labels, size, colors=None):
    """Annotate points, cycling offset directions to avoid overlapping labels."""
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels, strict=True)):
        dx, dy = _LABEL_OFFSETS[i % len(_LABEL_OFFSETS)]
        ax.annotate(
            lab,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=size,
            ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
            color=colors[i] if colors is not None else "0.15",
        )


#: Marker cycle used when ``marker_by`` splits the set into groups (models, runs, ...).
#: Shape carries one dimension and colour another, so a single diagram can show, say,
#: three models across six regions without becoming unreadable.
_MARKERS = ("o", "^", "*", "s", "D", "v", "P", "X")


def pretty_level(field, value) -> str:
    """Legend text for one level: short variable names, units on depths.

    Module level and public to the package because the interactive renderer builds its
    own legend and has to spell the levels identically — a legend reading
    ``sea_water_temperature`` in one renderer and ``temperature`` in the other is the
    same plot disagreeing with itself.
    """
    from ocean_skill.vars import short_name

    if field == "variable":
        return short_name(str(value))
    if field == "depth":
        return f"{value:g} m" if isinstance(value, int | float) else str(value)
    return str(value)


def _group_styles(recs, color_by=None, marker_by=None, colors=None):
    """Assign a colour and marker to every record; return styles plus legend handles.

    ``color_by``/``marker_by`` name any field present in the metric records
    (``variable``, ``depth``, ``test``, ``reference``, ...). Colour defaults to one per
    point when no field is given, which is right for a small fan-out.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("tab10")
    _pretty = pretty_level

    def _levels(field):
        seen = []
        for r in recs:
            v = r.get(field)
            if v not in seen:
                seen.append(v)
        return seen

    if marker_by:
        mlevels = _levels(marker_by)
        marks = [
            _MARKERS[mlevels.index(r.get(marker_by)) % len(_MARKERS)] for r in recs
        ]
    else:
        mlevels, marks = [], ["o"] * len(recs)

    if color_by:
        levels = _levels(color_by)
        cols = [cmap(levels.index(r.get(color_by)) % 10) for r in recs]
    elif colors:
        cols = colors
    elif marker_by:
        # Colour follows the marker groups so the legend's swatches match the points.
        # Colouring per point here instead would vary colour with nothing explaining it,
        # against a legend whose entries are one per *marker* level.
        cols = [cmap(mlevels.index(r.get(marker_by)) % 10) for r in recs]
    else:
        cols = [cmap(i % 10) for i in range(len(recs))]

    handles = []
    if color_by:
        for lev in _levels(color_by):
            c = cmap(_levels(color_by).index(lev) % 10)
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="",
                    marker="o",
                    mfc=c,
                    mec=c,
                    ms=7,
                    label=_pretty(color_by, lev),
                )
            )
    if marker_by:
        for i, lev in enumerate(mlevels):
            m = _MARKERS[i % len(_MARKERS)]
            # Grey only when colour already carries its own dimension; otherwise the
            # swatch takes the group's actual colour so it matches the points.
            c = "0.35" if color_by else cmap(i % 10)
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="",
                    marker=m,
                    mfc=c,
                    mec=c,
                    ms=7,
                    label=_pretty(marker_by, lev),
                )
            )
    if not handles:
        # No grouping field named: one entry per comparison, each drawn exactly as its
        # own point. Without this a legend would only ever be possible when the caller
        # happened to pass color_by/marker_by, which is the common case for a small set.
        handles = [
            Line2D([], [], ls="", marker=mk, mfc=c, mec=c, ms=7, label=r["label"])
            for r, c, mk in zip(recs, cols, marks, strict=True)
        ]
    return cols, marks, handles


#: How a diagram identifies its points. ``"legend"`` puts a key below the axes,
#: ``"annotate"`` writes each label beside its marker, ``None`` does neither.
LABEL_MODES = ("legend", "annotate")


def _resolve_labels(labels):
    """Normalize the ``labels`` argument, rejecting typos loudly."""
    if labels is None or labels is False or labels == "none":
        return None
    if labels not in LABEL_MODES:
        raise ValueError(
            f"labels={labels!r} is not one of {LABEL_MODES} or None — "
            "'legend' keys the points below the axes, 'annotate' writes each "
            "label beside its marker."
        )
    return labels


def _reference_handle():
    """Legend entry for the reference point, drawn as a black star on both diagrams."""
    from matplotlib.lines import Line2D

    return Line2D([], [], ls="", marker="*", mfc="k", mec="k", ms=9, label="reference")


#: Most columns a key beneath the axes is laid out in before it starts wrapping.
_LEGEND_MAX_COLS = 5


def _fit_text(fig) -> None:
    """Shrink any label still too long for its own axes — the field renderer's pass.

    Imported here rather than duplicated so the two renderers cannot disagree about what
    counts as overflowing. Deferred because ``matplotlib_renderer`` reaches back into
    module from ``render()``; a module-level import either way round would be circular.
    """
    from ocean_skill.plot.matplotlib_renderer import _fit_text_widths

    _fit_text_widths(fig)


#: Air (display points) left between the key and the lowest thing above it.
_LEGEND_PAD = 6.0


def _lowest_artist_bottom(fig, renderer) -> float:
    """Display-y of the bottom of the lowest axis label / tick label in the figure.

    What the key has to clear is not the axes box — it is whatever hangs below it, which
    is the x tick labels and then the x axis label, both of whose heights are set by the
    type scale and so move when the level does.
    """
    bottoms = []
    for ax in fig.axes:
        bottoms.append(ax.get_window_extent(renderer).y0)
        texts = [ax.xaxis.label, *ax.get_xticklabels()]
        bottoms += [
            t.get_window_extent(renderer).y0
            for t in texts
            if t.get_visible() and t.get_text()
        ]
    return min(bottoms) if bottoms else 0.0


def _legend_below(fig, handles, size):
    """Draw one key beneath the axes, clear of them, in as many columns as fit.

    Below rather than inside because these diagrams put data in every corner — Taylor
    fills the upper right at high correlation, and target points scatter around the
    origin — so any in-axes placement collides with the data for some input.

    **Vertical placement is measured, not assumed.** This used to sit at a fixed
    ``y=-0.04`` in figure fractions while the x-axis labels it has to clear are a fixed
    *height*, so raising the type level walked the key straight up into them — the same
    bug, and the same fix, as the row label that used to sit at a constant ``x=-0.18``
    (see ``matplotlib_renderer._clear_row_labels``). Placing it below the lowest drawn
    label instead holds at any level.

    **Width is bought back with columns before font size.** Five columns is a lot, and
    the entries are whatever the caller's comparisons happen to be called, which no type
    scale can know: ten labels like ``ROMS-GOM-hindcast-run00-nitrate`` come to half
    again the figure's width. Dropping the column count keeps every entry readable;
    only if a single column is still too wide does the text shrink, floored at
    ``MIN_PT``.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    pad_px = _LEGEND_PAD * fig.dpi / 72.0
    # figure fraction of the top of the key: just below everything already drawn
    top = (_lowest_artist_bottom(fig, renderer) - pad_px) / fig_h_px

    def draw(ncol):
        return fig.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc="upper center",
            ncol=ncol,
            fontsize=size,
            frameon=False,
            numpoints=1,
            bbox_to_anchor=(0.5, top),
        )

    limit = fig.get_size_inches()[0] * fig.dpi
    legend, width = None, 0.0
    for ncol in range(min(len(handles), _LEGEND_MAX_COLS), 0, -1):
        if legend is not None:
            legend.remove()
        legend = draw(ncol)
        fig.canvas.draw()
        width = legend.get_window_extent(fig.canvas.get_renderer()).width
        if width <= limit or ncol == 1:
            break
    if legend is not None and width > limit > 0:
        shrunk = max(size * limit / width, MIN_PT)
        for text in legend.get_texts():
            text.set_fontsize(shrunk)
    return legend


def taylor(
    comparisons,
    *,
    title: str | None = None,
    normalize: bool = True,
    save: str | Path | None = None,
    colors=None,
    color_by: str | None = None,
    marker_by: str | None = None,
    fig=None,
    rect: int = 111,
    labels: str | None = "legend",
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    scale: dict[str, float] | None = None,
):
    """Taylor diagram with one point per comparison.

    ``normalize=True`` divides each standard deviation by its own reference, so
    comparisons in different units (or of different variables) share one diagram; the
    reference then sits at radius 1. Turn it off only when every comparison shares a
    reference and you want the native units.

    ``color_by``/``marker_by`` name a field of the metric records (``variable``,
    ``depth``, ``test``, ...) so several groups can share one diagram — colour for one
    dimension, marker shape for another (three models across six regions, say).

    ``labels`` chooses how points are identified: ``"legend"`` (a key below the axes)
    or ``"annotate"`` (each label written beside its marker); ``None`` for neither.
    Annotation is the better choice for a handful of points, a legend once there are
    enough that the labels would collide.

    Text sizes follow the figure size rather than being fixed, so a diagram drawn at
    twice the default is not a diagram with half-size labels; ``font_scale`` multiplies
    them all. ``scale`` takes a ready-made :func:`_scale` result, which is how
    :func:`paired` gives both of its panels the sizes of the figure they *share* rather
    than the sizes each would pick alone.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot._taylor import TaylorDiagram

    labels = _resolve_labels(labels)
    recs = _records(comparisons)
    if not recs:
        raise ValueError("no comparisons to plot")

    refstd = 1.0 if normalize else recs[0]["std_reference"]
    figsize = figsize or _diagram_figsize(TAYLOR_FIGSIZE, size=size, zoom=zoom)
    scale = _scale(figsize, font_scale=font_scale, override=scale)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    stds = [
        (r["std_test"] / r["std_reference"]) if normalize else r["std_test"]
        for r in recs
    ]
    dia = TaylorDiagram(
        refstd,
        fig=fig,
        rect=rect,
        label="reference",
        srange=(0, max(1.6, 1.15 * max(stds))),
    )
    cols, marks, handles = _group_styles(recs, color_by, marker_by, colors)

    for rec, sd, col, mk in zip(recs, stds, cols, marks, strict=True):
        dia.add_sample(
            sd,
            rec["corr"],
            marker=mk,
            ms=9,
            ls="",
            mfc=col,
            mec=col,
            label=rec["label"],
        )
    contours = dia.add_contours(levels=5, colors="0.6", linewidths=0.7)
    plt.clabel(contours, inline=1, fontsize=scale["contour_label"], fmt="%.2f")
    dia.add_grid(color="0.85", linewidth=0.5)
    dia._ax.axis[:].major_ticks.set_tick_out(True)
    for axis in dia._ax.axis.values():
        # the floating polar axes' own correlation/stddev labelling, which mpl_toolkits
        # leaves at the rcParams default — the one part of this diagram that would still
        # be sized independently of the rest of it
        axis.major_ticklabels.set_fontsize(scale["tick_label"])
        axis.label.set_fontsize(scale["axes_label"])

    if labels == "legend":
        _legend_below(fig, [*handles, _reference_handle()], scale["legend"])
    elif labels == "annotate":
        # The aux axes are polar: a sample sits at (arccos(corr), stddev), which is
        # exactly where add_sample put it.
        _offset_labels(
            dia.ax,
            [np.arccos(r["corr"]) for r in recs],
            stds,
            [r["label"] for r in recs],
            scale["annotation"],
            colors=cols,
        )
    if title:
        dia._ax.set_title(title, fontsize=scale["title"], pad=18)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def target(
    comparisons,
    *,
    title: str | None = None,
    save: str | Path | None = None,
    colors=None,
    color_by: str | None = None,
    marker_by: str | None = None,
    circles=(0.5, 1.0),
    ax=None,
    labels: str | None = "annotate",
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    scale: dict[str, float] | None = None,
):
    """Target diagram (Jolliff et al. 2009) with one point per comparison.

    x is the centred RMSD **signed** by ``sign(std_test − std_reference)`` — negative
    means the model is under-dispersed — and y is the bias; both normalized by the
    reference standard deviation. Distance from the origin is the normalized total RMSD,
    so points inside the unit circle out-perform the observed mean as a predictor.

    ``labels`` chooses how points are identified — ``"legend"`` below the axes or
    ``"annotate"`` beside each marker — exactly as for :func:`taylor`, so the two can be
    made to match. It defaults to ``"annotate"`` here because target points cluster near
    the origin when a model is good, and a label beside the marker stays readable there.

    ``font_scale``/``scale`` size the text from the figure, as in :func:`taylor`.
    """
    import matplotlib.pyplot as plt

    labels_mode = _resolve_labels(labels)
    recs = _records(comparisons)
    if not recs:
        raise ValueError("no comparisons to plot")

    sref = np.array([r["std_reference"] for r in recs])
    x = np.array([r["crmsd"] for r in recs]) / sref
    x *= np.sign(np.array([r["std_test"] for r in recs]) - sref)  # the signed part
    y = np.array([r["bias"] for r in recs]) / sref
    point_labels = [r["label"] for r in recs]

    lim = max(1.15 * float(np.max(np.hypot(x, y))), max(circles) * 1.25, 1.2)
    figsize = figsize or _diagram_figsize(TARGET_FIGSIZE, size=size, zoom=zoom)
    scale = _scale(figsize, font_scale=font_scale, override=scale)
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        # drawn into someone else's figure (paired); they fit the text once at the end
        fig = ax.figure

    for radius in circles:
        ax.add_patch(
            plt.Circle(
                (0, 0),
                radius,
                fill=False,
                color="0.55",
                ls="--" if radius == 1.0 else ":",
                lw=0.9,
                zorder=1,
            )
        )
        ax.annotate(
            f"{radius:g}",
            (radius * 0.71, radius * 0.71),
            fontsize=scale["contour_label"],
            color="0.45",
            ha="left",
            va="bottom",
        )
    ax.axhline(0, color="0.7", lw=0.7, zorder=1)
    ax.axvline(0, color="0.7", lw=0.7, zorder=1)
    ax.plot(0, 0, marker="*", ms=11, color="k", zorder=3, label="reference")

    cols, marks, handles = _group_styles(recs, color_by, marker_by, colors)
    for xi, yi, ci, mi in zip(x, y, cols, marks, strict=True):
        ax.scatter(
            xi,
            yi,
            s=70,
            color=ci,
            marker=mi,
            zorder=4,
            edgecolor="white",
            linewidth=0.6,
        )
    # Axes labelling before the key: _legend_below places itself below the lowest label
    # already drawn, so the x label has to exist by then or the key lands on top of it.
    ax.set(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        xlabel="signed centred RMSD / $\\sigma_{ref}$  (← under | over →)",
        ylabel="bias / $\\sigma_{ref}$",
    )
    ax.xaxis.label.set_size(scale["axes_label"])
    ax.yaxis.label.set_size(scale["axes_label"])
    ax.tick_params(labelsize=scale["tick_label"])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=scale["title"])

    if labels_mode == "annotate":
        _offset_labels(ax, x, y, point_labels, scale["annotation"], colors=cols)
    elif labels_mode == "legend":
        _legend_below(fig, [*handles, _reference_handle()], scale["legend"])

    if owns_figure:
        # That x label is longer than the axes it belongs to at any generous type level,
        # and it is one particular string rather than a scale problem — so it gets the
        # same measure-and-shrink treatment the field renderer gives a long panel title.
        _fit_text(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def paired(
    comparisons,
    *,
    title: str | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    labels: str | None = "legend",
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    scale: dict[str, float] | None = None,
    **kwargs,
):
    """Taylor and Target side by side — the pairing they are usually read as.

    Pure composition: :func:`taylor` draws into the left half of the figure and
    :func:`target` into the right, both accepting the same ``color_by``/``marker_by``
    style arguments, so the two panels stay visually consistent. Neither function knows
    about the other.

    ``labels`` applies to **both** panels, since the diagrams show the same points and
    identifying them two different ways in one figure reads as two unrelated plots. With
    ``"legend"`` the key is drawn once beneath both panels rather than twice.

    One type scale is computed here for the two-column figure and handed to both panels,
    rather than each sizing itself: called alone they are square and near page width, so
    left to themselves they would pick the type for a figure twice the size of the half
    they actually get, and the pair would read as two plots pasted together. ``scale``
    overrides individual roles in it, as for :func:`taylor` — declared here rather than
    left to ``**kwargs`` because the panels are handed this figure's scale explicitly,
    which a forwarded ``scale=`` would collide with.
    """
    import matplotlib.pyplot as plt

    labels = _resolve_labels(labels)
    figsize = figsize or _diagram_figsize(PAIRED_FIGSIZE, size=size, zoom=zoom)
    scale = _scale(figsize, ncols=2, font_scale=font_scale, override=scale)
    fig = plt.figure(figsize=figsize)
    # Panels never draw their own key: with "legend" it is shared (below), and with
    # "annotate" each panel labels its own markers.
    panel_labels = "annotate" if labels == "annotate" else None
    taylor(
        comparisons,
        fig=fig,
        rect=121,
        labels=panel_labels,
        title="Taylor",
        scale=scale,
        **kwargs,
    )
    ax_t = fig.add_subplot(122)
    target(
        comparisons,
        ax=ax_t,
        labels=panel_labels,
        title="Target",
        scale=scale,
        **kwargs,
    )

    if title:
        fig.suptitle(title, fontsize=scale["suptitle"])
    # Both of these move the axes, so they come before the key, which is placed by
    # measuring where the axes and their labels actually ended up.
    fig.subplots_adjust(wspace=0.35)
    if labels == "legend":
        # One shared key beneath both panels, so it cannot collide with either title
        recs = _records(comparisons)
        _, _, handles = _group_styles(
            recs, kwargs.get("color_by"), kwargs.get("marker_by")
        )
        _legend_below(fig, [*handles, _reference_handle()], scale["legend"])
    _fit_text(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
