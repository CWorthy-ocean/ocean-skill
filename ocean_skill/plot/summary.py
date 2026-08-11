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

__all__ = ["paired", "target", "taylor"]

PAGE_W = 8.5


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


def _offset_labels(ax, xs, ys, labels, size=6.5, colors=None):
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


def _legend_below(fig, handles, y=-0.04):
    """Draw one key beneath the axes.

    Below rather than inside because these diagrams put data in every corner — Taylor
    fills the upper right at high correlation, and target points scatter around the
    origin — so any in-axes placement collides with the data for some input.
    """
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        ncol=min(len(handles), 5),
        fontsize=7,
        frameon=False,
        numpoints=1,
        bbox_to_anchor=(0.5, y),
    )


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
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot._taylor import TaylorDiagram

    labels = _resolve_labels(labels)
    recs = _records(comparisons)
    if not recs:
        raise ValueError("no comparisons to plot")

    refstd = 1.0 if normalize else recs[0]["std_reference"]
    if fig is None:
        fig = plt.figure(figsize=figsize or (PAGE_W * 0.62, PAGE_W * 0.62))
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
    plt.clabel(contours, inline=1, fontsize=6, fmt="%.2f")
    dia.add_grid(color="0.85", linewidth=0.5)
    dia._ax.axis[:].major_ticks.set_tick_out(True)

    if labels == "legend":
        _legend_below(fig, [*handles, _reference_handle()])
    elif labels == "annotate":
        # The aux axes are polar: a sample sits at (arccos(corr), stddev), which is
        # exactly where add_sample put it.
        _offset_labels(
            dia.ax,
            [np.arccos(r["corr"]) for r in recs],
            stds,
            [r["label"] for r in recs],
            colors=cols,
        )
    if title:
        dia._ax.set_title(title, fontsize=9, pad=18)
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
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (PAGE_W * 0.55, PAGE_W * 0.55), constrained_layout=True
        )
    else:
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
            fontsize=6,
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
    if labels_mode == "annotate":
        _offset_labels(ax, x, y, point_labels, colors=cols)
    elif labels_mode == "legend":
        _legend_below(fig, [*handles, _reference_handle()])

    ax.set(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        xlabel="signed centred RMSD / $\\sigma_{ref}$  (← under | over →)",
        ylabel="bias / $\\sigma_{ref}$",
    )
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10)
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
    """
    import matplotlib.pyplot as plt

    labels = _resolve_labels(labels)
    fig = plt.figure(figsize=figsize or (PAGE_W, PAGE_W * 0.5))
    # Panels never draw their own key: with "legend" it is shared (below), and with
    # "annotate" each panel labels its own markers.
    panel_labels = "annotate" if labels == "annotate" else None
    taylor(
        comparisons, fig=fig, rect=121, labels=panel_labels, title="Taylor", **kwargs
    )
    ax_t = fig.add_subplot(122)
    target(comparisons, ax=ax_t, labels=panel_labels, title="Target", **kwargs)

    if labels == "legend":
        # One shared key beneath both panels, so it cannot collide with either title
        recs = _records(comparisons)
        _, _, handles = _group_styles(
            recs, kwargs.get("color_by"), kwargs.get("marker_by")
        )
        _legend_below(fig, [*handles, _reference_handle()])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(wspace=0.35)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
