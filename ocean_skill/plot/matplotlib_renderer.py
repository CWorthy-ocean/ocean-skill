"""Static matplotlib renderer (PNG/JPG/PDF, and mp4/gif via FuncAnimation).

Currently implements the **field row**: ``test | reference | difference`` maps for a
gridded comparison. Test and reference share one colour scale (so they are visually
comparable) taken from the 10th–90th percentile of the pair; the difference panel uses
a diverging map centred on zero. Metrics go in a corner box, leaving the title for
identity. Registers itself under ``"matplotlib"``.

Every family here draws its panels through :func:`_draw_map`, and the two movie
families are the two static map families played rather than laid out:
:func:`field_movie` animates :func:`field_grid`'s comparison rows, :func:`facet_movie`
animates :func:`field_facet`'s facet axis. So a frame of a movie is the still it would
have been.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ocean_skill import _stacklevel
from ocean_skill.colormaps import cmaps_for, norm_for
from ocean_skill.plot.registry import register_renderer
from ocean_skill.plot.typography import (
    MIN_PT,
    PAGE_H,
    PAGE_W,
    REFERENCE_GRID,
    auto_figsize,
    reference_scale,
    type_scale,
)

# aliased: field_grid already has a row_height *parameter*, which is the caller's
# override of exactly this
from ocean_skill.plot.typography import row_height as _typographic_row_height

__all__ = ["facet_labels", "field_facet", "field_grid", "field_row", "render"]

# PAGE_W/PAGE_H (the portrait page every figure has to fit) now live in typography,
# which is where they are used to decide sizes; re-exported under their old names.
__all__ += ["PAGE_H", "PAGE_W"]


def _limits(*arrays, lo: float = 10, hi: float = 90) -> tuple[float, float]:
    """Shared colour limits from percentiles across all arrays (robust to outliers)."""
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def _contour_levels(norm, n: int = 21):
    """Level edges matching ``norm`` — geometrically spaced under a ``LogNorm``."""
    import matplotlib.colors as mcolors

    if isinstance(norm, mcolors.LogNorm):
        return np.geomspace(norm.vmin, norm.vmax, n)
    return np.linspace(norm.vmin, norm.vmax, n)


#: Metric keys shown in the corner box by default — the first three of the set
#: metrics.compute() returns. Pass metric_keys=(...) to show a different subset/order.
DEFAULT_METRIC_KEYS = ("bias", "rmse", "corr")

# Every *_kwargs parameter below merges with these defaults rather than replacing them
# wholesale — passing e.g. title_kwargs={"fontsize": 10} doesn't require also
# supplying every other title property. Each maps onto exactly one matplotlib/cartopy
# call, so any kwarg that call accepts works, not just a hand-picked subset.
#
# Font *sizes* are deliberately absent from these dicts: they come from
# ocean_skill.plot.typography, which derives all of them from the figure's own geometry
# so that changing figsize (or the row count) does not silently leave eleven hand-tuned
# point sizes wrong. _style_defaults() below folds the two together. Everything that is
# genuinely a fixed choice — a colour, a line width, a colorbar's orientation — stays
# here as a literal.
#: ``y`` is pinned rather than left to matplotlib's automatic title placement, which
#: is broken over a cartopy GeoAxES carrying gridline labels: matplotlib 3.11 places
#: the title above the union of the axes' children's bboxes, cartopy contributes an
#: empty ``(inf, inf, -inf, -inf)`` one, and the title's y comes out infinite. Its
#: window extent is then NaN, which makes the whole *axes* report a NaN tight bbox,
#: which drops that axes out of the figure's tight bbox — so ``bbox_inches="tight"``,
#: used both by our own ``save=`` and by Jupyter's inline backend, silently crops the
#: leftmost column out of the figure. Supplying any explicit ``y`` skips the automatic
#: placement that computes the infinity. Identical output on 3.10, where it is a no-op.
DEFAULT_TITLE_KWARGS: dict[str, Any] = {"y": 1.0}
DEFAULT_GRIDLINE_KWARGS: dict[str, Any] = {
    "linewidth": 0.2,
    "color": "0.6",
    "alpha": 0.6,
}
DEFAULT_TICK_LABEL_KWARGS: dict[str, Any] = {}
DEFAULT_ROW_LABEL_KWARGS: dict[str, Any] = {
    "rotation": 90,
    "va": "center",
    "ha": "center",
    "weight": "normal",
}
DEFAULT_METRICS_KWARGS: dict[str, Any] = {
    "va": "bottom",
    "ha": "left",
    "bbox": {
        "facecolor": "white",
        "alpha": 0.75,
        "pad": 2,
        "edgecolor": "0.6",
        "linewidth": 0.4,
    },
}
#: The movie families only: the per-frame label (usually a timestamp), drawn in the
#: top-left of the panel — mirroring the metrics box in the bottom-left of the
#: difference panel, and the same box styling so the two read as one figure's notes.
#: ``monospace`` deliberately: a proportional font makes a counting timestamp jitter
#: sideways from frame to frame, which is distracting in a way it never is on a still.
DEFAULT_FRAME_LABEL_KWARGS: dict[str, Any] = {
    "va": "top",
    "ha": "left",
    "family": "monospace",
    "bbox": dict(DEFAULT_METRICS_KWARGS["bbox"]),
}
#: field_row draws one row per figure (page-width, horizontal bars below); field_grid
#: stacks several rows (vertical bars beside them) — same keys, different orientation.
#: ``shrink`` is 1.0 in both because :func:`_align_colorbars` re-fits each bar to the
#: panels it belongs to after the layout settles; a shrink below 1 now means "this
#: fraction of the panels' own extent", not of the grid cell they sit in.
DEFAULT_COLORBAR_KWARGS_ROW: dict[str, Any] = {
    "orientation": "horizontal",
    "pad": 0.04,
    "shrink": 1.0,
    "aspect": 30,
}
DEFAULT_COLORBAR_KWARGS_GRID: dict[str, Any] = {
    "orientation": "vertical",
    "pad": 0.015,
    "shrink": 1.0,
    "aspect": 15,
}
DEFAULT_SUPTITLE_KWARGS_ROW: dict[str, Any] = {}
DEFAULT_SUPTITLE_KWARGS_GRID: dict[str, Any] = {}

#: Colorbar aspect (length/thickness) for ``field_facet``'s single shared bar.
#: Both dicts above describe a bar spanning one row, and :func:`_align_colorbars`
#: derives thickness from the bar's own length — so a bar re-fitted to span *every*
#: row of a facet grid comes out as many times fatter as there are rows, and a 0.6in
#: slab down the side of the page reads as a second figure rather than as a scale.
#: Roughly the per-row aspect times the row count these grids typically have.
FACET_COLORBAR_ASPECT = 40

#: Reference sizes, i.e. what a default page-width ``field_row`` gets. There is no
#: literal default dict to point at any more, so this is what the docs quote and what
#: :func:`_nested_owner` matches option names against.
REFERENCE_SCALE: dict[str, float] = reference_scale()

#: Which nested ``*_kwargs`` dict each styling key belongs to, so an option passed one
#: level too high can be pointed at its home rather than just rejected. Built from the
#: non-font defaults plus the font keys, since neither alone lists every valid key.
_NESTED_KWARGS: dict[str, dict[str, Any]] = {
    "colorbar_kwargs": {
        **DEFAULT_COLORBAR_KWARGS_ROW,
        **DEFAULT_COLORBAR_KWARGS_GRID,
        "label_size": REFERENCE_SCALE["colorbar_label"],
        "tick_labelsize": REFERENCE_SCALE["colorbar_tick"],
    },
    "title_kwargs": {**DEFAULT_TITLE_KWARGS, "fontsize": REFERENCE_SCALE["title"]},
    "gridline_kwargs": DEFAULT_GRIDLINE_KWARGS,
    "tick_label_kwargs": {"size": REFERENCE_SCALE["tick_label"]},
    "row_label_kwargs": {
        **DEFAULT_ROW_LABEL_KWARGS,
        "fontsize": REFERENCE_SCALE["row_label"],
    },
    "metrics_kwargs": {
        **DEFAULT_METRICS_KWARGS,
        "fontsize": REFERENCE_SCALE["metrics"],
    },
    "frame_label_kwargs": {
        **DEFAULT_FRAME_LABEL_KWARGS,
        "fontsize": REFERENCE_SCALE["frame_label"],
    },
    "suptitle_kwargs": {"fontsize": REFERENCE_SCALE["suptitle"]},
}


def _scale_for(
    figsize: tuple[float, float], *, nrows: int = 1, font_scale: float = 1.0
) -> dict[str, float]:
    """Type scale for a figure of ``figsize`` holding ``nrows`` rows of three maps."""
    return type_scale(figsize, ncols=3, nrows=nrows, font_scale=font_scale)


def _style_defaults(
    scale: dict[str, float], *, horizontal_colorbar: bool
) -> dict[str, dict[str, Any]]:
    """Build the seven ``*_kwargs`` defaults: fixed choices plus figure-sized type.

    Returned rather than held at module level because half of each dict now depends on
    the figure being drawn. The caller's own ``*_kwargs`` merge on top of these, so an
    explicit ``title_kwargs={"fontsize": 11}`` still wins outright — automatic sizing is
    a better *default*, not a new constraint.
    """
    cbar = dict(
        DEFAULT_COLORBAR_KWARGS_ROW
        if horizontal_colorbar
        else DEFAULT_COLORBAR_KWARGS_GRID
    )
    # tick_labelsize joins label_size here rather than in the literal defaults because
    # it was previously absent altogether: the bar's tick labels fell back to rcParams'
    # 10pt while every other size in the figure was 5-9, which read as a different
    # figure's colorbar pasted on. Every text size in the figure now comes from one
    # scale, so that cannot recur silently.
    cbar["label_size"] = scale["colorbar_label"]
    cbar["tick_labelsize"] = scale["colorbar_tick"]
    return {
        "colorbar_kwargs": cbar,
        "title_kwargs": {**DEFAULT_TITLE_KWARGS, "fontsize": scale["title"]},
        "gridline_kwargs": dict(DEFAULT_GRIDLINE_KWARGS),
        "tick_label_kwargs": {
            **DEFAULT_TICK_LABEL_KWARGS,
            "size": scale["tick_label"],
        },
        "row_label_kwargs": {
            **DEFAULT_ROW_LABEL_KWARGS,
            "fontsize": scale["row_label"],
        },
        "metrics_kwargs": {**DEFAULT_METRICS_KWARGS, "fontsize": scale["metrics"]},
        "frame_label_kwargs": {
            **DEFAULT_FRAME_LABEL_KWARGS,
            "fontsize": scale["frame_label"],
        },
        "suptitle_kwargs": {"fontsize": scale["suptitle"]},
    }


def _merged(
    defaults: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Shallow-merge a ``*_kwargs`` override onto its defaults.

    ``overrides=None`` returns the defaults unchanged.
    """
    return {**defaults, **(overrides or {})}


def _draw_colorbar(
    fig, im, ax, label, colorbar_kwargs: dict[str, Any] | None, defaults
):
    """Draw one colorbar from a single merged kwargs dict, split by key prefix.

    ``label_*`` keys go to ``.set_label()``, ``tick_*`` keys go to
    ``.ax.tick_params()``, everything else goes to ``fig.colorbar()`` itself — one
    parameter for the caller, still three separate matplotlib calls underneath,
    since that's genuinely three different methods with non-overlapping kwargs.
    """
    merged = _merged(defaults, colorbar_kwargs)
    cbar_kw, label_kw, tick_kw = {}, {}, {}
    for k, v in merged.items():
        if k.startswith("label_"):
            label_kw[k.removeprefix("label_")] = v
        elif k.startswith("tick_"):
            tick_kw[k.removeprefix("tick_")] = v
        else:
            cbar_kw[k] = v
    cbar = fig.colorbar(im, ax=ax, **cbar_kw)
    if label:
        cbar.set_label(label, **label_kw)
    if tick_kw:
        cbar.ax.tick_params(**tick_kw)
    # Remember which panels this bar belongs to (and how it is oriented) so
    # _align_colorbars can re-fit it to them once the layout is final.
    cbar.ax._osk_cbar_parents = list(np.atleast_1d(ax).ravel())
    cbar.ax._osk_cbar_horizontal = cbar_kw.get("orientation") == "horizontal"
    cbar.ax._osk_cbar_shrink = float(cbar_kw.get("shrink", 1.0))
    cbar.ax._osk_cbar_aspect = float(cbar_kw.get("aspect", 20))
    return cbar


def _align_colorbars(fig, renderer=None) -> None:
    """Re-fit every colorbar to the drawn extent of the panels it describes.

    ``fig.colorbar(im, ax=...)`` under constrained_layout sizes the bar to the
    *gridspec cell*, which is taller (or wider) than the map inside it: the cell also
    holds the title above and the longitude labels below, and a cartopy GeoAxes has a
    fixed aspect, so the map shrinks inside its own slot as well. A vertical bar then
    overshoots the map at both ends — visibly so, since it is the map's own colour
    scale and reads as its ruler.

    Nothing in the layout engine expresses "match that box", so the fix is to measure
    the panels after the layout has settled and set the bar's long axis to their union:
    top to top, bottom to bottom, ignoring the title and axis labelling. ``shrink``
    still applies, now as a fraction of *that* extent, centred.

    Thickness then comes from ``aspect`` — but from the *shortest* bar's new length, one
    value for every bar of the same orientation in the figure, rather than each bar's
    own. A field row pairs one bar spanning two panels with one spanning a single panel,
    and length/aspect applied bar-by-bar makes the long one two and a half times
    fatter than its neighbour.

    The gap to the panels is levelled the same way, and for the same reason: ``pad`` is
    a fraction of the parent's *own* size along the direction the bar is stolen from, so
    the two-panel bar in a field grid is padded off a span twice as wide as the
    difference bar beside it and ends up with twice the gap. One gap in inches, the
    widest of the group, is used for every bar of that orientation.

    The layout engine is stood down first, otherwise the next draw — including the one
    inside ``savefig`` and Jupyter's inline backend — recomputes the positions and
    undoes this.
    """
    caxes = [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]
    if not caxes:
        return
    if renderer is None:
        fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    # get_position() is the *active* position, so it reflects the shrinking a
    # fixed-aspect GeoAxes does to itself — but only once a draw has applied it.
    fitted = {}
    for cax in caxes:
        boxes = [p.get_position() for p in cax._osk_cbar_parents]
        pos = cax.get_position()
        if cax._osk_cbar_horizontal:
            lo, hi = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
            # both families draw a horizontal bar *below* its panels and a vertical one
            # to their right, so the near edge and the sign of the gap are known
            gap_in = (min(b.y0 for b in boxes) - pos.y1) * fig_h
            near = min(b.y0 for b in boxes)
        else:
            lo, hi = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
            gap_in = (pos.x0 - max(b.x1 for b in boxes)) * fig_w
            near = max(b.x1 for b in boxes)
        mid, half = (lo + hi) / 2.0, (hi - lo) * cax._osk_cbar_shrink / 2.0
        fitted[cax] = (mid - half, mid + half, near, max(gap_in, 0.0))

    # one thickness and one gap (both in inches) per orientation: the thickness of the
    # shortest bar in the group, the gap of the most generously padded
    thickness, gap = {}, {}
    for cax, (lo, hi, _near, gap_in) in fitted.items():
        horiz = cax._osk_cbar_horizontal
        length_in = (hi - lo) * (fig_w if horiz else fig_h)
        thick_in = length_in / max(cax._osk_cbar_aspect, 1e-6)
        thickness[horiz] = min(thickness.get(horiz, thick_in), thick_in)
        gap[horiz] = max(gap.get(horiz, gap_in), gap_in)

    fig.set_layout_engine("none")
    for cax, (lo, hi, near, _gap_in) in fitted.items():
        # fig.colorbar() expresses `aspect` as a *box aspect* on the bar's axes, which
        # would shrink the box we set back down to that ratio (the long axis with it,
        # which is the one thing that must not move). We size both axes here, so the
        # constraint has nothing left to do.
        cax.set_box_aspect(None)
        if cax._osk_cbar_horizontal:
            height = thickness[True] / fig_h
            top = near - gap[True] / fig_h
            cax.set_position([lo, top - height, hi - lo, height])
        else:
            width = thickness[False] / fig_w
            cax.set_position([near + gap[False] / fig_w, lo, width, hi - lo])


def _add_row_label(ax, text: str, row_label_kwargs: dict[str, Any]) -> None:
    """Write a rotated label at the left edge of the row ``ax`` starts.

    The x here is provisional; :func:`_clear_row_labels` moves it once the layout is
    final and the latitude labels it must clear have a measurable width. Deliberately
    NOT ``set_ylabel``: constrained_layout does not see cartopy's gridline labels (they
    are free artists, not ytick labels), so it packs the axes against the ylabel alone
    and the latitude labels land on top of it no matter how large a labelpad is set. A
    free text artist takes no part in the layout, so it can be placed after the fact
    without the layout shifting back underneath it.
    """
    ax._osk_row_label = ax.text(
        -0.18, 0.5, text, transform=ax.transAxes, **row_label_kwargs
    )


def _draw_map(
    ax,
    da,
    *,
    label: str | None,
    cmap,
    norm,
    mark: str,
    domain: tuple[float, float, float, float] | None,
    gridline_kwargs: dict[str, Any],
    tick_label_kwargs: dict[str, Any],
    title_kwargs: dict[str, Any],
    left_labels: bool | None = None,
    bottom_labels: bool | None = None,
):
    """Draw one map panel into ``ax`` and return its mappable.

    Everything every map in this module has in common: the field, the land mask,
    coastlines, gridlines, the optional model-domain outline and the title. Shared by
    :func:`_draw_row` and :func:`field_facet` so that the two families cannot drift
    apart on what a map of this package looks like — a change to the coastline weight
    or the land grey should not be a change to only half the figures.

    ``left_labels``/``bottom_labels`` of ``None`` leave cartopy's ``draw_labels=True``
    default standing, i.e. every panel labels its own axes; ``True``/``False`` set them
    explicitly, which is how a grid shows each axis once.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.PlateCarree()
    draw = getattr(ax, "contourf" if mark == "contourf" else "pcolormesh")
    kw = {"levels": _contour_levels(norm)} if mark == "contourf" else {}
    im = draw(da["lon"], da["lat"], da, transform=proj, cmap=cmap, norm=norm, **kw)
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
    ax.coastlines(linewidth=0.4, zorder=3)
    gl = ax.gridlines(draw_labels=True, **gridline_kwargs)
    gl.top_labels = gl.right_labels = False
    if left_labels is not None:
        gl.left_labels = left_labels
    if bottom_labels is not None:
        gl.bottom_labels = bottom_labels
    gl.xlabel_style = gl.ylabel_style = dict(tick_label_kwargs)
    if domain:
        lo0, la0, lo1, la1 = domain
        ax.plot(
            [lo0, lo1, lo1, lo0, lo0],
            [la0, la0, la1, la1, la0],
            transform=proj,
            color="k",
            lw=0.6,
            ls="--",
            zorder=4,
        )
    if label is not None:
        ax.set_title(label, **title_kwargs)
    return im


def _draw_row(
    axes,
    aligned,
    *,
    test_name: str,
    reference_name: str,
    labels: tuple[str, str],
    units: str | None,
    standard_name: str | None,
    metrics: dict[str, Any] | None,
    mark: str,
    domain: tuple[float, float, float, float] | None,
    row_label: str | None = None,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    title_kwargs: dict[str, Any] | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    row_label_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    seq_norm: Any = None,
    div_norm: Any = None,
    shared_axis_labels: bool = True,
    is_bottom_row: bool = True,
    defaults: dict[str, dict[str, Any]] | None = None,
):
    """Draw one test|reference|difference row into three existing cartopy axes.

    ``seq_norm``/``div_norm``, if given, override the row's own percentile-derived
    colour limits — how :func:`field_grid`'s ``shared_limits=True`` makes every row
    share one scale instead of each computing its own.

    ``shared_axis_labels=True`` (the default) draws grid lines on every panel but
    only draws coordinate *labels* on the leftmost panel (latitude) and, if
    ``is_bottom_row``, every panel (longitude) — the usual convention for a grid of
    maps sharing axes, since three side-by-side copies of the same latitude labels
    say nothing three copies didn't already say once. Set ``False`` to label every
    panel's axes independently, as every version before this one did.

    ``defaults`` is the caller's :func:`_style_defaults` — the font sizes it derived
    from the figure's geometry, merged with the fixed style choices. Passed in rather
    than recomputed because the sizes belong to the whole figure, and a row cannot see
    how many other rows are sharing the page with it.
    """
    import matplotlib.colors as mcolors

    defaults = defaults or _style_defaults(reference_scale(), horizontal_colorbar=True)
    title_kwargs = _merged(defaults["title_kwargs"], title_kwargs)
    gridline_kwargs = _merged(defaults["gridline_kwargs"], gridline_kwargs)
    tick_label_kwargs = _merged(defaults["tick_label_kwargs"], tick_label_kwargs)
    row_label_kwargs = _merged(defaults["row_label_kwargs"], row_label_kwargs)
    metrics_kwargs = _merged(defaults["metrics_kwargs"], metrics_kwargs)

    t, r, d = aligned[test_name], aligned[reference_name], aligned["difference"]
    tl, rl = labels
    seq, div = cmaps_for(standard_name)
    if seq_norm is None:
        vmin, vmax = _limits(t, r)
        seq_norm = norm_for(standard_name, vmin, vmax)
    if div_norm is None:
        dmax = float(np.nanpercentile(np.abs(np.asarray(d)), 98)) or 1.0
        div_norm = mcolors.Normalize(vmin=-dmax, vmax=dmax)

    panels = [
        (t, tl, seq, seq_norm),
        (r, rl, seq, seq_norm),
        (d, "difference", div, div_norm),
    ]
    ims = []
    for j, (ax, (da, lab, cmap, norm)) in enumerate(zip(axes, panels, strict=True)):
        # None leaves every panel labelling its own axes, as every version before
        # shared_axis_labels existed did.
        ims.append(
            _draw_map(
                ax,
                da,
                label=lab,
                cmap=cmap,
                norm=norm,
                mark=mark,
                domain=domain,
                gridline_kwargs=gridline_kwargs,
                tick_label_kwargs=tick_label_kwargs,
                title_kwargs=title_kwargs,
                left_labels=(j == 0) if shared_axis_labels else None,
                bottom_labels=is_bottom_row if shared_axis_labels else None,
            )
        )

    if row_label:
        _add_row_label(axes[0], row_label, row_label_kwargs)
    if metrics:
        # stashed on the axes, as the row label is, so that field_movie can retext it
        # per frame rather than re-deriving where the box was put
        axes[2]._osk_metrics_text = axes[2].text(
            0.02,
            0.02,
            _metrics_text(metrics, metric_keys),
            transform=axes[2].transAxes,
            zorder=5,
            **metrics_kwargs,
        )
    return ims, (f"[{units}]" if units else "")


def _metrics_text(metrics: dict[str, Any] | None, metric_keys) -> str:
    """Return the corner box's text: one ``key=value`` line per requested metric.

    Its own function because :func:`field_movie` rewrites the box every frame and the
    two spellings of "what the box says" must not drift apart.
    """
    if not metrics:
        return ""
    return "\n".join(
        f"{k}={metrics[k]:.3g}"
        for k in metric_keys
        if isinstance(metrics.get(k), int | float)
    )


def field_row(
    aligned,
    *,
    test_name: str = "test",
    reference_name: str = "reference",
    labels: tuple[str, str] | None = None,
    title: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    metrics: dict[str, Any] | None = None,
    mark: str = "pcolormesh",
    save: str | Path | None = None,
    domain: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    shared_axis_labels: bool = True,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
):
    """Draw one ``test | reference | difference`` row for a gridded comparison.

    ``figsize`` defaults to a page-width row as tall as the maps' own aspect ratio
    wants, plus the room the type around them needs — see
    :func:`~ocean_skill.plot.typography.row_height`. Pass your own to override.
    ``metric_keys`` picks which of ``metrics.compute()``'s values appear in the
    corner box (default ``bias``/``rmse``/``corr``) — any subset/order, e.g.
    ``metric_keys=("corr", "sigma_ratio")``.

    Every font size — panel titles, the suptitle, latitude/longitude labels, the
    colorbars' labels and their tick labels, the metrics box — is derived from the
    figure's geometry by :func:`~ocean_skill.plot.typography.type_scale`, so a
    ``figsize`` half the default gets type to match rather than three maps crushed to a
    third of an inch by 8pt titles that no longer fit. ``font_scale`` multiplies all of
    them together (``font_scale=1.2`` for "the same figure, larger type"), keeping their
    proportions; a size passed explicitly in a ``*_kwargs`` dict still overrides
    outright.

    ``shared_axis_labels=True`` (the default) draws grid lines on every panel but
    only labels the leftmost panel's latitude axis, since the other two show the
    same latitudes — longitude is labelled on all three either way, this being the
    only (and so also the bottom) row. Set ``False`` to label every panel fully.

    The ``*_kwargs`` parameters each merge onto their current defaults and map
    straight onto one matplotlib/cartopy call: ``colorbar_kwargs`` ->
    ``fig.colorbar()`` (``label_*``/``tick_*``-prefixed keys split off to
    ``.set_label()``/``.ax.tick_params()`` instead — e.g.
    ``colorbar_kwargs={"shrink": 1.0, "label_size": 9}``), ``title_kwargs`` ->
    ``ax.set_title()``, ``gridline_kwargs`` -> ``ax.gridlines()``,
    ``tick_label_kwargs`` -> the gridliner's ``xlabel_style``/``ylabel_style``,
    ``suptitle_kwargs`` -> ``fig.suptitle()``.

    ``align_colorbars=True`` (the default) re-fits each colorbar to the drawn extent
    of the panels it belongs to once the layout is final, so a horizontal bar spans
    exactly its panels' left and right edges rather than the wider grid cell that also
    holds their titles and axis labels (see :func:`_align_colorbars`). ``shrink`` then
    means a fraction of that extent. Set ``False`` to leave placement entirely to
    ``constrained_layout``.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    # A single row's colorbars are horizontal and sit below the maps, so they come out
    # of the row's height rather than its width — auto_figsize has to be told, and the
    # type scale sized against a cell that already accounts for them.
    figsize = figsize or auto_figsize(
        _map_aspect([{"aligned": aligned}], reference_name),
        nrows=1,
        font_scale=font_scale,
        horizontal_colorbar=True,
    )
    scale = _scale_for(figsize, nrows=1, font_scale=font_scale)
    defaults = _style_defaults(scale, horizontal_colorbar=True)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    ims, lab = _draw_row(
        axes,
        aligned,
        test_name=test_name,
        reference_name=reference_name,
        labels=labels or ("test", "reference"),
        units=units,
        standard_name=standard_name,
        metrics=metrics,
        mark=mark,
        domain=domain,
        metric_keys=metric_keys,
        title_kwargs=title_kwargs,
        gridline_kwargs=gridline_kwargs,
        tick_label_kwargs=tick_label_kwargs,
        metrics_kwargs=metrics_kwargs,
        shared_axis_labels=shared_axis_labels,
        is_bottom_row=True,
        defaults=defaults,
    )
    _draw_colorbar(
        fig, ims[1], axes[:2], lab, colorbar_kwargs, defaults["colorbar_kwargs"]
    )
    _draw_colorbar(
        fig,
        ims[2],
        axes[2],
        f"difference {lab}",
        colorbar_kwargs,
        defaults["colorbar_kwargs"],
    )

    # after the suptitle, so the margin is fitted to the layout the figure ends with
    if title:
        fig.suptitle(title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs))
    _fit_left_margin(fig)
    _fit_text_widths(fig)
    if align_colorbars:
        _align_colorbars(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


#: Air (display points) left between a row label and the latitude labels it clears.
_ROW_LABEL_PAD = 4.0

#: Air (display points) left between the leftmost label and the canvas edge, once
#: :func:`_fit_left_margin` has had to make room.
_LEFT_MARGIN_PAD = 3.0


def _left_label_artists(ax):
    """Every text artist drawn to the left of ``ax``: latitude labels, row label.

    Cartopy exposes a gridliner's labels through ``left_label_artists``; the row
    label is our own free text. Both are what a too-tight left margin eats.
    """
    for artist in ax.artists:
        yield from (
            text
            for text in getattr(artist, "left_label_artists", []) or []
            if text.get_visible() and text.get_text()
        )
    if (label := getattr(ax, "_osk_row_label", None)) is not None:
        yield label


def _clear_row_labels(fig, renderer=None) -> None:
    """Move each row label just left of its own latitude labels.

    The offset cannot be a constant. It was ``x=-0.18`` in *axes fraction*, i.e. a
    share of the panel width, while the latitude labels it has to clear are a fixed
    text width — so it overlapped at every figure size (-4px at 8.5in, worsening to
    -31px at 3.5in). Nor can it be a labelpad on the axes' own ylabel: cartopy draws
    gridline labels as free artists rather than ytick labels, so constrained_layout
    never accounts for them and simply re-packs the axes, preserving the collision.

    Measuring after the layout has settled avoids both traps, and a free text artist
    takes no part in the layout, so nothing shifts back underneath it.
    """
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        label = getattr(ax, "_osk_row_label", None)
        if label is None:
            continue
        lefts = [
            text.get_window_extent(renderer).x0
            for artist in ax.artists
            for text in getattr(artist, "left_label_artists", []) or []
            if text.get_visible() and text.get_text()
        ]
        if not lefts:
            continue
        pad_px = _ROW_LABEL_PAD * fig.dpi / 72.0
        # the label is rotated 90°, so its *width* is what intrudes horizontally
        half_width = label.get_window_extent(renderer).width / 2.0
        x_display = min(lefts) - pad_px - half_width
        x_axes = ax.transAxes.inverted().transform((x_display, 0))[0]
        label.set_position((x_axes, 0.5))


def _fit_left_margin(fig, *, passes: int = 3) -> None:
    """Place the row labels, then widen the left margin until nothing spills off-canvas.

    Whether the left margin is wide enough is not something the layout engine can be
    trusted to know: what hangs off the left of the leftmost panel is cartopy's
    latitude labels and our own row label, neither of which the engine reliably
    counts as taking up room. When it reserves nothing, both are drawn at a negative
    x and clipped off the canvas, leaving the maps themselves looking intact.

    This is a backstop rather than the cure for any particular bug — the matplotlib
    3.11 clipping that prompted it is fixed at its source in ``DEFAULT_TITLE_KWARGS``
    — and it costs nothing when the margin is already adequate, which is the usual
    case: the first pass measures and returns.

    ``bbox_inches="tight"`` is not the escape hatch it looks like: cartopy's gridline
    labels are not in the tight bbox either, so the saved PNG loses them too.

    Measuring the drawn result and pushing constrained_layout's ``rect`` in from the
    left settles it in the one way that holds on either cartopy. Each pass re-places
    the row labels first, since they are positioned relative to the latitude labels
    and so move with the axes; two passes normally converge, and ``passes`` bounds it
    either way.
    """
    engine = fig.get_layout_engine()
    for _ in range(passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        _clear_row_labels(fig, renderer)
        lefts = [
            text.get_window_extent(renderer).x0
            for ax in fig.axes
            for text in _left_label_artists(ax)
        ]
        pad_px = _LEFT_MARGIN_PAD * fig.dpi / 72.0
        if not lefts or min(lefts) >= pad_px:
            return
        if engine is None or not hasattr(engine, "get"):
            return  # no constrained layout to push on; leave the figure as drawn
        rect = tuple(engine.get().get("rect", (0, 0, 1, 1)))
        gutter = (pad_px - min(lefts)) / (fig.get_size_inches()[0] * fig.dpi)
        # rect is (left, bottom, width, height): take the space off the width too, or
        # the right-hand colorbar labels walk off the other edge instead.
        engine.set(rect=(rect[0] + gutter, rect[1], rect[2] - gutter, rect[3]))


#: Share of its own box one line of text may fill before :func:`_fit_text_widths`
#: shrinks it. Not 1.0, because a title that reaches exactly to the panel's edges reads
#: as overflowing even when it technically doesn't.
_TEXT_FIT_FRACTION = 0.98


def _shrink_to_fit(text, limit_px: float, renderer, *, along: str = "width") -> None:
    """Reduce ``text``'s font size until it fits ``limit_px``, no further than MIN_PT.

    One shot rather than a loop: a font size is a linear scaling of the glyphs, so the
    rendered extent is very nearly proportional to it and the first correction lands.
    """
    if not text.get_text() or limit_px <= 0:
        return
    extent = getattr(text.get_window_extent(renderer), along)
    if extent <= limit_px or extent <= 0:
        return
    size = text.get_fontsize()
    shrunk = max(size * limit_px / extent, MIN_PT)
    if shrunk < size - 0.05:
        text.set_fontsize(shrunk)


def _fit_text_widths(fig, renderer=None) -> None:
    """Shrink any single label still too long for the thing it labels.

    The type scale sizes text against the *space* available (see
    :mod:`ocean_skill.plot.typography`), which is the right default but cannot know how
    many characters the caller will put in it. Ocean variable names and units are long
    -- a difference bar labelled ``test − reference [mmol m-3]`` under a 2-inch panel,
    or a CF standard name as a panel title -- and no choice of base size fixes that,
    because the problem is one particular string rather than the scale.

    So this measures the drawn result and shrinks only what actually overflows, leaving
    everything else at the size the scale chose. Same draw-measure-adjust shape as
    :func:`_fit_left_margin` and :func:`_align_colorbars`, and safe in the same way:
    text can only get smaller here, which only ever frees layout space.

    Runs before :func:`_align_colorbars`, which stands the layout engine down -- so the
    redraw that follows still gets to reclaim the room a shrunken title gave back.
    """
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    fig_w_px = fig.get_size_inches()[0] * fig.dpi

    if (suptitle := getattr(fig, "_suptitle", None)) is not None:
        _shrink_to_fit(suptitle, fig_w_px * _TEXT_FIT_FRACTION, renderer)

    for ax in fig.axes:
        box = ax.get_window_extent(renderer)
        if getattr(ax, "_osk_cbar_parents", None):
            # the bar's label runs along the bar, so it is the bar's own length that
            # bounds it -- and for a vertical bar the label is rotated, which makes its
            # *height* the dimension that has to fit
            horizontal = ax._osk_cbar_horizontal
            label = (ax.xaxis if horizontal else ax.yaxis).label
            limit = (box.width if horizontal else box.height) * _TEXT_FIT_FRACTION
            _shrink_to_fit(
                label, limit, renderer, along="width" if horizontal else "height"
            )
            continue
        _shrink_to_fit(ax.title, box.width * _TEXT_FIT_FRACTION, renderer)
        if (row_label := getattr(ax, "_osk_row_label", None)) is not None:
            # rotated 90 degrees up the left edge of the row: its height is its length
            _shrink_to_fit(
                row_label, box.height * _TEXT_FIT_FRACTION, renderer, along="height"
            )


def _aspect_of(da) -> float:
    """Return one field's ``lon_span / lat_span`` — the shape a panel wants to be."""
    try:
        lon_span = float(np.ptp(np.asarray(da["lon"])))
        lat_span = float(np.ptp(np.asarray(da["lat"])))
        return lon_span / max(lat_span, 1e-6)
    except Exception:  # pragma: no cover - fall back to a square-ish panel
        return 1.0


def _map_aspect(comparisons, reference_name: str) -> float:
    """Return the maps' ``lon_span / lat_span`` — the shape a panel wants to be.

    Read off the reference grid, which both renderers can see; ``clamp_aspect`` in
    :mod:`~ocean_skill.plot.typography` bounds it, so a degenerate span here only has
    to be caught, not corrected.
    """
    try:
        return _aspect_of(comparisons[0]["aligned"][reference_name])
    except Exception:  # pragma: no cover - fall back to a square-ish panel
        return 1.0


def _row_height(comparisons, reference_name: str, n: int, font_scale: float = 1.0):
    """Row height (inches) matched to the map's own aspect ratio.

    A fixed height leaves a tall empty band above and below wide domains — and the
    colorbars, which span the whole cell, then tower over the maps. Sizing the row to
    ``lon_span / lat_span`` keeps the bars the same height as the maps beside them.

    The text above and below the map is the other term, and it is now measured in ems
    of the type the row will actually get rather than the flat 0.62in this used to add:
    a constant is wrong at both ends, leaving a tall row's titles adrift in white space
    and a short row's crowded, and it made ``font_scale`` a way to squeeze the maps
    instead of a way to enlarge the type. ``typography.row_height`` resolves the
    circularity (the type depends on the row height it is helping decide).
    """
    return _typographic_row_height(
        _map_aspect(comparisons, reference_name),
        nrows=n,
        ncols=3,
        page_w=PAGE_W,
        page_h=PAGE_H,
        font_scale=font_scale,
    )


def _shared_norms(comparisons, test_name: str, reference_name: str):
    """Return one ``(seq_norm, div_norm)`` computed across *every* row.

    For ``field_grid(..., shared_limits=True)``: colour limits derived from all
    rows' test+reference values combined, and one difference range from all rows'
    diffs — rather than each row scaling to its own data.
    """
    import matplotlib.colors as mcolors

    standard_name = comparisons[0].get("standard_name")
    all_t = [np.asarray(c["aligned"][test_name]) for c in comparisons]
    all_r = [np.asarray(c["aligned"][reference_name]) for c in comparisons]
    vmin, vmax = _limits(*all_t, *all_r)
    seq_norm = norm_for(standard_name, vmin, vmax)

    all_d = np.concatenate(
        [np.asarray(c["aligned"]["difference"]).ravel() for c in comparisons]
    )
    finite = all_d[np.isfinite(all_d)]
    dmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    div_norm = mcolors.Normalize(vmin=-(dmax or 1.0), vmax=dmax or 1.0)
    return seq_norm, div_norm


def field_grid(
    comparisons: list[dict[str, Any]],
    *,
    test_name: str = "test",
    reference_name: str = "reference",
    labels: tuple[str, str] | None = None,
    title: str | None = None,
    mark: str = "pcolormesh",
    save: str | Path | None = None,
    domain: tuple[float, float, float, float] | None = None,
    row_height: float | None = None,
    figsize: tuple[float, float] | None = None,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    row_label_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    shared_limits: bool = False,
    shared_axis_labels: bool = True,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
):
    """Stack one ``test | reference | difference`` row per comparison.

    Each item is a dict with ``aligned`` and optionally ``row_label``, ``units``,
    ``standard_name``, ``metrics`` and ``labels``. Every row gets its own colour
    scales by default (variables have different ranges), its own colorbars, and its
    own column titles from its own ``labels`` — rows commonly come from *different*
    reference sources in a ``compare()`` fan-out (nitrate from one WOA entry,
    phosphate from another), so reusing one shared pair of titles for every row
    would mislabel all but the first. The top-level ``labels`` is only the fallback
    for a row that doesn't carry its own. Row height follows the map's aspect ratio
    plus the room its type needs (override with ``row_height``), and the total is
    capped at the 11-inch page — or set ``figsize`` to size the whole figure yourself.
    ``metric_keys`` picks which of ``metrics.compute()``'s values appear in each row's
    corner box (default ``bias``/``rmse``/``corr``).

    Font sizes are derived from the figure's geometry rather than fixed, so a grid of
    eight rows gets smaller panel type than a grid of two without being asked, while its
    suptitle — which labels the whole figure, not one row — does not shrink with the
    rows. ``font_scale`` multiplies them all, and adds the height that needs; a size
    passed explicitly in a ``*_kwargs`` dict still overrides outright. See
    :mod:`ocean_skill.plot.typography`.

    ``shared_limits=True`` makes every row's colour scale (and its difference
    range) span *all* rows' data instead of each computing its own — meaningful
    only when every row is the same variable (e.g. one depth per row), since
    different variables have unrelated ranges and units; sharing across those would
    make every row's colours meaningless relative to the numbers on the bar. Warns
    if the rows' ``standard_name``s actually differ.

    ``shared_axis_labels=True`` (the default) draws grid lines on every panel but
    only labels the leftmost column's latitude axis and the bottom row's longitude
    axis — the usual convention for a grid of maps, since every other panel would
    otherwise repeat labels a neighbour already shows. Set ``False`` to label every
    panel's axes independently, as every version before this one did.

    ``align_colorbars=True`` (the default) makes each row's vertical bars start and
    end level with that row's maps — top with the top of the axes, bottom with the
    bottom, excluding the title above and the longitude labels below, which the grid
    cell the bar is otherwise sized to includes. See :func:`_align_colorbars`.

    The ``*_kwargs`` parameters each merge onto their current defaults and map onto
    one matplotlib/cartopy call — see :func:`field_row`'s docstring for the full
    list; the same names mean the same thing here, applied per row.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    n = len(comparisons)
    proj = ccrs.PlateCarree()
    row_h = row_height or _row_height(comparisons, reference_name, n, font_scale)
    figsize = figsize or (PAGE_W, row_h * n)
    scale = _scale_for(figsize, nrows=n, font_scale=font_scale)
    defaults = _style_defaults(scale, horizontal_colorbar=False)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=figsize,
        subplot_kw={"projection": proj},
        constrained_layout=True,
        squeeze=False,
    )

    shared_seq_norm = shared_div_norm = None
    if shared_limits:
        import warnings

        names = {c.get("standard_name") for c in comparisons}
        if len(names) > 1:
            warnings.warn(
                f"shared_limits=True but rows use different variables "
                f"({sorted(nm for nm in names if nm)}); their ranges/units differ, "
                "so one shared colour scale won't mean the same thing on every row.",
                stacklevel=2,
            )
        shared_seq_norm, shared_div_norm = _shared_norms(
            comparisons, test_name, reference_name
        )

    for i, comp in enumerate(comparisons):
        ims, lab = _draw_row(
            axes[i],
            comp["aligned"],
            test_name=test_name,
            reference_name=reference_name,
            labels=comp.get("labels") or labels or ("test", "reference"),
            units=comp.get("units"),
            standard_name=comp.get("standard_name"),
            metrics=comp.get("metrics"),
            mark=mark,
            domain=domain,
            row_label=comp.get("row_label"),
            metric_keys=metric_keys,
            title_kwargs=title_kwargs,
            gridline_kwargs=gridline_kwargs,
            tick_label_kwargs=tick_label_kwargs,
            row_label_kwargs=row_label_kwargs,
            metrics_kwargs=metrics_kwargs,
            seq_norm=shared_seq_norm,
            div_norm=shared_div_norm,
            shared_axis_labels=shared_axis_labels,
            is_bottom_row=(i == n - 1),
            defaults=defaults,
        )
        _draw_colorbar(
            fig, ims[1], axes[i][:2], lab, colorbar_kwargs, defaults["colorbar_kwargs"]
        )
        _draw_colorbar(
            fig,
            ims[2],
            axes[i][2],
            f"test − reference {lab}",
            colorbar_kwargs,
            defaults["colorbar_kwargs"],
        )

    # after the suptitle, so the margin is fitted to the layout the figure ends with
    if title:
        fig.suptitle(title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs))
    _fit_left_margin(fig)
    _fit_text_widths(fig)
    if align_colorbars:
        _align_colorbars(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


#: Facet coordinates that name a vertical level rather than a time. ``z`` is what
#: :func:`ocean_skill.roms.to_depth` produces; the rest are what observational products
#: call the same axis, matching :data:`ocean_skill.cf._COORD_FALLBACKS`.
_DEPTH_COORDS = ("z", "depth", "depth_surface", "lev")


def facet_labels(coord) -> list[str]:
    """Panel labels for a facet coordinate, spelled to say which reduction made it.

    The shapes :mod:`ocean_skill.operators` can leave standing are distinguishable from
    the coordinate alone, and are deliberately labelled so that the figure says which
    one it is:

    * timestamps (a ``resample``) -> ``"Jan 2012"``. The year is not optional here: it
      is the only thing on the page distinguishing six consecutive months from six
      months of a climatology, and a reader who cannot tell those apart is reading the
      wrong figure without knowing it.
    * integer months (``{"groupby": "month"}``) -> ``"Jan"``, no year, because there
      isn't one — the panel is every January of the record.
    * a vertical level -> ``"50 m"``. Taken through ``abs`` because the model's own
      axis is negative-down (:func:`ocean_skill.roms.to_depth` interpolates onto
      ``-depths``) while the depth the caller asked for, and the one a reader expects
      on a label, is positive-down.
    * anything else (``{"groupby": "season"}`` gives ``"DJF"``) -> its own value.
    """
    import calendar

    name = str(coord.name)
    values = list(np.atleast_1d(coord.values))
    try:
        # covers numpy datetime64 and cftime alike, which is why this goes through
        # xarray's accessor rather than pandas or datetime directly -- a ROMS run on a
        # 360-day calendar carries cftime objects that pd.Timestamp cannot parse.
        return [str(v) for v in coord.dt.strftime("%b %Y").values]
    except (TypeError, AttributeError):
        pass
    if name == "month":
        try:
            return [calendar.month_abbr[int(v)] for v in values]
        except (ValueError, TypeError, IndexError):  # pragma: no cover - odd coord
            pass
    if name in _DEPTH_COORDS:
        try:
            return [f"{abs(float(v)):g} m" for v in values]
        except (ValueError, TypeError):  # pragma: no cover - odd coord
            pass
    return [str(v) for v in values]


def frame_labels(coord) -> list[str]:
    """Return a movie's frame labels: :func:`facet_labels`, refined until they differ.

    A facet grid's panels are almost always a reduction — six monthly means, twelve
    climatological months — so ``"%b %Y"`` names them exactly. A movie is as often the
    *unreduced* axis: every step of a run, which at daily cadence gives 31 panels all
    called ``Jan 2012`` and at hourly cadence 744. On a still that is merely a repeated
    caption; interactively the labels are the slider's values, and duplicates collapse
    frames on top of each other silently.

    So this starts from :func:`facet_labels` — a monthly movie is labelled exactly as
    the equivalent grid is, which is the point — and escalates the resolution only when
    the labels would not tell one frame from another. Non-datetime axes (levels,
    seasons) come back unchanged, having nothing finer to fall back to.
    """
    labels = facet_labels(coord)
    if len(set(labels)) == len(labels):
        return labels
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            # same accessor facet_labels uses, so cftime calendars work here too
            finer = [str(v) for v in coord.dt.strftime(fmt).values]
        except (TypeError, AttributeError):
            return labels
        labels = finer
        if len(set(finer)) == len(finer):
            break
    return labels


def field_facet(
    field,
    *,
    facet_dim: str | None = None,
    row_dim: str | None = None,
    title: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    mark: str = "pcolormesh",
    save: str | Path | None = None,
    domain: tuple[float, float, float, float] | None = None,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    row_label_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    shared_limits: bool = False,
    shared_axis_labels: bool = True,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
):
    """Draw one map per value of ``facet_dim``: a single field over time, in order.

    The model-only counterpart of :func:`field_grid`. There is no reference and so no
    difference panel and no metrics box; what varies across the panels is the facet
    axis — most usefully time, via ``aggregate={"time": {"resample": "1MS", "reduce":
    "mean"}}``, which gives one panel per month of the run.

    **One facet axis.** The panels are a single series, so the grid is free and
    ``ncols`` defaults to :func:`~ocean_skill.plot.typography.facet_layout`, which reads
    the orientation off the domain's own aspect ratio: a wide box stacks down the page,
    a tall one spreads across it. Every panel shares one colour scale and one colorbar
    — the whole point of the family rather than a default worth changing, since these
    panels are the same quantity at different times and per-panel scaling would make a
    doubling between March and April look like no change at all. The colorbar follows
    the grid — horizontal beneath a wide, short one, vertical beside a tall one — since
    a bar on the long edge stays the same length as the panels it describes.

    **Two facet axes.** Pass ``row_dim`` as well (``select={"depth": [0, 50, 100]}``
    leaves a vertical axis standing beside the monthly one) and the grid stops being a
    free choice: it is ``len(row_dim)`` by ``len(facet_dim)``, so ``ncols`` and the
    aspect-ratio rule no longer apply. Each row then gets **its own colour scale and its
    own colorbar**, exactly as :func:`field_grid`'s rows do and for the same reason:
    nitrate at 100 m and nitrate at the surface have unrelated ranges, and one scale
    across both would flatten every surface panel to the bottom of the bar. Set
    ``shared_limits=True`` for a single scale across the whole grid, which is right only
    when the rows genuinely share a range.

    Panel titles come from the facet coordinate itself (see :func:`facet_labels`), so a
    consecutive-month figure is labelled ``Jan 2012`` and a climatology ``Jan``, and the
    two cannot be confused for one another on the page. With a ``row_dim`` the titles
    appear on the top row only — every row below shows the same months — and each row is
    named down the left edge instead (``50 m``), the same rotated label
    :func:`field_grid` uses.

    The ``*_kwargs`` parameters and ``font_scale`` mean exactly what they do in
    :func:`field_row`; ``metrics_kwargs`` has no counterpart here, there being no
    metrics, and ``row_label_kwargs`` applies only when there is a ``row_dim``.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    from ocean_skill.plot.typography import facet_figsize, facet_layout

    for name, value in (("facet_dim", facet_dim), ("row_dim", row_dim)):
        if value is not None and value not in field.dims:
            raise ValueError(
                f"{name} {value!r} is not a dimension of the field ({list(field.dims)})"
            )
    if row_dim is not None and row_dim == facet_dim:
        raise ValueError(
            f"facet_dim and row_dim are both {facet_dim!r}; one axis cannot be both "
            "the rows and the columns"
        )

    n = int(field.sizes[facet_dim]) if facet_dim else 1
    aspect = _aspect_of(field)
    if row_dim is not None:
        # Two axes fix the grid: rows are the levels, columns the periods. Nothing is
        # left for the aspect-ratio rule to choose, and an ncols that disagrees with
        # the data would drop panels rather than re-flow them, so it is refused here
        # instead of being quietly ignored.
        if ncols is not None and int(ncols) != n:
            raise ValueError(
                f"ncols={ncols} contradicts row_dim={row_dim!r}: with two facet axes "
                f"the grid is {field.sizes[row_dim]} x {n} (one column per "
                f"{facet_dim!r}), so there is no column count left to choose."
            )
        nrows, ncols = int(field.sizes[row_dim]), n
    elif ncols is None:
        ncols, nrows = facet_layout(n, aspect)
    else:
        ncols = max(int(ncols), 1)
        nrows = -(-n // ncols)
    # A bar on the grid's long edge stays the same length as the panels it describes;
    # on the short edge it would be a stub beside a tall column, or a rule under a
    # wide one. Per-row bars are always vertical, beside their row.
    per_row_bars = row_dim is not None and not shared_limits
    horizontal = False if per_row_bars else ncols > nrows

    figsize = figsize or facet_figsize(
        aspect,
        nrows=nrows,
        ncols=ncols,
        # with two axes only the top row is titled, so the rows below need a gap
        # rather than a title's worth of room as well
        title_every_row=row_dim is None,
        font_scale=font_scale,
    )
    scale = type_scale(
        figsize,
        ncols=ncols,
        nrows=nrows,
        font_scale=font_scale,
        # the suptitle spans the page, so it is sized as every other family's is
        # rather than off this grid's column count -- see type_scale
        figure_ncols=REFERENCE_GRID[0],
    )
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    if not per_row_bars:
        # One bar refitted across every row comes out as many times fatter as there are
        # rows; a per-row bar spans one row, which is what the grid default already
        # describes. See FACET_COLORBAR_ASPECT.
        defaults["colorbar_kwargs"] = {
            **defaults["colorbar_kwargs"],
            "aspect": FACET_COLORBAR_ASPECT,
        }
    merged_title = _merged(defaults["title_kwargs"], title_kwargs)
    merged_gridline = _merged(defaults["gridline_kwargs"], gridline_kwargs)
    merged_tick = _merged(defaults["tick_label_kwargs"], tick_label_kwargs)
    merged_row_label = _merged(defaults["row_label_kwargs"], row_label_kwargs)

    cmap, _ = cmaps_for(standard_name)

    def _norm_of(sub):
        vmin, vmax = _limits(sub)
        return norm_for(standard_name, vmin, vmax)

    # Computed before drawing so each panel is drawn against its scale rather than
    # corrected afterwards. One norm per row, unless there is only one scale to have.
    if per_row_bars:
        norms = [_norm_of(field.isel({row_dim: r})) for r in range(nrows)]
    else:
        norms = [_norm_of(field)] * nrows

    labels = (
        facet_labels(field[facet_dim])
        if facet_dim and facet_dim in field.coords
        else [None] * n
    )
    row_labels = (
        facet_labels(field[row_dim])
        if row_dim and row_dim in field.coords
        else [None] * nrows
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
        squeeze=False,
    )
    flat = list(axes.ravel())
    used, ims, im = [], [], None
    n_panels = nrows * ncols if row_dim is not None else n
    for i in range(n_panels):
        ax = flat[i]
        row, col = divmod(i, ncols)
        if row_dim is not None:
            panel = field.isel({row_dim: row, facet_dim: col})
            # Below the top row every panel repeats its column's period; the row is
            # named down the left edge instead.
            label = labels[col] if row == 0 else None
        else:
            panel = field.isel({facet_dim: i}) if facet_dim else field
            label = labels[i]
        im = _draw_map(
            ax,
            panel,
            label=label,
            cmap=cmap,
            norm=norms[row],
            mark=mark,
            domain=domain,
            gridline_kwargs=merged_gridline,
            tick_label_kwargs=merged_tick,
            title_kwargs=merged_title,
            # The bottom row is ragged when n does not fill the grid, so "is there a
            # panel below me?" is the question, not "am I in the last row?".
            left_labels=(col == 0) if shared_axis_labels else None,
            bottom_labels=(i + ncols >= n_panels) if shared_axis_labels else None,
        )
        used.append(ax)
        ims.append(im)
        if col == 0 and row_labels[row]:
            _add_row_label(ax, row_labels[row], merged_row_label)

    # Cells past the last panel carry no map, no gridlines and so no label artists —
    # hidden rather than deleted, which keeps the remaining panels on the grid they
    # were sized for instead of letting the layout engine expand them into the gap.
    for ax in flat[n_panels:]:
        ax.set_visible(False)

    bar_label = f"[{units}]" if units else ""
    if per_row_bars:
        for row in range(nrows):
            span = slice(row * ncols, (row + 1) * ncols)
            _draw_colorbar(
                fig,
                ims[span][0],
                used[span],
                bar_label,
                colorbar_kwargs,
                defaults["colorbar_kwargs"],
            )
    elif im is not None:
        _draw_colorbar(
            fig, im, used, bar_label, colorbar_kwargs, defaults["colorbar_kwargs"]
        )

    if title:
        fig.suptitle(title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs))
    _fit_left_margin(fig)
    _fit_text_widths(fig)
    if align_colorbars:
        _align_colorbars(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


#: Output formats :func:`field_movie` writes, and which matplotlib writer does each.
#: A gif needs nothing beyond Pillow, which matplotlib already requires, so it always
#: works; mp4 goes through ffmpeg, an external binary that may not be installed.
MOVIE_FORMATS: dict[str, str] = {
    ".mp4": "ffmpeg",
    ".m4v": "ffmpeg",
    ".mov": "ffmpeg",
    ".gif": "pillow",
}

#: Frames per second a movie defaults to. Model output at daily or monthly cadence reads
#: better slowly than smoothly — an eighth of a second is long enough to take a frame in
#: and short enough that the motion is still motion.
DEFAULT_FPS = 8

#: Dots per inch a movie's frames are rasterized at, below the 150 a saved PNG gets: a
#: movie is watched rather than examined, and every frame pays the cost twice over (in
#: encoding time and in file size).
DEFAULT_MOVIE_DPI = 110

#: Frame count past which :func:`field_movie` says so before spending the time. Not a
#: cap: a year of hourly output really is 8760 frames, and refusing to draw what was
#: asked for would be worse than taking a while over it. But every frame is a full
#: cartopy redraw, so an accidental extra axis is worth catching before the render and
#: not after it.
FRAME_WARN_AT = 200


def _select_frames(frames: list, every: int) -> list:
    """Take every ``every``-th frame, warning if a lot are left.

    Striding is a plotting decision, not a data one — it thins what is *shown* without
    touching the reduction that produced it, which is why it lives here rather than in
    a ``select``/``aggregate`` spec. ``every=24`` turns hourly output into daily frames
    without re-preparing anything.

    Takes any list, since a comparison movie's frames are spec items while a facet
    movie's are indices into the facet axis; only the count matters here.
    """
    if every < 1:
        raise ValueError(f"every must be 1 or more, got {every}")
    frames = frames[::every]
    if len(frames) > FRAME_WARN_AT:
        import warnings

        seconds = len(frames) / max(DEFAULT_FPS, 1)
        warnings.warn(
            f"{len(frames)} frames is a long movie ({seconds / 60:.0f}m "
            f"{seconds % 60:.0f}s at {DEFAULT_FPS} fps) and every frame is a full "
            "redraw. Narrow it with select= (e.g. {'time': '2012-01'}), collapse it "
            "with aggregate= (e.g. a daily or monthly mean), or thin it here with "
            f"every= (every={max(len(frames) // FRAME_WARN_AT, 2)} would give "
            f"{len(frames) // max(len(frames) // FRAME_WARN_AT, 2)}).",
            stacklevel=_stacklevel.find(),
        )
    return frames


def _movie_writer(path: Path, fps: int):
    """Return the matplotlib writer for ``path``'s extension.

    Chosen from the extension rather than a ``format=`` parameter because the caller has
    already said which they want by naming the file, and two ways to say it could
    disagree.
    """
    kind = MOVIE_FORMATS.get(path.suffix.lower())
    if kind is None:
        raise ValueError(
            f"cannot write a movie to {path.name!r}: unknown extension "
            f"{path.suffix!r}. Use one of {', '.join(sorted(MOVIE_FORMATS))}."
        )
    if kind == "pillow":
        from matplotlib.animation import PillowWriter

        return PillowWriter(fps=fps)

    from matplotlib.animation import FFMpegWriter

    if not FFMpegWriter.isAvailable():
        raise RuntimeError(
            f"writing {path.name} needs ffmpeg, which matplotlib cannot find. Either\n"
            "  install it:  conda install -c conda-forge ffmpeg   (or: module load "
            "ffmpeg)\n"
            f"  or write a .gif instead: save={str(path.with_suffix('.gif'))!r}, which "
            "needs nothing beyond matplotlib itself."
        )
    return FFMpegWriter(
        fps=fps,
        metadata={"artist": "ocean-skill"},
        codec="libx264",
        extra_args=[
            # yuv420p is what players that will not touch anything else want, which is
            # most of them. It requires even pixel dimensions, and a figure sized in
            # inches from the data's own aspect ratio lands on an odd one often enough
            # that padding the odd edge is cheaper than constraining every figsize.
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white",
            "-preset",
            "fast",
        ],
    )


def _one_facet_axis(field, facet_dim: str | None) -> str:
    """Return ``facet_dim`` once confirmed to be the field's *only* non-spatial axis.

    A movie plays one axis, so anything else left standing means a frame is not a single
    map — most often a second facet axis (``select={"depth": [0, 50, 100]}`` beside a
    monthly one), which :func:`field_facet` can lay out as rows and a movie cannot play.
    Caught here rather than left to pcolormesh, which fails on the dimensionality with
    nothing to say about the cause. Shared with the interactive renderer so the two
    refuse the same fields for the same stated reason.
    """
    if facet_dim is None:
        got = ", ".join(f"{d}={field.sizes[d]}" for d in field.dims) or "no dimensions"
        raise ValueError(
            f"a movie needs an axis to play, but this field is a single map ({got}) — "
            "every axis was collapsed, either by an aggregate= that reduces them all "
            "or by a select= that picked one value. Leave time standing:\n"
            "  aggregate=None (or {})                                    every step\n"
            '  aggregate={"time": {"resample": "1MS", "reduce": "mean"}}  one per '
            "month\n"
            '  aggregate={"time": {"groupby": "month", "reduce": "mean"}} a '
            "climatology\n"
            "Inspect what you actually got with `.data` (a DataArray) or "
            "`.facet_dims`, and if it disagrees with the call, re-prepare with "
            "`.prepare(refresh=True)` — a lane cached under an older meaning of "
            "aggregate= is the one way this can surprise you.\n"
            "Or use .plot() for the single map you have."
        )
    if facet_dim not in field.dims:
        raise ValueError(
            f"facet_dim {facet_dim!r} is not a dimension of the field "
            f"({list(field.dims)})"
        )
    from ocean_skill.align import _lat_name, _lon_name

    spatial: set[str] = set()
    for name in (_lon_name(field), _lat_name(field)):
        if name is not None:
            spatial |= {str(d) for d in field[name].dims}
    extra = [str(d) for d in field.dims if d not in spatial and str(d) != facet_dim]
    if extra:
        raise ValueError(
            f"{extra} still stands beyond {facet_dim!r} and the horizontal axes, so a "
            "frame is not a single map. A movie plays one axis: collapse the others "
            'with aggregate= or narrow them with select= (e.g. {"depth": 50}). '
            ".plot() can lay a second axis out as rows instead."
        )
    return facet_dim


def _update_field(ax, im, da, *, mark: str, proj):
    """Point ``im`` at ``da`` — a new frame of the same field — and return the artist.

    A ``QuadMesh`` holds its values in an array that can simply be swapped, which is
    why ``pcolormesh`` is the mark to animate. A filled contour set is geometry rather
    than an image and has no array to swap, so it is removed and redrawn at the same
    cmap/norm — hence a *new* artist, which is why this returns one.
    """
    if mark == "contourf":
        cmap, norm = im.cmap, im.norm
        im.remove()
        return ax.contourf(
            da["lon"],
            da["lat"],
            da,
            transform=proj,
            cmap=cmap,
            norm=norm,
            levels=_contour_levels(norm),
        )
    im.set_array(np.asarray(da))
    return im


def field_movie(
    frames: list[dict[str, Any]],
    *,
    save: str | Path | None = None,
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_MOVIE_DPI,
    every: int = 1,
    test_name: str = "test",
    reference_name: str = "reference",
    labels: tuple[str, str] | None = None,
    title: str | None = None,
    mark: str = "pcolormesh",
    domain: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    frame_label_kwargs: dict[str, Any] | None = None,
    frame_label: bool = True,
    shared_limits: bool = True,
    shared_axis_labels: bool = True,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
    progress: bool = True,
):
    """Animate one ``test | reference | difference`` row over a sequence of frames.

    Each item of ``frames`` is shaped exactly as a :func:`field_grid` row (``aligned``
    plus optional ``metrics``, ``units``, ``standard_name``, ``labels``) with one
    addition: ``frame_label``, the text identifying that frame — a timestamp, usually —
    drawn in the top-left of the test panel. So the same items that stack down a page as
    a grid play as a movie here, and a frame looks exactly like the row
    :func:`field_row` draws, because it is drawn by the same code.

    ``save`` names the output file and, by its extension, the format:
    ``.mp4``/``.m4v``/``.mov`` through ffmpeg, ``.gif`` through Pillow (see
    :data:`MOVIE_FORMATS`). Only ffmpeg is an external dependency, so a gif is the
    fallback when it is missing. Without ``save`` nothing is written and the animation
    is only returned — useful in a notebook, where ``HTML(ani.to_jshtml())`` plays it
    inline.

    ``every=N`` keeps every Nth frame — ``every=24`` turns hourly output into daily
    frames without re-preparing anything, since it thins what is *shown* rather than
    what was computed. A movie longer than :data:`FRAME_WARN_AT` frames says so before
    spending the time on it, and suggests a stride; it is a warning and not a cap, since
    a year of hourly output legitimately is 8760 frames.

    ``shared_limits=True`` (the default) takes the colour scale from **every** frame at
    once, so a value is the same colour throughout; ``False`` takes it from the first
    frame alone. Either way the scale is fixed for the whole movie — a per-frame scale
    makes an animation unreadable, since every frame's colours would mean something
    different and the eye cannot tell a change in the field from a change in the ruler.

    Only the values, the frame label and the metrics box are redrawn per frame: the
    figure, its layout, its colorbars and its axis labelling are built once from the
    first frame, so nothing shifts or resizes as the movie plays. Every other parameter
    means what it does in :func:`field_row`, which is where they are documented;
    ``frame_label_kwargs`` styles the frame label (an ``Axes.text``, see
    :data:`DEFAULT_FRAME_LABEL_KWARGS`) and ``frame_label=False`` omits it.

    Returns the :class:`matplotlib.animation.FuncAnimation`. Keep a reference to it:
    an animation whose last reference is dropped stops.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    if not frames:
        raise ValueError("a movie needs at least one frame, got none")
    frames = _select_frames(frames, every)
    shapes = {np.shape(f["aligned"][reference_name]) for f in frames}
    if len(shapes) > 1:
        raise ValueError(
            f"every frame must be on the same grid, got shapes {sorted(shapes)}. "
            "Frames are redrawn into one figure, so a grid that changes mid-movie has "
            "nowhere to go — compare each grid as its own movie."
        )

    first = frames[0]
    figsize = figsize or auto_figsize(
        _map_aspect(frames, reference_name),
        nrows=1,
        font_scale=font_scale,
        horizontal_colorbar=True,
    )
    scale = _scale_for(figsize, nrows=1, font_scale=font_scale)
    defaults = _style_defaults(scale, horizontal_colorbar=True)
    proj = ccrs.PlateCarree()

    seq_norm = div_norm = None
    if shared_limits and len(frames) > 1:
        seq_norm, div_norm = _shared_norms(frames, test_name, reference_name)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    ims, lab = _draw_row(
        axes,
        first["aligned"],
        test_name=test_name,
        reference_name=reference_name,
        labels=first.get("labels") or labels or ("test", "reference"),
        units=first.get("units"),
        standard_name=first.get("standard_name"),
        metrics=first.get("metrics"),
        mark=mark,
        domain=domain,
        metric_keys=metric_keys,
        title_kwargs=title_kwargs,
        gridline_kwargs=gridline_kwargs,
        tick_label_kwargs=tick_label_kwargs,
        metrics_kwargs=metrics_kwargs,
        seq_norm=seq_norm,
        div_norm=div_norm,
        shared_axis_labels=shared_axis_labels,
        is_bottom_row=True,
        defaults=defaults,
    )
    _draw_colorbar(
        fig, ims[1], axes[:2], lab, colorbar_kwargs, defaults["colorbar_kwargs"]
    )
    _draw_colorbar(
        fig,
        ims[2],
        axes[2],
        f"difference {lab}",
        colorbar_kwargs,
        defaults["colorbar_kwargs"],
    )

    label_text = None
    if frame_label and any(f.get("frame_label") for f in frames):
        label_text = axes[0].text(
            0.02,
            0.98,
            str(first.get("frame_label") or ""),
            transform=axes[0].transAxes,
            zorder=5,
            **_merged(defaults["frame_label_kwargs"], frame_label_kwargs),
        )

    if title:
        fig.suptitle(title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs))
    _fit_left_margin(fig)
    _fit_text_widths(fig)
    if align_colorbars:
        _align_colorbars(fig)

    metrics_text = getattr(axes[2], "_osk_metrics_text", None)
    keys = (test_name, reference_name, "difference")

    def update(index: int):
        frame = frames[index]
        aligned = frame["aligned"]
        for j, (ax, key) in enumerate(zip(axes, keys, strict=True)):
            ims[j] = _update_field(ax, ims[j], aligned[key], mark=mark, proj=proj)
        if label_text is not None:
            label_text.set_text(str(frame.get("frame_label") or ""))
        if metrics_text is not None:
            metrics_text.set_text(_metrics_text(frame.get("metrics"), metric_keys))
        return [*ims, label_text, metrics_text]

    return _animate(
        fig, update, len(frames), save=save, fps=fps, dpi=dpi, progress=progress
    )


def _animate(fig, update, n_frames: int, *, save, fps: int, dpi: int, progress: bool):
    """Wrap ``update`` in a ``FuncAnimation`` and, if asked, encode it to ``save``.

    The half of a movie that has nothing to do with what is being drawn, shared by
    :func:`field_movie` and :func:`facet_movie` so that "which writer, at what rate,
    reporting progress how" is answered once for both.
    """
    from matplotlib.animation import FuncAnimation

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / max(fps, 1),
        # every frame reads its own values off the frame list, so there is nothing to
        # cache; caching would hold every rendered frame in memory for no gain
        cache_frame_data=False,
        blit=False,
    )
    if not save:
        return ani
    save = Path(save).expanduser()
    save.parent.mkdir(parents=True, exist_ok=True)
    writer = _movie_writer(save, fps)
    callback = None
    if progress:

        def callback(index, total):
            end = "\n" if index + 1 == total else ""
            print(f"\r  frame {index + 1}/{total}", end=end, flush=True)

    ani.save(str(save), writer=writer, dpi=dpi, progress_callback=callback)
    print(f"ocean-skill: movie written to {save}")
    return ani


def facet_movie(
    field,
    *,
    facet_dim: str | None = None,
    save: str | Path | None = None,
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_MOVIE_DPI,
    every: int = 1,
    title: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    mark: str = "pcolormesh",
    domain: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    frame_label_kwargs: dict[str, Any] | None = None,
    frame_label: bool = True,
    shared_limits: bool = True,
    font_scale: float = 1.0,
    progress: bool = True,
):
    """Play one source's facet axis instead of laying it out: a movie of one field.

    The animated counterpart of :func:`field_facet`, and the model-only counterpart of
    :func:`field_movie` — one map, no reference, no difference panel, no metrics box.
    The axis that becomes the panels there becomes the frames here, so the same
    ``Field`` reads either way::

        run = osk.field("GOM_bgc", "salinity",
                        select={"time": "2012", "depth": "surface"})
        run.plot()                      # every step as a panel
        run.movie(save="salt.mp4")      # every step as a frame

    Frame labels come from the facet coordinate through :func:`frame_labels`, so they
    are spelled exactly as the static panels' titles are — ``Jan 2012`` for consecutive
    months, ``Jan`` for a climatology, ``50 m`` for a level — except where that would
    not tell one frame from another, since a movie is as often over the unreduced axis
    as over a reduction.

    ``row_dim`` has no counterpart: a movie has one axis to play, and two facet axes
    would need one to become the panels — which is what :func:`field_facet` is for.
    Every other parameter means what it does there, or in :func:`field_movie` for the
    movie-specific ones (``save``, ``fps``, ``dpi``, ``every``, ``frame_label``).
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    from ocean_skill.plot.typography import REFERENCE_GRID, facet_figsize

    facet_dim = _one_facet_axis(field, facet_dim)
    indices = _select_frames(list(range(int(field.sizes[facet_dim]))), every)
    labels = frame_labels(field[facet_dim]) if facet_dim in field.coords else None
    aspect = _aspect_of(field)
    # One panel, so the grid's long edge is the map's own: a wide domain takes a
    # horizontal bar beneath it, a tall one a vertical bar beside it. Same rule
    # field_facet applies to its grid, which for a single cell *is* the map.
    horizontal = aspect > 1.0
    figsize = figsize or facet_figsize(aspect, nrows=1, ncols=1, font_scale=font_scale)
    scale = type_scale(
        figsize,
        ncols=1,
        nrows=1,
        font_scale=font_scale,
        # the suptitle spans the page whatever the panel count — see field_facet
        figure_ncols=REFERENCE_GRID[0],
    )
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)

    # One scale for the whole movie, from every frame or just the first. Mandatory in
    # spirit either way: a scale re-derived per frame would make the ruler move with the
    # field. field_facet shares one scale across its panels for the same reason.
    scope = field if shared_limits else field.isel({facet_dim: indices[0]})
    vmin, vmax = _limits(scope)
    norm = norm_for(standard_name, vmin, vmax)
    cmap, _ = cmaps_for(standard_name)

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    im = _draw_map(
        ax,
        field.isel({facet_dim: indices[0]}),
        # An empty title rather than no title, so that set_title runs and pins ``y``
        # from DEFAULT_TITLE_KWARGS. A frame is identified by the label box inside the
        # panel (fixed position, so the layout it was built with still holds), which
        # leaves nothing for the title to say — but skipping set_title leaves
        # matplotlib's automatic title placement switched on, and over a cartopy
        # GeoAxes carrying gridline labels that computes an infinite y on matplotlib
        # 3.11: the title's extent comes out NaN, the axes' tight bbox with it, and the
        # map then drops out of the figure's tight bbox altogether. bbox_inches="tight"
        # — which Jupyter's inline backend uses — thereupon crops the map away and
        # leaves only the colorbar. See DEFAULT_TITLE_KWARGS for the full account.
        label="",
        cmap=cmap,
        norm=norm,
        mark=mark,
        domain=domain,
        gridline_kwargs=_merged(defaults["gridline_kwargs"], gridline_kwargs),
        tick_label_kwargs=_merged(defaults["tick_label_kwargs"], tick_label_kwargs),
        title_kwargs=_merged(defaults["title_kwargs"], title_kwargs),
    )
    _draw_colorbar(
        fig,
        im,
        ax,
        f"[{units}]" if units else "",
        colorbar_kwargs,
        defaults["colorbar_kwargs"],
    )

    label_text = None
    if frame_label and labels:
        label_text = ax.text(
            0.02,
            0.98,
            labels[indices[0]],
            transform=ax.transAxes,
            zorder=5,
            **_merged(defaults["frame_label_kwargs"], frame_label_kwargs),
        )

    if title:
        fig.suptitle(title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs))
    _fit_left_margin(fig)
    _fit_text_widths(fig)
    _align_colorbars(fig)

    proj = ccrs.PlateCarree()
    artists = [im]

    def update(frame: int):
        index = indices[frame]
        artists[0] = _update_field(
            ax, artists[0], field.isel({facet_dim: index}), mark=mark, proj=proj
        )
        if label_text is not None:
            label_text.set_text(labels[index])
        return [artists[0], label_text]

    return _animate(
        fig, update, len(indices), save=save, fps=fps, dpi=dpi, progress=progress
    )


def _nested_owner(key: str) -> str | None:
    """Which ``*_kwargs`` dict ``key`` belongs inside, if any.

    Styling here lives in nested dicts, so a plausible-looking option can be real and
    still be wrong at the top level — ``label_size`` is a colorbar key, not a
    ``field_grid`` parameter.
    """
    for name, defaults in _NESTED_KWARGS.items():
        if key in defaults:
            return name
    # colorbar_kwargs additionally forwards any label_*/tick_* key it is handed, so
    # those never appear in the defaults but are still colorbar options.
    if key.startswith(("label_", "tick_")):
        return "colorbar_kwargs"
    return None


def _check_options(fn, opts) -> None:
    """Reject unknown options with an error that says where they belong.

    Python's own ``TypeError: field_grid() got an unexpected keyword argument
    'label_size'`` names the key and stops there, which does not help when the key is a
    valid option one level down.
    """
    import inspect

    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return  # the function takes **kwargs; nothing is "unexpected"
    unknown = [k for k in opts if k not in params]
    if not unknown:
        return

    lines = []
    for key in sorted(unknown):
        owner = _nested_owner(key)
        if owner:
            lines.append(
                f"  {key!r} goes inside {owner}, e.g. {owner}={{{key!r}: ...}}"
            )
        else:
            lines.append(f"  {key!r} is not an option of {fn.__name__}()")
    accepted = ", ".join(sorted(k for k in params if not k.startswith("_")))
    raise TypeError(
        f"{fn.__name__}() got {len(unknown)} unusable option"
        f"{'' if len(unknown) == 1 else 's'}:\n"
        + "\n".join(lines)
        + f"\n\n{fn.__name__}() accepts: {accepted}"
    )


def render(spec, **kwargs: Any):
    """Draw a :class:`~ocean_skill.plot.spec.PlotSpec` with matplotlib.

    Dispatches on ``spec.family``; ``spec.options`` are merged with any keyword
    arguments, with the explicit keywords winning.
    """
    from ocean_skill.plot.summary import paired, target, taylor

    opts = {**spec.options, **kwargs}
    family = spec.family

    if family == "field_grid":
        _check_options(field_grid, opts)
    elif family == "field_row":
        _check_options(field_row, opts)
    elif family == "field_facet":
        _check_options(field_facet, opts)
    elif family == "field_movie":
        _check_options(field_movie, opts)
    elif family == "facet_movie":
        _check_options(facet_movie, opts)

    if family == "field_facet":
        item = spec.single
        return field_facet(
            item["field"],
            facet_dim=item.get("facet_dim"),
            row_dim=item.get("row_dim"),
            units=item.get("units"),
            standard_name=item.get("standard_name"),
            **opts,
        )
    if family == "facet_movie":
        # the same item field_facet takes, played rather than laid out. `row_dim` is
        # deliberately not passed on: a movie has one axis to play, and a second facet
        # axis would have to become panels, which is what field_facet is for.
        item = spec.single
        return facet_movie(
            item["field"],
            facet_dim=item.get("facet_dim"),
            units=item.get("units"),
            standard_name=item.get("standard_name"),
            **opts,
        )
    if family == "field_row":
        item = spec.single
        return field_row(
            item["aligned"],
            units=item.get("units"),
            standard_name=item.get("standard_name"),
            metrics=item.get("metrics"),
            **opts,
        )
    if family == "field_grid":
        return field_grid(spec.items, **opts)
    if family == "field_movie":
        return field_movie(spec.items, **opts)
    if family in ("taylor", "target", "paired"):
        # summary families work from metric records, which the spec carries per item
        fn = {"taylor": taylor, "target": target, "paired": paired}[family]
        return fn([_Record(i) for i in spec.items], **opts)
    raise NotImplementedError(f"matplotlib renderer: family {family!r} not implemented")


class _Record:
    """Adapt a spec item to the ``.metrics()``/``.label`` interface summaries need."""

    def __init__(self, item: dict[str, Any]):
        self._item = item

    def metrics(self) -> dict[str, Any]:
        return self._item.get("metrics", {})

    @property
    def label(self):
        return self._item.get("label")


register_renderer("matplotlib", render)
