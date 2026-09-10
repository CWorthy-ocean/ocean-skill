"""Static matplotlib renderer (PNG/JPG/PDF, and mp4/gif via FuncAnimation).

Currently implements the **field row**: ``test | reference | difference`` maps for a
gridded comparison. Test and reference share one colour scale (so they are visually
comparable) taken from the full range of the pair by default, or its 10th–90th
percentile with ``robust=True``; the difference panel uses a diverging map centred on
zero. Metrics go in a corner box, leaving the title for identity. Registers itself
under ``"matplotlib"``.

Every family here draws its panels through :func:`_draw_map`, and the two movie
families are the two static map families played rather than laid out:
:func:`field_movie` animates :func:`field_grid`'s comparison rows, :func:`facet_movie`
animates :func:`field_facet`'s facet axis. So a frame of a movie is the still it would
have been.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ocean_skill import _stacklevel
from ocean_skill.colormaps import cmaps_for, norm_for
from ocean_skill.plot.coastline import (
    DEFAULT_COASTLINE_RESOLUTION,
    is_gshhs,
    normalize_coastline_resolution,
)
from ocean_skill.plot.registry import register_renderer
from ocean_skill.plot.typography import (
    FACET_PANEL_W_FRACTION,
    MIN_PT,
    PAGE_H,
    PAGE_W,
    PANEL_W_FRACTION,
    PANEL_W_FRACTION_HORIZONTAL_CBAR,
    REFERENCE_GRID,
    ROW_OVERHEAD,
    ROW_OVERHEAD_HORIZONTAL_CBAR,
    SUPTITLE_ALLOWANCE,
    Canvas,
    auto_figsize,
    colorbar_is_horizontal,
    reference_scale,
    resolve_canvas,
    type_scale,
)

# aliased: field_grid already has a row_height *parameter*, which is the caller's
# override of exactly this
from ocean_skill.plot.typography import row_height as _typographic_row_height

__all__ = [
    "cross",
    "facet_labels",
    "field_facet",
    "field_grid",
    "field_map_grid",
    "field_row",
    "locations",
    "metric_panel_titles",
    "metric_panels",
    "metric_value_text",
    "profile",
    "render",
    "section",
    "section_row",
    "series",
    "skill_map",
    "time_depth",
    "time_depth_grid",
]

# PAGE_W/PAGE_H (the portrait page every figure has to fit) now live in typography,
# which is where they are used to decide sizes; re-exported under their old names.
__all__ += ["PAGE_H", "PAGE_W"]


def _limits(*arrays, robust: bool | float = False) -> tuple[float, float]:
    """Shared colour limits across all arrays.

    Default is the full finite range (min, max), so a colourbar's top always
    matches what :meth:`~ocean_skill.field.Field.extremum` and ``.series()``
    report at the same cell — nothing is clipped unless asked for. ``robust=True``
    clips to the 10th/90th percentile instead (the classic xarray-style "robust
    to outliers" scaling); a float ``q`` in ``(0, 1)`` clips to the central
    fraction ``q`` of the data (``q=0.8`` is the same as ``robust=True``).
    """
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if robust is False or robust is None:
        return float(np.min(vals)), float(np.max(vals))
    q = 0.8 if robust is True else float(robust)
    if not 0.0 < q < 1.0:
        raise ValueError(
            f"robust={robust!r} is not True/False or a central fraction in (0, 1) "
            "— pass the fraction itself (robust=0.8 for the 10th-90th percentile, "
            "same as robust=True), or robust=False for the plain min/max."
        )
    lo, hi = (1 - q) / 2 * 100, (1 + q) / 2 * 100
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
#: A series panel's lines and its legend. Separate dicts so ``line_kwargs`` can carry
#: anything ``Axes.plot`` takes without colliding with the legend's own keys.
DEFAULT_LINE_KWARGS: dict[str, Any] = {"linewidth": 1.2}
DEFAULT_LEGEND_KWARGS: dict[str, Any] = {"frameon": False}

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
    # Last deliberately: _nested_owner takes the first dict claiming a key, and these
    # two share several with the map families' dicts (`color`, `alpha` and `linewidth`
    # are gridline options too). At the end they answer only for keys nothing else has.
    "legend_kwargs": {**DEFAULT_LEGEND_KWARGS, "fontsize": REFERENCE_SCALE["legend"]},
    "line_kwargs": dict(DEFAULT_LINE_KWARGS),
}


def _scale_for(
    figsize: tuple[float, float], *, nrows: int = 1, font_scale: float = 1.0
) -> dict[str, float]:
    """Type scale for a figure of ``figsize`` holding ``nrows`` rows of three maps."""
    return type_scale(figsize, ncols=3, nrows=nrows, font_scale=font_scale)


#: Which key inside each ``*_kwargs`` dict carries a font size. Used to tell a size the
#: caller chose from one the type scale chose — see :func:`_pinned`.
_SIZE_KEYS: dict[str, tuple[str, ...]] = {
    "title_kwargs": ("fontsize", "size"),
    "tick_label_kwargs": ("size", "fontsize"),
    "row_label_kwargs": ("fontsize", "size"),
    "metrics_kwargs": ("fontsize", "size"),
    "suptitle_kwargs": ("fontsize", "size"),
    "colorbar_kwargs": ("label_size",),
    "legend_kwargs": ("fontsize", "size"),
    "line_kwargs": (),
}


def _pinned(kwargs: dict[str, Any] | None, which: str) -> bool:
    """Report whether the caller set a font size in this ``*_kwargs`` dict.

    An explicit size has to survive :func:`_fit_text_widths`, which otherwise shrinks
    anything that overflows its box — including a size that was asked for. Automatic
    sizing is a better default, not a new constraint, so the two are distinguished here
    and the fitting pass leaves the caller's choices alone.
    """
    return bool(kwargs) and any(key in kwargs for key in _SIZE_KEYS[which])


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
        "legend_kwargs": dict(DEFAULT_LEGEND_KWARGS),
        "line_kwargs": dict(DEFAULT_LINE_KWARGS),
    }


def _merged(
    defaults: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Shallow-merge a ``*_kwargs`` override onto its defaults.

    ``overrides=None`` returns the defaults unchanged.
    """
    return {**defaults, **(overrides or {})}


def _apply_subplot_spacing(fig, *, wspace: float | None, hspace: float | None) -> None:
    """Override ``constrained_layout``'s panel gutter -- ``None`` leaves it alone.

    ``wspace``/``hspace`` are matplotlib's own :class:`.ConstrainedLayoutEngine`
    names (the fraction of a panel's own width/height reserved as a gutter,
    default ``0.02`` each) -- exposed as-is rather than translated, so a value
    a caller already knows from ``plt.subplots_adjust`` carries over unchanged.
    """
    if wspace is None and hspace is None:
        return
    engine = fig.get_layout_engine()
    kwargs = {}
    if wspace is not None:
        kwargs["wspace"] = wspace
    if hspace is not None:
        kwargs["hspace"] = hspace
    engine.set(**kwargs)


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
        # an explicitly requested label_size must survive _fit_text_widths
        pinned = _pinned(colorbar_kwargs, "colorbar_kwargs")
        for axis in (cbar.ax.xaxis, cbar.ax.yaxis):
            axis.label._osk_size_pinned = pinned
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


def _basemap(
    ax,
    *,
    gridline_kwargs: dict[str, Any],
    tick_label_kwargs: dict[str, Any],
    left_labels: bool | None = None,
    bottom_labels: bool | None = None,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
):
    """Land fill, coastlines and labelled gridlines — what every map panel shares.

    Extracted from :func:`_draw_map` so a family with no field to draw (the
    ``locations`` map) still gets exactly this package's basemap rather than a
    near-copy that drifts. Returns the gridliner.

    ``coastline_resolution`` (see :mod:`ocean_skill.plot.coastline`) picks the
    dataset: ``"auto"`` (the default) leaves cartopy's own ``AdaptiveScaler`` to pick
    a Natural Earth scale from the panel's extent, a fixed ``"110m"``/``"50m"``/
    ``"10m"`` pins one, and a GSHHS scale (``"coarse"``..``"full"``) draws from that
    finer dataset instead — GSHHS ``levels=[1]`` is land, drawn twice (a filled
    polygon, then its edge) since :class:`~cartopy.feature.GSHHSFeature` has no
    separate coastline-only feature the way Natural Earth does.

    ``land`` controls the grey land fill, which otherwise paints over any data drawn
    underneath it: ``True`` (the default) is today's opaque ``"0.85"`` fill, a float
    in ``[0, 1]`` fades that fill to the given opacity (data shows through) while
    still drawing the coastline outline, and ``False`` draws neither fill nor
    outline — a completely bare map.
    """
    import cartopy.feature as cfeature

    if land is not True and land is not False:
        if not isinstance(land, (int, float)) or not 0.0 <= land <= 1.0:
            raise ValueError("land must be True, False, or a number in [0, 1]")
    resolution = normalize_coastline_resolution(coastline_resolution)
    if land is not False:
        fill_alpha = 1.0 if land is True else float(land)
        if is_gshhs(resolution):
            land_feature = cfeature.GSHHSFeature(scale=resolution, levels=[1])
            if fill_alpha > 0:
                ax.add_feature(
                    land_feature, facecolor="0.85", edgecolor="none", alpha=fill_alpha, zorder=2
                )
            ax.add_feature(land_feature, facecolor="none", edgecolor="black", linewidth=0.4, zorder=3)
        else:
            land_feature = cfeature.LAND if resolution == "auto" else cfeature.LAND.with_scale(resolution)
            if fill_alpha > 0:
                ax.add_feature(land_feature, facecolor="0.85", alpha=fill_alpha, zorder=2)
            ax.coastlines(resolution=resolution, linewidth=0.4, zorder=3)
    gl = ax.gridlines(draw_labels=True, **gridline_kwargs)
    gl.top_labels = gl.right_labels = False
    if left_labels is not None:
        gl.left_labels = left_labels
    if bottom_labels is not None:
        gl.bottom_labels = bottom_labels
    gl.xlabel_style = gl.ylabel_style = dict(tick_label_kwargs)
    return gl


def _map_projection(*fields):
    """Return a PlateCarree centred so every field's longitudes stay contiguous.

    A lane straddling the antimeridian (a 0-360 domain, as a Pacific model has)
    splits at the edges of the default ``central_longitude=0`` frame — the basin torn
    across both edges with a blank Atlantic between — so centring the frame on 180 puts
    the seam back outside the data. Only the *axes* projection moves; the ``transform=``
    handed to pcolormesh stays plain :class:`~cartopy.crs.PlateCarree`, because the
    coordinates are geographic degrees wherever the frame is centred.

    Straddling is decided by :func:`~ocean_skill.align.natural_convention`, the same
    span-based test ``align`` uses — not by ``lon.max() > 180``, which only sees a
    straddle when the coordinates happen to be *stored* in 0-360. A pair aligned onto a
    ±180 reference grid keeps a straddling Pacific domain's longitudes in ±180
    (``lon.max() <= 180``) while it still straddles, and the raw-value test missed
    exactly that case.
    """
    import cartopy.crs as ccrs

    from ocean_skill.align import natural_convention
    from ocean_skill.plot.proj_check import warn_projection_skew

    # Every geo panel drawn here goes through cartopy, so this is the one place to
    # catch a broken cartopy/PROJ pairing (see ocean_skill.plot.proj_check) for the
    # whole static family — cheap, since the check itself is cached per process.
    warn_projection_skew()
    for field in fields:
        if field is None or "lon" not in getattr(field, "coords", ()):
            continue
        if natural_convention(field) == "0-360":
            return ccrs.PlateCarree(central_longitude=180.0)
    return ccrs.PlateCarree()


def domain_ring(domain) -> np.ndarray | None:
    """Normalize a ``domain`` plot option to a closed ``(N, 2)`` ``[lon, lat]`` ring.

    Accepts either spelling the option is documented to take: the ``(lon_min,
    lat_min, lon_max, lat_max)`` bbox every family has always drawn, or the ``(N, 2)``
    vertex ring :func:`ocean_skill.comparison._outline_of` hands back for a
    curvilinear source's true (possibly rotated) grid shape. ``None`` stays ``None``
    (no outline drawn). Shared with :mod:`ocean_skill.plot.holoviews_renderer` so the
    two renderers agree on what the option means.
    """
    if domain is None:
        return None
    arr = np.asarray(domain, dtype="float64")
    if arr.shape == (4,):
        lo0, la0, lo1, la1 = arr
        return np.array([[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]])
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
        if not np.allclose(arr[0], arr[-1]):
            arr = np.vstack([arr, arr[:1]])
        return arr
    raise ValueError(
        "domain must be a (lon_min, lat_min, lon_max, lat_max) bbox or an (N, 2) "
        f"[lon, lat] ring, got an array of shape {arr.shape}."
    )


def _draw_map(
    ax,
    da,
    *,
    label: str | None,
    cmap,
    norm,
    mark: str,
    domain: tuple[float, float, float, float] | np.ndarray | None,
    gridline_kwargs: dict[str, Any],
    tick_label_kwargs: dict[str, Any],
    title_kwargs: dict[str, Any],
    left_labels: bool | None = None,
    bottom_labels: bool | None = None,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
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

    ``coastline_resolution``/``land`` are forwarded to :func:`_basemap` — see there.
    """
    import cartopy.crs as ccrs

    proj = ccrs.PlateCarree()
    draw = getattr(ax, "contourf" if mark == "contourf" else "pcolormesh")
    kw = {"levels": _contour_levels(norm)} if mark == "contourf" else {}
    im = draw(da["lon"], da["lat"], da, transform=proj, cmap=cmap, norm=norm, **kw)
    _basemap(
        ax,
        gridline_kwargs=gridline_kwargs,
        tick_label_kwargs=tick_label_kwargs,
        left_labels=left_labels,
        bottom_labels=bottom_labels,
        coastline_resolution=coastline_resolution,
        land=land,
    )
    ring = domain_ring(domain)
    if ring is not None:
        from matplotlib.lines import Line2D

        # add_artist rather than ax.plot: the ring is context, not data the view
        # should frame itself around -- ax.plot folds it into dataLim, so a small
        # spatial subset's axes would autoscale out to the whole model domain the
        # moment its (much larger) outline is drawn. add_artist skips that update
        # while still leaving the ring itself drawn and enumerable (ax.get_lines()
        # still finds it) -- the matplotlib equivalent of holoviews'
        # apply_ranges=False on the same overlay
        # (ocean_skill.plot.holoviews_renderer._domain_overlay).
        ax.add_artist(
            Line2D(
                ring[:, 0],
                ring[:, 1],
                transform=proj._as_mpl_transform(ax),
                color="k",
                lw=0.6,
                ls="--",
                zorder=4,
            )
        )
    # Always set_title, even to "": the point is not the text but the explicit ``y`` in
    # title_kwargs, which clears matplotlib's ``_autotitlepos`` and so skips the
    # automatic placement that goes infinite over a cartopy GeoAxes (see
    # DEFAULT_TITLE_KWARGS). An axes that never had set_title called keeps automatic
    # placement, reports a NaN tight bbox on matplotlib 3.11, and is dropped from
    # ``bbox_inches="tight"`` -- which is every panel below the top row of a two-axis
    # facet grid, where the label is deliberately None.
    ax.set_title(label or "", **title_kwargs)
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
    domain: tuple[float, float, float, float] | np.ndarray | None,
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
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
    robust: bool | float = False,
):
    """Draw one test|reference|difference row into three existing cartopy axes.

    ``seq_norm``/``div_norm``, if given, override the row's own colour limits —
    how :func:`field_grid`'s ``shared_limits=True`` makes every row share one
    scale instead of each computing its own. ``robust`` means what it does in
    :func:`_limits`, and is ignored once ``seq_norm`` is given.

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
    title_pinned = _pinned(title_kwargs, "title_kwargs")
    row_label_pinned = _pinned(row_label_kwargs, "row_label_kwargs")
    title_kwargs = _merged(defaults["title_kwargs"], title_kwargs)
    gridline_kwargs = _merged(defaults["gridline_kwargs"], gridline_kwargs)
    tick_label_kwargs = _merged(defaults["tick_label_kwargs"], tick_label_kwargs)
    row_label_kwargs = _merged(defaults["row_label_kwargs"], row_label_kwargs)
    metrics_kwargs = _merged(defaults["metrics_kwargs"], metrics_kwargs)

    t, r, d = aligned[test_name], aligned[reference_name], aligned["difference"]
    tl, rl = labels
    seq, div = cmaps_for(standard_name)
    if seq_norm is None:
        vmin, vmax = _limits(t, r, robust=robust)
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
                coastline_resolution=coastline_resolution,
                land=land,
            )
        )
        ax.title._osk_size_pinned = title_pinned

    if row_label:
        _add_row_label(axes[0], row_label, row_label_kwargs)
        axes[0]._osk_row_label._osk_size_pinned = row_label_pinned
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


def _draw_section_row(
    axes,
    values: dict[str, Any],
    geometry,
    *,
    labels: tuple[str, str],
    units: str | None,
    standard_name: str | None,
    metrics: dict[str, Any] | None,
    mark: str,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    title_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    shared_axis_labels: bool = True,
    scale: dict[str, float],
    defaults: dict[str, dict[str, Any]],
    robust: bool | float = False,
):
    """Draw one test|reference|difference section row into three existing axes.

    The section counterpart of :func:`_draw_row`: the same shared/symmetric colour
    norms and corner metrics box, but drawn as :func:`section` draws its one panel
    (grey facecolor for below-bathymetry/off-domain cells, positive-down depth with
    the y-axis inverted) rather than as a map — these are plain Cartesian axes, not
    cartopy ``GeoAxes``, so there is no gridliner, no coastline, no domain ring.

    ``values``/``geometry`` are :func:`ocean_skill.plot.section.prepare_section_row`'s
    own return, unpacked by the caller so this function stays a pure drawing step.
    ``robust`` means what it does in :func:`_limits`.
    """
    import matplotlib.colors as mcolors

    title_pinned = _pinned(title_kwargs, "title_kwargs")
    title_kwargs = _merged(defaults["title_kwargs"], title_kwargs)
    metrics_kwargs = _merged(defaults["metrics_kwargs"], metrics_kwargs)

    t, r, d = values["test"], values["reference"], values["difference"]
    tl, rl = labels
    seq, div = cmaps_for(standard_name)
    vmin, vmax = _limits(t, r, robust=robust)
    seq_norm = norm_for(standard_name, vmin, vmax)
    dmax = float(np.nanpercentile(np.abs(np.asarray(d)), 98)) or 1.0
    div_norm = mcolors.Normalize(vmin=-dmax, vmax=dmax)

    panels = [
        (t, tl, seq, seq_norm),
        (r, rl, seq, seq_norm),
        (d, "difference", div, div_norm),
    ]
    ims = []
    for j, (ax, (da, lab, cmap, norm)) in enumerate(zip(axes, panels, strict=True)):
        ax.set_facecolor("0.85")
        draw = ax.contourf if mark == "contourf" else ax.pcolormesh
        kw = {"levels": _contour_levels(norm)} if mark == "contourf" else {}
        im = draw(
            da[geometry.x_name], da[geometry.y_name], da, cmap=cmap, norm=norm, **kw
        )
        ax.invert_yaxis()
        ax.set_xlabel(geometry.x_label, fontsize=scale["axes_label"])
        # Only the leftmost panel labels depth -- the other two share the same axis,
        # the same convention _draw_row uses for latitude on a row of maps.
        if not shared_axis_labels or j == 0:
            ax.set_ylabel(geometry.y_label, fontsize=scale["axes_label"])
        ax.tick_params(axis="both", labelsize=scale["tick_label"])
        if shared_axis_labels and j != 0:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(lab, **title_kwargs)
        ax.title._osk_size_pinned = title_pinned
        ims.append(im)

    if metrics:
        axes[2]._osk_metrics_text = axes[2].text(
            0.02,
            0.02,
            _metrics_text(metrics, metric_keys),
            transform=axes[2].transAxes,
            zorder=5,
            **metrics_kwargs,
        )
    return ims, (f"[{units}]" if units else "")


def metric_value_text(metrics: dict[str, Any] | None, name: str) -> str:
    """Return one metric's value formatted for a label, or ``""`` if there isn't one.

    Bools are excluded rather than formatted: ``isinstance(True, int)`` is ``True`` in
    Python, so a naive numeric test renders the metric record's ``weighted`` flag as
    ``1`` — a plausible-looking number that is not a metric at all.

    Shared with the interactive renderer, which puts the same value in a panel title
    instead of a box, so the two cannot disagree about what a metric reads as.
    """
    value = (metrics or {}).get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return ""
    return f"{value:.3g}"


#: Where a statistics box sits inside its corner, in axes coordinates.
_CORNER_XY = {
    "upper left": (0.02, 0.98, "left", "top"),
    "upper right": (0.98, 0.98, "right", "top"),
    "lower left": (0.02, 0.02, "left", "bottom"),
    "lower right": (0.98, 0.02, "right", "bottom"),
}


def _x_axis(
    ax,
    scale: dict[str, float],
    tick_label_kwargs,
    *,
    date: bool = True,
    ticks: tuple[tuple[float, str], ...] | None = None,
) -> None:
    """Label a series/``time_depth`` panel's x axis: date, fixed groupby ticks, or plain.

    ``date=True`` (a real time axis, the ordinary case) installs
    ``ConciseDateFormatter`` so a date axis does not need the 45-degree tilt --
    ``fig.autofmt_xdate()``, the usual reflex, both rotates *and* hides all but
    the bottom axes in a way that fights an explicit ``sharex``.

    ``date=False`` is a time groupby's surviving axis instead (see
    :func:`ocean_skill.operators.time_axis_dim`) -- a bare integer index, not a
    date, so ``AutoDateLocator`` would read a month ``3`` as days since 1970.
    ``ticks`` -- ``((position, label), ...)`` from
    :func:`ocean_skill.plot.series.groupby_ticks` -- installs those exact
    labels (a ``month`` axis's ``Jan``..``Dec``) via a fixed locator/formatter
    pair; with no ``ticks`` (``year``, ``hour``, ...), a plain integer locator
    with no thousands-style offset keeps a year reading as ``2012`` rather
    than an axis-wide ``+2.012e3`` label.
    """
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    if date:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    elif ticks:
        ax.xaxis.set_major_locator(mticker.FixedLocator([pos for pos, _ in ticks]))
        ax.xaxis.set_major_formatter(
            mticker.FixedFormatter([label for _, label in ticks])
        )
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.tick_params(axis="both", labelsize=scale["tick_label"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if tick_label_kwargs:
            label.set(**tick_label_kwargs)


#: How a series panel draws a line. ``mark`` is a family-independent concept here (a map
#: family's marks are "pcolormesh"/"contourf"/"scatter"), which is why this family is
#: named `series` rather than `line`.
SERIES_MARKS = ("line", "line+marker", "marker", "step")


def _draw_series_lines(
    ax, lines, line_kwargs: dict[str, Any], *, mark: str = "line"
) -> list:
    """Draw one panel's lines, marking a subsample rather than every point."""
    from ocean_skill.plot.series import time_values
    from ocean_skill.plot.style import markevery_indices

    drawn = []
    for line in lines:
        values = line.spec.values
        kwargs: dict[str, Any] = {}
        wants_marker = mark in ("line+marker", "marker") or line.marker is not None
        if wants_marker:
            # Every sample marked is a filled band rather than a line, so a subsample is
            # marked -- the same indices bokeh gets, having no `markevery` of its own.
            kwargs = {
                "marker": line.marker or "o",
                "markevery": markevery_indices(int(values.sizes[values.dims[0]])),
                "markersize": 4,
            }
        if mark == "marker":
            kwargs["linestyle"] = "none"
        (artist,) = ax.plot(
            time_values(values),
            np.asarray(values.values, dtype="float64"),
            color=line.color,
            label=line.label,
            drawstyle="steps-mid" if mark == "step" else "default",
            **{"linestyle": line.linestyle, **kwargs},
            **line_kwargs,
        )
        drawn.append(artist)
    return drawn


def _metrics_box(ax, panel, metrics_kwargs: dict[str, Any]) -> None:
    """Put the statistics box in the corner :mod:`ocean_skill.plot.series` measured."""
    if not panel.metrics_text:
        return
    x, y, ha, va = _CORNER_XY.get(panel.metrics_corner, _CORNER_XY["upper left"])
    kwargs = {**metrics_kwargs, "ha": ha, "va": va}
    ax._osk_metrics_text = ax.text(
        x, y, panel.metrics_text, transform=ax.transAxes, zorder=5, **kwargs
    )


def _warn_if_overplotted(layout, canvas: Canvas | None) -> None:
    """Say so before drawing when a figure is being asked for more than it can show.

    Before rather than after, unlike :func:`_warn_if_cramped`: the counts are known from
    the layout, so there is no reason to spend the render first. Both still draw — the
    caller may be exporting at ``size="free"`` and know exactly what they asked for.
    """
    import warnings

    height = getattr(canvas, "max_height", None)
    if height is None or len(layout.panels) <= 1:
        return
    # Rows, not panel count: a wrapped grid stacks its rows vertically, so that is
    # what divides the canvas height. A single row falls back to the panel count,
    # matching today's (admittedly loose) per-panel budget for that shape.
    rows = layout.nrows if layout.nrows > 1 else len(layout.panels)
    per_panel = (height - SUPTITLE_ALLOWANCE) / rows
    if per_panel < 1.0:
        warnings.warn(
            f"{len(layout.panels)} panels on a canvas capped at {height:.1f}in leaves "
            f"{per_panel:.2f}in each, less than the labelling needs. Drawing anyway; "
            'size="free" lets the figure grow instead.',
            stacklevel=_stacklevel.find(),
        )


def series(
    items,
    *,
    title: str | None = None,
    rows: str | None = None,
    cols: str | None = None,
    secondary_y: bool = True,
    encode: dict[str, str | None] | None = None,
    residual: bool = False,
    metrics_loc: str = "auto",
    metrics_labels: Sequence[str] | None = None,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    mark: str = "line",
    legend: bool | str = True,
    line_labels: Sequence[str] | None = None,
    colors=None,
    ylim: tuple[float, float] | None = None,
    panel_aspect: float | None = None,
    labels: tuple[str, str] | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    font_scale: float = 1.0,
    fit_text: bool = True,
    sharex: bool = True,
    sharey: bool = False,
    ncols: int | None = None,
    nrows: int | None = None,
    wspace: float | None = None,
    hspace: float | None = None,
    title_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
):
    """Draw time series: one panel per group, both lanes of each comparison overlaid.

    The line counterpart of :func:`field_grid`, and the family a comparison whose lanes
    reduce to one time axis gets by default. ``reference`` is drawn solid and ``test``
    dashed — by *role*, so model-versus-model works the same way and reverses when the
    roles do — with colour carrying the variable and markers a varying depth. See
    :mod:`ocean_skill.plot.style` for the whole policy and ``encode=`` for overriding a
    channel.

    Composition follows the defaults in :func:`ocean_skill.plot.series.compose`: one
    variable overlays in a single panel, two put the second on a right-hand y axis
    (``secondary_y=False`` to stack them instead), three or more become one row each.
    ``rows=``/``cols=`` facet on ``variable``/``source``/``depth``/``comparison``
    instead; one or the other, not both. Faceting on ``variable`` also drops it from
    every legend entry -- the panel title already says it. ``ncols=``/``nrows=`` wrap
    the panels into a rectangular grid instead of the default single row/column --
    orthogonal to the facet choice, and refused together with ``residual=True``,
    whose strip only stacks in a single column.

    ``legend=`` is ``True``/``False`` for the usual auto/off, or a string for something
    more specific: ``"below"``/``"right"`` force one combined key outside the axes
    (deduplicated across every panel, whether or not they agree), and a corner name
    (``"upper left"``, ...) forces every panel's own key into that corner. ``"auto"``
    (the ``True`` default) draws the combined key below only when every panel's labels
    already agree, and otherwise one key per panel in whichever corner the data leaves
    emptiest. ``line_labels=`` overrides the legend text itself, one string per unique
    line in first-appearance order -- pass the wrong count and the ``ValueError`` lists
    the current labels, ready to copy and edit. ``colors=`` pins the auto colour cycle
    to specific values instead; see :func:`ocean_skill.plot.style.resolve`.

    Each panel's statistics box prefixes its rows automatically with whichever
    field(s) actually distinguish them -- the variable when a panel holds several,
    else depth/source/season/time when a panel's rows share one variable but differ
    some other way (three depth bands in one panel, say). ``metrics_labels=``
    overrides that prefix by hand, one string per metrics-box row figure-wide, panel
    by panel -- same wrong-count-lists-the-current-labels UX as ``line_labels=``; see
    :func:`ocean_skill.plot.series._resolve_metrics_labels`.

    ``residual=True`` adds a short ``test − reference`` strip under each panel, sharing
    its time axis. It is off by default: a difference *map* needs a panel of its own
    because it needs its own colour scale, while a difference *series* is a note on the
    panel above it, and drawing it always would double the axes on every figure.

    ``sharex=True`` (the default) links every panel's time axis, dropping inner tick
    labels the way a grid's shared edge normally does; ``sharey=False`` (the default)
    leaves each panel's value axis to its own data, since two panels are rarely the
    same quantity on the same scale the way a shared time axis is. Pass ``sharey=True``
    to line them up when they are (:func:`ocean_skill.plot.profile`'s twin, depth,
    defaults the other way -- shared, since every panel there reads the same axis).

    ``wspace``/``hspace`` tighten or loosen the gap between panels -- the fraction
    of a panel's own width/height ``constrained_layout`` reserves as a gutter
    (matplotlib's own names; default ``0.02`` each, left alone when unset).

    Sized like every other family — ``size``/``zoom``/``figsize``, type from geometry
    (:mod:`ocean_skill.plot.typography`) — with the statistics box placed in whichever
    corner the data leaves emptiest, since a line panel, unlike a map, does not fill
    its axes.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot import series as _series_layout
    from ocean_skill.plot.typography import (
        RESIDUAL_FRACTION,
        SERIES_ASPECT,
        SERIES_OVERHEAD,
        SERIES_PANEL_W_FRACTION,
    )

    if mark not in SERIES_MARKS:
        raise ValueError(
            f"mark={mark!r} is not a series mark; expected one of {SERIES_MARKS}. "
            '(A map family\'s marks -- "pcolormesh", "contourf", "scatter" -- draw a '
            "field, not a line.)"
        )
    if residual and sharey:
        raise ValueError(
            "sharey=True would share one y-axis between each panel's own value "
            "range and its residual strip's difference range below it -- not the "
            "same quantity. Drop residual=True, or leave sharey at its default."
        )
    layout = _series_layout.compose(
        items,
        rows=rows,
        cols=cols,
        secondary_y=secondary_y,
        encode=encode,
        residual=residual,
        metric_keys=metric_keys,
        metrics_loc=metrics_loc,
        metrics_labels=metrics_labels,
        legend=legend,
        line_labels=line_labels,
        colors=colors,
        ncols=ncols,
        nrows=nrows,
    )
    canvas = resolve_canvas(size, zoom)
    _warn_if_overplotted(layout, canvas)
    aspect = panel_aspect or SERIES_ASPECT
    figsize = figsize or auto_figsize(
        aspect,
        nrows=layout.nrows,
        ncols=layout.ncols,
        canvas=canvas,
        font_scale=font_scale,
        panel_w_fraction=SERIES_PANEL_W_FRACTION,
        overhead=SERIES_OVERHEAD,
    )
    # figure_ncols pins the *suptitle* to the reference grid: a one-column figure asking
    # for the figure base off its own cell gets a 17pt suptitle where every other figure
    # in the same report has 9. Same fix field_facet carries.
    scale = type_scale(
        figsize,
        ncols=layout.ncols,
        nrows=layout.nrows,
        font_scale=font_scale,
        figure_ncols=REFERENCE_GRID[0],
    )
    defaults = _style_defaults(scale, horizontal_colorbar=False)
    title_kwargs = _merged(defaults["title_kwargs"], title_kwargs)
    tick_label_kwargs = _merged(defaults["tick_label_kwargs"], tick_label_kwargs)
    metrics_kwargs = _merged(defaults["metrics_kwargs"], metrics_kwargs)
    suptitle_kwargs = _merged(defaults["suptitle_kwargs"], suptitle_kwargs)
    legend_kwargs = _merged(defaults["legend_kwargs"], legend_kwargs)
    line_kwargs = _merged(defaults["line_kwargs"], line_kwargs)

    if residual:
        # compose() refuses residual=True with more than one column, so this is
        # always a single stacked column with a short strip under every panel.
        heights = []
        for _ in layout.panels:
            heights.append(1.0)
            heights.append(RESIDUAL_FRACTION)
        fig, axes = plt.subplots(
            nrows=len(heights),
            ncols=1,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            gridspec_kw={"height_ratios": heights},
            layout="constrained",
        )
    else:
        fig, axes = plt.subplots(
            nrows=layout.nrows,
            ncols=layout.ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            layout="constrained",
        )
    _apply_subplot_spacing(fig, wspace=wspace, hspace=hspace)
    flat = list(axes.ravel())

    per_panel: list[tuple[Any, list]] = []
    for index, panel in enumerate(layout.panels):
        ax = flat[index * (2 if residual else 1)]
        handles = _draw_series_lines(ax, panel.lines, line_kwargs, mark=mark)
        per_panel.append((ax, handles))
        ax.set_title(
            panel.title, fontsize=scale["title"], **_without_font(title_kwargs)
        )
        ax.set_ylabel(panel.ylabel, fontsize=scale["axes_label"])
        if panel.ylabel_color:
            ax.yaxis.label.set_color(panel.ylabel_color)
            ax.tick_params(axis="y", labelcolor=panel.ylabel_color)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if panel.secondary:
            twin = ax.twinx()
            handles += _draw_series_lines(twin, panel.secondary, line_kwargs, mark=mark)
            per_panel[-1] = (ax, handles)
            twin.set_ylabel(panel.secondary_ylabel or "", fontsize=scale["axes_label"])
            twin.tick_params(labelsize=scale["tick_label"])
            if panel.secondary_ylabel_color:
                twin.yaxis.label.set_color(panel.secondary_ylabel_color)
                twin.tick_params(axis="y", labelcolor=panel.secondary_ylabel_color)
        _metrics_box(ax, panel, metrics_kwargs)
        _x_axis(ax, scale, tick_label_kwargs, date=layout.date_axis, ticks=layout.xticks)
        if panel.residual:
            strip = flat[index * 2 + 1]
            _draw_series_lines(strip, panel.residual, line_kwargs, mark=mark)
            strip.axhline(0.0, color="0.7", linewidth=0.7, zorder=1)
            # Spelled exactly as field_grid labels its difference colorbar, so the two
            # families name the same quantity the same way.
            units = panel.ylabel.partition("[")[2].rstrip("]")
            # Wrapped, not one line: the strip is a third of a panel high, and a rotated
            # label of this length on it is either shrunk to nothing or clipped.
            strip.set_ylabel(
                "test − reference" + (f"\n[{units}]" if units else ""),
                fontsize=scale["axes_label"],
            )
            _x_axis(
                strip, scale, tick_label_kwargs, date=layout.date_axis, ticks=layout.xticks
            )

    n_panels = len(layout.panels)
    if layout.ncols == 1 or (layout.nrows == 1 and layout.ncols == n_panels):
        flat[-1].set_xlabel(layout.xlabel, fontsize=scale["axes_label"])
    else:
        # A wrapped grid's bottom row is ragged when n_panels does not fill it, so
        # "is there a panel below me?" is the question, not "am I in the last
        # row?" -- field_facet's own rule for the same situation. sharex=True
        # otherwise hides tick labels on every row but the last, which would
        # leave a panel sitting above a hidden blank cell with no dates at all.
        for index, ax in enumerate(flat[:n_panels]):
            if index + layout.ncols >= n_panels:
                ax.set_xlabel(layout.xlabel, fontsize=scale["axes_label"])
                ax.xaxis.set_tick_params(labelbottom=True)
        for ax in flat[n_panels:]:
            ax.set_visible(False)
    if title:
        fig.suptitle(title, **suptitle_kwargs)
    if layout.legend_placement != "off":
        _series_legend(fig, per_panel, layout, scale, legend_kwargs)
    _warn_if_cramped(
        fig,
        ncols=layout.ncols,
        canvas=canvas,
        nrows=layout.nrows,
        panels=[ax for ax in flat if ax.lines],
    )
    if fit_text:
        _fit_text_widths(fig)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


#: How a profile panel draws a line. No ``"step"`` -- unlike a time axis (whose
#: instantaneous or period-mean values a step honestly represents as holding until
#: the next sample), a profile's levels are irregularly spaced model or instrument
#: depths with nothing between them the value legitimately holds at.
PROFILE_MARKS = ("line", "line+marker", "marker")


def _draw_profile_lines(
    ax, lines, line_kwargs: dict[str, Any], *, mark: str = "line"
) -> list:
    """Draw one panel's lines: value on x, depth on y -- the transpose of
    :func:`_draw_series_lines`, marking the same subsample the same way.

    A mean±spread envelope, when a line carries one, draws first in its own
    pass -- a horizontal fill (:func:`~matplotlib.axes.Axes.fill_betweenx`,
    value on x, depth on y, matching the panel's own orientation) in every
    line's own colour, split into :func:`~ocean_skill.plot.style.band_runs`'
    contiguous finite runs. Not appended to ``drawn``: a band earns no legend
    entry of its own, and matplotlib's default z-order (collections below
    lines) already puts it beneath every line regardless.
    """
    from ocean_skill.plot.profile import vertical_values
    from ocean_skill.plot.style import BAND_ALPHA, band_runs, markevery_indices

    for line in lines:
        if line.spec.spread is None:
            continue
        depth = vertical_values(line.spec.values)
        values = np.asarray(line.spec.values.values, dtype="float64")
        for axis, lo, hi in band_runs(depth, values, line.spec.spread):
            ax.fill_betweenx(
                axis, lo, hi, color=line.color, alpha=BAND_ALPHA, linewidth=0
            )

    drawn = []
    for line in lines:
        values = line.spec.values
        depth = vertical_values(values)
        kwargs: dict[str, Any] = {}
        finite = np.isfinite(np.asarray(values.values, dtype="float64")) & np.isfinite(
            depth
        )
        wants_marker = (
            mark in ("line+marker", "marker")
            or line.marker is not None
            # A single-depth cast (a ragged timeSeriesProfile station's own
            # single-bottle visit, or an already-collapsed profile) has nothing
            # for a *line* to connect -- one point under the default mark="line"
            # draws a zero-length segment, invisible. Marked regardless of `mark`
            # so the cast is not silently dropped from the panel.
            or int(np.count_nonzero(finite)) < 2
        )
        if wants_marker:
            kwargs = {
                "marker": line.marker or "o",
                "markevery": markevery_indices(depth.size),
                "markersize": 4,
            }
        if mark == "marker":
            kwargs["linestyle"] = "none"
        (artist,) = ax.plot(
            np.asarray(values.values, dtype="float64"),
            depth,
            color=line.color,
            label=line.label,
            **{"linestyle": line.linestyle, **kwargs},
            **line_kwargs,
        )
        drawn.append(artist)
    return drawn


def profile(
    items,
    *,
    title: str | None = None,
    rows: str | None = None,
    cols: str | None = None,
    secondary_x: bool = True,
    encode: dict[str, str | None] | None = None,
    metrics_loc: str = "auto",
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    metrics_stacked: bool = False,
    metrics_labels: Sequence[str] | None = None,
    colors=None,
    mark: str = "line",
    legend: bool | str = True,
    line_labels: Sequence[str] | None = None,
    titles: Sequence[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    panel_aspect: float | None = None,
    labels: tuple[str, str] | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    font_scale: float = 1.0,
    fit_text: bool = True,
    sharex: bool = False,
    sharey: bool = True,
    ncols: int | None = None,
    nrows: int | None = None,
    wspace: float | None = None,
    hspace: float | None = None,
    title_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
):
    """Draw vertical profiles: value on x, depth on y, surface at the top.

    The vertical counterpart of :func:`series` — a station's compared value read
    down the water column rather than through time — built from the same
    :mod:`ocean_skill.plot.style` policy: ``reference`` solid, ``test`` dashed by
    *role*, colour carrying the variable, and now a varying *cast* (``marker <-
    time``) where ``series`` marks a varying depth, since here depth is the axis
    itself rather than something to distinguish lines by. See
    :mod:`ocean_skill.plot.profile` for the whole policy and ``encode=`` for
    overriding a channel (``depth`` is refused there: it names the axis, not a
    channel).

    Composition follows :func:`ocean_skill.plot.profile.compose`'s defaults: one
    variable overlays every source/cast in a single panel; two variables merge
    onto that one panel too, the second drawn against a *top* x axis
    (``secondary_x`` — the profile twin of :func:`series`' ``secondary_y``,
    transposed because a profile's value axis is x rather than y);
    ``secondary_x=False`` gives each variable its own column instead, sharing
    the one depth axis (the standard CTD layout); three or more variables always
    fall back to one column each. ``rows=``/``cols=`` facet on
    ``variable``/``source``/``reference``/``time``/``comparison`` instead of the
    default; one or the other, not both, and a facet wins over ``secondary_x``.
    ``ncols=``/``nrows=`` wrap the resulting panels into a rectangular grid
    instead of the default single row/column -- orthogonal to the facet choice,
    which only decides what goes in each panel.

    ``legend=``/``line_labels=`` match :func:`series` exactly: ``True``/``False``
    for the usual auto/off, ``"below"``/``"right"`` for one combined key, a corner
    name to force every panel's own key there, and ``line_labels=`` to override the
    legend text itself -- see :func:`series`' own docstring for the full rule.

    A ``cols="comparison"``/``rows="comparison"`` facet (one panel per station)
    auto-promotes the station into its panel's title and drops it from that
    line's legend entry, since the title now already says it; ``titles=``
    overrides the result by hand afterward, one string per panel in panel
    order -- see :func:`ocean_skill.plot.profile.compose`.

    ``xlim`` bounds the (primary) value axis; with ``secondary_x`` merging a
    second variable in, it bounds only the bottom axis, the same rule ``ylim``
    follows for :func:`series`' twin. ``ylim`` bounds depth in the same
    positive-down metres every profile draws in — ``(shallow, deep)``, e.g.
    ``(0, 200)`` — not axes order, since the axis is always inverted regardless
    of what is passed.

    ``sharey=True`` (the default) reads every panel down the same depth range,
    since a grid of casts is usually meant to compare like-for-like; pass
    ``sharey=False`` to let each panel autoscale to its own deepest sample --
    a shallow station no longer inherits a deep neighbour's mostly-empty axis.
    ``sharex=False`` (the default) leaves the value axis per-panel, the same as
    :func:`series`' value axis; pass ``sharex=True`` to line panels up on it
    when they share the one quantity.

    ``metrics_stacked=True`` draws the statistics box narrow-and-tall (one metric
    per line) instead of the default single wide line -- fits a narrow, portrait
    panel that a page-wide box would otherwise overrun into its neighbours; see
    :func:`ocean_skill.plot.series._metrics_text`. ``metrics_labels=`` overrides
    each row's automatic prefix by hand, matching :func:`series` exactly -- see
    :func:`ocean_skill.plot.series._resolve_metrics_labels`. Panel width itself is not the
    box's doing either way -- it comes from ``panel_aspect``/``figsize`` divided
    across ``ncols``, the same as any other panel dimension; ``panel_aspect``
    only ever solves for figure *height* (this renderer sets no axes aspect), so
    raising it will not widen a cramped grid.

    ``wspace``/``hspace`` tighten or loosen the gap between panels -- the fraction
    of a panel's own width/height ``constrained_layout`` reserves as a gutter
    (matplotlib's own names; default ``0.02`` each, left alone when unset). A
    ``sharey=False`` grid draws its own depth tick numbers on every panel, which
    already sets a floor under how tight ``wspace`` can pull columns together.

    ``colors=`` pins the auto colour cycle to specific values instead; see
    :func:`ocean_skill.plot.style.resolve`. A band's fill follows for free.

    Sized like every other line family — ``size``/``zoom``/``figsize``, type from
    geometry (:mod:`ocean_skill.plot.typography`) — with the statistics box placed
    in whichever corner the data leaves emptiest, since a profile panel, unlike a
    map, does not fill its axes.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot import profile as _profile_layout
    from ocean_skill.plot.typography import (
        PROFILE_ASPECT,
        SERIES_OVERHEAD,
        SERIES_PANEL_W_FRACTION,
    )

    if mark not in PROFILE_MARKS:
        raise ValueError(
            f"mark={mark!r} is not a profile mark; expected one of {PROFILE_MARKS}. "
            '("step" belongs to a value that holds between samples, which a '
            "profile's irregularly spaced levels are not.)"
        )
    layout = _profile_layout.compose(
        items,
        rows=rows,
        cols=cols,
        secondary_x=secondary_x,
        encode=encode,
        metric_keys=metric_keys,
        metrics_loc=metrics_loc,
        metrics_stacked=metrics_stacked,
        metrics_labels=metrics_labels,
        colors=colors,
        legend=legend,
        line_labels=line_labels,
        titles=titles,
        ncols=ncols,
        nrows=nrows,
    )
    canvas = resolve_canvas(size, zoom)
    _warn_if_overplotted(layout, canvas)
    aspect = panel_aspect or PROFILE_ASPECT
    figsize = figsize or auto_figsize(
        aspect,
        nrows=layout.nrows,
        ncols=layout.ncols,
        canvas=canvas,
        font_scale=font_scale,
        panel_w_fraction=SERIES_PANEL_W_FRACTION,
        overhead=SERIES_OVERHEAD,
    )
    scale = type_scale(
        figsize,
        ncols=layout.ncols,
        nrows=layout.nrows,
        font_scale=font_scale,
        figure_ncols=REFERENCE_GRID[0],
    )
    defaults = _style_defaults(scale, horizontal_colorbar=False)
    title_kwargs = _merged(defaults["title_kwargs"], title_kwargs)
    tick_label_kwargs = _merged(defaults["tick_label_kwargs"], tick_label_kwargs)
    metrics_kwargs = _merged(defaults["metrics_kwargs"], metrics_kwargs)
    suptitle_kwargs = _merged(defaults["suptitle_kwargs"], suptitle_kwargs)
    legend_kwargs = _merged(defaults["legend_kwargs"], legend_kwargs)
    line_kwargs = _merged(defaults["line_kwargs"], line_kwargs)

    fig, axes = plt.subplots(
        nrows=layout.nrows,
        ncols=layout.ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        layout="constrained",
    )
    _apply_subplot_spacing(fig, wspace=wspace, hspace=hspace)
    flat = list(axes.ravel())

    # One depth range for the whole figure when sharey -- explicit set_ylim on
    # every axis rather than relying on invert_yaxis()'s toggle state, which
    # sharey=True would otherwise flip twice on every axis but the first. With
    # sharey=False each panel instead gets its own range from its own lines,
    # inside the loop below.
    shared_depth = None
    if sharey:
        all_lines = [
            line for panel in layout.panels for line in panel.lines + panel.secondary
        ]
        shared_depth = _profile_layout.depth_range(all_lines, ylim=ylim)

    per_panel: list[tuple[Any, list]] = []
    for index, panel in enumerate(layout.panels):
        ax = flat[index]
        if panel.blank:
            # An empty cell in a two-axis rows=/cols= grid (see Panel.blank) --
            # hidden exactly like a trailing cell past the panel count below,
            # just interior rather than trailing.
            ax.set_visible(False)
            per_panel.append((ax, []))
            continue
        handles = _draw_profile_lines(ax, panel.lines, line_kwargs, mark=mark)
        per_panel.append((ax, handles))
        panel_title_kwargs = _without_font(title_kwargs)
        if panel.secondary and "pad" not in panel_title_kwargs:
            # A twin's own ticks and axis label are about to be drawn above the
            # shared top spine -- exactly where the title's default pad would
            # otherwise land it. Clear them by the twin's own vertical extent
            # (tick length + tick labels + axis label, in points) before the
            # constrained-layout engine ever measures either.
            panel_title_kwargs = {
                **panel_title_kwargs,
                "pad": 17 + 1.2 * (scale["tick_label"] + scale["axes_label"]),
            }
        ax.set_title(panel.title, fontsize=scale["title"], **panel_title_kwargs)
        # The label reads the same on every panel ("salinity", "Depth [m]") --
        # only the grid's outer edge needs to say so once. "Is there a panel
        # directly below/left of me?" is the question (a wrapped grid's last
        # row can be ragged), not "am I in the last row/first column?" -- the
        # same rule field_facet and series() already use for this. Tick
        # *numbers* are a different thing and stay on every panel regardless.
        if index + layout.ncols >= len(layout.panels):
            ax.set_xlabel(panel.xlabel or "", fontsize=scale["axes_label"])
        if panel.xlabel_color:
            ax.xaxis.label.set_color(panel.xlabel_color)
            ax.tick_params(axis="x", labelcolor=panel.xlabel_color)
        if layout.ncols == 1 or index % layout.ncols == 0:
            ax.set_ylabel(panel.ylabel, fontsize=scale["axes_label"])
        if xlim is not None:
            ax.set_xlim(*xlim)
        y_bottom, y_top = shared_depth or _profile_layout.depth_range(
            panel.lines + panel.secondary, ylim=ylim
        )
        ax.set_ylim(y_bottom, y_top)
        ax.tick_params(axis="both", labelsize=scale["tick_label"])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            if tick_label_kwargs:
                label.set(**tick_label_kwargs)
        if panel.secondary:
            # A top x axis, not twinx()'s right-hand y: a profile's value axis is
            # x, so its twin grows the same way series' grows a twin y -- placed
            # after set_ylim. With sharey=True the twin inherits (y_bottom, y_top)
            # by sharing the parent's y axis; with sharey=False there is no shared
            # axis to inherit from, so it is set explicitly instead.
            twin = ax.twiny()
            handles += _draw_profile_lines(twin, panel.secondary, line_kwargs, mark=mark)
            per_panel[-1] = (ax, handles)
            if not sharey:
                twin.set_ylim(y_bottom, y_top)
            twin.set_xlabel(panel.secondary_xlabel or "", fontsize=scale["axes_label"])
            twin.tick_params(labelsize=scale["tick_label"])
            if panel.secondary_xlabel_color:
                twin.xaxis.label.set_color(panel.secondary_xlabel_color)
                twin.tick_params(axis="x", labelcolor=panel.secondary_xlabel_color)
        _metrics_box(ax, panel, metrics_kwargs)

    # A grid wider or taller than there are panels leaves trailing cells blank --
    # hidden rather than removed, so the rest of the grid keeps the shape it was
    # sized for (field_facet's own rule for the same situation).
    for ax in flat[len(layout.panels) :]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, **suptitle_kwargs)
    if layout.legend_placement != "off":
        _series_legend(fig, per_panel, layout, scale, legend_kwargs)
    _warn_if_cramped(
        fig,
        ncols=layout.ncols,
        canvas=canvas,
        nrows=layout.nrows,
        panels=[ax for ax in flat if ax.lines],
    )
    if fit_text:
        _fit_text_widths(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def _without_font(kwargs: dict[str, Any]) -> dict[str, Any]:
    """``kwargs`` without its font-size key, which is passed explicitly."""
    return {k: v for k, v in kwargs.items() if k not in ("fontsize", "size")}


def _series_legend(fig, per_panel, layout, scale, legend_kwargs) -> None:
    """Draw the key(s) ``layout.legend_placement`` asks for.

    ``"below"``/``"right"`` (forced, or "auto" once every panel's labels happen to
    agree) draw one combined key outside the axes; a shared key is also the one thing
    bokeh cannot do (it has no figure-level legend) — the stated divergence for this
    family. The *entries* are identical in both renderers either way; only their
    placement is not.

    Anything else draws one key per panel, in ``panel.legend_corner`` — a forced
    corner is already baked in there by :func:`ocean_skill.plot.series.compose`, so
    this function does not need to know the difference. Per-panel keys carry that
    panel's own lines, not the figure's: with one variable per panel, a shared key
    would list every variable under each of them.

    A blank or underscore-prefixed label (:mod:`ocean_skill.plot.profile`'s own
    auto station-title drops a line's label entirely once its identity moved to
    the panel title) is left out here too, matching matplotlib's own "don't
    legend this" convention -- unlike :meth:`Axes.legend`'s *implicit* handle
    discovery, an *explicit* ``(handles, labels)`` call like this one draws
    whatever it is given, blank rows included, so the filter has to be explicit.
    """
    from ocean_skill.plot.summary import _legend_below, _legend_right

    def _labelled(handle) -> bool:
        label = handle.get_label()
        return bool(label) and not label.startswith("_")

    auto_shared = layout.shared_legend and len(layout.panels) > 1
    combined = layout.legend_placement == "below" or (
        layout.legend_placement == "auto" and auto_shared
    )
    if combined or layout.legend_placement == "right":
        seen: dict[str, Any] = {}
        for _, handles in per_panel:
            for handle in filter(_labelled, handles):
                seen.setdefault(handle.get_label(), handle)
        if seen:
            right = layout.legend_placement == "right"
            draw = _legend_right if right else _legend_below
            draw(fig, list(seen.values()), scale["legend"])
        return
    for (ax, handles), panel in zip(per_panel, layout.panels, strict=True):
        handles = list(filter(_labelled, handles))
        if not handles:
            continue
        seen = {}
        for handle in handles:
            seen.setdefault(handle.get_label(), handle)
        # An explicit corner, not loc="best": "best" minimises overlap with the *data*
        # and knows nothing about the statistics box, which is how the two came to be
        # drawn on top of each other. compose() ranks the corners and hands out two.
        ax.legend(
            list(seen.values()),
            list(seen.keys()),
            loc=panel.legend_corner,
            fontsize=scale["legend"],
            **legend_kwargs,
        )


def _metrics_text(metrics: dict[str, Any] | None, metric_keys) -> str:
    """Return the corner box's text: one ``key=value`` line per requested metric.

    Its own function because :func:`field_movie` rewrites the box every frame and the
    two spellings of "what the box says" must not drift apart.
    """
    if not metrics:
        return ""
    return "\n".join(
        f"{key}={text}"
        for key in metric_keys
        if (text := metric_value_text(metrics, key))
    )


def _warn_if_interactive_only(rasterize, hover) -> None:
    """Warn that ``rasterize``/``hover`` are the interactive renderer's, not this one.

    Accepted here only so ``renderer="both"`` can pass one option set to each renderer
    — the same accommodation :func:`locations` makes for ``tiles``. Bokeh needs
    ``rasterize`` to avoid a per-cell Python loop on a large curvilinear mesh and
    ``hover`` to draw a readout tool; matplotlib's ``pcolormesh`` is vectorized
    regardless of mesh size and has no interactive readout to switch on, so neither
    option has anything to do here. To shrink a static figure's draw time, thin the
    data itself (``every=``, a coarser ``aggregate=``) rather than the mesh's rendering.
    """
    import warnings

    given = (("rasterize", rasterize), ("hover", hover))
    passed = [name for name, value in given if value is not None]
    if passed:
        warnings.warn(
            f"{passed} only affect the interactive renderer and have no effect here "
            "— pass renderer='holoviews' for them to apply.",
            stacklevel=_stacklevel.find(),
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
    depth: str | None = None,
    time: str | None = None,
    region: str | None = None,
    metrics: dict[str, Any] | None = None,
    mark: str = "pcolormesh",
    save: str | Path | None = None,
    domain: tuple[float, float, float, float] | np.ndarray | None = None,
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
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
    robust: bool | float = False,
):
    """Draw one ``test | reference | difference`` row for a gridded comparison.

    Sized automatically: ``size`` names the canvas the figure has to fit (``"page"`` by
    default — see :data:`~ocean_skill.plot.typography.CANVASES` for ``"slide"``,
    ``"free"``, ``"column"``, or pass a width in inches / a ``(width, max_height)``
    pair), and ``zoom`` multiplies it. The height then follows the maps' own aspect
    ratio plus the room the type needs, so ``zoom=1.5`` gives a figure half again as
    large with panels and type to match rather than a ``figsize`` you had to work out.
    ``figsize``
    overrides both outright. ``metric_keys`` picks which of ``metrics.compute()``'s
    values appear in the corner box (default ``bias``/``rmse``/``corr``) — any
    subset/order, e.g. ``metric_keys=("corr", "sigma_ratio")``.

    Every font size — panel titles, the suptitle, latitude/longitude labels, the
    colorbars' labels and their tick labels, the metrics box — is derived from the
    figure's geometry by :func:`~ocean_skill.plot.typography.type_scale`, so a canvas
    half the default gets type to match rather than three maps crushed to a third of an
    inch by titles that no longer fit. ``font_scale`` multiplies all of them together
    (``font_scale=1.2`` for "the same figure, larger type", as against ``zoom``, which
    grows the figure and lets type follow); a size passed explicitly in a ``*_kwargs``
    dict overrides outright, and is exempt from the ``fit_text`` pass below.

    ``fit_text=True`` (the default) measures the drawn labels and shrinks any *single*
    one still too long for the box it labels — a 50-character CF standard name in a
    2-inch panel, which no choice of scale can accommodate. Sizes you set yourself are
    never shrunk. Set ``False`` to leave every label at its nominal size and let long
    ones overhang.

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

    ``title`` defaults, when not given, to the variable this comparison names followed
    by the depth, time and region a ``select=`` has collapsed to one map —
    ``chlorophyll · 0–10 m · 2010-01-22 · 45–55°N, 165°E–155°W`` — built through the
    same :func:`suptitle_text` a one-field figure uses (:func:`field_suptitle`), so a
    comparison and a plain field name the same quantity the same way. A single row has
    no left-edge row label to carry the variable (that is :func:`field_grid`'s doing,
    and only when it stacks several), so without this the figure said only *which
    sources*, never *what*. Pass ``title=""`` to drop it, or any string to replace it.

    ``rasterize``/``hover`` are accepted only so ``renderer="both"`` can pass one option
    set to each renderer (see :func:`_warn_if_interactive_only`) — they are the
    interactive renderer's fix for a large mesh and do nothing here.

    ``coastline_resolution`` (see :mod:`ocean_skill.plot.coastline`) picks the
    coastline/land dataset — ``"auto"`` (the default) scales Natural Earth to each
    panel's extent, ``"110m"``/``"50m"``/``"10m"`` pin a Natural Earth scale, and
    ``"coarse"``..``"full"`` draw from GSHHS, finer than Natural Earth's own limit but
    a one-time download the first time a given scale is used.

    ``land`` controls the grey land fill, which otherwise paints over any data drawn
    under it — the usual reason to reach for this is data hugging or crossing the
    coastline. ``True`` (the default) is today's opaque fill, a float in ``[0, 1]``
    fades it to that opacity while keeping the coastline outline, and ``False`` draws
    neither fill nor outline.

    ``robust`` means what it does in :func:`_limits`: the sequential colour scale
    (test/reference) spans the full range of the pair by default, or its 10th–90th
    percentile with ``robust=True`` — useful when a few outlier cells would
    otherwise crush the rest of the map against one end of the bar.
    """
    import matplotlib.pyplot as plt

    _warn_if_interactive_only(rasterize, hover)
    if title is None:
        title = suptitle_text(standard_name, (depth, time, region))

    # Horizontal bars sit below the maps and so come out of the row's height; vertical
    # ones sit beside and come out of its width. Which it is has to be settled before
    # sizing, not after, and the same answer used for both the allowance and the bars.
    aspect = _map_aspect([{"aligned": aligned}], reference_name)
    horizontal = colorbar_is_horizontal(
        aspect,
        default_horizontal=True,  # one row: bars below, panels get the full cell width
        requested=(colorbar_kwargs or {}).get("orientation"),
    )
    figsize = figsize or auto_figsize(
        aspect,
        nrows=1,
        canvas=resolve_canvas(size, zoom),
        font_scale=font_scale,
        horizontal_colorbar=horizontal,
    )
    scale = _scale_for(figsize, nrows=1, font_scale=font_scale)
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        subplot_kw={"projection": _map_projection(aligned)},
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
        coastline_resolution=coastline_resolution,
        land=land,
        robust=robust,
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
        sup = fig.suptitle(
            title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs)
        )
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    _fit_left_margin(fig)
    if align_colorbars:
        _align_colorbars(fig)
    # after alignment, which is what finally decides how long each colorbar is
    if fit_text:
        _fit_text_widths(fig)
        # a shrunken row label is a *narrower* one, and its placement was measured
        # against the old width — re-place it against what is actually drawn now
        _clear_row_labels(fig)
    _warn_if_cramped(fig, canvas=resolve_canvas(size, zoom), nrows=1)
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

    A size the caller set explicitly is left alone (see :func:`_pinned`): they asked for
    18pt, and quietly serving 5pt because the string is long would make ``*_kwargs`` an
    advisory rather than an override.
    """
    if getattr(text, "_osk_size_pinned", False):
        return
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

    Runs **after** :func:`_align_colorbars`, and that ordering is the point rather
    than a detail. Alignment shortens each bar to the panels it describes, so measuring
    a vertical bar's rotated label before that happens compares it against a box it will
    not end up in: the label came out too big for the final bar and was clipped by the
    figure edge, which is exactly what raising the type level made visible.
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
        # axis labels run along their own axis; the y label is rotated, so its height
        # is its length. Summary diagrams are where these bite -- a target diagram's
        # "signed centred RMSD / σ_ref" label is longer than its own axes.
        _shrink_to_fit(ax.xaxis.label, box.width * _TEXT_FIT_FRACTION, renderer)
        _shrink_to_fit(
            ax.yaxis.label, box.height * _TEXT_FIT_FRACTION, renderer, along="height"
        )
        if (row_label := getattr(ax, "_osk_row_label", None)) is not None:
            # rotated 90 degrees up the left edge of the row: its height is its length
            _shrink_to_fit(
                row_label, box.height * _TEXT_FIT_FRACTION, renderer, along="height"
            )


def _centre_suptitle(fig, renderer=None) -> None:
    """Centre the suptitle over the panels rather than over the canvas.

    matplotlib centres a suptitle on the *figure*, which is the same thing as centring
    it over the panels only when the panels fill the figure's width. A tall facet grid
    is where they do not: one narrow column of maps beside a vertical colorbar on a
    page-width canvas leaves the drawn block well right of centre, and a title at
    ``x=0.5`` sits off in the margin to the left of everything it names.

    Measured after the layout has settled, since that is when the panels are where they
    will be drawn. Colorbar axes are left out of the span deliberately — the title names
    the field, and the bar is scenery beside it — and a figure whose panels *are*
    centred lands back on 0.5, so this costs the common case nothing but a measurement.
    """
    sup = getattr(fig, "_suptitle", None)
    if sup is None or not sup.get_text():
        return
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    boxes = [
        ax.get_window_extent(renderer)
        for ax in fig.axes
        if ax.get_visible() and not getattr(ax, "_osk_cbar_parents", None)
    ]
    if not boxes:  # pragma: no cover - a figure with nothing but a colorbar
        return
    width = fig.get_size_inches()[0] * fig.dpi
    centre = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2 / width
    if np.isfinite(centre):
        sup.set_x(float(centre))


def _aspect_of(da) -> float:
    """Return one field's ``lon_span / lat_span`` — the shape a panel wants to be."""
    try:
        lon_span = float(np.ptp(np.asarray(da["lon"])))
        lat_span = float(np.ptp(np.asarray(da["lat"])))
        return lon_span / max(lat_span, 1e-6)
    except Exception:  # pragma: no cover - fall back to a square-ish panel
        return 1.0


#: Share of its grid cell a panel keeps before the figure is reported as too small for
#: its content. Well below the ~0.73 a healthy layout gives (``PANEL_W_FRACTION`` in
#: typography), so this fires only when the maps really have been crowded out.
_CRAMPED_PANEL_FRACTION = 0.5


def _warn_if_cramped(
    fig,
    ncols: int = 3,
    *,
    canvas: Canvas | None = None,
    nrows: int = 1,
    panels=None,
) -> None:
    """Say so when the canvas cannot hold its own labelling, and which knob to turn.

    Type is sized from the geometry and floored at ``MIN_PT``, so once the floor binds
    the figure runs out of moves: ``constrained_layout`` gives text priority over axes,
    and the maps take the difference. It still *draws*, which is the problem: the panels
    quietly become slivers and nothing says why.

    Two different constraints produce that, and they want opposite advice, so the
    message names the one that actually bound:

    * **too narrow** — three maps plus their labelling need roughly five inches of
      width; below that, widen the canvas.
    * **height-capped with many rows** — the figure is as tall as its canvas allows and
      the rows are splitting what is left. Widening does nothing here; lifting the cap
      (``size="free"``) or drawing fewer rows per figure does.

    ``panels`` names the axes to measure, for a family whose panels are not maps.
    """
    import warnings

    fig_w, fig_h = fig.get_size_inches()
    # A map panel is identified by its projection; a line panel has none, so a family
    # whose panels are not maps has to say which axes to measure -- without that this
    # returned silently and the cap went unenforced for it.
    if panels is None:
        panels = [ax for ax in fig.axes if hasattr(ax, "projection")]
    if not panels or fig_w <= 0:
        return
    cell_w = fig_w / max(ncols, 1)
    widest = max(ax.get_position().width * fig_w for ax in panels)
    if widest >= _CRAMPED_PANEL_FRACTION * cell_w:
        return

    cap = getattr(canvas, "max_height", None)
    # row_height() holds SUPTITLE_ALLOWANCE back from the cap, so a figure that has hit
    # it is that much shorter than max_height rather than equal to it
    capped = cap is not None and nrows > 1 and fig_h >= cap - SUPTITLE_ALLOWANCE - 0.05
    remedy = (
        'lift the height cap (size="free") or draw fewer rows per figure'
        if capped
        else "widen the canvas (size=, zoom= or figsize=)"
    )
    reason = (
        f"{nrows} rows are sharing a canvas capped at {cap:.1f}in"
        if capped
        else f"a {fig_w:.2f}in canvas leaves too little width"
        " once the titles, colorbars and coordinate labels have taken theirs"
    )
    warnings.warn(
        f"the maps are only {widest:.2f}in wide: {reason}, and the type is already at "
        f"its {MIN_PT:g}pt floor, so there is no room left to reclaim. Rather than "
        f"shrinking the text, {remedy}.",
        stacklevel=3,
    )


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


def _row_height(
    comparisons,
    reference_name: str,
    n: int,
    font_scale: float = 1.0,
    canvas: Canvas | None = None,
    horizontal_colorbar: bool = False,
):
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
        canvas=canvas or resolve_canvas(),
        font_scale=font_scale,
        horizontal_colorbar=horizontal_colorbar,
    )


def _shared_norms(
    comparisons, test_name: str, reference_name: str, *, robust: bool | float = False
):
    """Return one ``(seq_norm, div_norm)`` computed across *every* row.

    For ``field_grid(..., shared_limits=True)``: colour limits derived from all
    rows' test+reference values combined, and one difference range from all rows'
    diffs — rather than each row scaling to its own data. ``robust`` means what it
    does in :func:`_limits`.
    """
    import matplotlib.colors as mcolors

    standard_name = comparisons[0].get("standard_name")
    all_t = [np.asarray(c["aligned"][test_name]) for c in comparisons]
    all_r = [np.asarray(c["aligned"][reference_name]) for c in comparisons]
    vmin, vmax = _limits(*all_t, *all_r, robust=robust)
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
    domain: tuple[float, float, float, float] | np.ndarray | None = None,
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
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
    robust: bool | float = False,
):
    """Stack one ``test | reference | difference`` row per comparison.

    Each item is a dict with ``aligned`` and optionally ``row_label``, ``units``,
    ``standard_name``, ``metrics`` and ``labels``. Every row gets its own colour
    scales by default (variables have different ranges), its own colorbars, and its
    own column titles from its own ``labels`` — rows commonly come from *different*
    reference sources in a ``compare()`` fan-out (nitrate from one WOA entry,
    phosphate from another), so reusing one shared pair of titles for every row
    would mislabel all but the first. The top-level ``labels`` is only the fallback
    for a row that doesn't carry its own. ``metric_keys`` picks which of
    ``metrics.compute()``'s values appear in each row's corner box (default
    ``bias``/``rmse``/``corr``).

    Row height follows the map's aspect ratio plus the room its type needs (override one
    row with ``row_height``, or the whole figure with ``figsize``). Whether the total is
    *capped* belongs to the canvas ``size`` names: the default ``"page"`` keeps the
    figure inside a portrait page, squeezing the panels once there are enough rows —
    unavoidable for something that has to paginate. ``size="free"`` lifts the cap, so
    a many-row grid keeps every panel at full height and simply gets longer, which is
    what you want in a notebook. ``zoom`` multiplies whichever canvas you chose.

    Font sizes are derived from the figure's geometry rather than fixed, so a grid of
    eight rows gets smaller panel type than a grid of two without being asked, while its
    suptitle — which labels the whole figure, not one row — does not shrink with the
    rows. ``font_scale`` multiplies them all and buys the height that needs; a size
    passed explicitly in a ``*_kwargs`` dict overrides outright and is exempt from
    ``fit_text``. See :mod:`ocean_skill.plot.typography` and :func:`field_row`.

    ``shared_limits=True`` makes every row's colour scale (and its difference
    range) span *all* rows' data instead of each computing its own — meaningful
    only when every row is the same variable (e.g. one depth per row), since
    different variables have unrelated ranges and units; sharing across those would
    make every row's colours meaningless relative to the numbers on the bar. Warns
    if the rows' ``standard_name``s actually differ.

    ``robust`` means what it does in :func:`_limits`: each row's (or, with
    ``shared_limits=True``, the whole grid's) sequential colour scale spans the
    full data range by default, or the 10th–90th percentile with ``robust=True``.

    ``shared_axis_labels=True`` (the default) draws grid lines on every panel but
    only labels the leftmost column's latitude axis and the bottom row's longitude
    axis — the usual convention for a grid of maps, since every other panel would
    otherwise repeat labels a neighbour already shows. Set ``False`` to label every
    panel's axes independently, as every version before this one did.

    ``align_colorbars=True`` (the default) makes each row's vertical bars start and
    end level with that row's maps — top with the top of the axes, bottom with the
    bottom, excluding the title above and the longitude labels below, which the grid
    cell the bar is otherwise sized to includes. See :func:`_align_colorbars`.

    ``title`` defaults, when not given, to whatever identity every row shares — the
    variable, the depth, the instant a ``select=`` fixed for all of them — through
    :func:`grid_suptitle`; the part the rows *differ* in is already their left-edge row
    label, so it is left off the top title rather than repeated. A grid whose rows share
    nothing nameable draws no suptitle, as before. Pass ``title=""`` to drop it.

    The ``*_kwargs`` parameters each merge onto their current defaults and map onto
    one matplotlib/cartopy call — see :func:`field_row`'s docstring for the full
    list; the same names mean the same thing here, applied per row.

    ``rasterize``/``hover`` are accepted only so ``renderer="both"`` can pass one option
    set to each renderer (see :func:`_warn_if_interactive_only`) — they are the
    interactive renderer's fix for a large mesh and do nothing here.

    ``coastline_resolution``/``land`` pick the coastline/land dataset and the land
    fill's visibility for every row — see :func:`field_row`'s docstring.
    """
    import matplotlib.pyplot as plt

    _warn_if_interactive_only(rasterize, hover)

    if title is None:
        title = grid_suptitle(comparisons)

    n = len(comparisons)
    proj = _map_projection(*(c["aligned"] for c in comparisons))
    canvas = resolve_canvas(size, zoom)
    # One decision, used for both the layout allowance and the bars themselves.
    # Splitting them is how overriding the orientation used to cost 37% of the panel:
    # the bars moved below the maps while the row height reserved space beside them.
    horizontal = colorbar_is_horizontal(
        _map_aspect(comparisons, reference_name),
        # stacked rows: bars beside, since height is the dimension in short supply
        default_horizontal=False,
        requested=(colorbar_kwargs or {}).get("orientation"),
    )
    row_h = row_height or _row_height(
        comparisons,
        reference_name,
        n,
        font_scale,
        canvas=canvas,
        horizontal_colorbar=horizontal,
    )
    figsize = figsize or (canvas.width, row_h * n)
    scale = _scale_for(figsize, nrows=n, font_scale=font_scale)
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
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
            comparisons, test_name, reference_name, robust=robust
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
            coastline_resolution=coastline_resolution,
            land=land,
            robust=robust,
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
        sup = fig.suptitle(
            title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs)
        )
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    _fit_left_margin(fig)
    if align_colorbars:
        _align_colorbars(fig)
    # after alignment, which is what finally decides how long each colorbar is
    if fit_text:
        _fit_text_widths(fig)
        # a shrunken row label is a *narrower* one, and its placement was measured
        # against the old width — re-place it against what is actually drawn now
        _clear_row_labels(fig)
    _warn_if_cramped(fig, canvas=canvas, nrows=n)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


#: Facet coordinates that name a vertical level rather than a time. ``z`` is what
#: :func:`ocean_skill.roms.to_depth` produces; the rest are what observational products
#: call the same axis, matching :data:`ocean_skill.cf._COORD_FALLBACKS`. Deliberately
#: excludes ``pressure``/``pres`` (a different unit -- dbar, not metres) and the
#: model-native ``s_rho``/``z_rho`` spellings (handled before this in
#: :func:`facet_labels`), so this can't reuse :func:`ocean_skill.vocabulary.matches_axis`
#: outright -- neither its full ``Z`` token set (adds pressure) nor its ``direct_only``
#: one (drops ``depth_surface``/``lev``) is quite this list. Matched
#: case-insensitively (:func:`facet_labels` lowercases first) so a source's own
#: capitalization (WHOTS' ``DEPTH``, say) still gets its ``" m"`` label.
_DEPTH_COORDS = ("z", "depth", "depth_surface", "lev")


def facet_labels(coord) -> list[str]:
    """Panel labels for a facet coordinate, spelled to say which reduction made it.

    The shapes :mod:`ocean_skill.operators` can leave standing are distinguishable from
    the coordinate alone, and are deliberately labelled so that the figure says which
    one it is:

    * timestamps (a ``resample``) -> ``"Jan 2012"``, refined by :func:`_distinct` to
      ``"2012-01-16"`` (and further) when a month is not enough to tell one panel from
      the next. The year is not optional here: it is the only thing on the page
      distinguishing six consecutive months from six months of a climatology, and a
      reader who cannot tell those apart is reading the wrong figure without knowing it.
    * integer months (``{"groupby": "month"}``) -> ``"Jan"``, no year, because there
      isn't one — the panel is every January of the record.
    * a vertical level -> ``"50 m"``. Taken through ``abs`` because the model's own
      axis is negative-down (:func:`ocean_skill.roms.to_depth` interpolates onto
      ``-depths``) while the depth the caller asked for, and the one a reader expects
      on a label, is positive-down.
    * anything else (``{"groupby": "season"}`` gives ``"DJF"``) -> its own value.

    A ``level_labels`` attr on the coordinate wins outright: it is how a mixed
    vertical selection (``["surface", 50, 100]``) says its ``z=0.0`` row is the
    model's own surface and not an interpolated 0 m — the coordinate itself must stay
    numeric for the lane cache, so the spelling rides here (see
    :func:`ocean_skill.comparison._surface_and_levels`). Honoured only at full
    length: a subset of the axis no longer knows which label belongs to which level.
    """
    name = str(coord.name)
    values = list(np.atleast_1d(coord.values))
    labels = coord.attrs.get("level_labels")
    if labels is not None and len(labels) == len(values):
        return [str(lb) for lb in labels]
    try:
        # covers numpy datetime64 and cftime alike, which is why this goes through
        # xarray's accessor rather than pandas or datetime directly -- a ROMS run on a
        # 360-day calendar carries cftime objects that pd.Timestamp cannot parse.
        month_labels = [str(v) for v in coord.dt.strftime("%b %Y").values]
    except (TypeError, AttributeError):
        month_labels = None
    if month_labels is not None:
        return _distinct(coord, month_labels)
    if name == "month":
        from ocean_skill.plot.series import month_label

        try:
            return [month_label(v) for v in values]
        except (ValueError, TypeError, IndexError):  # pragma: no cover - odd coord
            pass
    if name == "sigma0":
        # An isopycnal axis (:func:`ocean_skill.roms.to_sigma0`) — a density, not a
        # depth, so it is spelled through the same label the rest of the package
        # uses for one rather than folded into _DEPTH_COORDS, which would append "m".
        from ocean_skill.comparison import _sigma_label

        try:
            return [_sigma_label(float(v)) for v in values]
        except (ValueError, TypeError):  # pragma: no cover - odd coord
            pass
    if name.lower() in _DEPTH_COORDS:
        try:
            return [f"{abs(float(v)):g} m" for v in values]
        except (ValueError, TypeError):  # pragma: no cover - odd coord
            pass
    return [str(v) for v in values]


#: Datetime spellings tried in turn when the one before it leaves two panels — or two
#: frames — saying the same thing. Coarsest first: a label is as short as it can be
#: while still naming which panel it sits above.
_FINER_TIME_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M")


def _distinct(coord, labels) -> list[str]:
    """Return ``labels``, spelled finer until no two panels carry the same one.

    ``"%b %Y"`` names a reduction exactly — six monthly means, twelve climatological
    months — and is what a reader wants above a panel that *is* a month. It is wrong
    the moment the axis is finer than the label: three days of January selected out of
    a run come out as three panels all called ``Jan 2012``, and a month of daily output
    as 31. Statically that is a caption repeated over panels that differ; interactively
    the labels are the slider's values, and duplicates collapse frames on top of each
    other silently.

    So the coarse spelling stands wherever it distinguishes the panels, and the
    resolution escalates only where it does not. Non-datetime axes (levels, seasons)
    come back unchanged, having nothing finer to fall back to.
    """
    if len(set(labels)) == len(labels):
        return labels
    for fmt in _FINER_TIME_FORMATS:
        try:
            # same accessor facet_labels uses, so cftime calendars work here too
            finer = [str(v) for v in coord.dt.strftime(fmt).values]
        except (TypeError, AttributeError):
            return labels
        labels = finer
        if len(set(finer)) == len(finer):
            break
    return labels


def frame_labels(coord) -> list[str]:
    """Return a movie's frame labels: exactly the panel titles the grid would carry.

    A movie is its facet grid played rather than laid out, so its frames are labelled by
    :func:`facet_labels` — including that function's refinement of a datetime axis too
    fine for ``"%b %Y"``, which movies need more often than grids do (every step of a
    run is a frame) but which is the same rule either way. Kept as its own name because
    the movie paths read better for it and both renderers call it.
    """
    return facet_labels(coord)


def field_title(standard_name) -> str:
    """The suptitle a one-field figure carries when the caller has not named one.

    The panels say *when*; nothing on the figure said *what*. A colorbar reading
    ``[mmol/m^3]`` narrows it to a concentration and no further, and the file the figure
    came from is not on the page, so a saved alkalinity figure and a saved nitrate one
    were indistinguishable once they left the session that drew them.

    Spelled through :func:`ocean_skill.vars.short_name`, as every legend and axis label
    in the package is — the same field must not be ``alkalinity`` on one figure and
    ``sea_water_alkalinity_expressed_as_mole_equivalent`` on the next. A field with no
    CF name to shorten (a derived expression, say) gets no title rather than a guess;
    ``title=""`` suppresses it explicitly, and ``title="..."`` still wins outright.

    Only the one-source families default this. A comparison's rows already name their
    variable down the left edge, and a grid of several would have no single name to
    carry.
    """
    from ocean_skill.plot.summary import pretty_level

    return pretty_level("variable", standard_name) if standard_name else ""


def _scalar_time_label(value) -> str:
    """``"2013-01-30"``, or ``"2013-01-30 14:00"`` when the time is not midnight."""
    import pandas as pd

    try:
        t = pd.Timestamp(value)
    except (TypeError, ValueError):
        t = value  # cftime: carries .hour/.minute/.second/.strftime itself
    fmt = "%Y-%m-%d" if not (t.hour or t.minute or t.second) else "%Y-%m-%d %H:%M"
    return t.strftime(fmt)


def field_suptitle(
    field,
    *,
    standard_name=None,
    depth: str | None = None,
    label: str | None = None,
    facet_dim: str | None = None,
    row_dim: str | None = None,
) -> str:
    """The suptitle a one-field figure carries: what a ``select=`` has taken off the page.

    :func:`field_title` alone says only the variable. A ``select={"depth": "surface",
    "time": ...}`` that collapses both axes to one map leaves nothing on the figure
    saying which level or which instant — the panels can no longer say *when* because
    there is only one panel, and no row says *how deep* because there is no row. This
    builds the fuller identity a collapsed field needs: source, variable, depth, time,
    joined with " · " and each part dropped when it is not this figure's to say.

    Depth is left out when it is still a facet or row axis — those panels already say
    it, down the rotated row label or across the columns. Time is included only when it
    survives as a *scalar* coordinate; a standing time dimension is the facet axis
    itself, and the panels already say when. A region is included when the field's own
    ``select`` cropped it to a box (``attrs["region"]``, set by
    :func:`ocean_skill.align.subset_to_box`) — the one-field counterpart of
    :func:`ocean_skill.comparison.Comparison.as_item`'s ``"region"`` key.
    """
    from ocean_skill.align import _time_name
    from ocean_skill.comparison import _region_label
    from ocean_skill.operators import resolve_dim

    extras = []

    vertical = resolve_dim(field, "Z")
    faceted_vertically = vertical is not None and vertical in (facet_dim, row_dim)
    if depth and not faceted_vertically:
        extras.append(depth)

    tname = _time_name(field)
    if tname is not None and tname in field.coords and field.coords[tname].ndim == 0:
        extras.append(_scalar_time_label(field.coords[tname].values.item()))

    region = field.attrs.get("region")
    if region is not None:
        extras.append(_region_label(region))

    return suptitle_text(standard_name, extras, label=label)


def suptitle_text(standard_name, extras, *, label: str | None = None) -> str:
    """Join a variable name with the context a collapsed figure has to spell out.

    The package's one spelling of a default figure title: the variable's short name
    (via :func:`field_title`) followed by whatever a ``select=`` has taken off the page
    — a depth, an instant — each part dropped when it is empty, joined with `` · ``. A
    ``label``, when given, prefixes the whole as ``label: subject``.

    Shared by :func:`field_suptitle` (which derives ``extras`` from a single field's own
    coords) and :func:`field_row` (which is handed the comparison's already-formatted
    depth and time), so a one-field figure and a ``test | reference | difference`` row
    name the same quantity the same way.
    """
    parts = [field_title(standard_name), *(e for e in extras if e)]
    subject = " · ".join(p for p in parts if p)
    return f"{label}: {subject}" if label and subject else subject


def grid_suptitle(items) -> str:
    """Overall title for a stacked grid: the identity every row already shares.

    A grid names down each row's left edge whatever the set's fan varied — the variable,
    the depth, the instant (:func:`ocean_skill.comparison.ComparisonSet._label_for`). The
    parts that *don't* vary are common to every row, so one title up top can carry them;
    the varying part is already the row label, so leaving it out is what keeps the two
    from duplicating. The shared ``standard_name``/``depth``/``time`` compose through the
    same :func:`suptitle_text` a single row uses, and a grid whose rows share nothing
    nameable (different variables, no common depth or time) gets ``""`` — no suptitle,
    exactly as before this default existed.
    """

    def shared(key):
        values = {item.get(key) for item in items}
        return next(iter(values)) if len(values) == 1 else None

    return suptitle_text(
        shared("standard_name"), (shared("depth"), shared("time"), shared("region"))
    )


def field_facet(
    field,
    *,
    facet_dim: str | None = None,
    row_dim: str | None = None,
    title: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    depth: str | None = None,
    label: str | None = None,
    mark: str = "pcolormesh",
    save: str | Path | None = None,
    domain: tuple[float, float, float, float] | np.ndarray | None = None,
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
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
    robust: bool | float = False,
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
    two cannot be confused for one another on the page. An axis finer than its label —
    three days of a January, say — is spelled out far enough to tell the panels apart
    (``2013-01-16``), since a title that names every panel names none. With a ``row_dim``
    the titles appear on the top row only — every row below shows the same months — and
    each row is named down the left edge instead (``50 m``), the same rotated label
    :func:`field_grid` uses. The panels having said *when*, the suptitle says *what* the
    panels no longer do: it defaults to the variable, depth and (if collapsed to one
    instant) time (see :func:`field_suptitle`), and ``title=""`` drops it.

    The ``*_kwargs`` parameters and ``font_scale`` mean exactly what they do in
    :func:`field_row`; ``metrics_kwargs`` has no counterpart here, there being no
    metrics, and ``row_label_kwargs`` applies only when there is a ``row_dim``.

    ``rasterize``/``hover`` are accepted only so ``renderer="both"`` can pass one option
    set to each renderer (see :func:`_warn_if_interactive_only`) — they are the
    interactive renderer's fix for a large mesh and do nothing here.

    ``coastline_resolution``/``land`` pick the coastline/land dataset and the land
    fill's visibility for every panel — see :func:`field_row`'s docstring.

    ``robust`` means what it does in :func:`_limits`: each row's colour scale spans
    the full range of its own data by default, or its 10th–90th percentile with
    ``robust=True``.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot.typography import facet_figsize, facet_layout

    _warn_if_interactive_only(rasterize, hover)
    canvas = resolve_canvas(size, zoom)
    title = (
        field_suptitle(
            field,
            standard_name=standard_name,
            depth=depth,
            label=label,
            facet_dim=facet_dim,
            row_dim=row_dim,
        )
        if title is None
        else title
    )
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
        ncols, nrows = facet_layout(n, aspect, canvas=canvas)
    else:
        ncols = max(int(ncols), 1)
        nrows = -(-n // ncols)
    # A bar on the grid's long edge stays the same length as the panels it describes;
    # on the short edge it would be a stub beside a tall column, or a rule under a
    # wide one. Per-row bars are always vertical, beside their row.
    per_row_bars = row_dim is not None and not shared_limits
    if row_dim is None and n == 1:
        # A single map has no grid to compare rows and columns of -- the long edge is
        # the map's own aspect, the same rule facet_movie's lone frame follows.
        horizontal = colorbar_is_horizontal(
            aspect,
            default_horizontal=True,
            requested=(colorbar_kwargs or {}).get("orientation"),
        )
    else:
        horizontal = False if per_row_bars else ncols > nrows

    figsize = figsize or facet_figsize(
        aspect,
        nrows=nrows,
        ncols=ncols,
        # with two axes only the top row is titled, so the rows below need a gap
        # rather than a title's worth of room as well
        title_every_row=row_dim is None,
        canvas=canvas,
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
        vmin, vmax = _limits(sub, robust=robust)
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
        subplot_kw={"projection": _map_projection(field)},
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
            coastline_resolution=coastline_resolution,
            land=land,
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
    # alignment first, then the fit: see _fit_text_widths, whose whole point is that it
    # measures each label against the box it will *end up* in
    if align_colorbars:
        _align_colorbars(fig)
    _fit_text_widths(fig)
    _clear_row_labels(fig)
    # last, for the same reason: the panels have to be where they will be drawn before
    # the title can be centred over them
    _centre_suptitle(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def metric_panels(skill, requested=None) -> list[str]:
    """Resolve and validate the metrics to draw from a skill Dataset.

    A requested metric the Dataset does not carry **raises**, naming what it does carry.
    Dropping the panel instead would be invisible: a figure of three maps where four
    were asked for looks exactly like a figure of three maps. Which metrics exist is a
    data-layer question — each is a full reduction over the scored axis and cannot be
    conjured at draw time — so the message points there rather than at a plot option.

    Shared with the interactive renderer, so both refuse the same request the same way.
    """
    available = [
        name
        for name, da in skill.data_vars.items()
        if da.ndim == 2 and da.dtype.kind in "fiu"
    ]
    if requested is None:
        return available
    names = [requested] if isinstance(requested, str) else list(requested)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            f"no pointwise map for {missing} — this comparison computed "
            f"{available}. Which metrics are computed is decided when the maps are "
            'prepared, not when they are drawn: pass metrics=("bias", "corr", ...) to '
            "compare() (or to Comparison.pointwise_metrics()) to add them."
        )
    return names


def metric_panel_titles(names) -> list[str]:
    """Return the panel titles for a list of metrics.

    The metric's own key, which is how every metric is already spelled in the corner
    box, in the CSV and in :data:`ocean_skill.metrics.REGISTRY` — one name for one thing
    across the whole package. Its own function so a prettier spelling later ("σ ratio")
    lands in both renderers at once, the way :func:`facet_labels` does for facet panels.
    """
    return [str(name) for name in names]


def metric_arrays(skill, names) -> dict[str, Any]:
    """Return the values each metric's colour limits should be derived from.

    Usually the metric's own map. The exception is a
    :data:`~ocean_skill.colormaps.METRIC_LIMIT_GROUPS` pair — ``mean_test`` with
    ``mean_reference``, ``std_test`` with ``std_reference`` — which get the *pooled*
    values of both members, because they are one physical quantity for two fields and a
    per-panel scale would make the only comparison worth making impossible to read.
    """
    from ocean_skill.colormaps import METRIC_LIMIT_GROUPS

    grouped = {name: group for group in METRIC_LIMIT_GROUPS for name in group}
    out = {}
    for name in names:
        members = [
            member for member in grouped.get(name, (name,)) if member in skill.data_vars
        ]
        out[name] = np.concatenate(
            [np.asarray(skill[member]).ravel() for member in members]
        )
    return out


def section(
    field,
    *,
    title: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    depth: str | None = None,
    label: str | None = None,
    mark: str = "pcolormesh",
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    robust: bool | float = False,
):
    """Draw one vertical section: depth against along-path distance.

    The model-only counterpart of :func:`field_facet`'s single-map case, minus the
    map itself: a section has no cartopy projection, since its two axes are depth
    and along-path distance rather than longitude and latitude. See
    :func:`ocean_skill.plot.section.prepare_section` for the axis conventions this
    draws against -- positive-down depth with the y-axis inverted, so 0 m sits at
    the top and the deepest cell at the bottom; the along-path axis in kilometres
    (or in degrees, for a fixed-longitude/latitude line, whose along coordinate
    still varies in the *other* direction).

    ``mark="pcolormesh"`` (default) or ``"contourf"``, the same two this package's
    map families accept. Cells below the modelled seafloor -- or wherever the path
    has left the source's domain -- carry no data, and draw as the same grey a
    map's land does (``ax.set_facecolor``), so the seafloor's shape is visible
    without singling those cells out.

    There is no ``domain``, ``gridline_kwargs`` or ``tick_label_kwargs``: a section
    has no map to outline or gridline, and its plain Cartesian ticks need no
    gridliner styling to carry. Everything else -- sizing, ``font_scale``,
    ``colorbar_kwargs``/``suptitle_kwargs`` -- means what it does in
    :func:`field_row`.

    ``rasterize``/``hover`` are accepted only so ``renderer="both"`` can pass one
    option set to each renderer (see :func:`_warn_if_interactive_only`) — a
    section's mesh is small enough that neither changes anything here.

    ``robust`` means what it does in :func:`_limits`: the colour scale spans the
    full range of the section by default, or its 10th–90th percentile with
    ``robust=True``.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot.section import prepare_section
    from ocean_skill.plot.typography import SECTION_ASPECT

    _warn_if_interactive_only(rasterize, hover)
    values, geometry = prepare_section(field)
    if title is None:
        title = suptitle_text(standard_name, (depth, geometry.path_note), label=label)

    canvas = resolve_canvas(size, zoom)
    horizontal = colorbar_is_horizontal(
        SECTION_ASPECT,
        default_horizontal=True,  # one panel: a bar below leaves it the full width
        requested=(colorbar_kwargs or {}).get("orientation"),
    )
    figsize = figsize or auto_figsize(
        SECTION_ASPECT,
        nrows=1,
        ncols=1,
        canvas=canvas,
        font_scale=font_scale,
        horizontal_colorbar=horizontal,
        overhead=ROW_OVERHEAD_HORIZONTAL_CBAR if horizontal else ROW_OVERHEAD,
    )
    scale = type_scale(
        figsize, ncols=1, nrows=1, font_scale=font_scale, figure_ncols=REFERENCE_GRID[0]
    )
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    suptitle_kwargs = _merged(defaults["suptitle_kwargs"], suptitle_kwargs)

    cmap, _ = cmaps_for(standard_name)
    vmin, vmax = _limits(values, robust=robust)
    norm = norm_for(standard_name, vmin, vmax)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.set_facecolor("0.85")  # the map families' land grey, doing the same job here:
    # a below-bathymetry (or off-domain) cell is genuinely absent data, not zero.
    draw = ax.contourf if mark == "contourf" else ax.pcolormesh
    kw = {"levels": _contour_levels(norm)} if mark == "contourf" else {}
    im = draw(
        values[geometry.x_name],
        values[geometry.y_name],
        values,
        cmap=cmap,
        norm=norm,
        **kw,
    )
    ax.invert_yaxis()
    ax.set_xlabel(geometry.x_label, fontsize=scale["axes_label"])
    ax.set_ylabel(geometry.y_label, fontsize=scale["axes_label"])
    ax.tick_params(axis="both", labelsize=scale["tick_label"])

    lab = units or ""
    _draw_colorbar(fig, im, ax, lab, colorbar_kwargs, defaults["colorbar_kwargs"])

    if title:
        sup = fig.suptitle(title, **suptitle_kwargs)
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    if align_colorbars:
        _align_colorbars(fig)
    if fit_text:
        _fit_text_widths(fig)
    _warn_if_cramped(fig, canvas=canvas, nrows=1, panels=[ax])
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def cross(
    items: list[dict[str, Any]],
    *,
    orientation: str = "vertical",
    title: str | None = None,
    mark: str = "pcolormesh",
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    robust: bool | float = False,
):
    """Draw two vertical sections through one point, one along each grid direction.

    :func:`section`'s two-panel sibling -- ``osk.field(..., select={"transect":
    {"cross": ...}}).plot()`` builds a :class:`~ocean_skill.field.Cross` of two
    independent sections (one along each grid dimension through the shared
    point, see :mod:`ocean_skill.transect`), and this draws them on one
    figure, sharing one colour scale (the same variable, so the two read as
    directly comparable) rather than a scale per panel. ``orientation``
    (``"vertical"``, the default) stacks the two panels down the page;
    ``"horizontal"`` lays them side by side instead.

    Each panel draws exactly as :func:`section` draws its own single panel --
    the same below-bathymetry grey, inverted positive-down depth axis,
    along-path distance axis -- but titled by its own item's ``label`` (which
    grid dimension it holds fixed) followed by its own ``path_note``
    (:attr:`ocean_skill.plot.section.SectionGeometry.path_note`), since the two
    panels cut through different parts of the domain and so have different
    endpoints to name. ``title`` is the one suptitle both panels share, naming
    the variable and depth (identical on both, since a cross's two directions
    share one variable and one vertical request) -- there is no per-panel
    ``standard_name``/``units``/``depth`` argument the way :func:`section` has,
    since both of ``items`` already carry the same ones.

    There is no ``domain``, ``metrics``/``metrics_kwargs`` or ``labels``: a
    cross panel has no map to outline and no reference to score against, and
    each item's own ``label`` already says which panel is which (see above).
    Everything else -- sizing (``size``/``zoom``/``figsize``), ``font_scale``,
    ``fit_text``, ``align_colorbars``, the ``*_kwargs`` dicts,
    ``rasterize``/``hover`` (interactive-only, see
    :func:`_warn_if_interactive_only`), ``robust`` -- means exactly what it does in
    :func:`section`, applied to the one scale the two panels share.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot.section import prepare_section
    from ocean_skill.plot.typography import SECTION_ASPECT

    if len(items) != 2:
        raise ValueError(
            f"cross needs exactly 2 items (one section per grid direction), "
            f"got {len(items)}."
        )
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"orientation={orientation!r} -- expected 'vertical' (stacked, the "
            "default) or 'horizontal' (side by side)."
        )
    _warn_if_interactive_only(rasterize, hover)

    prepared = [prepare_section(item["field"]) for item in items]
    standard_name = items[0].get("standard_name")
    units = items[0].get("units")
    depth = items[0].get("depth")
    if title is None:
        title = suptitle_text(standard_name, (depth,))

    nrows, ncols = (2, 1) if orientation == "vertical" else (1, 2)
    canvas = resolve_canvas(size, zoom)
    horizontal = colorbar_is_horizontal(
        SECTION_ASPECT,
        default_horizontal=(orientation == "vertical"),
        requested=(colorbar_kwargs or {}).get("orientation"),
    )
    figsize = figsize or auto_figsize(
        SECTION_ASPECT,
        nrows=nrows,
        ncols=ncols,
        canvas=canvas,
        font_scale=font_scale,
        horizontal_colorbar=horizontal,
        overhead=ROW_OVERHEAD_HORIZONTAL_CBAR if horizontal else ROW_OVERHEAD,
    )
    scale = type_scale(
        figsize,
        ncols=ncols,
        nrows=nrows,
        font_scale=font_scale,
        figure_ncols=REFERENCE_GRID[0],
    )
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    title_pinned = _pinned(title_kwargs, "title_kwargs")
    title_kwargs = _merged(defaults["title_kwargs"], title_kwargs)
    suptitle_kwargs = _merged(defaults["suptitle_kwargs"], suptitle_kwargs)

    cmap, _ = cmaps_for(standard_name)
    vmin, vmax = _limits(*(values for values, _ in prepared), robust=robust)
    norm = norm_for(standard_name, vmin, vmax)

    fig, axes_grid = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=True
    )
    axes = list(np.atleast_1d(axes_grid).ravel())
    ims = []
    for ax, item, (values, geometry) in zip(axes, items, prepared, strict=True):
        ax.set_facecolor("0.85")  # section()'s below-bathymetry/off-domain grey
        draw = ax.contourf if mark == "contourf" else ax.pcolormesh
        kw = {"levels": _contour_levels(norm)} if mark == "contourf" else {}
        im = draw(
            values[geometry.x_name],
            values[geometry.y_name],
            values,
            cmap=cmap,
            norm=norm,
            **kw,
        )
        ax.invert_yaxis()
        ax.set_xlabel(geometry.x_label, fontsize=scale["axes_label"])
        ax.set_ylabel(geometry.y_label, fontsize=scale["axes_label"])
        ax.tick_params(axis="both", labelsize=scale["tick_label"])
        label = item.get("label") or ""
        path_note = geometry.path_note
        panel_title = f"{label} — {path_note}" if label else path_note
        t = ax.set_title(panel_title, **title_kwargs)
        t._osk_size_pinned = title_pinned
        ims.append(im)

    lab = units or ""
    _draw_colorbar(
        fig, ims[-1], axes, lab, colorbar_kwargs, defaults["colorbar_kwargs"]
    )

    if title:
        sup = fig.suptitle(title, **suptitle_kwargs)
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    if align_colorbars:
        _align_colorbars(fig)
    if fit_text:
        _fit_text_widths(fig)
    _warn_if_cramped(fig, canvas=canvas, nrows=nrows, panels=axes)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def _draw_time_depth(ax, values, geometry, *, cmap, norm, mark: str) -> Any:
    """Draw one ``time_depth`` panel's scatter/mesh into ``ax``, return its mappable.

    ``values``/``geometry`` are :func:`~ocean_skill.plot.time_depth.prepare_time_depth`'s
    own return; the caller has already resolved ``mark`` (via
    :func:`~ocean_skill.plot.time_depth.default_mark`) and ``cmap``/``norm`` (its own
    range, or one shared across a grid's panels -- see
    :func:`time_depth_grid`'s ``shared_limits``). Shared by :func:`time_depth` (one
    panel, its own figure) and :func:`time_depth_grid` (several, one per axes) so the
    two can never draw a cell differently.
    """
    import xarray as xr

    ax.set_facecolor("0.85")  # the section family's absent-cell grey, doing the
    # same job here: a mesh cell with no reading at all is genuinely absent data.
    x = values[geometry.x_name]
    y = values[geometry.y_name]
    if mark == "scatter":
        # x/y are always broadcast to the value grid's own shape already (see
        # prepare_time_depth), so this is one scatter call over finite cells,
        # no reshaping needed here.
        xb, yb, vb = xr.broadcast(x, y, values)
        finite = np.asarray(vb.notnull())
        im = ax.scatter(
            np.asarray(xb)[finite],
            np.asarray(yb)[finite],
            c=np.asarray(vb)[finite],
            cmap=cmap,
            norm=norm,
            s=26,
            edgecolor="white",
            linewidth=0.4,
        )
    else:
        im = ax.pcolormesh(x, y, values, cmap=cmap, norm=norm)
    ax.invert_yaxis()
    return im


def time_depth(
    field,
    *,
    title: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    label: str | None = None,
    mark: str | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    robust: bool | float = False,
):
    """Draw one ``time_depth`` panel: depth against time, at one place.

    The default figure for a bare :class:`~ocean_skill.field.Field` whose select
    leaves both a time axis and a vertical axis standing at one point -- a
    ``timeSeriesProfile`` station's own shape, most often. See
    :func:`ocean_skill.plot.time_depth.prepare_time_depth` for the axis
    conventions this draws against -- positive-down depth with the y-axis
    inverted, so 0 m sits at the top and the deepest reading at the bottom; time
    on x, labelled concisely without the 45-degree tilt -- or, for a time
    groupby's surviving axis (``month``, ``year``, ...; see
    :func:`ocean_skill.operators.time_axis_dim`), its own integer values instead,
    ``month`` spelled ``Jan``..``Dec`` (see :func:`_x_axis`).

    ``mark`` defaults to :func:`ocean_skill.plot.time_depth.default_mark`'s own
    call: ``"scatter"`` for a ragged repeat-visit record (most of a bottle
    station's (time, depth) rectangle is holes no cast actually sampled), or
    ``"pcolormesh"`` for a dense one (a mooring's fixed levels, a model column).
    Pass it explicitly to override either way. A scatter marker's colour reads
    the value at that (time, depth) cell; a mesh cell with no reading at all
    (below where a shorter cast reached, say) draws as the same grey a map draws
    for land (``ax.set_facecolor``).

    ``tick_label_kwargs`` styles the date ticks, unlike :func:`section` (whose
    plain Cartesian ticks have no gridliner styling to carry) -- a date axis's
    ticks are exactly the labels worth styling here. There is no ``domain``: a
    ``time_depth`` panel has no map to outline.

    ``rasterize``/``hover`` are accepted only so ``renderer="both"`` can pass one
    option set to each renderer (see :func:`_warn_if_interactive_only`) — a
    ``time_depth`` panel's own mesh or scatter is small enough that neither
    changes anything here.

    ``robust`` means what it does in :func:`_limits`: the colour scale spans the
    full range of the panel by default, or its 10th–90th percentile with
    ``robust=True``.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot.time_depth import default_mark, prepare_time_depth
    from ocean_skill.plot.typography import SECTION_ASPECT

    _warn_if_interactive_only(rasterize, hover)
    values, geometry = prepare_time_depth(field)
    if mark is None:
        mark = default_mark(values)
    if title is None:
        title = suptitle_text(
            standard_name, (geometry.place_note, geometry.period_note), label=label
        )

    canvas = resolve_canvas(size, zoom)
    horizontal = colorbar_is_horizontal(
        SECTION_ASPECT,
        default_horizontal=True,  # one panel: a bar below leaves it the full width
        requested=(colorbar_kwargs or {}).get("orientation"),
    )
    figsize = figsize or auto_figsize(
        SECTION_ASPECT,
        nrows=1,
        ncols=1,
        canvas=canvas,
        font_scale=font_scale,
        horizontal_colorbar=horizontal,
        overhead=ROW_OVERHEAD_HORIZONTAL_CBAR if horizontal else ROW_OVERHEAD,
    )
    scale = type_scale(
        figsize, ncols=1, nrows=1, font_scale=font_scale, figure_ncols=REFERENCE_GRID[0]
    )
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    suptitle_kwargs = _merged(defaults["suptitle_kwargs"], suptitle_kwargs)

    cmap, _ = cmaps_for(standard_name)
    vmin, vmax = _limits(values, robust=robust)
    norm = norm_for(standard_name, vmin, vmax)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    im = _draw_time_depth(ax, values, geometry, cmap=cmap, norm=norm, mark=mark)
    ax.set_xlabel(geometry.x_label, fontsize=scale["axes_label"])
    ax.set_ylabel(geometry.y_label, fontsize=scale["axes_label"])
    _x_axis(
        ax, scale, tick_label_kwargs, date=geometry.date_axis, ticks=geometry.x_ticks
    )

    lab = units or ""
    _draw_colorbar(fig, im, ax, lab, colorbar_kwargs, defaults["colorbar_kwargs"])

    if title:
        sup = fig.suptitle(title, **suptitle_kwargs)
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    if align_colorbars:
        _align_colorbars(fig)
    if fit_text:
        _fit_text_widths(fig)
    _warn_if_cramped(fig, canvas=canvas, nrows=1, panels=[ax])
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def time_depth_grid(
    items: list[dict[str, Any]],
    *,
    title: str | None = None,
    mark: str | None = None,
    ncols: int | None = None,
    nrows: int | None = None,
    shared_limits: bool = False,
    sharex: bool | None = None,
    sharey: bool = False,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    robust: bool | float = False,
):
    """Stack several ``time_depth`` panels -- one per item -- in a single figure.

    The ``time_depth`` counterpart of :func:`series`/:func:`profile`: a
    :class:`~ocean_skill.field.FieldSet` of several stations/moorings, each its own
    ``time_depth`` item (see :meth:`ocean_skill.field.Field._time_depth_item`), drawn
    one panel per item rather than any overlay -- a mesh or scatter has no second
    channel (colour) free to carry a second source the way a line's colour does.

    Panels stack in a single column by default (``ncols=None``, ``nrows=None``), time
    on a shared x-axis, matching the layout :func:`~ocean_skill.plot.series.grid_shape`
    gives every other line family for the same shape. ``ncols=``/``nrows=`` wrap the
    panels into a rectangular grid instead -- shared with :func:`series`/:func:`profile`
    so the three families cannot pick different wraps.

    Each panel gets its own colour scale and colorbar by default (different moorings,
    different depths, different ranges); ``shared_limits=True`` computes one shared
    ``vmin``/``vmax`` (and a single warning if the items' ``standard_name``s actually
    differ) across every panel instead -- :func:`field_grid`'s own convention, so the
    two grid families agree on what the option means.

    ``title`` defaults to whatever identity every item shares (ordinarily the
    variable) via :func:`grid_suptitle`; each panel's own title is its identity instead
    -- the item's ``label`` (a mooring's source name, or whatever
    :meth:`~ocean_skill.field.Field._time_depth_item` gave it), plus place/period
    context only when no ``label`` says as much already.

    ``sharex=None`` (the default) links every panel's time axis when they draw the
    same way -- the default single stacked column, every panel a real date axis or
    every panel the same groupby kind (see :attr:`~ocean_skill.plot.time_depth
    .TimeDepthGeometry.date_axis`), exactly :func:`series`' own ``sharex=True``
    default. Several moorings deployed one after another (disjoint windows, not one
    long overlapping record) read as slivers of blank axis either side of their own
    data under that default; pass ``sharex=False`` to autoscale each panel to its own
    window instead, with its own date ticks -- the same option :func:`series` exposes,
    given here rather than defaulted the other way, since sharing is meaningful far
    more often than not for this family's own stacked-column shape.

    ``sharey=False`` (the default) leaves each panel's depth axis to its own
    instrument's range -- moorings at very different depths (22m vs 64m, say) each
    keep the range their own readings actually reach, rather than a shallow one
    inheriting a deep neighbour's mostly-empty axis. Pass ``sharey=True`` to line
    every panel up on one common depth range instead -- :func:`profile`'s own
    default direction, reversed here since this family's panels more often draw
    genuinely different instruments than :func:`profile`'s own casts at one place.

    ``rasterize``/``hover`` are accepted only so ``renderer="both"`` can pass one option
    set to each renderer (see :func:`_warn_if_interactive_only`) -- neither changes
    anything here.

    ``robust`` means what it does in :func:`_limits`: each panel's (or, with
    ``shared_limits=True``, every panel's shared) colour scale spans the full data
    range by default, or its 10th–90th percentile with ``robust=True``.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot.series import grid_shape, value_span
    from ocean_skill.plot.time_depth import default_mark, prepare_time_depth
    from ocean_skill.plot.typography import SECTION_ASPECT

    _warn_if_interactive_only(rasterize, hover)

    n = len(items)
    prepared = [prepare_time_depth(item["field"]) for item in items]
    marks = [mark or default_mark(values) for values, _ in prepared]

    if title is None:
        title = grid_suptitle(items)

    grid_nrows, grid_ncols = grid_shape(n, as_columns=False, ncols=ncols, nrows=nrows)

    canvas = resolve_canvas(size, zoom)
    horizontal = colorbar_is_horizontal(
        SECTION_ASPECT,
        default_horizontal=False,  # stacked panels: bars beside, height is scarce
        requested=(colorbar_kwargs or {}).get("orientation"),
    )
    figsize = figsize or auto_figsize(
        SECTION_ASPECT,
        nrows=grid_nrows,
        ncols=grid_ncols,
        canvas=canvas,
        font_scale=font_scale,
        horizontal_colorbar=horizontal,
        overhead=ROW_OVERHEAD_HORIZONTAL_CBAR if horizontal else ROW_OVERHEAD,
    )
    scale = type_scale(
        figsize,
        ncols=grid_ncols,
        nrows=grid_nrows,
        font_scale=font_scale,
        figure_ncols=REFERENCE_GRID[0],
    )
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    title_kwargs = _merged(defaults["title_kwargs"], title_kwargs)
    suptitle_kwargs = _merged(defaults["suptitle_kwargs"], suptitle_kwargs)

    if sharex is None:
        # auto: share only where it is physically meaningful -- one stacked column,
        # every panel's x axis the same kind (all real dates, or all the same
        # groupby index) -- see TimeDepthGeometry.date_axis.
        sharex = grid_ncols == 1 and len({g.date_axis for _, g in prepared}) == 1
    fig, axes = plt.subplots(
        grid_nrows,
        grid_ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        layout="constrained",
    )
    flat = list(axes.ravel())

    shared_cmap = shared_norm = None
    if shared_limits:
        import warnings

        names = {item.get("standard_name") for item in items}
        if len(names) > 1:
            warnings.warn(
                f"shared_limits=True but panels use different variables "
                f"({sorted(nm for nm in names if nm)}); their ranges/units differ, "
                "so one shared colour scale won't mean the same thing on every panel.",
                stacklevel=_stacklevel.find(),
            )
        standard_name = items[0].get("standard_name")
        shared_cmap, _ = cmaps_for(standard_name)
        vmin, vmax = _limits(*(values for values, _ in prepared), robust=robust)
        shared_norm = norm_for(standard_name, vmin, vmax)

    # One depth range for the whole figure when sharey -- explicit set_ylim on every
    # panel rather than relying on invert_yaxis()'s toggle state (_draw_time_depth's
    # own call), which sharey=True would otherwise flip twice on every panel but the
    # first -- the same fix ocean_skill.plot.profile.depth_range's own docstring
    # explains for that family's sharey. With sharey=False (the default) each panel
    # keeps exactly the range _draw_time_depth's own invert_yaxis() already gives it
    # -- untouched here, so a ragged (scatter) panel's axis still reflects only the
    # depths it actually has readings at, not its full depth coordinate.
    shared_depth = None
    if sharey:
        shared_depth = value_span(
            [np.asarray(values[geometry.y_name]) for values, geometry in prepared]
        )

    for index, (item, (values, geometry), panel_mark) in enumerate(
        zip(items, prepared, marks)
    ):
        ax = flat[index]
        if shared_limits:
            cmap, norm = shared_cmap, shared_norm
        else:
            cmap, _ = cmaps_for(item.get("standard_name"))
            vmin, vmax = _limits(values, robust=robust)
            norm = norm_for(item.get("standard_name"), vmin, vmax)
        im = _draw_time_depth(ax, values, geometry, cmap=cmap, norm=norm, mark=panel_mark)
        if shared_depth is not None:
            y_lo, y_hi = shared_depth
            ax.set_ylim(y_hi, y_lo)  # deep at the bottom, shallow at top
        panel_title = " · ".join(
            p for p in (item.get("label"), geometry.place_note, geometry.period_note) if p
        )
        ax.set_title(panel_title, fontsize=scale["title"], **_without_font(title_kwargs))
        ax.set_ylabel(geometry.y_label, fontsize=scale["axes_label"])
        _x_axis(
            ax, scale, tick_label_kwargs, date=geometry.date_axis, ticks=geometry.x_ticks
        )
        lab = item.get("units") or ""
        _draw_colorbar(fig, im, ax, lab, colorbar_kwargs, defaults["colorbar_kwargs"])

    if grid_ncols == 1 or (grid_nrows == 1 and grid_ncols == n):
        flat[-1].set_xlabel(prepared[-1][1].x_label, fontsize=scale["axes_label"])
    else:
        # A wrapped grid's bottom row is ragged when n does not fill it, so "is there
        # a panel below me?" is the question, not "am I in the last row?" -- the same
        # rule series()'s own wrapped grid uses for the same situation.
        for index, ax in enumerate(flat[:n]):
            if index + grid_ncols >= n:
                ax.set_xlabel(prepared[index][1].x_label, fontsize=scale["axes_label"])
                ax.xaxis.set_tick_params(labelbottom=True)
        for ax in flat[n:]:
            ax.set_visible(False)

    if title:
        sup = fig.suptitle(title, **suptitle_kwargs)
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    if align_colorbars:
        _align_colorbars(fig)
    if fit_text:
        _fit_text_widths(fig)
    _warn_if_cramped(fig, canvas=canvas, nrows=grid_nrows, panels=flat[:n])
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def _draw_time_depth_row(
    axes,
    values: dict[str, Any],
    geometry,
    *,
    labels: tuple[str, str],
    units: str | None,
    standard_name: str | None,
    metrics: dict[str, Any] | None,
    mark: str,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    title_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    shared_axis_labels: bool = True,
    scale: dict[str, float],
    defaults: dict[str, dict[str, Any]],
    robust: bool | float = False,
):
    """Draw one test|reference|difference ``time_depth`` row into three existing axes.

    The ``time_depth`` counterpart of :func:`_draw_section_row`: the same
    shared/symmetric colour norms and corner metrics box, but each panel drawn
    through :func:`_draw_time_depth` -- a station's own scatter-or-mesh
    convention (grey facecolor for a ``(time, depth)`` cell no cast reached,
    positive-down depth with the y-axis inverted, a date-aware x-axis) -- rather
    than :func:`_draw_section_row`'s own inline ``contourf``/``pcolormesh`` call,
    since a station's ragged rectangle can need the scatter branch neither a
    section nor a map ever does.

    ``values``/``geometry`` are
    :func:`ocean_skill.plot.time_depth.prepare_time_depth_row`'s own return,
    unpacked by the caller so this function stays a pure drawing step. ``mark``
    is resolved once by the caller (see :func:`ocean_skill.plot.time_depth
    .default_mark`) and applied to every panel alike, so test, reference and
    difference are never drawn two different ways in the same row. ``robust``
    means what it does in :func:`_limits`.
    """
    import matplotlib.colors as mcolors

    title_pinned = _pinned(title_kwargs, "title_kwargs")
    title_kwargs = _merged(defaults["title_kwargs"], title_kwargs)
    metrics_kwargs = _merged(defaults["metrics_kwargs"], metrics_kwargs)

    t, r, d = values["test"], values["reference"], values["difference"]
    tl, rl = labels
    seq, div = cmaps_for(standard_name)
    vmin, vmax = _limits(t, r, robust=robust)
    seq_norm = norm_for(standard_name, vmin, vmax)
    dmax = float(np.nanpercentile(np.abs(np.asarray(d)), 98)) or 1.0
    div_norm = mcolors.Normalize(vmin=-dmax, vmax=dmax)

    panels = [
        (t, tl, seq, seq_norm),
        (r, rl, seq, seq_norm),
        (d, "difference", div, div_norm),
    ]
    ims = []
    for j, (ax, (field, lab, cmap, norm)) in enumerate(zip(axes, panels, strict=True)):
        im = _draw_time_depth(ax, field, geometry, cmap=cmap, norm=norm, mark=mark)
        _x_axis(
            ax, scale, tick_label_kwargs, date=geometry.date_axis, ticks=geometry.x_ticks
        )
        ax.set_xlabel(geometry.x_label, fontsize=scale["axes_label"])
        # Only the leftmost panel labels depth -- the other two share the same axis,
        # the same convention _draw_section_row uses for its own leftmost panel.
        if not shared_axis_labels or j == 0:
            ax.set_ylabel(geometry.y_label, fontsize=scale["axes_label"])
        if shared_axis_labels and j != 0:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(lab, **title_kwargs)
        ax.title._osk_size_pinned = title_pinned
        ims.append(im)

    if metrics:
        axes[2]._osk_metrics_text = axes[2].text(
            0.02,
            0.02,
            _metrics_text(metrics, metric_keys),
            transform=axes[2].transAxes,
            zorder=5,
            **metrics_kwargs,
        )
    return ims, (f"[{units}]" if units else "")


def time_depth_row(
    aligned,
    *,
    labels: tuple[str, str] | None = None,
    title: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    depth: str | None = None,
    time: str | None = None,
    metrics: dict[str, Any] | None = None,
    mark: str | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    shared_axis_labels: bool = True,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    robust: bool | float = False,
):
    """Draw one ``test | reference | difference`` row of ``time_depth`` panels.

    :func:`section_row` with :func:`ocean_skill.plot.time_depth
    .prepare_time_depth_row`'s geometry substituted for the vertical section
    -- a comparison whose bare ``timeSeriesProfile`` reference pools both its
    own time and depth axes (see
    :attr:`ocean_skill.comparison.Comparison.is_time_depth`) gets this family
    instead, the same way one that reduces to a single time or depth axis gets
    :func:`series`/:func:`profile`. Test and reference share one colour scale
    (the full range of the pair by default, or its 10th-90th percentile with
    ``robust=True``); the difference panel uses a diverging map centred on
    zero; metrics go in the difference panel's corner box -- all exactly as
    :func:`section_row` draws its own row, just against time and depth instead
    of along-path distance and depth.

    ``mark`` defaults to :func:`ocean_skill.plot.time_depth.default_mark`'s own
    call on the *reference* lane -- the ragged, real-visits one -- and is then
    applied to all three panels alike, the same reasoning :func:`time_depth`
    gives for its own single panel. Pass it explicitly to override either way.

    There is no ``domain``, ``region`` or ``gridline_kwargs``: a ``time_depth``
    row has no map to outline or gridline, the same omission :func:`section_row`
    makes for the same reason. ``tick_label_kwargs`` styles the date ticks every
    panel draws, unlike :func:`section_row` (whose plain Cartesian ticks have no
    gridliner styling to carry) -- a date axis's ticks are exactly the labels
    worth styling here, the same option :func:`time_depth` exposes for its own
    lone panel.

    ``title`` defaults to the variable name followed by any depth/time a
    ``select=`` collapsed (ordinarily neither, for a bare pooled comparison),
    then the station's own place and visit period --
    :attr:`~ocean_skill.plot.time_depth.TimeDepthGeometry.place_note`/
    ``period_note`` standing in for :func:`section_row`'s own path endpoints.

    Everything else -- sizing (``size``/``zoom``/``figsize``), ``font_scale``,
    ``fit_text``, ``align_colorbars``, ``metric_keys``, the ``*_kwargs`` dicts,
    ``rasterize``/``hover`` (interactive-only, see
    :func:`_warn_if_interactive_only`) -- means exactly what it does in
    :func:`section_row`.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot.time_depth import default_mark, prepare_time_depth_row
    from ocean_skill.plot.typography import SECTION_ASPECT

    _warn_if_interactive_only(rasterize, hover)
    values, geometry = prepare_time_depth_row(aligned)
    if mark is None:
        mark = default_mark(values["reference"])
    if title is None:
        title = suptitle_text(
            standard_name, (depth, time, geometry.place_note, geometry.period_note)
        )

    canvas = resolve_canvas(size, zoom)
    horizontal = colorbar_is_horizontal(
        SECTION_ASPECT,
        default_horizontal=True,  # one row: bars below, panels get the full cell width
        requested=(colorbar_kwargs or {}).get("orientation"),
    )
    figsize = figsize or auto_figsize(
        SECTION_ASPECT,
        nrows=1,
        canvas=canvas,
        font_scale=font_scale,
        horizontal_colorbar=horizontal,
        overhead=ROW_OVERHEAD_HORIZONTAL_CBAR if horizontal else ROW_OVERHEAD,
    )
    scale = _scale_for(figsize, nrows=1, font_scale=font_scale)
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    ims, lab = _draw_time_depth_row(
        axes,
        values,
        geometry,
        labels=labels or ("test", "reference"),
        units=units,
        standard_name=standard_name,
        metrics=metrics,
        mark=mark,
        metric_keys=metric_keys,
        title_kwargs=title_kwargs,
        metrics_kwargs=metrics_kwargs,
        tick_label_kwargs=tick_label_kwargs,
        shared_axis_labels=shared_axis_labels,
        scale=scale,
        defaults=defaults,
        robust=robust,
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

    if title:
        sup = fig.suptitle(
            title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs)
        )
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    if align_colorbars:
        _align_colorbars(fig)
    if fit_text:
        _fit_text_widths(fig)
    _warn_if_cramped(fig, canvas=canvas, nrows=1, panels=list(axes))
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def field_map_grid(
    items: list[dict[str, Any]],
    *,
    title: str | None = None,
    mark: str = "pcolormesh",
    ncols: int | None = None,
    domain: tuple[float, float, float, float] | np.ndarray | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    shared_axis_labels: bool = True,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
    robust: bool | float = False,
):
    """Draw one map per item -- several *variables*, not one variable's own facet axis.

    The ``field_facet`` counterpart for a :class:`~ocean_skill.field.FieldSet` of
    several maps (see :meth:`ocean_skill.field.Field._map_item`): each item is a single,
    already-reduced-to-one-instant field, so unlike :func:`field_facet` there is no
    facet coordinate ordering the panels and nothing shared for them to share a colour
    scale over. Each panel gets **its own** colour scale and **its own** colorbar --
    :func:`skill_map`'s convention, not :func:`field_facet`'s -- since different
    variables carry different units and unrelated ranges; a shared bar across nitrate
    and temperature would be meaningless on both ends.

    The grid is free (these panels have no inherent order): ``ncols`` defaults to
    :func:`~ocean_skill.plot.typography.facet_layout`, which reads the orientation off
    the domain's own aspect ratio, the same rule :func:`field_facet`'s one-facet-axis
    case and :func:`skill_map`'s single-item case use.

    Each panel is titled by its own **variable** (:func:`field_title`) -- the inverse of
    :func:`field_facet`, where the panels say *when* and the variable rides in the
    suptitle, because here the panels are exactly what differs and the suptitle is
    whatever the whole set shares instead (``title`` defaults to :func:`grid_suptitle`,
    which composes only the depth/time/region every item has in common, dropping the
    variable since the items' ``standard_name``s differ by construction).

    Every other parameter means what it means in :func:`field_facet`/:func:`skill_map`,
    including ``robust`` (see :func:`_limits`) -- each panel's own colour scale spans
    its full data range by default, or its 10th–90th percentile with ``robust=True``.
    ``rasterize``/``hover`` are accepted only so ``renderer="both"`` can pass one option
    set to each renderer (see :func:`_warn_if_interactive_only`) -- they are the
    interactive renderer's fix for a large mesh and do nothing here.
    """
    import warnings

    import matplotlib.pyplot as plt

    from ocean_skill.plot.typography import facet_figsize, facet_layout

    _warn_if_interactive_only(rasterize, hover)
    if not items:
        raise ValueError("field_map_grid needs at least one field, got none")

    n = len(items)
    aspect = _aspect_of(items[0]["field"])
    canvas = resolve_canvas(size, zoom)
    if title is None:
        title = grid_suptitle(items)

    if ncols is None:
        ncols, nrows = facet_layout(n, aspect, canvas=canvas)
    else:
        ncols = max(int(ncols), 1)
        nrows = -(-n // ncols)

    # Vertical, one per panel -- see skill_map's identical choice:
    # colorbar_is_horizontal forces horizontal above a wide-domain aspect,
    # which is right for one bar shared across a row but would put a bar
    # under *every* panel here at fixed height, something facet_figsize is
    # not charged for below.
    horizontal = str((colorbar_kwargs or {}).get("orientation", "vertical")).startswith(
        "h"
    )
    if horizontal:
        warnings.warn(
            "colorbar_kwargs={'orientation': 'horizontal'} puts a bar under every "
            "panel, but this family's height is not re-charged for that (see "
            "facet_figsize). Pass figsize= or zoom= to compensate.",
            stacklevel=_stacklevel.find(),
        )
    figsize = figsize or facet_figsize(
        aspect,
        nrows=nrows,
        ncols=ncols,
        title_every_row=True,  # every panel names its own variable
        canvas=canvas,
        # PANEL_W_FRACTION (0.72), not the facet default: a bar beside every panel,
        # not one shared bar with nothing beside the maps -- see skill_map's own
        # identical comment on this choice.
        panel_w_fraction=(
            PANEL_W_FRACTION_HORIZONTAL_CBAR if horizontal else PANEL_W_FRACTION
        ),
        font_scale=font_scale,
    )
    scale = type_scale(
        figsize,
        ncols=ncols,
        nrows=nrows,
        font_scale=font_scale,
        # the suptitle spans the page, sized as every other family's is rather than
        # off this grid's own column count -- see type_scale
        figure_ncols=REFERENCE_GRID[0],
    )
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    # FACET_COLORBAR_ASPECT is deliberately not applied: it exists for one bar
    # refitted across a whole row, and each bar here spans exactly one panel --
    # which is what the grid default already describes.
    merged_title = _merged(defaults["title_kwargs"], title_kwargs)
    merged_gridline = _merged(defaults["gridline_kwargs"], gridline_kwargs)
    merged_tick = _merged(defaults["tick_label_kwargs"], tick_label_kwargs)
    merged_suptitle = _merged(defaults["suptitle_kwargs"], suptitle_kwargs)
    title_pinned = _pinned(title_kwargs, "title_kwargs")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": _map_projection(*(item["field"] for item in items))},
        constrained_layout=True,
        squeeze=False,
    )
    flat = list(axes.ravel())

    for i, item in enumerate(items):
        ax = flat[i]
        col = i % ncols
        field = item["field"]
        standard_name = item.get("standard_name")
        cmap, _ = cmaps_for(standard_name)
        vmin, vmax = _limits(field, robust=robust)
        norm = norm_for(standard_name, vmin, vmax)
        im = _draw_map(
            ax,
            field,
            label=field_title(standard_name),
            cmap=cmap,
            norm=norm,
            mark=mark,
            domain=domain,
            gridline_kwargs=merged_gridline,
            tick_label_kwargs=merged_tick,
            title_kwargs=merged_title,
            left_labels=(col == 0) if shared_axis_labels else None,
            # The bottom row is ragged when n does not fill the grid, so the question
            # is "is there a panel below me?", not "am I in the last row?".
            bottom_labels=(i + ncols >= n) if shared_axis_labels else None,
            coastline_resolution=coastline_resolution,
            land=land,
        )
        ax.title._osk_size_pinned = title_pinned
        bar_label = f"[{item['units']}]" if item.get("units") else ""
        _draw_colorbar(
            fig, im, ax, bar_label, colorbar_kwargs, defaults["colorbar_kwargs"]
        )

    # Cells past the last panel carry no map and so no label artists -- hidden
    # rather than deleted, which keeps the drawn panels on the grid they were
    # sized for instead of letting the layout engine expand them into the gap.
    for ax in flat[n:]:
        ax.set_visible(False)

    if title:
        sup = fig.suptitle(title, **merged_suptitle)
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    if align_colorbars:
        _align_colorbars(fig)
    if fit_text:
        _fit_text_widths(fig)
    _warn_if_cramped(fig, ncols, canvas=canvas, nrows=nrows)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def section_row(
    aligned,
    *,
    labels: tuple[str, str] | None = None,
    title: str | None = None,
    units: str | None = None,
    standard_name: str | None = None,
    depth: str | None = None,
    time: str | None = None,
    metrics: dict[str, Any] | None = None,
    mark: str = "pcolormesh",
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    shared_axis_labels: bool = True,
    align_colorbars: bool = True,
    font_scale: float = 1.0,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    robust: bool | float = False,
):
    """Draw one ``test | reference | difference`` row of vertical sections.

    :func:`field_row` with :func:`ocean_skill.plot.section.prepare_section`'s
    geometry substituted for the map — a comparison whose select cuts a transect
    (see :func:`ocean_skill.comparison.Comparison.is_section`) gets this family
    instead, the same way one that reduces to a single time axis gets
    :func:`series`. Test and reference share one colour scale (the full range of
    the pair by default, or its 10th-90th percentile with ``robust=True``); the
    difference panel uses a diverging map centred on zero; metrics go in the
    difference panel's corner box — all exactly as
    :func:`field_row` draws a gridded comparison, just against depth and
    along-path distance instead of longitude and latitude.

    There is no ``domain``, ``region``, ``gridline_kwargs``, ``tick_label_kwargs``
    or ``row_label``: a section has no map to outline or gridline, and this is
    always the only (and so also the bottom) row — see :func:`section` for the
    single-panel case these omissions also apply to.

    ``title`` defaults to the variable name followed by the depth list, time and
    the path's own endpoints (``29.0°N, 94.5°W → 27.5°N, 90.0°W``) —
    :func:`ocean_skill.plot.section.SectionGeometry.path_note` standing in for
    the region a gridded comparison's title names instead.

    Everything else — sizing (``size``/``zoom``/``figsize``), ``font_scale``,
    ``fit_text``, ``align_colorbars``, ``metric_keys``, the ``*_kwargs`` dicts,
    ``rasterize``/``hover`` (interactive-only, see
    :func:`_warn_if_interactive_only`) — means exactly what it does in
    :func:`field_row`.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot.section import prepare_section_row
    from ocean_skill.plot.typography import SECTION_ASPECT

    _warn_if_interactive_only(rasterize, hover)
    values, geometry = prepare_section_row(aligned)
    if title is None:
        title = suptitle_text(standard_name, (depth, time, geometry.path_note))

    canvas = resolve_canvas(size, zoom)
    horizontal = colorbar_is_horizontal(
        SECTION_ASPECT,
        default_horizontal=True,  # one row: bars below, panels get the full cell width
        requested=(colorbar_kwargs or {}).get("orientation"),
    )
    figsize = figsize or auto_figsize(
        SECTION_ASPECT,
        nrows=1,
        canvas=canvas,
        font_scale=font_scale,
        horizontal_colorbar=horizontal,
        overhead=ROW_OVERHEAD_HORIZONTAL_CBAR if horizontal else ROW_OVERHEAD,
    )
    scale = _scale_for(figsize, nrows=1, font_scale=font_scale)
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    ims, lab = _draw_section_row(
        axes,
        values,
        geometry,
        labels=labels or ("test", "reference"),
        units=units,
        standard_name=standard_name,
        metrics=metrics,
        mark=mark,
        metric_keys=metric_keys,
        title_kwargs=title_kwargs,
        metrics_kwargs=metrics_kwargs,
        shared_axis_labels=shared_axis_labels,
        scale=scale,
        defaults=defaults,
        robust=robust,
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

    if title:
        sup = fig.suptitle(
            title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs)
        )
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    if align_colorbars:
        _align_colorbars(fig)
    if fit_text:
        _fit_text_widths(fig)
    _warn_if_cramped(fig, canvas=canvas, nrows=1, panels=list(axes))
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def _tight_extent(
    items: list[dict[str, Any]], names: tuple[str, ...], *, margin: float = 0.08
) -> tuple[float, float, float, float] | None:
    """Return a ``(lon0, lon1, lat0, lat1)`` box hugging the drawn skill surface.

    The union, over every item and every drawn metric, of the lon/lat of the cells
    that carry a value (``NaN`` cells — masked land, or an interpolated surface's
    ``maxdist`` blob edge — do not count), grown by ``margin`` of its own span so the
    surface is not flush against the frame. ``None`` when nothing is finite (an
    empty surface), which leaves the caller on the default whole-grid autoscale.
    """
    lons: list[float] = []
    lats: list[float] = []
    for item in items:
        skill = item["skill"]
        lon = np.asarray(skill["lon"].values, dtype="float64")
        lat = np.asarray(skill["lat"].values, dtype="float64")
        finite = np.zeros(lon.shape, dtype=bool)
        for name in names:
            if name in skill:
                finite |= np.isfinite(np.asarray(skill[name].values))
        if finite.any():
            lons += [float(lon[finite].min()), float(lon[finite].max())]
            lats += [float(lat[finite].min()), float(lat[finite].max())]
    if not lons:
        return None
    lon0, lon1, lat0, lat1 = min(lons), max(lons), min(lats), max(lats)
    dlon = (lon1 - lon0) or 1.0
    dlat = (lat1 - lat0) or 1.0
    return (lon0 - margin * dlon, lon1 + margin * dlon,
            lat0 - margin * dlat, lat1 + margin * dlat)


def resolve_extent(
    extent, items, names
) -> tuple[float, float, float, float] | None:
    """Turn the ``extent`` plot option into a ``(lon0, lon1, lat0, lat1)`` box.

    ``None`` stays ``None`` (leave the caller's default whole-grid framing);
    ``"tight"`` becomes the drawn surface's own box (:func:`_tight_extent`, itself
    ``None`` when nothing is finite); a 4-tuple is validated and passed through.
    Shared with :mod:`ocean_skill.plot.holoviews_renderer` so both backends read the
    option the same way.
    """
    if extent is None:
        return None
    if isinstance(extent, str):
        if extent != "tight":
            raise ValueError(
                f"extent={extent!r} — expected 'tight', a (lon_min, lon_max, "
                "lat_min, lat_max) tuple, or None."
            )
        return _tight_extent(items, names)
    box = tuple(float(v) for v in extent)
    if len(box) != 4:
        raise ValueError(
            f"extent={extent!r} — a bbox must be (lon_min, lon_max, lat_min, lat_max)."
        )
    return box


def _apply_extent(axes, extent, items, names) -> None:
    """Set the view extent on every map panel from the ``extent`` plot option."""
    import cartopy.crs as ccrs

    box = resolve_extent(extent, items, names)
    if box is None:
        return
    for ax in axes:
        ax.set_extent(box, crs=ccrs.PlateCarree())


def skill_map(
    items: list[dict[str, Any]],
    *,
    metric_names: tuple[str, ...] | None = None,
    title: str | None = None,
    mark: str = "pcolormesh",
    save: str | Path | None = None,
    domain: tuple[float, float, float, float] | np.ndarray | None = None,
    extent: str | tuple[float, float, float, float] | None = None,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    title_kwargs: dict[str, Any] | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    row_label_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    suptitle_kwargs: dict[str, Any] | None = None,
    shared_axis_labels: bool = True,
    align_colorbars: bool = True,
    shared_limits: bool = False,
    layout: str = "rows",
    font_scale: float = 1.0,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    fit_text: bool = True,
    rasterize: bool | str | None = None,
    hover: bool | None = None,
    station_markers: bool = True,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
):
    """Draw one map per skill metric: where the model agrees, metric by metric.

    The figure for a comparison scored over an axis (``compare(..., over="time")``).
    Every panel is the *same* comparison judged by a different measure, each computed
    cell by cell along that axis — so bias says where the model runs high or low,
    correlation where it tracks the observations through time, and the variability
    ratio where it is over- or under-dispersed. There is no test/reference/difference
    row here because there is nothing to set beside anything: the maps *are* it.

    Each panel gets its own colour scale by default, unlike :func:`field_facet`, whose
    panels share one because they are one quantity at different times — a different
    *metric* (bias vs. a dimensionless correlation) has no shared scale to have, and
    there is no way to ask for one across metrics. The colours come from
    :func:`ocean_skill.colormaps.metric_colors`, which both renderers call, so a bias
    panel is symmetric about zero and a correlation panel spans (−1, 1) whichever
    backend drew it.

    **Across rows of the *same* metric**, though — several comparisons stacked, one
    row each — sharing a scale is exactly what makes them comparable by colour:
    ``shared_limits=True`` pools that metric's values over every row before choosing
    its limits, and draws one colorbar per metric spanning the rows instead of one per
    panel. The default (``False``) keeps each row's own scale, which is honest about
    that row's own range but means two rows of "the same" metric can carry different
    colours for the same shade.

    Each panel is also annotated with that metric's **overall** value — the same number
    reduced over space *and* the scored axis together, from ``metrics``' record — in
    the corner box a comparison row uses for the same purpose. The map and the single
    number are the same statistic at two resolutions, and reading one without the other
    is how a good average hides a bad region.

    Several items (a :func:`compare` fan-out) become rows: metrics across, comparisons
    down, each row named at its left edge as :func:`field_grid`'s are — this is
    ``layout="rows"``, the default. ``layout="columns"`` transposes it: comparisons
    across, metrics down, each row named at its left edge instead and each comparison
    titled at its column's top — the natural arrangement for putting two or three
    models side by side. ``layout`` only has an effect with more than one item; a
    single item's panels have no inherent order, so the grid is free and ``ncols``
    defaults to :func:`~ocean_skill.plot.typography.facet_layout`, which reads the
    orientation off the domain's shape exactly as :func:`field_facet` does.

    ``metric_names`` picks and orders the panels from what the item carries; a name it
    does not carry raises (see :func:`metric_panels`). Every other parameter means
    what it means in :func:`field_facet`.

    ``extent`` crops the *view* every panel shows, without touching the data or its
    interpolation. ``None`` (the default) frames the whole grid each panel was drawn
    on — the model's domain for ``grid="model"``, the padded station box for
    ``grid="regular"`` — which for a small cluster of stations leaves a lot of empty
    ocean. ``"tight"`` instead frames just the drawn skill surface (the non-``NaN``
    cells, which for an interpolated map is the ``maxdist`` blob around the stations)
    with a small margin, so the figure is filled by what it actually has a value for.
    A ``(lon_min, lon_max, lat_min, lat_max)`` tuple sets an exact window. Every panel
    gets the same extent, so a grid of them stays aligned.

    An item carrying ``stations`` (see :func:`ocean_skill.plot.map_metrics.build_items`
    — an interpolated surface fit through scattered per-station values, rather than a
    scored comparison's own cell-by-cell map) additionally draws each station's true
    value as a dot, in the same colour scale as the surface underneath it. That is the
    one thing distinguishing an interpolated metric map from a scored one here: where
    the surface has actual support, and where it is only filling a gap between
    stations. ``station_markers=False`` suppresses that dot overlay, leaving only the
    surface — its distance-masked extent already shows where the data is, and with
    thousands of near-coincident stations the dots smear into a mask that hides the
    very surface they annotate.

    ``rasterize``/``hover`` are accepted only so ``renderer="both"`` can pass one option
    set to each renderer (see :func:`_warn_if_interactive_only`) — they are the
    interactive renderer's fix for a large mesh and do nothing here.

    ``coastline_resolution``/``land`` pick the coastline/land dataset and the land
    fill's visibility for every panel — see :func:`field_row`'s docstring.
    """
    import warnings

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    from ocean_skill.colormaps import metric_colors
    from ocean_skill.plot.typography import facet_figsize, facet_layout

    _warn_if_interactive_only(rasterize, hover)
    if not items:
        raise ValueError("skill_map needs at least one comparison, got none")
    names = metric_panels(items[0]["skill"], metric_names)
    if not names:
        raise ValueError(
            "this comparison carries no 2-D metric maps to draw. It was probably not "
            'scored over an axis: build it with compare(..., over="time").'
        )
    for item in items[1:]:  # every row must be able to fill every column
        metric_panels(item["skill"], names)

    if layout not in ("rows", "columns"):
        raise ValueError(f"layout={layout!r} — expected 'rows' or 'columns'")
    titles = metric_panel_titles(names)
    aspect = _aspect_of(items[0]["skill"][names[0]])
    canvas = resolve_canvas(size, zoom)
    stacked = len(items) > 1
    if stacked:
        # two axes fix the grid, as field_facet's row_dim does. An ncols disagreeing
        # with the layout's own axis would drop panels.
        if layout == "columns":
            if ncols is not None and int(ncols) != len(items):
                raise ValueError(
                    f"ncols={ncols} contradicts layout='columns' with a "
                    f"{len(items)}-comparison set: the grid is {len(names)} x "
                    f"{len(items)} (one column per comparison), so there is no "
                    "column count left to choose."
                )
            nrows, ncols = len(names), len(items)
            panels = [(item_idx, name) for name in names for item_idx in range(len(items))]
        else:
            if ncols is not None and int(ncols) != len(names):
                raise ValueError(
                    f"ncols={ncols} contradicts a {len(items)}-comparison set: the grid is "
                    f"{len(items)} x {len(names)} (one column per metric), so there is no "
                    "column count left to choose."
                )
            nrows, ncols = len(items), len(names)
            panels = [(row, name) for row in range(nrows) for name in names]
    else:
        if ncols is None:
            ncols, nrows = facet_layout(len(names), aspect, canvas=canvas)
        else:
            ncols = max(int(ncols), 1)
            nrows = -(-len(names) // ncols)
        panels = [(0, name) for name in names]

    # Vertical, one per panel -- and *not* through colorbar_is_horizontal, which forces
    # horizontal above a 2.5 aspect (a Gulf-shaped domain) and would put a bar stack
    # under every row at ~1in of fixed height, which facet_figsize cannot charge for.
    horizontal = str((colorbar_kwargs or {}).get("orientation", "vertical")).startswith(
        "h"
    )
    if horizontal:
        warnings.warn(
            "colorbar_kwargs={'orientation': 'horizontal'} puts a bar under every "
            "panel, but this family's height is not re-charged for that (see "
            "facet_figsize), so the maps may be squeezed. Pass figsize= or zoom=.",
            stacklevel=_stacklevel.find(),
        )
    figsize = figsize or facet_figsize(
        aspect,
        nrows=nrows,
        ncols=ncols,
        # every panel is a different metric, so every row carries its own titles --
        # except when the rows are comparisons and the columns repeat down the page
        title_every_row=not stacked,
        # the canvas whole, rather than its width and a height defaulted to the page:
        # that spelling silently capped size="free" at the page, which is the one thing
        # an uncapped canvas is for
        canvas=canvas,
        # PANEL_W_FRACTION, not FACET_PANEL_W_FRACTION: 0.88 is the allowance for a grid
        # whose panels *share* one bar and so have nothing beside them. A bar in every
        # cell is what 0.72 describes, and getting this backwards silently squeezes the
        # maps -- the failure typography's own commentary is about.
        panel_w_fraction=(
            PANEL_W_FRACTION_HORIZONTAL_CBAR if horizontal else PANEL_W_FRACTION
        ),
        font_scale=font_scale,
    )
    scale = type_scale(
        figsize,
        ncols=ncols,
        nrows=nrows,
        font_scale=font_scale,
        # the suptitle spans the page, so it is sized as every other family's is rather
        # than off this grid's column count -- see type_scale
        figure_ncols=REFERENCE_GRID[0],
    )
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    # FACET_COLORBAR_ASPECT is deliberately *not* applied: it exists for one bar
    # refitted across every row, and each bar here spans exactly one panel -- which is
    # what the grid default already describes.
    merged_title = _merged(defaults["title_kwargs"], title_kwargs)
    merged_gridline = _merged(defaults["gridline_kwargs"], gridline_kwargs)
    merged_tick = _merged(defaults["tick_label_kwargs"], tick_label_kwargs)
    merged_row_label = _merged(defaults["row_label_kwargs"], row_label_kwargs)
    merged_metrics = _merged(defaults["metrics_kwargs"], metrics_kwargs)
    title_pinned = _pinned(title_kwargs, "title_kwargs")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": _map_projection(*(item["skill"] for item in items))},
        constrained_layout=True,
        squeeze=False,
    )
    flat = list(axes.ravel())
    arrays = {i: metric_arrays(item["skill"], names) for i, item in enumerate(items)}

    # One colour scale per metric, pooled over every row, when asked to share: fit
    # only once names are known and every row's array is in hand, before any panel
    # is drawn, so every row of a metric — however many — draws with the same norm.
    shared_colors: dict[str, Any] = {}
    if shared_limits and stacked:
        for name in names:
            pooled = np.concatenate(
                [np.asarray(arrays[i][name]).ravel() for i in range(len(items))]
            )
            shared_colors[name] = metric_colors(
                name, pooled, standard_name=items[0].get("standard_name")
            )

    # Colorbars are drawn per panel by default, but a shared scale wants exactly one
    # bar per metric spanning every row it appears in -- collected here as panels are
    # drawn, then (only in the shared case) issued once each after the loop.
    panel_axes: dict[str, list[Any]] = {name: [] for name in names}
    panel_mappable: dict[str, Any] = {}

    for i, (row_index, name) in enumerate(panels):
        ax = flat[i]
        row, col = divmod(i, ncols)
        item = items[row_index]
        colors = (
            shared_colors[name]
            if shared_limits and stacked
            else metric_colors(
                name, arrays[row_index][name], standard_name=item.get("standard_name")
            )
        )
        # Every row shows every metric once, so within a layout's own repeating axis
        # the label only needs to appear once: "rows" repeats metrics across columns,
        # so the metric title is shown on the top row only (row-label carries the
        # comparison down the left edge instead); "columns" transposes both roles.
        if layout == "columns" and stacked:
            label = item.get("row_label") if row == 0 else None
        else:
            label = titles[names.index(name)] if (not stacked or row == 0) else None
        im = _draw_map(
            ax,
            item["skill"][name],
            label=label,
            cmap=colors.cmap,
            norm=colors.norm(),
            mark=mark,
            domain=domain,
            gridline_kwargs=merged_gridline,
            tick_label_kwargs=merged_tick,
            title_kwargs=merged_title,
            left_labels=(col == 0) if shared_axis_labels else None,
            # the bottom row is ragged when the metrics do not fill the grid, so the
            # question is "is there a panel below me?", not "am I in the last row?"
            bottom_labels=(i + ncols >= len(panels)) if shared_axis_labels else None,
            coastline_resolution=coastline_resolution,
            land=land,
        )
        stations = item.get("stations")
        if station_markers and stations is not None and name in stations["values"]:
            # Same cmap/norm as the surface beneath: a dot and the patch of surface
            # under it are the same statistic, so they read as one colour scale, not
            # two. zorder above the domain outline (4) and below nothing else drawn
            # in this panel.
            ax.scatter(
                stations["lon"],
                stations["lat"],
                c=stations["values"][name],
                cmap=colors.cmap,
                norm=colors.norm(),
                s=26,
                transform=ccrs.PlateCarree(),
                # A dark edge (not white) so a station coloured near the fill's own
                # "good" end — e.g. a white-centred metric colormap — still shows as
                # a dot rather than disappearing into the surface beneath it.
                edgecolor="0.15",
                linewidth=0.6,
                zorder=5,
            )
        if label is not None:
            ax.title._osk_size_pinned = title_pinned
        edge_label = (
            titles[names.index(name)]
            if (layout == "columns" and stacked)
            else item.get("row_label")
        )
        if col == 0 and stacked and edge_label:
            _add_row_label(ax, edge_label, merged_row_label)
            ax._osk_row_label._osk_size_pinned = _pinned(
                row_label_kwargs, "row_label_kwargs"
            )
        panel_axes[name].append(ax)
        panel_mappable[name] = im
        # the metric's overall value, in the same corner box a comparison row uses for
        # the same reason -- stashed on the axes as that one is
        overall = _metrics_text(item.get("metrics"), (name,))
        if overall:
            ax._osk_metrics_text = ax.text(
                0.02,
                0.02,
                overall,
                transform=ax.transAxes,
                zorder=5,
                **merged_metrics,
            )
        if not (shared_limits and stacked):
            _draw_colorbar(
                fig,
                im,
                ax,
                _units_label(item["skill"][name]),
                colorbar_kwargs,
                defaults["colorbar_kwargs"],
            )

    if shared_limits and stacked:
        # One bar per metric, spanning every row it appears in -- _draw_colorbar
        # already accepts a list of parent axes (a field grid's own difference-panel
        # bar does the same), and _align_colorbars below re-fits it to their union.
        for name in names:
            _draw_colorbar(
                fig,
                panel_mappable[name],
                panel_axes[name],
                _units_label(items[0]["skill"][name]),
                colorbar_kwargs,
                defaults["colorbar_kwargs"],
            )

    # Cells past the last panel carry no map and so no label artists — hidden rather
    # than deleted, which keeps the drawn panels on the grid they were sized for.
    for ax in flat[len(panels) :]:
        ax.set_visible(False)

    # Crop the view after every panel is drawn (so "tight" can read the drawn
    # surface's own extent), on the visible map panels only.
    _apply_extent(flat[: len(panels)], extent, items, names)

    if title:
        sup = fig.suptitle(
            title, **_merged(defaults["suptitle_kwargs"], suptitle_kwargs)
        )
        sup._osk_size_pinned = _pinned(suptitle_kwargs, "suptitle_kwargs")
    _fit_left_margin(fig)
    if align_colorbars:
        _align_colorbars(fig)
    if fit_text:
        _fit_text_widths(fig)
        _clear_row_labels(fig)
    _warn_if_cramped(fig, ncols, canvas=canvas, nrows=nrows)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def _units_label(da) -> str:
    """Return one metric map's colorbar label: its units, or nothing for a number.

    Units go on each panel's own bar rather than into its title, because unlike
    :func:`field_facet` every panel here has a bar of its own and a title copy would
    say it twice.
    """
    units = str(da.attrs.get("units", "") or "")
    return f"[{units}]" if units else ""


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
    domain: tuple[float, float, float, float] | np.ndarray | None = None,
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
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    progress: bool = True,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
    robust: bool | float = False,
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
    # Exactly what field_row decides, and for the reason in this function's docstring: a
    # frame is meant to *be* that row. Hardcoding True here made that false for anything
    # not wide — a tall domain's bars sat below the maps in the movie and beside them in
    # the still, with the figure reshaped to match.
    aspect = _map_aspect(frames, reference_name)
    horizontal = colorbar_is_horizontal(
        aspect,
        default_horizontal=True,
        requested=(colorbar_kwargs or {}).get("orientation"),
    )
    figsize = figsize or auto_figsize(
        aspect,
        nrows=1,
        canvas=resolve_canvas(size, zoom),
        font_scale=font_scale,
        horizontal_colorbar=horizontal,
    )
    scale = _scale_for(figsize, nrows=1, font_scale=font_scale)
    defaults = _style_defaults(scale, horizontal_colorbar=horizontal)
    # the axes' frame follows the data; the transform below stays geographic degrees
    proj = ccrs.PlateCarree()

    seq_norm = div_norm = None
    if shared_limits and len(frames) > 1:
        seq_norm, div_norm = _shared_norms(
            frames, test_name, reference_name, robust=robust
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        subplot_kw={"projection": _map_projection(first["aligned"])},
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
        coastline_resolution=coastline_resolution,
        land=land,
        robust=robust,
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
    # alignment first, then the fit — see _fit_text_widths. It matters more here than
    # on a still: every frame is drawn into this one layout, so a label clipped by
    # measuring it against a pre-alignment bar is clipped for the whole movie.
    if align_colorbars:
        _align_colorbars(fig)
    _fit_text_widths(fig)
    _clear_row_labels(fig)

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
    domain: tuple[float, float, float, float] | np.ndarray | None = None,
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
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    progress: bool = True,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
    robust: bool | float = False,
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
    months, ``Jan`` for a climatology, ``50 m`` for a level — including where a month is
    too coarse to tell one frame from another, which a movie runs into more often than a
    grid does, being as often over the unreduced axis as over a reduction.

    The suptitle likewise says what the frame labels do not: it defaults to the
    variable's short name (see :func:`field_title`), as :func:`field_facet`'s does, and
    ``title=""`` drops it. It stays fixed while the frames play, being the one thing
    about the figure that does not change.

    ``row_dim`` has no counterpart: a movie has one axis to play, and two facet axes
    would need one to become the panels — which is what :func:`field_facet` is for.
    Every other parameter means what it does there, or in :func:`field_movie` for the
    movie-specific ones (``save``, ``fps``, ``dpi``, ``every``, ``frame_label``).
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    from ocean_skill.plot.typography import REFERENCE_GRID, facet_figsize

    title = field_title(standard_name) if title is None else title
    facet_dim = _one_facet_axis(field, facet_dim)
    indices = _select_frames(list(range(int(field.sizes[facet_dim]))), every)
    labels = frame_labels(field[facet_dim]) if facet_dim in field.coords else None
    aspect = _aspect_of(field)
    # One panel, so the grid's long edge is the map's own: a wide domain takes a
    # horizontal bar beneath it, a tall one a vertical bar beside it. Same rule
    # field_facet applies to its grid, which for a single cell *is* the map.
    horizontal = aspect > 1.0
    figsize = figsize or facet_figsize(
        aspect,
        nrows=1,
        ncols=1,
        canvas=resolve_canvas(size, zoom),
        font_scale=font_scale,
    )
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
    vmin, vmax = _limits(scope, robust=robust)
    norm = norm_for(standard_name, vmin, vmax)
    cmap, _ = cmaps_for(standard_name)

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": _map_projection(field)},
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
        coastline_resolution=coastline_resolution,
        land=land,
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
    # alignment first, then the fit — see _fit_text_widths and the note in field_movie
    _align_colorbars(fig)
    _fit_text_widths(fig)
    _clear_row_labels(fig)
    _centre_suptitle(fig)

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


def locations(
    items,
    *,
    title: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    legend: bool = True,
    marker_size: float = 80.0,
    tiles: str | bool | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    size: str | Canvas | tuple[float, float | None] | float | None = None,
    zoom: float = 1.0,
    font_scale: float = 1.0,
    title_kwargs: dict[str, Any] | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    tick_label_kwargs: dict[str, Any] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    coastline_resolution: str = DEFAULT_COASTLINE_RESOLUTION,
    land: bool | float = True,
):
    """Map where things sit: markers for points, dashed boxes for extents and
    domains, solid lines for selection slices.

    Items come from :func:`ocean_skill.plot.locations.build_items` (pure catalog
    metadata) and/or :func:`ocean_skill.plot.map_locations.build_map_items` (a
    plotted selection) — no field, no colormap and no colorbar either way; colour
    keys the item's ``featureType`` instead, off the shared constants and
    :func:`~ocean_skill.plot.locations.style_for` in
    :mod:`ocean_skill.plot.locations`, and the legend is the key to it.

    ``extent`` is ``(lon_min, lat_min, lon_max, lat_max)`` — the same bbox shape
    ``find(bbox=...)`` takes — and defaults to a frame around every item (set by
    :func:`~ocean_skill.plot.map_locations.map_locations`). ``tiles`` is accepted
    so ``renderer="both"`` can pass one set of options, but web tiles are the
    interactive renderer's; here it warns and draws the usual coastline basemap.

    ``coastline_resolution``/``land`` pick that basemap's coastline/land dataset and
    the land fill's visibility — see :func:`field_row`'s docstring.
    """
    import warnings

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ocean_skill.plot.locations import FEATURE_TYPE_ORDER, style_for
    from ocean_skill.plot.proj_check import warn_projection_skew
    from ocean_skill.plot.summary import _MARKERS

    warn_projection_skew()
    if tiles:
        warnings.warn(
            "tiles= only affects the interactive renderer; the static map draws "
            "coastlines. Pass renderer='holoviews' for a web basemap.",
            stacklevel=_stacklevel.find(),
        )

    if extent is None:
        from ocean_skill.plot.locations import _default_extent

        extent = _default_extent(items)
    lon0, lat0, lon1, lat1 = (float(v) for v in extent)

    aspect = max(lon1 - lon0, 1e-6) / max(lat1 - lat0, 1e-6)
    canvas = resolve_canvas(size, zoom)
    if figsize is None:
        # one panel, no colorbar: the facet fraction (a shared-bar grid's) is the
        # closest existing answer to "the map keeps nearly the whole cell"
        figsize = auto_figsize(
            aspect,
            nrows=1,
            ncols=1,
            canvas=canvas,
            font_scale=font_scale,
            panel_w_fraction=FACET_PANEL_W_FRACTION,
        )
    scale = type_scale(figsize, ncols=1, nrows=1, font_scale=font_scale)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": proj}, layout="constrained"
    )
    if (lon0, lat0, lon1, lat1) == (-180.0, -90.0, 180.0, 90.0):
        ax.set_global()
    else:
        ax.set_extent((lon0, lon1, lat0, lat1), crs=proj)
    _basemap(
        ax,
        gridline_kwargs=_merged(DEFAULT_GRIDLINE_KWARGS, gridline_kwargs),
        tick_label_kwargs=_merged(
            {**DEFAULT_TICK_LABEL_KWARGS, "size": scale["tick_label"]},
            tick_label_kwargs,
        ),
        coastline_resolution=coastline_resolution,
        land=land,
    )

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["featureType"], []).append(item)
    ordered = [ft for ft in FEATURE_TYPE_ORDER if ft in groups]
    ordered += [ft for ft in groups if ft not in FEATURE_TYPE_ORDER]

    handles = []
    for feature_type in ordered:
        style = style_for(feature_type)
        color = style["color"]
        linestyle = style["linestyle"]
        points = [i for i in groups[feature_type] if i["kind"] == "point"]
        extents = [i for i in groups[feature_type] if i["kind"] == "extent"]
        paths = [i for i in groups[feature_type] if i["kind"] in ("line", "ring")]
        if points:
            marker = style["marker"] or _MARKERS[style["marker_index"] % len(_MARKERS)]
            ax.scatter(
                [p["lon"] for p in points],
                [p["lat"] for p in points],
                transform=proj,
                color=color,
                marker=marker,
                s=marker_size,
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="",
                    marker=marker,
                    markersize=8,
                    color=color,
                    markeredgecolor="white",
                    label=feature_type,
                )
            )
        if extents:
            for item in extents:
                for lo, la, hi, ha in item["bboxes"]:
                    ax.plot(
                        [lo, hi, hi, lo, lo],
                        [la, la, ha, ha, la],
                        transform=proj,
                        color=color,
                        lw=1.0,
                        ls=linestyle,
                        zorder=4,
                    )
            handles.append(
                Line2D([], [], linestyle=linestyle, color=color, label=feature_type)
            )
        if paths:
            # "line" (a selection slice) draws solid and on top; "ring" (a domain
            # outline) draws dashed and beneath — the same convention the
            # extent/point split above keeps, so a mixed group's legend entry
            # still reads as one thing even though it drew two ways.
            for item in paths:
                solid = item["kind"] == "line"
                for seg in item["paths"]:
                    ax.plot(
                        seg[:, 0],
                        seg[:, 1],
                        transform=proj,
                        color=color,
                        lw=1.8 if solid else 1.0,
                        ls="-" if solid else linestyle,
                        zorder=5 if solid else 4,
                    )
            any_solid = any(item["kind"] == "line" for item in paths)
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="-" if any_solid else linestyle,
                    lw=1.8 if any_solid else 1.0,
                    color=color,
                    label=feature_type,
                )
            )

    if legend and handles:
        # framed, unlike the series default: this key floats over a map, and
        # unbacked text over coastlines and extent boxes is unreadable
        ax.legend(
            handles=handles,
            **_merged(
                {
                    "frameon": True,
                    "framealpha": 0.85,
                    "edgecolor": "0.6",
                    "fontsize": scale["legend"],
                },
                legend_kwargs,
            ),
        )
    ax.set_title(
        title or "",
        **_merged(
            {**DEFAULT_TITLE_KWARGS, "fontsize": scale["title"]}, title_kwargs
        ),
    )
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def _nested_owner(key: str) -> str | None:
    """Which ``*_kwargs`` dict ``key`` belongs inside, if any.

    Styling here lives in nested dicts, so a plausible-looking option can be real and
    still be wrong at the top level — ``label_size`` is a colorbar key, not a
    ``field_grid`` parameter.

    A name that *is* a top-level parameter is never redirected, even when a nested dict
    happens to use the same key. ``size`` is both the canvas parameter and
    ``tick_label_kwargs``' font-size key, so without this check the interactive renderer
    told callers that ``size="slide"`` belonged inside ``tick_label_kwargs``.
    """
    if key in _top_level_options():
        return None
    for name, defaults in _NESTED_KWARGS.items():
        if key in defaults:
            return name
    # colorbar_kwargs additionally forwards any label_*/tick_* key it is handed, so
    # those never appear in the defaults but are still colorbar options.
    if key.startswith(("label_", "tick_")):
        return "colorbar_kwargs"
    return None


@functools.cache
def _top_level_options() -> frozenset[str]:
    """Every keyword a top-level family accepts directly, rather than inside a dict.

    Read off the signatures rather than listed, so adding a parameter cannot leave this
    behind — which is exactly how ``size`` came to be misreported as a nested key.
    """
    import inspect

    return frozenset(
        name
        for fn in (
            field_row,
            field_grid,
            field_facet,
            series,
            section,
            section_row,
            cross,
            profile,
            skill_map,
            locations,
            time_depth,
            time_depth_row,
        )
        for name in inspect.signature(fn).parameters
    )


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
        if key == "secondary_y" and fn.__name__ == "profile":
            # secondary_y is a real option -- of series, not profile -- so
            # _nested_owner (which only redirects into *_kwargs dicts) has
            # nothing to say about it; name the actual spelling instead.
            lines.append(
                "  'secondary_y' is not an option of profile() -- a profile's "
                "value axis is x (depth is y), so its twin is secondary_x"
            )
        elif owner:
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
    from ocean_skill.plot.portrait import portrait
    from ocean_skill.plot.summary import paired, target, taylor

    opts = {**spec.options, **kwargs}
    family = spec.family

    if family == "field_grid":
        _check_options(field_grid, opts)
    elif family == "field_row":
        _check_options(field_row, opts)
    elif family == "field_facet":
        _check_options(field_facet, opts)
    elif family == "field_map_grid":
        _check_options(field_map_grid, opts)
    elif family == "field_movie":
        _check_options(field_movie, opts)
    elif family == "facet_movie":
        _check_options(facet_movie, opts)
    elif family == "series":
        _check_options(series, opts)
    elif family == "section":
        _check_options(section, opts)
    elif family == "section_row":
        _check_options(section_row, opts)
    elif family == "cross":
        _check_options(cross, opts)
    elif family == "time_depth":
        _check_options(time_depth_grid if len(spec.items) > 1 else time_depth, opts)
    elif family == "time_depth_row":
        _check_options(time_depth_row, opts)
    elif family == "profile":
        _check_options(profile, opts)
    elif family == "skill_map":
        _check_options(skill_map, opts)
    elif family == "locations":
        _check_options(locations, opts)
    elif family == "portrait":
        _check_options(portrait, opts)

    if family == "field_facet":
        item = spec.single
        return field_facet(
            item["field"],
            facet_dim=item.get("facet_dim"),
            row_dim=item.get("row_dim"),
            units=item.get("units"),
            standard_name=item.get("standard_name"),
            depth=item.get("depth"),
            label=item.get("label"),
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
            depth=item.get("depth"),
            time=item.get("time"),
            region=item.get("region"),
            metrics=item.get("metrics"),
            **opts,
        )
    if family == "skill_map":
        return skill_map(spec.items, **opts)
    if family == "field_map_grid":
        return field_map_grid(spec.items, **opts)
    if family == "locations":
        return locations(spec.items, **opts)
    if family == "field_grid":
        return field_grid(spec.items, **opts)
    if family == "field_movie":
        return field_movie(spec.items, **opts)
    if family == "series":
        return series(spec.items, **opts)
    if family == "profile":
        return profile(spec.items, **opts)
    if family == "section":
        item = spec.single
        return section(
            item["field"],
            units=item.get("units"),
            standard_name=item.get("standard_name"),
            depth=item.get("depth"),
            label=item.get("label"),
            **opts,
        )
    if family == "section_row":
        item = spec.single
        return section_row(
            item["aligned"],
            units=item.get("units"),
            standard_name=item.get("standard_name"),
            depth=item.get("depth"),
            time=item.get("time"),
            metrics=item.get("metrics"),
            **opts,
        )
    if family == "cross":
        return cross(spec.items, **opts)
    if family == "time_depth":
        if len(spec.items) > 1:
            return time_depth_grid(spec.items, **opts)
        item = spec.single
        return time_depth(
            item["field"],
            units=item.get("units"),
            standard_name=item.get("standard_name"),
            label=item.get("label"),
            **opts,
        )
    if family == "time_depth_row":
        item = spec.single
        return time_depth_row(
            item["aligned"],
            units=item.get("units"),
            standard_name=item.get("standard_name"),
            depth=item.get("depth"),
            time=item.get("time"),
            metrics=item.get("metrics"),
            **opts,
        )
    if family in ("taylor", "target", "paired", "portrait"):
        # summary families work from metric records, which the spec carries per item
        summary_fns = {
            "taylor": taylor,
            "target": target,
            "paired": paired,
            "portrait": portrait,
        }
        return summary_fns[family]([_Record(i) for i in spec.items], **opts)
    raise NotImplementedError(f"matplotlib renderer: family {family!r} not implemented")


class _Record:
    """Adapt a spec item to the ``.metrics()``/``.label``/``.units`` interface
    summaries need.
    """

    def __init__(self, item: dict[str, Any]):
        self._item = item

    def metrics(self) -> dict[str, Any]:
        return self._item.get("metrics", {})

    @property
    def label(self):
        return self._item.get("label")

    @property
    def units(self):
        return self._item.get("units")


register_renderer("matplotlib", render)
