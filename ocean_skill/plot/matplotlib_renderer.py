"""Static matplotlib renderer (PNG/JPG/PDF; mp4/gif later via FuncAnimation).

Currently implements the **field row**: ``test | reference | difference`` maps for a
gridded comparison. Test and reference share one colour scale (so they are visually
comparable) taken from the 10th–90th percentile of the pair; the difference panel uses
a diverging map centred on zero. Metrics go in a corner box, leaving the title for
identity. Registers itself under ``"matplotlib"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ocean_skill.colormaps import cmaps_for, norm_for
from ocean_skill.plot.registry import register_renderer

__all__ = ["field_grid", "field_row", "render"]

PAGE_W = 8.5  # inches; figures must fit a portrait page
PAGE_H = 11.0


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
#: ``y`` is pinned rather than left to matplotlib's automatic title placement, which
#: is broken over a cartopy GeoAxES carrying gridline labels: matplotlib 3.11 places
#: the title above the union of the axes' children's bboxes, cartopy contributes an
#: empty ``(inf, inf, -inf, -inf)`` one, and the title's y comes out infinite. Its
#: window extent is then NaN, which makes the whole *axes* report a NaN tight bbox,
#: which drops that axes out of the figure's tight bbox — so ``bbox_inches="tight"``,
#: used both by our own ``save=`` and by Jupyter's inline backend, silently crops the
#: leftmost column out of the figure. Supplying any explicit ``y`` skips the automatic
#: placement that computes the infinity. Identical output on 3.10, where it is a no-op.
DEFAULT_TITLE_KWARGS: dict[str, Any] = {"fontsize": 8, "y": 1.0}
DEFAULT_GRIDLINE_KWARGS: dict[str, Any] = {
    "linewidth": 0.2,
    "color": "0.6",
    "alpha": 0.6,
}
DEFAULT_TICK_LABEL_KWARGS: dict[str, Any] = {"size": 5}
DEFAULT_ROW_LABEL_KWARGS: dict[str, Any] = {
    "fontsize": 7,
    "rotation": 90,
    "va": "center",
    "ha": "center",
    "weight": "normal",
}
DEFAULT_METRICS_KWARGS: dict[str, Any] = {
    "fontsize": 5.5,
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
#: field_row draws one row per figure (page-width, horizontal bars below); field_grid
#: stacks several rows (vertical bars beside them, smaller) — same keys, different
#: starting point.
DEFAULT_COLORBAR_KWARGS_ROW: dict[str, Any] = {
    "orientation": "horizontal",
    "pad": 0.04,
    "shrink": 0.85,
    "aspect": 30,
    "label_size": 7,
}
DEFAULT_COLORBAR_KWARGS_GRID: dict[str, Any] = {
    "orientation": "vertical",
    "pad": 0.015,
    "shrink": 0.8,
    "aspect": 15,
    "label_size": 6,
}
DEFAULT_SUPTITLE_KWARGS_ROW: dict[str, Any] = {"fontsize": 9}
DEFAULT_SUPTITLE_KWARGS_GRID: dict[str, Any] = {"fontsize": 10}

#: Which nested ``*_kwargs`` dict each styling key belongs to, so an option passed one
#: level too high can be pointed at its home rather than just rejected.
_NESTED_KWARGS: dict[str, dict[str, Any]] = {
    "colorbar_kwargs": {**DEFAULT_COLORBAR_KWARGS_ROW, **DEFAULT_COLORBAR_KWARGS_GRID},
    "title_kwargs": DEFAULT_TITLE_KWARGS,
    "gridline_kwargs": DEFAULT_GRIDLINE_KWARGS,
    "tick_label_kwargs": DEFAULT_TICK_LABEL_KWARGS,
    "row_label_kwargs": DEFAULT_ROW_LABEL_KWARGS,
    "metrics_kwargs": DEFAULT_METRICS_KWARGS,
    "suptitle_kwargs": {
        **DEFAULT_SUPTITLE_KWARGS_ROW,
        **DEFAULT_SUPTITLE_KWARGS_GRID,
    },
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
    return cbar


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
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.colors as mcolors

    title_kwargs = _merged(DEFAULT_TITLE_KWARGS, title_kwargs)
    gridline_kwargs = _merged(DEFAULT_GRIDLINE_KWARGS, gridline_kwargs)
    tick_label_kwargs = _merged(DEFAULT_TICK_LABEL_KWARGS, tick_label_kwargs)
    row_label_kwargs = _merged(DEFAULT_ROW_LABEL_KWARGS, row_label_kwargs)
    metrics_kwargs = _merged(DEFAULT_METRICS_KWARGS, metrics_kwargs)

    t, r, d = aligned[test_name], aligned[reference_name], aligned["difference"]
    tl, rl = labels
    seq, div = cmaps_for(standard_name)
    if seq_norm is None:
        vmin, vmax = _limits(t, r)
        seq_norm = norm_for(standard_name, vmin, vmax)
    if div_norm is None:
        dmax = float(np.nanpercentile(np.abs(np.asarray(d)), 98)) or 1.0
        div_norm = mcolors.Normalize(vmin=-dmax, vmax=dmax)
    proj = ccrs.PlateCarree()

    panels = [
        (t, tl, seq, seq_norm),
        (r, rl, seq, seq_norm),
        (d, "difference", div, div_norm),
    ]
    ims = []
    for j, (ax, (da, lab, cmap, norm)) in enumerate(zip(axes, panels, strict=True)):
        draw = getattr(ax, "contourf" if mark == "contourf" else "pcolormesh")
        kw = {"levels": _contour_levels(norm)} if mark == "contourf" else {}
        ims.append(
            draw(da["lon"], da["lat"], da, transform=proj, cmap=cmap, norm=norm, **kw)
        )
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax.coastlines(linewidth=0.4, zorder=3)
        gl = ax.gridlines(draw_labels=True, **gridline_kwargs)
        gl.top_labels = gl.right_labels = False
        if shared_axis_labels:
            gl.left_labels = j == 0
            gl.bottom_labels = is_bottom_row
        # else: leave left/bottom labels at gridlines(draw_labels=True)'s default —
        # every panel gets its own, independent of position.
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
        ax.set_title(lab, **title_kwargs)

    if row_label:
        # Provisional x only; _clear_row_labels moves it once the layout is final and
        # the latitude labels it must clear have a measurable width. Deliberately NOT
        # set_ylabel: constrained_layout does not see cartopy's gridline labels (they
        # are free artists, not ytick labels), so it packs the axes against the ylabel
        # alone and the latitude labels land on top of it no matter how large a
        # labelpad is set. A free text artist takes no part in the layout, so it can be
        # placed after the fact without the layout shifting back underneath it.
        axes[0]._osk_row_label = axes[0].text(
            -0.18,
            0.5,
            row_label,
            transform=axes[0].transAxes,
            **row_label_kwargs,
        )
    if metrics:
        txt = "\n".join(
            f"{k}={metrics[k]:.3g}"
            for k in metric_keys
            if isinstance(metrics.get(k), int | float)
        )
        axes[2].text(
            0.02,
            0.02,
            txt,
            transform=axes[2].transAxes,
            zorder=5,
            **metrics_kwargs,
        )
    return ims, (f"[{units}]" if units else "")


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
):
    """Draw one ``test | reference | difference`` row for a gridded comparison.

    ``figsize`` defaults to a page-width row; pass your own to override.
    ``metric_keys`` picks which of ``metrics.compute()``'s values appear in the
    corner box (default ``bias``/``rmse``/``corr``) — any subset/order, e.g.
    ``metric_keys=("corr", "sigma_ratio")``.

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
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize or (PAGE_W, PAGE_W / 3.1),
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
    )
    _draw_colorbar(
        fig, ims[1], axes[:2], lab, colorbar_kwargs, DEFAULT_COLORBAR_KWARGS_ROW
    )
    _draw_colorbar(
        fig,
        ims[2],
        axes[2],
        f"difference {lab}",
        colorbar_kwargs,
        DEFAULT_COLORBAR_KWARGS_ROW,
    )

    # after the suptitle, so the margin is fitted to the layout the figure ends with
    if title:
        fig.suptitle(title, **_merged(DEFAULT_SUPTITLE_KWARGS_ROW, suptitle_kwargs))
    _fit_left_margin(fig)
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


def _row_height(comparisons, reference_name: str, n: int) -> float:
    """Row height (inches) matched to the map's own aspect ratio.

    A fixed height leaves a tall empty band above and below wide domains — and the
    colorbars, which span the whole cell, then tower over the maps. Sizing the row to
    ``lon_span / lat_span`` keeps the bars the same height as the maps beside them.
    """
    try:
        da = comparisons[0]["aligned"][reference_name]
        lon_span = float(np.ptp(np.asarray(da["lon"])))
        lat_span = float(np.ptp(np.asarray(da["lat"])))
        aspect = np.clip(lon_span / max(lat_span, 1e-6), 0.3, 4.0)
    except Exception:  # pragma: no cover - fall back to a square-ish panel
        aspect = 1.0
    panel_w = PAGE_W / 3.6  # three panels plus colorbar gutters
    return float(min(panel_w / aspect + 0.62, (PAGE_H - 0.8) / max(n, 1)))


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
    (override with ``row_height``), and the total is capped at the 11-inch page — or
    set ``figsize`` to size the whole figure yourself. ``metric_keys`` picks which of
    ``metrics.compute()``'s values appear in each row's corner box (default
    ``bias``/``rmse``/``corr``).

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

    The ``*_kwargs`` parameters each merge onto their current defaults and map onto
    one matplotlib/cartopy call — see :func:`field_row`'s docstring for the full
    list; the same names mean the same thing here, applied per row.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    n = len(comparisons)
    proj = ccrs.PlateCarree()
    row_h = row_height or _row_height(comparisons, reference_name, n)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=figsize or (PAGE_W, row_h * n),
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
        )
        _draw_colorbar(
            fig, ims[1], axes[i][:2], lab, colorbar_kwargs, DEFAULT_COLORBAR_KWARGS_GRID
        )
        _draw_colorbar(
            fig,
            ims[2],
            axes[i][2],
            f"test − reference {lab}",
            colorbar_kwargs,
            DEFAULT_COLORBAR_KWARGS_GRID,
        )

    # after the suptitle, so the margin is fitted to the layout the figure ends with
    if title:
        fig.suptitle(title, **_merged(DEFAULT_SUPTITLE_KWARGS_GRID, suptitle_kwargs))
    _fit_left_margin(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


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
