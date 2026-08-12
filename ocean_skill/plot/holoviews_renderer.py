"""Interactive holoviews/bokeh renderer.

Consumes the same :class:`~ocean_skill.plot.spec.PlotSpec` as the matplotlib renderer,
so switching between static and interactive output is only a change of renderer name.
Returns holoviews objects, which display inline in a notebook and can be saved to
standalone HTML with ``holoviews.save(obj, "plot.html")``.

Implemented: ``field_row``, ``field_grid`` and ``target``. **Taylor is delegated back to
matplotlib**: it is drawn on a floating polar axis (``mpl_toolkits.axisartist``) that
bokeh has no equivalent for, and rebuilding the curved correlation axis from primitives
is a project in itself rather than a port. ``paired`` therefore returns a static Taylor
next to an interactive Target only if asked for separately — as a family it also
delegates.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ocean_skill.colormaps import cmaps_for
from ocean_skill.plot.matplotlib_renderer import DEFAULT_METRIC_KEYS
from ocean_skill.plot.registry import register_renderer
from ocean_skill.plot.typography import bokeh_fontsize, bokeh_scale, frame_px

__all__ = ["render"]

#: Families bokeh cannot reasonably draw; these fall back to matplotlib with a warning.
_DELEGATED = {"taylor", "paired"}

#: Style parameters that only mean something to the static renderer (see
#: docs/plot_styling_reference.md) — each maps onto a matplotlib/cartopy call bokeh
#: has no equivalent for. Passing one here has no effect; render() warns once rather
#: than silently absorbing them with no signal at all. ``title`` and ``metric_keys``
#: are NOT here -- both renderers honor them the same way (see _field_row/_field_grid).
_STATIC_ONLY_KWARGS = {
    "title_kwargs",
    "colorbar_kwargs",
    "gridline_kwargs",
    "tick_label_kwargs",
    "row_label_kwargs",
    "metrics_kwargs",
    "suptitle_kwargs",
}


def _extension():
    """Activate the bokeh backend once, quietly."""
    import holoviews as hv

    if not hv.Store.renderers.get("bokeh"):
        hv.extension("bokeh", logo=False)
    return hv


#: Width (CSS pixels) of one interactive map panel; the height follows the data's own
#: aspect ratio, as the static renderer's panels do.
PANEL_WIDTH_PX = 260

#: The Target diagram's frame, square because its guide rings must stay circular.
TARGET_FRAME_PX = (400, 400)


def _panel_geometry(da, *, font_scale: float = 1.0):
    """``(frame_width, frame_height, fontsize)`` for one interactive map of ``da``.

    A fixed 260x200 frame letterboxed exactly the domains the static renderer fits, and
    fixed bokeh font sizes were a second set of numbers to keep in step with the static
    ones by hand. Both now come from :mod:`ocean_skill.plot.typography`, off the same
    aspect ratio and the same type scale, so the interactive plot is the static plot
    drawn interactively rather than a near-miss of it.
    """
    try:
        aspect = float(np.ptp(np.asarray(da["lon"]))) / max(
            float(np.ptp(np.asarray(da["lat"]))), 1e-6
        )
    except Exception:  # pragma: no cover - unlabelled coords; fall back to square
        aspect = 1.0
    px = frame_px(aspect, width_px=PANEL_WIDTH_PX)
    return px[0], px[1], bokeh_fontsize(px, font_scale=font_scale)


def _quadmesh(
    da, *, title, cmap, clim, units, geo=True, log=False, font_scale: float = 1.0
):
    """One interactive map panel with hover readout."""
    import hvplot.xarray  # noqa: F401  (registers the .hvplot accessor)

    frame_w, frame_h, fontsize = _panel_geometry(da, font_scale=font_scale)
    opts = {
        "x": "lon",
        "y": "lat",
        "cmap": cmap,
        "clim": clim,
        "title": title,
        "colorbar": True,
        "clabel": units or "",
        "hover": True,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "fontsize": fontsize,
        "rasterize": False,
        "logz": log,
    }
    if geo:
        opts |= {"geo": True, "coastline": "50m", "projection": None}
    try:
        return da.hvplot.quadmesh(**opts)
    except Exception:  # pragma: no cover - geoviews/cartopy unavailable
        opts.pop("geo", None)
        opts.pop("coastline", None)
        opts.pop("projection", None)
        return da.hvplot.quadmesh(**opts)


def _metrics_summary(metrics: dict[str, Any] | None, metric_keys) -> str:
    """Build a short ``"bias=.., rmse=.., corr=.."`` string for a panel title.

    Same numbers as the static renderer's corner box (see
    ``matplotlib_renderer._draw_row``), folded into the difference panel's title
    instead — bokeh has no equivalent of a free-floating text-box annotation that
    survives pan/zoom/resize as cleanly as a title does.
    """
    if not metrics:
        return ""
    return ", ".join(
        f"{k}={metrics[k]:.3g}"
        for k in metric_keys
        if isinstance(metrics.get(k), int | float)
    )


def _field_row(
    item: dict[str, Any],
    labels=("test", "reference"),
    geo=True,
    shared_axes: bool = True,
    metric_keys=DEFAULT_METRIC_KEYS,
    title: str | None = None,
    row_label: str | None = None,
    font_scale: float = 1.0,
    **_,
):
    """Test | reference | difference as three linked interactive maps.

    ``shared_axes=True`` (the default) links pan/zoom across all three panels —
    zooming one zooms the others, since they share the same underlying bokeh
    ``Range``. ``title``, if given, becomes the row's overall title, same kwarg as
    the static renderer's ``title=``. Metrics (bias/rmse/corr by default; see
    ``metric_keys``) are folded into the difference panel's own title, the closest
    interactive equivalent of the static renderer's corner box.

    ``row_label`` (the variable name the static renderer draws rotated at the row's
    left edge — bokeh has no equivalent of that floating text) is prefixed onto the
    first panel's title instead, which puts it in the same place visually.

    ``font_scale`` means what it means statically — every text size multiplied, their
    proportions kept — and reaches bokeh through the same type scale (see
    :func:`~ocean_skill.plot.typography.bokeh_fontsize`). The seven ``*_kwargs`` dicts
    remain matplotlib-only, since each names a matplotlib call; ``font_scale`` names a
    size, which bokeh does have.
    """
    from ocean_skill.colormaps import is_log
    from ocean_skill.plot.matplotlib_renderer import _limits

    hv = _extension()
    aligned = item["aligned"]
    t, r, d = aligned["test"], aligned["reference"], aligned["difference"]
    units = item.get("units") or ""
    standard_name = item.get("standard_name")
    seq, div = cmaps_for(standard_name)
    log = is_log(standard_name)
    vmin, vmax = _limits(t, r)
    if log:
        vmin = max(vmin, 1e-6)
    dmax = float(np.nanpercentile(np.abs(np.asarray(d)), 98)) or 1.0
    tl, rl = labels

    diff_title = "difference"
    summary = _metrics_summary(item.get("metrics"), metric_keys)
    if summary:
        diff_title = f"difference ({summary})"
    test_title = f"{row_label} — {tl}" if row_label else str(tl)

    panels = [
        _quadmesh(
            t,
            title=test_title,
            cmap=seq,
            clim=(vmin, vmax),
            units=units,
            geo=geo,
            log=log,
            font_scale=font_scale,
        ),
        _quadmesh(
            r,
            title=str(rl),
            cmap=seq,
            clim=(vmin, vmax),
            units=units,
            geo=geo,
            log=log,
            font_scale=font_scale,
        ),
        _quadmesh(
            d,
            title=diff_title,
            cmap=div,
            clim=(-dmax, dmax),
            units=f"test − reference {units}",
            geo=geo,
            font_scale=font_scale,
        ),
    ]
    row = panels[0] + panels[1] + panels[2]
    row = row.opts(hv.opts.Layout(shared_axes=shared_axes))
    if title:
        row = row.opts(title=str(title))
    return row


def _field_grid(
    items,
    labels=("test", "reference"),
    geo=True,
    shared_axes: bool = True,
    metric_keys=DEFAULT_METRIC_KEYS,
    title: str | None = None,
    font_scale: float = 1.0,
    **_,
):
    """One interactive row per comparison, stacked.

    ``shared_axes=True`` (the default) links pan/zoom across *every* panel in the
    grid, not just within a row — meaningful whenever every row shares the same
    geographic domain (the common case: one `compare()` fan-out, one model
    domain), same as `field_grid`'s own `shared_limits` is meaningful only when
    rows share a variable. Set `False` if rows genuinely cover different regions.
    ``title`` sets one overall title above the whole grid, same kwarg as the static
    renderer's ``title=`` (its per-panel ``suptitle_kwargs`` styling has no bokeh
    equivalent, so only the text carries over, not the font/size).

    Each row is titled from *its own* ``labels``, exactly as the static renderer
    does, falling back to the top-level ``labels`` only for a row that carries
    none. Rows in one ``compare()`` fan-out commonly come from *different*
    reference sources (nitrate from one WOA entry, phosphate from another), so
    reusing the first row's pair for every row would mislabel all but the first.

    Type sizes do **not** shrink with the row count the way the static renderer's do:
    a bokeh ``frame_width`` is fixed rather than a share of a page, so stacking more
    rows makes the page longer instead of making each panel smaller — the thing the
    static scale is compensating for does not happen here. ``font_scale`` still applies.
    """
    hv = _extension()
    rows = [
        _field_row(
            it,
            labels=it.get("labels") or labels,
            geo=geo,
            shared_axes=shared_axes,
            metric_keys=metric_keys,
            row_label=it.get("row_label"),
            font_scale=font_scale,
        )
        for it in items
    ]
    layout = rows[0]
    for extra in rows[1:]:
        layout = layout + extra
    layout = layout.cols(3).opts(hv.opts.Layout(shared_axes=shared_axes))
    if title:
        layout = layout.opts(title=str(title))
    return layout


#: tab10 in matplotlib's own order — the palette ``summary._group_styles`` assigns by
#: level index, so pinning the same hexes here keeps colours identical across renderers.
_TAB10 = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

#: matplotlib marker → bokeh marker, in the same order as ``summary._MARKERS``, so a
#: diagram keeps its shapes when the same call is rendered interactively.
_BOKEH_MARKERS = (
    "circle",
    "triangle",
    "star",
    "square",
    "diamond",
    "inverted_triangle",
    "plus",
    "x",
)


def _target(
    items,
    title=None,
    circles=(0.5, 1.0),
    labels="annotate",
    color_by=None,
    marker_by=None,
    font_scale: float = 1.0,
    **_,
):
    """Interactive Target diagram: hover a point for its full metric record.

    ``labels``, ``color_by`` and ``marker_by`` mean exactly what they do in
    :mod:`ocean_skill.plot.summary`, so one call renders the same way in either
    renderer — including the default (``"annotate"``), which matches the static target.
    ``font_scale`` likewise: text is sized from the frame by the shared type scale, so
    the point labels here and on the static target are the same size relative to the
    diagram.
    """
    import pandas as pd

    from ocean_skill.plot.summary import _resolve_labels, pretty_level

    hv = _extension()
    sizes = bokeh_scale(TARGET_FRAME_PX, font_scale=font_scale)
    fontsize = bokeh_fontsize(TARGET_FRAME_PX, font_scale=font_scale)
    labels_mode = _resolve_labels(labels)
    recs = [dict(i.get("metrics", {}), label=i.get("label") or "") for i in items]
    df = pd.DataFrame(recs)
    sref = df["std_reference"].to_numpy()
    df["x"] = df["crmsd"].to_numpy() / sref * np.sign(df["std_test"].to_numpy() - sref)
    df["y"] = df["bias"].to_numpy() / sref

    # Mirrors summary._group_styles: colour follows marker groups when no colour
    # dimension was named, so the legend has one entry per group rather than a
    # duplicate entry per point.
    if color_by in df.columns:
        color_dim = color_by
    elif marker_by in df.columns:
        color_dim = marker_by
    else:
        color_dim = "label"
    cols = [
        c
        for c in ("label", "bias", "rmse", "corr", "sigma_ratio", "n", color_dim)
        if c in df.columns
    ]
    cols = list(dict.fromkeys(cols))

    # One element per group with a *fixed* colour, rather than one element coloured by a
    # field. Bokeh cannot build a legend from a colour field and overlay labels at once
    # (E-1006: non-matching data sources), which is exactly what color_by + marker_by
    # asks for. Explicit groups also pin the colours to tab10 by level index, so a
    # diagram keeps its colours when the same call is rendered statically.
    color_levels = list(dict.fromkeys(df[color_dim]))
    grouped_by_marker = marker_by in df.columns
    marker_levels = list(dict.fromkeys(df[marker_by])) if grouped_by_marker else []

    def _label(color_level, marker_level):
        # pretty_level, not str(): the static legend spells levels through it, and the
        # two renderers must not disagree about what the same group is called.
        if marker_levels and color_by:
            return (
                f"{pretty_level(color_dim, color_level)} · "
                f"{pretty_level(marker_by, marker_level)}"
            )
        if marker_levels:
            return pretty_level(marker_by, marker_level)
        return pretty_level(color_dim, color_level)

    elements = []
    for mi, marker_level in enumerate(marker_levels or [None]):
        for ci, color_level in enumerate(color_levels):
            frame = df[df[color_dim] == color_level]
            if marker_levels:
                frame = frame[frame[marker_by] == marker_level]
            if frame.empty:
                continue
            elements.append(
                hv.Points(
                    frame,
                    kdims=["x", "y"],
                    vdims=cols,
                    label=_label(color_level, marker_level),
                ).opts(
                    size=11,
                    color=_TAB10[ci % len(_TAB10)],
                    marker=_BOKEH_MARKERS[mi % len(_BOKEH_MARKERS)],
                    tools=["hover"],
                    # Below, matching the static diagrams: target points scatter around
                    # the origin, so a legend inside the frame collides with the data.
                    legend_position="bottom",
                    show_legend=labels_mode == "legend",
                    line_color="white",
                    line_width=1,
                )
            )
    points = hv.Overlay(elements)

    if labels_mode == "annotate":
        points = points * hv.Labels(df, kdims=["x", "y"], vdims="label").opts(
            # bokeh's fontsize dict has no slot for a Labels element, so this one role
            # is taken from the scale directly rather than through _BOKEH_KEYS
            text_font_size=sizes["annotation"],
            yoffset=0.06,
            text_color="black",
        )
    lim = max(
        1.15 * float(np.max(np.hypot(df["x"], df["y"]))), max(circles) * 1.25, 1.2
    )

    theta = np.linspace(0, 2 * np.pi, 181)
    guides = hv.Overlay(
        [
            hv.Path([np.column_stack([rad * np.cos(theta), rad * np.sin(theta)])]).opts(
                color="grey",
                line_dash="dashed" if rad == 1.0 else "dotted",
                line_width=1,
            )
            for rad in circles
        ]
        + [
            hv.HLine(0).opts(color="lightgrey", line_width=1),
            hv.VLine(0).opts(color="lightgrey", line_width=1),
            # the reference sits at the origin, as in the static version
            hv.Scatter([(0.0, 0.0)]).opts(marker="star", size=16, color="black"),
        ]
    )
    return (guides * points).opts(
        # equal frame dims + data_aspect keeps the guide circles circular; fixed
        # width/height would fight the aspect and squash them into ellipses
        frame_width=TARGET_FRAME_PX[0],
        frame_height=TARGET_FRAME_PX[1],
        data_aspect=1,
        fontsize=fontsize,
        xlabel="signed centred RMSD / σ_ref  (← under | over →)",
        ylabel="bias / σ_ref",
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        title=title or "Target diagram",
    )


def render(spec, **kwargs: Any):
    """Draw a :class:`PlotSpec` interactively; delegate families bokeh cannot do."""
    opts = {**spec.options, **kwargs}
    family = spec.family

    if family in _DELEGATED:
        warnings.warn(
            f"{family!r} has no interactive form (its floating polar axis is "
            "matplotlib-specific); rendering it statically instead.",
            stacklevel=2,
        )
        from ocean_skill.plot.matplotlib_renderer import render as mpl_render

        return mpl_render(spec, **kwargs)

    ignored = _STATIC_ONLY_KWARGS & opts.keys()
    if ignored:
        warnings.warn(
            f"{sorted(ignored)} only affect the static (matplotlib) renderer and "
            "have no effect here — pass renderer='matplotlib' for them to apply.",
            stacklevel=2,
        )

    # The family functions take **_, so an option one level too high (label_size, which
    # belongs inside colorbar_kwargs) is absorbed in silence here where the static
    # renderer raises. Say so, or the same typo is a loud error in one renderer and no
    # error at all in the other.
    from ocean_skill.plot.matplotlib_renderer import _nested_owner

    for key in sorted(opts.keys() - _STATIC_ONLY_KWARGS):
        owner = _nested_owner(key)
        if owner and owner not in opts:
            warnings.warn(
                f"{key!r} is not a top-level plot option — it belongs inside "
                f"{owner}, e.g. {owner}={{{key!r}: ...}}. Ignoring it.",
                stacklevel=2,
            )

    # options the static renderer understands but bokeh has no use for
    for drop in (
        "figsize",
        "save",
        # bokeh attaches a colorbar to the plot frame, so it already starts and ends
        # level with the panel — align_colorbars is satisfied here by construction
        # rather than ignored, hence a silent drop and not _STATIC_ONLY_KWARGS.
        "align_colorbars",
        "row_height",
        "domain",
        "mark",
        "metrics",
        *_STATIC_ONLY_KWARGS,
    ):
        opts.pop(drop, None)

    if family == "field_row":
        return _field_row(spec.single, **opts)
    if family == "field_grid":
        return _field_grid(spec.items, **opts)
    if family == "target":
        return _target(spec.items, **opts)
    raise NotImplementedError(f"holoviews renderer: family {family!r} not implemented")


register_renderer("holoviews", render)
