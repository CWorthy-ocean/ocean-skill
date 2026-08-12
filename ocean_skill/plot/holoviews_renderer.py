"""Interactive holoviews/bokeh renderer.

Consumes the same :class:`~ocean_skill.plot.spec.PlotSpec` as the matplotlib renderer,
so switching between static and interactive output is only a change of renderer name.
Returns holoviews objects, which display inline in a notebook and can be saved to
standalone HTML with ``holoviews.save(obj, "plot.html")``.

The two movie families are where the renderers differ most in form and least in intent:
the static one encodes the frames into an mp4, this one puts them on a slider. Stepping
is the interactive form of playing — you can hold a frame, step back, and hover a cell
for its value — with an optional play/pause scrubber for when it should just run.

Implemented: ``field_row``, ``field_grid``, ``field_facet``, ``field_movie``,
``facet_movie`` and ``target``. **Taylor is delegated back to
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

#: The animated families. They are the only ones here with a file to write and a rate to
#: play at, so ``save``/``fps`` reach them where every other family leaves both to the
#: static renderer.
_MOVIES = {"field_movie", "facet_movie"}

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
    # a movie's frame label is an Axes.text statically; here it is the slider's own
    # value and a panel title, neither of which takes matplotlib Text properties
    "frame_label_kwargs",
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


def _field_facet(
    item: dict[str, Any],
    geo=True,
    shared_axes: bool = True,
    title: str | None = None,
    ncols: int | None = None,
    shared_limits: bool = False,
    font_scale: float = 1.0,
    **_,
):
    """One interactive map per value of the facet axis: a field over time, in order.

    The interactive twin of :func:`ocean_skill.plot.matplotlib_renderer.field_facet`,
    and the same two commitments hold. Every panel shares one colour scale, because
    the panels are one quantity at different times and per-panel scaling would hide
    exactly the change the figure exists to show. Panel titles come from the facet
    coordinate through the shared :func:`~ocean_skill.plot.matplotlib_renderer.
    facet_labels`, so a consecutive-month figure says ``Jan 2012`` here too and cannot
    be mistaken for the climatology that says ``Jan``.

    The column count also comes from the shared
    :func:`~ocean_skill.plot.typography.facet_layout`, so the two renderers arrange the
    same panels the same way. Its page-fitting argument is weaker here — a bokeh layout
    grows a scrollbar rather than a smaller panel — but a plot that rearranges itself
    when you switch renderer is not the same plot drawn interactively.

    With a second facet axis (``row_dim``, a depth) the grid is fixed at levels by
    periods and each row keeps its own colour range, as it does statically. The level
    joins each panel's title rather than sitting rotated at the row's left edge, bokeh
    having no equivalent of that floating text — the same substitution
    :func:`_field_row` makes for a field grid's ``row_label``.
    """
    from ocean_skill.colormaps import is_log
    from ocean_skill.plot.matplotlib_renderer import _aspect_of, _limits, facet_labels
    from ocean_skill.plot.typography import facet_layout

    hv = _extension()
    field = item["field"]
    facet_dim = item.get("facet_dim")
    row_dim = item.get("row_dim")
    for name, value in (("facet_dim", facet_dim), ("row_dim", row_dim)):
        if value is not None and value not in field.dims:
            raise ValueError(
                f"{name} {value!r} is not a dimension of the field ({list(field.dims)})"
            )
    n = int(field.sizes[facet_dim]) if facet_dim else 1
    nrows = int(field.sizes[row_dim]) if row_dim else 1
    units = item.get("units") or ""
    standard_name = item.get("standard_name")
    seq, _div = cmaps_for(standard_name)
    log = is_log(standard_name)

    def _clim(sub):
        lo, hi = _limits(sub)
        return (max(lo, 1e-6) if log else lo, hi)

    # One scale per row when the rows are levels, matching the static renderer: depths
    # have unrelated ranges, and one scale across them flattens the shallow rows.
    per_row = row_dim is not None and not shared_limits
    clims = [
        _clim(field.isel({row_dim: r})) if per_row else _clim(field)
        for r in range(nrows)
    ]

    labels = (
        facet_labels(field[facet_dim])
        if facet_dim and facet_dim in field.coords
        else [""] * n
    )
    row_labels = (
        facet_labels(field[row_dim])
        if row_dim and row_dim in field.coords
        else [""] * nrows
    )
    if row_dim is not None:
        ncols = n
    elif ncols is None:
        ncols, _nrows = facet_layout(n, _aspect_of(field))
    ncols = max(int(ncols), 1)

    def _panel(row, col):
        if row_dim is not None:
            sub = field.isel({row_dim: row, facet_dim: col})
            # bokeh has no rotated row label, so the level joins the panel's own
            # title -- the same move _field_row makes for a field grid's row_label
            title = f"{row_labels[row]} — {labels[col]}"
        else:
            sub = field.isel({facet_dim: col}) if facet_dim else field
            title = str(labels[col])
        return _quadmesh(
            sub,
            title=title,
            cmap=seq,
            clim=clims[row],
            units=units,
            geo=geo,
            log=log,
            font_scale=font_scale,
        )

    panels = [
        _panel(row, col)
        for row in range(nrows)
        for col in range(ncols if row_dim is not None else n)
    ]
    layout = panels[0]
    for extra in panels[1:]:
        layout = layout + extra
    layout = layout.cols(ncols).opts(hv.opts.Layout(shared_axes=shared_axes))
    if title:
        layout = layout.opts(title=str(title))
    return layout


#: Name of the dimension a movie's frames vary along, i.e. what the slider is labelled.
#: A movie's frames are usually time steps, but not necessarily — the static renderer
#: calls the per-frame text a ``frame_label`` for the same reason — so the neutral name
#: is the honest one.
FRAME_DIM = "frame"


def _unique_keys(labels) -> list[str]:
    """Make slider values unique, since a ``HoloMap`` key identifies a frame.

    Two frames sharing a key do not draw twice — the second replaces the first and the
    movie quietly loses a frame. Labels are usually unique already (see
    :func:`~ocean_skill.plot.matplotlib_renderer.frame_labels`, which refines datetimes
    until they are), so this is the backstop for the cases that cannot be refined: two
    comparisons in one set carrying the same label, say.
    """
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def _frame_keys(frames) -> list[str]:
    """One slider value per frame: its ``frame_label``, else its position.

    Strings rather than the labels' own types because a bokeh slider over discrete
    values shows them verbatim, and ``"2012-01-15"`` is what a viewer wants to read.
    Positions fill in for frames carrying no label, so the widget still steps.
    """
    return _unique_keys(
        [str(f.get("frame_label") or f"frame {i + 1}") for i, f in enumerate(frames)]
    )


def _facet_movie(
    item: dict[str, Any],
    geo=True,
    title: str | None = None,
    shared_limits: bool = True,
    every: int = 1,
    fps: int = 8,
    player: bool = False,
    save=None,
    font_scale: float = 1.0,
    **_,
):
    """One source's facet axis on a slider: the interactive twin of ``facet_movie``.

    Where :func:`_field_facet` lays the axis out as panels, this steps through it in
    place — the same field, the same labels, one panel at a time. For a long axis that
    is the more useful of the two: forty panels on a page are each too small to read,
    while forty frames are full size and a drag apart.

    One colour scale for the whole movie, as statically, and for the same reason — a
    scale that moved with the slider would make a change in the ruler look like a change
    in the field.
    """
    from ocean_skill.colormaps import is_log
    from ocean_skill.plot.matplotlib_renderer import (
        _limits,
        _one_facet_axis,
        _select_frames,
        frame_labels,
    )

    hv = _extension()
    field = item["field"]
    facet_dim = _one_facet_axis(field, item.get("facet_dim"))
    indices = _select_frames(list(range(int(field.sizes[facet_dim]))), every)
    labels = (
        frame_labels(field[facet_dim])
        if facet_dim in field.coords
        else [f"frame {i + 1}" for i in range(int(field.sizes[facet_dim]))]
    )
    keys = _unique_keys([labels[i] for i in indices])

    units = item.get("units") or ""
    standard_name = item.get("standard_name")
    seq, _div = cmaps_for(standard_name)
    log = is_log(standard_name)
    scope = field if shared_limits else field.isel({facet_dim: indices[0]})
    vmin, vmax = _limits(scope)
    if log:
        vmin = max(vmin, 1e-6)

    dim = hv.Dimension(FRAME_DIM, values=keys)
    panels = {
        key: _quadmesh(
            field.isel({facet_dim: index}),
            title=key,
            cmap=seq,
            clim=(vmin, vmax),
            units=units,
            geo=geo,
            log=log,
            font_scale=font_scale,
        )
        for key, index in zip(keys, indices, strict=True)
    }
    movie = hv.HoloMap(panels, kdims=[dim])
    if title:
        movie = movie.opts(title=str(title))
    if player:
        movie = _scrubber(movie, fps=fps)
    if save:
        _save_interactive(movie, save)
    return movie


def _field_movie(
    items,
    labels=("test", "reference"),
    geo=True,
    shared_axes: bool = True,
    metric_keys=DEFAULT_METRIC_KEYS,
    title: str | None = None,
    font_scale: float = 1.0,
    shared_limits: bool = True,
    every: int = 1,
    fps: int = 8,
    player: bool = False,
    save=None,
    **_,
):
    """Put the same row on a slider: the interactive counterpart of a movie.

    Where the static renderer encodes the frames into an mp4, this returns three
    ``HoloMap``s — test, reference and difference — sharing one ``frame`` dimension, so
    a single widget steps all three panels together. Stepping *is* the interactive form
    of playing: you can hold a frame, go back one, and hover a cell for its value, none
    of which an mp4 can do. ``player=True`` adds a play/pause scrubber for when you do
    want it to run on its own, at ``fps``.

    The colour scale is fixed across frames exactly as it is statically (see
    :func:`~ocean_skill.plot.matplotlib_renderer.field_movie`), and for the same
    reason — a scale that moves as you drag the slider makes the field look like it is
    changing when only the ruler is. ``shared_limits=True`` derives it from every frame,
    ``False`` from the first.

    ``save`` writes a self-contained HTML page, the interactive analogue of the static
    renderer's mp4; anything else is left to the static renderer, which is what can
    write a video.
    """
    from ocean_skill.colormaps import is_log
    from ocean_skill.plot.matplotlib_renderer import _limits, _select_frames

    hv = _extension()
    if not items:
        raise ValueError("a movie needs at least one frame, got none")
    items = _select_frames(list(items), every)
    keys = _frame_keys(items)

    first = items[0]
    units = first.get("units") or ""
    standard_name = first.get("standard_name")
    seq, div = cmaps_for(standard_name)
    log = is_log(standard_name)

    # One clim for the whole movie, from every frame or just the first. Computed here
    # rather than per panel because _quadmesh is called once per frame per panel and
    # would otherwise re-derive a different scale for each.
    scope = items if shared_limits else items[:1]
    vmin, vmax = _limits(
        *[f["aligned"]["test"] for f in scope],
        *[f["aligned"]["reference"] for f in scope],
    )
    if log:
        vmin = max(vmin, 1e-6)
    dmax = (
        float(
            np.nanpercentile(
                np.abs(
                    np.concatenate(
                        [np.asarray(f["aligned"]["difference"]).ravel() for f in scope]
                    )
                ),
                98,
            )
        )
        or 1.0
    )

    dim = hv.Dimension(FRAME_DIM, values=keys)
    panels: list[dict[str, Any]] = [{}, {}, {}]
    for key, item in zip(keys, items, strict=True):
        aligned = item["aligned"]
        tl, rl = item.get("labels") or labels
        summary = _metrics_summary(item.get("metrics"), metric_keys)
        # the frame label goes on the test panel's title as well as on the slider: the
        # static renderer draws it in the panel, and a saved HTML page is read the same
        # way a figure is
        panels[0][key] = _quadmesh(
            aligned["test"],
            title=f"{key} — {tl}",
            cmap=seq,
            clim=(vmin, vmax),
            units=units,
            geo=geo,
            log=log,
            font_scale=font_scale,
        )
        panels[1][key] = _quadmesh(
            aligned["reference"],
            title=str(rl),
            cmap=seq,
            clim=(vmin, vmax),
            units=units,
            geo=geo,
            log=log,
            font_scale=font_scale,
        )
        panels[2][key] = _quadmesh(
            aligned["difference"],
            title=f"difference ({summary})" if summary else "difference",
            cmap=div,
            clim=(-dmax, dmax),
            units=f"test − reference {units}",
            geo=geo,
            font_scale=font_scale,
        )

    maps = [hv.HoloMap(p, kdims=[dim]) for p in panels]
    layout = (maps[0] + maps[1] + maps[2]).opts(hv.opts.Layout(shared_axes=shared_axes))
    if title:
        layout = layout.opts(title=str(title))
    if player:
        layout = _scrubber(layout, fps=fps)
    if save:
        _save_interactive(layout, save)
    return layout


def _scrubber(layout, *, fps: int):
    """Wrap ``layout`` in a panel pane whose widget plays the frames at ``fps``.

    Holoviews' own widget for a ``HoloMap`` is a slider, which steps but does not run.
    Panel's ``"scrubber"`` gives the same dimension a ``Player`` — play, pause, step —
    and its interval is where ``fps`` lands interactively, so the same argument means
    the same thing in both renderers.
    """
    import panel as pn

    pn.extension()
    pane = pn.pane.HoloViews(layout, widget_type="scrubber", widget_location="bottom")
    for widget in pane.widget_box:
        if hasattr(widget, "interval"):
            widget.interval = int(1000 / max(fps, 1))
    return pane


def _save_interactive(obj, save) -> None:
    """Write ``obj`` to a standalone HTML page, refusing formats bokeh cannot write."""
    from pathlib import Path

    path = Path(save).expanduser()
    if path.suffix.lower() not in (".html", ".htm"):
        raise ValueError(
            f"the interactive renderer writes HTML, not {path.suffix or 'that'}: it "
            f"has no video encoder. Either save={str(path.with_suffix('.html'))!r} "
            f"here, or pass renderer='matplotlib' to write {path.name}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "save"):  # a panel pane (player=True) embeds its own widget state
        obj.save(str(path), embed=True)
    else:
        import holoviews as hv

        hv.save(obj, str(path))
    print(f"ocean-skill: interactive movie written to {path}")


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
    drops = [
        "figsize",
        # bokeh attaches a colorbar to the plot frame, so it already starts and ends
        # level with the panel — align_colorbars is satisfied here by construction
        # rather than ignored, hence a silent drop and not _STATIC_ONLY_KWARGS.
        "align_colorbars",
        "row_height",
        "domain",
        "mark",
        "metrics",
        # bokeh labels every panel's own axes and has no notion of borrowing a
        # neighbour's, so there is nothing for this to switch off
        "shared_axis_labels",
        *_STATIC_ONLY_KWARGS,
    ]
    if family not in _MOVIES:
        # a movie is the only family with something to write here (a standalone HTML
        # page, the interactive counterpart of an mp4) and the only one that plays at a
        # rate; everywhere else both are the static renderer's business
        drops += ["save", "fps", "dpi", "progress"]
    for drop in drops:
        opts.pop(drop, None)

    if family == "field_row":
        return _field_row(spec.single, **opts)
    if family == "field_grid":
        return _field_grid(spec.items, **opts)
    if family == "field_facet":
        return _field_facet(spec.single, **opts)
    if family == "field_movie":
        return _field_movie(spec.items, **opts)
    if family == "facet_movie":
        return _facet_movie(spec.single, **opts)
    if family == "target":
        return _target(spec.items, **opts)
    raise NotImplementedError(f"holoviews renderer: family {family!r} not implemented")


register_renderer("holoviews", render)
