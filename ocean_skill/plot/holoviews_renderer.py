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
``facet_movie``, ``skill_map`` and ``target``. **Taylor is delegated back to
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
from ocean_skill.plot.matplotlib_renderer import (
    DEFAULT_METRIC_KEYS,
    metric_value_text,
)
from ocean_skill.plot.registry import register_renderer
from ocean_skill.plot.typography import (
    PAGE_W,
    bokeh_fontsize,
    bokeh_scale,
    frame_px,
    resolve_canvas,
)

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


#: Width (CSS pixels) of one interactive map panel on the default canvas; the height
#: follows the data's own aspect ratio, as the static renderer's panels do. Sized for a
#: *row* of three panels side by side, which is what most families draw — see
#: :data:`SOLO_PANEL_WIDTH_PX` for the one that draws a single panel. ``size``/``zoom``
#: scale whichever of the two applies, via :func:`_canvas_factor`.
PANEL_WIDTH_PX = 260

#: Width for a family that draws **one** panel, i.e. a single-field movie. It has the
#: whole width to itself rather than a third of it, so inheriting the row's 260 wastes
#: most of the page and makes the one thing on it the smallest thing on it. Chosen to
#: leave room beside it for the colorbar and the frame widget in a normal notebook.
SOLO_PANEL_WIDTH_PX = 680

#: The Target diagram's frame, square because its guide rings must stay circular.
TARGET_FRAME_PX = (400, 400)


def _canvas_factor(size=None, zoom: float = 1.0) -> float:
    """How much wider than the default canvas ``size``/``zoom`` ask for.

    ``size``/``zoom`` are stated in inches statically, which bokeh has no notion of, so
    they arrive here as a ratio against the page and scale the frame in pixels instead.
    Doing it as a ratio rather than converting inches to pixels is deliberate: the two
    renderers agree about *relative* size (see
    :func:`~ocean_skill.plot.typography.bokeh_fontsize`), and a browser's pixel is not a
    physical length to convert to anyway.
    """
    return resolve_canvas(size, zoom).width / PAGE_W


def _panel_geometry(
    da,
    *,
    font_scale: float = 1.0,
    width_px: float = PANEL_WIDTH_PX,
    canvas_factor: float = 1.0,
):
    """``(frame_width, frame_height, fontsize)`` for one interactive map of ``da``.

    A fixed 260x200 frame letterboxed exactly the domains the static renderer fits, and
    fixed bokeh font sizes were a second set of numbers to keep in step with the static
    ones by hand. Both now come from :mod:`ocean_skill.plot.typography`, off the same
    aspect ratio and the same type scale, so the interactive plot is the static plot
    drawn interactively rather than a near-miss of it.

    ``width_px`` is the panel's share of the page — a third of it for a row of three,
    all of it for a lone panel (:data:`SOLO_PANEL_WIDTH_PX`). The type scale follows it,
    so a bigger frame gets proportionally bigger labels without being told.
    """
    try:
        aspect = float(np.ptp(np.asarray(da["lon"]))) / max(
            float(np.ptp(np.asarray(da["lat"]))), 1e-6
        )
    except Exception:  # pragma: no cover - unlabelled coords; fall back to square
        aspect = 1.0
    # the two compose: width_px is this family's share of the page, canvas_factor is
    # how much bigger a canvas the caller asked for than the default one
    px = frame_px(aspect, width_px=width_px * canvas_factor)
    return px[0], px[1], bokeh_fontsize(px, font_scale=font_scale)


def _quadmesh(
    da,
    *,
    title,
    cmap,
    clim,
    units,
    geo=True,
    log=False,
    font_scale: float = 1.0,
    canvas_factor: float = 1.0,
    width_px: float = PANEL_WIDTH_PX,
    axis_labels: tuple[str, str] | None = None,
):
    """One interactive map panel with hover readout.

    ``axis_labels`` overrides the axis titles, which otherwise come from the
    coordinates' own ``long_name``. ROMS spells those "longitude of rho-points (degrees
    East)": accurate, twice the width of the numbers it labels, and truncated by bokeh
    anyway — the static renderer never shows it because cartopy draws gridline labels
    instead.
    """
    import hvplot.xarray  # noqa: F401  (registers the .hvplot accessor)

    frame_w, frame_h, fontsize = _panel_geometry(
        da, font_scale=font_scale, width_px=width_px, canvas_factor=canvas_factor
    )
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
    if axis_labels is not None:
        opts["xlabel"], opts["ylabel"] = axis_labels
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
    survives pan/zoom/resize as cleanly as a title does. Formatted through the same
    shared helper, so a metric reads identically in either renderer.
    """
    if not metrics:
        return ""
    return ", ".join(
        f"{key}={text}"
        for key in metric_keys
        if (text := metric_value_text(metrics, key))
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
    size=None,
    zoom: float = 1.0,
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
    factor = _canvas_factor(size, zoom)
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
            canvas_factor=factor,
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
            canvas_factor=factor,
        ),
        _quadmesh(
            d,
            title=diff_title,
            cmap=div,
            clim=(-dmax, dmax),
            units=f"test − reference {units}",
            geo=geo,
            font_scale=font_scale,
            canvas_factor=factor,
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
    size=None,
    zoom: float = 1.0,
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
            size=size,
            zoom=zoom,
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


def _skill_map(
    items,
    geo=True,
    shared_axes: bool = True,
    title: str | None = None,
    ncols: int | None = None,
    metric_names=None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    **_,
):
    """One interactive map per skill metric: the interactive twin of ``skill_map``.

    Every commitment the static family makes holds here, because both come from the same
    two shared functions. Which metrics may be drawn, and the message when one was never
    computed, come from :func:`~ocean_skill.plot.matplotlib_renderer.metric_panels`; the
    colours and limits come from :func:`ocean_skill.colormaps.metric_colors`, so a bias
    panel is symmetric about zero and a correlation panel spans (−1, 1) in either
    backend. The column count comes from the shared
    :func:`~ocean_skill.plot.typography.facet_layout`, so the two renderers arrange the
    same panels the same way.

    Each metric's **overall** value — reduced over space and the scored axis together —
    joins its panel's title, which is the same substitution :func:`_field_row` makes
    for a comparison's corner box: bokeh has no free-floating annotation that survives
    pan/zoom as cleanly as a title. Units stay on each panel's own colorbar, as they do
    statically.
    """
    from ocean_skill.colormaps import metric_colors
    from ocean_skill.plot.matplotlib_renderer import (
        _aspect_of,
        metric_arrays,
        metric_panel_titles,
        metric_panels,
    )
    from ocean_skill.plot.typography import facet_layout

    hv = _extension()
    items = list(items)
    if not items:
        raise ValueError("skill_map needs at least one comparison, got none")
    names = metric_panels(items[0]["skill"], metric_names)
    if not names:
        raise ValueError(
            "this comparison carries no 2-D metric maps to draw. It was probably not "
            'scored over an axis: build it with compare(..., over="time").'
        )
    for item in items[1:]:
        metric_panels(item["skill"], names)

    titles = metric_panel_titles(names)
    stacked = len(items) > 1
    if stacked:
        ncols = len(names)
    elif ncols is None:
        ncols, _nrows = facet_layout(
            len(names), _aspect_of(items[0]["skill"][names[0]])
        )
    ncols = max(int(ncols), 1)
    factor = _canvas_factor(size, zoom)
    arrays = {i: metric_arrays(item["skill"], names) for i, item in enumerate(items)}

    panels = []
    for row, item in enumerate(items):
        for name in names:
            colors = metric_colors(
                name, arrays[row][name], standard_name=item.get("standard_name")
            )
            base = titles[names.index(name)]
            # bokeh has no rotated row label, so the comparison joins the panel's own
            # title -- the same move _field_row makes for a field grid's row_label
            if stacked and item.get("row_label"):
                base = f"{item['row_label']} — {base}"
            value = metric_value_text(item.get("metrics"), name)
            panels.append(
                _quadmesh(
                    item["skill"][name],
                    title=f"{base} ({value})" if value else base,
                    cmap=colors.cmap,
                    clim=colors.clim(),
                    units=str(item["skill"][name].attrs.get("units", "") or ""),
                    geo=geo,
                    log=colors.log,
                    font_scale=font_scale,
                    canvas_factor=factor,
                )
            )

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
    widget: str = "slider",
    save=None,
    font_scale: float = 1.0,
    width_px: float = SOLO_PANEL_WIDTH_PX,
    axis_labels: tuple[str, str] | None = ("longitude", "latitude"),
    **_,
):
    """One source's facet axis on a slider: the interactive twin of ``facet_movie``.

    Where :func:`_field_facet` lays the axis out as panels, this steps through it in
    place — the same field, the same labels, one panel at a time. For a long axis that
    is the more useful of the two: forty panels on a page are each too small to read,
    while forty frames are full size and a drag apart.

    Being the only panel on the page, it gets the page: the frame is
    :data:`SOLO_PANEL_WIDTH_PX` rather than the third-of-a-row every other family draws
    at (override with ``width_px``). Axis titles are shortened for the same reason — a
    ROMS coordinate's own ``long_name`` is "longitude of rho-points (degrees East)",
    which bokeh truncates anyway; pass ``axis_labels=None`` to keep whatever the
    coordinates say.

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
            width_px=width_px,
            axis_labels=axis_labels,
        )
        for key, index in zip(keys, indices, strict=True)
    }
    movie = hv.HoloMap(panels, kdims=[dim])
    if title:
        movie = movie.opts(title=str(title))
    movie = _with_widget(movie, widget=widget, fps=fps, frame_dim=FRAME_DIM)
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
    widget: str = "slider",
    save=None,
    **_,
):
    """Put the same row on a slider: the interactive counterpart of a movie.

    Where the static renderer encodes the frames into an mp4, this returns three
    ``HoloMap``s — test, reference and difference — sharing one ``frame`` dimension, so
    a single widget steps all three panels together. Stepping *is* the interactive form
    of playing: you can hold a frame, go back one, and hover a cell for its value, none
    of which an mp4 can do. ``widget="player"`` swaps the slider for a play/pause
    scrubber, at ``fps``, for when it should just run; ``"dropdown"`` gives holoviews'
    own default control.

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
    layout = _with_widget(layout, widget=widget, fps=fps, frame_dim=FRAME_DIM)
    if save:
        _save_interactive(layout, save)
    return layout


def _with_widget(obj, *, widget: str, fps: int, frame_dim: str | None = None):
    """Wrap ``obj`` so its frame dimension gets the widget the caller asked for.

    Holoviews picks the widget itself, and for a dimension whose values are strings —
    a date, a depth — it picks a *dropdown*. That is the wrong control for an ordered
    sequence: stepping to the next frame is two clicks and a search rather than one
    nudge, and dragging through the movie is impossible. Panel can be told otherwise.

    ``"slider"`` (the default) is a ``DiscreteSlider``: drag it, or arrow-key through
    the frames, with the label still reading "2010-01-29" rather than an index.
    ``"player"`` is a ``Player`` — play, pause, step — running at ``fps``.
    ``"dropdown"`` returns the holoviews object untouched, as every other family does.
    """
    if widget == "dropdown":
        return obj
    if widget not in ("slider", "player"):
        raise ValueError(
            f"unknown widget {widget!r}; expected 'slider', 'player' or 'dropdown'"
        )
    import panel as pn

    pn.extension()
    if widget == "player":
        pane = pn.pane.HoloViews(obj, widget_type="scrubber", widget_location="bottom")
    else:
        pane = pn.pane.HoloViews(
            obj,
            widget_location="bottom",
            widgets={frame_dim: pn.widgets.DiscreteSlider} if frame_dim else {},
        )
    for control in pane.widget_box:
        # a Player runs at the same rate the static renderer encodes at, so `fps` means
        # the same thing in both renderers
        if hasattr(control, "interval"):
            control.interval = int(1000 / max(fps, 1))
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
    if hasattr(obj, "save"):  # a panel pane embeds its own widget state
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
    size=None,
    zoom: float = 1.0,
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
    factor = _canvas_factor(size, zoom)
    # not `frame`: that name is taken below for a per-group slice of the DataFrame
    frame_size = (TARGET_FRAME_PX[0] * factor, TARGET_FRAME_PX[1] * factor)
    sizes = bokeh_scale(frame_size, font_scale=font_scale)
    fontsize = bokeh_fontsize(frame_size, font_scale=font_scale)
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
        frame_width=round(frame_size[0]),
        frame_height=round(frame_size[1]),
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
        # bokeh lays out its own text and never clips a long label the way a fixed
        # matplotlib axes does, so the shrink-to-fit pass has nothing to do here.
        "fit_text",
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
    if family == "skill_map":
        return _skill_map(spec.items, **opts)
    if family == "target":
        return _target(spec.items, **opts)
    raise NotImplementedError(f"holoviews renderer: family {family!r} not implemented")


register_renderer("holoviews", render)
