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
from itertools import pairwise
from typing import Any

import numpy as np

from ocean_skill.align import natural_convention
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
    diagram_scale_factor,
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
    # the series family's line and legend styling: both are matplotlib call signatures
    # (Axes.plot, Axes.legend), and bokeh's equivalents take neither. The *policy* they
    # would tweak -- which line is solid, which colour, where the key goes -- is honored
    # here; only the raw keyword pass-through is not.
    "legend_kwargs",
    "line_kwargs",
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
    aspect: float | None = None,
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

    ``aspect``, given, is used outright instead of measured off ``da["lon"]``/
    ``da["lat"]`` — the escape hatch a non-geographic panel needs (see
    :func:`_quadmesh`'s own note), since it has no lon/lat spans to measure at all.
    """
    if aspect is None:
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


def _output_projection(da):
    """Return the map frame for ``da``: centred on 180 when it straddles the dateline.

    A lane straddling the antimeridian (a Pacific model) splits at the edges of the
    default centre-0 frame, leaving the basin torn across both edges and a blank
    Atlantic in the middle; centring on 180 keeps it one piece. ``None`` (hvplot's own
    default frame) everywhere else.

    Straddling is decided by :func:`~ocean_skill.align.natural_convention`, the same
    span-based test ``align`` and ``_tiles_for`` use — not by ``lon.max() > 180``, which
    only sees a straddle when the coordinates happen to be *stored* in 0-360. A pair
    aligned onto a ±180 reference grid keeps a straddling Pacific domain's longitudes in
    ±180 (``lon.max() <= 180``) while it still straddles, and the raw-value test missed
    exactly that case — the one this frame exists for.
    """
    if "lon" not in getattr(da, "coords", ()):
        return None
    if natural_convention(da) == "0-360":
        import cartopy.crs as ccrs

        return ccrs.PlateCarree(central_longitude=180.0)
    return None


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
    hover: bool = True,
    rasterize: bool = False,
    tiles: str | bool | None = None,
    coastline: bool = True,
    project: bool = False,
    x: str = "lon",
    y: str = "lat",
    aspect: float | None = None,
    invert_y: bool = False,
    bgcolor: str | None = None,
):
    """One interactive map panel with hover readout.

    ``axis_labels`` overrides the axis titles, which otherwise come from the
    coordinates' own ``long_name``. ROMS spells those "longitude of rho-points (degrees
    East)": accurate, twice the width of the numbers it labels, and truncated by bokeh
    anyway — the static renderer never shows it because cartopy draws gridline labels
    instead.

    ``x``/``y`` name the coordinates to draw against, ``"lon"``/``"lat"`` by default;
    a non-geographic panel (a vertical section — see
    :func:`ocean_skill.plot.section.prepare_section`) passes its own pair with
    ``geo=False``, skipping every geographic option below. ``aspect``, given,
    overrides :func:`_panel_geometry`'s lon/lat-span measurement outright — a
    section's axes are kilometres and metres, spans with no ratio to *read*, so its
    shape is a design constant instead (:data:`~ocean_skill.plot.typography.
    SECTION_ASPECT`), the same reasoning that gives the static renderer's
    ``SERIES_ASPECT`` no data to measure either. ``invert_y``/``bgcolor`` are the
    other two a non-geographic panel needs and a geographic one never does: a
    section's y-axis reads shallow-at-top, and its off-domain/below-bathymetry
    cells want the same grey a map draws for land.
    """
    import hvplot.xarray  # noqa: F401  (registers the .hvplot accessor)

    frame_w, frame_h, fontsize = _panel_geometry(
        da,
        font_scale=font_scale,
        width_px=width_px,
        canvas_factor=canvas_factor,
        aspect=aspect,
    )
    opts = {
        "x": x,
        "y": y,
        "cmap": cmap,
        "clim": clim,
        "title": title,
        "colorbar": True,
        "clabel": units or "",
        "hover": hover,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "fontsize": fontsize,
        "rasterize": rasterize,
        "logz": log,
    }
    if bgcolor is not None:
        opts["bgcolor"] = bgcolor
    if rasterize:
        # hvplot returns rasterize=True as a *lazy* DynamicMap so that zooming
        # re-aggregates at the new extent. Nothing downstream can evaluate that lazily:
        # a movie's HoloMap has to hold concrete frames (see _frame_map), and a still
        # figure has to survive being saved and reopened with no kernel behind it — so
        # the aggregation is applied eagerly here for both. The cost is that zooming
        # magnifies the rasterized image rather than re-aggregating the mesh underneath
        # it — pass rasterize=False for a field small enough to zoom into properly.
        opts["dynamic"] = False
    if axis_labels is not None:
        opts["xlabel"], opts["ylabel"] = axis_labels
    if geo:
        projection = _output_projection(da)
        opts |= {"geo": True, "projection": projection}
        if coastline:
            # hvplot's coastline is a geoviews Feature, which the plot re-projects
            # from scratch every time it renders a frame. Fine for the single draw
            # every other family does; ruinous for an embedded movie, which renders
            # every frame — movies pass coastline=False and overlay a static,
            # once-projected path instead (see _movie_coastline).
            opts["coastline"] = "50m"
        if project or projection is not None:
            # project the data to the output projection now, once, instead of
            # letting the plot re-project it per rendered frame. Movies set
            # ``project`` to pay that cost a single time rather than per frame.
            #
            # A straddling field (``projection`` is the 180-centred frame) *must*
            # be projected regardless: with ``project=False`` geoviews leaves the
            # quadmesh in its raw 0-360 longitudes and only relabels the axes, so
            # every cell past 180 wraps back to the far side — the basin tears
            # across both edges with a blank Atlantic in the middle, exactly the
            # centre-0 layout the 180-centred frame was chosen to avoid. Projecting
            # the mesh vertices once keeps it one contiguous piece.
            opts["project"] = True
        # movie callers already downgrade via _tiles_for before reaching here (so this
        # is a no-op for them); a caller that hands tiles straight to _quadmesh still
        # gets the same seam protection rather than a silently broken map.
        tiles = _tiles_for(tiles, da)
        if tiles:
            # a basemap gives the eye real coastline and terrain where the field is
            # masked, which the 50m outline cannot. On by default for a movie, since a
            # notebook watching one is on the web already; tiles=False opts back out
            # for a notebook that has to work offline.
            opts["tiles"] = tiles
            opts.pop("projection", None)
            # A tile layer's own extent is the whole world, so it might seem the view
            # would open zoomed out to match. It does not: geoviews/hvplot frame the
            # view on the field's own extent regardless (verified against geoviews
            # 1.15.1 / hvplot 0.12.2). Do not add xlim/ylim here to "fix" this — an
            # xlim/ylim pair in degrees silently empties a *rasterized* field instead
            # (the image comes back with zero finite values), and one in Web Mercator
            # metres is just ignored, since hvplot's geo xlim/ylim are read as lon/lat.
    try:
        result = da.hvplot.quadmesh(**opts)
    except Exception:  # pragma: no cover - geoviews/cartopy unavailable
        opts.pop("geo", None)
        opts.pop("coastline", None)
        opts.pop("projection", None)
        opts.pop("project", None)
        # tiles-without-geo is a different hvplot code path (it swaps x/y into
        # dimensions to project them), which breaks on 2-D curvilinear lon/lat
        # coordinates -- there is no basemap to fall back to without geo anyway.
        opts.pop("tiles", None)
        result = da.hvplot.quadmesh(**opts)
    if invert_y:
        # Not folded into `opts` above: hvplot's own constructor-time option
        # resolution does not recognize "invert_yaxis" for a quadmesh (it warns
        # "option not found" and silently drops it, even though the *element's*
        # bokeh plot class accepts it fine) -- confirmed by applying it via
        # .opts() on the already-built element instead, which this does.
        result = result.opts(invert_yaxis=True)
    return result


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
    domain=None,
    hover: bool = True,
    rasterize: bool | str = "auto",
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

    ``rasterize="auto"`` (the default, see :func:`_should_rasterize`) ships an image
    instead of the raw mesh once a panel is past :data:`RASTERIZE_ABOVE_CELLS` — a
    curvilinear model grid drawn without it sends bokeh a Python loop over every cell
    (geoviews always projects to 2-D coordinates, which forces holoviews' irregular-mesh
    path), taking minutes instead of seconds. Resolved once from the test panel so all
    three panels rasterize together, same as :func:`_field_movie` decides once for every
    frame. Pass ``rasterize=False`` for a field small enough to zoom into sharply.
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
    raster = _should_rasterize(t, rasterize)

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
            hover=hover,
            rasterize=raster,
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
            hover=hover,
            rasterize=raster,
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
            hover=hover,
            rasterize=raster,
        ),
    ]
    outline = _domain_overlay(domain, t, geo=geo)
    if outline is not None:
        panels = [p * outline for p in panels]
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
    domain=None,
    hover: bool = True,
    rasterize: bool | str = "auto",
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
    equivalent, so only the text carries over, not the font/size). Left unset it
    defaults to the rows' shared identity, the same as the static grid — see
    :func:`~ocean_skill.plot.matplotlib_renderer.grid_suptitle`; ``title=""`` drops it.

    Each row is titled from *its own* ``labels``, exactly as the static renderer
    does, falling back to the top-level ``labels`` only for a row that carries
    none. Rows in one ``compare()`` fan-out commonly come from *different*
    reference sources (nitrate from one WOA entry, phosphate from another), so
    reusing the first row's pair for every row would mislabel all but the first.

    Type sizes do **not** shrink with the row count the way the static renderer's do:
    a bokeh ``frame_width`` is fixed rather than a share of a page, so stacking more
    rows makes the page longer instead of making each panel smaller — the thing the
    static scale is compensating for does not happen here. ``font_scale`` still applies.

    ``rasterize`` and ``hover`` pass straight through to every row (see
    :func:`_field_row`); each row's ``rasterize="auto"`` decision is its own, since rows
    can carry different-sized grids.
    """
    hv = _extension()
    title = _default_grid_title(items, title)
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
            domain=domain,
            hover=hover,
            rasterize=rasterize,
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


def _default_grid_title(items, title):
    """The grid's overall title, defaulted to the rows' shared identity when unset.

    Kept identical to the static renderer's default (:func:`grid_suptitle`) so a stacked
    comparison names the same shared variable · depth · time in both, and only when the
    caller named none — ``title=""`` still suppresses it.
    """
    from ocean_skill.plot.matplotlib_renderer import grid_suptitle

    return grid_suptitle(items) if title is None else title


def _field_facet(
    item: dict[str, Any],
    geo=True,
    shared_axes: bool = True,
    title: str | None = None,
    ncols: int | None = None,
    shared_limits: bool = False,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    domain=None,
    hover: bool = True,
    rasterize: bool | str = "auto",
    **_,
):
    """One interactive map per value of the facet axis: a field over time, in order.

    The interactive twin of :func:`ocean_skill.plot.matplotlib_renderer.field_facet`,
    and the same two commitments hold. Every panel shares one colour scale, because
    the panels are one quantity at different times and per-panel scaling would hide
    exactly the change the figure exists to show. Panel titles come from the facet
    coordinate through the shared :func:`~ocean_skill.plot.matplotlib_renderer.
    facet_labels`, so a consecutive-month figure says ``Jan 2012`` here too and cannot
    be mistaken for the climatology that says ``Jan`` — and three days of one January
    say ``2013-01-16`` here too rather than repeating a month three times. The layout's
    own title defaults to the variable, depth and (if collapsed to one instant) time,
    from the shared :func:`~ocean_skill.plot.matplotlib_renderer.field_suptitle`, so the
    two renderers name the same field the same way; ``title=""`` drops it.

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

    ``rasterize="auto"`` (the default, see :func:`_should_rasterize`) decides once, from
    one panel's worth of cells, whether every panel ships an image instead of the raw
    mesh — the fix for the same per-cell Python loop :func:`_field_row` avoids, since a
    facet grid draws just as many curvilinear panels as it has frames.
    """
    from ocean_skill.colormaps import is_log
    from ocean_skill.plot.matplotlib_renderer import (
        _aspect_of,
        _limits,
        facet_labels,
        field_suptitle,
    )
    from ocean_skill.plot.typography import facet_layout

    hv = _extension()
    factor = _canvas_factor(size, zoom)
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
    title = (
        field_suptitle(
            field,
            standard_name=standard_name,
            depth=item.get("depth"),
            label=item.get("label"),
            facet_dim=facet_dim,
            row_dim=row_dim,
        )
        if title is None
        else title
    )
    seq, _div = cmaps_for(standard_name)
    log = is_log(standard_name)
    outline = _domain_overlay(domain, field, geo=geo)
    # one panel's worth of cells, not the whole faceted field, which would overcount by
    # the number of panels and rasterize a grid whose individual maps are small
    one_panel = field.isel({d: 0 for d in (facet_dim, row_dim) if d})
    raster = _should_rasterize(one_panel, rasterize)

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
        ncols, _nrows = facet_layout(
            n, _aspect_of(field), canvas=resolve_canvas(size, zoom)
        )
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
        mesh = _quadmesh(
            sub,
            title=title,
            cmap=seq,
            clim=clims[row],
            units=units,
            geo=geo,
            log=log,
            font_scale=font_scale,
            canvas_factor=factor,
            hover=hover,
            rasterize=raster,
        )
        return mesh if outline is None else mesh * outline

    panels = [
        _panel(row, col)
        for row in range(nrows)
        for col in range(ncols if row_dim is not None else n)
    ]
    if len(panels) == 1:
        # A lone panel is an Overlay, which has no .cols() -- and stays a plain
        # pannable overlay rather than gaining a one-element Layout's chrome, the
        # same choice a single frame of _facet_movie/_field_movie makes.
        single = panels[0]
        if title:
            panel_label = labels[0] if (facet_dim and labels and labels[0]) else ""
            text = f"{title} — {panel_label}" if panel_label else str(title)
            single = single.opts(title=text)
        return single
    layout = panels[0]
    for extra in panels[1:]:
        layout = layout + extra
    if len(panels) > 1:
        layout = layout.cols(ncols).opts(hv.opts.Layout(shared_axes=shared_axes))
    if title:
        layout = layout.opts(title=str(title))
    return layout


def _section(
    item: dict[str, Any],
    title: str | None = None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    hover: bool = True,
    rasterize: bool | str = "auto",
    **_,
):
    """One interactive vertical section: depth against along-path distance.

    The interactive twin of
    :func:`ocean_skill.plot.matplotlib_renderer.section`. Draws through the same
    :func:`~ocean_skill.plot.section.prepare_section` the static renderer calls, so
    the two cannot disagree about which axis is depth, which sign it reads
    positive, or what the along-path axis is labelled — and through the same
    :func:`_quadmesh` every geographic family draws through, just with ``geo=False``
    (no tiles, no coastline, no cartopy projection: a section is not a map) and its
    non-geographic options (``aspect``, ``invert_y``, ``bgcolor``) instead.
    """
    from ocean_skill.colormaps import is_log
    from ocean_skill.plot.matplotlib_renderer import _limits, suptitle_text
    from ocean_skill.plot.section import prepare_section
    from ocean_skill.plot.typography import SECTION_ASPECT

    _extension()
    factor = _canvas_factor(size, zoom)
    field, geometry = prepare_section(item["field"])
    units = item.get("units") or ""
    standard_name = item.get("standard_name")
    if title is None:
        title = suptitle_text(
            standard_name, (item.get("depth"), geometry.path_note),
            label=item.get("label"),
        )
    seq, _div = cmaps_for(standard_name)
    log = is_log(standard_name)
    vmin, vmax = _limits(field)
    if log:
        vmin = max(vmin, 1e-6)
    raster = _should_rasterize(field, rasterize)

    return _quadmesh(
        field,
        title=title,
        cmap=seq,
        clim=(vmin, vmax),
        units=units,
        geo=False,
        log=log,
        font_scale=font_scale,
        canvas_factor=factor,
        width_px=SOLO_PANEL_WIDTH_PX,
        axis_labels=(geometry.x_label, geometry.y_label),
        hover=hover,
        rasterize=raster,
        x=geometry.x_name,
        y=geometry.y_name,
        aspect=SECTION_ASPECT,
        invert_y=True,
        bgcolor="#d9d9d9",
    )


def _section_row(
    item: dict[str, Any],
    labels=("test", "reference"),
    shared_axes: bool = True,
    metric_keys=DEFAULT_METRIC_KEYS,
    title: str | None = None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    hover: bool = True,
    rasterize: bool | str = "auto",
    **_,
):
    """Test | reference | difference vertical sections, as three linked interactive maps.

    The section counterpart of :func:`_field_row`: the same shared-axes linking and
    the same metrics-folded-into-the-difference-title convention, but drawn through
    :func:`ocean_skill.plot.section.prepare_section_row` and the same non-geographic
    :func:`_quadmesh` bundle :func:`_section` draws its own single panel through
    (``geo=False``, a section's own axis names/aspect/inverted-y/grey background)
    rather than the geographic one :func:`_field_row` uses.

    A comparison section is never stacked into a grid (see
    :class:`~ocean_skill.comparison.ComparisonSet`'s own refusal on more than one
    ``section_row``), so unlike :func:`_field_row` there is no ``row_label`` or
    ``domain`` to thread through here. The title default is likewise owned inside
    this function rather than a grid caller passing one in: it needs the path's own
    endpoints (:attr:`~ocean_skill.plot.section.SectionGeometry.path_note`), which
    only :func:`~ocean_skill.plot.section.prepare_section_row` can supply.
    """
    from ocean_skill.colormaps import is_log
    from ocean_skill.plot.matplotlib_renderer import _limits, suptitle_text
    from ocean_skill.plot.section import prepare_section_row
    from ocean_skill.plot.typography import SECTION_ASPECT

    hv = _extension()
    factor = _canvas_factor(size, zoom)
    values, geometry = prepare_section_row(item["aligned"])
    t, r, d = values["test"], values["reference"], values["difference"]
    units = item.get("units") or ""
    standard_name = item.get("standard_name")
    if title is None:
        title = suptitle_text(
            standard_name, (item.get("depth"), item.get("time"), geometry.path_note)
        )
    seq, div = cmaps_for(standard_name)
    log = is_log(standard_name)
    vmin, vmax = _limits(t, r)
    if log:
        vmin = max(vmin, 1e-6)
    dmax = float(np.nanpercentile(np.abs(np.asarray(d)), 98)) or 1.0
    tl, rl = labels
    raster = _should_rasterize(t, rasterize)

    diff_title = "difference"
    summary = _metrics_summary(item.get("metrics"), metric_keys)
    if summary:
        diff_title = f"difference ({summary})"

    section_opts = dict(
        geo=False,
        x=geometry.x_name,
        y=geometry.y_name,
        aspect=SECTION_ASPECT,
        invert_y=True,
        bgcolor="#d9d9d9",
        axis_labels=(geometry.x_label, geometry.y_label),
        font_scale=font_scale,
        canvas_factor=factor,
        hover=hover,
        rasterize=raster,
    )
    panels = [
        _quadmesh(
            t, title=str(tl), cmap=seq, clim=(vmin, vmax), units=units, log=log,
            **section_opts,
        ),
        _quadmesh(
            r, title=str(rl), cmap=seq, clim=(vmin, vmax), units=units, log=log,
            **section_opts,
        ),
        _quadmesh(
            d,
            title=diff_title,
            cmap=div,
            clim=(-dmax, dmax),
            units=f"test − reference {units}",
            **section_opts,
        ),
    ]
    row = panels[0] + panels[1] + panels[2]
    row = row.opts(hv.opts.Layout(shared_axes=shared_axes))
    if title:
        row = row.opts(title=str(title))
    return row


def _station_overlay(stations, name: str, colors, da, *, geo: bool):
    """Return one metric's station values as hoverable dots, or ``None``.

    Plain ``hv.Points`` rather than a geoviews element — the same choice
    :func:`_domain_overlay` makes and for the same reason: a static figure has no
    per-frame reprojection cost to worry about, but staying in one coordinate
    family with the outline it is drawn alongside is simpler than mixing two. The
    antimeridian shift matches :func:`_domain_overlay`'s: identity for every panel
    except one in :func:`_output_projection`'s shifted frame.
    """
    if stations is None or name not in stations.get("values", {}):
        return None
    import holoviews as hv
    import pandas as pd

    lon = np.asarray(stations["lon"], dtype="float64")
    lat = np.asarray(stations["lat"], dtype="float64")
    if geo and _output_projection(da) is not None:
        lon = (lon % 360.0) - 180.0
    frame = {"lon": lon, "lat": lat, "value": np.asarray(stations["values"][name])}
    names = stations.get("names")
    vdims = ["value"]
    if names is not None:
        frame["station"] = list(names)
        vdims.append("station")
    return hv.Points(pd.DataFrame(frame), kdims=["lon", "lat"], vdims=vdims).opts(
        color="value",
        cmap=colors.cmap,
        clim=colors.clim(),
        size=8,
        line_color="white",
        line_width=1,
        tools=["hover"],
        apply_ranges=False,
    )


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
    domain=None,
    hover: bool = True,
    rasterize: bool | str = "auto",
    shared_limits: bool = False,
    layout: str = "rows",
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
    same panels the same way. An item carrying ``stations`` (see
    :func:`ocean_skill.plot.map_metrics.build_items`) overlays each station's true value
    as a hoverable dot in the same colour scale (:func:`_station_overlay`).

    Each metric's **overall** value — reduced over space and the scored axis together —
    joins its panel's title, which is the same substitution :func:`_field_row` makes
    for a comparison's corner box: bokeh has no free-floating annotation that survives
    pan/zoom as cleanly as a title. Units stay on each panel's own colorbar, as they do
    statically.

    ``rasterize="auto"`` (the default, see :func:`_should_rasterize`) resolves once from
    the first metric's map, same fix as :func:`_field_row`'s: a metric map is a
    curvilinear mesh too, and hits the same per-cell loop past
    :data:`RASTERIZE_ABOVE_CELLS`.

    ``shared_limits=True`` pools each metric's values over every row before choosing
    its colour limits, the same as the static family — every panel of one metric then
    carries the same scale, comparable by colour. Each panel still shows its own
    colorbar (bokeh has no cross-panel bar the way a matplotlib figure does), but with
    identical limits they read as one. ``layout="columns"`` transposes the grid —
    comparisons across, metrics down — by reordering which panels tile before
    ``.cols()`` and does not otherwise change what each panel's title says, since a
    bokeh panel already carries both the metric and (when stacked) the comparison's
    own label in its own title.
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

    if layout not in ("rows", "columns"):
        raise ValueError(f"layout={layout!r} — expected 'rows' or 'columns'")
    titles = metric_panel_titles(names)
    stacked = len(items) > 1
    if stacked:
        ncols = len(items) if layout == "columns" else len(names)
    elif ncols is None:
        ncols, _nrows = facet_layout(
            len(names),
            _aspect_of(items[0]["skill"][names[0]]),
            canvas=resolve_canvas(size, zoom),
        )
    ncols = max(int(ncols), 1)
    factor = _canvas_factor(size, zoom)
    arrays = {i: metric_arrays(item["skill"], names) for i, item in enumerate(items)}
    outline = _domain_overlay(domain, items[0]["skill"][names[0]], geo=geo)
    raster = _should_rasterize(items[0]["skill"][names[0]], rasterize)

    # One colour scale per metric, pooled over every row, when asked to share -- the
    # interactive twin of the static family's shared_limits (each panel still draws
    # its own colorbar; bokeh has no cross-panel bar, but identical limits read as one).
    shared_colors: dict[str, Any] = {}
    if shared_limits and stacked:
        for name in names:
            pooled = np.concatenate(
                [np.asarray(arrays[i][name]).ravel() for i in range(len(items))]
            )
            shared_colors[name] = metric_colors(
                name, pooled, standard_name=items[0].get("standard_name")
            )

    # "rows" (default) tiles metrics across within a comparison's row; "columns"
    # transposes by tiling comparisons across within a metric's row instead -- paired
    # with ncols above, .cols() below wraps into the same transposed grid either way.
    order = (
        [(row, name) for name in names for row in range(len(items))]
        if (stacked and layout == "columns")
        else [(row, name) for row in range(len(items)) for name in names]
    )

    panels = []
    for row, name in order:
        item = items[row]
        colors = (
            shared_colors[name]
            if shared_limits and stacked
            else metric_colors(
                name, arrays[row][name], standard_name=item.get("standard_name")
            )
        )
        base = titles[names.index(name)]
        # bokeh has no rotated row label, so the comparison joins the panel's own
        # title -- the same move _field_row makes for a field grid's row_label
        if stacked and item.get("row_label"):
            base = f"{item['row_label']} — {base}"
        value = metric_value_text(item.get("metrics"), name)
        mesh = _quadmesh(
            item["skill"][name],
            title=f"{base} ({value})" if value else base,
            cmap=colors.cmap,
            clim=colors.clim(),
            units=str(item["skill"][name].attrs.get("units", "") or ""),
            geo=geo,
            log=colors.log,
            font_scale=font_scale,
            canvas_factor=factor,
            hover=hover,
            rasterize=raster,
        )
        points = _station_overlay(
            item.get("stations"), name, colors, item["skill"][name], geo=geo
        )
        for extra in (outline, points):
            if extra is not None:
                mesh = mesh * extra
        panels.append(mesh)

    result = panels[0]
    for extra in panels[1:]:
        result = result + extra
    if len(panels) > 1:
        result = result.cols(ncols).opts(hv.opts.Layout(shared_axes=shared_axes))
    if title:
        result = result.opts(title=str(title))
    return result


#: Name of the dimension a movie's frames vary along, i.e. what the slider is labelled.
#: A movie's frames are usually time steps, but not necessarily — the static renderer
#: calls the per-frame text a ``frame_label`` for the same reason — so the neutral name
#: is the honest one.
FRAME_DIM = "frame"


#: Cell count past which ``rasterize="auto"`` turns datashader on. Above roughly this,
#: bokeh is being asked to ship and hit-test more quads than a screen has pixels, so
#: rendering server-side to an image is both faster and no less accurate — the mesh was
#: never resolvable at that size anyway. Below it, the raw mesh is sharper on zoom.
RASTERIZE_ABOVE_CELLS = 100_000


def _check_tiles(tiles: str | bool | None) -> str | bool | None:
    """Return ``tiles`` once confirmed to name a real tile source.

    ``True`` means hvplot's own default basemap (OpenStreetMap) and needs no lookup;
    any other string is checked against :mod:`geoviews.tile_sources` because a typo is
    otherwise reported as ``"cannot swap from dimension 'lon'"`` from deep inside
    hvplot's projection handling, which names neither the argument at fault nor
    anything a reader could act on.
    """
    if not tiles or tiles is True:
        return tiles
    try:
        from geoviews import tile_sources
    except ImportError:  # pragma: no cover - geoviews ships in environment.yml
        return tiles
    known = sorted(tile_sources.tile_sources)
    if tiles not in known:
        raise ValueError(
            f"unknown tile source {tiles!r}. geoviews offers: {', '.join(known)}"
        )
    return tiles


def _tiles_for(tiles, *fields):
    """Return ``tiles``, downgraded to ``None`` when a field genuinely straddles.

    Web tiles are Web Mercator, whose frame is fixed at ±180 -- a domain that
    actually straddles the antimeridian (a Pacific model crossing the dateline)
    projects there as two disjoint pieces meeting at ±2e7 metres, so the tiled
    quadmesh renders split across the map rather than as one contiguous field.
    There is no tiled frame that keeps such a domain whole, so the basemap is
    dropped in favour of the offline coastline (:func:`_movie_coastline`) rather
    than shipping a broken map; pass ``tiles=False`` to silence the warning.

    Tested with :func:`~ocean_skill.align.natural_convention` rather than
    :func:`_output_projection`'s broader "any longitude past 180" check: a model
    stored 0-360 natively but sitting entirely east of it (a Gulf of Mexico grid at
    260-270°, say, reaching the renderer straight off a model-only
    :class:`~ocean_skill.field.Field` with no reference to harmonize it) also has
    ``lon.max() > 180`` without straddling anything -- Web Mercator wraps 260-270
    into one contiguous -100…-90 slice, no seam involved. Downgrading tiles for a
    domain like that would drop a feature that works fine; the natural-convention
    test only answers yes for a domain that would actually split.
    """
    if not tiles or not any(natural_convention(f) == "0-360" for f in fields):
        return tiles
    warnings.warn(
        "a web tile basemap is fixed in Web Mercator's ±180 frame, which splits "
        "this dateline-straddling domain at the seam -- falling back to the "
        "offline coastline outline instead. Pass tiles=False to silence this.",
        stacklevel=2,
    )
    return None


def _should_rasterize(da, rasterize) -> bool:
    """Resolve ``rasterize=True/False/"auto"`` for one frame.

    ``"auto"`` is a real choice rather than a hedge: the right answer depends on the
    mesh, and a movie of a model field is usually far past the size where shipping every
    quad to the browser stops being sensible, while the small synthetic grids in the
    tests are far below it. Datashader is optional, so a missing one means False rather
    than an import error at draw time.
    """
    if rasterize != "auto":
        return bool(rasterize)
    if int(np.prod(da.shape)) <= RASTERIZE_ABOVE_CELLS:
        return False
    try:
        import datashader  # noqa: F401
    except ImportError:  # pragma: no cover - datashader is present in environment.yml
        warnings.warn(
            f"{int(np.prod(da.shape)):,} cells per frame would be worth rasterizing, "
            "but datashader is not installed; drawing the raw mesh instead "
            "(conda install -c conda-forge datashader).",
            stacklevel=2,
        )
        return False
    return True


#: Frames bigger than this stay dask-backed rather than being loaded up front.
#: Below it, one parallel read replaces a read per frame (see _preload_frames); above
#: it, holding every raw frame in memory at once is the greater risk, so the movie
#: draws from the store frame by frame instead.
PRELOAD_FRAMES_BELOW_BYTES = 4 * 1024**3


def _preload_frames(da):
    """Load the movie's frames into memory once, when they fit.

    A dask-backed field goes back to its store for every frame drawn — after the colour
    limits already read the whole thing once — so a movie of N frames pays N+1 reads,
    one at a time. One ``.load()`` up front turns that into a single parallel read that
    every frame then slices in memory. Fields past
    :data:`PRELOAD_FRAMES_BELOW_BYTES` keep the lazy per-frame reads and say so, since
    an ``every=`` or a coarser ``aggregate=`` is the real fix at that size.
    """
    if getattr(da, "chunks", None) is None:
        return da
    if da.nbytes > PRELOAD_FRAMES_BELOW_BYTES:
        warnings.warn(
            f"the movie's frames are {da.nbytes / 1024**3:.1f} GB, too much to hold in "
            "memory at once, so each frame reads from the store as it is drawn — "
            "expect the movie to take a while to build. Thin it with every= or a "
            "coarser aggregate= for a quicker one.",
            stacklevel=2,
        )
        return da
    return da.load()


def _lon_pieces(lon0: float, lon1: float) -> list[tuple[float, float]]:
    """Split a longitude span into ``(west, east)`` clip pieces in the ±180 frame.

    Natural Earth geometry lives in −180…180 and so does the plot: the mesh is
    projected into that frame whatever convention the grid uses, so the coastline is
    clipped — and left — in it. A 0…360-style span becomes the equivalent pieces, two
    of them when it crosses the antimeridian, exactly where the projected mesh lands.
    """
    if -180 <= lon0 and lon1 <= 180:
        return [(lon0, lon1)]
    if lon0 < 180 < lon1:
        return [(lon0, 180.0), (-180.0, lon1 - 360.0)]
    if lon0 >= 180:
        return [(lon0 - 360.0, lon1 - 360.0)]
    return [(max(lon0, -180.0), min(lon1, 180.0))]


def _movie_coastline(*fields):
    """The 50m coastline as one static ``hv.Path``, clipped to the fields' extent.

    hvplot's own ``coastline`` overlay is a geoviews ``Feature``, and a Feature is
    lazy: rendering it projects the whole world's coastline geometry from scratch —
    which an embedded movie does once per frame, measured at 0.6–1.2 s and ~1.5 MB of
    page *per frame*, dominating everything else the movie does. The coastline never
    changes between frames, so it is built here exactly once: clipped to the fields'
    own extent (plus margin to pan into) and returned as a plain holoviews Path, in
    the movie's own output frame, that no geoviews machinery touches again. Movies
    with tiles don't need it at all — the basemap draws the coast — so this only ever
    joins the untiled (offline) movie.

    That output frame is plain PlateCarree degrees for most domains, but one
    straddling the antimeridian is drawn in :func:`_output_projection`'s 180-centred
    frame instead (the same test :func:`_tiles_for` uses to decide tiles cannot show
    such a domain at all) — so the coastline's x is shifted into that frame too, or
    it would land 180° away from the mesh it is meant to outline.
    ``apply_ranges=False`` keeps the clip margin from widening the view.

    Returns ``None`` when the coastline cannot be built (no cartopy, or Natural Earth
    data unavailable offline) — a movie without an outline still plays.
    """
    import holoviews as hv

    # 180 for a domain _output_projection would centre the mesh on (straddling the
    # antimeridian); 0 -- an identity shift below -- for every other domain, leaving
    # today's coordinates untouched.
    central = 180.0 if any(_output_projection(f) is not None for f in fields) else 0.0

    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box

        lons = np.concatenate([np.asarray(f["lon"]).ravel() for f in fields])
        lats = np.concatenate([np.asarray(f["lat"]).ravel() for f in fields])
        lon0, lon1 = float(np.nanmin(lons)), float(np.nanmax(lons))
        lat0, lat1 = float(np.nanmin(lats)), float(np.nanmax(lats))
        pad_x, pad_y = 0.5 * max(lon1 - lon0, 1e-3), 0.5 * max(lat1 - lat0, 1e-3)
        lat0, lat1 = max(lat0 - pad_y, -90.0), min(lat1 + pad_y, 90.0)
        geoms = list(
            cfeature.NaturalEarthFeature("physical", "coastline", "50m").geometries()
        )
        segments = []
        for west, east in _lon_pieces(lon0 - pad_x, lon1 + pad_x):
            clip = box(west, lat0, east, lat1)
            for geom in geoms:
                gx0, gy0, gx1, gy1 = geom.bounds
                if gx1 < west or gx0 > east or gy1 < lat0 or gy0 > lat1:
                    continue
                piece = geom.intersection(clip)
                if piece.is_empty:
                    continue
                for line in getattr(piece, "geoms", [piece]):
                    coords = np.asarray(line.coords)
                    if len(coords) >= 2:
                        segments.append(coords[:, :2])
    except Exception as err:  # pragma: no cover - depends on local NE cache/network
        warnings.warn(
            f"could not build the movie's coastline overlay ({err}); the frames play "
            "without one. Natural Earth data downloads on first use, so a machine "
            "that has never drawn a coastline needs to be online once.",
            stacklevel=2,
        )
        return None
    # one NaN-separated array rather than one array per segment: bokeh breaks the line
    # at NaN either way, and holoviews walks every segment of a Path on every frame it
    # renders — a few hundred of them (islands, mostly) cost ~0.2s per embedded frame
    nan_row = np.full((1, 2), np.nan)
    merged = np.concatenate(
        [arr for seg in segments for arr in (seg, nan_row)][:-1] or [np.empty((0, 2))]
    )
    if central:
        # NaN separator rows pass through unchanged: arithmetic and mod on NaN stay
        # NaN. Each clip piece already lies inside one contiguous span of the output
        # frame (see _lon_pieces), so this never introduces a new seam of its own.
        merged[:, 0] = ((merged[:, 0] - central + 180.0) % 360.0) - 180.0
    return hv.Path([merged]).opts(
        color="black", line_width=1, apply_ranges=False, show_legend=False
    )


def _to_mercator(lons, lats):
    """Project geographic ``lons``/``lats`` to Web Mercator (metres).

    Shared by :func:`_locations` (its extent rectangles) and :func:`_domain_overlay`
    (a domain outline) — both draw plain holoviews elements under web tiles, which
    take no part in geoviews's automatic projection and so need this done by hand.
    """
    import cartopy.crs as ccrs

    pts = ccrs.GOOGLE_MERCATOR.transform_points(
        ccrs.PlateCarree(),
        np.asarray(lons, dtype=float),
        # Web Mercator diverges at the poles; clamp to its own limit
        np.clip(np.asarray(lats, dtype=float), -85.06, 85.06),
    )
    return pts[:, 0], pts[:, 1]


def _domain_overlay(domain, da, *, geo: bool = True, tiles=None):
    """Return the model-domain outline as one dashed ``hv.Path`` in the panel's frame.

    Plain holoviews rather than a geoviews element — the same choice
    :func:`_movie_coastline` and :func:`_locations` make, and for the same reason: a
    geoviews element re-projects its geometry from scratch on every render, fine once
    but ruinous embedded in a movie. Under tiles the ring is projected to Web
    Mercator (:func:`_to_mercator`); for an untiled geo panel it is shifted into
    :func:`_output_projection`'s frame with the same formula :func:`_movie_coastline`
    uses for its coastline — a plain ``PlateCarree(central_longitude=180)`` frame is
    ``x = (lon % 360) - 180``, an identity shift for every other domain (one already
    in ``_output_projection``'s own default frame needs no correction). ``domain``
    accepts either spelling :func:`~ocean_skill.plot.matplotlib_renderer.domain_ring`
    does — the bbox this option has always taken, or the true-perimeter ring a
    curvilinear source's catalog entry supplies. Returns ``None`` when there is
    nothing to draw.
    """
    from ocean_skill.plot.matplotlib_renderer import domain_ring

    ring = domain_ring(domain)
    if ring is None:
        return None
    import holoviews as hv

    xs, ys = ring[:, 0].astype(float), ring[:, 1].astype(float)
    if geo and tiles:
        xs, ys = _to_mercator(xs, ys)
    elif geo and _output_projection(da) is not None:
        xs = (xs % 360.0) - 180.0
    return hv.Path([np.column_stack([xs, ys])]).opts(
        color="black",
        line_dash="dashed",
        line_width=1,
        apply_ranges=False,
        show_legend=False,
    )


def _frame_map(keys: list[str], draw, *, frame_dim: str | None = None):
    """Return a ``HoloMap`` holding every frame, drawn up front.

    A lazy ``DynamicMap`` was tried here instead: it draws only the frame you're looking
    at, so opening costs one frame however many there are. But it draws that frame by
    sending the slider's new value back through a live Panel↔kernel comm channel — a
    channel that has to be bound perfectly to work at all, and in practice often isn't
    (``pn.extension()`` running mid-cell, VS Code's own comm mode, a notebook that has
    been exported or reopened with no kernel behind it). When it isn't bound the slider
    moves and the plot doesn't, with no error to say why. Eager frames have no channel to
    lose: every frame is embedded in the display itself (see :func:`_with_widget`), so
    stepping through them is plain client-side JavaScript.

    What makes this affordable again is ``rasterize`` — past
    :data:`RASTERIZE_ABOVE_CELLS` each frame is a small fixed-size image rather than a
    full quadmesh, so building and embedding all of them costs a fraction of what it
    would for the raw mesh. The cost that remains is display time and page size, both
    proportional to frame count; ``every=`` and the static renderer's mp4 are the levers
    for a movie long enough for that to matter (see docs/movies.md).
    """
    import holoviews as hv

    dim = hv.Dimension(frame_dim or FRAME_DIM, values=keys)
    return hv.HoloMap({key: draw(i) for i, key in enumerate(keys)}, kdims=[dim])


def _subject(item: dict[str, Any]) -> str:
    """Return what the movie is *of*: variable, depth and source, as the item has them.

    A frame label alone ("2010-01-29") says which frame you are looking at and nothing
    about what it shows — fine on a figure with a caption, thin on a plot you scrolled
    back to a week later. The static renderer has a suptitle for this; here the panel
    title is the only place, so it carries both, and ``title=`` still overrides.

    Every part is optional, because an item assembled by hand may carry none of them.
    """
    from ocean_skill.plot.matplotlib_renderer import field_title

    # field_title is the package's one spelling of "which variable is this" (it goes
    # through vars.short_name, as every legend and axis label does); the depth and the
    # source are what a movie can add to it, having only the panel title to say them in
    parts = [field_title(item.get("standard_name")), item.get("depth")]
    subject = ", ".join(str(p) for p in parts if p)
    source = item.get("label")
    if source and subject:
        return f"{source}: {subject}"
    return subject or (str(source) if source else "")


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
    size=None,
    zoom: float = 1.0,
    hover: bool = False,
    rasterize: bool | str = "auto",
    tiles: str | bool | None = True,
    domain=None,
    **_,
):
    """One source's facet axis on a slider: the interactive twin of ``facet_movie``.

    Where :func:`_field_facet` lays the axis out as panels, this steps through it in
    place — the same field, the same labels, one panel at a time. For a long axis that
    is the more useful of the two: forty panels on a page are each too small to read,
    while forty frames are full size and a drag apart.

    Every frame is drawn and embedded up front (see :func:`_frame_map`), so the slider
    works with no live kernel behind it — in an exported notebook exactly as in a
    running one.

    Being the only panel on the page, it gets the page: the frame is
    :data:`SOLO_PANEL_WIDTH_PX` rather than the third-of-a-row every other family draws
    at (override with ``width_px``). Axis titles are shortened for the same reason — a
    ROMS coordinate's own ``long_name`` is "longitude of rho-points (degrees East)",
    which bokeh truncates anyway; pass ``axis_labels=None`` to keep whatever the
    coordinates say.

    Three knobs exist because a movie is watched rather than inspected, and the defaults
    that suit a single still do not suit a hundred frames of a model grid:

    * ``hover=False`` — a hover readout makes bokeh hit-test every quad. Worth it on one
      map you are reading values off; pure cost on a movie you are watching. Pass
      ``True`` to get it back.
    * ``rasterize="auto"`` — render the mesh to an image with datashader once it is
      bigger than :data:`RASTERIZE_ABOVE_CELLS`. See :func:`_should_rasterize`.
    * ``tiles=True`` — a basemap under the field (OpenStreetMap by default), on by
      default since a notebook watching a movie is on the web already. Pass a
      :mod:`geoviews.tile_sources` name (e.g. ``tiles="EsriTerrain"`` or
      ``"CartoLight"``) for a different one, or ``tiles=False`` for a notebook that has
      to work offline — which swaps the basemap for a static 50m coastline outline
      (see :func:`_movie_coastline` for why a movie never uses hvplot's own).

    One colour scale for the whole movie, as statically, and for the same reason — a
    scale that moved with the slider would make a change in the ruler look like a change
    in the field.

    What the movie is *of* joins each frame's title (``GOM_bgc: alkalinity, surface —
    2013-01-16``) rather than sitting above it as the static suptitle does: bokeh's only
    title here is the panel's own, and a fixed one set on the map would replace the
    frame labels rather than joining them. Same substitution :func:`_field_row` makes
    for a row label. The variable is spelled by
    :func:`~ocean_skill.plot.matplotlib_renderer.field_title`, so it reads the same here
    as on a static figure; the depth and source are what a movie adds, having nowhere
    else to put them. An explicit ``title=`` replaces the subject outright and
    ``title=""`` leaves the frame labels bare.
    """
    from ocean_skill.colormaps import is_log
    from ocean_skill.plot.matplotlib_renderer import (
        _limits,
        _one_facet_axis,
        _select_frames,
        frame_labels,
    )

    _extension()
    factor = _canvas_factor(size, zoom)
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
    frames_da = _preload_frames(field.isel({facet_dim: indices}))
    if shared_limits:
        # the selected frames *are* the whole field unless every= thinned them, so the
        # loaded copy can feed the colour limits too, instead of a second full read of
        # a lazy store
        scope = frames_da if len(indices) == int(field.sizes[facet_dim]) else field
    else:
        scope = frames_da.isel({facet_dim: 0})
    vmin, vmax = _limits(scope)
    if log:
        vmin = max(vmin, 1e-6)
    raster = _should_rasterize(frames_da.isel({facet_dim: 0}), rasterize)
    tiles = _tiles_for(_check_tiles(tiles), frames_da)
    # with tiles the basemap draws the coast; without them a static, once-built
    # outline stands in for the per-frame Feature that hvplot would overlay
    coast = _movie_coastline(frames_da) if geo and not tiles else None
    outline = _domain_overlay(domain, frames_da, geo=geo, tiles=tiles)
    subject = _subject(item) if title is None else title

    def draw(position: int):
        frame = frames_da.isel({facet_dim: position})
        mesh = _quadmesh(
            frame,
            # what is being shown, then which frame of it — the subject stays put while
            # the timestamp changes, so the eye is not re-reading the whole title every
            # frame to find the part that moved
            title=f"{subject} — {keys[position]}" if subject else keys[position],
            cmap=seq,
            clim=(vmin, vmax),
            units=units,
            geo=geo,
            log=log,
            font_scale=font_scale,
            canvas_factor=factor,
            width_px=width_px,
            axis_labels=axis_labels,
            hover=hover,
            rasterize=raster,
            tiles=tiles,
            coastline=False,
            project=True,
        )
        for overlay in (coast, outline):
            if overlay is not None:
                mesh = mesh * overlay
        return mesh

    movie = _frame_map(keys, draw)
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
    size=None,
    zoom: float = 1.0,
    hover: bool = False,
    rasterize: bool | str = "auto",
    tiles: str | bool | None = True,
    domain=None,
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

    ``tiles=True`` (the default) puts a basemap under all three panels, including the
    difference — the field there is largely empty near the coast, which a basemap fills
    in the same way it does for :func:`_facet_movie`. Pass a :mod:`geoviews.tile_sources`
    name for a different map, or ``tiles=False`` for a notebook that has to work
    offline, which swaps the basemap for one static coastline outline shared by all
    three panels (see :func:`_movie_coastline`).

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
    factor = _canvas_factor(size, zoom)
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
    raster = _should_rasterize(items[0]["aligned"]["reference"], rasterize)
    tiles = _tiles_for(
        _check_tiles(tiles), first["aligned"]["test"], first["aligned"]["reference"]
    )
    # one static coastline for all three panels and every frame (the frames share the
    # aligned grid); with tiles the basemap draws the coast instead
    coast = (
        _movie_coastline(first["aligned"]["test"], first["aligned"]["reference"])
        if geo and not tiles
        else None
    )
    outline = _domain_overlay(domain, first["aligned"]["test"], geo=geo, tiles=tiles)
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

    def draw(position: int, panel: int):
        mesh = _panel_mesh(position, panel)
        for overlay in (coast, outline):
            if overlay is not None:
                mesh = mesh * overlay
        return mesh

    def _panel_mesh(position: int, panel: int):
        item = items[position]
        aligned = item["aligned"]
        tl, rl = item.get("labels") or labels
        if panel == 0:
            # the frame label goes on the test panel's title as well as on the slider:
            # the static renderer draws it in the panel, and a saved page is read the
            # same way a figure is
            return _quadmesh(
                aligned["test"],
                title=f"{keys[position]} — {tl}",
                cmap=seq,
                clim=(vmin, vmax),
                units=units,
                geo=geo,
                log=log,
                font_scale=font_scale,
                canvas_factor=factor,
                hover=hover,
                rasterize=raster,
                tiles=tiles,
                coastline=False,
                project=True,
            )
        if panel == 1:
            return _quadmesh(
                aligned["reference"],
                title=str(rl),
                cmap=seq,
                clim=(vmin, vmax),
                units=units,
                geo=geo,
                log=log,
                font_scale=font_scale,
                canvas_factor=factor,
                hover=hover,
                rasterize=raster,
                tiles=tiles,
                coastline=False,
                project=True,
            )
        summary = _metrics_summary(item.get("metrics"), metric_keys)
        return _quadmesh(
            aligned["difference"],
            title=f"difference ({summary})" if summary else "difference",
            cmap=div,
            clim=(-dmax, dmax),
            units=f"test − reference {units}",
            geo=geo,
            font_scale=font_scale,
            canvas_factor=factor,
            hover=hover,
            rasterize=raster,
            tiles=tiles,
            coastline=False,
            project=True,
        )

    # three maps over one shared frame dimension, so a single widget steps all three
    maps = [_frame_map(keys, lambda i, p=panel: draw(i, p)) for panel in range(3)]
    layout = (maps[0] + maps[1] + maps[2]).opts(hv.opts.Layout(shared_axes=shared_axes))
    if title:
        layout = layout.opts(title=str(title))
    layout = _with_widget(layout, widget=widget, fps=fps, frame_dim=FRAME_DIM)
    if save:
        _save_interactive(layout, save)
    return layout


#: Cached on first use — (pane class, player class) whose notebook repr embeds every
#: frame instead of asking a live kernel to draw the next one. Built lazily because
#: panel, like the rest of this module, is imported only once bokeh is actually needed.
_EMBED_TYPES: tuple[type, type] | None = None


def _embed_types() -> tuple[type, type]:
    """Return (pane, player) classes whose repr bakes in every frame.

    A stock ``pn.pane.HoloViews`` repr draws only the current frame and leaves the rest
    to be requested over a live comm channel — see :func:`_frame_map` for why that
    channel is the thing actually breaking the slider. Forcing ``embed=True`` for the
    repr runs the same machinery ``pane.save(embed=True)`` already uses, which bakes
    every frame's data into the page up front: no comm, no kernel, nothing to lose.

    ``comms="default"`` has to be forced alongside it: panel checks for a VS Code or
    ipywidgets comm *before* it checks ``config.embed``, and takes that live-kernel path
    instead if one is available (VS Code is, whenever ``jupyter_bokeh`` is installed) —
    which would put the very channel this is working around back in the loop.

    A stock ``pn.widgets.Player``'s embedded states are also one short: it lists
    ``range(start, end, step)``, which is end-exclusive, so the last frame of the movie
    never makes it into the page. The subclass below lists ``end + step`` instead.
    """
    global _EMBED_TYPES
    if _EMBED_TYPES is None:
        import panel as pn

        class _EmbeddedHoloViews(pn.pane.HoloViews):
            def _repr_mimebundle_(self, include=None, exclude=None):
                with pn.config.set(embed=True, comms="default"):
                    return super()._repr_mimebundle_(include, exclude)

        class _EmbedPlayer(pn.widgets.Player):
            def _get_embed_state(self, root, values=None, max_opts=3):
                if values is None:
                    values = list(range(self.start, self.end + self.step, self.step))
                return (
                    self,
                    self._models[root.ref["id"]][0],
                    values,
                    lambda x: x.value,
                    "value",
                    "cb_obj.value",
                )

        _EMBED_TYPES = (_EmbeddedHoloViews, _EmbedPlayer)
    return _EMBED_TYPES


def _with_widget(obj, *, widget: str, fps: int, frame_dim: str | None = None):
    """Wrap ``obj`` so its frame dimension gets the widget the caller asked for.

    Holoviews picks the widget itself, and for a dimension whose values are strings —
    a date, a depth — it picks a *dropdown*. That is the wrong control for an ordered
    sequence: stepping to the next frame is two clicks and a search rather than one
    nudge, and dragging through the movie is impossible. Panel can be told otherwise.

    ``"slider"`` (the default) is a ``DiscreteSlider``: drag it, or arrow-key through
    the frames, with the label still reading "2010-01-29" rather than an index.
    ``"player"`` is a ``Player`` — play, pause, step — running at ``fps``.
    ``"dropdown"`` returns the holoviews object untouched; holoviews' own display
    embeds a bare ``HoloMap`` the same way :func:`_embed_types` does explicitly here,
    so it needs no special handling.

    Both the slider and player panes use the embedding classes from
    :func:`_embed_types`: every frame is baked into the notebook cell's own output, so
    stepping through them is client-side JavaScript with no kernel behind it — it
    survives a shut-down kernel or an exported notebook exactly as a saved HTML page
    does (see :func:`_save_interactive`).
    """
    if widget == "dropdown":
        return obj
    if widget not in ("slider", "player"):
        raise ValueError(
            f"unknown widget {widget!r}; expected 'slider', 'player' or 'dropdown'"
        )
    import panel as pn

    pane_cls, player_cls = _embed_types()
    pn.extension()
    if widget == "player":
        pane = pane_cls(
            obj,
            widget_type="scrubber",
            widget_location="bottom",
            default_widgets={"scrubber": player_cls},
        )
    else:
        pane = pane_cls(
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
    """Write ``obj`` to a standalone HTML page, refusing formats bokeh cannot write.

    A page has no kernel behind it, so every frame has to be rendered and embedded in
    it — the same thing a notebook display now does (see :func:`_frame_map` and
    :func:`_with_widget`), just written to disk instead of a cell. That is the whole
    movie in one file, and it grows with the frame count — which is exactly why the
    static renderer's mp4 exists.
    """
    from pathlib import Path

    import holoviews as hv

    path = Path(save).expanduser()
    if path.suffix.lower() not in (".html", ".htm"):
        raise ValueError(
            f"the interactive renderer writes HTML, not {path.suffix or 'that'}: it "
            f"has no video encoder. Either save={str(path.with_suffix('.html'))!r} "
            f"here, or pass renderer='matplotlib' to write {path.name}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "save"):  # a panel pane; embed=True evaluates every frame
        obj.save(str(path), embed=True)
    else:  # widget="dropdown": a bare HoloMap, which hv.save embeds directly
        hv.save(obj, str(path))
    print(f"ocean-skill: interactive movie written to {path}")


# _TAB10 (from ocean_skill.plot.locations) colours the *locations map*'s fixed,
# small featureType set — a separate concern from the diagrams below. The Target
# diagram's group colours come from ``ocean_skill.plot.style.COLOR_CYCLE`` instead, the
# same 20-colour tab20-derived cycle ``summary._group_styles`` assigns by level index,
# so a Taylor/Target diagram and a series panel of the same comparisons agree and don't
# repeat until the 21st group.

#: matplotlib marker → bokeh marker, in the same order as ``summary._MARKERS``, so a
#: diagram keeps its shapes when the same call is rendered interactively.
_BOKEH_MARKERS = (
    "circle",
    "triangle",
    "square",
    "diamond",
    "inverted_triangle",
    "plus",
    "x",
)


#: Width (CSS pixels) of one interactive series panel on the default canvas — wider than
#: a map panel, since a time axis carrying years of dates needs the room.
SERIES_WIDTH_PX = 620

#: Corner name -> bokeh's ``legend_position``, so the key lands where the static
#: renderer puts it (both read it off ``Panel.legend_corner``).
_BOKEH_LEGEND_POSITION = {
    "upper left": "top_left",
    "upper right": "top_right",
    "lower left": "bottom_left",
    "lower right": "bottom_right",
}


def _series_geometry(*, font_scale: float = 1.0, canvas_factor: float = 1.0, aspect):
    """``(frame_width, frame_height, fontsize)`` for one interactive series panel.

    The line counterpart of :func:`_panel_geometry`, which reads a map's aspect off its
    lon/lat span. A line panel has no such ratio to read — x is time and y is a physical
    quantity — so the family's chosen aspect is passed in instead.
    """
    px = frame_px(aspect, width_px=SERIES_WIDTH_PX * canvas_factor)
    return px[0], px[1], bokeh_fontsize(px, font_scale=font_scale)


def _series_curve(hv, line, dimensions, *, mark: str):
    """Return one line as a bokeh element, marked where the static renderer marks it."""
    from ocean_skill.plot.series import time_values
    from ocean_skill.plot.style import markevery_indices

    values = line.spec.values
    x = time_values(values)
    y = np.asarray(values.values, dtype="float64")
    element = hv.Curve((x, y), *dimensions, label=line.label).opts(
        color=line.color, line_dash=line.line_dash, line_width=1.4
    )
    if mark == "step":
        element = element.opts(interpolation="steps-mid")
    if mark == "marker":
        element = hv.Scatter((x, y), *dimensions, label=line.label).opts(
            color=line.color, marker=line.bokeh_marker or "circle", size=5
        )
    elif line.marker is not None or mark == "line+marker":
        # bokeh's Curve has no `markevery`, so the markers are their own overlay -- on
        # the *same* indices the static renderer uses, or the two would mark different
        # samples of one line.
        keep = markevery_indices(len(x))
        element = element * hv.Scatter((x[keep], y[keep]), *dimensions).opts(
            color=line.color, marker=line.bokeh_marker or "circle", size=5
        )
    return element


def _axis_label_color_hook(panel):
    """Bokeh finalize hook: colour each y axis's label like the lines it scales.

    Bokeh's ``multi_y`` axes are not real twins the way matplotlib's are, so there is
    no per-axis option for this -- only a hook reaching into the rendered figure after
    the fact. Axes are matched to the panel by ``axis_label``, the one identity both
    this hook and the static renderer's own labels share.
    """
    wanted = {
        panel.ylabel: panel.ylabel_color,
        (panel.secondary_ylabel or ""): panel.secondary_ylabel_color,
    }

    def hook(plot, element):
        axes = getattr(getattr(plot, "state", None), "yaxis", None) or ()
        for axis in axes:
            color = wanted.get(axis.axis_label)
            if color:
                axis.axis_label_text_color = color
                axis.major_label_text_color = color

    return hook


def _target_arrows_hook(segments):
    """Bokeh finalize hook: draw one ``Arrow`` annotation per drift segment.

    Bokeh has no arrow glyph an ``hv.Points``/``hv.Segments`` element can render, only
    the ``Arrow`` *annotation* added straight onto a figure -- hence a hook, the same
    escape hatch :func:`_axis_label_color_hook` uses for a bokeh feature holoviews has
    no element-level option for. Being an annotation rather than a glyph, it does not
    interfere with point hover. ``segments`` is ``(x0, y0, x1, y1, color)`` per arrow,
    in the same data coordinates the target diagram's points are drawn in.
    """
    from bokeh.models import Arrow, VeeHead

    def hook(plot, element):
        for x0, y0, x1, y1, color in segments:
            plot.state.add_layout(
                Arrow(
                    end=VeeHead(size=8, fill_color=color, line_color=color),
                    x_start=x0,
                    y_start=y0,
                    x_end=x1,
                    y_end=y1,
                    line_color=color,
                    line_width=1.6,
                    line_alpha=0.85,
                )
            )

    return hook


def _series(
    items,
    title=None,
    rows=None,
    cols=None,
    secondary_y=True,
    encode=None,
    residual=False,
    metrics_loc="auto",
    metric_keys=DEFAULT_METRIC_KEYS,
    legend=True,
    ylim=None,
    panel_aspect=None,
    labels=None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    mark: str = "line",
    **_,
):
    """Draw the ``series`` family interactively — the same layout, drawn with bokeh.

    Composition, styling, labels, titles and the statistics text all come from
    :mod:`ocean_skill.plot.series` and :mod:`ocean_skill.plot.style`, the same as the
    static renderer, so the two cannot disagree about anything but the drawing call.

    Two things differ, both stated rather than silent: the key is drawn inside each
    panel (bokeh has no figure-level legend, so a shared key below the figure is not
    available), and the statistics box is an ``hv.Text`` in data coordinates, so it pans
    and zooms with the data instead of staying pinned to the axes.
    """
    hv = _extension()

    from ocean_skill.plot.series import compose
    from ocean_skill.plot.typography import SERIES_ASPECT

    layout = compose(
        items,
        rows=rows,
        cols=cols,
        secondary_y=secondary_y,
        encode=encode,
        residual=residual,
        metric_keys=metric_keys,
        metrics_loc=metrics_loc,
    )
    width, height, fontsize = _series_geometry(
        font_scale=font_scale,
        canvas_factor=_canvas_factor(size, zoom),
        aspect=panel_aspect or SERIES_ASPECT,
    )
    plots = []
    for panel in layout.panels:
        x_dim = hv.Dimension("time", label="time")
        # label=, never unit=: hv spells `unit` as "name (unit)" where matplotlib writes
        # "name [unit]", and the two renderers must print one axis label, not two.
        y_dim = hv.Dimension("value", label=panel.ylabel)
        overlay = hv.Overlay(
            [_series_curve(hv, line, (x_dim, y_dim), mark=mark) for line in panel.lines]
        )
        if panel.secondary:
            second = hv.Dimension("secondary", label=panel.secondary_ylabel or "")
            overlay = overlay * hv.Overlay(
                [
                    _series_curve(hv, line, (x_dim, second), mark=mark)
                    for line in panel.secondary
                ]
            )
        if panel.metrics_text:
            overlay = overlay * _series_metrics_text(hv, panel)
        plot = overlay.opts(
            hv.opts.Curve(
                frame_width=width,
                frame_height=height,
                fontsize=fontsize,
                show_grid=True,
                tools=["hover"],
            ),
            hv.opts.Overlay(
                title=panel.title,
                show_legend=bool(legend),
                legend_position=_BOKEH_LEGEND_POSITION.get(
                    panel.legend_corner, "top_right"
                ),
                multi_y=bool(panel.secondary),
                fontsize=fontsize,
                hooks=[_axis_label_color_hook(panel)],
            ),
        )
        if ylim is not None:
            plot = plot.opts(hv.opts.Curve(ylim=tuple(ylim)))
        plots.append(plot)
        if panel.residual:
            strip = hv.Overlay(
                [
                    _series_curve(
                        hv,
                        line,
                        (x_dim, hv.Dimension("residual", label="test − reference")),
                        mark=mark,
                    )
                    for line in panel.residual
                ]
            ) * hv.HLine(0.0).opts(color="0.7", line_width=1)
            plots.append(
                strip.opts(
                    hv.opts.Curve(
                        frame_width=width,
                        frame_height=int(height * 0.35),
                        fontsize=fontsize,
                    ),
                    hv.opts.Overlay(show_legend=False, fontsize=fontsize),
                )
            )

    out = hv.Layout(plots).cols(layout.ncols)
    return out.opts(title=title or "")


def _series_metrics_text(hv, panel):
    """Return the statistics box as an ``hv.Text``, in the corner ``compose`` chose.

    Placed in *data* coordinates, which is the one real cost of the interactive form:
    the box pans and zooms with the data rather than staying pinned to the panel. bokeh
    has no axes-fraction annotation, and folding the numbers into the title — what the
    map families do here — is not available either, since the title is reserved for
    identity.
    """
    from ocean_skill.plot.series import time_values

    values = panel.lines[0].spec.values
    x = time_values(values)
    ys = np.concatenate(
        [np.asarray(line.spec.values.values, dtype="float64") for line in panel.lines]
    )
    low, high = float(np.nanmin(ys)), float(np.nanmax(ys))
    vertical, horizontal = panel.metrics_corner.split()
    x_at = x[int(0.02 * (len(x) - 1))] if horizontal == "left" else x[-1]
    y_at = high if vertical == "upper" else low
    return hv.Text(
        x_at,
        y_at,
        panel.metrics_text,
        halign=horizontal,
        valign="top" if vertical == "upper" else "bottom",
    )


def _profile_curve(hv, line, dimensions, *, mark: str):
    """Return one line as a bokeh element: value on x, depth on y.

    The transpose of :func:`_series_curve`, marking the same subsample the same
    way. There is no ``"step"`` branch -- :func:`ocean_skill.plot.matplotlib_
    renderer.profile` refuses that mark outright (a profile's levels are
    irregularly spaced, with nothing between them a step-hold represents
    honestly); an interactive-only caller who passes it anyway simply gets an
    ordinary line, matching every other mark this function does not recognize.
    """
    from ocean_skill.plot.profile import vertical_values
    from ocean_skill.plot.style import markevery_indices

    values = line.spec.values
    x = np.asarray(values.values, dtype="float64")
    y = vertical_values(values)
    element = hv.Curve((x, y), *dimensions, label=line.label).opts(
        color=line.color, line_dash=line.line_dash, line_width=1.4
    )
    if mark == "marker":
        element = hv.Scatter((x, y), *dimensions, label=line.label).opts(
            color=line.color, marker=line.bokeh_marker or "circle", size=5
        )
    elif line.marker is not None or mark == "line+marker":
        keep = markevery_indices(len(x))
        element = element * hv.Scatter((x[keep], y[keep]), *dimensions).opts(
            color=line.color, marker=line.bokeh_marker or "circle", size=5
        )
    return element


def _profile_bands(hv, line, dimensions) -> list:
    """Return one ``hv.Polygons`` per finite run of one line's mean±spread envelope.

    HoloViews' ``hv.Area`` only fills *vertically* (between two y arrays along
    x), and a profile's envelope needs the transpose of that -- value on x,
    depth on y -- so each run is built as its own closed ``hv.Polygons``
    instead, one *element* per :func:`~ocean_skill.plot.style.band_runs` run
    (not one multi-patch element for the whole line): the static renderer
    draws one ``fill_betweenx`` collection per run the same way, which is what
    lets the two renderers' band *count* agree, not just their shapes. Each is
    in the line's own colour and carries no ``label`` -- unlabeled, so it earns
    no legend entry, matching the static renderer's un-legended
    ``fill_betweenx``.

    ``kdims`` must be a *list*, not the ``dimensions`` tuple as given: HoloViews
    reads a bare 2-tuple of dimensions as its own ``(name, label)`` shorthand
    for a single dimension, which is not what this is.
    """
    from ocean_skill.plot.profile import vertical_values
    from ocean_skill.plot.style import BAND_ALPHA, band_runs

    depth = vertical_values(line.spec.values)
    values = np.asarray(line.spec.values.values, dtype="float64")
    bands = []
    for axis, lo, hi in band_runs(depth, values, line.spec.spread):
        xs = np.concatenate([lo, hi[::-1]])
        ys = np.concatenate([axis, axis[::-1]])
        patch = np.column_stack([xs, ys])
        bands.append(
            hv.Polygons([patch], list(dimensions)).opts(
                fill_color=line.color, fill_alpha=BAND_ALPHA, line_alpha=0
            )
        )
    return bands


def _value_range(lines, limit=None) -> tuple[float, float]:
    """A value axis's ``(low, high)``, padded 5% like matplotlib's own default margin.

    HoloViews auto-ranges an ``Overlay``'s x axis over *every* element it contains,
    secondary-axis curves included, so the primary range has to be pinned explicitly
    once a twin axis joins the panel (see :func:`_secondary_x_hook`) -- computed here
    the same way for both axes so neither renderer frames its data more tightly than
    the other.
    """
    if limit is not None:
        return float(limit[0]), float(limit[1])
    arrays = []
    for line in lines:
        values = np.asarray(line.spec.values.values, dtype="float64")
        arrays.append(values)
        if line.spec.spread is not None:
            spread = np.asarray(line.spec.spread, dtype="float64")
            arrays += [values - spread, values + spread]
    xs = np.concatenate(arrays) if arrays else np.array([])
    finite = xs[np.isfinite(xs)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    pad = 0.05 * (hi - lo) or 0.5
    return lo - pad, hi + pad


def _secondary_x_hook(panel, primary_range, secondary_range):
    """Bokeh finalize hook: grow a second x axis along the top of the frame.

    HoloViews' own ``multi_y`` cannot do this even with ``invert_axes=True`` --
    it always creates the extra range under ``extra_y_ranges`` and never adds a
    matching axis to the layout, an orphaned range with nothing drawn against it
    (checked against the installed holoviews/bokeh; a dead path, not a missing
    feature to wait for). This hook does by hand what the static renderer's
    ``ax.twiny()`` does for free: a second ``Range1d`` registered under
    ``extra_x_ranges``, a ``LinearAxis`` added "above" the frame, and every
    secondary glyph re-pointed at it by ``x_range_name``.

    Secondary glyphs are found by *dimension name* -- ``"secondary"``, the one
    identity a line's Curve, its unlabeled markevery Scatter, and its unlabeled
    Polygons bands all carry (labels don't: the marker overlay and the bands
    have none). Any future secondary element built without that kdim would
    silently draw against the primary range instead.
    """
    from bokeh.models import LinearAxis, Range1d

    def hook(plot, element):
        fig = plot.state
        fig.extra_x_ranges = {
            "secondary": Range1d(start=secondary_range[0], end=secondary_range[1])
        }
        top = LinearAxis(
            x_range_name="secondary", axis_label=panel.secondary_xlabel or ""
        )
        if panel.secondary_xlabel_color:
            top.axis_label_text_color = panel.secondary_xlabel_color
            top.major_label_text_color = panel.secondary_xlabel_color
        fig.add_layout(top, "above")
        fig.x_range.start, fig.x_range.end = primary_range
        if panel.xlabel_color:
            # fig.xaxis now holds both the bottom and the top axis just added --
            # matched by label, the same way _axis_label_color_hook matches a
            # series twin's y axes.
            for axis in fig.xaxis:
                if axis.axis_label == (panel.xlabel or ""):
                    axis.axis_label_text_color = panel.xlabel_color
                    axis.major_label_text_color = panel.xlabel_color
        for subplot in plot.subplots.values():
            frame = subplot.current_frame
            if frame is None or not frame.kdims or frame.kdims[0].name != "secondary":
                continue
            glyph = subplot.handles.get("glyph_renderer")
            if glyph is not None:
                glyph.x_range_name = "secondary"

    return hook


def _profile_metrics_text(hv, panel, y_range):
    """Return the statistics box as an ``hv.Text``, in the corner ``compose`` chose.

    The transpose of :func:`_series_metrics_text`: "upper" means shallow (small
    depth, drawn at the top once the axis inverts) and "lower" means deep -- the
    same mapping :func:`ocean_skill.plot.profile._free_corners` uses to rank the
    corners in the first place. ``y_range`` is the whole figure's shared depth
    range (see :func:`_profile`), not this one panel's own lines, so the box lands
    at the panel's actual top/bottom edge even when every panel in the figure
    shares one axis.
    """
    xs = np.concatenate(
        [np.asarray(line.spec.values.values, dtype="float64") for line in panel.lines]
    )
    x_low, x_high = float(np.nanmin(xs)), float(np.nanmax(xs))
    vertical, horizontal = panel.metrics_corner.split()
    x_at = x_low if horizontal == "left" else x_high
    y_shallow, y_deep = min(y_range), max(y_range)
    y_at = y_shallow if vertical == "upper" else y_deep
    return hv.Text(
        x_at,
        y_at,
        panel.metrics_text,
        halign=horizontal,
        valign="top" if vertical == "upper" else "bottom",
    )


def _profile(
    items,
    title=None,
    rows=None,
    cols=None,
    secondary_x=True,
    encode=None,
    metrics_loc="auto",
    metric_keys=DEFAULT_METRIC_KEYS,
    legend=True,
    xlim=None,
    ylim=None,
    panel_aspect=None,
    labels=None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    mark: str = "line",
    **_,
):
    """Draw the ``profile`` family interactively — the same layout, drawn with bokeh.

    Composition, styling, labels, titles and the statistics text all come from
    :mod:`ocean_skill.plot.profile` and :mod:`ocean_skill.plot.style`, the same as
    the static renderer, so the two cannot disagree about anything but the drawing
    call. The depth axis reads surface-at-top, seafloor (or the deepest sample) at
    the bottom, the same way the static renderer's ``ax.set_ylim(deep, shallow)``
    achieves it -- ``ylim=(deep, shallow)`` on the bokeh side, deliberately *not*
    also ``invert_yaxis=True``, which would flip an already-descending range back
    to ascending. One shared depth range across every panel in the figure,
    computed once here exactly as
    :func:`ocean_skill.plot.matplotlib_renderer.profile` computes it, rather than
    left to each panel's own data range.

    A ``secondary_x`` panel's top axis is drawn by :func:`_secondary_x_hook` --
    HoloViews' ``multi_y`` has no x-axis counterpart, and even ``invert_axes=True``
    does not make one (a dead path in the installed holoviews/bokeh, not a missing
    feature). The secondary range is a fixed ``Range1d``, so a box-zoom (which
    acts on ``fig.x_range`` only) does not stretch or shear the twin along with it.
    """
    hv = _extension()

    from ocean_skill.plot.profile import compose, vertical_values
    from ocean_skill.plot.typography import PROFILE_ASPECT

    layout = compose(
        items,
        rows=rows,
        cols=cols,
        secondary_x=secondary_x,
        encode=encode,
        metric_keys=metric_keys,
        metrics_loc=metrics_loc,
    )
    width, height, fontsize = _series_geometry(
        font_scale=font_scale,
        canvas_factor=_canvas_factor(size, zoom),
        aspect=panel_aspect or PROFILE_ASPECT,
    )

    all_lines = [
        line for panel in layout.panels for line in panel.lines + panel.secondary
    ]
    if ylim is not None:
        y_range = (float(ylim[1]), float(ylim[0]))
    elif all_lines:
        depths = np.concatenate(
            [vertical_values(line.spec.values) for line in all_lines]
        )
        finite = depths[np.isfinite(depths)]
        lo, hi = (
            (float(np.nanmin(finite)), float(np.nanmax(finite)))
            if finite.size
            else (0.0, 1.0)
        )
        y_range = (hi, lo) if hi > lo else (lo + 1.0, lo)
    else:
        y_range = (1.0, 0.0)

    plots = []
    for panel in layout.panels:
        x_dim = hv.Dimension("value", label=panel.xlabel or "")
        y_dim = hv.Dimension("depth", label=panel.ylabel)
        dims = (x_dim, y_dim)
        # Bands first, so every line's envelope sits beneath every line -- the
        # same two-pass order the static renderer's fill_betweenx-then-plot
        # draws in, and here it also matters for z-order within the Overlay.
        bands = [
            band
            for line in panel.lines
            if line.spec.spread is not None
            for band in _profile_bands(hv, line, dims)
        ]
        curves = [_profile_curve(hv, line, dims, mark=mark) for line in panel.lines]
        if panel.secondary:
            # "secondary" is the dimension *name* _secondary_x_hook keys on to
            # find which glyphs to re-point at the top axis -- it must match
            # exactly, on every element type a secondary line can produce.
            second_dims = (
                hv.Dimension("secondary", label=panel.secondary_xlabel or ""),
                y_dim,
            )
            bands += [
                band
                for line in panel.secondary
                if line.spec.spread is not None
                for band in _profile_bands(hv, line, second_dims)
            ]
            curves += [
                _profile_curve(hv, line, second_dims, mark=mark)
                for line in panel.secondary
            ]
        overlay = hv.Overlay(bands + curves)
        if panel.metrics_text:
            overlay = overlay * _profile_metrics_text(hv, panel, y_range)
        curve_opts = dict(
            frame_width=width,
            frame_height=height,
            fontsize=fontsize,
            show_grid=True,
            tools=["hover"],
            ylim=y_range,
        )
        if xlim is not None and not panel.secondary:
            # With a twin, xlim reaches the primary axis through the hook's
            # pinned range instead -- applying it as a Curve opt here would
            # clamp the shared pre-hook range under the secondary curves too.
            curve_opts["xlim"] = tuple(xlim)
        overlay_opts = dict(
            title=panel.title,
            show_legend=bool(legend),
            legend_position=_BOKEH_LEGEND_POSITION.get(
                panel.legend_corner, "top_right"
            ),
            fontsize=fontsize,
        )
        if panel.secondary:
            overlay_opts["hooks"] = [
                _secondary_x_hook(
                    panel,
                    _value_range(panel.lines, xlim),
                    _value_range(panel.secondary),
                )
            ]
        plot = overlay.opts(
            hv.opts.Curve(**curve_opts),
            hv.opts.Overlay(**overlay_opts),
        )
        plots.append(plot)

    out = hv.Layout(plots).cols(layout.ncols)
    return out.opts(title=title or "")


def _target(
    items,
    title=None,
    normalize: bool = True,
    circles=None,
    labels="annotate",
    color_by=None,
    marker_by=None,
    groups=None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    colors=None,
    marker_scale: float = 1.0,
    alpha: float | None = None,
    overlay=None,
    overlay_marker_scale: float | dict = 1.8,
    overlay_alpha: float | dict = 1.0,
    summary_points: bool | str = False,
    arrows: bool | str | None = None,
    **_,
):
    """Interactive Target diagram: hover a point for its full metric record.

    ``labels``, ``color_by``, ``marker_by``, ``groups``, ``colors``, ``marker_scale``
    and ``alpha`` mean exactly what they do in :mod:`ocean_skill.plot.summary`, so one
    call renders the same way in either renderer — including the default
    (``"annotate"``), which matches the static target. ``font_scale`` likewise: text is
    sized from the frame by the shared type scale, so the point labels here and on the
    static target are the same size relative to the diagram.

    ``overlay``/``overlay_marker_scale``/``overlay_alpha``/``summary_points`` also mean
    exactly what they do statically — see :func:`ocean_skill.plot.summary.taylor`'s
    docstring for the full explanation. A centroid's marker is drawn as a bokeh
    ``"hex"`` here (the static family's ``"h"`` translated to this renderer's own
    marker vocabulary); everything else about the overlay layer is unchanged.

    ``normalize``/``circles`` mean exactly what they do in
    :func:`ocean_skill.plot.summary.target` — including the mixed-variable and
    mixed-reference warnings, and axis labels named with the shared units when
    ``normalize=False``.

    ``arrows`` means exactly what it does statically — see
    :func:`ocean_skill.plot.summary.target`'s docstring. Bokeh has no arrow glyph, so
    each segment is a separate ``Arrow`` annotation added via a finalize hook (the
    same mechanism the series renderer uses for twin-axis label colour); a chain's
    first point is drawn hollow the same way the static renderer does.
    """
    import pandas as pd

    from ocean_skill.plot.summary import (
        TARGET_FIGSIZE,
        _arrow_chains,
        _overlay_point_specs,
        _resolve_arrows,
        _resolve_colors,
        _resolve_labels,
        _resolve_overlay_style,
        _resolve_per_level,
        _scalar_scale,
        _Styles,
        _summary_point_specs,
        _target_labels,
        _target_rings,
        _target_xy,
        _warn_mixed_variables,
        pretty_level,
    )

    hv = _extension()
    # diagram_scale_factor, not _canvas_factor: this figure is square by construction
    # (its rings have to stay circular), so it *fits inside* the canvas rather than
    # taking its shape. The width-only ratio ignored the canvas height, so size="slide"
    # gave a square that only happened not to overflow a 7.5in slide.
    factor = diagram_scale_factor(TARGET_FIGSIZE, size=size, zoom=zoom)
    # not `frame`: that name is taken below for a per-group slice of the DataFrame
    frame_size = (TARGET_FRAME_PX[0] * factor, TARGET_FRAME_PX[1] * factor)
    sizes = bokeh_scale(frame_size, font_scale=font_scale)
    fontsize = bokeh_fontsize(frame_size, font_scale=font_scale)
    labels_mode = _resolve_labels(labels)
    recs = [
        dict(i.get("metrics", {}), label=i.get("label") or "", units=i.get("units"))
        for i in items
    ]
    df = pd.DataFrame(recs)
    if groups:
        # Mirrors summary._records: keyed by each record's own reference (its
        # catalog source name), falling back to the label for a hand-built table.
        df["group"] = [
            groups.get(rec.get("reference"), groups.get(rec["label"], "other"))
            for rec in recs
        ]
        if not color_by and not marker_by:
            color_by = "group"
    if not normalize:
        _warn_mixed_variables(recs, "target")
    xy = np.array([_target_xy(r, normalize) for r in recs])
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    arrows_field = _resolve_arrows(arrows)
    chains = _arrow_chains(recs, arrows_field) if arrows_field else []
    df["_arrow_start"] = df.index.isin({idxs[0] for idxs in chains})

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
    # asks for. Explicit groups also pin the colours to COLOR_CYCLE by level index, so a
    # diagram keeps its colours when the same call is rendered statically.
    color_levels = list(dict.fromkeys(df[color_dim]))
    grouped_by_marker = marker_by in df.columns
    marker_levels = list(dict.fromkeys(df[marker_by])) if grouped_by_marker else []

    # Mirrors summary._group_styles exactly: colour/alpha/size style by color_dim, the
    # same field the points are already grouped and legended by, so a dict keys the
    # same levels a caller sees in the legend either renderer draws. With no grouping
    # at all color_dim is "label", one level per point in record order, so this
    # reproduces the static per-point styling exactly.
    level_colors = _resolve_colors(colors, color_levels, color_dim)
    level_alphas = _resolve_per_level(
        alpha, color_levels, color_dim, default=None, param="alpha"
    )
    level_scales = _resolve_per_level(
        marker_scale, color_levels, color_dim, default=1.0, param="marker_scale"
    )

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
        for color_level in color_levels:
            frame = df[df[color_dim] == color_level]
            if marker_levels:
                frame = frame[frame[marker_by] == marker_level]
            if frame.empty:
                continue
            level_alpha = level_alphas[color_level]
            alpha_opts = (
                {}
                if level_alpha is None
                else {"fill_alpha": level_alpha, "line_alpha": level_alpha}
            )
            marker = _BOKEH_MARKERS[mi % len(_BOKEH_MARKERS)]
            label = _label(color_level, marker_level)
            # A chain's first point is hollow, exactly like the static renderer --
            # split into two elements (bokeh has no per-point fill toggle) so the
            # legend still gets one filled swatch per group. If a group is *all*
            # chain starts (arrows on a set with no later point to fill), the hollow
            # element carries the legend instead of dropping the group from it.
            rest = frame[~frame["_arrow_start"]] if arrows_field else frame
            starts = frame[frame["_arrow_start"]] if arrows_field else frame.iloc[0:0]
            if not rest.empty:
                elements.append(
                    hv.Points(
                        rest,
                        kdims=["x", "y"],
                        vdims=cols,
                        label=label,
                    ).opts(
                        size=11 * level_scales[color_level],
                        color=level_colors[color_level],
                        marker=marker,
                        tools=["hover"],
                        # Below, matching the static diagrams: target points scatter
                        # around the origin, so a legend inside the frame collides
                        # with the data.
                        legend_position="bottom",
                        show_legend=labels_mode == "legend",
                        line_color="white",
                        line_width=1,
                        **alpha_opts,
                    )
                )
            if not starts.empty:
                elements.append(
                    hv.Points(
                        starts,
                        kdims=["x", "y"],
                        vdims=cols,
                        label=label,
                    ).opts(
                        size=11 * level_scales[color_level],
                        marker=marker,
                        tools=["hover"],
                        legend_position="bottom",
                        show_legend=(labels_mode == "legend") and rest.empty,
                        fill_alpha=0,
                        line_color=level_colors[color_level],
                        line_width=1.5,
                        **(
                            {"line_alpha": level_alpha}
                            if level_alpha is not None
                            else {}
                        ),
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
    rings, boundary = _target_rings(circles, normalize, recs)
    ring_floor = max(rings) * 1.25 if rings else 0.0
    lim = max(
        1.15 * float(np.max(np.hypot(df["x"], df["y"]))),
        ring_floor,
        1.2 if normalize else 0.0,
    )

    theta = np.linspace(0, 2 * np.pi, 181)
    guides = hv.Overlay(
        [
            hv.Path([np.column_stack([rad * np.cos(theta), rad * np.sin(theta)])]).opts(
                color="grey",
                line_dash=(
                    "dashed"
                    if boundary is not None and np.isclose(rad, boundary)
                    else "dotted"
                ),
                line_width=1,
            )
            for rad in rings
        ]
        + [
            hv.HLine(0).opts(color="lightgrey", line_width=1),
            hv.VLine(0).opts(color="lightgrey", line_width=1),
            # the reference sits at the origin, as in the static version. A dict
            # marker_scale has no single level to key the reference on, so it stays
            # unscaled (_scalar_scale's fallback), matching the static reference star.
            hv.Scatter([(0.0, 0.0)]).opts(
                marker="star", size=16 * _scalar_scale(marker_scale), color="black"
            ),
        ]
    )

    # A second, emphasized layer on top of everything -- a highlighted subset
    # (overlay=), a per-group centroid (summary_points=), or both. Each overlay
    # entry is its own small hv.Scatter rather than one grouped hv.Points per
    # (marker_level, color_level) the way the base layer batches -- an overlay is
    # inherently a handful of points, so there is no batching to gain from.
    overlay_specs = []
    if overlay is not None:
        overlay_specs += _overlay_point_specs(
            overlay,
            groups,
            lambda r: _target_xy(r, normalize)[0],
            lambda r: _target_xy(r, normalize)[1],
        )
    if summary_points:
        overlay_specs += _summary_point_specs(
            recs, df["x"].to_numpy(), df["y"].to_numpy(), color_dim, summary_points
        )
    overlay_layer = None
    if overlay_specs:
        marker_map = (
            {
                lev: _BOKEH_MARKERS[i % len(_BOKEH_MARKERS)]
                for i, lev in enumerate(marker_levels)
            }
            if grouped_by_marker
            else {}
        )
        base_styles = _Styles(
            colors=[level_colors[r.get(color_dim)] for r in recs],
            markers=(
                [marker_map[r.get(marker_by)] for r in recs]
                if grouped_by_marker
                else [_BOKEH_MARKERS[0]] * len(recs)
            ),
            alphas=[],
            scales=[],
            handles=[],
        )
        overlay_recs = [spec[2] for spec in overlay_specs]
        overlay_styles = _resolve_overlay_style(
            overlay_recs,
            recs,
            base_styles,
            style_field=color_dim,
            marker_by=marker_by,
            overlay_marker_scale=overlay_marker_scale,
            overlay_alpha=overlay_alpha,
        )
        overlay_elements = []
        for (xi, yi, _rec, mk_override), col, mk, al, scl in zip(
            overlay_specs,
            overlay_styles.colors,
            overlay_styles.markers,
            overlay_styles.alphas,
            overlay_styles.scales,
            strict=True,
        ):
            marker = "hex" if mk_override == "h" else mk
            alpha_opts = {} if al is None else {"fill_alpha": al, "line_alpha": al}
            overlay_elements.append(
                hv.Scatter([(xi, yi)]).opts(
                    size=16 * scl,
                    color=col,
                    marker=marker,
                    line_color="black",
                    line_width=1.2,
                    **alpha_opts,
                )
            )
        overlay_layer = hv.Overlay(overlay_elements)

    result = guides * points
    if overlay_layer is not None:
        result = result * overlay_layer
    xlabel, ylabel = _target_labels(normalize, recs, tex=False)
    opts = dict(
        # equal frame dims + data_aspect keeps the guide circles circular; fixed
        # width/height would fight the aspect and squash them into ellipses
        frame_width=round(frame_size[0]),
        frame_height=round(frame_size[1]),
        data_aspect=1,
        fontsize=fontsize,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        title=title or "Target diagram",
    )
    if chains:
        # Colour follows the *end* point of each segment, same as the static
        # renderer — well-defined even when `arrows` and color_dim name the same
        # field (a colour-graded chain).
        point_colors = [level_colors[v] for v in df[color_dim]]
        xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
        segments = [
            (xs[i], ys[i], xs[j], ys[j], point_colors[j])
            for idxs in chains
            for i, j in pairwise(idxs)
        ]
        opts["hooks"] = [_target_arrows_hook(segments)]
    return result.opts(**opts)


def _locations(
    items,
    *,
    title: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    tiles: str | bool | None = "CartoLight",
    legend: bool = True,
    marker_size: float = 9.0,
    size=None,
    zoom: float = 1.0,
    font_scale: float = 1.0,
    **_,
):
    """Interactive dataset-location map: hoverable markers and extent boxes.

    Items come from :func:`ocean_skill.plot.locations.build_items`; hovering any
    marker (or a grid's dashed extent rectangle) reads that dataset's metadata —
    the :data:`~ocean_skill.plot.locations.HOVER_FIELDS` record, pre-formatted by
    the builder so both renderers agree on it.

    Web tiles are on by default (``"CartoLight"``): a locations map exists to be
    panned and zoomed into, which bare 50m coastlines serve poorly. Pass
    ``tiles=None`` for the offline coastline basemap the other map families
    default to. Extent boxes are plain ``hv.Rectangles`` deliberately —
    ``gv.Rectangles`` + hover crashes in geoviews's bokeh hover handling (its
    ``_process_hover_geo`` assumes two kdims) — so with tiles on, their corners
    (and the ``extent`` limits) are projected to Web Mercator here, since plain
    holoviews elements take no part in geoviews's automatic projection. Lon/lat
    limits on a tiled plot silently blank it (see :func:`_quadmesh`); projecting
    the bounds is the real fix, and lets the map open framed on the datasets
    rather than at world extent.
    """
    import geoviews as gv
    import pandas as pd
    from bokeh.models import HoverTool

    from ocean_skill.plot.locations import FEATURE_TYPE_ORDER, HOVER_FIELDS, style_for

    hv = _extension()
    if tiles is True:
        tiles = "CartoLight"
    tiles = _check_tiles(tiles or None)

    def hover_tool():
        # A fresh HoverTool per element — a bokeh model belongs to one renderer.
        # Explicit tooltips rather than bokeh's defaults, which would append the
        # kdims: raw Web Mercator metres for a rectangle corner under tiles.
        return HoverTool(tooltips=[(f, f"@{{{f}}}") for f in HOVER_FIELDS])

    to_mercator = _to_mercator

    if extent is None:
        from ocean_skill.plot.locations import _default_extent

        extent = _default_extent(items)
    lon0, lat0, lon1, lat1 = (float(v) for v in extent)
    aspect = max(lon1 - lon0, 1e-6) / max(lat1 - lat0, 1e-6)
    px = frame_px(aspect, width_px=SOLO_PANEL_WIDTH_PX * _canvas_factor(size, zoom))
    fontsize = bokeh_fontsize(px, font_scale=font_scale)

    if tiles:
        from geoviews import tile_sources

        overlay = tile_sources.tile_sources[tiles]
    else:
        overlay = gv.feature.coastline.opts(scale="50m")

    def _ordered(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
        ordered = [ft for ft in FEATURE_TYPE_ORDER if ft in groups]
        ordered += [ft for ft in groups if ft not in FEATURE_TYPE_ORDER]
        return ordered

    extent_groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if item["kind"] == "extent":
            extent_groups.setdefault(item["featureType"], []).append(item)
    for feature_type in _ordered(extent_groups):
        style = style_for(feature_type)
        rect_rows = [
            {"lon0": lo, "lat0": la, "lon1": hi, "lat1": ha}
            | {field: item[field] for field in HOVER_FIELDS}
            for item in extent_groups[feature_type]
            for lo, la, hi, ha in item["bboxes"]
        ]
        rect_df = pd.DataFrame(rect_rows)
        if tiles:
            for lon_col, lat_col in (("lon0", "lat0"), ("lon1", "lat1")):
                rect_df[lon_col], rect_df[lat_col] = to_mercator(
                    rect_df[lon_col], rect_df[lat_col]
                )
        overlay = overlay * hv.Rectangles(
            rect_df,
            kdims=["lon0", "lat0", "lon1", "lat1"],
            vdims=list(HOVER_FIELDS),
            label=feature_type,
        ).opts(
            fill_alpha=0,
            line_dash="solid" if style["linestyle"] == "-" else "dashed",
            line_color=style["color"],
            line_width=1.5,
            tools=[hover_tool()],
            show_legend=legend,
        )

    point_groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if item["kind"] == "point":
            point_groups.setdefault(item["featureType"], []).append(item)
    for feature_type in _ordered(point_groups):
        style = style_for(feature_type)
        marker = style["bokeh_marker"] or _BOKEH_MARKERS[
            style["marker_index"] % len(_BOKEH_MARKERS)
        ]
        frame = pd.DataFrame(point_groups[feature_type])[
            ["lon", "lat", *HOVER_FIELDS]
        ]
        # One element per featureType with a fixed colour, as the Target diagram
        # groups its points — it is also what gives the legend one entry per type.
        overlay = overlay * gv.Points(
            frame,
            kdims=["lon", "lat"],
            vdims=list(HOVER_FIELDS),
            label=feature_type,
        ).opts(
            color=style["color"],
            marker=marker,
            size=marker_size,
            tools=[hover_tool()],
            line_color="white",
            line_width=1,
            show_legend=legend,
        )

    # "line" (a selection slice) and "ring" (a domain outline) are plain paths, not
    # geoviews elements -- the same reason the extent rectangles above are plain
    # hv.Rectangles: a geoviews element re-projects its geometry from scratch on
    # every render, and gv + hover crashes for a shape with no point-per-glyph
    # record to show. No hover on these in v1, matching _domain_overlay's own
    # domain ring, for the same reason: there is nothing per-glyph to report.
    path_groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if item["kind"] in ("line", "ring"):
            path_groups.setdefault(item["featureType"], []).append(item)
    for feature_type in _ordered(path_groups):
        style = style_for(feature_type)
        group_items = path_groups[feature_type]
        any_solid = any(item["kind"] == "line" for item in group_items)
        segments = []
        for item in group_items:
            for seg in item["paths"]:
                xs, ys = seg[:, 0].astype(float), seg[:, 1].astype(float)
                if tiles:
                    xs, ys = to_mercator(xs, ys)
                segments.append(np.column_stack([xs, ys]))
        overlay = overlay * hv.Path(segments, label=feature_type).opts(
            color=style["color"],
            line_dash="solid" if any_solid else "dashed",
            line_width=1.8 if any_solid else 1.0,
            show_legend=legend,
        )

    opts: dict[str, Any] = {
        "frame_width": px[0],
        "frame_height": px[1],
        "fontsize": fontsize,
        "title": title or "",
    }
    if legend:
        opts["legend_position"] = "right"
    if tiles:
        xs, ys = to_mercator([lon0, lon1], [lat0, lat1])
        opts["xlim"], opts["ylim"] = (xs[0], xs[1]), (ys[0], ys[1])
    else:
        opts["xlim"], opts["ylim"] = (lon0, lon1), (lat0, lat1)
    return overlay.opts(**opts)


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

    # options the static renderer understands but bokeh has no use for. Split by family
    # because this list is *not* universal: `mark` is meaningless for a map here (bokeh
    # draws a quadmesh either way) and load-bearing for a line panel, which honors
    # mark="line+marker"/"step". Dropped for every family, it would be accepted and
    # silently discarded for the one that implements it -- the exact failure
    # tests/test_renderers.py exists to catch.
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
        "mark",
        "metrics",
        # bokeh labels every panel's own axes and has no notion of borrowing a
        # neighbour's, so there is nothing for this to switch off
        "shared_axis_labels",
        *_STATIC_ONLY_KWARGS,
    ]
    if family in ("series", "profile"):
        # A line panel has no colormap, no map and no fixed axes, so the map-only drops
        # do not apply to it -- and `mark` and `metrics_kwargs` do.
        drops = [d for d in drops if d not in ("mark", "metrics")]
    if family not in _MOVIES:
        # a movie is the only family with something to write here (a standalone HTML
        # page, the interactive counterpart of an mp4) and the only one that plays at a
        # rate; everywhere else both are the static renderer's business
        drops += ["save", "fps", "dpi", "progress"]
    for drop in drops:
        opts.pop(drop, None)

    if family == "field_row":
        # Default the row's overall title to variable · depth · time, exactly as the
        # static field_row does (a single row has no left-edge row label to name the
        # variable — that is field_grid's, and only when it stacks several). Kept in the
        # dispatch, not _field_row, so _field_grid's per-row calls never pick it up and
        # title every row where the grid means to carry one name up top.
        from ocean_skill.plot.matplotlib_renderer import suptitle_text

        item = spec.single
        opts.setdefault(
            "title",
            suptitle_text(
                item.get("standard_name"),
                (item.get("depth"), item.get("time"), item.get("region")),
            ),
        )
        return _field_row(item, **opts)
    if family == "field_grid":
        return _field_grid(spec.items, **opts)
    if family == "field_facet":
        return _field_facet(spec.single, **opts)
    if family == "field_movie":
        return _field_movie(spec.items, **opts)
    if family == "facet_movie":
        return _facet_movie(spec.single, **opts)
    if family == "series":
        return _series(spec.items, **opts)
    if family == "profile":
        if "secondary_y" in opts:
            warnings.warn(
                "'secondary_y' is not an option of profile -- a profile's value "
                "axis is x (depth is y), so its twin is secondary_x. Ignoring it.",
                stacklevel=2,
            )
            opts.pop("secondary_y", None)
        return _profile(spec.items, **opts)
    if family == "section":
        if "domain" in opts:
            warnings.warn(
                "'domain' is not an option of section -- a section has no map to "
                "outline. Ignoring it.",
                stacklevel=2,
            )
            opts.pop("domain", None)
        return _section(spec.single, **opts)
    if family == "section_row":
        if "domain" in opts:
            warnings.warn(
                "'domain' is not an option of section_row -- a section has no map "
                "to outline. Ignoring it.",
                stacklevel=2,
            )
            opts.pop("domain", None)
        return _section_row(spec.single, **opts)
    if family == "skill_map":
        return _skill_map(spec.items, **opts)
    if family == "locations":
        return _locations(spec.items, **opts)
    if family == "target":
        return _target(spec.items, **opts)
    raise NotImplementedError(f"holoviews renderer: family {family!r} not implemented")


register_renderer("holoviews", render)
