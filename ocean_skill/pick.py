"""An interactive lon/lat waypoint picker: the one deliberately kernel-full widget.

Every other interactive plot this package draws (see
:mod:`ocean_skill.plot.holoviews_renderer`) ships as eagerly-embedded, kernel-free
output — that module's ``_frame_map`` explains why a lazy ``DynamicMap`` was tried
and rejected there: a slider's Panel<->kernel comm channel silently breaks, and the
plot just stops moving with no error to say why. :func:`pick_path` is the
deliberate exception, for a reason that rule does not cover: it is an *input*
device, not a rendered figure. It is meaningless without a live kernel by
definition (a clicked point has to reach Python somehow), it is never part of a
saved figure, and its failure mode — clicks that do not register — is immediately
visible, rather than a plot that looks fine and quietly isn't.

**Live Jupyter only. An exported or reopened notebook shows a dead map** — the
same comm dependency ``_frame_map`` warns about, just spent on an input instead of
an output. Deliberately kept OUTSIDE :mod:`ocean_skill.plot.registry` so no
``.plot()`` call can ever reach it by accident, and imports nothing from
:mod:`ocean_skill.plot` at module load time (that package's top level pulls in
matplotlib; this one stays import-light so ``import ocean_skill`` never drags a
plotting stack along for a feature nobody asked for this session)::

    picker = osk.pick_path("pac_dt_ramp")   # last expression of a cell: displays
    # ... click waypoints on the map; the point-draw tool is already active ...
    osk.field("pac_dt_ramp", "temperature", select=picker.as_select()).plot()
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["PathPicker", "pick_path"]

#: One clickable panel's frame, in CSS pixels — matches the reasoning behind
#: ocean_skill.plot.holoviews_renderer.SOLO_PANEL_WIDTH_PX (a lone panel takes the
#: whole width rather than a row's third of it) without importing that module.
_WIDTH_PX, _HEIGHT_PX = 680, 420


def _extension():
    """Activate the bokeh backend once, quietly.

    Duplicates ``ocean_skill.plot.holoviews_renderer._extension``'s few lines
    rather than importing it — see the module docstring on why this file stays
    import-light.
    """
    import holoviews as hv

    if not hv.Store.renderers.get("bokeh"):
        hv.extension("bokeh", logo=False)
    return hv


def _domain_paths(source: str) -> list[np.ndarray]:
    """The domain to draw and click on: catalog metadata only, nothing opened.

    A ladder of three: the catalog's own ``domain_outline`` ring (a curvilinear
    source's true, possibly rotated, perimeter, written at catalog-build time —
    see :func:`ocean_skill.align.perimeter_of`); else the ``geospatial_*``
    bounding box, closed into a rectangle; else a clear error naming what
    metadata would fill the gap. Both re-express the ring/box in whichever
    longitude convention keeps it contiguous (see
    :func:`ocean_skill.comparison._outline_of`'s own ``convention=None``
    default), so a seam-straddling domain (``pac_dt_ramp``, 77°E–316°E) still
    draws as one unbroken piece rather than tearing at ±180.

    ``source`` is resolved directly first, rather than only through
    :func:`~ocean_skill.comparison._outline_of`/:func:`~ocean_skill.comparison.
    _domain_of` (which both swallow a lookup miss into "no outline found"), so
    an unknown or ambiguous name surfaces :func:`ocean_skill.catalog.resolve`'s
    own "Did you mean...?" instead of this function's unhelpful silence.
    """
    from ocean_skill.catalog import resolve
    from ocean_skill.comparison import _domain_of, _outline_of

    resolve(source)  # let an unknown/ambiguous name raise its own clear KeyError

    ring = _outline_of(source)
    if ring is not None:
        return [ring]

    bbox = _domain_of(source)
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        rect = np.array(
            [
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min],
            ]
        )
        return [rect]

    raise ValueError(
        f"{source!r} declares neither a 'domain_outline' ring nor a "
        "geospatial extent ('geospatial_lon_min'/'geospatial_lat_min'/"
        "'geospatial_lon_max'/'geospatial_lat_max') in its catalog metadata, "
        "so there is no domain to click waypoints on. Probe the catalog to "
        "fill an extent in (see ocean_skill.build)."
    )


class PathPicker:
    """A domain map with a click-to-add-waypoints tool, from :func:`pick_path`.

    Displays as the domain outline overlaid with the clickable points — make it
    the last expression of a notebook cell (or call ``display(picker)``) to see
    it. Read :attr:`waypoints` (or call :meth:`as_select`) once you've clicked
    the path you want; drag an existing point to move it, select one and press
    backspace to delete it (the bokeh point-draw tool's own bindings).
    """

    def __init__(self, source: str, overlay: Any, stream: Any):
        self.source = source
        self.overlay = overlay
        self.stream = stream

    @property
    def waypoints(self) -> list[list[float]]:
        """Clicked points, in click order, as ``[[lon, lat], ...]``.

        Longitudes are returned exactly as clicked, in whichever convention
        the domain outline was drawn in — the transect grammar wraps them to
        the model grid's own convention at apply time (see
        :func:`ocean_skill.align.natural_convention`), so no conversion
        happens here.
        """
        data = self.stream.data or {}
        lons, lats = data.get("lon", ()), data.get("lat", ())
        if len(lons) == 0:
            raise RuntimeError(
                "no waypoints yet. Display the picker (make it the last "
                "expression of a cell) and click points on the map -- the "
                "point-draw tool is already the active one. If clicked "
                "points are visible on the map but this still raises, the "
                "widget's comm channel back to the kernel is broken: the "
                "usual causes are hv.extension()/pn.extension() not having "
                "run in this kernel, VS Code's own comm handling, or a "
                "notebook that was exported or reopened with no kernel "
                "behind it -- this map is an input device and is dead "
                "without one by design (see ocean_skill.pick's module "
                "docstring)."
            )
        return [[float(lo), float(la)] for lo, la in zip(lons, lats, strict=True)]

    def as_select(self, *, spacing_km: float | None = None) -> dict[str, Any]:
        """``{"transect": {"waypoints": self.waypoints, ...}}``, ready to hand
        to :func:`ocean_skill.field.field`/:func:`ocean_skill.comparison.compare`
        as (or merged into) ``select=``.
        """
        transect: dict[str, Any] = {"waypoints": self.waypoints}
        if spacing_km is not None:
            transect["spacing_km"] = spacing_km
        return {"transect": transect}

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.overlay._repr_mimebundle_(include, exclude)

    def __repr__(self) -> str:
        try:
            n = len(self.waypoints)
            state = f"{n} waypoint{'' if n == 1 else 's'}"
        except RuntimeError:
            state = "no waypoints yet"
        return f"PathPicker(source={self.source!r}, {state})"


def pick_path(source: str) -> PathPicker:
    """Click transect waypoints on ``source``'s domain, in a live notebook.

    Displays the source's own domain outline (from catalog metadata alone —
    nothing is opened) with a point-draw tool already active; each click adds
    an ordered waypoint. Read :attr:`PathPicker.waypoints` (or call
    :meth:`PathPicker.as_select`) once done, and pass it straight into
    ``select=``::

        picker = osk.pick_path("pac_dt_ramp")
        # ... click a few points along the transect you want ...
        osk.field("pac_dt_ramp", "temperature", select=picker.as_select()).plot()

    **Live Jupyter only.** See this module's docstring for why — unlike every
    other interactive plot this package draws, this one needs a live kernel by
    design, and an exported or reopened notebook shows a dead map.
    """
    hv = _extension()
    paths = _domain_paths(source)
    outline = hv.Path(paths, kdims=["lon", "lat"]).opts(
        color="black", line_dash="dashed", line_width=1
    )
    points = hv.Points([], kdims=["lon", "lat"]).opts(
        size=8, color="crimson", active_tools=["point_draw"]
    )
    stream = hv.streams.PointDraw(source=points)
    overlay = (outline * points).opts(
        frame_width=_WIDTH_PX,
        frame_height=_HEIGHT_PX,
        padding=0.05,
        title=f"click transect waypoints: {source}",
    )
    return PathPicker(source, overlay, stream)
