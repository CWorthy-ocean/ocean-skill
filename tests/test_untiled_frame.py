"""An untiled interactive map draws everything in one frame, in degrees.

hvplot pins a plot ``projection`` only when one is passed, or on the coastline
``Feature`` it attaches; without either, geoviews' bokeh plot defaults to Web Mercator.
So an untiled panel that carried no ``Feature`` (``tiles=False`` with ``land=False``, or
any untiled movie, whose coastline is a hand-built path) rendered its mesh in Mercator
metres while every plain-holoviews overlay -- the offline coastline, the domain ring,
the station dots, a ``skill_map`` extent window -- was drawn in degrees and landed near
(0, 0), off the map. Nothing raised; the figure just came back with the field in one
place and its context somewhere else.

The renderer now pins the frame itself: an untiled geo panel is always PlateCarree
degrees (180-centred, ``x = (lon % 360) - 180``, for a dateline-straddling field), no
interactive map carries a geoviews ``Feature``, and every plain overlay is moved *into*
that frame -- a 0-360 ring over a mesh geoviews wrapped into +-180, an ``xlim`` over a
straddling frame -- with ``apply_ranges=False`` so the mesh alone sets the view.

What these tests read is the bokeh figure itself, not the holoviews objects that built
it: the regression is a disagreement between two *rendered* coordinate systems, which no
assertion about one element can see. Every helper below resolves a glyph's x data
through its field specs into the column data source, exactly as bokeh's JS would, and
:func:`_assert_one_frame` is the whole claim -- the view is in degrees and every glyph
(coast, ring, station dot, marker) lies on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

pytest.importorskip("geoviews")
pytest.importorskip("cartopy")

# Every test renders interactive maps through geoviews/bokeh, several of them movies.
# Skipped by default (`pytest.ini`'s `-m "not slow"`); run with `pytest -m ""`.
pytestmark = pytest.mark.slow

NITRATE = "mole_concentration_of_nitrate_in_sea_water"

#: A model-domain box spelled in 0-360 on purpose: the figures below draw fields in
#: -100..-90 (or 260..270), so the ring only lands on the map if the renderer wraps it
#: into the frame the mesh is actually in.
DOMAIN = (261.0, 19.0, 269.0, 25.0)

#: Degrees of slack when comparing a glyph's span to the view or to another glyph's.
TOL = 1e-6


# --- builders ------------------------------------------------------------------------


def _grid(lon0, lon1, lat0=18.0, lat1=26.0, *, offset=0.0, ny=16, nx=20):
    """Build one tiny regular lat/lon field, small enough that nothing rasterizes."""
    return xr.DataArray(
        offset + np.linspace(-1.0, 1.0, ny * nx).reshape(ny, nx),
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(lat0, lat1, ny),
            "lon": np.linspace(lon0, lon1, nx),
        },
        attrs={"units": "mmol m-3"},
    )


def _row_item(lon0=-100.0, lon1=-90.0, **over):
    """One ``field_row`` item, shaped as ``Comparison.as_item()`` builds it."""
    test, ref = _grid(lon0, lon1, offset=1.0), _grid(lon0, lon1)
    return {
        "aligned": {"test": test, "reference": ref, "difference": test - ref},
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "depth": "0-10 m",
        "time": "2010-01-22",
        "metrics": {"bias": 0.125, "rmse": 0.5, "corr": 0.98},
        "labels": ("GOM_bgc", "woa23_nitrate"),
        **over,
    }


def _facet_item(lon0=-100.0, lon1=-90.0, lat0=18.0, lat1=26.0, *, days=3):
    """One ``field_facet`` / ``facet_movie`` item, shaped as ``Field.as_item()``."""
    ny, nx = 10, 20
    rng = np.random.default_rng(0)
    field = xr.DataArray(
        rng.normal(5.0, 1.0, (days, ny, nx)).astype("float32"),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2012-01-01", periods=days, freq="D"),
            "lat": np.linspace(lat0, lat1, ny),
            "lon": np.linspace(lon0, lon1, nx),
        },
        attrs={"units": "mmol m-3"},
    )
    return {
        "field": field,
        "facet_dim": "time",
        "row_dim": None,
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "label": "GOM_bgc",
    }


def _map_item(label, value, lon0=-100.0, lon1=-90.0):
    """One ``field_map_grid`` member: a single reduced-to-one-instant map."""
    field = _grid(lon0, lon1) * 0 + value
    field.name = NITRATE
    return {
        "field": field,
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "label": label,
        "time": None,
    }


def _skill_item(lon0=-100.0, lon1=-90.0, lat0=18.0, lat1=26.0, *, stations=True):
    """One ``skill_map`` item, with station dots spelled like the field's longitudes."""
    bias = _grid(lon0, lon1, lat0, lat1)
    rmse = _grid(lon0, lon1, lat0, lat1) * 0.5 + 1.0
    item = {
        "skill": xr.Dataset({"bias": bias, "rmse": rmse}),
        "metric_names": ("bias", "rmse"),
    }
    if stations:
        span_lon, span_lat = lon1 - lon0, lat1 - lat0
        item["stations"] = {
            "lon": np.array([lon0 + 0.3 * span_lon, lon0 + 0.7 * span_lon]),
            "lat": np.array([lat0 + 0.3 * span_lat, lat0 + 0.7 * span_lat]),
            "values": {"bias": np.array([0.4, -0.3])},
            "names": ["st_a", "st_b"],
        }
    return item


def _movie_frame(index):
    """One ``field_movie`` frame, shaped as ``ComparisonSet.movie()`` builds it."""
    test, ref = _grid(-100.0, -90.0, offset=1.0 + 0.5 * index), _grid(-100.0, -90.0)
    return {
        "aligned": {"test": test, "reference": ref, "difference": test - ref},
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "metrics": {"bias": 0.1 * index, "rmse": 0.5, "corr": 0.98},
        "labels": ("GOM_bgc", "woa23_nitrate"),
        "frame_label": f"2012-01-{index + 1:02d}",
    }


def _draw(family, items, **options):
    """Render ``items`` through the public path, interactively."""
    return render(
        PlotSpec(family=family, items=items, options=options), renderer="holoviews"
    )


# --- reading a rendered bokeh figure -------------------------------------------------


def _figures(obj):
    """Bokeh-render ``obj`` and return every ``Plot`` in it, in document order."""
    import holoviews as hv
    from bokeh.models import Plot

    root = hv.render(obj, backend="bokeh")
    return [root] if isinstance(root, Plot) else list(root.select({"type": Plot}))


def _floats(values):
    """Flatten a column (an array, or lists of arrays with NaN breaks) to one array."""
    if isinstance(values, (list, tuple)) or (
        isinstance(values, np.ndarray) and values.dtype == object
    ):
        parts = [_floats(v) for v in values]
        return np.concatenate(parts) if parts else np.array([])
    return np.asarray(values, dtype="float64").ravel()


def _column(renderer, attr):
    """Return the data behind one glyph property, or ``None`` if it has none.

    A bokeh property is a field spec: a plain column name (``"xs"``), a dict or object
    carrying a ``field``, or a constant (``value``) that names no column at all.
    """
    spec = getattr(renderer.glyph, attr, None)
    if isinstance(spec, dict):
        field, value = spec.get("field"), spec.get("value")
    else:
        field = spec if isinstance(spec, str) else getattr(spec, "field", None)
        value = None if isinstance(spec, str) else getattr(spec, "value", None)
    data = renderer.data_source.data
    if field is not None and field in data:
        return _floats(data[field])
    if isinstance(value, (int, float)):
        return np.array([float(value)])
    return None


def _x_span(renderer):
    """Return the finite ``(min, max)`` x of one glyph renderer, or ``None``.

    ``None`` for a glyph with no x data to read (a hline, a label) and for one whose
    x data is all NaN -- an empty clip is not a glyph sitting somewhere wrong.
    """
    kind = type(renderer.glyph).__name__
    if kind in ("MultiLine", "Patches"):
        columns = [_column(renderer, "xs")]
    elif kind == "Quad":
        columns = [_column(renderer, "left"), _column(renderer, "right")]
    elif kind == "Image":
        x, dw = _column(renderer, "x"), _column(renderer, "dw")
        columns = [x] if x is None or dw is None else [x, x + dw]
    else:
        columns = [_column(renderer, "x")]
    parts = [c for c in columns if c is not None]
    if not parts:
        return None
    xs = np.concatenate(parts)
    xs = xs[np.isfinite(xs)]
    return (float(xs.min()), float(xs.max())) if xs.size else None


def _is_ring(renderer):
    """Return whether ``renderer`` draws the dashed domain ring."""
    if type(renderer.glyph).__name__ != "MultiLine":
        return False
    dash = renderer.glyph.line_dash
    return dash != "solid" if isinstance(dash, str) else bool(dash)


def _rings(fig):
    return [r for r in fig.renderers if _is_ring(r)]


def _coasts(fig):
    """Return the solid-line ``MultiLine`` renderers: the coastline."""
    return [
        r
        for r in fig.renderers
        if type(r.glyph).__name__ == "MultiLine" and not _is_ring(r)
    ]


def _scatters(fig):
    return [
        r for r in fig.renderers if type(r.glyph).__name__ in ("Scatter", "Circle")
    ]


def _mesh_span(fig):
    """Return the ``(min, max)`` x of the field itself (a mesh, or its image)."""
    spans = [
        _x_span(r)
        for r in fig.renderers
        if type(r.glyph).__name__ in ("Patches", "Image")
    ]
    spans = [s for s in spans if s is not None]
    assert spans, "no field mesh in the figure"
    return min(s[0] for s in spans), max(s[1] for s in spans)


def _assert_one_frame(fig):
    """Assert the view is in degrees and every glyph of the figure lies on it.

    The whole claim in one place: nothing is left in another coordinate system. A
    glyph whose span misses the view entirely was drawn in a different frame; a station
    dot or ring that is merely *partly* outside is as wrong, so those two must sit
    inside it.
    """
    start, end = float(fig.x_range.start), float(fig.x_range.end)
    assert np.isfinite([start, end]).all(), f"x_range=({start}, {end})"
    assert abs(start) <= 360.0 and abs(end) <= 360.0, (
        f"the view is not in degrees: x_range=({start}, {end})"
    )
    for renderer in fig.renderers:
        span = _x_span(renderer)
        if span is None:
            continue
        kind = type(renderer.glyph).__name__
        assert span[0] <= end + TOL and span[1] >= start - TOL, (
            f"{kind} spans x={span}, off the view ({start}, {end})"
        )
        if kind in ("Scatter", "Circle") or _is_ring(renderer):
            assert start - TOL <= span[0] and span[1] <= end + TOL, (
                f"{kind} spans x={span}, not inside the view ({start}, {end})"
            )


# --- skill_map: the repro ------------------------------------------------------------


@pytest.mark.parametrize("land", [True, False], ids=["land", "no-land"])
@pytest.mark.parametrize(
    "lon",
    [(-100.0, -90.0), (260.0, 270.0)],
    ids=["spelled-pm180", "spelled-0-360"],
)
def test_an_untiled_skill_map_draws_everything_in_one_frame(land, lon):
    """The bug: a coast-less untiled skill_map came back in Web Mercator metres.

    Without the coastline ``Feature`` (``land=False``) nothing pinned the projection,
    so the mesh sat at x ~ -1.1e7 while the station dots and the domain ring, plain
    degree geometry, sat near the origin. A 0-360 spelling is the second half: the mesh
    is wrapped to -100..-90 by geoviews, but the ring (261..269) and the dots (263, 267)
    stayed at their raw longitudes, a whole turn away from the field.
    """
    item = _skill_item(*lon, 18.0, 26.0)
    figs = _figures(
        _draw("skill_map", [item], tiles=False, land=land, domain=DOMAIN)
    )
    assert len(figs) == 2  # bias and rmse
    for fig in figs:
        _assert_one_frame(fig)
        assert _rings(fig), "the domain ring is missing"
        if land:
            coasts = _coasts(fig)
            assert coasts, "land=True drew no coastline"
            assert any(_x_span(c) is not None for c in coasts), "the coastline is empty"
    # only the bias panel has station values
    assert sum(len(_scatters(fig)) for fig in figs) == 1


# --- the other still families --------------------------------------------------------


def _family_items(family):
    if family == "field_row":
        return [_row_item()]
    if family == "field_facet":
        return [_facet_item()]
    return [_map_item("run_a", 5.0), _map_item("run_b", 6.0)]


@pytest.mark.parametrize("family", ["field_row", "field_facet", "field_map_grid"])
def test_an_untiled_still_map_keeps_its_domain_ring_on_the_field(family):
    """The same frame bug, in every other still family that can draw a domain ring.

    ``tiles=False`` with ``land=False`` is the bare case that lost its projection; the
    0-360 ring has to be wrapped into the -100..-90 frame of every panel.
    """
    figs = _figures(
        _draw(family, _family_items(family), tiles=False, land=False, domain=DOMAIN)
    )
    assert figs
    for fig in figs:
        _assert_one_frame(fig)
        assert _rings(fig), f"a {family} panel has no domain ring"


def test_a_straddling_untiled_facet_lands_on_the_180_centred_frame():
    """A dateline-straddling field is framed 180-centred, and so are its overlays.

    The field spans 150..250 (plot x -30..70 in the 180-centred frame). The coastline
    and the ring (160..240, written in 0-360) both have to be carried to that frame:
    left at raw longitudes they sit past 100 degrees from the mesh, and the view -- if
    the frame were the world -- would run the whole -180..180.
    """
    item = _facet_item(150.0, 250.0, -15.0, 15.0)
    figs = _figures(
        _draw(
            "field_facet",
            [item],
            tiles=False,
            land=True,
            domain=(160.0, -10.0, 240.0, 10.0),
        )
    )
    assert len(figs) == 3
    for fig in figs:
        _assert_one_frame(fig)
        start, end = float(fig.x_range.start), float(fig.x_range.end)
        assert -40.0 <= start
        assert end <= 80.0, f"the world, not the basin: x_range=({start}, {end})"
        mesh = _mesh_span(fig)
        coasts = [s for s in map(_x_span, _coasts(fig)) if s is not None]
        assert coasts, "the coastline is missing or empty"
        assert any(s[0] <= mesh[1] and s[1] >= mesh[0] for s in coasts), (
            f"the coastline {coasts} misses the mesh {mesh}"
        )
        rings = [_x_span(r) for r in _rings(fig)]
        assert len(rings) == 1, "the domain ring is missing"
        assert mesh[0] - TOL <= rings[0][0] and rings[0][1] <= mesh[1] + TOL


# --- skill_map's extent window -------------------------------------------------------


@pytest.mark.parametrize(
    ("lon", "lat", "box", "centre"),
    [
        ((-100.0, -90.0), (18.0, 26.0), (-98.0, -92.0, 19.0, 25.0), -95.0),
        ((260.0, 270.0), (18.0, 26.0), (262.0, 268.0, 19.0, 25.0), -95.0),
        ((150.0, 250.0), (-15.0, 15.0), (160.0, 240.0, -10.0, 10.0), 20.0),
    ],
    ids=["pm180", "0-360", "straddling"],
)
def test_an_untiled_skill_map_extent_is_moved_into_the_frame(lon, lat, box, centre):
    """``extent=`` frames the window in the panel's own coordinates, not raw degrees.

    An ``xlim`` is taken verbatim in the plot's frame, so a box spelled 0-360 missed a
    mesh wrapped into +-180, and a box on a straddling field missed its 180-centred
    frame (160..240 is plot x -20..60: centre 20, not 200). One shift moves both ends.
    Bokeh pads one axis to keep the aspect, but symmetrically, so the *centre* is the
    box's exactly.
    """
    item = _skill_item(*lon, *lat)
    (fig, *_) = _figures(
        _draw("skill_map", [item], tiles=False, land=False, extent=box)
    )
    xc = (fig.x_range.start + fig.x_range.end) / 2
    yc = (fig.y_range.start + fig.y_range.end) / 2
    assert xc == pytest.approx(centre)
    assert yc == pytest.approx((box[2] + box[3]) / 2)


# --- untiled movies ------------------------------------------------------------------


def _first_frames(movie):
    """Return the first frame of every ``HoloMap`` a movie object carries."""
    import holoviews as hv

    obj = getattr(movie, "object", movie)  # panel wraps it for its slider widget
    maps = [el for el in obj.traverse() if isinstance(el, hv.HoloMap)]
    return [hm[hm.kdims[0].values[0]] for hm in maps]


@pytest.mark.parametrize("land", [True, False], ids=["land", "no-land"])
@pytest.mark.parametrize("family", ["field_movie", "facet_movie"])
def test_an_untiled_movie_frame_is_in_degrees(family, land):
    """The headline repro: an untiled movie's mesh was Mercator, its coast degrees.

    A movie draws its coastline as a once-built path and gives hvplot
    ``coastline=False`` per frame, so *no* movie frame carried the ``Feature`` that
    pinned the projection.
    Over lon -100..-90 the view was x ~ -1.1e7 .. -9.9e6 metres while the coastline
    spanned -105..-85 -- the field and its own coast on opposite sides of the page.
    """
    if family == "field_movie":
        items = [_movie_frame(i) for i in range(3)]
    else:
        items = [_facet_item()]
    frames = _first_frames(_draw(family, items, tiles=False, land=land, domain=DOMAIN))
    assert frames
    for frame in frames:
        (fig,) = _figures(frame)
        _assert_one_frame(fig)
        assert _rings(fig), "the domain ring is missing"
        if land:
            coasts = [s for s in map(_x_span, _coasts(fig)) if s is not None]
            assert coasts, "land=True drew no coastline"
            assert all(-180.0 <= s[0] and s[1] <= 180.0 for s in coasts)


# --- _locations ----------------------------------------------------------------------


@pytest.mark.parametrize("land", [True, False], ids=["land", "no-land"])
def test_an_untiled_locations_map_pins_degrees(land):
    """``_locations(tiles=None)`` was Mercator with degree limits: markers at ~-1e7 m.

    The limits were plain degrees and so were the extent boxes, but the gv.Points
    markers are projected by geoviews into whatever frame the plot has -- Web Mercator
    unless one is pinned -- so the point landed at -1.06e7 m, far from its own box.
    """
    from ocean_skill.plot.holoviews_renderer import _locations
    from ocean_skill.plot.locations import HOVER_FIELDS

    hover = {field: "x" for field in HOVER_FIELDS}
    items = [
        {
            **hover,
            "kind": "point",
            "featureType": "timeSeries",
            "lon": -95.0,
            "lat": 22.0,
        },
        {
            **hover,
            "kind": "extent",
            "featureType": "grid",
            "bboxes": [(-98.0, 19.0, -92.0, 24.0)],
        },
    ]
    (fig,) = _figures(
        _locations(items, extent=(-100.0, 18.0, -90.0, 26.0), tiles=None, land=land)
    )
    assert (float(fig.x_range.start), float(fig.x_range.end)) == pytest.approx(
        (-100.0, -90.0)
    )
    _assert_one_frame(fig)
    (marker,) = _scatters(fig)
    assert _x_span(marker) == pytest.approx((-95.0, -95.0))


# --- the guard: no geoviews Feature, one shared coastline ----------------------------


@pytest.mark.parametrize(
    "family",
    ["field_row", "field_grid", "field_facet", "field_map_grid", "skill_map"],
)
def test_no_untiled_interactive_map_carries_a_geoviews_feature(family):
    """A ``Feature`` is what used to pin the projection, and it re-projects per render.

    Every untiled map now draws its coast as a plain path instead; a ``Feature``
    creeping back in would mean the projection is being pinned (or not) by accident
    again, and that bokeh re-projects the world's coastline once per panel.
    """
    items = {
        "field_row": lambda: [_row_item()],
        "field_grid": lambda: [
            _row_item(row_label="nitrate"),
            _row_item(row_label="phosphate", labels=("GOM_bgc", "woa23_phosphate")),
        ],
        "field_facet": lambda: [_facet_item()],
        "field_map_grid": lambda: _family_items("field_map_grid"),
        "skill_map": lambda: [_skill_item()],
    }[family]()
    obj = _draw(family, items, tiles=False)
    kinds = [type(node).__name__ for node in obj.traverse()]
    assert "Feature" not in kinds, kinds


def test_every_panel_of_a_row_shares_one_coastline_path():
    """Three panels, one coastline: the path is built once and overlaid on each.

    Without a domain the only ``Path`` in a row is the coast, so the distinct objects
    among them are the number of times it was built -- which must not scale with the
    panel count, since that cost (a clip of the world's coastline) is what a per-panel
    ``Feature`` made every untiled figure pay.
    """
    import holoviews as hv

    obj = _draw("field_row", [_row_item()], tiles=False, domain=None)
    paths = [node for node in obj.traverse() if isinstance(node, hv.Path)]
    assert len(paths) == 3, "one coastline per panel, three panels"
    assert len({id(p) for p in paths}) == 1, "each panel built its own coastline"
