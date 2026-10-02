"""Tests for drawing ``locations``-family items over a field map.

``Field.plot(locations=...)`` puts what :func:`ocean_skill.map_locations` draws --
markers, transect lines, extent boxes -- on top of a field map, so a bathymetry (or any
other) map can carry the stations and sections being analysed. Three claims carry it,
and each is a way to draw a plausible figure of the wrong thing if it breaks:

* the items land in the right *place* on a 180-centred (dateline-straddling) map --
  the data transform is plain PlateCarree, the axes projection is the one that moves;
* they are context, not data -- a station outside the field must not widen the view;
* the two renderers agree on what they draw.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.plot.locations import HOVER_FIELDS, _seam_split, style_for
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

PLAIN_META = {
    "geospatial_lon_min": -100.0,
    "geospatial_lat_min": -40.0,
    "geospatial_lon_max": -70.0,
    "geospatial_lat_max": 40.0,
    "featureType": "grid",
}


def _hover(name: str) -> dict[str, str]:
    return {f: (name if f == "name" else "") for f in HOVER_FIELDS}


def _map(lon) -> xr.DataArray:
    lat = np.linspace(-40.0, 40.0, 20)
    z = np.add.outer(np.linspace(0.0, 1.0, lat.size), np.linspace(0.0, 1.0, len(lon)))
    return xr.DataArray(
        z, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs={"units": "m"}
    )


#: an ordinary ±180 map, and a dateline-straddling one stored in ±180 (the case a raw
#: ``lon.max() > 180`` test misses, and the Pacmed grid's shape)
PLAIN = _map(np.linspace(-98.0, -80.0, 30))
STRADDLER = _map(
    np.unique(
        np.concatenate([np.linspace(80.0, 180.0, 40), np.linspace(-180.0, -44.0, 30)])
    )
)

PLAIN_ITEMS = [
    {
        **_hover("stn"),
        "kind": "point",
        "featureType": "timeSeries",
        "lon": -90.0,
        "lat": 5.0,
    },
    {
        **_hover("section"),
        "kind": "line",
        "featureType": "selection",
        "paths": [np.array([[-95.0, -10.0], [-85.0, 10.0]])],
    },
    {
        **_hover("box"),
        "kind": "extent",
        "featureType": "grid",
        "bboxes": [(-96.0, -5.0, -88.0, 5.0)],
    },
]

#: a transect from 160E across the dateline to 150W, seam-split exactly as the builder
#: does, a station on the west side of the seam, and a box split into its two halves
STRADDLE_ITEMS = [
    {
        **_hover("Eq"),
        "kind": "line",
        "featureType": "selection",
        "paths": _seam_split(np.array([160.0, 210.0]), np.array([0.0, 0.0])),
    },
    {
        **_hover("SWP"),
        "kind": "point",
        "featureType": "timeSeries",
        "lon": 170.0,
        "lat": -15.0,
    },
    {
        **_hover("box"),
        "kind": "extent",
        "featureType": "grid",
        "bboxes": [(170.0, 5.0, 180.0, 15.0), (-180.0, 5.0, -170.0, 15.0)],
    },
]


def _facet_item(field, facet_dim=None) -> dict:
    return {
        "field": field,
        "facet_dim": facet_dim,
        "row_dim": None,
        "units": "m",
        "standard_name": "sea_floor_depth_below_geoid",
    }


def _static(field, items, **opts):
    return render(
        PlotSpec("field_facet", [_facet_item(field)], {"location_items": items, **opts})
    )


def _map_axes(fig):
    return next(ax for ax in fig.axes if hasattr(ax, "projection"))


def _points(ax):
    from matplotlib.collections import PathCollection

    return [c for c in ax.collections if isinstance(c, PathCollection)]


# -- static renderer ------------------------------------------------------------------


def test_static_overlay_draws_every_kind_with_a_key():
    ax = _map_axes(_static(PLAIN, PLAIN_ITEMS))

    assert len(_points(ax)) == 1
    (selection,) = [ln for ln in ax.lines if ln.get_linestyle() == "-"]
    (box,) = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert selection.get_color() == style_for("selection")["color"]
    assert box.get_color() == style_for("grid")["color"]
    assert sorted(t.get_text() for t in ax.get_legend().get_texts()) == [
        "grid",
        "selection",
        "timeSeries",
    ]


def test_static_no_items_is_the_map_it_always_was():
    for items in (None, []):
        ax = _map_axes(_static(PLAIN, items))
        assert not _points(ax)
        assert ax.get_legend() is None


def test_static_legend_sits_in_a_corner_not_best():
    # loc="best" transforms every mesh cell through cartopy on each layout pass --
    # minutes on a basin grid -- so the key takes a fixed corner instead.
    legend = _map_axes(_static(PLAIN, PLAIN_ITEMS)).get_legend()
    assert legend._loc == legend.codes["upper right"]


def test_static_legend_corner_can_be_overridden():
    legend = _map_axes(
        _static(PLAIN, PLAIN_ITEMS, legend_kwargs={"loc": "lower left"})
    ).get_legend()
    assert legend._loc == legend.codes["lower left"]


def test_static_legend_can_be_switched_off():
    ax = _map_axes(_static(PLAIN, PLAIN_ITEMS, legend=False))
    assert ax.get_legend() is None
    assert len(_points(ax)) == 1  # ... but the items are still drawn


def test_static_overlay_does_not_widen_the_view():
    """A station or transect outside the field is context, not something to frame."""
    far = {
        **_hover("far"),
        "kind": "point",
        "featureType": "profile",
        "lon": 20.0,
        "lat": 0.0,
    }
    far_line = {
        **_hover("far line"),
        "kind": "line",
        "featureType": "selection",
        "paths": [np.array([[0.0, -60.0], [30.0, 60.0]])],
    }
    bare = _map_axes(_static(PLAIN, None)).get_extent()
    over = _map_axes(_static(PLAIN, [far, far_line])).get_extent()
    np.testing.assert_allclose(over, bare)


def test_static_straddling_map_centres_on_180_and_items_land_where_they_belong():
    """The data transform is plain PlateCarree; only the *axes* projection is shifted.

    Reusing the axes projection for the items would put a station at 170E on the far
    side of the planet -- the bug this pins down. Checked in display space: where the
    scatter draws must equal where the axes projection puts 170E, -15.
    """
    ccrs = pytest.importorskip("cartopy.crs")
    fig = _static(STRADDLER, STRADDLE_ITEMS)
    ax = _map_axes(fig)
    assert ax.projection.proj4_params["lon_0"] == 180.0

    fig.canvas.draw()
    (points,) = _points(ax)
    drawn = points.get_offset_transform().transform(points.get_offsets())
    expected = ax.transData.transform(
        ax.projection.transform_point(170.0, -15.0, ccrs.PlateCarree())
    )
    np.testing.assert_allclose(drawn[0], expected, atol=1e-6)

    # the seam-split transect meets in the middle of the frame, with no gap
    ends = sorted(
        (ln.get_xdata()[0], ln.get_xdata()[-1])
        for ln in ax.lines
        if ln.get_linestyle() == "-"
    )
    assert ends == [(-180.0, -150.0), (160.0, 180.0)]


def test_static_facets_draw_on_every_panel_and_key_once():
    time = pd.date_range("2012-01-01", periods=2, freq="MS")
    field = xr.concat([PLAIN, PLAIN + 1.0], dim="time").assign_coords(time=time)
    fig = render(
        PlotSpec(
            "field_facet",
            [_facet_item(field, facet_dim="time")],
            {"location_items": PLAIN_ITEMS},
        )
    )
    panels = [ax for ax in fig.axes if hasattr(ax, "projection")]
    assert len(panels) == 2
    assert all(len(_points(ax)) == 1 for ax in panels)
    assert [ax.get_legend() is not None for ax in panels] == [True, False]


# -- interactive renderer -------------------------------------------------------------


def _hv(field, items, **opts):
    from ocean_skill.plot.holoviews_renderer import _field_facet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _field_facet(
            _facet_item(field), tiles=False, location_items=items, **opts
        )


def _elements(overlay, name):
    return [e for e in overlay if type(e).__name__ == name]


def test_interactive_overlay_draws_every_kind_as_context():
    overlay = _hv(PLAIN, PLAIN_ITEMS)
    for kind in ("Rectangles", "Points", "Path"):
        (element,) = _elements(overlay, kind)
        assert element.opts.get().kwargs["apply_ranges"] is False


def test_interactive_no_items_is_the_map_it_always_was():
    assert [type(e).__name__ for e in _hv(PLAIN, None)] == [
        type(e).__name__ for e in _hv(PLAIN, [])
    ]
    assert not _elements(_hv(PLAIN, None), "Path")


def test_interactive_plain_map_is_not_shifted():
    (path,) = _elements(_hv(PLAIN, PLAIN_ITEMS), "Path")
    (segment,) = path.split()
    assert segment.dimension_values(0).tolist() == [-95.0, -85.0]


def test_interactive_straddling_map_shifts_plain_geometry_but_not_geoviews_points():
    """Plain rectangles and paths are shifted by hand; ``gv.Points`` must not be.

    Rectangles and paths are plain holoviews, so they are moved into the 180-centred
    frame here; ``gv.Points`` is projected by geoviews itself and must be left in
    degrees, or it is moved twice.
    """
    overlay = _hv(STRADDLER, STRADDLE_ITEMS)

    (path,) = _elements(overlay, "Path")
    spans = [
        (float(s.dimension_values(0).min()), float(s.dimension_values(0).max()))
        for s in path.split()
    ]
    assert spans == [(-20.0, 0.0), (0.0, 30.0)]

    (rects,) = _elements(overlay, "Rectangles")
    assert list(zip(rects.dimension_values(0), rects.dimension_values(2))) == [
        (-10.0, 0.0),
        (0.0, 10.0),
    ]

    (points,) = _elements(overlay, "Points")
    assert points.dimension_values(0).tolist() == [170.0]


def test_interactive_transform_follows_the_frame():
    from ocean_skill.plot.holoviews_renderer import (
        _identity_xform,
        _location_xform,
        _shift_to_180_frame,
        _to_mercator,
    )

    assert _location_xform(STRADDLER, None) is _shift_to_180_frame
    assert _location_xform(PLAIN, None) is _identity_xform
    assert _location_xform(PLAIN, "EsriOceanBase") is _to_mercator
    assert _location_xform(STRADDLER, None, geo=False) is _identity_xform


def test_interactive_facets_key_the_first_panel_only():
    time = pd.date_range("2012-01-01", periods=2, freq="MS")
    field = xr.concat([PLAIN, PLAIN + 1.0], dim="time").assign_coords(time=time)
    from ocean_skill.plot.holoviews_renderer import _field_facet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layout = _field_facet(
            _facet_item(field, facet_dim="time"),
            tiles=False,
            location_items=PLAIN_ITEMS,
        )
    keyed = [
        _elements(panel, "Points")[0].opts.get().kwargs["show_legend"]
        for panel in layout
    ]
    assert keyed == [True, False]


# -- Field.plot(locations=...) --------------------------------------------------------


def _resolver(meta_by_name):
    def fake_resolve(source):
        if source in meta_by_name:
            return SimpleNamespace(metadata=meta_by_name[source])
        raise KeyError(source)

    return fake_resolve


@pytest.fixture
def stub(monkeypatch):
    """Swap ``comparison.prepare_source`` for one prepared field, as test_facet does."""
    from ocean_skill import comparison

    def use(field_da):
        monkeypatch.setattr(
            comparison, "prepare_source", lambda *a, **k: (field_da, None)
        )

    return use


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _transect_field():
    from ocean_skill.field import Field

    return Field(
        "grid_src",
        "nitrate",
        select={"transect": {"waypoints": [[-95.0, -10.0], [-85.0, 10.0]]}},
    )


def test_field_plot_draws_a_transect_field_on_the_map(stub):
    """The section you analyse and the line on the map are the same request."""
    from ocean_skill.field import field as make_field

    stub(PLAIN)
    with patch("ocean_skill.catalog.resolve", _resolver({"grid_src": PLAIN_META})):
        fig = make_field("stub", "nitrate").plot(locations=[_transect_field()])

    ax = _map_axes(fig)
    (line,) = [ln for ln in ax.lines if ln.get_linestyle() == "-"]
    assert np.column_stack([line.get_xdata(), line.get_ydata()]).tolist() == [
        [-95.0, -10.0],
        [-85.0, 10.0],
    ]
    # the field map draws its own model outline through ``domain=``; the overlay does
    # not repeat it
    assert not [ln for ln in ax.lines if ln.get_linestyle() == "--"]


def test_field_plot_locations_works_interactively_too(stub):
    from ocean_skill.field import field as make_field

    stub(PLAIN)
    with patch("ocean_skill.catalog.resolve", _resolver({"grid_src": PLAIN_META})):
        obj = make_field("stub", "nitrate").plot(
            renderer="holoviews", tiles=False, locations=[_transect_field()]
        )
    assert _elements(obj, "Path")


def test_field_plot_colors_recolours_the_transect_line_and_its_key(stub):
    from matplotlib.colors import to_rgba

    from ocean_skill.field import field as make_field

    stub(PLAIN)
    with patch("ocean_skill.catalog.resolve", _resolver({"grid_src": PLAIN_META})):
        fig = make_field("stub", "nitrate").plot(
            locations=[_transect_field()], colors="k"
        )

    ax = _map_axes(fig)
    (line,) = [ln for ln in ax.lines if ln.get_linestyle() == "-"]
    assert to_rgba(line.get_color()) == to_rgba("k")
    (handle,) = ax.get_legend().legend_handles
    assert to_rgba(handle.get_color()) == to_rgba("k")


def test_field_plot_colors_works_interactively_too(stub):
    from ocean_skill.field import field as make_field

    stub(PLAIN)
    with patch("ocean_skill.catalog.resolve", _resolver({"grid_src": PLAIN_META})):
        obj = make_field("stub", "nitrate").plot(
            renderer="holoviews",
            tiles=False,
            locations=[_transect_field()],
            colors="navy",
        )
    (path,) = _elements(obj, "Path")
    assert path.opts.get().kwargs["color"] == "navy"


def test_field_plot_locations_is_refused_where_there_is_no_map(stub):
    from ocean_skill.field import field as make_field

    time = pd.date_range("2015-01-01", periods=6, freq="MS")
    series = xr.DataArray(
        np.arange(6.0), dims=("time",), coords={"time": time}, attrs={"units": "m"}
    ).assign_coords(lon=-90.0, lat=25.0)
    stub(series)
    fld = make_field("stub", "nitrate")
    assert fld.family == "series"
    with pytest.raises(ValueError, match=r"locations= draws on a map.*'series'"):
        fld.plot(locations=[_transect_field()])


def test_location_items_names_its_caller_in_the_error():
    from ocean_skill.plot.map_locations import location_items

    with pytest.raises(TypeError, match=r"locations= cannot place 42"):
        location_items([42], who="locations=")
    with pytest.raises(TypeError, match=r"map_locations\(\) cannot place 42"):
        location_items([42])
