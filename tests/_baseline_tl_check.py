"""Tests for the dataset-locations map: item building, both renderers, the API."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ocean_skill.plot.locations import (
    GROUP_STYLES,
    HOVER_FIELDS,
    TAB10,
    _default_extent,
    _normalized_geometry,
    _seam_split,
    _split_bbox,
    build_items,
    style_for,
)

# -- fake catalog index (same isolation pattern as tests/test_catalog.py) --------------


def _fake_index(monkeypatch, entries):
    """Point discover() at a hand-built index so nothing touches real catalogs."""
    from ocean_skill import catalog

    refs = {
        name: catalog.SourceRef(name=name, catalog=cat, path=None, metadata=meta)
        for name, (cat, meta) in entries.items()
    }
    monkeypatch.setattr(catalog, "discover", lambda *a, **k: refs)
    return catalog


ROMS_GRID = {  # declared 0-360, ROMS' native convention
    "featureType": "grid",
    "geospatial_lon_min": 262.0,
    "geospatial_lon_max": 282.0,
    "geospatial_lat_min": 18.0,
    "geospatial_lat_max": 31.0,
    "grid_resolution_deg": 0.25,
    "grid_resolution_km": 27.799,
    "grid_regular": True,
    "time_coverage_start": "2012-01-01T00:00:00",
    "time_coverage_end": "2013-01-01T00:00:00",
    "time_resolution_s": 86400.0,
}
PACIFIC_GRID = {  # straddles the anti-meridian
    "featureType": "grid",
    "geospatial_lon_min": 150.0,
    "geospatial_lon_max": 210.0,
    "geospatial_lat_min": -10.0,
    "geospatial_lat_max": 10.0,
}
MOORING = {  # a fixed point: min == max
    "featureType": "timeSeries",
    "geospatial_lon_min": -144.8022,
    "geospatial_lon_max": -144.8022,
    "geospatial_lat_min": 50.069483,
    "geospatial_lat_max": 50.069483,
    "time_resolution_s": 3600.0,
    "geospatial_vertical_min": 0.0,
    "geospatial_vertical_max": 4500.0,
    "vertical_levels": 102,
    "variables": [f"var_{i}" for i in range(8)],
    "institution": "OOI",
    "title": "Station Papa mooring",
}
CLIMATOLOGY = {
    "featureType": "grid",
    "climatology": True,
    "climatology_period": "month01",
    "geospatial_lon_min": -180.0,
    "geospatial_lon_max": 180.0,
    "geospatial_lat_min": -90.0,
    "geospatial_lat_max": 90.0,
}
NO_EXTENT = {"featureType": "timeSeries"}


@pytest.fixture
def index(monkeypatch):
    return _fake_index(
        monkeypatch,
        {
            "roms_gulf": ("GOM run", ROMS_GRID),
            "pacific": ("Pacific run", PACIFIC_GRID),
            "papa": ("OOI Station Papa", MOORING),
            "woa_jan": ("WOA23", CLIMATOLOGY),
            "unprobed": ("OOI Station Papa", NO_EXTENT),
        },
    )


# -- geometry normalisation -------------------------------------------------------


def test_0_360_declaration_wraps_into_pm180():
    geometry = _normalized_geometry(ROMS_GRID)
    assert geometry["bboxes"] == [(-98.0, 18.0, -78.0, 31.0)]
    assert geometry["midpoint"] == (-88.0, 24.5)


def test_antimeridian_extent_splits_at_the_seam():
    geometry = _normalized_geometry(PACIFIC_GRID)
    assert geometry["bboxes"] == [
        (150.0, -10.0, 180.0, 10.0),
        (-180.0, -10.0, -150.0, 10.0),
    ]
    # circular midpoint sits on the seam, not in the Atlantic at (lo+hi)/2
    assert abs(abs(geometry["midpoint"][0]) - 180.0) < 1e-9
    assert geometry["midpoint"][1] == 0.0


def test_global_extent_short_circuits_before_wrapping():
    geometry = _normalized_geometry(CLIMATOLOGY)
    assert geometry["bboxes"] == [(-180.0, -90.0, 180.0, 90.0)]
    assert geometry["midpoint"] == (0.0, 0.0)


def test_missing_extent_is_none():
    assert _normalized_geometry(NO_EXTENT) is None


# -- shared box/path splitting (reused by ocean_skill.plot.map_locations) -----------


def test_split_bbox_ordinary_box_passes_through():
    assert _split_bbox(10.0, 0.0, 20.0, 5.0) == [(10.0, 0.0, 20.0, 5.0)]


def test_split_bbox_declared_0_360_wraps_without_splitting():
    # a box that stays contiguous once wrapped needs no second piece
    assert _split_bbox(262.0, 18.0, 282.0, 31.0) == [(-98.0, 18.0, -78.0, 31.0)]


def test_split_bbox_straddling_splits_at_the_seam():
    assert _split_bbox(150.0, -10.0, 210.0, 10.0) == [
        (150.0, -10.0, 180.0, 10.0),
        (-180.0, -10.0, -150.0, 10.0),
    ]


def test_seam_split_non_straddling_path_is_one_piece():
    lons, lats = [-140.0, -145.0, -150.0], [50.0, 51.0, 52.0]
    pieces = _seam_split(lons, lats)
    assert len(pieces) == 1
    assert pieces[0][:, 0].tolist() == lons
    assert pieces[0][:, 1].tolist() == lats


def test_seam_split_straddling_ring_stays_within_pm180_and_touches_the_seam():
    # a small closed rectangle whose east edge sits past 180 in an unwrapped,
    # contiguous trace (170 -> 190), the shape align.perimeter_of would trace for
    # a grid straddling the antimeridian
    ring_lons = [170.0, 190.0, 190.0, 170.0, 170.0]
    ring_lats = [-10.0, -10.0, 10.0, 10.0, -10.0]
    pieces = _seam_split(ring_lons, ring_lats)
    assert len(pieces) >= 2
    all_lons = [lon for piece in pieces for lon in piece[:, 0]]
    assert all(-180.0 <= lon <= 180.0 for lon in all_lons)
    assert any(abs(lon - 180.0) < 1e-9 for lon in all_lons)
    assert any(abs(lon + 180.0) < 1e-9 for lon in all_lons)
    # no piece silently reintroduces a >180 jump between its own vertices
    for piece in pieces:
        deltas = np.diff(piece[:, 0])
        assert all(abs(d) <= 180.0 for d in deltas)


# -- selection/domain styling and extent framing (shared with map_locations) -------


def test_style_for_selection_and_domain_are_fixed():
    selection = style_for("selection")
    assert selection["color"] == GROUP_STYLES["selection"]["color"] == "crimson"
    assert selection["marker_index"] is None
    domain = style_for("domain")
    assert domain["color"] == GROUP_STYLES["domain"]["color"] == "black"
    assert domain["linestyle"] == "--"


def test_style_for_catalog_featuretype_is_index_based():
    style = style_for("grid")
    assert style["color"] == TAB10[0]
    assert style["marker_index"] == 0
    assert style["marker"] is None  # renderer looks its own marker table up


def test_default_extent_includes_ring_and_line_paths():
    items = [
        {
            "kind": "ring",
            "paths": [
                np.array([[170.0, -10.0], [180.0, -10.0]]),
                np.array([[-180.0, -10.0], [-170.0, 10.0]]),
            ],
        },
        {"kind": "line", "paths": [np.array([[-150.0, 40.0], [-150.0, 60.0]])]},
    ]
    lon0, lat0, lon1, lat1 = _default_extent(items)
    assert lon0 <= -180.0 or lon0 < -170.0  # ring's western piece is included
    assert lat1 >= 60.0 - 1e-9  # line's northern end is included


# -- item building ------------------------------------------------------------------


def test_build_items_kinds_and_hover_fields(index):
    with pytest.warns(UserWarning, match="unprobed"):
        items, extent = build_items()
    by_name = {item["name"]: item for item in items}

    assert by_name["roms_gulf"]["kind"] == "extent"
    assert by_name["papa"]["kind"] == "point"
    assert by_name["papa"]["lon"] == pytest.approx(-144.8022)
    assert by_name["papa"]["lat"] == pytest.approx(50.069483)
    # every hover field is present and a string on every item
    for item in items:
        for field in HOVER_FIELDS:
            assert isinstance(item[field], str)

    assert by_name["papa"]["cadence"] == "hourly"
    assert by_name["papa"]["depth"] == "0–4500 m (102 levels)"
    # eight variables elide past six
    assert by_name["papa"]["variables"].endswith("… (+2)")
    assert by_name["roms_gulf"]["cadence"] == "daily"
    assert by_name["roms_gulf"]["resolution"] == "0.25° (~27.799 km)"
    assert by_name["roms_gulf"]["time_coverage"] == "2012-01-01 → 2013-01-01"
    assert by_name["woa_jan"]["time_coverage"] == "climatology: month01"
    assert "unprobed" not in by_name  # skipped, not mapped

    # the global climatology snaps the default extent to the whole world
    assert extent == (-180.0, -90.0, 180.0, 90.0)


def test_build_items_accepts_names_and_catalog_filter(index):
    items, _ = build_items(["papa"])
    assert [item["name"] for item in items] == ["papa"]

    items, _ = build_items(catalog="*run*")
    assert sorted(item["name"] for item in items) == ["pacific", "roms_gulf"]


def test_build_items_nothing_mappable_raises(index):
    with pytest.raises(ValueError, match="no datasets"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_items(["unprobed"])


def test_default_extent_min_span_for_a_lone_mooring(index):
    items, extent = build_items(["papa"])
    lon0, lat0, lon1, lat1 = extent
    assert lon1 - lon0 == pytest.approx(10.0)
    assert lat1 - lat0 == pytest.approx(10.0)
    # centred on the mooring
    assert (lon0 + lon1) / 2 == pytest.approx(items[0]["lon"])


def test_build_items_resolves_names_against_one_snapshot(index, monkeypatch):
    from ocean_skill import catalog

    real_discover = catalog.discover
    calls = []

    def counting():
        calls.append(1)
        return real_discover()

    monkeypatch.setattr(catalog, "discover", counting)

    items, _ = build_items(["papa", "roms_gulf", "pacific"])
    assert len(calls) == 1  # one discover() for the whole name list, not one per name
    assert sorted(item["name"] for item in items) == ["pacific", "papa", "roms_gulf"]


def test_build_items_qualified_and_unknown_names(index):
    items, _ = build_items(["OOI Station Papa:papa"])
    assert [item["name"] for item in items] == ["papa"]

    with pytest.raises(KeyError):
        build_items(["does_not_exist"])


def test_build_items_surfaces_the_shadow_warning(monkeypatch):
    from pathlib import Path

    from ocean_skill import catalog

    shadowed = catalog.SourceRef(
        name="dupe",
        catalog="B",
        path=Path("B.yaml"),
        metadata=MOORING,
        shadowed_path=Path("A.yaml"),
    )
    monkeypatch.setattr(catalog, "discover", lambda: {"dupe": shadowed})

    with pytest.warns(UserWarning, match="shadows"):
        items, _ = build_items(["dupe"])
    assert [item["name"] for item in items] == ["dupe"]


# -- spec ----------------------------------------------------------------------------


def test_locations_is_a_family():
    from ocean_skill.plot.spec import FAMILIES, PlotSpec

    assert "locations" in FAMILIES
    PlotSpec(family="locations", items=[])  # validates
    with pytest.raises(ValueError):
        PlotSpec(family="location_map")


# -- static renderer -------------------------------------------------------------


def _items(index):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_items()


def _hover_stub(name: str) -> dict[str, str]:
    """A minimal all-string hover record for a hand-built ``locations`` item."""
    return {field: (name if field == "name" else "") for field in HOVER_FIELDS}


def test_matplotlib_locations_draws_ring_and_line_kinds():
    from ocean_skill.plot.matplotlib_renderer import locations

    ring = np.array([[170.0, -10.0], [180.0, -10.0], [180.0, 10.0], [170.0, 10.0]])
    line = np.array([[-150.0, 40.0], [-150.0, 60.0]])
    items = [
        {
            **_hover_stub("model domain"),
            "kind": "ring",
            "paths": [ring],
            "featureType": "domain",
        },
        {
            **_hover_stub("meridional slice"),
            "kind": "line",
            "paths": [line],
            "featureType": "selection",
        },
    ]
    fig = locations(items, extent=(-180.0, -20.0, -120.0, 70.0))
    ax = fig.axes[0]

    labels = sorted(t.get_text() for t in ax.get_legend().get_texts())
    assert labels == ["domain", "selection"]

    solid = [ln for ln in ax.lines if ln.get_linestyle() == "-"]
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(solid) == 1  # the selection slice line, drawn solid
    assert len(dashed) == 1  # the domain ring, drawn dashed
    assert solid[0].get_color() == GROUP_STYLES["selection"]["color"]
    assert dashed[0].get_color() == GROUP_STYLES["domain"]["color"]


def test_matplotlib_locations_mixes_catalog_and_selection_items(index):
    """A selection map's items sit alongside catalog items without disturbing them."""
    from ocean_skill.plot.matplotlib_renderer import locations

    catalog_items, extent = _items(index)
    ring = np.array([[-100.0, 15.0], [-95.0, 15.0], [-95.0, 25.0], [-100.0, 25.0]])
    selection_items = [
        {
            **_hover_stub("model domain"),
            "kind": "ring",
            "paths": [ring],
            "featureType": "domain",
        },
        {
            **_hover_stub("requested point"),
            "kind": "point",
            "lon": -98.0,
            "lat": 20.0,
            "featureType": "selection",
        },
    ]
    fig = locations(catalog_items + selection_items, extent=extent)
    ax = fig.axes[0]
    labels = sorted(t.get_text() for t in ax.get_legend().get_texts())
    assert labels == ["domain", "grid", "selection", "timeSeries"]
    # catalog rings unaffected: still dashed, still one per bbox (roms 1 + pacific
    # split 2 + climatology 1), plus the new domain ring makes 5
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(dashed) == 5


def test_matplotlib_locations_smoke(index):
    from matplotlib.figure import Figure

    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items, extent = _items(index)
    fig = render(
        PlotSpec(family="locations", items=items, options={"extent": extent}),
        renderer="matplotlib",
    )
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["grid", "timeSeries"]
    # one dashed ring per bbox: roms (1) + pacific split (2) + climatology (1)
    rings = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(rings) == 4


def test_matplotlib_locations_extent_and_tiles_warning(index):
    from ocean_skill.plot.matplotlib_renderer import locations

    items, _ = _items(index)
    extent = (-150.0, 40.0, -130.0, 60.0)
    with pytest.warns(UserWarning, match="tiles"):
        fig = locations(items, extent=extent, tiles="EsriOceanBase")
    lon0, lon1, lat0, lat1 = fig.axes[0].get_extent()
    assert (lon0, lat0, lon1, lat1) == pytest.approx(extent)


# -- interactive renderer ----------------------------------------------------------


def _bokeh_hover_tools(obj):
    import holoviews as hv
    from bokeh.models import HoverTool

    return list(hv.render(obj, backend="bokeh").select({"type": HoverTool}))


def test_holoviews_locations_hover_carries_the_metadata(index):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items, extent = _items(index)
    obj = render(
        PlotSpec(family="locations", items=items, options={"extent": extent}),
        renderer="holoviews",
    )
    hovers = _bokeh_hover_tools(obj)
    assert hovers  # points and rectangles both carry one
    for hover in hovers:
        assert [name for name, _ in hover.tooltips] == list(HOVER_FIELDS)


def test_holoviews_locations_tiles(index):
    from ocean_skill.plot.holoviews_renderer import _locations

    items, extent = _items(index)
    with pytest.raises(ValueError, match="unknown tile source"):
        _locations(items, extent=extent, tiles="NotATile")
    # untiled renders with degree limits
    import holoviews as hv

    fig = hv.render(_locations(items, extent=extent, tiles=None), backend="bokeh")
    assert (fig.x_range.start, fig.x_range.end) == (extent[0], extent[2])
    # tiled (the default) projects the limits: Mercator metres, not degrees
    fig = hv.render(_locations(items, extent=extent), backend="bokeh")
    assert abs(fig.x_range.start) > 1000.0


def test_holoviews_locations_default_tiles_avoid_carto(index):
    """Carto's unkeyed tile endpoints now render 'API KEY REQUIRED' watermarks —
    the default source must not silently regress back to one.
    """
    import geoviews as gv

    from ocean_skill.plot.holoviews_renderer import _locations

    items, extent = _items(index)
    overlay = _locations(items, extent=extent)
    wmts = [
        el for el in overlay.traverse(lambda x: x) if isinstance(el, gv.element.WMTS)
    ]
    assert wmts
    assert "carto" not in wmts[0].data.lower()


def test_holoviews_locations_carto_tiles_warn(index):
    from ocean_skill.plot.holoviews_renderer import _locations

    items, extent = _items(index)
    with pytest.warns(UserWarning, match="API key"):
        _locations(items, extent=extent, tiles="CartoLight")


def test_holoviews_locations_warns_on_broken_proj_pairing(index, monkeypatch):
    """A tiled locations map must say so, not just draw the wrong map silently.

    It is exactly the case a broken cartopy/PROJ pairing corrupts (see
    ocean_skill.plot.proj_check).
    """
    import cartopy.crs as ccrs
    import numpy as np

    from ocean_skill.plot import proj_check
    from ocean_skill.plot.holoviews_renderer import _locations

    real = ccrs.GOOGLE_MERCATOR.transform_points

    def broken(src_crs, x, y):
        pts = np.array(real(src_crs, x, y), dtype=float)
        pts[:, 1] += 25_000.0
        return pts

    monkeypatch.setattr(ccrs.GOOGLE_MERCATOR, "transform_points", broken)
    proj_check.projection_skew.cache_clear()
    try:
        items, extent = _items(index)
        with pytest.warns(UserWarning, match="cartopy"):
            _locations(items, extent=extent)
    finally:
        proj_check.projection_skew.cache_clear()


def _selection_map_items():
    """A ring, a line, a point and a box — one of each new/reused kind."""
    ring = np.array([[170.0, -10.0], [180.0, -10.0], [180.0, 10.0], [170.0, 10.0]])
    line = np.array([[-150.0, 40.0], [-150.0, 60.0]])
    return [
        {
            **_hover_stub("model domain"),
            "kind": "ring",
            "paths": [ring],
            "featureType": "domain",
        },
        {
            **_hover_stub("meridional slice"),
            "kind": "line",
            "paths": [line],
            "featureType": "selection",
        },
        {
            **_hover_stub("requested point"),
            "kind": "point",
            "lon": -144.8,
            "lat": 50.07,
            "featureType": "selection",
        },
        {
            **_hover_stub("region"),
            "kind": "extent",
            "featureType": "selection",
            "bboxes": [(-100.0, 15.0, -95.0, 25.0)],
        },
    ]


def test_holoviews_locations_ring_and_line_kinds():
    import holoviews as hv

    from ocean_skill.plot.holoviews_renderer import _locations

    items = _selection_map_items()
    extent = (-180.0, -20.0, -80.0, 70.0)

    # untiled: rendered as a legend-carrying element, in plain degrees
    fig = hv.render(_locations(items, extent=extent, tiles=None), backend="bokeh")
    legend_labels = {it.label["value"] for it in fig.legend[0].items}
    assert {"domain", "selection"} <= legend_labels

    # tiled: paths are hand-projected to Web Mercator, same as the extent boxes
    fig = hv.render(_locations(items, extent=extent), backend="bokeh")
    assert abs(fig.x_range.start) > 1000.0

    # the point and the box hover with the full metadata record; the ring/line
    # paths carry none — there is nothing per-glyph on a path to report
    hovers = _bokeh_hover_tools(_locations(items, extent=extent, tiles=None))
    assert len(hovers) == 2  # rectangles element + points element, not the paths
    for hover in hovers:
        assert [name for name, _ in hover.tooltips] == list(HOVER_FIELDS)


# -- public API ---------------------------------------------------------------------


def test_map_locations_by_name_and_find_map(index):
    from matplotlib.figure import Figure

    from ocean_skill.plot.map_locations import map_locations

    fig = map_locations(["papa", "roms_gulf"])
    assert isinstance(fig, Figure)

    names = index.find(featureType="timeSeries")
    assert type(names).__name__ == "SourceNames"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(names.map(), Figure)
