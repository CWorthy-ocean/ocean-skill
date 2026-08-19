"""Tests for the dataset-locations map: item building, both renderers, the API."""

from __future__ import annotations

import warnings

import pytest

from ocean_skill.plot.locations import (
    HOVER_FIELDS,
    _default_extent,
    _normalized_geometry,
    build_items,
    map_datasets,
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
    with pytest.raises(ValueError, match="no datasets"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build_items(["unprobed"])


def test_default_extent_min_span_for_a_lone_mooring(index):
    items, extent = build_items(["papa"])
    lon0, lat0, lon1, lat1 = extent
    assert lon1 - lon0 == pytest.approx(10.0)
    assert lat1 - lat0 == pytest.approx(10.0)
    # centred on the mooring
    assert (lon0 + lon1) / 2 == pytest.approx(items[0]["lon"])


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


def test_matplotlib_locations_smoke(index):
    import matplotlib

    matplotlib.use("Agg")
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
    import matplotlib

    matplotlib.use("Agg")
    from ocean_skill.plot.matplotlib_renderer import locations

    items, _ = _items(index)
    extent = (-150.0, 40.0, -130.0, 60.0)
    with pytest.warns(UserWarning, match="tiles"):
        fig = locations(items, extent=extent, tiles="CartoLight")
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


# -- public API ---------------------------------------------------------------------


def test_map_datasets_and_find_map(index):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = map_datasets(["papa", "roms_gulf"])
    assert isinstance(fig, Figure)

    names = index.find(featureType="timeSeries")
    assert type(names).__name__ == "SourceNames"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(names.map(), Figure)
