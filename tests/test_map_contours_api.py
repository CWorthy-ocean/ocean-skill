"""``contours:`` on a map page: a second variable's isolines over a suite's map figure.

The suite path of the map overlay (the Python API's own pairing is in
``test_contour_pairing.py``): a ``field:`` or ``compare:`` page whose figure is a map
takes ``contours: true`` (the plotted object's own lines) or a mapping (a second object
built the same way, with the mapping's keys overriding the page's own). Two layers:

* the fake-object layer (``_fake_osk``-style stubs) checks what ``build`` hands to
  ``plot()`` for a map family -- nothing about it is section-specific;
* the real-object layer runs ``expand`` + ``build`` against the real field/comparison
  machinery on small synthetic sources and captures the :class:`PlotSpec` the renderer
  registry is handed, to see the contour keys land on the map items.

The ``end_to_end`` cases draw the figure itself; they need both renderers' map-contour
support and are the ones to run after the renderers land.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog
from ocean_skill.suites import pages as P
from tests.test_combine import captured  # noqa: F401  (fixture)
from tests.test_section_comparison import (  # noqa: F401  (patched_sources: fixture)
    HC,
    _climatology,
    _roms_run,
    patched_sources,
)
from tests.test_suites_pages import (  # noqa: F401  (fixtures + helpers)
    INDEX,
    _fake_field,
    _fake_osk,
    _suite,
    _time_index,
)

VAR = "sea_water_potential_temperature"
LINES = "sea_water_salinity"


def _map_dataset() -> xr.Dataset:
    """Return a small three-month climatology: two variables, two levels."""
    time = INDEX[:3]
    lon = np.arange(180.0, 201.0, 5.0)
    lat = np.arange(-10.0, 11.0, 5.0)
    depth = np.array([0.0, 250.0])
    t, d, la, lo = np.meshgrid(np.arange(3), depth, lat, lon, indexing="ij")
    coords = {
        "time": time,
        "depth": ("depth", depth, {"positive": "down"}),
        "lat": lat,
        "lon": lon,
    }
    dims = ("time", "depth", "lat", "lon")
    temp = 25.0 - d / 50.0 + la / 10.0 + t
    salt = 34.0 + d / 500.0 + lo / 100.0
    return xr.Dataset(
        {
            VAR: (dims, temp, {"units": "degC"}),
            LINES: (dims, salt, {"units": "1"}),
        },
        coords=coords,
    )


@pytest.fixture
def patched_map_source(monkeypatch):
    ds = _map_dataset()
    monkeypatch.setattr(osk, "read", lambda name, **kw: ds)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata={}))
    return "clim_x"


def _field_page(contours, **extra):
    return {
        "title": "map",
        "field": {
            "variables": [VAR],
            "select": {"depth": 0.0},
            "aggregate": {"time": "mean"},
        },
        "contours": contours,
        **extra,
    }


# -- fake objects: what build() hands plot() for a map family -----------------


def test_build_compare_contours_true_on_a_map_row_passes_the_bucket(monkeypatch):
    calls = _fake_osk(monkeypatch, test_families=["field_row", "field_row"])
    page = {
        "title": "x",
        "compare": {
            "reference": ["woa"],
            "variables": ["temperature"],
            "select": {"test": {"depth": 0}, "reference": {}},
        },
        "contours": True,
        "plot": {"contour_levels": [0]},
    }
    (p,) = P.expand(_suite([page], defaults={"test": "stub"}))
    figs = P.build(p)
    assert [s for s, _ in figs] == [""]
    ((bucket, opts),) = calls["plot"]
    assert opts["contours"] is bucket and opts["contour_levels"] == [0]


def test_build_compare_contours_mapping_pairs_map_rows_by_family(monkeypatch):
    calls = _fake_osk(
        monkeypatch,
        test_families=["field_row", "field_row"],
        contour_families=["field_row", "field_row"],
    )
    page = {
        "title": "x",
        "compare": {
            "reference": ["woa"],
            "variables": ["temperature"],
            "select": {"test": {"depth": 0}, "reference": {}},
        },
        "contours": {"variables": ["salinity"]},
    }
    (p,) = P.expand(_suite([page], defaults={"test": "stub"}))
    P.build(p)
    first, second = calls["compare"]
    assert second["variables"] == ["salinity"] and first["variables"] == ["temperature"]
    ((_bucket, opts),) = calls["plot"]
    assert [c.tag for c in opts["contours"]] == ["contour", "contour"]


def test_build_compare_contours_refuses_a_section_overlay_for_map_rows(monkeypatch):
    _fake_osk(
        monkeypatch, test_families=["field_row"], contour_families=["section_row"]
    )
    page = {
        "title": "x",
        "compare": {"reference": ["woa"], "variables": ["temperature"]},
        "contours": {"variables": ["salinity"]},
    }
    (p,) = P.expand(_suite([page], defaults={"test": "stub"}))
    with pytest.raises(ValueError, match=r"contours: built no 'field_row'"):
        P.build(p)


def test_build_field_contours_on_a_map_page_follow_the_section_rules(monkeypatch):
    calls = _fake_field(monkeypatch)
    page = _field_page({"variable": "salinity"})
    page["field"]["variables"] = ["temperature"]
    (p,) = P.expand(_suite([page], defaults={"test": "stub"}))
    P.build(p)
    main, lines = calls
    assert main.opts["contours"] is lines
    assert lines.kwargs["select"]["depth"] == 0.0


# -- real objects: the contour keys land on the plotted map items ---------------------


def _suite_of(page, source):
    return _suite([page], defaults={"test": source})


def test_a_field_map_page_with_contours_true_carries_its_own_lines(
    patched_map_source, captured
):
    (page,) = P.expand(_suite_of(_field_page(True), patched_map_source))
    P.build(page)
    spec = captured["spec"]
    assert spec.family == "field_facet"
    (item,) = spec.items
    assert isinstance(item["contour"], xr.DataArray)
    assert item["contour"].identical(item["field"])
    assert item["contour_units"] == "degC"


def test_a_field_map_page_with_a_contours_mapping_builds_the_second_field(
    patched_map_source, captured
):
    page = _field_page({"variable": LINES, "select": {"depth": 250.0}})
    page["field"]["variables"] = [VAR]
    (p,) = P.expand(_suite_of(page, patched_map_source))
    assert p.contours["select"]["depth"] == 250.0
    P.build(p)
    (item,) = captured["spec"].items
    assert item["contour"].name == LINES or LINES in str(item["contour_standard_name"])
    assert item["contour_units"] == "1"
    # the overlay is its own variable, not the plotted one
    assert not item["contour"].identical(item["field"])


def test_a_faceted_field_map_page_pairs_a_time_faceted_overlay(
    patched_map_source, captured
):
    page = _field_page({"variable": LINES})
    del page["field"]["aggregate"]
    page["field"]["select"] = {"depth": 0.0}
    (p,) = P.expand(_suite_of(page, patched_map_source))
    P.build(p)
    (item,) = captured["spec"].items
    assert item["contour"].sizes["time"] == 3 == item["field"].sizes["time"]


def _compare_page(contours, **extra):
    return {
        "title": "cmp",
        "compare": {
            "reference": ["woa_ref"],
            "variables": [VAR],
            "select": {"test": {"depth": 50.0}, "reference": {}},
        },
        "contours": contours,
        **extra,
    }


@pytest.fixture
def map_comparison_sources(patched_sources, monkeypatch):
    """Serve a ROMS run (2-D lon/lat) and a climatology, which compare as a map row."""
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    return "roms_test"


def test_a_compare_map_page_with_contours_true_carries_test_and_reference(
    map_comparison_sources, captured
):
    (p,) = P.expand(_suite_of(_compare_page(True), map_comparison_sources))
    P.build(p)
    spec = captured["spec"]
    assert spec.family == "field_row"
    (item,) = spec.items
    assert set(item["contour"]) == {"test", "reference"}


def test_a_compare_map_page_with_a_contours_mapping_pairs_the_second_comparison(
    map_comparison_sources, captured
):
    page = _compare_page({"select": {"test": {"depth": 200.0}, "reference": {}}})
    (p,) = P.expand(_suite_of(page, map_comparison_sources))
    P.build(p)
    spec = captured["spec"]
    assert spec.family == "field_row"
    (item,) = spec.items
    assert set(item["contour"]) == {"test", "reference"}
    # the lines are the second (deeper) comparison's, not the plotted row's own
    assert not item["contour"]["test"].identical(item["aligned"]["test"])


# -- end to end: the figure itself (needs both renderers' map-contour support) -------


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_end_to_end_a_field_map_page_draws_its_isolines(
    patched_map_source, renderer, monkeypatch
):
    (page,) = P.expand(_suite_of(_field_page({"variable": LINES}), patched_map_source))
    original = osk.field

    def with_renderer(*a, **k):
        obj = original(*a, **k)
        plot = obj.plot
        obj.plot = lambda **kw: plot(renderer=renderer, **kw)
        return obj

    monkeypatch.setattr(osk, "field", with_renderer)
    ((_, fig),) = P.build(page)
    assert fig is not None


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_end_to_end_a_compare_map_row_draws_its_isolines(
    map_comparison_sources, renderer
):
    comparison = osk.Comparison(
        label="row",
        test=map_comparison_sources,
        reference="woa_ref",
        variable=VAR,
        select={"depth": 50.0},
        cache=False,
    )
    fig = comparison.plot(contours=True, renderer=renderer, contour_levels=[14.0, 15.0])
    assert fig is not None


def test_a_scalar_lon_lat_overlay_is_refused_by_name():
    from ocean_skill.plot.map_contours import check_map_overlay

    grid = xr.DataArray(
        np.zeros((3, 4)),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0, 3.0]},
    )
    with pytest.raises(ValueError, match="single values"):
        check_map_overlay(grid.isel(lat=0, lon=0), grid)


def test_a_global_overlay_is_cut_to_the_panels_window_before_it_is_read():
    from ocean_skill.plot.map_contours import map_contour_arrays

    world = xr.DataArray(
        np.arange(181 * 360, dtype=float).reshape(181, 360),
        dims=("lat", "lon"),
        coords={"lat": np.arange(-90.0, 91.0), "lon": np.arange(-180.0, 180.0)},
    )
    # a dateline-straddling panel, 150E-110W: the window is in 0-360 and contiguous
    lon = np.r_[np.arange(150.0, 180.0, 2.0), np.arange(-180.0, -110.0, 2.0)]
    panel = xr.DataArray(
        np.zeros((11, lon.size)),
        dims=("lat", "lon"),
        coords={"lat": np.arange(-10.0, 10.1, 2.0), "lon": lon},
    )
    x, y, z = map_contour_arrays(world, panel=panel)
    assert x.shape == (21, 99) and np.isfinite(z).all()
    assert (x.min(), x.max(), y.min(), y.max()) == (150.0, 248.0, -10.0, 10.0)
