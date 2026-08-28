"""Tests for :mod:`ocean_skill.plot.map_locations`: where a plotted selection sits.

Builds :class:`~ocean_skill.comparison.Comparison` objects with ``catalog.resolve``
(and, where the footprint fallback needs it, ``catalog.discover``) patched by hand —
the same isolation pattern ``tests/test_antimeridian.py``'s ``_stub_comparison``
uses — so these tests exercise ``build_map_items``/``map_locations`` without ever
touching a real catalog or opening data.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ocean_skill import catalog
from ocean_skill.comparison import Comparison, ComparisonSet
from ocean_skill.field import Field, FieldSet
from ocean_skill.plot.map_locations import (
    build_map_items,
    footprint_item,
    map_locations,
)

# A domain declared 0-360 (77.4E - 316.2E), straddling the antimeridian the long
# way -- the same stress case tests/test_antimeridian.py builds around.
STRADDLING_META = {
    "geospatial_lon_min": 77.4,
    "geospatial_lat_min": -20.0,
    "geospatial_lon_max": 316.2,
    "geospatial_lat_max": 20.0,
    "featureType": "grid",
}

# An ordinary, non-straddling domain for the simpler cases.
PLAIN_META = {
    "geospatial_lon_min": -170.0,
    "geospatial_lat_min": 10.0,
    "geospatial_lon_max": -100.0,
    "geospatial_lat_max": 60.0,
    "featureType": "grid",
}


def _resolver(meta_by_name):
    def fake_resolve(source):
        if source in meta_by_name:
            return SimpleNamespace(metadata=meta_by_name[source])
        raise KeyError(source)

    return fake_resolve


def _comparison(*, meta=PLAIN_META, test_name="test_src", **kwargs):
    """A Comparison whose test source resolves; the reference never does."""
    return Comparison(
        reference="unresolvable_ref", test=test_name, variable="nitrate", **kwargs
    )


@pytest.fixture(autouse=True)
def _quiet_variable_resolution_warning():
    """Silence the one-time ``variable="nitrate"`` standard_name resolution warning.

    Irrelevant to every test here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        yield


# -- geometry: point, region band, lone-lon/lat, footprint fallback -----------------


def test_point_select_draws_one_selection_point_and_the_domain_ring():
    c = _comparison(select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        items = build_map_items(c)
    kinds = {it["kind"] for it in items}
    assert kinds == {"point", "ring"}
    point = next(it for it in items if it["kind"] == "point")
    assert (point["lon"], point["lat"]) == pytest.approx((-144.3, 50.0))
    assert point["featureType"] == "selection"


def test_point_select_never_triggers_alignment():
    c = _comparison(select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        with patch(
            "ocean_skill.align.align",
            side_effect=AssertionError("align() must never be called"),
        ):
            build_map_items(c)
    assert c._aligned is None


def test_pair_spec_of_two_points_draws_both():
    c = _comparison(
        select={
            "test": {"lon": -144.3, "lat": 50.0},
            "reference": {"lon": -144.1, "lat": 50.9},
        }
    )
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        items = build_map_items(c)
    points = [it for it in items if it["kind"] == "point"]
    assert len(points) == 2
    positions = {(p["lon"], p["lat"]) for p in points}
    assert positions == {(-144.3, 50.0), (-144.1, 50.9)}


def test_pair_spec_of_the_same_shared_point_still_dedupes():
    same = {"lon": -144.3, "lat": 50.0}
    c = _comparison(select={"test": dict(same), "reference": dict(same)})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        items = build_map_items(c)
    points = [it for it in items if it["kind"] == "point"]
    assert len(points) == 1


def test_one_sided_band_clamps_to_the_test_domain():
    c = _comparison(select={"lon": {"min": -150.0}})  # no max -> clamp to domain
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        items = build_map_items(c)
    extent = next(it for it in items if it["kind"] == "extent")
    (lo, la, hi, ha) = extent["bboxes"][0]
    assert lo == pytest.approx(-150.0)
    assert hi == pytest.approx(PLAIN_META["geospatial_lon_max"])
    assert la == pytest.approx(PLAIN_META["geospatial_lat_min"])
    assert ha == pytest.approx(PLAIN_META["geospatial_lat_max"])


def test_straddling_region_band_splits_at_the_seam():
    c = _comparison(
        meta=STRADDLING_META,
        select={
            "lat": {"min": -10.0, "max": 10.0},
            "lon": {"min": 170.0, "max": 200.0},
        },
    )
    with patch(
        "ocean_skill.catalog.resolve", _resolver({"test_src": STRADDLING_META})
    ):
        items = build_map_items(c)
    extent = next(it for it in items if it["kind"] == "extent")
    assert extent["bboxes"] == [
        (170.0, -10.0, 180.0, 10.0),
        (-180.0, -10.0, -160.0, 10.0),
    ]


def test_lone_lon_draws_a_meridian_line_spanning_the_domain():
    c = _comparison(select={"lon": -150.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        items = build_map_items(c)
    line = next(it for it in items if it["kind"] == "line")
    assert len(line["paths"]) == 1
    (path,) = line["paths"]
    assert path[:, 0].tolist() == [-150.0, -150.0]
    assert sorted(path[:, 1].tolist()) == [
        PLAIN_META["geospatial_lat_min"],
        PLAIN_META["geospatial_lat_max"],
    ]


def test_lone_lat_across_a_straddling_domain_splits_into_two_segments():
    c = _comparison(meta=STRADDLING_META, select={"lat": 0.0})
    with patch(
        "ocean_skill.catalog.resolve", _resolver({"test_src": STRADDLING_META})
    ):
        items = build_map_items(c)
    line = next(it for it in items if it["kind"] == "line")
    assert len(line["paths"]) == 2
    all_lons = [lon for path in line["paths"] for lon in path[:, 0]]
    assert all(-180.0 <= lon <= 180.0 for lon in all_lons)
    assert any(abs(lon - 180.0) < 1e-9 for lon in all_lons)
    assert any(abs(lon + 180.0) < 1e-9 for lon in all_lons)


def test_no_horizontal_select_falls_back_to_each_sources_footprint():
    papa_meta = {
        "featureType": "timeSeries",
        "geospatial_lon_min": -144.8,
        "geospatial_lon_max": -144.8,
        "geospatial_lat_min": 50.07,
        "geospatial_lat_max": 50.07,
    }
    c = _comparison(select={"time": "2012-01"}, test_name="grid_src")
    c.reference_name = "papa"  # a resolvable reference, for this one test
    papa_ref = catalog.SourceRef(
        name="papa", catalog="OOI", path=None, metadata=papa_meta
    )
    grid_ref = catalog.SourceRef(
        name="grid_src", catalog="stub", path=None, metadata=PLAIN_META
    )
    resolver = _resolver({"grid_src": PLAIN_META, "papa": papa_meta})
    with patch("ocean_skill.catalog.resolve", resolver), patch(
        "ocean_skill.catalog.discover",
        lambda: {"grid_src": grid_ref, "papa": papa_ref},
    ):
        items = build_map_items(c)
    names = {it["name"] for it in items if it["kind"] in ("point", "extent")}
    assert names == {"grid_src", "papa"}


def test_unresolvable_reference_warns_and_is_skipped():
    c = _comparison(select={"time": "2012-01"})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        with pytest.warns(UserWarning, match="not a known catalog source"):
            items = build_map_items(c)
    assert all(it.get("name") != "unresolvable_ref" for it in items)


def test_footprint_item_warns_and_returns_none_for_an_extentless_source():
    ref = catalog.SourceRef(
        name="bare", catalog="stub", path=None, metadata={"featureType": "grid"}
    )
    with patch("ocean_skill.catalog.discover", lambda: {"bare": ref}):
        with pytest.warns(UserWarning, match="no declared geospatial extent"):
            assert footprint_item("bare") is None


# -- domain ring: default, suppressed, overridden ------------------------------------


def test_domain_none_suppresses_the_ring():
    c = _comparison(select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        items = build_map_items(c, domain=None)
    assert all(it["kind"] != "ring" for it in items)


def test_a_user_bbox_overrides_the_domain_ring():
    c = _comparison(select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        items = build_map_items(c, domain=(0.0, 0.0, 5.0, 5.0))
    ring = next(it for it in items if it["kind"] == "ring")
    (path,) = ring["paths"]
    assert path.tolist() == [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]


def test_outside_declared_bbox_point_warns():
    c = _comparison(select={"lon": 10.0, "lat": 80.0})  # far outside PLAIN_META
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        with pytest.warns(UserWarning, match="outside 'test_src'"):
            build_map_items(c)


def test_inside_declared_bbox_point_does_not_warn():
    c = _comparison(select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_map_items(c)
    assert not any("outside" in str(w.message) for w in caught)


# -- ComparisonSet / Field / FieldSet -------------------------------------------------


def test_comparisonset_dedupes_a_shared_point_and_domain_ring():
    c1 = _comparison(select={"lon": -144.3, "lat": 50.0})
    c2 = Comparison(
        reference="unresolvable_ref",
        test="test_src",
        variable="salinity",
        select={"lon": -144.3, "lat": 50.0},
    )
    cs = ComparisonSet([c1, c2])
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        items = build_map_items(cs)
    assert len([it for it in items if it["kind"] == "point"]) == 1
    assert len([it for it in items if it["kind"] == "ring"]) == 1


def test_field_builds_footprint_and_ring_with_no_select():
    ref = catalog.SourceRef(
        name="grid_src", catalog="stub", path=None, metadata=PLAIN_META
    )
    f = Field("grid_src", "nitrate")
    with patch("ocean_skill.catalog.resolve", _resolver({"grid_src": PLAIN_META})):
        with patch("ocean_skill.catalog.discover", lambda: {"grid_src": ref}):
            items = build_map_items(f)
    kinds = {it["kind"] for it in items}
    assert kinds == {"extent", "ring"}


def test_field_with_a_point_select_draws_it():
    f = Field("grid_src", "nitrate", select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"grid_src": PLAIN_META})):
        items = build_map_items(f)
    point = next(it for it in items if it["kind"] == "point")
    assert (point["lon"], point["lat"]) == pytest.approx((-144.3, 50.0))


def test_fieldset_covers_every_member():
    fs = FieldSet(
        [
            Field("grid_src", "nitrate", select={"lon": -144.3, "lat": 50.0}),
            Field("grid_src", "salinity", select={"lon": -144.3, "lat": 50.0}),
        ]
    )
    with patch("ocean_skill.catalog.resolve", _resolver({"grid_src": PLAIN_META})):
        items = build_map_items(fs)
    # the two members share a select -> one deduped point, plus one domain ring
    assert len([it for it in items if it["kind"] == "point"]) == 1
    assert len([it for it in items if it["kind"] == "ring"]) == 1


def test_nothing_placeable_raises():
    ref = catalog.SourceRef(
        name="bare", catalog="stub", path=None, metadata={"featureType": "grid"}
    )
    f = Field("bare", "nitrate")
    with patch("ocean_skill.catalog.resolve", _resolver({})):
        with patch("ocean_skill.catalog.discover", lambda: {"bare": ref}):
            with pytest.warns(UserWarning):
                with pytest.raises(ValueError, match="nothing to map"):
                    build_map_items(f)


def test_build_map_items_rejects_an_unsupported_type():
    with pytest.raises(TypeError, match="Comparison, ComparisonSet, Field"):
        build_map_items(object())


# -- top-level map_locations(): mixed input, rendering -------------------------------


def test_map_locations_mixes_a_name_and_an_object():
    from matplotlib.figure import Figure

    papa_meta = {
        "featureType": "timeSeries",
        "geospatial_lon_min": -144.8,
        "geospatial_lon_max": -144.8,
        "geospatial_lat_min": 50.07,
        "geospatial_lat_max": 50.07,
    }
    papa_ref = catalog.SourceRef(
        name="papa", catalog="OOI", path=None, metadata=papa_meta
    )
    c = _comparison(select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        with patch("ocean_skill.catalog.discover", lambda: {"papa": papa_ref}):
            fig = map_locations([c, "papa"])
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    labels = sorted(t.get_text() for t in ax.get_legend().get_texts())
    assert labels == ["domain", "selection", "timeSeries"]


def test_map_locations_object_alone_renders_both_backends():
    from matplotlib.figure import Figure

    c = _comparison(select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        fig = map_locations(c)
        assert isinstance(fig, Figure)
        obj = map_locations(c, renderer="holoviews")
    import holoviews as hv

    rendered = hv.render(obj, backend="bokeh")
    legend_labels = {it.label["value"] for it in rendered.legend[0].items}
    assert legend_labels == {"selection", "domain"}


def test_map_locations_rejects_an_unsupported_list_item():
    with pytest.raises(TypeError, match="cannot place"):
        map_locations([1.0])


# -- the four .map_locations() methods ------------------------------------------------


def test_comparison_map_locations_method():
    from matplotlib.figure import Figure

    c = _comparison(select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        fig = c.map_locations()
    assert isinstance(fig, Figure)
    assert c._aligned is None


def test_comparisonset_map_locations_method():
    from matplotlib.figure import Figure

    c1 = _comparison(select={"lon": -144.3, "lat": 50.0})
    c2 = Comparison(
        reference="unresolvable_ref",
        test="test_src",
        variable="salinity",
        select={"lon": -144.3, "lat": 50.0},
    )
    cs = ComparisonSet([c1, c2])
    with patch("ocean_skill.catalog.resolve", _resolver({"test_src": PLAIN_META})):
        fig = cs.map_locations()
    assert isinstance(fig, Figure)


def test_field_map_locations_method():
    from matplotlib.figure import Figure

    f = Field("grid_src", "nitrate", select={"lon": -144.3, "lat": 50.0})
    with patch("ocean_skill.catalog.resolve", _resolver({"grid_src": PLAIN_META})):
        fig = f.map_locations()
    assert isinstance(fig, Figure)


def test_fieldset_map_locations_method():
    from matplotlib.figure import Figure

    fs = FieldSet(
        [
            Field("grid_src", "nitrate", select={"lon": -144.3, "lat": 50.0}),
            Field("grid_src", "salinity", select={"lon": -144.3, "lat": 50.0}),
        ]
    )
    with patch("ocean_skill.catalog.resolve", _resolver({"grid_src": PLAIN_META})):
        fig = fs.map_locations()
    assert isinstance(fig, Figure)
