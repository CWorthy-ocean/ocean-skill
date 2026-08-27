"""Tests for :mod:`ocean_skill.pick`: the interactive waypoint picker.

``pick_path`` is the one widget in this package that needs a live kernel by
design (see the module's own docstring for why) -- everything else the interactive
renderer draws is deliberately kernel-free. That does not mean it needs a browser
to test, though: every check here fabricates the ``PointDraw`` stream's ``.data``
the same way holoviews' own ``CDSCallback`` sets it from a real click, so the whole
suite runs headless. Catalog isolation follows ``tests/test_map_locations.py``'s
pattern -- ``catalog.resolve`` patched by hand, nothing real opened.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

hv = pytest.importorskip("holoviews")

from ocean_skill.pick import PathPicker, _domain_paths, pick_path  # noqa: E402

# A domain declared 0-360, straddling the antimeridian the long way -- the same
# stress case tests/test_antimeridian.py builds around (pac_dt_ramp-style).
STRADDLING_META = {"domain_outline": [[170.0, -10.0], [200.0, -10.0], [200.0, 10.0], [170.0, 10.0], [170.0, -10.0]]}

BBOX_META = {
    "geospatial_lon_min": -96.0,
    "geospatial_lat_min": 20.0,
    "geospatial_lon_max": -90.0,
    "geospatial_lat_max": 30.0,
}


def _resolver(meta_by_name):
    def fake_resolve(source):
        if source in meta_by_name:
            return SimpleNamespace(metadata=meta_by_name[source])
        raise KeyError(f"Unknown source {source!r}. Did you mean...?")

    return fake_resolve


# -- _domain_paths: the outline/bbox fallback ladder ---------------------------------


def test_outline_ring_stays_contiguous_in_its_own_convention():
    with patch("ocean_skill.catalog.resolve", _resolver({"pac": STRADDLING_META})):
        paths = _domain_paths("pac")
    assert len(paths) == 1
    lon = paths[0][:, 0]
    # one contiguous piece: no jump back across the seam anywhere in the ring
    assert np.all(np.abs(np.diff(lon)) < 180.0)
    assert lon.max() > 180.0  # genuinely drawn past the seam, not wrapped to it


def test_bbox_fallback_draws_a_closed_rectangle():
    with patch("ocean_skill.catalog.resolve", _resolver({"gom": BBOX_META})):
        paths = _domain_paths("gom")
    assert len(paths) == 1
    ring = paths[0]
    assert tuple(ring[0]) == tuple(ring[-1])  # closed
    lons, lats = ring[:, 0], ring[:, 1]
    assert lons.min() == pytest.approx(-96.0)
    assert lons.max() == pytest.approx(-90.0)
    assert lats.min() == pytest.approx(20.0)
    assert lats.max() == pytest.approx(30.0)


def test_missing_metadata_names_the_keys_it_needs():
    with patch("ocean_skill.catalog.resolve", _resolver({"nothing": {}})):
        with pytest.raises(ValueError, match="domain_outline") as exc:
            _domain_paths("nothing")
    assert "geospatial_lon_min" in str(exc.value)


def test_unknown_source_propagates_catalogs_own_keyerror():
    with patch("ocean_skill.catalog.resolve", _resolver({})):
        with pytest.raises(KeyError, match="Did you mean"):
            _domain_paths("typo")


# -- pick_path: the overlay it builds -------------------------------------------------


def test_overlay_is_a_path_and_points_with_lon_lat_kdims():
    with patch("ocean_skill.catalog.resolve", _resolver({"gom": BBOX_META})):
        picker = pick_path("gom")
    kinds = {type(el).__name__: el for el in picker.overlay}
    assert set(kinds) == {"Path", "Points"}
    for el in kinds.values():
        assert [d.name for d in el.kdims] == ["lon", "lat"]
    assert picker.stream.source is kinds["Points"]


def test_point_draw_tool_is_created_and_active():
    with patch("ocean_skill.catalog.resolve", _resolver({"gom": BBOX_META})):
        picker = pick_path("gom")
    plot = hv.renderer("bokeh").get_plot(picker.overlay)
    fig = plot.state
    from bokeh.models import PointDrawTool

    draw_tools = [t for t in fig.toolbar.tools if isinstance(t, PointDrawTool)]
    assert len(draw_tools) == 1
    assert fig.toolbar.active_tap is draw_tools[0]


def test_picker_is_not_reachable_through_the_renderer_registry():
    from ocean_skill.plot.registry import _RENDERERS

    assert "pick_path" not in _RENDERERS
    assert set(_RENDERERS) <= {"matplotlib", "holoviews"}


# -- PathPicker.waypoints / as_select --------------------------------------------------


def _picker() -> PathPicker:
    with patch("ocean_skill.catalog.resolve", _resolver({"gom": BBOX_META})):
        return pick_path("gom")


def test_waypoints_preserve_click_order_and_are_plain_floats():
    picker = _picker()
    picker.stream.update(
        data={"lon": np.array([-95.0, -94.0, -93.0]), "lat": np.array([24.0, 25.0, 26.0])}
    )
    wp = picker.waypoints
    assert wp == [[-95.0, 24.0], [-94.0, 25.0], [-93.0, 26.0]]
    assert all(isinstance(v, float) for pair in wp for v in pair)


def test_waypoints_accept_plain_lists_too():
    picker = _picker()
    picker.stream.update(data={"lon": [-95.0, -94.0], "lat": [24.0, 25.0]})
    assert picker.waypoints == [[-95.0, 24.0], [-94.0, 25.0]]


def test_waypoints_lons_are_not_rewrapped():
    picker = _picker()
    picker.stream.update(data={"lon": [200.5], "lat": [10.0]})
    assert picker.waypoints == [[200.5, 10.0]]


def test_empty_stream_raises_before_any_click(monkeypatch):
    picker = _picker()
    assert picker.stream.data is None
    with pytest.raises(RuntimeError, match="comm"):
        picker.waypoints


def test_empty_stream_raises_after_a_render_with_no_clicks():
    picker = _picker()
    picker.stream.update(data={"lon": np.array([]), "lat": np.array([])})
    with pytest.raises(RuntimeError, match="comm"):
        picker.waypoints


def test_as_select_emits_the_transect_spec():
    picker = _picker()
    picker.stream.update(data={"lon": [-95.0, -94.0], "lat": [24.0, 25.0]})
    assert picker.as_select() == {
        "transect": {"waypoints": [[-95.0, 24.0], [-94.0, 25.0]]}
    }
    assert picker.as_select(spacing_km=10.0) == {
        "transect": {
            "waypoints": [[-95.0, 24.0], [-94.0, 25.0]],
            "spacing_km": 10.0,
        }
    }


def test_repr_never_raises_empty_or_populated():
    empty = _picker()
    assert "no waypoints yet" in repr(empty)

    populated = _picker()
    populated.stream.update(data={"lon": [-95.0], "lat": [24.0]})
    assert "1 waypoint" in repr(populated)
