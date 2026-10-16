"""``locations=`` takes labelled select dicts: ``{label: select}``.

In a suite YAML only catalog names could be written under ``locations:``. A
``{label: select}`` mapping (alone, or beside names/objects in a list) draws each
select's geometry -- the same grammar ``compare(select=...)`` reads -- labelled with the
key, so a bathymetry map can carry the equator transect, the lon-mean box slabs and the
T-S boxes without any data behind them. The domain these exist for straddles the
dateline in 0-360 degrees: every shape must come out seam-split, and the label must be
what ``legend="annotate"`` writes.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot.locations import annotation_anchors, legend_groups
from ocean_skill.plot.map_locations import location_items, map_locations
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

EQUATOR = {"lat": 0, "lon": {"min": 140, "max": 280}}
SLAB = {"lon": {"min": 180, "max": 200}, "lat": {"min": -30, "max": 30}}
POINT = {"lon": 200, "lat": 20}
PATH = {"transect": {"waypoints": [[160, -20], [200, 0], [250, 10]]}}


def _by_label(items):
    return {i["legend_label"]: i for i in items}


def test_a_mapping_draws_one_labelled_selection_item_per_entry():
    items = location_items(
        {"Equator": EQUATOR, "Slab": SLAB, "Stn": POINT, "Section": PATH}
    )
    got = _by_label(items)
    assert list(got) == ["Equator", "Slab", "Stn", "Section"]
    assert {k: v["kind"] for k, v in got.items()} == {
        "Equator": "line",
        "Slab": "extent",
        "Stn": "point",
        "Section": "line",
    }
    assert all(i["featureType"] == "selection" for i in items)
    assert all(i["name"] == i["legend_label"] for i in items)


def test_a_zonal_line_across_the_dateline_is_split_at_the_seam():
    (item,) = location_items({"Equator": EQUATOR}).__iter__()
    ends = sorted((p[0, 0], p[-1, 0]) for p in item["paths"])
    assert ends == [(-180.0, -80.0), (140.0, 180.0)]
    assert all(np.all(p[:, 1] == 0.0) for p in item["paths"])


def test_a_box_in_0_360_degrees_lands_in_the_pm180_frame():
    slab = location_items({"Slab": SLAB})[0]
    assert slab["bboxes"] == [(-180.0, -30.0, -160.0, 30.0)]
    # a box straddling the seam is two pieces, not one backwards rectangle
    ts = location_items(
        {"TS": {"lon": {"min": 170, "max": 190}, "lat": {"min": 5, "max": 15}}}
    )[0]
    assert ts["bboxes"] == [(170.0, 5.0, 180.0, 15.0), (-180.0, 5.0, -170.0, 15.0)]


def test_a_transect_path_across_the_dateline_is_split_too():
    item = location_items({"Section": PATH})[0]
    assert len(item["paths"]) == 2
    assert item["paths"][0][-1, 0] == 180.0 and item["paths"][1][0, 0] == -180.0


def test_a_point_is_wrapped():
    item = location_items({"Stn": POINT})[0]
    assert (item["lon"], item["lat"]) == (-160.0, 20)


def test_a_list_mixes_mappings_with_names(monkeypatch):
    catalog_item = {
        "kind": "point",
        "lon": 1.0,
        "lat": 2.0,
        "name": "papa",
        "featureType": "timeSeries",
    }
    seen = {}

    def fake_build(names, catalog=None):
        seen["names"] = names
        return [catalog_item], (0, 0, 1, 1)

    monkeypatch.setattr("ocean_skill.plot.locations.build_items", fake_build)
    items = location_items(["papa", {"Equator": EQUATOR}, {"Slab": SLAB}])
    assert seen["names"] == ["papa"]
    assert [i["name"] for i in items] == ["papa", "Equator", "Slab"]


def test_the_labels_are_distinct_legend_groups_in_distinct_colors():
    items = location_items({"Equator": EQUATOR, "Slab": SLAB, "Slab 2": SLAB})
    groups = legend_groups(items)
    assert [g[0] for g in groups] == ["Equator", "Slab", "Slab 2"]
    assert len({g[1]["color"] for g in groups}) == 3


def test_annotate_places_each_label_on_its_shape():
    items = location_items({"Equator": EQUATOR, "Slab": SLAB, "Stn": POINT})
    anchors, rest = annotation_anchors(items)
    assert [a["text"] for a in anchors] == ["Equator", "Slab", "Stn"]
    assert rest == []
    slab = next(a for a in anchors if a["text"] == "Slab")
    assert slab["lon"] == pytest.approx(-170.0) and slab["lat"] == 30.0


@pytest.mark.parametrize(
    "bad, error",
    [
        ({"": EQUATOR}, TypeError),
        ({"x": "not a select"}, TypeError),
        ({"Depth only": {"depth": 10}}, ValueError),
        ({"Grid": {"transect": {"xi_rho": 3}}}, ValueError),
    ],
)
def test_an_entry_with_nothing_to_draw_names_itself(bad, error):
    with pytest.raises(error) as exc:
        location_items(bad, who="locations=")
    label = next(iter(bad))
    assert "locations=" in str(exc.value) and (not label or label in str(exc.value))


def test_the_standalone_map_draws_and_annotates():
    fig = map_locations(
        {"Equator": EQUATOR, "Slab": SLAB},
        legend="annotate",
        extent=(-180, -40, 180, 40),
    )
    ax = next(a for a in fig.axes if hasattr(a, "projection"))
    labels = {t.get_text() for t in ax.texts}
    assert {"Equator", "Slab"} <= labels
    assert ax.get_legend() is None  # every group is named in place


def _straddler():
    lon = np.unique(
        np.concatenate([np.linspace(140.0, 180.0, 30), np.linspace(-180.0, -70.0, 50)])
    )
    lat = np.linspace(-40.0, 40.0, 20)
    z = np.add.outer(np.linspace(0, 1, lat.size), np.linspace(0, 1, lon.size))
    return xr.DataArray(
        z, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs={"units": "m"}
    )


def _facet(field):
    return {
        "field": field,
        "facet_dim": None,
        "row_dim": None,
        "units": "m",
        "standard_name": "sea_floor_depth_below_geoid",
    }


def test_the_static_field_map_draws_the_lines_and_boxes_in_place():
    """On the dateline-straddling map the shapes land where the projection puts them."""
    ccrs = pytest.importorskip("cartopy.crs")
    items = location_items({"Equator": EQUATOR, "Slab": SLAB}, domain=None)
    fig = render(
        PlotSpec(
            "field_facet",
            [_facet(_straddler())],
            {"location_items": items, "legend": "annotate"},
        )
    )
    ax = next(a for a in fig.axes if hasattr(a, "projection"))
    assert ax.projection.proj4_params["lon_0"] == 180.0
    fig.canvas.draw()
    solid = [ln for ln in ax.lines if ln.get_linestyle() == "-"]
    # both halves of the equator, meeting at the seam, and the slab's four sides
    equator = [ln for ln in solid if ln.get_ydata()[0] == 0]
    xs = sorted((ln.get_xdata()[0], ln.get_xdata()[-1]) for ln in equator)
    assert xs == [(-180.0, -80.0), (140.0, 180.0)]
    assert {t.get_text() for t in ax.texts} >= {"Equator", "Slab"}
    # the slab's west edge (180E) is the map's centre meridian: display x of the
    # equator's seam end equals that of lon=180 under the axes projection
    centre = ax.transData.transform(
        ax.projection.transform_point(180.0, 0.0, ccrs.PlateCarree())
    )[0]
    west = next(ln for ln in solid if ln.get_xdata()[0] == 140.0)
    x_end = ax.transData.transform(
        ax.projection.transform_point(
            west.get_xdata()[-1], west.get_ydata()[-1], ccrs.PlateCarree()
        )
    )[0]
    assert x_end == pytest.approx(centre, abs=1.0)


def test_the_interactive_field_map_renders_the_same_items():
    import holoviews as hv

    from ocean_skill.plot.holoviews_renderer import _field_facet

    items = location_items({"Equator": EQUATOR, "Slab": SLAB}, domain=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlay = _field_facet(
            _facet(_straddler()),
            tiles=False,
            land=False,
            location_items=items,
            legend="annotate",
        )
        hv.render(overlay, backend="bokeh")
    kinds = [type(e).__name__ for e in overlay]
    assert "Path" in kinds
    assert "Labels" in kinds or "Text" in kinds
