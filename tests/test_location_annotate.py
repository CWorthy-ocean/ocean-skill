"""``legend="annotate"`` for ``locations=``: names written beside their shapes.

The pure placement rules (:func:`ocean_skill.plot.locations.annotation_anchors`) and
the static renderer that draws them. A labelled selection is named in place; every
other group still goes in a framed key, drawn only when one exists.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.plot.locations import (
    HOVER_FIELDS,
    annotation_anchors,
    resolve_location_legend,
)
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec


def _hover(name):
    return {f: (name if f == "name" else "") for f in HOVER_FIELDS}


def _sel(kind, label, **geo):
    return {
        **_hover(label or "sel"),
        "kind": kind,
        "featureType": "selection",
        "legend_label": label,
        **geo,
    }


POINT = _sel("point", "SG", lon=-90.0, lat=5.0)
BOX = _sel("extent", "NWP", bboxes=[(-96.0, -5.0, -88.0, 4.0)])
LINE = _sel(
    "line",
    "Eq",
    paths=[np.array([[-95.0, 0.0], [-85.0, 0.0]]), np.array([[1, 2], [3, 4.0]])],
)
STATION = {
    **_hover("stn"),
    "kind": "point",
    "featureType": "timeSeries",
    "lon": -92.0,
    "lat": -3.0,
}


# -- pure logic -----------------------------------------------------------------------


def test_resolve_location_legend():
    assert resolve_location_legend(True) is True
    assert resolve_location_legend(False) is False
    assert resolve_location_legend(None) is False
    assert resolve_location_legend("annotate") == "annotate"
    with pytest.raises(ValueError, match="annotate"):
        resolve_location_legend("sideways")


def test_anchor_geometry_point_box_line():
    anchors, rest = annotation_anchors([POINT, BOX, LINE])
    assert rest == []
    by = {a["text"]: a for a in anchors}
    assert [a["text"] for a in anchors] == ["SG", "NWP", "Eq"]
    sg, nwp, eq = by["SG"], by["NWP"], by["Eq"]
    assert (sg["lon"], sg["lat"], sg["ha"], sg["va"], sg["dx"], sg["dy"]) == (
        -90.0,
        5.0,
        "left",
        "center",
        1,
        0,
    )
    assert (nwp["lon"], nwp["lat"], nwp["ha"], nwp["va"], nwp["dx"], nwp["dy"]) == (
        -92.0,
        4.0,
        "center",
        "bottom",
        0,
        1,
    )
    # last vertex of the LAST path
    assert (eq["lon"], eq["lat"], eq["ha"], eq["va"], eq["dx"], eq["dy"]) == (
        3.0,
        4.0,
        "right",
        "bottom",
        0,
        1,
    )
    assert {a["color"] for a in anchors} and all(a["color"] for a in anchors)


def test_anchor_seam_split_box_is_centred_across_the_seam():
    box = _sel(
        "extent", "X", bboxes=[(170.0, 5.0, 180.0, 15.0), (-180.0, 5.0, -170.0, 15.0)]
    )
    (a,), _ = annotation_anchors([box])
    assert abs(a["lon"]) == 180.0
    assert a["lat"] == 15.0
    # a box crossing the seam off-centre: 160..-150 -> midpoint 185 -> -175
    box = _sel(
        "extent", "Y", bboxes=[(160.0, 0.0, 180.0, 9.0), (-180.0, 0.0, -150.0, 9.0)]
    )
    (a,), _ = annotation_anchors([box])
    assert a["lon"] == pytest.approx(-175.0)


def test_unlabelled_and_unplaceable_groups_stay_in_the_key():
    ring = _sel("ring", "R", paths=[np.array([[0, 0], [1, 1.0]])])
    unlabelled_sel = _sel("point", None, lon=1.0, lat=1.0)
    anchors, rest = annotation_anchors([STATION, unlabelled_sel, ring, POINT])
    assert [a["text"] for a in anchors] == ["SG"]
    assert [label for label, _s, _m in rest] == ["timeSeries", "selection", "R"]


def test_anchor_colour_follows_colors():
    (a,), _ = annotation_anchors([POINT], colors={"SG": "purple"})
    assert a["color"] == "purple"


# -- static renderer ------------------------------------------------------------------


def _map(lon):
    lat = np.linspace(-40.0, 40.0, 20)
    z = np.add.outer(np.linspace(0.0, 1.0, lat.size), np.linspace(0.0, 1.0, len(lon)))
    return xr.DataArray(
        z, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs={"units": "m"}
    )


PLAIN = _map(np.linspace(-98.0, -80.0, 30))
STRADDLER = _map(
    np.unique(
        np.concatenate([np.linspace(80.0, 180.0, 40), np.linspace(-180.0, -44.0, 30)])
    )
)


def _facet_item(field, facet_dim=None):
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
    return [ax for ax in fig.axes if hasattr(ax, "projection")]


def test_static_labels_replace_the_key_when_every_group_is_labelled():
    fig = _static(PLAIN, [POINT, BOX, LINE], legend="annotate")
    (ax,) = _map_axes(fig)
    assert sorted(t.get_text() for t in ax.texts) == ["Eq", "NWP", "SG"]
    assert ax.get_legend() is None


def test_static_mixed_keeps_a_key_with_only_the_unlabelled_group():
    fig = _static(PLAIN, [STATION, POINT, BOX], legend="annotate")
    (ax,) = _map_axes(fig)
    assert sorted(t.get_text() for t in ax.texts) == ["NWP", "SG"]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["timeSeries"]


def test_static_label_style_and_annot_kwargs_override():
    fig = _static(PLAIN, [POINT], legend="annotate", colors={"SG": "purple"})
    (t,) = _map_axes(fig)[0].texts
    assert t.get_color() == "purple"
    assert t.get_fontweight() == "bold"
    fig = _static(PLAIN, [POINT], legend="annotate", annot_kwargs={"color": "k"})
    (t,) = _map_axes(fig)[0].texts
    assert t.get_color() == "k"


def test_static_labels_do_not_widen_the_view():
    plain = _map_axes(_static(PLAIN, [POINT]))[0]
    fig = _static(PLAIN, [POINT, BOX], legend="annotate")
    ax = _map_axes(fig)[0]
    fig.canvas.draw()
    assert ax.get_xlim() == plain.get_xlim()
    assert ax.get_ylim() == plain.get_ylim()


def test_static_label_lands_at_the_anchor_on_a_180_centred_axes():
    ccrs = pytest.importorskip("cartopy.crs")
    pt = _sel("point", "SWP", lon=170.0, lat=-15.0)
    fig = _static(STRADDLER, [pt], legend="annotate")
    (ax,) = _map_axes(fig)
    assert ax.projection.proj4_params["lon_0"] == 180.0
    fig.canvas.draw()
    (t,) = ax.texts
    expected = ax.transData.transform(
        ax.projection.transform_point(170.0, -15.0, ccrs.PlateCarree())
    )
    # the text sits 4 points east of the marker: same row, a few pixels to the right
    box = t.get_window_extent()
    x = box.x0
    px = 4 * fig.dpi / 72
    assert x == pytest.approx(expected[0] + px, abs=2.0)
    assert abs((box.y0 + box.y1) / 2 - expected[1]) < 3


def test_static_facets_label_every_panel_and_key_once():
    time = pd.date_range("2012-01-01", periods=2, freq="MS")
    field = xr.concat([PLAIN, PLAIN + 1.0], dim="time").assign_coords(time=time)
    fig = render(
        PlotSpec(
            "field_facet",
            [_facet_item(field, facet_dim="time")],
            {"location_items": [STATION, POINT], "legend": "annotate"},
        )
    )
    panels = _map_axes(fig)
    assert len(panels) == 2
    assert [[t.get_text() for t in ax.texts] for ax in panels] == [["SG"], ["SG"]]
    assert [ax.get_legend() is not None for ax in panels] == [True, False]


def test_static_bad_legend_value_raises():
    with pytest.raises(ValueError, match="annotate"):
        _static(PLAIN, [POINT], legend="sideways")


def test_standalone_locations_annotate():
    from ocean_skill.plot.matplotlib_renderer import locations

    fig = locations(
        [POINT, BOX], extent=(-100.0, -10.0, -80.0, 10.0), legend="annotate"
    )
    ax = _map_axes(fig)[0]
    assert sorted(t.get_text() for t in ax.texts) == ["NWP", "SG"]
    assert ax.get_legend() is None
