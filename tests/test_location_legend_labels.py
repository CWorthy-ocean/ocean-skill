"""A labelled selection is its own legend entry, in its own colour.

``Field``/``Comparison`` selections all carry ``featureType == "selection"``, so three
labelled transects and boxes laid over one map used to read "selection", "selection",
"selection" -- twice each in the static key, since a group's box and line drew a handle
apiece. ``label=`` now keys the legend instead (``legend_label`` on the item, grouped by
:func:`ocean_skill.plot.locations.legend_groups`), and each labelled selection takes the
next :data:`~ocean_skill.plot.locations.SELECTION_PALETTE` colour; an unlabelled
selection is the one crimson "selection" entry it always was. The two renderers draw
from the same grouping, so each claim is checked on both.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ocean_skill.comparison import Comparison
from ocean_skill.field import Field
from ocean_skill.plot.locations import (
    HOVER_FIELDS,
    SELECTION_PALETTE,
    legend_groups,
    style_for,
)
from ocean_skill.plot.map_locations import (
    build_map_items,
    location_items,
    map_locations,
)

META = {
    "geospatial_lon_min": 150.0,
    "geospatial_lat_min": -30.0,
    "geospatial_lon_max": 250.0,
    "geospatial_lat_max": 30.0,
    "featureType": "grid",
}


@pytest.fixture(autouse=True)
def _resolvable_source():
    """Every source resolves to ``META``; the variable-resolution warning is noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch(
            "ocean_skill.catalog.resolve", lambda s: SimpleNamespace(metadata=META)
        ):
            yield


def _box(label, lo, hi):
    return Field(
        "src",
        "temperature",
        label=label,
        select={"lon": {"min": lo, "max": hi}, "lat": {"min": -5.0, "max": 5.0}},
    )


def _three_fields():
    """Three labelled selections: a lat=0 transect line and two boxes."""
    eq = Field("src", "temperature", label="Eq", select={"lat": 0.0})
    return [eq, _box("180-160", 180.0, 200.0), _box("160-120", 200.0, 240.0)]


def _static_axes(fig):
    return next(ax for ax in fig.axes if ax.get_legend() is not None)


def _legend(ax):
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()], [
        h.get_color() for h in legend.legend_handles
    ]


def _drawn_colors(ax):
    return {ln.get_color() for ln in ax.lines}


def _bokeh_legend(obj):
    import holoviews as hv

    return [it.label["value"] for it in hv.render(obj, backend="bokeh").legend[0].items]


def _bokeh_colors(obj):
    """Each legend entry's glyph line colour, by label."""
    import holoviews as hv

    out = {}
    for it in hv.render(obj, backend="bokeh").legend[0].items:
        glyph = it.renderers[0].glyph
        out[it.label["value"]] = getattr(glyph, "line_color", None)
    return out


# -- items carry the label ------------------------------------------------------------


def test_selection_items_carry_legend_label_and_footprint_items_do_not():
    items = location_items(_three_fields())
    selections = [it for it in items if it["featureType"] == "selection"]
    assert [it["legend_label"] for it in selections] == ["Eq", "180-160", "160-120"]
    assert all("legend_label" not in it for it in items if it not in selections)


def test_unlabelled_field_has_no_legend_label_but_keeps_its_hover_name():
    (item,) = [
        it
        for it in build_map_items(Field("src", "temperature", select={"lat": 0.0}))
        if it["featureType"] == "selection"
    ]
    assert item["legend_label"] is None
    assert item["name"]  # the variable-name fallback is hover-only


def test_pair_spec_comparison_suffixes_each_lane_of_the_label():
    c = Comparison(
        reference="ref_src",
        test="src",
        variable="temperature",
        label="x",
        select={"test": {"lat": 0.0}, "reference": {"lat": 2.0}},
    )
    items = [it for it in build_map_items(c) if it["featureType"] == "selection"]
    assert [it["legend_label"] for it in items] == ["x (test)", "x (reference)"]


def test_unlabelled_pair_spec_comparison_has_no_legend_label():
    c = Comparison(
        reference="ref_src",
        test="src",
        variable="temperature",
        select={"test": {"lat": 0.0}, "reference": {"lat": 2.0}},
    )
    items = [it for it in build_map_items(c) if it["featureType"] == "selection"]
    assert [it["legend_label"] for it in items] == [None, None]


# -- static renderer ------------------------------------------------------------------


def test_static_three_labelled_selections_are_three_distinct_entries():
    ax = _static_axes(map_locations(_three_fields()))

    labels, colors = _legend(ax)
    # each field adds its own domain ring; the selections still read as one block
    assert labels == ["Eq", "180-160", "160-120", "domain"]
    assert len(labels) == len(set(labels))  # the line/box pair no longer doubles up
    sel_colors = [c for lab, c in zip(labels, colors, strict=True) if lab != "domain"]
    assert sel_colors == list(SELECTION_PALETTE[:3])
    # ... and those are the colours the shapes were actually drawn in
    assert set(sel_colors) <= _drawn_colors(ax)
    assert "selection" not in labels


def test_static_labelled_and_unlabelled_selection_are_two_entries():
    fields = [
        _box("west", 180.0, 200.0),
        Field("src", "temperature", select={"lat": 0.0}),
    ]
    ax = _static_axes(map_locations(fields))

    labels, colors = _legend(ax)
    by_label = dict(zip(labels, colors, strict=True))
    assert [lab for lab in labels if lab != "domain"] == ["west", "selection"]
    assert by_label["west"] != by_label["selection"]
    assert {by_label["west"], by_label["selection"]} <= _drawn_colors(ax)


def test_static_lone_unlabelled_selection_is_the_crimson_selection_entry():
    ax = _static_axes(map_locations(Field("src", "temperature", select={"lat": 0.0})))

    labels, colors = _legend(ax)
    assert dict(zip(labels, colors, strict=True))["selection"] == "crimson"


def test_static_lone_labelled_selection_stays_crimson():
    ax = _static_axes(map_locations(_box("only", 180.0, 200.0)))

    labels, colors = _legend(ax)
    assert dict(zip(labels, colors, strict=True))["only"] == "crimson"


def test_static_pair_spec_lanes_get_test_and_reference_entries():
    c = Comparison(
        reference="ref_src",
        test="src",
        variable="temperature",
        label="x",
        select={"test": {"lat": 0.0}, "reference": {"lat": 2.0}},
    )
    ax = _static_axes(map_locations(c))

    labels, colors = _legend(ax)
    assert [lab for lab in labels if lab != "domain"] == ["x (test)", "x (reference)"]
    assert len(set(colors)) == len(colors)


def test_static_field_map_overlay_uses_the_labels_too():
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    lat = np.linspace(-30.0, 30.0, 12)
    lon = np.linspace(150.0, 250.0, 20)
    import xarray as xr

    da = xr.DataArray(
        np.add.outer(lat, lon) * 0.0,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "m"},
    )
    facet = {
        "field": da,
        "facet_dim": None,
        "row_dim": None,
        "units": "m",
        "standard_name": "sea_floor_depth_below_geoid",
    }
    items = location_items(_three_fields(), domain=None)
    fig = render(PlotSpec("field_facet", [facet], {"location_items": items}))

    ax = next(a for a in fig.axes if hasattr(a, "projection"))
    labels, _ = _legend(ax)
    assert labels == ["Eq", "180-160", "160-120"]


# -- interactive renderer -------------------------------------------------------------


def test_interactive_three_labelled_selections_are_one_entry_each():
    obj = map_locations(_three_fields(), renderer="holoviews", tiles=None)

    labels = _bokeh_legend(obj)
    assert [lab for lab in labels if lab != "domain"] == ["Eq", "180-160", "160-120"]
    assert len(labels) == len(set(labels))
    colors = _bokeh_colors(obj)
    assert [colors[lab] for lab in ("Eq", "180-160", "160-120")] == list(
        SELECTION_PALETTE[:3]
    )


def test_interactive_labelled_and_unlabelled_selection():
    fields = [
        _box("west", 180.0, 200.0),
        Field("src", "temperature", select={"lat": 0.0}),
    ]
    obj = map_locations(fields, renderer="holoviews", tiles=None)

    labels = _bokeh_legend(obj)
    assert [lab for lab in labels if lab != "domain"] == ["west", "selection"]
    colors = _bokeh_colors(obj)
    assert colors["west"] != colors["selection"]


def test_interactive_pair_spec_lanes():
    c = Comparison(
        reference="ref_src",
        test="src",
        variable="temperature",
        label="x",
        select={"test": {"lat": 0.0}, "reference": {"lat": 2.0}},
    )
    labels = _bokeh_legend(map_locations(c, renderer="holoviews", tiles=None))
    assert [lab for lab in labels if lab != "domain"] == ["x (test)", "x (reference)"]


def test_interactive_group_spanning_two_kinds_is_still_one_legend_entry():
    """Bokeh files same-label glyphs under one legend item; no extra bookkeeping."""
    import holoviews as hv

    from ocean_skill.plot.holoviews_renderer import _extension, _location_elements

    _extension()
    hover = dict.fromkeys(HOVER_FIELDS, "")
    items = [
        {
            **hover,
            "kind": "extent",
            "featureType": "selection",
            "legend_label": "X",
            "bboxes": [(0.0, 0.0, 5.0, 5.0)],
        },
        {
            **hover,
            "kind": "line",
            "featureType": "selection",
            "legend_label": "X",
            "paths": [np.array([[0.0, 0.0], [3.0, 3.0]])],
        },
        {
            **hover,
            "kind": "point",
            "featureType": "selection",
            "legend_label": "Y",
            "lon": 4.0,
            "lat": 4.0,
        },
    ]
    elements = _location_elements(
        items, xform=lambda x, y: (x, y), marker_size=9.0, legend=True
    )
    legend = hv.render(hv.Overlay(elements), backend="bokeh").legend[0]
    assert [it.label["value"] for it in legend.items] == ["X", "Y"]


# -- legend_groups --------------------------------------------------------------------


def _item(kind, feature_type, legend_label=None, **extra):
    item = {"kind": kind, "featureType": feature_type, **extra}
    if legend_label is not None:
        item["legend_label"] = legend_label
    return item


def test_legend_groups_orders_feature_types_then_selections_then_rest():
    items = [
        _item("ring", "domain"),
        _item("line", "selection", "B"),
        _item("point", "timeSeries"),
        _item("extent", "grid"),
        _item("line", "selection", "A"),
        _item("line", "selection"),
    ]
    groups = legend_groups(items)

    assert [g[0] for g in groups] == [
        "grid",
        "timeSeries",
        "B",
        "A",
        "selection",
        "domain",
    ]
    # members are the items themselves, grouped
    assert [len(g[2]) for g in groups] == [1, 1, 1, 1, 1, 1]


def test_legend_groups_groups_by_label_across_kinds():
    items = [
        _item("extent", "selection", "X"),
        _item("line", "selection", "X"),
        _item("line", "selection", "Y"),
    ]
    (x, y) = legend_groups(items)
    assert x[0] == "X" and [i["kind"] for i in x[2]] == ["extent", "line"]
    assert y[0] == "Y"


def test_legend_groups_cycles_selection_colours_only_among_selection_groups():
    items = [_item("point", "timeSeries"), _item("ring", "domain")] + [
        _item("line", "selection", f"s{i}") for i in range(8)
    ]
    groups = {g[0]: g[1] for g in legend_groups(items)}

    assert groups["timeSeries"] == style_for("timeSeries")
    assert groups["domain"] == style_for("domain")  # not drawn from the palette
    colors = [groups[f"s{i}"]["color"] for i in range(8)]
    assert colors == [SELECTION_PALETTE[i % len(SELECTION_PALETTE)] for i in range(8)]
    # marker / linestyle stay the selection group's own
    assert groups["s3"]["marker"] == style_for("selection")["marker"]
    assert groups["s3"]["linestyle"] == "-"


def test_legend_groups_unlabelled_selection_counts_as_a_selection_group():
    items = [_item("line", "selection", "A"), _item("line", "selection")]
    groups = {g[0]: g[1]["color"] for g in legend_groups(items)}
    assert groups == {"A": "crimson", "selection": SELECTION_PALETTE[1]}


def test_legend_groups_lone_selection_is_crimson_and_none_label_falls_back():
    (group,) = legend_groups([_item("line", "selection", None)])
    assert group[0] == "selection"
    assert group[1]["color"] == "crimson"


def test_legend_groups_styles_are_copies():
    (group,) = legend_groups([_item("line", "selection")])
    group[1]["color"] = "pink"
    assert style_for("selection")["color"] == "crimson"
    assert legend_groups([]) == []


def test_selection_palette_avoids_tab10():
    from ocean_skill.plot.locations import TAB10

    assert not set(SELECTION_PALETTE) & set(TAB10)
