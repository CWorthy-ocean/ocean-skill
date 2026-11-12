"""``contours=``: a second variable's section, paired with the panels it is drawn over.

Only the pairing is tested here -- which overlay lands on which plotted item, and the
refusals -- by capturing the :class:`~ocean_skill.plot.spec.PlotSpec` handed to the
renderer registry. Drawing the lines is the renderers' business (see the section
contour tests).
"""

from __future__ import annotations

import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill.comparison import ComparisonSet
from ocean_skill.field import FieldSet
from tests.test_combine import (  # noqa: F401  (captured is a fixture)
    _section_comparisons,
    _slab_fields,
    captured,
)
from tests.test_section_comparison import (  # noqa: F401  (patched_sources: fixture)
    HC,
    _climatology,
    _roms_run,
    patched_sources,
)
from tests.test_slab_section import BOX, DEPTHS, VAR, patched_woa  # noqa: F401


def _map(patched_woa, *, depth):
    """Return a map (or a depth-faceted map) of the patched climatology at ``depth``."""
    return osk.field(
        patched_woa(), VAR, select={**BOX, "depth": depth}, cache=False
    )


def test_a_field_overlay_rides_on_the_section_item(patched_woa, captured):
    fill, lines = _slab_fields(patched_woa)
    fill.plot(contours=lines, contour_levels=[1.0, 2.0])
    (item,) = captured["spec"].items
    assert item["contour"] is lines.data
    assert item["contour_standard_name"] == lines.standard_name
    assert item["contour_units"] == lines.data.attrs.get("units")
    assert captured["spec"].options["contour_levels"] == [1.0, 2.0]


def test_a_fieldset_overlay_pairs_by_position(patched_woa, captured):
    first, second = _slab_fields(patched_woa)
    FieldSet([first, second]).plot(contours=FieldSet([second, first]))
    items = captured["spec"].items
    assert items[0]["contour"] is second.data
    assert items[1]["contour"] is first.data


def test_a_list_overlay_is_flattened_like_a_set(patched_woa, captured):
    first, second = _slab_fields(patched_woa)
    FieldSet([first, second]).plot(contours=[FieldSet([second]), first])
    items = captured["spec"].items
    assert items[0]["contour"] is second.data
    assert items[1]["contour"] is first.data


def test_an_overlay_of_the_wrong_length_is_refused(patched_woa, captured):
    first, second = _slab_fields(patched_woa)
    with pytest.raises(ValueError, match=r"1 field.* for 2 panel"):
        FieldSet([first, second]).plot(contours=[first])


def test_an_overlay_of_the_wrong_kind_is_refused(
    patched_woa, patched_sources, captured
):
    first, _ = _slab_fields(patched_woa)
    transect, _ = _section_comparisons(patched_sources)
    with pytest.raises(TypeError, match="same kind of object"):
        first.plot(contours=transect)


def test_a_section_overlay_over_a_map_is_refused(patched_woa, captured):
    surface = _map(patched_woa, depth=0.0)
    assert surface.family == "field_facet"
    first, _ = _slab_fields(patched_woa)
    with pytest.raises(ValueError, match="over a map takes an overlay that is a map"):
        surface.plot(contours=first)


def test_a_map_overlay_over_a_section_is_refused(patched_woa, captured):
    first, _ = _slab_fields(patched_woa)
    surface = _map(patched_woa, depth=0.0)
    with pytest.raises(ValueError, match="over a section takes an overlay"):
        first.plot(contours=surface)


def test_a_map_overlay_rides_on_the_facet_item(patched_woa, captured):
    fill = _map(patched_woa, depth=0.0)
    lines = _map(patched_woa, depth=50.0)
    fill.plot(contours=lines, contour_levels=[1.0, 2.0])
    spec = captured["spec"]
    assert spec.family == "field_facet"
    (item,) = spec.items
    assert isinstance(item["contour"], xr.DataArray)
    assert item["contour"].identical(lines._facet_item_or_refuse()["field"])
    assert item["contour_standard_name"] == lines.standard_name
    assert item["contour_units"] == "degC"
    assert spec.options["contour_levels"] == [1.0, 2.0]


def test_a_map_overlay_pairs_a_faceted_map_panel_by_panel(patched_woa, captured):
    fill = _map(patched_woa, depth=[0.0, 50.0])
    lines = _map(patched_woa, depth=[50.0, 200.0])
    fill.plot(contours=lines)
    (item,) = captured["spec"].items
    assert item["contour"].sizes["depth"] == 2
    assert item["contour"].identical(lines._facet_item_or_refuse()["field"])


def test_a_map_overlay_with_other_facet_axes_is_refused(patched_woa, captured):
    fill = _map(patched_woa, depth=[0.0, 50.0])
    with pytest.raises(ValueError, match="same axes as the map"):
        fill.plot(contours=_map(patched_woa, depth=50.0))
    with pytest.raises(ValueError, match="same axes as the map"):
        fill.plot(contours=_map(patched_woa, depth=[0.0, 50.0, 200.0]))


def test_a_fieldset_of_maps_pairs_overlays_by_position(patched_woa, captured):
    first = _map(patched_woa, depth=0.0)
    second = _map(patched_woa, depth=50.0)
    FieldSet([first, second]).plot(contours=FieldSet([second, first]))
    spec = captured["spec"]
    assert spec.family == "field_map_grid"
    assert spec.items[0]["contour"].identical(second._facet_item_or_refuse()["field"])
    assert spec.items[1]["contour"].identical(first._facet_item_or_refuse()["field"])
    assert all(i["contour_units"] == "degC" for i in spec.items)


def test_a_fieldset_of_maps_split_into_levels_pairs_level_by_level(
    patched_woa, captured
):
    # a vertical axis standing on a set member splits it into one map per level
    first = _map(patched_woa, depth=[0.0, 50.0])
    second = _map(patched_woa, depth=[0.0, 50.0])
    lines = _map(patched_woa, depth=[50.0, 200.0])
    FieldSet([first, second]).plot(contours=[lines, lines])
    items = captured["spec"].items
    assert len(items) == 4
    level = lines._map_items_of(levels=True)
    assert items[0]["contour"].identical(level[0]["field"])
    assert items[1]["contour"].identical(level[1]["field"])
    assert items[2]["contour"].identical(level[0]["field"])


def test_a_member_and_overlay_with_different_level_counts_are_refused(
    patched_woa, captured
):
    first = _map(patched_woa, depth=[0.0, 50.0])
    second = _map(patched_woa, depth=[0.0, 50.0])
    lines = _map(patched_woa, depth=[0.0, 50.0, 200.0])
    with pytest.raises(ValueError, match="draws 2 panel"):
        FieldSet([first, second]).plot(contours=[lines, lines])


def test_a_lone_map_member_hands_its_overlay_to_the_field(patched_woa, captured):
    fill = _map(patched_woa, depth=0.0)
    lines = _map(patched_woa, depth=50.0)
    FieldSet([fill]).plot(contours=FieldSet([lines]))
    spec = captured["spec"]
    assert spec.family == "field_facet"
    assert spec.items[0]["contour"].identical(lines._facet_item_or_refuse()["field"])


def test_a_bare_vertical_map_overlay_gets_the_surface_default(patched_woa, captured):
    # the grid preamble: a catalogued grid source with no vertical request draws the
    # surface, for the map and for its overlay alike
    name = patched_woa(meta={"featureType": "grid"})
    box = {**BOX}
    fill = osk.field(name, VAR, select=box, cache=False)
    lines = osk.field(name, VAR, select=box, cache=False)
    fill.plot(contours=lines)
    (item,) = captured["spec"].items
    assert "depth" not in item["contour"].dims
    assert item["contour"].identical(item["field"])


def test_a_map_overlay_over_a_series_is_refused(patched_woa, captured):
    from tests.test_field_time_depth import _point_time_depth

    ds = _point_time_depth().rename("v").to_dataset()
    name = patched_woa(ds)
    series = osk.field(name, "v", select={"depth": [0.0, 50.0, 100.0]}, cache=False)
    with pytest.raises(ValueError, match="over a vertical section or a map"):
        series.plot(contours=series)


def test_contours_true_is_the_plotted_section_itself(patched_woa, captured):
    first, second = _slab_fields(patched_woa)
    first.plot(contours=True)
    assert captured["spec"].items[0]["contour"] is first.data
    FieldSet([first, second]).plot(contours=True)
    items = captured["spec"].items
    assert items[0]["contour"] is first.data
    assert items[1]["contour"] is second.data
    osk.plot([first, second], contours=True)
    assert captured["spec"].items[1]["contour"] is second.data
    first.plot(contours=False)
    assert "contour" not in captured["spec"].items[0]


def test_contours_true_is_the_plotted_map_itself(patched_woa, captured):
    first = _map(patched_woa, depth=0.0)
    second = _map(patched_woa, depth=50.0)
    first.plot(contours=True)
    (item,) = captured["spec"].items
    assert item["contour"].identical(item["field"])
    FieldSet([first, second]).plot(contours=True)
    items = captured["spec"].items
    assert all(i["contour"].identical(i["field"]) for i in items)
    osk.plot([first, second], contours=True)
    assert captured["spec"].family == "field_map_grid"
    assert all(i["contour"].identical(i["field"]) for i in captured["spec"].items)
    osk.plot({"A": first, "B": second}, contours=True)
    assert len(captured["spec"].items) == 2
    first.plot(contours=False)
    assert "contour" not in captured["spec"].items[0]


def test_osk_plot_pairs_a_dict_overlay_by_label(patched_woa, captured):
    first, second = _slab_fields(patched_woa)
    osk.plot(
        {"A": first, "B": second}, contours={"B": first, "A": second}, mark="contourf"
    )
    items = captured["spec"].items
    assert items[0]["contour"] is second.data
    assert items[1]["contour"] is first.data
    assert captured["spec"].options["mark"] == "contourf"


def test_osk_plot_refuses_a_dict_overlay_with_other_labels(patched_woa, captured):
    first, second = _slab_fields(patched_woa)
    with pytest.raises(ValueError, match="missing \\['B'\\]"):
        osk.plot({"A": first, "B": second}, contours={"A": second})


def test_a_comparison_overlay_carries_its_test_and_reference(
    patched_sources, captured
):
    transect, _ = _section_comparisons(patched_sources)
    transect.plot(contours=transect)
    (item,) = captured["spec"].items
    assert set(item["contour"]) == {"test", "reference"}
    assert item["contour"]["test"].identical(transect.aligned["test"])
    assert item["contour"]["reference"].identical(transect.aligned["reference"])


def test_a_comparison_set_overlay_pairs_rows_by_position(patched_sources, captured):
    transect, slab = _section_comparisons(patched_sources)
    osk.plot([transect, slab], contours=[ComparisonSet([slab]), transect])
    items = captured["spec"].items
    assert items[0]["contour"]["test"].identical(slab.aligned["test"])
    assert items[1]["contour"]["test"].identical(transect.aligned["test"])


def _map_comparisons(patched_sources, *, depth=50.0):
    """Two map rows (model vs climatology at ``depth``), as comparisons."""
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )

    def make(label, select):
        return osk.Comparison(
            label=label,
            test="roms_test",
            reference="woa_ref",
            variable=VAR,
            select=select,
            cache=False,
        )

    return make("a", {"depth": depth}), make("b", {"depth": 200.0})


def test_a_comparison_map_row_carries_its_test_and_reference(
    patched_sources, captured
):
    row, other = _map_comparisons(patched_sources)
    assert row.family == "field_row"
    row.plot(contours=other)
    spec = captured["spec"]
    assert spec.family == "field_row"
    (item,) = spec.items
    assert set(item["contour"]) == {"test", "reference"}
    assert item["contour"]["test"].identical(other.aligned["test"])
    assert item["contour"]["reference"].identical(other.aligned["reference"])
    assert item["contour_standard_name"] == other.standard_name


def test_a_comparison_set_of_map_rows_pairs_per_row(patched_sources, captured):
    row, other = _map_comparisons(patched_sources)
    ComparisonSet([row, other]).plot(contours=ComparisonSet([other, row]))
    spec = captured["spec"]
    assert spec.family == "field_grid"
    assert spec.items[0]["contour"]["test"].identical(other.aligned["test"])
    assert spec.items[1]["contour"]["reference"].identical(row.aligned["reference"])
    # a lone row keeps its own family and its contour keys
    ComparisonSet([row]).plot(contours=ComparisonSet([other]))
    assert captured["spec"].family == "field_row"
    assert set(captured["spec"].items[0]["contour"]) == {"test", "reference"}


def test_comparison_contours_true_is_the_plotted_row(patched_sources, captured):
    row, other = _map_comparisons(patched_sources)
    row.plot(contours=True)
    assert captured["spec"].items[0]["contour"]["test"].identical(row.aligned["test"])
    osk.plot([row, other], contours=True)
    items = captured["spec"].items
    assert items[1]["contour"]["reference"].identical(other.aligned["reference"])
    row.plot(contours=False)
    assert "contour" not in captured["spec"].items[0]


def test_a_section_comparison_overlay_over_a_map_row_is_refused(
    patched_sources, captured
):
    row, _ = _map_comparisons(patched_sources)
    transect, _ = _section_comparisons(patched_sources)
    with pytest.raises(ValueError, match="over a map takes an overlay"):
        row.plot(contours=transect)


def test_contours_true_on_a_map_reuses_the_maps_own_field(patched_woa, captured):
    # the overlay is the field drawn, not a second prepare of the same variable
    fld = _map(patched_woa, depth=0)
    fld.plot(contours=True)
    (item,) = captured["spec"].items
    assert item["contour"] is item["field"]
