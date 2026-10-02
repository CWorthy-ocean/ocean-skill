"""``contours=``: a second variable's section, paired with the panels it is drawn over.

Only the pairing is tested here -- which overlay lands on which plotted item, and the
refusals -- by capturing the :class:`~ocean_skill.plot.spec.PlotSpec` handed to the
renderer registry. Drawing the lines is the renderers' business (see the section
contour tests).
"""

from __future__ import annotations

import pytest

import ocean_skill as osk
from ocean_skill.comparison import ComparisonSet
from ocean_skill.field import FieldSet
from tests.test_combine import (  # noqa: F401  (captured is a fixture)
    _section_comparisons,
    _slab_fields,
    captured,
)
from tests.test_section_comparison import patched_sources  # noqa: F401  (fixture)
from tests.test_slab_section import BOX, DEPTHS, VAR, patched_woa  # noqa: F401


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


def test_an_overlay_on_a_map_is_refused(patched_woa, captured):
    name = patched_woa()
    surface = osk.field(name, VAR, select={**BOX, "depth": DEPTHS[0]}, cache=False)
    assert surface.family != "section"
    first, _ = _slab_fields(patched_woa)
    with pytest.raises(ValueError, match="over a vertical section"):
        surface.plot(contours=first)


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
