"""``osk.plot``: comparisons (or fields) built separately, drawn as one figure.

The function only decides which set to build -- :class:`~ocean_skill.comparison
.ComparisonSet` for comparisons, :class:`~ocean_skill.field.FieldSet` for fields --
and forwards everything else, so these tests check that routing and its refusals,
capturing the :class:`~ocean_skill.plot.spec.PlotSpec` the set hands the renderer
registry rather than depending on a renderer to draw it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog
from ocean_skill.comparison import ComparisonSet
from ocean_skill.field import Field, FieldSet
from ocean_skill.plot import registry
from tests.test_section_comparison import (  # noqa: F401  (patched_sources is a fixture)
    HC,
    _climatology,
    _roms_run,
    patched_sources,
)
from tests.test_slab_section import (  # noqa: F401  (fixtures)
    BOX,
    DEPTHS,
    MEAN_LON,
    ROMS_BOX,
    VAR,
    _woa_like,
    patched_woa,
)


@pytest.fixture
def captured(monkeypatch):
    """Stand in for the renderer registry: record the spec, draw nothing."""
    seen: dict = {}

    def fake_render(spec, **kwargs):
        seen["spec"] = spec
        seen["kwargs"] = kwargs
        return "figure"

    monkeypatch.setattr(registry, "render", fake_render)
    return seen


def _section_comparisons(patched_sources):
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    kwargs = dict(
        test="roms_test", reference="woa_ref", variable=VAR, cache=False
    )
    transect = osk.Comparison(
        label="line",
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
        **kwargs,
    )
    slab = osk.Comparison(
        label="band",
        select={**ROMS_BOX, "depth": [50.0, 200.0]},
        aggregate=MEAN_LON,
        **kwargs,
    )
    return transect, slab


# -- comparisons -----------------------------------------------------------------------


def test_a_list_of_comparisons_is_one_section_row_figure(patched_sources, captured):
    transect, slab = _section_comparisons(patched_sources)
    out = osk.plot([transect, slab], renderer="matplotlib", shared_limits=True)
    assert out == "figure"
    spec = captured["spec"]
    assert spec.family == "section_row"
    assert [item["row_label"] for item in spec.items] == ["line", "band"]
    assert spec.options["shared_limits"] is True
    assert captured["kwargs"] == {"renderer": "matplotlib"}


def test_a_dict_of_comparisons_labels_the_rows_by_its_keys(patched_sources, captured):
    transect, slab = _section_comparisons(patched_sources)
    osk.plot({"Eq": transect, "180-160": slab})
    assert [item["row_label"] for item in captured["spec"].items] == ["Eq", "180-160"]


def test_comparison_sets_and_comparisons_can_be_mixed_in_one_list(
    patched_sources, captured
):
    transect, slab = _section_comparisons(patched_sources)
    osk.plot([ComparisonSet([transect]), slab])
    assert len(captured["spec"].items) == 2


def test_a_single_comparison_is_drawn_as_itself(patched_sources, captured):
    transect, _ = _section_comparisons(patched_sources)
    osk.plot(transect)
    assert len(captured["spec"].items) == 1


def test_the_renderer_choice_reaches_the_registry(patched_sources, captured):
    transect, slab = _section_comparisons(patched_sources)
    osk.plot([transect, slab], renderer="holoviews")
    assert captured["kwargs"] == {"renderer": "holoviews"}


# -- fields -----------------------------------------------------------------------------


def _slab_fields(patched_woa):
    name = patched_woa()
    first = osk.field(
        name, VAR, select={**BOX, "depth": DEPTHS}, aggregate=MEAN_LON, cache=False
    )
    second = osk.field(
        name,
        VAR,
        select={
            "lon": {"min": 200, "max": 240},
            "lat": {"min": -30, "max": 30},
            "depth": DEPTHS,
        },
        aggregate=MEAN_LON,
        cache=False,
    )
    return first, second


def test_a_list_of_section_fields_is_one_section_figure(patched_woa, captured):
    first, second = _slab_fields(patched_woa)
    assert osk.plot([first, second], shared_limits=True) == "figure"
    spec = captured["spec"]
    assert spec.family == "section"
    assert len(spec.items) == 2
    assert spec.options["shared_limits"] is True
    assert all("field" in item for item in spec.items)


def test_a_fieldset_in_the_list_is_flattened(patched_woa, captured):
    first, second = _slab_fields(patched_woa)
    osk.plot([FieldSet([first]), second])
    assert len(captured["spec"].items) == 2


def test_fieldset_plot_stacks_section_fields_directly(patched_woa, captured):
    first, second = _slab_fields(patched_woa)
    FieldSet([first, second]).plot()
    spec = captured["spec"]
    assert spec.family == "section"
    assert [item["label"] for item in spec.items] == ["woa_x", "woa_x"]


def test_a_dict_of_fields_labels_by_key_without_touching_the_originals(
    patched_woa, captured
):
    first, second = _slab_fields(patched_woa)
    first.label = "mine"
    prepared = first.data  # already prepared: the relabelled copy shares it
    osk.plot({"180-200": first, "200-240": second})
    assert [item["label"] for item in captured["spec"].items] == ["180-200", "200-240"]
    # the callers' own fields keep the labels they had, and their prepared data
    assert first.label == "mine"
    assert second.label is None
    assert first._data is prepared


def test_a_dict_group_of_several_fields_is_labelled_key_colon_own(
    patched_woa, captured
):
    first, second = _slab_fields(patched_woa)
    second.label = "b"
    first.label = "a"
    osk.plot({"set": FieldSet([first, second])})
    assert [item["label"] for item in captured["spec"].items] == ["set: a", "set: b"]


def test_a_lone_section_field_draws_through_its_own_plot(patched_woa, captured):
    first, _ = _slab_fields(patched_woa)
    osk.plot([first])
    spec = captured["spec"]
    assert spec.family == "section"
    assert len(spec.items) == 1


def test_a_section_field_set_with_a_further_axis_names_it(monkeypatch, captured):
    ds = _woa_like()
    ds = xr.concat([ds, ds + 1.0], dim="time").assign_coords(time=[0, 1])
    monkeypatch.setattr(osk, "read", lambda n, **kw: ds)
    monkeypatch.setattr(catalog, "resolve", lambda n: SimpleNamespace(metadata={}))
    fields = [
        osk.field(
            "woa_x", VAR, select={**BOX, "depth": DEPTHS}, aggregate=MEAN_LON, cache=False
        )
        for _ in range(2)
    ]
    with pytest.raises(ValueError, match="time"):
        osk.plot(fields)


def test_sections_mixed_with_other_shapes_are_refused_and_say_so(
    patched_woa, captured
):
    first, _ = _slab_fields(patched_woa)
    point = osk.field(
        "woa_x",
        VAR,
        select={"lon": 190.0, "lat": 0.0},
        aggregate={"depth": "mean"},
        cache=False,
    )
    with pytest.raises(ValueError, match="section.*not supported yet"):
        osk.plot([first, point])


# -- refusals -----------------------------------------------------------------------------


def test_an_empty_list_is_refused():
    with pytest.raises(ValueError, match="nothing to draw"):
        osk.plot([])
    with pytest.raises(ValueError, match="nothing to draw"):
        osk.plot({})


def test_comparisons_and_fields_cannot_share_a_figure(patched_sources, patched_woa):
    transect, _ = _section_comparisons(patched_sources)
    field = Field.__new__(Field)  # never prepared: the mix is refused on type alone
    with pytest.raises(ValueError, match="comparisons and fields"):
        osk.plot([transect, field])


def test_something_that_is_neither_is_refused(patched_sources):
    transect, _ = _section_comparisons(patched_sources)
    with pytest.raises(TypeError, match=r"item \[1\] is a str"):
        osk.plot([transect, "temp"])
    with pytest.raises(TypeError, match="item \\['b'\\] is a int"):
        osk.plot({"a": transect, "b": 3})
    with pytest.raises(TypeError, match="takes a list"):
        osk.plot("temp")


def test_osk_plot_does_not_shadow_the_plot_package():
    """``osk.plot`` is callable *and* still the plotting subpackage."""
    import ocean_skill.plot.registry as module_form  # the form a plain function breaks

    assert module_form is registry
    assert osk.plot.registry is registry
    assert callable(osk.plot)
