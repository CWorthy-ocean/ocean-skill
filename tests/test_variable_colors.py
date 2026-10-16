"""A variable keeps one colour on every page of a report, in both renderers.

``color_by="variable"`` (the default of the summary diagrams and the series family)
used to hand out colours by position in the figure's own legend, so nitrate was green
on one page and chlorophyll green on the next. The colour now comes from which
variable it is (:func:`ocean_skill.plot.style.variable_colors`): a fixed palette for
the variables the package knows, a deterministic hash for the rest.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest

from ocean_skill.plot import style
from ocean_skill.plot.summary import _resolve_colors, target
from tests.test_summary_style import _FakeComparison

NAMES = [
    "sea_water_potential_temperature",
    "sea_water_practical_salinity",
    "mole_concentration_of_nitrate_in_sea_water",
    "sea_water_alkalinity_expressed_as_mole_equivalent",
    "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
    "mass_concentration_of_chlorophyll_a_in_sea_water",
]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_known_variable_has_one_colour_whatever_else_is_in_the_figure():
    everything = style.variable_colors(NAMES)
    for name in NAMES:
        alone = style.variable_colors([name])
        assert alone[name] == everything[name]
        reversed_ = style.variable_colors(list(reversed(NAMES)))
        assert reversed_[name] == everything[name]
    assert len(set(everything.values())) == len(NAMES)


def test_one_variable_spelled_several_ways_is_one_colour():
    spellings = ["nitrate", "mole_concentration_of_nitrate_in_sea_water", "NO3"]
    assert len({style.variable_colors([s])[s] for s in spellings}) == 1
    assert style.variable_colors(["temperature"])["temperature"] == style.COLOR_CYCLE[0]


def test_chlorophyll_and_nitrate_never_share_a_colour_by_page_order():
    page_a = style.variable_colors(["nitrate", "chlorophyll_a_in_sea_water"])
    page_b = style.variable_colors(
        ["chlorophyll_a_in_sea_water", "alkalinity", "nitrate"]
    )
    assert page_a["nitrate"] == page_b["nitrate"]
    assert page_a["chlorophyll_a_in_sea_water"] == page_b["chlorophyll_a_in_sea_water"]


def test_an_unknown_variable_is_deterministic_and_clear_of_the_known_ones():
    first = style.variable_colors(["temperature", "my_tracer_x"])
    again = style.variable_colors(["my_tracer_x", "temperature"])
    assert first["my_tracer_x"] == again["my_tracer_x"]
    taken = {style.COLOR_CYCLE[i] for i in style.VARIABLE_COLOR_SLOTS.values()}
    assert first["my_tracer_x"] not in taken
    # pinned: md5, not hash() -- Python randomizes that per process
    assert style.variable_colors(["my_tracer_x"])["my_tracer_x"] == first["my_tracer_x"]


def test_unknown_variables_that_collide_are_moved_apart_within_a_figure():
    many = [f"tracer_{i}" for i in range(25)]
    colors = style.variable_colors(many)
    assert len(set(colors.values())) == len(style.COLOR_CYCLE)  # every slot used
    assert len(set(list(colors.values())[: len(style.COLOR_CYCLE)])) == len(
        style.COLOR_CYCLE
    )


def test_a_level_that_is_not_a_name_keeps_the_positional_colour():
    band = {"min": 0, "max": 10}
    out = _resolve_colors(None, ["temperature", band], "variable")
    assert out["temperature"] == style.COLOR_CYCLE[0]
    assert out[band] == style.COLOR_CYCLE[1]
    # and other fields are untouched: still positional
    other = _resolve_colors(None, ["b", "a"], "source")
    assert [other["b"], other["a"]] == list(style.COLOR_CYCLE[:2])


def _comparisons(variables):
    return [
        _FakeComparison(f"c{i}", 0.9, 1.0, 0.1 * i, 0.3, v)
        for i, v in enumerate(variables)
    ]


def _static_colors(variables):
    fig = target(_comparisons(variables), color_by="variable", legend_style=None)
    cols = [mcolors.to_hex(c.get_facecolor()[0]) for c in fig.axes[0].collections]
    return dict(zip(variables, cols, strict=True))


def _interactive_colors(variables):
    import holoviews as hv

    from ocean_skill.plot.holoviews_renderer import _target

    items = [
        {"label": c.label, "metrics": c.metrics()} for c in _comparisons(variables)
    ]
    obj = _target(items, color_by="variable", legend_style="legend")
    points = [e for e in obj if isinstance(e, hv.Points)]
    hv.render(obj, backend="bokeh")  # a holoviews bug can surface only at render time
    return {
        v: p.opts.get("style").kwargs["color"]
        for v, p in zip(variables, points, strict=True)
    }


def test_target_colours_a_variable_the_same_on_every_page_in_both_renderers():
    page_a = ["nitrate", "chlorophyll_a_in_sea_water"]
    page_b = [
        "chlorophyll_a_in_sea_water",
        "sea_water_potential_temperature",
        "nitrate",
    ]
    for colors_of in (_static_colors, _interactive_colors):
        a, b = colors_of(page_a), colors_of(page_b)
        assert mcolors.to_hex(a["nitrate"]) == mcolors.to_hex(b["nitrate"])
        assert mcolors.to_hex(a["chlorophyll_a_in_sea_water"]) == mcolors.to_hex(
            b["chlorophyll_a_in_sea_water"]
        )
    static = _static_colors(page_b)
    interactive = _interactive_colors(page_b)
    assert {k: mcolors.to_hex(v) for k, v in static.items()} == {
        k: mcolors.to_hex(v) for k, v in interactive.items()
    }


def test_series_lines_take_the_same_colours():
    from ocean_skill.plot.series import line_specs
    from tests.test_series_renderers import SALINITY, TEMPERATURE, _item

    def colours(items):
        specs = [s for i, item in enumerate(items) for s in line_specs(item, i)]
        return {line.spec.variable: line.color for line in style.resolve(specs)}

    both = colours([_item(TEMPERATURE), _item(SALINITY, units="PSU")])
    swapped = colours([_item(SALINITY, units="PSU"), _item(TEMPERATURE)])
    assert both == swapped
    assert both[TEMPERATURE] == style.COLOR_CYCLE[0]
    assert both[SALINITY] == style.COLOR_CYCLE[1]
