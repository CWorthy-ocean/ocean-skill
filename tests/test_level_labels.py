"""``level_labels=``: rename legend/axis levels for one summary figure.

A ``{field: {level: text}}`` dict, accepted by ``summary``/``taylor``/``target``/
``portrait`` in both renderers. A key matches a level's raw record value or its default
display spelling (case- and space-insensitive); a level no key matches keeps its text.
The interactive tests render through bokeh (``hv.render``) rather than just inspecting
the holoviews object, because holoviews bugs can surface only at render time.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pytest

from ocean_skill.plot.portrait import portrait
from ocean_skill.plot.summary import (
    paired,
    pretty_level,
    target,
    taylor,
)


class _FakeComparison:
    """Minimal stand-in: the diagrams call ``metrics()`` and read ``label``."""

    def __init__(self, label, *, variable, time, test="run", **metrics):
        self.label = label
        self._metrics = {
            "variable": variable,
            "time": time,
            "test": test,
            "corr": 0.9,
            "std_test": 1.0,
            "std_reference": 1.0,
            "bias": 0.1,
            "crmsd": 0.3,
            "sigma_ratio": 1.0,
            **metrics,
        }

    def metrics(self):
        return self._metrics


@pytest.fixture
def comparisons():
    """Two variables x two time windows: a 2 x 2 colour x marker grid."""
    return [
        _FakeComparison(
            "t1", variable="sea_water_temperature", time="MAMJJASON", corr=0.95
        ),
        _FakeComparison(
            "t2", variable="sea_water_temperature", time="DJF", corr=0.85, bias=-0.1
        ),
        _FakeComparison("a1", variable="alkalinity", time="MAMJJASON", corr=0.9),
        _FakeComparison(
            "a2", variable="alkalinity", time="DJF", corr=0.8, test="other"
        ),
    ]


@pytest.fixture
def items(comparisons):
    return [{"label": c.label, "metrics": c.metrics()} for c in comparisons]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


LABELS = {"time": {"MAMJJASON": "Mar–Nov"}, "variable": {"alkalinity": "TA"}}


def _legend_texts(fig) -> set[str]:
    return {t.get_text() for lg in fig.legends for t in lg.get_texts()}


def _tick_texts(fig, axis="x") -> list[str]:
    ax = fig.axes[0]
    ticks = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    return [t.get_text() for t in ticks]


# ---------------------------------------------------------------- pretty_level


def test_pretty_level_unchanged_without_labels():
    assert pretty_level("variable", "sea_water_temperature") == "temperature"
    assert pretty_level("variable", "sea_water_temperature", None) == "temperature"
    assert pretty_level("variable", "sea_water_temperature", {}) == "temperature"


def test_pretty_level_matches_raw_or_default_spelling():
    raw = {"variable": {"sea_water_temperature": "T"}}
    pretty = {"variable": {"temperature": "T"}}
    assert pretty_level("variable", "sea_water_temperature", raw) == "T"
    assert pretty_level("variable", "sea_water_temperature", pretty) == "T"


def test_pretty_level_matching_ignores_case_and_spaces():
    labels = {"time": {" mamJJA son": "Mar–Nov"}}
    assert pretty_level("time", "MAMJJASON", labels) == "Mar–Nov"
    labels = {"variable": {"TEMPERATURE": "T"}}
    assert pretty_level("variable", "sea_water_temperature", labels) == "T"
    labels = {"test": {"Run A": "Baseline"}}
    assert pretty_level("test", "run a", labels) == "Baseline"


def test_pretty_level_unmatched_keeps_default_text():
    labels = {"variable": {"alkalinity": "TA"}}
    assert pretty_level("variable", "sea_water_salinity", labels) == "salinity"
    assert pretty_level("test", "run", labels) == "run"


def test_pretty_level_depth_band_does_not_choke():
    band = {"min": 0, "max": 5}
    assert pretty_level("depth", band, {"depth": {"0-5m": "x"}}) == str(band)


# ------------------------------------------------------------ static legends


@pytest.mark.parametrize("diagram", [taylor, target, paired])
def test_flat_legend_uses_the_new_names(diagram, comparisons):
    fig = diagram(
        comparisons,
        color_by="variable",
        marker_by="time",
        legend_style="legend",
        level_labels=LABELS,
    )
    texts = _legend_texts(fig)
    assert {"TA", "Mar–Nov", "temperature", "DJF"} <= texts
    assert "alkalinity" not in texts
    assert "MAMJJASON" not in texts


@pytest.mark.parametrize("diagram", [taylor, target, paired])
def test_grid_legend_uses_the_new_names(diagram, comparisons):
    fig = diagram(
        comparisons,
        color_by="variable",
        marker_by="time",
        legend_style="grid",
        level_labels=LABELS,
    )
    texts = _legend_texts(fig)
    assert {"TA", "Mar–Nov", "temperature", "DJF"} <= texts
    assert "alkalinity" not in texts
    assert "MAMJJASON" not in texts


@pytest.mark.parametrize("diagram", [taylor, target])
def test_marker_only_legend_is_renamed(diagram, comparisons):
    fig = diagram(
        comparisons, marker_by="time", legend_style="legend", level_labels=LABELS
    )
    texts = _legend_texts(fig)
    assert "Mar–Nov" in texts and "MAMJJASON" not in texts


def test_no_level_labels_keeps_todays_text(comparisons):
    fig = taylor(
        comparisons, color_by="variable", marker_by="time", legend_style="grid"
    )
    texts = _legend_texts(fig)
    assert {"alkalinity", "MAMJJASON", "temperature", "DJF"} <= texts


def test_paired_warns_once_for_an_unmatched_level(comparisons):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paired(
            comparisons,
            color_by="variable",
            marker_by="time",
            legend_style="legend",
            level_labels={"time": {"MAMJJSON": "typo"}},
        )
    hits = [w for w in caught if "level_labels" in str(w.message)]
    assert len(hits) == 1


# ------------------------------------------------------------------- portrait


def test_portrait_axis_ticks_use_the_new_names(comparisons):
    fig = portrait(
        comparisons,
        row_by="variable",
        col_by="time",
        metric_names="bias",
        level_labels=LABELS,
    )
    assert _tick_texts(fig, "y") == ["temperature", "TA"]
    assert _tick_texts(fig, "x") == ["Mar–Nov", "DJF"]


def test_portrait_matches_the_default_spelling(comparisons):
    fig = portrait(
        comparisons,
        col_by="time",
        metric_names="bias",
        level_labels={"variable": {"temperature": "Temperature"}},
    )
    assert _tick_texts(fig, "y") == ["Temperature", "alkalinity"]


def test_portrait_without_labels_is_unchanged(comparisons):
    fig = portrait(comparisons, col_by="time", metric_names="bias")
    assert _tick_texts(fig, "y") == ["temperature", "alkalinity"]


# ------------------------------------------------------ keys: raw/pretty/case


@pytest.mark.parametrize(
    "key",
    ["temperature", "sea_water_temperature", "TEMPERATURE", " Sea_Water_ Temperature"],
)
def test_variable_key_spellings_all_match(key, comparisons):
    fig = target(
        comparisons,
        color_by="variable",
        legend_style="legend",
        level_labels={"variable": {key: "T"}},
    )
    assert "T" in _legend_texts(fig) and "temperature" not in _legend_texts(fig)


# ------------------------------------------------------------ errors/warnings


@pytest.mark.parametrize("diagram", [taylor, target, paired, portrait])
def test_unknown_field_raises(diagram, comparisons):
    with pytest.raises(ValueError, match=r"'colour'.*not a field"):
        diagram(comparisons, level_labels={"colour": {"x": "y"}})


@pytest.mark.parametrize("diagram", [taylor, target, paired, portrait])
def test_non_dict_value_raises(diagram, comparisons):
    with pytest.raises(TypeError, match=r"level_labels\['variable'\].*dict"):
        diagram(comparisons, level_labels={"variable": "TA"})


def test_non_dict_level_labels_raises(comparisons):
    with pytest.raises(TypeError, match="level_labels must be"):
        taylor(comparisons, level_labels=["TA"])


def test_groups_field_is_known(comparisons):
    fig = taylor(
        comparisons,
        groups={"x": "lower"},
        legend_style="legend",
        level_labels={"group": {"other": "Everything"}},
    )
    assert "Everything" in _legend_texts(fig)


@pytest.mark.parametrize("diagram", [taylor, target])
def test_unmatched_level_warns_naming_levels_present(diagram, comparisons):
    with pytest.warns(UserWarning, match="levels present") as record:
        diagram(
            comparisons,
            color_by="variable",
            legend_style="legend",
            level_labels={"variable": {"alkalinty": "TA"}},
        )
    msg = str(record[0].message)
    assert "'alkalinty'" in msg and "alkalinity" in msg and "temperature" in msg


def test_portrait_unmatched_level_warns(comparisons):
    with pytest.warns(UserWarning, match="levels present"):
        portrait(
            comparisons,
            col_by="time",
            metric_names="bias",
            level_labels={"variable": {"alkalinty": "TA"}},
        )


def test_unmatched_level_still_draws_with_default_text(comparisons):
    with pytest.warns(UserWarning, match="level_labels"):
        fig = taylor(
            comparisons,
            color_by="variable",
            legend_style="legend",
            level_labels={"variable": {"alkalinty": "TA"}},
        )
    assert {"temperature", "alkalinity"} <= _legend_texts(fig)


def test_matched_levels_do_not_warn(comparisons):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        taylor(
            comparisons,
            color_by="variable",
            marker_by="time",
            legend_style="legend",
            level_labels=LABELS,
        )
    assert not [w for w in caught if "level_labels" in str(w.message)]


# ----------------------------------------------------------- interactive


def _bokeh_legend_labels(obj) -> set[str]:
    import holoviews as hv
    from bokeh.models import Legend

    fig = hv.render(obj, backend="bokeh")
    out = set()
    for legend in fig.select({"type": Legend}):
        for item in legend.items:
            label = item.label
            out.add(str(getattr(label, "value", label)))
    return out


def _interactive_target(items, **kwargs):
    from ocean_skill.plot.holoviews_renderer import _target

    return _target(items, **kwargs)


def test_interactive_target_legend_uses_the_new_names(items):
    obj = _interactive_target(
        items,
        color_by="variable",
        marker_by="time",
        legend_style="legend",
        level_labels=LABELS,
    )
    labels = _bokeh_legend_labels(obj)
    assert {"TA · Mar–Nov", "temperature · DJF"} <= labels
    assert not any("alkalinity" in s or "MAMJJASON" in s for s in labels)


def test_interactive_target_color_only_legend(items):
    obj = _interactive_target(
        items, color_by="variable", legend_style="legend", level_labels=LABELS
    )
    labels = _bokeh_legend_labels(obj)
    assert {"TA", "temperature"} <= labels and "alkalinity" not in labels


def test_interactive_target_without_labels_is_unchanged(items):
    obj = _interactive_target(
        items, color_by="variable", marker_by="time", legend_style="legend"
    )
    labels = _bokeh_legend_labels(obj)
    assert {"alkalinity · MAMJJASON", "temperature · DJF"} <= labels


def test_interactive_target_matches_default_spelling_and_case(items):
    obj = _interactive_target(
        items,
        color_by="variable",
        legend_style="legend",
        level_labels={"variable": {"TEMPERATURE": "T"}},
    )
    assert "T" in _bokeh_legend_labels(obj)


def test_interactive_target_errors_and_warnings(items):
    with pytest.raises(ValueError, match="not a field"):
        _interactive_target(items, level_labels={"colour": {"x": "y"}})
    with pytest.raises(TypeError, match="dict"):
        _interactive_target(items, level_labels={"variable": "TA"})
    with pytest.warns(UserWarning, match="levels present"):
        _interactive_target(
            items,
            color_by="variable",
            legend_style="legend",
            level_labels={"variable": {"alkalinty": "TA"}},
        )


def test_interactive_target_groups_field_is_known(items):
    obj = _interactive_target(
        items,
        groups={"x": "lower"},
        legend_style="legend",
        level_labels={"group": {"other": "Everything"}},
    )
    assert "Everything" in _bokeh_legend_labels(obj)


def test_summary_through_the_registry_in_both_renderers(items):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    spec = PlotSpec(
        family="target",
        items=items,
        options={
            "color_by": "variable",
            "marker_by": "time",
            "legend_style": "legend",
            "level_labels": LABELS,
        },
    )
    static = render(spec, renderer="matplotlib")
    assert {"TA", "Mar–Nov"} <= _legend_texts(static)
    assert "TA · Mar–Nov" in _bokeh_legend_labels(render(spec, renderer="holoviews"))


def test_interactive_taylor_delegates_and_keeps_the_labels(items):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    spec = PlotSpec(
        family="taylor",
        items=items,
        options={
            "color_by": "variable",
            "marker_by": "time",
            "legend_style": "grid",
            "level_labels": LABELS,
        },
    )
    with pytest.warns(UserWarning, match="no interactive form"):
        fig = render(spec, renderer="holoviews")
    assert {"TA", "Mar–Nov"} <= _legend_texts(fig)


def _heatmap_categories(obj):
    import holoviews as hv

    fig = hv.render(obj, backend="bokeh")
    from bokeh.models import FactorRange

    xs, ys = [], []
    for rng in fig.select({"type": FactorRange}):
        xs.append(list(rng.factors))
    return xs, ys


def test_interactive_portrait_categories_use_the_new_names(items):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    spec = PlotSpec(
        family="portrait",
        items=items,
        options={
            "row_by": "variable",
            "col_by": "time",
            "metric_names": "bias",
            "level_labels": LABELS,
        },
    )
    factors, _ = _heatmap_categories(render(spec, renderer="holoviews"))
    flat = {f for fs in factors for f in fs}
    assert {"Mar–Nov", "DJF", "TA", "temperature"} <= flat
    assert not ({"MAMJJASON", "alkalinity"} & flat)


def test_interactive_portrait_errors(items):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    spec = PlotSpec(
        family="portrait",
        items=items,
        options={"metric_names": "bias", "level_labels": {"colour": {"a": "b"}}},
    )
    with pytest.raises(ValueError, match="not a field"):
        render(spec, renderer="holoviews")


def test_a_dict_reused_across_subsets_stays_silent_when_some_levels_match(comparisons):
    # one level_labels dict shared across .sel() subsets names levels a given figure
    # lacks; that's only a typo when none of the field's keys match
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        taylor(
            comparisons,
            color_by="variable",
            legend_style="legend",
            level_labels={"variable": {"alkalinity": "TA", "not_in_this_subset": "X"}},
        )
    assert not [w for w in caught if "level_labels" in str(w.message)]
