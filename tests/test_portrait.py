"""The portrait plot: a heatmap scoreboard, one cell per (row_by, col_by) pair.

Static and interactive draw from the same shared helpers (:func:`_records`,
:func:`_grid`, :func:`_shared_standard_name`, :func:`_fmt_value`) so the two cannot
disagree on cell values, level order, or which cells are missing — this file tests the
static drawing and those shared helpers directly (fast); ``test_portrait_interactive``
covers the holoviews path (slow, since it imports holoviews/bokeh).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ocean_skill.plot.portrait import (
    _fmt_value,
    _grid,
    _resolve_metric_names,
    _shared_standard_name,
    portrait,
)


class _FakeComparison:
    """Minimal stand-in: portrait only calls ``metrics()`` and reads ``label``."""

    def __init__(self, label, *, test, variable, **metrics):
        self.label = label
        self._metrics = {"test": test, "variable": variable, **metrics}

    def metrics(self):
        return self._metrics


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def comparisons():
    """2 runs x 3 variables, one cell (runB/salinity) deliberately missing."""
    return [
        _FakeComparison(
            "A-temp", test="runA", variable="sea_water_temperature",
            bias=0.1, crmsd=0.3, corr=0.9, sigma_ratio=1.1,
            std_test=1.1, std_reference=1.0,
        ),
        _FakeComparison(
            "B-temp", test="runB", variable="sea_water_temperature",
            bias=0.05, crmsd=0.2, corr=0.95, sigma_ratio=1.0,
            std_test=1.0, std_reference=1.0,
        ),
        _FakeComparison(
            "A-salt", test="runA", variable="sea_water_salinity",
            bias=-0.2, crmsd=0.4, corr=0.8, sigma_ratio=0.9,
            std_test=0.9, std_reference=1.0,
        ),
        # runB/salinity: no comparison -- the deliberately missing cell
        _FakeComparison(
            "A-no3", test="runA", variable="nitrate",
            bias=0.3, crmsd=0.5, corr=0.7, sigma_ratio=1.3,
            std_test=1.3, std_reference=1.0,
        ),
        _FakeComparison(
            "B-no3", test="runB", variable="nitrate",
            # corr came back undefined (too few valid pairs) -- masks the same as
            # a genuinely missing cell, not as a plotted zero
            bias=-0.15, crmsd=0.25, corr=float("nan"), sigma_ratio=0.95,
            std_test=0.95, std_reference=1.0,
        ),
    ]


def _images(fig) -> list:
    return [im for ax in fig.axes for im in ax.get_images()]


def _cell_texts(ax) -> set[str]:
    return {t.get_text() for t in ax.texts}


def _items(comparisons: list) -> list[dict]:
    """Build the ``{metrics, label, units}`` spec items ``_metric_items`` would."""
    return [
        {"metrics": c.metrics(), "label": c.label, "units": None} for c in comparisons
    ]


# --------------------------------------------------------------------------- #
# _grid / _resolve_metric_names / _shared_standard_name / _fmt_value
# --------------------------------------------------------------------------- #


def test_grid_orders_levels_by_first_appearance(comparisons):
    from ocean_skill.plot.summary import _records

    recs = _records(comparisons)
    row_levels, col_levels, _ = _grid(recs, "variable", "test", "bias")
    assert row_levels == ["sea_water_temperature", "sea_water_salinity", "nitrate"]
    assert col_levels == ["runA", "runB"]


def test_grid_masks_a_missing_combination(comparisons):
    from ocean_skill.plot.summary import _records

    recs = _records(comparisons)
    row_levels, col_levels, matrix = _grid(recs, "variable", "test", "bias")
    i = row_levels.index("sea_water_salinity")
    j = col_levels.index("runB")
    assert np.ma.is_masked(matrix[i, j]), "runB/salinity was never given, must mask"


def test_grid_masks_a_present_but_nonfinite_value(comparisons):
    from ocean_skill.plot.summary import _records

    recs = _records(comparisons)
    row_levels, col_levels, matrix = _grid(recs, "variable", "test", "corr")
    i = row_levels.index("nitrate")
    j = col_levels.index("runB")
    assert np.ma.is_masked(matrix[i, j]), "a NaN metric reads as 'no data', not a value"


def test_grid_refuses_two_comparisons_sharing_one_cell(comparisons):
    from ocean_skill.plot.summary import _records

    dup = [
        *comparisons,
        _FakeComparison(
            "A-temp-2", test="runA", variable="sea_water_temperature",
            bias=0.9, crmsd=0.9, corr=0.5, sigma_ratio=0.5,
            std_test=0.5, std_reference=1.0,
        ),
    ]
    recs = _records(dup)
    with pytest.raises(ValueError, match="more than one comparison"):
        _grid(recs, "variable", "test", "bias")


def test_resolve_metric_names_accepts_a_single_string(comparisons):
    from ocean_skill.plot.summary import _records

    recs = _records(comparisons)
    assert _resolve_metric_names(recs, "bias") == ("bias",)


def test_resolve_metric_names_rejects_a_metric_no_record_carries(comparisons):
    from ocean_skill.plot.summary import _records

    recs = _records(comparisons)
    with pytest.raises(ValueError, match="no metric"):
        _resolve_metric_names(recs, "not_a_real_metric")


def test_shared_standard_name_is_none_across_several_variables(comparisons):
    from ocean_skill.plot.summary import _records

    assert _shared_standard_name(_records(comparisons)) is None


def test_shared_standard_name_is_the_one_variable_when_there_is_only_one():
    from ocean_skill.plot.summary import _records

    recs = _records(
        [
            _FakeComparison(
                "A", test="runA", variable="sea_water_temperature",
                bias=0.1, corr=0.9,
            ),
            _FakeComparison(
                "B", test="runB", variable="sea_water_temperature",
                bias=0.2, corr=0.8,
            ),
        ]
    )
    assert _shared_standard_name(recs) == "sea_water_temperature"


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("corr", 0.9123, "0.91"),  # dimensionless: 2 dp
        ("sigma_ratio", 1.0499, "1.05"),
        ("bias", 0.123456, "0.123"),  # same-units-as-variable: 3 sig figs
        ("n", 42.0, "42"),  # count: plain integer
        ("bias", float("nan"), ""),  # non-finite: nothing to write
    ],
)
def test_fmt_value_formats_by_the_metrics_own_units(name, value, expected):
    assert _fmt_value(name, value) == expected


def test_fmt_value_handles_a_masked_cell():
    matrix = np.ma.masked_invalid([np.nan])
    assert _fmt_value("bias", matrix[0]) == ""


# --------------------------------------------------------------------------- #
# portrait() -- the static grid
# --------------------------------------------------------------------------- #


def test_a_single_metric_draws_one_panel_with_one_colorbar(comparisons):
    fig = portrait(comparisons, metric_names="bias")
    assert len(fig.axes) == 2, "one heatmap axes, one colorbar axes"
    assert len(_images(fig)) == 1


def test_several_metrics_draw_small_multiples(comparisons):
    fig = portrait(comparisons, metric_names=("bias", "corr", "crmsd"))
    assert len(_images(fig)) == 3


def test_default_metrics_are_the_four_canonical_statistics(comparisons):
    from ocean_skill.metrics import DEFAULT_MAP_METRICS

    fig = portrait(comparisons)
    assert len(_images(fig)) == len(DEFAULT_MAP_METRICS)


def test_annotate_writes_each_cells_value(comparisons):
    fig = portrait(comparisons, metric_names="bias", annotate=True)
    (ax,) = [a for a in fig.axes if a.get_images()]
    texts = _cell_texts(ax)
    assert "0.1" in texts
    assert "-0.2" in texts
    assert "0.3" in texts
    # the missing (runB/salinity) cell writes nothing
    assert "" not in texts or len(texts) == 5


def test_annotate_off_by_default_writes_no_cell_text(comparisons):
    fig = portrait(comparisons, metric_names="bias")
    (ax,) = [a for a in fig.axes if a.get_images()]
    assert not _cell_texts(ax)


def test_a_missing_cell_reads_as_missing_color_not_a_value(comparisons):
    fig = portrait(comparisons, metric_names="bias", missing_color="#123456")
    (ax,) = [a for a in fig.axes if a.get_images()]
    assert ax.get_facecolor() != (1.0, 1.0, 1.0, 1.0)  # not left at the default white


def test_row_and_column_labels_are_the_pretty_short_names(comparisons):
    fig = portrait(comparisons, metric_names="bias")
    (ax,) = [a for a in fig.axes if a.get_images()]
    row_labels = {t.get_text() for t in ax.get_yticklabels()}
    assert "salinity" in row_labels  # short_name, not "sea_water_salinity"


def test_row_by_and_col_by_accept_any_record_field():
    recs = [
        _FakeComparison("a", test="runA", variable="v", depth=0, bias=0.1, corr=0.9),
        _FakeComparison("b", test="runA", variable="v", depth=10, bias=0.2, corr=0.8),
    ]
    fig = portrait(recs, row_by="depth", col_by="variable", metric_names="bias")
    assert len(_images(fig)) == 1


def test_no_comparisons_is_refused():
    with pytest.raises(ValueError, match="at least one comparison"):
        portrait([], metric_names="bias")


def test_unknown_option_is_rejected_by_the_renderer_dispatch(comparisons):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = _items(comparisons)
    with pytest.raises(TypeError, match="not an option"):
        render(
            PlotSpec(family="portrait", items=items, options={"bogus": 1}),
            renderer="matplotlib",
        )


def test_portrait_is_reachable_through_plotspec_and_render(comparisons):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = _items(comparisons)
    fig = render(
        PlotSpec(family="portrait", items=items, options={"metric_names": "corr"}),
        renderer="matplotlib",
    )
    assert len(_images(fig)) == 1


def test_comparisonset_portrait_and_summary_kind_reach_the_same_family(comparisons):
    from ocean_skill.comparison import _SUMMARY_KINDS, ComparisonSet

    assert _SUMMARY_KINDS["portrait"] == "portrait"

    cset = ComparisonSet.__new__(ComparisonSet)
    cset.comparisons = comparisons
    cset.labels = None
    fig = cset.portrait(metric_names="bias")
    assert len(_images(fig)) == 1


# --------------------------------------------------------------------------- #
# interactive (holoviews/bokeh) -- slow: real imports, real construction
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_interactive_portrait_returns_a_heatmap(comparisons):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = _items(comparisons)
    obj = render(
        PlotSpec(family="portrait", items=items, options={"metric_names": "bias"}),
        renderer="holoviews",
    )
    import holoviews as hv

    assert isinstance(obj, hv.HeatMap)


@pytest.mark.slow
def test_interactive_portrait_several_metrics_returns_a_layout(comparisons):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = _items(comparisons)
    spec = PlotSpec(family="portrait", items=items, options={})
    obj = render(spec, renderer="holoviews")
    import holoviews as hv

    assert isinstance(obj, hv.Layout)


@pytest.mark.slow
def test_interactive_missing_cell_uses_the_missing_color(comparisons):
    import holoviews as hv

    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = _items(comparisons)
    obj = render(
        PlotSpec(
            family="portrait",
            items=items,
            options={"metric_names": "bias", "missing_color": "#123456"},
        ),
        renderer="holoviews",
    )
    bokeh_fig = hv.render(obj, backend="bokeh")
    from bokeh.models import LinearColorMapper

    (mapper,) = bokeh_fig.select(dict(type=LinearColorMapper))
    assert mapper.nan_color == "#123456"


@pytest.mark.slow
def test_interactive_portrait_warns_on_a_static_only_kwarg(comparisons):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = _items(comparisons)
    with pytest.warns(UserWarning, match="annot_kwargs"):
        render(
            PlotSpec(
                family="portrait",
                items=items,
                options={"metric_names": "bias", "annot_kwargs": {"fontsize": 5}},
            ),
            renderer="holoviews",
        )


@pytest.mark.slow
def test_portrait_is_not_delegated_to_matplotlib(comparisons):
    """Unlike taylor/paired, portrait has a real interactive form -- no fallback."""
    import warnings

    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = _items(comparisons)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        render(
            PlotSpec(family="portrait", items=items, options={"metric_names": "bias"}),
            renderer="holoviews",
        )
