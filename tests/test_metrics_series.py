"""Metrics series: ``ComparisonSet.metrics_series()`` -- metrics as lines against time.

Fed by a ``compare(..., times=...)`` fan-out: one comparison per time bin, each metric
record carrying that bin's ``time``. Offline throughout -- like ``test_pooling`` and
``test_summary_arrows`` the diagrams read only ``metrics()``/``label``, so a stand-in
carrying a metric record exercises the real code paths, the real ``ComparisonSet``
method and both renderers included.

Both renderers are drawn from one :func:`~ocean_skill.plot.metrics_series.build_layout`,
so most of the contract (grouping, order, panels) is asserted there; each renderer's own
test then checks that what it drew matches. The holoviews tests call ``hv.render``:
some holoviews failures only surface when bokeh builds the figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.comparison import ComparisonSet
from ocean_skill.plot.metrics_series import build_layout, time_of

TEMP = "sea_water_temperature"
SALT = "sea_water_salinity"
MONTHS = ["2010-01", "2010-02", "2010-03", "2010-04"]


class _FakeComparison:
    """A comparison's metric record plus a label, and nothing else."""

    def __init__(self, label, *, units=None, **record):
        self.label = label
        self.units = units
        # what pooling's identity check (ocean_skill.comparison._identity) reads, so
        # the stand-ins survive ComparisonSet's flattening as distinct comparisons
        merged = {"variable": TEMP, "depth": 0, "test": "runA", "reference": "runB"}
        merged.update(record)
        self.variable = self.standard_name = merged["variable"]
        self.select = {"depth": merged["depth"]}
        if record.get("time") is not None:
            self.select["time"] = record["time"]
        self.test_name, self.reference_name = merged["test"], merged["reference"]
        self.aggregate = None
        self.method = "conservative_normed"
        self.over = None
        self.time_method = "auto"
        self.tolerance = None
        self.bin_anchor = "auto"
        if units is not None:
            # what ComparisonSet._metric_items reads the reference's units from
            self.aligned = {"reference": xr.DataArray(0.0, attrs={"units": units})}
        self._record = {
            "variable": TEMP,
            "depth": 0,
            "test": "runA",
            "reference": "runB",
            **record,
        }

    def metrics(self):
        return self._record

    def as_item(self):
        return {"metrics": self._record, "label": self.label}


def _fan(variable=TEMP, months=MONTHS, base=1.0, units=None, **extra):
    """One variable's monthly fan-out: rmse grows with time, bias shrinks."""
    return [
        _FakeComparison(
            f"{variable} {month}",
            units=units,
            variable=variable,
            time=month,
            rmse=base * (i + 1),
            bias=base * (2.0 - i),
            corr=0.5 + 0.1 * i,
            n=10 + i,
            **extra,
        )
        for i, month in enumerate(months)
    ]


def _records(comparisons):
    return [{**c.metrics(), "label": c.label, "units": c.units} for c in comparisons]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# -- the shared layout -----------------------------------------------------------------


def test_time_of_reads_periods_at_their_midpoint():
    assert time_of("2010-01") == pd.Timestamp("2010-01-16 12:00")
    assert time_of({"min": "2010-01-01", "max": "2010-01-31"}) == pd.Timestamp(
        "2010-01-16"
    )
    assert time_of("2010-01-01–2010-01-31") == pd.Timestamp("2010-01-16")
    assert time_of("2010-01-15 06:00") == pd.Timestamp("2010-01-15 06:00")


def test_time_of_reads_a_360_day_calendar_date():
    assert time_of("2000-02-30").year == 2000


def test_time_of_refuses_a_season_name():
    with pytest.raises(ValueError, match="not a date"):
        time_of("MAM")


def test_one_panel_per_metric_in_order_with_time_ordered_lines():
    # records deliberately shuffled: the line must still run chronologically
    comps = _fan()
    shuffled = [comps[2], comps[0], comps[3], comps[1]]
    layout = build_layout(_records(shuffled), metrics=("rmse", "bias"))
    assert [p.metric for p in layout.panels] == ["rmse", "bias"]
    assert (layout.nrows, layout.ncols) == (2, 1)
    (line,) = layout.panels[0].lines
    assert list(line.y) == [1.0, 2.0, 3.0, 4.0]
    assert np.all(np.diff(line.x.astype("int64")) > 0)
    (bias,) = layout.panels[1].lines
    assert list(bias.y) == [2.0, 1.0, 0.0, -1.0]


def test_color_by_variable_gives_one_line_per_variable():
    comps = _fan(TEMP) + _fan(SALT, base=0.1)
    layout = build_layout(_records(comps), metrics=("corr",))
    assert len(layout.panels) == 1
    assert len(layout.panels[0].lines) == 2
    assert {ln.label for ln in layout.panels[0].lines} == {"temp", "salinity"} or len(
        layout.legend
    ) == 2
    assert len({ln.color for ln in layout.panels[0].lines}) == 2


def test_different_units_split_a_unit_bearing_metric_into_columns():
    recs = _records(_fan(TEMP, units="°C") + _fan(SALT, units="psu"))
    layout = build_layout(recs, metrics=("rmse", "corr"))
    by_metric = {}
    for p in layout.panels:
        by_metric.setdefault(p.metric, []).append(p)
    # rmse has units: one panel per unit string; corr is unitless: one panel, both lines
    assert len(by_metric["rmse"]) == 2
    assert [len(p.lines) for p in by_metric["rmse"]] == [1, 1]
    assert [p.col for p in by_metric["rmse"]] == [0, 1]
    assert "°C" in by_metric["rmse"][0].title
    assert "psu" in by_metric["rmse"][1].title
    assert len(by_metric["corr"]) == 1 and len(by_metric["corr"][0].lines) == 2
    assert by_metric["corr"][0].title == "corr"
    assert (layout.nrows, layout.ncols) == (2, 2)
    assert not layout.uniform


def test_shared_units_stay_on_one_axis():
    recs = _records(_fan(TEMP, units="°C") + _fan("sea_water_x", units="°C"))
    layout = build_layout(recs, metrics=("rmse",))
    assert len(layout.panels) == 1 and len(layout.panels[0].lines) == 2


def test_rows_variable_transposes_the_grid():
    recs = _records(_fan(TEMP, units="°C") + _fan(SALT, units="psu"))
    layout = build_layout(recs, metrics=("rmse", "bias"), rows="variable")
    assert (layout.nrows, layout.ncols) == (2, 2)
    assert [(p.row, p.col, p.metric) for p in layout.panels] == [
        (0, 0, "rmse"),
        (0, 1, "bias"),
        (1, 0, "rmse"),
        (1, 1, "bias"),
    ]
    assert "°C" in layout.panels[0].title and "psu" in layout.panels[2].title


def test_marker_by_splits_lines_further():
    comps = _fan(test="runA") + _fan(test="runB", base=2.0)
    layout = build_layout(_records(comps), metrics=("rmse",), marker_by="test")
    lines = layout.panels[0].lines
    assert len(lines) == 2
    assert {ln.marker_index for ln in lines} == {0, 1}
    # one colour (one variable), told apart by marker alone
    assert len({ln.color for ln in lines}) == 1


def test_color_by_depth_draws_a_line_per_depth():
    comps = _fan(depth=0) + _fan(depth=50, base=3.0)
    layout = build_layout(_records(comps), metrics=("rmse",), color_by="depth")
    assert len(layout.panels[0].lines) == 2


def test_two_records_at_one_time_in_one_line_raise():
    comps = _fan(depth=0) + _fan(depth=50)
    with pytest.raises(ValueError, match="same time"):
        build_layout(_records(comps), metrics=("rmse",))


def test_records_without_a_time_raise_naming_times():
    comps = [_FakeComparison("a", rmse=1.0, bias=0.0, time=None)]
    with pytest.raises(ValueError, match=r"times="):
        build_layout(_records(comps))
    with pytest.raises(ValueError, match=r"times="):
        ComparisonSet(comps).metrics_series()


def test_unknown_metric_and_rows_raise():
    recs = _records(_fan())
    with pytest.raises(ValueError, match="no metric"):
        build_layout(recs, metrics=("nonesuch",))
    with pytest.raises(ValueError, match="rows="):
        build_layout(recs, rows="bogus")


def test_undefined_metric_value_is_a_gap_not_a_crash():
    comps = _fan()
    comps[1]._record["rmse"] = float("nan")
    (line,) = build_layout(_records(comps), metrics=("rmse",)).panels[0].lines
    assert np.isnan(line.y[1]) and line.y[0] == 1.0


# -- static renderer -------------------------------------------------------------------


def _axes_with_data(fig):
    return [ax for ax in fig.axes if ax.get_visible() and ax.lines]


def _data_lines(ax):
    """Return the axes' metric lines: all but the bias panel's faint zero line."""
    return [ln for ln in ax.lines if ln.get_gid() != "zero"]


def test_static_draws_one_axes_per_metric_with_a_line_per_variable():
    comps = _fan(TEMP) + _fan(SALT, base=0.1)
    fig = ComparisonSet(comps).metrics_series()
    assert isinstance(fig, plt.Figure)
    rmse, bias = _axes_with_data(fig)
    assert rmse.get_title() == "rmse" and bias.get_title() == "bias"
    assert len(_data_lines(rmse)) == 2
    for ln in _data_lines(rmse):
        assert len(ln.get_xdata()) == len(MONTHS)
        assert np.all(np.diff(np.asarray(ln.get_xdata()).astype("int64")) > 0)
    # bias gets a faint zero line besides its two data lines
    assert len(bias.lines) == 3 and len(_data_lines(bias)) == 2
    # a shared time axis, and a figure-level key
    assert rmse.get_shared_x_axes().joined(rmse, bias)
    assert fig.legends and len(fig.legends[0].get_texts()) == 2


def test_static_points_follow_the_records_in_time_order():
    comps = _fan()
    shuffled = [comps[3], comps[1], comps[0], comps[2]]
    fig = ComparisonSet(shuffled).metrics_series(metrics="rmse")
    (ax,) = _axes_with_data(fig)
    assert list(ax.lines[0].get_ydata()) == [1.0, 2.0, 3.0, 4.0]


def test_static_units_split_gives_a_grid_with_a_hidden_cell():
    comps = _fan(TEMP, units="°C") + _fan(SALT, units="psu")
    fig = ComparisonSet(comps).metrics_series(metrics=("rmse", "corr"))
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3  # rmse x 2 units, corr x 1; the 4th cell hidden


def test_static_title_save_and_sizes(tmp_path):
    out = tmp_path / "ms.png"
    fig = ComparisonSet(_fan()).metrics_series(
        title="monthly skill", save=out, size="slide", font_scale=1.2
    )
    assert out.exists() and out.stat().st_size > 0
    assert fig._suptitle.get_text() == "monthly skill"
    wide = ComparisonSet(_fan()).metrics_series(size="slide")
    page = ComparisonSet(_fan()).metrics_series()
    assert wide.get_size_inches()[0] > page.get_size_inches()[0]
    explicit = ComparisonSet(_fan()).metrics_series(figsize=(5, 4))
    assert tuple(explicit.get_size_inches()) == (5.0, 4.0)


def test_static_unknown_option_is_refused():
    with pytest.raises(TypeError, match="label_size"):
        ComparisonSet(_fan()).metrics_series(label_size=3)


# -- interactive renderer --------------------------------------------------------------


@pytest.fixture
def hv():
    return pytest.importorskip("holoviews")


def _curves(obj, hv):
    return list(obj.traverse(lambda e: e, [hv.Curve]))


def test_holoviews_layout_has_one_panel_per_metric_and_renders(hv):
    comps = _fan(TEMP) + _fan(SALT, base=0.1)
    obj = ComparisonSet(comps).metrics_series(renderer="holoviews")
    assert isinstance(obj, hv.Layout) and len(obj) == 2
    curves = _curves(obj, hv)
    assert len(curves) == 4  # (rmse, bias) x (temp, salinity)
    for curve in curves:
        times = pd.to_datetime(curve.dimension_values("time"))
        assert len(times) == len(MONTHS) and times.is_monotonic_increasing
    assert hv.render(obj) is not None


def test_holoviews_one_metric_is_an_overlay_and_renders(hv):
    obj = ComparisonSet(_fan()).metrics_series(metrics="rmse", renderer="holoviews")
    assert isinstance(obj, hv.Overlay)
    assert hv.render(obj) is not None


def test_holoviews_matches_static_values_and_order(hv):
    comps = _fan(TEMP) + _fan(SALT, base=0.1)
    static = ComparisonSet(comps).metrics_series()
    live = ComparisonSet(comps).metrics_series(renderer="holoviews")
    ys_static = [
        list(ln.get_ydata()) for ax in _axes_with_data(static) for ln in _data_lines(ax)
    ]
    ys_live = [list(c.dimension_values(1)) for c in _curves(live, hv)]
    assert ys_live == ys_static
    colors_live = [c.opts.get("style").kwargs["color"] for c in _curves(live, hv)]
    colors_static = [
        ln.get_color() for ax in _axes_with_data(static) for ln in _data_lines(ax)
    ]
    assert colors_live == colors_static


def test_holoviews_units_split_and_marker_by_render(hv):
    comps = _fan(TEMP, units="°C", test="runA") + _fan(SALT, units="psu", test="runB")
    obj = ComparisonSet(comps).metrics_series(
        metrics=("rmse", "corr"), marker_by="test", renderer="holoviews"
    )
    assert isinstance(obj, hv.Layout)
    assert hv.render(obj) is not None


def test_holoviews_ragged_unit_split_places_panels_by_row_and_col(hv):
    # corr is unitless (one panel); rmse splits by units into two: a ragged grid whose
    # cells the static figure fills by each panel's (row, col), not in a flat run
    comps = _fan(TEMP, units="°C") + _fan(SALT, units="psu")
    # the records a real plot is built from carry display-spelled units (psu -> PSU)
    from ocean_skill.plot.summary import _records as display_records

    layout = build_layout(display_records(comps), metrics=("corr", "rmse"))
    want = {(p.row, p.col): p.title for p in layout.panels}
    assert want == {(0, 0): "corr", (1, 0): "rmse (°C)", (1, 1): "rmse (PSU)"}

    obj = ComparisonSet(comps).metrics_series(
        metrics=("corr", "rmse"), renderer="holoviews"
    )
    assert isinstance(obj, hv.Layout) and len(obj) == 4  # the empty cell is filled
    from bokeh.models import GridPlot

    grid = hv.render(obj)
    assert isinstance(grid, GridPlot)
    got = {(row, col): fig.title.text for fig, row, col in grid.children}
    assert got == want


def test_holoviews_rows_variable_and_title_render(hv):
    comps = _fan(TEMP) + _fan(SALT)
    obj = ComparisonSet(comps).metrics_series(
        rows="variable", title="skill", renderer="holoviews"
    )
    assert len(obj) == 4
    assert hv.render(obj) is not None


def test_holoviews_without_time_raises(hv):
    comps = [_FakeComparison("a", rmse=1.0, bias=0.0)]
    with pytest.raises(ValueError, match=r"times="):
        ComparisonSet(comps).metrics_series(renderer="holoviews")
