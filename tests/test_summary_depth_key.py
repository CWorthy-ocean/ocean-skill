"""A summary coloured by variable says which depth each point is.

Two scores of one variable (surface and 100 m) share a colour under ``color_by=
"variable"``: two dots a reader cannot tell apart, in a key that never mentions depth.
When a colour group spans several depths and the caller named no marker field, the
marker shape carries depth and the key lists it -- in every diagram and both renderers.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from ocean_skill.plot.summary import default_marker_by, paired, target, taylor
from tests.test_summary_style import _FakeComparison


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _at_depths():
    out = []
    for variable, base in (("nitrate", 0.0), ("oxygen", 0.1)):
        for depth in ("surface", 100, 200):
            out.append(
                _FakeComparison(
                    f"{variable} {depth}",
                    0.9,
                    1.0 + base,
                    base,
                    0.4,
                    variable,
                    depth=depth,
                )
            )
    return out


def _records(comparisons):
    return [dict(c.metrics(), label=c.label) for c in comparisons]


def _legend_texts(fig):
    return [t.get_text() for leg in fig.legends for t in leg.get_texts()] + [
        t.get_text()
        for ax in fig.axes
        if (leg := ax.get_legend()) is not None
        for t in leg.get_texts()
    ]


def test_depth_becomes_the_marker_when_a_colour_spans_depths():
    recs = _records(_at_depths())
    assert default_marker_by(recs, "variable", None) == "depth"


@pytest.mark.parametrize(
    "color_by, marker_by",
    [
        (None, None),  # every point is its own legend entry already
        ("variable", "test"),  # the caller chose a marker field
        ("depth", None),  # colour already says depth
    ],
)
def test_a_caller_who_chose_the_encoding_is_left_alone(color_by, marker_by):
    assert default_marker_by(_records(_at_depths()), color_by, marker_by) == marker_by


def test_one_depth_everywhere_needs_no_depth_key():
    flat = [
        _FakeComparison("a", 0.9, 1.0, 0.0, 0.4, "nitrate", depth=100),
        _FakeComparison("b", 0.9, 1.0, 0.1, 0.4, "oxygen", depth=100),
    ]
    assert default_marker_by(_records(flat), "variable", None) is None


@pytest.mark.parametrize("diagram", [target, taylor, paired])
def test_the_static_key_lists_each_depth_beside_the_variables(diagram):
    fig = diagram(_at_depths(), color_by="variable", legend_style="legend")
    texts = _legend_texts(fig)
    assert {"nitrate", "oxygen"} <= set(texts)
    assert {"surface", "100 m", "200 m"} <= set(texts)


def test_a_depth_band_is_spelled_like_every_other_depth_label():
    bands = [
        _FakeComparison(
            f"v{i}", 0.9, 1.0, 0.1 * i, 0.4, "nitrate", depth={"min": lo, "max": hi}
        )
        for i, (lo, hi) in enumerate([(0, 10), (10, 50)])
    ]
    texts = _legend_texts(target(bands, color_by="variable", legend_style="legend"))
    assert "0-10 m" in texts and "10-50 m" in texts


def test_the_interactive_target_keys_depth_too():
    pytest.importorskip("holoviews")
    import holoviews as hv

    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = [
        {"metrics": c.metrics(), "label": c.label, "units": None}
        for c in _at_depths()
    ]
    obj = render(
        PlotSpec(
            family="target",
            items=items,
            options={"color_by": "variable", "legend_style": "legend"},
        ),
        renderer="holoviews",
    )
    figure = hv.render(obj, backend="bokeh")
    labels = " ".join(
        str(getattr(item.label, "value", item.label))
        for legend in figure.legend
        for item in legend.items
    )
    assert "surface" in labels and "100 m" in labels
