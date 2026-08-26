"""Marker sizing, transparency, and explicit colours on the summary diagrams.

``marker_scale``/``alpha`` are style knobs on every point drawn — sample points, the
reference star, and legend swatches all move together, so a figure at
``marker_scale=2`` doesn't have a key sized for the default. ``colors`` is an explicit
per-point palette that both renderers must honour identically, including inside
:func:`paired`'s shared legend, which used to build its swatches from ``COLOR_CYCLE``
regardless of an explicit ``colors=`` — see ``test_paired_legend_swatches_honor_explicit_colors``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest

from ocean_skill.plot.summary import _group_styles, paired, target, taylor


class _FakeComparison:
    """Minimal stand-in: the diagrams only call ``metrics()`` and read ``label``."""

    def __init__(self, label, corr, std_test, bias, crmsd, variable):
        self.label = label
        self._metrics = {
            "corr": corr,
            "std_test": std_test,
            "std_reference": 1.0,
            "bias": bias,
            "crmsd": crmsd,
            "variable": variable,
            "depth": 0,
        }

    def metrics(self):
        return self._metrics


@pytest.fixture
def comparisons():
    return [
        _FakeComparison("temp GOM", 0.95, 1.10, 0.30, 0.35, "sea_water_temperature"),
        _FakeComparison("salt GOM", 0.88, 0.82, -0.20, 0.50, "sea_water_salinity"),
        _FakeComparison("no3 GOM", 0.91, 0.95, -0.10, 0.40, "nitrate"),
    ]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _taylor_lines(fig):
    """Every Line2D drawn on a Taylor diagram, including its parasite (aux) axes.

    Mirrors ``test_summary_labels._annotated_texts``: Taylor's sample points and
    reference star live on ``get_aux_axes``'s parasite, which ``fig.findobj`` and
    ``fig.axes`` alike do not reach.
    """
    out = []
    for ax in fig.axes:
        for target_ax in (ax, *getattr(ax, "parasites", [])):
            out += list(target_ax.lines)
    return out


# ------------------------------------------------------------------ marker_scale


def test_taylor_marker_scale_scales_points_and_star(comparisons):
    fig = taylor(comparisons, marker_scale=2.0, labels=None)

    lines = _taylor_lines(fig)
    samples = [ln for ln in lines if ln.get_marker() == "o"]
    stars = [ln for ln in lines if ln.get_marker() == "*"]

    assert samples, "expected sample points on the diagram"
    assert all(ln.get_markersize() == pytest.approx(18.0) for ln in samples)
    assert stars, "expected the reference star"
    assert stars[0].get_markersize() == pytest.approx(20.0)


def test_target_marker_scale_scales_points_star_and_swatches(comparisons):
    fig = target(comparisons, marker_scale=2.0, labels="legend")
    ax = fig.axes[0]

    assert ax.collections[0].get_sizes() == pytest.approx([70 * 2.0**2])
    star = next(ln for ln in ax.lines if ln.get_marker() == "*")
    assert star.get_markersize() == pytest.approx(22.0)

    swatches = {
        h.get_label(): h.get_markersize() for h in fig.legends[0].legend_handles
    }
    assert swatches["temp GOM"] == pytest.approx(14.0)
    assert swatches["reference"] == pytest.approx(18.0)


def test_paired_forwards_marker_scale_and_alpha(comparisons):
    fig = paired(comparisons, marker_scale=2.0, alpha=0.5, labels=None)
    target_ax = fig.axes[1]

    collection = target_ax.collections[0]
    assert collection.get_sizes() == pytest.approx([70 * 2.0**2])
    assert collection.get_alpha() == pytest.approx(0.5)


def test_paired_legend_swatches_honor_explicit_colors(comparisons):
    """Regression: the shared legend used to ignore ``colors``, keying against COLOR_CYCLE."""
    explicit = ["#1b9e77", "#d95f02", "#7570b3"]
    fig = paired(comparisons, colors=explicit, labels="legend")

    swatches = {h.get_label(): h.get_mfc() for h in fig.legends[0].legend_handles}
    for label, color in zip(("temp GOM", "salt GOM", "no3 GOM"), explicit, strict=True):
        assert mcolors.to_hex(swatches[label]) == color


# ------------------------------------------------------------------------- alpha


@pytest.mark.parametrize("diagram", [taylor, target])
def test_alpha_reaches_the_points_but_not_the_star(diagram, comparisons):
    fig = diagram(comparisons, alpha=0.5, labels=None)

    if diagram is taylor:
        lines = _taylor_lines(fig)
        samples = [ln for ln in lines if ln.get_marker() == "o"]
        stars = [ln for ln in lines if ln.get_marker() == "*"]
        assert all(ln.get_alpha() == pytest.approx(0.5) for ln in samples)
        assert stars[0].get_alpha() is None
    else:
        ax = fig.axes[0]
        assert ax.collections[0].get_alpha() == pytest.approx(0.5)
        star = next(ln for ln in ax.lines if ln.get_marker() == "*")
        assert star.get_alpha() is None


# --------------------------------------------------------- the interactive renderer


@pytest.fixture
def items(comparisons):
    """Comparisons in the shape ``PlotSpec.items`` carries them."""
    return [{"label": c.label, "metrics": c.metrics()} for c in comparisons]


def _interactive_target(items, **kwargs):
    from ocean_skill.plot.holoviews_renderer import _target

    return _target(items, **kwargs)


def _points(obj):
    import holoviews as hv

    return [e for e in obj if isinstance(e, hv.Points)]


def test_interactive_target_honors_explicit_colors(items):
    explicit = ["#1b9e77", "#d95f02", "#7570b3"]
    obj = _interactive_target(items, colors=explicit, labels="legend")

    colors = {e.label: e.opts.get("style").kwargs["color"] for e in _points(obj)}
    for label, color in zip(("temp GOM", "salt GOM", "no3 GOM"), explicit, strict=True):
        assert colors[label] == color


def test_interactive_color_by_wins_over_colors(comparisons, items):
    """color_by is a field-driven grouping; an explicit palette must not override it."""
    from ocean_skill.plot.style import COLOR_CYCLE

    obj = _interactive_target(
        items, colors=["#111111", "#222222", "#333333"], color_by="variable"
    )
    colors = [e.opts.get("style").kwargs["color"] for e in _points(obj)]

    assert colors == [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(len(colors))]


def test_both_renderers_use_the_same_explicit_colors(comparisons, items):
    explicit = ["#1b9e77", "#d95f02", "#7570b3"]
    cols, _, _ = _group_styles(
        [dict(c.metrics(), label=c.label) for c in comparisons], colors=explicit
    )
    static = [mcolors.to_hex(c) for c in cols]

    obj = _interactive_target(items, colors=explicit, labels="legend")
    interactive = [e.opts.get("style").kwargs["color"] for e in _points(obj)]

    assert static == interactive


def test_interactive_marker_scale(items):
    obj = _interactive_target(items, marker_scale=2.0)

    assert all(e.opts.get("style").kwargs["size"] == 22.0 for e in _points(obj))


def test_interactive_alpha(items):
    default = _interactive_target(items)
    faded = _interactive_target(items, alpha=0.5)

    default_style = _points(default)[0].opts.get("style").kwargs
    faded_style = _points(faded)[0].opts.get("style").kwargs

    assert "fill_alpha" not in default_style and "line_alpha" not in default_style
    assert faded_style["fill_alpha"] == pytest.approx(0.5)
    assert faded_style["line_alpha"] == pytest.approx(0.5)
