"""Drift arrows: ``arrows=`` on the Target and Taylor diagrams.

Connects comparisons that share every identifying field but one (typically ``time``,
via a ``compare(..., times=...)`` fan-out) with an arrow per consecutive pair, so a
run's trajectory reads at a glance rather than as an unordered cloud. A chain's first
point is drawn hollow; later points, filled as usual.

Both renderers have to agree on where a segment starts and ends and what colour it
gets — the same requirement ``test_summary_style.py`` enforces for plain points — so
several tests here compare the static and interactive output directly.
"""

from __future__ import annotations

from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ocean_skill.plot.summary import _target_xy, paired, target, taylor


class _FakeComparison:
    """A comparison's metric record plus a label — the diagrams' whole interface."""

    def __init__(
        self,
        label,
        *,
        corr,
        std_test,
        bias,
        crmsd,
        test,
        time,
        std_reference=1.0,
        variable="sea_water_temperature",
        depth=0,
        reference="obs",
    ):
        self.label = label
        self._record = {
            "corr": corr,
            "std_test": std_test,
            "std_reference": std_reference,
            "bias": bias,
            "crmsd": crmsd,
            "variable": variable,
            "depth": depth,
            "test": test,
            "reference": reference,
            "time": time,
        }

    def metrics(self):
        return self._record


#: Two runs drifting toward the reference over three time steps — distinct enough
#: (different corr/std/bias/crmsd) that overlapping points never mask a bug.
_SERIES = {
    "runA": [
        ("2010", 0.80, 1.20, 0.40, 0.60),
        ("2015", 0.88, 1.10, 0.20, 0.45),
        ("2020", 0.93, 1.05, 0.05, 0.30),
    ],
    "runB": [
        ("2010", 0.70, 1.40, -0.30, 0.80),
        ("2015", 0.82, 1.25, -0.15, 0.55),
        ("2020", 0.90, 1.10, -0.02, 0.35),
    ],
}


def _comparisons(series=_SERIES, **extra):
    return [
        _FakeComparison(
            f"{test} {time}",
            corr=corr,
            std_test=std,
            bias=bias,
            crmsd=crmsd,
            test=test,
            time=time,
            **extra,
        )
        for test, points in series.items()
        for time, corr, std, bias, crmsd in points
    ]


@pytest.fixture
def comparisons():
    return _comparisons()


@pytest.fixture
def items(comparisons):
    """Comparisons in the shape ``PlotSpec.items`` carries them."""
    return [{"label": c.label, "metrics": c.metrics()} for c in comparisons]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _arrows(ax):
    """Every arrow annotation drawn on ``ax``.

    An ``Annotation`` with an ``arrow_patch`` set, as opposed to a plain ring/point
    label or (on Taylor) a ``clabel`` contour label, which is a bare ``Text`` with no
    such attribute.
    """
    return [t for t in ax.texts if getattr(t, "arrow_patch", None) is not None]


def _all_arrows(fig):
    """Arrows on every real axes in ``fig``, including a Taylor panel's parasite."""
    out = []
    for ax in fig.axes:
        for real_ax in (ax, *getattr(ax, "parasites", [])):
            out += _arrows(real_ax)
    return out


# --------------------------------------------------------------------- static target


def test_target_arrows_connects_time_ordered_points_per_run(comparisons):
    fig = target(comparisons, color_by="test", arrows=True, labels=None)
    arrows = _arrows(fig.axes[0])

    # 2 runs x 2 consecutive segments each (3 time steps per run)
    assert len(arrows) == 4

    by_run = {name: sorted(pts, key=lambda p: p[0]) for name, pts in _SERIES.items()}
    expected_segments = []
    for name, points in by_run.items():
        recs = [
            {"crmsd": c, "bias": b, "std_test": s, "std_reference": 1.0}
            for _, corr, s, b, c in points
        ]
        xy = [_target_xy(r, True) for r in recs]
        expected_segments += list(pairwise(xy))

    got_segments = {(tuple(a.xyann), tuple(a.xy)) for a in arrows}
    assert got_segments == {(tuple(x0), tuple(x1)) for x0, x1 in expected_segments}


def test_target_arrows_start_point_is_hollow(comparisons):
    fig = target(comparisons, color_by="test", arrows=True, labels=None)
    collections = fig.axes[0].collections  # one per point, in record order

    # record order: runA 2010, 2015, 2020, runB 2010, 2015, 2020 -- indices 0 and 3
    # are each chain's earliest time and so its hollow start.
    for i, coll in enumerate(collections):
        face = coll.get_facecolor()
        edge = coll.get_edgecolor()
        if i in (0, 3):
            assert len(face) == 0, f"point {i} should be hollow"
            assert tuple(edge[0][:3]) != pytest.approx((1, 1, 1))
        else:
            assert len(face) == 1
            assert tuple(edge[0][:3]) == pytest.approx((1, 1, 1))  # white edge


def test_target_arrows_true_means_time(comparisons):
    fig_true = target(comparisons, color_by="test", arrows=True, labels=None)
    fig_name = target(comparisons, color_by="test", arrows="time", labels=None)
    by_true = _arrows(fig_true.axes[0])
    by_name = _arrows(fig_name.axes[0])
    assert {(tuple(a.xyann), tuple(a.xy)) for a in by_true} == {
        (tuple(a.xyann), tuple(a.xy)) for a in by_name
    }


def test_target_arrows_color_matches_group(comparisons):
    fig = target(comparisons, color_by="test", arrows=True, labels=None)
    arrows = _arrows(fig.axes[0])
    collections = fig.axes[0].collections

    # runA's points are collections 0-2, runB's are 3-5; an arrow's colour should
    # match its *end* point's face/edge colour (the hollow start has no face, so its
    # edge is what carries the group colour).
    runA_color = tuple(collections[1].get_facecolor()[0][:3])
    runB_color = tuple(collections[4].get_facecolor()[0][:3])
    seen_colors = {tuple(a.arrow_patch.get_edgecolor()[:3]) for a in arrows}
    assert seen_colors == {runA_color, runB_color}


def test_target_arrows_sorts_out_of_order_times(comparisons):
    shuffled = [comparisons[2], comparisons[0], comparisons[1], *comparisons[3:]]
    fig = target(shuffled, color_by="test", arrows=True, labels=None)
    arrows = _arrows(fig.axes[0])

    runA_recs = [c.metrics() for c in comparisons[:3]]  # already chronological
    xy = [_target_xy(r, True) for r in runA_recs]
    expected = {(tuple(xy[0]), tuple(xy[1])), (tuple(xy[1]), tuple(xy[2]))}
    got = {(tuple(a.xyann), tuple(a.xy)) for a in arrows}
    assert expected <= got  # runA's segments still go early -> late despite shuffling


def test_target_arrows_min_max_window_sorts_by_start(comparisons):
    windowed = [
        _FakeComparison(
            f"runA {w['min']}",
            corr=0.9,
            std_test=1.0,
            bias=b,
            crmsd=0.1,
            test="runA",
            time=w,
        )
        for w, b in [
            ({"min": "2015-01", "max": "2015-12"}, 0.2),
            ({"min": "2010-01", "max": "2010-12"}, 0.4),
        ]
    ]
    fig = target(windowed, arrows=True, labels=None)
    (arrow,) = _arrows(fig.axes[0])
    # the 2010 window (input index 1) sorts before the 2015 one (input index 0)
    assert arrow.xyann[1] == pytest.approx(0.4)
    assert arrow.xy[1] == pytest.approx(0.2)


def test_target_arrows_unsortable_values_keep_input_order():
    unsortable = [
        _FakeComparison(
            f"runA {t}", corr=0.9, std_test=1.0, bias=b, crmsd=0.1, test="runA", time=t
        )
        for t, b in [("phase-two", 0.2), ("phase-one", 0.4)]
    ]
    fig = target(unsortable, arrows=True, labels=None)
    (arrow,) = _arrows(fig.axes[0])
    # input order preserved: phase-two (bias 0.2) is first, phase-one (0.4) second
    assert arrow.xyann[1] == pytest.approx(0.2)
    assert arrow.xy[1] == pytest.approx(0.4)


def test_target_arrows_unknown_field_raises(comparisons):
    with pytest.raises(ValueError, match="arrows='bogus'"):
        target(comparisons, arrows="bogus")


def test_target_arrows_nothing_to_connect_warns(comparisons):
    # every comparison has a distinct `reference`, so no two share the other identity
    # fields while differing only in it -- nothing to chain.
    for i, c in enumerate(comparisons):
        c._record["reference"] = f"obs{i}"
    with pytest.warns(UserWarning, match="nothing to connect"):
        fig = target(comparisons, arrows="reference", labels=None)
    assert _arrows(fig.axes[0]) == []
    # no hollow points either -- nothing to draw as a chain start
    assert all(len(c.get_facecolor()) == 1 for c in fig.axes[0].collections)


def test_target_without_arrows_is_unchanged(comparisons):
    fig = target(comparisons, color_by="test", labels=None)
    assert _arrows(fig.axes[0]) == []
    assert all(len(c.get_facecolor()) == 1 for c in fig.axes[0].collections)


# --------------------------------------------------------------------- static taylor


def _taylor_ax(fig):
    """Return the polar aux axes; see ``test_summary_style._taylor_lines``."""
    (ax,) = fig.axes
    (parasite,) = ax.parasites
    return parasite


def test_taylor_arrows_draws_on_the_polar_axes(comparisons):
    fig = taylor(comparisons, color_by="test", arrows=True, labels=None)
    arrows = _arrows(_taylor_ax(fig))
    assert len(arrows) == 4

    recs = [c.metrics() for c in comparisons[:3]]  # runA, chronological
    thetas = [np.arccos(r["corr"]) for r in recs]
    stds = [r["std_test"] / r["std_reference"] for r in recs]
    expected = {
        (tuple((thetas[i], stds[i])), tuple((thetas[i + 1], stds[i + 1])))
        for i in range(2)
    }
    got = {(tuple(a.xyann), tuple(a.xy)) for a in arrows}
    assert expected <= got


def test_taylor_arrows_start_sample_is_hollow(comparisons):
    fig = taylor(comparisons, color_by="test", arrows=True, labels=None)
    lines = [
        ln
        for ax in fig.axes
        for real_ax in (ax, *ax.parasites)
        for ln in real_ax.lines
        if ln.get_marker() == "o"
    ]
    # sample order matches record order: runA 2010/2015/2020, runB 2010/2015/2020
    assert lines[0].get_mfc() == "none"
    assert lines[3].get_mfc() == "none"
    assert lines[1].get_mfc() != "none"
    assert lines[2].get_mfc() != "none"


def test_taylor_arrows_unknown_field_raises(comparisons):
    with pytest.raises(ValueError, match="arrows='bogus'"):
        taylor(comparisons, arrows="bogus")


# -------------------------------------------------------------------------- paired


def test_paired_forwards_arrows_to_both_panels(comparisons):
    fig = paired(comparisons, color_by="test", arrows=True, labels=None)
    assert len(_all_arrows(fig)) == 8  # 4 on the target panel, 4 on the taylor panel


# ---------------------------------------------------------- the interactive renderer


def _interactive_target(items, **kwargs):
    from ocean_skill.plot.holoviews_renderer import _target

    return _target(items, **kwargs)


def _bokeh_arrows(obj):
    import holoviews as hv
    from bokeh.models import Arrow

    fig = hv.render(obj, backend="bokeh")  # runs the finalize hook
    return [r for r in fig.center if isinstance(r, Arrow)]


def _rounded(segments):
    return {
        tuple(round(v, 9) for v in (*start, *end)) for start, end in segments
    }


def test_interactive_target_arrows_match_static(comparisons, items):
    static_fig = target(comparisons, color_by="test", arrows=True, labels=None)
    static_segments = _rounded(
        (a.xyann, a.xy) for a in _arrows(static_fig.axes[0])
    )

    obj = _interactive_target(items, color_by="test", arrows=True, labels=None)
    arrows = _bokeh_arrows(obj)
    interactive_segments = _rounded(
        ((a.x_start, a.y_start), (a.x_end, a.y_end)) for a in arrows
    )

    assert len(arrows) == 4
    assert static_segments == interactive_segments


def test_interactive_target_arrows_color_matches_static(comparisons, items):
    import matplotlib.colors as mcolors

    static_fig = target(comparisons, color_by="test", arrows=True, labels=None)
    static_arrows = _arrows(static_fig.axes[0])
    static_colors = {
        mcolors.to_hex(a.arrow_patch.get_edgecolor()) for a in static_arrows
    }

    obj = _interactive_target(items, color_by="test", arrows=True, labels=None)
    interactive_colors = {a.line_color for a in _bokeh_arrows(obj)}

    assert static_colors == interactive_colors


def test_interactive_target_arrows_hollow_start_element(items):
    import holoviews as hv

    obj = _interactive_target(items, color_by="test", arrows=True)
    hollow = [
        e
        for e in obj
        if isinstance(e, hv.Points)
        and e.opts.get("style").kwargs.get("fill_alpha") == 0
    ]
    # one hollow element per group (runA, runB), each holding exactly one point
    assert len(hollow) == 2
    for e in hollow:
        assert len(e.data) == 1


def test_interactive_target_all_starts_group_keeps_legend(items):
    """A level made entirely of chain-starts must keep its legend entry.

    ``color_by="time"`` makes the earliest time level entirely chain-starts (both
    runs' first point).
    """
    import holoviews as hv

    two_step = [i for i in items if i["metrics"]["time"] in ("2010", "2015")]
    obj = _interactive_target(two_step, color_by="time", arrows=True, labels="legend")
    hollow_2010 = [
        e
        for e in obj
        if isinstance(e, hv.Points)
        and e.opts.get("style").kwargs.get("fill_alpha") == 0
        and e.label == "2010"
    ]
    assert len(hollow_2010) == 1
    assert hollow_2010[0].opts.get("plot").kwargs["show_legend"] is True


def test_interactive_target_arrows_unknown_field_raises(items):
    with pytest.raises(ValueError, match="arrows='bogus'"):
        _interactive_target(items, arrows="bogus")


def test_both_renderers_raise_the_same_arrows_error(comparisons, items):
    with pytest.raises(ValueError, match="arrows='bogus'") as static_exc:
        target(comparisons, arrows="bogus")
    with pytest.raises(ValueError, match="arrows='bogus'") as interactive_exc:
        _interactive_target(items, arrows="bogus")
    assert str(static_exc.value) == str(interactive_exc.value)


def test_interactive_target_without_arrows_is_unchanged(items):
    obj = _interactive_target(items, color_by="test")
    assert _bokeh_arrows(obj) == []
