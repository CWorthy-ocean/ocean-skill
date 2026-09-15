"""Comparisons with undefined metrics don't crash the summary diagrams.

A comparison scored from a single matched ``(time, depth)`` pair (or otherwise too few
samples to define a variance) leaves ``std_reference`` exactly ``0.0`` and ``corr``
``nan`` — not merely the "weakly constrained" case :func:`ocean_skill.metrics.compute`
already warns about (few, but more than one, matched samples: noisy but finite
numbers), but genuinely undefined geometry: there is no correlation angle or standard
deviation ratio for Taylor, no signed centred RMSD for Target. ``taylor()``/``target()``
used to divide by that zero straight through, raising ``ZeroDivisionError`` deep inside
a whole-suite ``ComparisonSet.summary()`` call. They now drop such comparisons, with one
warning naming which and why, and still draw everything else.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from ocean_skill.plot.summary import paired, target, taylor


class _FakeComparison:
    """A comparison's metric record plus a label and units — the diagrams' interface."""

    def __init__(
        self,
        *,
        label,
        corr,
        std_test,
        std_reference,
        bias,
        crmsd,
        n=30,
        variable="sea_water_temperature",
        units="degC",
        reference="ref1",
    ):
        self.label = label
        self.units = units
        self._record = {
            "corr": corr,
            "std_test": std_test,
            "std_reference": std_reference,
            "bias": bias,
            "crmsd": crmsd,
            "variable": variable,
            "reference": reference,
            "n": n,
        }

    def metrics(self):
        return self._record


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _good(label="good", n=30):
    return _FakeComparison(
        label=label, corr=0.9, std_test=1.1, std_reference=1.0, bias=0.1, crmsd=0.2, n=n
    )


def _degenerate(label="onepair"):
    """Build a comparison scored from a single pair: std collapses to zero, corr nan."""
    return _FakeComparison(
        label=label, corr=float("nan"), std_test=0.0, std_reference=0.0, bias=0.0,
        crmsd=0.0, n=1,
    )


def _n_points(fig):
    return sum(len(c.get_offsets()) for c in fig.axes[0].collections)


def _taylor_samples(fig):
    """Every sample-marker line on a Taylor diagram, including its parasite axes."""
    out = []
    for ax in fig.axes:
        for target_ax in (ax, *getattr(ax, "parasites", [])):
            out += [ln for ln in target_ax.lines if ln.get_marker() == "o"]
    return out


# --------------------------------------------------------------------------- taylor


def test_taylor_drops_degenerate_comparison_and_warns():
    comparisons = [_good(), _degenerate()]
    with pytest.warns(UserWarning, match="undefined.*'onepair' \\(1 pairs\\)"):
        fig = taylor(comparisons, legend_style=None)
    assert len(_taylor_samples(fig)) == 1


def test_taylor_all_degenerate_raises():
    comparisons = [_degenerate("a"), _degenerate("b")]
    with pytest.raises(ValueError, match="every comparison's metrics are undefined"):
        taylor(comparisons, legend_style=None)


def test_taylor_keeps_weakly_constrained_but_finite_points(recwarn):
    """Few-pair but finite metrics (the ordinary 'weakly constrained' case) are kept."""
    comparisons = [_good(n=30), _good("shaky", n=12)]
    fig = taylor(comparisons, legend_style=None)
    assert not [w for w in recwarn.list if "undefined" in str(w.message)]
    assert len(_taylor_samples(fig)) == 2


# --------------------------------------------------------------------------- target


def test_target_drops_degenerate_comparison_and_warns():
    comparisons = [_good(), _degenerate()]
    with pytest.warns(UserWarning, match="undefined.*'onepair' \\(1 pairs\\)"):
        fig = target(comparisons, legend_style=None)
    assert _n_points(fig) == 1


def test_target_all_degenerate_raises():
    comparisons = [_degenerate("a"), _degenerate("b")]
    with pytest.raises(ValueError, match="every comparison's metrics are undefined"):
        target(comparisons, legend_style=None)


def test_target_keeps_weakly_constrained_but_finite_points(recwarn):
    comparisons = [_good(n=30), _good("shaky", n=12)]
    fig = target(comparisons, legend_style=None)
    assert not [w for w in recwarn.list if "undefined" in str(w.message)]
    assert _n_points(fig) == 2


def test_target_interactive_drops_degenerate_comparison_and_warns():
    from ocean_skill.plot.holoviews_renderer import _target

    comparisons = [_good(), _degenerate()]
    items = [
        {"label": c.label, "metrics": c.metrics(), "units": c.units}
        for c in comparisons
    ]
    with pytest.warns(UserWarning, match="undefined.*'onepair' \\(1 pairs\\)"):
        obj = _target(items)

    import holoviews as hv

    points = [e for e in obj if isinstance(e, hv.Points)]
    assert sum(len(p.data) for p in points) == 1


# --------------------------------------------------------------------------- paired


def test_paired_drops_degenerate_comparison_and_warns():
    comparisons = [_good(), _degenerate()]
    with pytest.warns(UserWarning, match="undefined"):
        paired(comparisons)
