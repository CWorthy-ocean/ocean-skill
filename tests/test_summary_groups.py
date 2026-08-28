"""Grouping Taylor/target points by something not already in the metric record.

``color_by``/``marker_by`` split on whatever a metric record already carries
(``variable``, ``depth``, ``test``, ``reference``, ...) — but a caller often wants to
split by something that is not a column at all: which region a mooring sits in, which
cruise a cast came from. ``groups={reference_name: label}`` supplies exactly that at
plot time, without injecting it into every comparison's own metrics first.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from ocean_skill.plot.summary import paired, target, taylor


class _FakeComparison:
    """Minimal stand-in: the diagrams only call ``metrics()`` and read ``label``."""

    def __init__(self, label, reference, corr, std_test, bias, crmsd):
        self.label = label
        self._metrics = {
            "reference": reference,
            "corr": corr,
            "std_test": std_test,
            "std_reference": 1.0,
            "bias": bias,
            "crmsd": crmsd,
        }

    def metrics(self):
        return self._metrics


GROUPS = {
    "mooring_a": "upper inlet",
    "mooring_b": "lower inlet",
    "mooring_c": "lower inlet",
}


@pytest.fixture
def comparisons():
    return [
        _FakeComparison("station A", "mooring_a", 0.95, 1.10, 0.30, 0.35),
        _FakeComparison("station B", "mooring_b", 0.88, 0.82, -0.20, 0.50),
        _FakeComparison("station C", "mooring_c", 0.91, 0.95, -0.10, 0.40),
    ]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _legend_texts(fig) -> set[str]:
    return {t.get_text() for lg in fig.legends for t in lg.get_texts()}


@pytest.mark.parametrize("diagram", [taylor, target])
def test_groups_defaults_color_by_to_group(diagram, comparisons):
    fig = diagram(comparisons, groups=GROUPS, labels="legend")

    assert {"upper inlet", "lower inlet"} <= _legend_texts(fig)
    # One entry per *group*, since nothing else was named to split on.
    assert "station A" not in _legend_texts(fig)


@pytest.mark.parametrize("diagram", [taylor, target])
def test_an_explicit_color_by_is_not_overridden_by_groups(diagram, comparisons):
    fig = diagram(comparisons, groups=GROUPS, color_by="label", labels="legend")

    assert {"station A", "station B", "station C"} <= _legend_texts(fig)
    assert not ({"upper inlet", "lower inlet"} & _legend_texts(fig))


def test_groups_falls_back_to_the_label_with_no_reference_key():
    """A hand-built record with no ``reference`` (not from a real comparison)."""

    class _NoReference:
        label = "one-off"

        def metrics(self):
            return {
                "corr": 0.9,
                "std_test": 1.0,
                "std_reference": 1.0,
                "bias": 0.1,
                "crmsd": 0.2,
            }

    fig = target([_NoReference()], groups={"one-off": "special"}, labels="legend")
    assert "special" in _legend_texts(fig)


def test_paired_shares_one_group_legend(comparisons):
    fig = paired(comparisons, groups=GROUPS, labels="legend")

    assert len(fig.legends) == 1, "one key for both panels, not one per panel"
    assert {"upper inlet", "lower inlet"} <= _legend_texts(fig)


# ------------------------------------------------------- the interactive renderer


def _interactive_target(items, **kwargs):
    from ocean_skill.plot.holoviews_renderer import _target

    return _target(items, **kwargs)


def _points(obj):
    import holoviews as hv

    return [e for e in obj if isinstance(e, hv.Points)]


@pytest.fixture
def items(comparisons):
    return [{"label": c.label, "metrics": c.metrics()} for c in comparisons]


def test_interactive_target_groups_defaults_color_by_to_group(items):
    obj = _interactive_target(items, groups=GROUPS, labels="legend")

    assert {e.label for e in _points(obj)} == {"upper inlet", "lower inlet"}


def test_interactive_target_explicit_color_by_is_not_overridden(items):
    obj = _interactive_target(items, groups=GROUPS, color_by="label", labels="legend")

    assert {e.label for e in _points(obj)} == {"station A", "station B", "station C"}
