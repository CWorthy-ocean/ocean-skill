"""Pooling comparisons you already have onto one summary diagram.

The claim these defend is that a summary can be assembled from existing objects without
re-running anything and without disturbing them. Two halves: what ``_flatten`` accepts
and drops, and what the pooled points end up being *called* — because ``compare`` labels
only what varies inside its own fan-out, so two sets that each fanned over depth arrive
with the same labels and would otherwise be indistinguishable on the diagram.

Offline throughout: the labelling rule reads a comparison's specification, never its
data, so a stand-in carrying that specification exercises the real code paths.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from ocean_skill.comparison import (
    ComparisonSet,
    _flatten,
    _identity,
    _pooled_labels,
    summary,
)

NO3 = "mole_concentration_of_nitrate_in_sea_water"
PO4 = "mole_concentration_of_phosphate_in_sea_water"


class _FakeComparison:
    """A comparison's specification plus a metrics record, and nothing else.

    The pooling code reads the specification (``variable``, ``select``, source names)
    and the diagrams read ``metrics()``/``label`` — see ``_FakeComparison`` in
    ``test_summary_labels``, which stands in for the same interface from the other side.
    """

    def __init__(
        self,
        *,
        variable=NO3,
        depth="surface",
        test="GOM_bgc",
        reference="woa23_nitrate_month01",
        label="no3",
        corr=0.9,
        aggregate=None,
    ):
        self.variable = variable
        self.select = {"depth": depth}
        self.test_name = test
        self.reference_name = reference
        self.label = label
        self.aggregate = aggregate
        self.method = "conservative_normed"
        self.over = None
        self.time_method = "auto"
        self.tolerance = None
        self.bin_anchor = "auto"
        self._record = {
            "corr": corr,
            "std_test": 1.05,
            "std_reference": 1.0,
            "bias": 0.2,
            "crmsd": 0.3,
            "variable": variable,
            "depth": depth,
        }

    def metrics(self):
        return self._record

    def as_item(self):
        return {"metrics": self._record, "label": self.label}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --- what pooling accepts -----------------------------------------------------------


def test_flatten_takes_any_nesting_of_comparisons_and_sets():
    a, b, c = (
        _FakeComparison(label=n, depth=d) for n, d in (("a", 0), ("b", 1), ("c", 2))
    )
    assert _flatten([ComparisonSet([a]), b, [[c]]]) == [a, b, c]


def test_flatten_takes_a_mapping_by_its_values():
    a, b = _FakeComparison(label="a", depth=0), _FakeComparison(label="b", depth=1)
    assert _flatten({"hindcast": a, "forecast": ComparisonSet([b])}) == [a, b]


def test_flatten_refuses_something_that_is_not_a_comparison():
    with pytest.raises(TypeError, match="expected comparisons"):
        _flatten(["woa23_nitrate_month01"])


def test_an_exact_repeat_is_dropped_rather_than_drawn_twice(capsys):
    """Two sets built for different figures commonly share a pair."""
    first = _FakeComparison(label="no3")
    same = _FakeComparison(label="no3 again")  # same specification, different label
    pooled = _flatten([first, same])

    assert pooled == [first]
    assert "duplicate" in capsys.readouterr().out


def test_identity_is_the_specification_not_the_label():
    assert _identity(_FakeComparison(label="x")) == _identity(
        _FakeComparison(label="y")
    )
    assert _identity(_FakeComparison(depth=0)) != _identity(_FakeComparison(depth=5))


# --- what the pooled points get called ----------------------------------------------


def test_colliding_depth_labels_are_relabelled_by_what_varies_across_the_pool():
    """Both fan-outs called a point 'surface'; pooled, the variable tells them apart."""
    nutrients = ComparisonSet(
        [
            _FakeComparison(variable=NO3, depth="surface", label="no3"),
            _FakeComparison(variable=PO4, depth="surface", label="po4"),
        ]
    )
    depths = ComparisonSet(
        [
            _FakeComparison(variable=NO3, depth=50, label="50 m"),
            _FakeComparison(variable=NO3, depth=100, label="100 m"),
        ]
    )

    pooled = nutrients + depths

    assert pooled.labels == [
        "nitrate surface",
        "phosphate surface",
        "nitrate 50 m",
        "nitrate 100 m",
    ]


def test_a_dimension_that_does_not_vary_stays_out_of_the_label():
    pooled = ComparisonSet([_FakeComparison(variable=NO3, depth=0)]) + ComparisonSet(
        [_FakeComparison(variable=NO3, depth=50)]
    )
    assert pooled.labels == ["0 m", "50 m"]


def test_a_depth_selected_by_another_name_is_still_read_as_a_depth():
    """``select={"Z": 100}`` is as valid as ``{"depth": 100}``, and is not surface."""
    shallow, deep = _FakeComparison(depth=0), _FakeComparison(depth=100)
    deep.select = {"Z": 100}
    assert _pooled_labels([shallow, deep]) == ["0 m", "100 m"]


def test_a_dimension_that_varies_but_says_nothing_new_stays_out():
    """Sources that vary in lockstep with the variable would only lengthen the label."""
    pooled = _flatten(
        [
            _FakeComparison(variable=NO3, test="gom_no3", reference="woa_no3"),
            _FakeComparison(variable=PO4, test="gom_po4", reference="woa_po4"),
        ]
    )
    assert _pooled_labels(pooled) == ["nitrate", "phosphate"]


def test_pooling_names_the_model_when_that_is_what_varies():
    pooled = _flatten(
        [_FakeComparison(test="GOM_bgc"), _FakeComparison(test="NEP_bgc")]
    )
    assert _pooled_labels(pooled) == ["GOM_bgc", "NEP_bgc"]


def test_with_nothing_varying_each_comparison_keeps_its_own_label():
    """Differing only in how they were aggregated: no dimension names them apart."""
    pooled = _flatten(
        [
            _FakeComparison(label="monthly", aggregate={"time": "mean"}),
            _FakeComparison(label="annual", aggregate={"time": "median"}),
        ]
    )
    assert _pooled_labels(pooled) == ["monthly", "annual"]


def test_labels_still_colliding_are_suffixed_rather_than_left_ambiguous(capsys):
    pooled = _flatten(
        [
            _FakeComparison(variable=NO3, label="a", aggregate={"time": "mean"}),
            _FakeComparison(variable=NO3, label="b", aggregate={"time": "median"}),
            _FakeComparison(variable=PO4, label="c"),
        ]
    )
    assert _pooled_labels(pooled) == ["nitrate", "nitrate (2)", "phosphate"]
    assert "suffixed" in capsys.readouterr().out


def test_named_groups_take_their_key():
    a, b = _FakeComparison(label="no3"), _FakeComparison(depth=50, label="50 m")
    pooled = ComparisonSet({"hindcast": a, "forecast": ComparisonSet([b])})
    assert pooled.labels == ["hindcast", "forecast"]


def test_a_named_group_of_several_prefixes_its_members():
    pooled = ComparisonSet(
        {
            "hindcast": [
                _FakeComparison(depth=0, label="surface"),
                _FakeComparison(depth=50, label="50 m"),
            ],
            "forecast": _FakeComparison(depth=100, label="100 m"),
        }
    )
    assert pooled.labels == ["hindcast: surface", "hindcast: 50 m", "forecast"]


def test_a_dict_and_explicit_labels_are_refused_together():
    with pytest.raises(TypeError, match="already the labels"):
        ComparisonSet({"a": _FakeComparison()}, labels=["b"])


def test_labels_must_be_one_per_comparison():
    with pytest.raises(ValueError, match="one per comparison"):
        ComparisonSet([_FakeComparison()], labels=["a", "b"])


def test_pooling_leaves_the_originals_alone():
    """The whole point of reusing objects: their own row and frame labels survive."""
    a = _FakeComparison(variable=NO3, depth=0, label="mine")
    b = _FakeComparison(variable=PO4, depth=50, label="theirs")

    pooled = ComparisonSet([a]) + ComparisonSet([b])

    assert pooled.labels != [a.label, b.label]
    assert (a.label, b.label) == ("mine", "theirs")


def test_a_set_without_overrides_still_draws_its_comparisons_own_labels():
    a = _FakeComparison(label="mine")
    assert ComparisonSet([a])._metric_items()[0]["label"] == "mine"


def test_metric_items_units_default_to_none_without_an_aligned_pair():
    """A hand-built comparison (no ``aligned``) has nowhere to read units from — the
    absolute-axes diagrams treat that the same as units genuinely being unknown."""
    a = _FakeComparison(label="mine")
    assert ComparisonSet([a])._metric_items()[0]["units"] is None


# --- the front door ------------------------------------------------------------------


def test_summary_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="'both', 'taylor', 'target'"):
        summary([_FakeComparison()], kind="paired")


@pytest.mark.parametrize("kind", ["both", "taylor", "target"])
def test_summary_draws_the_pooled_labels(kind):
    fig = summary(
        [
            ComparisonSet([_FakeComparison(variable=NO3, depth=0, label="no3")]),
            _FakeComparison(variable=PO4, depth=0, label="po4"),
        ],
        kind=kind,
        labels="legend",
    )
    texts = {t.get_text() for t in fig.findobj(plt.Text)}
    assert {"nitrate", "phosphate"} <= texts


def test_summary_of_named_groups_draws_the_keys():
    fig = summary(
        {
            "hindcast": _FakeComparison(variable=NO3, depth=0),
            "forecast": _FakeComparison(variable=NO3, depth=50),
        },
        kind="target",
        labels="legend",
    )
    texts = {t.get_text() for t in fig.findobj(plt.Text)}
    assert {"hindcast", "forecast"} <= texts
