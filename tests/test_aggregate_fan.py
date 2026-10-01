"""A top-level ``aggregate=[spec, spec, ...]`` fans like ``variables=``/``depths=``.

The motivating case is the PACMED paper's Figure 4: the mean and the seasonal-cycle
variance of a model against a WOA climatology, at the surface and at 200 m, as one
call. Each spec is a whole aggregate (plain or ``{"test", "reference"}`` pair), run
through the ordinary ``compare()`` machinery on its own, so these tests mock the
catalog and ``Comparison.align`` the way ``test_select_aggregate_pair_spec.py`` does
and read off what each fanned member was built with. A list *under* an axis is a
different thing -- a chain of steps, see ``test_aggregate_chain_specs.py`` -- and
the last section pins that the two are never confused.
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from ocean_skill import comparison
from ocean_skill.comparison import (
    Comparison,
    ComparisonSet,
    _aggregate_fan,
    _aggregate_label,
)
from ocean_skill.config import SuiteConfig
from ocean_skill.workflows import pages as P

MEAN = {"time": "mean"}
SEASONAL_VAR = {"time": [{"groupby": "month", "reduce": "mean"}, "var"]}

DECLARED = {"model": {"variables": []}, "obs": {"variables": []}}


def _compare(**kwargs):
    """Run compare() against stub sources, returning the aligned-in-name-only set."""
    kwargs = {
        "reference": "obs",
        "test": "model",
        "variables": ["temperature"],
        **kwargs,
    }
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=DECLARED[n])
        ),
        mock.patch.object(Comparison, "align", lambda self, refresh=False: None),
    ):
        return comparison.compare(**kwargs)


# -- the fan itself ---------------------------------------------------------------


def test_two_specs_make_two_members_each_with_its_own_aggregate():
    out = _compare(aggregate=[MEAN, SEASONAL_VAR])
    assert isinstance(out, ComparisonSet)
    assert [c.aggregate for c in out] == [MEAN, SEASONAL_VAR]


def test_the_fan_crosses_the_depth_fan():
    out = _compare(depths=["surface", 200], aggregate=[MEAN, SEASONAL_VAR])
    assert len(out) == 4
    assert [(c.aggregate, c.select["depth"]) for c in out] == [
        (MEAN, "surface"),
        (MEAN, 200),
        (SEASONAL_VAR, "surface"),
        (SEASONAL_VAR, 200),
    ]


def test_the_fan_crosses_the_variable_fan():
    out = _compare(
        variables=["temperature", "salinity"], aggregate=[MEAN, SEASONAL_VAR]
    )
    assert len(out) == 4
    assert len({c.variable for c in out}) == 2
    assert {str(c.aggregate) for c in out} == {str(MEAN), str(SEASONAL_VAR)}
    assert [c.aggregate for c in out[:2]] == [MEAN, MEAN]  # spec-major order


def test_a_pair_spec_entry_is_normalized_and_kept_whole():
    pair = {
        "reference": {"time": "mean"},
        "test": {"time": {"groupby": "month", "reduce": "mean"}},
    }
    out = _compare(aggregate=[MEAN, pair])
    assert out[0].aggregate == MEAN
    assert out[1].aggregate == {
        "test": {"time": {"groupby": "month", "reduce": "mean"}},
        "reference": {"time": "mean"},
    }


def test_a_one_sided_pair_entry_is_refused_before_anything_runs():
    with pytest.raises(ValueError, match="needs both 'test' and 'reference'"):
        _compare(aggregate=[MEAN, {"test": {"time": "mean"}}])


def test_a_one_element_list_still_returns_a_set_and_leaves_the_label_alone():
    fanned = _compare(aggregate=[SEASONAL_VAR])
    plain = _compare(aggregate=SEASONAL_VAR)
    assert isinstance(fanned, ComparisonSet)
    assert [c.aggregate for c in fanned] == [SEASONAL_VAR]
    assert [c.label for c in fanned] == [c.label for c in plain]


def test_a_tuple_fans_like_a_list():
    assert len(_compare(aggregate=(MEAN, SEASONAL_VAR))) == 2


def test_an_empty_list_is_an_error():
    with pytest.raises(ValueError, match=r"aggregate=\[\]"):
        _compare(aggregate=[])


def test_exact_repeats_are_dropped_with_a_note(capsys):
    reordered = {"time": "mean"}  # same spec, separately built
    out = _compare(aggregate=[MEAN, SEASONAL_VAR, reordered])
    assert len(out) == 2
    assert "aggregate[2] repeats aggregate[0]; dropped" in capsys.readouterr().out


def test_repeats_are_found_through_key_order_in_a_pair_spec(capsys):
    a = {"test": {"time": "mean"}, "reference": {"time": "mean"}}
    b = {"reference": {"time": "mean"}, "test": {"time": "mean"}}
    assert len(_compare(aggregate=[a, b])) == 1
    assert "repeats" in capsys.readouterr().out


def test_a_deduped_list_that_leaves_one_spec_leaves_the_label_alone():
    out = _compare(aggregate=[MEAN, {"time": "mean"}])
    assert [c.label for c in out] == [c.label for c in _compare(aggregate=MEAN)]


# -- times= ---------------------------------------------------------------------


def test_a_time_entry_in_the_list_is_refused_with_resample_times():
    with pytest.raises(ValueError, match="times="):
        _compare(
            times={"resample": "1MS", "reduce": "mean"},
            aggregate=[{"lat": "mean"}, MEAN],
        )


def test_a_time_entry_in_the_list_is_refused_with_season_times():
    with pytest.raises(ValueError, match="times="):
        _compare(
            times={"groupby": "season", "reduce": "mean"},
            aggregate=[MEAN, SEASONAL_VAR],
        )


def test_the_refusal_comes_before_any_member_is_built():
    built = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=DECLARED[n])
        ),
        mock.patch.object(
            Comparison, "align", lambda self, refresh=False: built.append(self)
        ),
        pytest.raises(ValueError),
    ):
        comparison.compare(
            reference="obs",
            test="model",
            variables=["temperature"],
            times={"groupby": "season", "reduce": "mean"},
            aggregate=[{"lat": "mean"}, MEAN],
        )
    assert built == []


def test_times_list_follows_the_single_spec_rule():
    """An explicit times=[...] list was always allowed beside a time aggregate."""
    out = _compare(times=["2010-01", "2010-02"], aggregate=[MEAN, {"time": "max"}])
    assert len(out) == 4


# -- labels -----------------------------------------------------------------------


def test_labels_name_the_statistic():
    out = _compare(aggregate=[MEAN, SEASONAL_VAR])
    assert [c.label for c in out] == [
        "temperature mean",
        "temperature variance of monthly means",
    ]


def test_labels_keep_the_depth_part_and_stay_unique():
    out = _compare(depths=["surface", 200], aggregate=[MEAN, SEASONAL_VAR])
    labels = [c.label for c in out]
    assert len(set(labels)) == 4
    assert labels[0].endswith(" mean")
    assert labels[2].endswith("variance of monthly means")
    assert "200" in labels[1]


@pytest.mark.parametrize(
    ("spec", "label"),
    [
        ({"time": "mean"}, "mean"),
        ({"time": "var"}, "variance"),
        ({"time": "max"}, "maximum"),
        ({"time": {"groupby": "month", "reduce": "mean"}}, "monthly means"),
        (SEASONAL_VAR, "variance of monthly means"),
        ({"time": "mean", "lat": "mean"}, "mean, lat mean"),
        (None, "unaggregated"),
        (
            {"test": SEASONAL_VAR, "reference": {"time": "mean"}},
            "test variance of monthly means, reference mean",
        ),
        ({"test": MEAN, "reference": {"time": "mean"}}, "mean"),
    ],
)
def test_aggregate_label_spells_every_axis(spec, label):
    assert _aggregate_label(spec) == label


def test_distinct_specs_that_read_alike_are_numbered():
    fan = _aggregate_fan(
        [{"time": "mean"}, {"time": {"reduce": "mean"}}], caller="compare()"
    )
    assert [label for _, label in fan] == ["mean", "mean (2)"]


# -- identity / caching -----------------------------------------------------------


def test_members_are_keyed_by_their_own_spec():
    out = _compare(aggregate=[MEAN, SEASONAL_VAR])
    keys = {c._cache_key for c in out}
    assert len(keys) == 2
    # ...and each is the key the same call with that one spec would have had.
    assert out[1]._cache_key == _compare(aggregate=SEASONAL_VAR)[0]._cache_key


def test_pooling_keeps_both_members():
    out = _compare(aggregate=[MEAN, SEASONAL_VAR])
    assert len(ComparisonSet([out, out])) == 2


# -- a list under an axis is a chain, not a fan -----------------------------------


def test_a_chain_under_an_axis_is_one_member():
    out = _compare(aggregate=SEASONAL_VAR)
    assert len(out) == 1
    assert out[0].aggregate == SEASONAL_VAR


def test_a_chain_spelled_as_the_top_level_list_is_refused_with_the_fix():
    with pytest.raises(TypeError, match="put the list under the axis"):
        _compare(aggregate=[{"groupby": "month", "reduce": "mean"}, "var"])


# -- field() ----------------------------------------------------------------------


def test_field_fans_into_a_fieldset():
    import ocean_skill as osk

    out = osk.field("model", "temperature", aggregate=[MEAN, SEASONAL_VAR])
    assert isinstance(out, osk.FieldSet)
    assert [f.aggregate for f in out] == [MEAN, SEASONAL_VAR]
    assert [f.label for f in out] == ["model mean", "model variance of monthly means"]


def test_field_fan_crosses_the_variable_fan():
    import ocean_skill as osk

    out = osk.field(
        "model", ["temperature", "salinity"], aggregate=[MEAN, SEASONAL_VAR]
    )
    assert len(out) == 4
    assert len({f.label for f in out}) == 2  # same source, label varies by statistic


def test_field_one_element_list_still_returns_a_fieldset():
    import ocean_skill as osk

    out = osk.field("model", "temperature", aggregate=[MEAN])
    assert isinstance(out, osk.FieldSet)
    assert len(out) == 1
    assert out[0].label is None


def test_field_label_is_extended_not_replaced():
    import ocean_skill as osk

    out = osk.field("model", "temperature", aggregate=[MEAN, SEASONAL_VAR], label="run")
    assert [f.label for f in out] == ["run mean", "run variance of monthly means"]


def test_field_drops_repeats_with_a_note(capsys):
    import ocean_skill as osk

    out = osk.field("model", "temperature", aggregate=[MEAN, {"time": "mean"}])
    assert len(out) == 1
    assert "repeats" in capsys.readouterr().out


def test_field_empty_list_is_an_error():
    import ocean_skill as osk

    with pytest.raises(ValueError, match=r"aggregate=\[\]"):
        osk.field("model", "temperature", aggregate=[])


def test_field_still_refuses_a_pair_spec_member():
    import ocean_skill as osk

    pair = {"test": {"time": "mean"}, "reference": {"time": "mean"}}
    with pytest.raises(TypeError, match="pair-spec"):
        osk.field("model", "temperature", aggregate=[MEAN, pair])


def test_field_refuses_the_fan_beside_a_cross_select():
    import ocean_skill as osk

    with pytest.raises(ValueError, match="cross"):
        osk.field(
            "model",
            "temperature",
            select={"transect": {"cross": {"lon": -150, "lat": 50}}},
            aggregate=[MEAN, SEASONAL_VAR],
        )


# -- suite pages ------------------------------------------------------------------

INDEX = pd.date_range("2010-01-05", periods=10, freq="7D")


@pytest.fixture
def _stub_time_index(monkeypatch):
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: INDEX)


def _fig4_suite():
    return SuiteConfig.model_validate(
        {
            "name": "fig4",
            "defaults": {"test": "model"},
            "pages": [
                {
                    "title": "Figure 4",
                    "compare": {
                        "reference": ["obs"],
                        "variables": ["temperature"],
                        "depths": ["surface", 200],
                        "aggregate": [
                            {"time": "mean"},
                            {"time": [{"groupby": "month", "reduce": "mean"}, "var"]},
                        ],
                    },
                }
            ],
        }
    )


def test_a_suite_page_carries_the_aggregate_list_through_expand(_stub_time_index):
    (page,) = P.expand(_fig4_suite())
    assert page.kwargs["aggregate"] == [MEAN, SEASONAL_VAR]
    assert page.cache is True  # the injected whole-run window still decides this


def test_a_suite_page_builds_a_fanned_set(_stub_time_index):
    (page,) = P.expand(_fig4_suite())
    plotted = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=DECLARED[n])
        ),
        mock.patch.object(Comparison, "align", lambda self, refresh=False: None),
        mock.patch.object(Comparison, "metrics", lambda self: {"label": self.label}),
        mock.patch.object(
            Comparison, "family", new_callable=mock.PropertyMock, return_value="map"
        ),
        mock.patch.object(
            ComparisonSet,
            "plot",
            lambda self, **kw: plotted.append(self) or "fig",
        ),
    ):
        ((suffix, fig),) = P.build(page)
    assert (suffix, fig) == ("", "fig")
    (drawn,) = plotted
    assert len(drawn) == 4
    assert len({c.label for c in drawn}) == 4
    assert [c.aggregate for c in drawn[:2]] == [MEAN, MEAN]
    assert page.metrics_records and len(page.metrics_records) == 4


def test_a_then_chain_counts_the_aggregate_fan_as_members(_stub_time_index):
    suite = SuiteConfig.model_validate(
        {
            "name": "t",
            "defaults": {"test": "model"},
            "pages": [
                {
                    "title": "x",
                    "field": {
                        "variables": ["temperature"],
                        "aggregate": [MEAN, SEASONAL_VAR],
                    },
                    "then": ["extremum"],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="member"):
        P.expand(suite)
