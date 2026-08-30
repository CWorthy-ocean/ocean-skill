"""Tests for select/aggregate pair-specs and the deferred (post-aggregate) select.

The motivating case: a model spanning several months compared against a WOA monthly
climatology. The model needs ``{"time": {"groupby": "month", "reduce": "mean"}}``
then one month picked out of the result; the climatology (its ``time`` read with
``decode_times=False``, since a climatology has no calendar year) just needs its one
time step meaned away -- a shared groupby would crash on it (no ``.dt`` accessor on
an undecoded numeric axis), and a shared ``select={"time": "2010-01"}`` fails the
same way trying to match a date string against it.

This mirrors ``test_pair_spec.py`` (the variable pair-spec this generalizes) in
structure and mocking style.
"""

from __future__ import annotations

import warnings
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from ocean_skill import cache as _cache
from ocean_skill import comparison
from ocean_skill.comparison import (
    Comparison,
    _display_depth,
    _identity,
    _prepare,
    aggregate_for,
    select_for,
)

NITRATE = "mole_concentration_of_nitrate_in_sea_water"


# -- select_for/aggregate_for: mirror variable_for exactly ---------------------


def test_select_for_picks_the_named_side():
    pair = {"test": {"month": 1}, "reference": {}}
    assert select_for(pair, "test") == {"month": 1}
    assert select_for(pair, "reference") == {}


def test_select_for_passes_a_flat_select_through_unchanged():
    flat = {"depth": "surface", "time": "2012-01"}
    assert select_for(flat, "test") == flat
    assert select_for(flat, "reference") == flat


def test_aggregate_for_picks_the_named_side():
    pair = {
        "test": {"time": {"groupby": "month", "reduce": "mean"}},
        "reference": {"time": "mean"},
    }
    assert aggregate_for(pair, "test") == {
        "time": {"groupby": "month", "reduce": "mean"}
    }
    assert aggregate_for(pair, "reference") == {"time": "mean"}


def test_aggregate_for_passes_a_flat_or_none_aggregate_through():
    assert aggregate_for({"time": "mean"}, "test") == {"time": "mean"}
    assert aggregate_for(None, "reference") is None


# -- validation: pair intent is caught up front, with a clear message ----------


def test_a_one_sided_select_names_the_missing_key():
    with pytest.raises(ValueError, match="reference"):
        Comparison(reference="r", test="t", variable="a", select={"test": {"month": 1}})


def test_a_one_sided_aggregate_names_the_missing_key():
    with pytest.raises(ValueError, match="test"):
        Comparison(
            reference="r",
            test="t",
            variable="a",
            aggregate={"reference": {"time": "mean"}},
        )


def test_a_select_pair_spec_refuses_an_extra_key():
    with pytest.raises(ValueError, match="extra"):
        Comparison(
            reference="r",
            test="t",
            variable="a",
            select={"test": {}, "reference": {}, "depth": "surface"},
        )


def test_an_aggregate_pair_side_must_be_a_dict_or_none():
    with pytest.raises(TypeError, match="dict or None"):
        Comparison(
            reference="r",
            test="t",
            variable="a",
            aggregate={"test": "mean", "reference": {"time": "mean"}},
        )


def test_a_plain_select_naming_no_pair_keys_is_untouched():
    """The ordinary case -- select={"depth": ...} -- must not be mistaken for a pair."""
    c = Comparison(reference="r", test="t", variable="a", select={"depth": "surface"})
    assert c.select == {"depth": "surface"}


# -- Comparison normalizes and resolves per lane --------------------------------


def test_comparison_normalizes_a_pair_select():
    c = Comparison(
        reference="r",
        test="t",
        variable="a",
        select={"test": {"month": 1}, "reference": {}},
    )
    assert c.select == {"test": {"month": 1}, "reference": {}}


def test_prepare_lane_resolves_each_sides_own_select_and_aggregate():
    """The mechanism the feature exists for.

    Each lane's prepare() call gets only its own side of a pair select/aggregate,
    flat, by the time it reaches prepare_source -- everything below stays unaware a
    pair was ever involved.
    """
    c = Comparison(
        reference="woa23_nitrate_month01",
        test="model",
        variable=NITRATE,
        select={"test": {"month": 1}, "reference": {}},
        aggregate={
            "test": {"time": {"groupby": "month", "reduce": "mean"}},
            "reference": {"time": "mean"},
        },
    )
    calls = {}

    def fake_prepare_source(source, variable, select, aggregate, **kwargs):
        calls[kwargs.get("require_reduced") or "test"] = (select, aggregate)
        return None, None

    with mock.patch("ocean_skill.comparison.prepare_source", fake_prepare_source):
        c._prepare_lane("model", True, False, role="test")
        c._prepare_lane("woa23_nitrate_month01", True, False, role="reference")

    assert calls["test"] == (
        {"month": 1},
        {"time": {"groupby": "month", "reduce": "mean"}},
    )
    assert calls["reference"] == ({}, {"time": "mean"})


# -- the deferred select: a key naming an axis the aggregate creates -----------


@pytest.fixture
def monthly():
    """Build a dataset whose value is its own month, so picking one is checkable."""
    time = xr.date_range("2012-01-01", periods=24, freq="MS")
    # month k (1-indexed) is worth k, repeated across the two years -- so a groupby
    # mean by month recovers exactly k for month k, not some blend of two years.
    values = np.array([m.month for m in time], dtype=float)
    da = xr.DataArray(
        values[:, None, None] * np.ones((1, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [1.0, 2.0], "lon": [1.0, 2.0]},
        attrs={"units": "1"},
    )
    return xr.Dataset({"a": da})


def test_a_select_key_the_aggregate_creates_is_applied_after_it(monthly):
    """This is the deferred second try.

    select={"month": 3} matches nothing before the groupby runs (there is no
    `month` dim yet) and everything after.
    """
    da, _ = _prepare(
        monthly,
        {},
        "a",
        {"month": 3},
        {"time": {"groupby": "month", "reduce": "mean"}},
    )
    assert "month" not in da.dims
    assert float(da.isel(lat=0, lon=0)) == pytest.approx(3.0)


def test_a_resample_select_is_applied_once_not_twice(monthly):
    """Regression guard for the refactor.

    A select key that *does* match before the aggregate (resample keeps the dim
    named "time", unlike groupby) must not be re-applied a second time after --
    narrowing to May twice is still just May, so this only fails if the second pass
    narrowed something already-resampled to an empty result instead of leaving it
    alone.
    """
    da, _ = _prepare(
        monthly,
        {},
        "a",
        {"time": "2012-05"},
        {"time": {"resample": "1MS", "reduce": "mean"}},
    )
    # one monthly bin, standing -- resample doesn't drop it
    assert da.sizes["time"] == 1
    assert float(da.isel(time=0, lat=0, lon=0)) == pytest.approx(5.0)


def test_a_season_select_key_the_aggregate_creates_is_applied_after_it(monthly):
    """The season analogue of the month test above -- and the mechanism the WOA
    per-lane recipe in this module's own docstring needs for a seasonal test
    lane: {"season": "JJA"} matches nothing before the groupby runs and exactly
    one field after."""
    da, _ = _prepare(
        monthly,
        {},
        "a",
        {"season": "JJA"},
        {"time": {"groupby": "season", "reduce": "mean"}},
    )
    assert "season" not in da.dims
    # month k is worth k; JJA averages months 6, 7, 8
    assert float(da.isel(lat=0, lon=0)) == pytest.approx((6 + 7 + 8) / 3)


def test_seasonal_test_lane_against_a_time_meaned_reference():
    """The seasonal analogue of the WOA month recipe this module documents: a
    model spanning several years needs a season picked out of its own
    climatology, while a reference already reduced to one field just needs its
    lone time step meaned away -- both through the same per-lane aggregate/select
    pair-spec machinery, unchanged."""
    c = Comparison(
        reference="woa23_nitrate_month01",
        test="model",
        variable=NITRATE,
        select={"test": {"season": "JJA"}, "reference": {}},
        aggregate={
            "test": {"time": {"groupby": "season", "reduce": "mean"}},
            "reference": {"time": "mean"},
        },
    )
    calls = {}

    def fake_prepare_source(source, variable, select, aggregate, **kwargs):
        calls[kwargs.get("require_reduced") or "test"] = (select, aggregate)
        return None, None

    with mock.patch("ocean_skill.comparison.prepare_source", fake_prepare_source):
        c._prepare_lane("model", True, False, role="test")
        c._prepare_lane("woa23_nitrate_month01", True, False, role="reference")

    assert calls["test"] == (
        {"season": "JJA"},
        {"time": {"groupby": "season", "reduce": "mean"}},
    )
    assert calls["reference"] == ({}, {"time": "mean"})


def test_a_key_matching_neither_pass_warns_but_does_not_raise(monthly):
    with pytest.warns(UserWarning, match="matched no axis"):
        da, _ = _prepare(
            monthly, {}, "a", {"nonexistent_axis": 1}, {"time": "mean"}
        )
    assert "time" not in da.dims  # the aggregate still ran; only the bad key warned


def test_a_shared_select_absent_from_this_variable_alone_still_works(monthly):
    """The ordinary case select= is designed for.

    An axis this particular variable doesn't have is not treated as an error, only
    warned about if it never matches at all; here "lat" is real, so no warning and
    no crash either way.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        da, _ = _prepare(monthly, {}, "a", {"lat": 1.0}, {"time": "mean"})
    assert "lat" not in da.dims


# -- compare()'s depths=/select vertical sugar, pair-aware ----------------------


def test_compare_fans_depth_into_both_sides_of_a_pair_select():
    formed = []
    declared = {"model": {"variables": []}, "obs": {"variables": []}}
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.select),
        ),
    ):
        comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=["temperature"],
            select={"test": {"month": 1}, "reference": {}},
            depths=(50,),
        )
    assert formed == [{"test": {"month": 1, "depth": 50}, "reference": {"depth": 50}}]


def test_compare_refuses_a_pair_select_that_disagrees_on_depth():
    declared = {"model": {"variables": []}, "obs": {"variables": []}}
    with mock.patch(
        "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
    ):
        with pytest.raises(ValueError, match="disagrees"):
            comparison.compare(
                reference=["obs"],
                test=["model"],
                variables=["temperature"],
                select={"test": {"depth": 50}, "reference": {"depth": 100}},
            )


def test_compare_derives_depths_default_from_an_agreeing_pair_select():
    """depths=/select= sugar defaults from the vertical entry already in select=.

    Still works when that entry lives inside a pair-spec select.
    """
    formed = []
    declared = {"model": {"variables": []}, "obs": {"variables": []}}
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.select),
        ),
    ):
        comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=["temperature"],
            select={"test": {"depth": 75}, "reference": {}},
        )
    assert formed == [{"test": {"depth": 75}, "reference": {"depth": 75}}]


# -- cache keys and pooling identity ---------------------------------------------


def test_cache_key_is_distinct_for_a_pair_select_vs_a_flat_one():
    common = dict(test="t", reference="r", variable="a", method="conservative_normed")
    flat_key = _cache.key_for(select={"depth": "surface"}, **common)
    pair_key = _cache.key_for(
        select={"test": {"depth": "surface"}, "reference": {}}, **common
    )
    assert flat_key != pair_key


def test_cache_key_changes_when_either_side_of_a_pair_select_changes():
    common = dict(test="t", reference="r", variable="a", method="conservative_normed")
    base = _cache.key_for(select={"test": {"month": 1}, "reference": {}}, **common)
    other_test = _cache.key_for(
        select={"test": {"month": 2}, "reference": {}}, **common
    )
    other_ref = _cache.key_for(
        select={"test": {"month": 1}, "reference": {"depth": 50}}, **common
    )
    assert len({base, other_test, other_ref}) == 3


def test_identity_is_insensitive_to_pair_select_key_order():
    kw = dict(reference="r", test="t", variable="a")
    c1 = Comparison(**kw, select={"test": {"month": 1}, "reference": {}})
    c2 = Comparison(**kw, select={"reference": {}, "test": {"month": 1}})
    assert _identity(c1) == _identity(c2)


# -- Field: no second lane to give the other side of a pair to ------------------


def test_field_refuses_a_pair_select():
    from ocean_skill.field import Field

    with pytest.raises(TypeError, match="pair-spec select"):
        Field("model", "temperature", select={"test": {"month": 1}, "reference": {}})


def test_field_refuses_a_pair_aggregate():
    from ocean_skill.field import Field

    with pytest.raises(TypeError, match="pair-spec aggregate"):
        Field(
            "model",
            "temperature",
            aggregate={"test": {"time": "mean"}, "reference": {"time": "mean"}},
        )


def test_field_refuses_a_one_sided_select_the_same_way_comparison_does():
    from ocean_skill.field import Field

    with pytest.raises(ValueError, match="reference"):
        Field("model", "temperature", select={"test": {"month": 1}})


# -- labels: a pair select reports the test side's depth ------------------------


def test_display_depth_on_a_pair_select_reads_the_test_side():
    pair_select = {"test": {"depth": 50}, "reference": {}}
    assert _display_depth("temperature", pair_select) == 50
