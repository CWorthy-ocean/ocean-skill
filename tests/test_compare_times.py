"""``compare(times=...)``: one comparison per time bin, ``depths=``'s time analogue.

A dict form derives bins from the **test** source's own time axis (in the
``aggregate={"time": ...}`` vocabulary); a list form names an explicit set of
values, one comparison per entry, with no bin derivation and no aggregate merged
in. Mirrors ``test_select_aggregate_pair_spec.py`` in structure and mocking style:
fan-shape/label assertions mock ``catalog.resolve`` and stub ``Comparison.align``
so nothing here touches real data or the disk cache, and validation errors that
fire before any catalog lookup are asserted with no mocking at all.
"""

from __future__ import annotations

import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import comparison
from ocean_skill.comparison import (
    _display_time,
    _fanned_season_select,
    _fanned_time_select,
    _has_season_entry,
    _has_time_entry,
    _merged_season_aggregate,
    _merged_time_aggregate,
    _normalize_time_value,
    _normalize_times,
    _selected_time,
    _time_bins,
    _time_label,
    _time_select_value,
)

TEMPERATURE = "sea_water_potential_temperature"


# -- _normalize_times: what times= means -----------------------------------------


def test_none_means_no_fan():
    assert _normalize_times(None) is None


def test_a_dict_with_resample_and_reduce_is_the_bins_form():
    assert _normalize_times({"resample": "1MS", "reduce": "mean"}) == (
        "bins",
        {"resample": "1MS", "reduce": "mean"},
    )


def test_a_dict_missing_resample_or_reduce_is_refused():
    with pytest.raises(ValueError, match="needs both 'resample'"):
        _normalize_times({"resample": "1MS"})
    with pytest.raises(ValueError, match="needs both 'resample'"):
        _normalize_times({"reduce": "mean"})


def test_a_groupby_dict_is_refused_as_a_climatology_not_a_fan():
    with pytest.raises(ValueError, match="climatology"):
        _normalize_times({"groupby": "month", "reduce": "mean"})


def test_a_groupby_season_dict_is_the_seasons_form():
    from ocean_skill.operators import DEFAULT_SEASONS

    assert _normalize_times({"groupby": "season", "reduce": "mean"}) == (
        "seasons",
        {"groupby": "season", "reduce": "mean", "seasons": list(DEFAULT_SEASONS)},
    )


def test_custom_seasons_are_validated_and_order_preserved():
    kind, spec = _normalize_times(
        {"groupby": "season", "reduce": "mean", "seasons": ["jfma", "mjja", "sond"]}
    )
    assert kind == "seasons"
    assert spec["seasons"] == ["JFMA", "MJJA", "SOND"]


def test_seasons_form_requires_reduce():
    with pytest.raises(ValueError, match="needs 'reduce'"):
        _normalize_times({"groupby": "season"})


def test_seasons_form_refuses_resample_alongside():
    with pytest.raises(ValueError, match="different ways to fan"):
        _normalize_times({"groupby": "season", "resample": "1MS", "reduce": "mean"})


def test_an_invalid_season_fails_before_any_source_is_opened():
    with pytest.raises(ValueError, match="not a run of consecutive months"):
        _normalize_times({"groupby": "season", "reduce": "mean", "seasons": ["XYZ"]})


def test_a_pair_spec_shaped_dict_is_refused():
    with pytest.raises(ValueError, match="takes no"):
        _normalize_times({"test": {}, "reference": {}})


def test_a_bare_string_is_a_one_entry_list():
    assert _normalize_times("2010-01") == ("list", ("2010-01",))


def test_a_list_of_strings_passes_through():
    assert _normalize_times(["2010-01", "2010-02"]) == (
        "list",
        ("2010-01", "2010-02"),
    )


def test_a_timestamp_in_the_list_is_normalized_to_a_string():
    kind, values = _normalize_times([pd.Timestamp("2010-01-15")])
    assert kind == "list"
    assert values == ("2010-01-15 00:00:00",)


def test_a_min_max_window_in_the_list_normalizes_its_own_values_to_strings():
    window = {"min": pd.Timestamp("2010-01-01"), "max": "2010-01-31"}
    kind, values = _normalize_times([window])
    assert kind == "list"
    assert values == ({"min": "2010-01-01 00:00:00", "max": "2010-01-31"},)


# -- _normalize_time_value / _time_label / _has_time_entry -----------------------


def test_normalize_time_value_leaves_a_plain_string_alone():
    assert _normalize_time_value("2010-01") == "2010-01"


def test_time_label_of_a_plain_value_is_itself():
    assert _time_label("2010-01") == "2010-01"


def test_time_label_of_a_window_joins_both_ends():
    assert (
        _time_label({"min": "2010-01-01T00:00:00", "max": "2010-03-31T00:00:00"})
        == "2010-01-01–2010-03-31"
    )


def test_has_time_entry_reads_either_side_of_a_pair_spec():
    assert _has_time_entry({"time": "2010-01"}) is True
    assert _has_time_entry({"depth": 50}) is False
    assert _has_time_entry(None) is False
    assert _has_time_entry({"test": {"time": "2010-01"}, "reference": {}}) is True
    assert _has_time_entry({"test": {}, "reference": {}}) is False


# -- _selected_time / _display_time: the label/repr/metrics reader ----------------


def test_selected_time_reads_any_accepted_spelling():
    assert _selected_time({"time": "2010-01"}) == "2010-01"
    assert _selected_time({"T": "2010-01"}) == "2010-01"
    assert _selected_time({"depth": 50}) is None


def test_display_time_reads_the_test_side_of_a_pair_spec():
    pair = {"test": {"time": "2010-01"}, "reference": {}}
    assert _display_time(pair) == "2010-01"


def test_selected_time_reads_a_scalar_season_first():
    """A season fan's distinguishing identity, even alongside a time window that
    only narrows which years feed the climatology."""
    assert _selected_time({"season": "JJA"}) == "JJA"
    assert _selected_time({"season": "JJA", "time": {"min": "2010", "max": "2012"}}) == "JJA"


# -- _fanned_time_select: strip-then-write-back, like _fanned_select -------------


def test_fanned_time_select_replaces_a_flat_selects_time_entry():
    assert _fanned_time_select({"depth": 50, "time": "2010-01"}, "2010-02") == {
        "depth": 50,
        "time": "2010-02",
    }


def test_fanned_time_select_writes_both_sides_of_a_pair_spec():
    sel = {"test": {"depth": 50}, "reference": {}}
    assert _fanned_time_select(sel, "2010-02") == {
        "test": {"depth": 50, "time": "2010-02"},
        "reference": {"time": "2010-02"},
    }


# -- _merged_time_aggregate --------------------------------------------------------


def test_merged_time_aggregate_folds_into_a_flat_aggregate():
    entry = {"resample": "1MS", "reduce": "mean"}
    # 'resample' is dropped: the per-bin select already isolates the period, so
    # the aggregate only has to collapse it -- a plain reduction, not a resample
    # (which would keep a size-1 time axis _require_reduced rejects).
    reduction = {"reduce": "mean"}
    assert _merged_time_aggregate(None, entry) == {"time": reduction}
    assert _merged_time_aggregate({"Z": "mean"}, entry) == {
        "Z": "mean",
        "time": reduction,
    }


def test_merged_time_aggregate_folds_into_both_sides_of_a_pair_spec():
    entry = {"resample": "1MS", "reduce": "mean"}
    reduction = {"reduce": "mean"}
    pair = {"test": {"Z": "mean"}, "reference": None}
    assert _merged_time_aggregate(pair, entry) == {
        "test": {"Z": "mean", "time": reduction},
        "reference": {"time": reduction},
    }


# -- _fanned_season_select / _merged_season_aggregate / _has_season_entry --------


def test_has_season_entry_reads_either_side_of_a_pair_spec():
    assert _has_season_entry({"season": "JJA"}) is True
    assert _has_season_entry({"depth": 50}) is False
    assert _has_season_entry({"test": {"season": "JJA"}, "reference": {}}) is True
    assert _has_season_entry({"test": {}, "reference": {}}) is False


def test_fanned_season_select_writes_a_flat_select():
    assert _fanned_season_select({"depth": 50}, "JJA") == {"depth": 50, "season": "JJA"}


def test_fanned_season_select_writes_both_sides_of_a_pair_spec():
    sel = {"test": {"depth": 50}, "reference": {}}
    assert _fanned_season_select(sel, "JJA") == {
        "test": {"depth": 50, "season": "JJA"},
        "reference": {"season": "JJA"},
    }


def test_fanned_season_select_keeps_a_time_window():
    """Unlike the bins form's strip-then-replace, a season select narrows an axis
    the aggregate creates -- an existing time entry (which years feed the
    climatology) is orthogonal to it and survives untouched."""
    sel = {"time": {"min": "2010", "max": "2012"}}
    assert _fanned_season_select(sel, "JJA") == {
        "time": {"min": "2010", "max": "2012"},
        "season": "JJA",
    }


def test_merged_season_aggregate_narrows_to_one_season_per_comparison():
    entry = {"groupby": "season", "reduce": "mean", "seasons": ["DJF", "JJA"]}
    assert _merged_season_aggregate(None, entry, "JJA") == {
        "time": {"groupby": "season", "reduce": "mean", "seasons": ["JJA"]}
    }
    assert _merged_season_aggregate({"Z": "mean"}, entry, "JJA") == {
        "Z": "mean",
        "time": {"groupby": "season", "reduce": "mean", "seasons": ["JJA"]},
    }


def test_merged_season_aggregate_keeps_groupby_unlike_the_bins_form():
    """The bins form drops 'resample' because select already isolates the bin;
    a season select narrows an axis the aggregate itself has to create, so
    'groupby' has to survive for that axis to exist at all."""
    entry = {"groupby": "season", "reduce": "mean", "seasons": ["DJF"]}
    reduction = _merged_season_aggregate(None, entry, "DJF")["time"]
    assert reduction["groupby"] == "season"


def test_merged_season_aggregate_folds_into_both_sides_of_a_pair_spec():
    entry = {"groupby": "season", "reduce": "mean", "seasons": ["JJA"]}
    pair = {"test": {"Z": "mean"}, "reference": None}
    assert _merged_season_aggregate(pair, entry, "JJA") == {
        "test": {"Z": "mean", "time": {"groupby": "season", "reduce": "mean", "seasons": ["JJA"]}},
        "reference": {"time": {"groupby": "season", "reduce": "mean", "seasons": ["JJA"]}},
    }


def test_merged_season_aggregate_preserves_spread():
    entry = {"groupby": "season", "reduce": "mean", "spread": "std", "seasons": ["JJA"]}
    reduction = _merged_season_aggregate(None, entry, "JJA")["time"]
    assert reduction["spread"] == "std"


# -- _time_bins / _time_select_value: reading a source's own time axis -----------


@pytest.fixture
def daily_dataset():
    """Three months of daily data on a decoded calendar time axis."""
    time = pd.date_range("2010-01-01", "2010-03-31", freq="D")
    return xr.Dataset(
        {"v": (("time",), np.arange(len(time), dtype=float))}, coords={"time": time}
    )


def test_monthly_bins_cover_each_month_present(monkeypatch, daily_dataset):
    monkeypatch.setattr(
        "ocean_skill.sources.read", lambda source, **k: daily_dataset
    )
    bins = _time_bins("fake", "1MS", None)
    starts = [str(s)[:7] for s, _ in bins]
    assert starts == ["2010-01", "2010-02", "2010-03"]


def test_a_window_narrows_which_bins_are_returned(monkeypatch, daily_dataset):
    monkeypatch.setattr(
        "ocean_skill.sources.read", lambda source, **k: daily_dataset
    )
    bins = _time_bins("fake", "1MS", "2010-02")
    assert [str(s)[:7] for s, _ in bins] == ["2010-02"]


def test_a_period_window_matching_nothing_raises_valueerror(monkeypatch, daily_dataset):
    monkeypatch.setattr(
        "ocean_skill.sources.read", lambda source, **k: daily_dataset
    )
    with pytest.raises(ValueError, match="matched no data"):
        _time_bins("fake", "1MS", "2099-01")


def test_a_slice_window_matching_nothing_raises_valueerror(monkeypatch, daily_dataset):
    monkeypatch.setattr(
        "ocean_skill.sources.read", lambda source, **k: daily_dataset
    )
    with pytest.raises(ValueError, match="matched no data"):
        _time_bins("fake", "1MS", {"min": "2099-01-01", "max": "2099-06-01"})


def test_no_time_axis_at_all_raises_valueerror(monkeypatch):
    static = xr.Dataset({"v": (("lat",), np.arange(5, dtype=float))})
    monkeypatch.setattr("ocean_skill.sources.read", lambda source, **k: static)
    with pytest.raises(ValueError, match="has no time axis"):
        _time_bins("static", "1MS", None)


def test_an_undecoded_numeric_time_axis_raises_valueerror(monkeypatch):
    numeric = xr.Dataset(
        {"v": (("time",), np.arange(12, dtype=float))},
        coords={"time": np.arange(12, dtype=float)},
    )
    monkeypatch.setattr("ocean_skill.sources.read", lambda source, **k: numeric)
    with pytest.raises(ValueError, match="not a decoded calendar axis"):
        _time_bins("woa", "1MS", None)


def test_a_cftime_axis_bins_the_same_way_as_a_numpy_one(monkeypatch):
    time = xr.date_range("2010-01-01", periods=90, freq="D", use_cftime=True)
    cf = xr.Dataset(
        {"v": (("time",), np.arange(90, dtype=float))}, coords={"time": time}
    )
    monkeypatch.setattr("ocean_skill.sources.read", lambda source, **k: cf)
    bins = _time_bins("roms", "1MS", None)
    labels = [_time_select_value(s, last, "1MS") for s, last in bins]
    assert labels == ["2010-01", "2010-02", "2010-03"]


def test_time_select_value_uses_a_partial_date_for_whole_calendar_units(
    monkeypatch, daily_dataset
):
    monkeypatch.setattr(
        "ocean_skill.sources.read", lambda source, **k: daily_dataset
    )
    (start, last), *_ = _time_bins("fake", "1MS", None)
    assert _time_select_value(start, last, "1MS") == "2010-01"


def test_time_select_value_falls_back_to_a_window_for_other_frequencies(
    monkeypatch, daily_dataset
):
    monkeypatch.setattr(
        "ocean_skill.sources.read", lambda source, **k: daily_dataset
    )
    (start, last), *_ = _time_bins("fake", "3MS", None)
    value = _time_select_value(start, last, "3MS")
    assert isinstance(value, dict) and {"min", "max"} <= set(value)


def test_an_empty_time_axis_raises_valueerror_before_resampling(monkeypatch):
    """A degenerate source, not a windowing outcome -- pins the earlier check."""
    empty = xr.Dataset(
        {"v": (("time",), np.array([], dtype=float))},
        coords={"time": np.array([], dtype="datetime64[ns]")},
    )
    monkeypatch.setattr("ocean_skill.sources.read", lambda source, **k: empty)
    with pytest.raises(ValueError, match="has no data along its time axis"):
        _time_bins("fake", "1MS", None)


# -- compare()'s times= fan: list form --------------------------------------------


@pytest.fixture
def stubbed_fan():
    """Mock the catalog and record each fanned comparison's select/aggregate/label."""
    formed = []
    declared = {"model": {"variables": []}, "obs": {"variables": []}}
    patches = (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(
                (self.select, self.aggregate, self.label)
            ),
        ),
    )
    with patches[0], patches[1]:
        yield formed


def test_a_list_fans_one_comparison_per_entry(stubbed_fan):
    comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        times=["2010-01", "2010-02"],
    )
    assert stubbed_fan == [
        ({"depth": "surface", "time": "2010-01"}, None, "2010-01"),
        ({"depth": "surface", "time": "2010-02"}, None, "2010-02"),
    ]


def test_a_single_entry_list_does_not_earn_its_own_label(stubbed_fan):
    """Mirrors depths=(50,) alone: one value never disambiguates anything."""
    comparison.compare(
        reference=["obs"], test=["model"], variables=["temperature"], times=["2010-01"]
    )
    assert stubbed_fan == [
        ({"depth": "surface", "time": "2010-01"}, None, "temperature")
    ]


def test_a_list_writes_time_into_both_sides_of_a_pair_spec_select(stubbed_fan):
    comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        select={"test": {"depth": 50}, "reference": {}},
        times=["2010-01"],
    )
    # depths=/select= sugar (unrelated to times=) already writes the depth it
    # derives into both sides, same as with no times= involved at all.
    assert stubbed_fan == [
        (
            {
                "test": {"depth": 50, "time": "2010-01"},
                "reference": {"depth": 50, "time": "2010-01"},
            },
            None,
            "temperature",
        )
    ]


def test_a_list_drops_a_disagreeing_pair_spec_select_time_entry_with_a_warning(
    stubbed_fan,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=["temperature"],
            select={"test": {"time": "2010-01"}, "reference": {"time": "2010-02"}},
            times=["2011-05"],
        )
    assert any("select= time entry" in str(w.message) for w in caught)
    assert stubbed_fan == [
        (
            {
                "test": {"depth": "surface", "time": "2011-05"},
                "reference": {"depth": "surface", "time": "2011-05"},
            },
            None,
            "temperature",
        )
    ]


def test_variable_depth_and_time_all_varying_compose_the_label(stubbed_fan):
    comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature", "salinity"],
        depths=(0, 50),
        times=["2010-01", "2010-02"],
    )
    labels = [label for _, _, label in stubbed_fan]
    assert "temperature 0 m 2010-01" in labels
    assert "salinity 50 m 2010-02" in labels
    assert len(stubbed_fan) == 2 * 2 * 2  # variable x depth x time


def test_times_none_leaves_the_fan_and_its_labels_unchanged(stubbed_fan):
    """The regression guard: times=None must reproduce today's exact behaviour."""
    comparison.compare(
        reference=["obs"], test=["model"], variables=["temperature"], depths=(50,)
    )
    assert stubbed_fan == [({"depth": 50}, None, "temperature")]


# -- compare()'s times= fan: seasons form -----------------------------------------


def test_seasons_fan_one_comparison_per_default_season_in_calendar_order(stubbed_fan):
    comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        times={"groupby": "season", "reduce": "mean"},
    )
    assert [label for _, _, label in stubbed_fan] == ["DJF", "MAM", "JJA", "SON"]
    for season, (sel, agg, _label) in zip(
        ["DJF", "MAM", "JJA", "SON"], stubbed_fan, strict=True
    ):
        assert sel == {"depth": "surface", "season": season}
        assert agg == {
            "time": {"groupby": "season", "reduce": "mean", "seasons": [season]}
        }


def test_custom_seasons_fan_in_given_order(stubbed_fan):
    comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        times={"groupby": "season", "seasons": ["JFMA", "MJJA", "SOND"], "reduce": "mean"},
    )
    assert [label for _, _, label in stubbed_fan] == ["JFMA", "MJJA", "SOND"]


def test_a_single_season_earns_no_label(stubbed_fan):
    comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        times={"groupby": "season", "seasons": ["JJA"], "reduce": "mean"},
    )
    assert [label for _, _, label in stubbed_fan] == ["temperature"]


def test_seasons_fan_writes_both_sides_of_a_pair_spec_select(stubbed_fan):
    comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        select={"test": {"depth": 50}, "reference": {}},
        times={"groupby": "season", "seasons": ["JJA"], "reduce": "mean"},
    )
    sel, _agg, _label = stubbed_fan[0]
    # depths=/select= sugar (unrelated to times=) already writes the depth it
    # derives into both sides, same as with no times= involved at all.
    assert sel == {
        "test": {"depth": 50, "season": "JJA"},
        "reference": {"depth": 50, "season": "JJA"},
    }


def test_a_select_season_entry_is_replaced_with_a_warning(stubbed_fan):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=["temperature"],
            select={"season": "DJF"},
            times={"groupby": "season", "seasons": ["JJA"], "reduce": "mean"},
        )
    assert any("select= 'season' entry" in str(w.message) for w in caught)
    sel, _agg, _label = stubbed_fan[0]
    assert sel == {"depth": "surface", "season": "JJA"}


def test_seasons_fan_and_aggregate_time_entry_conflict():
    with pytest.raises(ValueError, match="got both times="):
        comparison.compare(
            reference="dummy_ref",
            test="dummy_test",
            variables=["temperature"],
            times={"groupby": "season", "reduce": "mean"},
            aggregate={"time": "mean"},
        )


def test_seasons_fan_and_over_time_conflict():
    with pytest.raises(ValueError, match="got both times="):
        comparison.compare(
            reference="dummy_ref",
            test="dummy_test",
            variables=["temperature"],
            times={"groupby": "season", "reduce": "mean"},
            over="time",
        )


def test_a_dataless_season_is_skipped_not_fatal():
    """One season's aggregate raises KeyError (operators' all-missing case); the
    others still form, matching a missing-variable/missing-bin skip."""
    from ocean_skill.comparison import select_for

    formed = []
    declared = {"model": {"variables": []}, "obs": {"variables": []}}

    def fake_align(self, refresh=False):
        if select_for(self.select, "test").get("season") == "DJF":
            raise KeyError("no data")
        formed.append(select_for(self.select, "test").get("season"))

    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(comparison.Comparison, "align", fake_align),
    ):
        out = comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=["temperature"],
            times={"groupby": "season", "reduce": "mean"},
        )
    assert formed == ["MAM", "JJA", "SON"]
    assert len(out) == 3


def test_a_dataless_season_reraises_with_skip_missing_false():
    declared = {"model": {"variables": []}, "obs": {"variables": []}}

    def fake_align(self, refresh=False):
        raise KeyError("no data")

    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(comparison.Comparison, "align", fake_align),
    ):
        with pytest.raises(KeyError):
            comparison.compare(
                reference=["obs"],
                test=["model"],
                variables=["temperature"],
                times={"groupby": "season", "reduce": "mean"},
                skip_missing=False,
            )


# -- compare()'s times= fan: dict (bins) form -------------------------------------


@pytest.fixture
def stubbed_bins():
    """Serve two fixed monthly bins in place of a real time axis, and record the fan."""
    formed = []
    declared = {"model": {"variables": []}, "obs": {"variables": []}}
    bins = [("2010-01-01", "2010-01-31"), ("2010-02-01", "2010-02-28")]
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(comparison, "_time_bins", lambda source, freq, window: bins),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(
                (self.select, self.aggregate, self.label)
            ),
        ),
    ):
        yield formed


def test_a_dict_fans_one_comparison_per_derived_bin(stubbed_bins):
    comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        times={"resample": "1MS", "reduce": "mean"},
    )
    # The bin lives in the select; the aggregate is a plain reduction that
    # collapses it (no 'resample' -- that would keep a size-1 time axis).
    assert stubbed_bins == [
        (
            {"depth": "surface", "time": "2010-01"},
            {"time": {"reduce": "mean"}},
            "2010-01",
        ),
        (
            {"depth": "surface", "time": "2010-02"},
            {"time": {"reduce": "mean"}},
            "2010-02",
        ),
    ]


def test_a_dict_merges_reduction_kwargs_into_the_aggregate(stubbed_bins):
    comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        times={"resample": "1MS", "reduce": "quantile", "q": 0.9},
    )
    _, aggregate, _ = stubbed_bins[0]
    assert aggregate == {"time": {"reduce": "quantile", "q": 0.9}}


def test_a_select_time_entry_becomes_the_window_not_a_conflict(monkeypatch):
    """The dict form's select= entry narrows which bins are derived -- no warning."""
    seen_windows = []
    declared = {"model": {"variables": []}, "obs": {"variables": []}}

    def fake_bins(source, freq, window):
        seen_windows.append(window)
        return [("2010-02-01", "2010-02-28")]

    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(comparison, "_time_bins", fake_bins),
        mock.patch.object(
            comparison.Comparison, "align", lambda self, refresh=False: None
        ),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # the canonical standard_name, not "temperature" -- resolving an
            # alias itself warns, which would otherwise trip this test's own
            # "no warnings at all" check for an unrelated reason
            comparison.compare(
                reference=["obs"],
                test=["model"],
                variables=[TEMPERATURE],
                select={"time": "2010-02"},
                times={"resample": "1MS", "reduce": "mean"},
            )
    assert seen_windows == ["2010-02"]


def test_bins_are_resolved_once_per_test_source_across_depths(monkeypatch):
    """The whole reason for the per-test-source memo: repeat opens cost nothing."""
    calls = []
    declared = {"model": {"variables": []}, "obs": {"variables": []}}

    def fake_bins(source, freq, window):
        calls.append(source)
        return [("2010-01-01", "2010-01-31")]

    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(comparison, "_time_bins", fake_bins),
        mock.patch.object(
            comparison.Comparison, "align", lambda self, refresh=False: None
        ),
    ):
        comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=["temperature"],
            depths=(0, 50, 100),
            times={"resample": "1MS", "reduce": "mean"},
        )
    assert calls == ["model"]  # once, not once per depth


# -- conflicts and validation: raised before any catalog lookup ------------------


def test_times_dict_and_aggregate_time_entry_conflict():
    with pytest.raises(ValueError, match="got both times="):
        comparison.compare(
            reference="dummy_ref",
            test="dummy_test",
            variables=["temperature"],
            times={"resample": "1MS", "reduce": "mean"},
            aggregate={"time": "mean"},
        )


def test_times_and_over_time_conflict():
    with pytest.raises(ValueError, match="got both times="):
        comparison.compare(
            reference="dummy_ref",
            test="dummy_test",
            variables=["temperature"],
            times=["2010-01"],
            over="time",
        )


def test_times_pair_spec_select_disagreement_is_refused():
    with pytest.raises(ValueError, match="disagrees"):
        comparison.compare(
            reference="dummy_ref",
            test="dummy_test",
            variables=["temperature"],
            select={"test": {"time": "2010-01"}, "reference": {"time": "2010-02"}},
            times={"resample": "1MS", "reduce": "mean"},
        )


def test_times_none_leaves_a_disagreeing_pair_spec_select_untouched():
    """The bug this guards: the WOA pattern must still work without times=."""
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
            select={"test": {"time": "2010-01"}, "reference": {"time": "2010-02"}},
            aggregate={"time": "mean"},
        )
    assert formed == [
        {
            "test": {"time": "2010-01", "depth": "surface"},
            "reference": {"time": "2010-02", "depth": "surface"},
        }
    ]


# -- skip_missing: a bin-resolution failure is skippable, like a missing variable -


def test_skip_missing_catches_a_bin_resolution_failure():
    declared = {"model": {"variables": []}, "obs": {"variables": []}}

    def boom(source, freq, window):
        raise ValueError(f"{source!r} has no time axis")

    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(comparison, "_time_bins", boom),
    ):
        out = comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=["temperature"],
            times={"resample": "1MS", "reduce": "mean"},
        )
    assert len(out) == 0


def test_skip_missing_false_reraises_a_bin_resolution_failure():
    declared = {"model": {"variables": []}, "obs": {"variables": []}}

    def boom(source, freq, window):
        raise ValueError(f"{source!r} has no time axis")

    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(comparison, "_time_bins", boom),
    ):
        with pytest.raises(ValueError, match="no time axis"):
            comparison.compare(
                reference=["obs"],
                test=["model"],
                variables=["temperature"],
                times={"resample": "1MS", "reduce": "mean"},
                skip_missing=False,
            )


def test_an_unresolvable_source_name_is_never_skippable():
    """Unlike a missing variable or a bin-resolution failure, a source name that
    resolves nowhere can never contribute a comparison no matter how the fan-out
    proceeds -- so it is checked once, up front, and raised even under the
    default ``skip_missing=True`` rather than silently dropping every pair and
    surfacing later as an empty, unexplained ComparisonSet.
    """
    declared = {"model": mock.Mock(metadata={"variables": []})}

    def resolve(name):
        try:
            return declared[name]
        except KeyError:
            raise KeyError(f"Unknown source {name!r}.") from None

    with mock.patch("ocean_skill.catalog.resolve", resolve):
        with pytest.raises(KeyError, match="woa23_nitrate_typo"):
            comparison.compare(
                reference=["woa23_nitrate_typo"],
                test=["model"],
                variables=["nitrate"],
                aggregate={"time": "mean"},
            )


# -- metrics/__repr__/pooling: a fanned time entry survives past the fan ---------


def _field(value):
    lat = np.linspace(18, 30, 6)
    lon = np.linspace(-97, -82, 8)
    return xr.DataArray(
        np.full((len(lat), len(lon)), float(value)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "degC"},
    )


@pytest.fixture
def stub_lane(monkeypatch):
    """Serve a fixed 2-D field for any source, like test_skill_maps.py's own stub."""
    monkeypatch.setattr(
        comparison, "prepare_source", lambda source, *a, **k: (_field(10.0), None)
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)


def test_metrics_reports_the_selected_time(stub_lane):
    c = comparison.Comparison(
        reference="obs", test="model", variable=TEMPERATURE, select={"time": "2010-01"}
    )
    assert c.metrics()["time"] == "2010-01"


def test_metrics_reports_none_when_no_time_was_selected(stub_lane):
    c = comparison.Comparison(reference="obs", test="model", variable=TEMPERATURE)
    assert c.metrics()["time"] is None


def test_repr_names_the_selected_time():
    c = comparison.Comparison(
        reference="obs", test="model", variable=TEMPERATURE, select={"time": "2010-01"}
    )
    assert "@ 2010-01" in repr(c)


def test_repr_omits_time_when_none_was_selected():
    c = comparison.Comparison(reference="obs", test="model", variable=TEMPERATURE)
    assert "@ surface" in repr(c)
    assert "2010" not in repr(c)


def test_metrics_reports_the_season(stub_lane):
    c = comparison.Comparison(
        reference="obs", test="model", variable=TEMPERATURE, select={"season": "JJA"}
    )
    assert c.metrics()["time"] == "JJA"


def test_repr_names_the_season():
    c = comparison.Comparison(
        reference="obs", test="model", variable=TEMPERATURE, select={"season": "JJA"}
    )
    assert "@ JJA" in repr(c)


def test_pooling_names_the_month_once_it_is_the_only_thing_that_varies(stub_lane):
    jan = comparison.Comparison(
        reference="obs", test="model", variable=TEMPERATURE, select={"time": "2010-01"}
    )
    feb = comparison.Comparison(
        reference="obs", test="model", variable=TEMPERATURE, select={"time": "2010-02"}
    )
    pooled = comparison.ComparisonSet([jan]) + comparison.ComparisonSet([feb])
    assert pooled.labels == ["2010-01", "2010-02"]


def test_pooling_ignores_time_when_it_never_varies(stub_lane):
    """The regression guard.

    Adding a "time" label dim must not perturb an existing pool where nothing
    carries a time entry at all.
    """
    a = comparison.Comparison(reference="obs", test="dev", variable=TEMPERATURE)
    b = comparison.Comparison(
        reference="obs", test="dev_marblsub", variable=TEMPERATURE
    )
    pooled = comparison.ComparisonSet([a]) + comparison.ComparisonSet([b])
    assert pooled.labels == ["dev", "dev_marblsub"]
