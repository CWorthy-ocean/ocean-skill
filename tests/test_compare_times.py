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
    _fanned_time_select,
    _has_time_entry,
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
    assert _merged_time_aggregate(None, entry) == {"time": entry}
    assert _merged_time_aggregate({"Z": "mean"}, entry) == {"Z": "mean", "time": entry}


def test_merged_time_aggregate_folds_into_both_sides_of_a_pair_spec():
    entry = {"resample": "1MS", "reduce": "mean"}
    pair = {"test": {"Z": "mean"}, "reference": None}
    assert _merged_time_aggregate(pair, entry) == {
        "test": {"Z": "mean", "time": entry},
        "reference": {"time": entry},
    }


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
    assert stubbed_bins == [
        (
            {"depth": "surface", "time": "2010-01"},
            {"time": {"resample": "1MS", "reduce": "mean"}},
            "2010-01",
        ),
        (
            {"depth": "surface", "time": "2010-02"},
            {"time": {"resample": "1MS", "reduce": "mean"}},
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
    assert aggregate == {
        "time": {"resample": "1MS", "reduce": "quantile", "q": 0.9}
    }


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
