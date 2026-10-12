"""A gap in the model's record is not bridged: time matching refuses, it does not guess.

A real hindcast store can be missing days (corrupted daily files), and a CTD cast taken
inside one used to be answered anyway -- ``time_method="interp"`` interpolated between
steps either side of the hole, and the default took whichever step was nearest, up to a
day away. Two rules now say no:

* interpolation (:func:`ocean_skill.align._time_target_steps`, shared by
  ``subset_to_time_targets`` and ``preselect_time_targets``) drops a target whose
  bracketing steps are more than :data:`~ocean_skill.align.TIME_GAP_FACTOR` of the
  lane's usual step apart, with a warning of its own;
* a single-instant profile (:func:`ocean_skill.align._sample_test_at_instant`) refuses a
  nearest step farther than :data:`~ocean_skill.align.NEAREST_TOLERANCE_FRACTION` of
  the lane's usual step -- the reach ``over="time"`` matching already has -- as a
  ``NoValidData``, which ``compare(skip_missing=True)`` skips pair by pair.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import align, catalog, comparison
from ocean_skill.align import (
    NoValidData,
    _sample_test_at_instant,
    preselect_time_targets,
    subset_to_time_targets,
)
from ocean_skill.comparison import Comparison, compare
from tests._tidal_roms import tidal_roms

T0 = np.datetime64("2024-07-01T00:00", "ns")
HOUR = np.timedelta64(1, "h")


def _at(*hours: float) -> np.ndarray:
    """Return datetime64 stamps ``hours`` after :data:`T0`."""
    return np.array([T0 + np.timedelta64(int(h * 60), "m") for h in hours])


def _gappy(hours=(0, 1, 2, 3, 8, 9, 10, 11)) -> xr.Dataset:
    """Return an hourly lane missing hours 4-7, valued by its own hour."""
    return xr.Dataset(
        {"x": ("time", np.asarray(hours, float))}, coords={"time": _at(*hours)}
    )


def _interp(ds, targets):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = subset_to_time_targets(ds, targets, method="interp")
    return out, [str(w.message) for w in caught]


# -- interpolation: a target in a gap is dropped ---------------------------------


def test_interp_drops_a_target_in_a_gap_with_a_warning():
    out, said = _interp(_gappy(), _at(1.5, 5.5))
    assert out["x"].values.tolist() == [1.5]  # the in-gap 5.5 h was not bridged
    (gap,) = [m for m in said if "gap" in m]
    assert "1 target time" in gap and "2024-07-01 05:30" in gap


def test_the_gap_warning_counts_every_target_and_names_the_first_few():
    out, said = _interp(_gappy(), _at(1.5, 4.5, 5.5, 6.5, 7.5, 9.5))
    assert out["x"].values.tolist() == [1.5, 9.5]
    (gap,) = [m for m in said if "gap" in m]
    assert "4 target time" in gap and "..." in gap


def test_a_target_exactly_on_a_step_beside_a_gap_is_kept():
    out, said = _interp(_gappy(), _at(3, 8))
    assert out["x"].values.tolist() == [3.0, 8.0]
    assert not said


def test_a_target_between_ordinary_steps_is_interpolated_without_a_warning():
    out, said = _interp(_gappy(), _at(0.25, 10.5))
    np.testing.assert_allclose(out["x"].values, [0.25, 10.5])
    assert not said


def test_a_hole_is_judged_against_the_lanes_median_step():
    # steps of 1, 1, 1 then 5 h: the median is 1 h, so a bracket may span 1.5 h at most
    ds = _gappy((0, 1, 2, 3, 8))
    out, said = _interp(ds, _at(2.5, 6.0))
    assert out["x"].values.tolist() == [2.5]
    assert any("gap" in m for m in said)


def test_a_lane_stamped_by_the_calendar_month_has_no_gaps():
    stamps = pd.date_range("2000-01-16", periods=13, freq="MS") + pd.Timedelta(days=15)
    stamps = stamps.values.astype("datetime64[ns]")
    ds = xr.Dataset({"x": ("time", np.arange(13.0))}, coords={"time": stamps})
    targets = np.array(["2000-02-29", "2000-08-01"], "datetime64[ns]")
    out, said = _interp(ds, targets)
    assert out.sizes["time"] == 2 and not said


def test_every_target_in_a_gap_leaves_the_lane_alone_and_says_so():
    ds = _gappy()
    out, said = _interp(ds, _at(5.5))
    assert out.sizes["time"] == ds.sizes["time"]
    assert any("gap" in m for m in said)


def test_preselect_and_subset_drop_the_same_targets_and_warn_once():
    ds, targets = _gappy(), _at(1.5, 5.5, 9.25)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        narrowed, left = preselect_time_targets(ds, targets, method="interp")
        via = subset_to_time_targets(narrowed, left, method="interp")
        direct = subset_to_time_targets(ds, targets, method="interp")
    assert left.tolist() == _at(1.5, 9.25).tolist()
    xr.testing.assert_allclose(via["x"], direct["x"], atol=0, rtol=0)
    # once from the preselection, once more only from the direct call on the whole lane
    assert sum("gap" in str(w.message) for w in caught) == 2


def test_nearest_pruning_never_drops_a_gap_target():
    """The prune is a superset by design; alignment's own tolerance decides after."""
    out = subset_to_time_targets(_gappy(), _at(5.0), method="nearest")
    assert out["x"].values.tolist() == [3.0]


# -- a single-instant profile: the nearest step must be near -----------------------


def _profile(when: str) -> xr.Dataset:
    return xr.Dataset(
        {"obs": ("depth", [1.0, 2.0])},
        coords={
            "depth": [3.0, 5.0],
            "time": np.datetime64(when, "ns"),
            "lon": 0.0,
            "lat": 0.0,
        },
    )


def test_a_cast_in_a_gap_is_refused_naming_the_cast_and_the_step():
    with pytest.raises(NoValidData) as err:
        _sample_test_at_instant(_gappy(), _profile("2024-07-01T05:00"), over="Z")
    message = str(err.value)
    assert "2024-07-01T05:00" in message  # the cast
    assert "2024-07-01T03:00" in message  # the nearest step
    assert "gap" in message


def test_a_cast_close_to_a_step_still_takes_it_and_records_the_offset():
    out = _sample_test_at_instant(_gappy(), _profile("2024-07-01T01:20"), over="Z")
    assert float(out["x"]) == 1.0
    assert out.attrs["time_to_reference_hours"] == pytest.approx(1 / 3)


def test_the_reach_is_half_the_lanes_step_inclusive():
    out = _sample_test_at_instant(_gappy(), _profile("2024-07-01T11:30"), over="Z")
    assert float(out["x"]) == 11.0
    with pytest.raises(NoValidData):
        _sample_test_at_instant(_gappy(), _profile("2024-07-01T11:31"), over="Z")


def test_a_cast_past_the_end_of_the_record_is_refused_too():
    with pytest.raises(NoValidData):
        _sample_test_at_instant(_gappy(), _profile("2024-07-01T14:00"), over="Z")


def test_a_tolerance_in_seconds_widens_the_reach():
    cast = _profile("2024-07-01T05:00")  # 2 h after step 3, 3 h before step 8
    out = _sample_test_at_instant(_gappy(), cast, over="Z", tolerance=2.5 * 3600)
    assert float(out["x"]) == 3.0


def _monthly(*months: int) -> xr.Dataset:
    """Return a lane stamped the 16th of each given month of 2000, valued by month."""
    stamps = [np.datetime64(f"2000-{m:02d}-16", "ns") for m in months]
    x = np.asarray(months, float)
    return xr.Dataset({"x": ("time", x)}, coords={"time": stamps})


def test_a_cast_mid_interval_of_a_monthly_lane_is_matched_not_refused():
    # Jul 16 to Aug 16 is 31 days against a median step of 30.5: the midpoint is 15.5
    # days from both neighbours, further than half the median, yet there is no gap
    lane = _monthly(6, 7, 8, 9, 10, 11, 12)
    out = _sample_test_at_instant(lane, _profile("2000-07-31T12:00"), over="Z")
    assert float(out["x"]) in (7.0, 8.0)
    assert out.attrs["time_to_reference_hours"] == pytest.approx(15.5 * 24)


def test_a_monthly_lane_missing_months_is_still_a_gap():
    lane = _monthly(6, 7, 10, 11, 12)  # August and September are gone
    with pytest.raises(NoValidData, match="gap"):
        _sample_test_at_instant(lane, _profile("2000-09-01T00:00"), over="Z")


def test_a_cast_before_the_start_of_the_record_says_so():
    with pytest.raises(NoValidData, match="before the first step"):
        _sample_test_at_instant(_gappy(), _profile("2024-06-30T20:00"), over="Z")


def test_a_lane_with_one_step_has_no_cadence_to_judge_by():
    one = _gappy((0,))
    out = _sample_test_at_instant(one, _profile("2024-07-05T00:00"), over="Z")
    assert float(out["x"]) == 0.0


# -- through a Comparison -----------------------------------------------------------

POINT = (200.01, 50.01)  # the middle cell of the tidal fixture's 3 x 3 grid
HEIGHT_SPEC = {
    "test": "height",
    "reference": "obs",
    "standard_name": "sea_water_temperature",
}
KEPT = [0, 1, 2, 3, 8, 9, 10, 11]  # the model record: hours 4-7 are missing


def _cast(time, depths=(3.0, 5.0)):
    return pd.DataFrame(
        {
            "time": pd.Timestamp(time),
            "lon": POINT[0],
            "lat": POINT[1],
            "depth (m)": list(depths),
            "obs (m)": [0.0] * len(depths),
        }
    )


def _install(monkeypatch, casts):
    ds, meta = tidal_roms(zeta=3.0 * np.array([1, 0, -1, 0] * 3))
    ds = ds.isel(time=KEPT)
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))
    sources = {"his": (ds, meta)}
    for name, when in casts.items():
        sources[name] = (_cast(when), {"featureType": "profile"})

    def read(name, **kw):
        return sources[name][0]

    monkeypatch.setattr(osk, "read", read)
    monkeypatch.setattr("ocean_skill.sources.read", read)
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)


def _comparison(reference="cast", cache=False, **kw):
    kwargs = dict(
        select={"depth": [3.0, 5.0]}, over="Z", depth_method="interp", cache=cache
    )
    kwargs.update(kw)
    return Comparison(reference=reference, test="his", variable=HEIGHT_SPEC, **kwargs)


@pytest.mark.parametrize("time_method", ["interp", "auto"])
def test_a_cast_in_a_gap_is_refused_by_the_comparison(monkeypatch, time_method):
    _install(monkeypatch, {"cast": "2024-07-01 05:30"})
    c = _comparison(time_method=time_method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NoValidData):
            c.align()


@pytest.mark.parametrize("time_method", ["interp", "auto"])
def test_a_cast_beside_the_gap_is_still_compared(monkeypatch, time_method):
    _install(monkeypatch, {"cast": "2024-07-01 01:20"})
    c = _comparison(time_method=time_method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isfinite(np.asarray(c.aligned["test"])).all()


def test_compare_skips_the_cast_in_the_gap_and_keeps_the_other(monkeypatch, capsys):
    _install(monkeypatch, {"inside": "2024-07-01 01:20", "gap": "2024-07-01 05:30"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compare(
            reference=["inside", "gap"],
            test="his",
            variables=[HEIGHT_SPEC],
            depth_method="interp",
            cache=False,
            skip_missing=True,
        )
    assert [c.reference_name for c in out] == ["inside"]
    printed = capsys.readouterr().out
    assert "1 skipped" in printed and "falls in a gap" in printed
    # a gap is the expected kind of skip, not an unexpected error
    assert not [
        w for w in caught if "failed with an error other than" in str(w.message)
    ]


def test_a_refused_cast_is_not_written_to_the_cache(monkeypatch):
    from ocean_skill import cache

    _install(monkeypatch, {"cast": "2024-07-01 05:30"})
    c = _comparison(cache=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NoValidData):
            c.align()
    assert cache.load(c._cache_key) is None


# -- the cache: interp is re-keyed by the gap rule, nearest is not ----------------


def test_the_gap_rule_is_part_of_an_interp_comparisons_keys(monkeypatch):
    _install(monkeypatch, {"cast": "2024-07-01 01:20"})
    before = _comparison(time_method="interp")._cache_key
    nearest = _comparison(time_method="auto")._cache_key
    monkeypatch.setattr(align, "TIME_GAP_FACTOR", align.TIME_GAP_FACTOR * 2)
    assert _comparison(time_method="interp")._cache_key != before
    assert _comparison(time_method="auto")._cache_key == nearest


def test_the_prepared_lane_key_carries_the_gap_rule_for_interp_only(monkeypatch):
    from ocean_skill import cache

    def lane_tags(method):
        seen = []
        real = cache.key_for_prepared

        def spy(**kw):
            seen.append(dict(kw["select"]))
            return real(**kw)

        monkeypatch.setattr(cache, "key_for_prepared", spy)
        try:
            comparison.prepare_source(
                "his",
                "height",
                {"depth": 3.0},
                None,
                time_targets=_at(1.5),
                time_targets_method=method,
                use_cache=False,
            )
        finally:
            monkeypatch.setattr(cache, "key_for_prepared", real)
        return seen[-1].get("_time_targets_method")

    _install(monkeypatch, {})
    assert lane_tags("nearest") is None
    tag = lane_tags("interp")
    assert tag and str(align.TIME_GAP_FACTOR) in tag
    monkeypatch.setattr(align, "TIME_GAP_FACTOR", 3.0)
    assert lane_tags("interp") != tag


# -- interp that cannot apply says so, once --------------------------------------


PAIR_SPEC = {"test": {"depth": [3.0, 5.0]}, "reference": {"depth": [3.0, 5.0]}}


def _interp_warnings(caught):
    return [w for w in caught if "cannot interpolate" in str(w.message)]


def test_a_pair_spec_select_says_interp_became_nearest_once(monkeypatch):
    _install(monkeypatch, {"cast": "2024-07-01 00:20"})
    c = _comparison(time_method="interp", select=PAIR_SPEC)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c._cache_key  # asks the question once; align() asks it again
        c.align()
    (said,) = _interp_warnings(caught)
    assert "pair-spec" in str(said.message) and "nearest" in str(said.message)
    # and the nearest step is what was used: 00:00 has zeta = +3
    np.testing.assert_allclose(np.asarray(c.aligned["test"]).reshape(-1), [0.0, -2.0])


@pytest.mark.parametrize("select", [None, "pair"])
def test_nothing_is_said_when_interp_applies_or_was_not_asked_for(monkeypatch, select):
    _install(monkeypatch, {"cast": "2024-07-01 00:20"})
    kw = {"time_method": "interp"} if select is None else {"select": PAIR_SPEC}
    c = _comparison(**kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c.align()
    assert not _interp_warnings(caught)


def test_a_section_of_casts_interpolates_each_cast_and_does_not_warn(monkeypatch):
    _install(monkeypatch, {"a": "2024-07-01 00:20", "b": "2024-07-01 02:40"})
    c = _comparison(
        reference="a+b",
        select={"transect": {"from": "reference"}, "depth": [3.0, 5.0]},
        over=None,
        section_casts=["a", "b"],
        time_method="interp",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test = np.asarray(c.aligned["test"])
    assert not _interp_warnings(caught)
    # the surface is at +2 m at 00:20 and -1 m at 02:40, the layers 3 and 5 m below it
    np.testing.assert_allclose(np.sort(test, axis=None), [-6.0, -4.0, -3.0, -1.0])
