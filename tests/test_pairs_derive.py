"""Deriving a comparison's variants from one saved base of matched pairs.

The bases are built by hand, mirroring ``tests/test_profile_comparison.py``: a pair
is just ``test``/``reference``/``difference``, so nothing here needs a catalog, a model,
or ``align()``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import pairs

LEVELS = np.array([2.0, 5.0, 10.0, 20.0, 50.0, 100.0])


def _dataset(test, reference, **attrs) -> xr.Dataset:
    test.attrs = reference.attrs = {"units": "degC"}
    ds = xr.Dataset(
        {
            "test": test,
            "reference": reference,
            "difference": test - reference,
        },
        attrs={
            "station_lon": -158.0,
            "station_lat": 22.75,
            "test_standard_name": "sea_water_temperature",
            "reference_standard_name": "sea_water_temperature",
            **attrs,
        },
    )
    ds["difference"].attrs = {"long_name": "test − reference", "units": "degC"}
    return ds


def profile_base() -> xr.Dataset:
    """Build one cast: six levels, the model 1.0 warmer, one hole in each lane."""
    reference = xr.DataArray(
        20.0 - 0.1 * LEVELS, dims=("DEPTH",), coords={"DEPTH": LEVELS}
    ).assign_coords(lon=-158.0, lat=22.75)
    test = reference + 1.0
    test[2] = np.nan  # the model is missing at 10 m
    reference[4] = np.nan  # the cast is missing at 50 m
    return _dataset(test, reference, scored_over="DEPTH", match_method="interp")


def series_base(hours: int = 24 * 35, step: str = "1h") -> xr.Dataset:
    """Build a mooring pair: no depth axis, a tide plus a slow signal in both lanes."""
    t = pd.date_range("2020-01-01", periods=hours, freq=step)
    hrs = np.arange(hours) * pd.Timedelta(step).total_seconds() / 3600.0
    tide = 2.0 * np.sin(2 * np.pi * hrs / 12.42)
    slow = 0.01 * hrs
    reference = xr.DataArray(15.0 + slow + tide, dims=("time",), coords={"time": t})
    test = xr.DataArray(
        16.0 + slow + 0.5 * tide + np.sin(hrs / 3.0), dims=("time",), coords={"time": t}
    )
    return _dataset(test, reference, scored_over="time")


def station_base() -> xr.Dataset:
    """Build repeat casts: ten visits (every 12 h) by four levels."""
    t = pd.date_range("2020-01-01", periods=10, freq="12h")
    z = np.array([5.0, 10.0, 20.0, 40.0])
    rng = np.random.default_rng(1)
    reference = xr.DataArray(
        10.0 + rng.standard_normal((10, 4)),
        dims=("time", "depth"),
        coords={"time": t, "depth": z},
    )
    test = reference + 0.5 + 0.1 * rng.standard_normal((10, 4))
    return _dataset(test, reference, scored_over=["time", "depth"])


def derive(base, feature_type, select=None, aggregate=None, detide=False):
    return pairs.derive(
        base,
        select=select or {},
        aggregate=aggregate or {},
        detide=detide,
        feature_type=feature_type,
    )


def _both(detide=True):
    return {"test": {"T": 33.0}, "reference": {"T": 33.0}} if detide else False


# -- time ---------------------------------------------------------------------------


def test_time_slice_subsets_the_pairs():
    out = derive(
        series_base(), "timeSeries", select={"time": slice("2020-01-05", "2020-01-06")}
    )
    assert out.sizes["time"] == 48
    assert out["time"].min() >= np.datetime64("2020-01-05")
    assert out.attrs["station_lon"] == -158.0


def test_time_select_accepts_a_partial_date_dict_and_instant():
    base = series_base()
    month = derive(base, "timeSeries", select={"time": "2020-01"})
    assert month.sizes["time"] == base.sizes["time"] - 24 * 4  # Jan 1-31 only
    window = derive(
        base, "timeSeries", select={"T": {"min": "2020-01-02", "max": "2020-01-03"}}
    )
    assert window.sizes["time"] == 48  # a partial date end takes its whole day
    instant = derive(base, "timeSeries", select={"time": "2020-01-02T06:10"})
    assert "time" not in instant.dims  # nearest step, scalar


def test_time_select_that_leaves_nothing_is_an_error():
    with pytest.raises((ValueError, KeyError)):
        derive(series_base(), "timeSeries", select={"time": "2025-01"})


# -- depth --------------------------------------------------------------------------


def test_depth_list_keeps_those_levels_in_the_requested_order():
    out = derive(profile_base(), "profile", select={"depth": [20.0, 5.0]})
    assert list(out["DEPTH"].values) == [20.0, 5.0]
    assert out["difference"].dropna("DEPTH").size == 2
    assert out.attrs["scored_over"] == "DEPTH"


def test_scalar_depth_drops_the_axis_and_labels_the_level():
    out = derive(profile_base(), "profile", select={"Z": 20})
    assert out["test"].ndim == 0
    assert out.attrs["actual_depth"] == 20.0
    assert "scored_over" not in out.attrs
    assert float(out["difference"]) == pytest.approx(1.0)


def test_depth_not_sampled_is_not_derivable():
    ok, why = pairs.derivable(
        {"depth": [5.0, 7.5]},
        {},
        feature_type="profile",
        obs_levels=LEVELS,
        detide=False,
    )
    assert not ok and "7.5" in why and "not sampled" in why
    with pytest.raises(ValueError, match="not sampled"):
        derive(profile_base(), "profile", select={"depth": 7.5})


@pytest.mark.parametrize("key", ["depth", "Z", "z", "vertical"])
def test_every_vertical_key_spelling_is_recognized(key):
    ok, _ = pairs.derivable(
        {key: 10.0}, {}, feature_type="profile", obs_levels=LEVELS, detide=False
    )
    assert ok


def test_a_base_with_levels_has_no_surface_to_derive():
    # a cast or a repeat-visit station has no surface measurement: there is nothing to
    # pair the model's top cell with, so "surface" is refused (the caller raises an
    # actionable error before it gets here), never answered with the shallowest level
    ok, why = derivable_profile({"depth": "surface"}, {})
    assert not ok and "no surface measurement" in why
    with pytest.raises(ValueError, match="no surface measurement"):
        derive(profile_base(), "profile", select={"depth": "surface"})
    with pytest.raises(ValueError, match="no surface measurement"):
        derive(station_base(), "timeSeriesProfile", select={"depth": ["surface"]})


def test_no_depth_named_keeps_every_level_of_a_cast():
    base = profile_base()
    out = derive(base, "profile")
    assert list(out["DEPTH"].values) == list(LEVELS)
    xr.testing.assert_equal(out["test"], base["test"])
    mean = derive(base, "profile", aggregate={"Z": "mean"})
    joint = base["test"].notnull() & base["reference"].notnull()
    assert float(mean["test"]) == pytest.approx(float(base["test"].where(joint).mean()))


def test_surface_of_a_station_at_the_surface_is_its_base_unchanged():
    # a surface buoy's base has no depth axis and *is* the surface comparison
    base = series_base()
    for select in ({"depth": "surface"}, {"depth": ["surface"]}, {}):
        out = derive(base, "timeSeries", select=select)
        xr.testing.assert_equal(out["test"], base["test"])
        xr.testing.assert_equal(out["reference"], base["reference"])
        xr.testing.assert_allclose(out["difference"], base["difference"])


def test_one_band_means_the_pairs_inside_it_with_a_joint_mask():
    base = profile_base()
    out = derive(
        base,
        "profile",
        select={"depth": {"min": 0, "max": 25}},
        aggregate={"Z": "mean"},
    )
    # levels 2, 5, 10, 20; the model is NaN at 10 m, so 10 m leaves both means
    kept = [0, 1, 3]
    assert float(out["test"]) == pytest.approx(np.mean(base["test"].values[kept]))
    assert float(out["reference"]) == pytest.approx(
        np.mean(base["reference"].values[kept])
    )
    assert float(out["difference"]) == pytest.approx(1.0)
    assert out.attrs["actual_depth"] == pytest.approx(np.mean([2.0, 5.0, 10.0, 20.0]))
    assert out.attrs["depth_band"] == [0.0, 25.0]
    assert "DEPTH" not in out["test"].dims and "scored_over" not in out.attrs


def test_joint_mask_changes_the_answer_a_naive_mean_would_give():
    base = profile_base()
    out = derive(
        base,
        "profile",
        select={"depth": {"min": 40, "max": 120}},
        aggregate={"depth": "mean"},
    )
    # the cast has no 50 m value: only the 100 m pair counts for either lane
    assert float(out["reference"]) == pytest.approx(10.0)
    assert float(out["test"]) == pytest.approx(11.0)
    naive = float(base["test"].isel(DEPTH=[4, 5]).mean())
    assert float(out["test"]) != pytest.approx(naive) or naive == pytest.approx(11.0)


def test_two_bands_keep_an_axis_of_midpoints_with_bounds():
    out = derive(
        profile_base(),
        "profile",
        select={"depth": [{"min": 0, "max": 6}, {"min": 15, "max": 30}]},
        aggregate={"Z": "mean"},
    )
    assert list(out["DEPTH"].values) == [3.0, 22.5]
    assert out["DEPTH_bounds"].values.tolist() == [[0.0, 6.0], [15.0, 30.0]]
    assert out["DEPTH"].attrs["bounds"] == "DEPTH_bounds"
    assert out["test"].dims == ("DEPTH",) and out.attrs["scored_over"] == "DEPTH"
    assert "actual_depth" not in out.attrs
    assert np.allclose(out["difference"].values, 1.0)


def test_a_band_between_levels_has_no_data_and_is_not_derivable():
    # a cast has nothing in a layer none of its levels falls in; the nearest level
    # would silently pool an out-of-range depth into the layer's average
    select, aggregate = {"depth": {"min": 30, "max": 35}}, {"Z": "mean"}
    ok, why = derivable_profile(select, aggregate)
    assert not ok
    assert "none of the observations' levels falls in the 30–35 m layer" in why
    with pytest.raises(ValueError, match="none of the observations' levels falls in"):
        derive(profile_base(), "profile", select=select, aggregate=aggregate)


def test_every_band_empty_is_not_derivable_either():
    ok, why = derivable_profile(
        {"depth": [{"min": 30, "max": 35}, {"min": 60, "max": 70}]}, {"Z": "mean"}
    )
    assert not ok and "the 30–35, 60–70 m layers" in why


def test_an_empty_band_among_others_is_nan_with_one_warning():
    select = {"depth": [{"min": 0, "max": 6}, {"min": 30, "max": 35}]}
    ok, _ = derivable_profile(select, {"Z": "mean"})
    assert ok
    base = profile_base()
    with pytest.warns(UserWarning, match="no data in the 30–35 m layer") as record:
        out = derive(base, "profile", select=select, aggregate={"Z": "mean"})
    assert len(record) == 1  # one warning, naming the empty band
    assert list(out["DEPTH"].values) == [3.0, 32.5]  # both bands stay on the axis
    for name in ("test", "reference", "difference"):
        assert np.isnan(out[name].values[1])
    # the other band is the plain mean of its levels (2 and 5 m), as before
    np.testing.assert_allclose(
        out["reference"].values[0], base["reference"].values[:2].mean()
    )
    np.testing.assert_allclose(out["difference"].values[0], 1.0)
    assert out["DEPTH_bounds"].values.tolist() == [[0.0, 6.0], [30.0, 35.0]]


def test_band_without_a_vertical_mean_is_not_derivable():
    ok, why = derivable_profile({"depth": {"min": 0, "max": 10}}, {})
    assert not ok and "vertical mean" in why


def test_depth_list_with_a_vertical_mean_collapses_the_levels():
    out = derive(
        profile_base(),
        "profile",
        select={"depth": [5.0, 20.0]},
        aggregate={"Z": "mean"},
    )
    assert out["test"].ndim == 0
    assert out.attrs["actual_depth"] == pytest.approx(12.5)


def test_band_on_repeat_casts_averages_depth_but_keeps_time():
    out = derive(
        station_base(),
        "timeSeriesProfile",
        select={"depth": {"min": 0, "max": 25}},
        aggregate={"Z": "mean"},
    )
    assert out["test"].dims == ("time",)
    assert out.attrs["scored_over"] == ["time"]


# -- time aggregates ----------------------------------------------------------------


def test_monthly_resample_gives_one_pair_per_month():
    base = series_base()
    out = derive(
        base,
        "timeSeries",
        aggregate={"time": {"resample": "1MS", "reduce": "mean"}},
        select={"depth": "surface"},
    )
    assert out.sizes["time"] == 2  # Jan and the first days of Feb
    jan = base["test"].sel(time="2020-01").mean()
    assert float(out["test"].isel(time=0)) == pytest.approx(float(jan))
    assert out.attrs["scored_over"] == "time"


SEASONAL = {"groupby": "season", "seasons": ["DJF"], "reduce": "mean"}


def test_a_season_select_picks_the_group_the_seasonal_aggregate_made():
    base = series_base()
    out = derive(
        base,
        "timeSeries",
        select={"depth": "surface", "season": "DJF"},
        aggregate={"time": SEASONAL},
    )
    assert "time" not in out["test"].dims
    assert out["test"].ndim == 0
    assert float(out["test"]) == pytest.approx(float(base["test"].mean()))
    assert float(out["reference"]) == pytest.approx(float(base["reference"].mean()))


def test_a_season_select_without_a_seasonal_aggregate_is_not_derivable():
    for aggregate in (
        {},
        {"time": "mean"},
        {"time": {"groupby": "month", "reduce": "mean"}},
        {"time": [SEASONAL, "max"]},
    ):
        ok, why = pairs.derivable(
            {"season": "DJF"},
            aggregate,
            feature_type="timeSeries",
            obs_levels=None,
            detide=False,
        )
        assert not ok
        assert "seasonal time aggregate" in why


def test_climatology_groupby_renames_the_axis():
    out = derive(
        series_base(),
        "timeSeries",
        select={"depth": "surface"},
        aggregate={"time": {"groupby": "month", "reduce": "mean"}},
    )
    assert "month" in out["test"].dims and "time" not in out["test"].dims
    assert "scored_over" not in out.attrs


def test_difference_is_recomputed_not_aggregated():
    base = series_base()
    out = derive(
        base, "timeSeries", select={"depth": "surface"}, aggregate={"time": "std"}
    )
    std_of_diff = float((base["test"] - base["reference"]).std())
    assert float(out["difference"]) == pytest.approx(
        float(base["test"].std() - base["reference"].std())
    )
    assert float(out["difference"]) != pytest.approx(std_of_diff)


def test_time_mean_masks_jointly_first():
    base = series_base()
    base["reference"][:100] = np.nan
    out = derive(
        base, "timeSeries", select={"depth": "surface"}, aggregate={"time": "mean"}
    )
    both = base["test"].isel(time=slice(100, None)).mean()
    assert float(out["test"]) == pytest.approx(float(both))
    assert float(out["difference"]) == pytest.approx(
        float(out["test"] - out["reference"])
    )


def test_a_chain_of_time_steps_is_accepted():
    out = derive(
        series_base(),
        "timeSeries",
        aggregate={"time": [{"groupby": "month", "reduce": "mean"}, "std"]},
    )
    assert out["test"].ndim == 0


def test_spread_rides_beside_test_and_reference():
    out = derive(
        series_base(),
        "timeSeries",
        aggregate={"time": {"resample": "1MS", "reduce": "mean", "spread": "std"}},
    )
    assert {"test_spread", "reference_spread"} <= set(out.data_vars)


def test_time_aggregate_on_repeat_casts_after_a_depth_band():
    out = derive(
        station_base(),
        "timeSeriesProfile",
        select={"depth": {"min": 0, "max": 15}},
        aggregate={"Z": "mean", "time": "mean"},
    )
    assert out["test"].ndim == 0
    assert "scored_over" not in out.attrs


# -- detide -------------------------------------------------------------------------


def test_detide_on_an_hourly_series_removes_the_tide_before_aggregating():
    pytest.importorskip("oceans")
    base = series_base()
    hrs = np.arange(base.sizes["time"])
    out = derive(base, "timeSeries", select={"depth": "surface"}, detide=_both())
    # the filter's NaN edges are part of the pairs, not an error
    assert out["reference"].isnull().any()
    inner = out["reference"].isel(time=slice(60, -60)).values
    assert np.abs(inner - (15.0 + 0.01 * hrs[60:-60])).max() < 0.2  # 2 degC tide gone
    # detide runs before the aggregate: the std is the detided series' own
    agg = derive(
        base,
        "timeSeries",
        select={"depth": "surface"},
        aggregate={"time": "std"},
        detide=_both(),
    )
    assert float(agg["reference"]) == pytest.approx(
        float(out["reference"].where(out["test"].notnull()).std()), rel=1e-6
    )


def test_detide_only_the_sides_that_ask():
    pytest.importorskip("oceans")
    base = series_base()
    out = derive(base, "timeSeries", detide={"test": {"T": 33.0}, "reference": None})
    xr.testing.assert_equal(out["reference"], base["reference"])
    assert out["test"].isnull().any()


def test_detide_on_a_coarse_series_warns_and_leaves_it_alone():
    pytest.importorskip("oceans")
    base = series_base(hours=60, step="1D")
    with pytest.warns(UserWarning, match="nothing was detided"):
        out = derive(base, "timeSeries", detide=_both())
    np.testing.assert_array_equal(out["test"].values, base["test"].values)
    assert not out["test"].isnull().any()


def test_detide_on_an_irregular_series_warns():
    pytest.importorskip("oceans")
    base = series_base().isel(time=np.r_[0:200, 210:400, 405:700])
    with pytest.warns(UserWarning, match="not regularly spaced"):
        derive(base, "timeSeries", detide=_both())


def test_detide_on_a_cast_warns_there_is_no_time_axis():
    with pytest.warns(UserWarning, match="no time dimension"):
        derive(profile_base(), "profile", select={"depth": 5.0}, detide=_both())


def test_detide_is_derivable_for_a_series():
    ok, _ = pairs.derivable(
        {}, {}, feature_type="timeSeries", obs_levels=None, detide=_both()
    )
    assert ok


# -- what is not derivable ----------------------------------------------------------


def derivable_profile(select, aggregate):
    return pairs.derivable(
        select, aggregate, feature_type="profile", obs_levels=LEVELS, detide=False
    )


@pytest.mark.parametrize(
    "select, aggregate, needle",
    [
        ({"depth": "column"}, {}, "column"),
        ({"lon": -158.0, "lat": 22.75}, {}, "horizontal"),
        ({"lon": {"min": 1, "max": 2}}, {}, "horizontal"),
        ({"sigma0": 25.0}, {}, "unrecognized"),
        ({"season": "JJA"}, {}, "one cast"),
        (
            {"season": "JJA"},
            {"time": {"groupby": "season", "seasons": ["JJA"], "reduce": "mean"}},
            "one cast",
        ),
        ({}, {"lon": "mean"}, "horizontal"),
        ({}, {"depth": "max"}, "plain mean"),
        ({"depth": "deep"}, {}, "not one the saved pairs answer"),
        ({"depth": [5.0, "surface"]}, {}, "not one the saved pairs answer"),
        ({"depth": {"min": 0, "max": 10, "step": 1}}, {"Z": "mean"}, "plain"),
        ({"time": "2020-01"}, {}, "one cast"),
        ({}, {"time": "mean"}, "one cast"),
        ({"test": {"depth": 5.0}, "reference": {"depth": 10.0}}, {}, "per-lane"),
        ({}, {"test": {"time": "mean"}, "reference": {}}, "per-lane"),
    ],
)
def test_conservative_refusals(select, aggregate, needle):
    ok, why = derivable_profile(select, aggregate)
    assert not ok
    assert needle in why


def test_an_unknown_feature_type_is_not_derivable():
    ok, _ = pairs.derivable(
        {}, {}, feature_type="trajectory", obs_levels=None, detide=False
    )
    assert not ok


def test_unknown_obs_levels_make_literal_depths_not_derivable():
    ok, _ = pairs.derivable(
        {"depth": 10.0},
        {},
        feature_type="timeSeriesProfile",
        obs_levels=None,
        detide=False,
    )
    assert not ok


def test_a_mooring_depth_is_derivable_only_at_its_own_level():
    args = {
        "feature_type": "timeSeries",
        "obs_levels": np.array([10.0]),
        "detide": False,
    }
    assert pairs.derivable({"depth": 10.0}, {}, **args)[0]
    assert not pairs.derivable({"depth": 25.0}, {}, **args)[0]
    # ...and a station with a level has no surface to answer "surface" with
    assert not pairs.derivable({"depth": "surface"}, {}, **args)[0]
    assert pairs.derivable(
        {"depth": "surface"}, {}, **{**args, "obs_levels": np.array([])}
    )[0]


def test_derive_refuses_what_derivable_refuses():
    with pytest.raises(ValueError, match="cannot derive"):
        derive(profile_base(), "profile", select={"depth": "column"})


def test_derive_leaves_the_base_untouched():
    base = profile_base()
    snapshot = base.copy(deep=True)
    derive(
        base,
        "profile",
        select={"depth": {"min": 0, "max": 25}},
        aggregate={"Z": "mean"},
    )
    xr.testing.assert_identical(base, snapshot)


def test_derive_keeps_the_pairs_attrs():
    out = derive(profile_base(), "profile", select={"depth": [5.0, 20.0]})
    assert out["test"].attrs["units"] == "degC"
    assert out.attrs["test_standard_name"] == "sea_water_temperature"
    assert out.attrs["match_method"] == "interp"
    assert out["difference"].attrs["long_name"] == "test − reference"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        derive(profile_base(), "profile", select={"depth": [5.0, 20.0]})
