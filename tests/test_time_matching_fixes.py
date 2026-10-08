"""Time matching that warned (or paired) wrongly on real observation records.

Four findings from the Iceland ROMS-vs-SEANOE comparison, each a false alarm or a
misattributed sample rather than a crash: a half-hourly mooring whose decoded stamps
jitter by milliseconds binned unevenly into hourly bins; short-bin warnings for sparse
cruises (empty months are sampling, not a part-period mean); bottle samples at 13:00
paired with the *next* day's model average; and a units warning that said ``None``
without saying whose.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import align as A
from ocean_skill.operators import _warn_short_bins

LAT = np.linspace(18, 26, 3)
LON = np.linspace(-98, -90, 3)


def _field(time, seed: int = 0, *, attrs=None):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(5.0, 1.0, (len(time), len(LAT), len(LON))),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": LAT, "lon": LON},
        attrs={"units": "mmol m-3"} if attrs is None else attrs,
    )


def _series(time, attrs=None):
    return xr.DataArray(
        np.arange(len(time), dtype=float),
        dims="time",
        coords={"time": time},
        attrs=attrs or {},
    )


# --- 1. jitter-robust binning ---------------------------------------------------------


def test_stamp_jitter_does_not_make_bin_counts_uneven():
    """A 30-min record with +/-29 ms noise lands 2 samples in every interior bin."""
    rng = np.random.default_rng(0)
    base = pd.date_range("2024-01-01", periods=48 * 40, freq="30min")
    jitter = rng.choice([-0.029, 0.0, 0.029], size=base.size)
    jittered = (base + pd.to_timedelta(jitter, unit="s")).values
    # the model: hourly, centre-anchored, so the bin edges are the hh:30 stamps
    model = _field(pd.date_range("2024-01-01T01:00", periods=24 * 38, freq="h"))
    obs = _series(jittered, {"units": "mmol m-3", "cell_methods": "time: point"})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, report = A.match_axis(
            obs, model, over="time", method="mean"
        )  # mean forced: obs (test) is the finer lane
    assert report["steps_per_bin"] == 2
    assert report["bins_short"] == 0


def test_jitter_moves_no_stamp_across_an_edge():
    """Per-sample: each jittered stamp bins exactly where its clean twin does."""
    base = pd.date_range("2024-01-01", periods=48 * 5, freq="30min")
    noisy = base + pd.to_timedelta(
        np.tile([-0.029, 0.029, 0.0], base.size // 3 + 1)[: base.size], unit="s"
    )
    model = _field(pd.date_range("2024-01-01T01:00", periods=24 * 4, freq="h"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {}
        for name, stamps in (("clean", base), ("noisy", noisy)):
            obs = _series(stamps.values, {"units": "mmol m-3"})
            obs = obs.copy(data=np.arange(obs.size, dtype=float))
            out[name] = A.match_axis(obs, model, over="time", method="mean")[0]
    np.testing.assert_array_equal(out["clean"].values, out["noisy"].values)


# --- 2. short bins: edges only, time coverage, regular data only ----------------------


def _hourly_coord(start, periods):
    return _series(pd.date_range(start, periods=periods, freq="h").values)["time"]


def test_resample_warns_for_a_month_the_record_starts_in_the_middle_of():
    coord = _hourly_coord("2024-04-10", 24 * 120)  # to 2024-08-07: full May/Jun/Jul
    with pytest.warns(UserWarning, match="fewer samples") as record:
        _warn_short_bins(coord, "MS", "time")
    message = str(record[0].message)
    assert "2024-04-01" in message
    assert "2024-05-01" not in message and "2024-06-01" not in message
    # August stops on the 7th, so it is the other partial edge
    assert "2024-08-01" in message and "leaves 2 of 5 bins" in message


def test_a_record_ending_on_a_month_boundary_does_not_warn_for_its_last_month():
    coord = _hourly_coord("2024-04-10", 24 * 82)  # ends 2024-06-30 23:00
    with pytest.warns(UserWarning, match="fewer samples") as record:
        _warn_short_bins(coord, "MS", "time")
    message = str(record[0].message)
    assert "leaves 1 of 3 bins" in message and "2024-04-01" in message


def test_whole_months_of_hourly_data_do_not_warn():
    coord = _hourly_coord("2024-01-01", 24 * 91)  # Jan-Mar, ends 2024-03-31 23:00
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_short_bins(coord, "MS", "time")


def test_a_modest_start_offset_is_not_a_part_period_mean():
    """Starting on the 4th covers 90% of April: above the 80% bar."""
    coord = _hourly_coord("2024-04-04", 24 * 88)  # ends 2024-06-30 23:00
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_short_bins(coord, "MS", "time")


def test_sparse_cruises_with_empty_months_do_not_warn():
    stamps = pd.to_datetime(
        [
            "2024-01-12",
            "2024-01-13",
            "2024-02-10",
            "2024-03-14",
            "2024-03-15",
            "2024-04-09",
            "2024-06-11",
            "2024-06-12",
            "2024-07-10",
            "2024-08-13",
            "2024-09-12",
            "2024-10-09",
            "2025-01-14",
        ]
    ).values
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_short_bins(_series(stamps)["time"], "MS", "time")


def test_daily_data_into_weekly_bins_is_left_alone():
    """Under ten samples a bin is sampling, however ragged the first one is."""
    coord = _series(pd.date_range("2024-01-03", periods=60, freq="D").values)["time"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_short_bins(coord, "7D", "time")


def test_align_warns_for_the_one_bin_a_record_starts_in_the_middle_of():
    model = _field(pd.date_range("2024-01-01", periods=24 * 3, freq="h"))
    # 5-minute obs from 06:50 -- the 06:30-07:30 bin holds only its last 40 minutes
    obs = _series(
        pd.date_range("2024-01-01T06:50", "2024-01-03T12:25", freq="5min").values,
        {"units": "mmol m-3", "cell_methods": "time: point"},
    )
    with pytest.warns(UserWarning, match="caught fewer than") as record:
        A.match_axis(obs, model, over="time", method="mean")
    shorts = [r for r in record if "caught fewer than" in str(r.message)]
    assert len(shorts) == 1
    assert str(shorts[0].message).startswith("1 of the reference's bins")
    assert "usually the first and last" not in str(shorts[0].message)


def test_align_does_not_warn_for_a_record_that_covers_its_bins():
    model = _field(pd.date_range("2024-01-01", periods=24 * 3, freq="h"))
    obs = _series(
        pd.date_range("2023-12-31T23:30", "2024-01-03T23:25", freq="5min").values,
        {"units": "mmol m-3"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        A.match_axis(obs, model, over="time", method="mean")


def test_align_does_not_warn_for_sparse_data_in_the_frame_bins():
    """Four samples a day into daily bins: ragged counts are sampling, not coverage."""
    model = _field(pd.date_range("2024-01-01", periods=10, freq="D"))
    stamps = pd.date_range("2024-01-01T03:00", periods=36, freq="6h").values[1:]
    obs = _series(stamps, {"units": "mmol m-3"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", message=".*caught fewer.*")
        A.match_axis(obs, model, over="time", method="mean", bin_anchor="start")


# --- 3. instants against a period average: containment --------------------------------


def _daily_mean(days=40, start="2024-05-08", cell_methods="time: mean"):
    attrs = {"units": "mmol m-3"}
    if cell_methods:
        attrs["cell_methods"] = cell_methods
    return _field(pd.date_range(start, periods=days, freq="D"), attrs=attrs)


#: two samples on the same day, then a cruise every ten days -- coarse against a daily
#: model, which is what puts the reference on the "sampled" side of the choice
BOTTLES = [
    "2024-05-10T10:30",
    "2024-05-10T13:00",
    "2024-05-20T11:00",
    "2024-05-30T09:00",
]


def _bottles(stamps=BOTTLES):
    return _series(
        pd.to_datetime(stamps).values,
        {"units": "mmol m-3", "cell_methods": "time: point"},
    )


def test_point_samples_take_the_day_that_contains_them():
    model = _daily_mean()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        test, reference, report = A.match_axis(
            model, _bottles(), over="time", min_overlap=1
        )
    assert report["match_method"] == "contained"
    assert "contains" in report["match_reason"]
    assert test.sizes["time"] == reference.sizes["time"] == 4
    # 10:30 and 13:00 on the 10th both take the 10th, not the 11th
    want = model.sel(time="2024-05-10").squeeze(drop=True)
    for i in range(2):
        np.testing.assert_array_equal(test.isel(time=i).values, want.values)
    np.testing.assert_array_equal(test["time"].values, reference["time"].values)


def test_a_sample_outside_every_day_is_dropped_and_said():
    model = _daily_mean(days=30)  # ends 2024-06-06
    bottles = _bottles([*BOTTLES, "2024-07-20T12:00"])
    with pytest.warns(UserWarning, match="outside every test bin"):
        test, _, report = A.match_axis(model, bottles, over="time", min_overlap=1)
    assert test.sizes["time"] == 4 and report["steps_unmatched"] == 1


def test_a_test_that_does_not_say_what_it_is_keeps_the_nearest_pairing():
    model = _daily_mean(cell_methods=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, report = A.match_axis(model, _bottles(), over="time", min_overlap=1)
    assert report["match_method"] == "nearest"


def test_resolve_match_method_routes_a_composite_test_to_containment():
    day = np.arange(40) * 86400.0
    cruises = np.array([10.5, 13.0, 250.0, 490.0]) * 3600
    method, _, _, target = A.resolve_match_method(
        day, cruises, composite=False, test_composite=True
    )
    assert (method, target) == ("contained", "reference")
    method, *_ = A.resolve_match_method(
        day, cruises, composite=False, test_composite=None
    )
    assert method == "nearest"


# --- 4. the units message names the side ---------------------------------------------


def _units_pair(t_units, r_units):
    test, ref = _series(np.arange(3)), _series(np.arange(3))
    for da, u in ((test, t_units), (ref, r_units)):
        if u is not None:
            da.attrs["units"] = u
    return test, ref


@pytest.mark.parametrize(
    ("t_units", "r_units", "fragment"),
    [
        (
            "Celsius",
            None,
            "the reference has no units attribute (the test is 'Celsius')",
        ),
        (None, "PSU", "the test has no units attribute (the reference is 'PSU')"),
        (None, None, "neither the test nor the reference has a units attribute"),
        ("$$bad", "mmol/m^3", "the test's units '$$bad' are not recognised"),
        ("mmol/m^3", "$$bad", "the reference's units '$$bad' are not recognised"),
    ],
)
def test_the_units_warning_says_which_side_is_the_problem(t_units, r_units, fragment):
    test, ref = _units_pair(t_units, r_units)
    with pytest.warns(UserWarning, match="cannot verify units") as record:
        A._check_units(test, ref)
    message = str(record[0].message)
    assert message.startswith("cannot verify units")
    assert fragment in message
    assert "None" not in message.replace("(None", "")


# --- 5. millisecond shifts are noise, not a stamping convention ----------------------


def _hourly_point_vs_half_hourly(offset_s):
    test = _field(
        pd.date_range("2024-01-01", periods=24 * 5, freq="h"),
        attrs={"units": "mmol m-3", "cell_methods": "time: point"},
    )
    stamps = pd.date_range("2024-01-01", periods=48 * 5, freq="30min")
    stamps = stamps + pd.to_timedelta(offset_s, unit="s")
    obs = _series(stamps.values, {"units": "mmol m-3", "cell_methods": "time: point"})
    return test, obs


def _nearest_warnings(record):
    return [r for r in record if "by nearest match" in str(r.message)]


def test_millisecond_nearest_shifts_are_recorded_but_not_warned():
    jitter = np.tile([-0.029, 0.029, 0.0], 80)[: 48 * 5]
    test, obs = _hourly_point_vs_half_hourly(jitter)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, _, report = A.match_axis(test, obs, over="time")
    assert report["match_method"] == "nearest"
    assert not _nearest_warnings(record)
    assert 0 < report["offset_max"] < 1


def test_a_ten_minute_nearest_shift_still_warns():
    test, obs = _hourly_point_vs_half_hourly(600.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, _, report = A.match_axis(test, obs, over="time")
    assert len(_nearest_warnings(record)) == 1
    assert report["offset_max"] == pytest.approx(600.0)
