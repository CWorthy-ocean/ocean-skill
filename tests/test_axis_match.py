"""Matching two lanes along the axis a comparison scores over.

Nothing in this package matched two sources in time before, so this is the new mechanism
and every failure mode here is one that produces a *plausible number* rather than an
error: model hours averaged into the wrong day, two products paired off by position, an
overlap that silently comes out empty.

The rule under test is the one alignment already follows in space — test → reference,
the reference is the frame — with the method chosen by which lane is coarser: a finer
test is averaged into the reference's bins, a comparable one paired step for step.
None of these need xesmf: matching happens before the regrid, deliberately.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import align as A

LAT = np.linspace(18, 26, 6)
LON = np.linspace(-98, -90, 5)


def _field(time, seed: int = 0, *, lat=LAT, lon=LON, attrs=None, name="time"):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(5.0, 1.0, (len(time), len(lat), len(lon))),
        dims=(name, "lat", "lon"),
        coords={name: time, "lat": lat, "lon": lon},
        attrs={"units": "mmol m-3"} if attrs is None else attrs,
    )


def _hourly(hours: int = 72, **kw):
    return _field(pd.date_range("2012-01-01", periods=hours, freq="h"), **kw)


def _daily(days: int = 3, start="2012-01-01", seed: int = 1, **kw):
    return _field(pd.date_range(start, periods=days, freq="D"), seed, **kw)


# --- the binning ---------------------------------------------------------------------


def test_hourly_output_becomes_the_reference_daily_means():
    """The case the feature exists for: a model sampled far finer than the product."""
    hourly, daily = _hourly(72), _daily(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test, reference, report = A.match_axis(hourly, daily, over="time")

    assert report["match_method"] == "mean"
    assert test.sizes["time"] == reference.sizes["time"] == 3
    assert report["steps_per_bin"] == 24
    for i, day in enumerate(("2012-01-01", "2012-01-02", "2012-01-03")):
        assert np.allclose(test.isel(time=i), hourly.sel(time=day).mean("time")), day


def test_units_survive_the_binning():
    """align() checks units next, and a reduction drops attrs unless they go back."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test, _, _ = A.match_axis(_hourly(48), _daily(2), over="time")
    assert test.attrs["units"] == "mmol m-3"


@pytest.mark.parametrize(
    ("stamps", "expected"),
    [
        (pd.date_range("2012-01-01", periods=4, freq="D"), "start"),
        (pd.date_range("2012-01-01T12:00", periods=4, freq="D"), "center"),
        (pd.date_range("2012-01-01", periods=4, freq="8D"), "start"),
        (pd.date_range("2012-01-01", periods=4, freq="MS"), "start"),
        (
            pd.date_range("1982-01-15", periods=4, freq="MS") + pd.Timedelta(days=14),
            "center",
        ),
    ],
    ids=["daily-midnight", "daily-noon", "8-day", "month-start", "mid-month"],
)
def test_the_anchoring_is_read_off_the_stamps(stamps, expected):
    """A product that labels a bin with its first instant lands on a period boundary.

    Getting this wrong is invisible and halves the data in every bin: centre-anchored
    bins around a midnight-stamped daily composite average noon to noon.
    """
    assert A.infer_bin_anchor(np.asarray(stamps, dtype="datetime64[ns]")) == expected


def test_a_reference_stamped_mid_period_still_bins_its_own_period():
    """Noon-stamped daily means are centred on noon, so the bin is the calendar day."""
    reference = _field(pd.date_range("2012-01-01T12:00", periods=3, freq="D"), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, report = A.match_axis(_hourly(72), reference, over="time")
    assert report["bin_anchor"] == "center"
    assert report["steps_per_bin"] == 24


def test_an_empty_bin_is_dropped_and_counted():
    hourly = _hourly(72)
    gapped = hourly.drop_sel(time=hourly.time[24:48])  # a whole day missing
    daily = _daily(3)
    with pytest.warns(UserWarning, match="had no test data"):
        test, reference, report = A.match_axis(gapped, daily, over="time")
    assert report["bins_empty"] == 1
    assert test.sizes["time"] == reference.sizes["time"] == 2


def test_a_part_period_bin_says_so():
    """The first and last bin of a selection are the usual culprits."""
    hourly = _hourly(72).isel(time=slice(6, None))  # January 1 starts at 06:00
    with pytest.warns(UserWarning, match="part of a period"):
        _, _, report = A.match_axis(hourly, _daily(3), over="time")
    assert report["bins_short"] == 1


# --- the pairing ---------------------------------------------------------------------


def test_daily_stamps_offset_by_a_convention_are_paired_not_rebinned():
    """A ROMS daily average at 12:00 against an L3 composite stamped 00:00."""
    noon = _field(pd.date_range("2012-01-01T12:00", periods=5, freq="D"))
    midnight = _daily(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test, reference, report = A.match_axis(noon, midnight, over="time")
    assert report["match_method"] == "nearest"
    assert test.sizes["time"] == reference.sizes["time"] == 5
    assert report["offset_max"] == 12 * 3600
    # the reference's own stamps become the shared axis, as in space
    assert (test.time.values == midnight.time.values).all()


def test_a_step_with_nothing_near_it_is_dropped():
    reference = _field(pd.to_datetime(["2012-01-01", "2012-01-02", "2012-06-01"]), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, ref, report = A.match_axis(
            _daily(3), reference, over="time", method="nearest", tolerance=43200
        )
    assert report["steps_unmatched"] == 1
    assert ref.sizes["time"] == 2


# --- choosing the method -------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell_methods", "expected"),
    [("time: point", "nearest"), ("time: mean", "mean")],
)
def test_the_reference_says_whether_it_is_a_composite_or_an_instant(
    cell_methods, expected
):
    reference = _daily(3, attrs={"units": "mmol m-3", "cell_methods": cell_methods})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, report = A.match_axis(_hourly(72), reference, over="time")
    assert report["match_method"] == expected


def test_a_catalog_period_settles_it_when_the_variable_does_not():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, report = A.match_axis(
            _hourly(72), _daily(3), over="time", metadata={"period": "daily"}
        )
    assert report["match_method"] == "mean"

    _, _, snap = A.match_axis(
        _hourly(72), _daily(3), over="time", metadata={"period": "snapshot"}
    )
    assert snap["match_method"] == "nearest"


def test_an_undeclared_reference_assumes_a_composite_and_says_so():
    """Warn-and-proceed, as align() already does for units it cannot verify."""
    with pytest.warns(UserWarning, match="taken to be a composite") as record:
        _, _, report = A.match_axis(_hourly(72), _daily(3), over="time")
    assert report["match_method"] == "mean"
    message = str(record[0].message)
    assert "time_method='nearest'" in message, "the override has to be named"
    assert "period" in message and "cell_methods" in message, "and the permanent fix"


def test_a_reference_finer_than_the_test_is_refused():
    """Coarsening the reference would change the thing being scored against."""
    with pytest.raises(ValueError) as excinfo:
        A.match_axis(_daily(3), _hourly(72), over="time")
    message = str(excinfo.value)
    assert "1 hour" in message and "1 day" in message, "both cadences"
    assert "resample" in message, "the deliberate way to coarsen it"
    assert "swap the roles" in message, "or the other way out"


def test_an_explicit_method_is_not_second_guessed():
    _, _, report = A.match_axis(
        _hourly(72), _daily(3), over="time", method="nearest", tolerance=3600
    )
    assert report["match_method"] == "nearest"
    assert "as asked" in report["match_reason"]


# --- refusals ------------------------------------------------------------------------


def test_no_overlap_names_both_spans():
    with pytest.raises(ValueError) as excinfo:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A.match_axis(_hourly(72), _daily(3, start="2015-01-01"), over="time")
    message = str(excinfo.value)
    assert "2012-01-01" in message and "2015-01-01" in message


def test_a_bare_axis_with_no_coordinate_is_refused():
    """Pairing by position would line up step 0 with step 0 for no reason at all."""
    bare = _daily(3).drop_vars("time")
    with pytest.raises(ValueError, match="reference_date"):
        A.match_axis(_hourly(72), bare, over="time")


def test_a_model_calendar_cannot_be_matched_against_real_dates():
    cftime = pytest.importorskip("cftime")
    stamps = np.array(
        [cftime.Datetime360Day(2012, 1, day) for day in range(1, 4)], dtype=object
    )
    with pytest.raises(ValueError, match="convert_calendar"):
        A.match_axis(_field(stamps), _daily(3), over="time")


def test_a_lane_with_no_such_axis_says_to_leave_over_unset():
    flat = _daily(3).mean("time")
    with pytest.raises(ValueError, match="over="):
        A.match_axis(_hourly(72), flat, over="time")


def test_too_few_matched_steps_warns_but_computes():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, _, report = A.match_axis(_hourly(72), _daily(3), over="time")
    assert report["n_matched"] == 3
    assert any("noise" in str(w.message) for w in record)


# --- what matching must not touch ----------------------------------------------------


def test_matching_leaves_the_horizontal_grids_alone():
    """Two lanes on different grids is the whole reason a regrid follows.

    Without ``exclude=``, an exact inner join also joins lat/lon and the overlap comes
    back empty — the silent-empty-overlap failure by a different door.
    """
    times = pd.date_range("2012-01-01", periods=4, freq="D")
    fine = _field(times, lat=np.linspace(18, 26, 40), lon=np.linspace(-98, -90, 33))
    coarse = _field(times, 1, lat=np.linspace(19, 25, 7), lon=np.linspace(-97, -91, 6))
    test, reference, report = A.match_axis(fine, coarse, over="time", method="exact")
    assert report["n_matched"] == 4
    assert test.sizes["lat"] == 40 and reference.sizes["lat"] == 7


def test_the_axis_may_be_named_differently_on_each_side():
    """ROMS calls it ocean_time; the result takes the reference's name."""
    roms = _field(
        pd.date_range("2012-01-01T12:00", periods=4, freq="D"), name="ocean_time"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test, _, report = A.match_axis(roms, _daily(4), over="time")
    assert report["axis"] == "time"
    assert "time" in test.dims and "ocean_time" not in test.dims


def test_a_numeric_axis_matches_too():
    """`over=` is not time-specific: a depth axis pairs by value the same way."""
    depths = np.array([0.0, 10.0, 20.0, 50.0])
    test = _field(depths, name="depth")
    reference = _field(depths + 0.4, 1, name="depth")
    _, _, report = A.match_axis(
        test, reference, over="depth", method="nearest", tolerance=2.0
    )
    assert report["n_matched"] == 4
    assert report["offset_max"] == pytest.approx(0.4)
