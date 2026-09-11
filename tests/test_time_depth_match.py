"""``align.match_axis(over=TIME_DEPTH_OVER)``: composing a time match with a vertical
match, for a bare ``timeSeriesProfile`` reference that keeps both axes standing.

The two-axis sibling of ``tests/test_axis_match.py`` (time alone) and
``tests/test_vertical_match.py`` (depth alone): the sentinel does not invent new 2-D
interpolation, it runs the two existing 1-D matchers in sequence (time first, depth
second) -- so these tests check the composition, not the individual matchers again.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import align as A
from ocean_skill.align import TIME_DEPTH_OVER


def _model_column(times, z, seed: int = 0):
    """A model-shaped lane: a time axis and a vertical axis, no horizontal dims."""
    rng = np.random.default_rng(seed)
    base = 20.0 - 0.1 * np.abs(z)
    values = (
        base[None, :]
        + 0.1 * np.arange(len(times))[:, None]
        + rng.normal(0, 0.01, (len(times), len(z)))
    )
    return xr.DataArray(
        values,
        dims=("time", "z"),
        coords={"time": times, "z": z},
        attrs={"units": "degC"},
    ).assign_coords(lon=-94.0, lat=25.0)


def _ragged_station(times, depth, sampled: dict[int, list[float]]):
    """A (time, depth) rectangle with a real value only where ``sampled`` names it.

    ``sampled`` maps a visit index to the list of depths actually sampled on that
    visit -- the shape ``tabular._timeseriesprofile_dataset`` builds for a real
    repeat-visit station: a NaN wherever a visit did not sample a given level.
    """
    depth = np.asarray(depth, dtype="float64")
    values = np.full((len(times), len(depth)), np.nan)
    for i, levels in sampled.items():
        for d in levels:
            j = int(np.argmin(np.abs(depth - d)))
            values[i, j] = 20.0 - 0.1 * d + 0.1 * i
    return xr.DataArray(
        values, dims=("time", "depth"), coords={"time": times, "depth": depth}
    ).assign_coords(lon=-94.01, lat=25.01)


TIMES = pd.date_range("2024-01-01", periods=4, freq="MS")
DEPTH = np.array([5.0, 20.0, 60.0])


def test_both_axes_land_on_the_stations_own_coordinates():
    z = -np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0])
    model_times = pd.date_range("2024-01-01", periods=8, freq="MS")
    test = _model_column(model_times, z)
    reference = _ragged_station(
        TIMES, DEPTH, {0: [5.0, 20.0], 1: [60.0], 2: [5.0], 3: [20.0, 60.0]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matched_test, matched_reference, report = A.match_axis(
            test, reference, over=TIME_DEPTH_OVER
        )
    assert matched_test.dims == matched_reference.dims == ("time", "depth")
    assert list(matched_test["time"].values) == list(TIMES.values)
    assert list(matched_test["depth"].values) == list(DEPTH)
    assert report["axis"] == ["time", "depth"]
    assert report["time_axis"] == "time"
    assert report["depth_axis"] == "depth"
    assert report["depth_match_method"] == "nearest"


def test_unsampled_time_depth_pairs_are_nan_not_dropped():
    """A ragged station's own holes survive onto the aligned pair as NaN -- they
    simply drop out of a metric computed with dim=None later, no special-casing.
    """
    z = -np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0])
    model_times = pd.date_range("2024-01-01", periods=8, freq="MS")
    test = _model_column(model_times, z)
    reference = _ragged_station(
        TIMES, DEPTH, {0: [5.0, 20.0], 1: [60.0], 2: [5.0], 3: [20.0, 60.0]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matched_test, matched_reference, _ = A.match_axis(
            test, reference, over=TIME_DEPTH_OVER
        )
    finite_ref = np.isfinite(matched_reference.values)
    # exactly the 6 (visit, depth) pairs named in `sampled` above are finite
    assert int(finite_ref.sum()) == 6
    # the model itself is finite everywhere it was asked to be sampled (no holes
    # of its own) -- only the reference carries the station's raggedness
    assert np.isfinite(matched_test.values[finite_ref]).all()


def test_nearest_vs_interp_give_different_depth_matches():
    """depth_method="nearest" (the default) snaps to the model's closest real
    level; "interp" linearly interpolates instead -- a real difference when the
    model's own levels do not coincide with the station's.
    """
    z = -np.array([0.0, 10.0, 30.0, 100.0])  # no level at 20 m
    model_times = pd.date_range("2024-01-01", periods=4, freq="MS")
    test = _model_column(model_times, z, seed=2)
    depth = np.array([20.0])
    reference = _ragged_station(TIMES[:1], depth, {0: [20.0]})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nearest_test, _, nearest_report = A.match_axis(
            test, reference, over=TIME_DEPTH_OVER, depth_method="nearest"
        )
        interp_test, _, interp_report = A.match_axis(
            test, reference, over=TIME_DEPTH_OVER, depth_method="interp"
        )

    assert nearest_report["depth_match_method"] == "nearest"
    assert interp_report["depth_match_method"] == "interp"
    nearest_value = float(nearest_test.isel(time=0, depth=0).values)
    interp_value = float(interp_test.isel(time=0, depth=0).values)
    # nearest snaps to the model's own 10 m or 30 m level; interp blends them --
    # the two are not the same number for a level the model does not carry itself
    assert nearest_value != pytest.approx(interp_value)
    expected_interp = 20.0 - 0.1 * 20.0  # linear in depth, by _model_column's own recipe
    assert interp_value == pytest.approx(expected_interp, abs=0.05)


def test_both_axes_match_when_the_references_depth_coordinate_is_named_differently():
    """A real ADCP mooring shape: the vertical *dimension* is ``DEPTH``, with no
    same-named coordinate -- the real metres live in a coordinate named lowercase
    ``depth`` riding on that dimension. This must still resolve, not be refused as
    "native s-coordinates" -- see ``ocean_skill.operators.vertical_coord_on``.
    """
    z = -np.array([0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0])
    model_times = pd.date_range("2024-01-01", periods=8, freq="MS")
    test = _model_column(model_times, z)
    depth = DEPTH
    values = np.stack(
        [20.0 - 0.1 * depth + 0.1 * i for i in range(len(TIMES))], axis=0
    )
    reference = (
        xr.DataArray(values, dims=("time", "DEPTH"), coords={"time": TIMES})
        .assign_coords(depth=("DEPTH", depth))
        .assign_coords(lon=-94.01, lat=25.01)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matched_test, matched_reference, report = A.match_axis(
            test, reference, over=TIME_DEPTH_OVER
        )
    assert matched_test.dims == matched_reference.dims == ("time", "DEPTH")
    assert list(matched_test["DEPTH"].values) == list(DEPTH)
    assert report["axis"] == ["time", "DEPTH"]


def test_a_lane_with_no_vertical_axis_after_the_time_match_is_refused():
    """A time_depth match needs a real vertical axis on both lanes -- a test lane
    with no depth/z coordinate at all fails clearly, the same way match_axis's
    ordinary single-axis branch does for a missing axis.
    """
    model_times = pd.date_range("2024-01-01", periods=4, freq="MS")
    test = xr.DataArray(
        np.arange(4, dtype="float64"), dims=("time",), coords={"time": model_times}
    ).assign_coords(lon=-94.0, lat=25.0)
    reference = _ragged_station(TIMES, DEPTH, {0: [5.0]})
    with (
        warnings.catch_warnings(),
        pytest.raises(ValueError, match="no vertical axis"),
    ):
        warnings.simplefilter("ignore")
        A.match_axis(test, reference, over=TIME_DEPTH_OVER)
