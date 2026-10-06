"""A resample keeps only the bins that held samples.

A sparse series (a CTD station visited a few times) binned daily would otherwise come
back with a bin for every day between the first and last visit, nearly all NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ocean_skill.operators import SPREAD_COORD, TIME_RESAMPLE_ATTR, aggregate

SPEC = {"time": {"resample": "1D", "reduce": "mean"}}


def _series(times, values):
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims="time",
        coords={"time": pd.DatetimeIndex(times)},
        name="temp",
        attrs={"units": "degC"},
    )


def _sparse():
    times = ["2020-04-01 06:00", "2020-04-01 18:00", "2020-05-10", "2020-07-01"]
    return _series(times, [1.0, 3.0, 5.0, 7.0])


def test_sparse_series_keeps_only_occupied_bins():
    out = aggregate(_sparse(), SPEC)
    assert out.sizes["time"] == 3
    assert list(out["time"].values) == list(
        pd.DatetimeIndex(["2020-04-01", "2020-05-10", "2020-07-01"])
    )
    np.testing.assert_allclose(out.values, [2.0, 5.0, 7.0])


def test_dense_series_matches_plain_resample():
    t = pd.date_range("2020-01-01", periods=24 * 4, freq="h")
    da = _series(t, np.arange(len(t)))
    out = aggregate(da, SPEC)
    ref = da.resample(time="1D").mean()
    assert out.sizes["time"] == ref.sizes["time"] == 4
    np.testing.assert_allclose(out.values, ref.values)


def test_spread_stays_aligned_with_values():
    out = aggregate(_sparse(), {"time": {**SPEC["time"], "spread": "std"}})
    assert out[SPREAD_COORD].sizes["time"] == out.sizes["time"] == 3
    assert not np.isnan(out.values).any()
    np.testing.assert_allclose(out[SPREAD_COORD].values, [1.0, 0.0, 0.0])


def test_resample_marker_survives():
    out = aggregate(_sparse(), SPEC)
    assert out["time"].attrs[TIME_RESAMPLE_ATTR] == "1D"


def test_bin_with_only_nan_samples_is_kept():
    times = ["2020-04-01", "2020-04-05", "2020-04-20"]
    out = aggregate(_series(times, [1.0, np.nan, 3.0]), SPEC)
    assert out.sizes["time"] == 3
    assert np.isnan(out.values[1])
