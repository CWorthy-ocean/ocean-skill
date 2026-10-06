"""``average()`` warns when its members share too few timestamps to average anything."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.comparison import Comparison, ComparisonSet

TEMPERATURE = "sea_water_potential_temperature"
PHRASE = "hold data from only one"


def _aligned(times, values, *, lon: float) -> xr.Dataset:
    if times is None:
        dims, coords = ("depth",), {"depth": [0.0, 10.0, 20.0]}
    else:
        dims, coords = ("time",), {"time": times}
    ref = xr.DataArray(np.asarray(values, float), dims=dims, coords=coords)
    ref = ref.assign_coords(lon=lon, lat=20.0)
    test = ref + 0.5
    return xr.Dataset(
        {"test": test, "reference": ref, "difference": test - ref},
        attrs={"station_lon": lon, "station_lat": 20.0},
    )


def _pair(a: xr.Dataset, b: xr.Dataset) -> ComparisonSet:
    comps = []
    for name, ds in (("S1", a), ("S2", b)):
        c = Comparison(reference=name, test="his", variable=TEMPERATURE, cache=False)
        c._aligned = ds
        comps.append(c)
    return ComparisonSet(comps)


def _no_overlap_warning(cs: ComparisonSet) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cs.average(by="variable")


def test_minutes_apart_timestamps_warn():
    t1 = pd.date_range("2024-01-01 10:00", periods=4, freq="h")
    t2 = t1 + pd.Timedelta(minutes=7)
    cs = _pair(_aligned(t1, [1, 2, 3, 4], lon=-150.0), _aligned(t2, [5, 6, 7, 8], lon=-151.0))
    with pytest.warns(UserWarning, match=PHRASE) as rec:
        cs.average(by="variable")
    assert 'resample' in str(rec[0].message)
    assert rec[0].filename == __file__


def test_identical_timestamps_do_not_warn():
    t = pd.date_range("2024-01-01", periods=4, freq="D")
    _no_overlap_warning(
        _pair(_aligned(t, [1, 2, 3, 4], lon=-150.0), _aligned(t, [5, 6, 7, 8], lon=-151.0))
    )


def test_no_time_dimension_does_not_warn():
    _no_overlap_warning(
        _pair(_aligned(None, [1, 2, 3], lon=-150.0), _aligned(None, [4, 5, 6], lon=-151.0))
    )


def test_partial_overlap_above_threshold_does_not_warn():
    t1 = pd.date_range("2024-01-01", periods=4, freq="D")
    t2 = pd.date_range("2024-01-02", periods=4, freq="D")  # 3 of 5 shared
    _no_overlap_warning(
        _pair(_aligned(t1, [1, 2, 3, 4], lon=-150.0), _aligned(t2, [5, 6, 7, 8], lon=-151.0))
    )
