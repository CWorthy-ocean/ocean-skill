"""A composite (daily-mean) station is scored against the model's mean over each day.

A fixed-position reference compared ``over="time"`` has the model lane pruned to the
step nearest each of its times before it is read (``Comparison.
_reference_time_targets``) -- lossless when alignment would only have picked those
steps anyway. It is *not* when the reference is a period average coarser than the
model: alignment then averages the model's steps into each of the reference's bins
(``align.resolve_match_method`` -> ``"mean"``), and a lane already cut down to one step
per day has no day left to average -- the "mean" is a single snapshot, and a station of
daily means is scored against whichever hourly instant fell nearest its stamp.

The model here is hourly with a steady trend plus a 12.42 h tide, so a day's mean and
the snapshot nearest its stamp are different numbers.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

TEMPERATURE = "sea_water_potential_temperature"
LON, LAT = -158.0, 22.75
DAYS = 8


def _signal(time) -> np.ndarray:
    """Return a trend plus a 12.42 h tide, at ``time`` (a DatetimeIndex)."""
    hours = np.asarray((time - pd.Timestamp("2024-01-01")) / pd.Timedelta(hours=1))
    return 10.0 + 0.05 * hours + 2.0 * np.sin(2 * np.pi * hours / 12.42)


def _model(hours: pd.DatetimeIndex) -> xr.Dataset:
    lon = np.array([LON - 0.05, LON, LON + 0.05])
    lat = np.array([LAT - 0.05, LAT, LAT + 0.05])
    values = _signal(hours)[:, None, None] + np.zeros((hours.size, lat.size, lon.size))
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": hours, "lat": lat, "lon": lon},
        name="TEMP",
        attrs={"units": "degC"},
    ).to_dataset()


def _station(stamps: pd.DatetimeIndex, values) -> xr.Dataset:
    return xr.Dataset(
        {"TEMP": ("time", np.asarray(values, float), {"units": "degC"})},
        coords={"time": stamps},
    ).assign_coords(lon=LON, lat=LAT)


def _station_metadata() -> dict:
    return {
        "featureType": "timeSeries",
        "axes": {"T": "time"},
        "standard_names": {"TEMP": TEMPERATURE},
        "depth_convention": {"support": "surface"},
        "geospatial_lon_min": LON,
        "geospatial_lon_max": LON,
        "geospatial_lat_min": LAT,
        "geospatial_lat_max": LAT,
    }


def _install(monkeypatch, station: xr.Dataset, station_meta: dict, model=None):
    import ocean_skill as osk
    from ocean_skill import catalog

    hours = pd.date_range("2024-01-01", periods=24 * DAYS, freq="h")
    lanes = {"stn": station, "his": model if model is not None else _model(hours)}
    metas = {
        "stn": {**_station_metadata(), **station_meta},
        "his": {
            "standard_names": {"TEMP": TEMPERATURE},
            "geospatial_lon_min": LON - 0.05,
            "geospatial_lon_max": LON + 0.05,
            "geospatial_lat_min": LAT - 0.05,
            "geospatial_lat_max": LAT + 0.05,
        },
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )


def _daily_mean_station(period="1D"):
    """Return a station of true daily means of the model, stamped at noon."""
    hours = pd.date_range("2024-01-01", periods=24 * DAYS, freq="h")
    daily = pd.Series(_signal(hours), index=hours).resample("1D").mean()
    stamps = daily.index + pd.Timedelta(hours=12)
    meta = {"period": period} if period else {}
    return _station(stamps, daily.values), meta, daily.values


def _compare(**kw):
    from ocean_skill.comparison import Comparison

    return Comparison(
        reference="stn",
        test="his",
        variable=TEMPERATURE,
        over="time",
        cache=False,
        **kw,
    )


def _aligned_test(c) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(c.aligned["test"]).reshape(-1)


def test_the_model_is_averaged_into_a_daily_mean_references_days(monkeypatch):
    station, meta, daily = _daily_mean_station()
    _install(monkeypatch, station, meta)
    got = _aligned_test(_compare())
    np.testing.assert_allclose(got, daily)


def test_the_daily_mean_is_not_the_snapshot_nearest_its_stamp(monkeypatch):
    """Guard on the fixture: the two answers differ, so the first test can tell."""
    station, _, daily = _daily_mean_station()
    noon = _signal(pd.DatetimeIndex(station["time"].values))
    assert np.abs(noon - daily).max() > 0.1


def test_cell_methods_alone_marks_the_reference_a_composite(monkeypatch):
    station, _, daily = _daily_mean_station(period=None)
    station["TEMP"].attrs["cell_methods"] = "time: mean"
    _install(monkeypatch, station, {})
    np.testing.assert_allclose(_aligned_test(_compare()), daily)


def test_an_instantaneous_reference_is_still_pruned_to_its_nearest_steps(monkeypatch):
    """The prune is the point for a snapshot station: nothing to average into."""
    station, meta, _ = _daily_mean_station(period="snapshot")
    _install(monkeypatch, station, meta)
    c = _compare()
    targets = c._reference_time_targets()
    assert targets is not None and len(targets) == DAYS
    noon = _signal(pd.DatetimeIndex(station["time"].values))
    np.testing.assert_allclose(_aligned_test(c), noon)


def test_pruning_changes_the_cache_key_when_it_changes(monkeypatch):
    station, meta, _ = _daily_mean_station()
    _install(monkeypatch, station, meta)
    composite = _compare()._cache_key
    snap_station, snap_meta, _ = _daily_mean_station(period="snapshot")
    _install(monkeypatch, snap_station, snap_meta)
    assert _compare()._cache_key != composite
