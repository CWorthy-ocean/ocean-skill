"""A single cast whose ship drifts ~1 km still has a position.

``tabular._scalar_position`` returns ``None`` once the lon/lat spread passes
``FIXED_POSITION_TOLERANCE`` (1e-4 degrees) -- right for telling a fixed station from a
trajectory, but a ``featureType: profile`` entry is *declared* to be one cast, and a
ship drifts ~0.001 degrees during one. The profile then carried no lon/lat, so
``align.point_of`` saw a field instead of a station and the comparison failed with an
opaque indexer error. A declared profile now takes the median position for a spread
under ~0.01 degrees (about 1 km) and warns once, naming the spread in metres; a wider
spread is still no position, and the plain time build is untouched.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import align, tabular

N = 6
PROFILE = {"featureType": "profile"}


def cast(lon_drift: float = 0.0, lat_drift: float = 0.0) -> pd.DataFrame:
    """One CTD cast on ``(-96.5, 27.5)`` whose position drifts over the downcast."""
    return pd.DataFrame(
        {
            "Time[UTC]": np.full(N, "2024-04-04T12:00:00Z"),
            "Latitude[degrees_north]": 27.5 + np.linspace(0.0, lat_drift, N),
            "Longitude[degrees_east]": -96.5 + np.linspace(0.0, lon_drift, N),
            "Depth[m]": np.linspace(0.0, 50.0, N),
            "Temperature[degC]": np.linspace(20.0, 15.0, N),
        }
    )


def test_a_drifting_cast_takes_the_median_position_and_warns_once_in_metres():
    frame = cast(lon_drift=0.001, lat_drift=0.0005)

    with pytest.warns(UserWarning, match=r"spread up to ~\d+ m") as record:
        ds = tabular.to_dataset(frame, PROFILE)

    assert len([w for w in record if "spread up to" in str(w.message)]) == 1
    assert float(ds["lon"]) == pytest.approx(frame["Longitude[degrees_east]"].median())
    assert float(ds["lat"]) == pytest.approx(frame["Latitude[degrees_north]"].median())
    assert align.point_of(ds) == (float(ds["lon"]), float(ds["lat"]))


def test_drift_along_one_axis_only_still_gives_a_whole_position():
    """Lat drifts, lon holds still: the fixed axis must not be returned alone."""
    with pytest.warns(UserWarning, match="spread up to"):
        ds = tabular.to_dataset(cast(lat_drift=0.001), PROFILE)

    assert float(ds["lon"]) == pytest.approx(-96.5)
    assert float(ds["lat"]) == pytest.approx(27.5005)


def test_a_cast_that_moves_past_a_kilometre_still_has_no_position():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no drift warning: this is not one station
        ds = tabular.to_dataset(cast(lon_drift=0.05, lat_drift=0.05), PROFILE)

    assert "lon" not in ds.coords and "lat" not in ds.coords
    assert align.point_of(ds) is None


def test_a_fixed_cast_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = tabular.to_dataset(cast(), PROFILE)

    assert float(ds["lon"]) == pytest.approx(-96.5)


def test_a_transect_casts_own_position_wins_without_a_drift_warning():
    meta = {**PROFILE, "cast": {"lon": -96.4, "lat": 27.6}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = tabular.to_dataset(cast(lon_drift=0.001), meta)

    assert (float(ds["lon"]), float(ds["lat"])) == (-96.4, 27.6)


def test_the_time_series_build_still_calls_a_moving_platform_a_trajectory():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-04-04", periods=N, freq="h").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "latitude": 27.5 + np.linspace(0.0, 0.001, N),
            "longitude": np.full(N, -96.5),
            "temperature (degC)": np.linspace(20.0, 15.0, N),
        }
    )

    with pytest.warns(UserWarning, match="rather than a fixed station"):
        ds = tabular.to_dataset(
            frame, {"featureType": "timeSeries", "nominal_depth_m": 5.0}
        )

    assert align.point_of(ds) is None


def test_a_drifting_cast_aligns_to_a_model_instead_of_failing():
    """The reported failure: the reference had no station position to sample at."""
    with pytest.warns(UserWarning, match="spread up to"):
        ref = tabular.to_dataset(cast(lon_drift=0.001, lat_drift=0.001), PROFILE)
    test = xr.DataArray(
        np.random.default_rng(0).random((4, 11, 11)) + 15.0,
        dims=("depth", "lat", "lon"),
        coords={
            "depth": [0.0, 10.0, 25.0, 50.0],
            "lat": np.linspace(27.0, 28.0, 11),
            "lon": np.linspace(-97.0, -96.0, 11),
        },
        name="Temperature",
        attrs={"units": "degC"},
    )

    out = align.align(test, ref["Temperature"], over="Z", method="nearest")

    assert "depth" in out.dims
