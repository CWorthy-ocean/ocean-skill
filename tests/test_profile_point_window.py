"""``Comparison.align`` shrinks the test lane's point window at a profile station.

The end-to-end counterpart of ``tests/test_comparison.py``'s
``test_point_window_cells_shrinks_the_read_and_the_cache_key`` (which drives
``prepare_source`` directly): here a real ``timeSeriesProfile``/``profile``
station comparison runs through ``Comparison.align`` itself, against a test grid
fine enough that the ordinary 5-cell and the shrunk 1-cell window keep visibly
different numbers of columns -- confirming ``align()`` actually threads
``point_window_cells=`` for this shape, not just that ``prepare_source`` honours
it when asked directly. See the standing "profile-at-a-station climatology"
feature this window shrink was built for.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.align import NEAREST_POINT_WINDOW_CELLS, POINT_WINDOW_CELLS

TEMPERATURE = "sea_water_temperature"
STATION_LON, STATION_LAT = -158.0, 22.75


def _station_dataset(n_time: int = 3) -> xr.Dataset:
    time = pd.date_range("2015-01-01", periods=n_time, freq="MS")
    depth = np.array([5.0, 25.0])
    base = 24.0 - 0.05 * depth
    values = base[None, :] + 0.1 * np.arange(n_time)[:, None]
    return xr.Dataset(
        {"TEMP": (("time", "depth"), values, {"units": "degC"})},
        coords={"time": time, "depth": depth},
    ).assign_coords(lon=STATION_LON, lat=STATION_LAT)


def _fine_test_grid(n_time: int = 3) -> xr.Dataset:
    """A 0.05-degree grid around the station -- fine enough that the ordinary
    (11x11) and shrunk (3x3) point windows keep a different number of columns.
    """
    lon = np.round(np.arange(STATION_LON - 1.0, STATION_LON + 1.0, 0.05), 4)
    lat = np.round(np.arange(STATION_LAT - 1.0, STATION_LAT + 1.0, 0.05), 4)
    time = pd.date_range("2015-01-01", periods=n_time, freq="MS")
    depth = np.array([5.0, 25.0])
    base = 24.5 - 0.05 * depth
    values = (
        base[None, :, None, None]
        + 0.1 * np.arange(n_time)[:, None, None, None]
        + np.zeros((1, 1, lat.size, lon.size))
    )
    da = xr.DataArray(
        values,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        name=TEMPERATURE,
        attrs={"units": "degC"},
    )
    return da.to_dataset()


@pytest.fixture
def station_and_fine_model(monkeypatch):
    import ocean_skill as osk
    from ocean_skill import catalog

    lanes = {"ctd_station": _station_dataset(), "his": _fine_test_grid()}
    metas = {
        "ctd_station": {
            "featureType": "timeSeriesProfile",
            "axes": {"T": "time", "Z": "depth"},
            "standard_names": {"TEMP": TEMPERATURE},
            # _reference_narrowing needs a declared position to derive the test
            # lane's crop bbox from -- a station's own geospatial_* extent is a
            # single point, so min == max on each axis.
            "geospatial_lon_min": STATION_LON,
            "geospatial_lon_max": STATION_LON,
            "geospatial_lat_min": STATION_LAT,
            "geospatial_lat_max": STATION_LAT,
        },
        "his": {"standard_names": {TEMPERATURE: TEMPERATURE}},
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    return lanes


def _sizes_seen(monkeypatch):
    """Spy on prepare_source to record the test lane's own (lon, lat) sizes."""
    import ocean_skill.comparison as comparison

    seen = []
    real_prepare_source = comparison.prepare_source

    def spy(source, *args, **kwargs):
        da, depth = real_prepare_source(source, *args, **kwargs)
        if source == "his" and da is not None:
            seen.append((da.sizes.get("lon"), da.sizes.get("lat")))
        return da, depth

    monkeypatch.setattr(comparison, "prepare_source", spy)
    return seen


def test_a_nearest_profile_comparison_reads_the_shrunk_window(
    station_and_fine_model, monkeypatch
):
    """The default method (conservative_normed -> nearest at a point) against a
    timeSeriesProfile station shrinks the test lane to the 3x3 window."""
    from ocean_skill.comparison import Comparison

    seen = _sizes_seen(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="ctd_station",
            test="his",
            variable=TEMPERATURE,
            select={"depth": [5.0, 25.0]},
            aggregate={"time": {"groupby": "month", "reduce": "mean", "spread": "std"}},
            cache=False,
        )
        c.align()
    assert c.over == "Z"
    expected = 2 * NEAREST_POINT_WINDOW_CELLS + 1
    assert seen and seen[0] == (expected, expected)


def test_a_bilinear_profile_comparison_keeps_the_ordinary_window(
    station_and_fine_model, monkeypatch
):
    """An explicit interpolating method needs its full stencil, not the
    nearest-only shrink -- the ordinary 11x11 window is kept."""
    from ocean_skill.comparison import Comparison

    seen = _sizes_seen(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="ctd_station",
            test="his",
            variable=TEMPERATURE,
            select={"depth": [5.0, 25.0]},
            aggregate={"time": {"groupby": "month", "reduce": "mean", "spread": "std"}},
            method="bilinear",
            cache=False,
        )
        c.align()
    expected = 2 * POINT_WINDOW_CELLS + 1
    assert seen and seen[0] == (expected, expected)


def test_a_mooring_style_reference_keeps_the_ordinary_window(
    station_and_fine_model, monkeypatch
):
    """The window shrink is scoped to a profile/timeSeriesProfile reference (see
    Comparison.align's own note) -- a timeSeries/point/station featureType, whose
    reference-lane bbox crop is not provably a no-op the same way, keeps the
    ordinary window even at the package default (nearest) method."""
    import ocean_skill.comparison as comparison
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(comparison, "_feature_type", lambda source: "timeSeries")
    seen = _sizes_seen(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="ctd_station",
            test="his",
            variable=TEMPERATURE,
            select={"depth": 5.0},
            cache=False,
        )
        c.align()
    expected = 2 * POINT_WINDOW_CELLS + 1
    assert seen and seen[0] == (expected, expected)
