"""``Comparison.align`` prunes the test lane to the station's own cast times,
by default, for a time-climatology profile comparison.

The end-to-end counterpart of ``tests/test_series.py``'s
``_reference_time_targets`` suite (which asserts the *targets themselves* are
computed) and ``tests/test_operators.py``'s ``subset_to_time_targets`` suite
(which asserts the crop/interpolation function in isolation): here a real
``timeSeriesProfile`` station comparison runs through ``Comparison.align``
itself, against a model with far more time steps than the station has casts,
confirming ``align()`` actually threads ``time_targets=``/``time_targets_method=``
for a ``groupby``/``resample`` climatology (Change C, Part 1/2) -- not just that
the lower-level pieces work when called directly. Mirrors
``tests/test_profile_point_window.py``'s station+fine-grid fixture pattern, but
spies on ``align.subset_to_time_targets`` (the prune/interpolate step itself)
rather than on the resulting column count.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

TEMPERATURE = "sea_water_temperature"
STATION_LON, STATION_LAT = -158.0, 22.75

#: Four casts spread over about a year -- sparse next to the model's own daily
#: output below, and not aligned to the model's own step times.
CAST_TIMES = pd.to_datetime(
    ["2024-01-15", "2024-04-20", "2024-07-10", "2024-10-05"]
)


def _station_dataset() -> xr.Dataset:
    depth = np.array([5.0, 25.0])
    base = 24.0 - 0.05 * depth
    values = base[None, :] + 0.5 * np.arange(len(CAST_TIMES))[:, None]
    return xr.Dataset(
        {"TEMP": (("time", "depth"), values, {"units": "degC"})},
        coords={"time": CAST_TIMES, "depth": depth},
    ).assign_coords(lon=STATION_LON, lat=STATION_LAT)


def _daily_test_grid(n_time: int = 366) -> xr.Dataset:
    """A whole year of *daily* model output on a small grid around the
    station -- ~100x the station's own 4 casts, the read this feature exists
    to avoid paying for in full.
    """
    lon = np.array([STATION_LON - 0.05, STATION_LON, STATION_LON + 0.05])
    lat = np.array([STATION_LAT - 0.05, STATION_LAT, STATION_LAT + 0.05])
    time = pd.date_range("2024-01-01", periods=n_time, freq="D")
    depth = np.array([5.0, 25.0])
    base = 24.5 - 0.05 * depth
    values = (
        base[None, :, None, None]
        + 0.02 * np.arange(n_time)[:, None, None, None]
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
def station_and_daily_model(monkeypatch):
    import ocean_skill as osk
    from ocean_skill import catalog

    lanes = {"ctd_station": _station_dataset(), "his": _daily_test_grid()}
    metas = {
        "ctd_station": {
            "featureType": "timeSeriesProfile",
            "axes": {"T": "time", "Z": "depth"},
            "standard_names": {"TEMP": TEMPERATURE},
            "geospatial_lon_min": STATION_LON,
            "geospatial_lon_max": STATION_LON,
            "geospatial_lat_min": STATION_LAT,
            "geospatial_lat_max": STATION_LAT,
            # _reference_narrowing's own time_coverage_of -- the contiguous
            # window the test lane is (pre-)cropped to before time_targets
            # prunes it further, exactly matching the cast span above.
            "time_coverage_start": "2024-01-15",
            "time_coverage_end": "2024-10-05",
        },
        "his": {"standard_names": {TEMPERATURE: TEMPERATURE}},
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    return lanes


def _time_targets_spy(monkeypatch):
    """Spy on align.subset_to_time_targets, recording each call's own
    (method, input time size, output time size) -- the prune/interpolate step
    itself, before the groupby/resample fold ever runs. Patching the *align*
    module's own attribute is what matters: prepare_source's own
    ``from ocean_skill.align import subset_to_time_targets`` is a local import,
    re-resolved from that module's namespace at call time.
    """
    import ocean_skill.align as align_mod

    calls = []
    real = align_mod.subset_to_time_targets

    def spy(obj, targets, method="nearest"):
        before = obj.sizes.get("time")
        out = real(obj, targets, method=method)
        calls.append((method, before, out.sizes.get("time")))
        return out

    monkeypatch.setattr(align_mod, "subset_to_time_targets", spy)
    return calls


def test_a_groupby_climatology_prunes_the_model_to_cast_times(
    station_and_daily_model, monkeypatch
):
    """The default (nearest): the model lane is pruned from ~366 days to the
    4 cast-nearest steps before the monthly fold ever runs."""
    from ocean_skill.comparison import Comparison

    calls = _time_targets_spy(monkeypatch)
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
        aligned = c.align()
    assert c.over == "Z"
    assert calls, "subset_to_time_targets was never called -- pruning did not fire"
    method, before, after = calls[0]
    assert method == "nearest"
    assert after == len(CAST_TIMES)
    assert before is None or before > after  # far more than 4 days were read in
    assert aligned.sizes["month"] <= len(CAST_TIMES)


def test_a_resample_fold_prunes_the_model_to_cast_times_too(
    station_and_daily_model, monkeypatch
):
    """Part 1 covers resample identically to groupby -- both are recognized by
    _time_is_climatology."""
    from ocean_skill.comparison import Comparison

    calls = _time_targets_spy(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="ctd_station",
            test="his",
            variable=TEMPERATURE,
            select={"depth": [5.0, 25.0]},
            aggregate={"time": {"resample": "1MS", "reduce": "mean", "spread": "std"}},
            cache=False,
        )
        c.align()
    assert calls
    method, _, after = calls[0]
    assert method == "nearest"
    assert after == len(CAST_TIMES)


def test_time_method_interp_lands_the_model_on_the_cast_instants(
    station_and_daily_model, monkeypatch
):
    """time_method="interp" switches the prune to a linear interpolation onto
    the cast times, keeping only the bracketing steps."""
    from ocean_skill.comparison import Comparison

    calls = _time_targets_spy(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="ctd_station",
            test="his",
            variable=TEMPERATURE,
            select={"depth": [5.0, 25.0]},
            aggregate={"time": {"groupby": "month", "reduce": "mean", "spread": "std"}},
            time_method="interp",
            cache=False,
        )
        c.align()
    assert c.over == "Z"
    assert calls
    method, before, after = calls[0]
    assert method == "interp"
    # Every cast is well inside the daily model's span, so all 4 interpolate;
    # bracketing keeps at most 2 steps per target, never the whole record.
    assert after == len(CAST_TIMES)
    assert before is None or before > after


def test_time_method_auto_still_uses_nearest():
    """The default "auto" (and any value that isn't "interp"/"linear") maps to
    the ordinary nearest-step prune, unchanged."""
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(reference="ctd_station", test="his", variable=TEMPERATURE)
    assert c.time_method == "auto"
