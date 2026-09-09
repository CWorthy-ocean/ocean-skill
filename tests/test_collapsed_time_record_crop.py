"""``Comparison.align`` crops the *reference* to the *test* source's own
catalog-declared time record when a plain (non-climatology) time aggregate
collapses the time axis.

Companion to ``tests/test_profile_climatology_time_targets.py`` (which covers
the ``groupby``/``resample`` climatology case, deliberately left untouched by
this feature) -- here the aggregate is a plain reducer
(``{"time": {"reduce": "mean", "spread": "std"}}``), so both lanes' mean/spread
are computed once, over the whole record each keeps. Before this feature, the
*test* (model) lane was already pruned to the cast-nearest steps
(``_reference_time_targets``), but the *reference* (station) lane was not
symmetrically cropped to the test's own record -- a cast entirely outside the
model run still fed the reference's mean/std, and still dragged in the model's
boundary step as its "nearest" match. These tests exercise the fix: casts
outside the test source's declared coverage are excluded from both lanes'
statistics, with a warning naming how many, or a clear error when none
overlap at all.
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
DEPTHS = np.array([5.0, 25.0])

#: Two casts inside the model's 2024 record, two a year later, well outside it.
CAST_TIMES = pd.to_datetime(
    ["2024-04-20", "2024-07-10", "2025-01-15", "2025-04-20"]
)
#: A per-cast offset, uniform across depth, so the *mean* offset alone tells
#: apart "only the first two casts" (mean offset 1.0) from "all four" (mean
#: offset 3.0) -- and the *std* likewise (1.0 vs sqrt(5)).
CAST_OFFSETS = np.array([0.0, 2.0, 4.0, 6.0])
BASE_BY_DEPTH = 24.0 - 0.05 * DEPTHS  # [23.75, 22.75]


def _station_dataset(cast_times=CAST_TIMES, offsets=CAST_OFFSETS) -> xr.Dataset:
    values = BASE_BY_DEPTH[None, :] + offsets[:, None]
    return xr.Dataset(
        {"TEMP": (("time", "depth"), values, {"units": "degC"})},
        coords={"time": cast_times, "depth": DEPTHS},
    ).assign_coords(lon=STATION_LON, lat=STATION_LAT)


def _daily_test_grid(start="2024-01-01", periods=366) -> xr.Dataset:
    """A year of daily model output -- declared coverage below matches this
    range exactly, so the crop this feature adds is exact too."""
    lon = np.array([STATION_LON - 0.05, STATION_LON, STATION_LON + 0.05])
    lat = np.array([STATION_LAT - 0.05, STATION_LAT, STATION_LAT + 0.05])
    time = pd.date_range(start, periods=periods, freq="D")
    base = 24.5 - 0.05 * DEPTHS
    values = (
        base[None, :, None, None]
        + np.zeros((periods, 1, lat.size, lon.size))
    )
    da = xr.DataArray(
        values,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": DEPTHS, "lat": lat, "lon": lon},
        name=TEMPERATURE,
        attrs={"units": "degC"},
    )
    return da.to_dataset()


def _install(monkeypatch, *, station: xr.Dataset, his_coverage: bool):
    import ocean_skill as osk
    from ocean_skill import catalog

    lanes = {"ctd_station": station, "his": _daily_test_grid()}
    his_meta = {"standard_names": {TEMPERATURE: TEMPERATURE}}
    if his_coverage:
        his_meta["time_coverage_start"] = "2024-01-01"
        his_meta["time_coverage_end"] = "2024-12-31"
    metas = {
        "ctd_station": {
            "featureType": "timeSeriesProfile",
            "axes": {"T": "time", "Z": "depth"},
            "standard_names": {"TEMP": TEMPERATURE},
            "geospatial_lon_min": STATION_LON,
            "geospatial_lon_max": STATION_LON,
            "geospatial_lat_min": STATION_LAT,
            "geospatial_lat_max": STATION_LAT,
        },
        "his": his_meta,
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    return lanes


def _time_targets_spy(monkeypatch):
    import ocean_skill.align as align_mod

    calls = []
    real = align_mod.subset_to_time_targets

    def spy(obj, targets, method="nearest"):
        out = real(obj, targets, method=method)
        calls.append((method, np.asarray(targets).copy(), out.sizes.get("time")))
        return out

    monkeypatch.setattr(align_mod, "subset_to_time_targets", spy)
    return calls


def _plain_comparison(**overrides):
    from ocean_skill.comparison import Comparison

    kwargs = dict(
        reference="ctd_station",
        test="his",
        variable=TEMPERATURE,
        select={"depth": [5.0, 25.0]},
        aggregate={"time": {"reduce": "mean", "spread": "std"}},
        cache=False,
    )
    kwargs.update(overrides)
    return Comparison(**kwargs)


def test_straddling_casts_are_cropped_to_the_test_record(monkeypatch):
    """2 of 4 casts fall outside "his"'s declared 2024 coverage: the obs
    mean/spread, and the model's own time_targets, both drop to the 2 in-range
    casts, with one warning naming the exclusion."""
    _install(monkeypatch, station=_station_dataset(), his_coverage=True)
    calls = _time_targets_spy(monkeypatch)
    c = _plain_comparison()

    with pytest.warns(UserWarning, match=r"2 of 4 casts"):
        aligned = c.align()

    # Only the 2 in-record casts ever reach the model's own nearest-step prune.
    assert calls
    _, targets_seen, after = calls[0]
    assert after == 2
    assert set(pd.Timestamp(t) for t in targets_seen) == set(CAST_TIMES[:2])

    # The reference mean/spread reflect only the 2 in-record casts (offsets
    # 0.0, 2.0 -> mean offset 1.0, std 1.0), not all 4 (mean offset 3.0).
    ref = aligned["reference"]
    expected = BASE_BY_DEPTH + 1.0
    order = np.argsort(np.abs(ref["depth"].values))  # depth axis may be signed
    np.testing.assert_allclose(
        sorted(ref.values.tolist()), sorted(expected.tolist()), atol=1e-8
    )
    assert "reference_spread" in aligned
    np.testing.assert_allclose(
        aligned["reference_spread"].values, np.full(2, 1.0), atol=1e-8
    )


def test_no_catalog_coverage_on_test_fails_open(monkeypatch):
    """When the test source declares no time coverage at all, this feature is
    a no-op -- today's (pre-fix) behavior, unchanged: every cast, in or out of
    the model's actual span, still feeds the reference's mean/spread."""
    _install(monkeypatch, station=_station_dataset(), his_coverage=False)
    c = _plain_comparison()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aligned = c.align()
    assert not any("time aggregate collapses time" in str(w.message) for w in caught)

    ref = aligned["reference"]
    expected_all_four = BASE_BY_DEPTH + CAST_OFFSETS.mean()
    np.testing.assert_allclose(
        sorted(ref.values.tolist()), sorted(expected_all_four.tolist()), atol=1e-8
    )


def test_all_casts_inside_is_unaffected(monkeypatch):
    """Every cast within the declared record: no warning, and the result
    matches the plain, uncropped mean/std -- nothing was excluded."""
    in_range_times = pd.to_datetime(["2024-04-20", "2024-07-10"])
    in_range_offsets = np.array([0.0, 2.0])
    _install(
        monkeypatch,
        station=_station_dataset(in_range_times, in_range_offsets),
        his_coverage=True,
    )
    c = _plain_comparison()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aligned = c.align()
    assert not any("time aggregate collapses time" in str(w.message) for w in caught)

    ref = aligned["reference"]
    expected = BASE_BY_DEPTH + 1.0
    np.testing.assert_allclose(
        sorted(ref.values.tolist()), sorted(expected.tolist()), atol=1e-8
    )


def test_all_casts_outside_raises(monkeypatch):
    """No cast falls inside the test's declared record: raise, rather than
    silently average zero casts (or fall through to a confusing NaN)."""
    out_of_range = pd.to_datetime(["2025-01-15", "2025-04-20"])
    _install(
        monkeypatch,
        station=_station_dataset(out_of_range, np.array([4.0, 6.0])),
        his_coverage=True,
    )
    c = _plain_comparison()

    with pytest.raises(ValueError, match=r"none of .* cast times fall within"):
        c.align()


def test_climatology_fold_is_left_untouched(monkeypatch):
    """A groupby/resample climatology is explicitly out of scope: no crop, no
    new warning, and every cast (in or out of the test's record) still feeds
    the model's own time_targets prune, exactly as before this feature."""
    _install(monkeypatch, station=_station_dataset(), his_coverage=True)
    calls = _time_targets_spy(monkeypatch)
    c = _plain_comparison(
        aggregate={"time": {"groupby": "month", "reduce": "mean", "spread": "std"}}
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c.align()
    assert not any("time aggregate collapses time" in str(w.message) for w in caught)

    assert calls
    _, targets_seen, after = calls[0]
    # All 4 cast times are still handed to the model's own prune -- none
    # dropped for being outside "his"'s declared record. (`after` can still
    # be fewer than 4: subset_to_time_targets itself dedupes two casts that
    # share the same nearest model step -- unrelated to, and unaffected by,
    # this feature.)
    assert len(targets_seen) == len(CAST_TIMES)
    assert set(pd.Timestamp(t) for t in targets_seen) == set(CAST_TIMES)
    assert after <= len(CAST_TIMES)
