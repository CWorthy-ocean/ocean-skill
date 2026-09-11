"""A repeat-visit station whose *declared* record runs past the model's own
is clipped to the overlap, not dropped whole.

The motivating bug: 12 ``timeSeriesProfile`` CTD stations compared against a
``bgc`` model, with a depth band collapsed and time kept standing
(``depths=[{"min": 0, "max": 5}], aggregate={"Z": "mean"}`` -- ``over="time"``,
the mooring-at-a-depth reading). Ten of twelve stations vanished from the
result and the survivors' time range collapsed to a single month, because
each station's catalog-declared coverage (``time_coverage_end``) ran a year
past the model's own (a real 13-month repeat-visit record; the model covers
one season). ``Comparison.align`` builds a crop window from the *reference's*
declared coverage and applies it to the *test* lane
(``Comparison._reference_narrowing``) -- and until this fix, that crop
(``ocean_skill.align.subset_to_time``) used a label ``.sel`` slice, which
``KeyError``s when the window's bound lands past a non-monotonic test time
axis (a plausible multi-file-concat artifact) rather than clipping to the
overlap. A second, closely related crop just downstream
(``ocean_skill.align.subset_to_time_targets``, pruning to the reference's own
cast-nearest steps) separately ``ValueError``ed on the same non-monotonic
axis, since ``pandas``' own nearest-step lookup requires a sorted index --
fixing the first crop alone was not enough to clear the whole path.
``compare()``'s ``skip_missing=True`` then silently drops the whole station
rather than surfacing either crop failure.

The test lane's time axis is built deliberately non-monotonic below: on a
plain monotonic ``datetime64`` axis, a label slice's out-of-range bound is
tolerated (silently clipped) and both crops are no-ops here, so a monotonic
fixture would not reproduce the crash this guards against -- see
``tests/test_subset_to_time.py`` for that distinction pinned directly on each
crop function. This module instead exercises the full
``Comparison.align``/``compare()`` path so a regression here is caught even
if some other layer starts relying on either crop's old behavior.

Companion to ``tests/test_collapsed_time_record_crop.py``, which covers the
same "reference record outruns the test's" question for the *other* branch
(a time aggregate that collapses the axis, ``over`` unset/not time) --
deliberately left untouched by this fix.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

TEMPERATURE = "sea_water_temperature"
STATION_LON, STATION_LAT = -158.0, 22.75
DEPTHS = np.array([2.0, 4.0])

#: Mirrors the real HV1 station: casts well inside the model's season, plus
#: casts a year later the model was never run for.
CAST_TIMES = pd.to_datetime(
    ["2024-04-20", "2024-07-10", "2024-10-08", "2025-01-15", "2025-04-20"]
)
CAST_OFFSETS = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
BASE_BY_DEPTH = np.array([12.0, 11.5])


def _station_dataset() -> xr.Dataset:
    values = BASE_BY_DEPTH[None, :] + CAST_OFFSETS[:, None]
    return xr.Dataset(
        {"TEMP": (("time", "depth"), values, {"units": "degC"})},
        coords={"time": CAST_TIMES, "depth": DEPTHS},
    ).assign_coords(lon=STATION_LON, lat=STATION_LAT)


def _bgc_grid() -> xr.Dataset:
    """A one-season model run, its time axis deliberately out of strict sorted
    order -- the shape that turns an out-of-range crop bound into a KeyError
    (see the module docstring) rather than a silent, tolerant clip."""
    lon = np.array([STATION_LON - 0.05, STATION_LON, STATION_LON + 0.05])
    lat = np.array([STATION_LAT - 0.05, STATION_LAT, STATION_LAT + 0.05])
    time = pd.to_datetime(
        [
            "2024-02-01",
            "2024-07-10",
            "2024-04-20",  # out of order relative to the previous step
            "2024-10-08",
            "2024-11-29",
        ]
    )
    values = (
        BASE_BY_DEPTH[None, :, None, None]
        + np.zeros((time.size, 1, lat.size, lon.size))
    )
    da = xr.DataArray(
        values,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": DEPTHS, "lat": lat, "lon": lon},
        name="TEMP",
        attrs={"units": "degC"},
    )
    return da.to_dataset()


def _install(monkeypatch):
    import ocean_skill as osk
    from ocean_skill import catalog

    lanes = {"ctd_station": _station_dataset(), "bgc": _bgc_grid()}
    metas = {
        "ctd_station": {
            "featureType": "timeSeriesProfile",
            "axes": {"T": "time", "Z": "depth"},
            "standard_names": {"TEMP": TEMPERATURE},
            "geospatial_lon_min": STATION_LON,
            "geospatial_lon_max": STATION_LON,
            "geospatial_lat_min": STATION_LAT,
            "geospatial_lat_max": STATION_LAT,
            # The declared coverage that outruns the model -- the real bug's
            # trigger. Padded by _TIME_COVERAGE_PAD_DAYS on top of this.
            "time_coverage_start": "2024-04-04",
            "time_coverage_end": "2025-04-28",
        },
        "bgc": {
            "standard_names": {"TEMP": TEMPERATURE},
            "geospatial_lon_min": STATION_LON - 0.05,
            "geospatial_lon_max": STATION_LON + 0.05,
            "geospatial_lat_min": STATION_LAT - 0.05,
            "geospatial_lat_max": STATION_LAT + 0.05,
            "time_coverage_start": "2024-02-01",
            "time_coverage_end": "2024-11-29",
        },
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    return lanes


def _plain_comparison(**overrides):
    from ocean_skill.comparison import Comparison

    kwargs = dict(
        reference="ctd_station",
        test="bgc",
        variable=TEMPERATURE,
        select={"depth": {"min": 0, "max": 5}},
        aggregate={"Z": "mean"},
        cache=False,
    )
    kwargs.update(overrides)
    return Comparison(**kwargs)


def test_station_with_a_declared_record_past_the_model_is_not_dropped(monkeypatch):
    """Before the fix: KeyError during align(), swallowed by compare()'s
    skip_missing=True, dropping the whole station. After: the comparison
    succeeds, restricted to the casts inside the model's own record."""
    _install(monkeypatch)
    c = _plain_comparison()
    assert c.over == "time"  # sanity: this is the time-standing path the bug hit

    aligned = c.align()  # must not raise

    times = pd.to_datetime(aligned["time"].values)
    assert times.max() <= pd.Timestamp("2024-11-29")
    assert (times.year == 2025).sum() == 0


def test_compare_keeps_the_station_instead_of_skipping_it(monkeypatch, capsys):
    from ocean_skill import comparison

    _install(monkeypatch)
    out = comparison.compare(
        reference="ctd_station",
        test="bgc",
        variables=[TEMPERATURE],
        depths=[{"min": 0, "max": 5}],
        aggregate={"Z": "mean"},
        cache=False,
    )
    assert len(out) == 1  # not skipped
    printed = capsys.readouterr().out
    assert "0 skipped" in printed
    assert "skipped ctd_station" not in printed


def test_a_pair_spec_variable_is_not_dropped_either(monkeypatch):
    """The chl pair-spec case from the real bug: the gate that could skip this
    clip lives on self.select (plain here), not on the variable, so a
    pair-spec variable takes the same path and must succeed the same way.
    (An identity pair-spec -- both sides name the same field -- is enough to
    exercise ``is_pair_spec(self.variable)``; the units mismatch that made the
    real chl comparison's own numbers wrong is a separate, already-documented
    issue, not this fix's concern.)"""
    _install(monkeypatch)
    c = _plain_comparison(
        variable={"test": "TEMP", "reference": "TEMP", "standard_name": TEMPERATURE}
    )
    aligned = c.align()  # must not raise
    assert pd.to_datetime(aligned["time"].values).max() <= pd.Timestamp("2024-11-29")
