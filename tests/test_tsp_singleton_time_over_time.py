"""A ``timeSeriesProfile`` reference narrowed to exactly one surviving cast, in
an ``over="time"`` comparison, keeps that cast as a real dimension rather than
being squeezed to a scalar coordinate.

The motivating bug: comparing repeat-visit CTD stations against a model with
``depths=[{"min": 0, "max": 5}], aggregate={"Z": "mean"}`` (``over="time"``, the
mooring-at-a-depth reading) crashed with

    ValueError: the reference lane has no 'time' axis to score over (its
    dimensions are []). For a comparison of single maps, leave over= unset.

for a station whose casts happened to narrow down to exactly one inside the
test's own record. ``compare()``'s ``skip_missing=True`` does not catch a bare
``ValueError`` (only ``KeyError``/``NoValidData``), so this aborted the whole
batch of comparisons rather than skipping the one problematic station.

Root cause: ``_prepare()`` (``ocean_skill/comparison.py``) has always
unconditionally squeezed a ``timeSeriesProfile`` lane's time dimension to a
non-dimensional scalar *coordinate* whenever narrowing happens to leave exactly
one step standing -- written for the genuinely single-instant case (a period
string select like ``"2024-06-11"``, scored with ``over="Z"``/``None``, where
``ocean_skill.align._sample_test_at_instant`` needs that stamp kept but not as
a standing axis). It never checked whether the *caller* actually wanted time
kept standing (``over="time"``, or the ``TIME_DEPTH_OVER`` sentinel) -- in
which case a single surviving cast is still one point to score, not zero
dimensions.

This bug was previously masked for many such stations by an unrelated,
already-fixed defect: a station whose declared record ran past the test
source's own used to ``KeyError`` earlier in ``align()`` (see
``tests/test_station_record_past_model.py``), which *is* caught by
``skip_missing=True``. Once that earlier crash was fixed, execution proceeds
further for those stations and reaches this squeeze bug instead -- this
module's fixtures narrow a reference to one cast directly (no reliance on that
other narrowing chain), so they exercise this bug in isolation.

Companion to ``tests/test_tsp_end_to_end.py``, whose
``test_a_pinned_visit_reads_as_one_profile_on_its_own_depths`` exercises the
*intended* squeeze (``over="Z"``, a period-string select) -- deliberately left
squeezing, and re-run here to confirm this fix does not disturb it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

TEMPERATURE = "sea_water_temperature"


# -- _prepare() itself: the exact mechanism, in isolation ----------------------


def _singleton_tsp_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"TEMP": (("time", "depth"), [[12.0, 11.5]], {"units": "degC"})},
        coords={"time": pd.to_datetime(["2024-04-10"]), "depth": [1.0, 4.0]},
    ).assign_coords(lon=-158.0, lat=22.75)


@pytest.mark.parametrize("over", [None, "Z"])
def test_prepare_still_squeezes_when_time_is_not_the_scored_axis(over):
    """Unchanged behavior: over=None/"Z" still collapses the singleton time
    dimension to a scalar coordinate, exactly as before this fix."""
    from ocean_skill.comparison import _prepare

    meta = {"featureType": "timeSeriesProfile", "standard_names": {"TEMP": TEMPERATURE}}
    with pytest.warns(UserWarning):
        da, _ = _prepare(
            _singleton_tsp_dataset(), meta, TEMPERATURE,
            {"depth": {"min": 0, "max": 5}}, {"Z": "mean"}, over=over,
        )
    assert "time" not in da.dims
    assert "time" in da.coords, "the stamp itself is kept, just not as a dimension"


@pytest.mark.parametrize("over_name", ["time", "TIME_DEPTH_OVER"])
def test_prepare_keeps_time_standing_when_it_is_the_scored_axis(over_name):
    """The fix: over="time" (or the TIME_DEPTH_OVER sentinel, which also keeps
    time standing) leaves the singleton time dimension in place -- one point
    to score, not zero dimensions."""
    from ocean_skill.align import TIME_DEPTH_OVER
    from ocean_skill.comparison import _prepare

    over = TIME_DEPTH_OVER if over_name == "TIME_DEPTH_OVER" else over_name
    meta = {"featureType": "timeSeriesProfile", "standard_names": {"TEMP": TEMPERATURE}}
    with pytest.warns(UserWarning):
        da, _ = _prepare(
            _singleton_tsp_dataset(), meta, TEMPERATURE,
            {"depth": {"min": 0, "max": 5}}, {"Z": "mean"}, over=over,
        )
    assert da.sizes["time"] == 1


# -- Full Comparison.align()/compare() path -------------------------------------


def _hv2_one_visit_frame() -> pd.DataFrame:
    """A repeat-visit station's read that happens to carry exactly one visit --
    the shape _prepare's squeeze fires on (featureType=timeSeriesProfile, time
    dim size 1), reached here directly rather than via some other narrowing
    chain, to isolate this bug from any other fix."""
    rows = [
        ("2024-04-10", 1, -158.0, 22.75, 12.0),
        ("2024-04-10", 4, -158.0, 22.75, 11.5),
    ]
    return pd.DataFrame(
        rows, columns=["time", "depth (m)", "lon", "lat", "Temperature (degC)"]
    )


def _his_grid() -> xr.Dataset:
    depth = np.array([1.0, 4.0])
    lon = np.array([-158.05, -158.0, -157.95])
    lat = np.array([22.7, 22.75, 22.8])
    time = pd.date_range("2024-04-08", periods=5, freq="D")  # covers the one visit
    base = 12.0 - 0.1 * depth
    values = base[None, :, None, None] + np.zeros((5, 1, lat.size, lon.size))
    da = xr.DataArray(
        values, dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        name=TEMPERATURE, attrs={"units": "degC"},
    )
    return da.to_dataset()


@pytest.fixture
def one_visit_station_and_his(monkeypatch):
    import ocean_skill as osk
    from ocean_skill import catalog

    lanes = {"ctd_station_HV2": _hv2_one_visit_frame(), "his": _his_grid()}
    metas = {
        "ctd_station_HV2": {
            "featureType": "timeSeriesProfile",
            "featureType_source": "declared",
            "datasetID": "ctd_station_HV2",
            "standard_names": {"Temperature (degC)": TEMPERATURE},
            "geospatial_lon_min": -158.0, "geospatial_lon_max": -158.0,
            "geospatial_lat_min": 22.75, "geospatial_lat_max": 22.75,
        },
        "his": {
            "standard_names": {TEMPERATURE: TEMPERATURE},
            "geospatial_lon_min": -158.05, "geospatial_lon_max": -157.95,
            "geospatial_lat_min": 22.7, "geospatial_lat_max": 22.8,
        },
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    return lanes


def test_a_single_surviving_cast_scores_instead_of_crashing(one_visit_station_and_his):
    from ocean_skill.comparison import Comparison

    c = Comparison(
        reference="ctd_station_HV2", test="his", variable=TEMPERATURE,
        select={"depth": {"min": 0, "max": 5}}, aggregate={"Z": "mean"}, cache=False,
    )
    assert c.over == "time"

    with pytest.warns(UserWarning):
        aligned = c.align()  # must not raise
    assert aligned.sizes["time"] == 1
    assert set(aligned.data_vars) >= {"test", "reference", "difference"}


def test_compare_does_not_abort_the_whole_batch(one_visit_station_and_his, capsys):
    from ocean_skill import comparison

    with pytest.warns(UserWarning):
        out = comparison.compare(
            reference="ctd_station_HV2", test="his", variables=[TEMPERATURE],
            depths=[{"min": 0, "max": 5}], aggregate={"Z": "mean"}, cache=False,
        )
    assert len(out) == 1
    assert "0 skipped" in capsys.readouterr().out
