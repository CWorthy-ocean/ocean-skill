"""``times={"resample": ..., "reduce": "pool"}``: a bin's steps are samples.

``reduce: "mean"`` averages each bin's steps into one map per lane and scores *that*.
``reduce: "pool"`` keeps every step as samples of one score -- bias, RMSE, correlation
over every valid cell of every step, area-weighted as a single map is -- while the
comparison still draws the bin's mean map. The two differ whenever the steps' errors do
not line up, which the synthetic data here arranges on purpose: the model is 1 degC too
warm half the year and 1 degC too cold the other half, so a year's mean-map RMSE is ~0
and its pooled RMSE is 1.

Sources are synthetic gridded fields stubbed the way ``tests/test_climatology_follows_
month.py`` does; the expected numbers are computed independently with numpy.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog
from ocean_skill.comparison import (
    Comparison,
    _is_calendar_time_value,
    _normalize_times,
    compare,
)

VAR = "sea_water_potential_temperature"
LON = np.array([200.0, 200.5, 201.0, 201.5])
LAT = np.array(
    [10.0, 30.0, 50.0, 70.0]
)  # wide on purpose: cos(lat) weights differ a lot
# 24 monthly steps, Jan 2010 .. Dec 2011, one per month
TIMES = pd.DatetimeIndex(
    [pd.Timestamp(2010 + i // 12, i % 12 + 1, 16) for i in range(24)]
)
SIGN = np.where(TIMES.month.isin([3, 4, 5, 6, 7, 8]), 1.0, -1.0)

pytestmark = pytest.mark.filterwarnings(
    "ignore:only .* valid cells", "ignore:only .* section cells"
)


def _meta(times):
    return {
        "variables": [VAR],
        "featureType": "grid",
        "geospatial_lon_min": float(LON.min()),
        "geospatial_lon_max": float(LON.max()),
        "geospatial_lat_min": float(LAT.min()),
        "geospatial_lat_max": float(LAT.max()),
        "standard_names": {VAR: VAR},
        "time_coverage_start": str(times.min().date()),
        "time_coverage_end": str(times.max().date()),
    }


def _field(values, times=TIMES):
    return xr.Dataset(
        {
            VAR: (
                ("time", "lat", "lon"),
                values,
                {"units": "degC", "standard_name": VAR},
            )
        },
        coords={"time": times, "lat": LAT, "lon": LON},
    )


@pytest.fixture
def pair(monkeypatch):
    """Return ``(t, r)``: ``t`` is ``r`` plus a seasonal error of +/- 1."""
    rng = np.random.default_rng(3)
    ref = 15.0 + rng.normal(0.0, 0.5, (24, LAT.size, LON.size))
    test = ref + SIGN[:, None, None]
    sources = {"t": (_field(test), _meta(TIMES)), "r": (_field(ref), _meta(TIMES))}
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )
    return test, ref


def _weighted(test, ref, rows):
    """Area-weighted pooled bias/rmse/n over the time steps ``rows``, in numpy."""
    d = (test - ref)[rows]
    w = np.broadcast_to(np.cos(np.deg2rad(LAT))[None, :, None], d.shape)
    return {
        "bias": float((w * d).sum() / w.sum()),
        "rmse": float(np.sqrt((w * d**2).sum() / w.sum())),
        "n": d.size,
    }


def _year(comps, label):
    return next(c for c in comps if c.label == label)


# -- grammar ---------------------------------------------------------------------------


def test_pool_is_a_reduce_of_the_bins_form():
    assert _normalize_times({"resample": "1YS", "reduce": "pool"}) == (
        "bins",
        {"resample": "1YS", "reduce": "pool"},
    )


def test_pool_takes_no_other_key():
    with pytest.raises(ValueError, match="takes no other key"):
        _normalize_times({"resample": "1YS", "reduce": "pool", "q": 0.5})


def test_pool_refuses_over_and_a_time_aggregate(pair):
    with pytest.raises(ValueError, match="Pick one"):
        compare(
            reference="r",
            test="t",
            variables=[VAR],
            times={"resample": "1YS", "reduce": "pool"},
            over="time",
        )
    with pytest.raises(ValueError, match="aggregate= time entry"):
        compare(
            reference="r",
            test="t",
            variables=[VAR],
            times={"resample": "1YS", "reduce": "pool"},
            aggregate={"time": "mean"},
        )


@pytest.mark.parametrize(
    "aggregate",
    [
        {"time": [{"groupby": "month", "reduce": "mean"}, "var"]},
        {"test": {"time": "mean"}, "reference": {}},
        {"test": {}, "reference": {"time": "mean"}},
        [{"time": "mean"}],
        [{"lon": "mean"}, {"test": {"time": "mean"}, "reference": {}}],
    ],
    ids=["chain", "pair-test", "pair-reference", "list", "list-pair"],
)
def test_pool_refuses_a_time_aggregate_in_every_shape(pair, aggregate):
    """A chain, a pair-spec side or a list entry carrying ``time`` is refused too."""
    with pytest.raises(ValueError, match="time entry"):
        compare(
            reference="r",
            test="t",
            variables=[VAR],
            times={"resample": "1YS", "reduce": "pool"},
            aggregate=aggregate,
        )


def test_calendar_time_values_are_told_from_raw_positions():
    assert _is_calendar_time_value("2010-03")
    assert _is_calendar_time_value({"min": "2010-01", "max": None})
    assert _is_calendar_time_value({"season": "MAM"})
    assert not _is_calendar_time_value(396.5)
    assert not _is_calendar_time_value([396.5, 397.5])
    assert not _is_calendar_time_value({"min": 396.0, "max": 400.0})


# -- the numbers: pooled is not the score of the mean map ---------------------------


def test_the_pooled_score_is_over_every_cell_of_every_step(pair):
    test, ref = pair
    out = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1YS", "reduce": "pool"},
    )
    assert [c.label for c in out] == ["2010", "2011"]
    for label, rows in (("2010", slice(0, 12)), ("2011", slice(12, 24))):
        got = _year(out, label).metrics()
        want = _weighted(test, ref, rows)
        assert got["bias"] == pytest.approx(want["bias"])
        assert got["rmse"] == pytest.approx(want["rmse"])
        assert got["n"] == want["n"]
        assert got["time"] == label


def test_the_mean_map_score_is_a_different_number(pair):
    mean = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1YS", "reduce": "mean"},
    )
    pooled = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1YS", "reduce": "pool"},
    )
    # +1 for six months and -1 for six: the maps average the error away...
    assert _year(mean, "2010").metrics()["rmse"] < 0.01
    # ...and the pooled score keeps it
    assert _year(pooled, "2010").metrics()["rmse"] == pytest.approx(1.0, abs=1e-9)
    assert _year(pooled, "2010").metrics()["n"] == 12 * LAT.size * LON.size
    assert _year(mean, "2010").metrics()["n"] == LAT.size * LON.size


# -- it still draws: the bin's mean map -----------------------------------------------


def test_the_aligned_pair_is_the_bins_mean_map(pair):
    test, ref = pair
    out = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1YS", "reduce": "pool"},
    )
    comp = _year(out, "2010")
    aligned = comp.aligned
    assert set(aligned["test"].dims) == {"lat", "lon"}
    np.testing.assert_allclose(
        aligned["test"].values, test[:12].mean(axis=0), atol=1e-9
    )
    np.testing.assert_allclose(
        aligned["reference"].values, ref[:12].mean(axis=0), atol=1e-9
    )
    np.testing.assert_allclose(
        aligned["difference"].values, (test[:12] - ref[:12]).mean(axis=0), atol=1e-9
    )
    # drawable like any single map: a field_row item, with the bin on its title
    item = comp.as_item()
    assert item["time"] is not None
    assert comp.family == "field_row"


def test_a_pooled_bin_reports_no_effective_sample_size(pair):
    out = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1YS", "reduce": "pool"},
    )
    assert "n_eff" not in out[0].metrics()


def test_a_select_window_narrows_which_steps_pool(pair):
    test, ref = pair
    out = compare(
        reference="r",
        test="t",
        variables=[VAR],
        select={"time": {"min": "2010-03", "max": "2010-08"}},
        times={"resample": "1YS", "reduce": "pool"},
    )
    assert len(out) == 1
    got = out[0].metrics()
    assert got["time"] == "2010"
    want = _weighted(test, ref, slice(2, 8))
    assert got["n"] == want["n"] == 6 * LAT.size * LON.size
    assert got["bias"] == pytest.approx(1.0)  # March..August is all +1


def test_monthly_bins_pool_one_step_each_and_equal_the_mean_form(pair):
    mean = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1MS", "reduce": "mean"},
    )
    pooled = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1MS", "reduce": "pool"},
    )
    assert [c.label for c in pooled] == [c.label for c in mean]
    for a, b in zip(mean, pooled, strict=True):
        assert a.metrics()["bias"] == pytest.approx(b.metrics()["bias"])
        assert a.metrics()["rmse"] == pytest.approx(b.metrics()["rmse"])


# -- steps that cannot be compared --------------------------------------------------


def _install(monkeypatch, test, ref):
    sources = {"t": (_field(test), _meta(TIMES)), "r": (_field(ref), _meta(TIMES))}
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )


def test_a_step_the_reference_lacks_adds_no_samples(monkeypatch):
    rng = np.random.default_rng(4)
    ref = 15.0 + rng.normal(0.0, 0.5, (24, LAT.size, LON.size))
    test = ref + SIGN[:, None, None]
    ref_gappy = ref.copy()
    ref_gappy[4] = np.nan  # May 2010 has no reference at all
    _install(monkeypatch, test, ref_gappy)
    out = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1YS", "reduce": "pool"},
    )
    got = _year(out, "2010").metrics()
    assert got["n"] == 11 * LAT.size * LON.size
    want = _weighted(test, ref, [i for i in range(12) if i != 4])
    assert got["bias"] == pytest.approx(want["bias"])


def test_a_step_that_cannot_be_compared_is_left_out_with_a_warning(pair, monkeypatch):
    from ocean_skill.align import NoValidData

    real = Comparison.align

    def flaky(self, *, refresh=False):
        if str(self.select.get("time", "")).startswith("2010-05"):
            raise NoValidData("no model data")
        return real(self, refresh=refresh)

    monkeypatch.setattr(Comparison, "align", flaky)
    with pytest.warns(
        UserWarning, match="left 1 of the bin's 12 steps out of the pool"
    ):
        out = compare(
            reference="r",
            test="t",
            variables=[VAR],
            times={"resample": "1YS", "reduce": "pool"},
        )
    assert _year(out, "2010").metrics()["n"] == 11 * LAT.size * LON.size


def test_a_bin_with_no_comparable_step_is_skipped(pair, monkeypatch):
    from ocean_skill.align import NoValidData

    def never(self, *, refresh=False):
        raise NoValidData("no model data")

    monkeypatch.setattr(Comparison, "align", never)
    out = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1YS", "reduce": "pool"},
    )
    assert len(out) == 0


def test_pooling_refuses_a_series_member():
    stub = SimpleNamespace(
        is_series=True,
        is_profile=False,
        is_time_depth=False,
        test_name="t",
        reference_name="r",
    )
    with pytest.raises(ValueError, match="place through time or depth"):
        Comparison._pooled_over_steps([stub], select={}, label="x")


# -- a section: depth x along ------------------------------------------------------


def _four_d(offset):
    lon = np.arange(180.0, 201.0, 10.0)
    lat = np.arange(-30.0, 31.0, 10.0)
    depth = np.array([0.0, 50.0, 200.0])
    d, la, lo = np.meshgrid(depth, lat, lon, indexing="ij")
    base = d / 100 + la / 10 + lo / 1000
    values = np.stack([base + offset[i] for i in range(len(TIMES))])
    return xr.Dataset(
        {
            VAR: (
                ("time", "depth", "lat", "lon"),
                values,
                {"units": "degC", "standard_name": VAR},
            )
        },
        coords={
            "time": TIMES,
            "depth": ("depth", depth, {"positive": "down", "units": "m"}),
            "lat": lat,
            "lon": lon,
        },
    )


def test_a_slab_section_pools_its_steps_too(monkeypatch):
    meta = {
        **_meta(TIMES),
        "geospatial_lon_min": 180.0,
        "geospatial_lon_max": 200.0,
        "geospatial_lat_min": -30.0,
        "geospatial_lat_max": 30.0,
    }
    sources = {
        "t": (_four_d(SIGN), meta),
        "r": (_four_d(np.zeros(len(TIMES))), meta),
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )
    box = {
        "lon": {"min": 180, "max": 200},
        "lat": {"min": -30, "max": 30},
        "depth": [0.0, 50.0, 200.0],
    }
    kwargs = {
        "reference": "r",
        "test": "t",
        "variables": [VAR],
        "select": box,
        "aggregate": {"lon": "mean"},
    }
    mean = compare(**kwargs, times={"resample": "1YS", "reduce": "mean"})
    pooled = compare(**kwargs, times={"resample": "1YS", "reduce": "pool"})
    comp = _year(pooled, "2010")
    assert comp.is_section
    assert _year(mean, "2010").metrics()["rmse"] < 0.01
    assert comp.metrics()["rmse"] == pytest.approx(1.0)
    assert comp.metrics()["n"] == 12 * _year(mean, "2010").metrics()["n"]
    assert set(comp.aligned["test"].dims) == set(
        _year(mean, "2010").aligned["test"].dims
    )


# -- with a climatology reference (the two features together) ---------------------


def test_pooling_against_a_climatology_pairs_each_step_with_its_month(monkeypatch):
    clim_values = (8.0 + 0.2 * np.arange(12))[:, None, None] * np.ones(
        (12, LAT.size, LON.size)
    )
    clim = xr.Dataset(
        {
            VAR: (
                ("month", "lat", "lon"),
                clim_values,
                {"units": "degC", "standard_name": VAR},
            )
        },
        coords={"month": np.arange(1, 13), "lat": LAT, "lon": LON},
    )
    test_values = (8.0 + 0.2 * (np.asarray(TIMES.month) - 1) + SIGN)[
        :, None, None
    ] * np.ones((24, LAT.size, LON.size))
    meta_clim = {
        "variables": [VAR],
        "featureType": "grid",
        "standard_names": {VAR: VAR},
        "geospatial_lon_min": float(LON.min()),
        "geospatial_lon_max": float(LON.max()),
        "geospatial_lat_min": float(LAT.min()),
        "geospatial_lat_max": float(LAT.max()),
    }
    sources = {"t": (_field(test_values), _meta(TIMES)), "c": (clim, meta_clim)}
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )
    out = compare(
        reference="c",
        test="t",
        variables=[VAR],
        times={"resample": "1YS", "reduce": "pool"},
    )
    got = _year(out, "2010").metrics()
    # the test departs from its month's climatology by exactly +/-1: pooled RMSE is 1
    assert got["rmse"] == pytest.approx(1.0)
    assert got["bias"] == pytest.approx(0.0, abs=1e-9)
    assert got["n"] == 12 * LAT.size * LON.size
