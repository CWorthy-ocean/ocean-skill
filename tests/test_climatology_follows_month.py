"""A monthly-climatology reference follows the test lane's calendar months.

A reference that is twelve calendar months with no dated time axis (WOA23's monthly
stack, a roms-tools climatology file, Holte and Talley) has no instant to pair
a dated test step with. Against a dated test lane ``compare()`` lays it out along the
test's own steps, each taking its calendar month
(:func:`ocean_skill.align.follow_months`, driven by
:meth:`ocean_skill.comparison.Comparison._follow_steps`), so that

* each ``times=`` bin's reference is that bin's month(s) -- a bin spanning several is
  mean of those months *weighted by how many test steps fall in each*;
* ``over="time"`` pairs the climatology repeated at the test's months;
* ``metrics_series`` has one value per test time;
* every request that already worked -- an explicit ``month`` select/aggregate, a time
  aggregate over a WOA-style stack -- is untouched.

Both climatology shapes are covered: ``month`` as a dimension (``month_dim``) and twelve
undecoded steps on ``time`` with a ``month`` coordinate (``woa_like``). Sources are
synthetic and stubbed the way ``tests/test_compare_disjoint_time_reduced.py`` does.
"""

from __future__ import annotations

import contextlib
import warnings
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import align as A
from ocean_skill import catalog
from ocean_skill.comparison import compare
from tests.test_suite_smoke_fixture import (  # noqa: F401  (pacmed is a fixture)
    PACMED_SUITES,
    localized,
    pacmed,
)

VAR = "sea_water_potential_temperature"
LON = np.array([200.0, 200.5, 201.0])
LAT = np.array([50.0, 50.5, 51.0])
DEPTHS = np.array([0.0, 50.0, 200.0])

pytestmark = pytest.mark.filterwarnings(
    "ignore:only .* valid cells",
    "ignore:only .* section cells",
    "ignore:groupby 'month'",
)


def _clim_value(month):
    """Return the climatology's value in ``month``: 8 + 0.2 * (month - 1)."""
    return 8.0 + 0.2 * (np.asarray(month) - 1)


def _attrs():
    return {"units": "degC", "standard_name": VAR}


def _meta(times=None):
    meta = {
        "variables": [VAR],
        "featureType": "grid",
        "geospatial_lon_min": float(LON.min()),
        "geospatial_lon_max": float(LON.max()),
        "geospatial_lat_min": float(LAT.min()),
        "geospatial_lat_max": float(LAT.max()),
        "standard_names": {VAR: VAR},
    }
    if times is not None:
        meta["time_coverage_start"] = str(times.min().date())
        meta["time_coverage_end"] = str(times.max().date())
    return meta


def _dated(times, delta):
    """Test field: the climatology's value at each step's month plus ``delta[i]``."""
    base = _clim_value(times.month)
    values = (base + delta)[:, None, None] * np.ones((len(times), LAT.size, LON.size))
    return xr.Dataset(
        {VAR: (("time", "lat", "lon"), values, _attrs())},
        coords={"time": times, "lat": LAT, "lon": LON},
    )


def month_dim():
    """Build a climatology with ``month`` as the dimension (roms-tools, H&T)."""
    values = _clim_value(np.arange(1, 13))[:, None, None] * np.ones((12, 3, 3))
    return xr.Dataset(
        {VAR: (("month", "lat", "lon"), values, _attrs())},
        coords={"month": np.arange(1, 13), "lat": LAT, "lon": LON},
    )


def woa_like():
    """Build twelve undecoded ``time`` steps with a ``month`` coordinate (WOA)."""
    values = _clim_value(np.arange(1, 13))[:, None, None] * np.ones((12, 3, 3))
    return xr.Dataset(
        {VAR: (("time", "lat", "lon"), values, _attrs())},
        coords={
            "time": (
                "time",
                396.5 + np.arange(12.0),
                {"units": "months since 1955-01-01"},
            ),
            "month": ("time", np.arange(1, 13)),
            "lat": LAT,
            "lon": LON,
        },
    )


# 24 monthly steps, mid-month, Jan 2010 .. Dec 2011; the test is the climatology plus a
# per-step offset, so the bias of any bin is the mean of its offsets exactly.
TIMES = pd.DatetimeIndex(
    [pd.Timestamp(2010 + i // 12, i % 12 + 1, 16) for i in range(24)]
)
DELTA = 0.1 * np.arange(24)


@pytest.fixture
def install(monkeypatch):
    """Stub ``osk.read``/``catalog.resolve`` for named in-memory sources."""

    def _install(sources):
        monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
        monkeypatch.setattr(
            "ocean_skill.sources.read", lambda name, **kw: sources[name][0]
        )
        monkeypatch.setattr(
            catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
        )

    return _install


@pytest.fixture(params=["month_dim", "woa_like"])
def shape(request, install):
    """Install ``t`` (dated) and ``c`` (the climatology, in each of its two shapes)."""
    clim = month_dim() if request.param == "month_dim" else woa_like()
    meta = {**_meta(), "climatology": True, "climatology_period": "monthly"}
    install({"t": (_dated(TIMES, DELTA), _meta(TIMES)), "c": (clim, meta)})
    return request.param


# -- the two helpers ---------------------------------------------------------------


def test_month_axis_names_the_dimension_of_each_climatology_shape():
    assert A.month_axis(month_dim()) == "month"
    assert A.month_axis(woa_like()) == "time"


def test_month_axis_refuses_what_is_not_a_monthly_climatology():
    dated = _dated(TIMES, DELTA)
    assert A.month_axis(dated) is None
    assert A.month_axis(dated.assign_coords(month=("time", TIMES.month))) is None, (
        "a decoded calendar beside the months means the object is dated"
    )
    clim = month_dim()
    assert A.month_axis(clim.assign_coords(month=np.arange(0, 12))) is None
    assert (
        A.month_axis(clim.assign_coords(month=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11]))
        is None
    )
    assert A.month_axis(clim.assign_coords(month=np.arange(1, 13) + 0.5)) is None
    assert A.month_axis(pd.DataFrame({"month": [1, 2]})) is None


@pytest.mark.parametrize("make", [month_dim, woa_like])
def test_follow_months_repeats_the_climatology_at_each_steps_month(make):
    steps = np.array(["2011-03-05", "2011-04-10", "2012-03-20"], dtype="datetime64[ns]")
    out, dropped = A.follow_months(make(), steps)
    assert dropped == 0
    assert out[VAR].dims[0] == "time"
    np.testing.assert_array_equal(out["time"].values, steps)
    assert "month" not in out.coords
    np.testing.assert_allclose(
        out[VAR].isel(lat=0, lon=0).values, _clim_value([3, 4, 3])
    )


def test_follow_months_drops_steps_the_climatology_lacks_and_says_how_many():
    clim = month_dim().isel(month=slice(0, 4))  # Jan..Apr only
    steps = np.array(["2011-03-05", "2011-08-10", "2011-04-20"], dtype="datetime64[ns]")
    out, dropped = A.follow_months(clim, steps)
    assert dropped == 1
    assert out.sizes["time"] == 2
    with pytest.raises(ValueError, match="none of the"):
        A.follow_months(clim, np.array(["2011-08-10"], dtype="datetime64[ns]"))


def test_follow_months_hands_back_a_dated_object_untouched():
    dated = _dated(TIMES, DELTA)
    out, dropped = A.follow_months(
        dated, np.array(["2011-03-05"], dtype="datetime64[ns]")
    )
    assert out is dated and dropped == 0


# -- times= bins: each bin's reference is that bin's month(s) -----------------------


def test_each_monthly_bin_is_scored_against_its_own_calendar_month(shape):
    out = compare(
        reference="c",
        test="t",
        variables=[VAR],
        times={"resample": "1MS", "reduce": "mean"},
    )
    assert len(out) == 24
    # test = climatology(month) + delta_i, so bias is delta_i only if the reference
    # really was that step's month; any other pairing leaves a seasonal remainder
    biases = [c.metrics()["bias"] for c in out]
    np.testing.assert_allclose(biases, DELTA, atol=1e-9)
    assert all(c.metrics()["rmse"] == pytest.approx(b) for c, b in zip(out, biases))


def test_an_explicit_list_of_dates_follows_too(shape):
    out = compare(
        reference="c", test="t", variables=[VAR], times=["2010-03", "2011-07"]
    )
    got = {c.label: c.metrics()["bias"] for c in out}
    assert got["2010-03"] == pytest.approx(DELTA[2])
    assert got["2011-07"] == pytest.approx(DELTA[18])


def test_a_bin_spanning_months_is_the_mean_of_them_weighted_by_test_steps(install):
    # Two steps in March, one in April, all in 2010: the year's reference is
    # (2 * march + april) / 3, not the 12-month mean and not the plain (march+april)/2.
    steps = pd.DatetimeIndex(["2010-03-05", "2010-03-20", "2010-04-10"])
    test = _dated(steps, np.zeros(3))
    meta = {**_meta(), "climatology": True, "climatology_period": "monthly"}
    install({"t": (test, _meta(steps)), "c": (month_dim(), meta)})
    (comp,) = compare(
        reference="c",
        test="t",
        variables=[VAR],
        times={"resample": "1YS", "reduce": "mean"},
    )
    expected = (2 * _clim_value(3) + _clim_value(4)) / 3
    np.testing.assert_allclose(comp.aligned["reference"].values, expected)
    assert comp.metrics()["bias"] == pytest.approx(
        np.mean(_clim_value([3, 3, 4])) - expected
    )


def test_a_whole_record_time_mean_averages_the_months_the_test_covers(install):
    # month-dim climatology + a shared {"time": "mean"}: until now the reference kept
    # twelve months and the comparison was refused; now it is the mean of the months the
    # test actually has (here March..May only, not the annual mean)
    steps = pd.DatetimeIndex(["2010-03-05", "2010-04-10", "2010-05-10"])
    test = _dated(steps, np.zeros(3))
    install({"t": (test, _meta(steps)), "c": (month_dim(), _meta())})
    (comp,) = compare(
        reference="c", test="t", variables=[VAR], aggregate={"time": "mean"}
    )
    np.testing.assert_allclose(
        comp.aligned["reference"].values, np.mean(_clim_value([3, 4, 5]))
    )
    assert comp.metrics()["bias"] == pytest.approx(0.0)


# -- a page that keeps the time axis ----------------------------------------------


def test_over_time_pairs_the_climatology_repeated_at_the_tests_months(shape):
    (comp,) = compare(reference="c", test="t", variables=[VAR], over="time")
    ref = comp.aligned["reference"]
    assert comp.aligned.sizes["time"] == 24
    np.testing.assert_allclose(
        ref.isel(lat=0, lon=0).values, _clim_value(TIMES.month), atol=1e-9
    )
    np.testing.assert_allclose(
        comp.aligned["difference"].isel(lat=0, lon=0).values, DELTA, atol=1e-9
    )
    # January 2010 and January 2011 carry the same climatological January
    assert ref.isel(time=0, lat=0, lon=0) == ref.isel(time=12, lat=0, lon=0)


def test_metrics_series_has_one_value_per_test_time(shape):
    out = compare(
        reference="c",
        test="t",
        variables=[VAR],
        times={"resample": "1MS", "reduce": "mean"},
    )
    rec = [c.metrics() for c in out]
    assert [r["time"] for r in rec] == [f"{t.year}-{t.month:02d}" for t in TIMES]
    fig = out.metrics_series(metrics=("bias",))
    try:
        assert fig is not None
    finally:
        plt.close("all")


# -- what must not change --------------------------------------------------------------


def test_an_explicit_month_select_is_left_exactly_alone(install, capsys):
    install({"t": (_dated(TIMES, DELTA), _meta(TIMES)), "c": (month_dim(), _meta())})
    (comp,) = compare(
        reference="c",
        test="t",
        variables=[VAR],
        select={"test": {"time": "2010-03"}, "reference": {"month": 7}},
    )
    # July's climatology against the test's March step: the caller's explicit pairing
    assert comp.metrics()["bias"] == pytest.approx(
        _clim_value(3) + DELTA[2] - _clim_value(7)
    )
    assert "monthly climatology" not in capsys.readouterr().out


def test_a_test_time_mean_against_a_month_mean_works_as_before(install, capsys):
    install({"t": (_dated(TIMES, DELTA), _meta(TIMES)), "c": (month_dim(), _meta())})
    (comp,) = compare(
        reference="c",
        test="t",
        variables=[VAR],
        aggregate={"test": {"time": "mean"}, "reference": {"month": "mean"}},
    )
    assert comp.metrics()["bias"] == pytest.approx(
        np.mean(_clim_value(TIMES.month) + DELTA)
        - np.mean(_clim_value(np.arange(1, 13)))
    )
    assert "monthly climatology" not in capsys.readouterr().out


def test_a_woa_like_time_aggregate_still_means_all_twelve_months(install, capsys):
    # the stack's own time reduced by a plain aggregate worked before (the annual mean)
    # and stays that: following only starts where the request used to be refused
    install({"t": (_dated(TIMES, DELTA), _meta(TIMES)), "c": (woa_like(), _meta())})
    (comp,) = compare(
        reference="c", test="t", variables=[VAR], aggregate={"time": "mean"}
    )
    np.testing.assert_allclose(
        comp.aligned["reference"].values, np.mean(_clim_value(np.arange(1, 13)))
    )
    assert "monthly climatology" not in capsys.readouterr().out


def test_a_dated_reference_does_not_follow_anything(install, capsys):
    install(
        {
            "t": (_dated(TIMES, DELTA), _meta(TIMES)),
            "r": (_dated(TIMES, 0 * DELTA), _meta(TIMES)),
        }
    )
    out = compare(
        reference="r",
        test="t",
        variables=[VAR],
        times={"resample": "1MS", "reduce": "mean"},
    )
    np.testing.assert_allclose([c.metrics()["bias"] for c in out], DELTA, atol=1e-9)
    assert "monthly climatology" not in capsys.readouterr().out


def test_two_climatologies_have_no_dates_to_follow(install, capsys):
    install({"t": (month_dim(), _meta()), "c": (month_dim(), _meta())})
    out = compare(
        reference="c",
        test="t",
        variables=[VAR],
        aggregate={"month": "mean"},
    )
    assert len(out) == 1
    assert "monthly climatology" not in capsys.readouterr().out


def test_over_month_is_not_followed(install, capsys):
    install({"t": (_dated(TIMES, DELTA), _meta(TIMES)), "c": (month_dim(), _meta())})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # whatever became of this request before, it still does
        with contextlib.suppress(ValueError, KeyError):
            compare(reference="c", test="t", variables=[VAR], over="month")
    assert "monthly climatology" not in capsys.readouterr().out


# -- the log line ----------------------------------------------------------------------


def test_the_fan_says_it_once_not_once_per_bin(shape, capsys):
    compare(
        reference="c",
        test="t",
        variables=[VAR],
        times={"resample": "1MS", "reduce": "mean"},
    )
    out = capsys.readouterr().out
    assert out.count("is a monthly climatology") == 1
    assert "calendar month" in out


def test_a_single_comparison_says_it_once(install, capsys):
    install({"t": (_dated(TIMES, DELTA), _meta(TIMES)), "c": (month_dim(), _meta())})
    compare(reference="c", test="t", variables=[VAR], over="time")
    assert capsys.readouterr().out.count("is a monthly climatology") == 1


def _one_comparison(install):
    install({"t": (_dated(TIMES, DELTA), _meta(TIMES)), "c": (month_dim(), _meta())})
    return next(iter(compare(reference="c", test="t", variables=[VAR], over="time")))


@pytest.mark.parametrize("error", [OSError("gone"), KeyError("c"), ValueError("bad")])
def test_a_read_failure_stops_the_following_with_one_warning(
    install, monkeypatch, error
):
    from ocean_skill import comparison as _comparison

    comp = _one_comparison(install)

    def boom(self):
        raise error

    monkeypatch.setattr(_comparison.Comparison, "_follow_steps_uncached", boom)
    comp._follow_cache = _comparison._UNSET
    comp._quiet = False
    with pytest.warns(UserWarning, match="could not check whether reference 'c'") as w:
        results = [comp._follow_steps(), comp._follow_steps()]  # memoized: one warning
    assert results == [None, None]
    assert len([x for x in w if "monthly climatology" in str(x.message)]) == 1


def test_a_programming_error_in_the_follow_check_is_not_swallowed(
    install, monkeypatch
):
    from ocean_skill import comparison as _comparison

    comp = _one_comparison(install)

    def bug(self):
        raise AttributeError("a defect, not a data problem")

    monkeypatch.setattr(_comparison.Comparison, "_follow_steps_uncached", bug)
    comp._follow_cache = _comparison._UNSET
    with pytest.raises(AttributeError, match="a defect"):
        comp._follow_steps()


# -- a section (depth x along): a slab through a 4-D climatology ------------------


def _four_d(with_time):
    """Build ``depth/100 + lat/10 + lon/1000`` plus the month's value."""
    lon = np.arange(180.0, 201.0, 10.0)
    lat = np.arange(-30.0, 31.0, 10.0)
    d, la, lo = np.meshgrid(DEPTHS, lat, lon, indexing="ij")
    base = d / 100 + la / 10 + lo / 1000
    attrs = {"depth": {"positive": "down", "units": "m"}}
    if with_time is None:
        stack = np.stack([base + _clim_value(m) for m in range(1, 13)])
        dims, coords = ("month", "depth", "lat", "lon"), {"month": np.arange(1, 13)}
    else:
        stack = np.stack([base + _clim_value(t.month) + 0.5 for t in with_time])
        dims, coords = ("time", "depth", "lat", "lon"), {"time": with_time}
    coords.update(depth=("depth", DEPTHS, attrs["depth"]), lat=lat, lon=lon)
    return xr.Dataset({VAR: (dims, stack, _attrs())}, coords=coords)


def test_a_climatology_slab_follows_the_months_of_a_dated_slab(install):
    steps = pd.DatetimeIndex(["2010-03-16", "2010-04-16", "2010-05-16"])
    box = {
        "lon": {"min": 180, "max": 200},
        "lat": {"min": -30, "max": 30},
        "depth": [0.0, 50.0, 200.0],
    }
    meta = {
        "variables": [VAR],
        "featureType": "grid",
        "standard_names": {VAR: VAR},
        "geospatial_lon_min": 180.0,
        "geospatial_lon_max": 200.0,
        "geospatial_lat_min": -30.0,
        "geospatial_lat_max": 30.0,
    }
    install(
        {
            "t": (
                _four_d(steps),
                {
                    **meta,
                    "time_coverage_start": "2010-03-16",
                    "time_coverage_end": "2010-05-16",
                },
            ),
            "c": (_four_d(None), meta),
        }
    )
    out = compare(
        reference="c",
        test="t",
        variables=[VAR],
        select=box,
        aggregate={"lon": "mean"},
        times={"resample": "1MS", "reduce": "mean"},
    )
    assert len(out) == 3
    # the test is the climatology plus 0.5 everywhere, in every month: a section of
    # exactly 0.5 only if each bin's reference was that bin's own month
    for comp in out:
        assert comp.is_section
        np.testing.assert_allclose(
            comp.aligned["difference"].values[
                np.isfinite(comp.aligned["difference"].values)
            ],
            0.5,
            atol=1e-9,
        )
        assert comp.metrics()["bias"] == pytest.approx(0.5)


# -- no shipped suite page changes ---------------------------------------------------
#
# Following only starts where a comparison could not be formed before, so no page of the
# suites that ship with the package may follow -- except model-comparison's boundary
# BGC strip pages, which are written to (ESPER against the MLD climatology's month): each either names its month itself
# (``reference: {month: ...}``), reduces the climatology's own axis, or has no
# climatology. This runs the PACMED suites (the ones that read climatologies) on the
# synthetic PACMED and fails if any comparison lays a reference out along the test's
# months.


@pytest.mark.slow
@pytest.mark.parametrize("name", PACMED_SUITES)
def test_no_page_of_a_shipped_suite_follows_a_climatology(
    localized,  # noqa: F811
    name,
    monkeypatch,
):
    from ocean_skill.comparison import Comparison
    from ocean_skill.suites.run import run_suite

    followed: list[tuple[str, str]] = []
    real = Comparison._follow_steps

    def spy(self):
        steps = real(self)
        if steps is not None:
            followed.append((self.test_name, self.reference_name))
        return steps

    monkeypatch.setattr(Comparison, "_follow_steps", spy)
    run_suite(localized(name))
    # model-comparison follows on purpose, and only there: the ESPER boundary strip
    # against the MLD-interpolated climatology for the same calendar month.
    expected = (
        {("pyesper_bry_bgc_strip", "mld_bry_bgc_strip")}
        if name == "model-comparison"
        else set()
    )
    assert set(followed) == expected, (
        f"{name}: references that follow the test: {sorted(set(followed))}"
    )
