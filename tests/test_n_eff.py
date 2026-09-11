"""Effective-sample-size weighting (``n_eff``): the AR(1) math, per-feature-type
attachment on :meth:`~ocean_skill.comparison.Comparison.metrics`, and the
default-on-with-a-warning behaviour of ``summary_weights=``/``weights=`` on the
Taylor/Target diagrams and :func:`~ocean_skill.plot.map_metrics.interpolate_records`.

Ported from a CIOFS-project helper (``omsa_cache.py``'s ``_effective_n``/
``_lag1_autocorr_gap_aware``/``_lag1_autocorr_along_index``) so a mixed-feature-type
pool — a long, autocorrelated mooring record next to a handful of independent CTD
casts, say — weights fairly by default rather than by raw sample count alone.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from ocean_skill import metrics as m
from ocean_skill.comparison import Comparison
from ocean_skill.plot import _weighting
from ocean_skill.plot.summary import target, taylor

TEMPERATURE = "sea_water_temperature"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# -- the AR(1) math (ocean_skill.metrics) --------------------------------------------


def test_effective_n_is_unchanged_at_zero_autocorrelation():
    assert m.effective_n(100, 0.0) == pytest.approx(100.0)


def test_effective_n_falls_back_to_n_when_r1_is_unavailable():
    assert m.effective_n(100, None) == pytest.approx(100.0)


def test_effective_n_is_floored_at_one_not_zero_as_r1_approaches_one():
    # r1 is clipped to 0.99 before the formula, so the factor never reaches 0 --
    # but for a small enough n, n * factor still rounds under 1 and is floored there.
    assert m.effective_n(2, 0.999) == pytest.approx(1.0)


def test_effective_n_of_zero_samples_is_zero():
    assert m.effective_n(0, 0.5) == 0.0


def test_effective_n_never_exceeds_n_for_negative_autocorrelation():
    # A negative r1 inflates the raw factor past 1 (alternating series carry *more*
    # independent evidence than their count alone suggests) -- still clipped at n,
    # since n_eff can't exceed the samples that exist.
    assert m.effective_n(50, -0.9) == pytest.approx(50.0)


def test_effective_n_2d_matches_1d_on_one_flat_axis():
    # No deflation on the y axis (r_y=0) reduces exactly to the 1-D formula on x.
    assert m.effective_n_2d(1000, 0.9, 0.0) == pytest.approx(m.effective_n(1000, 0.9))


def test_effective_n_2d_compounds_both_axes():
    n_eff = m.effective_n_2d(1000, 0.9, 0.9)
    assert 1.0 <= n_eff < m.effective_n(1000, 0.9)


def test_effective_n_2d_falls_back_to_n_with_no_estimate_on_either_axis():
    assert m.effective_n_2d(1000, None, None) == pytest.approx(1000.0)


def test_lag1_autocorr_along_index_recovers_a_known_ar1_coefficient():
    rng = np.random.default_rng(0)
    n, phi = 4000, 0.8
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = phi * v[i - 1] + rng.normal(scale=0.1)
    assert m.lag1_autocorr_along_index(v) == pytest.approx(phi, abs=0.05)


def test_lag1_autocorr_along_index_is_near_zero_for_white_noise():
    rng = np.random.default_rng(1)
    assert abs(m.lag1_autocorr_along_index(rng.normal(size=2000))) < 0.1


@pytest.mark.parametrize("values", [np.ones(100), np.array([1.0, 2.0, 3.0])])
def test_lag1_autocorr_along_index_is_none_when_it_cannot_be_estimated(values):
    # A constant series (undefined correlation) and fewer than 4 finite points both
    # have nothing to estimate from.
    assert m.lag1_autocorr_along_index(values) is None


def test_lag1_autocorr_ignores_an_artificial_gap():
    """The gap-aware estimator should barely move when a chunk of regularly-spaced
    time is simply missing -- a naive lag-1 on the compacted values would instead
    treat the jump across the gap as one more "adjacent" step and bias the estimate.
    """
    rng = np.random.default_rng(2)
    n, phi = 1000, 0.9
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = phi * v[i - 1] + rng.normal(scale=0.1)
    times = np.array("2020-01-01", dtype="datetime64[h]") + np.arange(n)
    good = np.ones(n, dtype=bool)
    r1_full = m.lag1_autocorr(times, v, good)

    gappy_times = np.concatenate([times[:400], times[600:]])
    gappy_values = np.concatenate([v[:400], v[600:]])
    r1_gappy = m.lag1_autocorr(gappy_times, gappy_values, np.ones(len(gappy_times), dtype=bool))
    assert r1_gappy == pytest.approx(r1_full, abs=0.02)


def test_lag1_autocorr_is_none_for_a_constant_series():
    times = np.array("2020-01-01", dtype="datetime64[h]") + np.arange(100)
    assert m.lag1_autocorr(times, np.ones(100), np.ones(100, dtype=bool)) is None


# -- Comparison._n_eff: attachment per feature type ----------------------------------


def _ar1(n, phi, seed, scale=0.1):
    rng = np.random.default_rng(seed)
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = phi * v[i - 1] + rng.normal(scale=scale)
    return v


def _series_comparison(n=800, phi=0.9, seed=10):
    times = np.array("2020-01-01", dtype="datetime64[h]") + np.arange(n)
    ref = _ar1(n, phi, seed) + 20.0
    test = ref + np.random.default_rng(seed + 1).normal(scale=0.05, size=n)
    reference = xr.DataArray(ref, dims=("time",), coords={"time": times}).assign_coords(
        lon=-158.0, lat=22.75, depth=1.0
    )
    reference.attrs["units"] = "degC"
    testda = xr.DataArray(test, dims=("time",), coords={"time": times})
    aligned = xr.Dataset(
        {
            "test": testda.rename("test"),
            "reference": reference.rename("reference"),
            "difference": (testda - reference).rename("difference"),
        },
        attrs={
            "scored_over": "time",
            "match_method": "nearest",
            "station_lon": -158.0,
            "station_lat": 22.75,
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(reference="a", test="b", variable=TEMPERATURE, over="time")
    c._aligned = aligned
    return c


def _profile_comparison_with(n=40, phi=0.85, seed=20):
    depth = np.linspace(5.0, 500.0, n)
    ref = _ar1(n, phi, seed) + 20.0
    test = ref + np.random.default_rng(seed + 1).normal(scale=0.05, size=n)
    reference = xr.DataArray(ref, dims=("DEPTH",), coords={"DEPTH": depth}).assign_coords(
        lon=-158.0, lat=22.75
    )
    reference.attrs["units"] = "degC"
    testda = xr.DataArray(test, dims=("DEPTH",), coords={"DEPTH": depth})
    aligned = xr.Dataset(
        {
            "test": testda.rename("test"),
            "reference": reference.rename("reference"),
            "difference": (testda - reference).rename("difference"),
        },
        attrs={
            "scored_over": "DEPTH",
            "match_method": "interp",
            "station_lon": -158.0,
            "station_lat": 22.75,
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(reference="a", test="b", variable=TEMPERATURE, over="Z")
    c._aligned = aligned
    return c


def test_metrics_emits_n_eff_for_a_timeseries_and_deflates_it_below_n():
    c = _series_comparison(n=800, phi=0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = c.metrics()
    assert c.is_series
    assert "n_eff" in rec
    assert 1.0 <= rec["n_eff"] <= rec["n"]
    assert rec["n_eff"] < 0.5 * rec["n"]  # strongly autocorrelated -> strongly deflated


def test_metrics_n_eff_is_close_to_n_for_an_uncorrelated_timeseries():
    c = _series_comparison(n=800, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = c.metrics()
    assert rec["n_eff"] > 0.85 * rec["n"]


def test_metrics_emits_n_eff_for_a_profile():
    c = _profile_comparison_with(n=40, phi=0.85)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = c.metrics()
    assert c.is_profile
    assert "n_eff" in rec
    assert 1.0 <= rec["n_eff"] <= rec["n"]


def test_metrics_n_eff_for_time_depth_uses_time_axis_averaged_over_levels():
    nt, nz = 80, 8
    times = np.array("2020-01-01", dtype="datetime64[D]") + np.arange(nt)
    ref = np.zeros((nt, nz))
    for z in range(nz):
        ref[:, z] = _ar1(nt, 0.8, seed=30 + z)
    ref += 20.0
    test = ref + np.random.default_rng(99).normal(scale=0.05, size=(nt, nz))
    reference = xr.DataArray(ref, dims=("time", "z"), coords={"time": times}).assign_coords(
        lon=-158.0, lat=22.75
    )
    reference.attrs["units"] = "degC"
    testda = xr.DataArray(test, dims=("time", "z"), coords={"time": times})
    aligned = xr.Dataset(
        {
            "test": testda.rename("test"),
            "reference": reference.rename("reference"),
            "difference": (testda - reference).rename("difference"),
        },
        attrs={"station_lon": -158.0, "station_lat": 22.75},
    )
    from ocean_skill.align import TIME_DEPTH_OVER

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="a", test="b", variable=TEMPERATURE, over=TIME_DEPTH_OVER
        )
    c._aligned = aligned
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = c.metrics()
    assert c.is_time_depth
    assert "n_eff" in rec
    assert 1.0 <= rec["n_eff"] <= rec["n"]
    assert rec["n_eff"] < 0.6 * rec["n"]


def test_metrics_n_eff_for_a_gridded_field_is_spatial_and_bounded():
    from scipy.ndimage import uniform_filter

    ny, nx = 24, 30
    rng = np.random.default_rng(40)
    lon = np.linspace(200, 210, nx)
    lat = np.linspace(20, 30, ny)
    ref = uniform_filter(rng.normal(size=(ny, nx)), size=5) + 20.0
    test = ref + rng.normal(scale=0.05, size=(ny, nx))
    reference = xr.DataArray(ref, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    reference.attrs["units"] = "degC"
    testda = xr.DataArray(test, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    aligned = xr.Dataset(
        {
            "test": testda.rename("test"),
            "reference": reference.rename("reference"),
            "difference": (testda - reference).rename("difference"),
        },
        attrs={"match_method": "conservative_normed"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(reference="a", test="b", variable=TEMPERATURE)
    c._aligned = aligned
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = c.metrics()
    assert c.family == "field_row"
    assert rec["weighted"] is True  # cos(lat) area weighting is untouched
    assert "n_eff" in rec
    assert 1.0 <= rec["n_eff"] <= rec["n"]
    assert rec["n_eff"] < 0.5 * rec["n"]  # spatially smoothed -> strongly deflated


def test_metrics_n_eff_for_a_white_noise_grid_is_close_to_n():
    ny, nx = 24, 30
    rng = np.random.default_rng(41)
    lon = np.linspace(200, 210, nx)
    lat = np.linspace(20, 30, ny)
    ref = rng.normal(size=(ny, nx)) + 20.0
    test = ref + rng.normal(scale=0.05, size=(ny, nx))
    reference = xr.DataArray(ref, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    reference.attrs["units"] = "degC"
    testda = xr.DataArray(test, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    aligned = xr.Dataset(
        {
            "test": testda.rename("test"),
            "reference": reference.rename("reference"),
            "difference": (testda - reference).rename("difference"),
        },
        attrs={"match_method": "conservative_normed"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(reference="a", test="b", variable=TEMPERATURE)
    c._aligned = aligned
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = c.metrics()
    assert rec["n_eff"] > 0.9 * rec["n"]


def test_n_eff_is_not_computed_for_a_section():
    """Sections are deliberately left to the CIOFS-specific transect-repartitioning
    recipe (see the module docstring's Context) -- ``_n_eff`` dispatches on the same
    boolean properties :meth:`Comparison.metrics` already branches on, so a stub
    exercising only that dispatch is enough; no real section pipeline needed.
    """
    stub = SimpleNamespace(
        is_series=False,
        is_profile=False,
        is_time_depth=False,
        is_section=True,
        _metrics={"n": 100},
    )
    aligned = xr.Dataset({"reference": ("along", np.arange(10.0))})
    assert Comparison._n_eff(stub, aligned) is None


def test_n_eff_is_none_when_n_is_zero_or_missing():
    stub = SimpleNamespace(
        is_series=True, is_profile=False, is_time_depth=False, is_section=False,
        _metrics={"n": 0},
    )
    aligned = xr.Dataset({"reference": ("time", np.arange(10.0))})
    assert Comparison._n_eff(stub, aligned) is None


# -- default-on weighting: taylor()/target() (static) --------------------------------


class _FakeComparison:
    """Bare metrics-record stand-in -- see ``tests/test_summary_normalize.py``'s
    identical pattern. Isolates the diagrams' default-weighting behaviour from how
    ``n_eff`` actually gets computed (covered above).
    """

    def __init__(self, *, label, n_eff=None, corr=0.9, std_test=1.0, std_reference=1.0,
                 bias=0.1, crmsd=0.2, n=100):
        self.label = label
        self.units = "degC"
        self._record = {
            "corr": corr, "std_test": std_test, "std_reference": std_reference,
            "bias": bias, "crmsd": crmsd, "variable": "sea_water_temperature",
            "reference": label, "n": n,
        }
        if n_eff is not None:
            self._record["n_eff"] = n_eff

    def metrics(self):
        return self._record


def _mixed_set():
    # A "long mooring" (large n_eff) and a "short cast" (small n_eff) -- the exact
    # imbalance n_eff exists to correct for in a pooled summary star.
    return [
        _FakeComparison(label="mooring", n_eff=500.0, std_test=1.2, corr=0.8),
        _FakeComparison(label="cast", n_eff=5.0, std_test=0.6, corr=0.95),
    ]


@pytest.mark.parametrize("diagram", [taylor, target])
def test_default_summary_weights_warns_and_weights_by_n_eff(diagram):
    with pytest.warns(UserWarning, match="effective sample size 'n_eff'"):
        diagram(_mixed_set(), summary_points=True)


@pytest.mark.parametrize("diagram", [taylor, target])
def test_explicit_none_opts_out_silently(diagram):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagram(_mixed_set(), summary_points=True, summary_weights=None)
    assert not any("effective sample size" in str(x.message) for x in caught)


@pytest.mark.parametrize("diagram", [taylor, target])
def test_default_matches_explicit_n_eff(diagram):
    """The AUTO default should select exactly ``n_eff`` -- proven by comparing the
    resolved weights field each takes, via the shared resolver both go through.
    """
    recs = [c.metrics() for c in _mixed_set()]
    default = _weighting.resolve(
        any("n_eff" in r for r in recs), _weighting.AUTO, param_name="summary_weights",
        warn=False,
    )
    explicit = _weighting.resolve(
        any("n_eff" in r for r in recs), "n_eff", param_name="summary_weights",
    )
    assert default == explicit == "n_eff"


def test_no_warning_when_no_record_carries_n_eff():
    """Back-compat: a set with no n_eff at all (hand-built records, or predating
    this feature) draws exactly as before -- unweighted, silently.
    """
    plain = [_FakeComparison(label="a"), _FakeComparison(label="b")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        taylor(plain, summary_points=True)
    assert not any("effective sample size" in str(x.message) for x in caught)


def test_interactive_target_matches_the_static_default():
    from ocean_skill.plot.holoviews_renderer import _target

    items = [
        {"label": c.label, "metrics": c.metrics(), "units": c.units}
        for c in _mixed_set()
    ]
    with pytest.warns(UserWarning, match="effective sample size 'n_eff'"):
        _target(items, summary_points=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _target(items, summary_points=True, summary_weights=None)
    assert not any("effective sample size" in str(x.message) for x in caught)


# -- default-on weighting: map_metrics / interpolate_records --------------------------


def _station_records():
    return [
        {"lon": -150.0, "lat": 58.0, "corr": 0.8, "bias": 0.1, "crmsd": 0.3,
         "sigma_ratio": 1.0, "n_eff": 500.0},
        {"lon": -149.5, "lat": 58.2, "corr": 0.9, "bias": 0.05, "crmsd": 0.2,
         "sigma_ratio": 1.1, "n_eff": 20.0},
        {"lon": -149.0, "lat": 58.5, "corr": 0.7, "bias": 0.2, "crmsd": 0.4,
         "sigma_ratio": 0.9, "n_eff": 300.0},
        {"lon": -148.5, "lat": 58.8, "corr": 0.85, "bias": 0.15, "crmsd": 0.25,
         "sigma_ratio": 1.05, "n_eff": 15.0},
        {"lon": -150.2, "lat": 58.9, "corr": 0.75, "bias": 0.12, "crmsd": 0.35,
         "sigma_ratio": 0.95, "n_eff": 250.0},
    ]


def test_map_metrics_default_weights_and_warns_on_spline():
    from ocean_skill.plot.map_metrics import interpolate_records

    with pytest.warns(UserWarning, match="effective sample size 'n_eff'"):
        interpolate_records(_station_records(), method="spline")


def test_map_metrics_default_is_silent_when_method_ignores_weights():
    """knn with no block_spacing never uses weights -- AUTO should neither turn one
    on nor trip the separate "would have no effect" warning that an *explicit*
    weights= would (see the next test).
    """
    from ocean_skill.plot.map_metrics import interpolate_records

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        interpolate_records(_station_records(), method="knn")
    messages = [str(x.message) for x in caught]
    assert not any("effective sample size" in m for m in messages)
    assert not any("no effect" in m for m in messages)


def test_map_metrics_explicit_weights_with_a_method_that_ignores_them_still_warns():
    from ocean_skill.plot.map_metrics import interpolate_records

    with pytest.warns(UserWarning, match="no effect"):
        interpolate_records(_station_records(), method="knn", weights="n_eff")


def test_map_metrics_default_warns_with_block_spacing_even_off_spline():
    from ocean_skill.plot.map_metrics import interpolate_records

    with pytest.warns(UserWarning, match="effective sample size 'n_eff'"):
        interpolate_records(_station_records(), method="knn", block_spacing=5_000)


def test_map_metrics_weights_none_opts_out_silently():
    from ocean_skill.plot.map_metrics import interpolate_records

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        interpolate_records(_station_records(), method="spline", weights=None)
    assert not any("effective sample size" in str(x.message) for x in caught)


def test_map_metrics_weighted_surface_differs_from_unweighted():
    """A weighted least-squares fit should land on genuinely different spline
    coefficients than an unweighted one -- not merely tolerance-close, since a
    tolerant comparison (``np.allclose``) would pass even if weighting silently
    did nothing.
    """
    from ocean_skill.plot.map_metrics import interpolate_records

    weighted = interpolate_records(_station_records(), method="spline")
    unweighted = interpolate_records(_station_records(), method="spline", weights=None)
    assert not np.array_equal(weighted["corr"].values, unweighted["corr"].values)
