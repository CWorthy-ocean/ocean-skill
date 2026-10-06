"""Tidal amplitude/phase calculators (ocean_skill.tides)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

pyfes = pytest.importorskip("pyfes")

from ocean_skill import tides  # noqa: E402  (registers the calculators)
from ocean_skill.operators import resolve_variable  # noqa: E402

AMP_TYPE = tides.TIDAL_AMPLITUDE
WAVES = ["M2", "K1", "O1", "S2"]
RNG = np.random.default_rng(0)


def _ssh(amps, *, days=300, shape=(3, 4), mean=0.7, land=True, source=None):
    """Hourly SSH built from pyFES's own f, vu: h = mean + sum f A cos(vu - G).

    ``amps`` maps constituent -> amplitude scale; A and G vary over the grid.
    Returns (dataset, {name: (A, G)}).
    """
    times = np.datetime64("2020-01-01T00:00", "us") + np.arange(
        days * 24
    ) * np.timedelta64(1, "h")
    wt = pyfes.wave_table_factory(pyfes.DARWIN, list(amps))
    f, vu = wt.compute_nodal_modulations(times)
    truth, h = {}, np.full((*shape, times.size), mean)
    for k, name in enumerate(wt.constituents):
        a = amps[name] * (1 + RNG.random(shape))
        g = RNG.random(shape) * 360
        truth[name] = (a, g)
        h += a[..., None] * f[k] * np.cos(vu[k] - np.radians(g)[..., None])
    if land:
        h[0, 0] = np.nan
    ds = xr.Dataset({"zeta": (("eta", "xi", "time"), h)}, coords={"time": times})
    ds["zeta"] = ds["zeta"].transpose("time", "eta", "xi")
    ds["zeta"].attrs["standard_name"] = tides._SSH
    ds = ds.rename(zeta=tides._SSH)
    if source:
        ds.attrs["ocean_skill_source"] = source
    return ds, truth


def _lag_diff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def test_harmonic_constants_recover_amplitude_and_phase():
    ds, truth = _ssh({n: 0.1 for n in WAVES})
    res = tides.harmonic_constants(ds, constituents=WAVES)
    assert sorted(res.constituent.values) == sorted(WAVES)
    for name, (a, g) in truth.items():
        got = res.sel(constituent=name)
        np.testing.assert_allclose(got.amplitude.values[1:], a[1:], atol=1e-6)
        assert _lag_diff(got.phase.values, g)[1:].max() < 1e-4
    assert res.amplitude.isel(eta=0, xi=0).isnull().all()  # land


def test_k1_not_contaminated_by_o1():
    ds, truth = _ssh({"O1": 1.0, "K1": 0.02, "M2": 0.3, "S2": 0.1}, land=False)
    res = tides.harmonic_constants(ds, constituents=WAVES)
    np.testing.assert_allclose(
        res.sel(constituent="K1").amplitude.values, truth["K1"][0], atol=1e-6
    )


def test_calculator_returns_2d_map_with_attrs():
    ds, truth = _ssh({n: 0.1 for n in WAVES})
    amp = resolve_variable(ds, {"calculate": "tidal_amplitude", "constituent": "k1"})
    assert amp.dims == ("eta", "xi") and "constituent" not in amp.coords
    assert amp.attrs["standard_name"] == AMP_TYPE
    assert amp.attrs["units"] == "m" and amp.attrs["constituent"] == "K1"
    assert amp.attrs["long_name"] == "K1 tidal amplitude"
    np.testing.assert_allclose(amp.values[1:], truth["K1"][0][1:], atol=1e-6)
    ph = resolve_variable(ds, {"calculate": "tidal_phase", "constituent": "K1"})
    assert (
        ph.attrs["standard_name"] == tides.TIDAL_PHASE and ph.attrs["units"] == "degree"
    )
    assert ph.min() >= 0 and ph.max() < 360


def _forcing(labels, re, im, **attrs):
    dims = ("ntides", "eta_rho", "xi_rho")
    ds = xr.Dataset(
        {
            tides.TIDAL_RE: (dims, re, attrs),
            tides.TIDAL_IM: (dims, im, attrs),
        },
        coords={"ntides": labels},
    )
    return ds


def test_forcing_file_pair_lowercase_labels_and_zero_fill():
    re = np.zeros((2, 2, 3))
    im = np.zeros((2, 2, 3))
    re[1, 1:, :], im[1, 1:, :] = 3.0, -4.0  # k1: amplitude 5, phase atan2(4, 3)
    re[0], im[0] = 1.0, 1.0
    ds = _forcing(["m2", "k1"], re, im, units="m")
    with pytest.warns(UserWarning, match="nodal factor"):
        amp = resolve_variable(
            ds, {"calculate": "tidal_amplitude", "constituent": "K1"}
        )
    with pytest.warns(UserWarning, match="nodal factor"):
        ph = resolve_variable(ds, {"calculate": "tidal_phase", "constituent": "K1"})
    assert amp.dims == ("eta_rho", "xi_rho")
    np.testing.assert_allclose(amp.values[1:], 5.0)
    np.testing.assert_allclose(ph.values[1:], np.degrees(np.arctan2(4, 3)))
    assert np.isnan(amp.values[0]).all() and np.isnan(ph.values[0]).all()  # zero fill


def test_atlas_pair_byte_labels_integer_mm():
    dims = ("con", "lat", "lon")
    re = np.zeros((2, 2, 2), dtype="int32")
    im = np.zeros((2, 2, 2), dtype="int32")
    re[0], im[0] = 300, 400  # m2, mm
    re[0, 0, 0] = im[0, 0, 0] = 0
    ds = xr.Dataset(
        {
            tides.TIDAL_RE: (dims, re, {"units": "mm"}),
            tides.TIDAL_IM: (dims, im, {"units": "mm"}),
        },
        coords={"con": np.array([b"m2  ", b"k1  "]), "lat": [0.0, 1], "lon": [0.0, 1]},
    )
    with warnings.catch_warnings():
        # a raw atlas has no forcing start date folded in -- no nodal-factor caveat
        warnings.filterwarnings("error", message=".*nodal factor")
        amp = resolve_variable(
            ds, {"calculate": "tidal_amplitude", "constituent": "M2"}
        )
    ph = resolve_variable(ds, {"calculate": "tidal_phase", "constituent": "M2"})
    assert amp.values[1, 1] == pytest.approx(0.5)  # metres
    assert ph.values[1, 1] == pytest.approx(np.degrees(np.arctan2(-400, 300)) % 360)
    assert np.isnan(amp.values[0, 0])
    with pytest.raises(ValueError, match="not in this source"):
        resolve_variable(ds, {"calculate": "tidal_amplitude", "constituent": "S2"})


def test_partial_nan_cell_is_nan_with_one_summary_warning():
    ds, _ = _ssh({n: 0.1 for n in WAVES}, land=False)
    ds[tides._SSH][5, 1, 2] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = tides.harmonic_constants(ds, constituents=WAVES)
    summary = [w for w in caught if "some NaN" in str(w.message)]
    assert len(summary) == 1 and "1 of 12" in str(summary[0].message)
    assert res.amplitude.isel(eta=1, xi=2).isnull().all()
    assert res.amplitude.notnull().sum() == 11 * 4


def test_rayleigh_drops_unresolvable_constituents():
    ds, _ = _ssh({n: 0.1 for n in WAVES}, days=20, land=False)
    with pytest.warns(UserWarning, match=r"cannot separate .*P1"):
        res = tides.harmonic_constants(ds)  # default set includes P1
    assert "P1" not in res.constituent and "K1" in res.constituent
    with pytest.warns(UserWarning, match="unreliable"):
        res = tides.harmonic_constants(ds, keep=["P1"])
    assert "P1" in res.constituent


def test_one_analysis_serves_two_constituents(monkeypatch):
    ds, _ = _ssh({n: 0.1 for n in WAVES}, source="x")
    calls = []
    real = tides._fit_cell
    monkeypatch.setattr(
        tides, "_fit_cell", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    k1 = resolve_variable(ds, {"calculate": "tidal_amplitude", "constituent": "K1"})
    n = len(calls)
    assert n > 0
    m2 = resolve_variable(ds, {"calculate": "tidal_amplitude", "constituent": "M2"})
    assert len(calls) == n  # second constituent came from the cache
    assert k1.attrs["constituent"] == "K1" and m2.attrs["constituent"] == "M2"
    assert not np.allclose(k1.values[1:], m2.values[1:])


def test_dask_backed_input_matches_numpy():
    pytest.importorskip("dask")
    ds, _ = _ssh({n: 0.1 for n in WAVES})
    lazy = ds.chunk({"time": 500, "eta": 1})
    a = tides.harmonic_constants(ds, constituents=WAVES, stride=1, time_stride=2)
    b = tides.harmonic_constants(lazy, constituents=WAVES, stride=1, time_stride=2)
    xr.testing.assert_allclose(a, b)


def test_calculators_registered_with_constituent_fan():
    from ocean_skill import operators

    spec = {"calculate": "tidal_phase", "constituent": ["K1", "M2"]}
    assert [s["constituent"] for s in operators.expand_calculator_fans(spec)] == [
        "K1",
        "M2",
    ]
    assert operators.CALCULATOR_INPUTS["tidal_amplitude"]({})[0] == [
        tides.TIDAL_RE,
        tides.TIDAL_IM,
    ]
