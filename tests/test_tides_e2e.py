"""End to end: tidal amplitude compared against a forcing file, and the flux maps.

The two public calls the PACMED Fig. 11 variables need -- ``osk.compare`` of a
model's hourly SSH against its tidal forcing file, and ``osk.field`` of the
model's baroclinic pressure flux -- through catalog lookup, preparation, the
constituent fan, the calculated cache, and both renderers.
"""

from __future__ import annotations

from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import tides
from ocean_skill.internal_tides import X_FLUX, Y_FLUX
from ocean_skill.tides import TIDAL_IM, TIDAL_RE

pyfes = pytest.importorskip("pyfes")
matplotlib.use("Agg")

AMP = {"M2": 0.8, "K1": 0.3, "O1": 0.2, "S2": 0.1}
PHASE = {"M2": 40.0, "K1": 120.0, "O1": 200.0, "S2": 300.0}
NY, NX = 3, 4
GRID = {
    "lon": (("eta_rho", "xi_rho"), -30 + np.tile(np.arange(NX, dtype=float), (NY, 1))),
    "lat": (
        ("eta_rho", "xi_rho"),
        50 + np.tile(np.arange(NY, dtype=float)[:, None], (1, NX)),
    ),
}


def _model():
    times = pd.date_range("2000-01-01", periods=24 * 200, freq="h")
    wt = pyfes.wave_table_factory(pyfes.DARWIN, list(AMP))
    f, vu = wt.compute_nodal_modulations(times.values.astype("datetime64[us]"))
    series = 0.05 + sum(
        AMP[n] * f[i] * np.cos(vu[i] - np.deg2rad(PHASE[n]))
        for i, n in enumerate(wt.constituents)
    )
    zeta = np.broadcast_to(series[:, None, None], (times.size, NY, NX)).copy()
    up = np.ones((times.size, NY, NX - 1)) * 2.0
    vp = np.ones((times.size, NY - 1, NX)) * 3.0
    return xr.Dataset(
        {
            "sea_surface_height_above_geoid": (
                ("time", "eta_rho", "xi_rho"),
                zeta,
                {"units": "m"},
            ),
            X_FLUX: (("time", "eta_rho", "xi_u"), up, {"units": "m4 s-3"}),
            Y_FLUX: (("time", "eta_v", "xi_rho"), vp, {"units": "m4 s-3"}),
        },
        coords={
            "time": times,
            "angle": (("eta_rho", "xi_rho"), np.zeros((NY, NX))),
            **GRID,
        },
    )


def _forcing():
    names = list(AMP)
    a = np.array([AMP[n] for n in names])[:, None, None] * np.ones((1, NY, NX))
    g = np.deg2rad([PHASE[n] for n in names])[:, None, None]
    return xr.Dataset(
        {
            TIDAL_RE: (("ntides", "eta_rho", "xi_rho"), a * np.cos(g), {"units": "m"}),
            TIDAL_IM: (("ntides", "eta_rho", "xi_rho"), -a * np.sin(g), {"units": "m"}),
        },
        coords={"ntides": [n.lower() for n in names], **GRID},
    )


@pytest.fixture
def sources(monkeypatch):
    data = {"model": _model(), "frc": _forcing()}
    meta = {
        "model": {
            "featureType": "grid",
            "variables": ["sea_surface_height_above_geoid", X_FLUX, Y_FLUX],
        },
        "frc": {"featureType": "grid", "variables": [TIDAL_RE, TIDAL_IM]},
    }
    monkeypatch.setattr(
        "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=meta[n])
    )
    monkeypatch.setattr("ocean_skill.read", lambda n, **kw: data[n])
    return data


def test_compare_amplitudes_against_forcing_in_both_renderers(sources, monkeypatch):
    fits = []
    real = tides._fit_cell
    monkeypatch.setattr(
        tides, "_fit_cell", lambda h, **k: fits.append(1) or real(h, **k)
    )
    spec = {"calculate": "tidal_amplitude", "constituent": ["K1", "M2"]}
    with pytest.warns(UserWarning):  # the forcing file's nodal-factor caveat
        cs = osk.compare(test="model", reference="frc", variables=[spec])
    assert len(cs) == 2
    for comp, name in zip(cs, ["K1", "M2"], strict=True):
        aligned = comp.align()
        np.testing.assert_allclose(aligned["test"], AMP[name], atol=1e-3)
        np.testing.assert_allclose(aligned["reference"], AMP[name], atol=1e-6)
    cs.plot(renderer="matplotlib")
    cs.plot(renderer="holoviews")
    # K1 and M2 come from one analysis (one fit per cell); the rest read the cache.
    assert len(fits) == NY * NX


def test_flux_field_in_both_renderers(sources):
    fs = osk.field(
        "model",
        [
            {
                "calculate": "baroclinic_pressure_flux",
                "component": "eastward",
                "units": "kW m-1",
            },
            {
                "calculate": "baroclinic_pressure_flux",
                "component": "northward",
                "units": "kW m-1",
            },
        ],
    )
    east, north = (f.prepare()[0] for f in fs.fields)
    np.testing.assert_allclose(east, 2.0 * 1027.0 / 1e3)
    np.testing.assert_allclose(north, 3.0 * 1027.0 / 1e3)
    fs.plot(renderer="matplotlib")
    fs.plot(renderer="holoviews")
