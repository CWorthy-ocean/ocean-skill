"""``_materialize_point_column``: collapse a point lane's vertical transform.

The reported failure: an ADCP mooring sampled continuously against a ROMS
history file chunked one time step per chunk. With no ``aggregate=``, every
kept model step feeds the aligned comparison directly, so
:func:`ocean_skill.roms.to_depth`'s xgcm transform ran once per (tiny) time
chunk -- thousands of Python-level ``apply_ufunc`` calls for a multi-month
deployment, each repaying its own interpolator setup. A CTD's dozen sparse
casts never showed this: pruning to cast-nearest steps already left too few
chunks for the per-chunk overhead to matter.

Built from the same synthetic ROMS-shaped dataset ``test_roms_chunking.py``
uses, chunked one step per time here to reproduce the actual layout.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
import ocean_skill.roms as roms
from ocean_skill import catalog, comparison
from ocean_skill.comparison import prepare_source

META = {"model": "roms", "vertical": {"hc": 20.0, "s_dim": "s_rho"}}


def _roms_timeseries(nt=200, ns=12, ny=6, nx=8) -> xr.Dataset:
    """A ROMS-shaped tracer, chunked one time step per chunk like a real history file."""
    sigma = np.linspace(-1.0, 0.0, ns)
    rng = np.random.default_rng(0)
    h = 50.0 + 150.0 * rng.random((ny, nx))
    ds = xr.Dataset(
        {
            "temp": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                rng.normal(15.0, 2.0, (nt, ns, ny, nx)),
            ),
            "h": (("eta_rho", "xi_rho"), h),
            "sigma_r": ("s_rho", sigma),
            "Cs_r": ("s_rho", sigma),
        },
        coords={
            "time": np.arange(nt),
            "lon": (("eta_rho", "xi_rho"), np.tile(np.linspace(-98, -80, nx), (ny, 1))),
            "lat": (
                ("eta_rho", "xi_rho"),
                np.tile(np.linspace(18, 31, ny)[:, None], (1, nx)),
            ),
        },
    )
    return ds.chunk({"time": 1, "s_rho": ns})


# Squarely inside the synthetic grid, and exactly on a grid point so
# `_is_point_bbox` recognizes the degenerate bbox below.
_POINT_LON, _POINT_LAT = -90.0, 24.0


def _resolve_stub(monkeypatch):
    ds = _roms_timeseries()
    monkeypatch.setattr(osk, "read", lambda name, **kw: ds)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata=META))
    return ds


def test_a_point_lane_is_materialized_before_the_transform(monkeypatch):
    """The transform's own input is numpy for a point lane -- zero xgcm tasks."""
    _resolve_stub(monkeypatch)
    captured = {}
    real_to_depth = roms.to_depth

    def spy(sub, meta, targets):
        captured["chunks"] = sub["temp"].chunks
        return real_to_depth(sub, meta, targets)

    monkeypatch.setattr(roms, "to_depth", spy)

    prepare_source(
        "his",
        "temp",
        {"depth": 10.0},
        None,
        use_cache=False,
        bbox=(_POINT_LON, _POINT_LAT, _POINT_LON, _POINT_LAT),
    )
    assert captured["chunks"] is None


def test_a_gridded_lane_is_never_eagerly_loaded(monkeypatch):
    """No point crop -> the transform still sees a lazy (dask) array.

    The whole point of gating on ``point_window`` -- a full-domain lane must
    never be materialized here, or a real regional model would blow up memory
    instead of saving time.
    """
    _resolve_stub(monkeypatch)
    captured = {}
    real_to_depth = roms.to_depth

    def spy(sub, meta, targets):
        captured["chunks"] = sub["temp"].chunks
        return real_to_depth(sub, meta, targets)

    monkeypatch.setattr(roms, "to_depth", spy)

    prepare_source("his", "temp", {"depth": 10.0}, None, use_cache=False)
    assert captured["chunks"] is not None


def test_the_byte_ceiling_falls_back_to_lazy(monkeypatch):
    """A point lane over the ceiling keeps its laziness contract."""
    _resolve_stub(monkeypatch)
    monkeypatch.setattr(comparison, "POINT_COLUMN_MATERIALIZE_MAX_BYTES", 0)
    captured = {}
    real_to_depth = roms.to_depth

    def spy(sub, meta, targets):
        captured["chunks"] = sub["temp"].chunks
        return real_to_depth(sub, meta, targets)

    monkeypatch.setattr(roms, "to_depth", spy)

    prepare_source(
        "his",
        "temp",
        {"depth": 10.0},
        None,
        use_cache=False,
        bbox=(_POINT_LON, _POINT_LAT, _POINT_LON, _POINT_LAT),
    )
    assert captured["chunks"] is not None


def test_materializing_early_does_not_change_the_result(monkeypatch):
    """Pure performance change: same values whether or not the point lane loads early."""
    _resolve_stub(monkeypatch)
    materialized, _ = prepare_source(
        "his",
        "temp",
        {"depth": 10.0},
        None,
        use_cache=False,
        bbox=(_POINT_LON, _POINT_LAT, _POINT_LON, _POINT_LAT),
    )

    monkeypatch.setattr(comparison, "POINT_COLUMN_MATERIALIZE_MAX_BYTES", 0)
    lazy, _ = prepare_source(
        "his",
        "temp",
        {"depth": 10.0},
        None,
        use_cache=False,
        bbox=(_POINT_LON, _POINT_LAT, _POINT_LON, _POINT_LAT),
    )
    xr.testing.assert_allclose(materialized, lazy)
