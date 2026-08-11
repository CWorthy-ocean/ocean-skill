"""Metrics must not care how the aligned pair happens to be chunked.

xskillscore passes ``dim`` to ``apply_ufunc`` as *core* dimensions, and dask's
``parallelized`` mode refuses a core dimension split across chunks. Any lazily-read
reference trips this: a NetCDF opened with ``chunks={}`` inherits the file's internal
chunking, which is routinely several chunks in lat. It surfaced on a real comparison
(ROMS vs OceanSODA) as::

    ValueError: dimension lat on 0th function argument to apply_ufunc with
    dask='parallelized' consists of multiple chunks, but is also a core dimension.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill.metrics import compute


def _pair(chunks=None) -> xr.Dataset:
    rng = np.random.default_rng(0)
    coords = {"lat": np.linspace(18, 31, 40), "lon": np.linspace(-98, -80, 60)}
    fields = {}
    for name, seed in (("test", 1), ("reference", 2)):
        rng = np.random.default_rng(seed)
        da = xr.DataArray(
            rng.random((40, 60)) + 1.0, dims=("lat", "lon"), coords=coords
        )
        fields[name] = da.chunk(chunks) if chunks else da
    return xr.Dataset(fields)


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param({"lat": 10, "lon": 20}, id="multi-chunk-both"),
        pytest.param({"lat": 10, "lon": -1}, id="multi-chunk-lat"),
        pytest.param({"lat": -1, "lon": 20}, id="multi-chunk-lon"),
        pytest.param({"lat": -1, "lon": -1}, id="single-chunk"),
    ],
)
def test_metrics_work_on_any_chunking(chunks):
    record = compute(_pair(chunks))

    assert np.isfinite(record["bias"])
    assert np.isfinite(record["rmse"])


def test_chunking_does_not_change_the_numbers():
    """The reduction is over every spatial dim, so chunking is purely an artifact."""
    eager = compute(_pair())
    chunked = compute(_pair({"lat": 10, "lon": 20}))

    for key in ("bias", "rmse", "mae", "corr", "sigma_ratio", "n"):
        assert chunked[key] == pytest.approx(eager[key]), key


def test_weights_with_fewer_dims_than_the_field():
    """area_weights need not carry every dimension, so rechunking must not assume it."""
    aligned = _pair({"lat": 10, "lon": 20})
    weights = xr.DataArray(
        np.cos(np.deg2rad(aligned.lat.values)),
        dims=("lat",),
        coords={"lat": aligned.lat},
    ).chunk({"lat": 10})

    record = compute(aligned, weights=weights)

    assert np.isfinite(record["bias"])
