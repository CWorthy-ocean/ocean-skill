"""Depth interpolation against a source chunked in the vertical.

Whether a store is chunked along ``s_rho`` is a property of how it was written, not of
anything the caller asks for, so this failed only on real output and only once someone
requested a list of depths: xgcm hands the vertical to ``apply_ufunc`` as a *core*
dimension, and a core dimension spanning several dask chunks is a hard error.

Built from a synthetic ROMS-shaped dataset rather than the catalog fixture the rest of
``test_roms.py`` uses, so the chunking is the thing under test and is set here.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms

META = {"vertical": {"hc": 20.0, "s_dim": "s_rho"}}


def _roms_like(nt=2, ns=12, ny=6, nx=8) -> xr.Dataset:
    """Return a minimal Vtransform-2 ROMS dataset: a tracer plus z_rho's grid."""
    sigma = np.linspace(-1.0, 0.0, ns)
    rng = np.random.default_rng(0)
    h = 50.0 + 150.0 * rng.random((ny, nx))
    return xr.Dataset(
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


def test_to_depth_survives_a_vertically_chunked_source():
    """The reported failure: several depths from a store chunked along s_rho."""
    pytest.importorskip("xgcm")
    pytest.importorskip("dask")
    ds = _roms_like().chunk({"s_rho": 3, "time": 1})
    assert len(ds["temp"].chunks[1]) > 1, "the fixture must be split in the vertical"

    out = roms.to_depth(ds, META, [0.0, 20.0, 40.0])

    assert out.sizes["z"] == 3
    assert np.isfinite(out["temp"].isel(z=1)).any(), "20 m should be inside the column"


def test_the_rechunk_leaves_the_horizontal_alone():
    """Only the vertical is coalesced; horizontal chunking is what bounds memory."""
    pytest.importorskip("dask")
    ds = _roms_like(ny=8, nx=8).chunk({"s_rho": 3, "eta_rho": 4, "xi_rho": 4})
    rechunked = roms._contiguous_column(ds["temp"], "s_rho")

    dims = dict(zip(rechunked.dims, rechunked.chunks, strict=True))
    assert len(dims["s_rho"]) == 1, "the column must be contiguous"
    assert len(dims["eta_rho"]) == 2, "horizontal chunking must survive"
    assert len(dims["xi_rho"]) == 2


def test_an_eager_array_is_not_made_lazy():
    """A rechunk helper that quietly converts numpy to dask would be a trap."""
    ds = _roms_like()
    assert roms._contiguous_column(ds["temp"], "s_rho").chunks is None


def test_a_vertically_chunked_source_works_with_a_month_axis_standing():
    """The facet case: interpolate to depths with a monthly axis still on the field.

    ``field(..., select={"depth": [...]}, aggregate={"time": {"resample": ...}})``
    reaches ``to_depth`` with time *not* collapsed, which is the shape a comparison
    never produces.
    """
    pytest.importorskip("xgcm")
    pytest.importorskip("dask")
    ds = _roms_like(nt=6).chunk({"s_rho": 3, "time": 2})

    out = roms.to_depth(ds, META, [0.0, 20.0, 40.0])

    assert out.sizes["time"] == 6, "the facet axis must survive the transform"
    assert out.sizes["z"] == 3
