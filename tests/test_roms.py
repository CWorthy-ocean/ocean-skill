"""Tests for the ROMS adapter via the v2 catalog (roms-tools example, local data)."""

from __future__ import annotations

import os

import numpy as np
import pytest

import ocean_skill as osk
from ocean_skill import roms
from ocean_skill.catalog import resolve

COMBINED = "/Users/kthyng/packages/roms-tools/docs/OUTPUT/old/combined.nc"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.path.exists(COMBINED),
        reason="roms-tools combined example output not available locally",
    ),
]


def _finite_frac(da) -> float:
    return float(np.isfinite(da).mean())


def test_read_roms_standardized():
    ds = osk.read("roms_example_combined")
    # combined file carries physical fields; renamed to CF standard_names
    assert "sea_water_potential_temperature" in ds.data_vars
    assert "sea_water_practical_salinity" in ds.data_vars
    # self-contained grid -> horizontal coords present (2-D curvilinear)
    assert "lon" in ds.coords and "lat" in ds.coords
    assert set(ds["lon"].dims) == {"eta_rho", "xi_rho"}
    # time decoded to datetime64
    assert np.issubdtype(ds["time"].dtype, np.datetime64)
    # depth coordinate reconstructed
    assert "z_rho" in ds.coords
    # land masked -> NaNs at the surface level
    surf = ds["sea_water_potential_temperature"].isel(time=0, s_rho=-1)
    assert bool(np.isnan(surf).any())
    assert ds.attrs.get("featureType") == "grid"


def test_surface_top_level():
    ds = osk.read("roms_example_combined")
    top = roms.surface(ds)
    assert "s_rho" not in top.dims
    da = top["sea_water_potential_temperature"].isel(time=0)
    assert _finite_frac(da) > 0.2
    vals = da.values[np.isfinite(da.values)]
    assert 5.0 < float(vals.mean()) < 35.0  # GoM SST sanity (deg C)


def test_to_depth_subsurface():
    ds = osk.read("roms_example_combined")
    meta = resolve("roms_example_combined").metadata
    at_100m = roms.to_depth(ds, meta, 100.0)
    da = at_100m["sea_water_potential_temperature"].isel(time=0, z=0)
    assert set(da.dims) == {"eta_rho", "xi_rho"}
    assert _finite_frac(da) > 0.05
    vals = da.values[np.isfinite(da.values)]
    assert 3.0 < float(vals.mean()) < 30.0
