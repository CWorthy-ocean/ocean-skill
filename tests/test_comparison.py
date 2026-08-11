"""Tests for the compare layer: variable aliasing and the surface/depth=0 distinction.

Both regressions here were found via a real ``osk.compare(..., variables=[OXYGEN])``
call against the GOM MARBL output: the aliased-variable crash
(``ValueError: could not convert string to float: b'T'``, from a bare ``.mean("time")``
falling through to the *whole* dataset including non-numeric fields like ``spherical``)
and the silent conflation of "surface" with "depth=0". Both are reproduced here on a
small synthetic ROMS-shaped dataset — self-contained, no external run directory or
kerchunk reference needed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms
from ocean_skill.comparison import SURFACE, _depth_label, _prepare, is_surface_request

# WOA/GLODAP spell dissolved oxygen per-mass; ROMS/MARBL writes it per-volume (see
# ocean_skill.vocabulary.VOCABULARY) — comparing across that alias is what broke.
OXYGEN_PER_MASS = "moles_of_oxygen_per_unit_mass_in_sea_water"
OXYGEN_PER_VOLUME = "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"


@pytest.fixture(scope="module")
def gom_bgc():
    """Build a minimal self-contained ROMS/MARBL-shaped dataset.

    Carries a byte-string field (``spherical``) standing in for the real non-numeric
    global ROMS carries — the exact shape that broke a naive ``.mean("time")`` over
    the whole dataset.
    """
    rng = np.random.default_rng(0)
    nt, ns, ny, nx = 3, 4, 5, 6
    lon = np.linspace(260.0, 262.0, nx)[None, :] * np.ones((ny, 1))
    lat = np.linspace(20.0, 22.0, ny)[:, None] * np.ones((1, nx))
    h = np.full((ny, nx), 50.0)  # shallow: puts the top cell centre below z=0
    mask = np.ones((ny, nx))
    sigma_r = np.linspace(-1 + 1 / (2 * ns), -1 / (2 * ns), ns)  # cell centres
    cs_r = sigma_r.copy()  # Vtransform 2, theta_s=theta_b=0 => Cs_r == sigma_r
    o2 = 200 + 5 * rng.standard_normal((nt, ns, ny, nx))

    ds = xr.Dataset(
        {
            "O2": (("time", "s_rho", "eta_rho", "xi_rho"), o2),
            "zeta": (("time", "eta_rho", "xi_rho"), np.zeros((nt, ny, nx))),
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), mask),
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "Cs_r": (("s_rho",), cs_r),
            "sigma_r": (("s_rho",), sigma_r),
            "ocean_time": (("time",), np.arange(nt) * 86400.0),
            "spherical": np.bytes_(b"T"),  # the field that broke a whole-dataset mean
        }
    )
    meta = {
        "model": "roms",
        "self_contained_grid": True,
        "standard_names": {"O2": OXYGEN_PER_VOLUME},
        "vertical": {"s_dim": "s_rho", "hc": 300.0, "Vtransform": 2},
        "reference_date": "2000-01-01",
    }
    return roms.standardize(ds, meta), meta


def test_prepare_resolves_aliased_variable(gom_bgc):
    """A per-mass request must resolve to ROMS' per-volume variable.

    Otherwise it falls through to the whole dataset, where a bare ``.mean("time")``
    chokes on non-numeric fields like ``spherical``.
    """
    ds, meta = gom_bgc
    da, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {})
    assert da is not None
    assert np.isfinite(da.values).any()


def test_prepare_fails_closed_on_missing_variable(gom_bgc):
    """A genuinely absent variable must return (None, None).

    Not fall through to reducing the whole dataset — the same crash risk the alias
    bug hit, for a different reason: nothing found rather than the wrong name found.
    """
    ds, meta = gom_bgc
    da, depth = _prepare(ds, meta, "not_a_real_standard_name", {})
    assert da is None
    assert depth is None


def test_surface_and_depth_zero_are_distinct(gom_bgc):
    """Unset/``"surface"`` uses the model's own top level, with no warning.

    An explicit ``depth=0`` is a real interpolation request and may legitimately warn
    all-NaN.
    """
    ds, meta = gom_bgc
    assert is_surface_request(None)
    assert is_surface_request(SURFACE)
    assert not is_surface_request(0)
    assert not is_surface_request(0.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        da_surface, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {})
    assert not any("entirely NaN" in str(w.message) for w in caught)
    assert np.isfinite(da_surface.values).all()

    # The synthetic grid is 50 m deep everywhere with a top cell centre well below
    # 0 m, so an explicit request for the literal surface interpolates to nothing.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _prepare(ds, meta, OXYGEN_PER_MASS, {"depth": 0})
    assert any("entirely NaN" in str(w.message) for w in caught)


def test_depth_label():
    assert _depth_label(None) == "surface"
    assert _depth_label(SURFACE) == "surface"
    assert _depth_label(0) == "0 m"
    assert _depth_label(100.0) == "100 m"
