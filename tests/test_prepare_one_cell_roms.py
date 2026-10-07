"""A ROMS lane whose catalog entry says it is a profile still prepares.

``_prepare``'s vertical ladder binds ``zname`` (the observational depth axis) only in
its final ``else:`` -- the branch a ROMS lane never takes -- while the block that
prunes a profile's unsampled levels reads it whenever the entry's ``featureType`` is
one of :data:`~ocean_skill.comparison.PROFILE_FEATURE_TYPES`. A one-cell ROMS file that
an older catalog labelled ``timeSeriesProfile`` (or any calculated variable on a
profile lane) therefore died with ``UnboundLocalError`` instead of being prepared the
way the same file under ``featureType: grid`` is.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocean_skill import depth_convention
from ocean_skill.comparison import _prepare
from tests._tidal_roms import tidal_roms

SURFACE = {
    "origin": "surface",
    "datum_z_m": 0.0,
    "support": "point",
    "source": "declared",
}


def _one_cell(**kw):
    """Return the tidal fixture cut to its middle cell (eta_rho/xi_rho length one)."""
    ds, meta = tidal_roms(**kw)
    return ds.isel(eta_rho=slice(1, 2), xi_rho=slice(1, 2)), meta


def _lane(feature_type, variable="height", select=None, **kw):
    ds, meta = _one_cell()
    meta = {**meta, "featureType": feature_type}
    return _prepare(
        ds,
        meta,
        variable,
        {"depth": 3.0, **(select or {})},
        None,
        depth_method="interp",
        depth_convention=SURFACE,
        obs_convention=depth_convention.resolve({}),
        **kw,
    )[0]


@pytest.mark.parametrize("feature_type", ["timeSeriesProfile", "profile"])
def test_a_one_cell_roms_lane_labelled_as_a_profile_prepares(feature_type):
    da = _lane(feature_type)
    # 3 m below the surface, through the +/-3 m tide: z = zeta - 3
    np.testing.assert_allclose(np.asarray(da).reshape(-1), [0.0, -3.0, -6.0, -3.0])


@pytest.mark.parametrize("feature_type", ["timeSeriesProfile", "profile"])
def test_the_profile_label_changes_no_value(feature_type):
    grid = _lane("grid")
    labelled = _lane(feature_type)
    np.testing.assert_array_equal(np.asarray(labelled), np.asarray(grid))
    assert labelled.dims == grid.dims


def test_a_depth_list_on_a_profile_labelled_roms_lane_keeps_its_levels():
    da = _lane("timeSeriesProfile", select={"depth": [3.0, 5.0]})
    assert da.sizes["z"] == 2
    np.testing.assert_allclose(
        np.asarray(da.isel(eta_rho=0, xi_rho=0, time=0)), [0.0, -2.0]
    )
