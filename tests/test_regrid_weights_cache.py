"""Regridder weights persist to disk, so a cross-process/session miss is still cheap.

:data:`ocean_skill.align._REGRIDDER_MEMO` only survives one process; ESMF weight
generation is the pipeline's slowest step and depends only on the two grids'
geometry, so :func:`ocean_skill.align._build_regridder` also persists it under
:func:`ocean_skill.cache.path` ``("weights")`` -- these tests pin that a disk hit
skips ESMF (``xe.Regridder`` is still constructed, but with ``weights=`` rather than
computing fresh), that the reused regridder matches a fresh build numerically, that a
locstream (station/transect) target shares the same mechanism, and that a corrupt
weights file degrades to a rebuild rather than an error, mirroring
:func:`ocean_skill.cache.load`.
"""

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as A
from ocean_skill import cache


def _test_grid(n=8, value=10.0):
    lon = np.linspace(-160.0, -140.0, n)
    lat = np.linspace(10.0, 30.0, n)
    return xr.DataArray(
        np.full((n, n), float(value)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "mmol/m^3"},
    )


def _reference_grid(n=20, value=4.0):
    lon = np.linspace(-170.0, -130.0, n)
    lat = np.linspace(0.0, 40.0, n)
    return xr.DataArray(
        np.full((n, n), float(value)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "mmol/m^3"},
    )


@pytest.fixture
def counted_regridder(monkeypatch):
    """Wrap ``xesmf.Regridder`` to record every construction's kwargs."""
    import xesmf

    original = xesmf.Regridder
    calls = []

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(xesmf, "Regridder", counting)
    return calls


def test_a_process_restart_reuses_weights_from_disk(counted_regridder):
    """A cleared in-process memo still hits the disk layer, not ESMF."""
    test, reference = _test_grid(), _reference_grid()
    A.align(test, reference, method="conservative_normed")
    assert len(counted_regridder) == 1
    assert "weights" not in counted_regridder[0]  # the fresh build computes weights

    A.clear_regridder_memo()  # simulate a new process: the in-memory memo is gone
    A.align(test, reference, method="conservative_normed")
    assert len(counted_regridder) == 2
    assert "weights" in counted_regridder[1]  # the second reads the persisted file


def test_reused_weights_match_a_fresh_build_numerically():
    test, reference = _test_grid(), _reference_grid()
    fresh = A.align(test, reference, method="conservative_normed")
    A.clear_regridder_memo()
    reused = A.align(test, reference, method="conservative_normed")
    xr.testing.assert_allclose(fresh["test"], reused["test"])
    xr.testing.assert_allclose(fresh["reference"], reused["reference"])


def test_one_weights_file_is_written_per_grid_pair_and_method():
    test, reference = _test_grid(), _reference_grid()
    A.align(test, reference, method="conservative_normed")
    A.clear_regridder_memo()
    A.align(test, reference, method="conservative_normed")  # a disk hit, not a new file
    A.align(test, reference, method="bilinear")  # a different method, a new file
    assert len(cache.entries("weights")) == 2


def test_a_locstream_target_shares_the_same_disk_cache(counted_regridder):
    """A station/transect regridder (:func:`ocean_skill.align._interp_locstream`)
    persists too, not only the map path.
    """
    curvilinear = xr.DataArray(
        np.full((6, 7), 10.0),
        dims=("eta", "xi"),
        coords={
            "lon": (("eta", "xi"), np.linspace(-160, -150, 42).reshape(6, 7)),
            "lat": (("eta", "xi"), np.linspace(10, 20, 42).reshape(6, 7)),
        },
        attrs={"units": "mmol/m^3"},
    )
    lons, lats = np.array([-158.0, -156.0]), np.array([12.0, 14.0])
    A._interp_locstream(A._as_xesmf(curvilinear), lons, lats)
    assert len(counted_regridder) == 1
    assert "weights" not in counted_regridder[0]

    A.clear_regridder_memo()
    A._interp_locstream(A._as_xesmf(curvilinear), lons, lats)
    assert len(counted_regridder) == 2
    assert "weights" in counted_regridder[1]


def test_a_corrupt_weights_file_degrades_to_a_rebuild(counted_regridder):
    test, reference = _test_grid(), _reference_grid()
    A.align(test, reference, method="conservative_normed")
    [entry] = cache.entries("weights")
    entry.write_bytes(b"not a netcdf file")

    A.clear_regridder_memo()
    with pytest.warns(UserWarning, match="unreadable regridder weights"):
        out = A.align(test, reference, method="conservative_normed")
    assert np.isfinite(out["test"]).any()
    # three constructor calls total: the original fresh build, the failed attempt to
    # read the now-corrupt file (caught and warned about), and the rebuild that
    # follows it -- the corrupt entry is a miss, not a pipeline failure
    assert len(counted_regridder) == 3
    assert "weights" in counted_regridder[1]  # the failed read attempt
    assert "weights" not in counted_regridder[2]  # rebuilt fresh, then rewritten
    assert cache.entries("weights") != []  # the rebuild re-persisted a fresh file


def test_disabled_cache_neither_reads_nor_writes_weights(counted_regridder):
    test, reference = _test_grid(), _reference_grid()
    cache.disable()
    try:
        A.align(test, reference, method="conservative_normed")
        assert cache.entries("weights") == []
        A.clear_regridder_memo()
        A.align(test, reference, method="conservative_normed")
        assert len(counted_regridder) == 2
        assert "weights" not in counted_regridder[1]
    finally:
        cache.enable()
