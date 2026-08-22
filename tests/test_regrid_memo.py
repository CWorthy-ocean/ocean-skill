"""The regridder is memoized on grid content, so a fan pays for xesmf's weights once.

:func:`compare`'s ``times=`` fan builds one :class:`Comparison` per time bin, each
with its own freshly-read (but numerically identical) lane objects — these tests pin
that repeat builds against the same grid pair hit the memo
(:func:`ocean_skill.align._regridder_for`), that a different method or a different
grid still rebuilds, that direction (see ``test_regrid_target.py``) is folded into
the key correctly, that the LRU bound evicts, and that a memoized result is
numerically identical to a fresh one.
"""

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as A


def _test_grid(n=8, value=10.0):
    """A model-like lane, fine enough to stay the frame under the default hysteresis."""
    lon = np.linspace(-160.0, -140.0, n)
    lat = np.linspace(10.0, 30.0, n)
    return xr.DataArray(
        np.full((n, n), float(value)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "mmol/m^3"},
    )


def _reference_grid(n=20, value=4.0):
    """A coarser, wider reference lane -- climatology- or satellite-like."""
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
    """Wrap ``xesmf.Regridder`` to count real constructions while still building one."""
    import xesmf

    original = xesmf.Regridder
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(xesmf, "Regridder", counting)
    return calls


def test_same_grids_and_method_build_the_regridder_once(counted_regridder):
    test, reference = _test_grid(), _reference_grid()
    A.align(test, reference, method="bilinear")
    A.align(test, reference, method="bilinear")
    assert counted_regridder["n"] == 1


def test_a_different_method_rebuilds(counted_regridder):
    test, reference = _test_grid(), _reference_grid()
    A.align(test, reference, method="bilinear")
    A.align(test, reference, method="conservative_normed")
    assert counted_regridder["n"] == 2


def test_a_different_grid_rebuilds(counted_regridder):
    test = _test_grid()
    A.align(test, _reference_grid(n=20), method="bilinear")
    A.align(test, _reference_grid(n=21), method="bilinear")
    assert counted_regridder["n"] == 2


def test_a_direction_flip_pair_still_memoizes_on_the_resolved_direction(
    counted_regridder,
):
    """A coarse test against a fine reference flips target= to 'test'.

    See test_regrid_target.py; the memo must still hit on repeat despite the swap.
    """
    test, reference = _test_grid(n=11), _reference_grid(n=161, value=4.0)
    first = A.align(test, reference, method="conservative_normed")
    second = A.align(test, reference, method="conservative_normed")
    assert first.attrs["regrid_target"] == "test"
    assert counted_regridder["n"] == 1
    xr.testing.assert_allclose(first["test"], second["test"])
    xr.testing.assert_allclose(first["reference"], second["reference"])


def test_lru_eviction_past_the_cap(counted_regridder):
    """More distinct pairs than the memo holds evicts the oldest, not the newest."""
    test = _test_grid()
    pairs = [_reference_grid(n=20 + i) for i in range(A._REGRIDDER_MEMO_SIZE + 1)]
    for reference in pairs:
        A.align(test, reference, method="bilinear")
    assert counted_regridder["n"] == len(pairs)

    # the memo holds only the last _REGRIDDER_MEMO_SIZE entries -- revisiting the
    # very first pair again is therefore a miss, not a hit
    A.align(test, pairs[0], method="bilinear")
    assert counted_regridder["n"] == len(pairs) + 1

    # but the most recent pair is still warm
    A.align(test, pairs[-1], method="bilinear")
    assert counted_regridder["n"] == len(pairs) + 1


def test_memoized_result_matches_a_fresh_build_numerically(counted_regridder):
    test, reference = _test_grid(), _reference_grid()
    first = A.align(test, reference, method="bilinear")
    second = A.align(test, reference, method="bilinear")
    assert counted_regridder["n"] == 1
    xr.testing.assert_allclose(first["test"], second["test"])
    xr.testing.assert_allclose(first["difference"], second["difference"])


def test_clear_regridder_memo_forces_a_rebuild(counted_regridder):
    test, reference = _test_grid(), _reference_grid()
    A.align(test, reference, method="bilinear")
    A.clear_regridder_memo()
    A.align(test, reference, method="bilinear")
    assert counted_regridder["n"] == 2


def test_each_test_starts_with_an_empty_memo():
    """The autouse conftest fixture clears the memo -- pin that it actually ran."""
    assert len(A._REGRIDDER_MEMO) == 0
