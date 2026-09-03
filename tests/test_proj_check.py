"""Tests for the cartopy/PROJ mis-projection tripwire.

See :mod:`ocean_skill.plot.proj_check`.
"""

from __future__ import annotations

import pytest

from ocean_skill.plot import proj_check


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear ``projection_skew``'s cache around every test.

    It caches its one probe transform per process, so a monkeypatch in one
    test cannot otherwise leak its cached answer into the next.
    """
    proj_check.projection_skew.cache_clear()
    yield
    proj_check.projection_skew.cache_clear()


def test_healthy_environment_has_no_skew():
    """The tripwire's whole point is to stay silent on a correct pairing.

    This repo's test environment is one -- only a broken pairing should make
    it speak up.
    """
    assert proj_check.projection_skew() is None


def test_healthy_environment_warns_nothing(recwarn):
    proj_check.warn_projection_skew()
    assert not any("PROJ" in str(w.message) for w in recwarn.list)


def _break_transform(monkeypatch, offset_m=25_000.0):
    """Simulate the PROJ>=9.8 / cartopy<=0.25 pairing.

    ``transform_points`` comes back off by roughly the observed ~0.22°
    latitude skew.
    """
    import cartopy.crs as ccrs
    import numpy as np

    real = ccrs.GOOGLE_MERCATOR.transform_points

    def broken(src_crs, x, y):
        pts = np.array(real(src_crs, x, y), dtype=float)
        pts[:, 1] += offset_m
        return pts

    monkeypatch.setattr(ccrs.GOOGLE_MERCATOR, "transform_points", broken)


def test_broken_pairing_is_detected(monkeypatch):
    _break_transform(monkeypatch)
    message = proj_check.projection_skew()
    assert message is not None
    assert "cartopy" in message


def test_broken_pairing_message_names_both_remedies(monkeypatch):
    _break_transform(monkeypatch)
    message = proj_check.projection_skew()
    assert "proj<9.8" in message
    assert "cartopy" in message and "0.26" in message
    assert "cartopy#2645" in message


def test_broken_pairing_warns(monkeypatch):
    _break_transform(monkeypatch)
    with pytest.warns(UserWarning, match="cartopy"):
        proj_check.warn_projection_skew()


def test_skew_within_tolerance_is_not_reported(monkeypatch):
    # sub-metre noise (ordinary floating point / PROJ-pipeline jitter) must not trip it
    _break_transform(monkeypatch, offset_m=0.1)
    assert proj_check.projection_skew() is None
