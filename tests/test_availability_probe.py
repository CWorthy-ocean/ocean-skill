"""Tests for the reference-availability probe inside :meth:`Comparison.align`.

Regression for a real ``osk.compare(reference=<CTD mooring>, test=<ROMS run>,
variables=["temperature"])`` that ran the ROMS test lane's vertical transform for
~30 minutes before discovering the *reference* never carried the variable at all.
:func:`~ocean_skill.comparison._variable_available` answers that question first,
with a read and a resolve -- no crop, no transform, no ``.load()`` -- so the same
``KeyError`` fires before the expensive lane is ever touched.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import comparison

# fresh_availability_memo (tests/conftest.py) clears comparison._AVAILABILITY_MEMO
# around every test -- this file's tests reuse "obs"/"model"/"temperature" often
# enough that a leftover positive would silently short-circuit a later read.


def _salinity_only():
    return xr.Dataset(
        {"Salinity": (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": [10.0, 11.0], "lon": [200.0, 201.0]},
    )


def _ctd_frame():
    """Build a pandas DataFrame shaped like the real tabular CTD mooring source."""
    time = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame({"time": time, "Salinity": [35.0, 35.1, 35.2]})


NO_ROUTE_META = {"variables": [], "featureType": None}


def test_align_raises_before_the_test_lane_is_touched():
    """The KeyError fires from the probe, never reaching the test lane's own read."""

    def fake_read(name, **kwargs):
        if name == "model":
            raise AssertionError("test lane must not be read")
        return _salinity_only()

    with mock.patch.object(osk, "read", fake_read), mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata=NO_ROUTE_META),
    ):
        c = comparison.Comparison(
            reference="obs", test="model", variable="temperature", over="time"
        )
        with pytest.raises(KeyError, match="not available in 'obs'"):
            c.align()


def test_a_tabular_reference_is_converted_before_the_probe_resolves():
    """The probe mirrors _prepare's to_dataset step -- the CTD-shaped regression.

    A probe that skipped the frame->Dataset conversion would hand
    ``operators.resolve_variable`` a raw pandas DataFrame, fail to resolve
    anything on it, and (thanks to the fail-open contract) report the variable
    available regardless -- which would let this test's ``AssertionError`` sentinel
    on the model lane trip instead of the intended ``KeyError``.
    """
    meta = {"variables": [], "featureType": "timeSeries", "axes": {"T": "time"}}

    def fake_read(name, **kwargs):
        if name == "model":
            raise AssertionError("test lane must not be read")
        return _ctd_frame()

    with mock.patch.object(osk, "read", fake_read), mock.patch(
        "ocean_skill.catalog.resolve", lambda n: SimpleNamespace(metadata=meta)
    ):
        c = comparison.Comparison(
            reference="obs", test="model", variable="temperature", over="time"
        )
        with pytest.raises(KeyError, match="not available in 'obs'"):
            c.align()


def test_compare_skips_a_missing_reference_variable_without_preparing_the_test_lane(
    capsys,
):
    declared = {"obs": NO_ROUTE_META, "model": NO_ROUTE_META}
    reads: list[str] = []

    def fake_read(name, **kwargs):
        reads.append(name)
        if name == "model":
            raise AssertionError("test lane must not be read")
        return _salinity_only()

    with mock.patch.object(osk, "read", fake_read), mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata=declared[n]),
    ):
        out = comparison.compare(
            reference=["obs"], test=["model"], variables=["temperature"]
        )
    assert len(out) == 0
    assert reads == ["obs"]
    captured = capsys.readouterr()
    assert "skipped" in captured.out
    assert "not available in 'obs'" in captured.out


def test_compare_reraises_a_missing_reference_variable_with_skip_missing_false():
    declared = {"obs": NO_ROUTE_META, "model": NO_ROUTE_META}

    def fake_read(name, **kwargs):
        if name == "model":
            raise AssertionError("test lane must not be read")
        return _salinity_only()

    with mock.patch.object(osk, "read", fake_read), mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata=declared[n]),
    ), pytest.raises(KeyError, match="not available in 'obs'"):
        comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=["temperature"],
            skip_missing=False,
        )


def test_a_positive_probe_is_memoized_per_source_and_variable():
    reads: list[str] = []

    def fake_read(name, **kwargs):
        reads.append(name)
        return xr.Dataset(
            {"temperature": (("lat", "lon"), np.ones((2, 2)))},
            coords={"lat": [10.0, 11.0], "lon": [200.0, 201.0]},
        )

    with mock.patch.object(osk, "read", fake_read), mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata=NO_ROUTE_META),
    ):
        assert comparison._variable_available("obs", "temperature") is True
        assert comparison._variable_available("obs", "temperature") is True
        assert reads == ["obs"]  # second call served from the memo

        assert (
            comparison._variable_available("obs", "temperature", refresh=True)
            is True
        )
        assert reads == ["obs", "obs"]  # refresh= bypasses the memo


def test_the_probe_fails_open_on_a_reader_error():
    def fake_read(name, **kwargs):
        raise RuntimeError("flaky reader")

    with mock.patch.object(osk, "read", fake_read), mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata=NO_ROUTE_META),
    ):
        assert comparison._variable_available("obs", "temperature") is True


def test_a_calculate_spec_is_not_probed():
    def fake_read(name, **kwargs):
        raise AssertionError("a calculate-spec must not be read at all")

    with mock.patch.object(osk, "read", fake_read):
        assert comparison._variable_available("obs", {"calculate": "mld"}) is True


def test_roms_test_lane_offers_geographic_velocity_though_it_declares_grid_relative(
    capsys,
):
    """Regression for the fallout of the "east_velocity"/"x_velocity" vocabulary split.

    Before the split, eastward_sea_water_velocity was a plain alias of
    sea_water_x_velocity, so a ROMS entry declaring only the grid-relative pair
    matched by accident. After the split they are different quantities, so
    ``compare()``'s catalog pre-filter (``_offers``, which this exercises through
    the public entry point rather than reaching the private closure directly) would
    otherwise skip every pair with "no test offers ..." *before* the test lane --
    where :func:`ocean_skill.roms.standardize` actually derives the geographic
    velocity -- is ever read. Only ``_offers``'s ROMS-aware augmentation
    (:func:`ocean_skill.roms.derived_geographic_velocities`) keeps this working
    without a catalog rebuild.
    """
    declared = {
        "obs": {"variables": ["eastward_sea_water_velocity"], "featureType": None},
        "his": {
            "model": "roms",
            "variables": ["sea_water_x_velocity", "sea_water_y_velocity"],
            "featureType": None,
        },
    }

    def fake_read(name, **kwargs):
        # Standing in for what roms.standardize would have produced by read time --
        # this test is about the catalog-metadata gate letting the pair through,
        # not about re-proving the rotation itself (see tests/test_roms_velocity.py).
        value = 1.0 if name == "his" else 2.0
        return xr.Dataset(
            {"eastward_sea_water_velocity": (("lat", "lon"), np.full((2, 2), value))},
            coords={"lat": [10.0, 11.0], "lon": [200.0, 201.0]},
        )

    with mock.patch.object(osk, "read", fake_read), mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata=declared[n]),
    ):
        out = comparison.compare(
            reference=["obs"],
            test=["his"],
            variables=["eastward_sea_water_velocity"],
            cache=False,
        )
    assert len(out) == 1
    captured = capsys.readouterr()
    assert "no test offers" not in captured.out
    assert "no reference offers" not in captured.out


def test_the_augmentation_is_roms_only(capsys):
    """A non-ROMS source declaring the same raw names is not credited with derivation.

    Guards against widening ``_offers``' fix beyond ROMS -- a source that merely
    happens to declare ``sea_water_x_velocity``/``sea_water_y_velocity`` (a plain
    grid-relative product, not a ROMS run standardize() would rotate) must not be
    reported as offering geographic velocity it has no way to produce.
    """
    declared = {
        "obs": {"variables": ["eastward_sea_water_velocity"], "featureType": None},
        "not_roms": {
            "variables": ["sea_water_x_velocity", "sea_water_y_velocity"],
            "featureType": None,
        },
    }

    def fake_read(name, **kwargs):
        raise AssertionError("neither lane should be read: the test lane never offers")

    with mock.patch.object(osk, "read", fake_read), mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata=declared[n]),
    ):
        out = comparison.compare(
            reference=["obs"],
            test=["not_roms"],
            variables=["eastward_sea_water_velocity"],
            cache=False,
        )
    assert len(out) == 0
    captured = capsys.readouterr()
    assert "no test offers" in captured.out
