"""Carrying a ``spread`` (mean+std, most often) through ``align()``.

``operators.aggregate``'s ``spread`` option rides as a same-shape non-dimension
coordinate on the value it describes (see ``operators.SPREAD_COORD``) -- which
works while the two lanes are apart, but not once they meet in one ``xr.Dataset``:
the test and reference's own spreads are two different arrays, and two
coordinates sharing one name that disagree raise a ``MergeError`` on
construction. ``align._split_spread`` is what keeps that from happening, by
popping each lane's own spread coordinate off before construction and
re-attaching it after as a separate ``test_spread``/``reference_spread`` data
variable. Mirrors ``test_vertical_match.py`` in fixture style.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as A
from ocean_skill.operators import SPREAD_COORD

LON = np.array([-95.0, -94.0])
LAT = np.array([25.0, 26.0])


def _gridded_test(z, *, name="z", lon=LON, lat=LAT, seed: int = 0, spread=None):
    """A model-shaped lane: fixed vertical levels, uniform across the horizontal."""
    rng = np.random.default_rng(seed)
    base = 20.0 - 0.1 * np.abs(z)
    noise = rng.normal(0, 0.05, (len(z), len(lat), len(lon)))
    values = base[:, None, None] + noise
    da = xr.DataArray(
        values,
        dims=(name, "lat", "lon"),
        coords={name: z, "lat": lat, "lon": lon},
        attrs={"units": "degC"},
    )
    if spread is not None:
        da = da.assign_coords({SPREAD_COORD: (name, np.asarray(spread))})
    return da


def _station_reference(depth, *, name="DEPTH", lon=-94.5, lat=25.5, spread=None):
    """A station-shaped lane: real levels, one horizontal position."""
    values = 20.0 - 0.1 * depth
    da = xr.DataArray(
        values, dims=(name,), coords={name: depth}, attrs={"units": "degC"}
    ).assign_coords(lon=lon, lat=lat)
    if spread is not None:
        da = da.assign_coords({SPREAD_COORD: (name, np.asarray(spread))})
    return da


# -- _split_spread: the pop-then-attach mechanism itself -------------------------


def test_two_lanes_with_spread_do_not_mergeerror_and_land_as_data_variables():
    z = np.array([0.0, 10.0, 20.0])
    test = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="z", coords={"z": z})
    test = test.assign_coords({SPREAD_COORD: ("z", [0.1, 0.2, 0.3])})
    reference = xr.DataArray(np.array([4.0, 5.0, 6.0]), dims="z", coords={"z": z})
    reference = reference.assign_coords({SPREAD_COORD: ("z", [0.4, 0.5, 0.6])})

    test_clean, reference_clean, attach = A._split_spread(test, reference)
    assert SPREAD_COORD not in test_clean.coords
    assert SPREAD_COORD not in reference_clean.coords

    out = xr.Dataset(
        {
            "test": test_clean,
            "reference": reference_clean,
            "difference": test_clean - reference_clean,
        }
    )
    out = attach(out)
    assert list(out["test_spread"].values) == [0.1, 0.2, 0.3]
    assert list(out["reference_spread"].values) == [0.4, 0.5, 0.6]
    assert SPREAD_COORD not in out.coords


def test_split_spread_is_a_no_op_when_neither_lane_has_one():
    z = np.array([0.0, 10.0])
    test = xr.DataArray(np.array([1.0, 2.0]), dims="z", coords={"z": z})
    reference = xr.DataArray(np.array([3.0, 4.0]), dims="z", coords={"z": z})
    test_clean, reference_clean, attach = A._split_spread(test, reference)
    out = xr.Dataset({"test": test_clean, "reference": reference_clean})
    out = attach(out)
    assert "test_spread" not in out.data_vars
    assert "reference_spread" not in out.data_vars


def test_split_spread_handles_only_one_lane_carrying_it():
    z = np.array([0.0, 10.0])
    test = xr.DataArray(np.array([1.0, 2.0]), dims="z", coords={"z": z})
    test = test.assign_coords({SPREAD_COORD: ("z", [0.1, 0.2])})
    reference = xr.DataArray(np.array([3.0, 4.0]), dims="z", coords={"z": z})
    test_clean, reference_clean, attach = A._split_spread(test, reference)
    out = xr.Dataset({"test": test_clean, "reference": reference_clean})
    out = attach(out)
    assert list(out["test_spread"].values) == [0.1, 0.2]
    assert "reference_spread" not in out.data_vars


# -- end to end through align(), the profile priority path -----------------------


def test_spread_is_interpolated_onto_reference_levels():
    """The whole point: a season aggregate's ``spread`` coordinate on the test
    lane rides through ``_match_vertical``'s interpolation for free, and the
    reference's own spread survives untouched, landing as two separate data
    variables with no MergeError."""
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    test = _gridded_test(z, spread=[1.0, 1.1, 1.2, 1.3, 1.4])
    reference = _station_reference(depth, spread=[0.5, 0.6, 0.7])

    out = A.align(test, reference, over="Z", test_name="test", reference_name="reference")

    assert "test_spread" in out.data_vars
    assert "reference_spread" in out.data_vars
    assert list(out["reference_spread"].values) == [0.5, 0.6, 0.7]
    # linear interpolation of [1.0, 1.1, 1.2, 1.3, 1.4] (at |z|=0,10,25,50,100)
    # onto depths [5, 20, 60]
    expected = np.interp([5.0, 20.0, 60.0], [0.0, 10.0, 25.0, 50.0, 100.0], [1.0, 1.1, 1.2, 1.3, 1.4])
    assert np.allclose(out["test_spread"].values, expected)
    assert SPREAD_COORD not in out.coords


def test_no_spread_means_no_spread_variables():
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    out = A.align(
        _gridded_test(z), _station_reference(depth), over="Z",
        test_name="test", reference_name="reference",
    )
    assert "test_spread" not in out.data_vars
    assert "reference_spread" not in out.data_vars


def test_only_the_reference_carrying_a_spread_still_works():
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    test = _gridded_test(z)
    reference = _station_reference(depth, spread=[0.5, 0.6, 0.7])
    out = A.align(test, reference, over="Z", test_name="test", reference_name="reference")
    assert "reference_spread" in out.data_vars
    assert "test_spread" not in out.data_vars


# -- a regridded lane's spread is dropped, with a warning -------------------------


def test_drop_spread_before_regrid_is_a_no_op_without_one():
    da = xr.DataArray(np.array([1.0, 2.0]), dims="z", coords={"z": [0.0, 10.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = A._drop_spread_before_regrid(da, "test")
    assert out is da


def test_drop_spread_before_regrid_drops_it_with_a_warning():
    da = xr.DataArray(np.array([1.0, 2.0]), dims="z", coords={"z": [0.0, 10.0]})
    da = da.assign_coords({SPREAD_COORD: ("z", [0.1, 0.2])})
    with pytest.warns(UserWarning, match="not carried through a horizontal regrid"):
        out = A._drop_spread_before_regrid(da, "test")
    assert SPREAD_COORD not in out.coords
