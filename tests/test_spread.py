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
from ocean_skill import roms
from ocean_skill.operators import SPREAD_COORD

LON = np.array([-95.0, -94.0])
LAT = np.array([25.0, 26.0])

# -- grid for the roms.to_depth/to_sigma0 fixtures below -------------------------

N = 20
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


@pytest.fixture
def roms_column():
    """Shelf-to-abyss grid with a ``chl`` field equal to its own depth.

    Same grid ``tests/test_depth_average.py``/``tests/test_isopycnal.py`` use.
    Temperature cools linearly with depth (constant salinity) so sigma0 is
    monotonic in every column, for the to_sigma0 half of the test below.
    """
    h = np.array([[20.0, 100.0], [1000.0, 5000.0]])
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    ds = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((2, 2))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
        },
        coords={
            "lon": (("eta_rho", "xi_rho"), np.array([[-95.0, -94.0], [-95.0, -94.0]])),
            "lat": (("eta_rho", "xi_rho"), np.array([[25.0, 25.0], [26.0, 26.0]])),
        },
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    ds = roms.add_depth_coord(ds, meta)
    temp = 20.0 + 0.002 * ds["z_rho"]  # z_rho negative-down: colder with depth
    salt = xr.full_like(temp, 35.0)
    chl = -ds["z_rho"]  # value == its own depth, for an independent check
    ds = ds.assign(
        sea_water_potential_temperature=temp,
        sea_water_practical_salinity=salt,
        chl=chl,
    )
    return ds, meta


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
    variables with no MergeError.
    """
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    test = _gridded_test(z, spread=[1.0, 1.1, 1.2, 1.3, 1.4])
    reference = _station_reference(depth, spread=[0.5, 0.6, 0.7])

    out = A.align(
        test, reference, over="Z", method="nearest", test_name="test",
        reference_name="reference",
    )

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
        _gridded_test(z), _station_reference(depth), over="Z", method="nearest",
        test_name="test", reference_name="reference",
    )
    assert "test_spread" not in out.data_vars
    assert "reference_spread" not in out.data_vars


def test_only_the_reference_carrying_a_spread_still_works():
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    test = _gridded_test(z)
    reference = _station_reference(depth, spread=[0.5, 0.6, 0.7])
    out = A.align(
        test, reference, over="Z", method="nearest", test_name="test",
        reference_name="reference",
    )
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


# -- roms.to_depth/to_sigma0: the gap this diagnosis found -----------------------
#
# A profile comparison's model lane goes through to_depth (interpolating onto the
# reference's own levels) or to_sigma0 (an isopycnal slice) after the time
# aggregate has already attached a spread coordinate. Both rebuild their result
# with a fixed coordinate whitelist, which used to drop that coordinate silently
# -- a mean+spread envelope showed for the observational reference (never touching
# either transform) but not for the model. See ocean_skill.roms._transform_spread.


def test_to_depth_carries_spread_onto_the_new_vertical_axis(roms_column):
    ds, meta = roms_column
    # An affine function of the same z_rho chl already equals, so linear
    # interpolation reproduces it exactly at any reachable target depth --
    # letting this check itself rather than trusting the transform.
    spread = 3.0 + 0.5 * ds["chl"]
    ds = ds.assign_coords({SPREAD_COORD: spread})

    targets = [5.0, 15.0]  # within every column's range (shallowest h is 20 m)
    out = roms.to_depth(ds, meta, targets)

    assert SPREAD_COORD in out.coords
    assert SPREAD_COORD in out["chl"].coords  # rides with the variable downstream
    assert "z" in out[SPREAD_COORD].dims
    assert "s_rho" not in out[SPREAD_COORD].dims

    expected = 3.0 + 0.5 * out["chl"]
    np.testing.assert_allclose(
        out[SPREAD_COORD].values, expected.values, equal_nan=True
    )


def test_to_depth_without_a_spread_coordinate_is_unaffected(roms_column):
    ds, meta = roms_column
    out = roms.to_depth(ds, meta, [5.0, 15.0])
    assert SPREAD_COORD not in out.coords


def test_to_sigma0_carries_spread_onto_the_new_vertical_axis(roms_column):
    gsw = pytest.importorskip("gsw")
    ds, meta = roms_column
    spread = 3.0 + 0.5 * ds["chl"]
    ds = ds.assign_coords({SPREAD_COORD: spread})

    # A target strictly between two of the abyssal column's own sigma0 values, so
    # it is a genuine, reachable interpolation there -- same recipe
    # test_isopycnal.py's test_slice_recovers_the_known_depth_of_a_target_isopycnal
    # uses to pick a target guaranteed not to fall outside the water column.
    col = dict(eta_rho=1, xi_rho=1)
    z = ds["z_rho"].isel(**col).values
    temp = ds["sea_water_potential_temperature"].isel(**col).values
    salt = ds["sea_water_practical_salinity"].isel(**col).values
    lon, lat = float(ds["lon"].isel(**col)), float(ds["lat"].isel(**col))
    pressure = gsw.p_from_z(z, lat)
    sa = gsw.SA_from_SP(salt, pressure, lon, lat)
    ct = gsw.CT_from_pt(sa, temp)
    sigma0_profile = gsw.sigma0(sa, ct)
    order = np.argsort(sigma0_profile)
    mid = len(order) // 2
    target = float(0.5 * (sigma0_profile[order][mid] + sigma0_profile[order][mid - 1]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # other, shallower columns may not reach it
        out = roms.to_sigma0(ds, meta, target)

    assert SPREAD_COORD in out.coords
    assert "sigma0" in out[SPREAD_COORD].dims
    assert "s_rho" not in out[SPREAD_COORD].dims

    expected = float((3.0 + 0.5 * out["chl"]).isel(sigma0=0, **col))
    got = float(out[SPREAD_COORD].isel(sigma0=0, **col))
    assert got == pytest.approx(expected, rel=1e-6)
