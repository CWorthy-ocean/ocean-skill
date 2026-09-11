"""``align.match_axis(..., over="Z")``: bringing two lanes onto one vertical axis.

The vertical twin of ``tests/test_axis_match.py``, and deliberately narrower: a
water column has no "composite vs instantaneous" question the way a time axis does
(see ``ocean_skill.align._match_vertical``'s docstring), so ``mean``/``exact`` have
no vertical counterpart -- but a real choice remains between snapping the test lane
to its nearest real level (``depth_method="nearest"``, the default) and linearly
interpolating it onto the reference's own levels (``"interp"``/``"linear"``).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as A

LON = np.array([-95.0, -94.0])
LAT = np.array([25.0, 26.0])


def _gridded_test(z, *, name="z", lon=LON, lat=LAT, seed: int = 0):
    """A model-shaped lane: fixed vertical levels, uniform across the horizontal."""
    rng = np.random.default_rng(seed)
    base = 20.0 - 0.1 * np.abs(z)
    noise = rng.normal(0, 0.05, (len(z), len(lat), len(lon)))
    values = base[:, None, None] + noise
    return xr.DataArray(
        values,
        dims=(name, "lat", "lon"),
        coords={name: z, "lat": lat, "lon": lon},
        attrs={"units": "degC"},
    )


def _station_reference(depth, *, name="DEPTH", lon=-94.5, lat=25.5):
    """A station-shaped lane: real levels, one horizontal position."""
    values = 20.0 - 0.1 * depth
    return xr.DataArray(
        values, dims=(name,), coords={name: depth}, attrs={"units": "degC"}
    ).assign_coords(lon=lon, lat=lat)


# -- match_axis dispatches to nearest/interp, not binning, for a vertical axis ---------


def test_over_Z_dispatches_to_nearest_by_default_not_binning():
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])  # ROMS-style, negative-down
    depth = np.array([5.0, 20.0, 60.0])
    test, reference, report = A.match_axis(
        _gridded_test(z), _station_reference(depth), over="Z"
    )
    assert report["match_method"] == "nearest"
    assert report["match_target"] == "reference"
    assert report["axis"] == "DEPTH"
    assert list(test["DEPTH"].values) == list(depth)


def test_over_Z_can_interpolate_explicitly():
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    test, reference, report = A.match_axis(
        _gridded_test(z), _station_reference(depth), over="Z", depth_method="interp"
    )
    assert report["match_method"] == "interp"
    assert report["match_target"] == "reference"
    assert list(test["DEPTH"].values) == list(depth)


def test_unknown_depth_method_is_refused():
    z = -np.array([0.0, 10.0, 25.0])
    depth = np.array([5.0, 20.0])
    with pytest.raises(ValueError, match="unknown depth_method"):
        A.match_axis(
            _gridded_test(z), _station_reference(depth), over="Z", depth_method="bogus"
        )


def test_lowercase_z_and_depth_are_left_to_the_generic_axis_path():
    """The pre-existing generic-numeric-axis spelling -- unrelated, untouched."""
    z = np.array([0.0, 10.0, 25.0])  # a bare, already-comparable numeric axis
    da = xr.DataArray(
        np.array([1.0, 2.0, 3.0]), dims=("depth",), coords={"depth": z}
    )
    da2 = xr.DataArray(
        np.array([1.5, 2.5]), dims=("depth",), coords={"depth": [5.0, 30.0]}
    )
    # unaffected by _match_vertical: still nearest/mean-matched, no interpolation
    with pytest.warns(UserWarning):
        _, _, report = A.match_axis(da, da2, over="depth")
    assert report["match_method"] in ("nearest", "mean")


# -- sign conventions reconcile correctly, under either method -------------------------


def test_negative_down_z_matches_positive_down_depth_by_value_when_interpolating():
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    test, reference, _ = A.match_axis(
        _gridded_test(z), _station_reference(depth), over="Z", depth_method="interp"
    )
    expected = 20.0 - 0.1 * depth
    got = test.sel(lon=-94.0, lat=25.0).values
    np.testing.assert_allclose(got, expected, atol=0.2)


def test_negative_down_z_matches_positive_down_depth_by_value_when_nearest():
    """Nearest must compare like-signed values -- not read 5 m against -5 m."""
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])  # test levels 0/10/25/50/100 m down
    depth = np.array([3.0, 27.0, 90.0])  # closest to 0, 25, 100 m respectively
    test, reference, _ = A.match_axis(
        _gridded_test(z), _station_reference(depth), over="Z"
    )
    expected = 20.0 - 0.1 * np.array([0.0, 25.0, 100.0])  # the test's own snapped levels
    got = test.sel(lon=-94.0, lat=25.0).values
    np.testing.assert_allclose(got, expected, atol=0.2)


def test_the_output_keeps_the_references_own_literal_values():
    """Even if the reference itself were negative-down, the output matches it exactly."""
    z = -np.array([0.0, 20.0, 50.0])
    # a reference recorded negative-down too (unusual, but must not be assumed away) --
    # "lev", a real fallback-recognized vertical name distinct from the test's own "z"
    depth = -np.array([5.0, 30.0])
    reference = xr.DataArray(
        20.0 - 0.1 * np.abs(depth), dims=("lev",), coords={"lev": depth}
    ).assign_coords(lon=-94.5, lat=25.5)
    test, _, _ = A.match_axis(_gridded_test(z), reference, over="Z")
    assert list(test["lev"].values) == list(depth)


# -- targets outside the test's own range come back NaN, not extrapolated --------------


def test_a_reference_level_beyond_the_tests_range_is_nan():
    z = -np.array([0.0, 10.0, 25.0])
    depth = np.array([5.0, 500.0])  # 500 m is far past the test's 25 m floor
    test, _, _ = A.match_axis(_gridded_test(z), _station_reference(depth), over="Z")
    sampled = test.sel(lon=-94.0, lat=25.0)
    assert np.isfinite(sampled.isel(DEPTH=0))
    assert not np.isfinite(sampled.isel(DEPTH=1))


def test_no_overlap_at_all_warns():
    z = -np.array([0.0, 10.0, 25.0])
    depth = np.array([500.0, 600.0])  # entirely beyond the test's range
    with pytest.warns(UserWarning, match="nothing finite"):
        A.match_axis(_gridded_test(z), _station_reference(depth), over="Z")


# -- an axis present but empty (a fanned bin with no data) is skippable, not a crash ----


def test_an_empty_reference_axis_raises_no_valid_data_not_a_bare_value_error():
    """A fanned per-bin comparison (e.g. times= against a station with no cast that
    month) can leave the axis present but with nothing on it. That must come back as
    NoValidData -- which compare(skip_missing=True) already catches and skips -- not
    reach _match_vertical's "nothing finite" warning, whose np.nanmin/nanmax on a
    zero-size array raises an unrelated, uncaught ValueError.
    """
    z = -np.array([0.0, 10.0, 25.0])
    depth = np.array([])  # no casts survived this bin
    with pytest.raises(A.NoValidData, match="empty"):
        A.match_axis(_gridded_test(z), _station_reference(depth), over="Z")


def test_an_empty_test_axis_also_raises_no_valid_data():
    z = np.array([])  # the test lane itself has nothing on this axis
    depth = np.array([5.0, 20.0])
    with pytest.raises(A.NoValidData, match="empty"):
        A.match_axis(_gridded_test(z), _station_reference(depth), over="Z")


# -- native s-coordinates (no metres coordinate) are refused with a clear hint ---------


def test_native_s_levels_on_the_test_side_are_refused_with_a_hint():
    bare = xr.DataArray(
        np.linspace(5.0, 20.0, 10)[:, None, None] * np.ones((10, 2, 2)),
        dims=("s_rho", "lat", "lon"),
        coords={"lat": LAT, "lon": LON},  # s_rho itself carries no coordinate
    )
    depth = np.array([5.0, 20.0])
    with pytest.raises(ValueError, match="native s-coordinates"):
        A.match_axis(bare, _station_reference(depth), over="Z")


# -- a real metres coordinate under a different name than its dimension is still found -


def _mismatched_name_reference(depth, *, lon=-94.5, lat=25.5):
    """A station-shaped lane like an OceanSITES/EMODnet ADCP mooring: the vertical
    *dimension* is ``DEPTH``, but it has no same-named coordinate -- the real
    metres live in a coordinate named lowercase ``depth`` riding on that dimension
    (``xarray`` reports this as "Dimensions without coordinates: DEPTH").
    """
    values = 20.0 - 0.1 * depth
    return (
        xr.DataArray(values, dims=("DEPTH",), attrs={"units": "degC"})
        .assign_coords(depth=("DEPTH", depth))
        .assign_coords(lon=lon, lat=lat)
    )


def test_a_metres_coordinate_under_a_different_name_than_its_dimension_still_matches():
    """The vocab-backed lookup (``operators.vertical_coord_on``) finds a real 1-D
    metres coordinate riding under a different name than its dimension -- it is
    not refused as "native s-coordinates" the way a genuine s_rho column is.
    """
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    matched_test, matched_reference, report = A.match_axis(
        _gridded_test(z), _mismatched_name_reference(depth), over="Z"
    )
    assert matched_test.dims == ("DEPTH", "lat", "lon")
    assert matched_reference.dims == ("DEPTH",)
    assert list(matched_test["DEPTH"].values) == list(depth)
    assert report["axis"] == "DEPTH"
    assert report["match_method"] == "nearest"


def test_a_metres_coordinate_under_a_different_name_can_interpolate_too():
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    _, _, report = A.match_axis(
        _gridded_test(z),
        _mismatched_name_reference(depth),
        over="Z",
        depth_method="interp",
    )
    assert report["match_method"] == "interp"


# -- full align(), through the station branch -------------------------------------------


def test_align_over_Z_against_a_station_produces_a_profile_shaped_result():
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    out = A.align(_gridded_test(z), _station_reference(depth), over="Z", method="nearest")
    assert set(out.data_vars) >= {"test", "reference", "difference"}
    assert out["reference"].dims == ("DEPTH",)
    assert out["test"].dims == ("DEPTH",)
    assert out.attrs["scored_over"] == "DEPTH"
    assert out.attrs["match_method"] == "nearest"
    assert "station_lon" in out.attrs and "station_lat" in out.attrs


def test_align_over_Z_can_interpolate_explicitly():
    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    out = A.align(
        _gridded_test(z),
        _station_reference(depth),
        over="Z",
        method="nearest",
        depth_method="interp",
    )
    assert out.attrs["match_method"] == "interp"


def test_align_over_Z_does_not_warn_about_missing_depth():
    """_warn_if_depths_differ must recognize the matched (possibly uppercase) axis."""
    import warnings

    z = -np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    depth = np.array([5.0, 20.0, 60.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        A.align(_gridded_test(z), _station_reference(depth), over="Z", method="nearest")
