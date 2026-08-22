"""The spatial regrid lands on the *coarser* grid, whichever lane owns it.

The time axis has always chosen direction by resolution (a finer test is averaged
into the coarser reference's bins); space used to target the reference
unconditionally, so a coarse model scored against a 4 km satellite product was
*upsampled* onto millions of satellite cells — inventing subgrid structure and
paying for it. These tests pin the decision (:func:`ocean_skill.align._regrid_target`),
the hysteresis that keeps near-equal grids on the reference, the ``target=``
override, and that the lanes keep their identities — ``difference`` is
test − reference no matter which lane moved.
"""

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as A


def _grid(lon0, lon1, nlon, lat0, lat1, nlat, value=5.0, units="mmol/m^3"):
    """Build a regular lat/lon lane with a constant value."""
    lon = np.linspace(lon0, lon1, nlon)
    lat = np.linspace(lat0, lat1, nlat)
    return xr.DataArray(
        np.full((lat.size, lon.size), float(value)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": units},
    )


def _fine_test(value=10.0):
    """A quarter-degree model-like lane.

    Kept within ±180 (rather than the more model-typical 0-360) so its natural
    convention is a no-op: ``align()`` harmonizes both lanes to whichever convention
    keeps the test contiguous (:func:`ocean_skill.align.natural_convention`), and a
    domain that ties between the two keeps ±180. A 0-360 domain here would come back
    relabeled, breaking the literal-coordinate assertions below for no reason the
    test cares about; the antimeridian-specific tests further down use a genuinely
    straddling domain instead.
    """
    return _grid(-160.0, -140.0, 81, 10.0, 30.0, 81, value=value)


def _coarse_test(value=10.0):
    """A two-degree model-like lane."""
    return _grid(-160.0, -140.0, 11, 10.0, 30.0, 11, value=value)


def _fine_reference(value=4.0):
    """A quarter-degree satellite-like lane, wider than the test."""
    return _grid(-170.0, -130.0, 161, 0.0, 40.0, 161, value=value)


def _coarse_reference(value=4.0):
    """A two-degree climatology-like lane, wider than the test."""
    return _grid(-170.0, -130.0, 21, 0.0, 40.0, 21, value=value)


# -- the decision ---------------------------------------------------------------------


def test_reference_coarser_keeps_the_reference_as_the_frame():
    """The historical case, asserted unchanged: fine model onto coarse climatology."""
    out = A.align(_fine_test(), _coarse_reference(), method="conservative_normed")
    assert out.attrs["regrid_target"] == "reference"
    # the output grid is the (cropped) reference's two-degree axis
    assert float(np.median(np.diff(out.lon))) == pytest.approx(2.0)
    assert np.isfinite(out["test"]).any()
    assert float(out["difference"].mean()) == pytest.approx(6.0, abs=1e-6)


def test_test_coarser_pulls_the_reference_onto_the_test_grid():
    """The new case: a fine satellite reference lands on the coarse model grid."""
    test, reference = _coarse_test(), _fine_reference()
    out = A.align(test, reference, method="conservative_normed")
    assert out.attrs["regrid_target"] == "test"
    # the output lives on the test's own grid, untouched
    assert out.sizes == {"lat": 11, "lon": 11}
    np.testing.assert_array_equal(out.lon, test.lon)
    np.testing.assert_array_equal(out.lat, test.lat)
    xr.testing.assert_allclose(out["test"], test.rename())
    assert "regridded onto the test's coarser grid" in out.attrs["regrid_reason"]


def test_difference_stays_test_minus_reference_when_the_reference_moves():
    """Regridding the other lane must not flip the sign or the labels."""
    out = A.align(
        _coarse_test(value=10.0),
        _fine_reference(value=4.0),
        method="conservative_normed",
        test_name="model",
        reference_name="satellite",
    )
    assert out.attrs["regrid_target"] == "test"
    inner = out["difference"].isel(lat=slice(2, -2), lon=slice(2, -2))
    assert float(inner.mean()) == pytest.approx(10.0 - 4.0, abs=1e-6)
    assert out["difference"].attrs["long_name"] == "model − satellite"
    assert float(out["model"].mean()) == pytest.approx(10.0)


def test_near_equal_grids_keep_the_reference_without_flip_flopping():
    """Hysteresis: within COARSER_BY the frames are the same and the default holds."""
    # test cells ~1.43x the reference's (COARSER_BY is 1.5) -- coarser, but not
    # materially so, so the reference must stay the frame rather than flip
    test = _grid(-160.0, -140.0, 8, 10.0, 30.0, 8, value=10.0)
    out = A.align(test, _coarse_reference(), method="bilinear")
    assert out.attrs["regrid_target"] == "reference"


def test_target_override_forces_either_direction():
    """target= mirrors convention=: the auto choice, made by hand."""
    test, reference = _coarse_test(), _fine_reference()
    onto_ref = A.align(
        test, reference, method="conservative_normed", target="reference"
    )
    assert onto_ref.attrs["regrid_target"] == "reference"
    assert onto_ref.attrs["regrid_reason"] == "target='reference' as asked"
    assert float(np.median(np.diff(onto_ref.lon))) == pytest.approx(0.25)

    onto_test = A.align(
        _fine_test(), _coarse_reference(), method="conservative_normed", target="test"
    )
    assert onto_test.attrs["regrid_target"] == "test"
    assert onto_test.sizes == {"lat": 81, "lon": 81}

    with pytest.raises(ValueError, match="unknown target"):
        A.align(test, reference, target="the finer one")


# -- coverage on the new path ---------------------------------------------------------


def test_coverage_masks_the_regridded_reference_on_the_test_grid():
    """min_coverage still means the *source's* coverage of each target cell."""
    reference = _fine_reference()
    # hole in the satellite data over part of the domain, as clouds would leave
    reference = reference.where(
        ~((reference.lat > 18) & (reference.lat < 22) & (reference.lon > -155)
          & (reference.lon < -145))
    )
    out = A.align(
        _coarse_test(), reference, method="conservative_normed", min_coverage=0.5
    )
    assert out.attrs["regrid_target"] == "test"
    assert out["coverage"].sizes == {"lat": 11, "lon": 11}
    assert (
        out["coverage"].attrs["long_name"]
        == "fraction of the test cell covered by valid reference data"
    )
    # cells inside the hole lose their reference value; the test lane keeps its own
    hole = out["reference"].sel(lat=20.0, lon=-150.0, method="nearest")
    assert not np.isfinite(hole)
    assert float(out["test"].sel(lat=20.0, lon=-150.0, method="nearest")) == 10.0


# -- the antimeridian, in the new direction -------------------------------------------


def _coarse_pacific_test(ny=12, nx=20):
    """A curvilinear dateline-straddling lane, 0-360 native and ~5 degrees coarse."""
    lon = np.linspace(150.0, 250.0, nx)[None, :] * np.ones((ny, 1))
    lat = np.linspace(-20.0, 20.0, ny)[:, None] * np.ones((1, nx))
    return xr.DataArray(
        np.cos(np.deg2rad(lat)) * 10 + 5,
        dims=("eta", "xi"),
        coords={"lon": (("eta", "xi"), lon), "lat": (("eta", "xi"), lat)},
        attrs={"units": "mmol/m^3"},
    )


def _fine_global_reference(step=1.0):
    """A one-degree global grid in ±180, satellite-L3-style."""
    lat = np.arange(-89.5, 90.0, step)
    lon = np.arange(-179.5, 180.0, step)
    return xr.DataArray(
        np.full((lat.size, lon.size), 5.0),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "mmol/m^3"},
    )


@pytest.mark.parametrize("method", ["bilinear", "conservative_normed"])
def test_a_fine_global_reference_lands_on_the_straddling_test_grid(method):
    """The Pacific case with the new direction: convention and crop still hold."""
    test = _coarse_pacific_test()
    out = A.align(test, _fine_global_reference(), method=method)
    assert out.attrs["regrid_target"] == "test"
    assert out.attrs["lon_convention"] == "0-360"
    # the pair lives on the test's own curvilinear grid, inside its own band
    assert out.sizes == {"eta": 12, "xi": 20}
    assert float(out.lon.min()) == pytest.approx(150.0)
    assert float(out.lon.max()) == pytest.approx(250.0)
    # the regridded reference holds its own sane value across the domain
    r = out["reference"]
    assert np.isfinite(r).any()
    inner = r.isel(eta=slice(1, -1), xi=slice(1, -1))
    finite = inner.values[np.isfinite(inner.values)]
    np.testing.assert_allclose(finite, 5.0, atol=1e-6)
    # and the difference is still test − reference on that grid
    xr.testing.assert_allclose(
        out["difference"], (out["test"] - out["reference"]).rename("difference")
    )
