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


# -- a NaN source cell (land) no longer poisons a partially-covered destination cell --


def _test_with_land_band(cutoff_lat=20.0, value=10.0):
    """The fine test lane with everything at or north of ``cutoff_lat`` masked.

    The coarse reference cell centered on 20.0 (:func:`_coarse_reference`'s 2-degree
    grid) spans roughly 19-21, so this band leaves it genuinely partly covered --
    neither fully valid nor fully land -- which is exactly the coastal case the bug
    report was about.
    """
    test = _fine_test(value=value)
    return test.where(test.lat < cutoff_lat)


def test_a_partly_land_covered_cell_stays_finite_not_nan():
    """The bug report: a NaN source cell must not poison a cell it only partly covers."""
    out = A.align(
        _test_with_land_band(), _coarse_reference(),
        method="conservative_normed", min_coverage=1e-9,
    )
    partly_covered = out["test"].sel(lat=20.0, lon=-150.0, method="nearest")
    assert np.isfinite(partly_covered)
    # conservative_normed renormalizes over the valid fraction -- the mean of a
    # constant field is that same constant, not something smaller
    assert float(partly_covered) == pytest.approx(10.0)
    # a cell entirely inside the land band is still correctly NaN -- this isn't
    # "nothing is ever masked now", only "a NaN cell no longer poisons a cell it
    # merely overlaps"
    fully_land = out["test"].sel(lat=24.0, lon=-150.0, method="nearest")
    assert not np.isfinite(fully_land)


def test_coverage_is_a_true_fraction_not_nan_or_one():
    """``coverage`` must report how much of a cell is valid, not NaN-or-1."""
    out = A.align(
        _test_with_land_band(), _coarse_reference(),
        method="conservative_normed", min_coverage=1e-9,
    )
    open_ocean = out["coverage"].sel(lat=14.0, lon=-150.0, method="nearest")
    assert float(open_ocean) == pytest.approx(1.0)
    partly_covered = out["coverage"].sel(lat=20.0, lon=-150.0, method="nearest")
    assert 0.0 < float(partly_covered) < 1.0
    fully_land = out["coverage"].sel(lat=24.0, lon=-150.0, method="nearest")
    assert float(fully_land) == pytest.approx(0.0)


def test_min_coverage_threshold_still_drops_a_too_thinly_covered_cell():
    """The renormalized value is real, but still gated by ``min_coverage``."""
    common = dict(method="conservative_normed")
    strict = A.align(_test_with_land_band(), _coarse_reference(), min_coverage=0.5, **common)
    lenient = A.align(_test_with_land_band(), _coarse_reference(), min_coverage=0.3, **common)
    cell = dict(lat=20.0, lon=-150.0, method="nearest")
    assert not np.isfinite(strict["test"].sel(**cell))
    assert np.isfinite(lenient["test"].sel(**cell))


def test_plain_conservative_is_exempt_from_the_renormalization():
    """A NaN source cell still poisons plain ``"conservative"`` -- unlike ``_normed``.

    Renormalizing it too would silently turn plain ``"conservative"`` into
    ``"conservative_normed"`` even on fully valid data (skipna divides by the row's
    own weight sum, which for ``"conservative"`` is the geometric overlap fraction,
    not 1) -- verified separately against installed xesmf. Here: on data with a NaN
    land band, the two methods must disagree at the partly-covered cell.
    """
    normed = A.align(
        _test_with_land_band(), _coarse_reference(),
        method="conservative_normed", min_coverage=1e-9,
    )
    plain = A.align(
        _test_with_land_band(), _coarse_reference(),
        method="conservative", min_coverage=1e-9,
    )
    cell = dict(lat=20.0, lon=-150.0, method="nearest")
    assert np.isfinite(normed["test"].sel(**cell))
    assert not np.isfinite(plain["test"].sel(**cell))


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


# -- the same-grid bypass --------------------------------------------------------------


def test_identical_grids_skip_the_regridder_entirely(monkeypatch):
    """Two runs of the same model share a grid; nothing should be built or moved."""
    import xesmf

    def _no_regridder(*a, **k):
        raise AssertionError("xesmf.Regridder should not be called for identical grids")

    monkeypatch.setattr(xesmf, "Regridder", _no_regridder)
    test, reference = _fine_test(value=10.0), _fine_test(value=4.0)
    out = A.align(test, reference, method="conservative_normed")
    assert out.attrs["regrid_target"] == "none"
    assert "share one grid" in out.attrs["regrid_reason"]
    assert "coverage" not in out
    np.testing.assert_array_equal(out.lon, test.lon)
    np.testing.assert_array_equal(out.lat, test.lat)
    assert float(out["difference"].mean()) == pytest.approx(6.0, abs=1e-9)


def test_identical_grids_bypass_survives_a_longitude_convention_difference(monkeypatch):
    """The same grid, spelled in a different longitude convention, still bypasses."""
    import xesmf

    def _no_regridder(*a, **k):
        raise AssertionError("xesmf.Regridder should not be called for identical grids")

    monkeypatch.setattr(xesmf, "Regridder", _no_regridder)
    test = _fine_test(value=10.0)
    # the same cells, re-expressed in 0-360 -- align()'s own convention harmonizing
    # brings this back onto test's -160..-140 axis before the grid check runs
    reference = test.copy(data=np.full_like(test.values, 4.0))
    reference = reference.assign_coords(lon=("lon", np.asarray(test.lon) + 360.0))
    out = A.align(test, reference, method="conservative_normed")
    assert out.attrs["regrid_target"] == "none"
    assert float(out["difference"].mean()) == pytest.approx(6.0, abs=1e-9)


def test_near_equal_but_different_grids_do_not_bypass():
    """Off-by-one-cell grids are a genuinely different grid, not a shared one."""
    test = _grid(-160.0, -140.0, 8, 10.0, 30.0, 8, value=10.0)
    reference = _grid(-160.0, -140.0, 9, 10.0, 30.0, 9, value=4.0)
    out = A.align(test, reference, method="bilinear")
    assert out.attrs["regrid_target"] == "reference"
    assert "share one grid" not in out.attrs["regrid_reason"]
