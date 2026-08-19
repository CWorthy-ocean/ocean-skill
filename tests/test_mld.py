"""Tests for mixed layer depth (threshold methods): ocean_skill/mld.py.

Three layers, tested separately: the pure per-column crossing scan
(:func:`ocean_skill.mld._mld_threshold_1d`, hand-worked so the exact interpolated
answer is checkable by arithmetic), the vectorized xarray wrapper and calculators on
a synthetic water column, and the wiring into :func:`ocean_skill.operators.resolve_variable`
and :func:`ocean_skill.comparison._prepare`.

The density path is checked against gsw itself rather than a hand-derived number:
TEOS-10 potential density has no simpler formula to work out by hand, so the test
computes sigma0 independently with gsw and feeds it through the same crossing scan,
then asserts :func:`ocean_skill.mld.mld_density_threshold` reproduces that number.
That still catches real bugs -- a flipped pressure sign, the wrong SA/CT chaining
order, a broadcasting mistake -- without re-deriving TEOS-10.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import mld
from ocean_skill.comparison import _prepare
from ocean_skill.operators import resolve_variable, spec_names

# -- the per-column scan, hand-worked -----------------------------------------
#
# depths (positive-down) and temperatures for one column with a clean crossing:
# uniform 20 degC down to 10 m (with a warmer 21 degC skin cell above it, which a
# ref_depth=10 lookup never sees), cooling below. Worked by hand for ref_depth=10,
# threshold=0.2:
#   ref_value = 20.0 (the profile is exactly 20.0 at d=10)
#   diff = temp - 20.0 = [0(2m,ignored), 0(6m,ignored), 0, -0.1, -1.0, -5.0]
#   |diff| > 0.2 first at d=40 (diff=-1.0); interpolating from d=20 (diff=-0.1):
#   frac = (-0.2 - -0.1) / (-1.0 - -0.1) = 1/9  ->  mld = 20 + (1/9)*20 = 22.2222 m
DEPTH = np.array([2.0, 6.0, 10.0, 20.0, 40.0, 80.0])
TEMP = np.array([21.0, 20.0, 20.0, 19.9, 19.0, 15.0])
CROSSING_MLD = 20.0 + (1 / 9) * 20.0


def test_threshold_crossing_interpolates_between_levels():
    """The headline case: linear interpolation, not a snap to the nearer level."""
    out = mld._mld_threshold_1d(TEMP, DEPTH, threshold=0.2, ref_depth=10.0)
    assert out == pytest.approx(CROSSING_MLD)


def test_column_order_does_not_matter():
    """ROMS stores s_rho bottom-to-top -- depth *decreasing* with index -- so the
    internal sort has to actually do something, not rely on already-sorted input."""
    out = mld._mld_threshold_1d(TEMP[::-1], DEPTH[::-1], threshold=0.2, ref_depth=10.0)
    assert out == pytest.approx(CROSSING_MLD)


def test_a_shallow_column_is_nan():
    """A column that never reaches ref_depth has no reference value to measure from."""
    depth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # h = 6 m < ref_depth
    out = mld._mld_threshold_1d(np.full(6, 20.0), depth, threshold=0.2, ref_depth=10.0)
    assert np.isnan(out)


def test_a_fully_mixed_column_is_nan():
    """No crossing within the resolved column -- not a wrong answer, an unresolved one."""
    out = mld._mld_threshold_1d(np.full(6, 20.0), DEPTH, threshold=0.2, ref_depth=10.0)
    assert np.isnan(out)


def test_threshold_override_shifts_the_crossing():
    """threshold=1.0 exactly reaches the diff at 40 m, so the crossing lands there."""
    out = mld._mld_threshold_1d(TEMP, DEPTH, threshold=1.0, ref_depth=10.0)
    assert out == pytest.approx(40.0)


def test_ref_depth_override_changes_the_reference_value():
    """ref_depth=2 measures from the warmer skin cell the default ref_depth ignores."""
    # ref_value becomes 21.0; diff = [0, -1.0, -1.0, -1.1, -2.0, -6.0]; |diff|>0.2
    # first at d=6 (diff=-1.0) from d=2 (diff=0): frac=0.2 -> mld = 2 + 0.2*4 = 2.8
    out = mld._mld_threshold_1d(TEMP, DEPTH, threshold=0.2, ref_depth=2.0)
    assert out == pytest.approx(2.8)


# -- the xarray wrapper and calculators, on a synthetic water column ----------


def _water_column(depths, temps, salt=35.0, lon=-90.0, lat=25.0):
    """Build a minimal (s_rho, eta_rho=1, xi_rho=1) column already carrying z_rho.

    Bypasses roms.add_depth_coord's sigma/Cs_r/h reconstruction entirely -- the
    calculators only ever read ``z_rho`` off the dataset, so handing them one
    directly keeps the fixture's depths exact and readable.
    """
    depths = np.asarray(depths, dtype=float)
    temps = np.asarray(temps, dtype=float)
    shape = (depths.size, 1, 1)
    return xr.Dataset(
        {
            "sea_water_potential_temperature": (
                ("s_rho", "eta_rho", "xi_rho"),
                temps[::-1].reshape(shape),
            ),
            "sea_water_practical_salinity": (
                ("s_rho", "eta_rho", "xi_rho"),
                np.full(shape, salt),
            ),
        },
        coords={
            "z_rho": (("s_rho", "eta_rho", "xi_rho"), (-depths[::-1]).reshape(shape)),
            "lon": (("eta_rho", "xi_rho"), np.full((1, 1), lon)),
            "lat": (("eta_rho", "xi_rho"), np.full((1, 1), lat)),
        },
    )


def _stack(*columns: xr.Dataset) -> xr.Dataset:
    """Concatenate single-column fixtures along xi_rho, for one vectorized check."""
    return xr.concat(columns, dim="xi_rho")


def test_mld_temperature_threshold_matches_the_hand_worked_column():
    ds = _stack(
        _water_column(DEPTH, TEMP),  # crossing at CROSSING_MLD
        _water_column([1, 2, 3, 4, 5, 6], np.full(6, 20.0)),  # shallow
        _water_column(DEPTH, np.full(6, 20.0)),  # fully mixed
    )
    out = mld.mld_temperature_threshold(ds)
    assert "s_rho" not in out.dims
    assert float(out.isel(eta_rho=0, xi_rho=0)) == pytest.approx(CROSSING_MLD)
    assert np.isnan(float(out.isel(eta_rho=0, xi_rho=1)))
    assert np.isnan(float(out.isel(eta_rho=0, xi_rho=2)))


def test_mld_attrs_record_the_method_and_parameters():
    out = mld.mld_temperature_threshold(_water_column(DEPTH, TEMP), threshold=0.5)
    assert out.name == "ocean_mixed_layer_thickness"
    assert out.attrs["units"] == "m"
    assert out.attrs["standard_name"] == "ocean_mixed_layer_thickness"
    assert out.attrs["mld_method"] == "temperature_threshold"
    assert out.attrs["mld_threshold"] == 0.5
    assert out.attrs["mld_ref_depth"] == mld.REF_DEPTH


def test_mld_density_threshold_matches_gsw_computed_independently():
    """Ground truth from gsw itself, run through the same crossing scan by hand."""
    gsw = pytest.importorskip("gsw")

    temp_step = np.array([20.0, 20.0, 20.0, 20.0, 15.0, 15.0])
    salt, lon, lat = 35.0, -90.0, 25.0
    pressure = gsw.p_from_z(-DEPTH, lat)
    sa = gsw.SA_from_SP(salt, pressure, lon, lat)
    ct = gsw.CT_from_pt(sa, temp_step)
    sigma0 = gsw.sigma0(sa, ct)
    expected = mld._mld_threshold_1d(sigma0, DEPTH, threshold=0.03, ref_depth=10.0)
    assert np.isfinite(expected)  # the hand-picked step really does cross 0.03

    ds = _water_column(DEPTH, temp_step, salt=salt, lon=lon, lat=lat)
    out = mld.mld_density_threshold(ds)
    assert float(out.isel(eta_rho=0, xi_rho=0)) == pytest.approx(expected)


def test_calculate_mld_needs_an_explicit_method():
    """No default method: four exist in the product this is meant to match, and
    picking one silently would be the kind of looks-right number this project
    otherwise refuses to produce."""
    with pytest.raises(KeyError, match="mld needs method"):
        mld.calculate_mld(_water_column(DEPTH, TEMP))


def test_calculate_mld_names_the_hybrid_methods_explicitly():
    """Asking for a method that exists in Holte & Talley but not here says so."""
    ds = _water_column(DEPTH, TEMP)
    with pytest.raises(NotImplementedError, match="Holte"):
        mld.calculate_mld(ds, method="density_algorithm")
    with pytest.raises(NotImplementedError, match="Holte"):
        mld.calculate_mld(ds, method="temperature_algorithm")


def test_calculate_mld_rejects_an_unknown_method():
    with pytest.raises(KeyError, match="mld needs method"):
        mld.calculate_mld(_water_column(DEPTH, TEMP), method="bogus")


# -- wiring: operators.resolve_variable / spec_names / comparison._prepare ----


def test_resolve_variable_dispatches_to_the_mld_calculator():
    ds = _water_column(DEPTH, TEMP)
    spec = {"calculate": "mld", "method": "temperature_threshold"}
    out = resolve_variable(ds, spec)
    assert float(out.isel(eta_rho=0, xi_rho=0)) == pytest.approx(CROSSING_MLD)


def test_spec_names_reports_temperature_or_density_inputs_by_method():
    temp = "sea_water_potential_temperature"
    salt = "sea_water_practical_salinity"
    assert spec_names({"calculate": "mld", "method": "temperature_threshold"}) == [
        [temp]
    ]
    assert spec_names({"calculate": "mld", "method": "density_threshold"}) == [
        [temp, salt]
    ]
    # No method yet (spec still being built, or a typo) -- report the broader
    # requirement rather than nothing, since calculate_mld will fail loudly anyway.
    assert spec_names({"calculate": "mld"}) == [[temp, salt]]


def test_prepare_refuses_a_depth_selection_alongside_calculate():
    """MLD already collapses the vertical axis; select={'depth': ...} beside it is a
    contradiction worth naming rather than silently ignoring."""
    ds = _water_column(DEPTH, TEMP)
    spec = {"calculate": "mld", "method": "temperature_threshold"}
    with pytest.raises(ValueError, match="already reduces the vertical axis"):
        _prepare(ds, {"model": "roms"}, spec, {"depth": "surface"})


def test_prepare_accepts_calculate_with_no_depth_key():
    ds = _water_column(DEPTH, TEMP)
    spec = {"calculate": "mld", "method": "temperature_threshold"}
    da, actual_depth = _prepare(ds, {"model": "roms"}, spec, {})
    assert actual_depth is None
    assert float(da.isel(eta_rho=0, xi_rho=0)) == pytest.approx(CROSSING_MLD)
