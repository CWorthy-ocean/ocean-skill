"""Units of a statistic: a variance is squared, a spread is a difference.

``operators.aggregate`` leaves ``attrs["statistic"]`` on its result and runs the units
through :func:`ocean_skill.units.for_statistic`. These tests pin the three things that
goes wrong otherwise: the strings must be valid pint and survive a round trip; a
standard deviation, range or variance of a temperature must never pick up the 273.15 of
a K <-> degC conversion; and a variance of a per-volume concentration must pick up the
seawater density *squared* when it is expressed per mass. Finally the pinned chlorophyll
display range (0.01-10, log) must not be applied to the variance of chlorophyll.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as _align
from ocean_skill import units as u
from ocean_skill.colormaps import is_log, norm_for

CHL = "mass_concentration_of_chlorophyll_a_in_sea_water"


def _da(value, units, statistic=None):
    attrs = {"units": units}
    if statistic is not None:
        attrs["statistic"] = statistic
    return xr.DataArray(np.atleast_1d(np.asarray(value, dtype=float)), attrs=attrs)


@pytest.mark.parametrize(
    ("units", "statistic", "expected"),
    [
        ("degC", "var", "delta_degC^2"),
        ("degC", "variance", "delta_degC^2"),
        ("degC", "std", "delta_degC"),
        ("degC", "range", "delta_degC"),
        ("degF", "ptp", "delta_degF"),
        ("K", "var", "K^2"),
        ("K", "std", "K"),
        ("mmol/m^3", "var", "(mmol/m^3)^2"),
        ("mmol m-3", "var", "(mmol m-3)^2"),
        ("mg/m^3", "std", "mg/m^3"),
        ("mg/m^3", "range", "mg/m^3"),
        ("mg/m^3", "mean", "mg/m^3"),
        ("mg/m^3", "max", "mg/m^3"),
        ("degC", "mean", "degC"),
        ("degC", "median", "degC"),
        ("degC", "quantile", "degC"),
        ("degC", None, "degC"),
        ("degC", "VAR", "delta_degC^2"),
        ("1e-3", "var", "1e-06"),
    ],
)
def test_for_statistic_table(units, statistic, expected):
    assert u.for_statistic(units, statistic) == expected


@pytest.mark.parametrize("statistic", ["var", "std", "mean", None])
def test_empty_and_missing_units_pass_through(statistic):
    assert u.for_statistic(None, statistic) is None
    assert u.for_statistic("", statistic) == ""


def test_dimensionless_variance_is_still_dimensionless():
    assert u.compatible(u.for_statistic("dimensionless", "var"), "1")


def test_unparseable_units_do_not_crash():
    assert u.for_statistic("furlongs of fog", "var") == "(furlongs of fog)^2"
    assert u.for_statistic("furlongs of fog", "std") == "furlongs of fog"
    assert u.parse(u.for_statistic("furlongs of fog", "var")) is None


@pytest.mark.parametrize(
    "units", ["degC", "degF", "K", "mmol/m^3", "umol kg-1", "mg/m^3"]
)
@pytest.mark.parametrize("statistic", ["var", "std", "range"])
def test_statistic_units_round_trip_through_pint(units, statistic):
    out = u.for_statistic(units, statistic)
    assert u.parse(out) is not None
    assert u.parse(u.normalize(out)) == u.parse(out)
    # and the dimensions are what the statistic says they are
    power = 2 if statistic == "var" else 1
    assert u.parse(out).dimensionality == u.parse(units).dimensionality ** power


def test_var_of_degC_converts_to_K_squared_without_a_shift():
    var = _da([1.0, 4.0], u.for_statistic("degC", "var"), "var")
    out = u.to_units(var, "K^2")
    np.testing.assert_allclose(out.values, [1.0, 4.0])
    assert out.attrs["units"] == "K^2"


def test_var_in_K_squared_converts_to_delta_degF_squared_by_the_square_of_nine_fifths():
    out = u.to_units(_da(1.0, "K^2", "var"), u.for_statistic("degF", "var"))
    np.testing.assert_allclose(out.values, 1.8**2)


@pytest.mark.parametrize("statistic", ["std", "range", "ptp", "mad"])
def test_std_in_K_to_degC_is_not_shifted(statistic):
    out = u.to_units(_da([0.3, 2.0], "K", statistic), "degC")
    np.testing.assert_allclose(out.values, [0.3, 2.0])


def test_std_stored_as_plain_degC_still_does_not_shift_into_K():
    """A std that never went through for_statistic is still caught by its attribute."""
    out = u.to_units(_da(0.5, "degC", "std"), "K")
    np.testing.assert_allclose(out.values, 0.5)


def test_delta_degC_std_against_a_reference_std_written_plain_degC():
    """A product's own std field says ``degC``; pint refuses delta -> degC outright."""
    out = u.to_units(_da(0.5, "delta_degC", "std"), "degC")
    np.testing.assert_allclose(out.values, 0.5)


def test_a_mean_in_degC_still_shifts():
    """The guard must not turn the ordinary offset conversion into a scale."""
    np.testing.assert_allclose(
        u.to_units(_da(10.0, "degC", "mean"), "K").values, 283.15
    )
    np.testing.assert_allclose(u.to_units(_da(10.0, "degC"), "K").values, 283.15)


def test_variance_uses_density_squared_per_volume_to_per_mass():
    rho = u.RHO_SEAWATER
    var = _da(4.0, u.for_statistic("mmol/m^3", "var"), "var")  # 4 (mmol/m^3)^2
    out = u.to_units(var, "(umol/kg)^2")
    # 1 mmol/m^3 = 1000 umol / rho kg  ->  squared: (1000/rho)^2
    np.testing.assert_allclose(out.values, 4.0 * (1000.0 / rho) ** 2)
    back = u.to_units(out, "(mmol/m^3)^2")
    np.testing.assert_allclose(back.values, 4.0)


def test_std_uses_density_once():
    std = _da(2.0, "mmol/m^3", "std")
    out = u.to_units(std, "umol/kg")
    np.testing.assert_allclose(out.values, 2.0 * 1000.0 / u.RHO_SEAWATER)


def test_compatible_across_squared_and_delta_units():
    assert u.compatible("(mmol/m^3)^2", "(umol/kg)^2") is True
    assert u.compatible("delta_degC^2", "K^2") is True
    assert u.compatible("delta_degC", "K") is True
    # a variance is not the quantity it is a variance of
    assert u.compatible("(mmol/m^3)^2", "mmol/m^3") is False
    assert u.compatible("K^2", "K") is False


def test_convert_units_converts_a_variance_to_the_target_convention():
    var = _da(1.0, "(umol/kg)^2", "var")
    out = u.convert_units(var, "(mmol/m^3)^2")
    np.testing.assert_allclose(out.values, (u.RHO_SEAWATER / 1000.0) ** 2)


def test_check_units_aligns_a_variance_lane_and_refuses_a_mismatch():
    ref = _da(1.0, "K^2", "var")
    out = _align._check_units(_da(1.0, "delta_degC^2", "var"), ref)
    np.testing.assert_allclose(out.values, 1.0)
    with pytest.raises(ValueError, match="not the same physical quantity"):
        _align._check_units(_da(1.0, "K", "mean"), ref)


def test_check_units_does_not_shift_a_std_against_a_kelvin_reference():
    out = _align._check_units(_da(0.4, "degC", "std"), _da(0.3, "K", "std"))
    np.testing.assert_allclose(out.values, 0.4)


@pytest.mark.parametrize(
    ("unit_string", "label"),
    [
        ("delta_degC^2", "°C²"),
        ("delta_degC", "°C"),
        ("degF", "°F"),
        ("K^2", "K²"),
        ("(mg/m^3)^2", "(mg/m³)²"),
        ("(mmol/m^3)^2", "(mmol/m³)²"),
        ("1/s^-1", "1/s⁻¹"),
        ("mg/m^3", "mg/m³"),
        ("psu", "psu"),
        ("", ""),
        (None, ""),
    ],
)
def test_display_is_readable(unit_string, label):
    assert u.display(unit_string) == label


def test_is_spread():
    assert all(
        u.is_spread(s) for s in ("var", "variance", "std", "range", "ptp", "Std")
    )
    assert not any(
        u.is_spread(s) for s in ("mean", "max", "min", "median", None, "sum")
    )


def test_chlorophyll_variance_skips_the_pinned_range_and_log_scale():
    assert is_log(CHL) is True
    assert is_log(CHL, statistic="mean") is True
    for statistic in ("var", "std", "range"):
        assert is_log(CHL, statistic=statistic) is False
    pinned = norm_for(CHL, 1e-5, 3e-4)
    assert (pinned.vmin, pinned.vmax) == (0.01, 10.0)
    assert type(pinned).__name__ == "LogNorm"
    spread = norm_for(CHL, 1e-5, 3e-4, statistic="var")
    assert type(spread).__name__ == "Normalize"
    # a spread reads from zero (cmo.amp's white end is "no spread")
    assert (spread.vmin, spread.vmax) == (0.0, 3e-4)
    # an explicit user limit still wins
    assert norm_for(CHL, 1e-5, 3e-4, user_vmax=1.0, statistic="var").vmax == 1.0


def test_a_per_spelled_unit_is_bracketed_when_squared():
    """``micromoles_per_kilogram`` normalizes to a quotient, so its square needs brackets."""
    squared = u.for_statistic("micromoles_per_kilogram", "var")
    assert squared == "(micromoles_per_kilogram)^2"
    assert u.compatible(squared, "(mmol/m^3)^2")


def test_convert_units_moves_a_variance_to_the_square_of_a_plain_target():
    """The default per-lane target is a plain unit; a variance lands in its square."""
    var = _da(1.0, u.for_statistic("micromoles_per_kilogram", "var"), "var")
    out = u.convert_units(var)  # default target mmol/m^3
    assert out.attrs["units"] == "(mmol/m^3)^2"
    np.testing.assert_allclose(out.values, (u.RHO_SEAWATER / 1000.0) ** 2)
