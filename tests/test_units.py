"""Tests for pint-backed unit handling.

Three things matter here. Real CF/UDUNITS spellings must parse — the parametrized
list is taken from the actual WOA/GLODAP/MODIS/OOI catalogs, and pint's default
registry manages only a third of it unaided. Conversions must be right, including
the per-mass/per-volume one that needs seawater density. And mismatched units must
stop a comparison rather than quietly producing a difference that looks plausible.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as _align
from ocean_skill import units as u

#: Every unit spelling seen in the project's real catalogs and data files.
REAL_SPELLINGS = [
    "micro-mol kg-1",  # GLODAP
    "micromoles_per_kilogram",  # WOA
    "umol kg-1",
    "umol/kg",
    "mmol/m^3",  # ROMS/MARBL
    "mmol m-3",
    "mmol/m3",
    "meq/m^3",  # alkalinity
    "meq m-3",
    "mg m-3",  # chlorophyll
    "mg/m^3",
    "microg.L-1",  # OOI fluorometer
    "Celsius",
    "degrees celcius",  # sic, real misspelling
    "degrees_celsius",
    "degC",
    "PSU",
    "practical salinity units",
    "1e-3",  # WOA salinity
    "1",
    "m",
    "",
]


@pytest.mark.parametrize("spelling", REAL_SPELLINGS)
def test_every_real_spelling_parses(spelling):
    """Pint alone handles 7 of these; the rest is what normalize() is for."""
    assert u.parse(spelling) is not None, f"{spelling!r} did not parse"


def test_unparseable_units_return_none_rather_than_raising():
    """A spelling problem should degrade to "cannot check", not abort a comparison."""
    assert u.parse("not a unit at all $$") is None


def test_missing_units_are_not_dimensionless():
    """A variable that never recorded ``units`` (``None``) is unknown, not "".

    ``normalize(None)`` collapses to ``""`` -> "dimensionless", which used to make
    a units-less field parse as a real, empty unit -- indistinguishable from a
    variable that genuinely declared itself dimensionless.
    """
    assert u.parse(None) is None


# -- compatibility ------------------------------------------------------------


def test_per_mass_and_per_volume_are_compatible_through_density():
    """The conversion this module exists for; density makes them interconvertible."""
    assert u.compatible("umol/kg", "mmol/m^3") is True


def test_different_quantities_are_incompatible():
    """Mass concentration is not amount concentration, whatever the prefixes."""
    assert u.compatible("mg/m^3", "mmol/m^3") is False


def test_unknown_units_report_unknown_not_incompatible():
    """Three-valued on purpose: blocking on a spelling problem would be wrong."""
    assert u.compatible("$$", "mmol/m^3") is None


def test_a_missing_units_attr_also_reports_unknown():
    """Regression: a units-less to_depth() result made compatible(None, ...) return
    a hard False (via normalize(None) -> "dimensionless"), which made align()
    refuse a comparison it should have warned and proceeded with instead.
    """
    assert u.compatible(None, "mmol/m^3") is None


@pytest.mark.parametrize(
    ("spelling", "equivalent"),
    [
        ("micro-mol kg-1", "umol/kg"),
        ("micromoles_per_kilogram", "umol/kg"),
        ("mmol m-3", "mmol/m^3"),
        ("mmol/m3", "mmol/m^3"),
        ("mg m-3", "mg/m^3"),
    ],
)
def test_spellings_of_the_same_unit_agree(spelling, equivalent):
    assert u.parse(spelling) == u.parse(equivalent)


# -- conversion ---------------------------------------------------------------


def _field(value, units):
    return xr.DataArray(
        np.full((2, 2), value),
        dims=("lat", "lon"),
        coords={"lat": [1.0, 2.0], "lon": [1.0, 2.0]},
        attrs={"units": units},
    )


def test_per_mass_converts_to_per_volume_using_density():
    out = u.convert_units(_field(1.0, "micromoles_per_kilogram"))
    assert float(out.mean()) == pytest.approx(u.RHO_SEAWATER / 1000.0)
    assert out.attrs["units"] == "mmol/m^3"


def test_conversion_records_what_it_did():
    out = u.convert_units(_field(1.0, "umol/kg"))
    assert "unit_conversion" in out.attrs


def test_an_already_matching_unit_is_untouched():
    out = u.convert_units(_field(5.0, "mmol/m^3"))
    assert float(out.mean()) == 5.0


@pytest.mark.parametrize("units", ["mg m-3", "degC", "PSU", "m"])
def test_other_quantities_pass_through_silently(units):
    """Not being a nutrient is the normal case, not something to warn about.

    The string-matching version warned "unrecognized units" on every chlorophyll
    comparison, which trains people to ignore the warning that matters.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = u.convert_units(_field(1.0, units))
    assert not caught
    assert out.attrs["units"] == units


def test_to_units_leaves_unrelated_quantities_alone():
    out = u.to_units(_field(1.0, "degC"), "mmol/m^3")
    assert out.attrs["units"] == "degC"


# -- the align() gap ----------------------------------------------------------


def _pair(test_units, reference_units, value=100.0):
    coords = {"lat": np.linspace(10, 20, 4), "lon": np.linspace(-100, -90, 5)}
    make = lambda unit: xr.DataArray(  # noqa: E731
        np.full((4, 5), value),
        dims=("lat", "lon"),
        coords=coords,
        attrs={"units": unit},
    )
    return make(test_units), make(reference_units)


def test_align_refuses_to_difference_different_quantities():
    test, reference = _pair("umol/kg", "mg/m^3")
    with pytest.raises(ValueError, match="not the same physical quantity"):
        _align.align(test, reference, method="bilinear")


def test_align_converts_before_differencing():
    """Regression: 100 umol/kg minus 100 mmol/m3 used to report a difference of 0.0.

    They differ by the density factor, so the honest answer is ~2.5 mmol/m3. The old
    code subtracted the raw arrays and labelled the result with the reference's
    units — silent, plausible, wrong.
    """
    test, reference = _pair("umol/kg", "mmol/m^3")
    out = _align.align(test, reference, method="bilinear")
    expected = 100.0 * u.RHO_SEAWATER / 1000.0 - 100.0
    assert float(out["difference"].mean()) == pytest.approx(expected, rel=1e-3)


def test_align_warns_but_proceeds_when_units_are_unknown():
    test, reference = _pair("$$unparseable", "mmol/m^3")
    with pytest.warns(UserWarning, match="cannot verify units"):
        _align.align(test, reference, method="bilinear")


def test_align_warns_but_proceeds_when_units_are_missing_entirely():
    """Same as above, but the test lane never had a ``units`` attr at all -- the
    shape of the failure a units-dropping vertical transform (e.g. to_depth
    before it copied per-variable attrs) actually produces.
    """
    test, reference = _pair("$$unparseable", "mmol/m^3")
    del test.attrs["units"]
    with pytest.warns(UserWarning, match="cannot verify units"):
        _align.align(test, reference, method="bilinear")


def test_matching_units_need_no_conversion():
    test, reference = _pair("mmol/m^3", "mmol/m^3")
    out = _align.align(test, reference, method="bilinear")
    assert float(out["difference"].mean()) == pytest.approx(0.0, abs=1e-9)


# -- one registry across pandas and xarray ------------------------------------


def test_pandas_and_xarray_share_one_registry():
    """Mixing registries raises in pint, so timeseries and gridded must share one."""
    import pandas as pd
    import pint_pandas  # noqa: F401  (registers the pint dtype)

    series = pd.Series([1.0, 2.0], dtype="pint[umol/kg]")
    quantity = u.registry().Quantity(1.0, "mmol/m^3")
    assert series.pint.quantity._REGISTRY is quantity._REGISTRY


def test_a_dataframe_column_converts_with_the_same_density_context():
    import pandas as pd
    import pint_pandas  # noqa: F401

    series = pd.Series([1.0], dtype="pint[umol/kg]")
    with u.registry().context("seawater"):
        converted = series.pint.to("mmol/m^3")
    assert float(converted.iloc[0].magnitude) == pytest.approx(u.RHO_SEAWATER / 1000.0)


# -- coordinate ordering ------------------------------------------------------


def _global_grid(lat):
    return xr.DataArray(
        np.ones((len(lat), 361)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": np.linspace(-180, 180, 361)},
        attrs={"units": "mg/m^3"},
    )


def _gom_curvilinear():
    ny, nx = 20, 24
    lon = np.linspace(-100, -79, nx)[None, :] * np.ones((ny, 1))
    lat = np.linspace(16, 32, ny)[:, None] * np.ones((1, nx))
    return xr.DataArray(
        np.ones((ny, nx)),
        dims=("eta", "xi"),
        coords={"lon": (("eta", "xi"), lon), "lat": (("eta", "xi"), lat)},
        attrs={"units": "mg/m^3"},
    )


@pytest.mark.parametrize("descending", [False, True])
def test_bbox_subset_honours_coordinate_direction(descending):
    """`.sel` with a slice follows the stored order, so a descending axis got nothing.

    Satellite L3 products are stored north-to-south (MODIS runs 89.979 to -89.979),
    so this was the common case: the subset came back empty and the failure surfaced
    much later as an IndexError inside cell-corner derivation.
    """
    from ocean_skill.align import subset_to_bbox

    lat = np.linspace(90, -90, 181) if descending else np.linspace(-90, 90, 181)
    got = subset_to_bbox(_global_grid(lat), (-100.0, 16.0, -79.0, 32.0))
    assert got.sizes["lat"] == 19


def test_align_works_against_a_descending_latitude_reference():
    """Regression: model-vs-MODIS raised IndexError from an empty corner array."""
    out = _align.align(
        _gom_curvilinear(),
        _global_grid(np.linspace(90, -90, 181)),
        method="conservative_normed",
    )
    assert out["difference"].size > 0


def test_genuinely_disjoint_sources_say_so():
    """Empty overlap is a real condition and should be named, not crash downstream."""
    far_north = _global_grid(np.linspace(90, -90, 181)).sel(lat=slice(80, 70))
    with pytest.raises(ValueError, match="no overlap"):
        _align.align(_gom_curvilinear(), far_north, method="bilinear")


# -- salinity spellings ----------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["ppt", "PPT", "ppth", "psu", "PSU"])
def test_parts_per_thousand_spellings_are_salinity_not_pico_pints(spelling):
    """``ppt`` parses as *pico-pint* in pint — a real unit, dimensionally wrong.

    Silently so: ``compatible("1e-3", "ppt")`` came back ``False``, which made a
    mooring-versus-product salinity comparison refuse to run rather than convert.
    OceanSODA-ETHZ writes ``ppt``; the OOI CTDs write ``1e-3``.
    """
    assert u.parse(spelling).dimensionality == u.parse("1").dimensionality
    assert u.compatible("1e-3", spelling) is True
    assert u.compatible(spelling, "g/kg") is True


# -- QC flags are never the measurement ------------------------------------------------


def _flagged(name: str, standard_name: str = "sea_water_potential_temperature"):
    """Build a dataset whose only variable is a flag *claiming* a data standard_name.

    Not a straw man: this is how the cf-xarray path is reached, since it matches on the
    ``standard_name`` attribute rather than the variable's name, so no anchored name
    pattern closes it. Verified to match without the guard below.
    """
    return xr.Dataset(
        {name: (("time",), np.ones(3), {"standard_name": standard_name})},
        coords={"time": [1, 2, 3]},
    )


@pytest.mark.parametrize(
    "name",
    [
        "sea_water_temperature_qc_agg",
        "sea_water_temperature_qc_tests",
        "temperature_qc",
        "qc_temperature",
        "sea_water_temperature_qartod",
        "sea_water_temperature_flag",
    ],
)
def test_a_qc_flag_never_satisfies_a_request_for_the_measurement(name):
    with pytest.warns(UserWarning, match="QC flag"):
        assert u.find_variable(_flagged(name), "temperature") is None


def test_a_flag_named_after_the_variable_is_no_match_either():
    """The other route in, closed already by the vocabulary's anchored patterns.

    Kept as a test because the two protections are independent: this one works on the
    *name*, the guard above on a claimed ``standard_name`` attribute.
    """
    ds = xr.Dataset(
        {"sea_water_temperature_qc_agg": (("time",), np.ones(3))},
        coords={"time": [1, 2, 3]},
    )
    assert u.find_variable(ds, "temperature") is None


def test_asking_for_a_flag_by_name_still_finds_it():
    """The rule is about *substituting* a flag, not about hiding one."""
    ds = _flagged("sea_water_temperature_qc_agg", "sea_water_temperature_qc_agg")
    found = u.find_variable(ds, "sea_water_temperature_qc_agg")
    assert found is not None and found.name == "sea_water_temperature_qc_agg"


def test_a_name_merely_containing_the_letters_qc_is_not_a_flag():
    """Tokens, not substrings — otherwise a legitimate name is caught too."""
    assert not u.is_qc_name("qcm_index")
    assert not u.is_qc_name("sea_water_qcx")
    assert u.is_qc_name("sea_water_temperature_qc_agg")


def test_the_real_variable_still_wins_when_both_are_present():
    ds = _flagged("sea_water_temperature_qc_agg")
    ds["sea_water_temperature"] = ds["sea_water_temperature_qc_agg"] * 2
    assert u.find_variable(ds, "temperature").name == "sea_water_temperature"
