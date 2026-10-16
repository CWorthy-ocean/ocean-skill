"""``ocean_skill.units.display``: every spelling of a unit reads one way on a label.

Colorbars and axes in a report print units straight from files, which spell the same
quantity several ways (``degrees_celsius``, ``degree_Celsius``, ``degC``, ``°C``;
``mmol m-3``, ``mmol/m3``, ``millimole m-3``). ``display`` is the one place that
picks a spelling, so these pin the table, the case-insensitivity, the derived forms
the package itself produces (squared for variance, ``delta_`` for a difference), and
the two things a label must never do: change what the number means (``PSU`` is not
``g/kg``) or drift when run twice.
"""

from __future__ import annotations

import re

import pytest

from ocean_skill import units as u
from ocean_skill.plot._statistic import units_text

TEMPERATURE = [
    "degC",
    "degrees_celsius",
    "degree_Celsius",
    "Degrees_Celsius",
    "degrees_C",
    "degree_C",
    "deg_C",
    "deg C",
    "DEGC",
    "celsius",
    "Celsius",
    "degrees celsius",
    "degrees celcius",  # sic: in real WOA-derived files
    "°C",
]


@pytest.mark.parametrize("spelling", TEMPERATURE)
def test_every_celsius_spelling_reads_the_same(spelling):
    assert u.display(spelling) == "°C"


@pytest.mark.parametrize(
    "spelling", ["degF", "degrees_fahrenheit", "Degree_Fahrenheit"]
)
def test_fahrenheit(spelling):
    assert u.display(spelling) == "°F"


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        # salinity: spelling only -- PSU is never turned into g/kg, nor 1e-3 into PSU
        ("psu", "PSU"),
        ("PSU", "PSU"),
        ("Psu", "PSU"),
        ("practical_salinity_units", "PSU"),
        ("ppt", "ppt"),  # the same number as PSU, not the same label
        ("1e-3", "1e-3"),
        ("g/kg", "g/kg"),
        ("g kg-1", "g/kg"),
        # molar concentrations
        ("mmol/m3", "mmol/m³"),
        ("mmol m-3", "mmol/m³"),
        ("mmol/m^3", "mmol/m³"),
        ("mmol m^-3", "mmol/m³"),
        ("mmol.m-3", "mmol/m³"),
        ("millimole m-3", "mmol/m³"),
        ("millimoles_per_m3", "mmol/m³"),
        ("MMOL/M3", "mmol/m³"),
        ("mmol/m³", "mmol/m³"),
        ("mmol/L", "mmol/L"),
        ("umol kg-1", "µmol/kg"),
        ("umol/kg", "µmol/kg"),
        ("µmol/kg", "µmol/kg"),
        ("μmol/kg", "µmol/kg"),  # Greek mu, as some files spell it
        ("micromoles_per_kilogram", "µmol/kg"),
        ("micromol kg-1", "µmol/kg"),
        ("mol/kg", "mol/kg"),
        ("meq/m3", "meq/m³"),
        # mass concentrations (chlorophyll)
        ("mg m-3", "mg/m³"),
        ("mg/m3", "mg/m³"),
        ("mg/m^3", "mg/m³"),
        ("milligram m-3", "mg/m³"),
        ("ug/L", "µg/L"),
        # oxygen as a gas volume stays a gas volume
        ("mL/L", "mL/L"),
        ("ml l-1", "mL/L"),
        # lengths and speeds
        ("m", "m"),
        ("meter", "m"),
        ("meters", "m"),
        ("metre", "m"),
        ("m s-1", "m/s"),
        ("m/s", "m/s"),
        ("m.s-1", "m/s"),
        ("meters_per_second", "m/s"),
        ("cm s-1", "cm/s"),
        ("m2 s-2", "m²/s²"),
        ("W m-2", "W/m²"),
        ("kg m-3", "kg/m³"),
        ("mmol m-3 d-1", "mmol/(m³ d)"),
        ("dbar", "dbar"),
        ("K", "K"),
        ("%", "%"),
        # not units pint knows: left as they came
        ("1", "1"),
        ("unknown_thing", "unknown_thing"),
        ("", ""),
        (None, ""),
    ],
)
def test_spelling_table(spelling, expected):
    assert u.display(spelling) == expected


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        # what for_statistic hands the renderers
        ("delta_degC", "°C"),
        ("delta_degC^2", "°C²"),
        ("delta_degree_Celsius", "°C"),
        ("Delta_degrees_Celsius", "°C"),
        ("delta_degF", "°F"),
        ("K^2", "K²"),
        ("(mg/m^3)^2", "(mg/m³)²"),
        ("(mmol/m^3)^2", "(mmol/m³)²"),
        ("(mmol m-3)^2", "(mmol/m³)²"),
        ("(micromoles_per_kilogram)^2", "(µmol/kg)²"),
        ("psu^2", "PSU²"),
        ("1e-06", "1e-06"),  # the squared salinity 1e-3
        ("(1e-06)", "1e-06"),
    ],
)
def test_derived_forms(spelling, expected):
    assert u.display(spelling) == expected


@pytest.mark.parametrize(
    ("units", "statistic", "label"),
    [
        ("degrees_celsius", "var", "°C²"),
        ("degree_Celsius", "std", "°C"),
        ("degC", "range", "°C"),
        ("degC", "mean", "°C"),
        ("mmol m-3", "var", "(mmol/m³)²"),
        ("mmol m-3", "std", "mmol/m³"),
        ("mg m-3", "iqr", "mg/m³"),
        ("1e-3", "var", "1e-06"),
        ("psu", "mean", "PSU"),
        ("K", "var", "K²"),
    ],
)
def test_for_statistic_output_round_trips_to_a_label(units, statistic, label):
    assert u.display(u.for_statistic(units, statistic)) == label
    assert units_text(u.for_statistic(units, statistic), statistic) == label


@pytest.mark.parametrize(
    "spelling",
    [
        *TEMPERATURE,
        *["mmol m-3", "umol kg-1", "m s-1", "psu", "(mg/m^3)^2", "delta_degC^2"],
        *["mmol m-3 d-1", "ml/l", "ug/L", "unknown_thing", "1e-3"],
    ],
)
def test_display_is_idempotent(spelling):
    once = u.display(spelling)
    assert u.display(once) == once


def test_display_never_changes_the_meaning_of_a_unit():
    """A label's units must parse back to the unit the file declared."""
    for spelling in [
        "mmol m-3",
        "umol kg-1",
        "mg/m3",
        "m s-1",
        "W m-2",
        "mmol m-3 d-1",
        "dbar",
        "ml/l",
        "psu",
        "g/kg",
    ]:
        assert u.parse(u.display(spelling)) == u.parse(spelling), spelling


def test_psu_is_never_read_as_g_per_kg():
    assert u.display("psu") != u.display("g/kg")
    assert u.display("PSU") == "PSU"


def test_units_text_goes_through_display_for_plain_values():
    assert units_text("degrees_celsius") == "°C"
    assert units_text("degree_Celsius", "mean") == "°C"
    assert units_text("mmol m-3", None) == "mmol/m³"
    assert units_text(None) == ""


# Realistic unit strings from ROMS/MARBL output, WOA23, GLODAP, CMEMS, Holte & Talley
# roms-tools forcing. Each must read as the same quantity it was; where pint would be
# ambiguous (bare C/N/P, kt, ppt) the original words stay and only the exponents tidy.
REALISTIC = [
    # carbon, nitrogen and friends are labels on a mass or amount, never degC / newton
    ("mmol C/m^3", "mmol C/m³"),
    ("mmol C m-3", "mmol C/m³"),
    ("mg C m-3", "mg C/m³"),
    ("mg C m-3 d-1", "mg C/(m³ d)"),
    ("mmol C m-2 d-1", "mmol C/(m² d)"),
    ("mmol C/m^2/d", "mmol C/(m² d)"),
    ("mol C m-2 yr-1", "mol C/(m² yr)"),
    ("Pg C yr-1", "Pg C/yr"),
    ("Pg C/yr", "Pg C/yr"),
    ("Gt C yr-1", "Gt C/yr"),
    ("Tg C", "Tg C"),
    ("g C m-2", "g C/m²"),
    ("g C/m2/yr", "g C/(m² yr)"),
    ("PgC yr-1", "PgC yr-1"),  # fused: not a unit pint reads, left alone
    ("GtC", "GtC"),
    ("mmol N m-3", "mmol N/m³"),
    ("mmol N/m3", "mmol N/m³"),
    ("mmol N m-2 s-1", "mmol N/(m² s)"),
    ("mol N m-2", "mol N/m²"),
    ("mmol P m-3", "mmol P/m³"),
    ("mmol Si m-3", "mmol Si/m³"),
    ("mmol Fe m-3", "mmol Fe/m³"),
    ("umol Fe m-3", "µmol Fe/m³"),
    ("nmol Fe/L", "nmol Fe/L"),
    ("mmol CaCO3 m-3", "mmol CaCO3/m³"),
    ("mmol NO3 m-3", "mmol NO3/m³"),
    ("mmol O2 m-3", "mmol O2/m³"),
    ("umol O2/kg", "µmol O2/kg"),
    ("mol O2 m-2 yr-1", "mol O2/(m² yr)"),
    ("ml O2 /L", "mL O2/L"),
    ("mL O2 L-1", "mL O2/L"),
    ("mg Chl m-3", "mg Chl/m³"),
    ("mg Chl-a m-3", "mg Chl-a/m³"),
    ("mg Chl a m-3", "mg Chl a/m³"),
    ("mg chl m-3", "mg chl/m³"),
    ("(mmol C m-3)^2", "(mmol C/m³)²"),
    ("C", "C"),  # bare: not assumed to be degC
    ("Chl", "Chl"),
    # newtons stay newtons
    ("N/m2", "N/m²"),
    ("N m-2", "N/m²"),
    ("N", "N"),
    ("Pa", "Pa"),
    ("dyn/cm2", "dyn/cm²"),
    # amounts, masses, per-mass
    ("mmol/m^3", "mmol/m³"),
    ("mmol/m^3/s", "mmol/(m³ s)"),
    ("nmol/m^3", "nmol/m³"),
    ("nmol/L", "nmol/L"),
    ("meq/m^3", "meq/m³"),
    ("micromole/kg", "µmol/kg"),
    ("kg/m3", "kg/m³"),
    ("kg/m^2/s", "kg/(m² s)"),
    ("m2 s-1", "m²/s"),
    ("m^2/s", "m²/s"),
    ("m3/s", "m³/s"),
    ("W/m2", "W/m²"),
    ("W m-2 K-1", "W/(m² K)"),
    ("J/kg", "J/kg"),
    ("mS/cm", "mS/cm"),
    ("S m-1", "S/m"),
    ("Sv", "Sv"),
    ("hPa", "hPa"),
    ("mbar", "mbar"),
    # time and angles
    ("hours", "h"),
    ("day", "d"),
    ("year", "yr"),
    ("radian", "rad"),
    ("degree_north", "degrees_north"),
    ("degrees_north", "degrees_north"),
    ("degrees_east", "degrees_east"),
    # dimensionless things, left as the file wrote them
    ("percent", "%"),
    ("fraction", "fraction"),
    ("unitless", "unitless"),
    ("dimensionless", "dimensionless"),
    ("none", "none"),
    ("pH", "pH"),
    ("pH units", "pH units"),  # pint would read "units" as micro-nit
    ("ppm", "ppm"),
    ("uatm", "µatm"),
    ("m3 m-3", "m3 m-3"),
    ("kg kg-1", "kg kg-1"),
    ("per mil", "per mil"),
    ("‰", "‰"),
    # words with two readings are left as written
    ("kt", "kt"),  # knot to pint, kiloton in a carbon budget
    ("ppt", "ppt"),
    ("ppth", "ppth"),
    ("parts per thousand", "parts per thousand"),
    # times with an epoch are not units, but must not crash or be mangled
    ("days since 2000-01-01", "days since 2000-01-01"),
    ("seconds since 1970-01-01 00:00:00", "seconds since 1970-01-01 00:00:00"),
    ("hours since 1900-01-01", "hours since 1900-01-01"),
    # numeric factors are exact
    ("1e9 kg/s", "1e9 kg/s"),
    ("1e-3 kg/s", "1e-3 kg/s"),
    ("1e6 m3 s-1", "1e6 m³/s"),
    ("1e6 m3/s", "1e6 m³/s"),
    ("1 m-1", "1/m"),
    ("1/s", "1/s"),
    ("1e-3", "1e-3"),
    ("1e-06", "1e-06"),
    ("1", "1"),
    ("", ""),
    # still the temperature spellings, with or without a space
    ("deg C", "°C"),
    ("Degrees C", "°C"),
    ("degrees_C", "°C"),
    ("Kelvin", "K"),
    ("mmol/m^3 N", "mmol/m³ N"),  # a label after the denominator: left in order
]


@pytest.mark.parametrize(("spelling", "expected"), REALISTIC)
def test_realistic_unit_strings(spelling, expected):
    assert u.display(spelling) == expected


@pytest.mark.parametrize(("spelling", "expected"), REALISTIC)
def test_realistic_unit_strings_are_idempotent(spelling, expected):
    assert u.display(expected) == expected


@pytest.mark.parametrize(
    "spelling",
    [s for s, _ in REALISTIC if "since" not in s],
)
def test_a_label_never_gains_a_temperature_or_loses_a_word(spelling):
    shown = u.display(spelling)
    if not re.search(r"deg|celsius|°", spelling, re.IGNORECASE):
        assert "°" not in shown and "Δ" not in shown, (spelling, shown)
    # every alphabetic element word the file wrote is still on the label
    for word in re.findall(r"\b(?:Chl|CaCO3|NO3|O2|Fe|Si)\b", spelling):
        assert word in shown, (spelling, shown)


def test_numeric_factors_survive_unchanged():
    for factor in ("1e9", "1e-3", "1e6", "2.5", "10"):
        assert u.display(f"{factor} kg/s").startswith(f"{factor} "), factor


def test_ppt_is_not_turned_into_psu_but_psu_spellings_are():
    assert u.display("ppt") == "ppt"
    assert u.display("ppt") != u.display("psu")
    for spelling in (
        "psu",
        "PSU",
        "practical_salinity_units",
        "practical salinity units",
    ):
        assert u.display(spelling) == "PSU"
