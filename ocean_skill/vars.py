"""BGC variable registry: CF standard_name → units/model-conversion metadata.

Colormaps live in :mod:`ocean_skill.colormaps` (registered directly into xcmocean's
own tables); unit *conversion* lives in :mod:`ocean_skill.units`, which does it with
pint from each field's own ``units`` attribute. This module keeps only the nominal
units a variable is reported in.

It used to also carry a per-species ``model_conv`` factor (ALK/DIC x 1026/1000).
That was dead code, and wiring it up would have been wrong: it is the same
umol/kg -> mmol/m3 density conversion :func:`ocean_skill.units.convert_units`
already applies from the data's own units, so applying both would double-count.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["REGISTRY", "VarInfo", "lookup", "short_name"]


@dataclass(frozen=True)
class VarInfo:
    """Display/analysis metadata for one variable (by CF standard_name)."""

    standard_name: str
    units: str | None = None


REGISTRY: dict[str, VarInfo] = {
    "sea_surface_height_above_geoid": VarInfo(
        "sea_surface_height_above_geoid", units="m"
    ),
    "sea_water_alkalinity_expressed_as_mole_equivalent": VarInfo(
        "sea_water_alkalinity_expressed_as_mole_equivalent",
        units="mmol m-3",
    ),
    "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water": VarInfo(
        "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water",
        units="mmol m-3",
    ),
    "mole_concentration_of_nitrate_in_sea_water": VarInfo(
        "mole_concentration_of_nitrate_in_sea_water", units="mmol m-3"
    ),
    "mole_concentration_of_phosphate_in_sea_water": VarInfo(
        "mole_concentration_of_phosphate_in_sea_water", units="mmol m-3"
    ),
    "mole_concentration_of_silicate_in_sea_water": VarInfo(
        "mole_concentration_of_silicate_in_sea_water", units="mmol m-3"
    ),
    "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water": VarInfo(
        "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
        units="mmol m-3",
    ),
    "surface_downward_mole_flux_of_carbon_dioxide": VarInfo(
        "surface_downward_mole_flux_of_carbon_dioxide"
    ),
    "mass_concentration_of_chlorophyll_a_in_sea_water": VarInfo(
        "mass_concentration_of_chlorophyll_a_in_sea_water", units="mg m-3"
    ),
    "ocean_mixed_layer_thickness": VarInfo("ocean_mixed_layer_thickness", units="m"),
    # CF's criterion-specific mixed layer thicknesses, each its own key: this table is
    # looked up by exact standard_name (see lookup), so the generic entry above does
    # not cover them.
    "ocean_mixed_layer_thickness_defined_by_sigma_theta": VarInfo(
        "ocean_mixed_layer_thickness_defined_by_sigma_theta", units="m"
    ),
    "ocean_mixed_layer_thickness_defined_by_sigma_t": VarInfo(
        "ocean_mixed_layer_thickness_defined_by_sigma_t", units="m"
    ),
    "ocean_mixed_layer_thickness_defined_by_temperature": VarInfo(
        "ocean_mixed_layer_thickness_defined_by_temperature", units="m"
    ),
    "ocean_mixed_layer_thickness_defined_by_mixing_scheme": VarInfo(
        "ocean_mixed_layer_thickness_defined_by_mixing_scheme", units="m"
    ),
    "mole_concentration_of_nitrate_and_nitrite_in_sea_water": VarInfo(
        "mole_concentration_of_nitrate_and_nitrite_in_sea_water", units="mmol m-3"
    ),
    "mass_concentration_of_phaeopigments_in_sea_water": VarInfo(
        "mass_concentration_of_phaeopigments_in_sea_water", units="mg m-3"
    ),
    "downwelling_photosynthetic_photon_flux_in_sea_water": VarInfo(
        "downwelling_photosynthetic_photon_flux_in_sea_water",
        units="umol m-2 s-1",
    ),
    # osk-custom names (CF has none for tidal harmonic constants or this flux).
    "sea_surface_height_tidal_amplitude": VarInfo(
        "sea_surface_height_tidal_amplitude", units="m"
    ),
    "sea_surface_height_tidal_phase": VarInfo(
        "sea_surface_height_tidal_phase", units="degree"
    ),
    "sea_surface_height_tidal_harmonic_real_part": VarInfo(
        "sea_surface_height_tidal_harmonic_real_part", units="m"
    ),
    "sea_surface_height_tidal_harmonic_imaginary_part": VarInfo(
        "sea_surface_height_tidal_harmonic_imaginary_part", units="m"
    ),
    # ROMS' up/vp are W m-1 divided by rho0, hence m4 s-3.
    "x_baroclinic_pressure_flux": VarInfo("x_baroclinic_pressure_flux", units="m4 s-3"),
    "y_baroclinic_pressure_flux": VarInfo("y_baroclinic_pressure_flux", units="m4 s-3"),
    "eastward_baroclinic_pressure_flux": VarInfo(
        "eastward_baroclinic_pressure_flux", units="W m-1"
    ),
    "northward_baroclinic_pressure_flux": VarInfo(
        "northward_baroclinic_pressure_flux", units="W m-1"
    ),
}


def lookup(standard_name: str) -> VarInfo:
    """Return :class:`VarInfo` for a standard_name, else a bare default.

    Accepts anything :func:`ocean_skill.vocabulary.resolve_name` recognizes (a short
    vocabulary key or alias, not just the canonical standard_name) for standalone use;
    callers that already carry a resolved :class:`~ocean_skill.comparison.Comparison`
    variable are passing the canonical form already, so this is a no-op for them.
    """
    from ocean_skill.vocabulary import resolve_name

    standard_name = resolve_name(standard_name)
    return REGISTRY.get(standard_name, VarInfo(standard_name))


#: Chunks stripped when shortening a CF standard_name for a plot label, longest first so
#: the more specific phrases win.
_LABEL_NOISE = (
    "_per_unit_mass_in_sea_water",
    "_expressed_as_mole_equivalent",
    "mole_concentration_of_",
    "moles_of_",
    "mass_concentration_of_",
    "_in_sea_water",
    "sea_water_",
    "_above_geoid",
    "dissolved_",
    "molecular_",
)

#: Preferred short labels where stripping alone reads poorly.
_LABEL_OVERRIDES = {
    "sea_water_potential_temperature": "temperature",
    "sea_water_practical_salinity": "salinity",
    "sea_surface_height_above_geoid": "SSH",
    "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water": "DIC",
    "sea_water_alkalinity_expressed_as_mole_equivalent": "alkalinity",
    "surface_downward_mole_flux_of_carbon_dioxide": "CO2 flux",
    "mole_concentration_of_nitrate_and_nitrite_in_sea_water": "nitrate+nitrite",
    "downwelling_photosynthetic_photon_flux_in_sea_water": "PAR",
    # The criterion-specific mixed layer thicknesses: stripping alone would leave
    # "ocean mixed layer thickness defined by sigma theta", far too long for a legend
    # entry or a table column -- and the criterion is the one thing that tells two of
    # them apart, so it is kept, in brackets. The generic name
    # (``ocean_mixed_layer_thickness``) is deliberately *not* overridden: it names no
    # criterion, so a label claiming one would be wrong, and its plain stripped-down
    # label is what existing callers show.
    "ocean_mixed_layer_thickness_defined_by_sigma_theta": "MLD (σθ)",
    "ocean_mixed_layer_thickness_defined_by_sigma_t": "MLD (σt)",
    "ocean_mixed_layer_thickness_defined_by_temperature": "MLD (temperature)",
    "ocean_mixed_layer_thickness_defined_by_mixing_scheme": "MLD (mixing scheme)",
    # Stripping would leave "sea surface height tidal amplitude" etc.
    "sea_surface_height_tidal_amplitude": "tidal amplitude",
    "sea_surface_height_tidal_phase": "tidal phase",
    "sea_surface_height_tidal_harmonic_real_part": "tidal harmonic (real)",
    "sea_surface_height_tidal_harmonic_imaginary_part": "tidal harmonic (imaginary)",
    "eastward_baroclinic_pressure_flux": "eastward baroclinic pressure flux",
    "northward_baroclinic_pressure_flux": "northward baroclinic pressure flux",
}


def short_name(standard_name: str) -> str:
    """Return a compact, human-readable label for a CF standard_name.

    ``mole_concentration_of_nitrate_in_sea_water`` -> ``nitrate``;
    ``sea_water_potential_temperature`` -> ``temperature``. Used for plot labels and
    table columns, where the full name is unreadable and naive truncation produces
    things like "sea_water_po". Accepts any spelling
    :func:`ocean_skill.vocabulary.resolve_name` recognizes, same as :func:`lookup`.
    """
    from ocean_skill.vocabulary import resolve_name

    standard_name = resolve_name(standard_name)
    if standard_name in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[standard_name]
    out = standard_name
    for chunk in _LABEL_NOISE:
        out = out.replace(chunk, "")
    return out.strip("_").replace("_", " ") or standard_name
