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
