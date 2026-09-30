"""Variable -> cmocean colormap assignments in :mod:`ocean_skill.colormaps`.

Each entry in ``_SEQUENTIAL_CMAPS`` is matched by ``re.search`` against the *resolved*
canonical standard_name, so a key must actually be a substring of that name -- "par"
is not a substring of ``downwelling_photosynthetic_photon_flux_in_sea_water``, which is
why that entry is keyed by the full standard_name instead. This test locks in the
mapping (and would have caught that gotcha) for turbidity/fluorescence/PAR, and
(with the tests below) for ammonium/iron colliding on xcmocean's "dye" default and for
sea-level anomaly colliding on its "vel" pattern. It also pins the mixed layer
thickness names -- the generic one and CF's four ``..._defined_by_<criterion>`` ones --
to the MLD map: those embed their criterion variable's name ("sigma_theta"), so which
map they get depends on the *order* of the table, not just on what it contains.
"""

from __future__ import annotations

import pytest

from ocean_skill.colormaps import cmaps_for

#: The BGC species this module's whole policy exists to keep visually distinct (see
#: the ``_SEQUENTIAL_CMAPS`` module-level comment in :mod:`ocean_skill.colormaps`).
_BGC_SPECIES = (
    "nitrate",
    "phosphate",
    "silicate",
    "ammonium",
    "iron",
    "oxygen",
    "dissolved_inorganic_carbon",
    "alkalinity",
    "chlorophyll",
    "par",
    "turbidity",
    "ph",
)


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("turbidity", "turbid"),
        ("Turbidity_CTD", "turbid"),
        ("fluorescence", "algae"),
        ("Fluor_CTD", "algae"),
        ("PAR", "solar"),
        ("PAR_CTD", "solar"),
        ("ammonium", "dense"),
        ("NH4", "dense"),
        ("iron", "amp"),
        ("Fe", "amp"),
        ("nitrate", "deep"),
        ("phosphate", "rain"),
        ("sea_level_anomaly", "balance"),
        ("eastward_wind", "speed"),
        ("northward_wind", "speed"),
        ("sea_ice", "ice"),
        ("sigma_theta", "dense"),
        # the density anomaly itself, spelled as the resolved CF name (the short
        # "sigma_theta" above resolves to it): listing "mixed_layer" ahead of it in the
        # table (see the MLD cases below) must not change what it gets
        ("sea_water_sigma_theta", "dense"),
        ("conductivity", "haline"),
        ("mld", "deep"),
        # The criterion-specific mixed layer thicknesses, by full CF name (the short
        # forms resolve to these). Each embeds the name of its criterion variable, and
        # "sigma_theta" is a *substring* of the first one: first-match-wins over the
        # table would hand it the density map (dense) unless "mixed_layer" is listed
        # ahead of "sigma_theta". All four are a thickness in metres -- the MLD map.
        ("ocean_mixed_layer_thickness_defined_by_sigma_theta", "deep"),
        ("ocean_mixed_layer_thickness_defined_by_sigma_t", "deep"),
        ("ocean_mixed_layer_thickness_defined_by_temperature", "deep"),
        ("ocean_mixed_layer_thickness_defined_by_mixing_scheme", "deep"),
        ("pressure", "deep"),
        ("ph", "speed_r"),
        ("kd490", "turbid"),
    ],
)
def test_new_sequential_cmaps(spelling, expected):
    seq, _ = cmaps_for(spelling)
    assert seq.name == expected


def test_bgc_species_pairwise_distinct():
    """No two BGC species may share a sequential colormap.

    This is the test that would have caught ammonium and iron both falling through to
    xcmocean's "dye" default (``cmo.matter``) before either had a dedicated entry.
    """
    names = {species: cmaps_for(species)[0].name for species in _BGC_SPECIES}
    assert len(set(names.values())) == len(names), names


def test_sla_is_signed():
    """Sea-level anomaly must not pick up xcmocean's velocity map.

    xcmocean's built-in "vel" pattern matches the substring "vel" in "sea_le-vel-",
    which -- absent our own entry -- silently hands SLA a magnitude (speed) colormap
    for a quantity that is actually signed.
    """
    seq, _ = cmaps_for("sea_level_anomaly")
    assert seq.name == "balance"


# --- bathymetry -----------------------------------------------------------------------
#
# ROMS calls its seafloor depth ``h`` and keeps it under that name (``to_depth`` reads
# it), so ``field(src, "h").plot()`` asks for a colormap by a one-letter string. That
# can only be matched exactly: as a substring pattern it would colour half the
# vocabulary. And it must not be a vocabulary alias either -- ``find_variable`` would
# then look for ``sea_floor_depth_below_geoid`` in a dataset that only has ``h``.


@pytest.mark.parametrize(
    "name",
    ["h", "H", "bathymetry", "sea_floor_depth", "sea_floor_depth_below_geoid"],
)
def test_bathymetry_spellings_get_the_deep_map(name):
    assert cmaps_for(name)[0].name == "deep"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("hs", "viridis"),  # contains an h, is not bathymetry: falls to the default
        ("ph", "speed_r"),
        ("phosphate", "rain"),
        ("chlorophyll", "algae"),
        ("temperature", "thermal"),
        ("sea_surface_height_above_geoid", "balance"),
    ],
)
def test_the_bathymetry_pattern_is_anchored_and_captures_nothing_else(name, expected):
    assert cmaps_for(name)[0].name == expected


def test_h_is_not_a_vocabulary_alias():
    """``h`` stays what the dataset calls it; only its colour is looked up by name."""
    from ocean_skill.vocabulary import is_known, resolve_name

    assert resolve_name("h") == "h"
    assert not is_known("h")
