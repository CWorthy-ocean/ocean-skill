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
        ("eastward_wind", "balance"),
        ("northward_wind", "balance"),
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
        ("sea_surface_height_above_geoid", "plasma"),
    ],
)
def test_the_bathymetry_pattern_is_anchored_and_captures_nothing_else(name, expected):
    assert cmaps_for(name)[0].name == expected


def test_h_is_not_a_vocabulary_alias():
    """``h`` stays what the dataset calls it; only its colour is looked up by name."""
    from ocean_skill.vocabulary import is_known, resolve_name

    assert resolve_name("h") == "h"
    assert not is_known("h")


# --- oxygen: low is dark ------------------------------------------------------------


def _luminance(rgba) -> float:
    """Return the Rec. 709 relative luminance of an RGBA colour (0 black, 1 white)."""
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@pytest.mark.parametrize(
    "name",
    [
        "oxygen",
        "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
        "oxygen_saturation",
    ],
)
def test_oxygen_is_dark_at_the_low_end_and_light_at_the_high_end(name):
    """Low oxygen must not draw white (the old ``gray_r`` did).

    Compared by luminance of the resolved matplotlib colormap itself at its two ends,
    not by its name, so a future map that is merely *called* something else but runs
    the wrong way still fails.
    """
    seq, _ = cmaps_for(name)
    assert _luminance(seq(0.0)) < _luminance(seq(1.0))
    assert _luminance(seq(0.0)) < 0.2, "the low end is near black"
    assert _luminance(seq(1.0)) > 0.8, "the high end is near white"


def test_oxygen_keeps_a_map_of_its_own():
    """Flipping the direction must not collide it with another BGC species' map."""
    seq, _ = cmaps_for("oxygen")
    assert seq.name == "gray"
    others = {s: cmaps_for(s)[0].name for s in _BGC_SPECIES if s != "oxygen"}
    assert "gray" not in others.values(), others


# --- ADT, spreads, centring, matplotlib names ----------------------------------------

SLA = "sea_surface_height_above_sea_level"
ADT = "sea_surface_height_above_geoid"
CHL = "mass_concentration_of_chlorophyll_a_in_sea_water"


def test_adt_is_plasma_and_sla_stays_balance():
    """ADT has an arbitrary datum (plasma); SLA is a signed anomaly (balance)."""
    assert cmaps_for(ADT)[0].name == "plasma"
    assert cmaps_for(SLA)[0].name == "balance"


def test_a_matplotlib_name_in_the_table_resolves_and_does_not_become_matter():
    """A non-``cmo.`` entry is a matplotlib name, not a silent cmo.matter."""
    import matplotlib

    from ocean_skill import colormaps

    assert colormaps._resolve_cmap("plasma").name == matplotlib.colormaps["plasma"].name
    assert colormaps._resolve_cmap("cmo.amp").name == "amp"
    # the registered xcmocean table holds the matplotlib map itself
    colormaps._register_colormaps()
    from xcmocean.options import SEQ

    assert SEQ[ADT].name == "plasma"
    with pytest.raises(KeyError):  # a typo is loud, not matter
        colormaps._resolve_cmap("not_a_colormap")


@pytest.mark.parametrize("statistic", ["std", "var", "range"])
def test_a_spread_gets_the_amp_map_from_zero(statistic):
    from ocean_skill.colormaps import norm_for

    assert cmaps_for("nitrate", statistic=statistic)[0].name == "amp"
    assert cmaps_for(None, statistic=statistic)[0].name == "amp"  # the fallback too
    assert cmaps_for("nitrate", statistic="mean")[0].name == "deep"
    # diverging side unchanged
    assert cmaps_for(SLA, statistic=statistic)[1].name == cmaps_for(SLA)[1].name
    norm = norm_for("nitrate", 3.1, 7.4, statistic=statistic)
    assert (norm.vmin, norm.vmax) == (0.0, 7.4)
    # a user's vmin still wins over the zero floor
    assert norm_for("nitrate", 3.1, 7.4, user_vmin=1.0, statistic=statistic).vmin == 1.0
    # a mean is untouched
    assert norm_for("nitrate", 3.1, 7.4, statistic="mean").vmin == 3.1


def test_center_for_is_zero_for_signed_anomalies_only():
    from ocean_skill.colormaps import center_for

    assert center_for(SLA) == 0.0
    assert center_for("sea_level_anomaly") == 0.0  # a short vocabulary spelling
    assert center_for("surface_downward_mole_flux_of_carbon_dioxide") == 0.0
    assert center_for(ADT) is None
    assert center_for("nitrate") is None
    assert center_for(SLA, statistic="std") is None  # a spread is not an anomaly


def test_sla_limits_are_symmetric_about_zero_and_users_win():
    from ocean_skill.colormaps import norm_for, variable_limits

    assert variable_limits(SLA, -0.1, 0.4) == (-0.4, 0.4)
    assert variable_limits(SLA, -0.6, 0.2) == (-0.6, 0.6)
    # the half-width is rounded like a centred metric's, not left at 0.4137
    assert variable_limits(SLA, -0.1, 0.4137) == (-0.42, 0.42)
    norm = norm_for(SLA, -0.1, 0.4)
    assert (norm.vmin, norm.vmax) == (-0.4, 0.4)
    # one user end is mirrored about the centre; the scale is always equal-sided
    assert variable_limits(SLA, -0.1, 0.4, user_vmin=-0.2) == (-0.2, 0.2)
    assert variable_limits(SLA, -0.1, 0.4, user_vmax=0.1) == (-0.1, 0.1)
    # both ends: honoured as given, even when lopsided
    assert variable_limits(SLA, -0.1, 0.4, user_vmin=-0.2, user_vmax=0.1) == (-0.2, 0.1)
    # a non-centred variable passes straight through
    assert variable_limits(ADT, 0.5, 1.4) == (0.5, 1.4)
    # a pinned range still applies (chlorophyll's), and a user's beats it
    assert variable_limits(CHL, 0.02, 3.0) == (0.01, 10.0)
    assert variable_limits(CHL, 0.02, 3.0, user_vmax=5.0) == (0.01, 5.0)


def test_a_centred_mean_panel_through_metric_colors_is_symmetric():
    import numpy as np

    from ocean_skill.colormaps import metric_colors

    colors = metric_colors("mean_test", np.linspace(-0.1, 0.4, 50), standard_name=SLA)
    assert colors.vmin == -colors.vmax


VELOCITY_COMPONENTS = [
    "eastward_sea_water_velocity",
    "northward_sea_water_velocity",
    "sea_water_x_velocity",
    "sea_water_y_velocity",
    "upward_sea_water_velocity",
    "eastward_wind",
    "northward_wind",
]


def test_the_sequential_map_is_diverging_iff_the_variable_is_centred():
    """``_CENTERED`` alone decides which variables are zero-meaningful (diverging)."""
    from pathlib import Path

    import yaml

    import ocean_skill
    from ocean_skill.colormaps import _CENTERED, center_for
    from ocean_skill.vocabulary import resolve_name

    path = Path(ocean_skill.__file__).parent / "vocab" / "vocabulary.yaml"
    keys = yaml.safe_load(path.read_text())
    assert keys
    for key in keys:
        diverging = cmaps_for(key)[0].name == "balance"
        centred = resolve_name(key) in _CENTERED
        assert diverging == centred, key
        assert (center_for(key) is not None) == centred, key
    # every table entry is a real vocabulary variable
    assert {resolve_name(k) for k in keys} >= set(_CENTERED)


@pytest.mark.parametrize("name", VELOCITY_COMPONENTS)
def test_signed_components_are_balance_symmetric_and_their_std_is_amp(name):
    from ocean_skill.colormaps import norm_for

    assert cmaps_for(name)[0].name == "balance"
    norm = norm_for(name, -0.3, 1.1)
    assert norm.vmin == -norm.vmax and norm.vmax >= 1.1
    assert cmaps_for(name, statistic="std")[0].name == "amp"
    spread = norm_for(name, 0.2, 1.1, statistic="std")
    assert spread.vmin == 0.0


def test_true_speeds_stay_cmo_speed():
    assert cmaps_for("wind_speed")[0].name == "speed"
    assert cmaps_for("sea_water_speed")[0].name == "speed"
