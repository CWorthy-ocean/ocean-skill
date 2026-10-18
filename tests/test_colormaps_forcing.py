"""roms-tools surface forcing variables keep raw names; each gets its quantity's map."""

import pytest

from ocean_skill.colormaps import center_for, cmaps_for


@pytest.mark.parametrize(
    ("name", "cmap"),
    [
        ("Tair", "thermal"),
        ("qair", "ice_r"),
        ("rain", "rain"),
        ("swrad", "solar"),
        ("lwrad", "solar"),
        ("dust", "turbid"),
        ("nox", "deep"),
        ("nhy", "deep"),
        ("xco2_air", "ice_r"),
        ("xco2_air_alt", "ice_r"),
    ],
)
def test_surface_forcing_variables_take_their_quantitys_map(name, cmap):
    assert cmaps_for(name)[0].name == cmap
    assert center_for(name) is None


@pytest.mark.parametrize("name", ["uwnd", "vwnd"])
def test_forcing_winds_are_signed_and_centered(name):
    assert cmaps_for(name)[0].name == "delta"
    assert center_for(name) == 0.0


def test_short_forcing_names_match_only_exactly():
    # anchored: "rain" must not catch a longer name that merely contains it
    assert cmaps_for("drainage")[0].name != "rain"
