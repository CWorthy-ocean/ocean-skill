"""Variable -> cmocean colormap assignments in :mod:`ocean_skill.colormaps`.

Each entry in ``_SEQUENTIAL_CMAPS`` is matched by ``re.search`` against the *resolved*
canonical standard_name, so a key must actually be a substring of that name -- "par"
is not a substring of ``downwelling_photosynthetic_photon_flux_in_sea_water``, which is
why that entry is keyed by the full standard_name instead. This test locks in the
mapping (and would have caught that gotcha) for turbidity/fluorescence/PAR.
"""

from __future__ import annotations

import pytest

from ocean_skill.colormaps import cmaps_for


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("turbidity", "turbid"),
        ("Turbidity_CTD", "turbid"),
        ("fluorescence", "algae"),
        ("Fluor_CTD", "algae"),
        ("PAR", "solar"),
        ("PAR_CTD", "solar"),
    ],
)
def test_new_sequential_cmaps(spelling, expected):
    seq, _ = cmaps_for(spelling)
    assert seq.name == expected
