"""One spelling of a longitude: titles, section ticks and maps all say it the same way.

A 0-360 model stores 237.2 where a map's gridlines read 122.8°W; before these helpers an
extremum's suptitle said ``237.2°E`` over panels titled ``122.8°W``, and a section's x
axis printed bare numbers under a ``longitude (°E)`` label.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocean_skill.plot._geo_labels import format_lat, format_lon, wrap_lon
from ocean_skill.plot.series import lonlat_label


@pytest.mark.parametrize(
    "lon, expected",
    [(237.2, -122.8), (-122.8, -122.8), (180.0, 180.0), (-180.0, 180.0), (0.0, 0.0)],
)
def test_wrap_lon_is_plus_minus_180_with_the_dateline_positive(lon, expected):
    assert wrap_lon(lon) == pytest.approx(expected)


def test_ticks_read_as_cartopy_spells_them():
    assert format_lon(237.2) == "122.8°W"
    assert format_lon(-125) == "125°W"
    assert format_lon(140) == "140°E"
    # the bare meridians, as a map's gridlines print them
    assert [format_lon(v) for v in (0, 180, -180)] == ["0°", "180°", "180°"]
    assert format_lat(49) == "49°N" and format_lat(-20.5) == "20.5°S"
    assert format_lat(0) == "0°"


def test_a_position_title_wraps_a_zero_to_360_longitude():
    # what the extremum suptitle and the panel titles under it both print
    assert lonlat_label(237.2, 49.0) == "49.0°N 122.8°W"
    assert lonlat_label(-122.8, 49.0) == lonlat_label(237.2, 49.0)
    assert lonlat_label(100.3, 13.4) == "13.4°N 100.3°E"


def test_a_sections_path_note_names_a_dateline_crossing_in_one_convention():
    from ocean_skill.plot.section import _path_note
    from tests.test_section_x_axis import _path_field

    da = _path_field(np.linspace(170, 190, 9), np.linspace(5.0, 8.0, 9))
    assert _path_note(da, "lon", "lat") == "5.0°N, 170.0°E → 8.0°N, 170.0°W"


# One table of (value, expected) for both spellings: the static ticks run it directly,
# and the interactive formatter's source must carry the same rounding step.
LON_CASES = [
    (179.96, "180°"),
    (-179.96, "180°"),
    (-0.04, "0°"),
    (1e-14, "0°"),
    (359.99, "0°"),
    (180.0, "180°"),
    (237.2, "122.8°W"),
    (140.0, "140°E"),
    (179.94, "179.9°E"),
]
LAT_CASES = [
    (-0.04, "0°"),
    (1e-14, "0°"),
    (0.0, "0°"),
    (49.0, "49°N"),
    (-20.5, "20.5°S"),
]


@pytest.mark.parametrize("value, expected", LON_CASES)
def test_a_longitude_tick_is_recognised_as_a_meridian_after_rounding(value, expected):
    assert format_lon(value) == expected


@pytest.mark.parametrize("value, expected", LAT_CASES)
def test_a_latitude_tick_is_recognised_as_the_equator_after_rounding(value, expected):
    assert format_lat(value) == expected


def test_the_interactive_formatters_round_before_testing_for_the_meridians():
    from ocean_skill.plot._geo_labels import lat_tick_js, lon_tick_js

    # no JS engine here, so the shared table above pins the Python and this pins the
    # JS: the same toFixed(1) rounding, tested for 0 (and 180) on the rounded number
    assert "toFixed(1)" in lon_tick_js and "toFixed(1)" in lat_tick_js
    assert "n === 0 || n === 180" in lon_tick_js
    assert "n === 0" in lat_tick_js
    # and the unrounded tests that mishandled 179.96 / -0.04 are gone
    assert "v === 0" not in lon_tick_js and "tick === 0" not in lat_tick_js
