"""One spelling of a longitude or latitude, for titles and tick labels alike.

A map's gridline labels come from cartopy (``125°W``, ``0°``, ``180°``, ``140°E``): the
longitude wrapped to ±180 with a hemisphere letter. Everything else that names a
degree -- a station's title, a section's x axis, an extremum's suptitle -- reads off
these helpers instead, so the same longitude is never ``237.2°E`` on one panel and
``122.8°W`` on the next. A model stored in 0-360 hands over ``237.2`` all the same;
how it is *spelled* is not the data's business.
"""

from __future__ import annotations

__all__ = [
    "format_lat",
    "format_lon",
    "lat_tick_js",
    "lon_tick_js",
    "tick_count",
    "wrap_lon",
]

#: Inches one degree tick label needs (the text, ``175°E``, plus air on both sides).
_INCHES_PER_TICK = 0.65


def tick_count(width_inches: float) -> int:
    """How many longitude/latitude ticks an axis ``width_inches`` wide can label.

    At most seven, fewer on a narrow panel: a section crossing the dateline spells every
    tick in full (``175°E``, ``160°W``), and five of those on a two-inch panel run
    together. At least two, so an axis always says where it is.
    """
    return max(2, min(7, round(width_inches / _INCHES_PER_TICK)))



def wrap_lon(lon: float) -> float:
    """Return ``lon`` in ``(-180, 180]``: 237.2 reads -122.8, and -180 reads 180."""
    wrapped = ((float(lon) + 180.0) % 360.0) - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _degrees(value: float) -> str:
    """``125``, ``122.8``: one decimal at most, a trailing ``.0`` dropped."""
    return f"{abs(value):.1f}".removesuffix(".0")


def format_lon(lon: float) -> str:
    """Spell a longitude tick as cartopy does: ``"122.8°W"``, ``"140°E"``, ``"180°"``.

    The meridians 0 and 180 carry no hemisphere letter, which is what cartopy prints on
    a map's gridlines -- so a section's longitude axis and a map's agree character for
    character. They are recognised *as printed*: 179.96 rounds to ``"180°"`` and -0.04
    to ``"0°"``, rather than ``"180.0°E"`` or ``"0°W"``.
    """
    lon = wrap_lon(lon)
    shown = round(abs(lon), 1)
    if shown in (0.0, 180.0):
        return f"{_degrees(shown)}°"
    return f"{_degrees(lon)}°{'E' if lon >= 0 else 'W'}"


def format_lat(lat: float) -> str:
    """Spell a latitude tick as cartopy does: ``"49°N"``, ``"20.5°S"``, ``"0°"``."""
    lat = float(lat)
    if round(abs(lat), 1) == 0.0:
        return "0°"
    return f"{_degrees(lat)}°{'N' if lat >= 0 else 'S'}"


#: Bokeh ``CustomJSTickFormatter`` bodies for the same two spellings, so the
#: interactive renderer's longitude/latitude axes read exactly as the static ones do.
#: The 0/180 test is on the *rounded* number, as in :func:`format_lon`.
lon_tick_js = """
let v = ((tick + 180) % 360 + 360) % 360 - 180;
if (v === -180) { v = 180; }
const n = +Math.abs(v).toFixed(1);
if (n === 0 || n === 180) { return n + "°"; }
return n + "°" + (v > 0 ? "E" : "W");
"""

lat_tick_js = """
const n = +Math.abs(tick).toFixed(1);
if (n === 0) { return "0°"; }
return n + "°" + (tick > 0 ? "N" : "S");
"""
