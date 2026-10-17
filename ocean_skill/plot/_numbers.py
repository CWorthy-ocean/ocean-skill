"""One way to print a number on a figure or in a table.

Annotations (``max 8440``), table cells and summary lines all go through
:func:`format_value`, so a value reads the same wherever it appears: plain digits
for everyday magnitudes, scientific notation only when plain digits would be
unreadable (``3.4e-22``), never ``8.44e+03``.
"""

from __future__ import annotations

import math
from typing import Any

#: Magnitudes inside [_SMALL, _LARGE) print as plain digits.
_SMALL = 1e-4
_LARGE = 1e6


def format_value(value: Any, sig: int = 3) -> str:
    """Return ``value`` rounded to ``sig`` significant figures, for display.

    Plain digits for magnitudes from 0.0001 up to a million (``8440``, ``0.0412``,
    ``-2.32``), scientific notation outside that range (``3.4e-22``, ``1.2e+07``).
    Integers keep every digit (``776301``); whole numbers drop a trailing ``.0``.
    NaN reads ``NaN``; ``None`` and non-numbers are returned as ``str(value)``.
    """
    if value is None or isinstance(value, bool):
        return str(value)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0:
        return "0"
    if isinstance(value, int) or (hasattr(value, "dtype") and value.dtype.kind in "iu"):
        return f"{int(value)}"
    magnitude = abs(x)
    if magnitude < _SMALL or magnitude >= _LARGE:
        mantissa, exponent = f"{x:.{max(sig - 1, 0)}e}".split("e")
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{exponent}"
    digits = sig - 1 - math.floor(math.log10(magnitude))
    rounded = round(x, digits)
    if digits <= 0:
        return f"{int(rounded)}"
    text = f"{rounded:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text
