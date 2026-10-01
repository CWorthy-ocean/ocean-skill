"""What statistic a plotted field is, as both renderers need to know it.

An ``aggregate`` that reduces a field -- the variance of a seasonal cycle, the standard
deviation over a year -- leaves the result carrying ``attrs["statistic"]`` (the
reduction's name) and ``attrs["units"]`` already rewritten to match
(:func:`ocean_skill.units.for_statistic`: ``delta_degC^2``, ``(mg/m^3)^2``). Two things
a renderer decides depend on that and on nothing else it can see: whether the standard
name's pinned display range and log scale apply (they do not to a spread, see
:func:`ocean_skill.colormaps.norm_for`), and how the units read on a colour bar.

Both renderers ask here, in one place, so the static and interactive figures agree on
which statistic a panel shows and on how its units are spelled.
"""

from __future__ import annotations

from typing import Any

__all__ = ["statistic_of", "units_text"]

#: The DataArrays of an item's ``aligned`` Dataset, in the order worth believing: the
#: two sources first, then a single-source ``value``. ``difference`` is last because a
#: difference of two reductions has often been rebuilt without the attributes.
_ROLES = ("test", "reference", "value", "difference")

#: Item keys that hold one DataArray, for the single-source families.
_ARRAY_KEYS = ("field", "value")


def _own(obj) -> Any:
    attrs = getattr(obj, "attrs", None)
    return attrs.get("statistic") if isinstance(attrs, dict) else None


def statistic_of(obj) -> str | None:
    """Return the ``statistic`` a plotted thing carries, or ``None``.

    ``obj`` is whatever the caller has to hand: a ``PlotSpec`` item (the ``statistic``
    key, if the builder set one, wins; otherwise the ``attrs["statistic"]`` of its
    ``field`` or its ``aligned`` trio), the ``aligned`` Dataset or mapping itself, or
    one DataArray. ``None`` -- the answer for every plain field -- changes nothing
    downstream.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        stat = obj.get("statistic")
        if isinstance(stat, str) and stat:
            return stat
        for key in _ARRAY_KEYS:
            stat = _own(obj.get(key))
            if stat:
                return stat
        aligned = obj.get("aligned")
        if aligned is not None:
            return statistic_of(aligned)
        # a bare ``aligned`` mapping (the tests and field_row callers pass one)
        for role in _ROLES:
            stat = _own(obj.get(role))
            if stat:
                return stat
        return None
    stat = _own(obj)
    if stat:
        return str(stat)
    data_vars = getattr(obj, "data_vars", None)
    if data_vars is not None:  # a Dataset
        for role in _ROLES:
            if role in data_vars:
                stat = _own(data_vars[role])
                if stat:
                    return str(stat)
    return None


def units_text(units, statistic=None) -> str:
    """Return ``units`` as a colour bar or axis should print it.

    Unchanged for a plain field, so no existing label moves. For a *spread* statistic
    (:func:`ocean_skill.units.is_spread`) the units are the machine spelling
    ``for_statistic`` left (``delta_degC^2``) and are shown readably instead
    (:func:`ocean_skill.units.display`: ``°C²``). ``None``/empty units give ``""``.
    """
    if not units:
        return ""
    from ocean_skill.units import display, is_spread

    return display(str(units)) if is_spread(statistic) else str(units)
