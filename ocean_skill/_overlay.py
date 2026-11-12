"""Pair a plot's ``contours=`` overlay with the panels it is drawn over.

``po4.plot(contours=temp)`` draws temperature isotherms over a phosphate section -- a
second variable, built the same way as the first (same source, same ``select`` and
``aggregate``), so that it lands on the same section grid -- and ``omz.plot(
contours=o2)`` draws oxygen isolines over a map the same way (a map's overlay may sit
on another grid; see :mod:`ocean_skill.plot.map_contours`). The overlay is a whole
object of the same shape as the one plotted: a :class:`~ocean_skill.field.Field` over
a ``Field``, a :class:`~ocean_skill.field.FieldSet` over a ``FieldSet`` (paired by
position), a :class:`~ocean_skill.comparison.Comparison` over a ``Comparison`` (its
test over the test panel, its reference over the reference panel), and so on.

This module only pairs and checks; it reads nothing new. Each plotted item gains
three keys a renderer reads -- ``contour`` (the overlay's own section data: one
DataArray for a field, a ``{"test", "reference"}`` pair for a comparison),
``contour_standard_name`` and ``contour_units`` (for the interactive hover) -- and
on a section the renderer then puts the overlay on the panel's own grid and refuses
one that does not match (:func:`ocean_skill.plot.section.prepare_overlay`), rather
than regridding it silently. ``contours=True`` is resolved by each ``plot()`` to the
plotted object itself before it reaches here.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "COMPARISON_FAMILIES",
    "FIELD_FAMILIES",
    "comparison_contour",
    "contour_members",
    "field_contour",
    "map_contour",
]


def _members(obj: Any, kind: str) -> list[Any] | None:
    """Return ``obj``'s flat members of ``kind`` (field/comparison), or ``None``."""
    from ocean_skill.comparison import Comparison, ComparisonSet
    from ocean_skill.field import Field, FieldSet

    single, group, attr = (
        (Field, FieldSet, "fields")
        if kind == "field"
        else (Comparison, ComparisonSet, "comparisons")
    )
    if isinstance(obj, single):
        return [obj]
    if isinstance(obj, group):
        return list(getattr(obj, attr))
    if isinstance(obj, list | tuple):
        out: list[Any] = []
        for member in obj:
            flat = _members(member, kind)
            if flat is None:
                return None
            out.extend(flat)
        return out
    return None


def contour_members(contours: Any, n: int, *, kind: str) -> list[Any]:
    """Return ``contours`` as exactly ``n`` overlay members, one per plotted panel/row.

    ``kind`` is what was plotted -- ``"field"`` or ``"comparison"`` -- and the overlay
    must be the same kind: a field's panel has one value to contour, a comparison's
    row two (test and reference). A set, or a list of objects and sets, is flattened
    in order -- the order the plotted set flattens its own members in.
    """
    members = _members(contours, kind)
    if members is None:
        want = (
            "a Field, a FieldSet, or a list of them"
            if kind == "field"
            else "a Comparison, a ComparisonSet, or a list of them"
        )
        raise TypeError(
            f"contours= over a {kind} takes {want} -- the same kind of object as "
            f"the one plotted, built the same way -- got {type(contours).__name__}."
        )
    if len(members) != n:
        what = "panel" if kind == "field" else "row"
        raise ValueError(
            f"contours= has {len(members)} {kind}(s) for {n} {what}(s) -- it pairs "
            f"one overlay with each {what}, in the same order, so the two must "
            "have the same length."
        )
    return members


#: Plotted families that take an overlay: a field's section or map, and a comparison
#: row's section or map (test | reference | difference -- the difference panel never
#: gets lines). Movies, skill maps and depth-time panels draw none.
FIELD_FAMILIES = ("section", "field_facet")
COMPARISON_FAMILIES = ("section_row", "field_row")


def _refuse_family(family: str, reason: str, who: str) -> None:
    raise ValueError(
        f"contours= draws contour lines over a vertical section or a map, but {who} "
        f"draws as {family!r} ({reason}). Overlays on depth-time panels, skill maps "
        "and movies are not supported."
    )


def _check_families(overlay: Any, plotted: Any, families: tuple[str, ...], what: str):
    """Refuse a plotted family with no lines, or an overlay that draws differently."""
    if plotted.family not in families:
        _refuse_family(plotted.family, plotted.family_reason, f"the plotted {what}")
    if overlay.family != plotted.family:
        shape = "a section" if plotted.family.startswith("section") else "a map"
        raise ValueError(
            f"contours= over {shape} takes an overlay that is {shape} too, but the "
            f"overlay draws as {overlay.family!r} ({overlay.family_reason}) -- build "
            "it the way the plotted one was built (same source, select= and "
            "aggregate=) with the variable to draw as lines."
        )


def field_contour(overlay: Any, *, plotted: Any) -> dict[str, Any]:
    """Return the item keys that draw ``overlay``'s lines over field ``plotted``.

    For a section, ``contour`` is the overlay's own section data. For a map it is
    ``None`` here: which slice of the overlay a map draws (its surface default, a
    level split across a set's panels) is the plotted field's own business, so the
    caller fills it with :func:`map_contour` once it has built the map's item.
    """
    _check_families(overlay, plotted, FIELD_FAMILIES, "field")
    if plotted.family == "field_facet":
        return {
            "contour": None,
            "contour_standard_name": overlay.standard_name,
            "contour_units": None,
        }
    overlay._require_section_shape()
    return {
        "contour": overlay.data,
        "contour_standard_name": overlay.standard_name,
        "contour_units": overlay.data.attrs.get("units"),
    }


def map_contour(
    keys: dict[str, Any], overlay_map: Any, plotted_map: Any
) -> dict[str, Any]:
    """Fill :func:`field_contour`'s map keys with the overlay's own map, checked.

    ``overlay_map`` is the overlay's DataArray built exactly as the plotted item's
    ``field`` was (the same ``_facet_item_or_refuse``/``_map_items_of`` path, so the
    same surface default and level split); ``plotted_map`` is that ``field``. The two
    must stand on the same facet axes -- see
    :func:`ocean_skill.plot.map_contours.check_map_overlay`.
    """
    from ocean_skill.plot.map_contours import check_map_overlay

    check_map_overlay(overlay_map, plotted_map)
    return {
        **keys,
        "contour": overlay_map,
        "contour_units": overlay_map.attrs.get("units"),
    }


def comparison_contour(overlay: Any, *, plotted: Any) -> dict[str, Any]:
    """Return the item keys that draw ``overlay``'s lines over a comparison row.

    ``contour`` is ``{"test", "reference"}``, the overlay's own aligned pair: its test
    lines go over the test panel, its reference lines over the reference panel, and
    the difference panel draws none. On a map row each side is checked to be a map
    standing on the same axes as the panel it is drawn over.
    """
    _check_families(overlay, plotted, COMPARISON_FAMILIES, "comparison")
    aligned = overlay.aligned
    pair = {"test": aligned["test"], "reference": aligned["reference"]}
    if plotted.family == "field_row":
        from ocean_skill.plot.map_contours import check_map_overlay

        panels = plotted.aligned
        for side in ("test", "reference"):
            check_map_overlay(pair[side], panels[side])
    return {
        "contour": pair,
        "contour_standard_name": overlay.standard_name,
        "contour_units": aligned["reference"].attrs.get("units"),
    }
