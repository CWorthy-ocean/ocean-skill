"""Pair a plot's ``contours=`` overlay with the panels it is drawn over.

``po4.plot(contours=temp)`` draws temperature isotherms over a phosphate section -- a
second variable, built the same way as the first (same source, same ``select`` and
``aggregate``), so that it lands on the same section grid. The overlay is a whole
object of the same shape as the one plotted: a :class:`~ocean_skill.field.Field` over
a ``Field``, a :class:`~ocean_skill.field.FieldSet` over a ``FieldSet`` (paired by
position), a :class:`~ocean_skill.comparison.Comparison` over a ``Comparison`` (its
test over the test panel, its reference over the reference panel), and so on.

This module only pairs and checks; it reads nothing new. Each plotted item gains
three keys a renderer reads -- ``contour`` (the overlay's own section data: one
DataArray for a field, a ``{"test", "reference"}`` pair for a comparison),
``contour_standard_name`` and ``contour_units`` (for the interactive hover) -- and
the renderer then puts the overlay on the panel's own grid and refuses one that does
not match (:func:`ocean_skill.plot.section.prepare_overlay`), rather than
regridding it silently.
"""

from __future__ import annotations

from typing import Any

__all__ = ["comparison_contour", "contour_members", "field_contour"]


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


def _refuse_non_section(family: str, reason: str, who: str) -> None:
    raise ValueError(
        f"contours= draws contour lines over a vertical section, but {who} draws "
        f"as {family!r} ({reason}). Overlays on maps and depth-time panels are not "
        "supported yet."
    )


def field_contour(overlay: Any, *, plotted: Any) -> dict[str, Any]:
    """Return the item keys that draw ``overlay``'s lines over field ``plotted``."""
    if plotted.family != "section":
        _refuse_non_section(plotted.family, plotted.family_reason, "the plotted field")
    if overlay.family != "section":
        _refuse_non_section(overlay.family, overlay.family_reason, "the overlay")
    overlay._require_section_shape()
    return {
        "contour": overlay.data,
        "contour_standard_name": overlay.standard_name,
        "contour_units": overlay.data.attrs.get("units"),
    }


def comparison_contour(overlay: Any, *, plotted: Any) -> dict[str, Any]:
    """Return the item keys that draw ``overlay``'s lines over a comparison row."""
    if plotted.family != "section_row":
        _refuse_non_section(
            plotted.family, plotted.family_reason, "the plotted comparison"
        )
    if overlay.family != "section_row":
        _refuse_non_section(overlay.family, overlay.family_reason, "the overlay")
    aligned = overlay.aligned
    return {
        "contour": {"test": aligned["test"], "reference": aligned["reference"]},
        "contour_standard_name": overlay.standard_name,
        "contour_units": aligned["reference"].attrs.get("units"),
    }
