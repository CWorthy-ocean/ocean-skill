"""Shared semantics for a family's ``titles=`` override.

Every multi-panel family auto-generates its own panel titles from data and
metadata (the variable, the facet value, the station, the metric name, ...).
``titles=`` lets a caller override some or all of them by hand, identically
across every family and both renderers -- see :func:`resolve_titles`, the one
place that decides what a ``titles=`` list means once a family has worked out
its own auto titles.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["map_panel_titles", "map_title_parts", "resolve_titles", "skill_map_title"]


def resolve_titles(
    auto: Sequence[str], titles: Sequence[str | None] | None
) -> list[str]:
    """Apply a ``titles=`` override onto a family's auto-generated ``auto`` titles.

    ``titles=None`` leaves every title exactly as ``auto`` computed it.
    Otherwise ``titles`` needs one entry per panel, in the same order
    ``auto`` lists them (row-major: left-to-right, then top-to-bottom):
    ``None`` at a position keeps that panel's auto title, and any string
    (including ``""``, to blank it) replaces it. The wrong count raises a
    copy-pasteable ``ValueError`` listing the current auto titles.
    """
    if titles is None:
        return list(auto)
    titles = list(titles)
    if len(titles) != len(auto):
        current = "\n".join(f"  {i + 1}. {t!r}" for i, t in enumerate(auto))
        raise ValueError(
            f"titles needs one entry per panel -- this figure draws "
            f"{len(auto)}:\n{current}\ngot {len(titles)}. Copy the list "
            "above, edit the text (or leave an entry None to keep it), and "
            "pass it back in the same order."
        )
    return [t if t is not None else a for a, t in zip(auto, titles, strict=True)]


def map_title_parts(item) -> tuple[str, ...]:
    """Return what identifies one map of a set, in reading order: its title parts.

    The variable, the source (the member's own label, less the statistic phrase a
    statistics fan appended to it), the statistic (``"mean"``, ``"std of monthly
    means"``), the level and the instant -- each ``""`` when the item has none. Whether
    a part reaches a title is :func:`map_panel_titles`'s call: only the parts that
    differ between the maps do.
    """
    from ocean_skill.plot.matplotlib_renderer import field_title

    aggregate = item.get("aggregate") or ""
    return (
        field_title(item.get("standard_name")),
        item.get("source_label") or item.get("label") or "",
        aggregate,
        item.get("depth") or "",
        # an aggregate's own time phrase restates the statistic (and the window, which
        # every member of a statistics fan shares); the statistic already says it
        "" if aggregate else item.get("time") or "",
    )


def map_panel_titles(items) -> list[str]:
    """One title per map of an unfaceted set, naming only what tells the maps apart.

    Every part (:func:`map_title_parts`) that reads the same on all the maps is left to
    the suptitle (or to nothing, for the source), so a set fanned over variables is
    titled by variable as it always was, and a set fanned over statistics -- the mean
    and the monthly variability of one variable -- by statistic, rather than the same
    variable name over every panel. Parts that differ join with `` · `` in reading
    order. A set whose maps differ in no part at all (cannot be built from a
    :class:`~ocean_skill.field.FieldSet`, but a hand-made item list can) falls back to
    the variable.
    """
    from ocean_skill.plot.matplotlib_renderer import _elide

    parts = [map_title_parts(item) for item in items]
    varying = [
        k
        for k in range(len(parts[0]))
        if len({row[k] for row in parts}) > 1
    ]
    if not varying:
        return [row[0] for row in parts]
    return [" · ".join(_elide(row[k]) for k in varying if row[k]) for row in parts]


def skill_map_title(items) -> str:
    """Return the suptitle a ``skill_map`` carries when the caller named none.

    That is the variable.

    The panels are metrics (bias, crmsd, corr, ...) and a stacked figure's rows are
    named by whatever distinguishes the comparisons (the depth, the period), so
    nothing on the figure said *which variable* was scored -- a page of per-cell skill
    for alkalinity and one for oxygen differed only in their colorbar units. Named only
    when every row is of one variable; a single comparison, whose depth has no row label
    to carry it, also gets its depth. Spelled as every other title spells a variable
    (:func:`ocean_skill.plot.matplotlib_renderer.field_title`). ``""`` when the rows
    mix variables (or carry none), so nothing is guessed.
    """
    from ocean_skill.plot.matplotlib_renderer import field_title

    names = list(
        dict.fromkeys(field_title(item.get("standard_name")) for item in items)
    )
    if len(names) != 1 or not names[0]:
        return ""
    depth = items[0].get("depth") if len(items) == 1 else None
    return f"{names[0]} · {depth}" if depth else names[0]
