"""Facet resolution for a grid whose panels each draw exactly *one* item.

:mod:`ocean_skill.plot.series`'s own ``rows=``/``cols=`` facet a family whose
panels can hold *several* items apiece (an overlay of lines); its
:func:`~ocean_skill.plot.series.facet_grid` is the shared cross-product engine
both :mod:`~ocean_skill.plot.series` and :mod:`~ocean_skill.plot.profile`
already build on. This module is for the other shape -- a ``time_depth`` mesh
or a map, where every item carries exactly two identity facts, its variable
and its source (see :meth:`ocean_skill.field.Field._time_depth_item`/
:meth:`~ocean_skill.field.Field._map_item`), and a panel can only ever draw
one of them at a time. Naming *either* fact as ``rows=``/``cols=`` already
implies the other for the grid's opposite axis -- :func:`resolve_facets` is
that implication, built on :func:`~ocean_skill.plot.series.facet_grid`;
:func:`one_item_cells` and :func:`facet_grid_titles` are what a
one-item-per-panel family needs beyond it.
"""

from __future__ import annotations

from typing import Any

from ocean_skill.plot.series import facet_grid

__all__ = [
    "facet_grid_titles",
    "facet_key",
    "one_item_cells",
    "resolve_facets",
    "resolve_limit_groups",
]

#: The facet an item's own *other* identity fact implies, when only one of
#: ``rows=``/``cols=`` is given -- every item this module facets carries
#: exactly these two facts (see :func:`facet_key`), so one alone already
#: fixes the grid's opposite axis.
_COMPLEMENT = {"variable": "source", "source": "variable"}

#: ``shared_limits=``'s own accepted spellings, beyond ``True``/``False`` --
#: the same vocabulary ``rows=``/``cols=`` use (see :func:`facet_key`).
_LIMIT_VALUES = ("variable", "standard_name", "source")


def _normalize(by: str) -> str:
    return "variable" if by in ("variable", "standard_name") else by


def facet_key(item: dict[str, Any], by: str, *, family: str, axis_hint: bool = False):
    """Return the value ``item`` is grouped by, for a one-item-per-panel grid's
    ``rows=``/``cols=``/``shared_limits=``.

    Vocabulary matches ``FieldSet._SEL_KEYS``: ``variable`` (alias
    ``standard_name``) is the item's own field identity, ``source`` its
    station/source label -- the same two facts every item this module facets
    carries. ``axis_hint=True`` (the ``time_depth`` family only -- a map has
    no drawn time/depth axis to protect the same way) gives ``time``/``depth``
    their own dedicated refusal, mirroring
    :func:`ocean_skill.plot.profile._group_key`'s identical one for ``depth``.
    """
    normalized = _normalize(by)
    if normalized == "variable":
        return item.get("standard_name") or item.get("label")
    if normalized == "source":
        return item.get("label")
    if axis_hint and by in ("time", "depth"):
        raise ValueError(
            f"cannot facet {family} by {by!r}: it is the axis every panel "
            "already draws against, not a fact to split panels on. Facet on "
            "variable or source instead."
        )
    raise ValueError(
        f"cannot facet {family} by {by!r}; expected one of variable, source."
    )


def resolve_facets(
    items: list[dict[str, Any]],
    rows: str | None,
    cols: str | None,
    *,
    family: str,
    axis_hint: bool = False,
):
    """Return ``None`` (no facet given) or ``(cells, row_values, col_values,
    eff_rows, eff_cols)`` for a ``rows=``/``cols=`` pair on a one-item-per-panel
    grid.

    Naming only one axis implies the other from :data:`_COMPLEMENT` -- a
    ``cols="variable"`` set of stations x variables needs no ``rows="source"``
    to say what the grid's other axis is, there being nothing else it could
    be (see the module docstring). Naming both to the *same* underlying fact
    (``rows="variable", cols="standard_name"``) is refused: there is no
    second fact left to give the grid's other axis. ``eff_rows``/``eff_cols``
    are the normalized keys actually used (``"variable"``/``"source"``), for
    a caller that needs to know which axis holds which -- e.g. to decide
    which one to share an axis range along.

    ``cells``/``row_values``/``col_values`` come straight from
    :func:`~ocean_skill.plot.series.facet_grid`: a (row, column) combination
    nothing matched is an empty list, for :func:`one_item_cells` to draw as a
    hidden blank panel.
    """
    if rows is None and cols is None:
        return None
    if items:
        # Validate whichever key(s) were given against this family's own
        # vocabulary up front, against a real item -- so an invalid rows= or
        # cols= names itself in the error, rather than silently producing no
        # complement and blaming the *other* axis instead.
        sample = items[0]
        if rows is not None:
            facet_key(sample, rows, family=family, axis_hint=axis_hint)
        if cols is not None:
            facet_key(sample, cols, family=family, axis_hint=axis_hint)
    if rows is not None and cols is not None:
        eff_rows, eff_cols = _normalize(rows), _normalize(cols)
        if eff_rows == eff_cols:
            raise ValueError(
                f"rows={rows!r} and cols={cols!r} both name the same fact -- "
                f"a {family} needs two different facts (variable and source) "
                "to build a grid from."
            )
    elif rows is not None:
        eff_rows = _normalize(rows)
        eff_cols = _COMPLEMENT[eff_rows]
    else:
        eff_cols = _normalize(cols)
        eff_rows = _COMPLEMENT[eff_cols]

    indexed = list(enumerate(items))
    cells, row_values, col_values = facet_grid(
        indexed,
        lambda n, item: facet_key(item, eff_rows, family=family, axis_hint=axis_hint),
        lambda n, item: facet_key(item, eff_cols, family=family, axis_hint=axis_hint),
    )
    return cells, row_values, col_values, eff_rows, eff_cols


def one_item_cells(
    cells: list[list[tuple[int, dict]]],
    row_values: list[Any],
    col_values: list[Any],
    *,
    family: str,
) -> list[dict[str, Any] | None]:
    """Return ``cells`` (see :func:`~ocean_skill.plot.series.facet_grid`) as one
    item per grid cell, row-major, ``None`` at a blank -- refusing any cell
    two members landed in, since a mesh or map panel draws exactly one item,
    not an overlay the way a line panel does.
    """
    ncols = len(col_values)
    result: list[dict[str, Any] | None] = []
    for i, cell in enumerate(cells):
        if len(cell) > 1:
            r, c = divmod(i, ncols)
            raise ValueError(
                f"two members of this {family} land in the same cell "
                f"({row_values[r]!r}, {col_values[c]!r}) -- each cell draws "
                "one panel; give the duplicates distinct labels, or drop "
                "rows=/cols=."
            )
        result.append(cell[0][1] if cell else None)
    return result


def facet_grid_titles(
    components: list[tuple[str | None, ...] | None], nrows: int, ncols: int
) -> tuple[str, list[str]]:
    """Return ``(suptitle, per_cell_titles)`` for a faceted one-item-per-panel
    grid.

    ``components`` is ``nrows * ncols`` long, row-major, ``None`` at a blank
    cell; each other entry is that panel's own full identity, as an ordered
    tuple of parts (its variable, its source, and whatever else a family's
    own geometry carries -- see
    :func:`~ocean_skill.plot.matplotlib_renderer.time_depth_grid_titles`,
    which this generalizes past a flat, unfaceted list). A part that reads
    the same on every drawn panel lifts into one suptitle -- exactly that
    function's own rule; a part that instead reads the same across every row
    within one column (or every column within one row) appears once, on
    that column's (or row's) own first drawn cell; anything left stays on
    every drawn panel, in its own original order. A degenerate one-row or
    one-column grid (one facet key given, its complement holding a single
    value) reproduces the flat function's own output exactly -- the
    classification only ever asks whether a part varies, never whether the
    grid is actually two-dimensional.
    """
    from ocean_skill.plot.matplotlib_renderer import _elide

    drawn = [(i, parts) for i, parts in enumerate(components) if parts is not None]
    per_cell = ["" for _ in components]
    if not drawn:
        return "", per_cell
    n_parts = len(drawn[0][1])

    suptitle_parts: list[str | None] = []
    kind: list[str] = []  # "global" | "col" | "row" | "free", one per part index
    for k in range(n_parts):
        values = {parts[k] for _, parts in drawn if parts[k]}
        if len(values) <= 1:
            suptitle_parts.append(next(iter(values), None))
            kind.append("global")
            continue
        suptitle_parts.append(None)
        col_seen: dict[int, Any] = {}
        row_seen: dict[int, Any] = {}
        col_consistent = row_consistent = True
        for i, parts in drawn:
            r, c = divmod(i, ncols)
            v = parts[k]
            if col_seen.setdefault(c, v) != v:
                col_consistent = False
            if row_seen.setdefault(r, v) != v:
                row_consistent = False
        kind.append("col" if col_consistent else "row" if row_consistent else "free")

    # The first (row-major) drawn cell of each column/row -- where a
    # column-bound (or row-bound) part actually gets printed.
    head_of_col: dict[int, int] = {}
    head_of_row: dict[int, int] = {}
    for i, _ in drawn:
        r, c = divmod(i, ncols)
        head_of_col.setdefault(c, i)
        head_of_row.setdefault(r, i)

    for i, parts in drawn:
        r, c = divmod(i, ncols)
        shown = []
        for k, v in enumerate(parts):
            if not v or kind[k] == "global":
                continue
            if kind[k] == "col" and head_of_col[c] != i:
                continue
            if kind[k] == "row" and head_of_row[r] != i:
                continue
            shown.append(v)
        per_cell[i] = " · ".join(_elide(p) for p in shown)

    suptitle = " · ".join(_elide(p) for p in suptitle_parts if p)
    return suptitle, per_cell


def resolve_limit_groups(
    items: list[dict[str, Any]], shared_limits: bool | str
) -> list[list[int]] | None:
    """Return ``None`` for ``shared_limits=False``; one group holding every
    item's index for ``True``; otherwise groups of indices sharing one
    ``facet_key(item, shared_limits)`` value, first-seen order -- the same
    vocabulary ``rows=``/``cols=`` use, so ``shared_limits="variable"`` pools
    each *variable*'s own panels onto one colour scale regardless of how (or
    whether) the grid itself is arranged -- a column built by ``cols=
    "variable"`` included, where it lands each column on its own scale for
    free -- and ``shared_limits="source"`` pools each source's own panels
    instead.

    Warns once per group that would mix ``standard_name``s onto one shared
    colour scale -- impossible by construction for ``"variable"`` itself (a
    group's key *is* that fact), so only ``True`` or ``"source"`` can trigger
    it; ``True``'s own message text is unchanged from before this existed.
    """
    import warnings

    from ocean_skill import _stacklevel

    if shared_limits is False:
        return None
    if shared_limits is True:
        groups = [list(range(len(items)))]
    elif shared_limits in _LIMIT_VALUES:
        order: list[Any] = []
        by_key: dict[Any, list[int]] = {}
        for i, item in enumerate(items):
            key = facet_key(item, shared_limits, family="shared_limits")
            if key not in by_key:
                order.append(key)
                by_key[key] = []
            by_key[key].append(i)
        groups = [by_key[k] for k in order]
    else:
        raise ValueError(
            'shared_limits must be False, True, "variable" (alias '
            f'"standard_name"), or "source"; got {shared_limits!r}.'
        )
    for group in groups:
        names = {items[i].get("standard_name") for i in group}
        if len(names) > 1:
            if shared_limits == "source":
                label = items[group[0]].get("label")
                warnings.warn(
                    f"shared_limits='source' but the {label!r} group mixes "
                    f"variables ({sorted(nm for nm in names if nm)}); their "
                    "ranges/units differ, so one shared colour scale won't "
                    "mean the same thing on every panel.",
                    stacklevel=_stacklevel.find(),
                )
            else:
                warnings.warn(
                    f"shared_limits={shared_limits!r} but panels use "
                    f"different variables ({sorted(nm for nm in names if nm)}); "
                    "their ranges/units differ, so one shared colour scale "
                    "won't mean the same thing on every panel.",
                    stacklevel=_stacklevel.find(),
                )
    return groups
