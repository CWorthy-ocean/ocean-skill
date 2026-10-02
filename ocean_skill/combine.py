"""Draw several comparisons, or several fields, together in one figure: ``osk.plot``.

:func:`compare` hands back a :class:`~ocean_skill.comparison.ComparisonSet` and
:func:`~ocean_skill.field.field` a :class:`~ocean_skill.field.Field` or
:class:`~ocean_skill.field.FieldSet`, each of which draws itself. What none of them
can do is draw results that were *built separately* -- three ``osk.compare(...)``
calls with three different selects are three sets, and nothing joins them. This
module is that join, nothing more::

    eq = osk.compare(test=model, reference=woa, variables=["temp"],
                     select={"transect": {"lat": 0, "lon": {"min": 143, "max": 267}},
                             "depth": depths},
                     aggregate={"time": "mean"})
    band = osk.compare(test=model, reference=woa, variables=["temp"],
                       select={"lon": {"min": 180, "max": 200},
                               "lat": {"min": -30, "max": 30}, "depth": depths},
                       aggregate={"time": "mean", "lon": "mean"})
    osk.plot({"Eq": eq, "180-160": band}, shared_limits=True)   # keys label the rows

It only decides *which set to build* and forwards the rest. Comparisons go to
:class:`~ocean_skill.comparison.ComparisonSet`, fields to
:class:`~ocean_skill.field.FieldSet`, and each set's own ``plot`` makes the figure
-- so what can sit beside what, and how, is exactly what those two already decide
(a lone member simply draws as itself; several sections stack as rows; a set that
mixes shapes is refused by the set with its own explanation). A figure cannot mix
comparisons with fields: a comparison is a ``test | reference | difference`` row
and a field is one panel, with nothing shared to lay out between them.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["plot"]


def _kind(obj: Any) -> str | None:
    """``"comparison"``/``"field"`` for a recognized object, else ``None``."""
    from ocean_skill.comparison import Comparison, ComparisonSet
    from ocean_skill.field import Field, FieldSet

    if isinstance(obj, Comparison | ComparisonSet):
        return "comparison"
    if isinstance(obj, Field | FieldSet):
        return "field"
    return None


def _flatten_fields(obj: Any) -> list[Any]:
    """Return the :class:`~ocean_skill.field.Field` members of a field or a set."""
    from ocean_skill.field import FieldSet

    return list(obj.fields) if isinstance(obj, FieldSet) else [obj]


def _relabelled(field: Any, label: str) -> Any:
    """Return ``field`` under another label, leaving the caller's own untouched.

    A shallow copy rather than a rebuilt :class:`~ocean_skill.field.Field`
    (:meth:`~ocean_skill.field.Field._replace`): the copy shares the already
    prepared data, so naming a row costs no second read, and the original keeps
    the label it was given.
    """
    out = copy.copy(field)
    out.label = label
    return out


def plot(items: Any, *, renderer: str = "matplotlib", **plot_kwargs: Any):
    """Draw several comparisons, or several fields, as one figure.

    Parameters
    ----------
    items
        A list or tuple of :class:`~ocean_skill.comparison.Comparison` /
        :class:`~ocean_skill.comparison.ComparisonSet` objects, or of
        :class:`~ocean_skill.field.Field` / :class:`~ocean_skill.field.FieldSet`
        objects -- not a mix of the two. A ``{label: item}`` dict instead names each
        member: the key becomes its row label (a member that is itself a set of
        several is labelled ``"key: own label"``). A single object is drawn as
        itself.
    renderer
        ``"matplotlib"`` (default, static) or ``"holoviews"`` (interactive).
    **plot_kwargs
        Plot options, forwarded unchanged to the set's own ``plot`` -- ``title``,
        ``shared_limits``, ``save``, the ``*_kwargs`` styling dicts, and the rest of
        ``docs/plot_styling_reference.md``.

    Comparisons are pooled with :class:`~ocean_skill.comparison.ComparisonSet`
    (so ``labels`` and nesting behave exactly as they do there) and fields with
    :class:`~ocean_skill.field.FieldSet`; the figure is whatever that set's
    ``plot`` returns. Which shapes can share a figure is therefore the sets' own
    rule: several vertical sections, for instance, stack as rows, while a set that
    mixes maps with sections says so and asks for them separately.

    Raises ``ValueError`` for an empty list or a mix of comparisons and fields, and
    ``TypeError`` for anything that is neither.
    """
    if isinstance(items, dict):
        names = list(items.keys())
        members = list(items.values())
    elif isinstance(items, list | tuple):
        names = None
        members = list(items)
    elif _kind(items) is not None:
        names = None
        members = [items]
    else:
        raise TypeError(
            "osk.plot() takes a list (or {label: item} dict) of comparisons or "
            f"fields, got {type(items).__name__}: {items!r}."
        )
    if not members:
        raise ValueError(
            "osk.plot() got nothing to draw: the list is empty. Pass the "
            "comparisons (osk.compare(...)) or fields (osk.field(...)) to put on "
            "one figure."
        )

    kinds = []
    for i, member in enumerate(members):
        kind = _kind(member)
        if kind is None:
            where = f"[{names[i]!r}]" if names is not None else f"[{i}]"
            raise TypeError(
                f"osk.plot() item {where} is a {type(member).__name__}, which is "
                "neither a comparison (Comparison/ComparisonSet, from "
                "osk.compare) nor a field (Field/FieldSet, from osk.field): "
                f"{member!r}."
            )
        kinds.append(kind)
    if len(set(kinds)) > 1:
        what = ", ".join(
            f"{names[i] if names is not None else i}: {type(m).__name__}"
            for i, m in enumerate(members)
        )
        raise ValueError(
            "osk.plot() cannot put comparisons and fields on one figure -- a "
            "comparison is a test | reference | difference row and a field is a "
            f"single panel, with no shared layout between them ({what}). Plot "
            "the comparisons together and the fields together, separately."
        )

    if kinds[0] == "comparison":
        from ocean_skill.comparison import ComparisonSet

        pooled = ComparisonSet(dict(zip(names, members)) if names else members)
        return pooled.plot(renderer=renderer, **plot_kwargs)

    from ocean_skill.field import FieldSet

    fields: list[Any] = []
    for i, member in enumerate(members):
        group = _flatten_fields(member)
        if names is not None:
            key = str(names[i])
            group = [
                _relabelled(
                    f, key if len(group) == 1 else f"{key}: {f.label or f.source}"
                )
                for f in group
            ]
        fields.extend(group)
    return FieldSet(fields).plot(renderer=renderer, **plot_kwargs)
