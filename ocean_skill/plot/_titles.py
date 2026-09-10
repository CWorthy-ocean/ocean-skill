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

__all__ = ["resolve_titles"]


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
