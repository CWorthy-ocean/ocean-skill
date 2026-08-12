"""Backend-agnostic plot specification.

A :class:`PlotSpec` says *what* to draw — a family, the prepared comparisons, and
styling options — but nothing about *how*. Renderers registered in
:mod:`ocean_skill.plot.registry` consume it, so the same spec can be drawn statically
with matplotlib or interactively with holoviews by naming a different renderer.

``items`` is the payload: one entry per comparison, each with the aligned pair and
the metadata a renderer needs to label it. That is deliberately the shape the plotting
functions already accept, so specs route rather than duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["FAMILIES", "PlotSpec"]

#: Plot families a renderer may implement. A renderer need not support all of them —
#: :func:`ocean_skill.plot.registry.render` reports clearly when one is missing, and the
#: holoviews renderer deliberately delegates ``taylor`` back to matplotlib.
#:
#: The two ``*_movie`` families are the animated forms of the two static map families,
#: paired as ``field_grid``/``field_movie`` (a comparison row, stacked down the page or
#: played) and ``field_facet``/``facet_movie`` (one source's facet axis, laid out or
#: played). Each is a family of its own rather than a ``mode=`` on its static twin
#: because the frames are a different payload: a movie's items *are* the sequence, so
#: the family has to say whether that sequence becomes panels or frames.
FAMILIES = (
    "field_row",
    "field_grid",
    "field_facet",
    "field_movie",
    "facet_movie",
    "taylor",
    "target",
    "paired",
)


@dataclass
class PlotSpec:
    """What to draw: a family, the comparisons to draw, and styling options.

    Parameters
    ----------
    family
        One of :data:`FAMILIES`.
    items
        One dict per comparison: ``aligned`` (the test/reference/difference Dataset),
        plus optional ``metrics``, ``units``, ``standard_name`` and ``label``. The
        ``field_facet`` family is the exception, carrying a single item with ``field``
        (one DataArray) and ``facet_dim`` instead of ``aligned`` — it draws one source
        rather than a pair, so there is no aligned trio to carry. ``facet_movie``
        carries that same single item, being ``field_facet`` played rather than laid
        out; ``field_movie`` carries ``field_grid``'s list, one entry per frame, each
        optionally naming its ``frame_label``.
    options
        Renderer-agnostic styling (title, labels, mark, colour grouping, figsize, ...).
        Renderers ignore options they do not understand rather than failing.
    """

    family: str
    items: list[dict[str, Any]] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(
                f"unknown plot family {self.family!r}; expected {FAMILIES}"
            )

    @property
    def single(self) -> dict[str, Any]:
        """The sole item, for families that draw exactly one comparison."""
        if not self.items:
            raise ValueError(f"{self.family!r} needs one comparison, got none")
        return self.items[0]
