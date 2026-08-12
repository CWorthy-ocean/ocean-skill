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
#: ``field_movie`` takes the same ``items`` as ``field_grid`` and stacks them in *time*
#: rather than down the page: one ``test | reference | difference`` row, redrawn per
#: item. That is why it is a family of its own rather than an option on ``field_row`` —
#: the payload is a list, as ``field_grid``'s is.
FAMILIES = ("field_row", "field_grid", "field_movie", "taylor", "target", "paired")


@dataclass
class PlotSpec:
    """What to draw: a family, the comparisons to draw, and styling options.

    Parameters
    ----------
    family
        One of :data:`FAMILIES`.
    items
        One dict per comparison: ``aligned`` (the test/reference/difference Dataset),
        plus optional ``metrics``, ``units``, ``standard_name`` and ``label``.
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
