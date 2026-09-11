"""Resolving the ``AUTO`` default shared by ``summary_weights=``
(:mod:`ocean_skill.plot.summary`) and ``weights=`` (:mod:`ocean_skill.plot.map_metrics`).

Both default to weighting by effective sample size (``"n_eff"``, computed
automatically by :meth:`ocean_skill.comparison.Comparison.metrics` -- see its
docstring) whenever the records being drawn carry it: a summary star or an
interpolated map that pools a long, highly-autocorrelated mooring record together
with a handful of independent CTD casts should not treat every one of them as
equally good evidence. But defaulting a figure to weighted is a real change in what
it shows, so it is never silent -- it warns once, naming what happened and how to
turn it off. One resolver, used by both the static summary/map code and the
interactive summary renderer, so the default and its warning can never drift
between them.
"""

from __future__ import annotations

import warnings
from typing import Any

from ocean_skill import _stacklevel

__all__ = ["AUTO", "resolve"]


class _Auto:
    """Sentinel type for :data:`AUTO` -- its own class only so ``repr()`` reads as
    ``AUTO`` in a signature or an error message, rather than some opaque object.
    """

    def __repr__(self) -> str:
        return "AUTO"


#: The default for ``summary_weights=``/``weights=``: "weight by effective sample
#: size when the records carry it, and say so; otherwise behave exactly as
#: ``None`` always has." Passing ``None`` (or ``False``) explicitly instead opts
#: out silently -- it is not this sentinel, so it is never mistaken for it.
AUTO: Any = _Auto()


def resolve(
    available: bool,
    requested: Any,
    *,
    param_name: str,
    field: str = "n_eff",
    warn: bool = True,
) -> str | None:
    """Resolve a ``summary_weights=``/``weights=`` argument to an actual column
    name (or ``None`` for unweighted).

    ``available`` says whether at least one record/row being drawn carries
    ``field`` -- the caller computes this (a records list vs. a DataFrame need
    different checks), so this function stays agnostic of either shape.

    * ``requested is AUTO`` (the default) and ``available`` -- returns ``field``,
      and (if ``warn``) warns once that the figure is weighted by effective
      sample size and names ``param_name=None`` as how to disable it.
    * ``requested is AUTO`` and not ``available`` -- returns ``None``, no
      warning: back-compat for records with no ``field`` at all (hand-built
      records, or predate this existing).
    * ``requested`` is ``None``/``False`` -- returns ``None``, no warning: a
      deliberate opt-out.
    * anything else -- returned unchanged, the explicit column name it always
      was.

    ``warn=False`` lets a caller resolve silently even when ``available`` --
    :func:`ocean_skill.plot.map_metrics.interpolate_records` uses this for a
    ``method`` that would ignore the weights anyway, so ``AUTO`` never raises the
    same "weights would have no effect" warning twice under two different
    messages.

    The warning itself is raised with :func:`ocean_skill._stacklevel.find`, not a
    hand-counted ``stacklevel=``, since this one function is called from several
    different nesting depths (directly from :func:`~ocean_skill.plot.summary.taylor`/
    :func:`~ocean_skill.plot.summary.target`, from
    :func:`~ocean_skill.plot.map_metrics.interpolate_records`, and from the
    interactive summary renderer) -- a fixed number right for one caller would
    misattribute the others.
    """
    if requested is AUTO:
        if not available:
            return None
        if warn:
            warnings.warn(
                f"Weighting by effective sample size {field!r} (pass "
                f"{param_name}=None to disable).",
                stacklevel=_stacklevel.find(),
            )
        return field
    if requested is None or requested is False:
        return None
    return requested
