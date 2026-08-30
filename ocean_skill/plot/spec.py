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
#: ``series`` is the line family — x=time, y=variable. It is named for the *shape of the
#: data* rather than for a mark, because ``mark`` is already an independent option here
#: and that family accepts ``mark="line"``/``"line+marker"``/``"step"``; a family called
#: ``line`` would invite reading the two as one thing.
FAMILIES = (
    "field_row",
    "field_grid",
    "field_facet",
    "field_movie",
    "facet_movie",
    "series",
    "section",
    "section_row",
    "profile",
    "skill_map",
    "taylor",
    "target",
    "paired",
    "locations",
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
        optionally naming its ``frame_label``. ``skill_map`` is the other exception: its
        items carry ``skill`` (a Dataset of one 2-D map per metric) and ``metric_names``
        in place of ``aligned``, because its panels are *metrics of* a comparison rather
        than the comparison's own fields — the ``metrics`` record is still there, as the
        overall value each panel is annotated with.

        The summary families (``taylor``, ``target``, ``paired``) are the smallest
        items of all: ``metrics`` and ``label``, with no field payload at all. That is
        not a subset of the others but the whole of what a diagram of points reads, and
        it is what lets those families pool comparisons the map families could not share
        a figure with — a scored map, an unscored one and a station series are all one
        record of scalars here. See
        :meth:`ocean_skill.comparison.ComparisonSet._metric_items`.

        ``skill_map`` items carry one further *optional* key, ``stations``: a dict of
        ``lon``, ``lat``, ``names`` and ``values`` (one array per metric), drawn as
        dots in each panel's own colour scale on top of the interpolated surface. Only
        :func:`ocean_skill.plot.map_metrics.build_items` sets it — a scored comparison's
        own maps have no scattered stations to mark, and an item without the key draws
        exactly as before.

        ``series`` carries ``field_grid``'s list too, with each ``aligned`` 1-D on
        ``time`` rather than 2-D on a grid (position and depth riding as scalar
        coordinates). Its items are *not* one per row, unlike ``field_grid``'s: the
        family bins them into panels by variable and overlays them by source (see
        :func:`ocean_skill.plot.series.compose`), so one item is a pair of lines rather
        than a row. ``aligned`` usually carries the ``test``/``reference``/
        ``difference`` trio, but a single source with nothing compared against it
        instead carries one variable named ``value`` — no ``difference``, ``metrics``
        is ``None``, and ``labels`` is a 1-tuple of the one source's name. Such an
        item draws one solid line (see
        :func:`ocean_skill.plot.series.item_roles`/``line_specs``), with no
        statistics box and no residual (``residual=True`` is refused if any item in
        the figure is this shape).

        ``section`` joins ``field_facet`` as a single-source exception: one item,
        carrying ``field`` (one DataArray) rather than ``aligned``. Its ``field``
        has two axes rather than ``field_facet``'s horizontal pair — a vertical
        axis (native s-levels or fixed depths) and one along-path axis carrying
        1-D ``lon``/``lat`` coordinates (see
        :func:`ocean_skill.align.path_of`) — so it draws as one depth-by-distance
        panel rather than map panels. Built by
        :meth:`ocean_skill.field.Field.as_item`.

        ``section_row`` is that comparison counterpart: ``field_row``'s item shape
        (``aligned`` carrying the ``test``/``reference``/``difference`` trio, plus
        ``metrics``/``units``/``standard_name``/``depth``/``time``/``labels``) but
        each lane two-dimensional on depth and along-path distance rather than
        longitude and latitude, fixed-depth (``z``, negative-down) on every lane —
        two sources' native verticals share no axis to compare on, so a comparison
        section is never native s-levels the way a model-only ``section`` may be.
        Built by :meth:`ocean_skill.comparison.Comparison.as_item` for a
        comparison whose ``select`` cuts a transect (see
        :attr:`~ocean_skill.comparison.Comparison.is_section`); has no
        ``field_grid``-style stack of its own (``section_grid`` is a follow-up —
        see :class:`~ocean_skill.comparison.ComparisonSet`'s refusal on more than
        one ``section_row``).

        ``profile`` carries ``field_grid``'s list, one item per station/cast, each
        ``aligned`` 1-D on the vertical axis (``z``, ``depth``, or ``sigma0`` for an
        isopycnal profile) rather than on ``time`` -- the vertical twin of
        ``series``, drawn with depth on the y-axis (inverted, surface at top) and
        the compared value on x. A profile item's ``aligned`` never carries
        ``attrs["actual_depth"]``: depth is the axis every line already draws
        against, not a single number to style a line by. A cast's own instant, if
        it has one, rides as a scalar ``time`` coordinate instead (read by
        :func:`ocean_skill.plot.profile.panel_title` for identity and by
        :mod:`ocean_skill.plot.style` as the marker channel, in the same slot
        ``series`` gives its own ``depth``) -- several casts at one place are
        several items, not one item with a surviving time axis, exactly as
        ``series`` fans a surviving depth axis into one item per level. A
        surviving ``season`` axis (``aggregate={"time": {"groupby": "season",
        ...}}``) is fanned the same way, into one item per season in
        coordinate (chronological) order, by
        :func:`ocean_skill.plot.profile.fan_season` — the one call site both
        :meth:`ocean_skill.field.Field.as_item` and
        :meth:`ocean_skill.comparison.Comparison.as_item` share, so a fanned
        item's own ``season`` also rides as a scalar coordinate, read the same
        way ``time`` is. A ``spread`` coordinate (``aggregate={..., "spread":
        "std"}``) rides alongside the value through the same fan and draws as a
        shaded mean±spread band, in both renderers, when present. Built by
        :meth:`ocean_skill.field.Field.as_item` (a model column at one point) and
        :meth:`ocean_skill.comparison.Comparison.as_item` (``over=`` a vertical
        axis, at a station).

        ``locations`` carries no field payload at all: one item per catalog *source*,
        with ``kind`` (``"point"`` or ``"extent"``), a lon/lat midpoint, ``bboxes``
        for the extent kind, and pre-formatted hover strings — built by
        :func:`ocean_skill.plot.locations.build_items` from catalog metadata alone,
        so drawing the map never opens a dataset. Two further kinds carry a path
        instead of a position: ``"ring"`` (a dashed model-domain outline) and
        ``"line"`` (a solid selection slice — a lone-lon/lat select drawn as a
        meridian or parallel), each with ``paths``: a list of ``(M, 2)`` ``[lon,
        lat]`` arrays, every longitude pre-wrapped to −180..180 and already split
        at the antimeridian where the path crosses it, so a renderer only ever
        draws segments, never decides where to cut one. Built alongside the
        catalog-metadata items by
        :func:`ocean_skill.plot.map_locations.build_map_items`, which turns a
        :class:`~ocean_skill.comparison.Comparison` (or ``ComparisonSet``,
        ``Field``, ``FieldSet``) — really, its *request* — into the same item
        shape: the requested point/box/slice, plus the source's domain outline as
        context. Style is fixed rather than index-based for these two groups (see
        :data:`ocean_skill.plot.locations.GROUP_STYLES`), so a selection can never
        collide with a catalog featureType's colour, and every ``locations`` map
        agrees on what a model footprint looks like.
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
