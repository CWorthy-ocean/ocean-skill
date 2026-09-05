"""Line-style policy: which visual channel carries which fact about a line.

Time-series panels overlay several lines, and what a reader can decode depends entirely
on the encoding being *predictable*: add a model and it should take the next dash
pattern, add a variable and it should take the next colour, every figure in the report
the same way round. So the policy lives here as data — a channel map and three cycles —
rather than being decided inside a renderer.

Two rules, in this order:

* **Role wins.** ``reference`` is solid, ``test`` is dashed. That is the whole "obs
  solid, model dashed" convention, and because it keys off the *role* rather than the
  source it keeps working for model-versus-model (baseline solid, candidate dashed) and
  reverses when the roles do. A lone, uncompared line has no role to win against --
  it carries role ``"value"`` and draws solid, from the same cycle a second one would
  step through.
* **Then the channels.** ``colour <- variable``, ``linestyle <- source``,
  ``marker <- depth``, overridable per channel with ``encode=``. A channel is only
  *drawn* when its field actually varies within the figure, so a single-variable panel
  does not spend its markers explaining a depth that never changes.

Both renderers import from here. Neither may reimplement any of it: a legend reading
``sea_water_temperature`` statically and ``temperature`` interactively is the same plot
disagreeing with itself, which is exactly how the two drifted apart before.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex

__all__ = [
    "BAND_ALPHA",
    "BOKEH_DASHES",
    "CHANNELS",
    "COLOR_CYCLE",
    "LINESTYLES",
    "LineSpec",
    "StyledLine",
    "ambiguous_sources",
    "band_runs",
    "linestyle_for",
    "markevery_indices",
    "resolve",
    "series_label",
    "spread_of",
    "varying_fields",
]

#: tab20 reordered darks-then-lights: indices 0-9 are exactly tab10 (matplotlib's own
#: order), 10-19 are tab20's lighter companions. This is also what ``summary._group_styles``
#: assigns by level index — so a Taylor/Target diagram and a series panel of the same
#: comparisons come out the same colours, and don't repeat until the 21st series/level.
#: Derived from the colormap (matplotlib is a core dependency) rather than pinned, so the
#: two renderers can't drift from it independently.
COLOR_CYCLE = tuple(
    to_hex(colormaps["tab20"](i)) for i in (*range(0, 20, 2), *range(1, 20, 2))
)

#: matplotlib line styles. Index 0 is the reference's; the rest are the test cycle.
LINESTYLES = ("-", "--", ":", "-.")

#: matplotlib linestyle -> bokeh ``line_dash``, so the two renderers draw one policy.
BOKEH_DASHES = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dashdot"}

#: matplotlib markers, and bokeh's names for the same shapes in the same order — the
#: cycle ``summary._MARKERS`` uses, kept in step for the same reason as the colours.
MARKERS = ("o", "^", "s", "D", "v", "P", "X")
BOKEH_MARKERS = (
    "circle",
    "triangle",
    "square",
    "diamond",
    "inverted_triangle",
    "plus",
    "x",
)

#: The deterministic default: which field feeds which channel. Override one with
#: ``encode={"linestyle": "depth"}``, or switch it off with ``encode={"marker": None}``.
CHANNELS: dict[str, str | None] = {
    "color": "variable",
    "linestyle": "source",
    "marker": "depth",
}

#: Fields a channel may be keyed on, i.e. what a :class:`LineSpec` knows about itself.
#: ``time`` is a profile's own field -- a station's several casts, one line each --
#: and stays ``None`` on every series line, so ``CHANNELS``'s default marker channel
#: (``depth`` there, ``time`` here -- see :mod:`ocean_skill.plot.profile`) never
#: mistakes one family's lines for the other's. ``season``/``month`` are the same
#: idea one level up: a surviving ``{"groupby": "season"}`` axis, or an explicit
#: ``select={"month": [...]}`` narrowing a ``{"groupby": "month"}`` axis, fans
#: into one line per value (:func:`ocean_skill.plot.profile.fan_season`), and
#: this is what lets a profile's default colour switch to season/month when one
#: is present (decided at compose time, not in :data:`CHANNELS`, since a series
#: line has no use for either yet).
FIELDS = ("variable", "source", "depth", "time", "role", "season", "month")

#: About how many markers a line should carry, however many samples it has. A marker per
#: sample on a 3000-point mooring series is a filled band, not a line.
MARKER_TARGET = 20

#: Fill opacity for a mean±spread envelope, in both renderers -- one number so a
#: static and an interactive band read as the same shade.
BAND_ALPHA = 0.25


@dataclass(frozen=True)
class LineSpec:
    """One drawable line: where it came from, and what it is of.

    ``values`` is the 1-D DataArray itself; everything else is the facts a channel may
    be keyed on. Frozen because a resolved style is a function of these — two lines with
    the same facts must get the same style, whoever asks.

    ``season``/``month``/``spread`` are not independent facts to add: ``season``
    and ``month`` are what a surviving season axis, or an explicitly-narrowed
    month axis, fan into (see :data:`FIELDS`), and ``spread`` (a same-length
    half-width array, or ``None``) is the envelope :func:`spread_of` reads off
    the same aligned data ``values`` came from -- carried here so a renderer
    draws the band from the same object it draws the line from, never a second
    lookup that could disagree.
    """

    role: str
    source: str
    variable: str | None = None
    #: A realized numeric depth for a scalar/list request; a band's range label
    #: ("0-5 m") instead, which has no single realized depth to report (see
    #: ocean_skill.plot.series._depth_channel).
    depth: float | str | None = None
    time: str | None = None
    units: str | None = None
    values: Any = None
    item: int = 0
    season: str | None = None
    month: int | None = None
    spread: Any = None

    def get(self, field: str):
        """Return the value of one channel field, by name."""
        return getattr(self, field, None)


@dataclass(frozen=True)
class StyledLine:
    """A :class:`LineSpec` with its channels resolved, ready for either renderer."""

    spec: LineSpec
    color: str
    linestyle: str
    marker: str | None
    label: str

    @property
    def line_dash(self) -> str:
        """The bokeh spelling of :attr:`linestyle`."""
        return BOKEH_DASHES.get(self.linestyle, "solid")

    @property
    def bokeh_marker(self) -> str | None:
        """The bokeh spelling of :attr:`marker`."""
        if self.marker is None:
            return None
        return BOKEH_MARKERS[MARKERS.index(self.marker) % len(BOKEH_MARKERS)]


def linestyle_for(role: str, level_index: int = 0) -> str:
    """Return the line style for a role, and its index within that role's cycle.

    Solid for the reference whatever else is true of it; the test side steps through the
    remaining patterns. ``level_index`` is the line's position among the *test* lines'
    levels of whatever field feeds the linestyle channel, so two models get ``--`` and
    ``:`` while both references stay solid — and swapping which source is the reference
    swaps the styles, because the role decides and the name does not.

    A lone, uncompared line carries role ``"value"`` rather than either of those --
    calling it a "test" or "reference" would claim a comparison that was never made.
    It has no role to *win against*, so it draws solid too, from the same full cycle
    a second uncompared source would step through (``dash_levels`` in :func:`resolve`
    already excludes only ``"reference"``, so several ``"value"`` lines in one figure
    still dash apart from each other).
    """
    if role in ("reference", "value"):
        return LINESTYLES[level_index % len(LINESTYLES)]
    tail = LINESTYLES[1:]
    return tail[level_index % len(tail)]


def _levels(specs, field: str | None) -> list:
    """Ordered distinct values of ``field`` across ``specs`` (first appearance wins)."""
    if field is None:
        return []
    seen: list = []
    for spec in specs:
        value = spec.get(field)
        if value not in seen:
            seen.append(value)
    return seen


def varying_fields(specs) -> set[str]:
    """Return the fields that actually differ across ``specs``.

    What varies is what needs labelling — both in the legend and in the panel title, and
    from one set so the two cannot end up saying the same thing twice. Mirrors the rule
    ``compare()`` already applies to its own row labels.
    """
    return {field for field in FIELDS if len(_levels(specs, field)) > 1}


def ambiguous_sources(specs) -> set[str]:
    """Return the source names that appear on both sides of a comparison.

    That is the one case where a source name alone does not say which line is
    which -- :func:`series_label` appends the role for those. Its own function
    so a caller that relabels lines *after* :func:`resolve` (dropping a facet's
    field from the legend, or applying ``line_labels=``) can recompute the same
    labels the same way, rather than re-deriving this set inline and risking a
    different answer.
    """
    return {
        s.source
        for s in specs
        if any(o.source == s.source and o.role != s.role for o in specs)
    }


def series_label(spec: LineSpec, *, varying, ambiguous_sources=()) -> str:
    """Return one line's legend entry: the facts that distinguish it, and no others.

    Levels are spelled through :func:`ocean_skill.plot.summary.pretty_level` — imported,
    never re-implemented — so ``"temperature"`` and ``"50 m"`` read the same here as in
    a Taylor legend.

    The role is appended only when the same source appears on both sides of the figure,
    which is the one case where the source name alone does not say which line is which.
    """
    from ocean_skill.plot.summary import pretty_level

    parts = []
    for field in ("source", "variable", "depth", "time", "season", "month"):
        if field in varying or (field == "source" and not varying):
            value = spec.get(field)
            if value is not None:
                parts.append(pretty_level(field, value))
    if spec.source in ambiguous_sources:
        parts.append(f"({spec.role})")
    return " · ".join(parts) if parts else spec.source


def markevery_indices(n: int, target: int = MARKER_TARGET) -> list[int]:
    """Return about ``target`` evenly spaced sample indices out of ``n``.

    Shared so both renderers mark the *same* samples: matplotlib has ``markevery`` and
    bokeh does not (its markers are a separate overlay), so without one list of indices
    the two would show markers in different places on the same line.
    """
    if n <= 0:
        return []
    if n <= target:
        return list(range(n))
    step = max(1, n // target)
    return list(range(0, n, step))


def resolve(specs, *, encode: dict[str, str | None] | None = None) -> list[StyledLine]:
    """Resolve every line's colour, dash pattern, marker and label.

    One pass over the whole figure's lines, because every channel is relative: a
    colour is an *index* among the variables present, and a marker is drawn at all only
    if depth varies. Renderers consume the result and decide nothing.
    """
    specs = list(specs)
    channels = {**CHANNELS, **(encode or {})}
    unknown = set(channels) - set(CHANNELS)
    if unknown:
        raise ValueError(
            f"encode={sorted(unknown)} is not a channel; expected any of "
            f"{sorted(CHANNELS)}. Values name a field to key it on "
            f"({', '.join(FIELDS)}) or None to switch the channel off."
        )
    bad = {f for f in channels.values() if f is not None and f not in FIELDS}
    if bad:
        raise ValueError(
            f"encode cannot key a channel on {sorted(bad)}; a line knows only "
            f"{', '.join(FIELDS)}."
        )

    varying = varying_fields(specs)
    colours = _levels(specs, channels["color"])
    # The linestyle cycle is indexed among the *test* lines only: the reference is solid
    # by role, so letting it consume a level would leave the first model dotted while
    # nothing was dashed.
    dash_levels = _levels(
        [s for s in specs if s.role != "reference"], channels["linestyle"]
    )
    marker_field = channels["marker"]
    marker_levels = _levels(specs, marker_field) if marker_field in varying else []
    ambiguous = ambiguous_sources(specs)

    out = []
    for spec in specs:
        colour_value = spec.get(channels["color"]) if channels["color"] else None
        index = colours.index(colour_value) if colours else 0
        dash_value = spec.get(channels["linestyle"]) if channels["linestyle"] else None
        level = dash_levels.index(dash_value) if dash_value in dash_levels else 0
        marker = None
        if marker_levels:
            value = spec.get(marker_field)
            marker = MARKERS[marker_levels.index(value) % len(MARKERS)]
        out.append(
            StyledLine(
                spec=spec,
                color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
                linestyle=linestyle_for(spec.role, level),
                marker=marker,
                label=series_label(spec, varying=varying, ambiguous_sources=ambiguous),
            )
        )
    return out


def with_values(line: StyledLine, values) -> StyledLine:
    """Return ``line`` carrying different data — for a residual sharing its style."""
    return replace(line, spec=replace(line.spec, values=values))


def spread_of(aligned, name: str, da) -> np.ndarray | None:
    """Return ``da``'s own spread (half-width) array, or ``None`` if it has none.

    Two shapes, probed in order:

    1. A same-shape ``"spread"`` coordinate riding on ``da`` itself -- a
       single-source item's value, straight off ``operators.aggregate``'s
       ``spread`` option (see ``operators.SPREAD_COORD``).
    2. A ``f"{name}_spread"`` data variable on ``aligned`` -- a comparison's
       test/reference pair, where ``align._split_spread`` moves each lane's own
       spread there rather than leaving two same-named coordinates that would
       disagree once the two lanes share one Dataset.

    ``None`` either way means: no envelope for this line, degrading silently to
    today's plain line.
    """
    from ocean_skill.operators import SPREAD_COORD

    coord = getattr(da, "coords", {}).get(SPREAD_COORD) if da is not None else None
    if coord is not None:
        return np.asarray(coord.values, dtype="float64")
    variable = aligned.get(f"{name}_spread") if hasattr(aligned, "get") else None
    if variable is not None:
        return np.asarray(variable.values, dtype="float64")
    return None


def band_runs(axis_values, values, spread):
    """Return contiguous all-finite ``(axis, lo, hi)`` runs for a mean±spread band.

    Splitting on non-finite samples here — rather than each renderer handling NaN
    its own way — is what lets a static and an interactive envelope provably draw
    the same shape: both read the identical runs from this one function. A run of
    a single finite point is dropped; there is no band to fill between one point
    and itself.
    """
    axis = np.asarray(axis_values, dtype="float64")
    values = np.asarray(values, dtype="float64")
    spread = np.asarray(spread, dtype="float64")
    lo, hi = values - spread, values + spread
    finite = np.isfinite(axis) & np.isfinite(lo) & np.isfinite(hi)
    runs = []
    start = None
    for i, ok in enumerate(finite):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > 1:
                runs.append((axis[start:i], lo[start:i], hi[start:i]))
            start = None
    if start is not None and len(finite) - start > 1:
        runs.append((axis[start:], lo[start:], hi[start:]))
    return runs
