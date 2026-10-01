"""Composition for the ``XY`` family: items in, a fully resolved layout out.

The property-property plot (T-S, N-P, Alk-S, ...): one variable against another, a
panel per region, a cloud of dots or a line per data source. Like
:mod:`ocean_skill.plot.series` and :mod:`ocean_skill.plot.profile`, everything
decidable is decided here, so a renderer only walks :class:`XYLayout` and draws:
which member is which colour and marker, what each panel's limits and labels are,
the shared ``color_by`` scale, the legend entries (and which of them are dots rather
than lines), the annotation positions, and the sigma-0 contour grid and its levels.

Nothing in this module imports a plotting backend's drawing API. The one
matplotlib object it hands over is a :class:`~matplotlib.colors.Colormap`, which
both renderers can sample (the interactive one by turning it into a list of colours).

Composition rule, in order:

=======================  ===============================================================
panels                   one per distinct ``item["region"]``, in first-seen order
colour                   first points member black, the rest :data:`style.COLOR_CYCLE`
marker                   varies only when more than one member is drawn as points
``color_by="depth"``     points members coloured by depth, one scale, one colour bar
limits                   per panel unless ``sharex``/``sharey``; annotations widen them
density contours         only when one axis is salinity and the other temperature
=======================  ===============================================================
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ocean_skill import _stacklevel
from ocean_skill.plot import _titles
from ocean_skill.plot import series as _series_layout
from ocean_skill.plot import style as _style

__all__ = [
    "COLOR_BY",
    "POINT_CAP",
    "ColorScale",
    "DensityGrid",
    "LegendEntry",
    "StyledXY",
    "XYLayout",
    "XYPanel",
    "compose",
    "density_grid",
    "density_levels",
    "normalize_annotations",
]

#: What ``color_by=`` may name: the per-point arrays an item can carry.
COLOR_BY = ("depth", "time")

#: A panel holding more points than this still draws, but slowly -- and the
#: interactive renderer, which ships every point to the browser, may not manage at
#: all. The warning says what to do about it; nothing is thinned automatically,
#: since silently dropping model cells would change what the figure shows.
POINT_CAP = 2_000_000

#: Fraction of a panel's data span added on each side when its limits are automatic.
_PAD = 0.02

#: The density contour grid is ``_GRID`` x ``_GRID`` cells over the panel's limits.
_GRID = 120

#: Candidate contour spacings in kg m-3, coarsest first. 1, 0.5 and 0.25 are the ones
#: automatic levels normally land on; the rest only matter for a panel whose density
#: range is unusually wide or narrow, where they keep the line count sensible.
_DENSITY_STEPS = (5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05)

#: How many contour lines automatic levels aim for: the coarsest step giving at least
#: the first number, which by construction never exceeds the second.
_DENSITY_LINES = (6, 14)

#: How many samples of a panel's data decide where its legend goes. Ranking corners
#: needs a distance to every point; a 2-million-point cloud does not need all of them.
_LEGEND_SAMPLE = 4000


@dataclass(frozen=True, eq=False)
class StyledXY:
    """One drawable member in one panel, with its channels resolved.

    ``item`` is the index into the items :func:`compose` was given; ``x``/``y`` are
    the (finite, equal-length) arrays to draw; ``mark`` is ``"points"`` or ``"line"``.
    ``color`` is the member's solid colour -- the dots' colour, the line's colour, and
    the legend swatch in either case. ``marker`` is a matplotlib marker for a points
    member (``"o"`` unless several members are points, when it steps through
    :data:`ocean_skill.plot.style.MARKERS`) and ``None`` for a line; ``linestyle`` is
    for lines. ``color_values`` is ``None`` for a solid member, and for a ``color_by``
    member the per-point scalar the renderer maps through :class:`ColorScale` --
    metres for ``"depth"``, days since 1970-01-01 for ``"time"``. ``depth``/``time``
    pass the item's own arrays through (either may be ``None``) for a hover readout.
    ``eq=False``: the fields are arrays, which have no truth value to compare by.
    """

    item: int
    label: str
    region: str | None
    mark: str
    x: np.ndarray
    y: np.ndarray
    color: str
    marker: str | None
    linestyle: str
    color_values: np.ndarray | None
    depth: np.ndarray | None
    time: np.ndarray | None
    source: str

    @property
    def bokeh_marker(self) -> str | None:
        """The bokeh spelling of :attr:`marker`."""
        if self.marker is None:
            return None
        return _style.BOKEH_MARKERS[_style.MARKERS.index(self.marker)]


@dataclass(frozen=True)
class LegendEntry:
    """One legend row. ``mark`` says whether to draw it as a dot or a line.

    A dot at the ``marker_size`` of a data point is invisible in a legend, so a
    renderer draws a fixed-size swatch from these fields rather than reusing the
    plotted artist.
    """

    label: str
    color: str
    mark: str
    marker: str | None
    linestyle: str


@dataclass(frozen=True)
class DensityGrid:
    """A panel's sigma-0 field, ready to contour.

    ``x`` (``nx``) and ``y`` (``ny``) are the grid's axes in the panel's own units,
    ascending, and ``sigma`` is ``(ny, nx)`` -- ``sigma[j, i]`` is the potential density
    anomaly in kg m-3 at ``(x[i], y[j])``, ``nan`` where the equation of state is
    undefined. ``levels`` are the contour values (possibly empty).
    """

    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    levels: tuple[float, ...]


@dataclass(frozen=True)
class ColorScale:
    """The one ``color_by`` scale shared by every panel.

    ``field`` is ``"depth"`` or ``"time"``; ``label`` is the colour bar's text.
    ``vmin``/``vmax`` bound the scale in the units of :attr:`StyledXY.color_values`
    (metres, or days since 1970-01-01 for time -- :attr:`is_time` says which), and
    ``cmap`` is a :class:`~matplotlib.colors.Colormap`. ``show`` is ``False`` when
    the caller passed ``colorbar=False``: the points are still coloured, the figure
    just carries no bar. ``inverted`` is ``True`` when a vertical bar should read with
    its low end at the top (depth: the surface up).
    """

    field: str
    label: str
    vmin: float
    vmax: float
    cmap: Any
    show: bool = True
    inverted: bool = False

    @property
    def is_time(self) -> bool:
        """Whether the scale is in days since 1970-01-01 rather than metres."""
        return self.field == "time"

    @property
    def norm(self):
        """The shared :class:`~matplotlib.colors.Normalize`."""
        import matplotlib.colors as mcolors

        return mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)


@dataclass(frozen=True)
class XYPanel:
    """One panel: what to draw in it, and how its axes read.

    ``key`` is the region (``None`` when the plot has none) and ``title`` the text
    above it. ``items`` are drawn in order -- dots under lines is the renderer's
    business, not an ordering here. ``xlim``/``ylim`` are final, whether automatic,
    shared across panels or the caller's own. ``annotations`` is ``((text, x, y),
    ...)`` in data coordinates, centred on the point. ``density`` is the sigma-0 grid
    or ``None``. ``legend`` lists this panel's own members; ``legend_corner`` is where
    a per-panel key goes. ``blank`` marks a cell to hide rather than draw; composition
    never sets it today, since a region with nothing to draw is not a panel at all.
    """

    key: str | None
    title: str
    items: tuple[StyledXY, ...]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    annotations: tuple[tuple[str, float, float], ...] = ()
    density: DensityGrid | None = None
    legend: tuple[LegendEntry, ...] = ()
    legend_corner: str = "upper right"
    blank: bool = False


@dataclass(frozen=True)
class XYLayout:
    """Everything a renderer needs to draw an ``XY`` figure.

    ``panels`` are row-major in an ``nrows`` x ``ncols`` grid (trailing cells are
    left empty). ``xlabel``/``ylabel`` are figure-wide and drawn on the outer edges.
    ``legend_entries`` is every member in member order, for a single combined key;
    ``shared_legend`` says every panel carries the same members, and
    ``legend_placement`` is one of ``"auto"``, ``"off"``, ``"below"``, ``"right"``, or
    ``"corner"`` (a corner was forced and is baked into each panel's
    ``legend_corner``) -- the same vocabulary :mod:`ocean_skill.plot.series` uses.
    ``colorbar`` is the shared ``color_by`` scale, or ``None`` when nothing is coloured
    by one.
    """

    panels: tuple[XYPanel, ...]
    nrows: int
    ncols: int
    xlabel: str
    ylabel: str
    shared_legend: bool = True
    legend_placement: str = "auto"
    colorbar: ColorScale | None = None
    legend_entries: tuple[LegendEntry, ...] = ()


def _warn(message: str) -> None:
    """Warn about the caller's request, blaming the caller's own code."""
    warnings.warn(message, stacklevel=_stacklevel.find())


def _members(items) -> list[str]:
    """Distinct member labels in order of first appearance."""
    return list(dict.fromkeys(item["label"] for item in items))


# --- annotations ---------------------------------------------------------------------


def _pair(pos) -> tuple[float, float] | None:
    """``pos`` as a finite ``(x, y)``, or ``None`` if it is not one."""
    if isinstance(pos, str) or not isinstance(pos, Sequence | np.ndarray):
        return None
    try:
        x, y = (float(v) for v in pos)
    except (TypeError, ValueError):
        return None
    return (x, y) if np.isfinite(x) and np.isfinite(y) else None


def _annotation_entries(
    mapping: Mapping, where: str
) -> tuple[tuple[str, float, float], ...]:
    """``((text, x, y), ...)`` from one ``{text: (x, y)}`` mapping, validated."""
    out = []
    for text, pos in mapping.items():
        pair = _pair(pos)
        if pair is None:
            raise ValueError(
                f"annotations{where}: {text!r} must map to an (x, y) pair of numbers "
                f"in data coordinates, got {pos!r}."
            )
        out.append((str(text), *pair))
    return tuple(out)


def normalize_annotations(
    annotations, panel_keys: Sequence[str | None]
) -> dict[str | None, tuple[tuple[str, float, float], ...]]:
    r"""Resolve ``annotations=`` to ``{panel key: ((text, x, y), ...)}``.

    Two spellings, never mixed: ``{text: (x, y)}`` puts the same labels on every
    panel, and ``{region: {text: (x, y)}}`` puts each region's labels on its own panel
    (a panel with no entry gets none). The per-region form needs regions to key on, so
    it is refused for a plot whose only panel key is ``None``; a region key that is
    not a panel is almost always a typo, so it warns, listing the real names, and is
    otherwise ignored. Every ``(x, y)`` is a pair of finite numbers in data
    coordinates; the text is drawn centred on it and may contain ``\n``.

    Every key of ``panel_keys`` is present in the result, with ``()`` for no labels.
    """
    keys = list(panel_keys)
    out: dict[str | None, tuple] = {key: () for key in keys}
    if not annotations:
        return out
    if not isinstance(annotations, Mapping):
        raise ValueError(
            "annotations must be a dict, {text: (x, y)} for every panel or "
            f"{{region: {{text: (x, y)}}}} for one panel each, got {annotations!r}."
        )
    nested = [isinstance(v, Mapping) for v in annotations.values()]
    if any(nested) and not all(nested):
        regions = [k for k, n in zip(annotations, nested, strict=True) if n]
        raise ValueError(
            "annotations mixes the two forms: use either {text: (x, y)} for every "
            "panel or {region: {text: (x, y)}} for one panel each, not both. "
            f"Keys with a dict value: {regions}."
        )
    if not all(nested):
        entries = _annotation_entries(annotations, "")
        return {key: entries for key in keys}
    if keys == [None]:
        raise ValueError(
            "annotations is keyed by region, but this plot has no regions (one "
            "panel, key None). Pass {text: (x, y)} for the single panel, or add "
            "regions= to the plot."
        )
    unknown = [k for k in annotations if k not in keys]
    if unknown:
        _warn(
            f"annotations names {unknown}, which is not a panel -- ignored. "
            f"The panels are: {keys}."
        )
    for key in keys:
        if key in annotations:
            out[key] = _annotation_entries(annotations[key], f"[{key!r}]")
    return out


# --- density -------------------------------------------------------------------------


def _multiples(lo: float, hi: float, step: float) -> tuple[float, ...]:
    """Return the multiples of ``step`` inside ``[lo, hi]``."""
    first, last = int(np.ceil(lo / step - 1e-9)), int(np.floor(hi / step + 1e-9))
    return tuple(float(round(k * step, 10)) for k in range(first, last + 1))


def density_levels(spec, lo: float, hi: float) -> tuple[float, ...]:
    """Contour levels for a density range ``lo``..``hi`` (kg m-3).

    ``spec`` is ``True`` for automatic levels -- the coarsest spacing of 1, 0.5 or
    0.25 that gives at least 6 lines (never more than 14 for an ordinary range) -- an
    ``int`` for roughly that many lines, or a sequence of explicit levels, which is
    returned sorted and untouched. A range that is not finite has no levels.
    """
    if spec is not True and not isinstance(spec, numbers.Integral):
        return tuple(sorted(float(v) for v in spec))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ()
    options = [_multiples(lo, hi, step) for step in _DENSITY_STEPS]
    if spec is True:
        enough = [levels for levels in options if len(levels) >= _DENSITY_LINES[0]]
        return enough[0] if enough else options[-1]
    # an int: whichever spacing lands nearest the requested count, coarser on a tie
    return min(options, key=lambda levels: abs(len(levels) - int(spec)))


def density_grid(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    lon: float | None,
    lat: float | None,
    levels=True,
    swap: bool = False,
) -> DensityGrid:
    """Sigma-0 on a ``120 x 120`` grid over ``xlim`` x ``ylim``.

    Salinity is practical salinity and temperature is taken as *potential*
    temperature, whatever the data's own definition (the figure warns when members
    disagree, it does not convert). Practical salinity becomes absolute salinity at
    the panel centre ``lon``/``lat`` (``gsw.SA_from_SP`` at the surface), potential
    temperature becomes conservative temperature (``gsw.CT_from_pt``), and
    ``gsw.sigma0`` gives the density. Without a position the salinity anomaly
    correction is skipped (``gsw.SR_from_SP``), a difference of a few thousandths of a
    kg m-3. ``swap=True`` means x is temperature and y is salinity. ``levels`` is
    anything :func:`density_levels` accepts.

    ``gsw`` is imported here rather than at module level: it is scoped to the
    calculators that need it, not the package core (see ``environment.yml``).
    """
    try:
        import gsw
    except ImportError as exc:
        raise ImportError(
            "density= draws sigma-0 contours with gsw (TEOS-10), which is not "
            "installed -- install it (conda install -c conda-forge gsw) or pass "
            "density=False."
        ) from exc

    xs = np.linspace(min(xlim), max(xlim), _GRID)
    ys = np.linspace(min(ylim), max(ylim), _GRID)
    gx, gy = np.meshgrid(xs, ys)
    salt, temp = (gy, gx) if swap else (gx, gy)
    if lon is None or lat is None:
        sa = gsw.SR_from_SP(salt)
    else:
        sa = gsw.SA_from_SP(salt, 0.0, lon, lat)
    sigma = np.asarray(gsw.sigma0(sa, gsw.CT_from_pt(sa, temp)), dtype="float64")
    sigma = np.where(np.isfinite(sigma), sigma, np.nan)
    finite = sigma[np.isfinite(sigma)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (np.nan, np.nan)
    return DensityGrid(xs, ys, sigma, density_levels(levels, lo, hi))


def _check_density(density) -> None:
    """Refuse a ``density=`` that is not ``True``, a line count or a list of levels."""
    if isinstance(density, bool):
        return
    if isinstance(density, numbers.Integral):
        if density < 1:
            raise ValueError(f"density={density!r}: a line count must be at least 1.")
        return
    if not isinstance(density, str) and isinstance(density, Sequence | np.ndarray):
        try:
            levels = [float(v) for v in density]
        except (TypeError, ValueError):
            levels = []
        if levels and all(np.isfinite(levels)):
            return
    raise ValueError(
        f"density={density!r} is not True/False, a number of lines, or a non-empty "
        "list of sigma-0 levels."
    )


def _density_orientation(items, density) -> bool | None:
    """``None`` when density is off, else whether x is temperature (``swap``).

    Refuses any other pair of axes: sigma-0 is a function of salinity and temperature
    and of nothing else, so asking for it against nitrate and phosphate has no
    meaning, and drawing the lines anyway would be a wrong figure rather than a plain
    one.
    """
    if density is None or density is False:
        return None
    _check_density(density)
    roles = {(item.get("x_role"), item.get("y_role")) for item in items}
    if roles == {("salinity", "temperature")}:
        return False
    if roles == {("temperature", "salinity")}:
        return True
    first = items[0]
    raise ValueError(
        "density=... needs one axis to be salinity and the other temperature, but "
        f"x is {first['x_name']!r} (role {first.get('x_role')!r}) and y is "
        f"{first['y_name']!r} (role {first.get('y_role')!r})"
        + (" for at least one member" if len(roles) > 1 else "")
        + ". Pass density=False, or plot salinity against temperature."
    )


# --- colour --------------------------------------------------------------------------


def _explicit_colors(colors, labels: list[str]) -> dict[str, str]:
    """Return the colours ``colors=`` names, as ``{member: colour}`` (maybe partial).

    A string colours every member alike, a list is a palette assigned in member order
    and a dict names only the members it wants to pin. Wording follows
    :func:`ocean_skill.plot.summary._resolve_colors`; the levels here are members.
    """
    if colors is None:
        return {}
    if isinstance(colors, str):
        return dict.fromkeys(labels, colors)
    if isinstance(colors, Mapping):
        unknown = [k for k in colors if k not in labels]
        if unknown:
            available = ", ".join(repr(lab) for lab in labels)
            raise ValueError(
                f"colors={{...}} names {', '.join(repr(u) for u in unknown)}, which is "
                f"not a member of this plot -- available members: {available}"
            )
        return dict(colors)
    colors = list(colors)
    if len(colors) < len(labels):
        raise ValueError(
            f"colors has {len(colors)} entries but there are {len(labels)} members "
            f"({', '.join(labels)}) -- give at least one colour per member, or pass a "
            "dict to style only some of them."
        )
    return {label: colors[i] for i, label in enumerate(labels)}


def _member_styles(items, colors) -> dict[str, tuple[str, str | None]]:
    """``{member: (colour, marker)}``.

    The first member drawn as points is black, so a dense model cloud reads as the
    dark body of the plot and the (usually thinner) lines stand out against it; every
    other member takes the next colour of :data:`ocean_skill.plot.style.COLOR_CYCLE`,
    counting from the first, in member order. A colour a caller pins does not shift
    its neighbours' defaults. Markers only vary when more than one member is points,
    since a lone cloud gains nothing from a symbol and a line has none.
    """
    labels = _members(items)
    mark_of = {}
    for item in items:
        mark_of.setdefault(item["label"], item["mark"])
    points = [label for label in labels if mark_of[label] == "points"]
    explicit = _explicit_colors(colors, labels)
    cycle = _style.COLOR_CYCLE
    out = {}
    others = 0
    for label in labels:
        if points and label == points[0]:
            default = "black"
        else:
            default = cycle[others % len(cycle)]
            others += 1
        marker = None
        if label in points:
            multiple = len(points) > 1
            marker = (
                _style.MARKERS[points.index(label) % len(_style.MARKERS)]
                if multiple
                else "o"
            )
        out[label] = (explicit.get(label, default), marker)
    return out


def _time_days(time) -> np.ndarray:
    """``datetime64`` as float days since 1970-01-01 (``NaT`` -> ``nan``)."""
    stamps = np.asarray(time, dtype="datetime64[ns]")
    return (stamps - np.datetime64(0, "ns")) / np.timedelta64(1, "D")


def _default_cmap(color_by: str):
    """Depth uses the colormap the package already gives bathymetry; time is viridis."""
    import matplotlib

    if color_by == "depth":
        from ocean_skill.colormaps import cmaps_for

        return cmaps_for("sea_floor_depth")[0]
    return matplotlib.colormaps["viridis"]


def _check_color_by(color_by) -> None:
    if color_by is not None and color_by not in COLOR_BY:
        raise ValueError(
            f"color_by={color_by!r} is not something an XY plot can colour by; "
            f"expected one of {COLOR_BY} (or None for solid colours)."
        )


# --- limits, labels, legend ----------------------------------------------------------


def _axis_span(arrays, lim) -> tuple[float, float]:
    """``lim`` as given, else the data span with :data:`_PAD` on each side."""
    if lim is not None:
        lo, hi = lim
        return float(lo), float(hi)
    lo, hi = _series_layout.value_span(arrays)
    pad = _PAD * (hi - lo)
    return float(lo - pad), float(hi + pad)


def _check_lim(name: str, lim) -> tuple[float, float] | None:
    if lim is None:
        return None
    try:
        lo, hi = (float(v) for v in lim)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}={lim!r} is not a (low, high) pair of numbers."
        ) from None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{name}={lim!r} must be finite.")
    return lo, hi


def _axis_label(items, axis: str) -> str:
    """``"temperature [degC]"``: the short name, plus units when every member agrees.

    A dimensionless unit (``"1"``, as practical salinity is often tagged) is not worth
    a bracket and does not count as disagreeing with a member that has none.
    """
    from ocean_skill.plot.summary import pretty_level

    names = list(
        dict.fromkeys(pretty_level("variable", i[f"{axis}_name"]) for i in items)
    )
    units = list(
        dict.fromkeys(
            i[f"{axis}_units"]
            for i in items
            if i.get(f"{axis}_units") not in (None, "", "1")
        )
    )
    label = " / ".join(names)
    if len(units) > 1:
        _warn(
            f"members disagree on the units of the {axis} axis ({', '.join(units)}); "
            "the label gives none. Nothing is converted."
        )
        return label
    return f"{label} [{units[0]}]" if units else label


#: Standard names that say the same thing for this check: CF's generic
#: ``sea_water_salinity`` is what most gridded products (GLORYS, WOA) label their
#: practical salinity with, so treating the pair as different would warn on every T-S
#: diagram for a distinction the data do not make.
_SAME_QUANTITY = {"sea_water_salinity": "sea_water_practical_salinity"}

#: What an offset between two definitions of one axis looks like, by kind.
_OFFSET_HINT = {
    "temperature": "in situ vs potential temperature differ by ~0.1 degC at 1000 m, "
    "more deeper",
    "salinity": "absolute salinity runs ~0.16 g/kg above practical salinity",
}


def _warn_standard_names(items) -> None:
    """Warn, once per axis, when members do not mean the same quantity.

    In situ and potential temperature are the case that matters: both are "temperature"
    to a reader and a label, differ by ~0.1 degC at 1000 m and more below, and sit in
    the same column of a catalog. Nothing is converted -- which one is right for a
    comparison is the caller's call -- but a figure that quietly mixes them is a
    figure whose offset someone will spend an afternoon chasing. Names that differ
    only in spelling (:data:`_SAME_QUANTITY`) are not a disagreement.
    """
    for axis in ("x", "y"):
        by_name: dict[str, list[str]] = {}
        for item in items:
            name = item.get(f"{axis}_standard_name")
            if name:
                members = by_name.setdefault(_SAME_QUANTITY.get(name, name), [])
                if item["label"] not in members:
                    members.append(item["label"])
        if len(by_name) > 1:
            detail = " vs ".join(
                f"{name!r} ({', '.join(members)})" for name, members in by_name.items()
            )
            role = next(
                (i.get(f"{axis}_role") for i in items if i.get(f"{axis}_role")), None
            )
            hint = f" ({_OFFSET_HINT[role]})" if role in _OFFSET_HINT else ""
            _warn(
                f"members disagree on what the {axis} axis is: {detail}. They are "
                "drawn on one axis as given, with no conversion -- expect an offset "
                f"between them{hint}."
            )


def _legend_corner(panel_items, annotations, xlim, ylim, placement) -> str:
    """Return the corner a per-panel key goes: forced, else the emptiest.

    Ranked by :func:`ocean_skill.plot.series._rank_corners` -- the same measure the
    series and profile families use -- from a thinned sample of the panel's data plus
    its annotation positions, all scaled into the panel's own limits.
    """
    if placement in _series_layout.CORNERS:
        return placement
    xs = [np.asarray(it.x) for it in panel_items] + [
        np.array([x for _, x, _ in annotations])
    ]
    ys = [np.asarray(it.y) for it in panel_items] + [
        np.array([y for _, _, y in annotations])
    ]
    x, y = np.concatenate(xs), np.concatenate(ys)
    if x.size > _LEGEND_SAMPLE:
        keep = np.linspace(0, x.size - 1, _LEGEND_SAMPLE).astype(int)
        x, y = x[keep], y[keep]
    if not x.size:
        return _series_layout.CORNERS[1]
    sx = (x - xlim[0]) / (xlim[1] - xlim[0] or 1.0)
    sy = (y - ylim[0]) / (ylim[1] - ylim[0] or 1.0)
    return _series_layout._rank_corners(sx, sy)[0]


# --- compose -------------------------------------------------------------------------


def _panel_position(items, key) -> tuple[float | None, float | None]:
    """Return the first ``(lon, lat)`` an item of panel ``key`` carries, or Nones."""
    for item in items:
        if item["region"] == key and item.get("lon") is not None:
            return item["lon"], item.get("lat")
    return None, None


def _validate(items) -> list[dict]:
    """Return the items as dicts with float arrays, or raise naming what is wrong."""
    items = list(items)
    if not items:
        raise ValueError("an XY plot needs at least one item to draw")
    out = []
    for i, item in enumerate(items):
        for key in ("label", "x", "y", "mark"):
            if key not in item:
                raise ValueError(f"XY item {i} has no {key!r}")
        if item["mark"] not in ("points", "line"):
            raise ValueError(
                f"XY item {item['label']!r} has mark={item['mark']!r}; expected "
                "'points' or 'line'."
            )
        x = np.asarray(item["x"], dtype="float64").ravel()
        y = np.asarray(item["y"], dtype="float64").ravel()
        if x.size != y.size:
            raise ValueError(
                f"XY item {item['label']!r}: x has {x.size} values but y has {y.size}."
            )
        out.append({**item, "region": item.get("region"), "x": x, "y": y})
    return out


def _color_values(item, color_by) -> np.ndarray | None:
    """Return the per-point scalar ``color_by`` maps through the scale, or ``None``."""
    if color_by is None or item["mark"] != "points":
        return None
    raw = item.get(color_by)
    if raw is None:
        return None
    values = _time_days(raw) if color_by == "time" else np.asarray(raw, dtype="float64")
    values = values.ravel()
    if values.size != item["x"].size:
        raise ValueError(
            f"XY item {item['label']!r}: {color_by} has {values.size} values but x "
            f"has {item['x'].size}."
        )
    return values


def compose(
    items,
    *,
    annotations: Mapping | None = None,
    density: bool | int | Sequence[float] = False,
    color_by: str | None = None,
    cmap=None,
    colorbar: bool = True,
    colors=None,
    legend: bool | str = True,
    titles: Sequence[str | None] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    ncols: int | None = None,
    nrows: int | None = None,
) -> XYLayout:
    """Group ``items`` into panels and resolve every member's style and every axis.

    ``items`` are the dicts documented in :class:`~ocean_skill.plot.spec.PlotSpec`
    (``family="XY"``). One panel per distinct ``region``, in first-seen order, titled
    by the region (or by ``region_note`` when there are no regions) and overridable
    with ``titles=`` -- see :func:`ocean_skill.plot._titles.resolve_titles`. The grid
    defaults to a single row of up to three panels, then three columns;
    ``ncols=``/``nrows=`` wrap it otherwise
    (:func:`ocean_skill.plot.series.grid_shape`).

    Colours follow :func:`_member_styles`; ``colors=`` is a string, a list in member
    order, or a dict keyed by member label. ``color_by="depth"|"time"`` colours the
    points members from their arrays on one scale shared by every panel (default map:
    the package's bathymetry map for depth, viridis for time; ``cmap=`` overrides);
    lines stay solid, and a points member with no such array warns and stays solid
    too. ``colorbar=False`` keeps the colours and drops the bar.

    Limits are per panel, from the data and the panel's annotations plus a 2% margin;
    ``sharex``/``sharey`` use one union of all panels instead, and ``xlim``/``ylim``
    are taken exactly as given. ``annotations`` is anything
    :func:`normalize_annotations` accepts. ``density`` is ``True``, a line count, or a
    list of sigma-0 levels (:func:`density_levels`); it needs salinity on one axis and
    temperature on the other, and raises ``ValueError`` otherwise. ``legend`` is as for
    :mod:`ocean_skill.plot.series`: a bool, ``"below"``, ``"right"`` or a corner.

    Warns when a panel holds more than :data:`POINT_CAP` points, and when members
    disagree on an axis's standard name or units.
    """
    items = _validate(items)
    placement = _series_layout._normalize_legend(legend)
    _check_color_by(color_by)
    swap = _density_orientation(items, density)
    xlim, ylim = _check_lim("xlim", xlim), _check_lim("ylim", ylim)
    _warn_standard_names(items)
    xlabel, ylabel = _axis_label(items, "x"), _axis_label(items, "y")

    keys = list(dict.fromkeys(item["region"] for item in items))
    notes = {}
    for item in items:
        if item.get("region_note"):
            notes.setdefault(item["region"], item["region_note"])
    auto_titles = [str(k) if k is not None else notes.get(k, "") for k in keys]
    panel_titles = _titles.resolve_titles(auto_titles, titles)
    annotated = normalize_annotations(annotations, keys)

    # --- styles -----------------------------------------------------------------
    styles = _member_styles(items, colors)
    values = [_color_values(item, color_by) for item in items]
    missing: list[str] = []
    if color_by is not None:
        missing = list(
            dict.fromkeys(
                item["label"]
                for item, v in zip(items, values, strict=True)
                if item["mark"] == "points" and v is None
            )
        )
        if missing:
            _warn(
                f"color_by={color_by!r}: {', '.join(map(repr, missing))} carries no "
                f"{color_by} values, so it is drawn in a solid colour."
            )
    coloured = [v for v in values if v is not None]
    scale = None
    if color_by is not None and coloured:
        lo, hi = _series_layout.value_span(coloured)
        label = "depth [m]" if color_by == "depth" else "time"
        scale = ColorScale(
            field=color_by,
            label=label,
            vmin=float(lo),
            vmax=float(hi),
            cmap=_resolve_cmap(cmap, color_by),
            show=bool(colorbar),
            inverted=color_by == "depth",
        )
    elif color_by is not None and not missing:
        _warn(f"color_by={color_by!r} colours points members, and there are none.")

    styled = [
        StyledXY(
            item=i,
            label=item["label"],
            region=item["region"],
            mark=item["mark"],
            x=item["x"],
            y=item["y"],
            color=styles[item["label"]][0],
            marker=styles[item["label"]][1] if item["mark"] == "points" else None,
            linestyle="-",
            color_values=values[i],
            depth=None if item.get("depth") is None else np.asarray(item["depth"]),
            time=None if item.get("time") is None else np.asarray(item["time"]),
            source=item.get("source", ""),
        )
        for i, item in enumerate(items)
    ]
    by_panel = {k: [s for s in styled if s.region == k] for k in keys}

    for key, members in by_panel.items():
        n_points = sum(s.x.size for s in members if s.mark == "points")
        if n_points > POINT_CAP:
            name = panel_titles[keys.index(key)] or "the panel"
            _warn(
                f"{name!r} holds {n_points:,} points (more than {POINT_CAP:,}). It "
                "will draw, but slowly, and the interactive renderer may not manage "
                "it; narrow the region or thin the data (a coarser select, or "
                "aggregate=) rather than expecting the plot to."
            )

    # --- limits -----------------------------------------------------------------
    def spans(axis: str, lim):
        pos = 1 if axis == "x" else 2
        per_panel = {
            k: [getattr(s, axis) for s in members]
            + [np.array([a[pos] for a in annotated[k]])]
            for k, members in by_panel.items()
        }
        share = sharex if axis == "x" else sharey
        if share:
            union = [a for arrays in per_panel.values() for a in arrays]
            return dict.fromkeys(keys, _axis_span(union, lim))
        return {k: _axis_span(arrays, lim) for k, arrays in per_panel.items()}

    x_spans, y_spans = spans("x", xlim), spans("y", ylim)

    # --- legend -----------------------------------------------------------------
    def entries(members) -> tuple[LegendEntry, ...]:
        first = {}
        for s in members:
            first.setdefault(s.label, s)
        return tuple(
            LegendEntry(s.label, s.color, s.mark, s.marker, s.linestyle)
            for s in first.values()
        )

    panels = []
    for k, title in zip(keys, panel_titles, strict=True):
        members = by_panel[k]
        grid = None
        if swap is not None:
            lon, lat = _panel_position(items, k)
            grid = density_grid(
                x_spans[k], y_spans[k], lon=lon, lat=lat, levels=density, swap=swap
            )
        panels.append(
            XYPanel(
                key=k,
                title=title,
                items=tuple(members),
                xlim=x_spans[k],
                ylim=y_spans[k],
                annotations=annotated[k],
                density=grid,
                legend=entries(members),
                legend_corner=_legend_corner(
                    members, annotated[k], x_spans[k], y_spans[k], placement
                ),
            )
        )

    n = len(panels)
    if ncols is None and nrows is None and n > 3:
        ncols = 3
    n_rows, n_cols = _series_layout.grid_shape(
        n, as_columns=True, ncols=ncols, nrows=nrows
    )
    label_sets = {tuple(e.label for e in p.legend) for p in panels}
    return XYLayout(
        panels=tuple(panels),
        nrows=n_rows,
        ncols=n_cols,
        xlabel=xlabel,
        ylabel=ylabel,
        shared_legend=len(label_sets) == 1,
        # A forced corner is already in every panel; "corner" tells a renderer not to
        # also run "auto"'s own shared-labels check, which could combine the keys anyway
        # and override the very corner the caller forced (profile does the same).
        legend_placement=(
            "corner" if placement in _series_layout.CORNERS else placement
        ),
        colorbar=scale,
        legend_entries=entries(styled),
    )


def _resolve_cmap(cmap, color_by: str):
    """Return ``cmap=`` as a Colormap, defaulting per ``color_by``."""
    import matplotlib

    if cmap is None:
        return _default_cmap(color_by)
    if isinstance(cmap, str):
        return matplotlib.colormaps[cmap]
    return cmap
