"""Summary diagrams: one point per comparison, for a whole :class:`ComparisonSet`.

Where a field row shows *one* comparison in detail, these compress each comparison to a
single point so a suite can be read at a glance. The two are complementary and are best
read together:

**Taylor** (Taylor 2001) places correlation on the angle and standard deviation on the
radius, so distance from the reference point is the centred RMSD. It shows how well the
*pattern* matches — and is blind to bias: a model uniformly 5 degrees too warm plots
perfectly.

**Target** (Jolliff et al. 2009) plots bias against *signed* centred RMSD, both
normalized by the reference standard deviation, so distance from the origin is the
normalized total RMSD. It shows bias, which Taylor omits, and the sign of the x axis
distinguishes an under-dispersed model (left) from an over-dispersed one (right).
Inside the unit circle the model beats the observed mean as a predictor.
"""

from __future__ import annotations

import warnings
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ocean_skill.plot.typography import (
    MIN_PT,
    PAGE_W,
    diagram_scale_factor,
    type_scale,
)

__all__ = ["paired", "target", "taylor"]

#: Default figure sizes: both diagrams are square (a Taylor diagram's radius is a
#: standard deviation and a Target's guide rings must stay circular, so neither survives
#: a non-square canvas), and ``paired`` sets them side by side across the page.
TAYLOR_FIGSIZE = (PAGE_W * 0.62, PAGE_W * 0.62)
TARGET_FIGSIZE = (PAGE_W * 0.55, PAGE_W * 0.55)
PAIRED_FIGSIZE = (PAGE_W, PAGE_W * 0.5)


def _scale(
    figsize,
    *,
    ncols: int = 1,
    font_scale: float = 1.0,
    override: dict[str, float] | None = None,
) -> dict[str, float]:
    """Type scale for a diagram of this size — see :mod:`ocean_skill.plot.typography`.

    These diagrams carry seven text sizes of their own (title, axis labels, tick labels,
    legend, point annotations, contour labels, the ring labels), which were seven more
    constants tuned for one figure size. Same table as the field renderers use, so a
    Taylor diagram and the field row beside it in a report agree about how big a label
    is when both are drawn at the same size.

    ``override`` merges *onto* the computed scale rather than replacing it, so a caller
    can name one role and leave the rest alone. These functions have no ``*_kwargs``
    dicts, which makes this their only per-role override — it has to accept a partial
    dict to be usable at all.
    """
    scale = type_scale(figsize, ncols=ncols, nrows=1, font_scale=font_scale)
    return {**scale, **(override or {})}


def _diagram_figsize(default, *, size=None, zoom: float = 1.0):
    """Scale a diagram's default figure size onto the canvas ``size``/``zoom`` name.

    These figures have fixed proportions — Taylor and Target are square because a radius
    is a standard deviation and the rings must stay circular, and ``paired`` is the two
    side by side — so they *fit inside* a canvas rather than taking its shape. See
    :func:`~ocean_skill.plot.typography.diagram_scale_factor`; ``size="page"`` leaves
    the default untouched.
    """
    factor = diagram_scale_factor(default, size=size, zoom=zoom)
    return (default[0] * factor, default[1] * factor)


def _records(comparisons, groups: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Metric records plus a short label for each comparison.

    ``groups`` maps a comparison's ``reference`` (its catalog source name — what
    every metric record already carries) to a group label of the caller's choosing:
    a region, a cruise, whatever ``color_by``/``marker_by`` should split on but
    nothing in the record itself names. Applied here, at plot time, rather than
    stored on the comparison — grouping a set someone already built never means
    recomputing it, and the same set groups differently for different figures.
    Falls back to trying the *label* as the key, for a record with no ``reference``
    (a plain metrics table someone assembled by hand).
    """
    out = []
    for c in comparisons:
        rec = dict(c.metrics())
        rec["label"] = getattr(c, "label", None) or rec.get("variable", "")
        aligned = getattr(c, "aligned", None)
        rec["units"] = (
            aligned["reference"].attrs.get("units")
            if aligned is not None
            else getattr(c, "units", None)
        )
        if groups:
            rec["group"] = groups.get(
                rec.get("reference"), groups.get(rec["label"], "other")
            )
        out.append(rec)
    return out


def _uniform_reference_std(recs, *, rtol: float = 1e-3) -> float | None:
    """The reference standard deviation every record shares, or ``None`` if they differ.

    ``normalize=False`` axes are only as meaningful as the assumption that one
    reference std describes every point — the Target diagram's guide rings and the
    Taylor diagram's reference star/arc/contours all need a single number, and there
    is no principled way to pick one when the records disagree.
    """
    srefs = [r["std_reference"] for r in recs]
    if srefs and np.allclose(srefs, srefs[0], rtol=rtol):
        return float(srefs[0])
    return None


def _warn_mixed_variables(recs, diagram: str) -> None:
    """Warn when ``normalize=False`` records don't all describe the same variable.

    Absolute axes are in each comparison's own units; pooling several variables (or
    the same variable from sources whose units disagree) onto one diagram makes the
    axes describe more than one unit at once. This only warns — the data caveat
    belongs in a message, not scribbled on the figure.
    """
    variables = {r.get("variable") for r in recs if r.get("variable")}
    if len(variables) > 1:
        warnings.warn(
            f"{diagram}(normalize=False): comparisons span multiple variables "
            f"{sorted(variables)!r} — the axes are in each comparison's own units, "
            "which may not agree.",
            stacklevel=3,
        )


def _shared_units(recs) -> str | None:
    """The one units string every record agrees on, or ``None`` if missing/mixed —
    or if the records don't even describe the same variable.

    A pipeline that converts everything onto one canonical unit (this project's own
    ``units.convert_units`` defaults every concentration to ``mmol/m^3``) routinely
    leaves DIC and alkalinity, say, agreeing on ``units`` while still being two
    different quantities on two different natural scales. Labeling the axis with
    that shared unit would read as reassurance that the mixed-variable warning is
    actively contradicting, so units are only shown for a genuinely single-variable
    diagram.
    """
    variables = {r.get("variable") for r in recs if r.get("variable")}
    if len(variables) > 1:
        return None
    units = {r["units"] for r in recs if r.get("units")}
    return units.pop() if len(units) == 1 else None


def _target_xy(rec, normalize: bool) -> tuple[float, float]:
    """A Target diagram's ``(x, y)`` for one metric record.

    x is the centred RMSD **signed** by ``sign(std_test − std_reference)`` — negative
    means the model is under-dispersed — and y is the bias. ``normalize=True`` divides
    both by the record's own reference standard deviation (Jolliff et al. 2009's
    convention, so comparisons in different units share one diagram); ``False`` leaves
    them in the variable's native units.
    """
    denom = rec["std_reference"] if normalize else 1.0
    sign = np.sign(rec["std_test"] - rec["std_reference"])
    return (rec["crmsd"] / denom) * sign, rec["bias"] / denom


def _target_rings(circles, normalize: bool, recs) -> tuple[tuple[float, ...], float | None]:
    """Resolve Target's guide-ring radii and the "beats-the-mean" dashed boundary.

    ``circles=None`` asks for the default rings, always in the axes' own units:
    ``(0.5, 1.0)`` normalized, or the same fractions of the shared reference standard
    deviation when ``normalize=False`` and every record shares one. With mixed
    references there is no single boundary to draw, so the default becomes no rings
    at all (with a warning) rather than guessing whose reference to use.

    Explicit ``circles`` are always honored, in the axes' own units, even with mixed
    references — they are the caller's own radii, not a boundary this function has
    to justify. The returned boundary (``None`` when it isn't well defined) is only
    used to decide which ring, if any, gets the dashed "reference" styling.
    """
    if normalize:
        return (tuple(circles) if circles is not None else (0.5, 1.0)), 1.0
    boundary = _uniform_reference_std(recs)
    if circles is not None:
        return tuple(circles), boundary
    if boundary is not None:
        return (0.5 * boundary, boundary), boundary
    warnings.warn(
        "target(normalize=False): comparisons have different reference standard "
        "deviations, so no default guide rings are drawn (there is no single "
        "'beats the observed mean' boundary to show); pass circles= explicitly, "
        "in the axes' own units, to draw rings anyway.",
        stacklevel=3,
    )
    return (), None


def _target_labels(normalize: bool, recs, *, tex: bool) -> tuple[str, str]:
    """Target's x/y axis labels — one source so both renderers agree on the wording.

    ``tex`` picks matplotlib mathtext (``$\\sigma_{ref}$``) over the plain-unicode
    spelling (``σ_ref``) holoviews needs instead; the two used to drift independently.
    """
    if normalize:
        sigma = "$\\sigma_{ref}$" if tex else "σ_ref"
        return f"signed centred RMSD / {sigma}  (← under | over →)", f"bias / {sigma}"
    units = _shared_units(recs)
    suffix = f" [{units}]" if units else ""
    return f"signed centred RMSD{suffix}  (← under | over →)", f"bias{suffix}"


#: Label offsets (display points) cycled by index so consecutive labels point in
#: different directions. Matplotlib has no label-repel, and target points cluster near
#: the origin precisely when a model is good — when labels most need to stay legible.
_LABEL_OFFSETS = ((9, 6), (9, -12), (-9, 6), (-9, -12), (0, 12), (0, -15))


def _offset_labels(ax, xs, ys, labels, size, colors=None):
    """Annotate points, cycling offset directions to avoid overlapping labels."""
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels, strict=True)):
        dx, dy = _LABEL_OFFSETS[i % len(_LABEL_OFFSETS)]
        ax.annotate(
            lab,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=size,
            ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
            color=colors[i] if colors is not None else "0.15",
        )


#: Marker cycle used when ``marker_by`` splits the set into groups (models, runs, ...).
#: Shape carries one dimension and colour another, so a single diagram can show, say,
#: three models across six regions without becoming unreadable. No star here: the
#: reference point owns ``"*"`` on every summary diagram, so a group must never draw one.
_MARKERS = ("o", "^", "s", "D", "v", "P", "X")


def _field_levels(recs, field) -> list:
    """Distinct values of ``field`` across ``recs``, in order of first appearance.

    Module level (not just inside :func:`_group_styles`) because :func:`_grid_handles`
    needs the same level ordering to line points up with their legend cell.
    """
    seen = []
    for r in recs:
        v = r.get(field)
        if v not in seen:
            seen.append(v)
    return seen


def pretty_level(field, value) -> str:
    """Legend text for one level: short variable names, units on depths.

    Module level and public to the package because the interactive renderer builds its
    own legend and has to spell the levels identically — a legend reading
    ``sea_water_temperature`` in one renderer and ``temperature`` in the other is the
    same plot disagreeing with itself.
    """
    from ocean_skill.vars import short_name

    if field == "variable":
        return short_name(str(value))
    if field == "depth":
        return f"{value:g} m" if isinstance(value, int | float) else str(value)
    return str(value)


class _Styles(NamedTuple):
    colors: list
    markers: list
    alphas: list
    scales: list
    handles: list


def _scalar_scale(marker_scale):
    """Reduce ``marker_scale`` to the one value used where there's no level to key on.

    The reference star and a grey marker-block legend swatch (when colour is carrying
    a different field than the marker) belong to no single group. A dict has no "the"
    scale for them, so it falls back to the default (unscaled) rather than guessing
    which entry applies; a scalar passes through unchanged.
    """
    return 1.0 if isinstance(marker_scale, dict) else marker_scale


def _unknown_level_error(unknown, levels, field, param):
    available = ", ".join(f"{lev!r} ({pretty_level(field, lev)})" for lev in levels)
    bad = ", ".join(repr(u) for u in sorted(unknown, key=str))
    return (
        f"{param}={{...}} names {bad}, which is not a level of {field!r} — "
        f"available levels: {available}"
    )


def _resolve_per_level(value, levels, field, *, default, param):
    """Normalize a scalar-or-dict style argument to one value per level.

    A scalar broadcasts to every level. A dict is validated against the actual levels
    — an unknown key is almost always a typo or a stale grouping, and failing loudly
    here beats drawing a plot that silently ignores half of what was asked for. A level
    the dict doesn't mention falls back to ``default``, so a caller styling one group
    (the whole point of the layering use case) doesn't have to name every group.
    """
    if value is None:
        return dict.fromkeys(levels, default)
    if isinstance(value, dict):
        unknown = set(value) - set(levels)
        if unknown:
            raise ValueError(_unknown_level_error(unknown, levels, field, param))
        return {lev: value.get(lev, default) for lev in levels}
    return dict.fromkeys(levels, value)


def _resolve_colors(colors, levels, field):
    """Normalize ``colors`` (``None``/str/list/dict) to one colour per level.

    Unlike :func:`_resolve_per_level`, an unset level's default isn't one fixed value —
    it's the next colour in :data:`~ocean_skill.plot.style.COLOR_CYCLE`, so a caller who
    names only a couple of levels still gets the other groups auto-coloured rather than
    all sharing one placeholder.
    """
    from ocean_skill.plot.style import COLOR_CYCLE

    cycle = COLOR_CYCLE
    if colors is None:
        return {lev: cycle[i % len(cycle)] for i, lev in enumerate(levels)}
    if isinstance(colors, str):
        return dict.fromkeys(levels, colors)
    if isinstance(colors, dict):
        unknown = set(colors) - set(levels)
        if unknown:
            raise ValueError(_unknown_level_error(unknown, levels, field, "colors"))
        return {
            lev: colors.get(lev, cycle[i % len(cycle)]) for i, lev in enumerate(levels)
        }
    # A plain sequence: a palette assigned to levels in the order they first appear —
    # not one colour per point, so it composes with color_by/marker_by/groups instead
    # of being silently overridden by them.
    colors = list(colors)
    if len(colors) < len(levels):
        pretty = ", ".join(pretty_level(field, lev) for lev in levels)
        raise ValueError(
            f"colors has {len(colors)} entries but there are {len(levels)} "
            f"{field!r} levels ({pretty}) — give at least one colour per level, or "
            "pass a dict to style only some of them."
        )
    return {lev: colors[i % len(colors)] for i, lev in enumerate(levels)}


def _group_styles(
    recs, color_by=None, marker_by=None, colors=None, marker_scale=1.0, alpha=None
):
    """Assign a colour, marker, alpha and size to every record; plus legend handles.

    ``color_by``/``marker_by`` name any field present in the metric records
    (``variable``, ``depth``, ``test``, ``reference``, ...). ``colors``/``alpha``/
    ``marker_scale`` style by the *same* field colour already uses — ``color_by`` if
    given, else ``marker_by``, else each comparison's own label — so a value passed
    for one point-styling argument names the same groups as the others, whether that's
    an explicit ``dict`` keying a few of them or a scalar/list covering them all. This
    is what lets one call layer less-noticeable markers under more-noticeable ones,
    e.g. ``alpha={"salt": 0.15}, marker_scale={"temp": 1.5}``.
    """
    from matplotlib.lines import Line2D

    _pretty = pretty_level

    def _levels(field):
        return _field_levels(recs, field)

    if marker_by:
        mlevels = _levels(marker_by)
        marks = [
            _MARKERS[mlevels.index(r.get(marker_by)) % len(_MARKERS)] for r in recs
        ]
    else:
        mlevels, marks = [], ["o"] * len(recs)

    # colour/alpha/size all key on the same field: color_by if named, else marker_by
    # (so the legend's swatches match the points it groups), else each comparison's
    # own label (one "level" per point, the small-fan-out default).
    style_field = color_by or marker_by or "label"
    style_levels = _levels(style_field)
    level_colors = _resolve_colors(colors, style_levels, style_field)
    level_alphas = _resolve_per_level(
        alpha, style_levels, style_field, default=None, param="alpha"
    )
    level_scales = _resolve_per_level(
        marker_scale, style_levels, style_field, default=1.0, param="marker_scale"
    )
    cols = [level_colors[r.get(style_field)] for r in recs]
    alphas = [level_alphas[r.get(style_field)] for r in recs]
    scales = [level_scales[r.get(style_field)] for r in recs]

    handles = []
    if color_by:
        for lev in _levels(color_by):
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="",
                    marker="o",
                    mfc=level_colors[lev],
                    mec=level_colors[lev],
                    ms=7 * level_scales[lev],
                    label=_pretty(color_by, lev),
                )
            )
    if marker_by:
        for i, lev in enumerate(mlevels):
            m = _MARKERS[i % len(_MARKERS)]
            if color_by:
                # Colour is carrying color_by, a different field than this swatch's
                # marker_by level — grey rather than guessing which colour applies,
                # sized uniformly since level_scales is keyed by color_by here too.
                c, ms = "0.35", 7 * _scalar_scale(marker_scale)
            else:
                # marker_by *is* the style field: the swatch takes the resolved
                # colour/scale for its own level, so it matches the points exactly.
                c, ms = level_colors[lev], 7 * level_scales[lev]
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="",
                    marker=m,
                    mfc=c,
                    mec=c,
                    ms=ms,
                    label=_pretty(marker_by, lev),
                )
            )
    if not handles:
        # No grouping field named: one entry per comparison, each drawn exactly as its
        # own point. Without this a legend would only ever be possible when the caller
        # happened to pass color_by/marker_by, which is the common case for a small set.
        handles = [
            Line2D(
                [],
                [],
                ls="",
                marker=mk,
                mfc=c,
                mec=c,
                ms=7 * scl,
                label=r["label"],
            )
            for r, c, mk, scl in zip(recs, cols, marks, scales, strict=True)
        ]
    return _Styles(cols, marks, alphas, scales, handles)


def _grid_handles(
    recs, color_by, marker_by, colors=None, marker_scale=1.0, alpha=None, star_scale=1.0
):
    """Legend handles laid out as a colour x marker matrix, column-major.

    ``fig.legend`` fills its columns top-to-bottom (confirmed against matplotlib's own
    layout, not assumed), so passing exactly ``ncols * (len(rows) + 1)`` handles — a row
    label column, one column per ``marker_by`` level, a trailing reference column — draws
    an exact grid: rows are ``color_by`` levels in their colour, columns are ``marker_by``
    levels in their marker shape, and each cell is the same glyph its points are drawn
    with. A (colour, marker) combination with no data gets an invisible blank the same
    size as its row instead of a glyph, so the grid doubles as a coverage matrix. Every
    blank is sized to match the row (or, for the header row, a neutral size) so a column's
    entries line up with its neighbours' regardless of ``marker_scale``.

    Requires both ``color_by`` and ``marker_by`` — callers are expected to have already
    handled the case where one or both is missing (there is no cross-product to grid).
    """
    from matplotlib.lines import Line2D

    rows = _field_levels(recs, color_by)
    cols = _field_levels(recs, marker_by)
    level_colors = _resolve_colors(colors, rows, color_by)
    level_scales = _resolve_per_level(
        marker_scale, rows, color_by, default=1.0, param="marker_scale"
    )
    present = {(r.get(color_by), r.get(marker_by)) for r in recs}
    header_size = 7 * _scalar_scale(marker_scale)

    def _blank(ms, label=""):
        return Line2D(
            [], [], ls="", marker="o", mfc="none", mec="none", ms=ms, label=label
        )

    handles = [_blank(header_size)]
    for lev in rows:
        handles.append(_blank(7 * level_scales[lev], pretty_level(color_by, lev)))

    for j, mlev in enumerate(cols):
        handles.append(_blank(header_size, pretty_level(marker_by, mlev)))
        m = _MARKERS[j % len(_MARKERS)]
        for lev in rows:
            ms = 7 * level_scales[lev]
            if (lev, mlev) in present:
                c = level_colors[lev]
                handles.append(
                    Line2D([], [], ls="", marker=m, mfc=c, mec=c, ms=ms, label="")
                )
            else:
                handles.append(_blank(ms))

    handles.append(_blank(header_size))
    handles.append(_reference_handle(star_scale))
    for lev in rows[1:]:
        handles.append(_blank(7 * level_scales[lev]))

    return handles, len(cols) + 2


#: A ``summary_points`` reduction accepts these spellings; ``True`` means the first.
_SUMMARY_REDUCERS = {"median": np.median, "mean": np.mean}


def _summary_point_specs(recs, coord1, coord2, style_field, summary_points):
    """One ``(coord1, coord2, rec, marker)`` tuple per group in ``recs`` — the
    per-group centroid overlay convenience. ``coord1``/``coord2`` are the same
    per-record coordinates the base cloud plots (Taylor's normalized std/corr, or
    Target's signed x/y), reduced (median by default, or ``summary_points="mean"``)
    across each group named by ``style_field``. ``marker`` is always ``"h"`` (hexagon),
    so a centroid never reads as just another individual point -- and never as the
    reference, which owns the star.
    """
    key = "median" if summary_points is True else summary_points
    if key not in _SUMMARY_REDUCERS:
        raise ValueError(
            f"summary_points={summary_points!r} — expected True, 'median', or 'mean'"
        )
    reduce = _SUMMARY_REDUCERS[key]
    groups: dict[Any, list[int]] = {}
    for i, r in enumerate(recs):
        groups.setdefault(r.get(style_field), []).append(i)
    specs = []
    for level, idxs in groups.items():
        c1 = float(reduce([coord1[i] for i in idxs]))
        c2 = float(reduce([coord2[i] for i in idxs]))
        label = pretty_level(style_field, level) if style_field != "label" else str(level)
        specs.append((c1, c2, {style_field: level, "label": label}, "h"))
    return specs


def _overlay_point_specs(overlay, groups, coord1_of, coord2_of):
    """One ``(coord1, coord2, rec, marker)`` tuple per explicitly-overlaid comparison
    — the highlight-a-subset use: ``overlay=`` takes the same kind of thing the main
    argument does (comparisons, a ``ComparisonSet``, hand-built records), goes through
    the same :func:`_records`, and is drawn on top of the base cloud rather than
    replacing it. ``marker`` is always ``None`` here (draw with the group's own
    marker, not a centroid's forced ``"h"``) — the two spec sources are concatenated
    before styling, so they compose in one call.
    """
    return [(coord1_of(r), coord2_of(r), r, None) for r in _records(overlay, groups)]


def _resolve_overlay_style(
    overlay_recs,
    base_recs,
    base_styles,
    *,
    style_field,
    marker_by,
    overlay_marker_scale,
    overlay_alpha,
):
    """Colour and marker for overlay records, looked up against the base groups they
    belong to — never independently cycled.

    Calling :func:`_group_styles` again on just the overlay records would be wrong:
    its colour cycle position comes from the *order levels are first seen*, and an
    overlay is usually a small subset (one highlighted station, one centroid per
    group) whose own encounter order rarely matches the base cloud's — a lone
    highlighted point would get the cycle's first colour regardless of which group it
    actually belongs to. Looking its group up in the base's own already-resolved
    colours/markers is what keeps a highlighted or summarized point the same colour as
    the cloud it came from, however small the overlay is.

    Alpha and size have no such identity to preserve — they are the overlay's own
    emphasis, not a group's colour — so they resolve independently
    (:func:`_resolve_per_level`) against whatever levels the overlay itself contains,
    defaulting to more opaque and larger than the base layer.
    """
    color_lookup = dict(zip((r.get(style_field) for r in base_recs), base_styles.colors))
    marker_lookup = (
        dict(zip((r.get(marker_by) for r in base_recs), base_styles.markers))
        if marker_by
        else {}
    )
    levels = list(dict.fromkeys(r.get(style_field) for r in overlay_recs))
    alphas = _resolve_per_level(
        overlay_alpha, levels, style_field, default=1.0, param="overlay_alpha"
    )
    scales = _resolve_per_level(
        overlay_marker_scale, levels, style_field, default=1.8, param="overlay_marker_scale"
    )
    return _Styles(
        colors=[color_lookup.get(r.get(style_field), "0.2") for r in overlay_recs],
        markers=(
            [marker_lookup.get(r.get(marker_by), "o") for r in overlay_recs]
            if marker_by
            else ["o"] * len(overlay_recs)
        ),
        alphas=[alphas[r.get(style_field)] for r in overlay_recs],
        scales=[scales[r.get(style_field)] for r in overlay_recs],
        handles=[],  # an overlay never introduces its own legend entry (see callers)
    )


#: The identity of a comparison, for :func:`_arrow_chains`: two records belong to the
#: same run's trajectory when they agree on every one of these *except* the field
#: ``arrows`` names. Mirrors :data:`ocean_skill.comparison._LABEL_DIMS`'s field names
#: (not imported — ``comparison`` imports this module, so the reverse would be
#: circular), deliberately the comparison's own identity rather than the style
#: grouping: ``color_by="test"`` over two variables must not fuse both variables'
#: drift into one bogus chain, and a diagram with no grouping at all (one legend entry
#: per point) must still connect a run's time series.
_ARROW_ID_FIELDS = ("variable", "depth", "time", "test", "reference")

#: A chain's sort key falls back to input order for the *whole chain* the moment any
#: one value fails to parse — a half-sorted arrow trail (some segments chronological,
#: one reversed because its value didn't match) is worse than a trail left in
#: whatever order the caller built it in.
_SEASON_ORDER = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


def _arrow_sort_key(value):
    """Best-effort chronological key for one record's ``arrows`` field value.

    Handles the shapes :func:`ocean_skill.comparison._display_time` actually
    produces: a ``{"min", "max"}`` window (keyed by its start), a ``slice`` (keyed by
    ``.start``), a season name (calendar order), or anything else parseable as a
    timestamp. Raises on anything else, which is exactly what tells the caller to
    give up sorting this chain and keep input order instead.
    """
    if isinstance(value, dict) and "min" in value:
        value = value["min"]
    elif isinstance(value, slice):
        value = value.start
    if isinstance(value, str) and value in _SEASON_ORDER:
        return _SEASON_ORDER[value]
    import pandas as pd

    return pd.Timestamp(str(value))


def _resolve_arrows(arrows: bool | str | None) -> str | None:
    """Normalize ``arrows`` to the field name it names, or ``None`` when off.

    ``True`` is sugar for ``"time"`` — the canonical use is a ``compare(...,
    times=...)`` fan-out, so ``arrows=True`` reads as "connect the time steps"
    without spelling out the field name every call.
    """
    if not arrows:
        return None
    return "time" if arrows is True else arrows


def _hashable(value):
    """Turn an identity-tuple element into something hashable.

    A ``{"min", "max"}`` time window is the one shape a metric record carries that
    plain ``dict`` grouping chokes on.
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _arrow_chains(recs, field: str) -> list[list[int]]:
    """Group ``recs`` into per-run trajectories and order each chronologically.

    A chain is every record sharing all of :data:`_ARROW_ID_FIELDS` except ``field``
    itself (missing fields compare as ``None``, so hand-built records lacking
    ``test``/``reference`` still chain on whatever identity they do carry). Chains of
    fewer than two points carry no story to draw and are dropped; if *every* chain
    drops, that's the whole diagram having nothing to connect, which is worth a
    warning rather than a silent no-op.
    """
    if field not in {k for r in recs for k in r}:
        available = ", ".join(sorted({k for r in recs for k in r}))
        raise ValueError(
            f"arrows={field!r} names no field of the metric records — "
            f"fields present: {available}"
        )
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(recs):
        key = tuple(_hashable(r.get(f)) for f in _ARROW_ID_FIELDS if f != field)
        groups.setdefault(key, []).append(i)
    chains = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        try:
            idxs = sorted(idxs, key=lambda i: _arrow_sort_key(recs[i].get(field)))
        except (ValueError, TypeError):
            pass  # unsortable values: keep input order rather than guess
        chains.append(idxs)
    if not chains:
        warnings.warn(
            f"arrows={field!r}: no two comparisons share every other identity "
            f"({', '.join(f for f in _ARROW_ID_FIELDS if f != field)}) while "
            "differing only in this field — nothing to connect.",
            stacklevel=3,
        )
    return chains


def _draw_arrows(ax, chains, points, colors, *, zorder: float) -> None:
    """Draw one arrow per consecutive pair in each chain, on ``ax``.

    ``points`` is one ``(x, y)`` per record, in the *same* coordinate space
    ``ax`` already draws its base cloud in — Target's cartesian ``(x, y)`` or
    Taylor's polar ``(arccos(corr), std)`` on its aux axes; ``ax.annotate`` draws a
    straight display-space arrow either way, so one implementation serves both.
    Arrow colour is the *end* point's resolved colour (well-defined even when
    ``arrows`` and ``color_by`` name the same field, i.e. a colour-graded chain).
    """
    for idxs in chains:
        for i, j in pairwise(idxs):
            ax.annotate(
                "",
                xy=points[j],
                xytext=points[i],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colors[j],
                    lw=1.6,
                    alpha=0.85,
                    shrinkA=3,
                    shrinkB=5,
                ),
                zorder=zorder,
                annotation_clip=False,
            )


#: How a diagram identifies its points. ``"legend"`` puts a key below the axes,
#: ``"annotate"`` writes each label beside its marker, ``"grid"`` keys the
#: color_by x marker_by cross-product as a matrix, ``None`` does neither.
LABEL_MODES = ("legend", "annotate", "grid")


def _resolve_labels(labels):
    """Normalize the ``labels`` argument, rejecting typos loudly."""
    if labels is None or labels is False or labels == "none":
        return None
    if labels not in LABEL_MODES:
        raise ValueError(
            f"labels={labels!r} is not one of {LABEL_MODES} or None — "
            "'legend' keys the points below the axes, 'annotate' writes each "
            "label beside its marker, 'grid' keys the color_by x marker_by "
            "cross-product as a matrix."
        )
    return labels


def _fallback_grid_without_both_channels(labels, color_by, marker_by):
    """``"grid"`` needs both channels to have a cross-product to tabulate.

    Called after any ``groups``-defaulting of ``color_by``/``marker_by`` has already
    happened, so ``groups=`` alone is enough to earn a real grid. With only one channel
    (or neither), a matrix would have one row or one column — exactly the flat legend
    — so this warns and falls back to it rather than drawing a degenerate grid or
    raising, which would be unfriendly to a caller whose grouping fields vary.
    """
    if labels == "grid" and not (color_by and marker_by):
        warnings.warn(
            'labels="grid" keys the color_by x marker_by cross-product, which needs '
            "both to be set; drawing the flat legend instead.",
            stacklevel=3,
        )
        return "legend"
    return labels


def _reference_handle(marker_scale=1.0):
    """Legend entry for the reference point, drawn as a black star on both diagrams."""
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        ls="",
        marker="*",
        mfc="k",
        mec="k",
        ms=9 * marker_scale,
        label="reference",
    )


#: Most columns a key beneath the axes is laid out in before it starts wrapping.
_LEGEND_MAX_COLS = 5


def _fit_text(fig) -> None:
    """Shrink any label still too long for its own axes — the field renderer's pass.

    Imported here rather than duplicated so the two renderers cannot disagree about what
    counts as overflowing. Deferred because ``matplotlib_renderer`` reaches back into
    module from ``render()``; a module-level import either way round would be circular.
    """
    from ocean_skill.plot.matplotlib_renderer import _fit_text_widths

    _fit_text_widths(fig)


#: Air (display points) left between the key and the lowest thing above it.
_LEGEND_PAD = 6.0


def _lowest_artist_bottom(fig, renderer) -> float:
    """Display-y of the bottom of the lowest axis label / tick label in the figure.

    What the key has to clear is not the axes box — it is whatever hangs below it, which
    is the x tick labels and then the x axis label, both of whose heights are set by the
    type scale and so move when the level does.
    """
    bottoms = []
    for ax in fig.axes:
        bottoms.append(ax.get_window_extent(renderer).y0)
        texts = [ax.xaxis.label, *ax.get_xticklabels()]
        bottoms += [
            t.get_window_extent(renderer).y0
            for t in texts
            if t.get_visible() and t.get_text()
        ]
    return min(bottoms) if bottoms else 0.0


def _legend_anchor_top(fig) -> float:
    """Figure-fraction y of the top of a key placed just below everything drawn so far.

    Shared by :func:`_legend_below` and :func:`_grid_legend_below` so a flat key and a
    grid key clear the axes labels identically — see :func:`_legend_below` for why this
    is measured rather than a fixed offset.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    pad_px = _LEGEND_PAD * fig.dpi / 72.0
    return (_lowest_artist_bottom(fig, renderer) - pad_px) / fig_h_px


def _legend_below(fig, handles, size):
    """Draw one key beneath the axes, clear of them, in as many columns as fit.

    Below rather than inside because these diagrams put data in every corner — Taylor
    fills the upper right at high correlation, and target points scatter around the
    origin — so any in-axes placement collides with the data for some input.

    **Vertical placement is measured, not assumed.** This used to sit at a fixed
    ``y=-0.04`` in figure fractions while the x-axis labels it has to clear are a fixed
    *height*, so raising the type level walked the key straight up into them — the same
    bug, and the same fix, as the row label that used to sit at a constant ``x=-0.18``
    (see ``matplotlib_renderer._clear_row_labels``). Placing it below the lowest drawn
    label instead holds at any level.

    **Width is bought back with columns before font size.** Five columns is a lot, and
    the entries are whatever the caller's comparisons happen to be called, which no type
    scale can know: ten labels like ``ROMS-GOM-hindcast-run00-nitrate`` come to half
    again the figure's width. Dropping the column count keeps every entry readable;
    only if a single column is still too wide does the text shrink, floored at
    ``MIN_PT``.
    """
    top = _legend_anchor_top(fig)

    def draw(ncol):
        return fig.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc="upper center",
            ncol=ncol,
            fontsize=size,
            frameon=False,
            numpoints=1,
            bbox_to_anchor=(0.5, top),
        )

    limit = fig.get_size_inches()[0] * fig.dpi
    legend, width = None, 0.0
    for ncol in range(min(len(handles), _LEGEND_MAX_COLS), 0, -1):
        if legend is not None:
            legend.remove()
        legend = draw(ncol)
        fig.canvas.draw()
        width = legend.get_window_extent(fig.canvas.get_renderer()).width
        if width <= limit or ncol == 1:
            break
    if legend is not None and width > limit > 0:
        shrunk = max(size * limit / width, MIN_PT)
        for text in legend.get_texts():
            text.set_fontsize(shrunk)
    return legend


def _grid_legend_below(fig, handles, ncols, size):
    """Draw the :func:`_grid_handles` matrix beneath the axes, clear of them.

    Unlike :func:`_legend_below`, ``ncols`` is not negotiable — it is the number of
    matrix columns, and matplotlib's column-major fill is what turns the handle list
    back into a grid, so dropping a column would scramble it rather than narrow it.
    Width overflow can therefore only be bought back with text size, exactly as
    :func:`_legend_below` does once it is down to one column.

    ``handlelength``/``handletextpad``/``columnspacing`` are tightened from
    matplotlib's defaults, which are sized for a legend with few, long entries — this
    one has many short (often empty) ones, and the header/row-label text otherwise
    sits noticeably further from its glyph column than the grid reads as intending.
    """
    top = _legend_anchor_top(fig)
    legend = fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="upper center",
        ncol=ncols,
        fontsize=size,
        frameon=False,
        numpoints=1,
        columnspacing=1.2,
        handlelength=1.4,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, top),
    )
    fig.canvas.draw()
    limit = fig.get_size_inches()[0] * fig.dpi
    width = legend.get_window_extent(fig.canvas.get_renderer()).width
    if width > limit > 0:
        shrunk = max(size * limit / width, MIN_PT)
        for text in legend.get_texts():
            text.set_fontsize(shrunk)
    return legend


def taylor(
    comparisons,
    *,
    title: str | None = None,
    normalize: bool = True,
    save: str | Path | None = None,
    colors=None,
    color_by: str | None = None,
    marker_by: str | None = None,
    groups: dict[str, Any] | None = None,
    fig=None,
    rect: int = 111,
    labels: str | None = "legend",
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    scale: dict[str, float] | None = None,
    marker_scale: float = 1.0,
    alpha: float | None = None,
    overlay=None,
    overlay_marker_scale: float | dict = 1.8,
    overlay_alpha: float | dict = 1.0,
    summary_points: bool | str = False,
    arrows: bool | str | None = None,
):
    """Taylor diagram with one point per comparison.

    ``normalize=True`` divides each standard deviation by its own reference, so
    comparisons in different units (or of different variables) share one diagram; the
    reference then sits at radius 1. Turn it off only when every comparison shares a
    reference and you want the native units — the radial axis is then labelled with
    them, but only for a genuinely single-variable diagram: comparisons that share a
    unit string while describing different variables (DIC and alkalinity both land on
    this project's own canonical ``"mmol/m^3"``, say, while remaining different
    quantities on different natural scales) still get an unlabelled axis, since naming
    one unit would read as reassurance the warning below is actively contradicting.
    With ``normalize=False``, a reference
    standard deviation that differs across comparisons (or comparisons that span more
    than one variable) draws the same diagram but warns, since the star, dashed arc,
    and RMS contours describe only the first comparison's reference.

    ``color_by``/``marker_by`` name a field of the metric records (``variable``,
    ``depth``, ``test``, ...) so several groups can share one diagram — colour for one
    dimension, marker shape for another (three models across six regions, say).
    ``groups`` names a grouping that is not already in the record — a
    ``{reference_name: label}`` mapping (see :func:`_records`) — for splitting on
    something like region without having to compute it into every comparison's
    metrics first; it defaults ``color_by`` to ``"group"`` when neither ``color_by``
    nor ``marker_by`` is otherwise given.

    ``labels`` chooses how points are identified: ``"legend"`` (a key below the axes),
    ``"annotate"`` (each label written beside its marker), or ``"grid"`` (a matrix key:
    one row per ``color_by`` level in that row's colour, one column per ``marker_by``
    level in that column's marker, each cell the exact glyph its points are drawn
    with — reading off a point's row and column tells you both groups it belongs to at
    once, which the flat ``"legend"`` leaves the reader to cross-reference); ``None``
    for neither. Annotation is the better choice for a handful of points, a legend once
    there are enough that the labels would collide, and grid once both ``color_by`` and
    ``marker_by`` are set and the reader would otherwise have to mentally combine two
    separate swatch blocks. ``"grid"`` needs both grouping fields — given only one (or
    neither) it warns and falls back to ``"legend"``, since a one-channel matrix is just
    the flat legend.

    Text sizes follow the figure size rather than being fixed, so a diagram drawn at
    twice the default is not a diagram with half-size labels; ``font_scale`` multiplies
    them all. ``scale`` takes a ready-made :func:`_scale` result, which is how
    :func:`paired` gives both of its panels the sizes of the figure they *share* rather
    than the sizes each would pick alone.

    ``marker_scale`` multiplies every marker — sample points, the reference star, and
    the legend swatches — together, keeping their proportions; it is the marker
    analogue of ``font_scale``. ``alpha`` fades the sample points (fill and edge) for
    overlapping sets, leaving the reference star and any labels opaque.

    ``colors``/``alpha``/``marker_scale`` all accept a ``{level: value}`` dict keyed by
    whichever field is grouping colour — ``color_by`` if given, else ``marker_by``, else
    each comparison's own label — so one call can style particular groups without
    touching the rest: ``taylor(..., color_by="variable", alpha={"salt": 0.15},
    marker_scale={"temp": 1.5})`` fades the salinity points and enlarges the
    temperature ones, everything else left at its default. A level the dict doesn't
    name keeps its usual colour/opacity/size; a level that isn't one of the field's
    values raises rather than being silently ignored.

    ``overlay=`` draws a second, emphasized layer on top of the base cloud — the same
    kind of thing ``comparisons`` accepts (a list, a ``ComparisonSet``, hand-built
    records), styled to match the group it belongs to (never independently
    re-cycled), by default bigger, opaque, and black-edged so it reads as "look here"
    against the fainter cloud beneath: fade the base with ``alpha=`` to make the
    contrast starker. This is the general mechanism for two related things —
    highlighting specific points (pass the subset you want to point out) and
    ``summary_points=True`` (or ``"median"``/``"mean"``), which instead builds one
    hexagon-marked centroid per group internally, the reduced (median by default) position
    of that group's own cloud. Both can be given at once. Neither introduces a new
    legend entry — an overlay point's group already has one from the base cloud.
    ``overlay_marker_scale``/``overlay_alpha`` size and fade the overlay layer
    specifically (defaults 1.8x and fully opaque), independent of the base layer's own
    ``marker_scale``/``alpha`` — and accept the same ``{level: value}`` dict form.

    ``arrows`` means exactly what it does in :func:`target` — connecting comparisons
    that agree on everything but the named field (``True``/``"time"`` for a
    ``compare(..., times=...)`` fan-out) with an arrow per consecutive pair, each
    chain's first sample drawn hollow. Static only: the interactive renderer
    delegates Taylor diagrams to matplotlib entirely, so this rides along with it.
    """
    import matplotlib.pyplot as plt

    from ocean_skill.plot._taylor import TaylorDiagram

    labels = _resolve_labels(labels)
    recs = _records(comparisons, groups)
    if not recs:
        raise ValueError("no comparisons to plot")
    if groups and not color_by and not marker_by:
        color_by = "group"
    labels = _fallback_grid_without_both_channels(labels, color_by, marker_by)
    style_field = color_by or marker_by or "label"
    arrows_field = _resolve_arrows(arrows)
    chains = _arrow_chains(recs, arrows_field) if arrows_field else []
    chain_starts = {idxs[0] for idxs in chains}

    if not normalize:
        _warn_mixed_variables(recs, "taylor")
        if _uniform_reference_std(recs) is None:
            warnings.warn(
                "taylor(normalize=False): comparisons have different reference "
                "standard deviations; the reference star, dashed arc, and RMS "
                "contours all use the first comparison's.",
                stacklevel=2,
            )

    refstd = 1.0 if normalize else recs[0]["std_reference"]
    figsize = figsize or _diagram_figsize(TAYLOR_FIGSIZE, size=size, zoom=zoom)
    scale = _scale(figsize, font_scale=font_scale, override=scale)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    stds = [
        (r["std_test"] / r["std_reference"]) if normalize else r["std_test"]
        for r in recs
    ]
    dia = TaylorDiagram(
        refstd,
        fig=fig,
        rect=rect,
        label="reference",
        # `srange` is multiples of `refstd` (see TaylorDiagram.__init__), but `stds`
        # is in raw units when `normalize=False` — dividing back by `refstd` keeps
        # this a no-op when normalized (refstd == 1) and correct otherwise.
        srange=(0, max(1.6, 1.15 * max(stds) / refstd)),
    )
    star_scale = _scalar_scale(marker_scale)
    if star_scale != 1.0:
        # The vendored diagram draws its own reference star at a fixed size; scaling it
        # afterwards (upstream's own idiom — see _taylor.py's use of set_color) keeps it
        # proportionate to points drawn at marker_scale without touching that file. A
        # dict marker_scale has no single "the" scale for the reference, so it stays
        # unscaled (_scalar_scale's fallback) rather than guessing which entry applies.
        dia.samplePoints[0].set_markersize(10 * star_scale)
    styles = _group_styles(recs, color_by, marker_by, colors, marker_scale, alpha)

    per_rec = zip(
        recs,
        stds,
        styles.colors,
        styles.markers,
        styles.alphas,
        styles.scales,
        strict=True,
    )
    for i, (rec, sd, col, mk, al, scl) in enumerate(per_rec):
        hollow = i in chain_starts
        dia.add_sample(
            sd,
            rec["corr"],
            marker=mk,
            ms=9 * scl,
            ls="",
            mfc="none" if hollow else col,
            mec=col,
            mew=1.2 if hollow else 1.0,
            alpha=al,
            label=rec["label"],
        )
    if chains:
        _draw_arrows(
            dia.ax,
            chains,
            list(zip((np.arccos(r["corr"]) for r in recs), stds, strict=True)),
            styles.colors,
            zorder=1.9,
        )

    overlay_specs = []
    if overlay is not None:
        overlay_specs += _overlay_point_specs(
            overlay,
            groups,
            lambda r: (r["std_test"] / r["std_reference"]) if normalize else r["std_test"],
            lambda r: r["corr"],
        )
    if summary_points:
        overlay_specs += _summary_point_specs(
            recs, stds, [r["corr"] for r in recs], style_field, summary_points
        )
    if overlay_specs:
        overlay_recs = [spec[2] for spec in overlay_specs]
        overlay_styles = _resolve_overlay_style(
            overlay_recs,
            recs,
            styles,
            style_field=style_field,
            marker_by=marker_by,
            overlay_marker_scale=overlay_marker_scale,
            overlay_alpha=overlay_alpha,
        )
        for (sd, cr, rec, mk_override), col, mk, al, scl in zip(
            overlay_specs,
            overlay_styles.colors,
            overlay_styles.markers,
            overlay_styles.alphas,
            overlay_styles.scales,
            strict=True,
        ):
            dia.add_sample(
                sd,
                cr,
                marker=mk_override or mk,
                ms=9 * scl,
                ls="",
                mfc=col,
                mec="k",
                alpha=al,
                zorder=10,
                label=rec.get("label"),
            )

    contours = dia.add_contours(levels=5, colors="0.6", linewidths=0.7)
    plt.clabel(contours, inline=1, fontsize=scale["contour_label"], fmt="%.2f")
    dia.add_grid(color="0.85", linewidth=0.5)
    dia._ax.axis[:].major_ticks.set_tick_out(True)
    for axis in dia._ax.axis.values():
        # the floating polar axes' own correlation/stddev labelling, which mpl_toolkits
        # leaves at the rcParams default — the one part of this diagram that would still
        # be sized independently of the rest of it
        axis.major_ticklabels.set_fontsize(scale["tick_label"])
        axis.label.set_fontsize(scale["axes_label"])
    if not normalize:
        units = _shared_units(recs)
        if units:
            dia._ax.axis["left"].label.set_text(f"Standard deviation [{units}]")

    if labels == "legend":
        _legend_below(
            fig, [*styles.handles, _reference_handle(star_scale)], scale["legend"]
        )
    elif labels == "grid":
        handles, ncols = _grid_handles(
            recs, color_by, marker_by, colors, marker_scale, alpha, star_scale
        )
        _grid_legend_below(fig, handles, ncols, scale["legend"])
    elif labels == "annotate":
        # The aux axes are polar: a sample sits at (arccos(corr), stddev), which is
        # exactly where add_sample put it.
        _offset_labels(
            dia.ax,
            [np.arccos(r["corr"]) for r in recs],
            stds,
            [r["label"] for r in recs],
            scale["annotation"],
            colors=styles.colors,
        )
    if title:
        dia._ax.set_title(title, fontsize=scale["title"], pad=18)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def target(
    comparisons,
    *,
    title: str | None = None,
    normalize: bool = True,
    save: str | Path | None = None,
    colors=None,
    color_by: str | None = None,
    marker_by: str | None = None,
    groups: dict[str, Any] | None = None,
    circles=None,
    ax=None,
    labels: str | None = "annotate",
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    scale: dict[str, float] | None = None,
    marker_scale: float = 1.0,
    alpha: float | None = None,
    overlay=None,
    overlay_marker_scale: float | dict = 1.8,
    overlay_alpha: float | dict = 1.0,
    summary_points: bool | str = False,
    arrows: bool | str | None = None,
):
    """Target diagram (Jolliff et al. 2009) with one point per comparison.

    x is the centred RMSD **signed** by ``sign(std_test − std_reference)`` — negative
    means the model is under-dispersed — and y is the bias. ``normalize=True`` (the
    default) divides both by the reference standard deviation, so distance from the
    origin is the normalized total RMSD and points inside the unit circle out-perform
    the observed mean as a predictor; comparisons in different units (or of different
    variables) then share one diagram. ``normalize=False`` leaves both in native units
    — the axis labels then name them for a single-variable diagram, but stay unlabelled
    once the comparisons span more than one variable (which also warns), even if those
    variables happen to share a unit string, since the two natural scales still differ.

    ``circles`` sets the guide-ring radii, always in the axes' own units — left at its
    default (``None``) that is ``(0.5, 1.0)`` normalized, or ``(0.5, 1.0)`` scaled by
    the shared reference standard deviation when ``normalize=False`` and every
    comparison shares one; with mixed references there is no single boundary to draw,
    so the default becomes no rings at all (with a warning). Pass ``circles``
    explicitly to draw rings regardless.

    ``color_by``/``marker_by``/``groups`` mean exactly what they do in :func:`taylor`.

    ``labels`` chooses how points are identified — ``"legend"`` below the axes,
    ``"annotate"`` beside each marker, or ``"grid"`` (a color_by x marker_by matrix
    key) — exactly as for :func:`taylor`, so the two can be made to match. It defaults
    to ``"annotate"`` here because target points cluster near the origin when a model
    is good, and a label beside the marker stays readable there.

    ``font_scale``/``scale`` size the text from the figure, as in :func:`taylor`.
    ``marker_scale``/``alpha`` mean exactly what they do in :func:`taylor`, including
    the ``{level: value}`` dict form for styling particular groups — see its docstring.

    ``overlay``/``overlay_marker_scale``/``overlay_alpha``/``summary_points`` mean
    exactly what they do in :func:`taylor` — a second, emphasized layer (a highlighted
    subset, a per-group hexagon centroid, or both) drawn on top of the base cloud, styled to
    match its own group's colour rather than re-cycled independently. See its
    docstring for the full explanation.

    ``arrows`` draws a run's drift over time: pass ``True`` (shorthand for
    ``"time"``) or the name of whichever metric-record field varies along a
    trajectory, and every pair of comparisons that agree on everything else (their
    variable, depth, test and reference source) but that field gets connected —
    typically the output of ``compare(..., times=...)``, so
    ``runs.target(color_by="test", arrows=True)`` draws one arrow per run per
    consecutive time step. Each chain's first point is drawn hollow (its later
    points filled, as usual) so a lone glance shows both where a run ends up and
    which way it got there; a chain of one point (nothing to connect) draws
    unchanged. Arrow colour follows the *end* point's colour, so this composes with
    ``color_by``/``colors`` rather than needing its own.
    """
    import matplotlib.pyplot as plt

    labels_mode = _resolve_labels(labels)
    recs = _records(comparisons, groups)
    if not recs:
        raise ValueError("no comparisons to plot")
    if groups and not color_by and not marker_by:
        color_by = "group"
    labels_mode = _fallback_grid_without_both_channels(labels_mode, color_by, marker_by)
    style_field = color_by or marker_by or "label"
    if not normalize:
        _warn_mixed_variables(recs, "target")
    arrows_field = _resolve_arrows(arrows)
    chains = _arrow_chains(recs, arrows_field) if arrows_field else []
    chain_starts = {idxs[0] for idxs in chains}

    xy = np.array([_target_xy(r, normalize) for r in recs])
    x, y = xy[:, 0], xy[:, 1]
    point_labels = [r["label"] for r in recs]

    rings, boundary = _target_rings(circles, normalize, recs)
    ring_floor = max(rings) * 1.25 if rings else 0.0
    lim = max(1.15 * float(np.max(np.hypot(x, y))), ring_floor, 1.2 if normalize else 0.0)
    figsize = figsize or _diagram_figsize(TARGET_FIGSIZE, size=size, zoom=zoom)
    scale = _scale(figsize, font_scale=font_scale, override=scale)
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        # drawn into someone else's figure (paired); they fit the text once at the end
        fig = ax.figure

    for radius in rings:
        dashed = boundary is not None and np.isclose(radius, boundary)
        ax.add_patch(
            plt.Circle(
                (0, 0),
                radius,
                fill=False,
                color="0.55",
                ls="--" if dashed else ":",
                lw=0.9,
                zorder=1,
            )
        )
        ax.annotate(
            f"{radius:.3g}",
            (radius * 0.71, radius * 0.71),
            fontsize=scale["contour_label"],
            color="0.45",
            ha="left",
            va="bottom",
        )
    star_scale = _scalar_scale(marker_scale)
    ax.axhline(0, color="0.7", lw=0.7, zorder=1)
    ax.axvline(0, color="0.7", lw=0.7, zorder=1)
    ax.plot(
        0, 0, marker="*", ms=11 * star_scale, color="k", zorder=3, label="reference"
    )

    styles = _group_styles(recs, color_by, marker_by, colors, marker_scale, alpha)
    per_point = zip(
        x, y, styles.colors, styles.markers, styles.alphas, styles.scales, strict=True
    )
    for i, (xi, yi, ci, mi, al, scl) in enumerate(per_point):
        # A chain's first point is hollow — "this is where the run started" — drawn
        # with the same marker and colour as any other point of its group, just
        # unfilled; every other point (including a chain's own later ones) is filled
        # exactly as without arrows.
        hollow = i in chain_starts
        ax.scatter(
            xi,
            yi,
            s=70 * scl**2,
            color="none" if hollow else ci,
            marker=mi,
            zorder=4,
            edgecolor=ci if hollow else "white",
            linewidth=1.2 if hollow else 0.6,
            alpha=al,
        )
    if chains:
        points = list(zip(x, y, strict=True))
        _draw_arrows(ax, chains, points, styles.colors, zorder=3.5)

    overlay_specs = []
    if overlay is not None:
        overlay_specs += _overlay_point_specs(
            overlay,
            groups,
            lambda r: _target_xy(r, normalize)[0],
            lambda r: _target_xy(r, normalize)[1],
        )
    if summary_points:
        overlay_specs += _summary_point_specs(recs, x, y, style_field, summary_points)
    if overlay_specs:
        overlay_recs = [spec[2] for spec in overlay_specs]
        overlay_styles = _resolve_overlay_style(
            overlay_recs,
            recs,
            styles,
            style_field=style_field,
            marker_by=marker_by,
            overlay_marker_scale=overlay_marker_scale,
            overlay_alpha=overlay_alpha,
        )
        for (xi, yi, rec, mk_override), col, mk, al, scl in zip(
            overlay_specs,
            overlay_styles.colors,
            overlay_styles.markers,
            overlay_styles.alphas,
            overlay_styles.scales,
            strict=True,
        ):
            ax.scatter(
                xi,
                yi,
                s=70 * scl**2,
                color=col,
                marker=mk_override or mk,
                zorder=10,
                edgecolor="k",
                linewidth=0.8,
                alpha=al,
            )

    # Axes labelling before the key: _legend_below places itself below the lowest label
    # already drawn, so the x label has to exist by then or the key lands on top of it.
    xlabel, ylabel = _target_labels(normalize, recs, tex=True)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.label.set_size(scale["axes_label"])
    ax.yaxis.label.set_size(scale["axes_label"])
    ax.tick_params(labelsize=scale["tick_label"])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=scale["title"])

    if labels_mode == "annotate":
        _offset_labels(
            ax, x, y, point_labels, scale["annotation"], colors=styles.colors
        )
    elif labels_mode == "legend":
        _legend_below(
            fig, [*styles.handles, _reference_handle(star_scale)], scale["legend"]
        )
    elif labels_mode == "grid":
        handles, ncols = _grid_handles(
            recs, color_by, marker_by, colors, marker_scale, alpha, star_scale
        )
        _grid_legend_below(fig, handles, ncols, scale["legend"])

    if owns_figure:
        # That x label is longer than the axes it belongs to at any generous type level,
        # and it is one particular string rather than a scale problem — so it gets the
        # same measure-and-shrink treatment the field renderer gives a long panel title.
        _fit_text(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def paired(
    comparisons,
    *,
    title: str | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    labels: str | None = "legend",
    font_scale: float = 1.0,
    size=None,
    zoom: float = 1.0,
    scale: dict[str, float] | None = None,
    **kwargs,
):
    """Taylor and Target side by side — the pairing they are usually read as.

    Pure composition: :func:`taylor` draws into the left half of the figure and
    :func:`target` into the right, both accepting the same
    ``color_by``/``marker_by``/``groups``/``colors``/``marker_scale``/``alpha``/
    ``arrows`` style arguments, so the two panels stay visually consistent. Neither
    function knows about the other.

    ``labels`` applies to **both** panels, since the diagrams show the same points and
    identifying them two different ways in one figure reads as two unrelated plots. With
    ``"legend"`` or ``"grid"`` the key is drawn once beneath both panels rather than
    twice.

    One type scale is computed here for the two-column figure and handed to both panels,
    rather than each sizing itself: called alone they are square and near page width, so
    left to themselves they would pick the type for a figure twice the size of the half
    they actually get, and the pair would read as two plots pasted together. ``scale``
    overrides individual roles in it, as for :func:`taylor` — declared here rather than
    left to ``**kwargs`` because the panels are handed this figure's scale explicitly,
    which a forwarded ``scale=`` would collide with.
    """
    import matplotlib.pyplot as plt

    labels = _resolve_labels(labels)
    color_by = kwargs.get("color_by")
    marker_by = kwargs.get("marker_by")
    if kwargs.get("groups") and not color_by and not marker_by:
        color_by = "group"
    labels = _fallback_grid_without_both_channels(labels, color_by, marker_by)
    figsize = figsize or _diagram_figsize(PAIRED_FIGSIZE, size=size, zoom=zoom)
    scale = _scale(figsize, ncols=2, font_scale=font_scale, override=scale)
    fig = plt.figure(figsize=figsize)
    # Panels never draw their own key: with "legend"/"grid" it is shared (below), and
    # with "annotate" each panel labels its own markers.
    panel_labels = "annotate" if labels == "annotate" else None
    taylor(
        comparisons,
        fig=fig,
        rect=121,
        labels=panel_labels,
        title="Taylor",
        scale=scale,
        **kwargs,
    )
    ax_t = fig.add_subplot(122)
    target(
        comparisons,
        ax=ax_t,
        labels=panel_labels,
        title="Target",
        scale=scale,
        **kwargs,
    )

    if title:
        fig.suptitle(title, fontsize=scale["suptitle"])
    # Both of these move the axes, so they come before the key, which is placed by
    # measuring where the axes and their labels actually ended up.
    fig.subplots_adjust(wspace=0.35)
    if labels in ("legend", "grid"):
        # One shared key beneath both panels, so it cannot collide with either title
        recs = _records(comparisons, kwargs.get("groups"))
        panel_marker_scale = kwargs.get("marker_scale", 1.0)
        if labels == "grid":
            handles, ncols = _grid_handles(
                recs,
                color_by,
                marker_by,
                kwargs.get("colors"),
                panel_marker_scale,
                kwargs.get("alpha"),
                _scalar_scale(panel_marker_scale),
            )
            _grid_legend_below(fig, handles, ncols, scale["legend"])
        else:
            styles = _group_styles(
                recs,
                color_by,
                marker_by,
                kwargs.get("colors"),
                panel_marker_scale,
                kwargs.get("alpha"),
            )
            _legend_below(
                fig,
                [*styles.handles, _reference_handle(_scalar_scale(panel_marker_scale))],
                scale["legend"],
            )
    _fit_text(fig)
    if save:
        save = Path(save).expanduser()
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
