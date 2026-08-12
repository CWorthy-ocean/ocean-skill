"""One type scale and one sizing rule, shared by every renderer.

Font sizes used to be eleven independent constants, each hand-tuned once against a
page-width row of three maps: title 8, latitude labels 5, colorbar label 7 (or 6 in a
grid), row label 7, metrics 5.5, suptitle 9 (or 10)... Any change of figure size or row
count left all of them wrong in different directions, and the only way to find out was
to draw the figure, squint, and pass a different ``*_kwargs`` dict. Worse, matplotlib
points are absolute: with ``constrained_layout`` giving text priority over axes, an 8pt
title on a 4-inch-wide row squeezes the three maps down to a third of an inch each --
the figure stops being a figure, and no amount of retuning the *fonts* fixes it, because
the fonts are the cause.

Two ideas replace the eleven constants:

**A modular scale.** Every role is one base size times :data:`STEP` raised to a small
integer power (see :data:`FONT_STEPS`), so the sizes stay in fixed proportion to each
other and there is exactly one number to move. The exponents were chosen to reproduce
the hand-tuned sizes at the default figure size to within a point, so existing figures
look as they did; what changes is that they now *track* the figure.

**A base size read off the geometry.** :func:`type_scale` derives that base from one
grid cell's size, so a smaller panel gets smaller type without being asked, and the
text takes a roughly constant share of the space rather than an absolute one.

The same two ideas also size the figure itself (:func:`auto_figsize`): a row is as tall
as its maps' aspect ratio wants, plus the text above and below them measured in ems of
the type it just chose -- so asking for larger type gives the row more room instead of
squeezing the maps, which is the failure above in miniature.

Renderer-agnostic on purpose: the interactive renderer draws the same plot and must not
disagree about how big its type is, so it reads the same table through
:func:`bokeh_fontsize` rather than keeping its own numbers (bokeh needs CSS units and
strings, hence a converter rather than shared literals).
"""

from __future__ import annotations

import math

__all__ = [
    "ASPECT_LIMITS",
    "BASE_EXPONENT",
    "BASE_PT_AT_1IN",
    "BLANK_CELL_WEIGHT",
    "FACET_PANEL_W_FRACTION",
    "FONT_STEPS",
    "MAX_BASE_PT",
    "MIN_BASE_PT",
    "MIN_PT",
    "PAGE_H",
    "PAGE_W",
    "PANEL_W_FRACTION",
    "REFERENCE_FIGSIZE",
    "REFERENCE_GRID",
    "ROW_GAP_EM",
    "ROW_TITLE_EM",
    "STEP",
    "SUPTITLE_ALLOWANCE",
    "auto_figsize",
    "bokeh_fontsize",
    "bokeh_pt",
    "bokeh_scale",
    "clamp_aspect",
    "facet_figsize",
    "facet_layout",
    "frame_px",
    "reference_scale",
    "row_height",
    "type_scale",
]

#: Point size of the ``title`` role for a grid cell one inch across, and the exponent
#: relating the two: ``base = BASE_PT_AT_1IN * length_in ** BASE_EXPONENT``.
#:
#: Type does not scale linearly with the panel it sits in. Doubling both leaves the type
#: *feeling* oversized, because what has to stay readable is the glyph itself, not its
#: ratio to the frame -- which is why every plotting library ships absolute point
#: defaults rather than relative ones. But absolute sizes are what broke here, so the
#: rule has to be somewhere in between, and the two ends of the range this package
#: actually spans pin the curve:
#:
#: * a dense page-width row of three maps (cell ~2.79in square) reads well at the 8pt
#:   titles hand-tuned in ``matplotlib_renderer`` before this module existed;
#: * a standalone square diagram at matplotlib's own default figure size, 6.4x4.8in
#:   (cell ~5.54in), reads well at matplotlib's own default 12pt title.
#:
#: A power law through both anchors gives exponent 0.59 and coefficient 4.32; 0.6 is
#: well inside the tolerance of "reads well" and easier to reason about. Being
#: sub-linear also fails safe in the direction people actually go: a figure made much
#: bigger for a poster gets somewhat bigger type, not proportionally bigger type.
BASE_PT_AT_1IN = 4.32
BASE_EXPONENT = 0.6

#: Bounds on the base size. The floor is legibility -- below ~5pt the smaller roles
#: stop being readable at all and a bigger figure is the only real fix, so clamping here
#: (and at :data:`MIN_PT` per role) keeps a too-small figure merely cramped rather than
#: illegible. The ceiling stops a deliberately huge canvas becoming three words.
MIN_BASE_PT = 5.0
MAX_BASE_PT = 24.0

#: Absolute floor for any single role, applied after its step is taken. The base floor
#: alone does not protect the small end of the scale: at ``MIN_BASE_PT`` the latitude
#: labels are three steps down, i.e. 3.3pt.
MIN_PT = 4.0

#: Ratio between adjacent steps of the scale. 1.15 is a conventional modular-scale
#: ratio -- large enough that neighbouring roles are visibly distinct, small enough that
#: three steps of separation still leaves the smallest role readable.
STEP = 1.15

#: ``role -> (base, steps)``: which base the role is measured against, and how many
#: steps from it. Two bases, because two different things are being labelled:
#:
#: ``"panel"`` -- the type inside or beside one map, sized to one grid cell. It has to
#: shrink when there are eight rows on the page, because its panel did.
#:
#: ``"figure"`` -- the type that labels the whole figure (its suptitle, a shared
#: legend). A suptitle spans the figure's width whatever the row count, so shrinking it
#: with the rows would be sizing it against something it is not attached to; it is
#: measured against the cell *width* alone, which row count does not change.
FONT_STEPS: dict[str, tuple[str, int]] = {
    "suptitle": ("figure", 1),
    "legend": ("figure", -1),
    "title": ("panel", 0),
    "axes_label": ("panel", -1),
    "colorbar_label": ("panel", -1),
    "row_label": ("panel", -1),
    "colorbar_tick": ("panel", -2),
    "metrics": ("panel", -2),
    "tick_label": ("panel", -3),
    "annotation": ("panel", -3),
    "contour_label": ("panel", -3),
}


#: The page these figures have to fit, in inches. Lives here because it is the outer
#: bound on every sizing decision below — and because it was previously a literal 8.5 in
#: both renderer modules, free to drift apart.
PAGE_W = 8.5
PAGE_H = 11.0


def _base(length_in: float) -> float:
    """Return the base point size for a cell ``length_in`` inches across."""
    raw = BASE_PT_AT_1IN * max(length_in, 1e-3) ** BASE_EXPONENT
    return min(max(raw, MIN_BASE_PT), MAX_BASE_PT)


def type_scale(
    figsize: tuple[float, float],
    *,
    ncols: int = 1,
    nrows: int = 1,
    font_scale: float = 1.0,
    figure_ncols: int | None = None,
) -> dict[str, float]:
    """Concrete point size for every role in :data:`FONT_STEPS`.

    ``figsize``, ``ncols`` and ``nrows`` describe the grid the type has to live in; one
    cell is ``(figsize[0] / ncols, figsize[1] / nrows)``. The panel base comes from the
    cell's geometric mean, so a cell that is wide but short -- eight rows of maps
    stacked down a page -- gets smaller type than its width alone would suggest, which
    is right, since what is short is the space the title and the longitude labels have
    to share with the map. The figure base comes from the cell width alone; see
    :data:`FONT_STEPS`.

    ``figure_ncols`` overrides the column count used for that *figure* base only. It
    exists because a suptitle spans the page whatever grid is beneath it, so its size
    should not move with the column count -- and for the families that pre-date facet
    grids it never did, there being no such thing as a figure here that was not three
    columns wide. A one-column facet grid asking for the figure base off its own cell
    would get a 17pt suptitle where every other figure in the same report has 9.
    Callers with a real grid pass :data:`REFERENCE_GRID`'s column count; callers whose
    "figure" genuinely is one cell (a bokeh frame, via :func:`bokeh_scale`) leave it
    alone.

    ``font_scale`` multiplies every result, for "all of this a bit bigger" without
    disturbing the proportions -- the one knob that replaces retuning eleven numbers.
    """
    cell_w = figsize[0] / max(ncols, 1)
    cell_h = figsize[1] / max(nrows, 1)
    figure_w = figsize[0] / max(figure_ncols or ncols, 1)
    bases = {
        "panel": _base(math.sqrt(max(cell_w, 1e-3) * max(cell_h, 1e-3))),
        "figure": _base(figure_w),
    }
    scale = {}
    for role, (base, steps) in FONT_STEPS.items():
        size = bases[base] * STEP**steps * font_scale
        scale[role] = round(max(size, MIN_PT * font_scale), 1)
    return scale


#: The figure size a page-width row of three maps defaults to, and the grid it forms:
#: the shape :func:`reference_scale` reports sizes for, and the shape the anchors in
#: :data:`BASE_PT_AT_1IN` were taken from.
REFERENCE_FIGSIZE = (PAGE_W, PAGE_W / 3.1)
REFERENCE_GRID = (3, 1)  # ncols, nrows


def reference_scale() -> dict[str, float]:
    """Return the scale at the default page-width row: what the documented sizes mean.

    Every ``*_kwargs`` default in the renderers is now computed per figure, so there is
    no literal dict for the docs (or for the "did you mean to put this inside
    ``title_kwargs``?" error) to point at. This is the one canonical instance: the sizes
    a default ``field_row`` actually gets, which are also the hand-tuned sizes the scale
    was calibrated to reproduce.
    """
    ncols, nrows = REFERENCE_GRID
    return type_scale(REFERENCE_FIGSIZE, ncols=ncols, nrows=nrows)


# --- figure sizing -----------------------------------------------------------------

#: Fraction of its grid cell that a map panel actually occupies horizontally, the rest
#: being the colorbar, the colorbar's labels and the latitude labels. Measured across
#: the figures this package draws -- page-width rows and 1-8 row grids of cartopy
#: GeoAxes -- where it holds to within a few percent, because most of what it accounts
#: for is a colorbar of fixed aspect rather than text. Used to turn the panel size the
#: data wants into the figure size that yields it.
PANEL_W_FRACTION = 0.73

#: Vertical room a row needs beyond the map itself, in ems of the row's own base font
#: size: the panel title above, the longitude labels below, and constrained_layout's
#: padding. Expressed in ems rather than inches so that asking for bigger type makes
#: the row taller instead of eating the map -- the whole point of deriving the two
#: together. The second figure adds the horizontal colorbar and its label, which a
#: single row puts below the maps where a grid puts them beside.
ROW_OVERHEAD_EM = 8.1
ROW_OVERHEAD_EM_HORIZONTAL_CBAR = 11.1

#: Aspect ratios (lon span / lat span) outside which a map is letterboxed rather than
#: sized to fit: past these the row would be a sliver or taller than the page, and a
#: band of white above and below the map is the lesser evil.
ASPECT_LIMITS = (0.3, 4.0)

#: Vertical inches held back from the page for a suptitle, i.e. what the maps may not
#: use. Named because :func:`row_height` and :func:`facet_layout` must agree about it —
#: a layout chosen against the full page and then drawn into a shorter one is a layout
#: chosen for a figure that does not exist.
SUPTITLE_ALLOWANCE = 0.8

#: Fraction of its cell a *facet* panel occupies horizontally, the counterpart of
#: :data:`PANEL_W_FRACTION` for a grid whose colorbar is shared by every panel rather
#: than drawn per row. With no bar in the cell, only the latitude labels and the
#: inter-panel padding come out of it, so the panel keeps materially more of its width.
FACET_PANEL_W_FRACTION = 0.88

#: How hard :func:`facet_layout` argues against blank cells, per unit of blank *share*
#: (blanks / cells). Pure aspect-matching will happily leave a 6-panel sequence as 5x2
#: — four of ten cells empty — because those cells happen to be the right shape; for an
#: ordered series of maps that reads as a mistake rather than as a layout. At 1.5 a
#: layout must be a good deal closer in aspect to justify each blank it adds, which
#: recovers the 3x2 a person would have drawn without forbidding ragged grids outright
#: (7 panels as 3x3 is still right).
BLANK_CELL_WEIGHT = 1.5


def clamp_aspect(aspect: float) -> float:
    """``aspect`` brought inside :data:`ASPECT_LIMITS`, guarding against zero spans."""
    lo, hi = ASPECT_LIMITS
    if not math.isfinite(aspect) or aspect <= 0:
        return 1.0
    return float(min(max(aspect, lo), hi))


def row_height(
    aspect: float,
    *,
    nrows: int = 1,
    ncols: int = 3,
    page_w: float = PAGE_W,
    page_h: float = PAGE_H,
    font_scale: float = 1.0,
    horizontal_colorbar: bool = False,
    panel_w_fraction: float = PANEL_W_FRACTION,
) -> float:
    """Height (inches) of one row of maps: the map, plus the type around it.

    ``aspect`` is the map's ``lon_span / lat_span``. The map's height follows from it
    and from the width a cell of the page has to spare (:data:`PANEL_W_FRACTION`); the
    rest of the row is text overhead, which is where this gets circular -- the overhead
    is a multiple of the font size, the font size comes from the cell size, and the cell
    size is what is being solved for. The loop below settles it: because the base goes
    as the cell's size to the power 0.6, and the overhead is a fraction of the row, each
    pass moves the answer by a fraction of the last move, so three passes agree to well
    under a tenth of a point.

    ``panel_w_fraction`` is how much of its cell the map itself gets, the rest being
    the colorbar and labelling; it is a parameter rather than the constant it used to
    be because a facet grid shares one colorbar across every panel instead of drawing
    one per row, so its panels keep more of their cell (:data:`FACET_PANEL_W_FRACTION`).

    Capped so ``nrows`` rows still fit ``page_h``, leaving room for a suptitle. Past
    that cap the maps are squeezed, which is the honest outcome: the alternative is a
    figure taller than the page it has to print on.
    """
    panel_w = page_w / max(ncols, 1) * panel_w_fraction
    panel_h = panel_w / clamp_aspect(aspect)
    em = ROW_OVERHEAD_EM_HORIZONTAL_CBAR if horizontal_colorbar else ROW_OVERHEAD_EM
    cell_w = page_w / max(ncols, 1)
    height = panel_h
    for _ in range(3):
        base = _base(math.sqrt(cell_w * height)) * font_scale
        height = panel_h + em * base / 72.0
    return float(min(height, (page_h - SUPTITLE_ALLOWANCE) / max(nrows, 1)))


#: Vertical room, in ems, that each row *after the first* needs in a facet grid: the
#: gap between one row's map and the next one's, and — when every row carries its own
#: title — the title too. Both are much less than :data:`ROW_OVERHEAD_EM`, which sizes
#: a row that is self-contained: a title above, longitude labels below, padding for
#: both. In a facet grid those decorations are shared (titles on the top row, longitude
#: labels on the bottom), so charging every row for a full set leaves the rows visibly
#: adrift from each other — most of a panel's height of white between them on a 3x6.
ROW_GAP_EM = 2.6
ROW_TITLE_EM = 2.4


def facet_figsize(
    aspect: float,
    *,
    nrows: int,
    ncols: int,
    title_every_row: bool = True,
    page_w: float = PAGE_W,
    page_h: float = PAGE_H,
    font_scale: float = 1.0,
    panel_w_fraction: float = FACET_PANEL_W_FRACTION,
) -> tuple[float, float]:
    """Figure size for a facet grid, charging shared decorations only once.

    :func:`auto_figsize` multiplies one self-contained row's height by the row count,
    which is right for a stack of independent rows and wrong for a grid whose rows
    share their titles and axis labels — there the second and later rows need a gap and
    (sometimes) a title, not a title *and* a set of longitude labels *and* the padding
    for both.

    Same circular solve as :func:`row_height`: the overhead is a multiple of the font
    size, the font size comes from the cell size, and the cell size is what is being
    solved for. Three passes settle it for the same reason.

    Capped at the page less :data:`SUPTITLE_ALLOWANCE`. A one-column grid of wide maps
    hits that cap and is squeezed exactly as it was before this function existed, so
    the saving only appears where the figure was not page-limited to begin with —
    which is where the slack actually was.
    """
    panel_w = page_w / max(ncols, 1) * panel_w_fraction
    panel_h = panel_w / clamp_aspect(aspect)
    nrows = max(nrows, 1)
    extra_em = ROW_GAP_EM + (ROW_TITLE_EM if title_every_row else 0.0)
    em = ROW_OVERHEAD_EM + (nrows - 1) * extra_em
    cell_w = page_w / max(ncols, 1)

    height = panel_h * nrows
    for _ in range(3):
        base = _base(math.sqrt(cell_w * height / nrows)) * font_scale
        height = panel_h * nrows + em * base / 72.0
    return (page_w, float(min(height, page_h - SUPTITLE_ALLOWANCE)))


def facet_layout(
    n: int,
    aspect: float,
    *,
    page_w: float = PAGE_W,
    page_h: float = PAGE_H,
    blank_weight: float = BLANK_CELL_WEIGHT,
) -> tuple[int, int]:
    """Return ``(ncols, nrows)`` for ``n`` panels of a map with the given aspect ratio.

    The orientation of a facet grid is not a free choice: a wide domain (a Gulf of
    Mexico box, ``aspect`` ~4) stacks down the page one panel per row, while a tall one
    (a California Current box, ~0.35) spreads across it, and getting this backwards
    wastes most of the page on white space. So rather than a ``ncols=`` parameter with a
    guessed default, every candidate is scored and the best one returned.

    The score is how far the resulting *cell* is from the shape the map wants, measured
    as a log ratio so that being twice as wide as wanted costs the same as being half as
    wide — an asymmetric error would bias every domain toward one orientation. Blank
    cells are then charged for (:data:`BLANK_CELL_WEIGHT`), because these panels are
    normally an ordered series and a grid with a third of its cells empty reads as a
    bug in a way that a merely imperfect aspect ratio does not.

    The page height available is the page less :data:`SUPTITLE_ALLOWANCE`, matching
    :func:`row_height`, so the layout is chosen against the space the figure will
    actually be drawn into.
    """
    if n <= 0:
        raise ValueError(f"a facet grid needs at least one panel, got {n}")
    want = clamp_aspect(aspect)
    usable_h = max(page_h - SUPTITLE_ALLOWANCE, 1e-3)

    best, best_cost = (1, n), math.inf
    for ncols in range(1, n + 1):
        nrows = math.ceil(n / ncols)
        cell_aspect = (page_w / ncols) / (usable_h / nrows)
        cells = ncols * nrows
        cost = abs(math.log(cell_aspect / want)) + blank_weight * (cells - n) / cells
        if cost < best_cost:
            best, best_cost = (ncols, nrows), cost
    return best


def auto_figsize(
    aspect: float,
    *,
    nrows: int = 1,
    ncols: int = 3,
    page_w: float = PAGE_W,
    page_h: float = PAGE_H,
    font_scale: float = 1.0,
    horizontal_colorbar: bool = False,
    panel_w_fraction: float = PANEL_W_FRACTION,
) -> tuple[float, float]:
    """Page-width figure size for ``nrows`` rows of maps of the given aspect ratio.

    Thin wrapper over :func:`row_height`; the width is the page, since these figures
    exist to be read at page width.
    """
    height = row_height(
        aspect,
        nrows=nrows,
        ncols=ncols,
        page_w=page_w,
        page_h=page_h,
        font_scale=font_scale,
        horizontal_colorbar=horizontal_colorbar,
        panel_w_fraction=panel_w_fraction,
    )
    return (page_w, height * max(nrows, 1))


# --- interactive (bokeh) -------------------------------------------------------------

#: CSS pixels per inch, the unit bokeh's ``frame_width``/``frame_height`` are in. Needed
#: to ask :func:`type_scale` -- which thinks in inches -- about a bokeh frame.
CSS_PX_PER_INCH = 96.0

#: Points per CSS pixel. Bokeh font sizes are CSS ``pt``, which is 4/3 of a CSS pixel,
#: whereas matplotlib's point is 1/72 inch. Without this conversion the same physical
#: size would come out a third too small interactively -- the two renderers visibly
#: disagreeing about a plot they are supposed to draw identically.
PT_PER_CSS_PX = 4.0 / 3.0

#: ``our role -> bokeh's`` name in a holoviews ``fontsize`` option dict. ``metrics`` has
#: no entry: the interactive renderer folds the metrics into a panel title (bokeh has no
#: equivalent of a free-floating text box), so they are already covered by ``title``.
_BOKEH_KEYS: dict[str, str] = {
    "title": "title",
    "axes_label": "labels",
    "tick_label": "ticks",
    "colorbar_label": "clabel",
    "colorbar_tick": "cticks",
    "legend": "legend",
}


def bokeh_scale(
    frame_px: tuple[float, float], *, font_scale: float = 1.0
) -> dict[str, str]:
    """Every role of :func:`type_scale`, as a bokeh CSS-point string, for one frame.

    Same scale as the static renderer, converted twice: pixels to inches to ask
    :func:`type_scale` for point sizes, then matplotlib points to bokeh's CSS points.
    A bokeh ``frame_width`` is the plot frame itself rather than a grid cell containing
    it, so it is passed as a one-cell grid.

    Keyed by *our* role names, so a renderer can also reach a role bokeh's own
    ``fontsize`` dict has no key for -- a ``hv.Labels`` element's ``text_font_size``,
    say. :func:`bokeh_fontsize` is this renamed into that dict.
    """
    figsize = (frame_px[0] / CSS_PX_PER_INCH, frame_px[1] / CSS_PX_PER_INCH)
    scale = type_scale(figsize, ncols=1, nrows=1, font_scale=font_scale)
    return {role: bokeh_pt(size) for role, size in scale.items()}


def bokeh_fontsize(
    frame_px: tuple[float, float],
    *,
    font_scale: float = 1.0,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Holoviews ``fontsize`` option dict for a frame ``frame_px`` CSS pixels in size.

    :func:`bokeh_scale` with the roles renamed to the keys holoviews knows, dropping the
    ones it has no slot for.
    """
    scale = bokeh_scale(frame_px, font_scale=font_scale)
    out = {bokeh: scale[role] for role, bokeh in _BOKEH_KEYS.items()}
    out.update(extra or {})
    return out


def bokeh_pt(size_pt: float) -> str:
    """One matplotlib point size as a bokeh CSS-point string, e.g. ``"7.1pt"``."""
    return f"{round(size_pt * PT_PER_CSS_PX, 1):g}pt"


def frame_px(
    aspect: float, *, width_px: float = 260.0, font_scale: float = 1.0
) -> tuple[int, int]:
    """Bokeh ``(frame_width, frame_height)`` for a map of the given aspect ratio.

    The static renderer sizes its panels to the data's aspect ratio rather than a fixed
    box (see :func:`row_height`); a fixed interactive frame would letterbox exactly the
    domains the static one fits. ``font_scale`` is accepted and ignored -- a bokeh frame
    excludes its own decorations, so bigger type grows the plot around the frame rather
    than taking room from it -- so that callers can pass it uniformly.
    """
    del font_scale  # frame_* excludes decorations; type does not eat into it
    height = width_px / clamp_aspect(aspect)
    return round(width_px), round(height)
