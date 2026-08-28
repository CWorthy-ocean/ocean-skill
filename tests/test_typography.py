"""Tests for the automatic type scale: sizes and figure sizes read off geometry.

Before :mod:`ocean_skill.plot.typography` there were eleven independent font-size
constants, each tuned once against a page-width row of three maps, and one figure-size
formula with a flat text allowance baked into it. Every one of them was wrong at any
other figure size, and the way you found out was to draw the figure and look.

So the tests here are mostly *relational* rather than pinned to numbers: that the sizes
move the right way when the geometry changes, that the proportions between roles hold,
that the two renderers agree about how big a label is relative to the panel it sits in.
A test asserting ``title == 8.0`` would just re-freeze the problem one figure size up.
The calibration point is pinned deliberately in one place, because reproducing the
hand-tuned appearance at the default size is a promise to existing notebooks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot import typography as tg
from ocean_skill.plot.matplotlib_renderer import field_grid, field_row

# Every test here draws a real figure and measures its geometry (layout depends on the
# backend's DPI, so this needs Agg -- set once, process-wide, in conftest.py). Skipped
# by default (`pytest.ini`'s `-m "not slow"`); run explicitly with `pytest -m slow` or
# `pytest -m ""`.
pytestmark = pytest.mark.slow

PANEL_ROLES = [r for r, (base, _) in tg.FONT_STEPS.items() if base == "panel"]
FIGURE_ROLES = [r for r, (base, _) in tg.FONT_STEPS.items() if base == "figure"]


# --- the scale itself --------------------------------------------------------------


def test_default_row_lands_on_the_chosen_level():
    """The level is a decision, so it is pinned — the one absolute assertion here.

    These sizes were chosen by rendering the same comparison at 8/11/12/14pt titles and
    picking. That makes them a deliberate choice rather than an accident, and the whole
    point of the previous round was that the sizes it shipped were *not*: they
    reproduced matplotlib-era defaults nobody had picked. If the coefficient drifts,
    this fails and someone has to look at a figure again rather than at arithmetic.
    """
    scale = tg.reference_scale()
    for role, chosen in {
        "title": 11.1,
        "suptitle": 12.9,
        "colorbar_label": 9.7,
        "row_label": 9.7,
        "tick_label": 7.3,
        "metrics": 8.4,
    }.items():
        assert scale[role] == pytest.approx(chosen, abs=0.3), role


def test_the_level_is_well_clear_of_the_old_inherited_defaults():
    """A guard against silently sliding back to the sizes that prompted this.

    The scale first shipped calibrated on 8pt titles and 5pt coordinate labels, taken
    from matplotlib-era literals. Those turned out to be too small for anything this
    package is used for, so the floor of the acceptable range is worth stating.
    """
    scale = tg.reference_scale()
    assert scale["title"] >= 10.0
    assert scale["tick_label"] >= 7.0


def test_panel_type_shrinks_with_the_cell_and_grows_with_it():
    """Monotone in cell size, which is the whole point of deriving it from geometry."""
    small = tg.type_scale((4.0, 1.3), ncols=3, nrows=1)
    default = tg.reference_scale()
    big = tg.type_scale((14.0, 4.5), ncols=3, nrows=1)
    for role in PANEL_ROLES:
        assert small[role] < default[role] < big[role], role


def test_type_is_sublinear_in_the_cell_size():
    """Doubling the canvas must not double the type.

    This is what makes a bigger figure buy *detail* rather than magnification, and what
    lets one calibration serve the whole range — the sizes chosen at page width stay
    reasonable on a poster instead of scaling into absurdity. A linear rule would keep
    the type a constant share of the panel, so the figure would look identical at every
    size and there would be no point growing it.
    """
    one = tg.type_scale((8.5, 2.74), ncols=3, nrows=1)["title"]
    two = tg.type_scale((17.0, 5.48), ncols=3, nrows=1)["title"]
    assert one < two < 2 * one
    # the specific exponent, stated where it can be checked: 2x canvas -> ~1.5x type
    assert two / one == pytest.approx(2**tg.BASE_EXPONENT, rel=0.02)


def test_roles_keep_their_proportions_at_every_size():
    """Ratios between roles are fixed; only the base moves.

    Eleven sizes that each moved independently is the state this replaced, so the
    invariant worth testing is that they now move *together*.
    """
    a = tg.type_scale((8.5, 2.74), ncols=3, nrows=1)
    b = tg.type_scale((20.0, 6.5), ncols=3, nrows=1)
    ratio = b["title"] / a["title"]
    for role in PANEL_ROLES:
        assert b[role] / a[role] == pytest.approx(ratio, rel=0.02), role


def test_font_scale_multiplies_everything_uniformly():
    plain = tg.type_scale((8.5, 2.74), ncols=3, nrows=1)
    bigger = tg.type_scale((8.5, 2.74), ncols=3, nrows=1, font_scale=1.5)
    for role in plain:
        assert bigger[role] == pytest.approx(plain[role] * 1.5, rel=0.02), role


def test_row_count_shrinks_panel_type_but_not_the_suptitle():
    """A suptitle labels the figure, so stacking rows must not shrink it.

    This is why there are two bases. Sizing the suptitle against the cell would make an
    eight-row grid's overall title smaller than a two-row grid's, even though both span
    the same 8.5 inches and answer the same question about the figure as a whole.
    """
    few = tg.type_scale((8.5, 4.7), ncols=3, nrows=2)
    many = tg.type_scale((8.5, 10.2), ncols=3, nrows=8)
    for role in PANEL_ROLES:
        assert many[role] < few[role], role
    for role in FIGURE_ROLES:
        assert many[role] == few[role], role


def test_nothing_falls_below_the_legibility_floor():
    """A figure too small for its content gets cramped type, never invisible type."""
    scale = tg.type_scale((1.0, 0.4), ncols=3, nrows=1)
    assert min(scale.values()) >= tg.MIN_PT


def test_the_base_is_capped_at_the_large_end():
    huge = tg.type_scale((400.0, 300.0))
    assert huge["title"] <= tg.MAX_BASE_PT


# --- figure sizing ------------------------------------------------------------------


def test_row_height_follows_the_maps_aspect_ratio():
    """A wide domain gets a short row and a tall one a tall row."""
    wide = tg.row_height(3.0, nrows=1)
    square = tg.row_height(1.0, nrows=1)
    tall = tg.row_height(0.4, nrows=1)
    assert wide < square < tall


def test_row_height_is_bounded_by_a_canvas_that_caps_its_height():
    """Eight rows still have to print, which is what the ``"page"`` cap is for."""
    assert tg.row_height(1.0, nrows=8) * 8 <= tg.PAGE_H


# --- facet grids: shared decorations cost less than self-contained rows -------------


def test_an_extra_facet_row_costs_less_than_the_first_one():
    """The invariant the old constants broke, stated where it can be checked.

    A facet grid's later rows share the top row's titles and the bottom row's longitude
    labels, so each should cost only a gap (plus its own title, if any) — less than the
    first row's full set. Stated in ems alone at 2.6 + 2.4, they charged 5.0 em
    where the first row was charged 1.6, which is the wrong way round and left ~0.6in of
    dead space at every row boundary of a three-row grid.
    """
    from itertools import pairwise

    heights = [tg.facet_figsize(1.385, nrows=n, ncols=3)[1] for n in (1, 2, 3, 4)]
    one_row = heights[0]
    extras = [b - a for a, b in pairwise(heights)]
    assert all(e < one_row for e in extras), (one_row, extras)
    # and each extra row costs about the same as the one before it
    assert max(extras) - min(extras) < 0.05


def test_a_facet_grid_amortises_its_shared_decorations_over_its_rows():
    """The height *per row* falls as rows are added, because the shared cost is fixed.

    That is what "charged once" means arithmetically, and it is the property that
    separates ``facet_figsize`` from multiplying :func:`row_height` by the row count. It
    also depends on the fixed term being right: ``facet_figsize`` used ``ROW_OVERHEAD``,
    the figure for a *self-contained* grid row, 0.36in short of what a facet grid needs
    once (it carries one colorbar for the figure where a grid row carries its own).
    """
    per_row = []
    for nrows in (1, 2, 3, 4):
        _, h = tg.facet_figsize(1.385, nrows=nrows, ncols=3)
        per_row.append(h / nrows)
    assert per_row == sorted(per_row, reverse=True), per_row
    # a self-contained row of the same maps costs more than any amortised facet row
    assert tg.row_height(1.385, nrows=1) < per_row[0]


def test_facet_layout_still_prefers_the_grid_a_person_would_draw():
    """9 panels is 3x3, not 2x5 — the case ``BLANK_CELL_WEIGHT`` exists to protect.

    Lowering ``SUPTITLE_ALLOWANCE`` to its measured value gave every candidate a little
    more height and tipped this one to 2x5, adding a blank cell where there had been
    none.
    """
    assert tg.facet_layout(9, 1.385) == (3, 3)
    assert tg.facet_layout(6, 1.385) == (2, 3)
    # a wide domain still stacks and a tall one still spreads
    assert tg.facet_layout(4, 4.0)[0] < tg.facet_layout(4, 0.35)[0]


# --- the canvas ---------------------------------------------------------------------


def test_an_uncapped_canvas_keeps_every_rows_panels_at_full_height():
    """``size="free"`` is the escape from the report constraint.

    The cap squeezes panels so a many-row grid fits a page — right for a PDF, wrong for
    a notebook, and previously unavoidable because 8.5x11 was hardwired into the sizing
    rule rather than being one canvas among several.
    """
    page, free = tg.CANVASES["page"], tg.CANVASES["free"]
    capped = tg.row_height(1.4, nrows=8, canvas=page)
    uncapped = tg.row_height(1.4, nrows=8, canvas=free)
    assert uncapped > capped
    # uncapped, the row is the height its aspect ratio asked for, whatever the row count
    assert uncapped == pytest.approx(tg.row_height(1.4, nrows=1, canvas=free))


def test_zoom_scales_the_canvas_and_the_type_follows_sub_linearly():
    """``zoom`` is the "make it bigger" knob; ``font_scale`` is the "more type" knob.

    They are separate because they answer different questions, and because zoom's type
    growth is deliberately sub-linear — a figure twice the size shows more detail rather
    than the same figure magnified.
    """
    plain = tg.resolve_canvas("page")
    doubled = tg.resolve_canvas("page", zoom=2.0)
    assert doubled.width == pytest.approx(2 * plain.width)
    assert doubled.max_height == pytest.approx(2 * plain.max_height)

    small = tg.type_scale((plain.width, 3.0), ncols=3)
    big = tg.type_scale((doubled.width, 6.0), ncols=3)
    assert 1.0 < big["title"] / small["title"] < 2.0


def test_size_accepts_a_name_a_pair_a_number_and_a_canvas():
    assert tg.resolve_canvas("slide") == tg.CANVASES["slide"]
    assert tg.resolve_canvas((6.5, None)) == tg.Canvas(6.5, None)
    assert tg.resolve_canvas(6.5) == tg.Canvas(6.5, None)
    assert tg.resolve_canvas(tg.Canvas(4.0, 9.0)) == tg.Canvas(4.0, 9.0)
    assert tg.resolve_canvas(None) == tg.CANVASES[tg.DEFAULT_SIZE]


@pytest.mark.parametrize(
    ("bad", "exc"),
    [
        ("A4", ValueError),  # not a preset — the message lists what is
        ((1.0, 2.0, 3.0), ValueError),
        (0.0, ValueError),
        (object(), TypeError),
    ],
)
def test_a_bad_size_is_rejected_by_name(bad, exc):
    with pytest.raises(exc):
        tg.resolve_canvas(bad)


def test_a_bad_zoom_is_rejected():
    with pytest.raises(ValueError):
        tg.resolve_canvas("page", zoom=0.0)


def test_extreme_aspects_are_letterboxed_rather_than_obeyed():
    """Past the limits, a band of white beats a figure taller than the page."""
    assert tg.row_height(20.0, nrows=1) == tg.row_height(tg.ASPECT_LIMITS[1], nrows=1)
    assert tg.row_height(0.01, nrows=1) == tg.row_height(tg.ASPECT_LIMITS[0], nrows=1)
    for degenerate in (0.0, -1.0, float("nan"), float("inf")):
        assert np.isfinite(tg.row_height(degenerate, nrows=1))


def test_larger_type_buys_a_taller_row_rather_than_a_smaller_map():
    """The text allowance is ems, not inches — the reason the two are solved together.

    With a flat inch allowance, ``font_scale`` would have been a way to *squeeze the
    maps*: the row height would not move, so every extra point of title came out of the
    panel. Growing the row instead is what makes the knob mean what it says.
    """
    plain = tg.row_height(1.4, nrows=1)
    bigger = tg.row_height(1.4, nrows=1, font_scale=1.6)
    assert bigger > plain


def test_a_horizontal_colorbar_needs_more_of_the_rows_height():
    """A single row puts its bars below the maps; a grid puts them beside."""
    assert tg.row_height(1.4, horizontal_colorbar=True) > tg.row_height(
        1.4, horizontal_colorbar=False
    )


# --- as actually drawn --------------------------------------------------------------


def _field(offset: float = 0.0, *, lon1: float = 270.0) -> xr.DataArray:
    return xr.DataArray(
        5.0 + offset + np.linspace(0, 1, 80).reshape(8, 10),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(18, 26, 8), "lon": np.linspace(260, lon1, 10)},
    )


def _aligned(**kwargs):
    test, ref = _field(1.0, **kwargs), _field(0.0, **kwargs)
    return {"test": test, "reference": ref, "difference": test - ref}


def _titles(fig) -> list[float]:
    return [ax.title.get_fontsize() for ax in fig.axes if ax.title.get_text()]


def _panel_box(fig):
    """Measure the first map panel, in inches, after the layout has settled."""
    fig.canvas.draw()
    w, h = fig.get_size_inches()
    ax = next(ax for ax in fig.axes if hasattr(ax, "projection"))
    pos = ax.get_position()
    return pos.width * w, pos.height * h


def test_a_canvas_far_too_small_still_draws_maps_and_says_it_is_too_small():
    """The honest limit of the whole approach, stated where it can be checked.

    matplotlib points are absolute and ``constrained_layout`` gives text priority over
    axes, so fixed 8pt titles on a 4-inch row squeezed the three maps to about a third
    of an inch — the text did not look wrong, it *ate the figure*.

    Deriving type from geometry does not rescue this case, and it is worth being clear
    about why: the ``MIN_PT`` floor binds long before the maps get their room back, so
    at four inches the panels stay about a third of an inch whatever the scale does. The
    remedy is a bigger canvas, which is exactly what the warning says. What this asserts
    is therefore the two things that *are* true: the figure is still a figure, and it
    tells you what is wrong instead of leaving you to wonder.
    """
    with pytest.warns(UserWarning, match="no room left to reclaim"):
        fig = field_row(_aligned(), figsize=(4.0, 4.0 / 3.1), title="t")
    panel_w, panel_h = _panel_box(fig)
    assert panel_w > 0.25 and panel_h > 0.15, "the maps were crowded out entirely"


def test_a_canvas_with_room_gets_panels_that_dominate_it():
    """Where the sizing rule does pay off: a normal canvas, mostly map.

    The counterpart to the test above — at page width the type takes a modest share and
    the panels get the rest, which is the outcome the scale exists to produce.
    """
    fig = field_row(_aligned(), title="t")
    panel_w, _ = _panel_box(fig)
    cell_w = fig.get_size_inches()[0] / 3
    assert panel_w > 0.6 * cell_w


def test_a_canvas_too_narrow_says_to_widen_it():
    """Below about five inches, three maps plus their labelling do not fit.

    No font size fixes that, because the canvas is the constraint — the type is already
    at its floor. Silently drawing sliver panels is the one outcome that leaves the user
    guessing, so the figure reports what to change instead.
    """
    with pytest.warns(UserWarning, match="widen the canvas"):
        field_row(_aligned(), figsize=(3.5, 1.2), title="t")


def test_a_narrow_but_uncapped_canvas_can_still_fit():
    """``size=3.5`` alone is not a problem: the row grows as tall as it needs to.

    Only a canvas that is narrow *and* short forces the squeeze — worth pinning so the
    warning does not start firing on every deliberately narrow figure.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        field_row(_aligned(), size=3.5, title="t")


def test_a_height_capped_grid_is_told_to_lift_the_cap_not_to_widen():
    """The two ways to run out of room want opposite advice.

    Nine rows on a page are cramped because the *height* cap is splitting between them,
    and widening does nothing for that — the fix is ``size="free"``. Naming the wrong
    knob would send someone off in the wrong direction, so the message picks by which
    constraint actually bound.
    """
    rows = [{"aligned": _aligned(), "units": "degC"} for _ in range(9)]
    with pytest.warns(UserWarning, match='size="free"'):
        field_grid(rows, title="t")
    # and with the cap lifted there is nothing to report
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        field_grid(rows, title="t", size="free")


def test_a_normal_canvas_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        field_row(_aligned(), title="t")
        field_grid([{"aligned": _aligned(), "units": "degC"}] * 3, title="t")


def test_drawn_type_tracks_the_figure_size():
    small = field_row(_aligned(), figsize=(4.5, 4.5 / 3.1), title="t")
    large = field_row(_aligned(), figsize=(14.0, 14.0 / 3.1), title="t")
    assert max(_titles(small)) < max(_titles(large))


def test_colorbar_tick_labels_are_sized_with_everything_else():
    """They used to be the one text in the figure nobody had sized.

    ``colorbar_kwargs`` carried ``label_size`` but no tick size, so the bar's numbers
    fell through to rcParams' 10pt while the titles around them were 8 and the latitude
    labels 5 — it read as another figure's colorbar pasted in.
    """
    fig = field_row(_aligned(), units="degC", title="t")
    fig.canvas.draw()
    bars = [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]
    assert bars
    sizes = [
        label.get_fontsize()
        for ax in bars
        for label in ax.get_xticklabels() + ax.get_yticklabels()
        if label.get_text()
    ]
    assert sizes
    assert max(sizes) < max(_titles(fig))


def test_font_scale_reaches_the_drawn_figure():
    plain = field_row(_aligned(), title="t")
    bigger = field_row(_aligned(), title="t", font_scale=1.5)
    assert max(_titles(bigger)) > 1.3 * max(_titles(plain))


def test_zoom_moves_figure_panels_and_type_together():
    """One knob, three consequences — the point of deriving them from one canvas."""
    plain = field_row(_aligned(), title="t")
    big = field_row(_aligned(), title="t", zoom=1.6)
    assert big.get_size_inches()[0] == pytest.approx(1.6 * plain.get_size_inches()[0])
    assert _panel_box(big)[0] > _panel_box(plain)[0]
    assert max(_titles(big)) > max(_titles(plain))


def test_a_named_size_reaches_the_drawn_figure():
    slide = field_row(_aligned(), title="t", size="slide")
    assert slide.get_size_inches()[0] == pytest.approx(tg.CANVASES["slide"].width)


def test_size_free_lets_a_many_row_grid_keep_its_panels():
    """The drawn counterpart of the uncapped-canvas test above."""
    rows = [{"aligned": _aligned(), "units": "degC"} for _ in range(8)]
    page = field_grid(rows, title="t")
    free = field_grid(rows, title="t", size="free")
    assert free.get_size_inches()[1] > page.get_size_inches()[1]
    assert _panel_box(free)[1] > _panel_box(page)[1]


def test_figsize_still_overrides_size_and_zoom():
    fig = field_row(_aligned(), title="t", size="slide", zoom=3.0, figsize=(7.0, 2.5))
    assert tuple(fig.get_size_inches()) == pytest.approx((7.0, 2.5))


def test_an_explicit_size_still_wins_over_the_automatic_one():
    """Automatic sizing is a better default, not a new constraint."""
    fig = field_row(_aligned(), title="t", title_kwargs={"fontsize": 17})
    assert _titles(fig) == [17.0] * len(_titles(fig))


def test_an_explicit_size_survives_an_overflow_that_would_shrink_it():
    """The override has to be absolute, including against the fitting pass.

    ``_fit_text_widths`` shrinks whatever overflows its box, and it used to do that to a
    size the caller had set: asking for 20pt with a long label got you 4.7pt on one
    panel and 20 on the others. That makes ``*_kwargs`` advisory rather than an
    override, so an explicitly chosen size is now exempt.
    """
    long = "sea_water_potential_temperature_at_sea_floor climatology"
    fig = field_row(
        _aligned(),
        labels=(long, "WOA"),
        units="degC",
        title="t",
        title_kwargs={"fontsize": 20},
    )
    fig.canvas.draw()
    assert set(_titles(fig)) == {20.0}


def test_every_font_role_can_be_overridden():
    """The whole point of the seven dicts: each names a size and each is honored."""
    fig = field_row(
        _aligned(),
        title="t",
        units="degC",
        metrics={"bias": 0.1, "rmse": 0.3, "corr": 0.9},
        title_kwargs={"fontsize": 13},
        suptitle_kwargs={"fontsize": 19},
        tick_label_kwargs={"size": 11},
        metrics_kwargs={"fontsize": 12},
        colorbar_kwargs={"label_size": 15, "tick_labelsize": 14},
    )
    fig.canvas.draw()
    assert set(_titles(fig)) == {13.0}
    assert fig._suptitle.get_fontsize() == 19.0
    bars = [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]
    assert bars
    for cax in bars:
        assert cax.xaxis.label.get_fontsize() == 15.0
        assert cax.get_xticklabels()[0].get_fontsize() == 14.0


def test_fit_text_can_be_turned_off_entirely():
    long = "sea_water_potential_temperature_at_sea_floor climatology"
    fitted = field_row(_aligned(), labels=(long, "WOA"), units="degC", title="t")
    left = field_row(
        _aligned(), labels=(long, "WOA"), units="degC", title="t", fit_text=False
    )
    assert max(_titles(left)) == min(_titles(left)), "nothing should have been resized"
    assert min(_titles(fitted)) < min(_titles(left))


def test_field_row_sizes_itself_to_the_domains_aspect_ratio():
    """``field_row`` used to ignore the aspect ratio that ``field_grid`` respected.

    Its figsize was the constant ``(8.5, 8.5/3.1)``, so a tall narrow domain — a coastal
    strip, a channel — was drawn into a short wide row and letterboxed, while the same
    data through ``field_grid`` was not. Both now go through the same rule.
    """
    wide = field_row(_aligned(lon1=290.0), title="t")  # lon span 30, lat span 8
    tall = field_row(_aligned(lon1=262.0), title="t")  # lon span 2
    assert tall.get_size_inches()[1] > wide.get_size_inches()[1]


def test_a_tall_grid_gets_smaller_panel_type_than_a_short_one():
    rows = [{"aligned": _aligned(), "units": "degC"} for _ in range(6)]
    few = field_grid(rows[:2], title="t")
    many = field_grid(rows, title="t")
    assert max(_titles(many)) < max(_titles(few))


def test_an_overlong_label_is_shrunk_to_fit_and_its_neighbours_are_not():
    """The backstop for what no global size can fix: one particular long string.

    The scale sizes text against the space available, which cannot account for a caller
    passing a 50-character CF standard name as a panel title. Measuring the drawn result
    and shrinking only what overflows leaves every label that does fit at the size the
    scale chose.

    The floor bounds how far this can go: a 56-character title over a 2.4in panel would
    need about 4pt to fit outright, and ``MIN_PT`` stops it at 6, so it comes down to
    the floor and still overhangs a little. That is the deliberate trade — a legible
    label slightly wider than its panel beats an illegible one inside it.
    """
    long = "sea_water_potential_temperature_at_sea_floor climatology"
    fig = field_row(_aligned(), labels=(long, "WOA"), units="degC", title="t")
    fig.canvas.draw()
    by_text = {
        ax.title.get_text(): (ax.title, ax) for ax in fig.axes if ax.title.get_text()
    }
    shrunk, _ = by_text[long]
    # its neighbours, compared within this figure rather than against the reference
    # geometry — this test's domain is a different aspect ratio, so its scale differs
    untouched = {by_text["difference"][0], by_text["WOA"][0]}
    kept = next(iter(untouched))
    sizes = {t.get_fontsize() for t in untouched}
    assert len(sizes) == 1, "a label that fits was resized anyway"
    assert shrunk.get_fontsize() < kept.get_fontsize(), "the long title was not shrunk"
    assert shrunk.get_fontsize() >= tg.MIN_PT, "shrunk past the legibility floor"


def test_a_label_that_fits_at_a_smaller_size_is_brought_inside_its_panel():
    """The case the backstop is actually for: fixable by shrinking, and fixed.

    Distinct from the test above, where the floor binds first. Here the label is long
    enough to overflow at the chosen level but short enough that a size above ``MIN_PT``
    fits, so it should end up inside its own panel.
    """
    fig = field_row(
        _aligned(), labels=("ROMS GOM hindcast v3", "WOA"), units="degC", title="t"
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not ax.title.get_text():
            continue
        assert (
            ax.title.get_window_extent(renderer).width
            <= ax.get_window_extent(renderer).width
        ), ax.title.get_text()


# --- the two renderers must agree ---------------------------------------------------


def test_bokeh_sizes_are_the_same_scale_in_bokeh_units():
    """Interactive is documented as the same plot; that has to include its type size.

    Bokeh font sizes are CSS points against a frame measured in CSS pixels, matplotlib's
    are 1/72 inch against a figure in inches. Without both conversions the same nominal
    number is a third too small in the browser, which is the two renderers disagreeing
    about a plot they claim to draw identically.
    """
    px = tg.frame_px(1.4)
    title_pt = float(tg.bokeh_fontsize(px)["title"].removesuffix("pt"))
    interactive_share = title_pt * tg.PT_PER_CSS_PX / px[0]

    static = tg.reference_scale()["title"]
    panel_in = tg.PAGE_W / 3 * tg.PANEL_W_FRACTION
    static_share = static / 72.0 / panel_in

    assert interactive_share == pytest.approx(static_share, rel=0.15)


def test_bokeh_scale_covers_every_role_and_fontsize_only_bokehs_own_keys():
    px = tg.frame_px(1.4)
    assert set(tg.bokeh_scale(px)) == set(tg.FONT_STEPS)
    assert set(tg.bokeh_fontsize(px)) <= {
        "title",
        "labels",
        "ticks",
        "clabel",
        "cticks",
        "legend",
    }
    assert all(v.endswith("pt") for v in tg.bokeh_fontsize(px).values())


def test_interactive_frames_follow_the_aspect_ratio_too():
    """A fixed frame letterboxed exactly the domains the static renderer fits."""
    wide, tall = tg.frame_px(3.0), tg.frame_px(0.5)
    assert wide[1] < tall[1]
    assert wide[0] == tall[0]  # width is what is pinned; height follows


def _bokeh_figures(obj):
    import holoviews as hv
    from bokeh.plotting import figure

    return list(hv.render(obj, backend="bokeh").select({"type": figure}))


def _interactive_row(**options):
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    item = {"aligned": _aligned(), "units": "degC"}
    spec = PlotSpec(family="field_row", items=[item], options=options)
    return _bokeh_figures(render(spec, renderer="holoviews"))[0]


#: Every family that draws maps, with a spec item shaped as its own renderer expects.
#: ``taylor``/``target``/``paired`` are excluded: they are metric diagrams, not maps,
#: and are square by construction rather than sized from a canvas.
def _map_family_specs():
    import numpy as np
    import xarray as xr

    def cube(n):
        base = _field()
        return xr.DataArray(
            np.stack([base.values + i * 0.1 for i in range(n)]),
            coords={"time": list(range(n)), "lat": base.lat, "lon": base.lon},
            dims=("time", "lat", "lon"),
        )

    row = {"aligned": _aligned(), "units": "degC"}
    return {
        "field_row": [row],
        "field_grid": [row, row],
        "field_facet": [{"field": cube(6), "facet_dim": "time", "units": "degC"}],
        "field_movie": [dict(row, frame_label=f"t{i}") for i in range(3)],
        "facet_movie": [{"field": cube(4), "facet_dim": "time", "units": "degC"}],
    }


def _drawn_width(family, items, **options):
    """Figure width (inches) for a static family; unwraps the movies' FuncAnimation."""
    import matplotlib.pyplot as plt

    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    out = render(
        PlotSpec(family=family, items=items, options=options), renderer="matplotlib"
    )
    fig = getattr(out, "_fig", out)  # the movie families return an animation
    width = float(fig.get_size_inches()[0])
    plt.close(fig)
    return width


@pytest.mark.parametrize("family", list(_map_family_specs()))
def test_size_and_zoom_reach_every_map_family(family):
    """The knobs have to land on all of them, not just the two they started on.

    ``size``/``zoom`` were added to ``field_row`` and ``field_grid`` and then extended
    family by family, which left ``field_facet``, ``field_movie`` and ``facet_movie``
    without them. A caller who had learnt ``zoom=1.4`` on a grid got a ``TypeError`` on
    a facet, and silence interactively, where those functions take ``**_``. All five now
    size from the same canvas.
    """
    items = _map_family_specs()[family]
    assert _drawn_width(family, items, zoom=1.5) > _drawn_width(family, items)
    assert _drawn_width(family, items, size="slide") == pytest.approx(
        tg.CANVASES["slide"].width
    )


@pytest.mark.parametrize("family", list(_map_family_specs()))
def test_size_and_zoom_reach_every_map_family_interactively(family):
    """Standing rule: a plot option lands in both renderers or in neither."""
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    def frame_width(**options):
        obj = render(
            PlotSpec(family=family, items=_map_family_specs()[family], options=options),
            renderer="holoviews",
        )
        # the movies come back wrapped in a slider: unwrap to the holoviews object
        for attr in ("object", "objects"):
            if not hasattr(obj, "traverse") and hasattr(obj, attr):
                inner = getattr(obj, attr)
                obj = inner[0] if isinstance(inner, list | tuple) and inner else inner
        if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
            try:  # a map of frames: any single frame carries the geometry
                obj = obj[next(iter(obj.keys()))]
            except Exception:
                pass
        return _bokeh_figures(obj)[0].frame_width

    assert frame_width(zoom=1.5) > frame_width()


def test_size_and_zoom_reach_the_interactive_renderer_too():
    """Standing rule: a plot option lands in both renderers or in neither.

    ``size``/``zoom`` are inches statically and CSS pixels interactively, so they cross
    as a ratio against the page rather than a conversion — but the same call still has
    to make the interactive plot bigger, or the two renderers are different plots.
    """
    plain = _interactive_row()
    big = _interactive_row(zoom=1.5)
    assert big.frame_width > plain.frame_width
    assert big.frame_height > plain.frame_height

    slide = _interactive_row(size="slide")
    assert slide.frame_width > plain.frame_width


def test_font_scale_reaches_the_interactive_renderer_too():
    plain = _interactive_row()
    bigger = _interactive_row(font_scale=1.5)
    as_pt = lambda f: float(f.title.text_font_size.removesuffix("pt"))  # noqa: E731
    assert as_pt(bigger) > 1.3 * as_pt(plain)


def test_fit_text_is_dropped_interactively_without_a_warning():
    """Bokeh lays out its own text, so the pass is satisfied by construction here."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _interactive_row(fit_text=False)


def test_colorbar_orientation_defers_to_the_family_except_at_the_extremes():
    """The aspect ratio only overrides where it is unambiguous.

    A bar wants the panel's longer edge, but the two orientations cost different things
    (horizontal is charged per row, vertical takes its width once), and aspect alone
    cannot weigh that — so between the limits each family keeps its own default. Pinning
    it here because my first attempt used a single 1.5 threshold, which flipped a
    page-width GoM row (aspect 1.385) to vertical bars and lost half its panel area.
    """
    horizontal = tg.colorbar_is_horizontal
    # a GoM-ish box sits between the limits: each family keeps its default
    assert horizontal(1.385, default_horizontal=True) is True
    assert horizontal(1.385, default_horizontal=False) is False
    # far enough either way and the shape decides regardless
    assert horizontal(3.5, default_horizontal=False) is True
    assert horizontal(0.4, default_horizontal=True) is False
    # and an explicit request always wins, whatever the shape
    assert horizontal(0.4, default_horizontal=False, requested="horizontal") is True
    assert horizontal(3.5, default_horizontal=True, requested="vertical") is False


def test_an_overridden_orientation_also_resizes_the_figure():
    """The override has to reach the *sizing*, not just the drawing.

    Horizontal bars come out of a row's height and vertical ones out of its width, so a
    figure sized for one and drawn with the other loses the difference. Asking a grid
    for horizontal bars used to cost 37% of its panel, with no warning.
    """
    rows = [{"aligned": _aligned(), "units": "degC"} for _ in range(3)]
    beside = field_grid(rows, title="t")
    below = field_grid(rows, title="t", colorbar_kwargs={"orientation": "horizontal"})
    # the figure grew to hold the bars rather than the panels shrinking to make room
    assert below.get_size_inches()[1] > beside.get_size_inches()[1]
    assert _panel_box(below)[0] >= _panel_box(beside)[0] * 0.98


def test_a_top_level_option_is_never_reported_as_a_nested_key():
    """``size`` is both a plot option and ``tick_label_kwargs``' font-size key.

    The "did you mean to put this inside ``*_kwargs``?" helper matched on key name
    alone, so it told callers that ``size="slide"`` belonged inside
    ``tick_label_kwargs``. Any top-level parameter sharing a nested key's name hits it.
    """
    from ocean_skill.plot.matplotlib_renderer import _nested_owner

    for option in ("size", "zoom", "fit_text", "font_scale", "figsize"):
        assert _nested_owner(option) is None, option
    # and the redirection still works for keys that really are nested
    assert _nested_owner("label_size") == "colorbar_kwargs"
    assert _nested_owner("shrink") == "colorbar_kwargs"

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _interactive_row(size="slide", zoom=1.2)


# --- the summary diagrams -----------------------------------------------------------


class _Rec:
    """A metric record shaped as the summary diagrams expect."""

    def __init__(self, label="run"):
        self.label = label

    def metrics(self):
        return {
            "bias": 0.1,
            "rmse": 0.4,
            "corr": 0.9,
            "crmsd": 0.3,
            "std_test": 1.1,
            "std_reference": 1.0,
        }


def test_a_fixed_proportion_diagram_fits_inside_a_canvas_rather_than_filling_it():
    """A square cannot fill 16:9, so ``size`` means *fit inside* for these.

    ``size="page"`` must come out at exactly the reviewed default, and a capped canvas
    must bind on height as well as width. Scaling by the width ratio alone (13.33/8.5 =
    1.57x) is what the interactive Target did, and it only avoided overflowing a 7.5in
    slide by luck.
    """
    default = (5.0, 5.0)
    assert tg.diagram_scale_factor(default, size="page") == pytest.approx(1.0)
    assert tg.diagram_scale_factor(default, zoom=1.5) == pytest.approx(1.5)

    slide = tg.CANVASES["slide"]
    factor = tg.diagram_scale_factor(default, size="slide")
    assert factor < slide.width / tg.PAGE_W, "the height must bind, not just the width"
    assert default[1] * factor <= slide.max_height - tg.SUPTITLE_ALLOWANCE + 1e-9


@pytest.mark.parametrize("diagram", ["taylor", "target", "paired"])
def test_the_summary_diagrams_take_size_and_zoom_in_both_renderers(diagram):
    """The asymmetry was mine: the interactive Target took them, the static one did not.

    Same call, different acceptance by renderer, which is the thing the both-renderers
    rule exists to prevent. ``paired``/``taylor`` are matplotlib-only by design (bokeh
    has no floating polar axis), so there "both renderers" means the delegation path.
    """
    from ocean_skill.plot import summary

    class Rec:
        label = "run"

        def metrics(self):
            return dict(
                bias=0.1, rmse=0.4, corr=0.9, crmsd=0.3, std_test=1.1, std_reference=1.0
            )

    fn = getattr(summary, diagram)
    plain = fn([Rec()], title="T")
    bigger = fn([Rec()], title="T", zoom=1.5)
    assert bigger.get_size_inches()[0] == pytest.approx(
        1.5 * plain.get_size_inches()[0]
    )
    # and a slide's height caps it rather than the width running away with it
    slide = fn([Rec()], title="T", size="slide")
    assert slide.get_size_inches()[1] <= (
        tg.CANVASES["slide"].max_height - tg.SUPTITLE_ALLOWANCE + 1e-9
    )


def test_summary_scale_accepts_a_partial_override():
    """``scale=`` is these diagrams' only per-role override, so it must take one role.

    They have no ``*_kwargs`` dicts, and this used to replace the computed scale
    outright rather than merge onto it — so naming one role raised ``KeyError`` on the
    first role you had not named, which made the parameter unusable for its only
    purpose.
    """
    from ocean_skill.plot.summary import paired, target, taylor

    for fn in (taylor, target, paired):
        fig = fn([_Rec()], title="T", scale={"title": 20.0})
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles, fn.__name__
        sizes = {ax.title.get_fontsize() for ax in fig.axes if ax.get_title()}
        assert 20.0 in sizes, fn.__name__


def test_the_summary_key_clears_the_axis_labels_at_any_level():
    """The key used to sit at a fixed offset while the labels it clears did not.

    ``y=-0.04`` in figure fractions against x labels of a fixed *height*: raising the
    level walked the key up into them. Same bug, same fix, as the row label that used to
    sit at a constant ``x=-0.18``.
    """
    from ocean_skill.plot.summary import paired

    for font_scale in (1.0, 1.6):
        fig = paired([_Rec("a"), _Rec("b")], title="T", font_scale=font_scale)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        key = fig.legends[0].get_window_extent(renderer)
        for ax in fig.axes:
            label = ax.xaxis.label
            if not label.get_text():
                continue
            assert key.y1 <= label.get_window_extent(renderer).y0 + 1.0, font_scale


def test_a_four_metric_grid_is_laid_out_by_the_domains_own_shape():
    """The ``skill_map`` family rests on this, and on the sizing that follows from it.

    Four metric panels have no inherent order, so nothing about the fold carries
    meaning and the aspect ratio decides it — a wide Gulf box stacks down the page one
    panel per row, a tall California box spreads across. Pinned because a hardcoded 2x2
    would halve the panel width for the wide case, and the difference is invisible in a
    figure that still draws every panel.
    """
    assert tg.facet_layout(4, 4.0) == (1, 4), "a wide domain stacks"
    assert tg.facet_layout(4, 1.4) == (2, 2), "a squarish one goes 2x2"
    assert tg.facet_layout(4, 0.35) == (4, 1), "a tall one spreads"


def test_facet_layout_folds_to_the_canvas_it_will_be_drawn_on():
    """The canvas has to be load-bearing here, not just a figure size.

    ``facet_layout`` scores each candidate grid by how close its *cell* is to the shape
    the map wants, and the cell shape depends on the canvas. Scored against a portrait
    page a sequence of wide maps stacks into a column — right for a report, and exactly
    wrong for a slide, where the same panels want to spread across 16:9. Before the
    canvas reached this function every family folded as if it were going on a page.
    """
    page, slide = tg.CANVASES["page"], tg.CANVASES["slide"]
    # six squarish panels: 2 across on a page, 3 across on a slide
    assert tg.facet_layout(6, 1.385, canvas=page) == (2, 3)
    assert tg.facet_layout(6, 1.385, canvas=slide) == (3, 2)
    # a wide domain stacks on a page and pairs up on a slide
    assert tg.facet_layout(6, 3.0, canvas=page) == (1, 6)
    assert tg.facet_layout(6, 3.0, canvas=slide) == (2, 3)
    # a narrow canvas has room for only one column
    assert tg.facet_layout(6, 1.385, canvas=tg.CANVASES["column"]) == (1, 6)


def test_an_uncapped_canvas_still_folds_but_never_squeezes():
    """``size="free"`` has no height to match, so the page's is used for *scoring* only.

    The figure itself is then free to grow past the page — which is the whole point of
    an uncapped canvas, and what ``facet_figsize`` previously could not express because
    its caller spelled the height as ``canvas.max_height or PAGE_H``.
    """
    free, page = tg.CANVASES["free"], tg.CANVASES["page"]
    assert tg.facet_layout(9, 1.385, canvas=free) == tg.facet_layout(
        9, 1.385, canvas=page
    )

    tall = tg.facet_figsize(0.5, nrows=6, ncols=1, canvas=free)[1]
    capped = tg.facet_figsize(0.5, nrows=6, ncols=1, canvas=page)[1]
    assert capped <= tg.PAGE_H - tg.SUPTITLE_ALLOWANCE
    assert tall > capped, "an uncapped canvas should grow rather than squeeze"


def test_zoom_scales_a_facet_figure():
    plain = tg.facet_figsize(1.385, nrows=3, ncols=3)
    big = tg.facet_figsize(1.385, nrows=3, ncols=3, canvas=tg.resolve_canvas(zoom=1.5))
    assert big[0] == pytest.approx(1.5 * plain[0])
    assert big[1] > plain[1]


def test_a_grid_with_a_bar_in_every_cell_still_fits_the_page():
    """``skill_map`` charges ``PANEL_W_FRACTION``, not the shared-bar allowance.

    Every panel there carries its own colorbar, so it keeps the 0.72 of its cell that a
    ``field_grid`` row does rather than the 0.88 a facet grid's shared bar leaves. This
    checks the resulting figure is still inside the page it has to fit.
    """
    width, height = tg.facet_figsize(
        4.0, nrows=4, ncols=1, panel_w_fraction=tg.PANEL_W_FRACTION
    )
    assert width == tg.PAGE_W
    assert height <= tg.PAGE_H - tg.SUPTITLE_ALLOWANCE
    # and the maps get materially more height than the shared-bar allowance implies
    shared = tg.facet_figsize(
        4.0, nrows=4, ncols=1, panel_w_fraction=tg.FACET_PANEL_W_FRACTION
    )[1]
    assert shared > height, "0.88 of the cell is a taller figure, not a wider map"


# -- the series family -----------------------------------------------------------------


def test_row_height_overhead_override_is_a_no_op_by_default():
    """The map families must be untouched by the parameter the series family needs."""
    from ocean_skill.plot.typography import ROW_OVERHEAD, row_height

    plain = row_height(2.0, nrows=2)
    explicit = row_height(2.0, nrows=2, overhead=ROW_OVERHEAD)
    assert plain == pytest.approx(explicit)


def test_a_series_figure_stays_inside_the_page_cap():
    from ocean_skill.plot.typography import (
        PAGE_H,
        SERIES_ASPECT,
        SERIES_OVERHEAD,
        SERIES_PANEL_W_FRACTION,
        auto_figsize,
        resolve_canvas,
    )

    _, height = auto_figsize(
        SERIES_ASPECT,
        nrows=6,
        ncols=1,
        canvas=resolve_canvas("page"),
        panel_w_fraction=SERIES_PANEL_W_FRACTION,
        overhead=SERIES_OVERHEAD,
    )
    assert height <= PAGE_H


def test_a_one_column_series_grid_does_not_get_a_giant_suptitle():
    """``figure_ncols`` pins the suptitle to the reference grid, as field_facet does.

    Without it, a one-column figure asks for the *figure* base off its own cell — three
    times as wide as a cell of the three-map row everything is calibrated against — and
    gets a suptitle twice the size of every other figure in the same report.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from ocean_skill.plot.matplotlib_renderer import series
    from ocean_skill.plot.typography import reference_scale

    time = pd.date_range("2015-01-01", periods=24, freq="MS")
    da = xr.DataArray(
        np.arange(24.0), coords={"time": time}, dims="time", attrs={"units": "degC"}
    ).assign_coords(lon=-144.0, lat=50.0)
    aligned = xr.Dataset({"reference": da, "test": da + 1, "difference": da * 0 + 1})
    item = {
        "aligned": aligned,
        "metrics": {"bias": 1.0},
        "units": "degC",
        "standard_name": "sea_water_temperature",
        "labels": ("model", "obs"),
    }
    fig = series([item], title="a title")
    drawn = fig._suptitle.get_fontsize()
    assert drawn == pytest.approx(reference_scale()["suptitle"], abs=1.5)
