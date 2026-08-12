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

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot import typography as tg
from ocean_skill.plot.matplotlib_renderer import field_grid, field_row

PANEL_ROLES = [r for r, (base, _) in tg.FONT_STEPS.items() if base == "panel"]
FIGURE_ROLES = [r for r, (base, _) in tg.FONT_STEPS.items() if base == "figure"]


# --- the scale itself --------------------------------------------------------------


def test_default_row_reproduces_the_hand_tuned_sizes():
    """The calibration promise: existing figures keep looking as they did.

    These are the sizes that were hand-tuned as literals in the renderers before the
    scale existed (title 8, suptitle 9, colorbar label 7, row label 7, latitude labels
    5, metrics 5.5). The scale has to land on them at the figure size they were tuned
    for, or adopting it silently restyles every notebook in the wild — which is a
    different change from the one being made.
    """
    scale = tg.reference_scale()
    for role, tuned in {
        "title": 8.0,
        "suptitle": 9.0,
        "colorbar_label": 7.0,
        "row_label": 7.0,
        "tick_label": 5.0,
        "metrics": 5.5,
    }.items():
        assert scale[role] == pytest.approx(tuned, abs=0.75), role


def test_panel_type_shrinks_with_the_cell_and_grows_with_it():
    """Monotone in cell size, which is the whole point of deriving it from geometry."""
    small = tg.type_scale((4.0, 1.3), ncols=3, nrows=1)
    default = tg.reference_scale()
    big = tg.type_scale((14.0, 4.5), ncols=3, nrows=1)
    for role in PANEL_ROLES:
        assert small[role] < default[role] < big[role], role


def test_type_is_sublinear_in_the_cell_size():
    """Doubling the figure must not double the type.

    A linear rule keeps the type a constant *share* of the panel, which is wrong at the
    large end: matplotlib's own 12pt default on a 6.4x4.8in figure is far below what a
    linear extrapolation from a dense page-width row would ask for. The exponent is what
    reconciles the two anchors, so this guards it against being quietly linearised.
    """
    one = tg.type_scale((8.5, 2.74), ncols=3, nrows=1)["title"]
    two = tg.type_scale((17.0, 5.48), ncols=3, nrows=1)["title"]
    assert one < two < 2 * one


def test_matplotlibs_own_default_figure_lands_on_matplotlibs_own_default_title():
    """The second calibration anchor: a standalone figure at mpl's default size.

    The curve is fitted through this point and the dense-row one, so if the coefficient
    or exponent drifts, one of the two ends stops agreeing with the convention it was
    matched to.
    """
    assert tg.type_scale((6.4, 4.8))["title"] == pytest.approx(12.0, abs=1.0)


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


def test_row_height_is_bounded_by_the_page():
    """Eight rows still have to print, which is why the cap exists at all."""
    assert tg.row_height(1.0, nrows=8) * 8 <= tg.PAGE_H


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


def test_a_small_figure_still_contains_maps():
    """The failure that motivated all of this, as a drawn figure.

    matplotlib points are absolute and ``constrained_layout`` gives text priority over
    axes, so fixed 8pt titles on a 4-inch row squeezed the three maps to about a third
    of an inch each — the text did not look wrong, it *ate the figure*. Sizing the type
    to the geometry is what keeps a small figure merely small.
    """
    fig = field_row(_aligned(), figsize=(4.0, 4.0 / 3.1), title="t")
    panel_w, _ = _panel_box(fig)
    cell_w = 4.0 / 3
    assert panel_w > 0.4 * cell_w, "the maps have been crowded out by their own labels"


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


def test_an_explicit_size_still_wins_over_the_automatic_one():
    """Automatic sizing is a better default, not a new constraint."""
    fig = field_row(_aligned(), title="t", title_kwargs={"fontsize": 17})
    assert _titles(fig) == [17.0] * len(_titles(fig))


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
    """
    long = "sea_water_potential_temperature_at_sea_floor climatology"
    fig = field_row(_aligned(), labels=(long, "WOA"), units="degC", title="t")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    by_text = {
        ax.title.get_text(): (ax.title, ax) for ax in fig.axes if ax.title.get_text()
    }
    shrunk, shrunk_ax = by_text[long]
    kept, _ = by_text["difference"]
    assert shrunk.get_fontsize() < kept.get_fontsize()
    assert kept.get_fontsize() == pytest.approx(tg.reference_scale()["title"], abs=1.0)
    assert (
        shrunk.get_window_extent(renderer).width
        <= shrunk_ax.get_window_extent(renderer).width
    )


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
