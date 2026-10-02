"""Static section figures: round filled bands, and labelled contour lines over them.

A paper-style section is a smooth ``contourf`` fill of one variable (phosphate) with
black, labelled contour lines of a second (temperature isotherms) on top. The static
renderer draws it from the shared decisions in :mod:`ocean_skill.plot.section` -- which
bands (:func:`fill_edges`), which lines (:func:`contour_levels`, once per figure) and
whether the overlay sits on the fill's mesh (:func:`prepare_overlay`) -- and these tests
check that the four section families (``section``, ``section_grid``, ``section_row``,
``section_row_grid``) act on them: the bands a figure really has, the lines each panel
really draws, and the refusals. Everything runs through ``render(PlotSpec(...))`` on
small synthetic sections.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.contour import ContourSet

from ocean_skill.align import ALONG_DIM
from ocean_skill.plot import matplotlib_renderer as mr
from ocean_skill.plot.registry import render
from ocean_skill.plot.section import (
    CONTOUR_WIDTH,
    DEFAULT_FILL_BANDS,
    contour_label,
    contour_levels,
    fill_edges,
    prepare_section,
)
from ocean_skill.plot.spec import PlotSpec

MINUS = "\N{MINUS SIGN}"


# -- fixtures -----------------------------------------------------------------------


def _raw(
    *, top: float, bottom: float, tilt: float = 0.0, n_z: int = 8, n_along: int = 14
) -> xr.DataArray:
    """Build a raw fixed-depth section running ``top`` to ``bottom``, tilted along-path.

    Linear in depth and in distance, so every contour line is one long straight run
    across the panel -- a line a label always fits on.
    """
    z = -np.linspace(0.0, 700.0, n_z)
    along = np.linspace(0.0, 210.0, n_along)
    down = np.linspace(0.0, 1.0, n_z)[:, None]
    across = np.linspace(0.0, 1.0, n_along)[None, :]
    da = xr.DataArray(
        top + (bottom - top) * down + tilt * across,
        dims=("z", ALONG_DIM),
        coords={
            "z": z,
            ALONG_DIM: along,
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, n_along)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, n_along)),
        },
    )
    da[ALONG_DIM].attrs["units"] = "km"
    return da


def _phosphate(tilt: float = 0.3, **grid) -> xr.DataArray:
    """Build the fill: 0.2 at the surface, 3 at the bottom."""
    return _raw(top=0.2, bottom=3.0, tilt=tilt, **grid)


def _temperature(shift: float = 0.0, **grid) -> xr.DataArray:
    """Build the lines: 26 at the surface down to 4, so 5 to 25 all cross."""
    return _raw(top=26.0, bottom=4.0, tilt=-2.0, **grid) + shift


def _section_item(field=None, contour=None, **extra) -> dict:
    item = {
        "field": _phosphate() if field is None else field,
        "units": "mmol m-3",
        "standard_name": None,
        "depth": None,
        "label": "roms",
        **extra,
    }
    if contour is not None:
        item["contour"] = contour
    return item


def _row_item(contour=None, **extra) -> dict:
    test, reference = _phosphate(0.3), _phosphate(-0.3)
    item = {
        "aligned": {
            "test": test,
            "reference": reference,
            "difference": test - reference,
        },
        "units": "mmol m-3",
        "standard_name": None,
        "depth": None,
        "time": None,
        "metrics": {"bias": 0.125, "rmse": 0.5, "corr": 0.98},
        "labels": ("model", "obs"),
        **extra,
    }
    if contour is not None:
        item["contour"] = contour
    return item


def _row_contour(shift: float = 0.0) -> dict:
    """Build a comparison row's overlay: a different temperature on each lane."""
    return {
        "test": _temperature(shift),
        "reference": _temperature(shift + 3.0),
    }


def _draw(family: str, items: list[dict], **options):
    return render(PlotSpec(family=family, items=items, options=options))


def _panels(fig) -> list:
    """Every panel axes of ``fig`` (no colour bars), in drawing order."""
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def _bars(fig) -> list:
    return [ax._colorbar for ax in fig.axes if ax.get_label() == "<colorbar>"]


def _filled(ax) -> list[ContourSet]:
    return [c for c in ax.collections if isinstance(c, ContourSet) and c.filled]


def _lines(ax) -> list[ContourSet]:
    return [c for c in ax.collections if isinstance(c, ContourSet) and not c.filled]


def _label_texts(lines: ContourSet) -> list[str]:
    return [t.get_text() for t in lines.labelTexts]


def _tick_values(bar) -> list[float]:
    """Return the numbers a bar's tick labels read (drawn first: they settle then)."""
    bar.ax.figure.canvas.draw()
    axis = bar.ax.xaxis if bar.orientation == "horizontal" else bar.ax.yaxis
    texts = [t.get_text() for t in axis.get_ticklabels()]
    return [float(text.replace(MINUS, "-")) for text in texts if text]


def _on_edges(values, edges) -> bool:
    return all(np.isclose(edges, v).any() for v in values)


#: The four section families, each as ``(family, items)``: a single section, a stacked
#: pair of them, a single comparison row, and a stacked pair of rows. Several items are
#: another code path in each renderer (a grid function of its own).
def _four_families(*, overlay: bool = False) -> dict:
    over = _temperature() if overlay else None
    row_over = _row_contour() if overlay else None
    return {
        "section": ("section", [_section_item(contour=over)]),
        "section[stacked]": (
            "section",
            [_section_item(contour=over), _section_item(contour=over, label="other")],
        ),
        "section_row": ("section_row", [_row_item(row_over)]),
        "section_row[stacked]": (
            "section_row",
            [_row_item(row_over, row_label="A"), _row_item(row_over, row_label="B")],
        ),
    }


FAMILIES = sorted(_four_families())


# -- the bands of a contourf fill ---------------------------------------------------


def test_a_contourf_section_is_cut_at_fill_edges_of_its_colour_range():
    fig = _draw("section", [_section_item()], mark="contourf")
    (ax,) = _panels(fig)
    (fill,) = _filled(ax)
    (bar,) = _bars(fig)
    expected = fill_edges(bar.norm.vmin, bar.norm.vmax, log=False)
    np.testing.assert_allclose(fill.levels, expected, rtol=0, atol=1e-12)
    # about the default target, not the 21 edges linspace used to give
    assert abs((len(expected) - 1) - DEFAULT_FILL_BANDS) <= DEFAULT_FILL_BANDS / 2
    assert len(fill.levels) > 21


def test_fill_levels_none_is_the_default_bands():
    default = _draw("section", [_section_item()], mark="contourf")
    spelled = _draw("section", [_section_item()], mark="contourf", fill_levels=None)
    np.testing.assert_array_equal(
        _filled(_panels(default)[0])[0].levels, _filled(_panels(spelled)[0])[0].levels
    )


@pytest.mark.parametrize("n", [5, 12, 100])
def test_an_int_fill_levels_asks_for_about_that_many_bands(n):
    fig = _draw("section", [_section_item()], mark="contourf", fill_levels=n)
    (fill,) = _filled(_panels(fig)[0])
    (bar,) = _bars(fig)
    expected = fill_edges(bar.norm.vmin, bar.norm.vmax, log=False, fill_levels=n)
    np.testing.assert_allclose(fill.levels, expected, rtol=0, atol=1e-12)
    default = _draw("section", [_section_item()], mark="contourf")
    n_default = len(_filled(_panels(default)[0])[0].levels)
    # fewer bands than the default when asked for fewer, more when asked for more
    assert (len(fill.levels) < n_default) == (n < DEFAULT_FILL_BANDS)


def test_a_list_fill_levels_is_exactly_those_edges():
    edges = [0.0, 0.5, 1.0, 2.0, 3.5]
    fig = _draw("section", [_section_item()], mark="contourf", fill_levels=edges)
    (fill,) = _filled(_panels(fig)[0])
    np.testing.assert_array_equal(fill.levels, edges)


@pytest.mark.parametrize(
    ("lo", "hi"), [(0.0213, 2.987), (-1.7, 1.7), (12.0, 31.0)], ids=str
)
@pytest.mark.parametrize("fill_levels", [None, 8, 100])
def test_every_colorbar_tick_sits_on_a_band_edge(lo, hi, fill_levels):
    field = _raw(top=lo, bottom=hi, tilt=0.1 * (hi - lo))
    fig = _draw(
        "section",
        [_section_item(field)],
        mark="contourf",
        fill_levels=fill_levels,
    )
    (fill,) = _filled(_panels(fig)[0])
    (bar,) = _bars(fig)
    ticks = _tick_values(bar)
    assert len(ticks) >= 3
    assert _on_edges(ticks, fill.levels), (ticks, list(fill.levels))


def test_a_stacked_grid_bands_every_panel():
    items = [_section_item(), _section_item(label="other")]
    fig = _draw("section", items, mark="contourf", fill_levels=10)
    panels = _panels(fig)
    assert len(panels) == 2
    (bar,) = _bars(fig)  # one shared bar: the two are the same variable
    expected = fill_edges(bar.norm.vmin, bar.norm.vmax, log=False, fill_levels=10)
    for ax in panels:
        (fill,) = _filled(ax)
        np.testing.assert_allclose(fill.levels, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("family", ["section_row", "section_row[stacked]"])
def test_the_difference_panel_is_filled_too(family):
    fam, items = _four_families()[family]
    fig = _draw(fam, items, mark="contourf", fill_levels=12)
    panels, bars = _panels(fig), _bars(fig)
    assert len(panels) == 3 * len(items) and len(bars) == 2 * len(items)
    for row in range(len(items)):
        test, ref, diff = panels[3 * row : 3 * row + 3]
        seq_bar, diff_bar = bars[2 * row : 2 * row + 2]
        for ax, bar in ((test, seq_bar), (ref, seq_bar), (diff, diff_bar)):
            (fill,) = _filled(ax)
            expected = fill_edges(
                bar.norm.vmin, bar.norm.vmax, log=False, fill_levels=12
            )
            np.testing.assert_allclose(fill.levels, expected, rtol=0, atol=1e-12)
        # the diverging scale is symmetric, so its edges are too
        np.testing.assert_allclose(
            _filled(diff)[0].levels, -_filled(diff)[0].levels[::-1], atol=1e-12
        )
        assert _on_edges(_tick_values(diff_bar), _filled(diff)[0].levels)


def test_map_and_movie_bands_keep_a_target_of_21_but_are_round():
    norm = mcolors.Normalize(vmin=0.0, vmax=3.0)
    kw = mr._contour_kw(norm)
    np.testing.assert_array_equal(
        kw["levels"], fill_edges(0.0, 3.0, log=False, fill_levels=21)
    )
    assert kw["extend"] == "neither"
    # round: every edge a multiple of 0.05, which 21 evenly spaced edges are not
    assert np.allclose(kw["levels"] / 0.05, np.round(kw["levels"] / 0.05))
    # a log scale stays geometric
    log = mcolors.LogNorm(vmin=0.1, vmax=10.0)
    np.testing.assert_allclose(
        mr._contour_levels(log), fill_edges(0.1, 10.0, log=True, fill_levels=21)
    )
    np.testing.assert_allclose(mr._contour_levels(log), np.geomspace(0.1, 10.0, 22))


# -- the lines over the fill --------------------------------------------------------


def test_a_section_overlay_is_one_set_of_lines_at_the_expected_levels():
    overlay = _temperature()
    fig = _draw("section", [_section_item(contour=overlay)], mark="contourf")
    (ax,) = _panels(fig)
    (lines,) = _lines(ax)
    expected = contour_levels(True, [prepare_section(overlay)[0]])
    assert expected == (5.0, 10.0, 15.0, 20.0, 25.0)
    assert tuple(lines.levels) == expected
    # one label per line, spelled as contour_label spells them (and the same text
    # objects are the axes' own, so a saved figure carries them)
    assert sorted(_label_texts(lines)) == sorted(contour_label(v) for v in expected)
    assert {t.get_text() for t in ax.texts} == {contour_label(v) for v in expected}
    assert all(t.get_visible() for t in lines.labelTexts)


def test_the_lines_are_the_overlays_own_not_the_fills():
    overlay = _temperature()
    # labels=False leaves each line whole (inline labels cut a gap out of it), so its
    # vertices can be compared to contouring the overlay on the panel's mesh by hand
    fig = _draw(
        "section",
        [_section_item(contour=overlay)],
        contour_kwargs={"labels": False},
    )
    (lines,) = _lines(_panels(fig)[0])
    prepared, geometry = prepare_section(overlay)
    fig_ref, ax_ref = plt.subplots()
    try:
        by_hand = ax_ref.contour(
            prepared[geometry.x_name],
            prepared[geometry.y_name],
            np.ma.masked_invalid(prepared),
            levels=list(lines.levels),
        )
        assert len(lines.allsegs) == len(by_hand.allsegs) == 5
        for ours, theirs in zip(lines.allsegs, by_hand.allsegs, strict=True):
            assert len(ours) == len(theirs) >= 1
            for a, b in zip(ours, theirs, strict=True):
                np.testing.assert_allclose(a, b)
    finally:
        plt.close(fig_ref)


def test_the_lines_are_black_thin_solid_and_above_the_fill():
    for mark in ("contourf", "pcolormesh"):
        fig = _draw("section", [_section_item(contour=_temperature())], mark=mark)
        (ax,) = _panels(fig)
        (lines,) = _lines(ax)
        assert np.allclose(lines.get_edgecolor(), [0.0, 0.0, 0.0, 1.0])
        assert set(lines.get_linewidth()) == {CONTOUR_WIDTH}
        assert {tuple(style) for style in lines.get_linestyle()} == {(0.0, None)}
        fills = [c for c in ax.collections if c is not lines]
        assert fills and all(lines.get_zorder() > c.get_zorder() for c in fills)
        assert all(t.get_zorder() > lines.get_zorder() for t in lines.labelTexts)


def test_a_negative_isotherm_is_solid_and_labelled_with_a_true_minus():
    cold = _raw(top=5.0, bottom=-3.0, tilt=0.0)
    fig = _draw("section", [_section_item(contour=cold)], mark="contourf")
    (lines,) = _lines(_panels(fig)[0])
    assert min(lines.levels) < 0 < max(lines.levels)
    # matplotlib alone would dash the negative levels of a one-colour set
    assert {tuple(style) for style in lines.get_linestyle()} == {(0.0, None)}
    labels = _label_texts(lines)
    assert any(text.startswith(MINUS) for text in labels)
    assert not any("-" in text for text in labels)


def test_the_y_axis_is_inverted_once_with_or_without_lines():
    for overlay in (None, _temperature()):
        fig = _draw("section", [_section_item(contour=overlay)], mark="contourf")
        (ax,) = _panels(fig)
        lo, hi = ax.get_ylim()
        assert ax.yaxis_inverted() and lo > hi
        assert (lo, hi) == pytest.approx((700.0, 0.0))


def test_contour_levels_say_which_lines():
    base = {"mark": "contourf"}
    item = _section_item(contour=_temperature())

    def drawn(**options):
        fig = _draw("section", [item], **base, **options)
        return _lines(_panels(fig)[0])

    (explicit,) = drawn(contour_levels=[10.0, 20.0])
    assert tuple(explicit.levels) == (10.0, 20.0)
    (count,) = drawn(contour_levels=2)
    assert 1 <= len(count.levels) <= 2
    (default,) = drawn()
    (spelled,) = drawn(contour_levels=True)
    five = (5.0, 10.0, 15.0, 20.0, 25.0)
    assert tuple(default.levels) == tuple(spelled.levels) == five
    assert drawn(contour_levels=False) == []
    assert drawn(contour_levels=[]) == []


def test_lines_over_a_pcolormesh_fill_work_too():
    fig = _draw("section", [_section_item(contour=_temperature())])
    (ax,) = _panels(fig)
    (lines,) = _lines(ax)
    assert tuple(lines.levels) == (5.0, 10.0, 15.0, 20.0, 25.0)
    assert any(type(c).__name__ == "QuadMesh" for c in ax.collections)
    assert _label_texts(lines)


def test_a_section_row_draws_lines_on_the_test_and_reference_panels_only():
    contour = _row_contour()
    fig = _draw("section_row", [_row_item(contour)], mark="contourf")
    test, reference, difference = _panels(fig)
    pooled = contour_levels(
        True, [prepare_section(contour[lane])[0] for lane in ("test", "reference")]
    )
    (lines_t,) = _lines(test)
    (lines_r,) = _lines(reference)
    assert tuple(lines_t.levels) == tuple(lines_r.levels) == pooled
    # nothing -- no lines, no label -- on the difference panel
    assert _lines(difference) == []
    assert [t for t in difference.texts if t is not difference._osk_metrics_text] == []
    # each lane's lines come from its own overlay: the reference is 3 degC warmer, so
    # a level sits lower in the water column there
    depth_of = {
        ax: {
            level: np.concatenate([np.asarray(s)[:, 1] for s in segs]).mean()
            for level, segs in zip(lines.levels, lines.allsegs, strict=True)
            if len(segs)
        }
        for ax, lines in ((test, lines_t), (reference, lines_r))
    }
    common = set(depth_of[test]) & set(depth_of[reference])
    assert common and all(depth_of[reference][v] > depth_of[test][v] for v in common)
    # every panel, the difference one included, is filled
    assert all(len(_filled(ax)) == 1 for ax in (test, reference, difference))


def test_a_stacked_section_grid_shares_one_set_of_levels_across_panels():
    cool, warm = _temperature(), _temperature(shift=10.0)
    items = [_section_item(contour=cool), _section_item(contour=warm, label="b")]
    fig = _draw("section", items, mark="contourf")
    first, second = _panels(fig)
    (lines_a,), (lines_b,) = _lines(first), _lines(second)
    pooled = contour_levels(True, [prepare_section(o)[0] for o in (cool, warm)])
    alone = contour_levels(True, [prepare_section(cool)[0]])
    assert pooled != alone  # the range is wider together, so the lines differ
    assert tuple(lines_a.levels) == tuple(lines_b.levels) == pooled
    # a level that crosses a panel's own range is labelled there, and only there
    assert 10.0 in pooled and 30.0 in pooled
    assert set(_label_texts(lines_a)) == {"10", "20"}
    assert set(_label_texts(lines_b)) == {"20", "30"}


def test_a_stacked_section_row_grid_shares_one_set_of_levels_across_rows():
    items = [
        _row_item(_row_contour(0.0), row_label="A"),
        _row_item(_row_contour(10.0), row_label="B"),
    ]
    fig = _draw("section_row", items, mark="contourf")
    panels = _panels(fig)
    assert len(panels) == 6
    arrays = [
        prepare_section(raw)[0] for item in items for raw in item["contour"].values()
    ]
    pooled = contour_levels(True, arrays)
    first_row_alone = contour_levels(
        True, [prepare_section(raw)[0] for raw in items[0]["contour"].values()]
    )
    assert pooled != first_row_alone
    for row in range(2):
        test, reference, difference = panels[3 * row : 3 * row + 3]
        assert tuple(_lines(test)[0].levels) == pooled
        assert tuple(_lines(reference)[0].levels) == pooled
        assert _lines(difference) == []


def test_an_item_without_an_overlay_just_has_no_lines_in_a_mixed_stack():
    items = [_section_item(contour=_temperature()), _section_item(label="b")]
    fig = _draw("section", items, mark="contourf")
    first, second = _panels(fig)
    assert len(_lines(first)) == 1
    assert _lines(second) == [] and not second.texts


def test_every_family_inverts_every_panel_once_with_lines():
    for key, (family, items) in _four_families(overlay=True).items():
        fig = _draw(family, items, mark="contourf")
        for ax in _panels(fig):
            lo, hi = ax.get_ylim()
            assert ax.yaxis_inverted() and lo > hi, key


def test_labels_go_on_even_when_nothing_else_has_drawn_the_figure_first():
    # fit_text/align_colorbars each draw the canvas, which settles the layout; with both
    # off the labels still wait for (and force) a settled layout rather than going on
    # against the unlaid-out one
    for family, items in (
        ("section", [_section_item(contour=_temperature())]),
        ("section_row", [_row_item(_row_contour())]),
    ):
        fig = _draw(
            family, items, mark="contourf", fit_text=False, align_colorbars=False
        )
        sets = [lines for ax in _panels(fig) for lines in _lines(ax)]
        assert sets and all(_label_texts(lines) for lines in sets)


def test_a_wrapped_grid_with_a_blank_cell_draws_lines_on_every_drawn_panel():
    items = [_section_item(contour=_temperature(), label=f"s{i}") for i in range(3)]
    fig = _draw("section", items, mark="contourf", ncols=2)
    panels = _panels(fig)
    assert len(panels) == 4
    visible = [ax for ax in panels if ax.get_visible()]
    assert len(visible) == 3
    assert all(len(_lines(ax)) == 1 and ax.texts for ax in visible)
    blank = next(ax for ax in panels if not ax.get_visible())
    assert list(blank.collections) == []


# -- styling the lines --------------------------------------------------------------


def test_contour_kwargs_colour_the_lines_and_silence_or_reformat_the_labels():
    item = _section_item(contour=_temperature())
    fig = _draw(
        "section",
        [item],
        mark="contourf",
        contour_kwargs={"colors": "white", "labels": False, "fmt": "%.1f"},
    )
    (ax,) = _panels(fig)
    (lines,) = _lines(ax)
    assert np.allclose(lines.get_edgecolor(), [1.0, 1.0, 1.0, 1.0])
    assert lines.labelTexts == [] and not ax.texts  # labels=False

    fig = _draw(
        "section",
        [item],
        mark="contourf",
        contour_kwargs={"colors": "white", "fmt": "%.1f"},
    )
    (lines,) = _lines(_panels(fig)[0])
    expected = {contour_label(v, "%.1f") for v in lines.levels}
    assert expected == {"5.0", "10.0", "15.0", "20.0", "25.0"}
    assert set(_label_texts(lines)) == expected
    assert all(mcolors.same_color(t.get_color(), "white") for t in lines.labelTexts)


def test_other_ax_contour_keywords_pass_through():
    item = _section_item(contour=_temperature())
    fig = _draw(
        "section",
        [item],
        contour_kwargs={"linewidths": 2.0, "linestyles": "dashed", "alpha": 0.5},
    )
    (lines,) = _lines(_panels(fig)[0])
    assert set(lines.get_linewidth()) == {2.0}
    assert lines.get_alpha() == 0.5
    assert all(style[1] is not None for style in lines.get_linestyle())  # dashed

    fig = _draw("section", [item], contour_kwargs={"cmap": "viridis"})
    (lines,) = _lines(_panels(fig)[0])
    assert lines.get_cmap().name == "viridis"  # a colour map replaces the default black


def test_contour_kwargs_style_every_row_panel_and_stack():
    for family, items in (
        ("section_row", [_row_item(_row_contour())]),
        ("section", [_section_item(contour=_temperature()) for _ in range(2)]),
        ("section_row", [_row_item(_row_contour()) for _ in range(2)]),
    ):
        fig = _draw(family, items, contour_kwargs={"colors": "white", "labels": False})
        sets = [lines for ax in _panels(fig) for lines in _lines(ax)]
        assert sets
        assert all(np.allclose(s.get_edgecolor(), [1, 1, 1, 1]) for s in sets)
        labels = [
            t
            for ax in _panels(fig)
            for t in ax.texts
            if t is not getattr(ax, "_osk_metrics_text", None)
        ]
        assert labels == []


# -- refusals -----------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_a_mismatched_overlay_grid_raises_before_any_figure_exists(family):
    bad = _temperature(n_z=5)  # five depth levels under a fill that has eight
    over = {"test": bad, "reference": bad} if "row" in family else bad
    fam, items = _four_families()[family]
    items = [
        {**item, "contour": over} if i == 0 else item for i, item in enumerate(items)
    ]
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="depth levels differ"):
        _draw(fam, items, mark="contourf")
    assert set(plt.get_fignums()) == before


def test_a_mismatch_on_the_reference_lane_alone_is_caught():
    contour = {"test": _temperature(), "reference": _temperature(n_along=9)}
    with pytest.raises(ValueError, match="positions along the section differ"):
        _draw("section_row", [_row_item(contour)])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    ("option", "value"), [("contour_levels", 5), ("contour_kwargs", {"colors": "r"})]
)
def test_contour_options_without_any_overlay_raise(family, option, value):
    fam, items = _four_families()[family]
    with pytest.raises(ValueError, match=f"{option}="):
        _draw(fam, items, mark="contourf", **{option: value})


@pytest.mark.parametrize("family", FAMILIES)
def test_fill_levels_with_pcolormesh_raises(family):
    fam, items = _four_families()[family]
    with pytest.raises(ValueError, match=r"fill_levels=.*pcolormesh"):
        _draw(fam, items, fill_levels=10)
    with pytest.raises(ValueError, match="fill_levels="):
        _draw(fam, items, mark="pcolormesh", fill_levels=[0.0, 1.0])


@pytest.mark.parametrize("bad", [0, [1.0], "forty", 2.5])
def test_a_bad_fill_levels_is_refused_before_a_figure_opens(bad):
    before = set(plt.get_fignums())
    with pytest.raises((TypeError, ValueError)):
        _draw("section", [_section_item()], mark="contourf", fill_levels=bad)
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize("bad", ["many", 0, [10.0, float("nan")]])
def test_a_bad_contour_levels_is_refused(bad):
    with pytest.raises((TypeError, ValueError)):
        _draw("section", [_section_item(contour=_temperature())], contour_levels=bad)


def test_contour_kwargs_must_be_a_dict_and_may_not_carry_levels():
    item = _section_item(contour=_temperature())
    with pytest.raises(TypeError, match="contour_kwargs="):
        _draw("section", [item], contour_kwargs=["colors", "red"])
    with pytest.raises(ValueError, match="contour_levels="):
        _draw("section", [item], contour_kwargs={"levels": [10]})


def test_the_new_options_are_accepted_by_every_section_signature():
    from ocean_skill.plot.matplotlib_renderer import _top_level_options

    for option in ("fill_levels", "contour_levels", "contour_kwargs"):
        assert option in _top_level_options()
    _draw(
        "section",
        [_section_item(contour=_temperature())],
        mark="contourf",
        fill_levels=20,
        contour_levels=3,
        contour_kwargs={"colors": "0.2"},
    )


# -- a figure that is saved ---------------------------------------------------------


def test_the_report_rasterization_leaves_the_labels_alone(tmp_path: Path):
    from ocean_skill.workflows.report import _rasterize

    fig = _draw(
        "section",
        [_section_item(contour=_temperature())],
        mark="contourf",
    )
    (ax,) = _panels(fig)
    labels = [t.get_text() for t in ax.texts]
    assert labels
    _rasterize(fig)  # skips contour sets; the label texts are not collections at all
    assert not any(t.get_rasterized() for t in ax.texts)
    assert [t.get_text() for t in ax.texts] == labels
    for name in ("section.png", "section.pdf"):
        fig.savefig(tmp_path / name, dpi=100, bbox_inches="tight")
        assert (tmp_path / name).stat().st_size > 0


@pytest.mark.skipif(
    shutil.which("pdftotext") is None,
    reason="pdftotext is not installed: a PDF's text cannot be read back",
)
def test_the_labels_are_text_in_a_pdf_saved_by_the_family_itself(tmp_path: Path):
    out = tmp_path / "saved" / "section.pdf"
    _draw(
        "section",
        [_section_item(contour=_temperature())],
        mark="contourf",
        contour_kwargs={"fmt": "T%g"},
        save=out,
    )
    text = subprocess.run(
        ["pdftotext", "-layout", str(out), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # saved by section() itself, so the labels went on before the figure was written
    for level in (5, 10, 15, 20, 25):
        assert f"T{level}" in text
