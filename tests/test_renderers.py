"""Tests that the two renderers agree where they claim to.

``renderer="holoviews"`` is documented as the same plot, drawn interactively — so
anything both renderers are supposed to honor (``title``, ``metric_keys``, and
crucially each row's *own* source labels) has to actually reach the output, not
just be accepted and dropped. The per-row labelling case is a regression guard:
the static renderer was fixed for it and the interactive one silently was not, so
every row in an interactive grid carried the *first* row's reference name.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec


def _field(offset: float = 0.0) -> xr.DataArray:
    return xr.DataArray(
        5.0 + offset + np.linspace(0, 1, 80).reshape(8, 10),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(18, 26, 8), "lon": np.linspace(260, 270, 10)},
    )


def _item(standard_name: str, reference: str, row_label: str) -> dict:
    """One spec item shaped exactly as ``Comparison.as_item()`` builds it."""
    test, ref = _field(1.0), _field(0.0)
    return {
        "aligned": {"test": test, "reference": ref, "difference": test - ref},
        "units": "mmol m-3",
        "standard_name": standard_name,
        "metrics": {"bias": 0.125, "rmse": 0.5, "corr": 0.98},
        "labels": ("GOM_bgc", reference),
        "row_label": row_label,
    }


@pytest.fixture
def two_rows():
    """Two rows from *different* reference sources — a real compare() fan-out.

    WOA ships one dataset per variable, so a multi-variable comparison pairs each
    variable with its own reference entry; the rows do not share a label pair.
    """
    return [
        _item("mole_concentration_of_nitrate_in_sea_water", "woa23_nitrate", "nitrate"),
        _item(
            "mole_concentration_of_phosphate_in_sea_water",
            "woa23_phosphate",
            "phosphate",
        ),
    ]


def _holoviews_panel_titles(obj) -> list[str]:
    """Every rendered bokeh figure title, in document order."""
    import holoviews as hv
    from bokeh.plotting import figure

    return [
        f.title.text for f in hv.render(obj, backend="bokeh").select({"type": figure})
    ]


def _matplotlib_panel_titles(fig) -> list[str]:
    return [ax.get_title() for ax in fig.axes if ax.get_title()]


# ComparisonSet.plot() setdefault()s `labels` to the *first* comparison's pair, so
# both renderers get a top-level fallback that is only correct for row 0.
_TOP_LEVEL = {"labels": ("GOM_bgc", "woa23_nitrate"), "title": "GOM vs WOA"}


def test_matplotlib_grid_labels_each_row_from_its_own_source(two_rows):
    import matplotlib

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)),
        renderer="matplotlib",
    )
    titles = _matplotlib_panel_titles(fig)
    assert "woa23_nitrate" in titles
    assert "woa23_phosphate" in titles


def test_holoviews_grid_labels_each_row_from_its_own_source(two_rows):
    """Regression: row 2 used to inherit row 1's reference name."""
    out = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)),
        renderer="holoviews",
    )
    titles = _holoviews_panel_titles(out)
    assert any("woa23_nitrate" in t for t in titles)
    assert any("woa23_phosphate" in t for t in titles), (
        f"row 2 lost its own reference label; got {titles}"
    )


def test_holoviews_grid_identifies_each_row_variable(two_rows):
    """The static renderer's rotated row label has to survive in some form."""
    out = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)),
        renderer="holoviews",
    )
    titles = _holoviews_panel_titles(out)
    assert any("nitrate" in t for t in titles)
    assert any("phosphate" in t for t in titles)


def test_holoviews_grid_shows_metrics(two_rows):
    out = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)),
        renderer="holoviews",
    )
    assert any("bias=0.125" in t for t in _holoviews_panel_titles(out))


def test_holoviews_metric_keys_selects_which_metrics(two_rows):
    out = render(
        PlotSpec(
            family="field_grid",
            items=two_rows,
            options={**_TOP_LEVEL, "metric_keys": ("corr",)},
        ),
        renderer="holoviews",
    )
    titles = _holoviews_panel_titles(out)
    assert any("corr=0.98" in t for t in titles)
    assert not any("bias" in t for t in titles)


def test_holoviews_grid_renders_the_overall_title(two_rows):
    import holoviews as hv
    from bokeh.models import Div

    out = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)),
        renderer="holoviews",
    )
    divs = list(hv.render(out, backend="bokeh").select({"type": Div}))
    assert any("GOM vs WOA" in (d.text or "") for d in divs)


def test_holoviews_grid_links_pan_and_zoom(two_rows):
    """Shared zoom means one shared bokeh Range object, not merely an option set."""
    import holoviews as hv
    from bokeh.plotting import figure

    out = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)),
        renderer="holoviews",
    )
    figs = list(hv.render(out, backend="bokeh").select({"type": figure}))
    assert len(figs) == 6
    assert len({id(f.x_range) for f in figs}) == 1
    assert len({id(f.y_range) for f in figs}) == 1


def test_holoviews_grid_can_unlink_pan_and_zoom(two_rows):
    import holoviews as hv
    from bokeh.plotting import figure

    out = render(
        PlotSpec(
            family="field_grid",
            items=two_rows,
            options={**_TOP_LEVEL, "shared_axes": False},
        ),
        renderer="holoviews",
    )
    figs = list(hv.render(out, backend="bokeh").select({"type": figure}))
    assert len({id(f.x_range) for f in figs}) > 1


def test_holoviews_warns_on_static_only_styling(two_rows):
    """Silently absorbing a matplotlib-only kwarg would look like it had applied."""
    with pytest.warns(UserWarning, match="static"):
        render(
            PlotSpec(
                family="field_grid",
                items=two_rows,
                options={**_TOP_LEVEL, "title_kwargs": {"fontsize": 20}},
            ),
            renderer="holoviews",
        )


# ---------------------------------------------------------------- row label placement


def _row_label_gap(fig) -> float:
    """Smallest horizontal gap (px) between a row label and its latitude labels.

    Negative means they overlap. Cartopy's gridline labels are free artists hanging off
    the Gridliner rather than the axes' ytick labels, so they have to be reached through
    ``ax.artists`` — which is also exactly why matplotlib's own label placement cannot
    see them.
    """
    renderer = fig.canvas.get_renderer()
    gaps = []
    for ax in fig.axes:
        label = getattr(ax, "_osk_row_label", None)
        if label is None:
            continue
        lefts = [
            text.get_window_extent(renderer).x0
            for artist in ax.artists
            for text in getattr(artist, "left_label_artists", []) or []
            if text.get_visible() and text.get_text()
        ]
        if lefts:
            gaps.append(min(lefts) - label.get_window_extent(renderer).x1)
    return min(gaps) if gaps else float("nan")


@pytest.mark.parametrize("width", [8.5, 6.5, 5.0, 3.5])
def test_row_labels_never_overlap_the_latitude_labels(two_rows, width):
    """At *any* figure width — the bug was an offset in axes fraction.

    ``x=-0.18`` is a share of the panel width, but the latitude labels it must clear
    are a fixed text width, so the two scaled differently: the label overlapped by 4px
    at 8.5in and 31px at 3.5in.
    """
    fig = render(
        PlotSpec(family="field_grid", items=two_rows, options={"figsize": (width, 4.4)})
    )
    assert _row_label_gap(fig) > 0


def _leftmost_label_x(fig) -> float:
    """Display x of the leftmost thing drawn beside the maps; < 0 means clipped."""
    from ocean_skill.plot.matplotlib_renderer import _left_label_artists

    renderer = fig.canvas.get_renderer()
    return min(
        text.get_window_extent(renderer).x0
        for ax in fig.axes
        for text in _left_label_artists(ax)
    )


def test_left_labels_stay_on_the_canvas(two_rows):
    fig = render(PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)))
    assert _leftmost_label_x(fig) >= 0


def test_every_panel_survives_a_tight_bbox_crop(two_rows):
    """No axes may report a non-finite tight bbox.

    One that does is dropped from the *figure's* tight bbox, and then
    ``bbox_inches="tight"`` — our own ``save=``, and Jupyter's inline backend —
    crops it out of the picture entirely. That is how a whole column of maps went
    missing on matplotlib 3.11 while every panel was still being drawn correctly:
    automatic title placement over a gridline-labelled GeoAxes put the title at
    y=inf, so its extent was NaN and it poisoned the axes bbox containing it.
    """
    fig = render(PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)))
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        bbox = ax.get_tightbbox(renderer)
        assert np.isfinite([bbox.x0, bbox.y0, bbox.x1, bbox.y1]).all(), (
            f"non-finite tight bbox for the {ax.get_title()!r} panel: {bbox}"
        )
    figure_bbox = fig.get_tightbbox(renderer)
    leftmost = min(ax.get_position().x0 for ax in fig.axes) * fig.get_size_inches()[0]
    assert figure_bbox.x0 <= leftmost, "the tight bbox starts right of the first column"


def test_left_margin_is_refitted_when_the_layout_reserves_none(two_rows):
    """Backstop: recover a left margin too narrow for the labels hanging off it.

    Not reachable through the public API now that the NaN title bbox behind the
    matplotlib 3.11 clipping is fixed at the source (see ``DEFAULT_TITLE_KWARGS``),
    which is why the clipped state has to be staged here by shifting ``rect`` off the
    left edge. Kept because the failure it recovers from is invisible in the numbers
    the renderer reports — the panels are all drawn, and only the labels outboard of
    the leftmost axes go missing.

    ``align_colorbars=False`` only because the colorbar refit stands the layout engine
    down when it is done (see ``_align_colorbars``), and staging this needs an engine
    whose ``rect`` can still be pushed around.
    """
    from ocean_skill.plot.matplotlib_renderer import _fit_left_margin

    fig = render(
        PlotSpec(
            family="field_grid",
            items=two_rows,
            options={**dict(_TOP_LEVEL), "align_colorbars": False},
        )
    )
    fig.get_layout_engine().set(rect=(-0.08, 0, 1.08, 1))
    fig.canvas.draw()
    assert _leftmost_label_x(fig) < 0, "failed to stage the clipped layout"

    _fit_left_margin(fig)
    assert _leftmost_label_x(fig) >= 0
    assert _row_label_gap(fig) > 0, "the refit must not push labels into each other"


def _colorbar_axes(fig):
    """Every colorbar axes, paired with the panels it describes."""
    return [
        (ax, parents)
        for ax in fig.axes
        if (parents := getattr(ax, "_osk_cbar_parents", None))
    ]


def test_colorbars_start_and_end_level_with_their_panels(two_rows):
    """The bar's long axis matches the *maps*, not the cell holding their labelling.

    ``fig.colorbar(im, ax=...)`` sizes the bar to the gridspec cell, which also holds
    the title above and the longitude labels below — and a cartopy GeoAxes shrinks
    itself inside its own slot to keep its aspect on top of that. So the bar overshot
    the map at both ends, which reads badly for something that is the map's own ruler.
    """
    fig = render(PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)))
    bars = _colorbar_axes(fig)
    assert len(bars) == 4  # two rows x (shared scale, difference)
    for cax, parents in bars:
        bar, boxes = cax.get_position(), [p.get_position() for p in parents]
        assert bar.y0 == pytest.approx(min(b.y0 for b in boxes), abs=1e-6)
        assert bar.y1 == pytest.approx(max(b.y1 for b in boxes), abs=1e-6)


def test_row_colorbars_span_their_panels_and_share_one_thickness():
    """The horizontal case: left/right edges, and no fat-bar-next-to-thin-bar.

    A field row pairs one bar over two panels with one over a single panel; sizing
    each one's thickness from its own length and ``aspect`` made the first two and a
    half times fatter than the second.
    """
    test, ref = _field(1.0), _field(0.0)
    aligned = {"test": test, "reference": ref, "difference": test - ref}
    fig = render(
        PlotSpec(
            family="field_row", items=[{"aligned": aligned}], options={"title": "row"}
        )
    )
    bars = _colorbar_axes(fig)
    assert len(bars) == 2
    for cax, parents in bars:
        bar, boxes = cax.get_position(), [p.get_position() for p in parents]
        assert bar.x0 == pytest.approx(min(b.x0 for b in boxes), abs=1e-6)
        assert bar.x1 == pytest.approx(max(b.x1 for b in boxes), abs=1e-6)
    heights = {round(cax.get_position().height, 9) for cax, _ in bars}
    assert len(heights) == 1


def test_colorbars_sit_the_same_distance_from_their_panels(two_rows):
    """``pad`` is a fraction of the parent's own width, so it has to be levelled too.

    A grid row's shared-scale bar is padded off a two-panel span and its difference bar
    off one panel, which left the far-right bar with roughly half the gap.
    """
    fig = render(PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)))
    gaps = {
        round(cax.get_position().x0 - max(p.get_position().x1 for p in parents), 9)
        for cax, parents in _colorbar_axes(fig)
    }
    assert len(gaps) == 1
    assert gaps.pop() > 0


def test_colorbar_alignment_survives_a_redraw(two_rows):
    """A later draw must not hand placement back to the layout engine.

    ``savefig`` and Jupyter's inline backend both draw again after we return the
    figure; with constrained_layout still live, that recomputes the positions and
    silently undoes the refit.
    """
    fig = render(PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)))
    before = [cax.get_position().frozen() for cax, _ in _colorbar_axes(fig)]
    fig.canvas.draw()
    after = [cax.get_position().frozen() for cax, _ in _colorbar_axes(fig)]
    for b, a in zip(before, after, strict=True):
        assert (a.y0, a.y1) == pytest.approx((b.y0, b.y1), abs=1e-6)


def test_colorbar_alignment_can_be_turned_off(two_rows):
    """``row_height`` is staged loose on purpose, and that is worth explaining.

    ``fig.colorbar(ax=...)`` sizes a bar to the gridspec *cell*, so the overshoot this
    checks for exists only insofar as the cell is taller than the panel inside it. Rows
    used to carry ~0.9in of slack each (a miscalibrated ``ROW_OVERHEAD`` -- see
    ``typography``), which is where that headroom came from. With rows sized to what
    they need, an un-refitted bar lands flush and the overshoot measures exactly zero.

    So the loose row is what makes the *absence* of the refit observable. The refit
    still matters at any row height: it equalises bar thickness across a row, and
    follows the way a fixed-aspect GeoAxes shrinks inside its own slot.
    """
    fig = render(
        PlotSpec(
            family="field_grid",
            items=two_rows,
            options={**dict(_TOP_LEVEL), "align_colorbars": False, "row_height": 2.6},
        )
    )
    bars = _colorbar_axes(fig)
    assert bars, "the parent bookkeeping is recorded either way"
    assert any(
        cax.get_position().y1 > max(p.get_position().y1 for p in parents) + 1e-3
        for cax, parents in bars
    ), "without the refit the bar should still overshoot its panels"


def test_row_labels_are_not_bold(two_rows):
    fig = render(PlotSpec(family="field_grid", items=two_rows, options={}))
    labels = [
        getattr(ax, "_osk_row_label")
        for ax in fig.axes
        if getattr(ax, "_osk_row_label", None) is not None
    ]
    assert labels
    assert all(lbl.get_weight() == "normal" for lbl in labels)


def test_row_label_styling_is_still_overridable(two_rows):
    fig = render(
        PlotSpec(
            family="field_grid",
            items=two_rows,
            options={"row_label_kwargs": {"weight": "bold", "fontsize": 11}},
        )
    )
    label = next(
        getattr(ax, "_osk_row_label")
        for ax in fig.axes
        if getattr(ax, "_osk_row_label", None) is not None
    )
    assert label.get_weight() == "bold"
    assert label.get_fontsize() == 11
    assert _row_label_gap(fig) > 0, "a bigger label must still clear the ticks"


# ------------------------------------------------------- misplaced styling options


def test_a_nested_styling_key_at_the_top_level_says_where_it_belongs(two_rows):
    """``label_size`` is a real colorbar key, so "unexpected keyword" is not enough.

    Python's own message names the key and stops, which leaves the user guessing that
    the option is valid but one level down.
    """
    with pytest.raises(TypeError, match=r"colorbar_kwargs"):
        render(PlotSpec(family="field_grid", items=two_rows, options={"label_size": 9}))


def test_an_option_with_no_home_is_still_rejected_clearly(two_rows):
    with pytest.raises(TypeError, match=r"not an option of field_grid"):
        render(PlotSpec(family="field_grid", items=two_rows, options={"colour": "red"}))


def test_the_error_lists_what_is_accepted(two_rows):
    with pytest.raises(TypeError, match=r"accepts:.*colorbar_kwargs"):
        render(PlotSpec(family="field_grid", items=two_rows, options={"label_size": 9}))


def test_correctly_nested_options_are_accepted(two_rows):
    """The fix the error suggests has to actually work."""
    fig = render(
        PlotSpec(
            family="field_grid",
            items=two_rows,
            options={"colorbar_kwargs": {"label_size": 9, "pad": 0.05}},
        )
    )
    assert fig is not None


def test_the_interactive_renderer_warns_instead_of_swallowing(two_rows):
    """It takes **_, so the same typo would error statically and vanish here."""
    with pytest.warns(UserWarning, match=r"colorbar_kwargs"):
        render(
            PlotSpec(family="field_grid", items=two_rows, options={"label_size": 9}),
            renderer="holoviews",
        )
