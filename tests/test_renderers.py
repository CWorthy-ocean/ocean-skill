"""Tests that the two renderers agree where they claim to.

``renderer="holoviews"`` is documented as the same plot, drawn interactively — so
anything both renderers are supposed to honor (``title``, ``metric_keys``, and
crucially each row's *own* source labels) has to actually reach the output, not
just be accepted and dropped. The per-row labelling case is a regression guard:
the static renderer was fixed for it and the interactive one silently was not, so
every row in an interactive grid carried the *first* row's reference name.
"""

from __future__ import annotations

import warnings

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


def _grid_item(standard_name, depth, time, reference):
    """A grid row carrying the depth/time an as_item() would, for title tests."""
    it = _item(standard_name, reference, standard_name)
    return {**it, "depth": depth, "time": time}


def test_grid_suptitle_composes_only_what_every_row_shares():
    """The variable/depth/time common to all rows makes the top title; whatever the fan
    varied is already the row label, so it is left out — no duplication, no stray part."""
    from ocean_skill.plot.matplotlib_renderer import grid_suptitle

    n = "mole_concentration_of_nitrate_in_sea_water"
    p = "mole_concentration_of_phosphate_in_sea_water"
    # one row: everything shared
    assert (
        grid_suptitle([_grid_item("chlorophyll", "0-10 m", "2010-01-22", "chl")])
        == "chlorophyll a · 0-10 m · 2010-01-22"
    )
    # variable fan-out: vars differ (they are the row labels), depth+time shared
    assert (
        grid_suptitle(
            [_grid_item(n, "surface", "2010-01", "woa_n"),
             _grid_item(p, "surface", "2010-01", "woa_p")]
        )
        == "surface · 2010-01"
    )
    # depth fan-out: one variable, depths differ (the row labels), time shared
    assert (
        grid_suptitle(
            [_grid_item(n, "surface", "2010-01", "woa"),
             _grid_item(n, "50 m", "2010-01", "woa")]
        )
        == "nitrate · 2010-01"
    )
    # nothing in common → no title at all
    assert grid_suptitle(
        [_grid_item(n, None, None, "a"), _grid_item(p, None, None, "b")]
    ) == ""


def test_a_stacked_grid_draws_the_shared_title_in_both_renderers():
    """A shared depth/time reaches the drawn figure as a suptitle, both renderers."""
    import holoviews as hv
    import matplotlib
    from bokeh.models import Div

    matplotlib.use("Agg")
    n = "mole_concentration_of_nitrate_in_sea_water"
    p = "mole_concentration_of_phosphate_in_sea_water"
    items = [_grid_item(n, "surface", "2010-01", "woa_n"),
             _grid_item(p, "surface", "2010-01", "woa_p")]

    fig = render(PlotSpec(family="field_grid", items=items, options={}))
    assert fig._suptitle.get_text() == "surface · 2010-01"

    out = render(PlotSpec(family="field_grid", items=items, options={}),
                 renderer="holoviews")
    divs = list(hv.render(out, backend="bokeh").select({"type": Div}))
    assert any("surface · 2010-01" in (d.text or "") for d in divs)


def test_a_grid_with_nothing_shared_still_draws_no_suptitle(two_rows):
    """The default must not invent a title where the rows have no common identity —
    two_rows are different variables with no depth/time, so the grid stays untitled."""
    import matplotlib

    matplotlib.use("Agg")
    fig = render(PlotSpec(family="field_grid", items=two_rows, options={}))
    assert fig._suptitle is None


def test_a_one_comparison_set_plots_as_a_single_row_not_a_grid(monkeypatch):
    """``compare()`` always returns a set, even of one. A lone comparison should draw as
    a single ``field_row`` — with its ``variable · depth · time`` suptitle — not a
    one-row grid whose only identity is a rotated left-edge label; two or more still
    stack as a grid."""
    from ocean_skill.comparison import ComparisonSet

    class _C:
        family = "field_row"

        def __init__(self, reference, label):
            self.test_name, self.reference_name, self.label = (
                "second_2wks",
                reference,
                label,
            )

        def metrics(self):  # _flatten keeps only objects with metrics + as_item
            return {"bias": 0.1}

        def as_item(self):
            t, r = _field(1.0), _field(0.0)
            return {
                "aligned": {"test": t, "reference": r, "difference": t - r},
                "standard_name": "chlorophyll",
                "depth": "0-10 m",
                "time": "2010-01-22",
                "units": "milligram m-3",
                "metrics": {"bias": 0.1},
                "labels": (self.test_name, self.reference_name),
            }

    captured = {}

    def fake_render(spec, **_):
        captured["family"] = spec.family
        return spec

    monkeypatch.setattr("ocean_skill.plot.registry.render", fake_render, raising=True)

    ComparisonSet([_C("chl_gapfree", "chlorophyll a")]).plot(domain=None)
    assert captured["family"] == "field_row"

    ComparisonSet([_C("woa_n", "nitrate"), _C("woa_p", "phosphate")]).plot(domain=None)
    assert captured["family"] == "field_grid"


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
    fig = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL))
    )
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
    fig = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL))
    )
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
    fig = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL))
    )
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


def _row_item(**over):
    """A single-comparison field_row item, shaped as ``Comparison.as_item()`` builds it."""
    test, ref = _field(1.0), _field(0.0)
    return {
        "aligned": {"test": test, "reference": ref, "difference": test - ref},
        "units": "mmol m-3",
        "standard_name": "chlorophyll",
        "depth": "0-10 m",
        "time": "2010-01-22",
        "labels": ("second_2wks", "chl_gapfree"),
        **over,
    }


def _mpl_row(**over):
    import matplotlib

    matplotlib.use("Agg")
    item = _row_item(**over)
    return render(PlotSpec(family="field_row", items=[item], options={}))


def _hv_row_title(**over):
    import holoviews as hv
    from bokeh.models import Div

    item = _row_item(**over)
    out = render(
        PlotSpec(family="field_row", items=[item], options={}), renderer="holoviews"
    )
    divs = list(hv.render(out, backend="bokeh").select({"type": Div}))
    return " ".join(d.text or "" for d in divs)


def test_a_single_comparison_row_titles_itself_from_variable_depth_time():
    """A lone row has no left-edge row label (that is field_grid's), so nothing said
    *what* — only which two sources. The suptitle now names variable · depth · time,
    the same spelling a one-field figure gets from field_suptitle, in both renderers."""
    expected = "chlorophyll a · 0-10 m · 2010-01-22"
    assert _mpl_row()._suptitle.get_text() == expected
    assert expected in _hv_row_title()


def test_an_explicit_row_title_wins_and_an_empty_one_drops_it():
    """The default only fills in when the caller named none — a string still overrides,
    and ``title=""`` suppresses the suptitle outright, in both renderers."""
    import holoviews as hv
    import matplotlib
    from bokeh.models import Div

    matplotlib.use("Agg")
    item = _row_item()
    spec = PlotSpec(family="field_row", items=[item], options={"title": "my run"})
    assert render(spec)._suptitle.get_text() == "my run"

    dropped = PlotSpec(family="field_row", items=[item], options={"title": ""})
    assert render(dropped)._suptitle is None

    out = render(spec, renderer="holoviews")
    divs = list(hv.render(out, backend="bokeh").select({"type": Div}))
    assert any("my run" in (d.text or "") for d in divs)


def test_a_row_with_no_depth_or_time_titles_from_the_variable_alone():
    """A pair nothing narrowed vertically or in time (or a diagnostic with no vertical
    axis, whose depth as_item drops) still gets the variable, with no stray separator."""
    assert _mpl_row(depth=None, time=None)._suptitle.get_text() == "chlorophyll a"


def test_a_grid_does_not_borrow_the_single_rows_auto_title(two_rows):
    """The auto suptitle is the *single* row's: a grid names its variable down each
    row's left edge and carries one title up top, so per-row titling must not leak in.
    Only the one overall title the caller passed should appear, once."""
    import holoviews as hv
    from bokeh.models import Div

    out = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL)),
        renderer="holoviews",
    )
    texts = [d.text or "" for d in hv.render(out, backend="bokeh").select({"type": Div})]
    assert sum("GOM vs WOA" in t for t in texts) == 1
    assert not any("·" in t for t in texts), "no row grew its own variable·depth title"


def test_colorbars_sit_the_same_distance_from_their_panels(two_rows):
    """``pad`` is a fraction of the parent's own width, so it has to be levelled too.

    A grid row's shared-scale bar is padded off a two-panel span and its difference bar
    off one panel, which left the far-right bar with roughly half the gap.
    """
    fig = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL))
    )
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
    fig = render(
        PlotSpec(family="field_grid", items=two_rows, options=dict(_TOP_LEVEL))
    )
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


# --- skill_map: one panel per metric, in both renderers ------------------------------
#
# Structurally the opposite of a facet grid on the one point that matters: those panels
# share a colour scale because they are one quantity at different times, while these
# cannot, being different quantities entirely. So the assertions below are the mirror
# images of test_facet.py's — one bar per panel, no two panels on one scale.

_SKILL_METRICS = ("bias", "crmsd", "corr", "sigma_ratio")


def _skill_dataset(names=_SKILL_METRICS, offset: float = 0.0) -> xr.Dataset:
    """One 2-D map per metric, with the sign and range each metric really has."""
    shape = (8, 10)
    ramp = np.linspace(0, 1, 80).reshape(shape)
    values = {
        # genuinely signed, or the symmetric-limits assertion cannot fail
        "bias": (ramp - 0.5) * 2.0 + offset,
        "crmsd": ramp * 1.5 + 0.1,  # strictly positive
        "corr": 0.6 + 0.35 * ramp,  # a realistic band well inside (-1, 1)
        "sigma_ratio": 0.7 + 0.6 * ramp,
        "rmse": ramp * 2.0 + 0.2,
        "n": np.round(ramp * 20 + 4),
    }
    units = {"bias": "mmol m-3", "crmsd": "mmol m-3", "rmse": "mmol m-3", "n": "count"}
    return xr.Dataset(
        {
            name: _field(0.0)
            .copy(data=values[name])
            .assign_attrs(units=units.get(name, ""))
            for name in names
        }
    )


def _skill_item(row_label: str = "nitrate", offset: float = 0.0) -> dict:
    """One spec item shaped as ``Comparison.as_item()`` builds it for a scored pair."""
    return {
        "skill": _skill_dataset(offset=offset),
        "metric_names": _SKILL_METRICS,
        "metrics": {
            "bias": 0.125,
            "crmsd": 0.5,
            "corr": 0.98,
            "sigma_ratio": 1.1,
            "weighted": True,
        },
        "units": "mmol m-3",
        "standard_name": "mole_concentration_of_nitrate_in_sea_water",
        "labels": ("GOM_bgc", "modis_chl"),
        "row_label": row_label,
    }


@pytest.fixture
def skill_item():
    return _skill_item()


def _skill_spec(items, **options):
    return PlotSpec(
        family="skill_map",
        items=items if isinstance(items, list) else [items],
        options=options,
    )


def test_skill_map_draws_the_same_panels_in_both_renderers(skill_item):
    """The parity contract: same panels, same titles, whichever backend drew them.

    Compared as sets because bokeh's document order is its own business (see
    ``test_facet.py``); the *count* and the *names* are what both renderers owe.
    """
    static = _matplotlib_panel_titles(render(_skill_spec(skill_item)))
    interactive = _holoviews_panel_titles(
        render(_skill_spec(skill_item), renderer="holoviews")
    )
    assert static == list(_SKILL_METRICS)
    assert len(interactive) == len(_SKILL_METRICS)
    assert {title.split(" (")[0] for title in interactive} == set(_SKILL_METRICS)


def test_skill_map_draws_a_single_metric_panel(skill_item):
    """One item, one requested metric collapses ``_skill_map`` to a single panel.

    Same underlying bug as the single-map facet case: with exactly one panel,
    ``layout`` stays a bare Overlay rather than becoming a Layout, and ``.cols()``
    / ``hv.opts.Layout`` only apply to the latter.
    """
    interactive = _holoviews_panel_titles(
        render(
            _skill_spec(skill_item, metric_names=("bias",)),
            renderer="holoviews",
        )
    )
    assert len(interactive) == 1


def test_the_overall_value_reaches_the_static_corner_box(skill_item):
    """The map and the single number are one statistic at two resolutions."""
    fig = render(_skill_spec(skill_item))
    boxes = {
        ax.get_title(): ax._osk_metrics_text.get_text()
        for ax in fig.axes
        if getattr(ax, "_osk_metrics_text", None) is not None
    }
    assert boxes["bias"] == "bias=0.125"
    assert boxes["corr"] == "corr=0.98"
    assert len(boxes) == len(_SKILL_METRICS), "every panel is annotated"


def test_the_overall_value_reaches_the_interactive_title(skill_item):
    titles = _holoviews_panel_titles(
        render(_skill_spec(skill_item), renderer="holoviews")
    )
    assert "bias (0.125)" in titles
    assert "corr (0.98)" in titles


def test_the_weighted_flag_is_never_rendered_as_a_number(skill_item):
    """``isinstance(True, int)`` is True, so a naive numeric test would print it."""
    item = {**skill_item, "metric_names": ("bias", "weighted")}
    with pytest.raises(ValueError, match="no pointwise map"):
        render(_skill_spec(item, metric_names=("bias", "weighted")))


def test_skill_map_draws_one_colorbar_per_panel(skill_item):
    """The inverse of ``field_facet``'s single shared bar, and the reason why.

    Bias in mmol m-3 and a dimensionless correlation have no shared scale to have, so
    each panel carries its own — and each bar describes exactly one panel.
    """
    fig = render(_skill_spec(skill_item))
    bars = _colorbar_axes(fig)
    assert len(bars) == len(_SKILL_METRICS)
    assert all(len(parents) == 1 for _, parents in bars)


def test_no_two_metric_panels_share_a_colour_scale(skill_item):
    """And each metric's limits are its own: pinned, symmetric or fixed as it needs."""
    from matplotlib.collections import QuadMesh

    fig = render(_skill_spec(skill_item))
    norms = {}
    for ax in fig.axes:
        if not ax.get_title():
            continue
        mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
        norms[ax.get_title()] = (mesh.norm.vmin, mesh.norm.vmax)
    assert len(set(norms.values())) == len(_SKILL_METRICS), "scales must not coincide"
    assert norms["crmsd"][0] == 0.0, "a magnitude's zero is pinned"
    assert norms["bias"][0] == pytest.approx(-norms["bias"][1]), "bias is symmetric"
    assert norms["corr"] == (-1.0, 1.0), "correlation has an absolute scale"
    assert sum(norms["sigma_ratio"]) == pytest.approx(2.0), "the ratio centres on 1"


def test_units_go_on_each_panels_own_colorbar_not_its_title(skill_item):
    fig = render(_skill_spec(skill_item))
    labels = {}
    for cax, parents in _colorbar_axes(fig):
        labels[parents[0].get_title()] = cax.get_ylabel() or cax.get_xlabel()
    assert labels["bias"] == "[mmol m-3]"
    assert labels["corr"] == "", "a correlation has no units to print"
    assert all("mmol" not in title for title in labels), "and no title repeats them"


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_metric_names_selects_and_orders_the_panels(skill_item, renderer):
    spec = _skill_spec(skill_item, metric_names=("corr", "bias"))
    drawn = render(spec, renderer=renderer)
    titles = (
        _matplotlib_panel_titles(drawn)
        if renderer == "matplotlib"
        else [t.split(" (")[0] for t in _holoviews_panel_titles(drawn)]
    )
    assert len(titles) == 2
    assert set(titles) == {"corr", "bias"}


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_a_metric_that_was_never_computed_is_refused(skill_item, renderer):
    """Dropping the panel would be invisible: three maps look like three maps."""
    with pytest.raises(ValueError) as excinfo:
        render(_skill_spec(skill_item, metric_names=("bias", "mae")), renderer=renderer)
    message = str(excinfo.value)
    assert "mae" in message
    assert "metrics=" in message, "the data-layer knob has to be named"


def test_metric_keys_is_not_a_skill_map_option(skill_item):
    """One metric per panel leaves nothing for the corner box to select."""
    with pytest.raises(TypeError) as excinfo:
        render(_skill_spec(skill_item, metric_keys=("bias",)))
    assert "metric_names" in str(excinfo.value), "and the right option is named"


def test_several_comparisons_become_rows_of_metrics(skill_item):
    items = [_skill_item("nitrate"), _skill_item("phosphate", offset=0.4)]
    fig = render(_skill_spec(items))
    titles = _matplotlib_panel_titles(fig)
    assert titles == list(_SKILL_METRICS), "titled once, on the top row"
    labels = [
        ax._osk_row_label.get_text()
        for ax in fig.axes
        if getattr(ax, "_osk_row_label", None) is not None
    ]
    assert labels == ["nitrate", "phosphate"]
    assert len(_colorbar_axes(fig)) == 2 * len(_SKILL_METRICS)


def test_skill_map_panels_are_not_squeezed(skill_item):
    """Catches the one sizing mistake nothing else here would notice.

    ``facet_figsize`` defaults to ``FACET_PANEL_W_FRACTION`` (0.88), the allowance for
    a grid whose panels *share* one colorbar and so have nothing beside them. A bar in
    every cell needs ``PANEL_W_FRACTION`` (0.72), and every other assertion in this file
    passes either way — the maps are simply smaller.
    """
    from ocean_skill.plot.typography import PANEL_W_FRACTION

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # _warn_if_cramped must not fire
        fig = render(_skill_spec(skill_item))
    fig.canvas.draw()
    fig_w = fig.get_size_inches()[0]
    panels = [ax for ax in fig.axes if ax.get_title()]
    ncols = len({round(ax.get_position().x0, 3) for ax in panels})
    widest = max(ax.get_position().width for ax in panels) * fig_w
    assert widest >= 0.9 * PANEL_W_FRACTION * fig_w / ncols


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


# --- size= / zoom= reach the interactive renderer too --------------------------------
#
# ``size``/``zoom`` are inches statically, which bokeh has no notion of, so they arrive
# as a ratio against the page and scale the frame in pixels (``_canvas_factor``). Every
# interactive family takes ``**_``, so one that simply forgot to name them absorbed them
# in silence and drew the default size — accepted, dropped, no warning, no clue.


def _hv_frame_widths(obj) -> list[int]:
    """Every rendered bokeh figure's frame width, in CSS pixels.

    ``getattr(obj, "object", obj)`` unwraps the ``pn.pane.HoloViews`` the movie families
    return for their widget (see ``tests/test_movie.py::_hv``); every other family hands
    back the bare holoviews object already.
    """
    import holoviews as hv
    from bokeh.plotting import figure

    plot = hv.render(getattr(obj, "object", obj), backend="bokeh")
    return [f.frame_width for f in plot.select({"type": figure}) if f.frame_width]


def _facet_field() -> xr.DataArray:
    """Build a four-period field: the payload the facet families take."""
    times = xr.date_range("2012-01-01", periods=4, freq="MS")
    base = _field(0.0)
    return xr.concat([base + i for i in range(4)], dim="time").assign_coords(time=times)


_INTERACTIVE_FAMILIES = {
    "field_row": lambda: [
        _item("mole_concentration_of_nitrate_in_sea_water", "woa", "n")
    ],
    "field_grid": lambda: [
        _item("mole_concentration_of_nitrate_in_sea_water", "woa", "n")
    ],
    "skill_map": lambda: [_skill_item()],
    "field_facet": lambda: [
        {
            "field": _facet_field(),
            "facet_dim": "time",
            "units": "mmol m-3",
            "standard_name": None,
        }
    ],
    "facet_movie": lambda: [
        {
            "field": _facet_field(),
            "facet_dim": "time",
            "units": "mmol m-3",
            "standard_name": None,
        }
    ],
    "field_movie": lambda: [
        {
            **_item("mole_concentration_of_nitrate_in_sea_water", "woa", "n"),
            "frame_label": "Jan 2012",
        },
        {
            **_item("mole_concentration_of_nitrate_in_sea_water", "woa", "n"),
            "frame_label": "Feb 2012",
        },
    ],
}


@pytest.mark.parametrize("family", sorted(_INTERACTIVE_FAMILIES))
def test_zoom_grows_the_interactive_frame_in_every_family(family):
    """A family that forgets to name ``zoom`` absorbs it into ``**_``, silently."""
    items = _INTERACTIVE_FAMILIES[family]()
    plain = _hv_frame_widths(
        render(PlotSpec(family=family, items=items), renderer="holoviews")
    )
    zoomed = _hv_frame_widths(
        render(
            PlotSpec(family=family, items=items, options={"zoom": 2.0}),
            renderer="holoviews",
        )
    )
    assert plain and len(zoomed) == len(plain), family
    assert all(z > p for z, p in zip(zoomed, plain, strict=True)), (
        f"{family}: zoom= was accepted and dropped ({plain} -> {zoomed})"
    )


@pytest.mark.parametrize("family", sorted(_INTERACTIVE_FAMILIES))
def test_a_named_canvas_reaches_the_interactive_frame_too(family):
    items = _INTERACTIVE_FAMILIES[family]()
    page = _hv_frame_widths(
        render(
            PlotSpec(family=family, items=items, options={"size": "page"}),
            renderer="holoviews",
        )
    )
    column = _hv_frame_widths(
        render(
            PlotSpec(family=family, items=items, options={"size": "column"}),
            renderer="holoviews",
        )
    )
    assert all(c < p for c, p in zip(column, page, strict=True)), family


# --- domain= draws the model-domain outline in both renderers ------------------------
#
# ``domain`` has always taken a ``(lon_min, lat_min, lon_max, lat_max)`` bbox; it now
# also takes an (N, 2) vertex ring — a curvilinear source's true, possibly rotated,
# perimeter (see ``ocean_skill.align.perimeter_of``) — and the interactive renderer
# used to drop the option outright (it sat in holoviews_renderer.render's ``drops``).

_DOMAIN_RING = [[261.0, 19.0], [269.0, 20.0], [268.0, 25.0], [262.0, 24.0]]
_DOMAIN_BBOX = (261.0, 19.0, 269.0, 25.0)


def _hv_paths(obj) -> list:
    """Every ``hv.Path`` element in ``obj``, unwrapping a movie's widget pane."""
    import holoviews as hv

    return getattr(obj, "object", obj).traverse(lambda x: x, [hv.Path])


def test_domain_draws_the_shape_it_was_given(two_rows):
    """One deep check, on ``field_grid``: ring, bbox, and ``None`` each draw right.

    Every family funnels a ``domain`` through the same ``_domain_overlay``/``mesh *
    outline`` composition, so the shape itself only needs proving once; the
    parametrized smoke test below is what catches a family that forgot to wire the
    option through at all.
    """
    ring_out = render(
        PlotSpec(family="field_grid", items=two_rows, options={"domain": _DOMAIN_RING}),
        renderer="holoviews",
    )
    assert len(_hv_paths(ring_out)) == 6  # 2 rows x 3 panels

    bbox_out = render(
        PlotSpec(family="field_grid", items=two_rows, options={"domain": _DOMAIN_BBOX}),
        renderer="holoviews",
    )
    assert len(_hv_paths(bbox_out)) == 6

    none_out = render(
        PlotSpec(family="field_grid", items=two_rows, options={"domain": None}),
        renderer="holoviews",
    )
    assert not _hv_paths(none_out)


@pytest.mark.parametrize("family", sorted(_INTERACTIVE_FAMILIES))
def test_domain_reaches_every_interactive_family(family):
    """Every family takes ``**_``, so one that forgot to name ``domain`` absorbs it.

    Same failure mode as the zoom=/size= checks above.
    """
    items = _INTERACTIVE_FAMILIES[family]()
    out = render(
        PlotSpec(family=family, items=items, options={"domain": _DOMAIN_RING}),
        renderer="holoviews",
    )
    assert _hv_paths(out), f"{family}: domain= was accepted and dropped"


def _matplotlib_dashed_lines(fig):
    return [
        line
        for ax in fig.axes
        for line in ax.get_lines()
        if line.get_linestyle() == "--"
    ]


def test_matplotlib_draws_the_domain_ring_on_every_panel(two_rows):
    import matplotlib

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(
            family="field_grid", items=two_rows, options={"domain": _DOMAIN_RING}
        ),
        renderer="matplotlib",
    )
    dashed = _matplotlib_dashed_lines(fig)
    # 2 rows x 3 panels (test | reference | difference) = 6 outlines
    assert len(dashed) == 6
    expected = np.asarray([*_DOMAIN_RING, _DOMAIN_RING[0]])
    for line in dashed:
        assert np.allclose(line.get_xydata(), expected)


def test_matplotlib_domain_bbox_still_draws_a_rectangle(two_rows):
    import matplotlib

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(
            family="field_grid", items=two_rows, options={"domain": _DOMAIN_BBOX}
        ),
        renderer="matplotlib",
    )
    dashed = _matplotlib_dashed_lines(fig)
    assert len(dashed) == 6
    lo0, la0, lo1, la1 = _DOMAIN_BBOX
    expected = np.asarray([[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]])
    for line in dashed:
        assert np.allclose(line.get_xydata(), expected)


def test_matplotlib_domain_none_draws_nothing(two_rows):
    import matplotlib

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(family="field_grid", items=two_rows, options={"domain": None}),
        renderer="matplotlib",
    )
    assert _matplotlib_dashed_lines(fig) == []
