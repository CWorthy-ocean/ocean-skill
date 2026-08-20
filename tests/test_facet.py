"""Tests for the model-only faceted path: consecutive periods, layout, both renderers.

The thing under test is a figure with no reference in it — one model run shown as a
series of maps over time. Three claims carry it, and each is a way to get a plausible
figure of the wrong thing if it breaks:

* ``resample`` and ``groupby`` are different reductions and stay distinguishable, on
  the page and not just in the call;
* a selection that does not land on period boundaries says so;
* the panels share one colour scale, and the grid is oriented by the domain's shape.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.operators import aggregate, select
from ocean_skill.plot import typography as tg
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

NITRATE = "mole_concentration_of_nitrate_in_sea_water"
MONTHLY = {"time": {"resample": "1MS", "reduce": "mean"}}
CLIMATOLOGY = {"time": {"groupby": "month", "reduce": "mean"}}


def _daily(days: int):
    """``days`` of daily output on a small lon/lat grid, starting 1 January 2012."""
    time = pd.date_range("2012-01-01", periods=days, freq="D")
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.normal(5.0, 1.0, (days, 12, 20)),
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": np.linspace(18, 31, 12),
            "lon": np.linspace(-98, -80, 20),
        },
        attrs={"units": "mmol m-3"},
    )


@pytest.fixture
def daily():
    """Six whole months of daily output — the case in hand.

    182 days, not 183: January through June 2012 is exactly 182 days, and one day more
    would open a seventh one-day bin that every panel-count assertion here would have
    to know about. (That the short-bin warning catches such a bin is
    :func:`test_a_selection_off_the_period_boundary_warns`'s job, not every test's.)
    """
    return _daily(182)


def _by_depth(daily, depths=(0.0, 50.0, 100.0)):
    """Monthly means at several levels — the two-facet-axis case.

    Built by hand rather than through ``select={"depth": [...]}`` because that route
    needs a ROMS grid to interpolate against; the shape it produces is what matters
    here. Each level is offset so the rows have genuinely different ranges, which is
    the reason they do not share a colour scale.
    """
    monthly = aggregate(daily, MONTHLY)
    levels = [monthly * (1.0 - i * 0.4) - i * 3.0 for i, _ in enumerate(depths)]
    return xr.concat(levels, dim=pd.Index(list(depths), name="depth"))


def _item(field, facet_dim, row_dim=None):
    return {
        "field": field,
        "facet_dim": facet_dim,
        "row_dim": row_dim,
        "units": "mmol m-3",
        "standard_name": NITRATE,
    }


def _mpl_titles(fig):
    return [
        ax.get_title()
        for ax in fig.axes
        if ax.get_visible()
        and not getattr(ax, "_osk_cbar_parents", None)
        and ax.get_title()
    ]


def _hv_titles(obj):
    """Every rendered bokeh figure title.

    Returned unordered on purpose: ``select`` walks the bokeh document, whose order is
    not the layout's, so asserting a sequence here would be asserting an implementation
    detail of bokeh rather than anything about the plot.
    """
    import holoviews as hv
    from bokeh.plotting import figure

    return [
        f.title.text for f in hv.render(obj, backend="bokeh").select({"type": figure})
    ]


# --- the two monthly reductions are not the same reduction ---------------------------


def test_resample_gives_consecutive_months_not_a_climatology(daily):
    """Six months of one run is six panels, not twelve bins of a climatology."""
    out = aggregate(daily, MONTHLY)
    assert out.sizes["time"] == 6
    assert list(out.time.dt.month.values) == [1, 2, 3, 4, 5, 6]
    assert out.attrs["units"] == "mmol m-3", "reductions drop attrs; units must survive"


def test_a_two_year_run_tells_the_two_reductions_apart(daily):
    """The case where they differ: 24 consecutive months against 12 climatological.

    On a single year they coincide, which is exactly why they have to be spelled
    differently — a reader cannot tell from the panel count alone.
    """
    time = pd.date_range("2012-01-01", periods=731, freq="D")
    da = xr.DataArray(
        np.ones((731, 3, 4)),
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": np.linspace(18, 31, 3),
            "lon": np.linspace(-98, -80, 4),
        },
    )
    assert aggregate(da, MONTHLY).sizes["time"] == 24
    assert aggregate(da, CLIMATOLOGY).sizes["month"] == 12


def test_groupby_and_resample_together_is_an_error(daily):
    both = {"time": {"groupby": "month", "resample": "1MS", "reduce": "mean"}}
    with pytest.raises(ValueError, match=r"sets both 'groupby'"):
        aggregate(daily, both)


def test_the_panels_say_which_reduction_made_them(daily):
    """A consecutive month carries its year; a climatological one has none to carry.

    This is the whole defence against confusing the two once the figure leaves the
    session that made it.
    """
    from ocean_skill.plot.matplotlib_renderer import facet_labels

    consecutive = aggregate(daily, MONTHLY)
    climatology = aggregate(daily, CLIMATOLOGY)
    assert facet_labels(consecutive["time"])[:2] == ["Jan 2012", "Feb 2012"]
    assert facet_labels(climatology["month"])[:2] == ["Jan", "Feb"]


def test_a_level_labels_attr_wins_the_row_labels():
    """A mixed vertical selection says its ``z=0.0`` row is the surface, not 0 m.

    The coordinate itself must stay numeric (the lane cache is zarr), so
    ``["surface", 50, 100]`` rides its spelling in a ``level_labels`` attr — see
    ``ocean_skill.comparison._surface_and_levels``. Honoured only at full length: a
    subset no longer knows which label belongs to which level, and falls back to the
    numeric spelling rather than mislabelling a row.
    """
    from ocean_skill.plot.matplotlib_renderer import facet_labels

    z = xr.DataArray(
        [0.0, -50.0, -100.0],
        dims="z",
        name="z",
        attrs={"level_labels": ["surface", "50 m", "100 m"]},
    )
    assert facet_labels(z) == ["surface", "50 m", "100 m"]
    assert facet_labels(z.isel(z=[1, 2])) == ["50 m", "100 m"]


def test_panels_finer_than_a_month_are_titled_finer_than_a_month(daily):
    """Three days of one January are three panels, so they need three titles.

    The unreduced case: ``select={"time": [d1, d2, d3]}`` leaves the model's own time
    axis standing, where ``"%b %Y"`` names the month all three share and none of the
    panels. A title that fits every panel identifies none of them.
    """
    from ocean_skill.plot.matplotlib_renderer import facet_labels

    days = select(daily, {"time": slice("2012-01-16", "2012-01-18")})
    assert facet_labels(days["time"]) == ["2012-01-16", "2012-01-17", "2012-01-18"]


def test_both_renderers_title_submonthly_panels_by_day(daily):
    """The labels reach the page, statically and interactively alike."""
    days = select(daily, {"time": slice("2012-01-16", "2012-01-18")})
    spec = PlotSpec(family="field_facet", items=[_item(days, "time")])
    expected = {"2012-01-16", "2012-01-17", "2012-01-18"}
    assert set(_mpl_titles(render(spec))) == expected
    assert set(_hv_titles(render(spec, renderer="holoviews"))) == expected


# --- a partial period is reported ----------------------------------------------------


def test_a_selection_off_the_period_boundary_warns(daily):
    """Feb 15 - May 15 gives two half-months labelled like whole ones."""
    sub = select(daily, {"time": slice("2012-02-15", "2012-05-15")})
    with pytest.warns(UserWarning, match="fewer samples"):
        out = aggregate(sub, MONTHLY)
    assert out.sizes["time"] == 4


def test_whole_periods_do_not_warn(daily):
    """Including February, which is genuinely shorter and must not be flagged."""
    sub = select(daily, {"time": slice("2012-01", "2012-06")})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert aggregate(sub, MONTHLY).sizes["time"] == 6


def test_the_short_bin_check_does_not_touch_the_data(daily):
    """It reads the time index, so a lazy array stays lazy through it.

    The check runs unconditionally; walking the data to count samples would make it a
    full read of every model run this is used on.
    """
    dask = pytest.importorskip("dask.array")
    lazy = daily.copy(data=dask.from_array(daily.values, chunks=(30, 12, 20)))
    out = aggregate(lazy, MONTHLY)
    assert hasattr(out.data, "compute"), "the reduction was computed eagerly"


# --- layout follows the domain -------------------------------------------------------


@pytest.mark.parametrize(
    ("aspect", "expected_ncols"),
    [
        (4.0, 1),  # wide: stack down the page, one panel per row
        (1.0, 2),  # square-ish: a compact grid
        (0.35, 3),  # tall: spread across the page
    ],
)
def test_layout_orientation_follows_the_domain(aspect, expected_ncols):
    ncols, nrows = tg.facet_layout(6, aspect)
    assert ncols == expected_ncols
    assert ncols * nrows >= 6


def test_layout_does_not_strand_panels_in_a_mostly_empty_grid():
    """Six tall panels must not land in a mostly empty grid.

    Pure aspect-matching picks 5x2 here; four blank cells out of ten reads as a bug
    rather than as a layout, which is what BLANK_CELL_WEIGHT is for.
    """
    ncols, nrows = tg.facet_layout(6, 0.35)
    assert ncols * nrows - 6 <= 1


def test_a_ragged_grid_is_still_allowed():
    """Seven panels have no rectangle; 3x3 with two blanks is the right answer."""
    ncols, nrows = tg.facet_layout(7, 1.0)
    assert ncols * nrows >= 7
    assert nrows == 3


def test_layout_needs_at_least_one_panel():
    with pytest.raises(ValueError, match="at least one panel"):
        tg.facet_layout(0, 1.0)


def test_the_suptitle_does_not_grow_on_a_one_column_grid():
    """A facet grid's overall title must match every other family's.

    Sizing the figure base off the cell width gives a one-column grid an 8.5in cell and
    a 17pt suptitle, twice what the same title gets on a field_row in the same report.
    """
    reference = tg.reference_scale()["suptitle"]
    one_col = tg.type_scale(
        (tg.PAGE_W, 10.2), ncols=1, nrows=6, figure_ncols=tg.REFERENCE_GRID[0]
    )
    assert one_col["suptitle"] == pytest.approx(reference, rel=0.01)


# --- the drawn figure ----------------------------------------------------------------


def test_matplotlib_draws_one_panel_per_period(daily):
    import matplotlib

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    fig = render(PlotSpec(family="field_facet", items=[_item(field, "time")]))
    assert _mpl_titles(fig) == [
        "Jan 2012",
        "Feb 2012",
        "Mar 2012",
        "Apr 2012",
        "May 2012",
        "Jun 2012",
    ]


def test_the_figure_names_the_variable_it_draws(daily):
    """The panels say *when*; without this nothing says *what*.

    A colorbar reading ``[mmol m-3]`` narrows a saved figure to "some concentration",
    and the source it came from is not on the page — so an alkalinity figure and a
    nitrate one were indistinguishable once out of the session that drew them.
    """
    import matplotlib

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    fig = render(PlotSpec(family="field_facet", items=[_item(field, "time")]))
    assert fig._suptitle.get_text() == "nitrate"
    # the short name every legend and axis label in the package uses, not the CF one
    assert NITRATE not in fig._suptitle.get_text()


def test_the_variable_name_is_a_default_and_not_a_fixture(daily):
    """An explicit title wins outright, and an empty one drops the suptitle."""
    import matplotlib

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    spec = PlotSpec(family="field_facet", items=[_item(field, "time")])
    assert render(spec, title="GOM run, 2012")._suptitle.get_text() == "GOM run, 2012"
    assert render(spec, title="")._suptitle is None


def test_the_interactive_twin_names_the_variable_too(daily):
    """Both renderers spell the field the same way — the shared field_title.

    Asserted on the rendered document rather than the options dict: bokeh draws a
    layout's title as a ``Div`` above the grid, and "the option was set" is not the
    same claim as "the name is on the page".
    """
    import holoviews as hv
    from bokeh.models import Div

    field = aggregate(daily, MONTHLY)
    spec = PlotSpec(family="field_facet", items=[_item(field, "time")])
    doc = hv.render(render(spec, renderer="holoviews"), backend="bokeh")
    assert any("nitrate" in d.text for d in doc.select({"type": Div}))


def test_the_title_sits_over_the_panels_not_over_the_canvas(daily):
    """A tall grid's maps are not centred on the page, and the title follows them.

    One narrow column of maps beside a vertical colorbar leaves the drawn block well
    right of the figure's middle, where matplotlib puts a suptitle. Unmoved, the title
    lands in the left margin, naming nothing.
    """
    import matplotlib

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    fig = render(PlotSpec(family="field_facet", items=[_item(field, "time")]), ncols=1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width = fig.get_size_inches()[0] * fig.dpi
    panels = [
        ax.get_window_extent(renderer)
        for ax in fig.axes
        if ax.get_visible() and not getattr(ax, "_osk_cbar_parents", None)
    ]
    middle = (min(b.x0 for b in panels) + max(b.x1 for b in panels)) / 2 / width
    assert fig._suptitle.get_position()[0] == pytest.approx(middle, abs=0.01)


def test_a_single_wide_map_keeps_its_title_close_to_the_map(daily):
    """A single map's colorbar goes under it, not beside it.

    A vertical bar beside a lone wide map leaves the figure sized for a panel title
    that is never drawn, and the fixed-aspect map centres in the surplus height —
    dropping it well below the suptitle. A bar on the map's own long edge sizes the
    figure honestly instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    field = daily.sel(time="2012-01-16")  # a scalar time coord: one wide map, no facet
    item = {
        "field": field,
        "facet_dim": None,
        "row_dim": None,
        "units": "mmol m-3",
        "standard_name": NITRATE,
    }
    fig = render(PlotSpec(family="field_facet", items=[item]))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    cbar_axes = [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]
    assert len(cbar_axes) == 1
    assert cbar_axes[0]._osk_cbar_horizontal is True

    panels = [
        ax
        for ax in fig.axes
        if ax.get_visible() and not getattr(ax, "_osk_cbar_parents", None)
    ]
    panel_top = max(ax.get_window_extent(renderer).y1 for ax in panels) / fig.dpi
    sup_bottom = fig._suptitle.get_window_extent(renderer).y0 / fig.dpi
    assert sup_bottom - panel_top < 0.2


def test_a_field_with_no_cf_name_gets_no_title_rather_than_a_guess(daily):
    """A derived expression has no standard_name to shorten; silence beats invention."""
    import matplotlib

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    item = {**_item(field, "time"), "standard_name": None}
    assert render(PlotSpec(family="field_facet", items=[item]))._suptitle is None


def test_the_suptitle_carries_the_source_and_depth_in_both_renderers(daily):
    """What a ``select=`` takes off the page belongs on the title instead.

    A monthly facet's panels already say *when*, so only the source and depth are
    missing from a plain "nitrate" — and both renderers compose them the same way.
    """
    import matplotlib
    import holoviews as hv
    from bokeh.models import Div

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    item = {**_item(field, "time"), "depth": "surface", "label": "ccs"}
    fig = render(PlotSpec(family="field_facet", items=[item]))
    assert fig._suptitle.get_text() == "ccs: nitrate · surface"

    doc = hv.render(render(PlotSpec(family="field_facet", items=[item]), renderer="holoviews"), backend="bokeh")
    assert any("ccs: nitrate · surface" in d.text for d in doc.select({"type": Div}))


def test_a_collapsed_single_map_also_carries_when_in_both_renderers(daily):
    """No panel and no row is left to say the depth or the instant; the title must.

    The case in hand: ``select={"depth": ..., "time": <one timestamp>}`` collapses
    both axes, so a lone map's only identifying text is its suptitle. This also pins
    the interactive single-panel fix — a lone panel used to crash instead of drawing.
    """
    import matplotlib
    import holoviews as hv
    from bokeh.plotting import figure

    matplotlib.use("Agg")
    field = daily.sel(time="2012-01-16")  # a scalar time coord, not a facet dim
    item = {
        "field": field,
        "facet_dim": None,
        "row_dim": None,
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "depth": "surface",
        "label": "ccs",
    }
    expected = "ccs: nitrate · surface · 2012-01-16"
    fig = render(PlotSpec(family="field_facet", items=[item]))
    assert fig._suptitle.get_text() == expected

    obj = render(PlotSpec(family="field_facet", items=[item]), renderer="holoviews")
    figs = list(hv.render(obj, backend="bokeh").select({"type": figure}))
    assert [f.title.text for f in figs] == [expected]


def test_a_faceted_vertical_suppresses_the_depth_part(daily):
    """A ``row_dim`` of levels already names the depth down the left edge."""
    import matplotlib

    matplotlib.use("Agg")
    item = {
        **_item(_by_depth(daily), "time", "depth"),
        "depth": "surface, 50 m, 100 m",
    }
    fig = render(PlotSpec(family="field_facet", items=[item]))
    assert fig._suptitle.get_text() == "nitrate"


# --- field_suptitle, in isolation from a drawn figure --------------------------------


def _no_dims(**coords):
    return xr.DataArray(
        np.zeros((3, 4)),
        dims=("lat", "lon"),
        coords={"lat": [1, 2, 3], "lon": [1, 2, 3, 4], **coords},
    )


def test_field_suptitle_drops_the_time_of_day_when_it_is_midnight():
    from ocean_skill.plot.matplotlib_renderer import field_suptitle

    field = _no_dims(time=np.datetime64("2013-01-30T00:00:00"))
    assert field_suptitle(field, standard_name=NITRATE) == "nitrate · 2013-01-30"


def test_field_suptitle_keeps_the_time_of_day_when_it_is_not_midnight():
    from ocean_skill.plot.matplotlib_renderer import field_suptitle

    field = _no_dims(time=np.datetime64("2013-01-30T14:00:00"))
    assert field_suptitle(field, standard_name=NITRATE) == "nitrate · 2013-01-30 14:00"


def test_field_suptitle_formats_a_cftime_scalar_without_raising():
    import cftime

    from ocean_skill.plot.matplotlib_renderer import field_suptitle

    field = _no_dims(time=cftime.DatetimeNoLeap(2013, 1, 30, 14, 30, 0))
    assert field_suptitle(field, standard_name=NITRATE) == "nitrate · 2013-01-30 14:30"


def test_field_suptitle_has_no_stray_separator_with_no_variable_name():
    from ocean_skill.plot.matplotlib_renderer import field_suptitle

    field = _no_dims()
    assert field_suptitle(field, standard_name=None, depth="surface") == "surface"
    assert field_suptitle(field, standard_name=None, depth=None) == ""


def test_field_suptitle_says_nothing_of_time_with_no_time_coord():
    from ocean_skill.plot.matplotlib_renderer import field_suptitle

    field = _no_dims()
    assert field_suptitle(field, standard_name=NITRATE, depth="surface") == (
        "nitrate · surface"
    )


def test_field_suptitle_prefixes_the_source_label():
    from ocean_skill.plot.matplotlib_renderer import field_suptitle

    field = _no_dims()
    assert field_suptitle(
        field, standard_name=NITRATE, depth="surface", label="ccs"
    ) == "ccs: nitrate · surface"
    # no label, no colon
    assert ":" not in field_suptitle(field, standard_name=NITRATE, depth="surface")


def test_every_panel_shares_one_colour_scale(daily):
    """The point of the family: a change between panels has to be visible as one.

    Per-panel scaling would draw a doubling from March to April as no change at all,
    since each panel would re-centre on its own range.
    """
    import matplotlib

    matplotlib.use("Agg")
    # a strong trend, so per-panel norms would be obviously different from shared ones
    field = aggregate(daily, MONTHLY)
    field = field + xr.DataArray(
        np.arange(field.sizes["time"]) * 10.0, dims="time", coords={"time": field.time}
    )
    fig = render(PlotSpec(family="field_facet", items=[_item(field, "time")]))
    from matplotlib.collections import QuadMesh

    # QuadMesh only: a cartopy GeoAxes also carries the land mask as a collection,
    # and that one has no colour limits of its own.
    norms = {
        (im.norm.vmin, im.norm.vmax)
        for ax in fig.axes
        if ax.get_visible() and not getattr(ax, "_osk_cbar_parents", None)
        for im in ax.collections
        if isinstance(im, QuadMesh)
    }
    assert len(norms) == 1, f"panels scaled independently: {norms}"
    (vmin, vmax) = next(iter(norms))
    assert vmax - vmin > 40, "the shared scale must span the trend across all panels"


def test_only_one_colorbar_is_drawn(daily):
    import matplotlib

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    fig = render(PlotSpec(family="field_facet", items=[_item(field, "time")]))
    bars = [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]
    assert len(bars) == 1
    assert len(bars[0]._osk_cbar_parents) == 6, "the bar must describe every panel"


def test_blank_cells_are_hidden_not_drawn():
    """Seven panels in an eight-cell grid: the spare cell carries no map."""
    import matplotlib

    matplotlib.use("Agg")
    # 213 days is January through July inclusive — seven whole months, no short bin
    seven = aggregate(_daily(213), MONTHLY)
    assert seven.sizes["time"] == 7
    fig = render(PlotSpec(family="field_facet", items=[_item(seven, "time")]))
    panels = [ax for ax in fig.axes if not getattr(ax, "_osk_cbar_parents", None)]
    assert sum(ax.get_visible() for ax in panels) == 7
    assert len(panels) == 8


def test_an_unknown_facet_dim_is_refused(daily):
    import matplotlib

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    with pytest.raises(ValueError, match="not a dimension"):
        render(PlotSpec(family="field_facet", items=[_item(field, "week")]))


def test_holoviews_draws_the_same_panels(daily):
    """Interactive is documented as the same plot; panels and labels are that."""
    field = aggregate(daily, MONTHLY)
    obj = render(
        PlotSpec(family="field_facet", items=[_item(field, "time")]),
        renderer="holoviews",
    )
    titles = _hv_titles(obj)
    assert len(titles) == 6
    assert set(titles) == {
        "Jan 2012",
        "Feb 2012",
        "Mar 2012",
        "Apr 2012",
        "May 2012",
        "Jun 2012",
    }


# --- the model-only lane -------------------------------------------------------------


@pytest.fixture
def prepared(monkeypatch, daily):
    """Stub the source out, so these exercise the lane rather than the catalog."""
    import ocean_skill.comparison as comparison

    field = aggregate(daily, MONTHLY)
    monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (field, None))
    return field


def test_the_leftover_axis_becomes_the_panels(prepared):
    """Nobody has to name the facet axis: what the reduction left standing is it."""
    from ocean_skill.field import field as make_field

    assert make_field("stub", "nitrate").facet_dim == "time"


def test_a_fully_collapsed_field_has_no_facet_axis(monkeypatch, daily):
    import ocean_skill.comparison as comparison
    from ocean_skill.field import field as make_field

    flat = aggregate(daily, {"time": "mean"})
    monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (flat, None))
    assert make_field("stub", "nitrate").facet_dim is None


def test_two_leftover_axes_become_a_grid_with_depth_down_the_rows(monkeypatch, daily):
    """The vertical axis takes the rows, whatever order the dims happen to be in.

    Depth reads top-to-bottom and time left-to-right; that is a convention, not
    something to be re-derived from whichever arrangement fits the page better.
    """
    import ocean_skill.comparison as comparison
    from ocean_skill.field import field as make_field

    monkeypatch.setattr(
        comparison, "prepare_source", lambda *a, **k: (_by_depth(daily), None)
    )
    assert make_field("stub", "nitrate").facet_dims == ("depth", "time")


def test_three_leftover_axes_are_refused(monkeypatch, daily):
    """A figure has rows and columns; a third axis would have to be dropped silently."""
    import ocean_skill.comparison as comparison
    from ocean_skill.field import field as make_field

    extra = _by_depth(daily).expand_dims(member=[1, 2])
    monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (extra, None))
    with pytest.raises(ValueError, match="only rows and columns"):
        _ = make_field("stub", "nitrate").facet_dims


@pytest.mark.parametrize("factory", ["field", "comparison"])
def test_a_bare_select_says_what_to_write_instead(factory):
    """``select="surface"`` reaches dict() and dies with a message about sequences.

    Both entry points, because the slip is the same one either side and a helpful
    error in only half of them is a coin toss from the caller's point of view.
    """
    from ocean_skill.comparison import Comparison
    from ocean_skill.field import field as make_field

    call = (
        (lambda: make_field("stub", "nitrate", select="surface"))
        if factory == "field"
        else (
            lambda: Comparison(reference="a", test="b", variable="x", select="surface")
        )
    )
    with pytest.raises(TypeError, match=r'select=\{"depth": .surface.\}'):
        call()


def test_a_missing_variable_is_reported_against_its_source(monkeypatch):
    import ocean_skill.comparison as comparison
    from ocean_skill.field import field as make_field

    monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (None, None))
    with pytest.raises(KeyError, match="stub"):
        make_field("stub", "nitrate").prepare()


def test_field_plot_draws_the_facet_family(prepared, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from ocean_skill.field import field as make_field

    # no catalog entry for the stub source, so there is no domain box to draw
    fig = make_field("stub", "nitrate", label="run A").plot(domain=None)
    assert len(_mpl_titles(fig)) == 6


# --- two facet axes: depth by month --------------------------------------------------


def _mpl_panels(fig):
    return [ax for ax in fig.axes if not getattr(ax, "_osk_cbar_parents", None)]


def test_depth_by_month_fills_a_determined_grid(daily):
    """Three levels by six months is 3x6 — no aspect-ratio choice left to make."""
    import matplotlib

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(family="field_facet", items=[_item(_by_depth(daily), "time", "depth")])
    )
    panels = _mpl_panels(fig)
    assert len(panels) == 18
    assert all(ax.get_visible() for ax in panels)


def test_the_month_is_titled_once_and_the_depth_named_down_the_side(daily):
    """Repeating "Jan 2012" on all three rows is noise; the row carries the level."""
    import matplotlib

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(family="field_facet", items=[_item(_by_depth(daily), "time", "depth")])
    )
    assert _mpl_titles(fig) == [
        "Jan 2012",
        "Feb 2012",
        "Mar 2012",
        "Apr 2012",
        "May 2012",
        "Jun 2012",
    ]
    labels = [
        ax._osk_row_label.get_text()
        for ax in _mpl_panels(fig)
        if getattr(ax, "_osk_row_label", None) is not None
    ]
    assert labels == ["0 m", "50 m", "100 m"]


def test_each_depth_row_keeps_its_own_colour_scale(daily):
    """Nitrate at 100 m and at the surface have unrelated ranges.

    One scale across both would push every surface panel to the bottom of the bar and
    hide the monthly change the figure exists to show — but *within* a row the months
    must still share, or the change is hidden the other way.
    """
    import matplotlib
    from matplotlib.collections import QuadMesh

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(family="field_facet", items=[_item(_by_depth(daily), "time", "depth")])
    )
    per_row = [
        {
            (im.norm.vmin, im.norm.vmax)
            for ax in _mpl_panels(fig)[r * 6 : (r + 1) * 6]
            for im in ax.collections
            if isinstance(im, QuadMesh)
        }
        for r in range(3)
    ]
    assert all(len(scales) == 1 for scales in per_row), "months differ within a row"
    assert len({next(iter(s)) for s in per_row}) == 3, "rows share one scale"


def test_one_colorbar_per_depth_row(daily):
    import matplotlib

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(family="field_facet", items=[_item(_by_depth(daily), "time", "depth")])
    )
    bars = [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]
    assert len(bars) == 3
    assert all(len(b._osk_cbar_parents) == 6 for b in bars)


def test_untitled_panels_still_pin_their_title_position(daily):
    """Every panel must skip matplotlib's automatic title placement, titled or not.

    Over a cartopy GeoAxes carrying gridline labels, matplotlib 3.11 computes an
    infinite title ``y``, which makes the axes report a NaN tight bbox and drop out of
    ``bbox_inches="tight"`` — silently, since the maps that survive look intact. The
    explicit ``y`` in DEFAULT_TITLE_KWARGS disarms it by clearing ``_autotitlepos``,
    but only on an axes ``set_title`` was actually called on; the twelve panels below
    the top row of a 3x6 grid are deliberately untitled and used not to be.

    Asserts the private ``_autotitlepos`` because that flag *is* the mechanism — the
    drawn output is identical on matplotlib 3.10, so nothing visible distinguishes a
    protected figure from a vulnerable one until the version that breaks.
    """
    import matplotlib

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(family="field_facet", items=[_item(_by_depth(daily), "time", "depth")])
    )
    panels = _mpl_panels(fig)
    untitled = [ax for ax in panels if not ax.get_title()]
    assert len(untitled) == 12, "the rows below the top should carry no month title"
    assert all(ax._autotitlepos is False for ax in panels)


def test_shared_limits_collapses_to_one_scale_and_one_bar(daily):
    import matplotlib
    from matplotlib.collections import QuadMesh

    matplotlib.use("Agg")
    fig = render(
        PlotSpec(
            family="field_facet",
            items=[_item(_by_depth(daily), "time", "depth")],
            options={"shared_limits": True},
        )
    )
    scales = {
        (im.norm.vmin, im.norm.vmax)
        for ax in _mpl_panels(fig)
        for im in ax.collections
        if isinstance(im, QuadMesh)
    }
    bars = [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]
    assert len(scales) == 1
    assert len(bars) == 1


def test_ncols_cannot_contradict_a_two_axis_grid(daily):
    import matplotlib

    matplotlib.use("Agg")
    with pytest.raises(ValueError, match="contradicts row_dim"):
        render(
            PlotSpec(
                family="field_facet",
                items=[_item(_by_depth(daily), "time", "depth")],
                options={"ncols": 4},
            )
        )


def test_one_axis_cannot_be_both_rows_and_columns(daily):
    import matplotlib

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    with pytest.raises(ValueError, match="cannot be both"):
        render(PlotSpec(family="field_facet", items=[_item(field, "time", "time")]))


def test_holoviews_draws_the_depth_by_month_grid_too(daily):
    """Same panels and same per-row scales, with the level folded into the title.

    Bokeh has no equivalent of the static renderer's rotated row label.
    """
    obj = render(
        PlotSpec(
            family="field_facet", items=[_item(_by_depth(daily), "time", "depth")]
        ),
        renderer="holoviews",
    )
    titles = _hv_titles(obj)
    assert len(titles) == 18
    assert "0 m — Jan 2012" in titles
    assert "100 m — Jun 2012" in titles


def test_both_renderers_arrange_the_panels_the_same_way(daily):
    """A plot that rearranges itself when you switch renderer is not the same plot."""
    import matplotlib

    matplotlib.use("Agg")
    field = aggregate(daily, MONTHLY)
    spec = PlotSpec(family="field_facet", items=[_item(field, "time")])
    fig = render(spec, renderer="matplotlib")
    panels = [ax for ax in fig.axes if not getattr(ax, "_osk_cbar_parents", None)]
    static_ncols = len({round(ax.get_subplotspec().colspan.start) for ax in panels})

    obj = render(spec, renderer="holoviews")
    assert obj._max_cols == static_ncols
