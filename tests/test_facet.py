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
