"""Grid fixes for the report pages.

Map titles, depth-by-statistic grids, row spacing and difference-only grids.

Four things a review of real PDF reports asked for, each checked in both renderers:

* a set of maps fanned over statistics titles each panel by its statistic, not by the
  one variable name they all share;
* a set fanned over levels *and* statistics is one grid, levels down and statistics
  across, one colorbar per statistic column;
* the stacked ``test | reference | difference`` rows of ``field_grid`` are evenly
  spaced and their colorbar labels do not run into the next row's;
* ``panels="difference"`` draws only the difference panels, on one shared symmetric
  scale (``ncols`` wraps them, ``diff_quantile`` sets how far the scale reaches).

Interactive checks call ``hv.render(..., backend="bokeh")``: some holoviews failures
only surface when bokeh builds the figure.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

TEMP = "sea_water_temperature"
IRON = "mole_concentration_of_dissolved_iron_in_sea_water"


def _bokeh(obj):
    import holoviews as hv

    return hv.render(obj, backend="bokeh")


def _bokeh_titles(obj) -> list[str]:
    from bokeh.plotting import figure

    return sorted(f.title.text for f in _bokeh(obj).select({"type": figure}))


def _color_mappers(obj) -> set[tuple[float, float]]:
    from bokeh.models import ColorBar

    return {
        (float(cb.color_mapper.low), float(cb.color_mapper.high))
        for cb in _bokeh(obj).select({"type": ColorBar})
    }


def _mpl_titles(fig) -> list[str]:
    return [ax.get_title() for ax in fig.axes if ax.get_title()]


# -- shared builders -----------------------------------------------------------------


def _pacific(offset: float = 0.0, *, ny: int = 8, nx: int = 14) -> xr.DataArray:
    """Return a small curvilinear grid straddling the dateline (100E-300E)."""
    lon = np.linspace(100.0, 300.0, nx)[None, :] * np.ones((ny, 1))
    lat = np.linspace(-45.0, 60.0, ny)[:, None] * np.ones((1, nx))
    values = 20 + offset + 5 * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon / 2))
    return xr.DataArray(
        values,
        dims=("eta", "xi"),
        coords={"lon": (("eta", "xi"), lon), "lat": (("eta", "xi"), lat)},
        attrs={"units": "Celsius"},
    )


def _row(label: str, offset: float, *, name: str = TEMP, units: str = "Celsius"):
    test, ref = _pacific(offset), _pacific(0.0)
    return {
        "aligned": xr.Dataset(
            {"test": test, "reference": ref, "difference": test - ref}
        ),
        "units": units,
        "standard_name": name,
        "metrics": {"bias": 0.1, "rmse": 0.5, "corr": 0.9},
        "labels": ("model", "woa"),
        "row_label": label,
        "label": label,
    }


def _grid(items, family="field_grid", **options):
    return PlotSpec(family=family, items=items, options=options)


# -- 1: map titles from member labels ------------------------------------------------


def _map_item(aggregate, *, depth=None, name=IRON, value=1.0):
    field = _pacific(value).rename("iron")
    return {
        "field": field,
        "units": "mmol m-3",
        "standard_name": name,
        "depth": depth,
        "time": "time mean",
        "label": f"run {aggregate}",
        "source_label": "run",
        "aggregate": aggregate,
    }


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_maps_of_one_variable_are_titled_by_their_statistic(renderer):
    items = [_map_item("mean"), _map_item("std of monthly means", value=2.0)]
    out = render(
        PlotSpec(family="field_map_grid", items=items, options={"tiles": False}),
        renderer=renderer,
    )
    titles = _mpl_titles(out) if renderer == "matplotlib" else _bokeh_titles(out)
    assert sorted(titles) == ["mean", "std of monthly means"]


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_maps_of_several_variables_keep_their_variable_titles(renderer):
    items = [
        {**_map_item(None), "standard_name": "sea_water_temperature"},
        {**_map_item(None), "standard_name": "sea_water_practical_salinity"},
    ]
    items = [{**it, "aggregate": None} for it in items]
    out = render(
        PlotSpec(family="field_map_grid", items=items, options={"tiles": False}),
        renderer=renderer,
    )
    titles = _mpl_titles(out) if renderer == "matplotlib" else _bokeh_titles(out)
    assert sorted(titles) == ["salinity", "temperature"]


def test_map_title_helper_names_only_the_parts_that_differ():
    from ocean_skill.plot._titles import map_panel_titles

    a = _map_item("mean", depth="0 m")
    b = _map_item("mean", depth="200 m")
    c = _map_item("std", depth="0 m")
    assert map_panel_titles([a, b]) == ["0 m", "200 m"]
    assert map_panel_titles([a, c]) == ["mean", "std"]
    assert map_panel_titles([a, b, c]) == ["mean · 0 m", "mean · 200 m", "std · 0 m"]
    # nothing differs (a hand-made list): fall back to the variable
    assert map_panel_titles([a, a]) == ["iron", "iron"]


# -- 2: depth x statistic grid -------------------------------------------------------


def _depth_stat_items():
    return [
        _map_item(stat, depth=depth, value=1.0 + 0.5 * k + 3 * j)
        for j, stat in enumerate(("mean", "std of monthly means"))
        for k, depth in enumerate(("0 m", "200 m"))
    ]


def _depth_stat_options(**over):
    return {
        "rows": "depth",
        "cols": "statistic",
        "shared_limits": "statistic",
        "colorbars": "column",
        "tiles": False,
        **over,
    }


def test_depth_by_statistic_is_a_grid_with_a_bar_per_statistic_column():
    from matplotlib.collections import QuadMesh

    fig = render(
        PlotSpec(
            family="field_map_grid",
            items=_depth_stat_items(),
            options=_depth_stat_options(),
        ),
        renderer="matplotlib",
    )
    gridspec = fig.axes[0].get_gridspec()
    assert (gridspec.nrows, gridspec.ncols) == (2, 2)
    # four panels + one colorbar per statistic column, not one per panel
    assert len(fig.axes) == 4 + 2
    # statistics across the top, titled once; levels down the left edge
    assert _mpl_titles(fig) == ["mean", "std of monthly means"]
    row_labels = [
        ax._osk_row_label.get_text()
        for ax in fig.axes[:4]
        if hasattr(ax, "_osk_row_label")
    ]
    assert row_labels == ["0 m", "200 m"]
    # the two panels of a column are on one scale; the columns are not
    norms = []
    for ax in fig.axes[:4]:
        mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
        norms.append((mesh.norm.vmin, mesh.norm.vmax))
    assert norms[0] == norms[2] and norms[1] == norms[3]
    assert norms[0] != norms[1]


def test_depth_by_statistic_renders_interactively_with_full_panel_titles():
    obj = render(
        PlotSpec(
            family="field_map_grid",
            items=_depth_stat_items(),
            options=_depth_stat_options(),
        ),
        renderer="holoviews",
    )
    assert _bokeh_titles(obj) == [
        "mean · 0 m",
        "mean · 200 m",
        "std of monthly means · 0 m",
        "std of monthly means · 200 m",
    ]


def test_column_bars_need_one_scale_per_column():
    spec = PlotSpec(
        family="field_map_grid",
        items=_depth_stat_items(),
        options=_depth_stat_options(shared_limits=False),
    )
    for renderer in ("matplotlib", "holoviews"):
        with pytest.raises(ValueError, match="one scale"):
            render(spec, renderer=renderer)
    with pytest.raises(ValueError, match="faceted grid"):
        render(
            PlotSpec(
                family="field_map_grid",
                items=_depth_stat_items(),
                options={"colorbars": "column", "tiles": False},
            ),
            renderer="matplotlib",
        )


def test_the_transposed_depth_statistic_grid_titles_by_its_margins_too():
    fig = render(
        PlotSpec(
            family="field_map_grid",
            items=_depth_stat_items(),
            options=_depth_stat_options(
                rows="statistic", cols="depth", shared_limits="depth"
            ),
        ),
        renderer="matplotlib",
    )
    assert _mpl_titles(fig) == ["0 m", "200 m"]
    labels = [
        ax._osk_row_label.get_text()
        for ax in fig.axes[:4]
        if hasattr(ax, "_osk_row_label")
    ]
    assert labels == ["mean", "std of monthly means"]


def _stub_fields(monkeypatch):
    """Stub a source whose iron field is cut at the depths ``select`` asks for."""
    from ocean_skill import comparison

    def fake(source, variable, select, aggregate, **kwargs):
        std = "reduce" in str(aggregate)
        asked = select["depth"]
        depths = list(asked) if isinstance(asked, list) else [asked]
        base = _pacific(0.0).expand_dims(depth=[float(d) for d in depths]).copy()
        base = base + np.array(depths, dtype=float).reshape(-1, 1, 1) / 100.0
        if not isinstance(asked, list):
            base = base.isel(depth=0)
        base.attrs = {"units": "mmol m-3", "statistic": "std" if std else "mean"}
        return base.rename("iron"), None

    monkeypatch.setattr(comparison, "prepare_source", fake)


_AGGREGATES = [
    {"time": "mean"},
    {"time": [{"groupby": "month", "reduce": "mean"}, {"reduce": "std", "ddof": 1}]},
]


def test_a_fieldset_over_levels_and_statistics_draws_the_grid(monkeypatch):
    from ocean_skill.field import field

    _stub_fields(monkeypatch)
    fs = field("run", "iron", select={"depth": [0, 200]}, aggregate=_AGGREGATES)
    fig = fs.plot(tiles=False)
    gridspec = fig.axes[0].get_gridspec()
    assert (gridspec.nrows, gridspec.ncols) == (2, 2)
    assert _mpl_titles(fig) == ["mean", "std of monthly means"]
    assert len(fig.axes) == 4 + 2
    assert fig._suptitle.get_text() == "iron"

    obj = fs.plot(renderer="holoviews", tiles=False)
    assert "mean · 0 m" in _bokeh_titles(obj)


def test_a_fieldset_that_does_not_factor_keeps_the_free_grid(monkeypatch):
    from ocean_skill.field import field

    _stub_fields(monkeypatch)
    # statistics only (one level each): the free grid, titled by statistic
    fig = field("run", "iron", select={"depth": 0}, aggregate=_AGGREGATES).plot(
        tiles=False
    )
    assert _mpl_titles(fig) == ["mean", "std of monthly means"]
    # a bar per panel, not one per column
    assert len([ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]) == 2
    assert not any(getattr(ax, "_osk_row_label", None) for ax in fig.axes)


def test_depth_by_statistic_is_left_to_a_caller_who_arranged_the_grid_themselves(
    monkeypatch,
):
    from ocean_skill.field import field

    _stub_fields(monkeypatch)
    fs = field("run", "iron", select={"depth": [0, 200]}, aggregate=_AGGREGATES)
    fig = fs.plot(tiles=False, ncols=1, shared_limits=False)
    assert fig.axes[0].get_gridspec().ncols == 1
    assert len([ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]) == 4
    assert _mpl_titles(fig) == [
        "mean · 0 m",
        "mean · 200 m",
        "std of monthly means · 0 m",
        "std of monthly means · 200 m",
    ]


# -- 3: stacked-row spacing and colorbar labels --------------------------------------


def _panel_axes(fig):
    return [ax for ax in fig.axes if ax.get_title() and ax.get_title() != "difference"]


def test_stacked_rows_are_evenly_spaced_whatever_their_colorbars_are():
    """Rows with extension arrows on their bars used to get extra room.

    So the gaps between rows differed for a reason nothing on the page explained.
    """
    items = [_row(f"row{i}", 1.0 + 0.7 * i) for i in range(6)]
    fig = render(_grid(items), renderer="matplotlib")
    firsts = [ax for ax in fig.axes if ax.get_title() == "model"]
    tops = sorted((ax.get_position().y1 for ax in firsts), reverse=True)
    steps = np.diff(tops)
    assert len(steps) == 5
    assert np.ptp(steps) < 1e-6, steps


def test_a_long_difference_bar_label_is_stacked_to_fit_its_bar():
    """Stack ``test − reference [units]`` when it is longer than its bar.

    A rotated label longer than its bar runs into the next row's; as two lines it fits.
    """
    items = [_row(f"row{i}", 1.0 + i, units="mmol m-3") for i in range(5)]
    fig = render(_grid(items, size=(7.5, 6.5)), renderer="matplotlib")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bars = [
        ax
        for ax in fig.axes
        if getattr(ax, "_osk_cbar_parents", None)
        and "test" in ax.yaxis.label.get_text()
    ]
    assert len(bars) == 5
    for bar in bars:
        assert "\n[" in bar.yaxis.label.get_text()
        assert (
            bar.yaxis.label.get_window_extent(renderer).height
            <= bar.get_window_extent(renderer).height + 1.0
        )
    boxes = sorted(
        (b.yaxis.label.get_window_extent(renderer) for b in bars), key=lambda b: -b.y1
    )
    for upper, lower in itertools.pairwise(boxes):
        assert upper.y0 >= lower.y1 - 1.0, "bar labels of adjacent rows overlap"


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_a_dateline_grid_of_rows_renders(renderer):
    items = [_row(f"row{i}", 1.0 + i) for i in range(3)]
    out = render(
        _grid(items, tiles=False) if renderer == "holoviews" else _grid(items),
        renderer=renderer,
    )
    if renderer == "holoviews":
        assert len(_bokeh_titles(out)) == 9


# -- 4: difference-only grids --------------------------------------------------------


def _diff_items(n=4, spread=(0.5, 1.0, 1.5, 2.0)):
    items = []
    for i in range(n):
        item = _row(
            f"{2011 + i // 2} {('DJF', 'JJA')[i % 2]}", spread[i % len(spread)] + i
        )
        # differences that spread out (so a quantile of them means something) and one
        # outlier, so the 98th percentile and the maximum differ
        rng = np.random.default_rng(i)
        diff = item["aligned"]["difference"].copy()
        diff[:] = rng.normal(0.0, spread[i % len(spread)], diff.shape)
        diff[0, 0] = 50.0
        item["aligned"] = item["aligned"].assign(difference=diff)
        items.append(item)
    return items


def _limit_of(fig):
    from matplotlib.collections import QuadMesh

    mesh = next(c for c in fig.axes[0].collections if isinstance(c, QuadMesh))
    return mesh.norm.vmax


def test_panels_all_keeps_three_columns_per_row():
    fig = render(_grid(_diff_items(2)), renderer="matplotlib")
    assert (
        len(_panel_axes(fig))
        + len([a for a in fig.axes if a.get_title() == "difference"])
        == 6
    )
    assert _mpl_titles(fig).count("difference") == 2


def test_difference_panels_share_one_symmetric_colorbar_and_wrap_into_ncols():
    items = _diff_items(4)
    fig = render(_grid(items, panels="difference", ncols=2), renderer="matplotlib")
    assert _mpl_titles(fig) == [it["label"] for it in items]
    gridspec = fig.axes[0].get_gridspec()
    assert (gridspec.nrows, gridspec.ncols) == (2, 2)
    assert len(fig.axes) == 4 + 1  # one colorbar for all four panels
    from matplotlib.collections import QuadMesh

    norms = {
        (m.norm.vmin, m.norm.vmax)
        for ax in fig.axes[:4]
        for m in ax.collections
        if isinstance(m, QuadMesh)
    }
    assert len(norms) == 1
    ((lo, hi),) = norms
    assert lo == -hi


def test_diff_quantile_sets_how_far_the_difference_scale_reaches():
    items = _diff_items(4)
    default = _limit_of(
        render(_grid(items, panels="difference"), renderer="matplotlib")
    )
    widest = _limit_of(
        render(
            _grid(items, panels="difference", diff_quantile=1.0), renderer="matplotlib"
        )
    )
    narrow = _limit_of(
        render(
            _grid(items, panels="difference", diff_quantile=0.5), renderer="matplotlib"
        )
    )
    assert narrow < default < widest
    assert widest >= 50.0  # the outlier


def test_robust_narrows_the_difference_scale_unless_diff_quantile_says_otherwise():
    items = _diff_items(4)
    plain = _limit_of(render(_grid(items, panels="difference"), renderer="matplotlib"))
    robust = _limit_of(
        render(_grid(items, panels="difference", robust=True), renderer="matplotlib")
    )
    pinned = _limit_of(
        render(
            _grid(items, panels="difference", robust=True, diff_quantile=0.98),
            renderer="matplotlib",
        )
    )
    assert robust < plain
    assert pinned == plain


def test_robust_and_diff_quantile_reach_the_stacked_rows_difference_panels():
    from matplotlib.collections import QuadMesh

    def diff_limit(**options):
        fig = render(_grid(_diff_items(2), **options), renderer="matplotlib")
        axes = [ax for ax in fig.axes if ax.get_title() == "difference"]
        mesh = next(c for c in axes[0].collections if isinstance(c, QuadMesh))
        return mesh.norm.vmax

    assert diff_limit(robust=True) < diff_limit() < diff_limit(diff_quantile=1.0)
    assert diff_limit(shared_limits=True, robust=True) < diff_limit(shared_limits=True)


@pytest.mark.parametrize("bad", [0.0, 1.5, -0.1])
def test_diff_quantile_must_be_a_fraction(bad):
    with pytest.raises(ValueError, match="diff_quantile"):
        render(
            _grid(_diff_items(2), panels="difference", diff_quantile=bad),
            renderer="matplotlib",
        )


def test_panels_and_ncols_are_validated():
    for renderer in ("matplotlib", "holoviews"):
        with pytest.raises(ValueError, match='panels must be "all"'):
            render(_grid(_diff_items(2), panels="nope"), renderer=renderer)
        with pytest.raises(ValueError, match="ncols="):
            render(_grid(_diff_items(2), ncols=2), renderer=renderer)


def test_difference_panels_warn_when_they_mix_variables():
    items = _diff_items(2)
    items[1] = {**items[1], "standard_name": "sea_water_practical_salinity"}
    with pytest.warns(UserWarning, match="different variables"):
        render(_grid(items, panels="difference"), renderer="matplotlib")


def test_difference_panels_interactively():
    items = _diff_items(4)
    obj = render(
        _grid(items, panels="difference", ncols=2, tiles=False), renderer="holoviews"
    )
    assert _bokeh_titles(obj) == sorted(it["label"] for it in items)
    (limits,) = _color_mappers(obj)
    assert limits[0] == -limits[1]
    wide = render(
        _grid(items, panels="difference", ncols=2, tiles=False, diff_quantile=1.0),
        renderer="holoviews",
    )
    (wide_limits,) = _color_mappers(wide)
    assert wide_limits[1] > limits[1]
    robust = render(
        _grid(items, panels="difference", tiles=False, robust=True),
        renderer="holoviews",
    )
    (robust_limits,) = _color_mappers(robust)
    assert robust_limits[1] < limits[1]


def _section_row_item(label, offset, *, corners=None):
    n_along, n_z = 12, 6
    along = np.linspace(0.0, 220.0, n_along)
    z = -np.array([0, 10, 25, 50, 100, 200], dtype="float64")
    values = 5.0 + np.linspace(0, 1, n_z * n_along).reshape(n_z, n_along)
    test = xr.DataArray(
        values + offset,
        dims=("z", "along"),
        coords={
            "z": ("z", z),
            "along": ("along", along, {"units": "km"}),
            "lon": ("along", np.linspace(260.0, 262.0, n_along)),
            "lat": ("along", np.linspace(20.0, 21.0, n_along)),
        },
    )
    if corners is not None:
        test["along"].attrs["boundary_corner_km"] = corners
    ref = test - offset
    return {
        "aligned": {"test": test, "reference": ref, "difference": test - ref},
        "units": "mmol m-3",
        "standard_name": None,
        "depth": None,
        "time": None,
        "metrics": {"bias": 0.1},
        "labels": ("model", "woa"),
        "row_label": label,
        "label": label,
    }


def test_section_difference_grids_in_both_renderers():
    items = [_section_row_item(f"s{i}", 0.5 * (i + 1)) for i in range(3)]
    spec = _grid(items, family="section_row", panels="difference", ncols=3)
    fig = render(spec, renderer="matplotlib")
    assert _mpl_titles(fig) == ["s0", "s1", "s2"]
    assert len(fig.axes) == 3 + 1
    obj = render(spec, renderer="holoviews")
    assert _bokeh_titles(obj) == ["s0", "s1", "s2"]
    (limits,) = _color_mappers(obj)
    assert limits[0] == -limits[1]
    # without panels= the same items are the stacked rows
    stacked = render(_grid(items, family="section_row"), renderer="matplotlib")
    assert _mpl_titles(stacked).count("difference") == 3


def test_a_lone_section_comparison_can_ask_for_its_difference_alone():
    item = _section_row_item("only", 0.5)
    spec = _grid([item], family="section_row", panels="difference")
    assert _mpl_titles(render(spec, renderer="matplotlib")) == ["only"]
    assert _bokeh_titles(render(spec, renderer="holoviews")) == ["only"]


def test_a_comparison_set_of_one_draws_its_difference_through_the_grid():
    from ocean_skill.comparison import ComparisonSet

    class _Comparison:
        family = "field_row"
        label = "lone"
        test_name = "model"
        reference_name = "woa"
        aligned = xr.Dataset(attrs={})

        def _require_drawable(self):
            pass

        def as_item(self):
            return _row("lone", 1.0)

        def metrics(self):
            return {}

    comparison_set = ComparisonSet([_Comparison()])
    fig = comparison_set.plot(panels="difference", domain=None)
    assert _mpl_titles(fig) == ["lone"]


# -- boundary-strip transects --------------------------------------------------------


def _vertical_lines(fig):
    return [
        line
        for ax in fig.axes
        for line in ax.get_lines()
        if line.get_gid() == "boundary_corner"
    ]


def test_a_section_draws_a_line_at_each_boundary_corner():
    corners = [60.0, 150.0]
    item = _section_row_item("strip", 0.5, corners=corners)
    spec = PlotSpec(
        family="section_row", items=[item], options={"section_x": "distance"}
    )
    fig = render(spec, renderer="matplotlib")
    lines = _vertical_lines(fig)
    # one per corner on each of the three panels
    assert len(lines) == 3 * len(corners)
    assert sorted({float(line.get_xdata()[0]) for line in lines}) == corners

    import holoviews as hv

    obj = render(spec, renderer="holoviews")
    vlines = obj.traverse(lambda x: x, [hv.VLine])
    assert sorted({float(v.data) for v in vlines}) == corners
    _bokeh(obj)  # renders


def test_a_section_without_the_attribute_draws_no_corner_lines():
    item = _section_row_item("plain", 0.5)
    fig = render(
        PlotSpec(family="section_row", items=[item], options={"section_x": "distance"}),
        renderer="matplotlib",
    )
    assert _vertical_lines(fig) == []
    import holoviews as hv

    obj = render(
        PlotSpec(family="section_row", items=[item], options={"section_x": "distance"}),
        renderer="holoviews",
    )
    assert obj.traverse(lambda x: x, [hv.VLine]) == []


def test_corners_outside_the_drawn_extent_are_not_drawn():
    item = _section_row_item("strip", 0.5, corners=[60.0, 900.0])
    fig = render(
        PlotSpec(family="section_row", items=[item], options={"section_x": "distance"}),
        renderer="matplotlib",
    )
    assert {float(line.get_xdata()[0]) for line in _vertical_lines(fig)} == {60.0}


# -- 5: a row label longer than its row must not crush the maps -----------------------


def _map_widths(fig) -> list[float]:
    width = fig.get_size_inches()[0]
    return [
        ax.get_position().width * width for ax in fig.axes if hasattr(ax, "projection")
    ]


def test_a_long_row_label_does_not_shrink_the_maps(recwarn):
    """The row label is rotated, so a long one is taller than the panel it names.

    Counted by constrained_layout like any child text, that overhang was reserved above
    and below the axes: the 8.5in chlorophyll page ("std of monthly means" row) drew its
    maps 0.23in wide.
    """
    items = [
        _row("chlorophyll a mean", 0.5),
        _row("chlorophyll a std of monthly means", 0.2),
    ]
    fig = render(_grid(items), renderer="matplotlib")
    assert min(_map_widths(fig)) > 1.0
    assert not [w for w in recwarn if "maps are only" in str(w.message)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        label = getattr(ax, "_osk_row_label", None)
        if label is not None:  # wrapped to sit beside its own row
            assert label.get_window_extent(renderer).height <= (
                ax.get_window_extent(renderer).height * 1.02
            )


def test_a_long_row_label_is_wrapped_at_word_boundaries():
    items = [_row("sea surface chlorophyll a std of monthly means", 0.2)] * 2
    fig = render(_grid(items), renderer="matplotlib")
    texts = [
        ax._osk_row_label.get_text()
        for ax in fig.axes
        if getattr(ax, "_osk_row_label", None) is not None
    ]
    assert all("\n" in t for t in texts)
    assert all(
        " ".join(t.split()) == "sea surface chlorophyll a std of monthly means"
        for t in texts
    )


def test_a_short_row_label_is_left_on_one_line():
    fig = render(_grid([_row("temp", 0.5), _row("salt", 0.2)]), renderer="matplotlib")
    assert all(
        "\n" not in ax._osk_row_label.get_text()
        for ax in fig.axes
        if getattr(ax, "_osk_row_label", None) is not None
    )


# -- 6: panels fill the canvas ---------------------------------------------------------
#
# The figure's width is fixed (a report page); its height was an estimate made before
# any text was measured. A short estimate left every aspect-locked map limited by its
# height, sitting in a wider cell, so the width came back as gutters between the columns
# -- 1.95in maps in 2.3in cells, and 1.3in maps in a 2x4 difference grid. The figure is
# now sized from the settled layout, so the panels take their whole cell and the height
# follows.


def _map_panels(fig):
    """Return the figure's drawn panels: every visible axes that is not a colorbar."""
    return [
        ax
        for ax in fig.axes
        if ax.get_visible()
        and not getattr(ax, "_osk_cbar_parents", None)
        and ax.get_subplotspec() is not None
    ]


def _fill(fig):
    """Return the worst panel's ``(width, height)`` as a share of its layout cell."""
    fig.canvas.draw()
    shares = []
    for ax in _map_panels(fig):
        cell, drawn = ax.get_position(original=True), ax.get_position()
        shares.append((drawn.width / cell.width, drawn.height / cell.height))
    return min(s[0] for s in shares), min(s[1] for s in shares)


def _widest_panel(fig):
    """Return the widest panel's width in inches."""
    fig.canvas.draw()
    width = fig.get_size_inches()[0]
    return max(ax.get_position().width for ax in _map_panels(fig)) * width


def _text_boxes(fig):
    """Named window extents of the text a figure's layout is responsible for."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    if fig._suptitle is not None and fig._suptitle.get_text():
        out.append(("suptitle", fig._suptitle.get_window_extent(renderer)))
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        if getattr(ax, "_osk_cbar_parents", None):
            horizontal = ax._osk_cbar_horizontal
            axis = ax.xaxis if horizontal else ax.yaxis
            texts = [axis.label, *(axis.get_ticklabels())]
            out += [
                (f"cbar{i}:{t.get_text()!r}", t.get_window_extent(renderer))
                for t in texts
                if t.get_visible() and t.get_text()
            ]
            continue
        if ax.get_title():
            out.append((f"title{i}", ax.title.get_window_extent(renderer)))
        label = getattr(ax, "_osk_row_label", None)
        if label is not None:
            out.append((f"row{i}", label.get_window_extent(renderer)))
        for artist in ax.artists:
            for t in (getattr(artist, "left_label_artists", None) or []) + (
                getattr(artist, "bottom_label_artists", None) or []
            ):
                if t.get_visible() and t.get_text():
                    box = t.get_window_extent(renderer)
                    out.append((f"grid{i}:{t.get_text()}", box))
    return out


def _assert_no_text_overlaps(fig):
    boxes = _text_boxes(fig)
    assert boxes
    for (a, box_a), (b, box_b) in itertools.combinations(boxes, 2):
        overlap_w = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
        overlap_h = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
        assert not (overlap_w > 1.0 and overlap_h > 1.0), f"{a} overlaps {b}"
    width, height = fig.canvas.get_width_height()
    for name, box in boxes:
        assert box.x0 >= -1 and box.y0 >= -1, f"{name} is off canvas"
        assert box.x1 <= width + 1 and box.y1 <= height + 1, f"{name} is off canvas"


def _four_row_grid(**options):
    rows = [
        _row(f"{depth} {stat}", 1.0 + 0.7 * i)
        for i, (depth, stat) in enumerate(
            [("0 m", "mean"), ("0 m", "std"), ("200 m", "mean"), ("200 m", "std")]
        )
    ]
    return render(_grid(rows, **options), renderer="matplotlib")


def test_a_compare_grids_maps_fill_their_columns():
    fig = _four_row_grid()
    # the maps take their whole cell in both directions: nothing is left over for
    # gutters
    # beyond the room the tick labels and colorbars asked for
    assert min(_fill(fig)) >= 0.97
    # and that cell is most of a third of the page (it was 1.95in of 2.83in)
    assert _widest_panel(fig) >= 0.74 * fig.get_size_inches()[0] / 3
    _assert_no_text_overlaps(fig)


def test_the_height_follows_the_larger_maps():
    from ocean_skill.plot.typography import SUPTITLE_ALLOWANCE

    fig = _four_row_grid()
    width, height = fig.get_size_inches()
    assert width == pytest.approx(8.5)
    # taller than the pre-measurement estimate (5.9in), and inside the page
    assert 6.0 < height <= 11.0 - SUPTITLE_ALLOWANCE
    # the maps are as tall as their width asks: no blank height inside the cells
    maps = _map_panels(fig)
    assert maps
    for ax in maps:
        drawn = ax.get_position()
        assert drawn.width * width / (drawn.height * height) == pytest.approx(
            maps[0].get_position().width * width
            / (maps[0].get_position().height * height)
        )


def test_an_explicit_figsize_is_left_exactly_as_asked():
    fig = _four_row_grid(figsize=(8.5, 5.0))
    assert tuple(fig.get_size_inches()) == pytest.approx((8.5, 5.0))


def test_a_capped_canvas_bounds_how_tall_the_figure_grows():
    from ocean_skill.plot.typography import SUPTITLE_ALLOWANCE

    fig = _four_row_grid(size=(8.5, 5.5))
    assert fig.get_size_inches()[1] <= 5.5 - SUPTITLE_ALLOWANCE + 1e-6
    _assert_no_text_overlaps(fig)


def test_a_difference_only_grid_fills_its_columns():
    fig = render(
        _grid(_diff_items(8), panels="difference", ncols=4), renderer="matplotlib"
    )
    assert min(_fill(fig)) >= 0.97
    # four columns on the page: the maps were 1.28in of a 2.1in column
    assert _widest_panel(fig) >= 0.8 * fig.get_size_inches()[0] / 4
    _assert_no_text_overlaps(fig)


def test_a_depth_by_statistic_grid_fills_its_columns():
    fig = render(
        PlotSpec(
            family="field_map_grid",
            items=_depth_stat_items(),
            options=_depth_stat_options(),
        ),
        renderer="matplotlib",
    )
    assert min(_fill(fig)) >= 0.97
    _assert_no_text_overlaps(fig)


def test_stacked_sections_reach_their_design_shape_and_the_page_width():
    from ocean_skill.plot.typography import SECTION_ASPECT

    items = [_section_row_item(f"s{i}", 0.5 * (i + 1)) for i in range(3)]
    fig = render(_grid(items, family="section_row"), renderer="matplotlib")
    fig.canvas.draw()
    width, height = fig.get_size_inches()
    panels = [ax for ax in fig.axes if not getattr(ax, "_osk_cbar_parents", None)]
    assert len(panels) == 9
    for ax in panels:
        box = ax.get_position()
        # no flatter than the shape sections are designed to have (they were 3.4:1)
        assert box.width * width / (box.height * height) <= SECTION_ASPECT * 1.05
    # a column of three panels and two bars needs ~2in for its decorations
    assert sum(ax.get_position().width for ax in panels[:3]) * width >= 6.0
    _assert_no_text_overlaps(fig)


def test_interactive_frames_already_follow_the_map_aspect():
    """Nothing to reclaim interactively: a bokeh frame is sized, not a share of a cell.

    Each panel is its own figure whose frame is the map's own shape in pixels, with a
    fixed border for its decorations -- there is no equal-width cell for a letterboxed
    map to sit centred in, so no gutter that grows with it.
    """
    from bokeh.plotting import figure

    from ocean_skill.plot.typography import clamp_aspect

    rows = [_row("a", 1.0), _row("b", 2.0)]
    obj = render(_grid(rows, tiles=False), renderer="holoviews")
    frames = [
        (f.frame_width, f.frame_height, f.min_border_left, f.min_border_right)
        for f in _bokeh(obj).select({"type": figure})
    ]
    assert len(frames) == 6
    assert len({f[:2] for f in frames}) == 1
    width, height, left, right = frames[0]
    assert width / height == pytest.approx(
        clamp_aspect(width / height), rel=0.02
    )
    assert left + right <= 0.1 * width
