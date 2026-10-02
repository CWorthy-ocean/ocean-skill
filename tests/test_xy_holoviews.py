"""Tests for the ``XY`` family's interactive renderer and its agreement with the static.

The standing rule is that a plot change lands in both renderers and a test asserts the
two **agree**, so most checks here build one set of hand-made items, draw it both ways
and compare what each drew: the panels, the legend's rows and colours, the shared
``color_by`` scale (point by point), the contour levels, the annotations, the limits.
Items are shaped as ``XY._items()`` builds them (``PlotSpec``'s docstring); the static
renderer's own checks live in ``tests/test_xy_renderers.py``.
"""

from __future__ import annotations

import warnings
import zlib

import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import same_color, to_hex
from matplotlib.contour import ContourSet

from ocean_skill.plot import style as _style
from ocean_skill.plot import xy as _xy
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

REGIONS = [
    "North West Pacific",
    "Subpolar Gyre",
    "California Current System",
    "South West Pacific",
    "South Pacific Gyre",
    "Peru Current",
]

MS_PER_DAY = 86_400_000.0


def _item(
    label: str = "ROMS",
    region: str | None = "North West Pacific",
    mark: str = "points",
    *,
    depth: bool = True,
    time: bool = True,
) -> dict:
    """One XY item, shaped as ``XY._items()`` builds it (Contract 1)."""
    n = 12 if mark == "line" else 200
    rng = np.random.default_rng(zlib.crc32(f"{label}|{region}|{mark}".encode()))
    d = np.sort(rng.uniform(5, 1000, n)) if mark == "line" else rng.uniform(5, 1000, n)
    return {
        "label": label,
        "region": region,
        "mark": mark,
        "x": 34.5 + 0.4 * np.cos(d / 300) + rng.normal(0, 0.01, n),
        "y": 10.0 + 8 * np.exp(-d / 300) + rng.normal(0, 0.05, n),
        "depth": d if depth else None,
        "time": (
            np.datetime64("2020-01-01")
            + rng.integers(0, 365, n).astype("timedelta64[D]")
            if time
            else None
        ),
        "x_name": "salinity",
        "x_units": None,
        "x_standard_name": "sea_water_practical_salinity",
        "y_name": "temperature",
        "y_units": "degC",
        "y_standard_name": "sea_water_potential_temperature",
        "x_role": "salinity",
        "y_role": "temperature",
        "lon": 155.8,
        "lat": 21.0,
        "region_note": None,
        "source": "test",
    }


def _ts_items(regions=REGIONS[:1], members=("ROMS", "WOA23", "GLORYS12")) -> list[dict]:
    """Build the Fig. 7 trio per region: a model cloud, then two profiles."""
    return [
        _item(label, region, "points" if label == "ROMS" else "line")
        for region in regions
        for label in members
    ]


def _interactive(items, **opts):
    return render(PlotSpec("XY", items, opts), renderer="holoviews")


def _static(items, **opts):
    return render(PlotSpec("XY", items, opts), renderer="matplotlib")


def _elements(obj, kind, *, keys: bool = False) -> list:
    """Return the data ``kind`` elements of ``obj`` (``keys``: the legend proxies)."""
    import holoviews as hv

    found = obj.traverse(lambda x: x, [getattr(hv, kind)])
    return [e for e in found if (e.group == "Legend") == keys]


def _figures(obj) -> dict:
    """Return the rendered bokeh figures of ``obj`` by panel title."""
    import holoviews as hv
    from bokeh.plotting import figure

    plot = hv.render(obj, backend="bokeh")
    return {f.title.text: f for f in plot.select({"type": figure})}


def _panel_axes(fig):
    return [
        ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"
    ]


def _scatter(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _cloud(figure):
    """Return the renderer of the 200-point model cloud among ``figure``'s glyphs.

    The legend proxies are one-point Scatters, so the size picks the data out.
    """
    (cloud,) = [
        r
        for r in figure.renderers
        if type(r.glyph).__name__ == "Scatter" and len(r.data_source.data["x"]) == 200
    ]
    return cloud


def _bokeh_colours(renderer) -> list[str]:
    """Return the hex colour bokeh gives each point of a ``color_by`` Scatter."""
    fill = renderer.glyph.fill_color
    mapper, values = fill.transform, renderer.data_source.data[fill.field]
    n = len(mapper.palette)
    norm = (np.asarray(values, dtype="float64") - mapper.low) / (
        mapper.high - mapper.low
    )
    index = np.clip(np.floor(norm * n).astype(int), 0, n - 1)
    return [mapper.palette[i].lower() for i in index]


def _static_colours(cloud) -> list[str]:
    """Return the hex colour matplotlib gives each point of a ``color_by`` scatter."""
    return [to_hex(c) for c in cloud.cmap(cloud.norm(cloud.get_array()))]


# --- panels and elements --------------------------------------------------------------


def test_six_regions_make_six_panels_in_a_three_column_grid():
    items = _ts_items(REGIONS)
    out = _interactive(items)
    assert len(out) == 6
    assert out.shape == (2, 3)
    assert set(_figures(out)) == set(REGIONS)
    # the same panels the static renderer draws
    static = [ax.get_title() for ax in _panel_axes(_static(items))]
    layout = _xy.compose(items)
    assert static == [p.title for p in layout.panels] == REGIONS


def test_ncols_and_nrows_wrap_the_grid():
    items = _ts_items(REGIONS[:3])
    assert _interactive(items, ncols=2).shape == (2, 2)
    assert _interactive(items, nrows=3).shape == (3, 1)


def test_a_panel_draws_one_cloud_under_two_profiles():
    import holoviews as hv

    (panel,) = _interactive(_ts_items(), annotations={"STSW": (34.8, 12.0)}).values()
    assert len(_elements(panel, "Points")) == 1
    assert len(_elements(panel, "Curve")) == 2
    assert len(_elements(panel, "Text")) == 1
    assert not _elements(panel, "Contours")
    # dots first, then lines, then the text, then the (empty) legend proxies
    kinds = [type(e).__name__ for e in panel.values()]
    assert kinds == ["Points", "Curve", "Curve", "Text", "Points", "Curve", "Curve"]
    assert [e.group for e in panel.values()][-3:] == ["Legend"] * 3
    assert isinstance(panel, hv.Overlay)


def test_the_data_elements_carry_the_composed_arrays():
    items = _ts_items()
    layout = _xy.compose(items)
    (panel,) = _interactive(items).values()
    cloud, *lines = layout.panels[0].items
    (points,) = _elements(panel, "Points")
    np.testing.assert_array_equal(points.dimension_values("x"), cloud.x)
    np.testing.assert_array_equal(points.dimension_values("y"), cloud.y)
    np.testing.assert_array_equal(points.dimension_values("depth"), cloud.depth)
    assert points.dimension_values("time").dtype.kind == "M"
    for curve, line in zip(_elements(panel, "Curve"), lines, strict=True):
        # a Curve keeps the order it is given (a T-S profile is not monotonic in x)
        np.testing.assert_array_equal(curve.dimension_values("x"), line.x)
        np.testing.assert_array_equal(curve.dimension_values("y"), line.y)
        np.testing.assert_array_equal(curve.dimension_values("depth"), line.depth)


def test_a_cloud_with_no_depth_or_time_still_draws():
    items = [_item("ROMS", depth=False, time=False)]
    (panel,) = _interactive(items).values()
    (points,) = _elements(panel, "Points")
    assert [d.name for d in points.vdims] == []
    assert len(points) == 200


def test_a_blank_cell_is_an_empty_placeholder(monkeypatch):
    import dataclasses

    import holoviews as hv

    real = _xy.compose

    def blank_second(items, **kwargs):
        layout = real(items, **kwargs)
        panels = list(layout.panels)
        panels[1] = dataclasses.replace(panels[1], blank=True)
        return dataclasses.replace(layout, panels=tuple(panels))

    monkeypatch.setattr(_xy, "compose", blank_second)
    out = _interactive(_ts_items(REGIONS[:3]))
    assert len(out) == 3
    assert sum(isinstance(v, hv.Empty) for v in out.values()) == 1


# --- density --------------------------------------------------------------------------


def test_density_contour_levels_equal_the_static_levels():
    pytest.importorskip("gsw")
    items = _ts_items(REGIONS[:3])
    interactive = _interactive(items, density=True)
    static = _static(items, density=True)
    for ax, panel in zip(_panel_axes(static), interactive.values(), strict=True):
        (contours,) = [c for c in ax.collections if isinstance(c, ContourSet)]
        (lines,) = _elements(panel, "Contours")
        # the Contours carry nan between a level's separate lines
        drawn = lines.dimension_values("sigma")
        np.testing.assert_allclose(
            np.unique(drawn[np.isfinite(drawn)]), contours.levels
        )
        assert 6 <= len(contours.levels) <= 14


def test_explicit_density_levels_are_drawn_as_given():
    pytest.importorskip("gsw")
    levels = [25.0, 26.0, 40.0]  # the last is far outside this panel's density range
    (panel,) = _interactive(_ts_items(), density=levels).values()
    (lines,) = _elements(panel, "Contours")
    drawn = lines.dimension_values("sigma")
    # a level that crosses nothing has no line, here as in the static ContourSet
    np.testing.assert_allclose(np.unique(drawn[np.isfinite(drawn)]), [25.0, 26.0])
    (ax,) = _panel_axes(_static(_ts_items(), density=levels))
    (contours,) = [c for c in ax.collections if isinstance(c, ContourSet)]
    assert [
        lv
        for lv, segs in zip(contours.levels, contours.allsegs, strict=True)
        if len(segs)
    ] == [25.0, 26.0]


def test_density_lines_are_light_grey_unlabelled_and_beneath_everything():
    pytest.importorskip("gsw")
    (panel,) = _interactive(_ts_items(), density=True).values()
    (lines,) = _elements(panel, "Contours")
    assert same_color(lines.opts.get("style").kwargs["color"], "0.6")
    assert lines.opts.get("plot").kwargs["show_legend"] is False
    assert lines.opts.get("plot").kwargs["color_index"] is None
    assert type(next(iter(panel.values()))).__name__ == "Contours"
    (figure,) = _figures(_interactive(_ts_items(), density=True)).values()
    # no legend row for them: the key lists the three members and only those
    assert [i.label.value for i in figure.legend[0].items] == [
        "ROMS",
        "WOA23",
        "GLORYS12",
    ]


def test_density_for_other_variables_is_a_clear_error():
    nutrients = [{**_item(), "x_role": None, "y_role": None, "x_name": "phosphate"}]
    with pytest.raises(ValueError, match="salinity and the other temperature"):
        _interactive(nutrients, density=True)
    assert _interactive(nutrients)  # the same plot without density draws


# --- color_by -------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["depth", "time"])
def test_color_by_colours_every_point_as_the_static_scatter_does(field):
    items = _ts_items(REGIONS[:3])
    static = _static(items, color_by=field)
    interactive = _interactive(items, color_by=field)
    layout = _xy.compose(items, color_by=field)
    figures = _figures(interactive)
    for ax, panel in zip(_panel_axes(static), layout.panels, strict=True):
        (cloud,) = _scatter(ax)
        assert _bokeh_colours(_cloud(figures[panel.title])) == _static_colours(cloud)


def test_color_by_channel_holds_the_composed_values_and_one_shared_scale():
    items = _ts_items(REGIONS[:3])
    layout = _xy.compose(items, color_by="depth")
    scale = layout.colorbar
    figures = _figures(_interactive(items, color_by="depth"))
    mappers = set()
    for panel in layout.panels:
        cloud = _cloud(figures[panel.title])
        np.testing.assert_array_equal(
            cloud.data_source.data["color"], panel.items[0].color_values
        )
        mapper = cloud.glyph.fill_color.transform
        mappers.add((mapper.low, mapper.high))
    # one scale for the figure, surface-at-top: the clim runs (deepest, shallowest)
    assert mappers == {(scale.vmax, scale.vmin)}


def test_a_time_scale_is_in_milliseconds_and_the_bar_reads_dates():
    from bokeh.models import ColorBar, DatetimeTickFormatter

    items = _ts_items(REGIONS[:2])
    layout = _xy.compose(items, color_by="time")
    scale = layout.colorbar
    figures = _figures(_interactive(items, color_by="time"))
    for panel in layout.panels:
        cloud = _cloud(figures[panel.title])
        np.testing.assert_allclose(
            cloud.data_source.data["color"], panel.items[0].color_values * MS_PER_DAY
        )
        mapper = cloud.glyph.fill_color.transform
        assert (mapper.low, mapper.high) == pytest.approx(
            (scale.vmin * MS_PER_DAY, scale.vmax * MS_PER_DAY)
        )
    bars = [b for f in figures.values() for b in f.right if isinstance(b, ColorBar)]
    (bar,) = bars
    assert isinstance(bar.formatter, DatetimeTickFormatter)


def test_there_is_exactly_one_colour_bar_on_the_last_panel_of_the_first_row():
    from bokeh.models import ColorBar

    out = _interactive(_ts_items(REGIONS), color_by="depth")
    holders = {
        title: [b for b in f.right if isinstance(b, ColorBar)]
        for title, f in _figures(out).items()
    }
    # a Layout has no shared bar, so one panel carries it
    assert {t for t, bars in holders.items() if bars} == {REGIONS[2]}
    drawing = [
        _elements(panel, "Points")[0].opts.get("plot").kwargs["colorbar"]
        for panel in out.values()
    ]
    assert drawing == [False, False, True, False, False, False]
    (cloud,) = _elements(list(out.values())[2], "Points")
    assert cloud.opts.get("plot").kwargs["clabel"] == "depth [m]"


def test_a_depth_bar_reads_surface_at_the_top():
    items = _ts_items()
    scale = _xy.compose(items, color_by="depth").colorbar
    assert scale.inverted
    (figure,) = _figures(_interactive(items, color_by="depth")).values()
    mapper = _cloud(figure).glyph.fill_color.transform
    # bokeh's bar puts `low` at the bottom: deepest there, so the surface is on top ...
    assert mapper.low > mapper.high
    # ... and the palette is reversed to match, so a value keeps its colour
    assert mapper.palette[0] == to_hex(scale.cmap(scale.cmap.N - 1)).lower()
    assert mapper.palette[-1] == to_hex(scale.cmap(0)).lower()


def test_colorbar_false_colours_without_a_bar():
    from bokeh.models import ColorBar

    items = _ts_items()
    (figure,) = _figures(_interactive(items, color_by="depth", colorbar=False)).values()
    assert not [b for b in figure.right if isinstance(b, ColorBar)]
    static = _static(items, color_by="depth", colorbar=False)
    (static_cloud,) = _scatter(_panel_axes(static)[0])
    assert _bokeh_colours(_cloud(figure)) == _static_colours(static_cloud)


def test_color_by_leaves_lines_solid_and_cmap_overrides():
    items = _ts_items()
    (panel,) = _interactive(items, color_by="depth", cmap="magma").values()
    scale = _xy.compose(items, color_by="depth", cmap="magma").colorbar
    (cloud,) = _elements(panel, "Points")
    palette = cloud.opts.get("style").kwargs["cmap"]
    # the colormap the caller named, reversed for the surface-at-top depth bar
    assert palette[0] == to_hex(scale.cmap(scale.cmap.N - 1))
    assert scale.cmap.name == "magma"
    for curve in _elements(panel, "Curve"):
        assert isinstance(curve.opts.get("style").kwargs["color"], str)
        assert curve.opts.get("style").kwargs["color"].startswith("#")


def test_color_by_without_depth_warns_and_stays_solid():
    items = [_item("ROMS", depth=False)]
    with pytest.warns(UserWarning, match="carries no depth values"):
        out = _interactive(items, color_by="depth")
    (panel,) = out.values()
    (cloud,) = _elements(panel, "Points")
    assert cloud.opts.get("style").kwargs["color"] == "#000000"


def _bars(out) -> list:
    """Return the bokeh colour bars among the figures of ``out``."""
    from bokeh.models import ColorBar

    return [
        b for f in _figures(out).values() for b in f.right if isinstance(b, ColorBar)
    ]


def _static_extend(items, **opts) -> str:
    """Return the ``extend`` the static colour bar was drawn with."""
    fig = _static(items, **opts)
    (bar,) = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    return bar._colorbar.extend


def test_a_depth_pin_clamps_deeper_dots_to_the_deepest_colour():
    items = _ts_items()
    layout = _xy.compose(items, color_by="depth", vmax=500)
    scale = layout.colorbar
    figures = _figures(_interactive(items, color_by="depth", vmax=500))
    (cloud,) = [_cloud(f) for f in figures.values()]
    mapper = cloud.glyph.fill_color.transform
    # surface at the top: the clim runs (deepest, shallowest), the palette reversed
    assert (mapper.low, mapper.high) == (500, scale.vmin)
    assert mapper.palette[0] == to_hex(scale.cmap(1.0)).lower()
    assert mapper.palette[-1] == to_hex(scale.cmap(0.0)).lower()
    # bokeh's mapper clamps beyond its ends unless given low_color/high_color
    assert mapper.low_color is None and mapper.high_color is None
    depth = layout.panels[0].items[0].color_values
    colours = np.array(_bokeh_colours(cloud))
    assert (depth > 500).any()
    assert set(colours[depth > 500]) == {to_hex(scale.cmap(1.0)).lower()}
    # and it is the colour the static scatter gives those dots
    static = _static(items, color_by="depth", vmax=500)
    (static_cloud,) = _scatter(_panel_axes(static)[0])
    assert list(colours) == _static_colours(static_cloud)


def _end_marks(bar) -> dict:
    """Return the forced ``≥``/``≤`` end labels among a bar's round tick labels."""
    return {
        value: text
        for value, text in bar.major_label_overrides.items()
        if text[:1] in ("≥", "≤")
    }


def test_a_depth_pin_labels_the_clipped_end_of_the_bar():
    out = _interactive(_ts_items(), color_by="depth", vmax=500)
    (bar,) = _bars(out)
    assert _end_marks(bar) == {500.0: "≥ 500"}
    assert type(bar.ticker).__name__ == "FixedTicker"
    assert 500.0 in bar.ticker.ticks
    # the unpinned top of the data (~5 m) snaps to a round 0 m; nothing past the pin
    assert all(0 <= t <= 500 for t in bar.ticker.ticks)
    (bar,) = _bars(_interactive(_ts_items(), color_by="depth", vmin=100, vmax=500))
    assert _end_marks(bar) == {100.0: "≤ 100", 500.0: "≥ 500"}


def test_a_bar_the_pins_leave_whole_marks_no_end():
    """Nothing clipped: round ticks (as on every bar), and no end marked."""
    for options in ({}, {"vmin": 0, "vmax": 5000}):
        (bar,) = _bars(_interactive(_ts_items(), color_by="depth", **options))
        assert type(bar.ticker).__name__ == "FixedTicker"
        assert bar.major_label_overrides and not _end_marks(bar)
    (bar,) = _bars(_interactive(_ts_items(), color_by="time"))
    assert type(bar.ticker).__name__ == "DatetimeTicker"
    assert not bar.major_label_overrides


def test_a_time_pin_labels_the_clipped_end_as_a_date():
    from bokeh.models import DatetimeTickFormatter

    items = _ts_items()
    pin = np.datetime64("2020-06-01")
    ms = float((pin - np.datetime64(0, "ms")) / np.timedelta64(1, "ms"))
    out = _interactive(items, color_by="time", vmax="2020-06-01")
    (bar,) = _bars(out)
    assert bar.major_label_overrides == {ms: "≥ 2020-06-01"}
    assert isinstance(bar.formatter, DatetimeTickFormatter)
    assert type(bar.ticker).__name__ == "FixedTicker"
    assert ms in bar.ticker.ticks
    assert len(bar.ticker.ticks) >= 3
    mapper = _cloud(next(iter(_figures(out).values()))).glyph.fill_color.transform
    assert mapper.high == ms
    both = _interactive(items, color_by="time", vmin="2020-03-01", vmax="2020-09-01")
    (bar,) = _bars(both)
    assert list(bar.major_label_overrides.values()) == ["≤ 2020-03-01", "≥ 2020-09-01"]


@pytest.mark.parametrize(
    "opts",
    [
        {"color_by": "depth"},
        {"color_by": "depth", "vmax": 500},
        {"color_by": "depth", "vmin": 100},
        {"color_by": "depth", "vmin": 100, "vmax": 500},
        {"color_by": "depth", "vmin": 0, "vmax": 5000},
        {"color_by": "time"},
        {"color_by": "time", "vmax": "2020-06-01"},
        {"color_by": "time", "vmin": "2020-03-01"},
        {"color_by": "time", "vmin": "2020-03-01", "vmax": "2020-09-01"},
        {"color_by": "time", "vmin": "2019-01-01", "vmax": "2021-01-01"},
    ],
)
def test_both_renderers_mark_the_same_ends_of_a_pinned_bar(opts):
    items = _ts_items()
    static = _static_extend(items, **opts)
    (bar,) = _bars(_interactive(items, **opts))
    text = list(bar.major_label_overrides.values())
    interactive = {
        (True, True): "both",
        (True, False): "min",
        (False, True): "max",
        (False, False): "neither",
    }[(any(t.startswith("≤") for t in text), any(t.startswith("≥") for t in text))]
    assert static == interactive


def test_pins_without_color_by_are_refused_interactively_too():
    with pytest.raises(ValueError, match=r"vmin=/vmax= pin the color_by scale"):
        _interactive(_ts_items(), vmin=0)


# --- members: colours, legend ---------------------------------------------------------


def test_member_colours_match_the_static_renderer():
    colors = {"ROMS": "black", "WOA23": "tab:red", "GLORYS12": "tab:blue"}
    items = _ts_items()
    (ax,) = _panel_axes(_static(items, colors=colors))
    (panel,) = _interactive(items, colors=colors).values()
    (cloud,) = _scatter(ax)
    (points,) = _elements(panel, "Points")
    # the static dots carry their alpha in the face colour; the colour is the rest
    assert same_color(
        points.opts.get("style").kwargs["color"], cloud.get_facecolor()[0][:3]
    )
    curves = _elements(panel, "Curve")
    assert len(curves) == len(ax.lines) == 2
    for curve, line in zip(curves, ax.lines, strict=True):
        assert same_color(curve.opts.get("style").kwargs["color"], line.get_color())


def test_legend_rows_and_swatch_colours_match_the_static_key():
    colors = {"ROMS": "black", "WOA23": "tab:red", "GLORYS12": "tab:blue"}
    items = _ts_items()
    (ax,) = _panel_axes(_static(items, colors=colors))
    static_key = ax.get_legend()
    static_labels = [t.get_text() for t in static_key.get_texts()]
    static_colours = [h.get_color() for h in static_key.legend_handles]
    static_dots = [h.get_linestyle() == "None" for h in static_key.legend_handles]

    out = _interactive(items, colors=colors)
    (panel,) = out.values()
    proxies = [e for e in panel.values() if e.group == "Legend"]
    assert [e.label for e in proxies] == static_labels
    assert [type(e).__name__ == "Points" for e in proxies] == static_dots
    for proxy, colour in zip(proxies, static_colours, strict=True):
        assert same_color(proxy.opts.get("style").kwargs["color"], colour)
    # bokeh's own key says the same, in the same order, and the data adds no rows
    (figure,) = _figures(out).values()
    assert [i.label.value for i in figure.legend[0].items] == static_labels


def test_the_legend_dot_is_a_real_swatch_not_the_plotted_speck():
    (panel,) = _interactive(_ts_items(), marker_size=2.0).values()
    (points,) = _elements(panel, "Points")
    (key,) = _elements(panel, "Points", keys=True)
    assert (
        key.opts.get("style").kwargs["size"]
        > 2 * points.opts.get("style").kwargs["size"]
    )
    assert key.opts.get("style").kwargs["alpha"] == 1.0
    # the static key makes the same choice
    (ax,) = _panel_axes(_static(_ts_items()))
    assert ax.get_legend().legend_handles[0].get_markersize() > 3


def test_two_dot_members_get_different_markers_in_the_key():
    items = [_item("A"), _item("B"), _item("C", mark="line")]
    (panel,) = _interactive(items).values()
    wanted = list(_style.BOKEH_MARKERS[:2])
    keys = _elements(panel, "Points", keys=True)
    assert [k.opts.get("style").kwargs["marker"] for k in keys] == wanted
    dots = _elements(panel, "Points")
    assert [p.opts.get("style").kwargs["marker"] for p in dots] == wanted


def test_legend_off_drops_the_key_and_its_proxies():
    out = _interactive(_ts_items(), legend=False)
    (panel,) = out.values()
    assert not [e for e in panel.values() if e.group == "Legend"]
    (figure,) = _figures(out).values()
    assert not any(key.items for key in figure.legend)


@pytest.mark.parametrize(
    ("corner", "position"),
    [
        ("lower right", "bottom_right"),
        ("upper left", "top_left"),
        ("lower left", "bottom_left"),
        ("upper right", "top_right"),
    ],
)
def test_a_forced_corner_reaches_every_panels_key(corner, position):
    out = _interactive(_ts_items(REGIONS[:2]), legend=corner)
    for panel in out.values():
        assert panel.opts.get("plot").kwargs["legend_position"] == position
    static = _static(_ts_items(REGIONS[:2]), legend=corner)
    assert all(ax.get_legend() is not None for ax in _panel_axes(static))


def test_each_panel_key_sits_in_the_corner_compose_chose():
    items = _ts_items(REGIONS[:3])
    layout = _xy.compose(items)
    out = _interactive(items)
    from ocean_skill.plot.holoviews_renderer import _BOKEH_LEGEND_POSITION

    assert [p.opts.get("plot").kwargs["legend_position"] for p in out.values()] == [
        _BOKEH_LEGEND_POSITION[p.legend_corner] for p in layout.panels
    ]


@pytest.mark.parametrize("side", ["below", "right"])
def test_a_key_pushed_outside_the_frame_leaves_the_data_clear(side):
    (figure,) = _figures(_interactive(_ts_items(), legend=side)).values()
    outside = figure.below if side == "below" else figure.right
    assert any(type(r).__name__ == "Legend" for r in outside)


# --- dots: size, alpha, hover ---------------------------------------------------------


def test_marker_size_becomes_the_diameter_it_would_be_on_screen():
    (small,) = _interactive(_ts_items(), marker_size=2.0).values()
    (large,) = _interactive(_ts_items(), marker_size=8.0).values()
    s_small = _elements(small, "Points")[0].opts.get("style").kwargs["size"]
    s_large = _elements(large, "Points")[0].opts.get("style").kwargs["size"]
    # s is points squared: the diameter is its square root, at 4/3 CSS pixels a point
    assert s_small == pytest.approx(2.0**0.5 * 4 / 3)
    assert s_large == pytest.approx(8.0**0.5 * 4 / 3)


def test_alpha_has_no_outline_and_the_marker_is_a_circle():
    (panel,) = _interactive(_ts_items(), alpha=0.25).values()
    (points,) = _elements(panel, "Points")
    style = points.opts.get("style").kwargs
    assert (style["alpha"], style["line_width"], style["marker"]) == (0.25, 0, "circle")


def test_hover_gives_dots_depth_and_time_and_lines_depth():
    from bokeh.models import HoverTool

    (figure,) = _figures(_interactive(_ts_items())).values()
    tips = [dict(h.tooltips) for h in figure.select({"type": HoverTool}) if h.tooltips]
    assert any("time" in t and "depth [m]" in t for t in tips)
    assert any("depth [m]" in t and "time" not in t for t in tips)
    for t in tips:
        assert {"salinity", "temperature [degC]"} <= set(t)


# --- annotations, limits, labels ------------------------------------------------------


def test_annotations_sit_where_the_static_renderer_puts_them():
    ann = {
        REGIONS[0]: {"STSW": (34.8, 12.0), "Northern\nwaters": (34.0, 11.0)},
        REGIONS[1]: {"NPIW": (34.4, 8.0)},
    }
    items = _ts_items(REGIONS[:2])
    static = _panel_axes(_static(items, annotations=ann))
    out = _interactive(items, annotations=ann)
    for ax, panel in zip(static, out.values(), strict=True):
        texts = _elements(panel, "Text")
        assert [(t.x, t.y, t.text) for t in texts] == [
            (*t.get_position(), t.get_text()) for t in ax.texts
        ]
        assert all(
            t.opts.get("style").kwargs["text_font_size"].endswith("pt") for t in texts
        )


def test_the_flat_annotation_form_goes_on_every_panel():
    out = _interactive(_ts_items(REGIONS[:3]), annotations={"STSW": (34.8, 12.0)})
    assert [len(_elements(p, "Text")) for p in out.values()] == [1, 1, 1]


def test_annotation_errors_surface_through_the_renderer():
    with pytest.raises(ValueError, match="mixes the two forms"):
        _interactive(_ts_items(), annotations={"A": (1, 2), REGIONS[0]: {"B": (1, 2)}})
    with pytest.raises(ValueError, match="no regions"):
        _interactive([_item(region=None)], annotations={"R": {"B": (1, 2)}})


@pytest.mark.parametrize(
    "opts",
    [
        {},
        {"sharex": True},
        {"sharey": True},
        {"sharex": True, "sharey": True},
        {"xlim": (34.0, 35.0), "ylim": (0.0, 30.0)},
        {"annotations": {"far": (36.0, 40.0)}},
    ],
)
def test_limits_agree_with_the_static_axes(opts):
    items = _ts_items(REGIONS[:3])
    static = {ax.get_title(): ax for ax in _panel_axes(_static(items, **opts))}
    figures = _figures(_interactive(items, **opts))
    for title, ax in static.items():
        figure = figures[title]
        assert (figure.x_range.start, figure.x_range.end) == pytest.approx(
            ax.get_xlim()
        )
        assert (figure.y_range.start, figure.y_range.end) == pytest.approx(
            ax.get_ylim()
        )


def test_axis_labels_go_on_the_outer_edges_only_and_read_as_the_static_ones():
    items = _ts_items(REGIONS)
    static = {ax.get_title(): ax for ax in _panel_axes(_static(items))}
    figures = _figures(_interactive(items))
    for title, ax in static.items():
        assert figures[title].xaxis[0].axis_label == ax.get_xlabel()
        assert figures[title].yaxis[0].axis_label == ax.get_ylabel()
    # first column and bottom row only
    assert [figures[r].yaxis[0].axis_label != "" for r in REGIONS] == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    assert [figures[r].xaxis[0].axis_label != "" for r in REGIONS] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_two_axes_with_one_label_still_draw():
    # two different variables can share a short name and units; holoviews draws nothing
    # for dimensions that share a label, so the axis labels are options, not dimensions
    item = {**_item(), "x_name": "temperature", "x_units": "degC"}
    out = _interactive([item])
    (points,) = _elements(next(iter(out.values())), "Points")
    assert len({d.label for d in points.kdims}) == 2
    (figure,) = _figures(out).values()
    assert figure.xaxis[0].axis_label == figure.yaxis[0].axis_label
    assert figure.xaxis[0].axis_label == "temperature [degC]"


def test_a_ragged_last_row_labels_the_panel_above_the_gap():
    figures = _figures(_interactive(_ts_items(REGIONS[:4]), ncols=3))
    assert figures[REGIONS[1]].xaxis[0].axis_label == "salinity"
    assert figures[REGIONS[2]].xaxis[0].axis_label == "salinity"
    assert figures[REGIONS[0]].xaxis[0].axis_label == ""


def test_titles_option_and_suptitle():
    import holoviews as hv

    out = _interactive(_ts_items(REGIONS[:2]), titles=["A", "B"], title="Water masses")
    assert set(_figures(out)) == {"A", "B"}
    assert (
        hv.Store.lookup_options("bokeh", out, "plot").kwargs["title"] == "Water masses"
    )


# --- geometry -------------------------------------------------------------------------


def _frame_widths(obj) -> set[int]:
    return {f.frame_width for f in _figures(obj).values()}


def _title_pt(figure) -> float:
    return float(figure.title.text_font_size.removesuffix("pt"))


def test_panel_width_shrinks_as_columns_are_added_but_never_below_a_map_panel():
    from ocean_skill.plot.holoviews_renderer import PANEL_WIDTH_PX

    items = _ts_items(REGIONS)
    one, two, three, six = (
        _frame_widths(_interactive(items, ncols=n)) for n in (1, 2, 3, 6)
    )
    for widths in (one, two, three, six):
        assert len(widths) == 1
    (w1,), (w2,), (w3,), (w6,) = one, two, three, six
    assert w1 > w2 > w3 > w6 == PANEL_WIDTH_PX


def test_panel_aspect_font_scale_size_and_zoom_reach_the_frame():
    items = _ts_items(REGIONS[:3])
    base = _figures(_interactive(items))[REGIONS[0]]
    wide = _figures(_interactive(items, panel_aspect=2.0))[REGIONS[0]]
    assert wide.frame_width / wide.frame_height == pytest.approx(2.0, rel=0.01)
    assert base.frame_width / base.frame_height == pytest.approx(0.9, rel=0.01)
    zoomed = _figures(_interactive(items, zoom=2.0))[REGIONS[0]]
    assert zoomed.frame_width == pytest.approx(2 * base.frame_width, abs=1)
    column = _figures(_interactive(items, size="column"))[REGIONS[0]]
    assert column.frame_width < base.frame_width
    big = _figures(_interactive(items, font_scale=1.5))[REGIONS[0]]
    assert _title_pt(big) > _title_pt(base)


# --- options routed through the shared dispatch ---------------------------------------


def test_domain_is_warned_about_and_dropped():
    with pytest.warns(UserWarning, match="not an option of XY"):
        out = _interactive(_ts_items(), domain=(0.0, 0.0, 1.0, 1.0))
    assert len(out) == 1


@pytest.mark.parametrize(
    "option",
    [
        "title_kwargs",
        "tick_label_kwargs",
        "suptitle_kwargs",
        "legend_kwargs",
        "line_kwargs",
        "annot_kwargs",
        "colorbar_kwargs",
    ],
)
def test_static_only_styling_warns_rather_than_vanishing(option):
    with pytest.warns(UserWarning, match=rf"\['{option}'\] only affect the static"):
        _interactive(_ts_items(), **{option: {"color": "red"}})


def test_wspace_and_hspace_warn_as_static_only():
    with pytest.warns(
        UserWarning, match=r"\['hspace', 'wspace'\] only affect the static"
    ):
        _interactive(_ts_items(REGIONS[:2]), wspace=0.1, hspace=0.1)


@pytest.mark.parametrize("option", ["figsize", "fit_text", "save"])
def test_layout_and_file_options_are_dropped_quietly(option, tmp_path):
    values = {"figsize": (5, 5), "fit_text": False, "save": str(tmp_path / "x.png")}
    value = values[option]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _interactive(_ts_items(), **{option: value})
    assert len(out) == 1
    assert not (tmp_path / "x.png").exists()


def test_every_contract_option_is_accepted():
    """Accept every option the static renderer does, so ``renderer="both"`` works."""
    import inspect

    from ocean_skill.plot.matplotlib_renderer import xy

    values = {
        "title": "t",
        "annotations": {"A": (34.5, 12.0)},
        "density": False,
        "color_by": "depth",
        "cmap": "viridis",
        "vmin": 0,
        "vmax": 1500,
        "colorbar": True,
        "colors": ["k", "r", "b"],
        "legend": True,
        "titles": ["T"],
        "xlim": (34.0, 35.0),
        "ylim": (5.0, 20.0),
        "sharex": True,
        "sharey": True,
        "marker_size": 3.0,
        "alpha": 0.4,
        "panel_aspect": 1.0,
        "ncols": 1,
        "nrows": 1,
        "size": "page",
        "zoom": 1.0,
        "font_scale": 1.1,
        "figsize": (6, 6),
        "save": None,
        "fit_text": True,
        "wspace": 0.1,
        "hspace": 0.1,
        "title_kwargs": {},
        "tick_label_kwargs": {},
        "suptitle_kwargs": {},
        "legend_kwargs": {},
        "line_kwargs": {},
        "annot_kwargs": {},
        "colorbar_kwargs": {},
    }
    assert set(values) == set(inspect.signature(xy).parameters) - {"items"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(_interactive(_ts_items(), **values)) == 1


def test_compose_warnings_reach_the_caller_once(monkeypatch):
    monkeypatch.setattr(_xy, "POINT_CAP", 100)
    items = _ts_items()
    items[1] = {**items[1], "y_standard_name": "sea_water_temperature"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _interactive(items)
    messages = [str(w.message) for w in caught]
    assert sum("holds 200 points" in m for m in messages) == 1
    assert sum("disagree on what the y axis is" in m for m in messages) == 1


def test_both_renderers_draw_from_one_option_set():
    out = render(
        PlotSpec("XY", _ts_items(REGIONS[:2]), {"ncols": 2, "color_by": "depth"}),
        renderer="both",
    )
    assert len(out) == 2  # panel Row: the static column and the interactive one
