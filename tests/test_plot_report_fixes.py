"""Figure fixes found in real suite reports, each pinned in both renderers.

Numbers on a figure read through one formatter (``max 8440``, never ``8.44e+03``); a
colorbar's label never lands on the next column's map; a dimensionless field draws no
``[]``; a one-day window reads as one date; a map grid is as many columns wide however
few the panels; a section's all-land ends are cut and its degree ticks never touch; a
figure with a single series has no one-entry legend; and every unit on a label goes
through ``units.display``.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import units as u
from ocean_skill.align import ALONG_DIM
from ocean_skill.plot import _titles
from ocean_skill.plot._numbers import format_value
from ocean_skill.plot._statistic import bracketed_units
from ocean_skill.plot.matplotlib_renderer import (
    _clip_text,
    field_map_grid,
    metric_value_text,
)
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

NITRATE = "mole_concentration_of_nitrate_in_sea_water"
SALINITY = "sea_water_practical_salinity"
CO2 = "mole_fraction_of_carbon_dioxide_in_air"


def _map_item(
    standard_name=NITRATE,
    *,
    units="mmol m-3",
    lon=(-130.0, -115.0),
    lat=(30.0, 48.0),
    base=1.0,
    time=None,
    depth=None,
):
    """One already-reduced map, shaped as ``Field._map_item`` builds it."""
    rng = np.random.default_rng(0)
    field = xr.DataArray(
        base * (1.0 + rng.random((24, 20))),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(*lat, 24), "lon": np.linspace(*lon, 20)},
        attrs={"units": units},
    )
    return {
        "field": field,
        "units": units,
        "standard_name": standard_name,
        "depth": depth,
        "time": time,
        "label": "model",
        "facet_dim": None,
        "row_dim": None,
    }


def _cbar_axes(fig):
    return [a for a in fig.axes if getattr(a, "_osk_cbar_parents", None)]


def _panel_axes(fig):
    return [
        a
        for a in fig.axes
        if a.get_visible() and not getattr(a, "_osk_cbar_parents", None)
    ]


# -- 1. numbers print through format_value --------------------------------------------


def test_clip_text_uses_the_shared_number_format():
    assert _clip_text("max", 8440.7) == "max 8440"
    assert _clip_text("max", 2890.0) == "max 2890"
    assert _clip_text("min", 3.4e-22) == "min 3.4e\N{MINUS SIGN}22"
    assert _clip_text("min", -1.83) == "min \N{MINUS SIGN}1.83"


def test_a_metric_in_the_corner_box_uses_the_shared_number_format():
    assert metric_value_text({"rmse": 8440.2}, "rmse") == format_value(8440.2) == "8440"
    assert metric_value_text({"bias": 0.04123}, "bias") == "0.0412"


def test_static_colorbar_ends_read_plain_digits_not_scientific():
    item = _map_item(base=1.0)
    values = item["field"].values
    values[0, 0], values[-1, -1] = 8440.0, 3.4e-22
    fig = field_map_grid([item], robust=True, colorbar_label_clipped=True)
    texts = sorted(t.get_text() for a in _cbar_axes(fig) for t in a.texts)
    assert texts == ["max 8440", "min 3.4e\N{MINUS SIGN}22"]


def test_interactive_clipped_tick_reads_plain_digits():
    pytest.importorskip("holoviews")
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts(
        (0.0, 3.0), (0.0, 8440.0), label_clipped=True
    )
    labels = list(opts["major_label_overrides"].values())
    assert any("max 8440" in label for label in labels)
    assert not any("e+03" in label for label in labels)


# -- 2. a colorbar label never lands on the next column -------------------------------


def _label_overlaps(fig) -> list[float]:
    """Pixels by which any bar's labelling reaches into a panel that is not its own."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    panels = _panel_axes(fig)
    for cax in _cbar_axes(fig):
        texts = [cax.yaxis.label, *cax.get_yticklabels()]
        boxes = [t.get_window_extent(renderer) for t in texts if t.get_text()]
        right = max(b.x1 for b in boxes)
        bar = cax.get_window_extent(renderer)
        for p in panels:
            if p in cax._osk_cbar_parents:
                continue
            box = p.get_window_extent(renderer)
            if (
                box.y1 > min(b.y0 for b in boxes)
                and box.y0 < max(b.y1 for b in boxes)
                and box.x0 > bar.x0
                and right > box.x0
            ):
                out.append(right - box.x0)
        if right > fig.bbox.width:
            out.append(right - fig.bbox.width)
    return out


def test_a_three_column_grid_with_long_units_overlaps_nothing():
    # tall, narrow panels in three columns: the case where a bar's label used to be
    # drawn over the next column's map
    items = [
        _map_item(
            NITRATE,
            units="mmol m-3",
            lon=(-130.0, -120.0),
            lat=(20.0, 50.0),
            base=5.0 * (i + 1),
        )
        for i in range(3)
    ]
    fig = field_map_grid(items, ncols=3)
    assert len(_cbar_axes(fig)) == 3
    assert _label_overlaps(fig) == []
    # the label reads in full: nothing was dropped to make it fit
    assert {a.yaxis.label.get_text() for a in _cbar_axes(fig)} == {"[mmol/m³]"}


def test_clearing_works_on_a_figure_whose_layout_engine_was_stood_down():
    # fig.set_layout_engine("none") on a constrained figure leaves a placeholder engine
    # object behind; that is not a live layout, and must not stop the labels clearing
    from ocean_skill.plot.matplotlib_renderer import _layout_is_live

    fig = field_map_grid([_map_item()])
    assert fig.get_layout_engine() is not None
    assert not _layout_is_live(fig)


def test_what_cannot_be_moved_clear_is_shrunk_never_left_on_the_neighbor():
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    from ocean_skill.plot.matplotlib_renderer import (
        _clear_colorbar_labels,
        _draw_colorbar,
        _style_defaults,
    )
    from ocean_skill.plot.typography import type_scale

    fig, (left, right) = plt.subplots(1, 2, figsize=(5, 3))
    fig.set_layout_engine("none")
    left.set_position([0.05, 0.15, 0.30, 0.7])
    right.set_position([0.52, 0.15, 0.30, 0.7])  # a gutter too narrow for 12pt type
    defaults = _style_defaults(type_scale((5, 3), ncols=2), horizontal_colorbar=False)
    im = ScalarMappable(cmap="viridis")
    im.set_clim(0.0, 100000.0)
    cbar = _draw_colorbar(fig, im, left, "[mmol/m³]", None, defaults["colorbar_kwargs"])
    cbar.ax.set_position([0.37, 0.15, 0.03, 0.7])
    fig.draw_without_rendering()
    before = cbar.ax.yaxis.label.get_fontsize()
    renderer = fig.canvas.get_renderer()
    for _ in range(3):
        _clear_colorbar_labels(fig, renderer)
        fig.draw_without_rendering()
    gutter = right.get_window_extent(renderer).x0
    labels = [cbar.ax.yaxis.label, *cbar.ax.get_yticklabels()]
    edge = max(t.get_window_extent(renderer).x1 for t in labels if t.get_text())
    assert edge <= gutter + 1.0
    assert cbar.ax.yaxis.label.get_fontsize() <= before


# -- 3. empty units draw no brackets --------------------------------------------------


@pytest.mark.parametrize("units", [None, "", "1"])
def test_empty_units_have_no_brackets(units):
    assert bracketed_units(units) == ""


def test_real_units_are_bracketed_through_display():
    assert bracketed_units("mmol m-3") == "[mmol/m³]"
    assert bracketed_units("psu") == "[PSU]"


def test_a_dimensionless_grid_labels_its_bar_with_nothing():
    fig = field_map_grid([_map_item(CO2, units="1")])
    (cax,) = _cbar_axes(fig)
    assert cax.yaxis.label.get_text() == ""


def test_a_dimensionless_difference_bar_says_just_difference():
    pytest.importorskip("cartopy")
    from ocean_skill.plot.matplotlib_renderer import field_row

    field = _map_item(CO2, units="1")["field"]
    aligned = {
        "test": field + 0.1,
        "reference": field,
        "difference": field * 0 + 0.1,
    }
    fig = field_row(aligned, units="1", standard_name=CO2)
    labels = [a.yaxis.label.get_text() for a in _cbar_axes(fig)]
    assert "difference" in labels
    assert not any("[]" in label for label in labels)


def test_dimensionless_units_are_blank_in_the_interactive_bar_too():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    item = _map_item(CO2, units="1")
    obj = render(
        PlotSpec(family="field_map_grid", items=[item]), renderer="holoviews"
    )
    meshes = obj.traverse(lambda x: x, [hv.QuadMesh])
    assert meshes
    assert all(not m.opts.get("plot").kwargs.get("clabel") for m in meshes)


# -- 4. single-day ranges read as one date ---------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("surface · 2013-01-01–2013-01-01", "surface · 2013-01-01"),
        ("2010-12-31–2010-12-31", "2010-12-31"),
        ("mean over 2012-05–2012-05", "mean over 2012-05"),
        ("2015-01 to 2015-01", "2015-01"),
        # a real range is left alone, and so is a similar-looking prefix
        ("2013-01-01–2013-01-31", "2013-01-01–2013-01-31"),
        ("2013-01-01–2013-01-011", "2013-01-01–2013-01-011"),
        ("2013-01-01–2013-01-01 12:00", "2013-01-01–2013-01-01 12:00"),
        ("no dates here", "no dates here"),
    ],
)
def test_collapse_single_day(text, expected):
    assert _titles.collapse_single_day(text) == expected


def test_a_map_grid_suptitle_names_a_single_day_once():
    items = [
        _map_item(NITRATE, time="2013-01-01–2013-01-01", depth="surface"),
        _map_item(SALINITY, time="2013-01-01–2013-01-01", depth="surface", units="PSU"),
    ]
    fig = field_map_grid(items)
    title = fig._suptitle.get_text()
    assert "2013-01-01" in title
    assert "2013-01-01–2013-01-01" not in title


def test_resolve_titles_collapses_an_auto_title():
    assert _titles.resolve_titles(["a · 2013-01-01–2013-01-01"], None) == [
        "a · 2013-01-01"
    ]


def test_the_series_period_of_one_month_is_one_month():
    from ocean_skill.plot.series import _period_of

    da = xr.DataArray(
        [1.0, 2.0],
        dims="time",
        coords={"time": pd.to_datetime(["2015-01-03", "2015-01-20"])},
    )
    assert _period_of(da) == "2015-01"


# -- 5. a map grid is as wide whatever the panel count ---------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 9])
def test_an_ordinary_domain_is_at_most_three_across(n):
    from ocean_skill.plot.typography import map_grid_layout

    ncols, nrows = map_grid_layout(n, 0.7)
    assert ncols == min(n, 3)
    assert ncols * nrows >= n


@pytest.mark.parametrize("n", [2, 3, 6, 9])
def test_a_very_wide_domain_is_two_across_never_one(n):
    from ocean_skill.plot.typography import map_grid_layout

    ncols, _ = map_grid_layout(n, 3.5)
    assert ncols == min(n, 2)


def test_two_variables_sit_side_by_side_not_stacked():
    items = [
        _map_item(CO2, units="1", lon=(-180, -60), lat=(-30, 30)) for _ in range(2)
    ]
    fig = field_map_grid(items)
    tops = {round(a.get_position().y1, 3) for a in _panel_axes(fig)}
    assert len(tops) == 1  # one row


def test_three_variables_of_a_wide_domain_do_not_stack_three_full_width_maps():
    items = [
        _map_item(NITRATE, lon=(80.0, 300.0), lat=(-30.0, 30.0)) for _ in range(3)
    ]
    fig = field_map_grid(items)
    columns = {round(a.get_position().x0, 2) for a in _panel_axes(fig)}
    assert len(columns) == 2


def test_the_interactive_grid_is_the_same_width():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    items = [
        _map_item(CO2, units="1", lon=(-180, -60), lat=(-30, 30)) for _ in range(2)
    ]
    obj = render(PlotSpec(family="field_map_grid", items=items), renderer="holoviews")
    assert isinstance(obj, hv.Layout)
    assert obj._max_cols == 2


# -- 6. sections ----------------------------------------------------------------------


def _line(lon, lat, *, lead=0, trail=0, bump=0.0) -> xr.DataArray:
    """Build a fixed-depth section along (lon, lat), NaN in the end columns."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    n = lon.size
    z = -np.array([0.0, 50.0, 100.0, 200.0])
    values = 20.0 - 0.05 * np.abs(z)[:, None] + 0.02 * np.arange(n)[None, :] + bump
    values[:, :lead] = np.nan
    if trail:
        values[:, n - trail :] = np.nan
    da = xr.DataArray(
        values,
        dims=("z", ALONG_DIM),
        coords={
            "z": z,
            ALONG_DIM: np.arange(n) * 25.0,
            "lon": (ALONG_DIM, lon),
            "lat": (ALONG_DIM, lat),
        },
    )
    da[ALONG_DIM].attrs["units"] = "km"
    return da


def _trio(**kwargs):
    test = _line(**kwargs)
    reference = _line(**kwargs, bump=0.5)
    return {"test": test, "reference": reference, "difference": test - reference}


def _coast(n=60):
    return {"lon": np.linspace(-135.0, -110.0, n), "lat": np.full(n, 44.0)}


def test_all_land_ends_are_trimmed_to_the_data():
    from ocean_skill.plot.section import prepare_section_row

    values, geometry = prepare_section_row(_trio(**_coast(), lead=10, trail=30))
    lo, hi = geometry.x_trim
    x = values["test"]["distance"].isel(z=0).values
    # the 20 columns in between survive, to the edges of their cells
    assert lo == pytest.approx(x[9:11].mean())
    assert hi == pytest.approx(x[29:31].mean())


def test_a_section_with_data_everywhere_is_not_trimmed():
    from ocean_skill.plot.section import prepare_section_row

    _, geometry = prepare_section_row(_trio(**_coast()))
    assert geometry.x_trim is None


def test_a_column_with_data_in_either_lane_is_kept():
    from ocean_skill.plot.section import prepare_section_row

    trio = _trio(**_coast(), trail=30)
    # the reference alone has data out to column 45: the row's extent follows it
    ref = trio["reference"].copy()
    ref[:, 30:45] = 5.0
    trio["reference"] = ref
    trio["difference"] = trio["test"] - ref
    values, geometry = prepare_section_row(trio)
    x = values["test"]["distance"].isel(z=0).values
    assert geometry.x_trim[1] == pytest.approx(x[44:46].mean())


def test_a_hole_in_the_middle_is_a_real_gap_and_stays():
    from ocean_skill.plot.section import prepare_section

    da = _line(**_coast())
    da[:, 20:30] = np.nan
    _, geometry = prepare_section(da)
    assert geometry.x_trim is None


def test_the_static_row_is_drawn_over_the_trimmed_range():
    from ocean_skill.plot.matplotlib_renderer import section_row

    fig = section_row(_trio(**_coast(), trail=40), standard_name=None, units="degC")
    panels = _panel_axes(fig)
    assert len(panels) == 3
    for ax in panels:
        _, hi = ax.get_xlim()
        assert hi < -120.0  # not out to the -110 the land runs to


def test_the_interactive_row_is_drawn_over_the_trimmed_range():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    item = {
        "aligned": _trio(**_coast(), trail=40),
        "units": "degC",
        "standard_name": None,
        "depth": None,
        "time": None,
        "metrics": {"bias": 0.1, "rmse": 0.5, "corr": 0.98},
        "labels": ("model", "obs"),
    }
    obj = render(PlotSpec(family="section_row", items=[item]), renderer="holoviews")
    plots = [hv.renderer("bokeh").get_plot(p) for p in obj]
    ends = [p.handles["x_range"].end for p in plots]
    assert all(end < -120.0 for end in ends)


def _dateline():
    east, west = np.linspace(113.0, 180.0, 20), np.linspace(-180.0, -78.0, 30)
    lon = np.concatenate([east, west[1:]])
    return {"lon": lon, "lat": np.full(lon.size, -30.0)}


def _tick_boxes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for ax in _panel_axes(fig):
        boxes = [
            t.get_window_extent(renderer)
            for t in ax.get_xticklabels()
            if t.get_visible() and t.get_text()
        ]
        out.append(sorted(boxes, key=lambda b: b.x0))
    return out


def test_dateline_tick_labels_never_touch():
    from ocean_skill.plot.matplotlib_renderer import section_row

    fig = section_row(_trio(**_dateline()), standard_name=None, units="degC")
    for boxes in _tick_boxes(fig):
        assert len(boxes) <= 7
        for a, b in pairwise(boxes):
            assert b.x0 > a.x1 + 2.0


def test_a_wider_panel_gets_more_ticks_up_to_seven():
    from ocean_skill.plot._geo_labels import tick_count

    assert tick_count(1.0) == 2
    assert tick_count(2.5) == 4
    assert tick_count(8.0) == 7
    assert tick_count(100.0) == 7


def test_the_interactive_degree_axis_asks_for_a_bounded_tick_count():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    item = {
        "aligned": _trio(**_dateline()),
        "units": "degC",
        "standard_name": None,
        "depth": None,
        "time": None,
        "metrics": {"bias": 0.1, "rmse": 0.5, "corr": 0.98},
        "labels": ("model", "obs"),
    }
    obj = render(PlotSpec(family="section_row", items=[item]), renderer="holoviews")
    meshes = obj.traverse(lambda x: x, [hv.QuadMesh])
    counts = [m.opts.get("plot").kwargs.get("xticks") for m in meshes]
    assert all(isinstance(c, int) and 2 <= c <= 7 for c in counts)


# -- 7. a single series has no one-entry legend ----------------------------------------


def _series_item(source="ccs_4km", n=12):
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    da = xr.DataArray(
        8.0 + np.sin(np.arange(n)),
        coords={"time": time},
        dims="time",
        attrs={"units": "degC"},
    ).assign_coords(lon=-144.0, lat=50.0)
    return {
        "aligned": xr.Dataset({"value": da}),
        "metrics": None,
        "units": "degC",
        "standard_name": "sea_water_temperature",
        "label": source,
        "labels": (source,),
    }


def test_one_series_draws_no_legend_in_either_renderer():
    from ocean_skill.plot import series as _series

    spec = PlotSpec(family="series", items=[_series_item()], options={})
    fig = render(spec, renderer="matplotlib")
    assert not fig.legends
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert _series.compose(spec.items).legend_placement == "off"

    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    obj = render(spec, renderer="holoviews")
    overlays = obj.traverse(lambda x: x, [hv.Overlay])
    shown = [o.opts.get("plot").kwargs.get("show_legend") for o in overlays]
    assert shown and not any(shown)


def test_two_series_keep_their_legend():
    spec = PlotSpec(
        family="series",
        items=[_series_item("a"), _series_item("b")],
        options={},
    )
    fig = render(spec, renderer="matplotlib")
    assert fig.legends or any(ax.get_legend() is not None for ax in fig.axes)


def test_a_forced_legend_is_kept_even_for_one_series():
    spec = PlotSpec(
        family="series", items=[_series_item()], options={"legend": "upper left"}
    )
    fig = render(spec, renderer="matplotlib")
    assert any(ax.get_legend() is not None for ax in fig.axes)


# -- 8. units go through display -------------------------------------------------------


@pytest.mark.parametrize(
    "spelling", ["psu", "PSU", "practical_salinity", "Practical Salinity"]
)
def test_every_salinity_spelling_displays_as_psu(spelling):
    assert u.display(spelling) == "PSU"


@pytest.mark.parametrize("spelling", ["1", "", None, " "])
def test_a_dimensionless_unit_displays_as_nothing(spelling):
    assert u.display(spelling) == ""


def test_a_scale_factor_is_not_dimensionless():
    assert u.display("1e-3") == "1e-3"


def test_a_series_axis_label_spells_its_units_through_display():
    from ocean_skill.plot.series import _ylabel, compose  # noqa: F401

    spec_item = _series_item()
    spec_item["units"] = "mmol m-3"
    spec_item["aligned"]["value"].attrs["units"] = "mmol m-3"
    layout = compose([spec_item])
    assert "[mmol/m³]" in layout.panels[0].ylabel
    assert "mmol m-3" not in layout.panels[0].ylabel


def test_no_plot_module_writes_a_raw_unit_into_a_label():
    # every [unit] a figure draws is built from units_text / display, not typed: no
    # string literal in the plot package is itself a machine spelling of a unit
    import ast
    import pathlib

    raw = {"mmol/m^3", "mmol m-3", "psu", "degC", "degrees_celsius", "m3/s", "m^3/s"}
    root = pathlib.Path(__file__).parents[1] / "ocean_skill" / "plot"
    offenders = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.strip("[] ") in raw:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
