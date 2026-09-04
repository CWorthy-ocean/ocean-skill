"""Tests for the ``profile`` family, in both renderers.

The vertical twin of ``tests/test_series_renderers.py``, and held to the same
standing rule: a plot change lands in the static *and* the interactive renderer,
and a test asserts the two **agree** rather than that each merely runs. Items are
built by hand, exactly as ``tests/test_renderers.py`` does, so a renderer is tested
without going through ``compare()``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot import profile as _profile
from ocean_skill.plot import style as _style
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

TEMPERATURE = "sea_water_temperature"
SALINITY = "sea_water_practical_salinity"


def _profile_item(
    variable: str = TEMPERATURE,
    *,
    test: str = "run_new",
    reference: str = "whots",
    units: str = "degC",
    offset: float = 0.6,
    zdim: str = "z",
    negative_down: bool = True,
    n: int = 8,
    max_depth: float = 150.0,
    time: str | None = None,
) -> dict:
    """One comparison item, shaped exactly as ``Comparison.as_item()`` builds it.

    ``zdim``/``negative_down`` cover both vertical-coordinate conventions this
    family has to read identically: ROMS's own ``z`` (negative-down, from
    :func:`ocean_skill.roms.to_depth`) and an observational product's ``depth``
    (already positive-down).
    """
    depths = np.linspace(5.0, max_depth, n)
    coord = -depths if negative_down else depths
    values = 20.0 - 0.08 * depths
    reference_da = xr.DataArray(
        values, coords={zdim: coord}, dims=zdim, attrs={"units": units}
    ).assign_coords(lon=-144.245, lat=49.978)
    if time is not None:
        reference_da = reference_da.assign_coords(time=np.datetime64(time))
    aligned = xr.Dataset(
        {
            "reference": reference_da,
            "test": reference_da + offset,
            "difference": reference_da * 0 + offset,
        }
    )
    aligned["reference"].attrs["units"] = units
    return {
        "aligned": aligned,
        "metrics": {
            "bias": offset,
            "rmse": abs(offset) + 0.1,
            "corr": 0.97,
            "n": n,
            "std_test": 2.8,
            "std_reference": 2.8,
            "crmsd": 0.1,
            "sigma_ratio": 1.0,
            "variable": variable,
        },
        "units": units,
        "standard_name": variable,
        "label": None,
        "labels": (test, reference),
    }


def _single_item(
    *,
    source: str = "run_new",
    variable: str = TEMPERATURE,
    units: str = "degC",
    n: int = 8,
    time: str | None = None,
) -> dict:
    """One single-source profile item, shaped as ``Field._profile_items`` builds it.

    No reference, no metrics -- one lone value read down the water column.
    """
    depths = np.linspace(5.0, 150.0, n)
    values = 20.0 - 0.08 * depths
    da = xr.DataArray(
        values, coords={"depth": depths}, dims="depth", attrs={"units": units}
    ).assign_coords(lon=-144.245, lat=49.978)
    if time is not None:
        da = da.assign_coords(time=np.datetime64(time))
    return {
        "aligned": xr.Dataset({"value": da}),
        "metrics": None,
        "units": units,
        "standard_name": variable,
        "label": source,
        "labels": (source,),
    }


def _seasonal_single_item(
    *,
    source: str = "run_new",
    variable: str = TEMPERATURE,
    units: str = "degC",
    seasons=("DJF", "MAM", "JJA", "SON"),
    n: int = 8,
    spread=None,
) -> dict:
    """A single-source profile item whose time axis was reduced to a season
    groupby -- the ``(season, depth)`` shape ``operators.aggregate({"time":
    {"groupby": "season", ...}})`` leaves standing on a model column.
    """
    depths = np.linspace(5.0, 150.0, n)
    base = 20.0 - 0.08 * depths
    values = base[None, :] + np.arange(len(seasons))[:, None]
    da = xr.DataArray(
        values,
        dims=("season", "depth"),
        coords={"season": list(seasons), "depth": depths},
        attrs={"units": units},
    ).assign_coords(lon=-144.245, lat=49.978)
    if spread is not None:
        da = da.assign_coords(
            spread=(("season", "depth"), np.broadcast_to(spread, values.shape).astype(float))
        )
    return {
        "aligned": xr.Dataset({"value": da}),
        "metrics": None,
        "units": units,
        "standard_name": variable,
        "label": source,
        "labels": (source,),
    }


def _seasonal_profile_item(
    variable: str = TEMPERATURE,
    *,
    test: str = "run_new",
    reference: str = "whots",
    units: str = "degC",
    offset: float = 0.6,
    seasons=("DJF", "MAM", "JJA", "SON"),
    n: int = 8,
    test_spread=None,
    reference_spread=None,
    metrics: dict | None = "auto",
) -> dict:
    """A comparison item whose aligned trio still has a season dim -- the shape
    ``compose()``'s :func:`~ocean_skill.plot.profile.fan_season` splits into
    one item per season. ``test_spread``/``reference_spread`` land as data
    variables on ``aligned``, exactly as ``align._split_spread`` writes them.
    """
    depths = np.linspace(5.0, 150.0, n)
    base = 20.0 - 0.08 * depths
    values = base[None, :] + np.arange(len(seasons))[:, None]
    reference_da = xr.DataArray(
        values,
        dims=("season", "z"),
        coords={"season": list(seasons), "z": depths},
        attrs={"units": units},
    ).assign_coords(lon=-144.245, lat=49.978)
    aligned = xr.Dataset(
        {
            "reference": reference_da,
            "test": reference_da + offset,
            "difference": reference_da * 0 + offset,
        }
    )
    aligned["reference"].attrs["units"] = units
    if reference_spread is not None:
        aligned["reference_spread"] = (
            ("season", "z"),
            np.broadcast_to(reference_spread, values.shape).astype(float),
        )
    if test_spread is not None:
        aligned["test_spread"] = (
            ("season", "z"),
            np.broadcast_to(test_spread, values.shape).astype(float),
        )
    if metrics == "auto":
        metrics = {
            "bias": offset,
            "rmse": abs(offset) + 0.1,
            "corr": 0.97,
            "n": n * len(seasons),
            "std_test": 2.8,
            "std_reference": 2.8,
            "crmsd": 0.1,
            "sigma_ratio": 1.0,
            "variable": variable,
        }
    return {
        "aligned": aligned,
        "metrics": metrics,
        "units": units,
        "standard_name": variable,
        "label": None,
        "labels": (test, reference),
    }


def _spec(items, **options) -> PlotSpec:
    return PlotSpec(
        family="profile",
        items=items if isinstance(items, list) else [items],
        options=options,
    )


def _matplotlib_lines(fig):
    """``[(label, colour, linestyle, marker), ...]`` in drawing order."""
    out = []
    for ax in fig.axes:
        for line in ax.get_lines():
            label = line.get_label()
            if label.startswith("_"):
                continue
            out.append(
                (label, line.get_color(), line.get_linestyle(), line.get_marker())
            )
    return out


def _holoviews_lines(obj):
    import holoviews as hv

    inverse = {v: k for k, v in _style.BOKEH_DASHES.items()}
    out = []
    for curve in obj.traverse(lambda x: x, [hv.Curve]):
        kwargs = curve.opts.get("style").kwargs
        if not curve.label:
            continue
        out.append(
            (
                curve.label,
                kwargs.get("color"),
                inverse.get(kwargs.get("line_dash"), kwargs.get("line_dash")),
            )
        )
    return out


def _matplotlib_titles(fig):
    return [ax.get_title() for ax in fig.axes if ax.get_title()]


def _holoviews_titles(obj):
    import holoviews as hv

    return [
        element.opts.get("plot").kwargs.get("title")
        for element in obj.traverse(lambda x: x, [hv.Overlay])
        if element.opts.get("plot").kwargs.get("title")
    ]


def _matplotlib_bands(fig):
    """``[(facecolor_hex, alpha, x_min, x_max), ...]`` for every filled band, in
    drawing order -- one per matched ``ax.fill_betweenx`` collection."""
    from matplotlib.colors import to_hex

    out = []
    for ax in fig.axes:
        for coll in ax.collections:
            facecolors = coll.get_facecolor()
            verts = np.concatenate([p.vertices for p in coll.get_paths()])
            out.append(
                (
                    to_hex(facecolors[0]) if len(facecolors) else None,
                    coll.get_alpha(),
                    float(verts[:, 0].min()),
                    float(verts[:, 0].max()),
                )
            )
    return out


def _holoviews_bands(obj):
    """``[(fill_color, fill_alpha, x_min, x_max), ...]`` for every ``hv.Polygons``."""
    import holoviews as hv

    out = []
    for poly in obj.traverse(lambda x: x, [hv.Polygons]):
        kwargs = poly.opts.get("style").kwargs
        coords = np.concatenate(list(poly.data)) if poly.data else np.empty((0, 2))
        out.append(
            (
                kwargs.get("fill_color"),
                kwargs.get("fill_alpha"),
                float(coords[:, 0].min()) if coords.size else None,
                float(coords[:, 0].max()) if coords.size else None,
            )
        )
    return out


def _bokeh_y_range(obj):
    import holoviews as hv
    from bokeh.plotting import figure

    plot = hv.render(obj, backend="bokeh")
    ranges = [f.y_range for f in plot.select({"type": figure})]
    return [(r.start, r.end) for r in ranges]


# -- the standing rule -----------------------------------------------------------------


def test_both_renderers_draw_the_same_lines_in_the_same_order():
    """The one test the whole two-module split exists to make possible."""
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3", offset=-0.2)]
    static = _matplotlib_lines(render(_spec(items), renderer="matplotlib"))
    interactive = _holoviews_lines(render(_spec(items), renderer="holoviews"))
    assert [(a, b, c) for a, b, c, _ in static] == interactive


def test_panel_titles_agree_and_carry_identity_only():
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3")]
    static = _matplotlib_titles(render(_spec(items), renderer="matplotlib"))
    interactive = _holoviews_titles(render(_spec(items), renderer="holoviews"))
    assert static == interactive
    assert static and all("bias" not in t and "rmse" not in t for t in static)


def test_axis_labels_carry_units_and_depth_identically():
    import holoviews as hv

    fig = render(_spec([_profile_item()]), renderer="matplotlib")
    obj = render(_spec([_profile_item()]), renderer="holoviews")
    curve = obj.traverse(lambda x: x, [hv.Curve])[0]
    assert fig.axes[0].get_xlabel() == curve.kdims[0].label == "temperature [degC]"
    assert fig.axes[0].get_ylabel() == curve.vdims[0].label == "Depth [m]"


# -- axis semantics: surface at top, both sign conventions ------------------------------


def test_the_y_axis_reads_surface_at_top_in_both_renderers():
    fig = render(_spec([_profile_item()]), renderer="matplotlib")
    bottom, top = fig.axes[0].get_ylim()
    assert bottom > top  # deep at the axes bottom, shallow at the top
    assert top == pytest.approx(5.0)
    assert bottom == pytest.approx(150.0)

    obj = render(_spec([_profile_item()]), renderer="holoviews")
    (start, end), = _bokeh_y_range(obj)
    assert start > end
    assert end == pytest.approx(5.0)
    assert start == pytest.approx(150.0)


def test_negative_down_z_and_positive_down_depth_render_identical_positive_depths():
    """ROMS's ``z`` (negative-down) and an obs product's ``depth`` (positive-down)
    must draw the very same y-values.
    """
    from_model = _profile_item(zdim="z", negative_down=True)
    from_obs = _profile_item(zdim="DEPTH", negative_down=False)
    fig_model = render(_spec([from_model]), renderer="matplotlib")
    fig_obs = render(_spec([from_obs]), renderer="matplotlib")
    model_ydata = [line.get_ydata() for line in fig_model.axes[0].get_lines()]
    obs_ydata = [line.get_ydata() for line in fig_obs.axes[0].get_lines()]
    for m, o in zip(model_ydata, obs_ydata, strict=True):
        np.testing.assert_allclose(sorted(m), sorted(o))
        assert (np.asarray(m) >= 0).all()


def test_sigma0_axis_labels_as_density_and_still_inverts():
    depths = np.linspace(5.0, 150.0, 8)
    sigma0 = 22.0 + 0.02 * depths  # denser (higher sigma0) with depth
    values = 20.0 - 0.08 * depths
    da = xr.DataArray(
        values, coords={"sigma0": sigma0}, dims="sigma0", attrs={"units": "degC"}
    ).assign_coords(lon=-144.245, lat=49.978)
    item = {
        "aligned": xr.Dataset({"value": da}),
        "metrics": None,
        "units": "degC",
        "standard_name": TEMPERATURE,
        "label": "run_new",
        "labels": ("run_new",),
    }
    fig = render(_spec([item]), renderer="matplotlib")
    assert fig.axes[0].get_ylabel() == "σ₀ [kg/m³]"
    bottom, top = fig.axes[0].get_ylim()
    assert bottom > top  # denser (higher sigma0) at the bottom, same convention


# -- the role -> style policy, reused verbatim from series -------------------------------


def test_the_reference_is_solid_and_the_test_is_dashed():
    lines = _matplotlib_lines(render(_spec([_profile_item()]), renderer="matplotlib"))
    styles = {label: dash for label, _, dash, _ in lines}
    assert styles["whots"] == "-"
    assert styles["run_new"] == "--"


# -- time replaces depth as the marker channel -------------------------------------------


def test_a_varying_time_earns_markers_and_a_constant_one_does_not():
    same = _profile.compose([_single_item(time="2015-01-15")]).panels[0].lines
    assert [line.marker for line in same] == [None]
    varied = _profile.compose(
        [_single_item(time="2015-01-15"), _single_item(time="2015-02-15")]
    ).panels[0].lines
    assert all(line.marker is not None for line in varied)


def test_multi_time_items_get_distinct_legend_labels():
    items = [
        _single_item(time="2015-01-15"),
        _single_item(time="2015-02-15"),
    ]
    lines = _matplotlib_lines(render(_spec(items), renderer="matplotlib"))
    labels = {label for label, _, _, _ in lines}
    assert len(labels) == 2
    assert any("2015-01" in label for label in labels)
    assert any("2015-02" in label for label in labels)


def test_profile_specs_never_carry_a_depth_channel():
    """Depth is the axis, never a style fact -- unlike series, whose LineSpec.depth
    is what earns a varying depth its markers.
    """
    specs = _profile._line_specs(_profile_item(), 0)
    assert all(spec.depth is None for spec in specs)


# -- depth is refused as a channel or a facet --------------------------------------------


def test_depth_encode_is_refused():
    with pytest.raises(ValueError, match="depth is the axis"):
        render(
            _spec([_profile_item()], encode={"marker": "depth"}), renderer="matplotlib"
        )


def test_depth_facet_is_refused():
    with pytest.raises(ValueError, match="depth is the axis"):
        render(_spec([_profile_item()], rows="depth"), renderer="matplotlib")


def test_rows_and_cols_together_are_refused():
    with pytest.raises(ValueError, match="one facet, not two"):
        render(
            _spec([_profile_item()], rows="variable", cols="source"),
            renderer="matplotlib",
        )


# -- composition: two variables merge onto one panel with a top axis by default ---------


def test_two_variables_go_to_a_top_axis_by_default():
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3")]
    fig = render(_spec(items), renderer="matplotlib")
    assert len(fig.axes) == 2  # the panel and its twin
    assert len(_matplotlib_titles(fig)) == 1
    assert fig.axes[1].xaxis.get_label_position() == "top"
    assert fig.axes[1].get_xlabel() == "salinity [1e-3]"

    obj = render(_spec(items), renderer="holoviews")
    assert len(_holoviews_titles(obj)) == 1
    import holoviews as hv
    from bokeh.plotting import figure

    plot = hv.render(obj, backend="bokeh")
    (bokeh_fig,) = plot.select({"type": figure})
    assert "secondary" in bokeh_fig.extra_x_ranges
    from bokeh.models import LinearAxis

    (top_axis,) = [a for a in bokeh_fig.above if isinstance(a, LinearAxis)]
    assert top_axis.axis_label == "salinity [1e-3]"


def test_secondary_x_false_gives_two_columns_sharing_the_depth_axis():
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3")]
    fig = render(_spec(items, secondary_x=False), renderer="matplotlib")
    assert len(fig.axes) == 2
    assert len(_matplotlib_titles(fig)) == 2
    # only the leftmost column carries the shared depth label
    assert fig.axes[0].get_ylabel() == "Depth [m]"
    assert fig.axes[1].get_ylabel() == ""

    obj = render(_spec(items, secondary_x=False), renderer="holoviews")
    assert len(_holoviews_titles(obj)) == 2


def test_secondary_glyphs_ride_the_secondary_range():
    import holoviews as hv
    from bokeh.plotting import figure

    items = [_profile_item(), _profile_item(SALINITY, units="1e-3", offset=-0.2)]
    obj = render(_spec(items), renderer="holoviews")
    plot = hv.render(obj, backend="bokeh")
    (bokeh_fig,) = plot.select({"type": figure})
    renderers = [r for r in bokeh_fig.renderers if hasattr(r, "x_range_name")]
    on_secondary = {r.x_range_name for r in renderers if "secondary" in r.x_range_name}
    on_default = {r.x_range_name for r in renderers if r.x_range_name == "default"}
    assert on_secondary == {"secondary"}
    assert on_default == {"default"}


def test_a_twin_panel_records_its_value_axis_label_colours():
    layout = _profile.compose(
        [_profile_item(), _profile_item(SALINITY, units="1e-3")]
    )
    panel = layout.panels[0]
    assert panel.xlabel_color == panel.lines[0].color
    assert panel.secondary_xlabel_color == panel.secondary[0].color
    assert panel.xlabel_color != panel.secondary_xlabel_color


def test_twin_axis_label_colours_match_their_lines_in_both_renderers():
    import holoviews as hv
    from bokeh.plotting import figure

    items = [_profile_item(), _profile_item(SALINITY, units="1e-3")]
    fig = render(_spec(items), renderer="matplotlib")
    ax, twin = fig.axes
    assert ax.xaxis.label.get_color() == ax.get_lines()[0].get_color()
    assert twin.xaxis.label.get_color() == twin.get_lines()[0].get_color()

    obj = render(_spec(items), renderer="holoviews")
    plot = hv.render(obj, backend="bokeh")
    (bokeh_fig,) = plot.select({"type": figure})
    by_label = {axis.axis_label: axis for axis in bokeh_fig.xaxis}
    assert by_label[ax.get_xlabel()].axis_label_text_color == ax.get_lines()[0].get_color()
    assert (
        by_label[twin.get_xlabel()].axis_label_text_color
        == twin.get_lines()[0].get_color()
    )


def test_an_axis_carrying_two_colours_leaves_value_labels_uncoloured():
    items = [
        _profile_item(test="run_new", reference="whots"),
        _profile_item(SALINITY, units="1e-3", test="run_old", reference="argo"),
    ]
    layout = _profile.compose(items, encode={"color": "source"})
    panel = layout.panels[0]
    assert panel.xlabel_color is None
    assert panel.secondary_xlabel_color is None


def test_single_variable_secondary_x_is_a_no_op():
    layout = _profile.compose([_profile_item()], secondary_x=True)
    assert len(layout.panels) == 1
    assert layout.panels[0].secondary == ()
    fig = render(_spec([_profile_item()]), renderer="matplotlib")
    assert len(fig.axes) == 1


def test_three_variables_ignore_secondary_x():
    items = [
        _profile_item(),
        _profile_item(SALINITY, units="1e-3"),
        _profile_item("sea_water_ph_reported_on_total_scale", units="1"),
    ]
    fig = render(_spec(items), renderer="matplotlib")
    assert len(fig.axes) == 3


def test_a_facet_wins_over_secondary_x():
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3")]
    fig = render(_spec(items, cols="variable"), renderer="matplotlib")
    assert len(fig.axes) == 2
    assert len(_matplotlib_titles(fig)) == 2


def test_merged_metrics_box_prefixes_each_variables_row():
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3")]
    fig = render(_spec(items, metric_keys=("bias",)), renderer="matplotlib")
    box = fig.axes[0].texts[0].get_text()
    assert box.count("\n") == 1  # two rows
    rows = box.split("\n")
    assert any("temperature" in r.lower() for r in rows)
    assert any("salinity" in r.lower() for r in rows)


def test_merged_panel_lines_agree_across_renderers():
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3", offset=-0.2)]
    static = _matplotlib_lines(render(_spec(items), renderer="matplotlib"))
    interactive = _holoviews_lines(render(_spec(items), renderer="holoviews"))
    assert [(a, b, c) for a, b, c, _ in static] == interactive


def test_spread_bands_follow_their_variable_onto_the_twin_axis():
    items = [
        _seasonal_single_item(spread=0.5, seasons=("JJA",)),
        _seasonal_single_item(
            variable=SALINITY, units="1e-3", spread=0.3, seasons=("JJA",)
        ),
    ]
    fig = render(_spec(items), renderer="matplotlib")
    ax, twin = fig.axes
    assert ax.collections  # temperature's band on the primary axis
    assert twin.collections  # salinity's band on the twin

    obj = render(_spec(items), renderer="holoviews")
    import holoviews as hv
    from bokeh.plotting import figure

    plot = hv.render(obj, backend="bokeh")
    (bokeh_fig,) = plot.select({"type": figure})
    from bokeh.models import MultiPolygons

    secondary_patches = [
        r
        for r in bokeh_fig.renderers
        if isinstance(r.glyph, MultiPolygons) and r.x_range_name == "secondary"
    ]
    assert secondary_patches


def test_depth_range_spans_both_variables():
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3", max_depth=300.0)]
    fig = render(_spec(items), renderer="matplotlib")
    bottom, _ = fig.axes[0].get_ylim()
    assert bottom == pytest.approx(300.0)


def test_the_title_clears_the_twin_axis_instead_of_overlapping_it():
    """A twin's own ticks and axis label are drawn above the shared top spine,
    exactly where the title's default pad would otherwise land it (matplotlib
    does not know to stack them on its own)."""
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3")]
    fig = render(_spec(items), renderer="matplotlib")
    ax, twin = fig.axes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bottom = ax.title.get_window_extent(renderer).y0
    twin_label_top = twin.xaxis.label.get_window_extent(renderer).y1
    assert title_bottom >= twin_label_top


def test_secondary_y_on_a_profile_hints_at_secondary_x():
    with pytest.raises(TypeError, match="secondary_x"):
        render(_spec([_profile_item()], secondary_y=True), renderer="matplotlib")
    with pytest.warns(UserWarning, match="secondary_x"):
        render(_spec([_profile_item()], secondary_y=True), renderer="holoviews")


def test_a_single_variable_stays_one_panel():
    fig = render(_spec([_profile_item()]), renderer="matplotlib")
    assert len(fig.axes) == 1


# -- the statistics box ------------------------------------------------------------------


def test_the_metrics_appear_off_the_title_in_both_renderers():
    import holoviews as hv

    fig = render(_spec([_profile_item()]), renderer="matplotlib")
    static = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("bias=" in t for t in static)

    obj = render(_spec([_profile_item()]), renderer="holoviews")
    interactive = [t.text for t in obj.traverse(lambda x: x, [hv.Text])]
    assert any("bias=" in t for t in interactive)


def test_the_box_and_the_legend_never_take_the_same_corner():
    layout = _profile.compose([_profile_item()], metric_keys=("bias",))
    panel = layout.panels[0]
    assert panel.metrics_corner != panel.legend_corner


def test_no_sample_counts_are_drawn_on_the_figure():
    fig = render(_spec([_profile_item()]), renderer="matplotlib")
    drawn = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert not [t for t in drawn if "n=" in t]


# -- a single source, no comparison -------------------------------------------------------


def test_a_single_source_item_draws_one_solid_line_in_both_renderers():
    item = _single_item()
    static = _matplotlib_lines(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_lines(render(_spec([item]), renderer="holoviews"))
    assert [(a, b, c) for a, b, c, _ in static] == interactive
    (label, _, dash, _), = static
    assert label == "run_new"
    assert dash == "-"


def test_a_single_source_item_has_no_statistics_box_in_either_renderer():
    import holoviews as hv

    fig = render(_spec([_single_item()]), renderer="matplotlib")
    assert not [t.get_text() for ax in fig.axes for t in ax.texts]

    obj = render(_spec([_single_item()]), renderer="holoviews")
    assert not obj.traverse(lambda x: x, [hv.Text])


def test_panel_title_reads_place():
    static = _matplotlib_titles(render(_spec([_single_item()]), renderer="matplotlib"))
    interactive = _holoviews_titles(
        render(_spec([_single_item()]), renderer="holoviews")
    )
    assert static == interactive
    assert any("144.2°W" in t and "50.0°N" in t for t in static)


def test_panel_title_shows_a_shared_single_time():
    item = _single_item(time="2015-06-15")
    static = _matplotlib_titles(render(_spec([item]), renderer="matplotlib"))
    assert any("2015-06-15" in t for t in static)


# -- axis limits -----------------------------------------------------------------------


def test_ylim_orders_shallow_to_deep_regardless_of_input_order():
    fig = render(_spec([_profile_item()], ylim=(0.0, 200.0)), renderer="matplotlib")
    bottom, top = fig.axes[0].get_ylim()
    assert (bottom, top) == (200.0, 0.0)

    obj = render(_spec([_profile_item()], ylim=(0.0, 200.0)), renderer="holoviews")
    (start, end), = _bokeh_y_range(obj)
    assert (start, end) == (200.0, 0.0)


def test_xlim_bounds_the_value_axis():
    fig = render(_spec([_profile_item()], xlim=(0.0, 30.0)), renderer="matplotlib")
    assert fig.axes[0].get_xlim() == (0.0, 30.0)


# -- mark ---------------------------------------------------------------------------------


def test_step_mark_is_refused():
    with pytest.raises(ValueError, match="not a profile mark"):
        render(_spec([_profile_item()], mark="step"), renderer="matplotlib")


def test_mark_reaches_both_renderers_rather_than_being_dropped():
    import holoviews as hv

    fig = render(_spec([_profile_item()], mark="line+marker"), renderer="matplotlib")
    assert any(
        line.get_marker() not in ("", "None", None) for line in fig.axes[0].lines
    )

    obj = render(_spec([_profile_item()], mark="line+marker"), renderer="holoviews")
    assert obj.traverse(lambda x: x, [hv.Scatter])


def test_a_single_depth_cast_draws_a_marker_in_both_renderers():
    """A single-bottle visit (a ragged timeSeriesProfile station's own
    single-depth cast, or an already-collapsed profile) has nothing for a bare
    *line* to connect -- under the default mark="line" a one-point line is a
    zero-length segment, invisible. Both renderers force a marker regardless of
    `mark` or whether time varies (the ordinary marker-earning condition) when a
    line has fewer than two finite points.
    """
    import holoviews as hv

    item = _single_item(n=1)
    fig = render(_spec([item]), renderer="matplotlib")
    (line,) = fig.axes[0].get_lines()
    assert line.get_marker() not in ("", "None", None)

    obj = render(_spec([item]), renderer="holoviews")
    assert obj.traverse(lambda x: x, [hv.Scatter])


# -- a surviving season axis fans into one line per season, in both renderers --------------


def test_seasonal_overlay_draws_one_line_per_season_in_both_renderers():
    item = _seasonal_single_item()
    static = _matplotlib_lines(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_lines(render(_spec([item]), renderer="holoviews"))
    assert [(a, b, c) for a, b, c, _ in static] == interactive
    assert [label for label, *_ in static] == ["DJF", "MAM", "JJA", "SON"]


def test_season_colors_follow_coordinate_order_not_alphabetical():
    """xarray's own groupby("time.season") sorts alphabetically (DJF, JJA, MAM,
    SON); a coordinate that already reads in a custom given order must colour
    in *that* order, not be re-sorted."""
    item = _seasonal_single_item(seasons=("JFMA", "MJJA", "SOND"))
    static = _matplotlib_lines(render(_spec([item]), renderer="matplotlib"))
    labels = [label for label, *_ in static]
    colors = [color for _, color, _, _ in static]
    assert labels == ["JFMA", "MJJA", "SOND"]
    assert colors == list(_style.COLOR_CYCLE[:3])


def test_seasonal_comparison_pairs_share_a_color_and_split_by_dash():
    item = _seasonal_profile_item()
    static = _matplotlib_lines(render(_spec([item]), renderer="matplotlib"))
    by_season: dict[str, list] = {}
    for label, color, dash, _ in static:
        season = label.split(" · ")[-1] if " · " in label else label
        by_season.setdefault(season, []).append((color, dash))
    assert len(by_season) == 4
    for pair in by_season.values():
        assert len(pair) == 2
        (color_a, dash_a), (color_b, dash_b) = pair
        assert color_a == color_b  # one season, one colour
        assert {dash_a, dash_b} == {"-", "--"}  # reference solid, test dashed


def test_cols_season_makes_one_panel_per_season_in_both_renderers():
    item = _seasonal_single_item()
    static = render(_spec([item], cols="season"), renderer="matplotlib")
    interactive = render(_spec([item], cols="season"), renderer="holoviews")
    assert len(static.axes) == 4
    assert _matplotlib_titles(static) == _holoviews_titles(interactive)
    assert all("DJF" in t or "MAM" in t or "JJA" in t or "SON" in t for t in _matplotlib_titles(static))


def test_rows_season_stacks_panels_in_chronological_order():
    item = _seasonal_single_item()
    fig = render(_spec([item], rows="season"), renderer="matplotlib")
    titles = _matplotlib_titles(fig)
    seasons_in_order = [t.split(" · ")[-1] for t in titles]
    assert seasons_in_order == ["DJF", "MAM", "JJA", "SON"]


def test_depth_facet_still_refused_with_seasons_present():
    with pytest.raises(ValueError, match="depth"):
        render(_spec([_seasonal_single_item()], rows="depth"), renderer="matplotlib")


def test_a_single_season_changes_nothing():
    """One season is no more distinguishing than one variable or one source --
    no season colouring, no season in the label."""
    item = _seasonal_single_item(seasons=("JJA",))
    static = _matplotlib_lines(render(_spec([item]), renderer="matplotlib"))
    assert len(static) == 1
    label, color, _, _ = static[0]
    assert label == "run_new"  # not "run_new · JJA"
    assert color == _style.COLOR_CYCLE[0]


def test_explicit_encode_beats_the_season_color_default():
    items = [
        _seasonal_single_item(source="run_new"),
        _seasonal_single_item(source="run_old"),
    ]
    static = _matplotlib_lines(
        render(_spec(items, encode={"color": "source"}), renderer="matplotlib")
    )
    colors = {color for _, color, _, _ in static}
    # coloured by source (2 sources), not season (4 seasons) -- exactly 2 colours
    assert len(colors) == 2


def test_fanned_comparison_shows_one_metrics_row_not_one_per_season():
    item = _seasonal_profile_item()
    fig = render(_spec([item], metric_keys=("bias",)), renderer="matplotlib")
    box_text = fig.axes[0].texts[0].get_text() if fig.axes[0].texts else ""
    assert box_text.count("\n") == 0  # one row, not four identical ones


# -- the mean±spread envelope, in both renderers --------------------------------------------


def test_seasonal_envelope_draws_the_same_bands_in_both_renderers():
    item = _seasonal_single_item(spread=0.5)
    static = _matplotlib_bands(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_bands(render(_spec([item]), renderer="holoviews"))
    assert len(static) == 4
    assert len(interactive) == 4
    for (s_color, s_alpha, s_lo, s_hi), (i_color, i_alpha, i_lo, i_hi) in zip(
        static, interactive, strict=True
    ):
        assert s_color == i_color
        assert s_alpha == pytest.approx(i_alpha)
        assert s_lo == pytest.approx(i_lo)
        assert s_hi == pytest.approx(i_hi)


def test_band_color_matches_its_own_lines_color():
    item = _seasonal_single_item(spread=0.5)
    lines = _matplotlib_lines(render(_spec([item]), renderer="matplotlib"))
    bands = _matplotlib_bands(render(_spec([item]), renderer="matplotlib"))
    line_colors = [color for _, color, _, _ in lines]
    band_colors = [color for color, *_ in bands]
    assert band_colors == line_colors


def test_band_alpha_is_the_shared_constant():
    item = _seasonal_single_item(spread=0.5)
    bands = _matplotlib_bands(render(_spec([item]), renderer="matplotlib"))
    assert all(alpha == pytest.approx(_style.BAND_ALPHA) for _, alpha, _, _ in bands)


def test_no_spread_means_no_band_in_either_renderer():
    item = _seasonal_single_item()  # no spread=
    static = _matplotlib_bands(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_bands(render(_spec([item]), renderer="holoviews"))
    assert static == []
    assert interactive == []


def test_bands_carry_no_legend_entry_in_either_renderer():
    item = _seasonal_single_item(spread=0.5)
    static = _matplotlib_lines(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_lines(render(_spec([item]), renderer="holoviews"))
    # still exactly one legend entry per season -- the band added no entries
    assert len(static) == 4
    assert len(interactive) == 4


def test_a_comparisons_envelope_carries_each_lanes_own_spread():
    item = _seasonal_profile_item(test_spread=0.3, reference_spread=0.6)
    static = _matplotlib_bands(render(_spec([item]), renderer="matplotlib"))
    assert len(static) == 8  # 4 seasons x 2 lanes


def test_a_nan_gap_splits_the_band_into_runs_in_both_renderers():
    item = _seasonal_single_item(seasons=("JJA",), n=6, spread=0.3)
    aligned = item["aligned"]
    aligned["value"].values[0, 2] = np.nan  # a gap in the middle of the water column
    static = _matplotlib_bands(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_bands(render(_spec([item]), renderer="holoviews"))
    # one contiguous run before the gap, one after -- never one run spanning it
    assert len(static) == 2
    assert len(interactive) == 2


# -- option plumbing -----------------------------------------------------------------------


def test_a_gridded_only_option_raises_statically():
    with pytest.raises(TypeError, match="profile"):
        render(
            _spec([_profile_item()], domain=(-150.0, 45.0, -140.0, 55.0)),
            renderer="matplotlib",
        )


def test_the_interactive_renderer_warns_for_static_only_line_styling():
    with pytest.warns(UserWarning, match="only affect the static"):
        render(
            _spec([_profile_item()], line_kwargs={"linewidth": 3}),
            renderer="holoviews",
        )


def test_profile_accepts_no_kwargs_catch_all():
    """A ``**kwargs`` in the signature would silently disable option validation."""
    import inspect

    from ocean_skill.plot.matplotlib_renderer import profile

    kinds = [p.kind for p in inspect.signature(profile).parameters.values()]
    assert inspect.Parameter.VAR_KEYWORD not in kinds


def test_profile_is_registered_everywhere_it_has_to_be():
    from ocean_skill.plot.matplotlib_renderer import _top_level_options
    from ocean_skill.plot.spec import FAMILIES

    assert "profile" in FAMILIES
    for option in (
        "xlim",
        "ylim",
        "panel_aspect",
        "legend",
        "encode",
        "mark",
        "secondary_x",
    ):
        assert option in _top_level_options(), option


# -- zoom/size reach the interactive frame -------------------------------------------------


def _hv_frame_widths(obj):
    import holoviews as hv
    from bokeh.plotting import figure

    plot = hv.render(obj, backend="bokeh")
    return [f.frame_width for f in plot.select({"type": figure}) if f.frame_width]


def test_zoom_grows_the_interactive_frame():
    items = [_profile_item()]
    plain = _hv_frame_widths(render(_spec(items), renderer="holoviews"))
    zoomed = _hv_frame_widths(render(_spec(items, zoom=2.0), renderer="holoviews"))
    assert plain and len(zoomed) == len(plain)
    assert all(z > p for z, p in zip(zoomed, plain, strict=True))


def test_a_named_canvas_reaches_the_interactive_frame():
    items = [_profile_item()]
    page = _hv_frame_widths(render(_spec(items, size="page"), renderer="holoviews"))
    column = _hv_frame_widths(
        render(_spec(items, size="column"), renderer="holoviews")
    )
    assert all(c < p for c, p in zip(column, page, strict=True))
