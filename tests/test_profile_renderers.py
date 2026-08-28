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
    time: str | None = None,
) -> dict:
    """One comparison item, shaped exactly as ``Comparison.as_item()`` builds it.

    ``zdim``/``negative_down`` cover both vertical-coordinate conventions this
    family has to read identically: ROMS's own ``z`` (negative-down, from
    :func:`ocean_skill.roms.to_depth`) and an observational product's ``depth``
    (already positive-down).
    """
    depths = np.linspace(5.0, 150.0, n)
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


# -- composition: variables become columns, not a secondary axis ------------------------


def test_two_variables_become_two_columns_sharing_the_depth_axis():
    items = [_profile_item(), _profile_item(SALINITY, units="1e-3")]
    fig = render(_spec(items), renderer="matplotlib")
    assert len(fig.axes) == 2
    assert len(_matplotlib_titles(fig)) == 2
    # only the leftmost column carries the shared depth label
    assert fig.axes[0].get_ylabel() == "Depth [m]"
    assert fig.axes[1].get_ylabel() == ""

    obj = render(_spec(items), renderer="holoviews")
    assert len(_holoviews_titles(obj)) == 2


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
    for option in ("xlim", "ylim", "panel_aspect", "legend", "encode", "mark"):
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
