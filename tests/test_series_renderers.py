"""Tests for the ``series`` family, in both renderers.

The standing rule for this package is that a plot change lands in the static *and* the
interactive renderer, and that a test asserts the two **agree** rather than that each
runs. So the first test here compares the drawn lines one by one, and the rest pin the
policy (:mod:`ocean_skill.plot.style`) independently of either backend — the two drifted
apart once before, on a legend that had been fixed in one of them.

Items are built by hand, exactly as ``tests/test_renderers.py`` does, so a renderer is
tested without going through ``compare()``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.plot import series as _series
from ocean_skill.plot import style as _style
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

TEMPERATURE = "sea_water_temperature"
SALINITY = "sea_water_practical_salinity"


def _item(
    variable: str = TEMPERATURE,
    *,
    test: str = "oceansoda_ethz",
    reference: str = "ooi-papa-ctd",
    units: str = "degC",
    offset: float = 0.6,
    depth: float | None = 8.0,
    n: int = 36,
) -> dict:
    """One comparison item, shaped exactly as ``Comparison.as_item()`` builds it."""
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    values = 8.0 + 4.0 * np.sin(np.arange(n) / 1.9)
    reference_da = xr.DataArray(
        values, coords={"time": time}, dims="time", attrs={"units": units}
    ).assign_coords(lon=-144.245, lat=49.978)
    if depth is not None:
        reference_da = reference_da.assign_coords(depth=depth)
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
    depth: float | None = None,
    lon: float = -144.245,
    lat: float = 49.978,
    lon_name: str = "lon",
    lat_name: str = "lat",
    n: int = 12,
) -> dict:
    """One single-source series item, shaped as ``Field._series_items`` builds it.

    No reference, no residual, no metrics -- one lone value, at a place, over time.
    """
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    values = 8.0 + 4.0 * np.sin(np.arange(n) / 1.9)
    da = xr.DataArray(
        values, coords={"time": time}, dims="time", attrs={"units": units}
    ).assign_coords(**{lon_name: lon, lat_name: lat})
    aligned = xr.Dataset({"value": da})
    if depth is not None:
        aligned.attrs["actual_depth"] = depth
    return {
        "aligned": aligned,
        "metrics": None,
        "units": units,
        "standard_name": variable,
        "label": source,
        "labels": (source,),
    }


def _spec(items, **options) -> PlotSpec:
    return PlotSpec(
        family="series",
        items=items if isinstance(items, list) else [items],
        options=options,
    )


def _matplotlib_lines(fig):
    """``[(label, colour, linestyle, marker), ...]`` in drawing order."""
    out = []
    for ax in fig.axes:
        for line in ax.get_lines():
            label = line.get_label()
            if label.startswith("_"):  # the residual strip's zero line
                continue
            out.append(
                (label, line.get_color(), line.get_linestyle(), line.get_marker())
            )
    return out


def _holoviews_lines(obj):
    """Return the same tuples from the interactive object, dashes translated back."""
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
                None,
            )
        )
    return out


def _holoviews_titles(obj):
    import holoviews as hv

    return [
        element.opts.get("plot").kwargs.get("title")
        for element in obj.traverse(lambda x: x, [hv.Overlay])
        if element.opts.get("plot").kwargs.get("title")
    ]


def _matplotlib_titles(fig):
    return [ax.get_title() for ax in fig.axes if ax.get_title()]


# -- the standing rule -----------------------------------------------------------------


def test_both_renderers_draw_the_same_lines_in_the_same_order():
    """The one test the whole two-module split exists to make possible."""
    items = [_item(), _item(SALINITY, units="1e-3", offset=-0.2)]
    static = _matplotlib_lines(render(_spec(items), renderer="matplotlib"))
    interactive = _holoviews_lines(render(_spec(items), renderer="holoviews"))
    assert [(a, b, c) for a, b, c, _ in static] == [
        (a, b, c) for a, b, c, _ in interactive
    ]


def test_panel_titles_agree_and_carry_identity_only():
    """The title says what the panel is; the numbers live in their own box."""
    items = [_item(), _item(SALINITY, units="1e-3")]
    static = _matplotlib_titles(
        render(_spec(items, secondary_y=False), renderer="matplotlib")
    )
    interactive = _holoviews_titles(
        render(_spec(items, secondary_y=False), renderer="holoviews")
    )
    assert static == interactive
    assert static and all("bias" not in t and "rmse" not in t for t in static)


def test_axis_labels_carry_the_units_identically():
    """``hv.Dimension(unit=..)`` prints ``(degC)``; matplotlib prints ``[degC]``."""
    import holoviews as hv

    fig = render(_spec([_item()]), renderer="matplotlib")
    obj = render(_spec([_item()]), renderer="holoviews")
    curve = obj.traverse(lambda x: x, [hv.Curve])[0]
    assert fig.axes[0].get_ylabel() == curve.vdims[0].label == "temperature [degC]"


# -- the role -> style policy ----------------------------------------------------------


def test_the_reference_is_solid_and_the_test_is_dashed():
    assert _style.linestyle_for("reference") == "-"
    assert _style.linestyle_for("test") == "--"
    lines = _matplotlib_lines(render(_spec([_item()]), renderer="matplotlib"))
    styles = {label: dash for label, _, dash, _ in lines}
    assert styles["ooi-papa-ctd"] == "-"
    assert styles["oceansoda_ethz"] == "--"


def test_model_versus_model_is_symmetric():
    """Role decides, not the source name — so swapping the roles swaps the dashes."""
    forward = _matplotlib_lines(
        render(_spec([_item(test="runB", reference="runA")]), renderer="matplotlib")
    )
    reverse = _matplotlib_lines(
        render(_spec([_item(test="runA", reference="runB")]), renderer="matplotlib")
    )
    assert dict((label, dash) for label, _, dash, _ in forward) == {
        "runA": "-",
        "runB": "--",
    }
    assert dict((label, dash) for label, _, dash, _ in reverse) == {
        "runA": "--",
        "runB": "-",
    }


def test_a_pair_of_one_variable_shares_one_colour():
    lines = _matplotlib_lines(render(_spec([_item()]), renderer="matplotlib"))
    assert len({colour for _, colour, _, _ in lines}) == 1


def test_two_variables_get_two_colours_and_keep_their_dashes():
    items = [_item(), _item(SALINITY, units="1e-3")]
    lines = _matplotlib_lines(render(_spec(items), renderer="matplotlib"))
    assert len({colour for _, colour, _, _ in lines}) == 2
    assert sorted(dash for _, _, dash, _ in lines) == ["-", "-", "--", "--"]


def test_a_second_source_takes_the_next_dash_not_the_next_colour():
    items = [_item(test="modelA"), _item(test="modelB")]
    lines = _matplotlib_lines(render(_spec(items), renderer="matplotlib"))
    dashes = {label: dash for label, _, dash, _ in lines}
    assert dashes["ooi-papa-ctd"] == "-"
    assert {dashes["modelA"], dashes["modelB"]} == {"--", ":"}


def test_encode_can_move_a_channel_but_not_the_role():
    specs = [
        _style.LineSpec(role="reference", source="obs", variable="a", depth=1.0),
        _style.LineSpec(role="test", source="model", variable="a", depth=1.0),
    ]
    resolved = _style.resolve(specs, encode={"linestyle": None})
    assert [line.linestyle for line in resolved] == ["-", "--"]
    with pytest.raises(ValueError, match="not a channel"):
        _style.resolve(specs, encode={"dash": "source"})


def test_the_colour_cycle_is_tab20_darks_then_lights():
    """20 colours, not 10, so a Taylor/Target diagram with >10 series stops repeating.

    ``summary._group_styles`` and the interactive Target both index this same cycle by
    level, so a series panel's colours stay identical to a Taylor diagram's for one
    comparison. The first 10 must stay exactly tab10 so existing plots with <=10 series
    are unaffected; matplotlib is now a core dependency (pyproject.toml), so the cycle is
    derived from the colormap rather than hand-transcribed.
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    assert len(_style.COLOR_CYCLE) == 20
    assert len(set(_style.COLOR_CYCLE)) == 20
    assert _style.COLOR_CYCLE[:10] == tuple(
        to_hex(colormaps["tab10"](i)) for i in range(10)
    )
    assert _style.COLOR_CYCLE == tuple(
        to_hex(colormaps["tab20"](i)) for i in (*range(0, 20, 2), *range(1, 20, 2))
    )


def test_markers_are_subsampled_the_same_way_for_both_renderers():
    """Bokeh has no ``markevery``, so the indices are shared, not reinvented."""
    assert _style.markevery_indices(4) == [0, 1, 2, 3]
    assert len(_style.markevery_indices(3000)) <= 21
    assert _style.markevery_indices(3000)[:2] == [0, 150]


def test_a_varying_depth_earns_markers_and_a_constant_one_does_not():
    same = _style.resolve(
        [
            _style.LineSpec(role="reference", source="obs", variable="a", depth=8.0),
            _style.LineSpec(role="test", source="model", variable="a", depth=8.0),
        ]
    )
    assert [line.marker for line in same] == [None, None]
    varied = _style.resolve(
        [
            _style.LineSpec(role="reference", source="obs", variable="a", depth=8.0),
            _style.LineSpec(role="reference", source="obs", variable="a", depth=30.0),
        ]
    )
    assert all(line.marker is not None for line in varied)


def test_legend_entries_spell_levels_through_pretty_level():
    items = [_item(), _item(SALINITY, units="1e-3")]
    labels = {
        label
        for label, _, _, _ in _matplotlib_lines(
            render(_spec(items), renderer="matplotlib")
        )
    }
    assert any(label.endswith("temperature") for label in labels)
    assert not any("sea_water_temperature" in label for label in labels)


# -- composition -----------------------------------------------------------------------


def test_two_variables_go_to_a_secondary_axis_by_default():
    items = [_item(), _item(SALINITY, units="1e-3")]
    fig = render(_spec(items), renderer="matplotlib")
    assert len(fig.axes) == 2  # one panel plus its twin
    assert len(_matplotlib_titles(fig)) == 1

    obj = render(_spec(items), renderer="holoviews")
    assert len(_holoviews_titles(obj)) == 1


def test_secondary_y_false_stacks_them_instead():
    items = [_item(), _item(SALINITY, units="1e-3")]
    fig = render(_spec(items, secondary_y=False), renderer="matplotlib")
    assert len(_matplotlib_titles(fig)) == 2


# -- axis label colour ------------------------------------------------------------------


def test_a_twin_panel_records_its_axis_label_colours():
    """Each label's colour is read off its own axis's lines, not the other's."""
    items = [_item(), _item(SALINITY, units="1e-3")]
    layout = _series.compose(items)
    panel = layout.panels[0]
    assert panel.ylabel_color == panel.lines[0].color
    assert panel.secondary_ylabel_color == panel.secondary[0].color
    assert panel.ylabel_color != panel.secondary_ylabel_color


def test_an_axis_carrying_two_colours_leaves_its_label_uncoloured():
    items = [_item(), _item(SALINITY, units="1e-3")]
    layout = _series.compose(items, encode={"color": "source"})
    assert layout.panels[0].ylabel_color is None
    assert layout.panels[0].secondary_ylabel_color is None


def test_a_lone_panel_keeps_its_label_uncoloured():
    """No twin axis means no ambiguity to resolve -- the label stays default."""
    layout = _series.compose([_item()])
    assert layout.panels[0].ylabel_color is None


def test_twin_axis_label_colours_match_their_lines_in_both_renderers():
    import holoviews as hv

    items = [_item(), _item(SALINITY, units="1e-3", offset=-0.2)]
    fig = render(_spec(items), renderer="matplotlib")
    primary, twin = fig.axes  # per test_two_variables_go_to_a_secondary_axis_by_default
    static = {
        primary.get_ylabel(): primary.yaxis.label.get_color(),
        twin.get_ylabel(): twin.yaxis.label.get_color(),
    }

    obj = render(_spec(items), renderer="holoviews")
    overlay = obj.traverse(lambda x: x, [hv.Overlay])[0]
    bokeh_fig = hv.render(overlay, backend="bokeh")  # runs the finalize hook
    # The metrics box is an hv.Text with its own default "y" dimension, which under
    # multi_y earns bokeh a third, unrelated axis (pre-existing, nothing to do with
    # colour) -- restrict the comparison to the two real data axes.
    interactive = {
        axis.axis_label: axis.axis_label_text_color
        for axis in bokeh_fig.yaxis
        if axis.axis_label in static
    }

    assert static == interactive
    assert set(static.values()) == {_style.COLOR_CYCLE[0], _style.COLOR_CYCLE[1]}

    # the tick numbers take the same colour as their label, in both renderers
    assert primary.yaxis.get_ticklabels()[0].get_color() == static[primary.get_ylabel()]
    assert twin.yaxis.get_ticklabels()[0].get_color() == static[twin.get_ylabel()]
    for axis in bokeh_fig.yaxis:
        if axis.axis_label in interactive:
            assert axis.major_label_text_color == interactive[axis.axis_label]


def test_a_single_axis_label_is_not_coloured():
    import holoviews as hv

    fig = render(_spec([_item()]), renderer="matplotlib")
    assert fig.axes[0].yaxis.label.get_color() not in _style.COLOR_CYCLE

    obj = render(_spec([_item()]), renderer="holoviews")
    overlay = obj.traverse(lambda x: x, [hv.Overlay])[0]
    bokeh_fig = hv.render(overlay, backend="bokeh")
    assert all(
        axis.axis_label_text_color not in _style.COLOR_CYCLE for axis in bokeh_fig.yaxis
    )


def test_three_variables_become_three_rows():
    items = [
        _item(),
        _item(SALINITY, units="1e-3"),
        _item("mass_concentration_of_chlorophyll_in_sea_water", units="mg m-3"),
    ]
    fig = render(_spec(items), renderer="matplotlib")
    assert len(_matplotlib_titles(fig)) == 3
    assert len(_holoviews_titles(render(_spec(items), renderer="holoviews"))) == 3


def test_rows_and_cols_together_are_refused():
    with pytest.raises(ValueError, match="one facet, not two"):
        render(_spec([_item()], rows="variable", cols="source"), renderer="matplotlib")


def test_faceting_on_an_unknown_field_says_what_is_allowed():
    with pytest.raises(ValueError, match="expected one of variable"):
        render(_spec([_item()], rows="platform"), renderer="matplotlib")


# -- the statistics box ----------------------------------------------------------------


def test_the_metrics_appear_off_the_title_in_both_renderers():
    import holoviews as hv

    fig = render(_spec([_item()]), renderer="matplotlib")
    static = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("bias=" in t for t in static)

    obj = render(_spec([_item()]), renderer="holoviews")
    interactive = [t.text for t in obj.traverse(lambda x: x, [hv.Text])]
    assert any("bias=" in t for t in interactive)


def test_the_box_and_the_legend_never_take_the_same_corner():
    """``loc="best"`` knows about the data and nothing about the box; it collided."""
    layout = _series.compose([_item()], metric_keys=("bias",))
    panel = layout.panels[0]
    assert panel.metrics_corner != panel.legend_corner


def test_the_emptiest_corner_is_chosen_from_the_data():
    """A rising line leaves the upper left and lower right empty."""
    rising = _item()
    rising["aligned"]["reference"].values[:] = np.linspace(0, 10, 36)
    rising["aligned"]["test"].values[:] = np.linspace(0, 10, 36)
    corner = _series.compose([rising], metric_keys=("bias",)).panels[0].metrics_corner
    assert corner in ("upper left", "lower right")


def test_too_many_comparisons_in_one_panel_drop_the_box_with_a_warning():
    items = [_item(test=f"model{i}") for i in range(4)]
    with pytest.warns(UserWarning, match="metrics CSV"):
        layout = _series.compose(items, metric_keys=("bias",))
    assert layout.panels[0].metrics_text == ""


def test_no_sample_counts_are_drawn_on_the_figure():
    """Data caveats are warnings, never annotations — the standing rule."""
    fig = render(_spec([_item()]), renderer="matplotlib")
    drawn = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert not [t for t in drawn if "n=" in t]


# -- legend placement and custom labels --------------------------------------------


def test_faceting_on_variable_drops_it_from_the_legend_and_shares_it():
    """The one thing that made a real figure never qualify for the combined key."""
    items = [_item(), _item(SALINITY, units="1e-3")]
    layout = _series.compose(items, rows="variable")
    labels = {line.label for panel in layout.panels for line in panel.lines}
    assert not any("temperature" in label or "salinity" in label for label in labels)
    assert layout.shared_legend
    fig = render(_spec(items, rows="variable"), renderer="matplotlib")
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in fig.axes)


def test_faceting_on_something_else_leaves_the_legend_alone():
    """Only ``variable`` is named anywhere else (the panel title) -- nothing else is."""
    items = [_item(test="modelA", depth=3.0), _item(test="modelB", depth=3.0)]
    unfaceted = {
        line.label
        for panel in _series.compose(items).panels
        for line in panel.lines
    }
    faceted = {
        line.label
        for panel in _series.compose(items, rows="source").panels
        for line in panel.lines
    }
    assert faceted == unfaceted  # nothing dropped, unlike the "variable" facet


def test_line_labels_overrides_the_text_in_both_renderers():
    items = [_item(depth=3.0), _item(depth=12.5)]
    layout = _series.compose(items, rows="variable")
    current = [line.label for panel in layout.panels for line in panel.lines]
    custom = [f"custom {i}" for i in range(len(current))]

    static = _matplotlib_lines(
        render(_spec(items, rows="variable", line_labels=custom), renderer="matplotlib")
    )
    interactive = _holoviews_lines(
        render(_spec(items, rows="variable", line_labels=custom), renderer="holoviews")
    )
    assert {label for label, *_ in static} == set(custom)
    assert {label for label, *_ in interactive} == set(custom)


def test_line_labels_wrong_length_lists_the_current_labels_to_copy():
    items = [_item(depth=3.0), _item(depth=12.5)]
    with pytest.raises(ValueError, match="needs one label per legend entry") as exc:
        render(
            _spec(items, rows="variable", line_labels=["only one"]),
            renderer="matplotlib",
        )
    # Every current auto label is quoted in the message, ready to copy and edit.
    layout = _series.compose(items, rows="variable")
    for panel in layout.panels:
        for line in panel.lines:
            assert repr(line.label) in str(exc.value)


def test_legend_below_combines_even_when_the_panels_disagree():
    items = [_item(test="modelA"), _item(test="modelB")]
    fig = render(_spec(items, rows="source", legend="below"), renderer="matplotlib")
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in fig.axes)


def test_legend_right_combines_outside_the_axes():
    items = [_item(test="modelA"), _item(test="modelB")]
    fig = render(_spec(items, rows="source", legend="right"), renderer="matplotlib")
    assert len(fig.legends) == 1
    assert fig.legends[0].get_bbox_to_anchor()._bbox.x0 == pytest.approx(1.0)
    assert all(ax.get_legend() is None for ax in fig.axes)


def test_legend_off_draws_nothing():
    items = [_item(), _item(SALINITY, units="1e-3")]
    fig = render(_spec(items, rows="variable", legend=False), renderer="matplotlib")
    assert not fig.legends
    assert all(ax.get_legend() is None for ax in fig.axes)


def test_legend_corner_forces_every_panel_even_when_labels_are_shared():
    """A forced corner must not fall into "auto"'s own combine-when-shared rule."""
    items = [_item(), _item(SALINITY, units="1e-3")]
    fig = render(
        _spec(items, rows="variable", legend="upper right"), renderer="matplotlib"
    )
    assert not fig.legends
    assert all(ax.get_legend() is not None for ax in fig.axes)
    assert all(ax.get_legend()._get_loc() == 1 for ax in fig.axes)  # 1 = "upper right"


def test_an_unknown_legend_placement_names_the_valid_ones():
    with pytest.raises(ValueError, match="'below', 'right', or a corner"):
        render(_spec([_item()], legend="sideways"), renderer="matplotlib")


def test_legend_below_and_right_still_move_bokehs_key_outside_the_frame():
    """Bokeh's per-panel divergence: pushed to the edge, not truly combined."""
    import holoviews as hv
    from bokeh.models import Legend

    for side in ("below", "right"):
        obj = render(_spec([_item()], legend=side), renderer="holoviews")
        overlay = obj.traverse(lambda x: x, [hv.Overlay])[0]
        bokeh_fig = hv.render(overlay, backend="bokeh")
        assert any(isinstance(r, Legend) for r in getattr(bokeh_fig, side))


# -- a single source, no comparison ------------------------------------------------


def test_a_single_source_item_draws_one_solid_line_in_both_renderers():
    item = _single_item()
    static = _matplotlib_lines(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_lines(render(_spec([item]), renderer="holoviews"))
    assert [(a, b, c) for a, b, c, _ in static] == [
        (a, b, c) for a, b, c, _ in interactive
    ]
    (label, _, dash, _), = static
    assert label == "run_new"
    assert dash == "-"


def test_linestyle_for_the_value_role_is_solid():
    assert _style.linestyle_for("value") == "-"


def test_a_single_source_item_has_no_statistics_box_in_either_renderer():
    import holoviews as hv

    fig = render(_spec([_single_item()]), renderer="matplotlib")
    static = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert not static

    obj = render(_spec([_single_item()]), renderer="holoviews")
    assert not obj.traverse(lambda x: x, [hv.Text])


def test_residual_is_refused_for_a_single_source_item_in_both_renderers():
    """There is no reference to difference against — the same refusal either way."""
    with pytest.raises(ValueError, match="needs a reference"):
        render(_spec([_single_item()], residual=True), renderer="matplotlib")
    with pytest.raises(ValueError, match="needs a reference"):
        render(_spec([_single_item()], residual=True), renderer="holoviews")


def test_faceting_a_single_source_item_by_reference_is_refused():
    items = [_single_item(), _single_item(source="run_other")]
    with pytest.raises(ValueError, match="no reference side"):
        render(_spec(items, rows="reference"), renderer="matplotlib")


def test_panel_title_reads_a_curvilinear_scalar_position():
    """A ROMS point (scalar lon_rho/lat_rho, not lon/lat) still titles its place."""
    item = _single_item(lon_name="lon_rho", lat_name="lat_rho")
    static = _matplotlib_titles(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_titles(render(_spec([item]), renderer="holoviews"))
    assert static == interactive
    assert any("144.2°W" in t and "50.0°N" in t for t in static)


def test_a_box_mean_comparison_reads_mean_over_the_region_not_a_place():
    """A box-mean lands on the box midpoint -- the same scalar-coord machinery a
    station's point sample uses (see ocean_skill.operators._horizontal_mean) -- but
    the title must say what it actually is, not claim a station that isn't there.
    """
    from ocean_skill.comparison import _region_label

    item = _item()
    region = [-149.0, 47.0, -142.0, 53.0]
    for name in ("reference", "test", "difference"):
        item["aligned"][name].attrs["region"] = region
    static = _matplotlib_titles(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_titles(render(_spec([item]), renderer="holoviews"))
    assert static == interactive
    expected = f"mean over {_region_label(region)}"
    assert any(expected in t for t in static)
    assert not any("°N " in t and "mean over" not in t for t in static)


def test_a_box_mean_single_source_item_reads_mean_over_the_region_too():
    from ocean_skill.comparison import _region_label

    item = _single_item()
    region = [-149.0, 47.0, -142.0, 53.0]
    item["aligned"]["value"].attrs["region"] = region
    static = _matplotlib_titles(render(_spec([item]), renderer="matplotlib"))
    interactive = _holoviews_titles(render(_spec([item]), renderer="holoviews"))
    assert static == interactive
    expected = f"mean over {_region_label(region)}"
    assert any(expected in t for t in static)


def test_a_list_region_attr_formats_too():
    """A zarr round trip turns the tuple region attr into a list -- must still format."""
    from ocean_skill.comparison import _region_label

    item = _item()
    region = [-149.0, 47.0, -142.0, 53.0]  # list, as a lane cache round-trip gives it
    item["aligned"]["reference"].attrs["region"] = region
    static = _matplotlib_titles(render(_spec([item]), renderer="matplotlib"))
    assert any(f"mean over {_region_label(region)}" in t for t in static)


def test_depth_fanned_single_source_items_get_markers_and_depth_labels():
    """One Field, several levels -- markers and legend labels tell them apart."""
    items = [_single_item(depth=10.0), _single_item(depth=50.0)]
    lines = _matplotlib_lines(render(_spec(items), renderer="matplotlib"))
    labels = {label for label, _, _, _ in lines}
    assert any("10" in label for label in labels)
    assert any("50" in label for label in labels)
    markers = {marker for _, _, _, marker in lines}
    assert markers != {None}, "depth varies, so the marker channel should engage"


def test_two_single_source_variables_agree_across_renderers():
    """The same composition rule single-source items get, not just comparisons'."""
    items = [_single_item(), _single_item(variable=SALINITY, units="1e-3")]
    static = _matplotlib_lines(render(_spec(items), renderer="matplotlib"))
    interactive = _holoviews_lines(render(_spec(items), renderer="holoviews"))
    assert [(a, b, c) for a, b, c, _ in static] == [
        (a, b, c) for a, b, c, _ in interactive
    ]
    fig = render(_spec(items), renderer="matplotlib")
    assert len(fig.axes) == 2  # one panel plus its twin
    assert len(_matplotlib_titles(fig)) == 1

    stacked = render(_spec(items, secondary_y=False), renderer="matplotlib")
    assert len(_matplotlib_titles(stacked)) == 2


# -- residual --------------------------------------------------------------------------


def test_residual_is_opt_in_and_adds_one_panel_in_both_renderers():
    plain = render(_spec([_item()]), renderer="matplotlib")
    with_strip = render(_spec([_item()], residual=True), renderer="matplotlib")
    assert len(with_strip.axes) == len(plain.axes) + 1
    labels = [ax.get_ylabel() for ax in with_strip.axes]
    assert any("test − reference" in label for label in labels)
    assert not any("test − reference" in ax.get_ylabel() for ax in plain.axes)

    interactive = render(_spec([_item()], residual=True), renderer="holoviews")
    assert len(interactive) == 2


# -- option plumbing -------------------------------------------------------------------


def test_a_gridded_only_option_raises_statically():
    with pytest.raises(TypeError, match="series"):
        render(
            _spec([_item()], domain=(-150.0, 45.0, -140.0, 55.0)),
            renderer="matplotlib",
        )


def test_a_nested_option_passed_too_high_is_named_not_swallowed():
    with pytest.raises(TypeError, match="legend_kwargs"):
        render(_spec([_item()], fontsize=8), renderer="matplotlib")


def test_the_interactive_renderer_warns_for_static_only_line_styling():
    with pytest.warns(UserWarning, match="only affect the static"):
        render(_spec([_item()], line_kwargs={"linewidth": 3}), renderer="holoviews")


def test_mark_reaches_both_renderers_rather_than_being_dropped():
    """``mark`` was dropped for every family; a line family honours it.

    The interactive functions take ``**_``, so a dropped option is accepted and ignored
    in silence — the failure this whole test module exists to prevent.
    """
    import holoviews as hv

    fig = render(_spec([_item()], mark="line+marker"), renderer="matplotlib")
    assert any(
        line.get_marker() not in ("", "None", None) for line in fig.axes[0].lines
    )

    obj = render(_spec([_item()], mark="line+marker"), renderer="holoviews")
    assert obj.traverse(lambda x: x, [hv.Scatter])


# -- season/spread plumbing lands, but series drawing is deliberately unchanged ---------


def test_series_specs_carry_spread_and_season_fields_but_draw_unchanged():
    """A pin for the deferred scope: ``line_specs`` already reads a scalar
    ``season`` coordinate and a ``spread`` coordinate off aligned data (the same
    readers :mod:`ocean_skill.plot.profile` uses), but neither a season facet
    nor an envelope is drawn for a series yet -- that plumbing landed once,
    for both families, ahead of the series-specific drawing work it is there
    to support later.
    """
    item = _item()
    item["aligned"]["reference"] = item["aligned"]["reference"].assign_coords(
        season="JJA"
    )
    item["aligned"]["test"] = item["aligned"]["test"].assign_coords(season="JJA")
    item["aligned"]["reference_spread"] = item["aligned"]["reference"] * 0 + 0.3
    item["aligned"]["test_spread"] = item["aligned"]["test"] * 0 + 0.4

    specs = _series.line_specs(item)
    by_role = {s.role: s for s in specs}
    assert by_role["reference"].season == "JJA"
    assert by_role["test"].season == "JJA"
    assert np.allclose(by_role["reference"].spread, 0.3)
    assert np.allclose(by_role["test"].spread, 0.4)

    # Drawing is unaffected: no band, no season-keyed colour default -- a series
    # figure looks exactly as it did before either field existed.
    fig = render(PlotSpec(family="series", items=[item]), renderer="matplotlib")
    assert len(fig.axes[0].collections) == 0
    assert len(fig.axes[0].lines) == 2


def test_series_accepts_no_kwargs_catch_all():
    """A ``**kwargs`` in the signature would silently disable option validation."""
    import inspect

    from ocean_skill.plot.matplotlib_renderer import series

    kinds = [p.kind for p in inspect.signature(series).parameters.values()]
    assert inspect.Parameter.VAR_KEYWORD not in kinds


def test_series_is_registered_everywhere_it_has_to_be():
    from ocean_skill.plot.matplotlib_renderer import _top_level_options
    from ocean_skill.plot.spec import FAMILIES

    assert "series" in FAMILIES
    for option in ("residual", "ylim", "panel_aspect", "legend", "encode"):
        assert option in _top_level_options(), option


# -- the summary families --------------------------------------------------------------


def test_taylor_and_target_accept_series_items_unchanged():
    """They read metric records, which are geometry-agnostic — asserted, not assumed."""
    items = [_item(), _item(SALINITY, units="1e-3")]
    for family in ("taylor", "target"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = render(PlotSpec(family=family, items=items), renderer="matplotlib")
        assert fig is not None
