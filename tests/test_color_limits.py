"""Field colorbars default to the true data range; ``robust=`` opts into clipping.

The bug this guards against: ``_limits`` used to *always* clip to the 10th-90th
percentile, so a map's colourbar could saturate well below the field's real maximum
while ``Field.extremum("max")``/``.series()`` (which read the raw values) reported the
true, higher number at the very same cell -- an apparent discrepancy that was really
just the colourbar lying about its own top. ``_limits`` now defaults to the full finite
range, with ``robust=True`` (or a float central fraction) as the opt-in for the old
clipped behaviour -- and every family that draws a variable's own colour scale accepts
and forwards that same ``robust`` keyword, in both renderers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.align import ALONG_DIM
from ocean_skill.plot.matplotlib_renderer import _limits
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

NITRATE = "mole_concentration_of_nitrate_in_sea_water"


# --- _limits itself: the one function every family's colour scale goes through --------


def test_default_is_the_full_finite_range():
    vals = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    assert _limits(vals) == (1.0, 100.0)


def test_robust_true_clips_to_the_10th_90th_percentile():
    vals = np.linspace(0.0, 100.0, 101)  # 0, 1, .., 100
    lo, hi = _limits(vals, robust=True)
    assert lo == pytest.approx(10.0)
    assert hi == pytest.approx(90.0)


def test_robust_point_eight_is_the_same_central_fraction_as_true():
    vals = np.linspace(0.0, 100.0, 101)
    assert _limits(vals, robust=0.8) == _limits(vals, robust=True)


def test_a_narrower_robust_fraction_clips_more():
    vals = np.linspace(0.0, 100.0, 101)
    lo_wide, hi_wide = _limits(vals, robust=0.8)
    lo_narrow, hi_narrow = _limits(vals, robust=0.5)
    assert lo_narrow > lo_wide
    assert hi_narrow < hi_wide


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_an_out_of_range_robust_fraction_is_refused(bad):
    with pytest.raises(ValueError, match="robust"):
        _limits(np.linspace(0.0, 10.0, 5), robust=bad)


def test_robust_false_and_none_both_mean_the_plain_range():
    vals = np.array([2.0, 4.0, 6.0])
    assert _limits(vals, robust=False) == _limits(vals)
    assert _limits(vals, robust=None) == _limits(vals)


def test_all_nan_or_empty_still_falls_back_to_zero_one():
    assert _limits(np.full(5, np.nan)) == (0.0, 1.0)
    assert _limits(np.array([])) == (0.0, 1.0)


def test_nan_values_are_dropped_not_counted():
    vals = np.array([1.0, 2.0, 3.0, np.nan])
    assert _limits(vals) == (1.0, 3.0)


def test_several_arrays_are_pooled_before_taking_the_range():
    a = np.array([1.0, 2.0])
    b = np.array([0.0, 100.0])
    assert _limits(a, b) == (0.0, 100.0)


# --- vmin/vmax: exact pinned limits, overriding both the plain range and robust -------


def test_vmin_pins_the_low_end_only():
    vals = np.array([2.0, 4.0, 6.0])
    assert _limits(vals, vmin=-10.0) == (-10.0, 6.0)


def test_vmax_pins_the_high_end_only():
    vals = np.array([2.0, 4.0, 6.0])
    assert _limits(vals, vmax=100.0) == (2.0, 100.0)


def test_vmin_and_vmax_together_ignore_the_data_entirely():
    vals = np.array([2.0, 4.0, 6.0])
    assert _limits(vals, vmin=0.0, vmax=1.0) == (0.0, 1.0)


def test_vmin_vmax_override_robust_only_for_the_end_they_pin():
    vals = np.linspace(0.0, 100.0, 101)
    lo, hi = _limits(vals, robust=True, vmin=-5.0)
    assert lo == -5.0
    assert hi == pytest.approx(90.0)  # the unpinned end still comes from robust


def test_vmin_applies_even_when_the_data_is_empty():
    assert _limits(np.array([]), vmin=2.0, vmax=3.0) == (2.0, 3.0)


def test_vmin_not_less_than_vmax_is_refused():
    with pytest.raises(ValueError, match="vmin"):
        _limits(np.array([1.0, 2.0]), vmin=5.0, vmax=1.0)


# --- norm_for: user vmin/vmax outrank a variable's own declared display range ---------

CHLOROPHYLL = "mass_concentration_of_chlorophyll_a_in_sea_water"


def test_norm_for_defaults_to_a_variables_declared_range():
    from ocean_skill.colormaps import norm_for

    norm = norm_for(CHLOROPHYLL, 1.0, 2.0)
    assert norm.vmin == pytest.approx(0.01)
    assert norm.vmax == pytest.approx(10.0)


def test_norm_for_user_limits_override_the_declared_range_but_keep_the_log_scale():
    import matplotlib.colors as mcolors

    from ocean_skill.colormaps import norm_for

    norm = norm_for(CHLOROPHYLL, 1.0, 2.0, user_vmin=0.5, user_vmax=50.0)
    assert isinstance(norm, mcolors.LogNorm)
    assert norm.vmin == pytest.approx(0.5)
    assert norm.vmax == pytest.approx(50.0)


# --- field_facet (matplotlib): the single-map path the bug report used ----------------


def _outlier_map(
    n_lat: int = 6, n_lon: int = 8, base: float = 5.0, outlier: float = 100.0
):
    """A field that is one value almost everywhere and much higher at one cell.

    Mirrors the reported case: a field whose 90th percentile sits far below its true
    maximum, so a percentile-clipped colourbar would saturate well short of it.
    """
    lat = np.linspace(20.0, 30.0, n_lat)
    lon = np.linspace(-100.0, -90.0, n_lon)
    vals = np.full((n_lat, n_lon), base)
    vals[0, 0] = outlier
    return xr.DataArray(
        vals,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "mmol m-3"},
    )


def _facet_item(field):
    return {
        "field": field,
        "facet_dim": None,
        "row_dim": None,
        "units": "mmol m-3",
        "standard_name": NITRATE,
    }


def test_field_facet_colourbar_defaults_to_the_true_maximum():
    field = _outlier_map()
    fig = render(PlotSpec(family="field_facet", items=[_facet_item(field)]))
    from matplotlib.collections import QuadMesh

    mesh = next(c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh))
    assert mesh.norm.vmax == pytest.approx(100.0)


def test_field_facet_robust_true_clips_the_outlier_below_its_true_value():
    field = _outlier_map()
    fig = render(
        PlotSpec(family="field_facet", items=[_facet_item(field)]), robust=True
    )
    from matplotlib.collections import QuadMesh

    mesh = next(c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh))
    assert mesh.norm.vmax < 100.0


def test_field_facet_vmin_vmax_pin_the_colourbar_exactly():
    field = _outlier_map()
    fig = render(
        PlotSpec(family="field_facet", items=[_facet_item(field)]),
        vmin=0.0,
        vmax=20.0,
    )
    from matplotlib.collections import QuadMesh

    mesh = next(c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh))
    assert mesh.norm.vmin == pytest.approx(0.0)
    assert mesh.norm.vmax == pytest.approx(20.0)


def test_field_facet_vmax_combines_with_robust_filling_the_unpinned_end():
    field = _outlier_map()
    fig = render(
        PlotSpec(family="field_facet", items=[_facet_item(field)]),
        robust=True,
        vmax=20.0,
    )
    from matplotlib.collections import QuadMesh

    mesh = next(c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh))
    assert mesh.norm.vmax == pytest.approx(20.0)
    assert mesh.norm.vmin == pytest.approx(5.0)  # robust's own 10th percentile here


def test_field_facet_holoviews_vmin_vmax_pin_the_colourbar_exactly():
    import holoviews as hv

    hv.extension("bokeh")
    field = _outlier_map()
    obj = render(
        PlotSpec(family="field_facet", items=[_facet_item(field)]),
        renderer="holoviews",
        vmin=0.0,
        vmax=20.0,
    )
    mesh = next(iter(obj.traverse(lambda x: x, [hv.QuadMesh])))
    lo, hi = mesh.range(mesh.vdims[0].name)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(20.0)


# --- field_map_grid (matplotlib): the mapview path in the bug report ------------------


def test_field_map_grid_each_panel_defaults_to_its_own_true_maximum():
    from matplotlib.collections import QuadMesh

    normal = _outlier_map(outlier=100.0)
    quiet = _outlier_map(outlier=6.0)  # no real outlier in this one
    items = [
        {"field": normal, "units": "mmol m-3", "standard_name": NITRATE},
        {"field": quiet, "units": "mmol m-3", "standard_name": NITRATE},
    ]
    fig = render(PlotSpec(family="field_map_grid", items=items))
    meshes = [
        next(c for c in ax.collections if isinstance(c, QuadMesh))
        for ax in fig.axes[:2]
    ]
    assert meshes[0].norm.vmax == pytest.approx(100.0)
    assert meshes[1].norm.vmax == pytest.approx(6.0)


def test_field_map_grid_robust_clips_the_outlier_panel_only():
    from matplotlib.collections import QuadMesh

    normal = _outlier_map(outlier=100.0)
    quiet = _outlier_map(outlier=6.0)
    items = [
        {"field": normal, "units": "mmol m-3", "standard_name": NITRATE},
        {"field": quiet, "units": "mmol m-3", "standard_name": NITRATE},
    ]
    fig = render(PlotSpec(family="field_map_grid", items=items), robust=True)
    meshes = [
        next(c for c in ax.collections if isinstance(c, QuadMesh))
        for ax in fig.axes[:2]
    ]
    assert meshes[0].norm.vmax < 100.0


# --- shared_limits="variable"/"source": per-group pooling, not all-or-nothing ---------

SILICATE = "mole_concentration_of_silicate_in_sea_water"


def _grouped_map_items():
    """2 variables x 2 sources; one nitrate panel carries the outlier -- pooling
    by ``"variable"`` should lift *both* nitrate panels to it and leave the
    silicate pair untouched.
    """
    return [
        {
            "field": _outlier_map(outlier=100.0),
            "units": "mmol m-3",
            "standard_name": NITRATE,
            "label": "run_a",
        },
        {
            "field": _outlier_map(outlier=6.0),
            "units": "mmol m-3",
            "standard_name": NITRATE,
            "label": "run_b",
        },
        {
            "field": _outlier_map(outlier=8.0),
            "units": "mmol m-3",
            "standard_name": SILICATE,
            "label": "run_a",
        },
        {
            "field": _outlier_map(outlier=9.0),
            "units": "mmol m-3",
            "standard_name": SILICATE,
            "label": "run_b",
        },
    ]


def test_field_map_grid_shared_limits_variable_pools_each_column_independently():
    from matplotlib.collections import QuadMesh

    items = _grouped_map_items()
    fig = render(
        PlotSpec(
            family="field_map_grid",
            items=items,
            options={"cols": "variable", "shared_limits": "variable"},
        )
    )
    meshes = [
        next(c for c in ax.collections if isinstance(c, QuadMesh))
        for ax in fig.axes[:4]
    ]
    # cell (r, c) at index r*2+c: nitrate (col 0) pools onto the outlier's
    # 100.0; silicate (col 1) pools onto 9.0, independent of nitrate's scale.
    assert meshes[0].norm.vmax == meshes[2].norm.vmax == pytest.approx(100.0)
    assert meshes[1].norm.vmax == meshes[3].norm.vmax == pytest.approx(9.0)
    assert meshes[0].norm.vmax != meshes[1].norm.vmax


def test_field_map_grid_shared_limits_variable_matches_in_both_renderers():
    import holoviews as hv

    hv.extension("bokeh")
    items = _grouped_map_items()
    obj = render(
        PlotSpec(
            family="field_map_grid",
            items=items,
            options={"cols": "variable", "shared_limits": "variable"},
        ),
        renderer="holoviews",
    )
    meshes = list(obj.traverse(lambda x: x, [hv.QuadMesh]))
    highs = [mesh.range(mesh.vdims[0].name)[1] for mesh in meshes]
    assert highs[0] == highs[2] == pytest.approx(100.0)
    assert highs[1] == highs[3] == pytest.approx(9.0)


def test_field_map_grid_shared_limits_true_still_warns_the_old_way():
    items = _grouped_map_items()
    with pytest.warns(UserWarning, match="shared_limits=True but panels use"):
        render(
            PlotSpec(family="field_map_grid", items=items, options={"shared_limits": True}),
        )


def test_field_map_grid_shared_limits_variable_never_warns():
    items = _grouped_map_items()
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        render(
            PlotSpec(
                family="field_map_grid",
                items=items,
                options={"shared_limits": "variable"},
            )
        )


def test_field_map_grid_shared_limits_source_warns_naming_the_mixed_label():
    items = _grouped_map_items()  # run_a mixes nitrate + silicate
    with pytest.warns(UserWarning, match="'run_a' group mixes variables"):
        render(
            PlotSpec(
                family="field_map_grid",
                items=items,
                options={"shared_limits": "source"},
            )
        )


def test_field_map_grid_shared_limits_unknown_string_is_refused():
    items = _grouped_map_items()
    with pytest.raises(ValueError, match='"variable".*"source"'):
        render(
            PlotSpec(
                family="field_map_grid", items=items, options={"shared_limits": "col"}
            )
        )


def test_field_map_grid_shared_limits_robust_still_clips_within_the_group():
    from matplotlib.collections import QuadMesh

    items = _grouped_map_items()
    fig = render(
        PlotSpec(
            family="field_map_grid",
            items=items,
            options={"cols": "variable", "shared_limits": "variable", "robust": True},
        )
    )
    meshes = [
        next(c for c in ax.collections if isinstance(c, QuadMesh))
        for ax in fig.axes[:4]
    ]
    assert meshes[0].norm.vmax < 100.0


def test_time_depth_grid_shared_limits_variable_pools_each_variable():
    from matplotlib.collections import QuadMesh

    def item(source, variable, outlier):
        time = pd.date_range("2020-01-01", periods=6, freq="MS")
        depth = np.array([5.0, 10.0])
        vals = np.full((depth.size, time.size), 5.0)
        vals[0, 0] = outlier
        field = xr.DataArray(
            vals,
            dims=("depth", "time"),
            coords={"depth": depth, "time": time},
            attrs={"units": "mmol m-3"},
        ).assign_coords(lon=-144.245, lat=49.978)
        return {
            "field": field,
            "units": "mmol m-3",
            "standard_name": variable,
            "label": source,
        }

    items = [
        item("station_a", NITRATE, 100.0),
        item("station_b", NITRATE, 6.0),
        item("station_a", SILICATE, 8.0),
        item("station_b", SILICATE, 9.0),
    ]
    fig = render(
        PlotSpec(
            family="time_depth",
            items=items,
            options={"cols": "variable", "shared_limits": "variable"},
        )
    )
    meshes = [
        next(c for c in ax.collections if isinstance(c, QuadMesh))
        for ax in fig.axes[:4]
    ]
    assert meshes[0].norm.vmax == meshes[2].norm.vmax == pytest.approx(100.0)
    assert meshes[1].norm.vmax == meshes[3].norm.vmax == pytest.approx(9.0)


# --- the same two families, interactively -- the two renderers must not disagree ------


def test_field_facet_holoviews_colourbar_also_defaults_to_the_true_maximum():
    import holoviews as hv

    hv.extension("bokeh")
    field = _outlier_map()
    obj = render(
        PlotSpec(family="field_facet", items=[_facet_item(field)]),
        renderer="holoviews",
    )
    mesh = next(iter(obj.traverse(lambda x: x, [hv.QuadMesh])))
    _lo, hi = mesh.range(mesh.vdims[0].name)
    assert hi == pytest.approx(100.0)


def test_field_facet_holoviews_robust_clips_the_outlier_too():
    import holoviews as hv

    hv.extension("bokeh")
    field = _outlier_map()
    obj = render(
        PlotSpec(family="field_facet", items=[_facet_item(field)]),
        renderer="holoviews",
        robust=True,
    )
    mesh = next(iter(obj.traverse(lambda x: x, [hv.QuadMesh])))
    _lo, hi = mesh.range(mesh.vdims[0].name)
    assert hi < 100.0


# --- section/cross/time_depth: vmin/vmax on the other single-field colorbar families --


def _outlier_section(base: float = 5.0, outlier: float = 100.0):
    n_along, n_z = 8, 6
    values = np.full((n_z, n_along), base)
    values[0, 0] = outlier
    return xr.DataArray(
        values,
        dims=("z", ALONG_DIM),
        coords={
            "z": -np.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0]),
            ALONG_DIM: np.linspace(0.0, 150.0, n_along),
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, n_along)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, n_along)),
        },
    )


def _section_item(field=None, **overrides):
    return {
        "field": field if field is not None else _outlier_section(),
        "units": "mmol m-3",
        "standard_name": None,
        "depth": None,
        "label": "roms_run",
        **overrides,
    }


def test_section_vmin_vmax_pin_the_colourbar_exactly():
    from matplotlib.collections import QuadMesh

    fig = render(
        PlotSpec(family="section", items=[_section_item()]), vmin=0.0, vmax=20.0
    )
    mesh = next(c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh))
    assert mesh.norm.vmin == pytest.approx(0.0)
    assert mesh.norm.vmax == pytest.approx(20.0)


def test_section_holoviews_vmin_vmax_pin_the_colourbar_exactly():
    import holoviews as hv

    hv.extension("bokeh")
    obj = render(
        PlotSpec(family="section", items=[_section_item()]),
        renderer="holoviews",
        vmin=0.0,
        vmax=20.0,
    )
    mesh = next(iter(obj.traverse(lambda x: x, [hv.QuadMesh])))
    lo, hi = mesh.range(mesh.vdims[0].name)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(20.0)


def test_cross_vmin_vmax_pin_the_shared_colourbar():
    from matplotlib.collections import QuadMesh

    items = [
        _section_item(label="lon fixed"),
        _section_item(label="lat fixed"),
    ]
    fig = render(PlotSpec(family="cross", items=items), vmin=0.0, vmax=20.0)
    meshes = [
        next(c for c in ax.collections if isinstance(c, QuadMesh))
        for ax in fig.axes[:2]
    ]
    for mesh in meshes:
        assert mesh.norm.vmin == pytest.approx(0.0)
        assert mesh.norm.vmax == pytest.approx(20.0)


def _outlier_time_depth(base: float = 5.0, outlier: float = 100.0):
    time = pd.date_range("2020-01-01", periods=6, freq="MS")
    depth = np.array([0.0, 10.0, 25.0])
    vals = np.full((time.size, depth.size), base)
    vals[0, 0] = outlier
    return xr.DataArray(
        vals,
        dims=("time", "depth"),
        coords={"time": time, "depth": depth, "lon": -144.0, "lat": 50.0},
        attrs={"units": "mmol m-3"},
    )


def _time_depth_item(field=None, **overrides):
    return {
        "field": field if field is not None else _outlier_time_depth(),
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "label": "GOM_bgc",
        **overrides,
    }


def test_time_depth_vmin_vmax_pin_the_colourbar_exactly():
    from matplotlib.collections import QuadMesh

    fig = render(
        PlotSpec(family="time_depth", items=[_time_depth_item()]), vmin=0.0, vmax=20.0
    )
    mesh = next(c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh))
    assert mesh.norm.vmin == pytest.approx(0.0)
    assert mesh.norm.vmax == pytest.approx(20.0)


def test_time_depth_holoviews_vmin_vmax_pin_the_colourbar_exactly():
    import holoviews as hv

    hv.extension("bokeh")
    obj = render(
        PlotSpec(family="time_depth", items=[_time_depth_item()]),
        renderer="holoviews",
        vmin=0.0,
        vmax=20.0,
    )
    mesh = next(iter(obj.traverse(lambda x: x, [hv.QuadMesh])))
    lo, hi = mesh.range(mesh.vdims[0].name)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(20.0)


def test_time_depth_grid_vmin_vmax_pin_every_panel_even_without_shared_limits():
    from matplotlib.collections import QuadMesh

    items = [
        _time_depth_item(_outlier_time_depth(outlier=100.0), label="station_a"),
        _time_depth_item(_outlier_time_depth(outlier=6.0), label="station_b"),
    ]
    fig = render(PlotSpec(family="time_depth", items=items), vmin=0.0, vmax=20.0)
    meshes = [
        next(c for c in ax.collections if isinstance(c, QuadMesh))
        for ax in fig.axes[:2]
    ]
    for mesh in meshes:
        assert mesh.norm.vmin == pytest.approx(0.0)
        assert mesh.norm.vmax == pytest.approx(20.0)
