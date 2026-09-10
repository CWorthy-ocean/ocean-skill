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

import numpy as np
import pytest
import xarray as xr

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
