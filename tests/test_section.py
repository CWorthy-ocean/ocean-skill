"""End-to-end tests for vertical sections: Field.family, _prepare wiring, plotting.

Stage A: a grid-aligned transect (``select={"transect": {"xi_rho": ...}}``) reduces a
model-only :class:`~ocean_skill.field.Field` to a (vertical, along-path) section,
drawn through the new ``section`` plot family in both renderers. See
``tests/test_transect.py`` for the pure extraction layer this builds on.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog, roms
from ocean_skill.align import ALONG_DIM
from ocean_skill.cache import key_for_prepared
from ocean_skill.field import Field
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

N = 12
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0
VAR = "sea_water_potential_temperature"


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


def _roms_run(*, with_time: bool = False) -> xr.Dataset:
    """A small standardized-shaped ROMS run: h/mask/sigma/Cs on a 5x3 rho grid."""
    ny, nx = 5, 3
    h = np.linspace(30.0, 2000.0, ny * nx).reshape(ny, nx)
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    lon_1d = np.linspace(-95.0, -93.0, nx)
    lat_1d = np.linspace(24.0, 28.0, ny)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    ds = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((ny, nx))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
        },
        coords={
            "lon": (("eta_rho", "xi_rho"), lon_2d),
            "lat": (("eta_rho", "xi_rho"), lat_2d),
        },
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    ds = roms.add_depth_coord(ds, meta)
    if with_time:
        time = pd.date_range("2012-01-01", periods=3, freq="D")
        base = 20.0 + 0.002 * ds["z_rho"]
        temp = xr.concat([base + 0.1 * i for i in range(3)], dim="time").assign_coords(
            time=time
        )
    else:
        temp = 20.0 + 0.002 * ds["z_rho"]
    ds = ds.assign({VAR: temp})
    ds[VAR].attrs["units"] = "degC"
    return ds


@pytest.fixture
def patched_read(monkeypatch):
    """Patch osk.read/catalog.resolve so a real osk.field() pipeline runs."""

    def _patch(ds: xr.Dataset, *, name: str = "roms_run"):
        monkeypatch.setattr(osk, "read", lambda n, **kw: ds if n == name else None)
        monkeypatch.setattr(
            catalog,
            "resolve",
            lambda n: SimpleNamespace(
                metadata={"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
            ),
        )
        return name

    return _patch


# -- native s-levels: the default when no depth key is given -----------------------


def test_a_transect_with_no_depth_key_leaves_native_levels_standing(patched_read):
    name = patched_read(_roms_run())
    f = osk.field(name, VAR, select={"transect": {"xi_rho": 1}}, cache=False)
    da = f.data
    assert set(da.dims) == {"s_rho", ALONG_DIM}
    assert "z_rho" in da.coords  # attached even though nothing was transformed
    assert f.family == "section"


def test_a_transect_does_not_hoist_to_the_surface_by_default(patched_read):
    """The plain-surface hoist (surface is the default depth) must not fire here."""
    name = patched_read(_roms_run())
    f = osk.field(name, VAR, select={"transect": {"xi_rho": 1}}, cache=False)
    assert f.data.sizes["s_rho"] == N  # every level survives, not just the top one


# -- fixed depths: the comparison-ready shape ---------------------------------------


def test_a_transect_with_a_depth_list_interpolates_to_fixed_z(patched_read):
    name = patched_read(_roms_run())
    f = osk.field(
        name,
        VAR,
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 500.0]},
        cache=False,
    )
    da = f.data
    assert set(da.dims) == {"z", ALONG_DIM}
    assert da.sizes["z"] == 2
    assert f.family == "section"


# -- arbitrary paths: waypoints and a fixed line, end to end ------------------------


def test_a_waypoint_transect_reaches_family_section(patched_read):
    name = patched_read(_roms_run())
    f = osk.field(
        name,
        VAR,
        select={"transect": {"waypoints": [[-95.5, 24.0], [-94.0, 27.0]]}},
        cache=False,
    )
    da = f.data
    assert set(da.dims) == {"s_rho", ALONG_DIM}
    assert f.family == "section"


def test_a_waypoint_transect_with_depths_matches_the_grid_aligned_shape(patched_read):
    """Same output contract as the grid-aligned pathway (Stage A) -- the plot layer
    (built once, against that contract) needs no changes to draw either one."""
    name = patched_read(_roms_run())
    f = osk.field(
        name,
        VAR,
        select={
            "transect": {"waypoints": [[-95.5, 24.0], [-94.0, 27.0]]},
            "depth": [50.0, 500.0],
        },
        cache=False,
    )
    da = f.data
    assert set(da.dims) == {"z", ALONG_DIM}
    assert da.sizes["z"] == 2


def test_a_fixed_lon_line_transect_reaches_family_section(patched_read):
    name = patched_read(_roms_run())
    f = osk.field(name, VAR, select={"transect": {"lon": -94.5}}, cache=False)
    assert f.family == "section"
    assert set(f.data.dims) == {"s_rho", ALONG_DIM}


def test_a_bounded_lat_line_transect_stays_inside_its_bounds(patched_read):
    name = patched_read(_roms_run())
    f = osk.field(
        name,
        VAR,
        select={"transect": {"lat": 25.0, "lon": {"min": -95.8, "max": -93.2}}},
        cache=False,
    )
    lon = np.asarray(f.data["lon"])
    assert lon.min() >= -96.0
    assert lon.max() <= -93.0


def test_a_waypoint_transect_renders_in_both_renderers():
    da = xr.DataArray(
        5.0 + np.linspace(0, 1, 6 * 8).reshape(6, 8),
        dims=("z", ALONG_DIM),
        coords={
            "z": -np.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0]),
            ALONG_DIM: np.linspace(0.0, 150.0, 8),
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, 8)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, 8)),
        },
    )
    item = {
        "field": da,
        "units": "degC",
        "standard_name": None,
        "depth": None,
        "label": "roms_run",
    }
    fig = render(PlotSpec(family="section", items=[item]), renderer="matplotlib")
    assert fig.axes
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    obj = render(PlotSpec(family="section", items=[item]), renderer="holoviews")
    assert obj is not None


# -- shape refusals: Field._require_section_shape -----------------------------------


def test_plot_refuses_a_section_with_no_vertical_axis_surviving():
    f = Field("roms_run", VAR)
    f._data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=ALONG_DIM,
        coords={
            ALONG_DIM: ("along", [0.0, 10.0, 20.0]),
            "lon": (ALONG_DIM, [-95.0, -94.0, -93.0]),
            "lat": (ALONG_DIM, [25.0, 25.0, 25.0]),
        },
    )
    assert f.family == "section"
    with pytest.raises(ValueError, match="no vertical axis surviving"):
        f.plot(renderer="matplotlib")


def test_plot_refuses_a_section_with_a_further_axis_surviving():
    f = Field("roms_run", VAR)
    f._data = xr.DataArray(
        np.zeros((2, 2, 4)),
        dims=("time", "z", ALONG_DIM),
        coords={
            "z": [-10.0, -50.0],
            ALONG_DIM: [0.0, 10.0, 20.0, 30.0],
            "lon": (ALONG_DIM, [-95.0, -94.5, -94.0, -93.5]),
            "lat": (ALONG_DIM, [25.0, 25.0, 25.0, 25.0]),
        },
    )
    assert f.family == "section"
    with pytest.raises(ValueError, match="still has"):
        f.plot(renderer="matplotlib")


def test_movie_refuses_a_section():
    f = Field("roms_run", VAR)
    f._data = xr.DataArray(
        np.zeros((2, 3)),
        dims=("z", ALONG_DIM),
        coords={
            "z": [-10.0, -50.0],
            ALONG_DIM: [0.0, 10.0, 20.0],
            "lon": (ALONG_DIM, [-95.0, -94.5, -94.0]),
            "lat": (ALONG_DIM, [25.0, 25.0, 25.0]),
        },
    )
    with pytest.raises(ValueError, match="vertical section"):
        f.movie()


# -- rendering: both renderers draw the section family without error ---------------


def _section_item():
    da = xr.DataArray(
        5.0 + np.linspace(0, 1, 6 * 8).reshape(6, 8),
        dims=("z", ALONG_DIM),
        coords={
            "z": -np.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0]),
            ALONG_DIM: np.linspace(0.0, 150.0, 8),
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, 8)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, 8)),
        },
    )
    return {
        "field": da,
        "units": "degC",
        "standard_name": None,
        "depth": None,
        "label": "roms_run",
    }


def test_section_renders_statically():
    fig = render(
        PlotSpec(family="section", items=[_section_item()]), renderer="matplotlib"
    )
    ax = fig.axes[0]
    ylim = ax.get_ylim()
    assert ylim[0] > ylim[1], "y-axis must be inverted: shallow at the top"


def test_section_renders_interactively():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    obj = render(
        PlotSpec(family="section", items=[_section_item()]), renderer="holoviews"
    )
    assert obj is not None


# -- cache-key stability -------------------------------------------------------------


def test_transect_select_keys_hash_stably_across_dict_orderings():
    a = key_for_prepared(
        source="s", variable="v", select={"transect": {"xi_rho": 1}, "depth": [5, 10]}
    )
    b = key_for_prepared(
        source="s", variable="v", select={"depth": [5, 10], "transect": {"xi_rho": 1}}
    )
    assert a == b


def test_different_transect_indices_produce_different_cache_keys():
    a = key_for_prepared(source="s", variable="v", select={"transect": {"xi_rho": 1}})
    b = key_for_prepared(source="s", variable="v", select={"transect": {"xi_rho": 2}})
    assert a != b


def test_waypoint_tuple_and_list_spellings_hash_identically():
    a = key_for_prepared(
        source="s",
        variable="v",
        select={"transect": {"waypoints": [[-95.0, 24.0], [-94.0, 25.0]]}},
    )
    b = key_for_prepared(
        source="s",
        variable="v",
        select={"transect": {"waypoints": ((-95.0, 24.0), (-94.0, 25.0))}},
    )
    assert a == b


def test_different_waypoints_spacing_and_method_each_produce_a_different_key():
    base = key_for_prepared(
        source="s",
        variable="v",
        select={"transect": {"waypoints": [[-95.0, 24.0], [-94.0, 25.0]]}},
    )
    different_points = key_for_prepared(
        source="s",
        variable="v",
        select={"transect": {"waypoints": [[-95.0, 24.0], [-93.0, 25.0]]}},
    )
    different_spacing = key_for_prepared(
        source="s",
        variable="v",
        select={
            "transect": {
                "waypoints": [[-95.0, 24.0], [-94.0, 25.0]],
                "spacing_km": 5.0,
            }
        },
    )
    different_method = key_for_prepared(
        source="s",
        variable="v",
        select={
            "transect": {
                "waypoints": [[-95.0, 24.0], [-94.0, 25.0]],
                "method": "bilinear",
            }
        },
    )
    assert len({base, different_points, different_spacing, different_method}) == 4


# -- section_row: the comparison counterpart of section, both renderers -------------
#
# Stage C: a comparison whose select cuts a transect (Comparison.is_section) draws as
# test | reference | difference sections through this family instead of section --
# structurally field_row with prepare_section's geometry substituted for the map.


def _section_row_item(*, labels=("roms_run", "woa23"), offset: float = 3.0):
    """One section_row spec item: a fixed-depth (z, along) trio, aligned by construction."""
    test = xr.DataArray(
        5.0 + np.linspace(0, 1, 6 * 8).reshape(6, 8),
        dims=("z", ALONG_DIM),
        coords={
            "z": -np.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0]),
            ALONG_DIM: np.linspace(0.0, 150.0, 8),
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, 8)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, 8)),
        },
    )
    reference = test + offset
    return {
        "aligned": {"test": test, "reference": reference, "difference": test - reference},
        "units": "degC",
        "standard_name": None,
        "depth": "0-200 m",
        "time": "2012-01",
        "metrics": {"bias": 0.125, "rmse": 0.5, "corr": 0.98},
        "labels": labels,
    }


def test_section_row_panels_are_all_inverted_with_positive_depth():
    fig = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="matplotlib",
    )
    for ax in fig.axes[:3]:
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1], "y-axis must be inverted: shallow at the top"
        assert ylim[0] >= 0, "depth reads positive-down"


def test_section_row_panels_are_grey_below_the_data():
    fig = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="matplotlib",
    )
    for ax in fig.axes[:3]:
        assert ax.get_facecolor() == (0.85, 0.85, 0.85, 1.0)


def test_section_row_test_and_reference_share_one_colour_scale():
    from matplotlib.collections import QuadMesh

    fig = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="matplotlib",
    )
    meshes = [
        next(c for c in ax.collections if isinstance(c, QuadMesh)) for ax in fig.axes[:3]
    ]
    test_norm, reference_norm, diff_norm = (m.norm for m in meshes)
    assert (test_norm.vmin, test_norm.vmax) == (reference_norm.vmin, reference_norm.vmax)
    assert diff_norm.vmin == pytest.approx(-diff_norm.vmax)
    assert diff_norm.vmin != test_norm.vmin


def test_section_row_metrics_land_in_the_difference_panels_corner_box():
    fig = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="matplotlib",
    )
    boxed = [ax for ax in fig.axes[:3] if getattr(ax, "_osk_metrics_text", None)]
    assert len(boxed) == 1
    text = boxed[0]._osk_metrics_text.get_text()
    assert "bias=0.125" in text
    assert "rmse=0.5" in text


def test_section_row_labels_become_panel_titles():
    # As real integration does (Comparison.plot()): `labels` reaches the spec through
    # options, not the item -- the item's own "labels" key is for a grid's per-row
    # fallback, which section_row (never stacked into one) has no caller for.
    fig = render(
        PlotSpec(
            family="section_row",
            items=[_section_row_item()],
            options={"labels": ("roms_run", "woa23")},
        ),
        renderer="matplotlib",
    )
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert "roms_run" in titles
    assert "woa23" in titles
    assert "difference" in titles


def test_section_row_suptitle_carries_the_path_note():
    fig = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="matplotlib",
    )
    assert fig._suptitle is not None
    text = fig._suptitle.get_text()
    assert "→" in text  # the path's own endpoints, from SectionGeometry.path_note


def test_section_row_depth_ylabel_only_on_the_first_panel():
    fig = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="matplotlib",
    )
    axes = fig.axes[:3]
    assert axes[0].get_ylabel() != ""
    assert axes[1].get_ylabel() == ""
    assert axes[2].get_ylabel() == ""


def test_section_row_renders_interactively():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    obj = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="holoviews",
    )
    qms = obj.traverse(lambda x: x, [hv.QuadMesh])
    assert len(qms) == 3


def test_section_row_panels_have_grey_background_and_inverted_depth():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    obj = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="holoviews",
    )
    for qm in obj.traverse(lambda x: x, [hv.QuadMesh]):
        plot_kwargs = qm.opts.get("plot").kwargs
        assert plot_kwargs.get("bgcolor") == "#d9d9d9"
        assert plot_kwargs.get("invert_yaxis") is True


def test_section_row_metrics_fold_into_the_difference_title():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    obj = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="holoviews",
    )
    titles = [
        qm.opts.get("plot").kwargs.get("title")
        for qm in obj.traverse(lambda x: x, [hv.QuadMesh])
    ]
    assert any("bias=0.125" in (t or "") for t in titles)


def test_section_row_shares_the_static_colour_limits_interactively():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    obj = render(
        PlotSpec(family="section_row", items=[_section_row_item()]),
        renderer="holoviews",
    )
    # hvplot's clim= sets the value dimension's own range rather than a plot-level
    # option, so it reads back off vdims[0].range, not .opts.get("plot").
    clims = [
        qm.vdims[0].range for qm in obj.traverse(lambda x: x, [hv.QuadMesh])
    ]
    test_clim, reference_clim, diff_clim = clims
    assert test_clim == reference_clim
    assert diff_clim[0] == pytest.approx(-diff_clim[1])


def test_section_row_refuses_a_native_s_trio():
    native = xr.DataArray(
        np.zeros((3, 4)),
        dims=("s_rho", ALONG_DIM),
        coords={
            ALONG_DIM: np.linspace(0.0, 30.0, 4),
            "lon": (ALONG_DIM, np.linspace(-95.0, -94.0, 4)),
            "lat": (ALONG_DIM, np.linspace(24.0, 25.0, 4)),
        },
    )
    aligned = {"test": native, "reference": native.copy(), "difference": native.copy()}
    item = {**_section_row_item(), "aligned": aligned}
    with pytest.raises(ValueError, match="fixed-depth"):
        render(PlotSpec(family="section_row", items=[item]), renderer="matplotlib")


def test_section_row_refuses_a_positive_down_z():
    field = xr.DataArray(
        5.0 + np.linspace(0, 1, 3 * 4).reshape(3, 4),
        dims=("z", ALONG_DIM),
        coords={
            "z": np.array([0.0, 50.0, 200.0]),  # positive-down: the bug this guards
            ALONG_DIM: np.linspace(0.0, 30.0, 4),
            "lon": (ALONG_DIM, np.linspace(-95.0, -94.0, 4)),
            "lat": (ALONG_DIM, np.linspace(24.0, 25.0, 4)),
        },
    )
    aligned = {"test": field, "reference": field.copy(), "difference": field.copy()}
    item = {**_section_row_item(), "aligned": aligned}
    with pytest.raises(ValueError, match="negative-down"):
        render(PlotSpec(family="section_row", items=[item]), renderer="matplotlib")


def test_domain_is_not_an_option_of_section_row():
    with pytest.raises(TypeError, match="not an option of section_row"):
        render(
            PlotSpec(
                family="section_row",
                items=[_section_row_item()],
                options={"domain": (0.0, 0.0, 1.0, 1.0)},
            ),
            renderer="matplotlib",
        )
