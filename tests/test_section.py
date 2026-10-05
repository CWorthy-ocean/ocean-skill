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
    (built once, against that contract) needs no changes to draw either one.
    """
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


# -- native-s land columns: prepare_section must never see a NaN depth coord -------
#
# roms.standardize masks the free-surface zeta over land, and z_rho/z_w are built
# from zeta -- so *without* correction a native-s section's depth coordinate would
# be NaN over every land column, unlike the all-wet synthetic grid every other test
# in this file uses. pcolormesh tolerates NaN in the *data* it colours (the
# below-bathymetry grey) but raises on NaN in its x/y coordinate arrays -- this
# used to reach real ROMS output (an `esper` run) as a bare ValueError.
#
# The fix lives upstream of prepare_section: ocean_skill.comparison._prepare builds
# a section's z_rho/z_w with roms.add_depth_coord(..., zero_zeta=True), which is
# finite everywhere -- land included -- because it is built from h (bathymetry,
# never masked) alone, dropping zeta rather than trying to fill around its absence.
# prepare_section itself no longer patches a NaN depth (see git history for the old
# mean-profile fill this replaced); confirmed here so a fill doesn't quietly
# reappear as a band-aid for some future path that reintroduces the NaN.


def _native_s_item_with_land():
    """Build a (z_rho, along) section item whose westernmost column is land.

    ``z_rho`` is NaN there, mirroring what ``roms.standardize``'s masked-zeta chain
    would produce *without* the zero_zeta correction -- the shape prepare_section
    must no longer be asked to fix.
    """
    n_s, n_along = 6, 5
    sigma = np.linspace(-0.95, -0.05, n_s)
    h = np.linspace(30.0, 500.0, n_along)
    z_rho = np.outer(sigma, h)  # (s, along), negative-down, deepest at sigma[0]
    z_rho[:, 0] = np.nan  # one land column
    da = xr.DataArray(
        5.0 + np.linspace(0, 1, n_s * n_along).reshape(n_s, n_along),
        dims=("s_rho", ALONG_DIM),
        coords={
            "z_rho": (("s_rho", ALONG_DIM), z_rho),
            ALONG_DIM: np.linspace(0.0, 100.0, n_along),
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, n_along)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, n_along)),
        },
    )
    da[ALONG_DIM].attrs["units"] = "km"
    da.values[:, 0] = np.nan  # the land column carries no data either
    return da


def test_prepare_section_no_longer_fills_a_nan_depth_column():
    """prepare_section trusts its input's depth mesh -- filling moved upstream.

    In practice prepare_section never receives this shape any more (see the
    module-level comment above), but confirming it propagates a NaN depth rather
    than silently patching one guards against the mean-profile fill (and the
    bathymetry "dips" it drew into land boundaries) quietly coming back.
    """
    from ocean_skill.plot.section import prepare_section

    da = _native_s_item_with_land()
    assert bool(np.isnan(da["z_rho"]).any())  # the land column is genuinely NaN
    values, geometry = prepare_section(da)
    assert geometry.native_s
    assert bool(np.isnan(np.asarray(values["depth"])).any())
    # the data is NaN there too, same as before -- only the fill is gone
    assert bool(np.isnan(np.asarray(values)).any())


def test_prepare_section_all_wet_section_is_finite():
    """An all-wet grid, or a fixed-depth section's 1-D z, draw with a finite mesh."""
    from ocean_skill.plot.section import prepare_section

    values, _ = prepare_section(_section_item()["field"])  # fixed-depth "z", all-wet
    assert bool(np.isfinite(np.asarray(values["depth"])).all())


def _roms_run_with_land() -> xr.Dataset:
    """Build a run like :func:`_roms_run`, but with land and a real free surface.

    Reproduces ``roms.standardize``'s own masked-zeta chain (zeta masked over land
    *before* ``add_depth_coord`` builds z_rho from it), rather than the zero-zeta,
    all-wet shape every other fixture in this file uses.
    """
    ny, nx = 5, 4
    h = np.linspace(30.0, 2000.0, ny * nx).reshape(ny, nx)
    mask = np.ones((ny, nx))
    mask[:, 0] = 0.0  # one land column along xi
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    lon_1d = np.linspace(-95.0, -93.0, nx)
    lat_1d = np.linspace(24.0, 28.0, ny)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    ds = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), mask),
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
    zeta = xr.zeros_like(ds["h"]).where(ds["mask_rho"] == 1)  # masked before z_rho
    ds = ds.assign(zeta=zeta)
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    ds = roms.add_depth_coord(ds, meta)
    temp = (20.0 + 0.002 * ds["z_rho"]).where(ds["mask_rho"] == 1)
    ds = ds.assign({VAR: temp})
    ds[VAR].attrs["units"] = "degC"
    return ds


def test_native_s_section_depth_over_land_matches_zero_zeta_formula(patched_read):
    """The land-column depth mesh is exactly the h-only (zeta=0) formula.

    Regression for the bathymetry "dips": before, a land column's depth
    coordinate was filled with the transect's *mean* depth profile -- a guess
    that could be far deeper than the real seafloor there, dragging a wet
    neighbor's meshed bottom edge down into a downward wedge. Now every column
    -- wet or land -- reads its own h through the same zero-zeta formula
    add_depth_coord already used when a run carries no free surface at all, so a
    shallow column next to land never meshes against a placeholder deeper than
    its own bathymetry.
    """
    ds = _roms_run_with_land()
    h_row = np.asarray(ds["h"])[2, :]  # eta_rho=2, the row the transect below cuts
    sigma_r = np.asarray(ds["sigma_r"])
    cs_r = np.asarray(ds["Cs_r"])

    name = patched_read(ds)
    f = osk.field(name, VAR, select={"transect": {"eta_rho": 2}}, cache=False)
    z_rho = np.asarray(f.data["z_rho"])  # (s_rho, along)
    assert bool(np.isfinite(z_rho).all())

    expected = (
        h_row[None, :]
        * (HC * sigma_r[:, None] + h_row[None, :] * cs_r[:, None])
        / (HC + h_row[None, :])
    )
    np.testing.assert_allclose(z_rho, expected)

    # no dip: every column's own depth stays within its own bathymetry -- the
    # land column (xi=0) is no deeper than its own h, so a wet neighbour's mesh
    # edge against it can never be pulled below the neighbour's real seafloor.
    depth = -z_rho
    assert bool((depth <= h_row[None, :] + 1e-6).all())


def test_a_native_s_transect_through_land_renders_statically(patched_read):
    """Pin the exact real-world failure, and its fix.

    A native-s section crossing land used to reach ``ax.pcolormesh`` with a NaN
    depth coordinate and raise; ``_prepare`` then patched it with a transect-mean
    profile, which drew bathymetry "dips" into the land boundary (a deep guessed
    depth meshed against a shallow real one). ``_roms_run_with_land`` reproduces
    the masked-zeta chain that *would* leave ``z_rho`` NaN over land, but
    ``osk.field`` (via ``comparison._prepare``'s ``zero_zeta=True`` section
    handling) rebuilds it from ``h`` alone first, so it is finite here already --
    the render below no longer needs to (and does not) fill anything.
    """
    name = patched_read(_roms_run_with_land())
    f = osk.field(name, VAR, select={"transect": {"eta_rho": 2}}, cache=False)
    assert bool(np.isfinite(np.asarray(f.data["z_rho"])).all())
    fig = f.plot()
    assert fig.axes


def test_a_native_s_transect_through_land_renders_interactively(patched_read):
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    name = patched_read(_roms_run_with_land())
    f = osk.field(name, VAR, select={"transect": {"eta_rho": 2}}, cache=False)
    obj = f.plot(renderer="holoviews")
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


def test_section_row_titles_overrides_by_hand_test_reference_difference():
    override = [None, "My Reference", "My Diff"]
    static = render(
        PlotSpec(
            family="section_row",
            items=[_section_row_item()],
            options={"labels": ("roms_run", "woa23"), "titles": override},
        ),
        renderer="matplotlib",
    )
    titles = [ax.get_title() for ax in static.axes if ax.get_title()]
    assert titles == ["roms_run", "My Reference", "My Diff"]

    with pytest.raises(ValueError, match="needs one entry per panel"):
        render(
            PlotSpec(
                family="section_row",
                items=[_section_row_item()],
                options={"titles": ["only one"]},
            ),
            renderer="matplotlib",
        )


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


def test_section_row_titles_overrides_interactively():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    override = [None, "My Reference", "My Diff"]
    obj = render(
        PlotSpec(
            family="section_row",
            items=[_section_row_item()],
            options={"labels": ("roms_run", "woa23"), "titles": override},
        ),
        renderer="holoviews",
    )
    titles = {
        qm.opts.get("plot").kwargs.get("title")
        for qm in obj.traverse(lambda x: x, [hv.QuadMesh])
    }
    assert titles == {"roms_run", "My Reference", "My Diff"}

    with pytest.raises(ValueError, match="needs one entry per panel"):
        render(
            PlotSpec(
                family="section_row",
                items=[_section_row_item()],
                options={"titles": ["only one"]},
            ),
            renderer="holoviews",
        )


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


# -- stacked section_row: one row per comparison, both renderers ---------------------
#
# A ComparisonSet of several section comparisons hands section_row several items; the
# renderers stack them as rows (section_row_grid / _section_row_grid), time_depth_row's
# own convention. Rows can run along different x quantities, so no x axis is shared
# between rows -- only depth's direction is common.


def _stacked_row_items(n: int = 2, *, same_path: bool = False) -> list[dict]:
    """``n`` section_row items with their own row labels and offsets.

    Each row's path is shifted in latitude (so the rows' ``path_note``s differ) unless
    ``same_path``; the offset also differs, so each row has its own colour range.
    """
    items = []
    for i in range(n):
        item = _section_row_item(
            labels=("roms_run", f"ref_{i}"), offset=1.0 + 2.0 * i
        )
        item["row_label"] = f"row {i}"
        if not same_path:
            item["aligned"] = {
                lane: da.assign_coords(
                    lat=(ALONG_DIM, np.linspace(24.0, 26.0, 8) + 5.0 * i)
                )
                for lane, da in item["aligned"].items()
            }
        items.append(item)
    return items


def _panels(fig):
    """Return the figure's data panels: every axes that is not a colorbar."""
    return [ax for ax in fig.axes if not getattr(ax, "_osk_cbar_parents", None)]


def _colorbars(fig):
    return [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]


@pytest.mark.parametrize("n", [2, 3])
def test_stacked_section_row_draws_n_by_3_panels_and_two_bars_per_row(n):
    fig = render(
        PlotSpec(family="section_row", items=_stacked_row_items(n)),
        renderer="matplotlib",
    )
    assert len(_panels(fig)) == 3 * n
    assert len(_colorbars(fig)) == 2 * n


def test_stacked_section_row_labels_each_row_and_titles_it_from_its_own_labels():
    fig = render(
        PlotSpec(family="section_row", items=_stacked_row_items(3)),
        renderer="matplotlib",
    )
    panels = _panels(fig)
    row_labels = [
        ax._osk_row_label.get_text() for ax in panels if hasattr(ax, "_osk_row_label")
    ]
    assert len(row_labels) == 3
    # the rows' paths differ, so each row's own path rides on its test-panel title --
    # not on the rotated label, which has no height in a short section row to spare
    assert row_labels == ["row 0", "row 1", "row 2"]
    titles = [ax.get_title() for ax in panels]
    assert all("→" in t for t in titles[0::3])
    assert titles[1::3] == ["ref_0", "ref_1", "ref_2"]  # each row's reference column
    assert titles[2::3] == ["difference"] * 3


def test_stacked_section_row_with_a_shared_path_names_it_once_in_the_suptitle():
    fig = render(
        PlotSpec(family="section_row", items=_stacked_row_items(2, same_path=True)),
        renderer="matplotlib",
    )
    assert "→" in fig._suptitle.get_text()
    labels = [
        ax._osk_row_label.get_text()
        for ax in _panels(fig)
        if hasattr(ax, "_osk_row_label")
    ]
    assert labels == ["row 0", "row 1"]  # nothing to add to the labels


def test_stacked_section_row_with_differing_paths_keeps_them_out_of_the_suptitle():
    fig = render(
        PlotSpec(family="section_row", items=_stacked_row_items(2)),
        renderer="matplotlib",
    )
    assert "→" not in (fig._suptitle.get_text() if fig._suptitle else "")


def test_stacked_section_row_panels_are_inverted_grey_and_do_not_share_x_across_rows():
    fig = render(
        PlotSpec(family="section_row", items=_stacked_row_items(2)),
        renderer="matplotlib",
    )
    panels = _panels(fig)
    for ax in panels:
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1] and ylim[0] >= 0  # positive-down, 0 m at the top
        assert ax.get_facecolor() == (0.85, 0.85, 0.85, 1.0)
    joined = panels[0].get_shared_x_axes()
    assert not joined.joined(panels[0], panels[3])  # row 0 vs row 1
    assert not panels[0].get_shared_y_axes().joined(panels[0], panels[3])


def test_stacked_section_row_takes_each_rows_x_label_from_its_own_geometry(monkeypatch):
    """A meridian row (degrees) beside a transect (km) keeps both x labels."""
    import dataclasses

    from ocean_skill.plot import section as section_module

    real = section_module.prepare_section_row
    calls = {"n": 0}

    def fake(aligned, x="auto"):
        values, geometry = real(aligned, x)
        calls["n"] += 1
        if calls["n"] == 2:
            geometry = dataclasses.replace(geometry, x_label="latitude (°N)")
        return values, geometry

    monkeypatch.setattr(section_module, "prepare_section_row", fake)
    fig = render(
        PlotSpec(family="section_row", items=_stacked_row_items(2)),
        renderer="matplotlib",
    )
    xlabels = [ax.get_xlabel() for ax in _panels(fig)]
    assert set(xlabels[:3]) == {"distance along transect (km)"}
    assert set(xlabels[3:]) == {"latitude (°N)"}


def _mesh_norms(fig):
    from matplotlib.collections import QuadMesh

    return [
        next(c for c in ax.collections if isinstance(c, QuadMesh)).norm
        for ax in _panels(fig)
    ]


def test_stacked_section_row_scales_each_row_on_its_own_by_default():
    fig = render(
        PlotSpec(family="section_row", items=_stacked_row_items(2)),
        renderer="matplotlib",
    )
    norms = _mesh_norms(fig)
    assert (norms[0].vmin, norms[0].vmax) == (norms[1].vmin, norms[1].vmax)  # in a row
    assert (norms[0].vmin, norms[0].vmax) != (norms[3].vmin, norms[3].vmax)  # across


def test_stacked_section_row_shared_limits_puts_every_row_on_one_scale():
    fig = render(
        PlotSpec(
            family="section_row",
            items=_stacked_row_items(3),
            options={"shared_limits": True},
        ),
        renderer="matplotlib",
    )
    norms = _mesh_norms(fig)
    seq = {(norms[i].vmin, norms[i].vmax) for i in (0, 1, 3, 4, 6, 7)}
    div = {(norms[i].vmin, norms[i].vmax) for i in (2, 5, 8)}
    assert len(seq) == 1
    assert len(div) == 1
    (lo, hi) = next(iter(div))
    assert lo == pytest.approx(-hi)


def test_stacked_section_row_metrics_box_sits_in_every_rows_difference_panel():
    fig = render(
        PlotSpec(family="section_row", items=_stacked_row_items(2)),
        renderer="matplotlib",
    )
    boxed = [ax for ax in _panels(fig) if getattr(ax, "_osk_metrics_text", None)]
    assert len(boxed) == 2
    assert all("bias=0.125" in ax._osk_metrics_text.get_text() for ax in boxed)


def test_stacked_section_row_titles_takes_three_per_row():
    fig = render(
        PlotSpec(
            family="section_row",
            items=_stacked_row_items(2),
            options={"titles": [None, None, None, "A", None, "B"]},
        ),
        renderer="matplotlib",
    )
    titles = [ax.get_title() for ax in _panels(fig)]
    assert titles[3] == "A"
    assert titles[5] == "B"
    assert titles[1] == "ref_0"

    with pytest.raises(ValueError, match="needs one entry per panel"):
        render(
            PlotSpec(
                family="section_row",
                items=_stacked_row_items(2),
                options={"titles": ["only three", None, None]},
            ),
            renderer="matplotlib",
        )


def test_a_single_item_section_row_still_draws_one_row_with_no_row_label():
    fig = render(
        PlotSpec(family="section_row", items=_stacked_row_items(1)),
        renderer="matplotlib",
    )
    assert len(_panels(fig)) == 3
    assert not any(hasattr(ax, "_osk_row_label") for ax in _panels(fig))


def test_domain_is_not_an_option_of_a_stacked_section_row():
    with pytest.raises(TypeError, match="not an option of section_row_grid"):
        render(
            PlotSpec(
                family="section_row",
                items=_stacked_row_items(2),
                options={"domain": (0.0, 0.0, 1.0, 1.0)},
            ),
            renderer="matplotlib",
        )


def test_stacked_section_row_renders_interactively_as_n_linked_rows():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    obj = render(
        PlotSpec(family="section_row", items=_stacked_row_items(3)),
        renderer="holoviews",
    )
    qms = obj.traverse(lambda x: x, [hv.QuadMesh])
    assert len(qms) == 9
    titles = [qm.opts.get("plot").kwargs.get("title") for qm in qms]
    assert titles[0].startswith("row 0 (") and titles[0].endswith("— roms_run")
    assert titles[3].startswith("row 1 (")
    assert [t for t in titles[1::3]] == ["ref_0", "ref_1", "ref_2"]
    for qm in qms:
        plot_kwargs = qm.opts.get("plot").kwargs
        assert plot_kwargs.get("bgcolor") == "#d9d9d9"
        assert plot_kwargs.get("invert_yaxis") is True


def test_stacked_section_row_interactive_shared_limits_matches_across_rows():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    obj = render(
        PlotSpec(
            family="section_row",
            items=_stacked_row_items(2),
            options={"shared_limits": True},
        ),
        renderer="holoviews",
    )
    clims = [qm.vdims[0].range for qm in obj.traverse(lambda x: x, [hv.QuadMesh])]
    assert clims[0] == clims[1] == clims[3] == clims[4]
    assert clims[2] == clims[5]


def test_stacked_section_row_interactive_keeps_rows_with_different_x_unlinked():
    """Rows with different along-path extents get distinct x names.

    So bokeh does not tie their x ranges together; rows with the same x share one name.
    """
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    same = render(
        PlotSpec(family="section_row", items=_stacked_row_items(2)),
        renderer="holoviews",
    )
    assert {
        qm.kdims[0].name for qm in same.traverse(lambda x: x, [hv.QuadMesh])
    } == {"distance"}

    items = _stacked_row_items(2)
    items[1]["aligned"] = {
        lane: da.assign_coords({ALONG_DIM: np.linspace(0.0, 400.0, 8)})
        for lane, da in items[1]["aligned"].items()
    }
    mixed = render(PlotSpec(family="section_row", items=items), renderer="holoviews")
    names = [qm.kdims[0].name for qm in mixed.traverse(lambda x: x, [hv.QuadMesh])]
    assert names[:3] == [names[0]] * 3 and names[3:] == [names[3]] * 3
    assert names[0] != names[3]
    labels = {qm.kdims[0].label for qm in mixed.traverse(lambda x: x, [hv.QuadMesh])}
    assert labels == {"distance"}  # the alias never reaches a reader


def test_domain_is_dropped_with_a_warning_for_a_stacked_section_row_interactively():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")

    with pytest.warns(UserWarning, match="not an option of section_row"):
        render(
            PlotSpec(
                family="section_row",
                items=_stacked_row_items(2),
                options={"domain": (0.0, 0.0, 1.0, 1.0)},
            ),
            renderer="holoviews",
        )


# -- stacked section: one panel per member (section_grid), both renderers ------------


def _stacked_section_items(n: int = 2, *, units=("degC",), **overrides) -> list[dict]:
    """``n`` section items: same variable by default, each shifted and labelled."""
    items = []
    for i in range(n):
        item = _section_item()
        item["field"] = item["field"] + 4.0 * i
        item["label"] = f"run {i}"
        item["units"] = units[i % len(units)]
        item["standard_name"] = "sea_water_potential_temperature"
        item["depth"] = None
        item.update(overrides)
        items.append(item)
    return items


def test_stacked_section_draws_a_panel_per_item_in_one_column():
    fig = render(
        PlotSpec(family="section", items=_stacked_section_items(3)),
        renderer="matplotlib",
    )
    panels = _panels(fig)
    assert len(panels) == 3
    lefts = {round(ax.get_position().x0, 3) for ax in panels}
    assert len(lefts) == 1  # one column
    for ax in panels:
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]
        assert ax.get_facecolor() == (0.85, 0.85, 0.85, 1.0)


def test_stacked_section_of_one_variable_shares_one_scale_and_one_colorbar():
    fig = render(
        PlotSpec(family="section", items=_stacked_section_items(3)),
        renderer="matplotlib",
    )
    assert len(_colorbars(fig)) == 1
    norms = _mesh_norms(fig)
    assert len({(n.vmin, n.vmax) for n in norms}) == 1
    # the shared scale spans every panel, not just the first
    assert norms[0].vmax > 5.0 + 1.0 + 4.0


def test_stacked_section_of_mixed_units_gets_a_scale_and_colorbar_per_panel():
    fig = render(
        PlotSpec(
            family="section", items=_stacked_section_items(2, units=("degC", "psu"))
        ),
        renderer="matplotlib",
    )
    assert len(_colorbars(fig)) == 2
    norms = _mesh_norms(fig)
    assert (norms[0].vmin, norms[0].vmax) != (norms[1].vmin, norms[1].vmax)


def test_stacked_section_shared_limits_false_forces_a_scale_per_panel():
    fig = render(
        PlotSpec(
            family="section",
            items=_stacked_section_items(2),
            options={"shared_limits": False},
        ),
        renderer="matplotlib",
    )
    assert len(_colorbars(fig)) == 2


def test_stacked_section_vmin_vmax_pin_the_shared_range():
    fig = render(
        PlotSpec(
            family="section",
            items=_stacked_section_items(2),
            options={"vmin": 0.0, "vmax": 30.0},
        ),
        renderer="matplotlib",
    )
    assert {(n.vmin, n.vmax) for n in _mesh_norms(fig)} == {(0.0, 30.0)}


def test_stacked_section_titles_each_panel_by_label_and_path_and_shares_the_rest():
    fig = render(
        PlotSpec(family="section", items=_stacked_section_items(2)),
        renderer="matplotlib",
    )
    titles = [ax.get_title() for ax in _panels(fig)]
    assert titles[0].startswith("run 0 — ") and "→" in titles[0]
    assert titles[1].startswith("run 1 — ")
    # the variable every panel shares is named once, on top
    assert fig._suptitle.get_text().lower() == "temperature"


def test_stacked_section_ncols_wraps_and_hides_the_blank_cell():
    fig = render(
        PlotSpec(
            family="section",
            items=_stacked_section_items(3),
            options={"ncols": 2},
        ),
        renderer="matplotlib",
    )
    visible = [ax for ax in _panels(fig) if ax.get_visible()]
    assert len(visible) == 3
    assert len(_panels(fig)) == 4  # 2 x 2 grid, one blank


def test_stacked_section_cols_variable_facets_a_grid():
    items = _stacked_section_items(4)
    for i, item in enumerate(items):
        item["standard_name"] = (
            "sea_water_potential_temperature",
            "sea_water_salinity",
        )[i % 2]
        item["label"] = ("north", "south")[i // 2]
        item["units"] = ("degC", "1")[i % 2]
    fig = render(
        PlotSpec(family="section", items=items, options={"cols": "variable"}),
        renderer="matplotlib",
    )
    panels = _panels(fig)
    assert len(panels) == 4
    assert len({round(ax.get_position().x0, 3) for ax in panels}) == 2  # 2 columns
    with pytest.raises(ValueError, match="already fix this grid's shape"):
        render(
            PlotSpec(
                family="section",
                items=items,
                options={"cols": "variable", "ncols": 2},
            ),
            renderer="matplotlib",
        )


def test_stacked_section_takes_each_panels_x_label_from_its_own_geometry(monkeypatch):
    import dataclasses

    from ocean_skill.plot import section as section_module

    real = section_module.prepare_section
    calls = {"n": 0}

    def fake(da, x="auto"):
        values, geometry = real(da, x)
        calls["n"] += 1
        if calls["n"] == 2:
            geometry = dataclasses.replace(geometry, x_label="latitude (°N)")
        return values, geometry

    monkeypatch.setattr(section_module, "prepare_section", fake)
    fig = render(
        PlotSpec(family="section", items=_stacked_section_items(2)),
        renderer="matplotlib",
    )
    assert [ax.get_xlabel() for ax in _panels(fig)] == [
        "distance along transect (km)",
        "latitude (°N)",
    ]
    ax0, ax1 = _panels(fig)
    assert not ax0.get_shared_x_axes().joined(ax0, ax1)


def test_a_single_item_section_is_unchanged_by_the_stacked_form():
    fig = render(
        PlotSpec(family="section", items=[_section_item()]), renderer="matplotlib"
    )
    assert len(_panels(fig)) == 1 and len(_colorbars(fig)) == 1


def test_domain_is_not_an_option_of_a_stacked_section():
    with pytest.raises(TypeError, match="not an option of section_grid"):
        render(
            PlotSpec(
                family="section",
                items=_stacked_section_items(2),
                options={"domain": (0.0, 0.0, 1.0, 1.0)},
            ),
            renderer="matplotlib",
        )


def test_stacked_section_renders_interactively_as_a_layout_of_panels():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    obj = render(
        PlotSpec(family="section", items=_stacked_section_items(3)),
        renderer="holoviews",
    )
    qms = obj.traverse(lambda x: x, [hv.QuadMesh])
    assert len(obj) == 3 and len(qms) == 3
    clims = {qm.vdims[0].range for qm in qms}
    assert len(clims) == 1  # one shared range (every panel is the same variable)
    for qm in qms:
        plot_kwargs = qm.opts.get("plot").kwargs
        assert plot_kwargs.get("bgcolor") == "#d9d9d9"
        assert plot_kwargs.get("invert_yaxis") is True
    titles = [qm.opts.get("plot").kwargs.get("title") for qm in qms]
    assert titles[0].startswith("run 0 — ")


def test_stacked_section_interactive_gives_mixed_units_a_range_per_panel():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    import holoviews as hv

    obj = render(
        PlotSpec(
            family="section", items=_stacked_section_items(2, units=("degC", "psu"))
        ),
        renderer="holoviews",
    )
    clims = [qm.vdims[0].range for qm in obj.traverse(lambda x: x, [hv.QuadMesh])]
    assert clims[0] != clims[1]


def test_domain_is_dropped_with_a_warning_for_a_stacked_section_interactively():
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")

    with pytest.warns(UserWarning, match="not an option of section"):
        render(
            PlotSpec(
                family="section",
                items=_stacked_section_items(2),
                options={"domain": (0.0, 0.0, 1.0, 1.0)},
            ),
            renderer="holoviews",
        )
