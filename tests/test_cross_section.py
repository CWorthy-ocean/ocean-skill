"""Tests for point-centered cross transects: two sections through one point.

``select={"transect": {"xi_rho": 30, "center": 40, "half_width": 15}}`` windows a
grid-aligned transect (see ``tests/test_transect.py``) to a stretch of the
surviving axis rather than the whole line; ``select={"transect": {"xi_rho": {"lon":
..., "lat": ...}, "half_width": 15}}`` resolves both the fixed index and the
window's center from one nearest-cell lookup. ``select={"transect": {"cross":
...}}`` is sugar for *two* such windows sharing one point, one along each grid
direction -- built by :func:`ocean_skill.field.field` into a
:class:`~ocean_skill.field.Cross` of two independent
:class:`~ocean_skill.field.Field` objects, drawn together by the new ``cross``
plot family.

Three layers, the house convention (see ``tests/test_transect.py``): the pure
grammar/extraction functions (:func:`ocean_skill.transect.as_transect`/
``grid_slice``), the wiring into :func:`ocean_skill.field.field` (building a
``Cross`` of two sections), and an end-to-end :class:`~ocean_skill.field.Field`
pipeline drawn through both renderers.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog, roms
from ocean_skill.align import ALONG_DIM
from ocean_skill.field import Cross
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec
from ocean_skill.transect import apply_transect, as_transect, grid_slice

N = 8
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


@pytest.fixture
def roms_grid():
    """Build a 21x15 ROMS-shaped grid.

    Big enough for a real ±3-cell window away from every edge, and for a ±15
    (default) window to actually clamp at one.
    """
    ny, nx = 21, 15
    h = np.linspace(30.0, 3000.0, ny * nx).reshape(ny, nx)
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    lon_1d = np.linspace(-96.0, -92.0, nx)
    lat_1d = np.linspace(20.0, 30.0, ny)
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
    chl = -ds["z_rho"]  # value == its own depth, for an independent check
    ds = ds.assign(chl=chl)
    return ds, meta


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


# -- as_transect: windowed-grid grammar ---------------------------------------------


def test_as_transect_reads_a_bare_dim_and_index_unchanged():
    """The plain, whole-line form is untouched by the new windowing/cross forms."""
    assert as_transect({"xi_rho": 30}) == {"kind": "grid", "dim": "xi_rho", "index": 30}


def test_as_transect_reads_an_index_windowed_transect():
    assert as_transect({"xi_rho": 30, "center": 40, "half_width": 5}) == {
        "kind": "grid",
        "dim": "xi_rho",
        "index": 30,
        "along_center": 40,
        "half_width": 5,
    }


def test_as_transect_windowed_needs_both_center_and_half_width():
    with pytest.raises(ValueError, match="both 'center'"):
        as_transect({"xi_rho": 30, "center": 40})
    with pytest.raises(ValueError, match="both 'center'"):
        as_transect({"xi_rho": 30, "half_width": 5})


def test_as_transect_reads_a_point_anchored_transect():
    parsed = as_transect({"xi_rho": {"lon": -94.0, "lat": 26.0}, "half_width": 10})
    assert parsed == {
        "kind": "grid",
        "dim": "xi_rho",
        "point": {"lon": -94.0, "lat": 26.0},
        "half_width": 10,
    }


def test_as_transect_point_anchored_defaults_half_width():
    parsed = as_transect({"xi_rho": {"lon": -94.0, "lat": 26.0}})
    assert parsed["half_width"] == 15  # _DEFAULT_HALF_WIDTH


def test_as_transect_point_anchored_rejects_center():
    with pytest.raises(ValueError, match="'center' does not apply"):
        as_transect({"xi_rho": {"lon": -94.0, "lat": 26.0}, "center": 5})


def test_as_transect_rejects_a_malformed_point():
    with pytest.raises(ValueError, match="a point must be exactly"):
        as_transect({"xi_rho": {"lon": -94.0}})


# -- as_transect: cross grammar -------------------------------------------------------


def test_as_transect_reads_a_cross_by_grid_indices():
    assert as_transect({"cross": {"eta_rho": 10, "xi_rho": 7}}) == {
        "kind": "cross",
        "point": {"eta_rho": 10, "xi_rho": 7},
        "half_width": 15,
    }


def test_as_transect_reads_a_cross_by_lon_lat():
    assert as_transect({"cross": {"lon": -94.0, "lat": 26.0}, "half_width": 5}) == {
        "kind": "cross",
        "point": {"lon": -94.0, "lat": 26.0},
        "half_width": 5,
    }


def test_as_transect_cross_accepts_custom_dims():
    parsed = as_transect(
        {"cross": {"lon": -94.0, "lat": 26.0}, "dims": ["eta_rho", "xi_rho"]}
    )
    assert parsed["point"]["dims"] == ("eta_rho", "xi_rho")


def test_as_transect_cross_rejects_a_malformed_point():
    with pytest.raises(ValueError, match="name the crossing point"):
        as_transect({"cross": {"lon": -94.0}})
    with pytest.raises(ValueError, match="name the crossing point"):
        as_transect({"cross": 5})


def test_as_transect_cross_rejects_dims_alongside_grid_indices():
    with pytest.raises(ValueError, match="only applies alongside"):
        as_transect({"cross": {"eta_rho": 10, "xi_rho": 7}, "dims": ["a", "b"]})


def test_as_transect_cross_rejects_unknown_sibling_keys():
    with pytest.raises(ValueError, match="do not apply alongside 'cross'"):
        as_transect({"cross": {"eta_rho": 10, "xi_rho": 7}, "method": "nearest"})


# -- grid_slice: windowing and point resolution ---------------------------------------


def test_grid_slice_windows_around_an_explicit_center(roms_grid):
    ds, _ = roms_grid
    out = grid_slice(ds, "xi_rho", 7, along_center=10, half_width=3)
    assert out.sizes[ALONG_DIM] == 7  # 3 either side + the center itself
    assert np.allclose(out["lat"].values, ds["lat"].values[7:14, 7])


def test_grid_slice_resolves_a_point_to_the_nearest_cell(roms_grid):
    ds, _ = roms_grid
    lon0, lat0 = float(ds["lon"].values[10, 7]), float(ds["lat"].values[10, 7])
    out = grid_slice(ds, "xi_rho", point={"lon": lon0, "lat": lat0}, half_width=3)
    assert out.sizes[ALONG_DIM] == 7
    expected = grid_slice(ds, "xi_rho", 7, along_center=10, half_width=3)
    assert np.allclose(out["lat"].values, expected["lat"].values)
    assert np.allclose(out["lon"].values, expected["lon"].values)


def test_grid_slice_clamps_a_window_at_the_domain_edge(roms_grid):
    ds, _ = roms_grid
    with pytest.warns(UserWarning, match="clamped"):
        out = grid_slice(ds, "xi_rho", 7, along_center=1, half_width=5)
    # center=1, half_width=5 -> [1-5, 1+5] clamped to [0, 6]: 7 cells, not 11
    assert out.sizes[ALONG_DIM] == 7
    assert np.allclose(out["lat"].values, ds["lat"].values[0:7, 7])


def test_grid_slice_out_of_range_center_raises(roms_grid):
    ds, _ = roms_grid
    with pytest.raises(ValueError, match="out of range"):
        grid_slice(ds, "xi_rho", 7, along_center=999, half_width=3)


def test_apply_transect_dispatches_windowed_and_point_forms(roms_grid):
    ds, _ = roms_grid
    windowed = apply_transect(ds, {"xi_rho": 7, "center": 10, "half_width": 3})
    assert windowed.sizes[ALONG_DIM] == 7
    lon0, lat0 = float(ds["lon"].values[10, 7]), float(ds["lat"].values[10, 7])
    anchored = apply_transect(
        ds, {"xi_rho": {"lon": lon0, "lat": lat0}, "half_width": 3}
    )
    assert anchored.sizes[ALONG_DIM] == 7


def test_apply_transect_refuses_a_cross_directly(roms_grid):
    ds, _ = roms_grid
    with pytest.raises(ValueError, match="only supported through osk\\.field"):
        apply_transect(ds, {"cross": {"eta_rho": 10, "xi_rho": 7}})


# -- field(): expanding a cross into two Fields ----------------------------------------


def test_field_cross_by_indices_builds_two_windowed_sections(patched_read, roms_grid):
    ds, _ = roms_grid
    name = patched_read(ds)
    c = osk.field(
        name,
        "chl",
        select={"transect": {"cross": {"eta_rho": 10, "xi_rho": 7}, "half_width": 3}},
        cache=False,
    )
    assert isinstance(c, Cross)
    assert c.along.select["transect"] == {"xi_rho": 7, "center": 10, "half_width": 3}
    assert c.across.select["transect"] == {"eta_rho": 10, "center": 7, "half_width": 3}
    assert c.labels == ("along eta_rho", "along xi_rho")
    assert c.along.family == "section"
    assert c.across.family == "section"
    assert c.along.data.sizes[ALONG_DIM] == 7
    assert c.across.data.sizes[ALONG_DIM] == 7


def test_field_cross_by_lon_lat_resolves_lazily_per_field(patched_read, roms_grid):
    ds, _ = roms_grid
    name = patched_read(ds)
    lon0, lat0 = float(ds["lon"].values[10, 7]), float(ds["lat"].values[10, 7])
    c = osk.field(
        name,
        "chl",
        select={"transect": {"cross": {"lon": lon0, "lat": lat0}, "half_width": 3}},
        cache=False,
    )
    # the raw lon/lat point is carried unresolved into each direction's own select
    assert c.along.select["transect"] == {
        "xi_rho": {"lon": lon0, "lat": lat0},
        "half_width": 3,
    }
    assert c.across.select["transect"] == {
        "eta_rho": {"lon": lon0, "lat": lat0},
        "half_width": 3,
    }
    assert c.along.data.sizes[ALONG_DIM] == 7
    assert c.across.data.sizes[ALONG_DIM] == 7


def test_field_cross_shares_the_rest_of_select_and_aggregate(patched_read, roms_grid):
    ds, _ = roms_grid
    name = patched_read(ds)
    c = osk.field(
        name,
        "chl",
        select={
            "transect": {"cross": {"eta_rho": 10, "xi_rho": 7}, "half_width": 3},
            "depth": [50.0, 500.0],
        },
        cache=False,
    )
    assert c.along.select["depth"] == [50.0, 500.0]
    assert c.across.select["depth"] == [50.0, 500.0]


def test_field_cross_refuses_source_or_variable_lists():
    with pytest.raises(ValueError, match="once per source/variable"):
        osk.field(
            ["a", "b"],
            "chl",
            select={"transect": {"cross": {"eta_rho": 10, "xi_rho": 7}}},
        )
    with pytest.raises(ValueError, match="once per source/variable"):
        osk.field(
            "a",
            ["chl", "temp"],
            select={"transect": {"cross": {"eta_rho": 10, "xi_rho": 7}}},
        )


# -- end-to-end: Cross.plot() in both renderers ----------------------------------------


def test_cross_plot_draws_two_panels_stacked_by_default(patched_read, roms_grid):
    ds, _ = roms_grid
    name = patched_read(ds)
    c = osk.field(
        name,
        "chl",
        select={"transect": {"cross": {"eta_rho": 10, "xi_rho": 7}, "half_width": 3}},
        cache=False,
    )
    fig = c.plot()
    # 2 panels + 1 shared colorbar
    assert len(fig.axes) == 3


def test_cross_plot_orientation_horizontal(patched_read, roms_grid):
    ds, _ = roms_grid
    name = patched_read(ds)
    c = osk.field(
        name,
        "chl",
        select={"transect": {"cross": {"eta_rho": 10, "xi_rho": 7}, "half_width": 3}},
        cache=False,
    )
    fig = c.plot(orientation="horizontal")
    assert len(fig.axes) == 3


@pytest.fixture
def roms_grid_with_land():
    """Like ``roms_grid``, but with a land strip and a real free surface.

    Reproduces ``roms.standardize``'s own masked-zeta chain (zeta masked over
    land *before* ``add_depth_coord`` builds ``z_rho`` from it) -- the gap
    ``roms_grid``'s zero-zeta, all-wet shape leaves untested. This is the exact
    shape that reached real ROMS output (an ``esper`` run) as a bare
    ``pcolormesh`` ``ValueError``: a native-s ``cross`` crossing land.
    """
    ny, nx = 21, 15
    h = np.linspace(30.0, 3000.0, ny * nx).reshape(ny, nx)
    mask = np.ones((ny, nx))
    mask[:3, :3] = 0.0  # a land corner near the cross center below
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    lon_1d = np.linspace(-96.0, -92.0, nx)
    lat_1d = np.linspace(20.0, 30.0, ny)
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
    ds = ds.assign(chl=(20.0 + 0.002 * ds["z_rho"]).where(ds["mask_rho"] == 1))
    return ds


def test_cross_through_land_renders_statically(patched_read, roms_grid_with_land):
    """The exact ``esper`` failure, and its fix: a native-s cross reaching land.

    ``roms_grid_with_land`` reproduces the masked-zeta chain that *would* leave
    ``z_rho`` NaN over land, but ``osk.field`` (via ``comparison._prepare``'s
    ``zero_zeta=True`` section handling) rebuilds it from ``h`` alone first, so
    it is finite here already -- no NaN depth reaches ``pcolormesh``, and no
    transect-mean placeholder is drawn into the land boundary either (the
    bathymetry "dips" that placeholder used to cause).
    """
    name = patched_read(roms_grid_with_land)
    c = osk.field(
        name,
        "chl",
        select={"transect": {"cross": {"eta_rho": 2, "xi_rho": 2}, "half_width": 3}},
        cache=False,
    )
    assert bool(np.isfinite(np.asarray(c.along.data["z_rho"])).all())
    fig = c.plot()
    assert len(fig.axes) == 3


def test_cross_through_land_renders_interactively(patched_read, roms_grid_with_land):
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    name = patched_read(roms_grid_with_land)
    c = osk.field(
        name,
        "chl",
        select={"transect": {"cross": {"eta_rho": 2, "xi_rho": 2}, "half_width": 3}},
        cache=False,
    )
    obj = c.plot(renderer="holoviews")
    assert obj is not None


def test_cross_plot_rejects_a_bad_orientation(patched_read, roms_grid):
    ds, _ = roms_grid
    name = patched_read(ds)
    c = osk.field(
        name,
        "chl",
        select={"transect": {"cross": {"eta_rho": 10, "xi_rho": 7}, "half_width": 3}},
        cache=False,
    )
    with pytest.raises(ValueError, match="expected 'vertical'"):
        c.plot(orientation="sideways")


def test_cross_plot_via_registry_with_hand_built_spec(patched_read, roms_grid):
    """Exercise the ``cross`` family through the registry directly.

    The same way ``tests/test_section.py`` checks ``section``.
    """
    ds, _ = roms_grid
    name = patched_read(ds)
    c = osk.field(
        name,
        "chl",
        select={"transect": {"cross": {"eta_rho": 10, "xi_rho": 7}, "half_width": 3}},
        cache=False,
    )
    items = [c.along.as_item(), c.across.as_item()]
    fig = render(PlotSpec(family="cross", items=items), renderer="matplotlib")
    assert len(fig.axes) == 3

    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    obj = render(PlotSpec(family="cross", items=items), renderer="holoviews")
    assert obj is not None


def test_cross_plot_interactive_renderer(patched_read, roms_grid):
    pytest.importorskip("holoviews")
    pytest.importorskip("hvplot")
    ds, _ = roms_grid
    name = patched_read(ds)
    c = osk.field(
        name,
        "chl",
        select={"transect": {"cross": {"eta_rho": 10, "xi_rho": 7}, "half_width": 3}},
        cache=False,
    )
    obj = c.plot(renderer="holoviews")
    assert obj is not None
    obj2 = c.plot(renderer="holoviews", orientation="horizontal")
    assert obj2 is not None


def test_cross_save_writes_a_figure_named_distinctly_from_a_plain_field(
    patched_read, roms_grid, tmp_path
):
    """Regression: a long standard_name must not push '_cross' off the stem.

    See the fix in ``Cross.save()``: truncate the base name *before* appending
    the suffix, not the concatenated string after.
    """
    from ocean_skill import outputs

    ds, _ = roms_grid
    name = patched_read(ds)
    outputs.set_base(tmp_path)
    try:
        c = osk.field(
            name,
            "chl",  # resolves to a long standard_name (>18 chars)
            select={
                "transect": {"cross": {"eta_rho": 10, "xi_rho": 7}, "half_width": 3}
            },
            cache=False,
        )
        result = c.save("proj")
        assert result["figure"].exists()
        assert result["figure"].name.endswith("_cross.png")
    finally:
        outputs.set_base(None)
