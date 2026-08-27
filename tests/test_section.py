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
