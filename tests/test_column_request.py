"""Tests for ``select={"depth": "column"}`` -- the whole water column, native levels
standing.

Complements ``tests/test_depth_average.py``'s band/average machinery: the same
synthetic grid (shelf to abyss, exercising both weighting regimes), but for the one
vertical request that keeps every native s-level rather than reducing to one number
or interpolating to fixed levels -- the request a profile at a point needs to reach
the model's own resolution.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

from ocean_skill import roms
from ocean_skill.comparison import COLUMN, _depth_label, _prepare, is_column_request

N = 20
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


@pytest.fixture
def roms_column():
    """A standardize()-shaped grid: h/mask_rho/sigma_*/Cs_* and z_rho as coords."""
    h = np.array([[20.0, 100.0], [1000.0, 5000.0]])
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    ds = xr.Dataset(
        {
            "temp": (
                ("s_rho", "eta_rho", "xi_rho"),
                np.linspace(5.0, 20.0, N)[:, None, None] * np.ones((N, 2, 2)),
                {"units": "degC"},
            )
        },
        coords={
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((2, 2))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
            "lon": (("eta_rho", "xi_rho"), np.array([[-95.0, -94.0], [-95.0, -94.0]])),
            "lat": (("eta_rho", "xi_rho"), np.array([[25.0, 25.0], [26.0, 26.0]])),
        },
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    ds = roms.add_depth_coord(ds, meta)  # standardize()'s own step
    return ds, meta


# -- the sentinel itself -----------------------------------------------------------


def test_is_column_request_recognizes_the_sentinel():
    assert is_column_request(COLUMN)
    assert is_column_request("column")
    assert is_column_request("COLUMN")
    assert not is_column_request(None)
    assert not is_column_request("surface")
    assert not is_column_request(50.0)
    assert not is_column_request({"min": 0, "max": 10})


def test_depth_label_spells_the_column_sentinel():
    assert _depth_label(COLUMN) == "full water column"


# -- through _prepare, at a point ---------------------------------------------------


def test_column_at_a_point_keeps_every_native_level(roms_column):
    ds, meta = roms_column
    da, actual = _prepare(
        ds, meta, "temp", {"depth": COLUMN, "eta_rho": 0, "xi_rho": 0}
    )
    assert da.dims == ("s_rho",)
    assert da.sizes["s_rho"] == N
    assert actual is None  # a surviving axis, not one number to label it with
    assert "actual_depth" not in da.attrs


def test_column_carries_z_rho_and_dz_for_a_profile_to_draw_against(roms_column):
    """The two things the plotting layer and a later vertical aggregate each need:
    z_rho (real depth per cell) and dz (thickness weights)."""
    ds, meta = roms_column
    da, _ = _prepare(ds, meta, "temp", {"depth": COLUMN, "eta_rho": 0, "xi_rho": 0})
    assert "z_rho" in da.coords
    assert "dz" in da.coords
    assert (np.asarray(da["z_rho"].values) <= 0).all()  # negative-down, ROMS's own


def test_column_levels_span_the_whole_column_not_just_a_band(roms_column):
    """Unlike depth_average's own band tests, nothing here is excluded: an unbounded
    band touches every cell from the surface to the seafloor."""
    ds, meta = roms_column
    # h[1, 1] == 5000.0 m -- the abyssal corner of this grid (see the fixture's h).
    da, _ = _prepare(ds, meta, "temp", {"depth": COLUMN, "eta_rho": 1, "xi_rho": 1})
    depths = -np.asarray(da["z_rho"].values)
    assert depths.min() < 20.0  # near the surface, not excluded by a band's floor
    assert depths.max() > 4000.0  # near this column's own ~5000 m seafloor


def test_a_vertical_aggregate_still_collapses_a_column_request(roms_column):
    """{"Z": "mean"} on a column is the same thickness-weighted mean depth_average
    gives a band -- the column is just an unbounded band."""
    ds, meta = roms_column
    da, _ = _prepare(
        ds,
        meta,
        "temp",
        {"depth": COLUMN, "eta_rho": 0, "xi_rho": 0},
        {"Z": "mean"},
    )
    assert "s_rho" not in da.dims
    assert 5.0 < float(da) < 20.0  # a real weighted mean of the column, not garbage


def test_column_and_sigma0_together_is_refused(roms_column):
    ds, meta = roms_column
    with pytest.raises(ValueError, match="cannot ask for both"):
        _prepare(ds, meta, "temp", {"depth": COLUMN, "sigma0": 25.0})


def test_column_on_an_observational_source_is_refused():
    ds = xr.Dataset(
        {"v": (("depth", "lat", "lon"), np.array([[[1.0]], [[3.0]]]), {})},
        coords={"depth": [0.0, 10.0], "lat": [1.0], "lon": [1.0]},
    )
    with pytest.raises(ValueError, match="native levels"):
        _prepare(ds, {}, "v", {"depth": COLUMN})


# -- through Field, end to end -------------------------------------------------------


def test_a_column_profile_draws_end_to_end_in_both_renderers(roms_column, monkeypatch):
    import ocean_skill.comparison as comparison
    from ocean_skill.field import field as make_field

    ds, meta = roms_column
    da, actual = _prepare(ds, meta, "temp", {"depth": COLUMN, "eta_rho": 0, "xi_rho": 0})
    monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (da, actual))

    f = make_field("stub_roms", "temperature")
    assert f.is_profile
    assert f.family == "profile"

    fig = f.plot()
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 1
    bottom, top = ax.get_ylim()
    assert bottom > top > 0  # deep at the bottom, shallow (but not 0) at the top

    obj = f.plot(renderer="holoviews")
    import holoviews as hv

    assert obj.traverse(lambda x: x, [hv.Curve])
