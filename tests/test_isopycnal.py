"""Tests for isopycnal slices: ocean_skill.roms.to_sigma0 and its comparison wiring.

A depth slice interpolates onto a surface of constant depth; this instead interpolates
onto a surface of constant potential density (sigma0, TEOS-10) -- physically the more
meaningful cut through a stratified column, since water masses move along density
surfaces rather than depth surfaces. The mechanism is the same xgcm vertical transform
:func:`ocean_skill.roms.to_depth` already uses, aimed at a different target coordinate.

Three layers, mirroring ``tests/test_mld.py``'s split for the same reason: the pure
transform (:func:`ocean_skill.roms.to_sigma0`, checked against gsw computed
independently), the wiring into :func:`ocean_skill.comparison._prepare` (including the
temperature/salinity re-attachment that has to go through the *same* select/aggregate
the sliced variable already did -- see the fixture and
``test_temperature_and_salinity_follow_the_lanes_reduction``), and the rendering path
(both renderers spell a density row as a density, not a depth).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import roms
from ocean_skill.comparison import _prepare, _sigma_label
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

N = 20
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


@pytest.fixture
def roms_column():
    """Build a shelf-to-abyss grid, stratified so sigma0 is monotonic in every column.

    Same grid ``tests/test_depth_average.py`` uses, plus temperature that cools
    linearly with depth (constant salinity) so potential density increases
    monotonically downward -- one crossing per target, nowhere to go wrong on
    which one xgcm's linear transform should pick. ``chl`` is set equal to each
    cell's own depth, so a slice's numeric answer can be checked independently
    with plain ``np.interp`` rather than trusting the transform to check itself.
    """
    h = np.array([[20.0, 100.0], [1000.0, 5000.0]])
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    ds = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((2, 2))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
        },
        coords={
            "lon": (("eta_rho", "xi_rho"), np.array([[-95.0, -94.0], [-95.0, -94.0]])),
            "lat": (("eta_rho", "xi_rho"), np.array([[25.0, 25.0], [26.0, 26.0]])),
        },
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    ds = roms.add_depth_coord(ds, meta)
    temp = 20.0 + 0.002 * ds["z_rho"]  # z_rho negative-down: colder with depth
    salt = xr.full_like(temp, 35.0)
    chl = -ds["z_rho"]  # value == its own depth, for an independent check
    ds = ds.assign(
        sea_water_potential_temperature=temp,
        sea_water_practical_salinity=salt,
        chl=chl,
    )
    return ds, meta


# -- the transform itself, against gsw computed independently -----------------


def test_slice_recovers_the_known_depth_of_a_target_isopycnal(roms_column):
    """``chl`` is depth, so the sliced value must equal the isopycnal's own depth.

    Computed independently with ``np.interp``, not by trusting the transform.
    """
    gsw = pytest.importorskip("gsw")
    ds, meta = roms_column
    col = dict(eta_rho=1, xi_rho=1)  # abyssal column: 20 levels resolve it finely
    z = ds["z_rho"].isel(**col).values
    temp = ds["sea_water_potential_temperature"].isel(**col).values
    salt = ds["sea_water_practical_salinity"].isel(**col).values
    lon, lat = float(ds["lon"].isel(**col)), float(ds["lat"].isel(**col))
    pressure = gsw.p_from_z(z, lat)
    sa = gsw.SA_from_SP(salt, pressure, lon, lat)
    ct = gsw.CT_from_pt(sa, temp)
    sigma0_profile = gsw.sigma0(sa, ct)

    order = np.argsort(sigma0_profile)
    mid = len(order) // 2
    # strictly between two real grid values, so this is a genuine interpolation
    # check and not a coincidental match to one of the knots.
    target = float(0.5 * (sigma0_profile[order][mid] + sigma0_profile[order][mid - 1]))
    expected_depth = np.interp(target, sigma0_profile[order], -z[order])

    out = roms.to_sigma0(ds, meta, target)["chl"].isel(sigma0=0, **col)
    assert float(out) == pytest.approx(expected_depth, rel=1e-6)


def test_target_outside_the_columns_range_is_nan_and_warns(roms_column):
    """A sigma0 denser than anything on this grid interpolates to nothing, per level.

    The same shape :func:`ocean_skill.roms.to_depth` uses beyond the water column.
    """
    ds, meta = roms_column
    with pytest.warns(UserWarning, match="outside this water column's sigma0 range"):
        out = roms.to_sigma0(ds, meta, 100.0)
    assert np.isnan(out["chl"].values).all()


def test_a_full_density_value_is_refused_rather_than_silently_sliced(roms_column):
    """1026.5 is a plausible in-situ density; sigma0 is that minus 1000.

    Refused rather than silently naming a surface far from the one meant (see
    roms.py's to_sigma0 docstring for why there is no 'rho'/'density' alias here).
    """
    ds, meta = roms_column
    with pytest.raises(ValueError, match="looks like a full density"):
        roms.to_sigma0(ds, meta, 1026.5)


def test_missing_salinity_names_the_standard_name_and_a_fix(roms_column):
    ds, meta = roms_column
    ds = ds.drop_vars("sea_water_practical_salinity")
    with pytest.raises(ValueError, match="sea_water_practical_salinity"):
        roms.to_sigma0(ds, meta, 26.0)


def test_gsw_missing_raises_an_install_hint(roms_column, monkeypatch):
    ds, meta = roms_column
    monkeypatch.setitem(sys.modules, "gsw", None)  # simulate "not installed"
    with pytest.raises(ImportError, match="conda install"):
        roms.to_sigma0(ds, meta, 26.0)


def test_units_and_provenance_survive(roms_column):
    pytest.importorskip("gsw")
    ds, meta = roms_column
    out = roms.to_sigma0(ds, meta, 25.0)
    assert out["sigma0"].attrs["units"] == "kg m-3"
    assert "TEOS-10" in out["chl"].attrs["isopycnal_slice"]


# -- through the compare layer -------------------------------------------------


def test_prepare_refuses_a_depth_and_a_density_together(roms_column):
    ds, meta = roms_column
    with pytest.raises(ValueError, match="cannot ask for both a depth"):
        _prepare(ds, meta, "chl", {"depth": 50, "sigma0": 26.0}, {})


def test_prepare_refuses_sigma0_on_a_non_roms_source():
    """No ROMS grid, no temperature/salinity to compute an isopycnal from at all."""
    ds = xr.Dataset(
        {"v": (("depth", "lat", "lon"), np.ones((2, 1, 1)))},
        coords={"depth": [0.0, 10.0], "lat": [1.0], "lon": [1.0]},
    )
    with pytest.raises(ValueError, match="needs a ROMS source"):
        _prepare(ds, {}, "v", {"sigma0": 26.0}, {})


def test_prepare_refuses_sigma0_alongside_a_calculator(roms_column):
    """Mixed layer depth already collapses the vertical axis.

    A density surface to slice it at is as much a contradiction as a fixed depth.
    """
    ds, meta = roms_column
    spec = {"calculate": "mld", "method": "temperature_threshold"}
    with pytest.raises(ValueError, match="already reduces the vertical axis"):
        _prepare(ds, meta, spec, {"sigma0": 26.0}, {})


def test_prepare_scalar_sigma0_collapses_the_axis(roms_column):
    pytest.importorskip("gsw")
    ds, meta = roms_column
    da, _ = _prepare(ds, meta, "chl", {"sigma0": 25.0}, {})
    assert set(da.dims) == {"eta_rho", "xi_rho"}


def test_prepare_a_list_of_isopycnals_keeps_the_axis_for_facet_rows(roms_column):
    pytest.importorskip("gsw")
    ds, meta = roms_column
    # both values sit inside the abyssal column's sigma0 range (~24.77-26.89), so
    # neither level is NaN everywhere and no warning fires.
    da, _ = _prepare(ds, meta, "chl", {"sigma0": [24.8, 25.0]}, {})
    assert da.sizes["sigma0"] == 2

    from ocean_skill.field import _facet_dims

    # the spatial dims lon/lat actually *ride on* -- a curvilinear ROMS grid, like a
    # real field's eta_rho/xi_rho, not the coordinate names themselves. One leftover
    # axis is the series case and becomes the column, not a row -- a second leftover
    # axis is what puts the vertical one on the rows instead, which
    # test_facet_dims_puts_sigma0_on_the_rows_like_a_depth checks directly.
    row_dim, col_dim = _facet_dims(da, {"eta_rho", "xi_rho"})
    assert (row_dim, col_dim) == (None, "sigma0")


def test_aggregate_over_sigma0_runs_after_the_slice_not_before(roms_column):
    """A guard on ``_vertical_only``/``_without_vertical``.

    Without it, ``aggregate={"sigma0": "mean"}`` is evaluated before the slice
    exists (where there is no such dimension yet, so it is silently skipped)
    rather than after.
    """
    pytest.importorskip("gsw")
    ds, meta = roms_column
    da, _ = _prepare(
        ds, meta, "chl", {"sigma0": [24.8, 25.0, 25.2]}, {"sigma0": "mean"}
    )
    assert set(da.dims) == {"eta_rho", "xi_rho"}


def test_temperature_and_salinity_follow_the_lanes_selection_and_aggregation(
    roms_column,
):
    """Regression test: temperature/salinity must go through the lane's own reduction.

    Re-attaching them *raw* (as the static grid variables are) would leave them
    carrying a time axis the sliced field no longer has once
    ``aggregate={"time": "mean"}`` has run -- xgcm's transform cannot reconcile a
    time-varying target against a time-collapsed field. Reducing the re-attached
    columns through the same select/aggregate first must give exactly the
    time-invariant answer, since temperature does not actually vary here.
    """
    pytest.importorskip("gsw")
    ds, meta = roms_column
    baseline, _ = _prepare(ds, meta, "chl", {"sigma0": 25.0}, {})

    times = xr.DataArray([0.0, 0.0], dims="time")
    ds_t = ds.assign(
        sea_water_potential_temperature=(
            ds["sea_water_potential_temperature"] + times
        ).transpose("time", "s_rho", "eta_rho", "xi_rho")
    )
    da, _ = _prepare(ds_t, meta, "chl", {"sigma0": 25.0}, {"time": "mean"})
    assert "time" not in da.dims
    assert "sigma0" not in da.dims
    xr.testing.assert_allclose(da, baseline)


def test_compare_refuses_both_depths_and_a_sigma0_select():
    from ocean_skill.comparison import compare

    with pytest.raises(ValueError, match="got both depths= and select"):
        compare(
            reference="dummy_ref",
            test="dummy_test",
            variables=["chl"],
            depths=(50,),
            select={"sigma0": 26.0},
        )


# -- labels: a density, spelled as a density, in both renderers ---------------


def test_sigma_label_formats_a_scalar_and_a_list():
    assert _sigma_label(26.5) == "σ₀ = 26.5 kg/m³"
    assert _sigma_label([24.0, 26.5]) == "σ₀ = 24 kg/m³, σ₀ = 26.5 kg/m³"


def test_facet_labels_reads_sigma0_as_a_density_not_a_depth():
    from ocean_skill.plot.matplotlib_renderer import facet_labels

    da = xr.DataArray([1.0, 2.0], dims="sigma0", coords={"sigma0": [26.0, 26.5]})
    assert facet_labels(da["sigma0"]) == ["σ₀ = 26 kg/m³", "σ₀ = 26.5 kg/m³"]


def test_facet_dims_puts_sigma0_on_the_rows_like_a_depth():
    from ocean_skill.field import _facet_dims

    da = xr.DataArray(
        np.zeros((2, 3, 2, 2)),
        dims=("sigma0", "time", "eta_rho", "xi_rho"),
        coords={"sigma0": [26.0, 26.5]},
    )
    row_dim, col_dim = _facet_dims(da, {"eta_rho", "xi_rho"})
    assert row_dim == "sigma0"
    assert col_dim == "time"


def _sigma0_grid(values=(24.0, 25.0), months=2):
    """Build a field with sigma0 rows and month columns, by hand.

    Only the shape and the dimension's name matter for the rendering path this
    exercises, not a real density transform (that is what the compare-layer tests
    above check).
    """
    time = pd.date_range("2012-01-01", periods=months, freq="MS")
    rng = np.random.default_rng(0)
    base = xr.DataArray(
        rng.normal(5.0, 1.0, (months, 12, 20)),
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": np.linspace(18, 31, 12),
            "lon": np.linspace(-98, -80, 20),
        },
        attrs={"units": "mmol m-3"},
    )
    levels = [base * (1.0 - i * 0.4) - i * 3.0 for i in range(len(values))]
    return xr.concat(levels, dim=pd.Index(list(values), name="sigma0"))


def _item(field, facet_dim, row_dim=None):
    return {
        "field": field,
        "facet_dim": facet_dim,
        "row_dim": row_dim,
        "units": "mmol m-3",
    }


@pytest.mark.slow
def test_static_renderer_labels_sigma0_rows_as_a_density():
    field = _sigma0_grid()
    fig = render(PlotSpec(family="field_facet", items=[_item(field, "time", "sigma0")]))
    panels = [ax for ax in fig.axes if not getattr(ax, "_osk_cbar_parents", None)]
    assert len(panels) == 4  # 2 isopycnals x 2 months
    labels = [
        ax._osk_row_label.get_text()
        for ax in panels
        if getattr(ax, "_osk_row_label", None) is not None
    ]
    assert labels == ["σ₀ = 24 kg/m³", "σ₀ = 25 kg/m³"]


@pytest.mark.slow
def test_holoviews_renderer_titles_sigma0_rows_as_a_density_too():
    """The same figure through the interactive renderer.

    Bokeh has no rotated row label, so the level is folded into each panel's own
    title instead.
    """
    import holoviews as hv
    from bokeh.plotting import figure as bokeh_figure

    field = _sigma0_grid()
    obj = render(
        PlotSpec(family="field_facet", items=[_item(field, "time", "sigma0")]),
        renderer="holoviews",
    )
    titles = [
        f.title.text
        for f in hv.render(obj, backend="bokeh").select({"type": bokeh_figure})
    ]
    assert any("σ₀ = 24 kg/m³" in t for t in titles)
    assert any("σ₀ = 25 kg/m³" in t for t in titles)
