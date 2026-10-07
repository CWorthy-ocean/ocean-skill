"""A slab -- a box averaged along one horizontal axis -- draws as a vertical section.

``aggregate={"lon": "mean"}`` over a lon range leaves latitude and depth standing: the
same shape of figure a ``transect`` draws, so the slab is re-spelled as one
(:func:`ocean_skill.comparison._slab_to_section`) and everything downstream that already
knows a section (:func:`ocean_skill.align.path_of`, ``_align_along_path``, the section
plots) handles it with no special case of its own.

Layers, the house convention: the spec-level decision (:func:`~ocean_skill.operators
.slab_axis`, a truth table), the curvilinear binned mean (checked against hand-computed
bin means), the real pipeline for a :class:`~ocean_skill.field.Field` and a
:class:`~ocean_skill.comparison.Comparison` on rectilinear and curvilinear sources, and
the section plot's x axis and depth-sign handling.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog
from ocean_skill.align import ALONG_DIM
from ocean_skill.operators import aggregate, slab_axis, slab_in_spec
from tests.test_section_comparison import (  # noqa: F401  (patched_sources is a fixture)
    HC,
    N_S,
    _climatology,
    _roms_run,
    patched_sources,
)

VAR = "sea_water_potential_temperature"
BOX = {"lon": {"min": 180, "max": 200}, "lat": {"min": -30, "max": 30}}
DEPTHS = [0.0, 50.0, 200.0]


def _woa_like(*, descending_lat: bool = False) -> xr.Dataset:
    """A rectilinear 0-360 climatology whose lon mean is known in closed form.

    ``value = depth / 100 + lat / 10 + lon / 1000``: the mean over lon 180..200
    (three columns, 180/190/200) is ``depth / 100 + lat / 10 + 0.19``.
    """
    lon = np.arange(150.0, 251.0, 10.0)
    lat = np.arange(-30.0, 31.0, 10.0)
    if descending_lat:
        lat = lat[::-1]
    depth = np.array([0.0, 50.0, 200.0])
    d, la, lo = np.meshgrid(depth, lat, lon, indexing="ij")
    values = d / 100 + la / 10 + lo / 1000
    da = xr.DataArray(
        values,
        dims=("depth", "lat", "lon"),
        coords={
            "depth": ("depth", depth, {"positive": "down", "units": "m"}),
            "lat": lat,
            "lon": lon,
        },
        name=VAR,
        attrs={"units": "degC"},
    )
    return da.to_dataset()


@pytest.fixture
def patched_woa(monkeypatch):
    """Patch osk.read/catalog.resolve so a real pipeline runs against ``_woa_like``."""

    def _patch(ds=None, *, meta=None):
        ds = _woa_like() if ds is None else ds
        meta = {} if meta is None else meta
        monkeypatch.setattr(osk, "read", lambda n, **kw: ds)
        monkeypatch.setattr(
            catalog, "resolve", lambda n: SimpleNamespace(metadata=meta)
        )
        return "woa_x"

    return _patch


# -- spec level: slab_axis ------------------------------------------------------------

MEAN_LON = {"lon": "mean"}
MEAN_LAT = {"lat": "mean"}


@pytest.mark.parametrize(
    ("select", "agg", "expected"),
    [
        # the canonical slab: lon band averaged, lat range kept
        (BOX, MEAN_LON, "lat"),
        (BOX, {"time": "mean", "lon": "mean"}, "lat"),
        (BOX, MEAN_LAT, "lon"),
        # {"reduce": "mean"} and the X/Y spellings are plain means too
        (BOX, {"lon": {"reduce": "mean"}}, "lat"),
        ({"X": {"min": 1, "max": 2}}, {"Y": "mean"}, "lon"),
        # the whole domain is fine: nothing says the other axis must be cropped
        ({}, MEAN_LON, "lat"),
        (None, MEAN_LON, "lat"),
        # a depth list or "column" keeps a vertical axis standing
        ({**BOX, "depth": [0, 50, 200]}, MEAN_LON, "lat"),
        ({**BOX, "depth": "column"}, MEAN_LON, "lat"),
        # both axes averaged is the joint area-weighted box mean, not a slab
        (BOX, {"lon": "mean", "lat": "mean"}, None),
        # neither axis averaged
        (BOX, {"time": "mean"}, None),
        (BOX, None, None),
        # not a plain mean
        (BOX, {"lon": "max"}, None),
        (BOX, {"lon": {"reduce": "mean", "groupby": "month"}}, None),
        (BOX, {"lon": ["mean", "max"]}, None),
        # the surviving axis is itself being reduced
        (BOX, {"lon": "mean", "lat": "max"}, None),
        # a scalar on either horizontal axis: a line or a point, not a band
        ({"lon": {"min": 180, "max": 200}, "lat": 0.0}, MEAN_LON, None),
        ({"lon": 190.0, "lat": {"min": -30, "max": 30}}, MEAN_LON, None),
        # a transect is a different cut through space
        ({**BOX, "transect": {"lat": 0}}, MEAN_LON, None),
        # no vertical axis can survive
        ({**BOX, "depth": "surface"}, MEAN_LON, None),
        ({**BOX, "depth": 50.0}, MEAN_LON, None),
        ({**BOX, "depth": [50.0]}, MEAN_LON, None),
        ({**BOX, "depth": {"min": 0, "max": 100}}, MEAN_LON, None),
        ({**BOX, "sigma0": 26.5}, MEAN_LON, None),
        (BOX, {"lon": "mean", "depth": "mean"}, None),
    ],
)
def test_slab_axis_truth_table(select, agg, expected):
    assert slab_axis(select, agg) == expected


def test_slab_in_spec_names_the_averaged_key():
    assert slab_in_spec(BOX, {"X": "mean", "time": "mean"}) == ("lat", "X")
    assert slab_in_spec(BOX, {"latitude": "mean"}) == ("lon", "latitude")
    assert slab_in_spec(BOX, {"time": "mean"}) is None


# -- curvilinear lone-axis mean: binned ------------------------------------------------


def _curvilinear(*, nz: int = 2) -> xr.DataArray:
    """A 6x4 grid whose lat steps 1 deg per eta row and lon 1 deg per xi column.

    ``value = 10 * eta + xi`` (+ 100 per level), so every bin mean is hand-computable.
    """
    ny, nx = 6, 4
    lat_1d = np.arange(ny, dtype=float)  # 0..5
    lon_1d = 200.0 + np.arange(nx, dtype=float)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    eta, xi = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    base = 10.0 * eta + xi
    values = np.stack([base + 100.0 * k for k in range(nz)])
    return xr.DataArray(
        values,
        dims=("z", "eta_rho", "xi_rho"),
        coords={
            "z": [-10.0, -20.0][:nz],
            "lon": (("eta_rho", "xi_rho"), lon2d),
            "lat": (("eta_rho", "xi_rho"), lat2d),
        },
        attrs={"units": "degC"},
        name="t",
    )


def test_a_lone_lon_mean_on_a_curvilinear_grid_bins_by_latitude():
    out = aggregate(_curvilinear(), {"lon": "mean"})
    assert set(out.dims) == {"z", "lat"}
    # no region attr: bins span the data extent (0..5) in steps of 1 -> 5 bins
    # (centres 0.5..4.5), the last row (lat == 5) folding into the top bin with row 4
    np.testing.assert_allclose(out["lat"], [0.5, 1.5, 2.5, 3.5, 4.5])
    # row r averages xi = 0..3 -> 10 r + 1.5; the top bin pools rows 4 and 5
    expected = [1.5, 11.5, 21.5, 31.5, 0.5 * ((40 + 1.5) + (50 + 1.5))]
    np.testing.assert_allclose(out.isel(z=0).values, expected)
    np.testing.assert_allclose(out.isel(z=1).values, np.asarray(expected) + 100.0)
    assert out.attrs["units"] == "degC"
    assert out["lat"].attrs["units"] == "degrees_north"


def test_the_bins_follow_the_box_the_select_stamped():
    da = _curvilinear()
    da.attrs["region"] = [200.0, 0.5, 203.0, 4.5]  # lon_min, lat_min, lon_max, lat_max
    out = aggregate(da, {"lon": "mean"})
    # edges 0.5..4.5 in unit steps: centres 1..4 -> rows 1..4 (lat 1,2,3,4 land in
    # bins [0.5,1.5), [1.5,2.5), ...), so each bin holds exactly one row
    np.testing.assert_allclose(out["lat"], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(out.isel(z=0).values, [11.5, 21.5, 31.5, 41.5])


def test_a_lone_lat_mean_on_a_curvilinear_grid_bins_by_longitude():
    out = aggregate(_curvilinear(), {"lat": "mean"})
    assert set(out.dims) == {"z", "lon"}
    assert out.sizes["lon"] == 3
    # each column averages rows 0..5 -> 25 + xi; the top bin pools xi = 2 and 3
    np.testing.assert_allclose(
        out.isel(z=0).values, [25.0, 26.0, 0.5 * (27.0 + 28.0)]
    )


def test_a_curvilinear_mean_keeps_nan_out_of_the_bins():
    da = _curvilinear()
    da[0, 1, :] = np.nan  # one whole row dry at the first level
    out = aggregate(da, {"lon": "mean"})
    assert np.isnan(out.isel(z=0, lat=1))
    assert np.isfinite(out.isel(z=1, lat=1))


def test_only_a_plain_lone_mean_is_binned():
    da = _curvilinear()
    # a different reduction, or both horizontal axes named, keep the old behaviour
    # (the key matches no dimension, so it is skipped) rather than being binned
    assert aggregate(da, {"lon": "max"}).dims == da.dims
    assert aggregate(da, {"lon": "mean", "lat": "max"}).dims == da.dims


def test_a_rectilinear_lone_mean_is_still_the_plain_dimension_mean():
    da = _woa_like()[VAR]
    out = aggregate(da.sel(lon=slice(180, 200)), {"lon": "mean"})
    assert set(out.dims) == {"depth", "lat"}
    np.testing.assert_allclose(
        out.sel(depth=50.0).values, 0.5 + out["lat"].values / 10 + 0.19
    )


# -- Field: rectilinear source ------------------------------------------------------


def _field(name, **kwargs):
    kwargs.setdefault("select", {**BOX, "depth": DEPTHS})
    kwargs.setdefault("aggregate", MEAN_LON)
    return osk.field(name, VAR, cache=False, **kwargs)


def test_a_lon_mean_slab_field_is_a_section_over_latitude(patched_woa):
    f = _field(patched_woa())
    da = f.data
    assert f.family == "section"
    assert set(da.dims) == {"depth", ALONG_DIM}
    np.testing.assert_allclose(da["lat"], np.arange(-30.0, 31.0, 10.0))
    # the averaged axis is a constant at the box midpoint, on a 1-D lon(along)
    assert da["lon"].dims == (ALONG_DIM,)
    np.testing.assert_allclose(da["lon"], 190.0)
    # the values are the lon mean, computed independently in closed form
    np.testing.assert_allclose(
        da.sel(depth=50.0).values, 0.5 + da["lat"].values / 10 + 0.19
    )
    # `along` is great-circle km, with the slab marked on its attrs
    assert da[ALONG_DIM].attrs["units"] == "km"
    assert da[ALONG_DIM].attrs["axis_coord"] == "lat"
    assert da[ALONG_DIM].attrs["band_axis"] == "lon"
    assert da[ALONG_DIM].attrs["band"] == [180.0, 200.0]
    assert float(da[ALONG_DIM][-1]) == pytest.approx(60 * 111.19, rel=0.01)


def test_a_slab_field_with_no_depth_list_keeps_the_observed_levels(patched_woa):
    f = _field(patched_woa(), select=BOX)
    assert f.family == "section"
    assert f.data.sizes["depth"] == 3


def test_a_lat_mean_slab_field_is_a_section_over_longitude(patched_woa):
    f = _field(
        patched_woa(),
        select={"lon": {"min": 160, "max": 220}, "lat": {"min": -10, "max": 10}},
        aggregate=MEAN_LAT,
    )
    da = f.data
    assert f.family == "section"
    np.testing.assert_allclose(da["lon"], np.arange(160.0, 221.0, 10.0))
    np.testing.assert_allclose(da["lat"], 0.0)
    assert da[ALONG_DIM].attrs["axis_coord"] == "lon"
    assert da[ALONG_DIM].attrs["band_axis"] == "lat"
    # mean of lat -10, 0, 10 contributes 0 to value = depth/100 + lat/10 + lon/1000
    np.testing.assert_allclose(
        da.sel(depth=0.0).values, da["lon"].values / 1000, atol=1e-12
    )


def test_a_north_to_south_grid_still_draws_left_to_right(patched_woa):
    f = _field(patched_woa(_woa_like(descending_lat=True)))
    lat = f.data["lat"].values
    assert np.all(np.diff(lat) > 0)
    np.testing.assert_allclose(
        f.data.sel(depth=50.0).values, 0.5 + lat / 10 + 0.19
    )


def test_a_slab_without_a_box_averages_the_whole_domain_and_records_its_extent(
    patched_woa,
):
    f = _field(patched_woa(), select={"depth": DEPTHS})
    assert f.family == "section"
    assert f.data[ALONG_DIM].attrs["band"] == [150.0, 250.0]
    np.testing.assert_allclose(f.data["lon"], 200.0)


def test_a_slab_spec_skips_the_grid_surface_default(patched_woa):
    """A catalogued 3-D grid is surfaced by default -- not when it is a slab."""
    meta = {"featureType": "grid", "vertical_levels": 3}
    f = _field(patched_woa(meta=meta), select=BOX)
    assert f._grid_metadata_if_eligible() is None
    assert f.family == "section"
    # ...whereas the same source with a plain time mean is still eligible
    plain = osk.field("woa_x", VAR, select=BOX, aggregate={"time": "mean"}, cache=False)
    assert plain._grid_metadata_if_eligible() == meta


def test_a_slab_with_a_surviving_extra_axis_is_refused_by_name(monkeypatch):
    ds = _woa_like()
    ds = xr.concat([ds, ds + 1.0], dim="time").assign_coords(time=[0, 1])
    monkeypatch.setattr(osk, "read", lambda n, **kw: ds)
    monkeypatch.setattr(catalog, "resolve", lambda n: SimpleNamespace(metadata={}))
    f = osk.field("woa_x", VAR, select={**BOX, "depth": DEPTHS}, aggregate=MEAN_LON, cache=False)
    with pytest.raises(ValueError, match="time"):
        f.plot()


# -- Field: curvilinear (ROMS) source -------------------------------------------------


@pytest.fixture
def patched_roms(monkeypatch):
    ds = _roms_run()
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    monkeypatch.setattr(osk, "read", lambda n, **kw: ds)
    monkeypatch.setattr(catalog, "resolve", lambda n: SimpleNamespace(metadata=meta))
    return "roms_x"


ROMS_BOX = {"lon": {"min": -95.5, "max": -92.5}, "lat": {"min": 23.5, "max": 28.5}}


def test_a_roms_slab_is_averaged_at_fixed_depths_then_binned(patched_roms):
    f = osk.field(
        patched_roms,
        VAR,
        select={**ROMS_BOX, "depth": [-0.0 + 50.0, 200.0]},
        aggregate=MEAN_LON,
        cache=False,
    )
    da = f.data
    assert f.family == "section"
    assert set(da.dims) == {"z", ALONG_DIM}
    assert da.sizes["z"] == 2
    # lat bins: 1-degree rows 24..28 inside the 23.5..28.5 box
    np.testing.assert_allclose(da["lat"], [24.0, 25.0, 26.0, 27.0, 28.0])
    np.testing.assert_allclose(da["lon"], -94.0)  # the box midpoint
    # temperature is linear in depth (20 + 0.002 z), so where the level exists the
    # lon mean of the fixed-depth values is exactly that, with no depth mixing
    np.testing.assert_allclose(
        da.sel(z=-200.0).dropna(ALONG_DIM).values, 20 - 0.4, atol=0.03
    )
    assert da[ALONG_DIM].attrs["band"] == [-95.5, -92.5]


def test_a_roms_slab_needs_fixed_depths(patched_roms):
    f = osk.field(
        patched_roms, VAR, select=ROMS_BOX, aggregate=MEAN_LON, cache=False
    )
    with pytest.raises(ValueError, match="fixed depths"):
        f.data


def test_a_lone_range_on_a_curvilinear_grid_is_refused(patched_roms):
    f = osk.field(
        patched_roms,
        VAR,
        select={"lon": {"min": -95.5, "max": -92.5}, "depth": [50.0, 200.0]},
        aggregate=MEAN_LON,
        cache=False,
    )
    with pytest.raises(ValueError, match="curvilinear"):
        f.data


# -- Comparison ----------------------------------------------------------------------


def _slab_comparison(patched_sources, **overrides):
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    kwargs = dict(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select={**ROMS_BOX, "depth": [50.0, 200.0]},
        aggregate=MEAN_LON,
        cache=False,
    )
    kwargs.update(overrides)
    return osk.Comparison(**kwargs)


def test_a_slab_comparison_is_a_section_row_that_keeps_its_depths(patched_sources):
    c = _slab_comparison(patched_sources)
    aligned = c.align()
    assert c.family == "section_row"
    assert c.is_section
    # depth survived (not surfaced) on both lanes, and the pair is on one lat axis
    assert aligned.sizes["z"] == 2
    assert set(aligned["test"].dims) == {"z", ALONG_DIM}
    np.testing.assert_allclose(aligned["lon"], aligned["lon"].values[0])
    # the slab marking survives alignment, binning and all, onto every lane
    assert aligned[ALONG_DIM].attrs["axis_coord"] == "lat"
    assert aligned["test"][ALONG_DIM].attrs["band_axis"] == "lon"


def test_both_lanes_share_the_collapsed_constant(patched_sources):
    from ocean_skill.align import SECTION_VERTICAL_DIMS

    c = _slab_comparison(patched_sources)
    t, _ = c._prepare_lane(
        "roms_test", False, False, role="test", keep=SECTION_VERTICAL_DIMS
    )
    r, _ = c._prepare_lane(
        "woa_ref", False, False, role="reference", keep=SECTION_VERTICAL_DIMS
    )
    np.testing.assert_allclose(t["lon"], -94.0)
    np.testing.assert_allclose(r["lon"], -94.0)


def test_a_slab_comparison_round_trips_the_cache(patched_sources):
    c = _slab_comparison(patched_sources, cache=True)
    first = c.align()
    again = _slab_comparison(patched_sources, cache=True)
    again.align()
    assert again.family == "section_row"
    assert again.aligned[ALONG_DIM].attrs["axis_coord"] == "lat"
    np.testing.assert_allclose(again.aligned["test"].values, first["test"].values)


def test_a_slab_comparison_needs_a_depth_list(patched_sources):
    # no vertical named at all would be a slab only at spec level; a scalar or a
    # one-level list leave no vertical axis, so those are not slabs and fall to the
    # ordinary map comparison's own handling -- only the unnamed case is refused here
    with pytest.raises(ValueError, match="explicit list of at least 2"):
        _slab_comparison(patched_sources, select=ROMS_BOX)


def test_the_two_lanes_must_be_the_same_slab(patched_sources):
    with pytest.raises(ValueError, match="same slab"):
        _slab_comparison(
            patched_sources,
            aggregate={"test": {"lon": "mean"}, "reference": {"lat": "mean"}},
        )


def test_over_is_refused_for_a_slab(patched_sources):
    with pytest.raises(ValueError, match="follow-up"):
        _slab_comparison(patched_sources, over="time")


def test_compare_refuses_depths_and_a_missing_depth_list_for_a_slab(patched_sources):
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    common = dict(test="roms_test", reference="woa_ref", variables=[VAR], cache=False)
    with pytest.raises(ValueError, match="explicit depth list"):
        osk.compare(select=ROMS_BOX, aggregate=MEAN_LON, **common)
    with pytest.raises(ValueError, match="depths="):
        osk.compare(
            select={**ROMS_BOX, "depth": [50.0, 200.0]},
            aggregate=MEAN_LON,
            depths=[50.0, 200.0],
            **common,
        )


def test_compare_builds_a_slab_section_row(patched_sources):
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    out = osk.compare(
        test="roms_test",
        reference="woa_ref",
        variables=[VAR],
        select={**ROMS_BOX, "depth": [50.0, 200.0]},
        aggregate=MEAN_LON,
        cache=False,
    )
    assert len(out) == 1
    assert out[0].family == "section_row"
    assert out[0].aligned.sizes["z"] == 2


def test_a_gridded_test_lane_aligns_against_a_gridded_reference(monkeypatch):
    """Both lanes observational: the test's positive-down depth is brought onto z."""
    test_ds = _woa_like()
    ref_ds = _woa_like()
    ref_ds[VAR] = ref_ds[VAR] + 0.5
    sources = {"test_grid": test_ds, "ref_grid": ref_ds}
    monkeypatch.setattr(osk, "read", lambda n, **kw: sources[n])
    monkeypatch.setattr(catalog, "resolve", lambda n: SimpleNamespace(metadata={}))
    c = osk.Comparison(
        test="test_grid",
        reference="ref_grid",
        variable=VAR,
        select={**BOX, "depth": DEPTHS},
        aggregate=MEAN_LON,
        cache=False,
    )
    aligned = c.align()
    assert c.family == "section_row"
    assert set(aligned["test"].dims) == {"z", ALONG_DIM}
    np.testing.assert_allclose(aligned["z"], [0.0, -50.0, -200.0])
    np.testing.assert_allclose(aligned["difference"], -0.5)


# -- the section plot -------------------------------------------------------------------


def test_a_slab_section_is_drawn_against_latitude(patched_woa):
    from ocean_skill.plot.section import prepare_section

    f = _field(patched_woa())
    field, geometry = prepare_section(f.data)
    assert geometry.x_label == "latitude (°N)"
    assert geometry.x_axis == "lat"
    assert geometry.path_note == "mean over 180–200°E"
    np.testing.assert_allclose(
        field[geometry.x_name].isel(depth=0).values, np.arange(-30.0, 31.0, 10.0)
    )


def test_a_lat_slab_is_labelled_by_longitude(patched_woa):
    from ocean_skill.plot.section import prepare_section

    f = _field(
        patched_woa(),
        select={"lon": {"min": 160, "max": 220}, "lat": {"min": -10, "max": 5}},
        aggregate=MEAN_LAT,
    )
    _, geometry = prepare_section(f.data)
    assert geometry.x_label == "longitude (°E)"
    assert geometry.path_note == "mean over 10°S–5°N"


def test_an_equatorial_transect_section_reads_longitude(patched_woa):
    """A lat=0 transect runs east-west, so ``"auto"`` labels it by longitude."""
    from ocean_skill.plot.section import prepare_section

    f = osk.field(
        patched_woa(),
        VAR,
        select={"transect": {"lat": 0, "lon": {"min": 160, "max": 220}}},
        cache=False,
    )
    field, geometry = prepare_section(f.data)
    assert geometry.x_label == "longitude (°E)"
    assert geometry.x_axis == "lon"
    assert field["distance"].attrs["units"] == "degrees_east"
    x = np.unique(field["distance"].values)
    assert x.min() == pytest.approx(160, abs=2)
    assert x.max() == pytest.approx(220, abs=2)

    # forcing distance gives the old kilometres reading back
    _, geometry = prepare_section(f.data, x="distance")
    assert geometry.x_label == "distance along transect (km)"
    assert geometry.x_axis == "distance"


def test_an_observational_positive_down_depth_prepares_as_a_section(patched_woa):
    """WOA's own ``depth`` axis (positive-down) is used as given, not refused."""
    from ocean_skill.plot.section import prepare_section

    f = osk.field(
        patched_woa(),
        VAR,
        select={"transect": {"lat": 0, "lon": {"min": 160, "max": 220}}},
        cache=False,
    )
    assert "depth" in f.data.dims
    field, geometry = prepare_section(f.data)
    np.testing.assert_allclose(
        np.unique(field["depth"].values), [0.0, 50.0, 200.0]
    )  # positive-down, not negated
    assert geometry.y_label == "depth (m)"


def test_a_positive_down_z_is_still_refused():
    from ocean_skill.plot.section import prepare_section

    da = xr.DataArray(
        np.ones((2, 3)),
        dims=("z", ALONG_DIM),
        coords={
            "z": [50.0, 200.0],
            ALONG_DIM: ("along", [0.0, 1.0, 2.0], {"units": "km"}),
            "lon": (ALONG_DIM, [0.0, 0.1, 0.2]),
            "lat": (ALONG_DIM, [0.0, 0.0, 0.0]),
        },
    )
    with pytest.raises(ValueError, match="negative-down"):
        prepare_section(da)


def test_a_negative_down_observational_axis_is_negated_to_positive_depth():
    from ocean_skill.plot.section import prepare_section

    da = xr.DataArray(
        np.ones((2, 3)),
        dims=("depth", ALONG_DIM),
        coords={
            "depth": ("depth", [-50.0, -200.0], {"positive": "up"}),
            ALONG_DIM: ("along", [0.0, 1.0, 2.0], {"units": "km"}),
            "lon": (ALONG_DIM, [0.0, 0.1, 0.2]),
            "lat": (ALONG_DIM, [0.0, 0.0, 0.0]),
        },
    )
    field, _ = prepare_section(da)
    np.testing.assert_allclose(np.unique(field["depth"].values), [50.0, 200.0])


def test_a_slab_section_draws_on_both_renderers(patched_woa, tmp_path):
    f = _field(patched_woa())
    f.plot(renderer="matplotlib", save=str(tmp_path / "slab.png"))
    assert (tmp_path / "slab.png").exists()
    f.plot(renderer="holoviews")


# -- a lon-less, single-longitude reference (a pre-averaged section) ------------------

EAST = "eastward_sea_water_velocity"
REF_BOX = {"lon": {"min": 180, "max": 200}, "lat": {"min": -4, "max": 4}}
REF_DEPTHS = [50.0, 100.0, 200.0]
SLAB_AGG = {"test": {"time": "mean", "lon": "mean"}, "reference": {"lon": "mean"}}


def _east_test_lane(*, units="m s-1", scale=0.01) -> xr.Dataset:
    """A gridded (time, depth, lat, lon) eastward velocity, 0-360, with time to average.

    ``value = scale * (depth + lat + 0.25 * time_index)``: constant in lon, so the lon
    mean is the value itself, and the time mean adds ``0.25 * mean(time_index)``.
    """
    time = np.arange(3)
    depth = np.array(REF_DEPTHS)
    lat = np.arange(-4.0, 5.0, 2.0)
    lon = np.arange(170.0, 211.0, 10.0)
    tt, dd, la, _ = np.meshgrid(time, depth, lat, lon, indexing="ij")
    return xr.DataArray(
        scale * (dd + la + 0.25 * tt),
        dims=("time", "depth", "lat", "lon"),
        coords={
            "time": np.datetime64("2000-01-01") + time.astype("timedelta64[D]"),
            "depth": ("depth", depth, {"positive": "down", "units": "m"}),
            "lat": lat,
            "lon": lon,
        },
        name=EAST,
        attrs={"units": units, "standard_name": EAST},
    ).to_dataset()


def _lonless_reference(
    *, lon=190.0, units="m s-1", scale=0.01, lat=None, depth=None, depth_name="depth"
) -> xr.Dataset:
    """A bare (depth, lat) section, no time and no lon, then given its one longitude.

    The same ``expand_dims`` the catalog's reader chain does for a Cravatte et al. file.
    ``value = scale * (depth + lat + 0.5)`` is checked against its own column.
    ``depth_name`` spells the vertical axis some other way, as a product's own file may.
    """
    depth = np.array(REF_DEPTHS if depth is None else depth, dtype="float64")
    lat = np.arange(-4.0, 5.0, 2.0) if lat is None else np.asarray(lat, "float64")
    da = xr.DataArray(
        scale * (depth[:, None] + lat[None, :] + 0.5),
        dims=(depth_name, "lat"),
        coords={
            depth_name: (depth_name, depth, {"positive": "down", "units": "m"}),
            "lat": lat,
        },
        name=EAST,
        attrs={"units": units, "standard_name": EAST},
    )
    return da.to_dataset().expand_dims(lon=[lon])


def _lonless_comparison(monkeypatch, reference, *, test=None, **overrides):
    sources = {
        "test_grid": _east_test_lane() if test is None else test,
        "ref_section": reference,
    }
    monkeypatch.setattr(osk, "read", lambda n, **kw: sources[n])
    monkeypatch.setattr(catalog, "resolve", lambda n: SimpleNamespace(metadata={}))
    kwargs = dict(
        test="test_grid",
        reference="ref_section",
        variable=EAST,
        select={**REF_BOX, "depth": REF_DEPTHS},
        aggregate=SLAB_AGG,
        cache=False,
    )
    kwargs.update(overrides)
    return osk.Comparison(**kwargs)


def test_a_lonless_single_longitude_reference_is_a_slab_comparison(monkeypatch):
    """A pre-averaged (depth, lat) section carrying one longitude joins a lon-mean slab.

    The reference has no time and its lon is a length-1 axis, so ``{"lon": "mean"}`` on
    it is the identity; the band the section records is the box, not the one longitude.
    """
    c = _lonless_comparison(monkeypatch, _lonless_reference())
    aligned = c.align()

    assert c.family == "section_row"
    assert set(aligned["reference"].dims) == {"z", ALONG_DIM}
    assert aligned[ALONG_DIM].attrs["axis_coord"] == "lat"
    assert aligned[ALONG_DIM].attrs["band_axis"] == "lon"
    assert aligned[ALONG_DIM].attrs["band"] == [180.0, 200.0]
    np.testing.assert_allclose(aligned["lat"], np.arange(-4.0, 5.0, 2.0))
    np.testing.assert_allclose(aligned["z"], [-50.0, -100.0, -200.0])

    # the reference's own column, untouched: scale * (depth + lat + 0.5)
    expected = 0.01 * (
        np.array(REF_DEPTHS)[:, None] + np.arange(-4.0, 5.0, 2.0)[None, :] + 0.5
    )
    np.testing.assert_allclose(aligned["reference"].values, expected)
    # and the test lane is its time- and lon-mean: scale * (depth + lat + 0.25)
    np.testing.assert_allclose(aligned["test"].values, expected - 0.01 * 0.25)


def test_a_box_that_excludes_the_references_one_longitude_is_refused(monkeypatch):
    """The test lane has cells in lon 200-240; the reference's single 190 is outside.

    The refusal names the lane that came up empty (the reference) and the box, rather
    than producing an empty or all-NaN section.
    """
    from ocean_skill.align import EmptySelection

    c = _lonless_comparison(
        monkeypatch,
        _lonless_reference(lon=190.0),
        select={
            "lon": {"min": 200, "max": 240},
            "lat": REF_BOX["lat"],
            "depth": REF_DEPTHS,
        },
    )
    with pytest.raises(EmptySelection, match=r"ref_section"):
        c.align()


def test_a_catalogued_cm_per_s_reference_is_converted_against_an_m_per_s_test(
    isolated_catalogs, tmp_path
):
    """The catalog's ``units`` map is what lets a ``unit: cm/s`` file compare at all.

    The Cravatte-style reference only spells its units ``unit``, so without the map the
    pair would be differenced unchecked; with it, the lanes are brought to a common unit
    (the test lane's m s-1 goes onto the reference's cm/s), so the aligned values and
    the difference are in cm/s.
    """
    from ocean_skill.build import build_catalog
    from tests.test_build_helpers import _build_cravatte

    _build_cravatte(isolated_catalogs, tmp_path)
    test_path = tmp_path / "test_grid.nc"
    _east_test_lane(scale=0.01).to_netcdf(test_path, format="NETCDF3_CLASSIC")
    build_catalog(
        {"test_grid": str(test_path)},
        isolated_catalogs / "test_grid.yaml",
        name_map=None,
        reader_kwargs={"engine": "scipy"},
    )
    c = osk.Comparison(
        test="test_grid",
        reference="cravatte",
        variable=EAST,
        select={**REF_BOX, "depth": REF_DEPTHS},
        aggregate=SLAB_AGG,
        cache=False,
    )
    aligned = c.align()
    assert c.family == "section_row"
    assert aligned["reference"].attrs["units"] == "cm/s"
    # the reference's own numbers, as the file has them: depth + lat + 0.5 (cm/s)
    expected = np.array(REF_DEPTHS)[:, None] + np.arange(-4.0, 5.0, 2.0)[None, :] + 0.5
    np.testing.assert_allclose(aligned["reference"].values, expected)
    # the test lane is 0.01 * (depth + lat + 0.25) m/s = depth + lat + 0.25 cm/s
    np.testing.assert_allclose(aligned["test"].values, expected - 0.25, atol=1e-9)
    np.testing.assert_allclose(aligned["difference"].values, -0.25, atol=1e-9)


def test_a_reference_with_its_own_vertical_axis_name_is_a_slab_section(monkeypatch):
    """``DEPTH`` is found as the vertical by axis detection, then put on ``depth``."""
    c = _lonless_comparison(monkeypatch, _lonless_reference(depth_name="DEPTH"))
    aligned = c.align()
    assert c.family == "section_row"
    np.testing.assert_allclose(aligned["z"], [-50.0, -100.0, -200.0])
    assert set(aligned["reference"].dims) == {"z", ALONG_DIM}


def _roms_run_with_velocity(*, angle: float, v: float) -> xr.Dataset:
    """:func:`_roms_run` plus the raw staggered velocity and a grid ``angle``.

    The post-``standardize`` state a catalogued ROMS source reads back in: the
    grid-relative components on their own ``xi_u``/``eta_v`` dims, ``angle`` a
    coordinate, and no ``eastward_sea_water_velocity`` yet -- that is derived on
    demand once a request names it. Both components are spatially uniform, so the
    rotated east velocity is a closed form: ``u cos(angle) - v sin(angle)``.
    """
    ds = _roms_run()
    ny, nx = ds["h"].shape
    u = np.full((N_S, ny, nx - 1), 0.0)
    vv = np.full((N_S, ny - 1, nx), v)
    return ds.assign(
        sea_water_x_velocity=(("s_rho", "eta_rho", "xi_u"), u),
        sea_water_y_velocity=(("s_rho", "eta_v", "xi_rho"), vv),
    ).assign_coords(angle=(("eta_rho", "xi_rho"), np.full((ny, nx), angle)))


def test_a_roms_east_velocity_slab_is_derived_on_demand_and_compared(monkeypatch):
    """The model lane derives east from u/v/angle inside the slab path.

    The grid's x axis points true north (``angle = pi/2``) and ``v = -0.2``, so east is
    ``-v = +0.2`` m/s at every cell and depth; with the fixed-depth interpolation and
    the lat binning in between, that has to come through unchanged. The reference is a
    pre-averaged single-longitude section at the box's own longitude.
    """
    test = _roms_run_with_velocity(angle=np.pi / 2, v=-0.2)
    assert EAST not in test  # nothing pre-derived: the slab path has to do it
    ref = _lonless_reference(
        lon=-94.0, lat=[24.0, 25.0, 26.0, 27.0, 28.0], depth=[50.0, 200.0]
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    sources = {"roms_test": (test, meta), "ref_section": (ref, {})}
    monkeypatch.setattr(osk, "read", lambda n, **kw: sources[n][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda n: SimpleNamespace(metadata=sources[n][1])
    )
    c = osk.Comparison(
        test="roms_test",
        reference="ref_section",
        variable=EAST,
        select={**ROMS_BOX, "depth": [50.0, 200.0]},
        aggregate=MEAN_LON,
        cache=False,
    )
    aligned = c.align()
    assert c.family == "section_row"
    assert aligned[ALONG_DIM].attrs["axis_coord"] == "lat"
    assert aligned[ALONG_DIM].attrs["band"] == [-95.5, -92.5]
    got = aligned["test"].values
    assert np.isfinite(got).any()  # the shallowest columns are below the 50 m level
    np.testing.assert_allclose(got[np.isfinite(got)], 0.2, atol=1e-9)
    # the reference's own numbers: 0.01 * (depth + lat + 0.5)
    np.testing.assert_allclose(
        aligned["reference"].sel(z=-50.0).values,
        0.01 * (50.0 + aligned["lat"].values + 0.5),
    )
