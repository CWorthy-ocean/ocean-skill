"""A horizontal aggregate on the test lane must not shrink the reference's crop.

:meth:`ocean_skill.comparison.Comparison.align` crops the reference lane to the test
lane's horizontal extent before the reference is read in full. When the test's own
aggregate reduces lon and/or lat (a domain mean), the *prepared* test lane is already
one centroid, and cropping the reference to that left its "domain mean" the value
near the middle of the region -- a warm-tropics/cold-poles field averaged at its warm
centre. The crop now comes from the test's extent after its select and before its
aggregate (:meth:`Comparison._test_extent_before_aggregation`).

Layers: the helper's extent against hand-built grids (rectilinear, curvilinear,
antimeridian, box select), then the end-to-end pair -- identical sources agree on a
domain mean, a larger global reference is averaged over the model's region (not the
centroid, not the globe), and a time mean is unchanged.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog, comparison, roms
from ocean_skill.comparison import _reduces_horizontal

VAR = "sea_water_potential_temperature"
DOMAIN_MEAN = {"lon": "mean", "lat": "mean"}


def _field(lon, lat):
    """Warm tropics, cold poles, warmer to the east: a function of absolute position."""
    return 28.0 - 0.004 * lat**2 + 0.01 * lon


def _rectilinear(lon0, lon1, lat0, lat1, step, *, nt=2) -> xr.Dataset:
    lon = np.arange(lon0, lon1 + 1e-9, step)
    lat = np.arange(lat0, lat1 + 1e-9, step)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    data = np.broadcast_to(_field(lo, la), (nt, *la.shape)).copy()
    return xr.Dataset(
        {VAR: (("time", "lat", "lon"), data, {"units": "degC", "standard_name": VAR})},
        coords={
            "time": np.array(["2012-01-01", "2012-02-01"], dtype="datetime64[ns]")[:nt],
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
    )


def _roms_run(lon0, lon1, *, nt=2, ns=4, ny=60, nx=120):
    """Build a curvilinear ROMS run over ``lon0..lon1``, 54S-66N."""
    sig = (np.arange(ns) - ns + 0.5) / ns
    sigw = np.linspace(-1, 0, ns + 1)
    lon = np.tile(np.linspace(lon0, lon1, nx), (ny, 1))
    lat = np.tile(np.linspace(-54, 66, ny)[:, None], (1, nx))
    temp = np.broadcast_to(_field(lon, lat), (nt, ns, ny, nx)).copy()
    raw = xr.Dataset(
        {
            "zeta": (
                ("time", "eta_rho", "xi_rho"),
                np.zeros((nt, ny, nx)),
                {"units": "m"},
            ),
            "temp": (("time", "s_rho", "eta_rho", "xi_rho"), temp, {"units": "degC"}),
            "h": (("eta_rho", "xi_rho"), np.full((ny, nx), 4000.0)),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((ny, nx))),
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "Cs_r": ("s_rho", sig),
            "sigma_r": ("s_rho", sig),
            "Cs_w": ("s_w", sigw),
            "sigma_w": ("s_w", sigw),
            "ocean_time": ("time", np.arange(nt) * 86400.0 * 30),
        }
    )
    meta = {
        "model": "roms",
        "self_contained_grid": True,
        "vertical": {"s_dim": "s_rho", "hc": 0.0, "Vtransform": 2},
        "reference_date": "2011-01-01",
    }
    return roms.standardize(raw, meta), meta


@pytest.fixture
def patched(monkeypatch):
    """Register ``{name: Dataset | (Dataset, metadata)}`` as readable sources."""

    def install(sources):
        pairs = {n: v if isinstance(v, tuple) else (v, {}) for n, v in sources.items()}
        monkeypatch.setattr(osk, "read", lambda n, **kw: pairs[n][0])
        import ocean_skill.sources as sources_module

        monkeypatch.setattr(sources_module, "read", lambda n, **kw: pairs[n][0])
        monkeypatch.setattr(
            catalog, "resolve", lambda n: SimpleNamespace(metadata=pairs[n][1])
        )
        monkeypatch.setattr(comparison, "_domain_of", lambda n: None)
        monkeypatch.setattr(comparison, "_outline_of", lambda n, convention=None: None)

    return install


def _metrics(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = osk.Comparison(cache=False, **kwargs)
        return c.metrics()


# -- the spec-level decision ---------------------------------------------------------


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        (DOMAIN_MEAN, True),
        ({"lon": "mean"}, True),
        ({"latitude": {"reduce": "max"}}, True),
        ({"lat": [{"reduce": "mean"}]}, True),
        ({"time": "mean"}, False),
        ({"time": "mean", "depth": "mean"}, False),
        ({}, False),
        (None, False),
    ],
)
def test_reduces_horizontal_truth_table(agg, expected):
    assert _reduces_horizontal(agg) is expected


# -- the extent the reference is cropped to -------------------------------------------


def _comparison(patched, sources, **overrides):
    patched(sources)
    kwargs = dict(
        test="a", reference="b", variable=VAR, aggregate=DOMAIN_MEAN, cache=False
    )
    kwargs.update(overrides)
    return osk.Comparison(**kwargs)


def test_the_extent_is_the_whole_grid_for_a_rectilinear_domain_mean(patched):
    c = _comparison(
        patched,
        {
            "a": _rectilinear(130, 250, -50, 60, 2),
            "b": _rectilinear(0, 360, -80, 80, 1),
        },
    )
    assert c._test_extent_before_aggregation() == (130.0, -50.0, 250.0, 60.0)


def test_the_extent_is_the_whole_grid_for_a_curvilinear_one(patched):
    run, meta = _roms_run(77.0, 316.0)
    c = _comparison(patched, {"a": (run, meta), "b": (run, meta)})
    lo, la, hi, ha = c._test_extent_before_aggregation()
    assert (lo, la, hi, ha) == pytest.approx((77.0, -54.0, 316.0, 66.0))


def test_a_box_select_narrows_the_extent_the_way_it_narrows_the_data(patched):
    box = {"lon": {"min": 150, "max": 200}, "lat": {"min": -20, "max": 30}}
    c = _comparison(
        patched,
        {
            "a": _rectilinear(130, 250, -50, 60, 2),
            "b": _rectilinear(0, 360, -80, 80, 1),
        },
        select=box,
    )
    assert c._test_extent_before_aggregation() == (150.0, -20.0, 200.0, 30.0)


def test_a_pm180_grid_across_the_dateline_is_re_expressed_not_spanning_the_globe(
    patched,
):
    # 170E..170W stored as 170..180 then -180..-170: nanmin/nanmax say -180..180.
    lon = np.r_[np.arange(170.0, 180.01, 2.0), np.arange(-180.0, -169.99, 2.0)]
    ds = _rectilinear(0, 1, -10, 10, 2).isel(lon=slice(0, 1)).drop_vars("lon")
    ds = xr.Dataset(
        {VAR: (("time", "lat", "lon"), np.zeros((2, 11, lon.size)), {"units": "degC"})},
        coords={
            "time": ds["time"],
            "lat": ("lat", np.arange(-10.0, 10.01, 2.0), {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
    )
    c = _comparison(patched, {"a": ds, "b": ds})
    lo, la, hi, ha = c._test_extent_before_aggregation()
    assert (lo, hi) == (170.0, 190.0)
    assert (la, ha) == (-10.0, 10.0)


def test_a_source_with_no_gridded_coordinates_has_no_extent(patched):
    ds = xr.Dataset(
        {VAR: ("time", np.zeros(3), {"units": "degC"})},
        coords={"time": np.array(["2012-01-01"] * 3, dtype="datetime64[ns]")},
    )
    c = _comparison(patched, {"a": ds, "b": ds}, aggregate={"time": "mean"})
    assert c._test_extent_before_aggregation() is None


# -- end to end: the same domain mean on both sides ---------------------------------


@pytest.mark.parametrize(
    ("lon0", "lon1"), [(130.0, 250.0), (10.0, 130.0), (77.0, 316.0)]
)
def test_two_identical_roms_runs_agree_on_a_domain_mean(patched, lon0, lon1):
    """The PACMED shape: same grid twice, across the dateline or not."""
    run_a, meta = _roms_run(lon0, lon1)
    run_b, _ = _roms_run(lon0, lon1)
    patched({"a": (run_a, meta), "b": (run_b, meta)})
    m = _metrics(
        test="a",
        reference="b",
        variable="temperature",
        select={"depth": "surface"},
        aggregate=DOMAIN_MEAN,
    )
    assert m["mean_test"] == pytest.approx(m["mean_reference"])
    assert m["bias"] == pytest.approx(0.0, abs=1e-9)
    # The domain mean really is a domain mean: it is the cold-poles average, not the
    # warm value at the centre of the grid (28 + 0.01 * lon at the equator).
    centre = _field(0.5 * (lon0 + lon1), 6.0)
    assert m["mean_test"] < centre - 1.0


def test_two_identical_rectilinear_sources_agree_on_a_domain_mean(patched):
    ds = _rectilinear(130, 250, -50, 60, 2)
    patched({"a": ds, "b": ds.copy(deep=True)})
    m = _metrics(test="a", reference="b", variable=VAR, aggregate=DOMAIN_MEAN)
    assert m["bias"] == pytest.approx(0.0, abs=1e-9)
    assert m["mean_test"] == pytest.approx(m["mean_reference"])


def test_a_box_select_domain_mean_covers_the_same_box_on_both_lanes(patched):
    box = {"lon": {"min": 150, "max": 200}, "lat": {"min": -20, "max": 30}}
    ds = _rectilinear(130, 250, -50, 60, 2)
    patched({"a": ds, "b": ds.copy(deep=True)})
    m = _metrics(
        test="a", reference="b", variable=VAR, aggregate=DOMAIN_MEAN, select=box
    )
    assert m["bias"] == pytest.approx(0.0, abs=1e-9)
    # ...and it is the box's own mean, not the centroid's value or the domain's.
    lon, lat = np.meshgrid(np.arange(150.0, 200.01, 2), np.arange(-20.0, 30.01, 2))
    weights = np.cos(np.deg2rad(lat))  # the joint domain mean is area-weighted
    assert m["mean_reference"] == pytest.approx(
        float(np.average(_field(lon, lat), weights=weights))
    )


def _area_mean(lon0, lon1, lat0, lat1, step):
    """Return the area-weighted mean over a box: what a joint domain mean is."""
    lon, lat = np.meshgrid(
        np.arange(lon0, lon1 + 1e-9, step), np.arange(lat0, lat1 + 1e-9, step)
    )
    return float(np.average(_field(lon, lat), weights=np.cos(np.deg2rad(lat))))


def test_a_larger_global_reference_is_averaged_over_the_models_region(patched):
    """A regional test vs a global reference: not the centroid, not the globe."""
    patched(
        {"a": _rectilinear(130, 250, -50, 60, 2), "b": _rectilinear(0, 360, -80, 80, 1)}
    )
    m = _metrics(test="a", reference="b", variable=VAR, aggregate=DOMAIN_MEAN)
    # The truth: the reference's own cells over (about) the model's region, give or
    # take the one-degree pad the crop keeps around the test's extent.
    region = _area_mean(130, 250, -50, 60, 1)
    centroid = float(_field(190.0, 5.0))
    globe = _area_mean(0, 360, -80, 80, 1)
    assert abs(centroid - region) > 1.0 and abs(globe - region) > 1.0
    assert m["mean_reference"] == pytest.approx(region, abs=0.15)
    assert m["mean_test"] == pytest.approx(_area_mean(130, 250, -50, 60, 2))


def test_a_time_mean_is_unchanged(patched):
    ds = _rectilinear(130, 250, -50, 60, 2)
    patched({"a": ds, "b": ds.copy(deep=True)})
    m = _metrics(test="a", reference="b", variable=VAR, aggregate={"time": "mean"})
    assert m["bias"] == pytest.approx(0.0, abs=1e-9)
    assert m["n"] > 1


def test_a_lone_lon_mean_slab_still_compares_clean(patched):
    """A band mean is a slab (a section lane); its reference was never cropped."""
    ds = _rectilinear(130, 250, -50, 60, 2)
    ds = ds.expand_dims(depth=[0.0, 50.0], axis=1).assign_coords(
        depth=("depth", [0.0, 50.0], {"positive": "down", "units": "m"})
    )
    ds[VAR].attrs.update({"units": "degC", "standard_name": VAR})
    patched({"a": ds, "b": ds.copy(deep=True)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = osk.Comparison(
            test="a",
            reference="b",
            variable=VAR,
            select={"depth": [0.0, 50.0], "time": "2012-01-01"},
            aggregate={"lon": "mean"},
            cache=False,
        )
        aligned = c.align()
    np.testing.assert_allclose(aligned["difference"], 0.0, atol=1e-9)


# -- the cache ----------------------------------------------------------------------


def test_a_horizontal_aggregate_carries_a_crop_marker_in_the_aligned_pair_key(
    patched, monkeypatch
):
    """An entry cached by the centroid-cropped build must not be served."""
    patched(
        {"a": _rectilinear(130, 250, -50, 60, 2), "b": _rectilinear(0, 360, -80, 80, 1)}
    )

    def key():
        return osk.Comparison(
            test="a", reference="b", variable=VAR, aggregate=DOMAIN_MEAN, cache=False
        )._cache_key

    with_marker = key()
    monkeypatch.setattr(comparison, "_reduces_horizontal", lambda agg: False)
    assert key() != with_marker
    # A time mean carries no marker, so its entries are untouched.
    monkeypatch.undo()
    patched(
        {"a": _rectilinear(130, 250, -50, 60, 2), "b": _rectilinear(0, 360, -80, 80, 1)}
    )
    time_key = osk.Comparison(
        test="a", reference="b", variable=VAR, aggregate={"time": "mean"}, cache=False
    )._cache_key
    monkeypatch.setattr(comparison, "_reduces_horizontal", lambda agg: False)
    assert (
        osk.Comparison(
            test="a",
            reference="b",
            variable=VAR,
            aggregate={"time": "mean"},
            cache=False,
        )._cache_key
        == time_key
    )
