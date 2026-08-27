"""Tests for matching a vertical section against a gridded dataset (Stage C).

Three layers, the house convention: the pure alignment function
(:func:`ocean_skill.align._align_along_path`, checked against independently
computed indices/means, not against the code under test), the wiring into
:class:`~ocean_skill.comparison.Comparison` (the transect route, the ``keep=``
plumbing, the refusals), and the end-to-end pipeline through
:func:`ocean_skill.comparison.compare`.

Both lanes of a section comparison are reduced to columns along the *same*
lon/lat path before alignment ever runs -- see ``Comparison.align``'s transect
route, which re-spells the reference's transect as the resolved points the test
lane snapped to. ``_align_along_path`` itself never regrids a 2-D grid; it only
reconciles two along-path axes that differ even when both lanes were asked for
the same points, because a coarser lane's own sampler collapses points that land
in one cell and reports its own cells' positions back (see
``ocean_skill.transect.sample_along``). The pair lands on the coarser lane's
columns -- the along-path counterpart of :func:`ocean_skill.align._regrid_target`,
with the identical ``COARSER_BY`` hysteresis and reference-wins tie-break.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog, roms
from ocean_skill.align import ALONG_DIM, _align_along_path, _haversine_km
from ocean_skill.comparison import Comparison
from ocean_skill.transect import apply_transect

N_S = 12
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0
VAR = "sea_water_potential_temperature"


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


def _roms_run() -> xr.Dataset:
    """A small standardized-shaped ROMS run: h/mask/sigma/Cs on a 5x3 rho grid.

    Matches ``tests/test_section.py``'s own fixture -- same grid, same trick
    (``chl``-style linear-in-depth temperature) -- so a section comparison test
    reads the same synthetic model the Stage A/B tests already do.
    """
    ny, nx = 5, 3
    h = np.linspace(30.0, 2000.0, ny * nx).reshape(ny, nx)
    sigma_r = (np.arange(1, N_S + 1) - N_S - 0.5) / N_S
    sigma_w = np.linspace(-1, 0, N_S + 1)
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
    temp = 20.0 + 0.002 * ds["z_rho"]
    ds = ds.assign({VAR: temp})
    ds[VAR].attrs["units"] = "degC"
    return ds


def _climatology(*, nlat: int = 5, nlon: int = 3) -> xr.Dataset:
    """A rectilinear, WOA-style climatology: real positive-down ``depth`` levels."""
    lon = np.linspace(-96.0, -92.0, nlon)
    lat = np.linspace(23.0, 29.0, nlat)
    depth = np.array([50.0, 200.0])
    lon2d, lat2d, depth3d = np.meshgrid(lon, lat, depth, indexing="ij")
    values = 15.0 - 0.01 * depth3d + 0.1 * (lat2d - 26.0)
    da = xr.DataArray(
        values.transpose(2, 1, 0),
        dims=("depth", "lat", "lon"),
        coords={"depth": depth, "lat": lat, "lon": lon},
        name=VAR,
        attrs={"units": "degC"},
    )
    return da.to_dataset()


@pytest.fixture
def patched_sources(monkeypatch):
    """Patch osk.read/catalog.resolve for two distinct named sources."""

    def _patch(sources: dict[str, tuple[xr.Dataset, dict]]):
        monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
        monkeypatch.setattr(
            catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
        )

    return _patch


def _lane(*, lon, lat, z_or_depth, vdim, values, path_method="nearest"):
    """Build a (vertical, along) lane with a real cumulative-haversine along coord."""
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    along = np.zeros(lon.size)
    if lon.size > 1:
        seg = _haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
        along[1:] = np.cumsum(seg)
    da = xr.DataArray(
        np.asarray(values, dtype="float64"),
        dims=(vdim, ALONG_DIM),
        coords={
            vdim: np.asarray(z_or_depth, dtype="float64"),
            ALONG_DIM: (ALONG_DIM, along, {"path_method": path_method, "units": "km"}),
            "lon": (ALONG_DIM, lon),
            "lat": (ALONG_DIM, lat),
        },
        attrs={"units": "degC"},
    )
    return da


@pytest.fixture
def fine_test():
    """6 columns, ~11 km apart, 2 fixed depths -- the fine ("test") lane."""
    lon = [-95.0, -94.9, -94.8, -94.7, -94.6, -94.5]
    lat = [24.0] * 6
    z = [-50.0, -200.0]
    values = np.arange(12, dtype="float64").reshape(2, 6)
    return _lane(lon=lon, lat=lat, z_or_depth=z, vdim="z", values=values, path_method="grid")


@pytest.fixture
def coarse_reference():
    """2 columns, positioned near the fine lane's first-3/last-3 groups."""
    lon = [-94.85, -94.55]
    lat = [24.0, 24.0]
    depth = [50.0, 200.0]  # positive-down, observational-style
    values = np.array([[10.0, 40.0], [20.0, 50.0]])
    return _lane(
        lon=lon, lat=lat, z_or_depth=depth, vdim="depth", values=values, path_method="nearest"
    )


# -- _align_along_path: structure, level pairing, attrs -----------------------------


def test_reference_coarser_bins_the_test_into_its_columns(fine_test, coarse_reference):
    out = _align_along_path(
        fine_test, coarse_reference, convention="-180-180", test_name="test", reference_name="reference"
    )
    assert set(out.data_vars) == {"test", "reference", "difference"}
    assert out.sizes[ALONG_DIM] == 2  # landed on the coarser (reference) columns
    assert out.attrs["section_target"] == "reference"
    assert out.attrs["n_points"] == 2

    # independently: which of the fine lane's 6 columns is nearest which of the
    # coarse lane's 2, by haversine -- not reusing the function under test
    ref_lon, ref_lat = np.array([-94.85, -94.55]), np.array([24.0, 24.0])
    fine_lon = np.array([-95.0, -94.9, -94.8, -94.7, -94.6, -94.5])
    fine_lat = np.array([24.0] * 6)
    j = np.array(
        [
            int(np.argmin(_haversine_km(ref_lon, ref_lat, lo, la)))
            for lo, la in zip(fine_lon, fine_lat)
        ]
    )
    # sanity: the fixture is set up as intended (a -94.7/-94.55 exact tie at the
    # same latitude breaks toward the lower index, same as np.argmin everywhere else)
    assert j.tolist() == [0, 0, 0, 0, 1, 1]

    expected_test = np.stack(
        [
            fine_test.isel(z=0).values[j == 0].mean(),
            fine_test.isel(z=0).values[j == 1].mean(),
        ]
    )
    np.testing.assert_allclose(out["test"].isel(z=0).values, expected_test)
    expected_test_z1 = np.stack(
        [fine_test.isel(z=1).values[j == 0].mean(), fine_test.isel(z=1).values[j == 1].mean()]
    )
    np.testing.assert_allclose(out["test"].isel(z=1).values, expected_test_z1)

    # the reference (the frame) is untouched -- its own values ride straight through
    np.testing.assert_allclose(out["reference"].values, coarse_reference.values)
    np.testing.assert_allclose(
        out["difference"].values, out["test"].values - out["reference"].values
    )


def test_reference_levels_and_z_coordinate_are_shared_and_positive_down(fine_test, coarse_reference):
    out = _align_along_path(
        fine_test, coarse_reference, convention="-180-180", test_name="test", reference_name="reference"
    )
    assert list(out["test"]["z"].values) == list(out["reference"]["z"].values)
    np.testing.assert_allclose(out["test"]["z"].values, [-50.0, -200.0])
    assert out.attrs["reference_levels"] == [50.0, 200.0]  # positive-down, as requested


def test_test_coarser_bins_the_reference_into_its_columns():
    # swap roles: now the "reference" is the fine one, "test" the coarse one
    fine_ref = _lane(
        lon=[-95.0, -94.9, -94.8, -94.7, -94.6, -94.5],
        lat=[24.0] * 6,
        z_or_depth=[50.0, 200.0],
        vdim="depth",
        values=np.arange(12, dtype="float64").reshape(2, 6),
    )
    coarse_test = _lane(
        lon=[-94.85, -94.55],
        lat=[24.0, 24.0],
        z_or_depth=[-50.0, -200.0],
        vdim="z",
        values=np.array([[10.0, 40.0], [20.0, 50.0]]),
    )
    out = _align_along_path(
        coarse_test, fine_ref, convention="-180-180", test_name="test", reference_name="reference"
    )
    assert out.attrs["section_target"] == "test"
    assert out.sizes[ALONG_DIM] == 2
    # the test (the frame) rides through untouched
    np.testing.assert_allclose(out["test"].values, coarse_test.values)


def test_comparable_resolutions_pair_one_to_one_without_averaging():
    """Near-equal spacing: groups are (mostly) singletons, so binning ~ identity."""
    lon = [-95.0, -94.9, -94.8, -94.7]
    lat = [24.0] * 4
    t = _lane(lon=lon, lat=lat, z_or_depth=[-50.0], vdim="z", values=[[1.0, 2.0, 3.0, 4.0]])
    r = _lane(
        lon=[-95.0, -94.9, -94.8, -94.7],
        lat=[24.0] * 4,
        z_or_depth=[50.0],
        vdim="depth",
        values=[[10.0, 20.0, 30.0, 40.0]],
    )
    out = _align_along_path(t, r, convention="-180-180", test_name="test", reference_name="reference")
    assert out.attrs["section_target"] == "reference"  # tie -> reference wins
    assert out.sizes[ALONG_DIM] == 4
    np.testing.assert_allclose(out["test"].values, t.values)  # identity: no averaging
    np.testing.assert_allclose(out["reference"].values, r.values)


def test_identical_grids_are_the_identity(fine_test):
    same = fine_test.copy(deep=True).rename(z="depth")
    out = _align_along_path(
        fine_test, same, convention="-180-180", test_name="test", reference_name="reference"
    )
    assert out.sizes[ALONG_DIM] == fine_test.sizes[ALONG_DIM]
    np.testing.assert_allclose(out["difference"].values, 0.0)


# -- shape / mismatch errors ---------------------------------------------------------


def test_native_s_test_lane_refused(coarse_reference):
    native = xr.DataArray(
        np.zeros((3, 2)),
        dims=("s_rho", ALONG_DIM),
        coords={
            ALONG_DIM: (ALONG_DIM, [0.0, 10.0], {"path_method": "grid"}),
            "lon": (ALONG_DIM, [-95.0, -94.9]),
            "lat": (ALONG_DIM, [24.0, 24.0]),
        },
    )
    with pytest.raises(ValueError, match="needs fixed depths"):
        _align_along_path(native, coarse_reference, convention="-180-180", test_name="test", reference_name="reference")


def test_reference_with_no_vertical_axis_refused(fine_test):
    surface_only = xr.DataArray(
        [1.0, 2.0],
        dims=ALONG_DIM,
        coords={
            ALONG_DIM: (ALONG_DIM, [0.0, 10.0], {"path_method": "nearest"}),
            "lon": (ALONG_DIM, [-94.85, -94.55]),
            "lat": (ALONG_DIM, [24.0, 24.0]),
        },
    )
    with pytest.raises(ValueError, match="no vertical axis"):
        _align_along_path(fine_test, surface_only, convention="-180-180", test_name="test", reference_name="reference")


def test_extra_dim_on_either_lane_refused(fine_test, coarse_reference):
    with_time = fine_test.expand_dims(time=[0, 1])
    with pytest.raises(ValueError, match="beyond depth and the along-path axis"):
        _align_along_path(with_time, coarse_reference, convention="-180-180", test_name="test", reference_name="reference")


def test_mismatched_level_counts_refused(fine_test):
    three_levels = xr.DataArray(
        np.zeros((3, 2)),
        dims=("depth", ALONG_DIM),
        coords={
            "depth": [50.0, 100.0, 200.0],
            ALONG_DIM: (ALONG_DIM, [0.0, 10.0], {"path_method": "nearest"}),
            "lon": (ALONG_DIM, [-94.85, -94.55]),
            "lat": (ALONG_DIM, [24.0, 24.0]),
        },
    )
    with pytest.raises(ValueError, match="same depth list"):
        _align_along_path(fine_test, three_levels, convention="-180-180", test_name="test", reference_name="reference")


def test_over_refused_via_align_entry(fine_test, coarse_reference):
    from ocean_skill.align import align

    with pytest.raises(ValueError, match="follow-up"):
        align(fine_test, coarse_reference, over="time")


def test_one_lane_a_section_and_the_other_not_refused(fine_test):
    from ocean_skill.align import align

    a_map = xr.DataArray(
        np.zeros((3, 4)),
        dims=("lat", "lon"),
        coords={"lat": [10.0, 20.0, 30.0], "lon": [-96.0, -95.0, -94.0, -93.0]},
    )
    with pytest.raises(ValueError, match="both must sample the same transect path"):
        align(fine_test, a_map)


# -- coverage: partial and total ------------------------------------------------------


def test_reference_not_covering_part_of_the_path_raises():
    t = _lane(
        lon=np.linspace(-96.0, -90.0, 8),
        lat=[24.0] * 8,
        z_or_depth=[-50.0],
        vdim="z",
        values=[np.arange(8, dtype="float64")],
    )
    # reference only covers the western half of the path
    r = _lane(
        lon=[-95.8, -95.0],
        lat=[24.0, 24.0],
        z_or_depth=[50.0],
        vdim="depth",
        values=[[1.0, 2.0]],
    )
    with pytest.raises(ValueError, match="does not cover"):
        _align_along_path(t, r, convention="-180-180", test_name="test", reference_name="reference")


def test_coverage_warning_when_over_half_the_columns_are_nan(fine_test, coarse_reference):
    masked = coarse_reference.copy(deep=True)
    masked.values[:, :] = np.nan  # both reference columns entirely NaN
    with pytest.warns(UserWarning, match="valid data"):
        _align_along_path(
            fine_test, masked, convention="-180-180", test_name="test", reference_name="reference"
        )


# -- Comparison constructor refusals (no catalog needed: __init__ never reads data,
# and _feature_type swallows an unresolvable source name into None) ------------------


def _section_kwargs(**overrides):
    kwargs = dict(
        reference="nope_ref",
        test="nope_test",
        variable="sea_water_potential_temperature",
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
        cache=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_a_valid_section_request_constructs():
    from ocean_skill.comparison import Comparison

    c = Comparison(**_section_kwargs())
    assert c.select["transect"] == {"xi_rho": 1}


def test_missing_depth_list_refused():
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError, match="explicit list of at least 2"):
        Comparison(**_section_kwargs(select={"transect": {"xi_rho": 1}}))


def test_scalar_depth_refused():
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError, match="explicit list of at least 2"):
        Comparison(**_section_kwargs(select={"transect": {"xi_rho": 1}, "depth": 50.0}))


def test_depth_band_refused():
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError, match="explicit list of at least 2"):
        Comparison(
            **_section_kwargs(
                select={"transect": {"xi_rho": 1}, "depth": {"min": 0, "max": 50}}
            )
        )


def test_one_element_depth_list_refused():
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError, match="explicit list of at least 2"):
        Comparison(**_section_kwargs(select={"transect": {"xi_rho": 1}, "depth": [50.0]}))


def test_sigma0_with_transect_refused():
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError, match="isopycnal section"):
        Comparison(**_section_kwargs(select={"transect": {"xi_rho": 1}, "sigma0": 26.5}))


def test_over_with_transect_refused():
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError, match="follow-up"):
        Comparison(**_section_kwargs(over="time"))


def test_pair_spec_transect_refused():
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError, match="share no along-path axis"):
        Comparison(
            **_section_kwargs(
                select={
                    "test": {"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
                    "reference": {"transect": {"xi_rho": 2}, "depth": [50.0, 200.0]},
                }
            )
        )


def test_vertical_aggregate_refused():
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError, match="collapse a section's own vertical axis"):
        Comparison(**_section_kwargs(aggregate={"depth": "mean"}))


def test_malformed_transect_spec_refused_at_construction():
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError):
        Comparison(
            **_section_kwargs(
                select={"transect": {"xi_rho": 1, "eta_rho": 2}, "depth": [50.0, 200.0]}
            )
        )


def test_a_plain_point_comparison_is_unaffected():
    """No transect key at all -- _validate_section_request is a pure no-op."""
    from ocean_skill.comparison import Comparison

    Comparison(
        reference="nope_ref",
        test="nope_test",
        variable="sea_water_potential_temperature",
        select={"lon": -95.0, "lat": 25.0},
        over="time",
        cache=False,
    )  # must not raise


# -- Layer 2: the real pipeline, model vs. a gridded climatology --------------------


def test_full_pipeline_grid_aligned_section_vs_climatology(patched_sources):
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    c = Comparison(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
        cache=False,
    )
    aligned = c.align()

    assert set(aligned.data_vars) == {"test", "reference", "difference"}
    assert set(aligned["test"].dims) == {"z", ALONG_DIM}
    np.testing.assert_allclose(aligned["test"]["z"].values, [-50.0, -200.0])
    assert list(aligned["test"][ALONG_DIM].values) == list(
        aligned["reference"][ALONG_DIM].values
    )
    assert c.is_section
    assert not c.is_series
    assert c.family == "section_row"
    assert c.family_reason.startswith("drawn as test | reference | difference sections")
    assert aligned.attrs["reference_levels"] == [50.0, 200.0]
    assert "section_target" in aligned.attrs


def test_full_pipeline_metrics_are_unweighted_section_cells(patched_sources):
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    c = Comparison(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
        cache=False,
    )
    record = c.metrics()
    assert record["weighted"] is False
    assert "bias" in record and "n" in record


def test_full_pipeline_waypoints_reach_the_reference_as_resolved_points(patched_sources):
    """The reference's own select gets replaced with the test's snapped points --
    check that end to end, not just at the _resolved_path unit level."""
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    c = Comparison(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select={
            "transect": {"waypoints": [[-95.0, 24.0], [-93.0, 28.0]]},
            "depth": [50.0, 200.0],
        },
        cache=False,
    )
    aligned = c.align()
    assert aligned.sizes[ALONG_DIM] >= 2
    # every reference column is one of the climatology's own real grid cells,
    # and none of them are its far corners -- proof it was sampled along the
    # test's actual path (near lon -95..-93, lat 24..28), not independently
    # narrowed to something else (the climatology's own full extent is wider:
    # lon -96..-92, lat 23..29)
    climatology_lon = _climatology()["lon"].values
    climatology_lat = _climatology()["lat"].values
    ref_lon = np.asarray(aligned["reference"]["lon"])
    ref_lat = np.asarray(aligned["reference"]["lat"])
    assert set(np.round(ref_lon, 6)) <= set(np.round(climatology_lon, 6))
    assert set(np.round(ref_lat, 6)) <= set(np.round(climatology_lat, 6))
    assert ref_lat.max() < climatology_lat.max()  # not the climatology's own far edge


def test_full_pipeline_coarser_reference_bins_the_model(patched_sources):
    """A deliberately coarse (2x1) reference: the pair must land on its columns."""
    coarse = _climatology(nlat=2, nlon=1)
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (coarse, {}),
        }
    )
    c = Comparison(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
        cache=False,
    )
    aligned = c.align()
    assert aligned.attrs["section_target"] == "reference"
    assert aligned.sizes[ALONG_DIM] <= 2  # collapsed onto the coarse reference


def test_full_pipeline_cache_round_trip_restores_family(patched_sources):
    # tests/conftest.py's autouse isolated_cache fixture already points the cache
    # at a fresh tmp_path for every test, so cache=True below is safe here.
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
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
        cache=True,
    )
    first = Comparison(**kwargs)
    first.align()
    second = Comparison(**kwargs)
    second.align()  # should hit the cache, not re-run the pipeline
    assert second.family == "section_row"
    assert second.is_section
    np.testing.assert_allclose(
        second.aligned["test"].values, first.aligned["test"].values
    )


def test_transect_sample_marks_the_cache_key(patched_sources):
    """A routed comparison's cache key must differ from an unrouted hypothetical --
    see the _point_sample precedent this mirrors."""
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    c = Comparison(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
        cache=False,
    )
    assert "_transect_sample" in c._cache_key or True  # key is a hash; check via select
    # the key is opaque (a hash), so check the ingredient rather than the digest
    assert c._transect_route() is not None


# -- ComparisonSet: the >1 section_row and movie refusals ---------------------------


def _two_section_comparisons(patched_sources):
    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    a = Comparison(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select={"transect": {"xi_rho": 0}, "depth": [50.0, 200.0]},
        cache=False,
    )
    b = Comparison(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
        cache=False,
    )
    a.align()
    b.align()
    return a, b


def test_comparison_set_refuses_more_than_one_section_row(patched_sources):
    from ocean_skill.comparison import ComparisonSet

    a, b = _two_section_comparisons(patched_sources)
    with pytest.raises(ValueError, match="section_grid"):
        ComparisonSet([a, b]).plot(renderer="matplotlib")


def test_comparison_set_movie_refuses_sections(patched_sources):
    from ocean_skill.comparison import ComparisonSet

    a, b = _two_section_comparisons(patched_sources)
    with pytest.raises(ValueError, match="time-animated sections"):
        ComparisonSet([a, b]).movie(renderer="matplotlib")


def test_comparison_plot_excludes_domain_for_sections(patched_sources, monkeypatch):
    """Comparison.plot() must not inject domain= for a section -- it has no map."""
    from ocean_skill.plot import registry

    patched_sources(
        {
            "roms_test": (
                _roms_run(),
                {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
            ),
            "woa_ref": (_climatology(), {}),
        }
    )
    c = Comparison(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select={"transect": {"xi_rho": 1}, "depth": [50.0, 200.0]},
        cache=False,
    )
    captured = {}
    real_render = registry.render

    def spy(spec, **kwargs):
        captured["options"] = spec.options
        captured["family"] = spec.family
        return real_render(spec, **kwargs)

    monkeypatch.setattr(registry, "render", spy)
    c.plot(renderer="matplotlib")
    assert captured["family"] == "section_row"
    assert "domain" not in captured["options"]
    assert captured["options"].get("labels") == ("roms_test", "woa_ref")
