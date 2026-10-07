"""The requested path rides beside the snapped cells: ``path_lon``/``path_lat``.

``transect.sample_along`` snaps each requested point to a source cell, and the cell
is where the *data* came from -- ``lon``/``lat``. Measuring the along-path distance
between those snapped positions, and sampling a second source at them, is wrong on
any line that runs along a cell boundary: the equator through a 1-degree grid whose
rows sit at +/-0.5, say. The snapped cells wobble a few km either side of the line,
the coarse source flips between its two rows with them, and every flip used to add
its full ~111 km to the along-path coordinate (a 13,800 km section reading 42,000).

These tests pin the fix, in three layers: the sampler (``sample_along`` on a
synthetic grid), the coordinates every section shape carries (grid slice, slab,
cache round trip), and the end-to-end comparison of a jittery fine lane against a
coarse reference.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import cache, catalog
from ocean_skill.align import ALONG_DIM, _haversine_km
from ocean_skill.comparison import (
    Comparison,
    _is_stale_pathless_section,
    prepare_source,
)
from ocean_skill.transect import grid_slice, sample_along

VAR = "sea_water_potential_temperature"
KM_PER_DEG = 6371.0088 * np.pi / 180.0  # the haversine's own degree


def _coarse_climatology(lon0=-62.5, lon1=66.5) -> xr.Dataset:
    """Build a 1-degree WOA-style grid with rows at +/-0.5, none *at* the equator.

    Each row carries its own value (``10 + lat``), so a section that wanders between
    the two rows nearest the equator shows up in the data, not just the coordinates.
    """
    lon = np.arange(lon0, lon1 + 0.01, 1.0)
    lat = np.arange(-4.5, 5.0, 1.0)
    depth = np.array([50.0, 200.0])
    values = 10.0 + lat[None, :, None] + 0.0 * depth[:, None, None] + 0.0 * lon
    da = xr.DataArray(
        values,
        dims=("depth", "lat", "lon"),
        coords={"depth": depth, "lat": lat, "lon": lon},
        name=VAR,
        attrs={"units": "degC"},
    )
    return da.to_dataset()


def _jittery_fine_grid(nx=1260, lon0=-60.0, step=0.1) -> xr.Dataset:
    """Build a curvilinear fine grid whose cell latitudes wobble +/-0.03 deg.

    The equator's nearest cell in every column is the ``j == 0`` row, whose latitude
    alternates +0.03 / -0.03 -- the sub-cell jitter that, snapped to, sends a coarse
    reference's sampler back and forth across the boundary between its rows.
    """
    ny = 5
    i = np.arange(nx)
    j = np.arange(ny) - ny // 2
    lon2d = lon0 + step * np.broadcast_to(i[None, :], (ny, nx))
    lat2d = step * j[:, None] + 0.03 * np.where(i % 2 == 0, 1.0, -1.0)[None, :]
    depth = np.array([50.0, 200.0])
    values = np.broadcast_to(
        (20.0 - 0.01 * depth)[:, None, None], (depth.size, ny, nx)
    ).copy()
    da = xr.DataArray(
        values,
        dims=("depth", "eta", "xi"),
        coords={
            "depth": depth,
            "lon": (("eta", "xi"), lon2d),
            "lat": (("eta", "xi"), lat2d),
        },
        name=VAR,
        attrs={"units": "degC"},
    )
    return da.to_dataset()


# -- sample_along: the sampler --------------------------------------------------------


def test_a_coarse_grid_sampled_along_its_row_boundary_has_the_true_length():
    """Requests jittering about lat=0 flip rows, but add no distance for it.

    These requests really do straddle the boundary between the +/-0.5 rows, so the
    sampled row alternating is the honest answer here -- what the old bookkeeping got
    wrong was charging ~111 km of along-path distance for every flip.
    """
    ds = _coarse_climatology()
    n = 2480  # 0.05-degree steps; a +/-0.001 degree wobble is ~0.2 km on a 5.5 km step
    lons = np.linspace(-60.0, 64.0, n)
    lats = np.where(np.arange(n) % 2 == 0, 0.001, -0.001)
    out = sample_along(ds, lons, lats, subject="coarse")

    along = np.asarray(out[ALONG_DIM])
    assert np.all(np.diff(along) > 0), "along must be strictly increasing"
    true_km = 124.0 * KM_PER_DEG
    assert along.max() == pytest.approx(true_km, rel=0.02)
    assert along.max() < true_km * 1.1, "no ~111 km added per row flip"
    np.testing.assert_allclose(np.asarray(out["path_lat"]), 0.0, atol=0.002)
    assert set(np.unique(np.asarray(out["lat"]))) == {-0.5, 0.5}


def test_a_coarse_grid_sampled_exactly_on_its_row_boundary_stays_on_one_row():
    """Requests at exactly lat=0 tie between the rows, and the tie resolves one way."""
    ds = _coarse_climatology()
    n = 1240
    out = sample_along(
        ds, np.linspace(-60.0, 64.0, n), np.zeros(n), subject="coarse"
    )
    assert np.unique(np.asarray(out["lat"])).size == 1
    assert np.all(np.diff(np.asarray(out[ALONG_DIM])) > 0)
    assert float(out[ALONG_DIM].max()) == pytest.approx(124.0 * KM_PER_DEG, rel=0.02)


def test_a_fine_jittered_equator_request_measures_the_requested_distance():
    """Distance comes from the requested lat=0 line, not from the zig-zag of cells."""
    ds = _jittery_fine_grid()
    n = 1240
    lons = np.linspace(-60.0, 64.0, n)
    lats = np.zeros(n)
    out = sample_along(ds, lons, lats, subject="fine")

    # The snapped cells really do zig-zag (that is the premise)...
    snapped_lat = np.asarray(out["lat"])
    assert snapped_lat.max() > 0.02 and snapped_lat.min() < -0.02
    snapped_km = float(
        _haversine_km(
            np.asarray(out["lon"])[:-1],
            snapped_lat[:-1],
            np.asarray(out["lon"])[1:],
            snapped_lat[1:],
        ).sum()
    )
    true_km = 124.0 * KM_PER_DEG
    assert snapped_km > true_km * 1.1, "the premise: snapped cells add distance"
    # ...and the along coordinate ignores it.
    assert float(out[ALONG_DIM].max()) == pytest.approx(true_km, rel=0.01)
    np.testing.assert_allclose(np.asarray(out["path_lat"]), 0.0, atol=1e-9)
    assert np.all(np.diff(np.asarray(out[ALONG_DIM])) > 0)


def test_a_run_of_requests_in_one_cell_collapses_to_the_middle_of_the_run():
    """Four requests in one coarse cell give one column at their mean km and lon."""
    lon = np.arange(0.0, 10.0, 1.0)
    lat = np.arange(-2.0, 3.0, 1.0)
    ds = xr.Dataset(
        {"t": (("lat", "lon"), np.random.default_rng(0).random((lat.size, lon.size)))},
        coords={"lat": lat, "lon": lon},
    )
    # Cell lon=2 owns [1.5, 2.5); lon=3 owns [2.5, 3.5).
    lons = np.array([1.6, 1.8, 2.0, 2.4, 3.0, 4.0])
    lats = np.zeros(6)
    out = sample_along(ds, lons, lats, subject="runs")

    assert out.sizes[ALONG_DIM] == 3
    np.testing.assert_allclose(np.asarray(out["lon"]), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(np.asarray(out["path_lon"])[0], np.mean(lons[:4]))
    km = np.concatenate(
        [[0.0], np.cumsum(_haversine_km(lons[:-1], lats[:-1], lons[1:], lats[1:]))]
    )
    np.testing.assert_allclose(np.asarray(out[ALONG_DIM])[0], km[:4].mean())
    np.testing.assert_allclose(np.asarray(out[ALONG_DIM])[1:], km[4:])


def test_a_run_across_the_seam_averages_longitude_unwrapped():
    lon = np.arange(-180.0, 180.0, 5.0)
    ds = xr.Dataset(
        {"t": (("lat", "lon"), np.ones((3, lon.size)))},
        coords={"lat": [-1.0, 0.0, 1.0], "lon": lon},
    )
    # All four requests snap to the lon=-180 cell; their mean must be 180/-180, not 0.
    lons = np.array([177.6, 179.0, -179.0, -177.6, -170.0, -165.0])
    out = sample_along(ds, lons, np.zeros(6), subject="seam")
    first = float(np.asarray(out["path_lon"])[0])
    assert abs(abs(first) - 180.0) < 1e-6, first


def test_bilinear_keeps_every_requested_point_as_its_own_column():
    lon = np.linspace(-96.0, -93.0, 7)
    lat = np.linspace(23.0, 28.0, 6)
    ds = xr.DataArray(
        np.random.default_rng(1).random((lat.size, lon.size)),
        dims=("lat", "lon"),
        coords={"lon": lon, "lat": lat},
        name="chl",
    ).to_dataset()
    lons = np.linspace(-95.5, -93.5, 40)  # far finer than the 0.5-degree cells
    lats = np.full(40, 25.0)
    out = sample_along(ds, lons, lats, method="bilinear", subject="bl")
    assert out.sizes[ALONG_DIM] == 40
    np.testing.assert_allclose(np.asarray(out["path_lon"]), lons)
    np.testing.assert_allclose(np.asarray(out["path_lat"]), lats)
    expected = np.concatenate(
        [[0.0], np.cumsum(_haversine_km(lons[:-1], lats[:-1], lons[1:], lats[1:]))]
    )
    np.testing.assert_allclose(np.asarray(out[ALONG_DIM]), expected)


# -- every section shape carries the same coordinates ---------------------------------


def test_a_grid_slice_carries_path_coords_equal_to_lon_lat():
    ds = _jittery_fine_grid(nx=30)
    out = grid_slice(ds, "eta", 2, subject="grid")
    assert ALONG_DIM in out.dims
    np.testing.assert_array_equal(out["path_lon"].values, out["lon"].values)
    np.testing.assert_array_equal(out["path_lat"].values, out["lat"].values)
    assert out["path_lon"].dims == (ALONG_DIM,)
    assert out["path_lon"].attrs["units"] == "degrees_east"
    assert out["path_lat"].attrs["units"] == "degrees_north"


def test_a_slab_section_carries_path_coords_equal_to_lon_lat():
    from ocean_skill.comparison import _slab_to_section

    lat = np.arange(-5.0, 6.0, 1.0)
    depth = np.array([50.0, 200.0])
    da = xr.DataArray(
        np.ones((depth.size, lat.size)),
        dims=("depth", "lat"),
        coords={"depth": depth, "lat": lat},
        name=VAR,
    )
    out = _slab_to_section(da, "lat", (160.0, 220.0))
    assert ALONG_DIM in out.dims
    np.testing.assert_array_equal(out["path_lon"].values, out["lon"].values)
    np.testing.assert_array_equal(out["path_lat"].values, out["lat"].values)


# -- the comparison: a jittery fine lane against a coarse reference -------------------


@pytest.fixture
def patched_sources(monkeypatch):
    def _patch(sources):
        monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
        monkeypatch.setattr(
            catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
        )

    return _patch


def _equator_comparison(patched_sources, **kwargs):
    patched_sources(
        {"fine": (_jittery_fine_grid(), {}), "coarse": (_coarse_climatology(), {})}
    )
    return Comparison(
        test="fine",
        reference="coarse",
        variable=VAR,
        select={
            "transect": {"lat": 0, "lon": {"min": -60.0, "max": 64.0}},
            "depth": [50.0, 200.0],
        },
        **kwargs,
    )


def test_an_equator_comparison_against_a_coarse_reference_has_its_true_length(
    patched_sources,
):
    c = _equator_comparison(patched_sources, cache=False)
    aligned = c.align()

    true_km = 124.0 * KM_PER_DEG
    assert float(aligned[ALONG_DIM].max()) == pytest.approx(true_km, rel=0.03)
    assert aligned.attrs["section_length_km"] == pytest.approx(true_km, rel=0.03)
    assert np.all(np.diff(np.asarray(aligned[ALONG_DIM])) > 0)
    # The request is lat=0 throughout, and the merged result says so...
    np.testing.assert_allclose(np.asarray(aligned["path_lat"]), 0.0, atol=1e-6)
    assert aligned["path_lon"].dims == (ALONG_DIM,)
    # ...and the reference stayed on one of its two rows instead of alternating.
    ref_values = np.unique(np.asarray(aligned["reference"].isel(z=0)).round(6))
    ref_values = ref_values[np.isfinite(ref_values)]
    assert ref_values.size == 1, f"reference alternated between rows: {ref_values}"


def test_the_reference_is_sampled_on_the_requested_path_not_the_snapped_cells(
    patched_sources,
):
    from ocean_skill.align import SECTION_VERTICAL_DIMS

    c = _equator_comparison(patched_sources, cache=False)
    t, _ = c._prepare_lane(
        "fine", False, False, role="test", keep=SECTION_VERTICAL_DIMS
    )
    extra, bbox = c._resolved_path(t, c._transect_route())
    pts = np.asarray(extra["transect"]["points"])
    np.testing.assert_allclose(pts[:, 1], 0.0, atol=1e-4)
    # The bbox still covers the snapped cells (which sit up to 0.03 deg off the line).
    assert bbox[1] <= -0.029 and bbox[3] >= 0.029


# -- the cache ------------------------------------------------------------------------


def test_a_cache_round_trip_keeps_the_path_coordinates(patched_sources):
    kwargs = {"cache": True}
    first = _equator_comparison(patched_sources, **kwargs)
    first.align()
    second = _equator_comparison(patched_sources, **kwargs)
    second.align()  # served from the aligned-pair cache
    assert "path_lon" in second.aligned.coords and "path_lat" in second.aligned.coords
    np.testing.assert_allclose(
        second.aligned["path_lat"].values, first.aligned["path_lat"].values
    )


def test_a_stale_section_lane_without_path_coords_is_recomputed(
    patched_sources, monkeypatch
):
    """A cached section lane from before ``path_lon`` existed is discarded, not served.

    Mirrors the stale-positionless-station test: overwrite the entry
    ``prepare_source`` filed under its key with the old shape (``along`` but no
    ``path_lon``), and the next call must warn, recompute, and overwrite -- after
    which the entry is good and is served without a recompute.
    """
    import warnings

    patched_sources({"fine": (_jittery_fine_grid(nx=60), {})})
    select = {
        "transect": {"lat": 0, "lon": {"min": -59.0, "max": -54.0}},
        "depth": [50.0, 200.0],
    }
    saved = {}
    real_save = cache.save_field

    def recording_save(key, da, depth):
        saved["key"] = key
        return real_save(key, da, depth)

    monkeypatch.setattr(cache, "save_field", recording_save)
    reads = []
    real_read = osk.read
    monkeypatch.setattr(
        osk, "read", lambda name, **kw: (reads.append(name), real_read(name, **kw))[1]
    )

    fresh, _ = prepare_source("fine", VAR, select, None)
    assert "path_lon" in fresh.coords and not _is_stale_pathless_section(fresh)

    stale = fresh.drop_vars(["path_lon", "path_lat"])
    assert _is_stale_pathless_section(stale), "the seeded entry must reproduce the bug"
    real_save(saved["key"], stale, None)

    n_reads = len(reads)
    with pytest.warns(UserWarning, match="fine.*path_lon"):
        again, _ = prepare_source("fine", VAR, select, None)
    assert "path_lon" in again.coords and "path_lat" in again.coords
    assert len(reads) > n_reads, "the stale lane must be recomputed"

    n_reads = len(reads)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare_source("fine", VAR, select, None)
    assert len(reads) == n_reads, "the repaired entry is served from the cache"


def test_a_stale_aligned_section_without_path_coords_is_recomputed(patched_sources):
    c = _equator_comparison(patched_sources, cache=True)
    c.align()
    key = c._cache_key
    stale = c.aligned.drop_vars(["path_lon", "path_lat"])
    cache.save(key, stale)

    again = _equator_comparison(patched_sources, cache=True)
    with pytest.warns(UserWarning, match="path_lon"):
        aligned = again.align()
    assert "path_lon" in aligned.coords
