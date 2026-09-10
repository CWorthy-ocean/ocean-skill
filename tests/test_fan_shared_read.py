"""Shared multi-point model read: `_build_shared_slabs` / `_shared_slab`.

The reported problem: `osk.compare(reference=<~14 ADCP moorings>, test="his",
variables=["eastward_sea_water_velocity"], select={"depth": 10})` decompressed the
same model time-chunks once per mooring (the moorings cluster in one spatial
region), because ocean-skill's read memo only shares the *lazy* dataset, never
decompressed data. `_build_shared_slabs` reads the ROMS test source once, decompressed,
over a window that covers every point-like reference `compare()` is about to fan
over, and stores it in the module-level `_SHARED_SLABS` registry; `prepare_source`
consults it (via `_shared_slab`) before falling back to `osk.read`.

This must be a pure optimization -- every value byte-identical to reading the test
lane fresh per pair -- so the tests here drive `_build_shared_slabs`/`prepare_source`
directly (the same pattern `test_point_column_materialize.py` and
`test_roms_velocity.py` use), rather than fighting the full `compare()` fan's
variable/featureType resolution machinery.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
import ocean_skill.roms as roms
from ocean_skill import catalog, comparison
from ocean_skill.comparison import (
    _build_shared_slabs,
    _SHARED_SLABS,
    _shared_slab,
    prepare_source,
)

N = 8  # s_rho levels

TEST_META = {"model": "roms", "vertical": {"hc": 20.0, "s_dim": "s_rho"}}


def _roms_multi_time(ny=10, nx=12, nt=6) -> xr.Dataset:
    """A curvilinear, time-varying ROMS-like Dataset, chunked one step per chunk
    (like a real history file), with the pre-derived geographic velocity a real
    `osk.read` would already carry -- exactly what `_build_shared_slabs` must drop
    before cropping+loading (see its own comment on why).
    """
    rng = np.random.default_rng(3)
    lon = np.tile(np.linspace(-90.0, -88.0, nx), (ny, 1))
    lat = np.tile(np.linspace(20.0, 22.0, ny)[:, None], (1, nx))
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    angle = 0.1 * i - 0.05 * j  # non-uniform, so a wrong cell gives a wrong rotation
    mask_rho = np.ones((ny, nx))
    sigma = np.linspace(-1.0, 0.0, N)
    h = np.full((ny, nx), 100.0)

    ds = xr.Dataset(
        {
            "sea_water_x_velocity": (
                ("time", "s_rho", "eta_rho", "xi_u"),
                rng.normal(0.1, 0.02, (nt, N, ny, nx - 1)),
            ),
            "sea_water_y_velocity": (
                ("time", "s_rho", "eta_v", "xi_rho"),
                rng.normal(0.05, 0.02, (nt, N, ny - 1, nx)),
            ),
            "h": (("eta_rho", "xi_rho"), h),
            "sigma_r": ("s_rho", sigma),
            "Cs_r": ("s_rho", sigma),
        },
        coords={
            "time": np.arange(nt),
            "lon": (("eta_rho", "xi_rho"), lon),
            "lat": (("eta_rho", "xi_rho"), lat),
            "angle": (("eta_rho", "xi_rho"), angle),
            "mask_rho": (("eta_rho", "xi_rho"), mask_rho),
        },
    )
    ds = ds.chunk({"time": 1, "s_rho": N})
    ds["sigma_r"] = ds["sigma_r"].chunk({"s_rho": -1})
    ds["Cs_r"] = ds["Cs_r"].chunk({"s_rho": -1})
    standardized = roms._add_geographic_velocity(ds)
    mask = standardized["mask_rho"] == 1
    for var in list(standardized.data_vars):
        da = standardized[var]
        if {"eta_rho", "xi_rho"} <= set(da.dims):
            standardized[var] = da.where(mask)
    return standardized


# Three station positions: two interior, one deliberately at the grid's own edge
# (iy=0) -- the case where a wrong union window would give a wrong trim/rotation.
_INTERIOR_A = (-89.6, 20.4)
_INTERIOR_B = (-88.6, 21.6)
_EDGE = (-89.0, 20.0)  # nearest row is eta_rho=0


def _catalog_stub(monkeypatch, grid: xr.Dataset, station_positions: dict[str, tuple]):
    """Route `osk.read`/`catalog.resolve` for one ROMS test source ``"his"`` and one
    fixed-point catalog entry per name in ``station_positions``, counting reads of
    ``"his"``.
    """
    read_calls = {"his": 0}

    def fake_read(name, **kw):
        if name == "his":
            read_calls["his"] += 1
            return grid
        raise AssertionError(f"unexpected osk.read({name!r}) in this test")

    def fake_resolve(name):
        if name == "his":
            return SimpleNamespace(metadata=TEST_META)
        lon, lat = station_positions[name]
        return SimpleNamespace(
            metadata={
                "featureType": "timeSeries",
                "geospatial_lon_min": lon,
                "geospatial_lon_max": lon,
                "geospatial_lat_min": lat,
                "geospatial_lat_max": lat,
            }
        )

    monkeypatch.setattr(osk, "read", fake_read)
    monkeypatch.setattr(catalog, "resolve", fake_resolve)
    return read_calls


@pytest.fixture(autouse=True)
def _clear_slabs():
    _SHARED_SLABS.clear()
    yield
    _SHARED_SLABS.clear()


def test_build_shared_slabs_reads_the_test_source_once(monkeypatch):
    grid = _roms_multi_time()
    positions = {"m1": _INTERIOR_A, "m2": _INTERIOR_B, "m3": _EDGE}
    read_calls = _catalog_stub(monkeypatch, grid, positions)

    _build_shared_slabs(list(positions), ["his"], qc=None)

    assert read_calls["his"] == 1
    slab = _shared_slab("his", None)
    assert slab is not None
    # fully realized -- no dask-backed variable left to decompress again later
    assert all(v.chunks is None for v in slab.variables.values())
    # the pre-derived pair must not survive into the slab -- see the builder's own
    # comment on why keeping them would reintroduce the full-domain graph cost
    assert not set(roms.GEOGRAPHIC_VELOCITY_NAMES) & set(slab.data_vars)
    # raw staggered components (what the velocity re-derive needs) do survive
    assert "sea_water_x_velocity" in slab.data_vars
    assert "sea_water_y_velocity" in slab.data_vars


def test_slab_not_built_for_a_single_reference(monkeypatch):
    grid = _roms_multi_time()
    positions = {"m1": _INTERIOR_A}
    _catalog_stub(monkeypatch, grid, positions)

    _build_shared_slabs(list(positions), ["his"], qc=None)

    assert _shared_slab("his", None) is None


def test_slab_not_built_when_a_reference_is_not_a_fixed_point(monkeypatch):
    grid = _roms_multi_time()
    positions = {"m1": _INTERIOR_A, "m2": _INTERIOR_B}
    read_calls = _catalog_stub(monkeypatch, grid, positions)

    def fake_resolve(name):
        if name == "his":
            return SimpleNamespace(metadata=TEST_META)
        if name == "m1":
            lon, lat = positions["m1"]
            return SimpleNamespace(
                metadata={
                    "geospatial_lon_min": lon,
                    "geospatial_lon_max": lon,
                    "geospatial_lat_min": lat,
                    "geospatial_lat_max": lat,
                }
            )
        # m2 is a REGION, not a point -- a slab sized to m1 alone could miss
        # whatever m2 actually needs, so no slab should be built at all.
        return SimpleNamespace(
            metadata={
                "geospatial_lon_min": -90.0,
                "geospatial_lon_max": -85.0,
                "geospatial_lat_min": 15.0,
                "geospatial_lat_max": 25.0,
            }
        )

    monkeypatch.setattr(catalog, "resolve", fake_resolve)

    _build_shared_slabs(list(positions), ["his"], qc=None)

    assert _shared_slab("his", None) is None
    assert read_calls["his"] == 0  # bailed before ever reading the test source


def test_prepare_source_uses_the_slab_instead_of_reading_again(monkeypatch):
    grid = _roms_multi_time()
    positions = {"m1": _INTERIOR_A, "m2": _INTERIOR_B, "m3": _EDGE}
    read_calls = _catalog_stub(monkeypatch, grid, positions)

    _build_shared_slabs(list(positions), ["his"], qc=None)
    assert read_calls["his"] == 1

    # osk.read now raises for "his" if called again -- proves prepare_source uses
    # the slab rather than re-reading, for every one of the three stations.
    def refuse_read(name, **kw):
        raise AssertionError(f"osk.read({name!r}) should not run -- the slab exists")

    monkeypatch.setattr(osk, "read", refuse_read)

    for lon, lat in positions.values():
        da, _ = prepare_source(
            "his",
            "eastward_sea_water_velocity",
            {"depth": "surface"},
            None,
            use_cache=False,
            bbox=(lon, lat, lon, lat),
        )
        assert da is not None
        assert bool(np.isfinite(da).any())


@pytest.mark.parametrize("name,pos", [("interior_a", _INTERIOR_A), ("interior_b", _INTERIOR_B), ("edge", _EDGE)])
def test_byte_identical_with_and_without_the_slab(monkeypatch, name, pos):
    """The whole point: a station sampled through the shared slab must match the
    same station sampled by reading the test source directly, edge case included
    (the union window's own trim math is the one place this could silently break --
    see `_build_shared_slabs`'s own comment on why it doesn't).
    """
    grid = _roms_multi_time()
    positions = {"interior_a": _INTERIOR_A, "interior_b": _INTERIOR_B, "edge": _EDGE}
    lon, lat = pos

    # -- with the slab --
    read_calls = _catalog_stub(monkeypatch, grid, positions)
    _build_shared_slabs(list(positions), ["his"], qc=None)
    assert _shared_slab("his", None) is not None
    da_slab, depth_slab = prepare_source(
        "his",
        "eastward_sea_water_velocity",
        {"depth": "surface"},
        None,
        use_cache=False,
        bbox=(lon, lat, lon, lat),
    )

    # -- without the slab: force a miss, read the (same) grid directly --
    _SHARED_SLABS.clear()
    da_direct, depth_direct = prepare_source(
        "his",
        "eastward_sea_water_velocity",
        {"depth": "surface"},
        None,
        use_cache=False,
        bbox=(lon, lat, lon, lat),
    )

    xr.testing.assert_allclose(da_slab, da_direct, atol=0, rtol=0)
    assert depth_slab == depth_direct


def test_non_velocity_variable_point_read_also_matches(monkeypatch):
    """A non-velocity variable sharing the batched slab must be unaffected by the
    east/north drop -- it never looked at those anyway.
    """
    grid = _roms_multi_time()
    positions = {"m1": _INTERIOR_A, "m2": _INTERIOR_B}
    _catalog_stub(monkeypatch, grid, positions)

    _build_shared_slabs(list(positions), ["his"], qc=None)
    lon, lat = _INTERIOR_A
    da_slab, _ = prepare_source(
        "his",
        "sea_water_x_velocity",
        {"depth": "surface"},
        None,
        use_cache=False,
        bbox=(lon, lat, lon, lat),
    )

    _SHARED_SLABS.clear()
    da_direct, _ = prepare_source(
        "his",
        "sea_water_x_velocity",
        {"depth": "surface"},
        None,
        use_cache=False,
        bbox=(lon, lat, lon, lat),
    )
    xr.testing.assert_allclose(da_slab, da_direct, atol=0, rtol=0)


def test_union_window_pads_past_an_unclamped_edge(monkeypatch):
    """Regression for a real off-by-one: two references landing on the SAME nearest
    cell (so the naive per-reference union adds no extra slack from a second,
    farther-apart station) must still match a direct read, including on the one
    axis (``xi``) whose window edge here falls strictly inside the domain rather
    than at its true boundary. Without an extra cell of slack past that edge, the
    slab's own boundary coincides with it by construction and a later per-pair
    crop misreads "real data just past my window" as "the true domain edge",
    silently discarding one valid rho point from the windowed velocity re-derive.
    """
    grid = _roms_multi_time()
    close_pair = {"c1": (-89.0, 21.0), "c2": (-89.02, 21.02)}
    _catalog_stub(monkeypatch, grid, close_pair)

    _build_shared_slabs(list(close_pair), ["his"], qc=None)
    assert _shared_slab("his", None) is not None

    lon, lat = close_pair["c1"]
    da_slab, _ = prepare_source(
        "his",
        "eastward_sea_water_velocity",
        {"depth": "surface"},
        None,
        use_cache=False,
        bbox=(lon, lat, lon, lat),
    )
    _SHARED_SLABS.clear()
    da_direct, _ = prepare_source(
        "his",
        "eastward_sea_water_velocity",
        {"depth": "surface"},
        None,
        use_cache=False,
        bbox=(lon, lat, lon, lat),
    )
    xr.testing.assert_allclose(da_slab, da_direct, atol=0, rtol=0)


def test_a_single_comparison_query_never_touches_the_slab_registry(monkeypatch):
    """`compare()`'s own guard (fewer than two references) must leave the general,
    single-comparison path completely untouched.
    """
    grid = _roms_multi_time()
    positions = {"m1": _INTERIOR_A}
    _catalog_stub(monkeypatch, grid, positions)

    _build_shared_slabs(list(positions), ["his"], qc=None)
    assert _SHARED_SLABS == {}
