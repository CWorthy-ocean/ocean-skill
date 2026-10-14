"""Bounded memory for the depth match on a store chunked like a real ROMS kerchunk.

``make_kerchunk`` subchunks a variable along its *leading* dimensions only, so a
restart run lands as ``(time=1, s_rho=5, eta, xi)`` -- each horizontal level one chunk.
The depth match needs the whole column in one chunk, and making ``s_rho`` one chunk
there gives ``n_s`` times a full level per chunk (1.3 GiB on a 962 x 1858 x 100 grid),
several times over in the kernel, one per core: an out-of-memory kill. The helper has to
split the horizontal as well, to dask's ``array.chunk-size``.

The grid here is small and ``array.chunk-size`` is lowered to match, so the same
proportions -- a column chunk many times the limit before the fix -- hold in a test that
runs in a fraction of a second.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

dask = pytest.importorskip("dask")

from ocean_skill import roms  # noqa: E402

META = {"vertical": {"hc": 20.0, "s_dim": "s_rho"}}
NT, NS, NY, NX = 3, 40, 48, 64
LIMIT = 256 * 1024  # bytes; one full column chunk here is NS*NY*NX*8 = 960 KiB


def _roms_like() -> xr.Dataset:
    sigma = np.linspace(-1.0, 0.0, NS)
    rng = np.random.default_rng(1)
    dims = ("time", "s_rho", "eta_rho", "xi_rho")
    return xr.Dataset(
        {
            "temp": (dims, rng.normal(15.0, 2.0, (NT, NS, NY, NX))),
            "salt": (dims, rng.normal(35.0, 0.3, (NT, NS, NY, NX))),
            "zeta": (("time", "eta_rho", "xi_rho"), rng.normal(0, 0.4, (NT, NY, NX))),
            "h": (("eta_rho", "xi_rho"), 50.0 + 150.0 * rng.random((NY, NX))),
            "sigma_r": ("s_rho", sigma),
            "Cs_r": ("s_rho", sigma),
        },
        coords={
            "time": np.arange(NT),
            "lon": (("eta_rho", "xi_rho"), np.tile(np.linspace(-98, -80, NX), (NY, 1))),
            "lat": (
                ("eta_rho", "xi_rho"),
                np.tile(np.linspace(18, 31, NY)[:, None], (1, NX)),
            ),
        },
    )


def _like_the_store(ds: xr.Dataset) -> xr.Dataset:
    """Chunk as ``make_kerchunk`` does: leading dims split, the horizontal one chunk."""
    return ds.chunk({"time": 1, "s_rho": 5, "eta_rho": -1, "xi_rho": -1})


@pytest.fixture
def small_chunks():
    with dask.config.set({"array.chunk-size": LIMIT}):
        yield


def _kernel_sizes(monkeypatch) -> list[int]:
    """Record the byte size of the largest array each kernel call is handed."""
    seen: list[int] = []
    for name in ("_match_columns", "_classify_columns"):
        original = getattr(roms, name)

        def wrapper(*args, _original=original, **kwargs):
            seen.append(max(a.nbytes for a in args if isinstance(a, np.ndarray)))
            return _original(*args, **kwargs)

        monkeypatch.setattr(roms, name, wrapper)
    return seen


def test_a_store_chunked_like_kerchunk_gives_bounded_column_chunks(
    small_chunks, monkeypatch
):
    ds = _like_the_store(_roms_like())
    assert NS * NY * NX * 8 > 3 * LIMIT, "the unsplit column must be far over budget"
    seen = _kernel_sizes(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the fixed-origin notes
        out = roms.to_depth(ds, META, [0.0, 20.0, 40.0])
        out["temp"].mean(["eta_rho", "xi_rho"]).compute()

    assert seen, "the kernel never ran"
    assert max(seen) <= LIMIT


def test_the_column_helper_splits_the_horizontal_but_not_time(small_chunks):
    da = _like_the_store(_roms_like())["temp"]
    column = roms._contiguous_column(da, "s_rho")

    sizes = dict(zip(column.dims, column.chunksizes.values(), strict=True))
    assert sizes["s_rho"] == (NS,), "the column must be contiguous"
    assert sizes["time"] == (1, 1, 1), "time stays one step a chunk"
    assert len(sizes["eta_rho"]) * len(sizes["xi_rho"]) > 1
    biggest = max(sizes["eta_rho"]) * max(sizes["xi_rho"]) * NS * 8
    assert biggest <= LIMIT


def test_variables_and_frame_share_their_horizontal_chunks(small_chunks):
    """Same rule on data and frame: nothing for dask to re-merge in the kernel."""
    ds = _like_the_store(_roms_like())
    ds = roms.add_depth_coord(ds, META)
    data = roms._contiguous_column(ds["temp"].reset_coords(drop=True), "s_rho")
    frame = roms._contiguous_column(ds["z_rho"].reset_coords(drop=True), "s_rho")
    for dim in ("time", "eta_rho", "xi_rho"):
        assert data.chunksizes[dim] == frame.chunksizes[dim]


def test_a_finely_chunked_horizontal_is_left_alone(small_chunks):
    ds = _roms_like().chunk({"time": 1, "s_rho": 5, "eta_rho": 8, "xi_rho": 8})
    column = roms._contiguous_column(ds["temp"], "s_rho")
    assert column.chunksizes["eta_rho"] == ds["temp"].chunksizes["eta_rho"]
    assert column.chunksizes["xi_rho"] == ds["temp"].chunksizes["xi_rho"]
    assert column.chunksizes["s_rho"] == (NS,)


def test_the_default_limit_leaves_a_small_source_in_one_horizontal_chunk():
    """No gratuitous splitting: this grid is far below dask's 128 MiB default."""
    ds = _like_the_store(_roms_like())
    column = roms._contiguous_column(ds["temp"], "s_rho")
    assert len(column.chunksizes["eta_rho"]) == 1
    assert len(column.chunksizes["xi_rho"]) == 1


@pytest.mark.parametrize("mode", ["interp", "nearest"])
def test_values_equal_the_unchunked_path(small_chunks, mode):
    eager = _roms_like()
    chunked = _like_the_store(eager)
    fn = roms.to_depth if mode == "interp" else roms.nearest_depth_levels
    depths = [0.0, 20.0, 60.0, 120.0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = fn(eager, META, depths)
        got = fn(chunked, META, depths)
        for name in ("temp", "salt"):
            np.testing.assert_array_equal(got[name].values, want[name].values)
        for key in ("depth_edge_top", "depth_edge_bottom", "depth_above_surface"):
            assert got.attrs[key] == want.attrs[key]


def test_no_chunk_explosion_warning(small_chunks):
    """Frame, bounds and data meet with matching chunks: nothing is multiplied up."""
    ds = _like_the_store(_roms_like())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("error", dask.array.core.PerformanceWarning)
        out = roms.to_depth(ds, META, [0.0, 20.0, 40.0])
        out["temp"].mean(["eta_rho", "xi_rho"]).compute()
        roms.nearest_depth_levels(ds, META, [10.0])["salt"].compute()


def _many_steps_store(nt: int = 12, ns: int = 40) -> xr.Dataset:
    """Return a restart-like store with *many* steps and a subchunked vertical.

    The shape that made dask warn "Increasing number of chunks by factor of 20" on the
    real run: per-step ``zeta`` chunks outer-multiplied with a subchunked ``sigma_r``.
    """
    sigma = np.linspace(-1.0, 0.0, ns)
    rng = np.random.default_rng(2)
    return xr.Dataset(
        {
            "zeta": (("time", "eta_rho", "xi_rho"), rng.normal(0, 0.4, (nt, NY, NX))),
            "h": (("eta_rho", "xi_rho"), 50.0 + 150.0 * rng.random((NY, NX))),
            "sigma_r": ("s_rho", sigma),
            "Cs_r": ("s_rho", sigma),
        }
    ).chunk({"time": 1, "s_rho": 2})


def test_depth_coordinate_is_not_built_as_an_outer_product_of_chunks(small_chunks):
    ds = _many_steps_store()
    with warnings.catch_warnings():
        warnings.simplefilter("error", dask.array.core.PerformanceWarning)
        out = roms.add_depth_coord(ds, META)

    z = out["z_rho"]
    assert z.chunksizes["s_rho"] == (40,), "the vertical is built in one piece"
    assert z.chunksizes["time"] == (1,) * 12
    biggest = max(z.chunksizes["eta_rho"]) * max(z.chunksizes["xi_rho"]) * 40 * 8
    assert biggest <= LIMIT, "and the frame is not one huge block per step"


def test_depth_coordinate_values_do_not_depend_on_chunking(small_chunks):
    ds = _many_steps_store()
    eager = roms.add_depth_coord(ds.compute(), META)["z_rho"]
    lazy = roms.add_depth_coord(ds, META)["z_rho"]
    np.testing.assert_array_equal(lazy.values, eager.values)
