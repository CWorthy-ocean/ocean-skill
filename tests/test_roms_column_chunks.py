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


# --- the frame is built inside the kernel task, not held whole ----------------------
#
# The depth match used to hand its kernel the lazy 3-D frame (z_rho) as a dask array of
# its own, so the scheduler computed every step's frame early and held them all until
# the slow data reads caught up. Now each task rebuilds its own block from zeta/h
# (``_frame_recipe`` / ``_rebuilt_frame``) wherever that is provably the same frame.
# These tests pin that the rebuilt path is bitwise the old one, is actually taken, and
# falls back where it cannot be proven.

FRAMES = {
    "fixed": {"origin": "fixed", "datum_z_m": 0.0, "source": "declared"},
    "fixed_datum": {"origin": "fixed", "datum_z_m": 1.5, "source": "declared"},
    "surface": {"origin": "surface", "source": "declared"},
}
# 0 m, interior depths, one deeper than most columns (h is 50-200 m)
TARGETS = [0.0, 7.5, 33.0, 90.0, 180.0]


def _meta(vtransform: int) -> dict:
    return {"vertical": {"hc": 20.0, "s_dim": "s_rho", "Vtransform": vtransform}}


def _roms_with_land(float32_vertical: bool = False) -> xr.Dataset:
    """Return :func:`_roms_like` with land (NaN zeta) and optional float32 sigma."""
    ds = _roms_like()
    zeta = ds["zeta"].values.copy()
    zeta[:, :6, :9] = np.nan  # a corner of land, every step
    zeta[1, 20, 30] = np.nan  # and one column masked at a single step
    ds["zeta"] = (ds["zeta"].dims, zeta)
    if float32_vertical:
        for name in ("sigma_r", "Cs_r"):
            ds[name] = ds[name].astype("float32")
    return ds


def _run(fn, ds, meta, convention, **kwargs) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the fixed-origin notes
        return fn(ds, meta, TARGETS, convention=convention, **kwargs).compute()


def _count_rebuilt(monkeypatch) -> list[int]:
    """Count calls of ``_rebuilt_frame`` (the in-task frame build)."""
    calls: list[int] = []
    original = roms._rebuilt_frame

    def wrapper(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(roms, "_rebuilt_frame", wrapper)
    return calls


def _no_recipe(monkeypatch) -> None:
    """Force the old path: the whole frame handed to the kernel."""
    monkeypatch.setattr(roms, "_frame_recipe", lambda *a, **k: None)


def _assert_identical(got: xr.Dataset, want: xr.Dataset) -> None:
    assert dict(got.sizes) == dict(want.sizes)
    assert set(got.data_vars) == set(want.data_vars)
    assert set(got.coords) == set(want.coords)
    for name in want.data_vars:
        assert got[name].dims == want[name].dims
        assert got[name].dtype == want[name].dtype
        np.testing.assert_array_equal(got[name].values, want[name].values)
        assert got[name].attrs == want[name].attrs
    for name in want.coords:
        np.testing.assert_array_equal(got[name].values, want[name].values)
        assert got[name].attrs == want[name].attrs
    assert got.attrs == want.attrs


@pytest.mark.parametrize("mode", ["interp", "nearest"])
@pytest.mark.parametrize("frame", list(FRAMES))
@pytest.mark.parametrize("vtransform", [1, 2])
def test_the_rebuilt_frame_is_bitwise_the_whole_frame_path(
    small_chunks, monkeypatch, mode, frame, vtransform
):
    fn = roms.to_depth if mode == "interp" else roms.nearest_depth_levels
    # float32 stretching vectors in one of the two Vtransforms, as the real store has
    ds = _like_the_store(_roms_with_land(float32_vertical=vtransform == 1))
    meta, convention = _meta(vtransform), FRAMES[frame]

    calls = _count_rebuilt(monkeypatch)
    got = _run(fn, ds, meta, convention)
    assert calls, "the frame was never built inside the task"

    _no_recipe(monkeypatch)
    calls.clear()
    want = _run(fn, ds, meta, convention)
    assert not calls

    assert np.isnan(want["temp"].values).any(), "the land must reach the result"
    _assert_identical(got, want)


def test_the_recipe_is_taken_and_the_frame_is_not_in_the_graph(
    small_chunks, monkeypatch
):
    ds = roms.add_depth_coord(_like_the_store(_roms_with_land()), META)
    z_name = ds["z_rho"].data.name
    convention = FRAMES["surface"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = roms.to_depth(ds, META, TARGETS, convention=convention)
        layers = set(out["temp"].data.__dask_graph__().layers)
        assert z_name not in layers, "the 3-D frame is still a layer of the match"
        # nor does anything that is built from it: only zeta/h feed the kernel
        graph = out["temp"].data.__dask_graph__()
        assert all(z_name not in deps for deps in graph.dependencies.values())

        # the old path does carry it (directly, or through its rechunk)
        _no_recipe(monkeypatch)
        old = roms.to_depth(ds, META, TARGETS, convention=convention)
        old_graph = old["temp"].data.__dask_graph__()
        assert z_name in old_graph.layers or any(
            z_name in deps for deps in old_graph.dependencies.values()
        )


def test_the_frame_recipe_of_a_plain_dataset_has_what_the_task_needs():
    ds = roms.add_depth_coord(_roms_like().chunk({"time": 1}), META)

    got = roms._frame_recipe(ds, META, "s_rho", "surface", 0.0)

    assert got is not None
    zeta, h, recipe = got
    assert zeta.dims == ("time", "eta_rho", "xi_rho")
    assert h.dims == ("eta_rho", "xi_rho")
    assert set(recipe) == {"sigma", "Cs", "hc", "vtransform", "origin", "datum"}
    assert recipe["origin"] == "surface"
    assert recipe["sigma"].shape == (NS,)


def test_a_frame_that_is_not_provably_the_datasets_own_has_no_recipe():
    base = _roms_like().chunk({"time": 1})
    ds = roms.add_depth_coord(base, META)
    args = (META, "s_rho", "fixed", 0.0)

    assert roms._frame_recipe(ds, *args) is not None
    # numpy-backed everything: nothing to schedule
    eager = roms.add_depth_coord(_roms_like(), META)
    assert roms._frame_recipe(eager, *args) is None
    # sliced after the frame was attached: z_rho no longer what the ds would build
    assert roms._frame_recipe(ds.isel(time=slice(0, 2)), *args) is None
    # altered some other way
    altered = ds.assign_coords(z_rho=ds["z_rho"] + 0.0)
    assert roms._frame_recipe(altered, *args) is None
    # no stretching vector to rebuild from
    assert roms._frame_recipe(ds.drop_vars("sigma_r"), *args) is None
    assert roms._frame_recipe(ds.drop_vars("Cs_r"), *args) is None
    assert roms._frame_recipe(ds.drop_vars("h"), *args) is None
    # no frame at all
    assert roms._frame_recipe(ds.drop_vars("z_rho"), *args) is None


def test_a_sliced_dataset_still_matches_through_the_whole_frame(
    small_chunks, monkeypatch
):
    ds = roms.add_depth_coord(_like_the_store(_roms_with_land()), META)
    sliced = ds.isel(time=slice(0, 2))
    assert roms._frame_recipe(sliced, META, "s_rho", "fixed", 0.0) is None
    calls = _count_rebuilt(monkeypatch)

    got = _run(roms.to_depth, sliced, META, FRAMES["fixed"])
    assert not calls

    # and equals the match of the same steps taken from the whole dataset
    want = _run(roms.to_depth, ds, META, FRAMES["fixed"]).isel(time=slice(0, 2))
    np.testing.assert_array_equal(got["temp"].values, want["temp"].values)


@pytest.mark.parametrize("frame", list(FRAMES))
def test_a_reference_time_never_builds_the_frame_in_the_task(
    small_chunks, monkeypatch, frame
):
    ds = _like_the_store(_roms_with_land())
    convention = FRAMES[frame]

    calls = _count_rebuilt(monkeypatch)
    got = _run(roms.nearest_depth_levels, ds, META, convention, ref_time=1)
    assert not calls, "the reference-time frame is loaded eagerly, not rebuilt"

    _no_recipe(monkeypatch)
    want = _run(roms.nearest_depth_levels, ds, META, convention, ref_time=1)
    _assert_identical(got, want)
