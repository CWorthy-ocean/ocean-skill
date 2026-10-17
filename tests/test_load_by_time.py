"""``_load_by_time``: load a long record of large steps a few steps per compute.

The reported failure: a whole-domain mean of a year of 4 km, 100-level restarts,
loaded in one compute, let dask's local threaded scheduler start every step's reads
long before the reductions that finish them and hold them all -- about 96 GB.
:func:`ocean_skill.comparison.prepare_source` now loads through ``_load_by_time``,
which computes whole time chunks a batch at a time (about ``LOAD_BATCH_BYTES`` of
source per compute, never more than ``MAX_LOAD_BATCHES`` computes) into one numpy
array.

What is guarded: the result is exactly what ``da.load()`` returns (the batching may
only change how many steps are in flight, never a value or a label); batches are
whole chunks, so no chunk is computed twice; the cap on computes holds; and every
case that should not batch (numpy-backed, no time axis, small record) is one plain
load. Tiny arrays throughout -- the two limits are monkeypatched to force batching.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import dask
import dask.array as dsa
import dask.callbacks
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog, comparison
from ocean_skill.comparison import _bytes_per_step, _load_by_time, prepare_source

STEP = 100  # the step_bytes handed to _load_by_time; LOAD_BATCH_BYTES is in these units


def _lazy(dims, shape, chunks, dtype="float64", seed=0, coords=True):
    """Build a dask-backed DataArray with a time axis, attrs and optional lazy coords.

    ``coords=True`` adds a lazy non-dim coordinate along time and a lazy 2-D one over
    the first two non-time dims (needs at least two); ``False`` leaves only the index.
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape).astype(dtype)
    values[(0,) * len(shape)] = np.nan
    nt = shape[dims.index("time")]
    da = xr.DataArray(
        values,
        dims=dims,
        coords={"time": np.arange(nt)},
        attrs={"units": "K", "long_name": "synthetic"},
        name="temp",
    )
    if coords:
        a, b = [d for d in dims if d != "time"][:2]
        da = da.assign_coords(
            stamp=("time", dsa.from_array(np.arange(nt) * 2.0, chunks=chunks)),
            plane=(
                (a, b),
                dsa.from_array(rng.random((da.sizes[a], da.sizes[b])), chunks=1),
            ),
        )
    return da.chunk({d: (chunks if d == "time" else -1) for d in dims})


def _counting(counter):
    """Return a dask Callback counting every compute started inside its context."""

    def start(dsk):
        counter["n"] += 1

    return dask.callbacks.Callback(start=start)


def _identical_to_load(da, step_bytes=STEP):
    expected = da.copy(deep=False).load()
    got = _load_by_time(da.copy(deep=False), "time", step_bytes)
    xr.testing.assert_identical(got, expected)
    assert got.dtype == expected.dtype
    assert np.array_equal(got.values, expected.values, equal_nan=True)
    assert got.chunks is None
    for name, coord in got.coords.items():
        assert coord.chunks is None, f"coordinate {name!r} still lazy"
    return got


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dims", [("time", "y", "x"), ("z", "time", "y")])
@pytest.mark.parametrize("chunks", [1, (2, 3, 1, 4), 10])
def test_result_equals_a_plain_load(monkeypatch, dtype, dims, chunks):
    """Values (NaN-equal), dtype, dims, coords, attrs and name all match da.load()."""
    monkeypatch.setattr(comparison, "LOAD_BATCH_BYTES", 3 * STEP)  # 3 steps a batch
    shape = tuple(10 if d == "time" else 4 for d in dims)
    da = _lazy(dims, shape, chunks, dtype=dtype)
    got = _identical_to_load(da)
    assert got.attrs == {"units": "K", "long_name": "synthetic"}
    assert got.name == "temp"
    assert {"stamp", "plane"} <= set(got.coords)


@pytest.mark.parametrize(
    "limit,chunks,expected",
    [
        (3 * STEP, 1, 4),  # 3,3,3,1 steps
        (4 * STEP, (2, 3, 1, 4), 3),  # [2] [3,1] [4]: 2+3 would overshoot 4
        (5 * STEP, (2, 3, 1, 4), 2),  # [2,3] [1,4]
        (1 * STEP, (2, 3, 1, 4), 4),  # a chunk bigger than a batch goes whole
    ],
)
def test_batches_are_counted_and_made_of_whole_chunks(
    monkeypatch, limit, chunks, expected
):
    """The number of computes follows LOAD_BATCH_BYTES // step_bytes; no chunk twice."""
    monkeypatch.setattr(comparison, "LOAD_BATCH_BYTES", limit)
    base = _lazy(("time", "y"), (10, 3), chunks, coords=False)  # one compute a batch
    calls = Counter()

    def tally(block, block_info=None):
        calls[block_info[None]["chunk-location"]] += 1
        return block

    counted = base.copy(
        data=dsa.map_blocks(
            tally, base.data, dtype=base.dtype, meta=np.array((), base.dtype)
        )
    )
    computes = {"n": 0}
    with dask.config.set(scheduler="synchronous"), _counting(computes):
        got = _load_by_time(counted, "time", STEP)
    assert computes["n"] == expected
    nchunks = len(counted.chunks[0])
    assert len(calls) == nchunks
    assert set(calls.values()) == {1}, f"a chunk ran more than once: {calls}"
    np.testing.assert_array_equal(got.values, base.values)


@pytest.mark.parametrize("chunks,n", [(1, 20), (1, 21), ((2, 3, 1, 4), 10)])
def test_max_load_batches_caps_the_computes(monkeypatch, chunks, n):
    """A tiny batch size cannot split the load into more than MAX_LOAD_BATCHES."""
    monkeypatch.setattr(comparison, "LOAD_BATCH_BYTES", 1)
    monkeypatch.setattr(comparison, "MAX_LOAD_BATCHES", 2)
    da = _lazy(("time", "y"), (n, 3), chunks, coords=False)
    computes = {"n": 0}
    with _counting(computes):
        got = _load_by_time(da, "time", STEP)
    assert 1 < computes["n"] <= 2
    np.testing.assert_array_equal(got.values, da.copy().load().values)


def _plain(monkeypatch, da, tdim, step_bytes, expected_computes):
    monkeypatch.setattr(comparison, "LOAD_BATCH_BYTES", 2 * STEP)
    reference = da.copy(deep=False).load()
    computes = {"n": 0}
    with _counting(computes):
        got = _load_by_time(da, tdim, step_bytes)
    assert computes["n"] == expected_computes
    xr.testing.assert_identical(got, reference)


def test_fallbacks_do_one_plain_load(monkeypatch):
    """Nothing to batch -> one da.load(): numpy-backed, no tdim, no step size, small."""
    lazy = _lazy(("time", "y"), (10, 3), 1, coords=False)
    numpy_backed = lazy.copy(deep=False).load()
    _plain(monkeypatch, numpy_backed, "time", STEP, 0)  # nothing lazy to compute
    _plain(monkeypatch, lazy.copy(deep=False), None, STEP, 1)
    _plain(monkeypatch, lazy.copy(deep=False), "depth", STEP, 1)  # not a dim
    _plain(monkeypatch, lazy.copy(deep=False), "time", 0, 1)
    # n <= per_batch: ten steps, 2*STEP // 1 = 200 steps a batch.
    _plain(monkeypatch, lazy.copy(deep=False), "time", 1, 1)


def test_bytes_per_step_picks_the_largest_lazy_variable():
    ds = xr.Dataset(
        {
            "small": (("time", "y"), np.zeros((10, 5))),  # 40 B a step
            "big": (("time", "z", "y"), np.zeros((10, 4, 5))),  # 160 B a step
            "eager": (("time", "w"), np.zeros((10, 40))),  # numpy: ignored
            "static": (("v",), np.zeros(400)),  # lazy, no time: ignored
        }
    )
    ds["small"] = ds["small"].chunk({"time": 1})
    ds["big"] = ds["big"].chunk({"time": 1})
    ds["static"] = ds["static"].chunk({"v": 100})
    assert _bytes_per_step(ds, "time") == 160
    assert _bytes_per_step(ds, "nope") == 0


def test_bytes_per_step_is_zero_without_a_lazy_variable():
    assert _bytes_per_step(pd.DataFrame({"a": [1.0, 2.0]}), "time") == 0
    assert _bytes_per_step(xr.Dataset(), "time") == 0
    eager = xr.Dataset({"a": (("time", "y"), np.zeros((4, 3)))})
    assert _bytes_per_step(eager, "time") == 0


# -- through prepare_source ---------------------------------------------------------

META = {"model": "roms", "vertical": {"hc": 20.0, "s_dim": "s_rho"}}


def _roms_series(nt=8, ns=6, ny=5, nx=7) -> xr.Dataset:
    """Build a ROMS-shaped tracer, one time step per chunk like a history file."""
    sigma = np.linspace(-1.0, 0.0, ns)
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "temp": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                rng.normal(15.0, 2.0, (nt, ns, ny, nx)),
            ),
            "h": (("eta_rho", "xi_rho"), 50.0 + 150.0 * rng.random((ny, nx))),
            "sigma_r": ("s_rho", sigma),
            "Cs_r": ("s_rho", sigma),
        },
        coords={
            "time": np.arange(nt),
            "lon": (("eta_rho", "xi_rho"), np.tile(np.linspace(-98, -80, nx), (ny, 1))),
            "lat": (
                ("eta_rho", "xi_rho"),
                np.tile(np.linspace(18, 31, ny)[:, None], (1, nx)),
            ),
        },
    )
    ds = ds.chunk({"time": 1})
    ds["sigma_r"] = ds["sigma_r"].chunk({"s_rho": -1})
    ds["Cs_r"] = ds["Cs_r"].chunk({"s_rho": -1})
    return ds


def test_prepare_source_batched_equals_one_compute(monkeypatch):
    """A whole-domain mean loaded in tiny batches equals the single-compute load.

    The synchronous scheduler on both sides: the area-weighted mean is not
    bit-reproducible across threads, and this is about the batching alone.
    """
    ds = _roms_series()
    monkeypatch.setattr(osk, "read", lambda name, **kw: ds)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata=META))
    seen = []
    real = comparison._load_by_time

    def spy(da, tdim, step_bytes):
        seen.append(step_bytes)
        return real(da, tdim, step_bytes)

    monkeypatch.setattr(comparison, "_load_by_time", spy)

    def run():
        return prepare_source(
            "his",
            "temp",
            {"depth": ["surface", 10.0]},
            {"lon": "mean", "lat": "mean"},
            use_cache=False,
        )

    with dask.config.set(scheduler="synchronous"):
        monkeypatch.setattr(comparison, "LOAD_BATCH_BYTES", 1 << 40)
        whole, depth_whole = run()
        computes = {"n": 0}
        monkeypatch.setattr(comparison, "LOAD_BATCH_BYTES", 1)
        with _counting(computes):
            batched, depth_batched = run()
    assert seen and all(s > 0 for s in seen)
    assert computes["n"] > 1, "the tiny batch size never split the load"
    xr.testing.assert_identical(batched, whole)
    assert depth_batched == depth_whole
