"""``build.open_reference``: kerchunk references open with one dask layer.

ocean-skill opened kerchunk references through intake with
``engine="kerchunk", chunks={}``. kerchunk's backend already hands back dask arrays (it
calls ``xr.open_zarr``), so xarray wrapped them in a *second* dask layer, and every
outer task then ``compute()``-d the inner array: about a second per storage chunk on a
long record, and a read of a few time steps paid for every chunk the outer task spanned.
:func:`ocean_skill.build.open_reference` opens with one layer instead
(``open_dataset_options={"chunks": None}`` inside the engine), sizes the dask chunks in
whole time steps by bytes (:data:`ocean_skill.build.KERCHUNK_CHUNK_BYTES`), and passes
``storage_options={"remote_protocol": ...}`` so fsspec does not scan the reference to
guess the protocol (a scan that raises ``TypeError: boolean value of NA is ambiguous``
on a parquet reference under pandas 3, depending on ``PYTHONHASHSEED``).

The references are tiny and built with :func:`ocean_skill.build.make_kerchunk` from a
few compressed, chunked netCDF files, in both ROMS layouts: classic (record dimension
``ocean_time``) and UCLA (record dimension ``time``, ``ocean_time`` a coordinate on
it).
No network: the ``https`` reference is only ever inspected, never opened.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import yaml
from fsspec.implementations.local import LocalFileSystem

import ocean_skill as osk
from ocean_skill import build, catalog

NS, NY, NX = 4, 6, 8
NFILES, STEPS_PER_FILE = 3, 2
NSTEPS = NFILES * STEPS_PER_FILE
# Storage chunk of ``temp`` as ``(s_rho, eta, xi)``: 2 x 2 x 2 = 8 chunks per step.
SUB = (2, 3, 4)
LAYOUTS = {"classic": "ocean_time", "ucla": "time"}
# What a bare ``xr.open_dataset`` needs so as not to hit the parquet NA flake.
FILE_PROTOCOL = {"remote_protocol": "file"}


def _part(
    path: Path,
    layout: str,
    first_step: int,
    nt: int = STEPS_PER_FILE,
    *,
    t_chunk: int = 1,
    temp_chunks: tuple[int, int, int] = SUB,
) -> Path:
    """Write one compressed, chunked ROMS-like file of ``nt`` records.

    ``temp`` is 4-D ``(time, s_rho, eta_rho, xi_rho)`` stored in chunks of
    ``(t_chunk, *temp_chunks)``; ``zeta`` is 3-D ``(time, eta_rho, xi_rho)``. The values
    are a function of the global step alone, so any read that mixes records up shows.
    """
    dim = LAYOUTS[layout]
    rng = np.random.default_rng(first_step)
    time = (first_step + np.arange(nt)) * 3600.0
    ds = xr.Dataset(
        {
            "temp": (
                (dim, "s_rho", "eta_rho", "xi_rho"),
                rng.normal(15.0, 2.0, (nt, NS, NY, NX)),
            ),
            "zeta": ((dim, "eta_rho", "xi_rho"), rng.normal(0.0, 1.0, (nt, NY, NX))),
        },
        coords={"ocean_time": (dim, time, {"units": "seconds since 1970-01-01"})},
    )
    ds["temp"][0, 0, 0, 0] = np.nan  # a masked cell, to keep the NaN comparison honest
    enc = {
        "temp": {
            "zlib": True,
            "chunksizes": (min(t_chunk, nt), *temp_chunks),
        },
        "zeta": {"zlib": True, "chunksizes": (min(t_chunk, nt), 3, 4)},
    }
    ds.to_netcdf(path, encoding=enc)
    return path


def _make_ref(
    tmp_path: Path,
    layout: str,
    *,
    suffix: str = ".json",
    nfiles: int = NFILES,
    nt: int = STEPS_PER_FILE,
    **part_kw,
) -> str:
    """Build a reference (as a path string) over ``nfiles`` files of ``nt`` records."""
    files = [
        _part(tmp_path / f"his_{i}.nc", layout, i * nt, nt, **part_kw)
        for i in range(nfiles)
    ]
    return str(build.make_kerchunk(files, out=tmp_path / f"refs{suffix}"))


@pytest.fixture(params=sorted(LAYOUTS))
def layout(request) -> str:
    return request.param


@pytest.fixture
def ref(tmp_path, layout) -> str:
    """Build a JSON reference over six one-record-chunk steps, in either layout."""
    return _make_ref(tmp_path, layout)


def _direct_open(path, **kw) -> xr.Dataset:
    """Open the old way: ``engine="kerchunk", chunks={}``, which nests dask."""
    return xr.open_dataset(
        str(path),
        engine="kerchunk",
        chunks={},
        decode_times=False,
        storage_options=FILE_PROTOCOL,
        **kw,
    )


def _wrapped_arrays(var: xr.DataArray) -> list[type]:
    """Return the chain of array types under the dask graph's ``original`` layer.

    The ``original-...`` key is where xarray hands dask the lazily-indexed backend
    array; following ``.array`` from it lists what a task wraps. A
    ``dask.array.Array`` in that chain is a second dask layer, whose ``compute()`` each
    outer task pays for.
    """
    graph = var.data.dask
    key = next(
        k
        for k in graph
        if str(k[0] if isinstance(k, tuple) else k).startswith("original")
    )
    arr, chain = graph[key], []
    while arr is not None:
        chain.append(type(arr))
        arr = getattr(arr, "array", None)
    return chain


def _per_step_bytes(ds: xr.Dataset, dim: str) -> int:
    """Largest storage-chunk bytes per record over the variables that have ``dim``."""
    sizes = []
    for var in ds.data_vars.values():
        chunks = var.encoding.get("chunks")
        if dim in var.dims and chunks:
            stored_t = chunks[var.dims.index(dim)]
            sizes.append(int(np.prod(chunks)) * var.dtype.itemsize // stored_t)
    return max(sizes)


# --------------------------------------------------------------------- one dask layer


def test_open_reference_builds_one_dask_layer(ref, layout):
    """No ``dask.array.Array`` hides under the graph's ``original`` layer."""
    ds = build.open_reference(ref)

    for name in ("temp", "zeta"):
        assert da.Array not in _wrapped_arrays(ds[name]), name


def test_the_direct_kerchunk_open_nests_a_second_dask_layer(ref):
    """CONTRAST (documents the bug): ``engine="kerchunk", chunks={}`` nests dask.

    This is not ``open_reference``: it is the open ocean-skill used to do, kept here so
    the test above is shown to detect the problem it guards against.
    """
    ds = _direct_open(ref)

    assert da.Array in _wrapped_arrays(ds["temp"])


# ------------------------------------------------------------------ byte-sized chunks


def test_time_chunks_follow_the_byte_target(tmp_path, layout, monkeypatch):
    """A target of three steps' worth of the largest variable gives 3-step chunks."""
    ref = _make_ref(tmp_path, layout, temp_chunks=(NS, NY, NX))
    dim = LAYOUTS[layout]
    per_step = _per_step_bytes(_direct_open(ref), dim)
    monkeypatch.setattr(build, "KERCHUNK_CHUNK_BYTES", 3 * per_step)

    ds = build.open_reference(ref)

    assert ds["temp"].chunks[ds["temp"].dims.index(dim)] == (3, 3)
    assert ds["zeta"].chunks[0] == (3, 3), "every variable on the dim shares the size"


def test_a_huge_byte_target_is_clipped_to_the_record_length(tmp_path, layout):
    """The default 256 MiB is far more than six tiny steps: one chunk spans them all."""
    ref = _make_ref(tmp_path, layout)
    dim = LAYOUTS[layout]

    ds = build.open_reference(ref)

    assert ds["temp"].chunks[0] == (NSTEPS,)
    assert build._time_chunks(ds) == {dim: NSTEPS}


def test_a_target_below_one_step_still_gives_whole_stored_time_chunks(
    tmp_path, layout, monkeypatch
):
    """The floor is the stored time chunk: dask never splits a chunk below it."""
    ref = _make_ref(tmp_path, layout)
    monkeypatch.setattr(build, "KERCHUNK_CHUNK_BYTES", 1)

    ds = build.open_reference(ref)

    assert ds["temp"].chunks[0] == (1,) * NSTEPS


def test_the_time_chunk_is_a_multiple_of_the_stored_time_chunk(
    tmp_path, layout, monkeypatch
):
    """Files storing two records per chunk give dask chunks of 2, 4, ... steps.

    Three files of four records, two to a chunk: a target worth three steps rounds down
    to the largest multiple of the stored chunk (two), never to three, which would make
    every task decompress half of a second storage chunk it then throws away.
    """
    ref = _make_ref(tmp_path, layout, nt=4, t_chunk=2, temp_chunks=(NS, NY, NX))
    dim = LAYOUTS[layout]
    per_step = _per_step_bytes(_direct_open(ref), dim)
    monkeypatch.setattr(build, "KERCHUNK_CHUNK_BYTES", 3 * per_step)

    ds = build.open_reference(ref)

    assert ds["temp"].chunks[0] == (2,) * 6
    assert build._time_chunks(ds) == {dim: 2}
    monkeypatch.setattr(build, "KERCHUNK_CHUNK_BYTES", 5 * per_step)
    assert build._time_chunks(ds) == {dim: 4}


def test_a_dataset_without_a_time_dimension_has_no_time_chunks():
    """Nothing to size: ``_time_chunks`` returns an empty dict, not an error."""
    ds = xr.Dataset({"h": (("eta_rho", "xi_rho"), np.zeros((NY, NX)))})

    assert build._time_chunks(ds) == {}


# ---------------------------------------------------------------- no read amplification


def test_reading_a_small_box_reads_only_the_storage_chunks_it_touches(
    tmp_path, layout, monkeypatch
):
    """Three steps of one horizontal chunk cost 3 x (vertical chunks) file reads.

    Dask chunks now span several time steps (here all six), so a slice must reach the
    zarr array and fetch only the storage chunks it needs, rather than load the whole
    dask chunk and cut it afterwards: that would be 6 x 8 reads here, not 6. (The old
    nested open does not amplify bytes at this size -- its cost was per-task overhead --
    so this guards the new, larger chunks, not the old bug.) The count is of
    ``cat_file`` calls on the local filesystem during ``.values`` alone (the open's own
    reads are not counted). ``temp`` is stored in 2 vertical x 2 x 2 chunks per step;
    the box lies inside one horizontal chunk and takes the whole vertical, so it touches
    2 chunks per step, over 3 steps.
    """
    ref = _make_ref(tmp_path, layout)
    dim = LAYOUTS[layout]
    calls: list[tuple] = []
    original = LocalFileSystem.cat_file

    def counting(self, *args, **kw):
        calls.append(args)
        return original(self, *args, **kw)

    # Patched before the open: the reference filesystem binds the local filesystem's
    # methods when it is built.
    monkeypatch.setattr(LocalFileSystem, "cat_file", counting)
    ds = build.open_reference(ref)
    assert ds["temp"].chunks[0] == (NSTEPS,), "one dask chunk spans all six steps"
    opened = len(calls)

    box = ds["temp"].isel(
        {dim: slice(1, 4), "eta_rho": slice(0, 2), "xi_rho": slice(1, 3)}
    )
    values = box.values

    assert values.shape == (3, NS, 2, 2)
    assert len(calls) - opened == 3 * 2


# ------------------------------------------------------------------------ same values


def test_open_reference_gives_the_same_values_as_the_direct_open(ref):
    """Only the dask layout differs: every variable's numbers are identical."""
    new = build.open_reference(ref, decode_times=False)
    old = _direct_open(ref)

    assert set(new.variables) == set(old.variables)
    for name in old.variables:
        assert new[name].dims == old[name].dims, name
        np.testing.assert_array_equal(new[name].values, old[name].values, err_msg=name)


# ------------------------------------------------------------------------ chunks=


def test_chunks_none_opens_lazily_without_dask(ref):
    """``chunks=None``: no dask arrays at all (a lazily-indexed backend array)."""
    ds = build.open_reference(ref, chunks=None)

    assert ds["temp"].chunks is None
    assert ds["temp"].shape == (NSTEPS, NS, NY, NX)


def test_an_explicit_chunks_dict_is_honoured(ref, layout):
    """``chunks={time: 1}`` is used as given, not replaced by the byte-sized size."""
    dim = LAYOUTS[layout]

    ds = build.open_reference(ref, chunks={dim: 1})

    assert ds["temp"].chunks[0] == (1,) * NSTEPS
    assert da.Array not in _wrapped_arrays(ds["temp"]), "still one dask layer"


def test_chunks_auto_and_an_empty_dict_both_mean_byte_sized_time_chunks(ref, layout):
    """The catalog's ``chunks: {}`` and the default ``"auto"`` size time alike."""
    dim = LAYOUTS[layout]

    auto = build.open_reference(ref, chunks="auto")
    empty = build.open_reference(ref, chunks={})

    assert auto["temp"].chunks == empty["temp"].chunks
    assert auto["temp"].chunks[0][0] == build._time_chunks(auto)[dim]


def test_keyword_arguments_reach_xarray(ref):
    """``decode_times=False``, an ``xr.open_dataset`` option, leaves numeric times."""
    ds = build.open_reference(ref, decode_times=False)
    decoded = build.open_reference(ref, decode_times=True)

    assert np.issubdtype(ds["ocean_time"].dtype, np.floating)
    assert np.issubdtype(decoded["ocean_time"].dtype, np.datetime64)


# --------------------------------------------------------------- _reference_protocol


@pytest.mark.parametrize("suffix", [".json", ".parquet"])
def test_a_local_reference_has_the_file_protocol(tmp_path, layout, suffix):
    ref = _make_ref(tmp_path, layout, suffix=suffix)

    assert build._reference_protocol(ref) == "file"
    assert build._reference_protocol(Path(ref)) == "file"


def test_a_reference_to_https_urls_has_the_https_protocol(tmp_path):
    """Read from the reference's own targets; nothing is fetched."""
    path = tmp_path / "remote.json"
    refs = {
        "version": 1,
        "refs": {
            ".zgroup": json.dumps({"zarr_format": 2}),
            "x/.zarray": json.dumps(
                {
                    "zarr_format": 2,
                    "shape": [10],
                    "chunks": [10],
                    "dtype": "<f8",
                    "compressor": None,
                    "fill_value": None,
                    "filters": None,
                    "order": "C",
                }
            ),
            "x/0": ["https://example.com/a.nc", 0, 10],
        },
    }
    path.write_text(json.dumps(refs))

    assert build._reference_protocol(path) == "https"


def test_an_unreadable_reference_has_no_protocol_and_does_not_raise(tmp_path):
    """Undeterminable is ``None``: a missing file, or one that is not a reference."""
    junk = tmp_path / "junk.json"
    junk.write_text("not json")

    assert build._reference_protocol(tmp_path / "missing.json") is None
    assert build._reference_protocol(junk) is None


def test_a_callers_remote_protocol_wins_over_the_detected_one(ref, monkeypatch):
    """Detection is a default: ``storage_options`` from the caller overrides it."""
    monkeypatch.setattr(build, "_reference_protocol", lambda url: "https")
    seen: list[dict] = []
    real_open = xr.open_dataset

    def spy(*args, **kw):
        seen.append(kw.get("storage_options"))
        return real_open(*args, **kw)

    monkeypatch.setattr(xr, "open_dataset", spy)

    # the detected "https" would fail to read the local file; the caller's "file" works
    ds = build.open_reference(ref, storage_options={"remote_protocol": "file"})
    assert ds["zeta"].isel({ds["zeta"].dims[0]: 0}).shape == (NY, NX)
    assert seen[0]["remote_protocol"] == "file"

    seen.clear()
    build.open_reference(ref, storage_options={"anon": True})
    assert seen[0] == {"remote_protocol": "https", "anon": True}, (
        "other caller options are merged with the detected protocol"
    )


# ---------------------------------------------------- through a catalog, end to end


def test_a_catalog_entry_reads_with_one_dask_layer_and_an_unchanged_definition(
    tmp_path, layout, isolated_catalogs
):
    """``build_catalog`` -> ``osk.read``: the new open is a read-time change only.

    The YAML still says ``chunks: {}`` (the reader definition is what the cache is keyed
    on), so the entry's fingerprint is identical before and after a read, and caches
    written under the old open stay valid. What differs is the dataset that comes back:
    one dask layer, with time chunks spanning more than one step.
    """
    ref = _make_ref(tmp_path, layout)
    out = build.build_catalog(
        {"run": str(ref)}, isolated_catalogs / "kc.yaml", title="kerchunk catalog"
    )
    doc = yaml.safe_load(Path(out).read_text())
    before = catalog.fingerprint("kc:run")
    assert before

    ds = osk.read("kc:run")

    assert len(ds.data_vars) == 2, list(ds.data_vars)
    for var in ds.data_vars.values():
        assert var.dims[0] == LAYOUTS[layout], var.dims
        assert da.Array not in _wrapped_arrays(var), var.name
        assert max(var.chunks[0]) > 1, var.name
    assert catalog.fingerprint("kc:run") == before
    text = Path(out).read_text()
    assert yaml.safe_load(text) == doc, "reading must not rewrite the catalog"
    assert "chunks: {}" in text, "the catalog definition still says chunks: {}"


# ------------------------------------------------------------------- parquet NA flake


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_parquet_reference_opens_under_every_hash_seed(tmp_path, seed):
    """No ``boolean value of NA is ambiguous``, whatever ``PYTHONHASHSEED``.

    fsspec's reference scan, run when no ``remote_protocol`` is given, trips over a
    pandas NA in a parquet reference only for some string-hash orderings. Passing the
    protocol skips the scan, so ``open_reference`` is immune; run in a fresh
    interpreter per seed because the hash seed is fixed at startup.
    """
    ref = _make_ref(tmp_path, "classic", suffix=".parquet", nfiles=2)
    code = (
        "import sys\n"
        "from ocean_skill.build import open_reference\n"
        "ds = open_reference(sys.argv[1])\n"
        "v = float(ds['zeta'].isel(ocean_time=0, eta_rho=0, xi_rho=0))\n"
        "assert v == v\n"
    )
    root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": root}

    proc = subprocess.run(
        [sys.executable, "-c", code, str(ref)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 0, proc.stderr[-2000:]
