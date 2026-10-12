# When is a kerchunk reference faster than reading the files?

During the CIOFS3 paper reruns, hourly model time series at moorings were read
directly from the daily netCDF files, because reading them through
`ciofs3_v5_kerchunk.parquet` was about 30× slower (about 3 s against 0.1 s per storage
chunk). This page measures where that gap came from, when a reference really is faster
or slower than the files it indexes, and what ocean-skill should do about it.

## Summary

- **The 30× was not kerchunk. It was dask inside dask.**
  - ocean-skill opens a reference with `xr.open_dataset(ref, engine="kerchunk", chunks={})`.
  - kerchunk's xarray backend builds its dataset with `xr.open_zarr`, which already
    returns dask arrays. xarray then wraps those in a *second* dask layer.
  - So every outer task (one per storage chunk) slices and `compute()`s the *inner*
    dask array, and that array has one chunk per time step of the whole record.
  - Each inner compute optimizes a graph over all of that array's chunks: ~1.8 million
    for `temp` in a 26-year hourly store.
  - Measured, that costs **1.1 s per storage chunk on lakes02** and **0.3 s on a
    laptop**, however few cells are requested.
- **With one dask layer, or none, the same reference is the fastest way to read a
  point time series.** On lakes02, reading the real CIOFS3 record:
  - through the reference: **18–42 ms per chunk cold** (17–26 ms warm);
  - with `netCDF4` straight from the daily files: **92–97 ms** (38–43 ms warm).
  - The bytes and the decompression are identical. The reference wins because zarr
    fetches and decodes chunks concurrently, while a netCDF4 reader handles them one at
    a time.
- **On high-latency storage, the number of reads in flight decides the speed.** anvil's
  GPFS takes ~0.45 s per cold 1 MB read. There, one dask layer with chunks spanning
  ~24 time steps reached **28 ms per chunk**, against:
  - 90–100 ms for plain zarr;
  - ~200 ms for one dask task per storage chunk;
  - 127–236 ms for netCDF4.
- **Opening is where a reference wins outright.** The full 26-year CIOFS3 reference opens
  in 0.2–1.5 s (5.5 s the way ocean-skill opened it before the fix). Opening the 9,374 daily files it points at costs about 0.15 s each,
  roughly 23 minutes before any data is read. On S3, one file takes 11 s to open, and a
  day of hourly data reads 9–38× faster through a reference.
- **The xarray version does not matter here; the dask array implementation does.**
  - xarray 2026.4 and 2026.9 behave the same.
  - Opting in to Dask's expression arrays (`dask-array`, `dask_array.xarray.register()`)
    makes the nested path 30× faster, because those arrays optimize without walking
    the whole graph. That is a workaround, not the fix.
- **For ocean-skill**, the fix is to open kerchunk references with **one** dask layer
  (chunked at the storage chunks by default; multi-step time blocks are opt-in, see the
  [correction](#correction-2026-10-12)), and to pass `storage_options` through for
  kerchunk entries, which also fixes the parquet `boolean value of NA is ambiguous`
  crash. Both are now done in `build.open_reference`. See
  [Recommendations](#recommendations-for-ocean-skill).

## What a reference changes, and what it does not

A kerchunk reference is a list of `(file, offset, length)` byte ranges, one per
storage chunk of the original files, plus the zarr metadata (shape, chunk shape,
codecs) needed to interpret them. It does **not** rechunk or recompress anything. The
CIOFS3 hindcast stores `temp` as `(1, 15, 522, 362)` chunks with shuffle + zlib-1
(about 3.5 MB compressed, 11 MB decoded), and so does the reference.

So for any request, both routes do the same work:
- read the same compressed chunks;
- decompress all of each one, even when only an 11 × 11 column is wanted.

This is why a single grid cell over a year costs as much as a 200 × 200 box inside the
same chunks: the unit of work is the storage chunk.

What differs between the two routes is everything around that work:

| | reading the files (netCDF4 / h5netcdf) | reading through a reference (fsspec → zarr) |
|---|---|---|
| open | parse each file's HDF5 metadata | read one JSON/parquet index |
| per chunk | HDF5 C library: B-tree lookup, read, decode | reference lookup, byte-range `cat`, numcodecs decode |
| concurrency | one chunk at a time per reader | zarr fetches and decodes many chunks at once (async) |
| dask on top | one layer when you ask for it | depends on how it is opened, which is the problem |

## How it was measured

[`docs/benchmarks/kerchunk_access.py`](benchmarks/kerchunk_access.py) is a standalone
script. It imports nothing from ocean-skill. It times one request through each layer of
the read path:

| path | what it does |
|---|---|
| P0 | raw byte ranges: open, seek and read per chunk |
| P0d | P0, then numcodecs decode, so the floor is bytes + decompression |
| P1 | `netCDF4.Dataset`, file by file, on the files the reference points at |
| P2 / P2m | `xr.open_dataset` per file (no dask) / `xr.open_mfdataset` over the window's files |
| P3 | `zarr.open_array` on the reference store (no xarray, no dask) |
| P4z | xarray `engine="zarr"` on the reference store, `chunks=None` (no dask) |
| P5z | xarray `engine="zarr"` on the reference store, `chunks={}` (one dask layer) |
| P5o | xarray `engine="kerchunk"`, `chunks={}`, `open_dataset_options={"chunks": None}` (one dask layer) |
| P4 | xarray `engine="kerchunk"`, `chunks=None` (the backend's own single dask layer) |
| **P5** | xarray `engine="kerchunk"`, `chunks={}`, **what ocean-skill did before the fix** (two dask layers) |
| P6_N | P5 with `chunks={time: N}` (two layers, fewer outer tasks) |
| P6z_N | xarray `engine="zarr"`, `chunks={time: N}` (one dask layer, N steps per task) |

Every path returns arrays identical to every other (checked per run).

Main request (R1), modelled on a mooring:
- `temp` at all 30 levels in an 11 × 11 box, plus `zeta`;
- over 24 or 240 hourly steps, i.e. 72 or 720 storage chunks.

"Cold" means a window not read before in that run; "warm" means the same window read
again.

| | E1 laptop | E2 lakes02 | E3 lakes02 → S3 | E4 anvil |
|---|---|---|---|---|
| storage | local SSD | `/mnt/vault`, GlusterFS FUSE | AWS S3 (NOAA NODD), ~5.7 MB/s per stream, ~0.3 s per request | GPFS (`/anvil/projects`), ~0.45 s per cold 1 MB read; dedicated `standard` node |
| data | synthetic CIOFS-shaped files (same shape, chunks, codecs; 3.1 MB/chunk compressed), 4 days cycled out to a 224,976-step reference | real CIOFS3 hindcast: the 26-year `ciofs3_v5_kerchunk.parquet` (109 variables, 224,976 steps, built with kerchunk's `MultiZarrToZarr`), plus a 1-year 2017 reference | CIOFS operational nowcast: 48 hourly files, **uncompressed** HDF5, chunks `(1, 10, 348, 242)` | Iceland/Hvalfjörður `his.parquet` from `make_kerchunk` (daily files, 7,272 steps, `temp` chunks `(1, 20, 172, 172)` zlib-1, 1.1 MB/chunk); PACMED `output_cdrgas` (143 daily CDF5 files, uncompressed 2-D fields, 14 MB each) |
| software | xarray 2026.4.0, dask 2026.3.0, zarr 3.2.1, kerchunk 0.2.10, pandas 3.0.5 | xarray 2025.8.0, dask 2025.7.0, zarr 3.1.2, kerchunk 0.2.9, pandas 2.3.2 | as E2 | xarray 2026.7.0, dask 2026.7.1, zarr 3.3.0, kerchunk 0.2.10, pandas 2.2.2 |

Dask ran with 4 threads unless noted.

## Results

### Point time series, 240 hourly steps (720 storage chunks), ms per chunk

| path | E1 laptop (warm) | E2 lakes02, 26-yr store, cold / warm | E2 lakes02, 1-yr store, cold / warm | E4 anvil, Iceland, cold / warm |
|---|---|---|---|---|
| P0 raw bytes | 0.4 | 65 / 3.3 | — | 641 / 620 |
| P0d bytes + decode | 13.3 | 97 / 43 | — | 646 / 628 |
| P1 netCDF4 on the files | 12.9 | 94–97 / 38–43 | 92 / 40 | 229–236 / 127–146 |
| P2 xarray per file | 13.0 | 99 / 40 | — | 165 / 138 |
| P2m `open_mfdataset` on the window | 5.5 | 113 / 46 | — | 303 / 320 |
| P3 zarr on the reference | 4.5 | 28–32 / 17–21 | 26 / 17 | 84–107 / 78–100 |
| P4z xarray, zarr engine, no dask | 4.4 | 28.5 / 17.4 | 26 / 17 | 87–98 / 86–98 |
| P5z one dask layer, one storage chunk per task | 6.5–6.8 | 39 / 25.5 | 38 / 23 | 203–212 / 170–197 |
| **P6z_24 one dask layer, 24 steps per task** | 4.4–5.1 | 18–36 / 18–21 ¹ | — | **28 / 32** |
| P6z_240 one dask layer, 240 steps per task | 4.4–4.6 | 18–26 / 17 ¹ | — | 30–39 / 36–38 |
| P5o kerchunk engine, one dask layer | 6.6 | 42 / 25 | 42 / 23 | 166–192 / 199–205 |
| P4 kerchunk engine, `chunks=None` | 6.9 | 45 / 26 | 39 / 23 | 156–198 / 194–206 |
| **P5 kerchunk engine, `chunks={}` (ocean-skill)** | **317** | **1,130** (cold) | **63 / 57** | **215–220 / 214** |
| P6_24 (nested, 24 steps per outer task) | 18 | 66–91 / 65–73 | — | 31–67 / 31–63 |
| P6_720 | 6.7 | 36 / 25 | — | 94 / 83 |

¹ These lakes02 windows came back already cached, so cold ≈ warm. Ranges include runs
with 4 and 16 dask threads.

What the numbers show:

- **The decode is the floor.** On lakes02, reading a 3.5 MB chunk from GlusterFS cold
  takes ~65 ms and decompressing it ~40–45 ms. Warm (from the page cache) the read
  drops to 3 ms. The netCDF4 baseline is exactly these two costs added together.
- **A reference read through zarr beats that sum** (26–32 ms cold, P3/P4z). It overlaps
  reading and decoding across many chunks at once; zarr's default async concurrency is
  10.
- **ocean-skill's path before the fix (P5) was 25–45× slower than any single-layer path**,
  because of the nested compute:
  - each outer dask task runs a full inner `compute()`;
  - on the laptop, one inner compute on the 1.8-million-chunk `temp` array takes
    470 ms, 405 ms of it in `dask.optimize`;
  - on the 1-year store (70k chunks) the same step takes 10 ms.
  - That is why the penalty grows with the length of the record: 57–63 ms per chunk on
    a 1-year CIOFS3 reference, 1,130 ms on the 26-year one.
- **`chunks={time: N}` on the nested path (P6) only reduces the number of outer
  tasks.** It helps, but on a long record it stays slower than one layer with the same
  chunks (P6z).
- **On anvil, cold reads are latency-bound (~0.45 s per 1 MB read), and that changes
  the ranking among the single-layer paths.** See
  [Concurrency](#concurrency-how-many-reads-are-in-flight) below.

### Concurrency: how many reads are in flight

When each read waits on storage, time per chunk is roughly the latency divided by the
number of reads in flight. The read paths differ in exactly that.

E4 anvil, Iceland, 240 hours (960 chunks), cold, ms per chunk:

| path | reads in flight | 4 dask threads | 16 dask threads |
|---|---|---|---|
| P3 / P4z zarr, no dask | zarr's `async.concurrency` (10) | 91–107 | 90–95 |
| P5z one dask layer, one storage chunk per task | one per dask thread | 212 | 57 |
| **P6z_24 one dask layer, 24 steps per task** | threads × zarr concurrency | **28** | **28** |
| P6z_240 one dask layer, 240 steps per task | threads × zarr concurrency, but fewer tasks | 30 | 39 |
| P6_24 nested, 24 steps per outer task | as P6z_24, minus the nesting cost | 67 | 31 |

- **One dask task per storage chunk caps the reads in flight at the thread count.** On
  GPFS, that makes it twice as slow as plain zarr.
- **Chunks spanning ~24 steps let every dask task run zarr's concurrent fetch,** so
  4 threads already keep ~40 reads in flight. That was the fastest path on anvil
  (28 ms per chunk: 3× plain zarr, 4–8× netCDF4). On lakes02 and the laptop it was
  within the noise of the fastest.
- **Raising zarr's `async.concurrency`** would likely help the no-dask path the same
  way. That was not measured.

### Opening

| | E1 laptop | E2 lakes02 | E3 lakes02 → S3 | E4 anvil (Iceland) |
|---|---|---|---|---|
| reference, kerchunk engine, `chunks={}` | 0.4–0.6 s | 5.5–5.9 s (14.4M dask chunks) | 1.4 s (48 files) | 1.3 s (1.0M dask chunks) |
| reference, zarr engine, `chunks={}` | 0.04 s | 1.5 s | 0.1 s | 0.03 s |
| reference, zarr engine, `chunks=None` | 0.01 s | 0.18 s | 0.01 s | 0.01 s |
| one file, `netCDF4.Dataset` | 0.001 s | 0.16 s | fails: netCDF-C byte-range mode refuses the S3 URL | 0.66 s |
| one file, `xr.open_dataset` | 0.002 s | 0.15 s | 10.9 s (h5netcdf over fsspec) | 0.02 s |

At 0.15 s per file, opening all 9,374 CIOFS3 daily files would take about 23 minutes
before reading anything. That cost is what a reference removes.

### Full-field snapshot (R2: surface level, 3 time steps, 12 chunks)

E2 lakes02, cold / warm:

| path | time |
|---|---|
| P1 netCDF4 | 1.12 s / 0.68 s |
| P3 zarr | 0.48 s / 0.30 s |
| P4z zarr engine, no dask | 0.62 s / 0.26 s |
| P5z zarr engine, one dask layer | 2.13 s / 1.84 s |
| P4 kerchunk engine, `chunks=None` | 2.21 s / 1.76 s |

E4 anvil, Iceland (27 chunks), cold / warm:

| path | time |
|---|---|
| P1 netCDF4 | 7.4 s / 0.72 s |
| P3 zarr | 2.3 s / 0.05 s |
| P4z zarr engine, no dask | 3.1 s / 0.06 s |
| P5z zarr engine, one dask layer | 7.9 s / 0.18 s |
| P5 (ocean-skill) | 6.7 s / 2.8 s |

With few chunks per request, the one remaining cost of dask shows. Every `compute()`
on a dask array with 1.8 million chunks pays ~1.5 s (lakes02) to optimize its graph,
however few chunks it reads. A lazily indexed (no-dask) array does not pay it.

### E3: lakes02 → S3

The data is 48 hourly CIOFS operational nowcast files from NOAA's public bucket
(`noaa-nos-ofs-pds`), 0.59 GB each. They are **uncompressed** HDF5 with `temp` chunks
`(1, 10, 348, 242)`, 3.4 MB each. The request is the same mooring column over 24 hours:
96 chunks, 315 MB fetched.

From lakes02, one S3 stream runs at about 5.7 MB/s with ~0.3 s per request, and
repeated runs varied by up to 2×.

| path | cold | warm |
|---|---|---|
| P0 raw byte ranges, one at a time | 72 s | 149 s |
| P2 `xr.open_dataset` per file (h5netcdf over fsspec) | 378 s | 572 s |
| P2m `xr.open_mfdataset` on the 24 files | 374 s | 682 s |
| **P3 zarr on the reference** | **10 s** | **20 s** |
| P4z / P5z / P5o (no dask, or one dask layer) | 23–41 s | 25–35 s |
| P5 (ocean-skill before the fix) | 27 s | 31 s |

- **The reference is 9–38× faster here**, comparing like with like (cold with cold, warm with warm). Going straight to the files pays about 11 s
  per file just to open it, because h5netcdf reads the HDF5 metadata through many small
  range requests. Then it reads chunk by chunk. The reference opens in 0.01–1.4 s and
  fetches the needed chunks concurrently.
- **netCDF4's own byte-range mode** (`…nc#mode=bytes`) was refused for the S3 URL
  (`NetCDF: S3 error`), with or without `aws.profile=no`, so h5netcdf is the direct
  baseline here.
- **The double-dask penalty is negligible** on this 48-step reference (P5 ≈ P5z). The
  penalty grows with the length of the record, not with the storage.
- **Building the reference** is a one-time cost of about 6 s per file from lakes02
  (271 s for 48 files with 8 threads), with fsspec's `cache_type="first"` and an 8 MB
  first block. From a home connection it did not finish in 20 minutes.
- **Uncompressed chunks** let a direct reader fetch just the needed bytes, but at
  ~0.3 s per request that loses to fetching whole chunks concurrently.

### E4: anvil, Iceland and PACMED

All E4 runs used dedicated `standard` nodes. A first attempt on a shared node (load
~137 from other users' jobs) was discarded.

**Iceland** (`his.parquet`, compressed daily files; full results in the tables above):

- **GPFS reads are latency-bound and barely cached for long series.**
  - A cold 1.1 MB read takes ~0.45–0.64 s one at a time.
  - Re-reading a 24-step window was cached (2 ms per read). Re-reading a 240-step
    window (~1 GB) was not (~0.62 s): the GPFS client cache held much less than that.
  - So a multi-month mooring series is effectively always read cold.
- **The best path,** one dask layer with 24-step chunks, ran at 28–32 ms per chunk.
  netCDF4 straight from the files ran at 127–236. ocean-skill's current path ran at
  ~215.
- **The double-dask penalty is smaller on this 7,272-step record.** With cold reads
  costing ~0.5 s, it is hidden. Warm, it still costs 59 ms per chunk against
  4–5 ms for one layer (24-step requests).

**PACMED** `output_cdrgas`:

- **Setup:** 143 daily CDF5 files, each one uncompressed `(962, 1858)` float64 field.
  The reference was built with `make_kerchunk`; kerchunk's own netCDF-3 reader cannot
  parse CDF5.
- **Bytes:** a reference stores the whole 14 MB field as one chunk, so it fetches all
  of it per step (1.37 GB for 96 days). A netCDF reader can fetch just the 11 × 11 box.
- **Timing caveat:** the record is too short for separate cold windows per path, so
  most timings are warm.
- **Warm results, per step:**
  - reference paths: 2–3 ms;
  - `xr.open_dataset` per file: 5.6 ms;
  - a `netCDF4.Dataset` loop: 75 ms.
- **Cold:** fetching a whole field one at a time took 146–255 ms per step.
- **Verdict:** for a short record of 2-D fields every route is fast, and the reference
  was not slower. Its 14 MB-per-step reads would cost real time on cold or remote
  storage over a long record. That is the one case measured here where reading the
  files directly could win.

## Does the xarray version matter?

Not for this. Measured on E1:

| setup | P5 (ocean-skill) | P5z | snapshot, P5z |
|---|---|---|---|
| xarray 2026.4.0, dask 2026.3.0 | 296–317 ms/chunk | 6.5 ms/chunk | 0.47 s |
| xarray 2026.9.0, dask 2026.8.0 | 298 ms/chunk | 6.5 ms/chunk | 0.52 s |
| xarray 2026.9.0 + `dask-array` 0.7.1, registered | **10 ms/chunk** | 6.0 ms/chunk | **0.12 s** |
| xarray 2026.4.0 + `dask-array` 0.7.1, registered | **10 ms/chunk** | 6.2 ms/chunk | — |

The relevant recent change is xarray 2026.07's support for Dask *expression* arrays
([pydata/xarray#11382](https://github.com/pydata/xarray/pull/11382) and
[pydata/xarray#11423](https://github.com/pydata/xarray/pull/11423)). Those arrays live in
the separate `dask-array` package, and you opt in with
`dask_array.xarray.register()`, which swaps xarray's dask chunk manager for the whole
process. They plan queries without materializing the whole graph, which removes both
the nested-compute penalty and the per-compute optimize cost. Even so, the nested path
stays slower than a single layer.

`dask-array` is version 0.7, replaces the chunk manager process-wide, and refuses to mix
legacy and expression arrays in one operation, so it is worth watching rather than
adopting now.

## Decision table

| situation | faster | why |
|---|---|---|
| opening a long multi-file record (years of daily/hourly files) | **reference** | one index read against N HDF5 metadata parses (0.15 s/file on lakes02, 11 s/file on S3) |
| point time series from compressed files, local/NFS/GlusterFS/GPFS, read with one dask layer (time-aggregated chunks) or none | **reference** | same bytes and decode, but zarr runs them concurrently: 2.2–3.6× faster than netCDF4 cold on lakes02, 4–8× on anvil |
| the same, through ocean-skill before the fix (`engine="kerchunk", chunks={}`) | files | the nested dask layer costs 0.3–1.1 s per chunk on long records |
| the same, through ocean-skill now (one layer at the storage chunks) | **reference** | no nesting; reads never exceed the bytes asked for. Multi-step time blocks (`chunks="time_blocks"`) are faster still for long contiguous series on GPFS, but read many times the bytes for snapshots and scattered times ([correction](#correction-2026-10-12)) |
| full-field snapshots at a few times | reference, opened lazily (no dask) | dask's per-compute graph cost dominates when few chunks are read |
| files on S3/HTTP | **reference** | 9–38× faster on S3 ([E3](#e3-lakes02--s3)); opening alone is 11 s per file directly |
| uncompressed files with large chunks (a whole 2-D field per chunk), small window, long record on cold/remote storage | possibly the files | a reference fetches the whole chunk (14 MB/step for PACMED); a netCDF reader fetches only the box. Short records: no difference measured ([E4](#e4-anvil-iceland-and-pacmed)) |

## Recommendations for ocean-skill

Items 1 and 2 are implemented; the rest are not. They are ranked by measured gain for
the CIOFS-style use.

### Correction (2026-10-12)

The first version of item 1 opened references with byte-sized, multi-step time chunks
(`KERCHUNK_CHUNK_BYTES` = 256 MiB). Testing on real data (Iceland ROMS `his.parquet` on
anvil: `temp` (7272, 60, 514, 514), storage chunks (1, 20, 172, 172)) found two
problems, and that default has been replaced.

1. **A sizing bug.** The block was derived from the largest stored time chunk over the
   data variables. In real UCLA output `ocean_time` is a 1-D *data variable* on `time`
   (not a coordinate), stored as **one chunk of the whole record**. Every variable then
   got a dask chunk of the entire record (3.4 GB), and `osk.read` point reads were
   OOM-killed at 30 GB, twice. The same store read 20–90 MiB in 2–8 s per pattern
   before. `_time_chunks` now ignores variables with fewer than two dimensions.
2. **A design problem, independent of the bug.** `roms.standardize` masks land with
   `.where(mask_rho)` on every rho-point variable, so every ROMS read does lazy
   arithmetic before any time selection, and dask loads each dask chunk it touches
   whole. Multi-step blocks therefore multiply the bytes read for snapshots and
   scattered times. Laptop, 26-year synthetic store, blocks of 23 steps: one full-field
   step after the arithmetic read **1.07 GB against 47 MiB**; three scattered steps
   read **3.2 GB against 139 MiB**. Blocks were faster only for long contiguous point
   series on GPFS.

**Now:** the default (`chunks="auto"`, and `{}` as baked into existing catalogs) is one
dask layer at the **storage chunks**, so the time chunk is the stored one (typically 1).
That never reads more bytes than the old nested open did, and it keeps the removal of
the per-chunk graph overhead (0.3–1.1 s per chunk on long records). Time blocks remain
available as the opt-in `chunks="time_blocks"` for a long contiguous point series on
high-latency storage, where the 24-step numbers above still apply. Read the
measurements below as describing that opt-in, not the default.

**Verified against the code before PR #303**, through `osk.read`, with each run on its
own time steps and old/new order alternated. Every pattern read the same bytes (to
within the different steps' compression) and gave identical values:

| Machine, store | Patterns (point, snapshot, scattered, strided, after arithmetic) | `osk.compare` HVIN1 ADCP vs `his` |
|---|---|---|
| anvil GPFS, Iceland `his` (7,272 steps) | 1.0–9.2 s → 0.3–1.4 s (3.5–9×) | 4,234 s → 1,013 s (4.2×), aligned results identical |
| laptop SSD, synthetic 26-year store | 6.6–296 s → 0.6–7.7 s (4–38×) | — |
| tanagra NFS, PACMED `smallgrid_physics` (144 steps) | same within noise (0.7–5 s) | — |

The gain grows with record length, since the nested graph cost does; a 144-step record
has almost none to remove. No pattern got slower.

1. **Open kerchunk references with one dask layer.** *(Originally: chunked over ~24
   time steps; see the correction above.)*
   - **Done:** `build.open_reference`, used by `osk.read` and catalog probes. Default:
     the storage chunks. Opt-in `chunks="time_blocks"`: blocks sized to
     `KERCHUNK_CHUNK_BYTES` = 256 MiB instead of a fixed 24 steps.
   - **Measured after the change, with time blocks** (laptop, synthetic 225k-step CIOFS-shaped reference,
     one process, `PYTHONHASHSEED=1`):
     - 24 hours at one point: old open 468–551 ms per chunk, `osk.read` 7–10 ms;
       240 hours 6–9 ms.
     - Values identical; dask chunks `(23, 15, 522, 362)`.
     - `osk.read` opens in 1.3 s.
   - **Change:** in `ocean_skill/build.py` `_reader_for`, replace `chunks={}` with
     `chunks={<time dim>: 24}` and add `open_dataset_options={"chunks": None}` (works
     with kerchunk 0.2.9 and 0.2.10). Equivalently, open
     `kerchunk.utils.refs_as_store(...)` with `engine="zarr"`.
   - **Measured gain** (P5 → P6z_24):
     - 26-year CIOFS3 store on lakes02: 1,130 → 18–36 ms per chunk (30–60×);
     - Iceland on anvil: ~215 → 28–32 ms (7×);
     - laptop: 317 → 4.4–5.1 ms (60–70×).
   - **Why storage chunks are the default despite this:** One dask layer with
     storage-sized chunks (`chunks={}`) fixes the nesting but caps reads in flight at
     the dask thread count, which is ~7× slower than blocks for a long contiguous series
     on GPFS. Blocks lose for every ROMS read that is not that (the land-mask
     arithmetic loads each block whole), so they are opt-in.
   - **Memory (time blocks):** a 24-step task holds up to 24 decoded storage chunks
     (CIOFS3 `temp`: ~270 MB before cropping).
   - **Effect:** the reference becomes the fastest route for point time series in every
     environment tested. No data or caching changes.
2. **Pass `storage_options` through for kerchunk entries, and set `remote_protocol`.**
   - **Done:** `build.open_reference` works out `remote_protocol` from where the
     reference's chunks live (`_reference_protocol`), and `_reader_for` now passes the
     entry's `storage_options` through.
   - **The bug:** `_reader_for` builds the kerchunk reader without the entry's
     `storage_options`, so `remote_protocol`, `remote_options` and the reference
     mapper's `cache_size` can't be set.
   - **Consequence:** with no `remote_protocol`, fsspec's
     `ReferenceFileSystem.__init__` scans `references.values()` to guess one
     (`fsspec/implementations/reference.py:763`). On a parquet reference under
     pandas 3, that scan can hit a missing-chunk row whose path is `pd.NA`, and
     `if ref[0]` raises `boolean value of NA is ambiguous`. Which row it hits first
     depends on the hash seed, so the failure is deterministic per seed but looks
     intermittent.
   - **Fix:** pass `remote_protocol` (`"file"` for local files, the URL scheme
     otherwise). That skips the scan; it is the workaround already noted in
     [kerchunk_remote.md](kerchunk_remote.md).
   - **Tests it would unflake:** the two parquet tests in `test_build_subchunk.py`, and
     `test_keep_latest_per_file_via_build_kerchunk`.
3. **Crop point lanes before chunking.**
   - **Change:** open lazily (`chunks=None`), do the time and space `isel` crop, then
     load, or chunk only the cropped result.
   - **Measured gain:** mainly for snapshots, 0.26 vs 1.84 s on lakes02 (7×), because
     each `compute()` on a 1.8-million-chunk array pays ~1.5 s of graph optimization.
     For time series, (1) already matches the no-dask path.
   - **Cost:** a larger change. `roms.standardize` masks and builds `z_rho` before the
     crop, assuming dask arrays, so the order of operations in `prepare_source` would
     have to change.
4. **Share decodes across stations.**
   - **Why it matters:** after (1) the floor is decompression, ~40–45 ms per chunk on
     lakes02. Two stations in the same horizontal chunk each decode it.
   - **Today:** the shared slab (`comparison._build_shared_slabs`) does this only for
     ROMS point references whose union box fits in 2 GiB.
   - **Gain:** a one-pass, per-chunk extraction for many stations would scale with the
     number of stations per chunk.
5. **For heavy, repeated multi-year station work, keep a time-series-chunked copy.**
   - This is the only way below the decode floor.
   - A zarr/icechunk copy chunked like `(8760, 30, 32, 32)` decodes ~180× less data per
     station column.
   - The one-time cost is reading every chunk once (CIOFS3: ~1.8M `temp` chunks ×
     ~45–100 ms, parallelizable).
6. **Reference format and length matter little once (1) and (2) are in.**
   - **JSON vs parquet, one year of CIOFS3 on lakes02:** both read at the same speed.
     Parquet opened faster (0.02 s vs 0.34 s lazily; the JSON is 9.4 MB).
   - **On the laptop:** parquet was up to 40% slower per chunk.
   - With `remote_protocol` passed, parquet's crash goes away, so choose by size. Keep
     one stream per reference.
   - Record length matters only to dask paths, since it sets the per-compute graph
     size.
7. **Watch `dask-array`** (see above). It removes the per-compute graph cost without
   any of the changes above, once it is stable.

**Not recommended:** switching CIOFS-style reads to the files themselves, for example by
using the manifest as an index to the daily files. With (1) in place, that route is the
slower one for compressed output.

## Reproducing

```bash
python docs/benchmarks/kerchunk_access.py run REF.parquet --out results.json \
    --var temp --zeta zeta --eta 261 --xi 181 --steps 24,240 \
    --start 175000 --stride 1000 --check-start 174000
```

- `compose` builds a narrowed reference from per-file references or netCDF files
  (optionally cycled out to a long record).
- `synth` writes CIOFS-shaped synthetic files.
- The script refuses to overwrite any existing path.
- Results for every environment are in [`docs/benchmarks/results/`](benchmarks/results/).
