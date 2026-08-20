# Kerchunking remote files

`build_kerchunk`/`make_kerchunk` accept `http://` and `https://` URLs as well as local
paths, so a few thousand daily files on a server can become **one** catalog entry
instead of a few thousand:

```python
from ocean_skill.build import make_kerchunk

urls = [f"{base}/AQUA_MODIS.{d}.L3m.DAY.CHL.chlor_a.9km.nc" for d in days]
make_kerchunk(urls, out="modis_daily_chl.json")
```

Everything else is unchanged: `concat_dim` and `loadable_variables` are still detected
from the first file, and the result is read through the same `engine="kerchunk"` path.

## The one hard requirement: byte ranges

Kerchunk stores *(file, offset, length)* triples and later re-reads those byte ranges.
The server must therefore serve the **raw file over HTTP range requests**
(`Accept-Ranges: bytes`, `206 Partial Content`).

**An OPeNDAP endpoint never qualifies.** DAP is a query protocol — you ask for a
variable and the server *computes* a response — so there is no stable byte layout to
point at, however happily `xr.open_dataset` reads it.

But the host can refuse ranges outright, which is the more common blocker and is easy
to misdiagnose as an OPeNDAP problem. Measured against `oceandata.sci.gsfc.nasa.gov`
(NASA OB.DAAC) on 2026-08-07:

| Request | Result |
|---|---|
| `GET /` | `200`, 36,740 bytes |
| `GET /` with `Range: bytes=0-99` | **`416 Range Not Satisfiable`** |
| `GET /ob/getfile/AQUA_MODIS...nc` with `Range:` | `416` |
| `GET /opendap/...nc` with `Range:` | `416` |

The plain HTML homepage refuses a range. So this is a **site-wide policy, not an
OPeNDAP limitation** — and it means no URL on that host is kerchunkable, including the
authenticated direct-download path. Earthdata credentials do not change this.

So a source you can open with xarray is not automatically a source you can kerchunk:

| Source | `open_dataset` | kerchunk |
|---|---|---|
| local path | ✅ | ✅ |
| HTTPS server honoring ranges | ✅ | ✅ |
| S3 / GCS / Azure object store | ✅ | ✅ |
| OPeNDAP / DAP endpoint | ✅ | ❌ |
| host that refuses ranges (OB.DAAC) | ✅ | ❌ |

When the archive refuses ranges, the options are to **download once and kerchunk
locally**, or to reach a **cloud copy** — object stores serve ranges by definition.
NASA's are in Earthdata Cloud (`obdaac-tea.earthdatacloud.nasa.gov/s3credentials`
issues temporary S3 keys), and obstore ships `S3Store`, `GCSStore` and `AzureStore`
alongside `HTTPStore`. `_store_for` currently builds only local and HTTP stores; pass a
pre-built store for anything else.

Where a host does need auth, `HTTPStore` carries headers:

```python
HTTPStore.from_url(base, client_options={"default_headers": {"Authorization": f"Bearer {token}"}})
```

Plain `http://` URLs additionally need `allow_http`, which `_store_for` sets
automatically — object_store refuses non-TLS URLs by default, and the error it raises
(`BadScheme`) names neither HTTP nor the option.

## Prefer `.json` over `.parquet` for now

The output format follows the extension. Both work, but **parquet reads back
intermittently**: roughly 1 run in 6 over identical inputs fails with

```
TypeError: boolean value of NA is ambiguous
```

from `fsspec/implementations/reference.py`, where a null column in the parquet
reference reaches a truth test. The parquet on disk is byte-identical between passing
and failing runs, so this is a read-side bug in fsspec/kerchunk, not a corrupt write.
`.json` round-tripped 6/6 on the same inputs.

Parquet remains the better target for very large references (JSON holds every chunk
record in memory); use it there and re-open on failure. For a few thousand daily files,
JSON is fine.

## Chunk size: the manifest inherits the source file's layout

A kerchunk reference does not choose a chunk shape — it records whatever the source
file already has. ROMS writes one time record per output file with the full water
column in it, so a variable shaped `(time, s_rho, eta_rho, xi_rho)` with 100 vertical
levels becomes a manifest with **one chunk of ~1.3 GiB per time step**, and every
downstream reader (dask, `xr.open_dataset(..., chunks={})`, `osk.compare`) inherits
that as the unit it reads — reading it, or reducing over it before narrowing it down,
pulls the whole thing into memory. `reader_kwargs={"chunks": ...}` on a catalog entry
cannot fix this: dask chunks smaller than the *storage* chunk still fetch and decode
the whole storage chunk per task, so a smaller dask chunk only adds read
amplification, never shrinks the actual I/O.

`make_kerchunk`/`build_kerchunk` split the manifest itself instead, which is possible
because an **uncompressed** chunk's bytes are one contiguous run: any leading axis
already down to one chunk (time, typically) makes the next axis (the vertical)
contiguous too, so it can be cut into byte-range sub-chunks with no data movement.
On by default (`target_chunk_mb=128`, splitting only the leading dims — never the
horizontal), or requested explicitly:

```python
make_kerchunk(files, "refs/gom_bgc.parquet", grid=GRID, subchunk={"s_rho": 5})
# (1, 100, 962, 1858) float64 (1.33 GiB) -> (1, 5, 962, 1858) (~68 MB)
```

**This does not extend to compressed output.** A compressed chunk is the unit its
codec decompresses — there is no byte range inside it that means anything on its own,
so `subchunk=`/`target_chunk_mb` silently leave a compressed variable's chunks as
stored (a warning names it). If future model output is written compressed, the chunk
shape has to be chosen **when the file is written**, not fixed up afterward: compression
already forces chunked (not contiguous) storage, so the choice is unavoidable at that
point anyway — picking something like `(1, 5-10 levels, full eta, full xi)` avoids
recreating the same one-chunk-per-timestep problem permanently. A file already written
with oversized compressed chunks can be repacked (`nccopy -c ocean_time/1,s_rho/5,... -d
4`, or `h5repack`) at the cost of rewriting it once.

A different (non-manifest) technique for compressed sub-chunk reads exists —
[CNES's `chunkindex`](https://github.com/CNES/netCDFchunkindex), described in
[Penard et al., GMD 19:1519 (2026)](https://gmd.copernicus.org/articles/19/1519/2026/):
a sidecar index of zlib re-entry points lets a reader decompress just the covering
segment of a compressed chunk. It is real and it works, but it is a standalone
h5py-based reader with no xarray/dask/zarr/kerchunk integration, no packaging, and
deflate-only support (shuffle costs it ~2×) — a watch item, not something this
package depends on. The zarr v3 sharding this project's icechunk conversion
(see `environment.yml`) would eventually offer is the integrated equivalent: small,
independently-compressed inner chunks inside larger storage objects.

## Verifying a server before a long build

Worth thirty seconds before kerchunking thousands of files:

```python
import obstore
from obstore.store import HTTPStore

store = HTTPStore.from_url(base)
print(obstore.head(store, name)["size"])                              # reachable at all?
print(len(bytes(obstore.get_range(store, name, start=0, end=100))))   # 100 → ranges work
```

If the second line raises `RangeNotSupported` or returns the whole file, kerchunk will
not work against that server no matter how the reference is built.
