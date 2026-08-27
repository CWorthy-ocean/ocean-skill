"""Helpers for building intake v2 catalogs with the ocean-skill metadata contract.

Two steps, deliberately simple:

1. :func:`make_kerchunk` — turn a list of files into one virtual store (VirtualiZarr →
   kerchunk reference). For ROMS the separate grid file can be merged in, so the
   store is self-contained.
2. :func:`add_source` — add that store (or any readable source) to a catalog,
   deriving metadata by *querying the data*: time/lon/lat/depth extents, the coordinate
   and variable mappings (cf-xarray), and a featureType guess from the dimensions.

Typical use::

    from ocean_skill.build import make_kerchunk, new_catalog, add_source, save

    make_kerchunk(sorted(out.glob("output_bgc.*.nc")), "refs/gom_bgc.parquet",
                  grid=GRID)

    cat = new_catalog(title="GOM offline run")
    add_source(cat, "GOM_bgc", "refs/gom_bgc.parquet")
    save(cat, "catalogs/gom.yaml")

Two one-call wrappers cover the shapes that repeat, so neither the per-stream
kerchunk dance nor a column of near-identical ``add_source`` lines has to be
written out each time:

:func:`build_kerchunk` — a model run's output streams, one reference each. Its
return value is exactly what :func:`build_catalog` takes, so the two compose::

    refs = build_kerchunk({"GOM_bgc": "output_bgc.*.nc", "GOM_his": "output_his.*.nc"},
                          root=run_dir, grid=GRID)
    # restart files hold more than one time record per file; keep="latest-per-file"
    # drops the earlier ones instead of letting them scramble the time axis
    refs |= build_kerchunk({"GOM_rst": "output_rst.*.nc"}, root=run_dir, grid=GRID,
                           keep="latest-per-file")
    build_catalog(refs, "catalogs/gom.yaml", title="GOM offline run")

:func:`build_catalog` — many URLs sharing one set of options::

    build_catalog(
        {"MODIS Aqua January Monthly Climatology": jan_url,
         "MODIS Aqua February Monthly Climatology": feb_url},
        "catalogs/modis_aqua.yaml",
        title="MODIS Aqua",
        storage_options={"simplecache": {"same_names": True}},
    )

Catalog authoring is expected to become its own project eventually; this is the minimum
that keeps ocean-skill self-sufficient.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import warnings
from pathlib import Path
from typing import Any

from ocean_skill.cf import find_coord

__all__ = [
    "ROMS_STANDARD_NAMES",
    "add_catalog",
    "add_copernicus_source",
    "add_erddap_source",
    "add_source",
    "add_sources",
    "build_catalog",
    "build_kerchunk",
    "detect_concat",
    "discover_opendap_files",
    "guess_feature_type",
    "make_kerchunk",
    "new_catalog",
    "save",
    "tolerant_hdf_attrs",
]

#: Fallback variable → CF standard_name map for ROMS/MARBL output, which mostly lacks
#: ``standard_name`` attributes. Variables carrying their own ``standard_name`` win.
#: Component chlorophylls (spChl/diatChl/…) are unmapped: CF's name means the total.
ROMS_STANDARD_NAMES: dict[str, str] = {
    "temp": "sea_water_potential_temperature",
    "salt": "sea_water_practical_salinity",
    "zeta": "sea_surface_height_above_geoid",
    "u": "sea_water_x_velocity",
    "v": "sea_water_y_velocity",
    "w": "upward_sea_water_velocity",
    "hbls": "ocean_mixed_layer_thickness",
    "NO3": "mole_concentration_of_nitrate_in_sea_water",
    "PO4": "mole_concentration_of_phosphate_in_sea_water",
    "SiO3": "mole_concentration_of_silicate_in_sea_water",
    "NH4": "mole_concentration_of_ammonium_in_sea_water",
    "Fe": "mole_concentration_of_dissolved_iron_in_sea_water",
    "O2": "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
    "DIC": "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water",
    "ALK": "sea_water_alkalinity_expressed_as_mole_equivalent",
    "FG_CO2": "surface_downward_mole_flux_of_carbon_dioxide",
}


# --------------------------------------------------------------------------- kerchunk


@contextlib.contextmanager
def tolerant_hdf_attrs():
    r"""Let the HDF parser survive attributes that are not valid UTF-8.

    ROMS ``tides.F90`` leaves its ``info`` buffer uninitialized and ignores the
    ``nf90_get_att`` status, so output carries a ``forcing_info`` global attribute with
    raw memory bytes (e.g. ``b'TIDES:\\xc0\\x89\\x0ck\\x01'``). virtualizarr's parser
    decodes attributes strictly and dies on it. Patching the files works but the bug
    recurs on every run until ROMS itself is fixed, so decode leniently here instead.
    """
    import numpy as np
    import virtualizarr.parsers.hdf.hdf as vzhdf

    hidden = {
        "REFERENCE_LIST",
        "CLASS",
        "DIMENSION_LIST",
        "NAME",
        "_Netcdf4Dimid",
        "_Netcdf4Coordinates",
        "_nc3_strict",
        "_NCProperties",
    }
    original = vzhdf._extract_attrs

    def lenient(h5obj):
        attrs = {}
        for name, v in h5obj.attrs.items():
            if name in hidden:
                continue
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace") or " "
            elif isinstance(v, np.ndarray | np.number | np.bool_):
                if v.dtype.kind == "S":
                    v = v.astype(str)
                elif v.size == 1:
                    v = v.flatten()[0]
                    if isinstance(v, np.ndarray | np.number | np.bool_):
                        v = v.tolist()
                else:
                    v = v.tolist()
            elif isinstance(v, vzhdf.h5py._hl.base.Empty):
                v = ""
            if isinstance(v, str) and v == "DIMENSION_SCALE":
                continue
            attrs[name] = v
        return attrs

    vzhdf._extract_attrs = lenient
    try:
        yield
    finally:
        vzhdf._extract_attrs = original


def _is_remote(target) -> bool:
    """Whether ``target`` is an http(s) URL rather than a local path."""
    return str(target).startswith(("http://", "https://"))


def _store_for(target):
    """Return ``(url, store)`` for one file, local or remote.

    virtualizarr reads chunks through an obstore backend, so a remote source needs
    an ``HTTPStore`` rather than a ``LocalStore``. Remote only works where the server
    serves the **raw file with byte ranges** — a plain OPeNDAP/DAP endpoint does not
    (it speaks a query protocol, not file bytes) and will fail here even though
    ``xarray.open_dataset`` reads it happily. See :func:`make_kerchunk`.
    """
    from obstore.store import HTTPStore, LocalStore

    url = str(target)
    if _is_remote(url):
        # allow_http is off by default in object_store, and a plain http:// URL then
        # fails to even build the request ("BadScheme"). Real archives are https, but
        # test servers and intranet hosts are not, and the default error names neither.
        return url, HTTPStore.from_url(
            url.rsplit("/", 1)[0],
            client_options={"allow_http": url.startswith("http://")},
        )
    # expanduser: "~/refs/grid.nc" reaches obstore as a *literal* tilde directory and
    # fails with "Unable to canonicalize filesystem root", which gives no hint that the
    # path simply was not expanded.
    path = Path(url).expanduser()
    return f"file://{path}", LocalStore(prefix=path.parent)


#: Leading bytes that identify a file's container format. NetCDF is two unrelated
#: formats wearing one extension: netCDF-4 is HDF5, netCDF-3 ("classic"/64-bit/CDF-5)
#: is not, and each needs a different virtualizarr parser. ROMS commonly writes
#: netCDF-4 output alongside a netCDF-3 grid file, so one store can need both.
_MAGIC = {
    b"\x89HDF": "hdf5",
    b"CDF\x01": "netcdf3",
    b"CDF\x02": "netcdf3",
    # CDF-5 is netCDF-3's 64-bit-data variant, kept distinct from "netcdf3" because
    # whether it can be read at all depends on which virtualizarr is installed: the
    # kerchunk-backed NetCDF3Parser is scipy-backed and scipy supports CDF-1/CDF-2
    # only, while the native parser reads all three. See _netcdf3_reads_cdf5.
    b"CDF\x05": "cdf5",
}


def _file_format(target) -> str | None:
    """Container format of ``target`` from its magic number; ``None`` if unreadable.

    Sniffing the bytes rather than trusting the extension: ``.nc`` says nothing about
    which of the two formats is inside, and guessing wrong surfaces as h5py's
    "file signature not found", which names neither the file nor the reason.
    """
    if _is_remote(target):
        return None  # would need a range request; callers fall back to the default
    try:
        with open(Path(target).expanduser(), "rb") as fh:
            head = fh.read(4)
    except OSError:
        return None
    return _MAGIC.get(head)


@functools.cache
def _netcdf3_reads_cdf5() -> bool:
    """Whether the installed ``NetCDF3Parser`` can read CDF-5.

    Two implementations ship under that name. The older one wraps
    ``kerchunk.netCDF3.NetCDF3ToZarr``, which subclasses scipy's header reader and so
    inherits scipy's CDF-1/CDF-2-only support; the native one parses the header itself
    and handles all three classic formats. Probing for the native module's
    ``_parse_header`` rather than comparing versions because the change is unreleased
    (VirtualiZarr PR #1086), so there is no version number to compare against yet.

    Replace this with a version floor in ``environment.yml`` once a release carries it.
    """
    try:
        netcdf3 = importlib.import_module("virtualizarr.parsers.netcdf3")
    except ImportError:  # pragma: no cover - virtualizarr is a hard dependency
        return False
    return hasattr(netcdf3, "_parse_header")


def _parser_for(target):
    """Return the virtualizarr parser matching ``target``'s container format.

    Raises on CDF-5 when the installed parser cannot read it, which is better than the
    ``IndexError`` deep inside scipy that it otherwise produces — that error names
    neither the file nor the format.
    """
    from virtualizarr.parsers import HDFParser, NetCDF3Parser

    fmt = _file_format(target)
    if fmt == "cdf5" and not _netcdf3_reads_cdf5():
        raise ValueError(
            f"{target} is netCDF-3 CDF-5 (64-bit data), which this virtualizarr "
            "cannot kerchunk: its NetCDF3Parser is scipy-backed and scipy reads "
            "CDF-1/CDF-2 only. Convert it first, then pass the converted file:\n"
            f"    nccopy -k netCDF-4 {target} converted.nc\n"
            "(ncdump -k names the format of any file.)"
        )
    return NetCDF3Parser() if fmt in ("netcdf3", "cdf5") else HDFParser()


def detect_concat(file) -> tuple[str, tuple[str, ...]]:
    """Return ``(concat_dim, loadable_variables)`` inferred from one output file.

    Replaces per-model configuration with one rule that holds for any model: files
    concatenate along whichever dimension the *time* variable lives on, and that
    variable must be loaded as real values rather than left virtual, because a
    concatenation needs its values to order the result.

    Finding "the time variable" is :func:`find_coord`'s job — cf-xarray first, which
    settles every CF-compliant model immediately, then the
    :data:`_COORD_FALLBACKS` name list. That fallback is what a model costs: one
    string, not a configuration block. (It is load-bearing for ROMS, whose output
    has *no* coordinate variables at all and writes ``units="second"``, so
    cf-xarray finds nothing — verified, not assumed.)

    Raises if no time variable can be found, rather than guessing a dimension:
    concatenating along the wrong axis silently scrambles the record order.
    """
    import xarray as xr

    # A remote file is opened through fsspec: xarray's netcdf4 engine takes a *path*
    # and would treat the URL as one. h5netcdf reads the file object fsspec returns.
    if _is_remote(file):
        import fsspec

        opener = fsspec.open(str(file))
        handle = opener.open()
        ds = xr.open_dataset(handle, decode_times=False, engine="h5netcdf")
    else:
        handle = None
        ds = xr.open_dataset(file, decode_times=False, engine="netcdf4")

    with contextlib.closing(ds):
        time = find_coord(ds, "time")
        if time is None:
            raise ValueError(
                f"cannot find a time variable in {file} to concatenate along; pass "
                "concat_dim= and loadable_variables= explicitly, or add its name to "
                "ocean_skill.cf._COORD_FALLBACKS['time']."
            )
        name = str(time.name)
        # The variable and its dimension are often named differently (ROMS:
        # ocean_time on dim "time"), so take the dimension from the variable.
        dim = str(time.dims[0]) if time.ndim == 1 else name
        result = (dim, (name,))
    if handle is not None:
        handle.close()
    return result


def _warn_if_concat_axis_is_disordered(vds, concat_dim, loadable_variables, paths):
    """Warn when the concatenated coordinate is not strictly increasing.

    ``combine="nested"`` joins in the order given, so nothing checks that the result
    makes sense as a series: two output streams globbed into one call produce an axis
    that runs forward, jumps back, and runs forward again, and the store reads back
    without complaint. A real case mixed ROMS ``cdr`` averages with ``rst`` restart
    files — the restarts each hold two records, and their second one repeated a cdr
    timestamp exactly.

    Warns rather than sorting or dropping, deliberately: the duplicates are a symptom
    of the wrong files being combined, and a silently repaired axis would hide that
    while leaving averaged and instantaneous fields sharing a coordinate. See
    :func:`build_kerchunk` for building one reference per stream, which is the fix.
    """
    import numpy as np

    for name in loadable_variables:
        var = vds.variables.get(name)
        if var is None or var.dims != (concat_dim,) or var.size < 2:
            continue
        steps = np.diff(np.asarray(var.values))
        repeats = int((steps == 0).sum())
        backwards = int((steps < 0).sum())
        if not (repeats or backwards):
            continue
        first = int(np.argmax(steps <= 0)) + 1
        # Raw ROMS times are seconds since an epoch named only in the long_name, so
        # the bare number says nothing about *when* the axis doubled back.
        decoded = _decode_times(vds, var)
        at = decoded[first] if decoded is not None else var.values[first].item()
        warnings.warn(
            f"{name} is not strictly increasing across the {len(paths)} files "
            f"concatenated on {concat_dim!r}: {repeats} repeated and {backwards} "
            f"out-of-order step(s), first at index {first} ({at}). "
            "Combining more than one output stream is the usual cause — build one "
            "reference per stream instead. Nothing was reordered or dropped.",
            stacklevel=3,
        )


def _keep_latest_per_file(concat_dim: str, loadable_variables: tuple[str, ...]):
    """Return a ``preprocess`` callable keeping only each file's latest record.

    ROMS restart files typically hold more than one time record per file, and with
    cycling restarts (``LcycleRST``) the newest record is not always written to the
    last slot. Selecting by the time variable's *value* — via ``argmax``, not a
    fixed position — is what makes cycled files come out right. A file with only
    one record is a no-op.
    """
    import numpy as np

    def preprocess(ds):
        name = next(
            (
                n
                for n in loadable_variables
                if (var := ds.variables.get(n)) is not None
                and var.dims == (concat_dim,)
            ),
            None,
        )
        if name is None:
            raise ValueError(
                "keep='latest-per-file' needs a time variable among "
                f"loadable_variables={loadable_variables!r} with dims=({concat_dim!r},); "
                "pass loadable_variables= explicitly if detection picked the wrong one."
            )
        i = int(np.argmax(np.asarray(ds[name].values)))
        return ds.isel({concat_dim: slice(i, i + 1)})

    return preprocess


def make_kerchunk(
    files,
    out: str | Path,
    *,
    grid: str | Path | None = None,
    concat_dim: str | None = None,
    loadable_variables: tuple[str, ...] | None = None,
    keep: str = "all",
    fmt: str | None = None,
    tolerant_attrs: bool = True,
    subchunk: dict[str, int] | None = None,
    target_chunk_mb: float | None = 128.0,
) -> Path:
    """Build a kerchunk reference over ``files``, optionally merging in a grid file.

    Parameters
    ----------
    files
        Files for **one** output stream (mixing streams with different variables fails).
    out
        Reference path to write (``.parquet`` or ``.json``).
    grid
        A separate static-coordinate file to merge in, so the store carries lon/lat
        and any grid parameters and needs no companion file. (For ROMS this is the
        grid file, giving lon/lat/h/mask plus the s-coordinate parameters.)
    concat_dim, loadable_variables
        Both **detected from the first file** by :func:`detect_concat` when left
        ``None`` — no per-model configuration. Pass them to override, e.g. for a
        model whose files should be joined along something other than time.
    keep
        Which records to keep from each file before concatenating. ``"all"``
        (default) keeps every record. ``"latest-per-file"`` keeps only the record
        with the latest time value in each file — the fix for ROMS restart files,
        which write more than one time record per file and, under cycling restarts,
        do not always write the newest one last. Selection happens per file, before
        concatenation, so :func:`_warn_if_concat_axis_is_disordered` still runs
        afterward and will warn about any overlap *between* files (e.g. a restart
        stream re-covering time an earlier run already wrote) — this only removes
        the within-file duplication, not that.
    target_chunk_mb
        Automatic manifest subchunking, on by default: any *uncompressed* variable
        whose stored chunk exceeds this many megabytes is split (see
        :func:`_auto_subchunk_spec`) along its leading dims only -- for ROMS-shaped
        output, that means time and/or the vertical, never the horizontal. A
        one-time-record-per-file chunk of ``(1, 100, 962, 1858)`` float64 (1.33 GiB)
        becomes ``(1, 5, 962, 1858)`` (~68 MB) at the default. Pass ``None`` to
        disable and keep whatever chunk grid the source files carry, unsplit.
        Silently leaves a compressed variable's chunks alone (splitting a
        compressed chunk is not possible after the fact — see ``subchunk`` below).
    subchunk
        Explicit manifest subchunking: ``{dim: resulting_chunk_length}``, the same
        shape as ``ds.chunk()``. Layers on top of (and overrides, per dim named)
        whatever ``target_chunk_mb`` chose automatically. Splitting only ever works
        along a variable's *outermost* dims -- once every dim before the one named
        is already a single chunk, that dim's bytes are contiguous and can be cut;
        a dim in the middle needs the ones before it named too (the error says so).
        This is why automatic mode never touches the trailing two (horizontal)
        dims, but ``subchunk`` can if a caller has reason to, e.g.
        ``{"eta_rho": 200}``. Raises on a spec that does not fit the store (wrong
        divisor, or a dim that is not actually outermost) rather than skipping —
        unlike a compressed variable, a bad request is something the caller can
        fix. Values are byte-identical to an unsplit build either way; only the
        read granularity changes.

    Notes
    -----
    Global attributes are merged with ``combine_attrs="drop_conflicts"``: attributes the
    output and grid agree on (``theta_s``/``theta_b``/``hc``) are kept, and ones that
    clash (e.g. ``title``) are dropped rather than raising.
    """
    import xarray as xr
    from obspec_utils.registry import ObjectStoreRegistry
    from virtualizarr import open_virtual_dataset, open_virtual_mfdataset

    if keep not in ("all", "latest-per-file"):
        raise ValueError(
            f"make_kerchunk: keep={keep!r} not recognized; use 'all' or "
            "'latest-per-file'"
        )
    # Deliberately not Path() for remote sources: Path collapses the "//" in a URL
    # to "/", so http://host/f.nc becomes http:/host/f.nc and every downstream check
    # then treats it as a local relative path.
    paths = [str(f) if _is_remote(f) else Path(f).expanduser() for f in files]
    if not paths:
        raise ValueError("make_kerchunk: no files given")
    if concat_dim is None or loadable_variables is None:
        detected_dim, detected_loadable = detect_concat(paths[0])
        concat_dim = detected_dim if concat_dim is None else concat_dim
        loadable_variables = (
            detected_loadable if loadable_variables is None else loadable_variables
        )
    parser = _parser_for(paths[0])
    if grid is not None and not _is_remote(grid):
        grid = Path(grid).expanduser()
    preprocess = (
        _keep_latest_per_file(concat_dim, loadable_variables)
        if keep == "latest-per-file"
        else None
    )

    ctx = tolerant_hdf_attrs() if tolerant_attrs else contextlib.nullcontext()
    with ctx:
        urls = dict(_store_for(p) for p in paths)
        vds = open_virtual_mfdataset(
            urls=list(urls),
            registry=ObjectStoreRegistry(urls),
            parser=parser,
            combine="nested",
            concat_dim=concat_dim,
            loadable_variables=list(loadable_variables),
            preprocess=preprocess,
        )
        _warn_if_concat_axis_is_disordered(vds, concat_dim, loadable_variables, paths)

        if grid is not None:
            gurl, gstore = _store_for(grid)
            # its own parser: a netCDF-3 grid beside netCDF-4 output is the norm
            # for ROMS, and the two formats need different readers
            vgrid = open_virtual_dataset(
                url=gurl,
                registry=ObjectStoreRegistry({gurl: gstore}),
                parser=_parser_for(grid),
            )
            vds = xr.merge(
                [vds, vgrid],
                compat="override",
                join="override",
                combine_attrs="drop_conflicts",
            )

    spec = (
        _auto_subchunk_spec(vds, target_chunk_mb * 1024**2)
        if target_chunk_mb is not None
        else {}
    )
    if subchunk:
        spec = {**spec, **subchunk}  # explicit wins per dim named; auto fills the rest
    if spec:
        vds = _subchunk(vds, spec)

    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    vds.vz.to_kerchunk(str(out), format=fmt or _kerchunk_format(out))
    return out


def _kerchunk_format(out: Path) -> str:
    """Kerchunk target format implied by ``out``'s extension.

    The extension has to drive this: a ``.json`` path written in parquet format
    silently produces a *directory* named ``x.json``, and the mismatch only surfaces
    much later as ``IsADirectoryError`` when something tries to read it back.
    """
    return "json" if out.suffix.lower() == ".json" else "parquet"


# ------------------------------------------------------------------------- subchunking


def _uncompressed(marr) -> bool:
    """Whether a :class:`~virtualizarr.manifests.ManifestArray`'s chunks are raw bytes.

    A stored chunk can only be split into byte-range sub-chunks if it carries no
    compression: an endian-only ``BytesCodec`` is fine (that's what an uncompressed
    HDF5 variable -- or a big-endian netCDF3 one -- carries), but any shuffle/deflate
    codec riding after it means the chunk is one opaque compressed blob, and splitting
    its byte range would slice into the compressed stream rather than the data.
    """
    import zarr

    codecs = marr.metadata.codecs
    return len(codecs) == 1 and isinstance(codecs[0], zarr.codecs.BytesCodec)


def _split_marr_axis(
    marr, axis: int, new_size: int, *, name: str, dim: str, dims: tuple[str, ...]
):
    """Split one axis of a ManifestArray's chunk grid into ``new_size``-length pieces.

    Works because a stored chunk is one contiguous run of bytes in C (row-major)
    order: once every axis before ``axis`` is already size 1 in the chunk grid, the
    bytes along ``axis`` are contiguous too, and can be cut into equal sub-ranges by
    offset arithmetic alone -- no bytes are read or moved, only the manifest's
    (path, offset, length) triples change. Everything after ``axis`` rides along
    unchanged in every sub-chunk. ``dims`` is ``marr``'s dimension names (``.zarray``
    metadata does not reliably carry them), used only to spell a clear error.

    Raises rather than skips on a bad request: an ineligible (compressed, inlined)
    variable is filtered out by the caller before this is ever reached (see
    :func:`_subchunk`), so reaching here with something unsplittable means the
    *request* was wrong, not the file -- worth failing loudly on rather than quietly
    doing nothing.
    """
    import numpy as np
    from virtualizarr.manifests import ChunkManifest, ManifestArray
    from virtualizarr.manifests.utils import copy_and_replace_metadata

    chunks = marr.metadata.chunks
    length = chunks[axis]
    if length % new_size != 0:
        raise ValueError(
            f"{name!r}: subchunk={{{dim!r}: {new_size}}} does not divide its stored "
            f"chunk length along {dim!r} ({length}) evenly."
        )
    factor = length // new_size
    if factor == 1:
        return marr  # already at (or coarser than) the requested size
    if any(c != 1 for c in chunks[:axis]):
        leading = dict(zip(dims[:axis], chunks[:axis], strict=True))
        raise ValueError(
            f"{name!r}: cannot subchunk {dim!r} on its own -- its chunk grid is not "
            f"size 1 along the dims before it ({leading}), so those bytes are not "
            "yet contiguous along this axis alone. This is the shape a "
            "*contiguously*-stored (unchunked) file carries: add the leading "
            f"dim(s) to subchunk= too, e.g. {{'{dims[axis - 1]}': 1, {dim!r}: "
            f"{new_size}}}."
        )

    man = marr.manifest
    if man._inlined:
        raise ValueError(
            f"{name!r} has inlined chunk data; subchunk= only rewrites byte-range "
            "references into the source file, so an inlined variable cannot be "
            "split."
        )
    paths, offsets, lengths = man._paths, man._offsets, man._lengths
    itemsize = marr.dtype.itemsize
    tail_shape = chunks[axis + 1 :]
    tail = int(np.prod(tail_shape, dtype=np.int64)) if tail_shape else 1
    sub_len = new_size * tail * itemsize

    # Every present chunk must be exactly prod(chunk_shape) * itemsize: a compressed
    # chunk's length varies record to record (that's what _uncompressed already
    # rules out), but this catches anything else unexpected -- a partial edge chunk,
    # say -- before it corrupts the split rather than after.
    expected = int(np.prod(chunks, dtype=np.int64)) * itemsize
    present = paths != ""
    if present.any() and not bool(np.all(lengths[present] == expected)):
        raise ValueError(
            f"{name!r}: not every stored chunk is exactly {expected} bytes -- "
            "refusing to subchunk it (an unexpectedly-shaped or partial chunk)."
        )

    g = paths.shape[axis]
    new_paths = np.repeat(paths, factor, axis=axis)
    new_lengths = np.full(new_paths.shape, np.uint64(sub_len), dtype=np.uint64)
    bcast_shape = [1] * paths.ndim
    bcast_shape[axis] = g * factor
    increments = (
        np.tile(np.arange(factor, dtype=np.uint64), g) * np.uint64(sub_len)
    ).reshape(bcast_shape)
    new_offsets = np.repeat(offsets, factor, axis=axis) + increments

    new_manifest = ChunkManifest.from_arrays(
        paths=new_paths, offsets=new_offsets, lengths=new_lengths, validate_paths=False
    )
    new_chunks = list(chunks)
    new_chunks[axis] = new_size
    new_metadata = copy_and_replace_metadata(marr.metadata, new_chunks=new_chunks)
    return ManifestArray(metadata=new_metadata, chunkmanifest=new_manifest)


def _subchunk(vds, spec):
    """Split each eligible variable's manifest chunks along the dims named in ``spec``.

    ``spec`` gives the *resulting* chunk length along each dim, ``ds.chunk()``-style
    -- e.g. ``{"s_rho": 5}`` turns a chunk of 100 into 20 pieces of 5. Applied per
    variable, dim by dim in ascending axis-index order (outermost first): splitting
    an axis needs every axis before it already collapsed to one chunk (see
    :func:`_split_marr_axis`), which is exactly why a contiguously-stored file needs
    its leading (record) dim named in ``spec`` too, ahead of the vertical.

    A variable that carries none of the named dims, or is not itself a
    :class:`~virtualizarr.manifests.ManifestArray` (a merged grid field, a
    ``loadable_variables`` entry already read into memory), passes through
    untouched. One that is compressed, or has inlined chunk data, is skipped with a
    warning rather than failing the whole build: that is a property of the source
    file the caller cannot change by asking differently, unlike a bad ``spec``
    (wrong dim, wrong divisor), which still raises.
    """
    from virtualizarr.manifests import ManifestArray

    out = vds.copy()
    for name, var in vds.variables.items():
        marr = var.data
        if not isinstance(marr, ManifestArray):
            continue
        axes = sorted(
            (var.dims.index(d), d, size) for d, size in spec.items() if d in var.dims
        )
        if not axes:
            continue
        if marr.manifest._inlined:
            warnings.warn(
                f"{name!r} has inlined chunk data and was left as stored -- "
                "subchunk= only rewrites byte-range references into the source "
                "file.",
                stacklevel=2,
            )
            continue
        if not _uncompressed(marr):
            warnings.warn(
                f"{name!r} is compressed and was left as stored -- a compressed "
                "chunk is the unit its codec decompresses, so it cannot be split "
                "after the fact. Choose a smaller chunk shape when the file "
                "itself is written instead.",
                stacklevel=2,
            )
            continue
        for axis, dim, size in axes:
            marr = _split_marr_axis(
                marr, axis, size, name=name, dim=dim, dims=var.dims
            )
        out[name].data = marr
    return out


def _auto_subchunk_spec(vds, target_bytes: float) -> dict[str, int]:
    """Derive a ``subchunk=`` spec that brings the largest chunk under ``target_bytes``.

    Automatic mode never touches the trailing two dims of a variable (the
    horizontal, for anything gridded) -- only the leading ones, time and/or depth in
    practice -- since those are what a user might reasonably want split without
    being asked, and a store is rarely written chunked across latitude/longitude in
    a way splitting could even help with. Works from the single worst-case variable
    (the biggest uncompressed, non-inlined chunk in the store): walks its dims
    outside-in, and for each picks the largest divisor of that dim's current chunk
    length that gets the chunk under target -- the coarsest split that works, not
    the finest -- moving to the next dim only if that one alone was not enough.

    Returns ``{}`` (no-op) if nothing needs splitting, or if the store has no
    eligible variable at all (all compressed, or none over target).
    """
    import numpy as np
    from virtualizarr.manifests import ManifestArray

    worst: tuple[str, tuple[str, ...], list[int], int, int] | None = None
    for name, var in vds.variables.items():
        marr = var.data
        if not isinstance(marr, ManifestArray):
            continue
        if marr.manifest._inlined or not _uncompressed(marr):
            continue
        itemsize = marr.dtype.itemsize
        nbytes = int(np.prod(marr.metadata.chunks, dtype=np.int64)) * itemsize
        if worst is None or nbytes > worst[4]:
            worst = (name, var.dims, list(marr.metadata.chunks), itemsize, nbytes)

    if worst is None or worst[4] <= target_bytes:
        return {}

    name, dims, chunks, itemsize, nbytes = worst
    spec: dict[str, int] = {}
    for axis in range(max(len(dims) - 2, 0)):
        if int(np.prod(chunks, dtype=np.int64)) * itemsize <= target_bytes:
            break
        length = chunks[axis]
        best = None
        for new_size in range(length, 0, -1):
            if length % new_size:
                continue
            trial = chunks.copy()
            trial[axis] = new_size
            if int(np.prod(trial, dtype=np.int64)) * itemsize <= target_bytes:
                best = new_size
                break
        chunks[axis] = best if best is not None else 1
        if chunks[axis] != length:
            spec[dims[axis]] = chunks[axis]

    if int(np.prod(chunks, dtype=np.int64)) * itemsize > target_bytes:
        warnings.warn(
            f"{name!r}'s chunk ({nbytes / 1024**2:.0f} MB) could not be brought "
            f"under target_chunk_mb={target_bytes / 1024**2:.0f} by splitting its "
            "leading dims alone -- pass an explicit subchunk= naming a horizontal "
            "dim to go further, or accept the remaining size.",
            stacklevel=2,
        )
    return spec


def _decode_times(ds, var):
    """Return ``var`` as datetime64 where possible, else ``None``.

    Handles three cases: already-decoded datetimes; standard CF units (via xarray's
    decoder); and ROMS' non-CF ``units="second"`` with the epoch buried in ``long_name``
    ("Time since 2000/01/01"). Returns ``None`` for units nothing can decode — notably
    climatologies like WOA's ``"months since 1965-01-01"``, where a date range isn't
    meaningful anyway (see ``climatology_period`` instead).
    """
    import re

    import numpy as np
    import xarray as xr

    vals = np.asarray(var)
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals
    units = str(var.attrs.get("units", ""))

    if " since " in units.lower():
        try:
            return np.asarray(
                xr.coding.times.decode_cf_datetime(
                    vals, units, var.attrs.get("calendar", "standard")
                )
            )
        except Exception:
            return None  # e.g. "months since ..." — not a fixed-length unit

    # ROMS: units="second", epoch only in the long_name text
    text = f"{var.attrs.get('long_name', '')} {units}"
    m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", text)
    if not m or not units.lower().startswith(("second", "s")):
        return None
    epoch = np.datetime64(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return epoch + vals.astype("timedelta64[s]")


# ---------------------------------------------------------------------- featureType

#: CF standard-name modifiers (Appendix C). A ``standard_name`` of "<name> <modifier>"
#: marks an auxiliary field (uncertainty, counts, flags) — not an independently
#: comparable variable, so these are recorded but kept out of ``variables``.
_SN_MODIFIERS = frozenset(
    {
        "detection_minimum",
        "number_of_observations",
        "standard_error",
        "status_flag",
    }
)

#: Canonical CF featureType spellings, keyed by lowercase (datasets are inconsistent —
#: WOA declares "Grid", CF wants camelCase for the point types).
_FEATURE_TYPES = {
    ft.lower(): ft
    for ft in (
        "point",
        "timeSeries",
        "trajectory",
        "profile",
        "timeSeriesProfile",
        "trajectoryProfile",
        "grid",
    )
}


def _canonical_feature_type(value: str) -> str:
    """Normalize a declared featureType to its canonical spelling ("Grid" -> "grid")."""
    return _FEATURE_TYPES.get(str(value).strip().lower(), str(value).strip())


def guess_feature_type(ds) -> tuple[str, str]:
    """Guess a CF ``featureType`` for ``ds``; returns ``(featureType, source)``.

    ``source`` is ``"declared"`` when the dataset says so itself, else ``"inferred"``.
    Inference uses the dimensionality of the cf-xarray-detected X/Y/Z/T axes:

    ==========================  ===  ===  =====================
    X/Y                         T    Z    featureType
    ==========================  ===  ===  =====================
    2-D, or two distinct dims   any  any  ``grid``
    fixed point (size 1)        yes  no   ``timeSeries``
    fixed point (size 1)        no   yes  ``profile``
    fixed point (size 1)        yes  yes  ``timeSeriesProfile``
    vary along one shared dim   yes  no   ``trajectory``
    vary along one shared dim   yes  yes  ``trajectoryProfile``
    ==========================  ===  ===  =====================

    ``trajectory`` and ``timeSeries`` have the same shape, so they are separated by
    whether the positions actually vary. The guess is a default — override it in the
    catalog when it matters.
    """
    import numpy as np

    declared = ds.attrs.get("featureType")
    if declared:
        return _canonical_feature_type(declared), "declared"

    x = find_coord(ds, "longitude")
    y = find_coord(ds, "latitude")
    t = find_coord(ds, "time")
    z = find_coord(ds, "vertical")

    if x is None or y is None:
        return "grid", "inferred"  # no positional info; treat as gridded

    xy_dims = set(x.dims) | set(y.dims)
    horizontal_dims = {d for d in xy_dims if ds.sizes.get(d, 1) > 1}
    has_t = t is not None and t.size > 1
    has_z = z is not None and z.size > 1

    # two independent horizontal dims (or 2-D curvilinear coords) => gridded
    if len(horizontal_dims) >= 2:
        return "grid", "inferred"

    if len(horizontal_dims) == 0:  # a fixed point
        if has_t and has_z:
            return "timeSeriesProfile", "inferred"
        if has_t:
            return "timeSeries", "inferred"
        if has_z:
            return "profile", "inferred"
        return "point", "inferred"

    # positions vary along exactly one dim: trajectory if they really move
    moves = False
    try:
        moves = bool(np.nanstd(np.asarray(x)) > 0 or np.nanstd(np.asarray(y)) > 0)
    except Exception:  # pragma: no cover - defensive
        moves = True
    if moves:
        return ("trajectoryProfile" if has_z else "trajectory"), "inferred"
    return ("timeSeriesProfile" if has_z else "timeSeries"), "inferred"


# ------------------------------------------------------------------------ resolution

#: km per degree of latitude (mean meridional degree on the WGS84 ellipsoid).
_KM_PER_DEG = 111.195

#: featureTypes with a horizontal grid. Everything else is a point, a profile or a
#: track, where a "horizontal resolution" would describe nothing.
_GRIDDED_FEATURE_TYPES = frozenset({"grid"})

#: fraction by which spacings may vary and still count as evenly spaced.
_REGULAR_TOL = 0.01


def _spacing(values) -> tuple[float, bool] | None:
    """Median absolute spacing of a coordinate, and whether it is even.

    The median rather than the mean because composite products are not evenly
    spaced and the mean quietly launders that: MODIS 8-day bins restart every
    January 1, so one bin a year spans 5 days, and hourly model output has
    outages. The median reports the spacing that actually dominates, and the
    ``regular`` flag says whether reporting a single number was honest at all.
    """
    import numpy as np

    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if arr.ndim > 1:  # curvilinear: step along the axis that actually varies
        arr = arr[0, :] if arr.shape[1] > 1 else arr[:, 0]
    if arr.size < 2:
        return None
    diffs = np.abs(np.diff(np.asarray(arr, dtype="float64")))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if not diffs.size:
        return None
    median = float(np.median(diffs))
    if median <= 0:
        return None
    regular = bool(np.all(np.abs(diffs - median) <= _REGULAR_TOL * median))
    return median, regular


def _iso_duration(seconds: float) -> str:
    """Seconds -> an ISO-8601 duration, snapped to the period a product means.

    Composite periods are never exact — a "monthly" mean steps 28-31 days and a
    "daily" one drifts by minutes when the files are stamped at acquisition time —
    so a literal conversion would render P1M as P30DT10H27M and make two monthly
    products look like different cadences.
    """
    days = seconds / 86400.0
    if 27 <= days <= 32:
        return "P1M"
    if 355 <= days <= 375:
        return "P1Y"
    if days >= 1:
        rounded = round(days)
        close = abs(days - rounded) <= 0.05 * days
        return f"P{rounded:g}D" if close else f"P{days:.1f}D"
    hours = seconds / 3600.0
    if hours >= 1:
        return f"PT{round(hours):g}H"
    minutes = seconds / 60.0
    return f"PT{round(minutes):g}M" if minutes >= 1 else f"PT{round(seconds):g}S"


def _resolution_metadata(
    ds, coords: dict[str, Any], feature_type: str
) -> dict[str, Any]:
    """Horizontal, temporal and vertical resolution, derived from the axes themselves.

    Read rather than believed, for the same reason the extents are: products
    misdeclare it. CoastWatch's Metop-C ASCAT product advertises 0.25 degrees in
    both its title and ``geospatial_lat_resolution`` while its latitude axis steps
    0.3333; AVISO's ``erdTAgeo1day`` is titled "1 Day Composite" and steps about six
    days. A catalog that copied the attribute would have published both.

    Horizontal resolution is recorded only for gridded sources. A moored time series
    has a location, not a spacing, and storing one would invite a
    ``find(resolution=...)`` to rank buoys against satellites.
    """
    import numpy as np

    md: dict[str, Any] = {}

    if feature_type in _GRIDDED_FEATURE_TYPES:
        steps: dict[str, float] = {}
        regular = True
        for kind, key in (("latitude", "lat"), ("longitude", "lon")):
            if (coord := coords.get(kind)) is None:
                continue
            if (found := _spacing(coord)) is None:
                continue
            steps[key], even = found
            regular &= even
        if steps:
            lat_deg = steps.get("lat", steps.get("lon"))
            # Latitude drives the scalar: a degree of longitude shrinks with cos(lat),
            # so a global product reporting its lon spacing would state a resolution
            # true only at the equator.
            md["grid_resolution_deg"] = round(lat_deg, 6)
            md["grid_resolution_km"] = round(lat_deg * _KM_PER_DEG, 3)
            md["grid_regular"] = regular
            # Both axes recorded only when they disagree -- on every product checked
            # they match, so carrying two near-identical numbers everywhere buys
            # nothing, while an anisotropic grid is worth seeing.
            if len(steps) == 2 and not np.isclose(
                steps["lat"], steps["lon"], rtol=1e-3
            ):
                md["grid_resolution_lat_deg"] = round(steps["lat"], 6)
                md["grid_resolution_lon_deg"] = round(steps["lon"], 6)

    if (time_coord := coords.get("time")) is not None:
        decoded = _decode_times(ds, time_coord)
        if decoded is not None and decoded.size > 1:
            as_seconds = np.asarray(decoded, dtype="datetime64[s]").astype("float64")
            if (found := _spacing(as_seconds)) is not None:
                step, _even = found
                md["time_resolution_s"] = round(step, 3)
                md["time_resolution"] = _iso_duration(step)

    if (vertical := coords.get("vertical")) is not None:
        arr = np.asarray(vertical)
        if arr.ndim == 1 and arr.size > 1:
            md["vertical_levels"] = int(arr.size)
            diffs = np.abs(np.diff(np.asarray(arr, dtype="float64")))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                # A range, not a scalar: ocean level spacing is stretched by design
                # (GLORYS is ~1 m at the surface and ~450 m at depth), so one number
                # would be wrong nearly everywhere.
                md["vertical_resolution_min"] = round(float(diffs.min()), 3)
                md["vertical_resolution_max"] = round(float(diffs.max()), 3)
    return md


# --------------------------------------------------------------------------- THREDDS


def discover_opendap_files(
    directory_url: str,
    pattern: str = "*.nc",
    *,
    recurse: bool = False,
    max_dirs: int = 500,
) -> list[str]:
    """List data-access URLs under a THREDDS/Hyrax OPeNDAP directory.

    Both server families publish a THREDDS ``catalog.xml`` for every directory —
    this fetches and parses it rather than scraping the HTML page, so the browser
    catalog URL (``.../catalog/path/catalog.html``) can be pasted in directly.
    Returns one OPeNDAP URL per leaf dataset (a real file, not a sub-directory)
    whose filename matches ``pattern`` (:mod:`fnmatch` syntax). The two families
    place data differently, and each leaf says which it is:

    - A true THREDDS Data Server (NCEI's ``thredds-ocean``, for example) serves
      catalogs under ``/catalog/`` but data under a separate OPeNDAP service base
      (``/thredds-ocean/dodsC/``); its leaves carry the data path in a ``urlPath``
      attribute, which is joined onto that declared base.
    - Hyrax (NASA's ``oceandata.sci.gsfc.nasa.gov``) serves ``catalog.xml``
      alongside the data itself and its leaves have no ``urlPath`` attribute, so
      the URL is the directory joined with the dataset's own name — the same form
      the directory's browser links use, which is more reliable than Hyrax's
      declared service base (installations have been seen to point it at the
      wrong path).

    Parameters
    ----------
    recurse
        Follow ``<catalogRef>`` sub-directory links too (breadth-first), not just
        list ``directory_url`` itself. **Scope this deliberately**: MODIS Aqua alone
        is a year -> 365 day directories -> ~500 files each, and that is one of
        several temporal products (daily/8-day/monthly/monthly-climatology/annual) at
        one of two resolutions — "the whole archive" is tens of millions of URLs, not
        a few thousand. Point ``directory_url`` at the narrowest directory that
        contains what you actually want (one year, one product) rather than the
        server root.
    max_dirs
        Hard cap on directories visited when ``recurse=True`` — a backstop against
        pointing this at something far larger than intended, not a target to reach
        for. Raises once exceeded rather than silently truncating the results, since
        a partial crawl that looks complete is worse than one that visibly stopped.

    Use this for a THREDDS/Hyrax server that lets you browse its directory structure;
    for a predictable per-date URL scheme (as MODIS L3 files have) a plain URL
    template plus a date range needs no network round-trip per file at all — see the
    OceanSODA/MODIS examples in ``docs/``.
    """
    import fnmatch
    import xml.etree.ElementTree as ET
    from collections import deque
    from urllib.parse import urljoin, urlsplit
    from urllib.request import urlopen

    ns = {"t": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}

    for suffix in ("catalog.html", "catalog.xml"):
        directory_url = directory_url.removesuffix(suffix)

    def _list_one(url: str):
        base = url.rstrip("/") + "/"
        with urlopen(base + "catalog.xml", timeout=30) as resp:
            root = ET.fromstring(resp.read())
        service_base = next(
            (
                svc.attrib["base"]
                for svc in root.iter(f"{{{ns['t']}}}service")
                if svc.attrib.get("serviceType", "").lower() == "opendap"
                and svc.attrib.get("base")
            ),
            None,
        )
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(base))
        files, subdirs = [], []
        # A leaf file carries <dataSize>; a <catalogRef> is a sub-directory link
        # (followed only if recurse=True) — everything else (the outer wrapping
        # <dataset>) is neither and is skipped.
        for ds in root.iter(f"{{{ns['t']}}}dataset"):
            name = ds.attrib.get("name", "")
            if ds.find("t:dataSize", ns) is None or not fnmatch.fnmatch(name, pattern):
                continue
            url_path = ds.attrib.get("urlPath")
            if url_path and service_base:  # TDS leaf; Hyrax leaves have no urlPath
                files.append(
                    urljoin(origin, service_base.rstrip("/") + "/")
                    + url_path.lstrip("/")
                )
            else:
                files.append(base + name)
        for ref in root.iter(f"{{{ns['t']}}}catalogRef"):
            href = ref.attrib.get("{http://www.w3.org/1999/xlink}href", "")
            if href:
                subdirs.append(urljoin(base, href).removesuffix("catalog.xml"))
        return files, subdirs

    if not recurse:
        return _list_one(directory_url)[0]

    urls: list[str] = []
    import warnings

    queue = deque([directory_url])
    visited = 0
    while queue:
        if visited >= max_dirs:
            raise RuntimeError(
                f"discover_opendap_files: stopped after {max_dirs} directories "
                f"({len(urls)} files found so far) — narrow directory_url, or raise "
                "max_dirs if you deliberately want a crawl this large."
            )
        current = queue.popleft()
        visited += 1
        try:
            files, subdirs = _list_one(current)
        except Exception as exc:  # a server hiccup on one dir shouldn't kill the crawl
            warnings.warn(f"skipping {current} ({exc})", stacklevel=2)
            continue
        urls.extend(files)
        queue.extend(subdirs)
    return urls


# -------------------------------------------------------------------------- catalog


def new_catalog(**metadata: Any):
    """Return an empty intake v2 catalog with catalog-level metadata."""
    import intake

    return intake.entry.Catalog(metadata=metadata)


def save(cat, path: str | Path) -> Path:
    """Write ``cat`` to ``path`` (intake v2 YAML), updating catalog-level extents."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    _rollup_metadata(cat)
    cat.to_yaml_file(str(path))
    _print_match_summary(path, cat)
    return path


def _print_match_summary(path: Path, cat) -> None:
    """Print a one-time vocabulary match summary for the catalog just saved.

    Nothing here is written into ``cat`` or the YAML file -- only printed, and only
    for this one build. A persisted ``{nickname: [variables]}`` map would go stale
    the moment the vocabulary gains a new alias or pattern and nobody rebuilds this
    catalog; see :class:`ocean_skill.vocabulary.MatchReport`. Rerun
    :func:`ocean_skill.match_report` on the saved catalog any time afterward for a
    report against whatever the vocabulary looks like *then* -- no rebuild needed.
    """
    from ocean_skill.vocabulary import match_report

    names = cat.metadata.get("standard_names") or []
    n_sources = len(list(cat) or list(getattr(cat, "entries", {})))
    report = match_report(names)
    n_matched = sum(len(v) for v in report.matched.values())
    print(
        f"ocean-skill: {path.name} — {len(names)} variables across {n_sources} "
        f"sources: {n_matched} matched, {len(report.unmatched)} unmatched"
    )
    if report.unmatched:
        print(f"  unmatched: {', '.join(report.unmatched)}")


def _rollup_metadata(cat) -> None:
    """Set catalog-level envelope (extents, featureTypes, names) from entries."""
    mins, maxs = {}, {}
    ftypes, snames = set(), set()
    for name in list(cat) or list(getattr(cat, "entries", {})):
        try:
            md = cat[name].metadata
        except Exception:  # pragma: no cover - defensive
            continue
        for k in (
            "geospatial_lat_min",
            "geospatial_lon_min",
            "geospatial_vertical_min",
        ):
            if k in md:
                mins[k] = min(mins.get(k, md[k]), md[k])
        for k in (
            "geospatial_lat_max",
            "geospatial_lon_max",
            "geospatial_vertical_max",
        ):
            if k in md:
                maxs[k] = max(maxs.get(k, md[k]), md[k])
        if md.get("featureType"):
            ftypes.add(md["featureType"])
        snames.update(md.get("variables") or [])
        for k, agg in (("time_coverage_start", min), ("time_coverage_end", max)):
            if k in md:
                cur = cat.metadata.get(k)
                cat.metadata[k] = agg(cur, md[k]) if cur else md[k]
    cat.metadata.update(mins)
    cat.metadata.update(maxs)
    if ftypes:
        cat.metadata["featureTypes"] = sorted(ftypes)
    if snames:
        cat.metadata["standard_names"] = sorted(snames)


def _reader_for(url: str, storage_options: dict[str, Any] | None = None, **kwargs):
    """Build an intake reader for ``url`` (kerchunk ref, OPeNDAP, or NetCDF).

    ``storage_options`` are fsspec options and belong on the *data* object, not the
    reader — that is where intake looks for them. Use them for caching and remote
    access, e.g. ``{"simplecache": {"cache_storage": "./cache", "same_names": True}}``
    with a ``simplecache::`` URL.

    ERDDAP is deliberately not one of the protocols dispatched here — see
    :func:`add_erddap_source`/:func:`add_erddap_catalog` instead, which build via
    ``intake_erddap``'s own reader classes directly rather than this function
    inventing a URL convention to parse.
    """
    from intake.readers import datatypes, readers

    # Local paths are stored absolute: a catalog holding relative paths only resolves
    # from the directory it was built in, which breaks as soon as a notebook or script
    # in a subdirectory opens it.
    if "://" not in str(url):
        candidate = Path(str(url)).expanduser()
        if candidate.exists():
            url = str(candidate.resolve())

    # decode_times=False by default: ocean data is full of non-CF time units (ROMS'
    # "second", WOA's "months since ...") xarray refuses. We decode in _decode_times.
    kwargs = {"decode_times": False, **kwargs}
    low = str(url).lower()
    so = storage_options or None

    if low.endswith((".parquet", ".json")):
        data = datatypes.HDF5(url=str(url))  # placeholder type; engine drives the read
        kwargs.setdefault("engine", "kerchunk")
        kwargs.setdefault("chunks", {})
        return readers.XArrayDatasetReader(data, **kwargs)
    # ERDDAP griddap *is* a DAP endpoint, so xarray opens it the same way — but the URL
    # says neither "opendap" nor "dods", and without this it falls through to the
    # NetCDF branch and fails. griddap matters because one griddap dataset is the whole
    # time series (7,115 daily MODIS steps in one entry), which is the alternative to a
    # catalog holding one entry per daily file.
    dap = ("griddap", "dodsc", "opendap")
    if any(token in low for token in dap) or low.endswith(".nc.dods"):
        return readers.XArrayDatasetReader(datatypes.OpenDAP(url=str(url)), **kwargs)

    # An ARCO Zarr store, which the HDF5 branch below would hand to h5netcdf and fail
    # on. ``engine`` is set here rather than left to the caller so that passing it in
    # ``reader_kwargs`` is not required; note that Copernicus Marine's stores are Zarr
    # **v2** and zarr-python 3 probes for v3 first, so those need
    # ``reader_kwargs={"zarr_format": 2}`` -- without it the 403 from the v3 probe
    # surfaces as an authentication failure rather than a version mismatch.
    if low.endswith(".zarr") or ".zarr/" in low:
        data = datatypes.Zarr(url=str(url), storage_options=so)
        kwargs.setdefault("engine", "zarr")
        return readers.XArrayDatasetReader(data, **kwargs)

    # Anything fsspec has to open (a remote URL, or a cache/chain like
    # "simplecache::https://...") arrives as a *file object*, which the netcdf4 C
    # backend rejects outright. h5netcdf reads file objects, so it is the right engine
    # for those; keep netcdf4 for plain local paths, where it is faster. An explicit
    # ``engine`` in ``reader_kwargs`` overrides this auto-detection — e.g. a remote
    # *classic-format* netCDF3 file (magic ``CDF\x01``) needs ``engine="scipy"``,
    # since h5netcdf can only read netCDF4/HDF5.
    remote = "://" in str(url)
    kwargs.setdefault("engine", "h5netcdf" if remote else "netcdf4")
    kwargs.setdefault("chunks", {})
    data = datatypes.HDF5(url=str(url), storage_options=so)
    return readers.XArrayDatasetReader(data, **kwargs)


def _roms_metadata(ds) -> dict[str, Any]:
    """Detect ROMS output and return the metadata its adapter needs, else ``{}``.

    Presence of the s-coordinate stretching arrays is the tell. Without this block
    :func:`ocean_skill.sources.read` would not route through
    :mod:`ocean_skill.roms`, so land would never be masked and no depth coordinate
    would be built — a silent correctness bug, since ROMS writes 0.0 (not NaN) on land.
    """
    import re

    if not {"Cs_r", "sigma_r"}.issubset(set(ds.variables)):
        return {}
    md: dict[str, Any] = {
        "model": "roms",
        "loader": "ocean_skill.roms",
        "self_contained_grid": "lon_rho" in ds.variables,
    }
    a = ds.attrs
    if all(k in a for k in ("theta_s", "theta_b", "hc")):
        md["vertical"] = {
            "s_dim": "s_rho",
            "theta_s": float(a["theta_s"]),
            "theta_b": float(a["theta_b"]),
            "hc": float(a["hc"]),
            "Vtransform": 2,
        }
    if "ocean_time" in ds.variables:
        t = ds["ocean_time"]
        text = f"{t.attrs.get('long_name', '')} {t.attrs.get('units', '')}"
        m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", text)
        md.update(
            time_coord="ocean_time",
            time_dim="time",
            time_units="seconds",
            reference_date=(
                f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "2000-01-01"
            ),
        )
    return md


def _domain_outline(lon, lat) -> list[list[float]] | None:
    """Return a YAML-plain ``domain_outline`` for 2-D ``lon``/``lat``, else ``None``.

    A 1-D (rectilinear) source gets nothing here — its ``geospatial_lon/lat_min/max``
    bbox already *is* its perimeter, so :func:`~ocean_skill.comparison._domain_of`
    covers it without this key. Only a curvilinear grid's true, possibly rotated,
    boundary needs recording. Values are plain Python floats rounded to 3 decimals
    (~100 m — far finer than the boundary needs to be, and small in the catalog
    YAML), never a numpy scalar, since ``ruamel``/``pyyaml`` don't know how to dump
    one and intake's ``to_yaml_file`` would otherwise fail on exactly the metadata
    this adds.
    """
    import numpy as np

    from ocean_skill.align import perimeter_of

    if lon is None or lat is None:
        return None
    lon, lat = np.asarray(lon), np.asarray(lat)
    if lon.ndim != 2 or lat.ndim != 2:
        return None
    ring = perimeter_of(lon, lat)
    if ring is None:
        return None
    return [[round(float(lo), 3), round(float(la), 3)] for lo, la in ring]


def _probe(ds, name_map: dict[str, str] | None) -> dict[str, Any]:
    """Derive extents, axis mapping, variable mapping and featureType from a dataset.

    Works for a gridded ``xarray.Dataset`` or a tabular (point) ``pandas.DataFrame``
    alike — dispatching here rather than each reader/protocol needing its own
    metadata-deriving code, so an ERDDAP source, say, is probed the same way as a
    kerchunk one: by reading it and looking, not by a source-specific shortcut.
    """
    if hasattr(ds, "columns"):
        return _probe_dataframe(ds)
    import numpy as np

    md: dict[str, Any] = {}
    coords = {
        k: find_coord(ds, k) for k in ("longitude", "latitude", "time", "vertical")
    }

    # --- coordinate/axis mapping ---
    axes = {
        ax: coords[kind].name
        for ax, kind in (
            ("X", "longitude"),
            ("Y", "latitude"),
            ("T", "time"),
            ("Z", "vertical"),
        )
        if coords[kind] is not None
    }
    if axes:
        md["axes"] = axes

    # --- extents ---
    def _extent(kind):
        v = coords[kind]
        if v is None:
            return None
        arr = np.asarray(v)
        if not arr.size or not np.issubdtype(arr.dtype, np.number):
            return None
        return float(np.nanmin(arr)), float(np.nanmax(arr))

    if (lon := _extent("longitude")) is not None:
        md["geospatial_lon_min"], md["geospatial_lon_max"] = lon
        md["lon_convention"] = "0-360" if lon[1] > 180 else "-180-180"
    if (lat := _extent("latitude")) is not None:
        md["geospatial_lat_min"], md["geospatial_lat_max"] = lat
    lon_coord, lat_coord = coords["longitude"], coords["latitude"]
    if (outline := _domain_outline(lon_coord, lat_coord)) is not None:
        md["domain_outline"] = outline
    if (dep := _extent("vertical")) is not None:
        md["geospatial_vertical_min"], md["geospatial_vertical_max"] = dep

    if coords["time"] is not None:
        t = _decode_times(ds, coords["time"])
        if t is not None and t.size:
            md["time_coverage_start"] = str(
                np.datetime_as_string(np.nanmin(t), unit="D")
            )
            md["time_coverage_end"] = str(np.datetime_as_string(np.nanmax(t), unit="D"))
    if "time_coverage_start" not in md:
        # No usable time axis -- but a product may still declare its coverage in
        # global attributes. MODIS L3-mapped files are the case in point: they carry
        # no time dimension at all, yet state time_coverage_start/end, so reading
        # only the axis left fourteen entries untimed for no reason.
        for key in ("time_coverage_start", "time_coverage_end"):
            value = ds.attrs.get(key)
            if value:
                md[key] = str(value)[:10]  # ISO date; the time of day adds nothing here

    # --- variable -> standard_name (declared attrs win, then the fallback map) ---
    std: dict[str, str] = {}
    auxiliary: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    claimed: set[str] = set()
    for var in ds.data_vars:
        sn = ds[var].attrs.get("standard_name") or (
            name_map.get(str(var)) if name_map else None
        )
        if not sn:
            continue
        _base, _, modifier = str(sn).partition(" ")
        if modifier and modifier in _SN_MODIFIERS:
            auxiliary[str(var)] = str(sn)  # uncertainty/count/flag field
        elif sn in claimed:
            # Several variables can claim one standard_name (WOA's n_an
            # objectively-analyzed vs n_mn statistical mean). First wins; the rest are
            # recorded but not renamed, so the mapping stays one-to-one.
            duplicates[str(var)] = str(sn)
        else:
            std[str(var)] = str(sn)
            claimed.add(str(sn))
    if std:
        md["standard_names"] = std
        md["variables"] = sorted(set(std.values()))
    if auxiliary:
        md["auxiliary_variables"] = auxiliary
    if duplicates:
        md["duplicate_standard_names"] = duplicates

    ftype, source = guess_feature_type(ds)
    md["featureType"] = ftype
    md["featureType_source"] = source
    md.update(_resolution_metadata(ds, coords, ftype))
    md.update(_roms_metadata(ds))  # model-specific block when this is ROMS output
    return md


def _probe_dataframe(df) -> dict[str, Any]:
    """Return the tabular counterpart of :func:`_probe` above.

    Same contract (axes, extents, standard_names, variables, featureType), derived
    by reading the actual data — no source-specific metadata endpoint, so any
    DataFrame-returning reader (ERDDAP or otherwise) is probed identically.

    Columns are read as ``"<name>"``, ``"<name> (<units>)"``, or ``"<name>[<units>]"``
    (the parenthesized form is ``intake_erddap``'s own convention; the bracketed one
    turns up on mooring CSVs that are not ERDDAP-sourced) — coordinate columns
    (time/lon/lat/depth) are recognized under any of those spellings, case-insensitive
    and by regex rather than an exact name, via :func:`ocean_skill.tabular
    .coord_column`. A ``<name>_qc_agg``/``<name>_qc_tests`` QARTOD pair alongside a
    column, or any column whose name contains the word "flag", is recognized and
    excluded (it describes a variable rather than being one), the same
    modifier-exclusion :func:`_probe` applies to a Dataset's auxiliary fields.

    That column vocabulary lives in :mod:`ocean_skill.tabular`, which is also what
    *reads* such a table into xarray — one spelling of the convention, so a catalog
    entry cannot describe a frame differently from how the pipeline later opens it.
    """
    from ocean_skill.tabular import (
        coord_column,
        decode_time_column,
        is_coordinate_column,
        is_qc_column,
        numeric_in_range,
        split_units,
    )

    md: dict[str, Any] = {}

    axes: dict[str, str] = {}
    if lon_col := coord_column(df, "X"):
        # numeric_in_range drops out-of-range fill values (e.g. a "not reported" row
        # written as 9999 rather than left blank) the same way it already drops
        # non-numeric/NaN -- otherwise one such row reports 9999 as this source's
        # easternmost longitude.
        lo = numeric_in_range(df[lon_col], "X").dropna()
        if not lo.empty:
            axes["X"] = lon_col
            md["geospatial_lon_min"], md["geospatial_lon_max"] = (
                float(lo.min()),
                float(lo.max()),
            )
            md["lon_convention"] = (
                "0-360" if md["geospatial_lon_max"] > 180 else "-180-180"
            )
    if lat_col := coord_column(df, "Y"):
        la = numeric_in_range(df[lat_col], "Y").dropna()
        if not la.empty:
            axes["Y"] = lat_col
            md["geospatial_lat_min"], md["geospatial_lat_max"] = (
                float(la.min()),
                float(la.max()),
            )
    if time_col := coord_column(df, "T"):
        # decode_time_column honors a CF "<n> since <date>" encoding stated in the
        # column's own name (e.g. "Time[days_since_1950-01-01T00:00:00Z]") -- plain
        # pd.to_datetime on the raw numbers instead reads them as nanoseconds since
        # the Unix epoch, landing every timestamp within a heartbeat of 1970-01-01.
        t = decode_time_column(df[time_col], time_col).dropna()
        if not t.empty:
            axes["T"] = time_col
            md["time_coverage_start"] = str(t.min().date())
            md["time_coverage_end"] = str(t.max().date())
    if depth_col := coord_column(df, "Z"):
        # Written for the catalog's own sake (search, map hover text -- see
        # plot/locations._format_depth) and as depth_of's rung-3 fallback for a
        # frame whose column depth_of's own (narrower, exact) alias list still
        # misses -- not consulted by coord_column/coord_axis_of again, so a
        # mismatch here can never desync axes from what to_dataset later excludes.
        finite = numeric_in_range(df[depth_col], "Z").dropna()
        if not finite.empty:
            axes["Z"] = depth_col
            md["geospatial_vertical_min"], md["geospatial_vertical_max"] = (
                float(finite.min()),
                float(finite.max()),
            )
    if axes:
        md["axes"] = axes

    std: dict[str, str] = {}
    units: dict[str, str] = {}
    for col in df.columns:
        if is_qc_column(col):
            continue
        base, unit = split_units(col)
        if is_coordinate_column(col):
            continue
        std[col] = base
        if unit:
            units[col] = unit
    if std:
        md["standard_names"] = std
        md["units"] = units
        md["variables"] = sorted(set(std.values()))

    # featureType by which axes actually vary -- the tabular counterpart of
    # guess_feature_type's point-family branch. A gridded product never arrives as a
    # DataFrame (grids go through _probe as a Dataset), so the grid/2-D split it makes
    # on shared dimensions has no analogue here; only the point family applies. A fixed
    # position with time varying is the common ERDDAP-mooring case (timeSeries); a fixed
    # position with depth varying and no spread in time is a CTD-style cast (profile);
    # both varying at a fixed position is a timeSeriesProfile; positions that actually
    # vary make it a trajectory (a profiler among them, a trajectoryProfile). Override
    # via add_source's **metadata when this guess is wrong.
    def _spread(axis: str, *, numeric: bool = True) -> bool:
        col = axes.get(axis)
        if col is None:
            return False
        # numeric_in_range/decode_time_column, not a bare pd.to_numeric/to_datetime:
        # an axis that is actually fixed but carries a stray fill-value row (see
        # numeric_in_range) would otherwise look like it "varies", misclassifying a
        # mooring as a trajectory.
        series = numeric_in_range(df[col], axis) if numeric else decode_time_column(
            df[col], col
        )
        return series.nunique(dropna=True) > 1

    position_varies = _spread("X") or _spread("Y")
    has_z = _spread("Z")
    has_t = _spread("T", numeric=False)
    if position_varies:
        ftype = "trajectoryProfile" if has_z else "trajectory"
    elif has_t and has_z:
        ftype = "timeSeriesProfile"
    elif has_z:
        ftype = "profile"
    else:
        ftype = "timeSeries"
    md["featureType"] = ftype
    md["featureType_source"] = "inferred"
    return md


def _build_reader(reader, url, storage_options, reader_kwargs):
    """Instantiate a named/importable intake reader with its own keyword arguments."""
    if isinstance(reader, str):
        module, _, attr = reader.partition(":")
        reader = getattr(importlib.import_module(module), attr)
    kwargs = dict(reader_kwargs)
    if url is not None:
        kwargs.setdefault("url", str(url))
    if storage_options:
        kwargs.setdefault("storage_options", storage_options)
    return reader(**kwargs)


def _is_catalog(obj) -> bool:
    """Whether ``obj`` is an already-built intake catalog rather than a spec mapping.

    Checked structurally rather than by import, so any object presenting intake's
    catalog surface works. A ``Catalog`` is deliberately *not* a dict/Mapping and has
    no ``.items()``, so it can never be confused with the ``{name: spec}`` form.
    """
    return (
        not isinstance(obj, dict | list | tuple)
        and hasattr(obj, "__getitem__")
        and hasattr(obj, "aliases")
    )


# How probing rides out a flaky remote server. A probe opens each source once, and
# against a server like CalOOS's `sensors.erddap` that read fails now and then for
# reasons that clear on their own -- a 500, a read timeout, a dropped connection. So
# the probe re-attempts a failed read this many times, waiting PROBE_RETRY_BACKOFF
# seconds and doubling each attempt, before the failure counts. Set at module scope
# rather than as a per-call argument so `build_catalog(discovered, out, probe=True)`
# needs nothing extra; assign `osk.build.PROBE_RETRIES = 0` to switch it off, or turn
# it up for an especially unreliable server. Retries only ever cost time on a read
# that was already failing, so a local build never pays for them.
PROBE_RETRIES = 3
PROBE_RETRY_BACKOFF = 1.0


def _read_with_retries(reader, name):
    """Call ``reader.read()``, re-attempting a transient failure a few times.

    A remote sweep against a flaky server — CalOOS's ``sensors.erddap`` is the case
    in point — fails a read now and then for reasons that clear on their own: a 500,
    a read timeout, a dropped connection. This re-attempts such a read with
    exponential backoff (:data:`PROBE_RETRIES` times, :data:`PROBE_RETRY_BACKOFF`
    seconds and doubling) before giving up, so one server hiccup does not cost a
    dataset that would have opened on the very next try.

    Only the *read* is retried, never the metadata derivation that follows it: a read
    failure is the network-shaped one worth re-attempting, whereas a probe failure is
    deterministic and would fail identically on every attempt. The counts are read
    from the module globals on each call, so setting ``PROBE_RETRIES = 0`` restores
    the plain single-attempt behaviour everywhere.
    """
    import time

    attempt = 0
    while True:
        try:
            return reader.read()
        except FileNotFoundError:
            # A missing path is a typo, not a server hiccup: it will be missing on
            # every retry, so fail fast rather than sleeping through backoff for it.
            raise
        except Exception as exc:
            if attempt >= PROBE_RETRIES:
                raise
            wait = PROBE_RETRY_BACKOFF * 2**attempt
            warnings.warn(
                f"read of {name!r} failed ({exc}); retrying in {wait:g}s "
                f"(attempt {attempt + 1} of {PROBE_RETRIES})",
                stacklevel=3,
            )
            time.sleep(wait)
            attempt += 1


def _attach(cat, name, reader, *, probe, name_map, metadata):
    """Probe a reader, attach metadata, and put it in ``cat`` under ``name``.

    The step every source shares once its reader exists — whether that reader was
    built from a URL, named explicitly, or came from an already-built catalog.

    Two different failures hide inside "probing failed", and they deserve opposite
    treatment:

    * the source cannot be **read** — a typo'd path, a dead URL. The entry is
      unusable, so this propagates and the caller decides (``skip_errors``).
      Swallowing it would bank a dead entry whose failure surfaces much later, far
      from the typo that caused it. A *transient* read failure gets a few more
      chances first (see :func:`_read_with_retries`), since on a flaky remote server
      the read that raised here often succeeds moments later.
    * the source reads but cannot be **probed** — odd axes, no recognizable
      coordinates. "Could not derive metadata" is not "this entry is invalid": it
      still reads fine, it is only less searchable. That warns and keeps the entry.
    """
    if probe:
        # unreadable => unusable; let the caller decide, but retry a transient failure first
        data = _read_with_retries(reader, name)
        try:
            reader.metadata.update(_probe(data, name_map))
        except Exception as exc:
            warnings.warn(
                f"read {name!r} but could not derive metadata from it ({exc}); "
                "adding it anyway, though osk.find() will not see its extents.",
                stacklevel=3,
            )
    reader.metadata.update(metadata)
    # A ROMS entry with its grid still a separate file (not merged into the store by
    # make_kerchunk's grid= at build time) never sees lon_rho/lat_rho in the probed
    # `data` above — self_contained_grid was already False by then, so :func:`_probe`
    # could derive neither the geospatial extent nor an outline. Now that the caller's
    # `grid=` path has landed in metadata, the grid file itself can supply the shape
    # the model output alone could not — the same file
    # :func:`ocean_skill.roms.standardize` opens at read time, just read here for its
    # lon_rho/lat_rho rather than its data.
    if (
        reader.metadata.get("model") == "roms"
        and not reader.metadata.get("self_contained_grid")
        and reader.metadata.get("grid")
        and "domain_outline" not in reader.metadata
    ):
        try:
            from ocean_skill.roms import _open_grid

            grid = _open_grid(reader.metadata)
            outline = _domain_outline(grid["lon_rho"], grid["lat_rho"])
        except Exception as exc:
            warnings.warn(
                f"{name!r} declares a separate ROMS grid file but it could not be "
                f"read for a domain outline ({exc}); comparisons will fall back to "
                "the bounding box.",
                stacklevel=3,
            )
        else:
            if outline is not None:
                reader.metadata["domain_outline"] = outline
    cat[name] = reader
    cat.aliases[name] = name  # otherwise list(cat) is empty
    return reader


def add_source(
    cat,
    name: str,
    url: str | Path | None = None,
    *,
    reader: Any = None,
    name_map: dict[str, str] | None = ROMS_STANDARD_NAMES,
    probe: bool = True,
    storage_options: dict[str, Any] | None = None,
    reader_kwargs: dict[str, Any] | None = None,
    **metadata: Any,
):
    """Add a source to ``cat``, deriving metadata by querying the data.

    Parameters
    ----------
    cat, name
        Catalog and the entry name (``osk.read(name)``).
    url
        A kerchunk reference (``.parquet``/``.json``), an OPeNDAP URL, or a NetCDF
        path — anything :func:`_reader_for` can pick a reader for. Omit it when
        passing ``reader``.
    reader
        An intake v2 reader class, or its ``"module:Class"`` import path, for a
        source no URL convention describes: a remote tarball
        (``"ocean_skill.readers:PoochTarNetCDF"``) or an ERDDAP table
        (``"intake_erddap.erddap:TableDAPReader"``). ``reader_kwargs`` is passed to
        it verbatim, so the reader's own arguments — ``member_glob``,
        ``dataset_id``, ``server`` — go there. Every source type then reaches a
        catalog through this one function, rather than each needing its own.
    name_map
        Fallback variable→standard_name map for data lacking ``standard_name`` attrs;
        defaults to :data:`ROMS_STANDARD_NAMES`. Pass ``None`` to rely on attrs only.
    probe
        Open the source once to derive extents/axes/variables/featureType. Opening is
        cheap locally; for remote sources it costs one round-trip per entry at build
        time (never at read time). Set ``False`` to skip it — worth doing when the
        source is large and a cache would download the whole file just to look. A
        probe read that fails transiently is re-attempted first (see
        :data:`PROBE_RETRIES`), so a flaky remote server does not cost the entry.
    storage_options
        fsspec options, e.g. ``{"simplecache": {"cache_storage": "./cache"}}`` paired
        with a ``simplecache::https://...`` URL.
    **metadata
        Extra metadata; caller values override derived ones.
    """
    if reader is None:
        if url is None:
            raise TypeError("add_source needs either a url or a reader")
        reader = _reader_for(
            str(url), storage_options=storage_options, **(reader_kwargs or {})
        )
    else:
        reader = _build_reader(reader, url, storage_options, reader_kwargs or {})
    return _attach(cat, name, reader, probe=probe, name_map=name_map, metadata=metadata)


def add_sources(
    cat,
    sources: dict[str, Any] | list[tuple[str, Any]],
    *,
    skip_errors: bool = False,
    **shared: Any,
) -> dict[str, Any]:
    """Add many sources to ``cat`` with one set of shared options.

    ``sources`` maps entry name -> URL, or entry name -> a dict of per-source
    keyword arguments containing ``url``. Anything in ``**shared`` applies to every
    entry, with a per-source dict overriding it — so the common case (one
    ``storage_options`` for a whole catalog) is stated once::

        add_sources(
            cat,
            {
                "MODIS Aqua January Monthly Climatology": jan_url,
                "MODIS Aqua February Monthly Climatology": feb_url,
                # a dict adds per-source keys on top of the shared ones
                "MODIS Aqua Daily Chlorophyll 2012-01-01": {
                    "url": day_url,
                    "time_coverage_start": "2012-01-01",
                },
            },
            storage_options={"simplecache": {"same_names": True}},
        )

    Keep ``probe=True`` (the default) unless a source is genuinely too large to open
    once at build time: probing is what fills in the extents, variables and
    featureType that make an entry findable by :func:`ocean_skill.find`, and an
    unprobed entry is nearly opaque to search.

    ``skip_errors=True`` warns and carries on past an entry that fails to open,
    which matters when the list is long and remote: one flaky OPeNDAP URL should
    not discard a build that has already opened twenty others. It is off by default
    so a silently short catalog is never the quiet outcome.

    A *transient* probe read is re-attempted before it counts as a failure (see
    :data:`PROBE_RETRIES`) — the complement to ``skip_errors``, which decides what to
    do once every retry is spent. On a flaky server the two go together: retry the
    reads that will recover, skip the few that genuinely will not.

    Returns ``{name: reader}`` for the entries actually added.
    """
    # An already-built catalog carries readers, so there is nothing to construct --
    # only the shared probe/attach step. This is what makes
    # `build_catalog(ERDDAPCatalogReader(...).read(), out)` work.
    from_catalog = _is_catalog(sources)
    if from_catalog:
        names = list(sources) or list(getattr(sources, "entries", {}))
        items = [(name, sources[name]) for name in names]
    elif isinstance(sources, dict):
        items = list(sources.items())
    else:
        items = list(sources)

    added: dict[str, Any] = {}
    for name, spec in items:
        opts = {**shared, **spec} if isinstance(spec, dict) else {**shared}
        url = (
            None
            if from_catalog
            else (spec.get("url") if isinstance(spec, dict) else spec)
        )
        opts.pop("url", None)
        try:
            if from_catalog:
                added[name] = _attach(
                    cat,
                    name,
                    spec,
                    probe=opts.get("probe", True),
                    name_map=opts.get("name_map"),
                    metadata={
                        k: v
                        for k, v in opts.items()
                        if k
                        not in ("probe", "name_map", "storage_options", "reader_kwargs")
                    },
                )
            else:
                added[name] = add_source(cat, name, url, **opts)
        except Exception as exc:
            if not skip_errors:
                raise RuntimeError(f"failed adding {name!r} ({url}): {exc}") from exc
            warnings.warn(f"skipping {name!r} ({url}): {exc}", stacklevel=2)
    return added


def build_catalog(
    sources: dict[str, Any] | list[tuple[str, Any]],
    out: str | Path,
    *,
    title: str | None = None,
    skip_errors: bool = False,
    catalog_metadata: dict[str, Any] | None = None,
    **shared: Any,
) -> Path:
    """Build and save a catalog from ``{name: url}`` plus shared options.

    Returns the catalog path.

    The whole build in one call::

        build_catalog(
            {
                "MODIS Aqua January Monthly Climatology": jan_url,
                "MODIS Aqua February Monthly Climatology": feb_url,
            },
            "catalogs/modis_aqua.yaml",
            title="MODIS Aqua",
            storage_options={"simplecache": {"same_names": True}},
        )

    replacing :func:`new_catalog`, a run of :func:`add_source` calls each repeating
    the same options, and :func:`save`. See :func:`add_sources` for per-source
    overrides and ``skip_errors``; ``catalog_metadata`` adds catalog-level keys
    beyond ``title``.

    Probing already re-attempts a transient read (see :data:`PROBE_RETRIES`), so a
    plain ``build_catalog(discovered, out, probe=True)`` rides out a flaky server's
    hiccups with nothing extra at the call site. Pair it with ``skip_errors=True``
    when a live sweep may still contain a few entries that never open at all.
    """
    cat = new_catalog(title=title or Path(out).stem, **(catalog_metadata or {}))
    add_sources(cat, sources, skip_errors=skip_errors, **shared)
    return save(cat, out)


def _resolve_files(spec, root: str | Path | None) -> list[Path]:
    """Turn a glob, a path, or an iterable of paths into a sorted file list."""
    if isinstance(spec, str | Path):
        pattern = Path(spec).expanduser()
        if pattern.is_absolute():
            return sorted(pattern.parent.glob(pattern.name))
        base = Path(root).expanduser() if root is not None else Path()
        return sorted(base.glob(str(spec)))
    return sorted(Path(f) for f in spec)


def build_kerchunk(
    streams: dict[str, Any],
    *,
    root: str | Path | None = None,
    grid: str | Path | None = None,
    out_dir: str | Path = "refs",
    **kerchunk_kwargs: Any,
) -> dict[str, Path]:
    """Build one kerchunk reference per named stream. Returns ``{name: reference}``.

    The model-representation half of catalog building; :func:`build_catalog` is the
    other half, and this returns exactly the ``{name: url}`` mapping it takes::

        refs = build_kerchunk(
            {"GOM_bgc": "output_bgc.*.nc", "GOM_his": "output_his.*.nc"},
            root=run_dir, grid=GRID,
        )
        build_catalog(refs, "catalogs/gom.yaml", title="GOM offline run")

    Kept separate rather than fused into one call because the two halves have very
    different costs: kerchunking a hundred files takes minutes, writing the YAML
    takes milliseconds. Split, you can fix a title or add metadata without
    rebuilding a single reference — and ``refs | {"woa": opendap_url}`` mixes
    virtual stores and remote URLs in one catalog with no special path.

    Parameters
    ----------
    streams
        Entry name -> the files for that stream: a glob (relative to ``root``, or
        absolute), a single path, or an explicit list. One entry per stream is
        required, not stylistic: :func:`make_kerchunk` combines files sharing a
        variable set, and mixing streams fails.
    root
        Directory that relative globs resolve against.
    grid
        A separate static-coordinate file merged into every reference.
    out_dir
        Where the references are written. They are *inputs* to the catalog, not
        regenerable scratch — it points at them by path, so they must outlive it
        (which is why they do not live in :mod:`ocean_skill.cache`).
    **kerchunk_kwargs
        Forwarded to :func:`make_kerchunk` (``concat_dim``, ``loadable_variables``,
        ``keep``, ``fmt``, ``tolerant_attrs``, ``target_chunk_mb``, ``subchunk``).
        Model differences belong here, as arguments — the defaults are detected per
        file, so nothing needs to know a model by name. Applied to every stream in
        this call, so a restart stream needing ``keep="latest-per-file"`` goes in
        its own call, merged with ``|`` — see the module docstring. Likewise a
        stream whose variables need a different ``subchunk=`` (different vertical
        axis length, say) than the rest.
    """
    out_dir = Path(out_dir).expanduser()
    resolved = {name: _resolve_files(spec, root) for name, spec in streams.items()}
    empty = [name for name, files in resolved.items() if not files]
    if empty:
        where = f" under {root}" if root is not None else ""
        raise FileNotFoundError(
            f"no files matched for stream(s) {empty}{where}: "
            f"{ {k: str(streams[k]) for k in empty} }"
        )
    return {
        name: make_kerchunk(
            files, out_dir / f"{name}.parquet", grid=grid, **kerchunk_kwargs
        )
        for name, files in resolved.items()
    }


def add_catalog(
    cat,
    external,
    *,
    probe: bool = True,
    name_map: dict[str, str] | None = None,
    **metadata: Any,
) -> list[str]:
    """Merge every entry of an already-built external intake catalog into ``cat``.

    Now a thin call onto :func:`add_sources`, which accepts a catalog as one of its
    source forms — see :func:`build_catalog` for the one-call version. Kept because
    "merge this catalog into that one" reads better than the general function at the
    call site, and because it returns just the names.

    ``external`` was built by whatever tool knows how to talk to that source
    (``intake_erddap.ERDDAPCatalogReader`` is the motivating case), and nothing here
    knows anything about that protocol: *building* is inherently source-specific,
    *enriching* never is.

    A transient probe read is already re-attempted before ``skip_errors`` retires the
    entry (see :data:`PROBE_RETRIES`) — the pairing that makes a flaky-server sweep
    like CalOOS's land most of its datasets: retry the reads that recover, skip only
    those that truly do not.
    """
    # skip_errors=True preserves this function's long-standing behaviour: a live
    # server sweep always has a few datasets that will not open, and losing the
    # other forty because of them would be useless.
    return list(
        add_sources(
            cat,
            external,
            probe=probe,
            name_map=name_map,
            skip_errors=True,
            **metadata,
        )
    )


def add_erddap_source(
    cat,
    name: str,
    server: str,
    dataset_id: str,
    *,
    variables: list[str] | None = None,
    mask_failed_qartod: bool = True,
    probe: bool = True,
    reader_kwargs: dict[str, Any] | None = None,
    **metadata: Any,
):
    """Add one ERDDAP dataset (a *known* ``dataset_id``) to ``cat``.

    Built via ``intake_erddap.TableDAPReader`` directly — the class that package
    itself provides for exactly this case. ``ERDDAPCatalogReader`` (see
    :func:`add_erddap_catalog`) has no equivalent "give me this one dataset_id" mode
    — it is a search interface (``search_for``/``bbox``/``standard_names``/...), so
    it is the right tool for discovery but an awkward one when you already know
    which dataset you want.

    Parameters
    ----------
    server
        ERDDAP base URL, e.g.
        ``"https://erddap.dataexplorer.oceanobservatories.org/erddap"``.
    dataset_id
        The ERDDAP dataset ID — find one via that server's
        ``/search/index.csv?searchFor=...``.
    mask_failed_qartod
        Apply OOI's own QARTOD aggregate flags on read (the default): failed
        observations come back as NaN instead of needing separate QC downstream.
    """
    from intake_erddap.erddap import TableDAPReader

    reader = TableDAPReader(
        server,
        dataset_id,
        variables=list(variables) if variables else None,
        mask_failed_qartod=mask_failed_qartod,
        **(reader_kwargs or {}),
    )
    md: dict[str, Any] = {"server": server, "dataset_id": dataset_id}
    if probe:
        md.update(_probe(reader.read(), None))
    md.update(metadata)
    reader.metadata.update(md)
    cat[name] = reader
    cat.aliases[name] = name
    return reader


#: The two ARCO chunking layouts Copernicus Marine publishes for each dataset, and how
#: each is surfaced in a catalog: the ``copernicusmarine`` ``service`` value that opens
#: it, the nickname ``suffix`` a user selects by, and the human-facing ``chunking``
#: label recorded in metadata. timeChunked (``arco-time-series``) chunks the full time
#: axis with a small spatial footprint -- fast for a time series at a point/small area;
#: geoChunked (``arco-geo-series``) chunks broad space with few time steps -- fast for
#: maps. See :class:`ocean_skill.readers.CopernicusMarineReader`.
COPERNICUS_SERVICES: dict[str, dict[str, str]] = {
    "arco-time-series": {"suffix": "timeseries", "chunking": "time-series"},
    "arco-geo-series": {"suffix": "geo", "chunking": "geo"},
}


def add_copernicus_source(
    cat,
    name: str,
    dataset_id: str,
    *,
    services: tuple[str, ...] = ("arco-time-series", "arco-geo-series"),
    dataset_version: str | None = None,
    probe: bool = True,
    reader_kwargs: dict[str, Any] | None = None,
    **metadata: Any,
):
    """Add a Copernicus Marine dataset to ``cat`` — one entry per ARCO chunking layout.

    Copernicus Marine publishes each dataset in two chunkings, and which one you want
    depends on the access pattern, so this adds a *separate, separately-named* entry for
    each requested ``service`` rather than picking one for you. The nickname gains a
    suffix the user selects by — ``"<name>_timeseries"`` (timeChunked; fast time series
    at a point/small area) and ``"<name>_geo"`` (geoChunked; fast spatial maps) — and
    each entry records its layout under the ``chunking`` metadata key (plus ``service``
    and ``dataset_id``) so the choice is visible in ``osk.find()`` and the written YAML.

    Each entry is a :class:`ocean_skill.readers.CopernicusMarineReader`, i.e. every read
    is delegated to ``copernicusmarine.open_dataset`` (authenticated); the anonymous
    store URL cannot be read for data (see that class). Only the stable ``dataset_id``
    is stored — the toolbox resolves the current version at read time — so pass
    ``dataset_version`` only to pin one.

    Parameters
    ----------
    cat, name
        Catalog and the *base* entry name; the service suffix is appended per entry.
    dataset_id
        The Copernicus Marine dataset ID, e.g.
        ``"cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"``.
    services
        Which ARCO layouts to add, from :data:`COPERNICUS_SERVICES`. Defaults to both.
        Pass a single-element tuple to add just one.
    probe
        Open each entry once at build time to derive variables/extents (the default).
        This needs ``copernicusmarine`` installed and a valid ``copernicusmarine
        login``, and costs one authenticated open per entry. Set ``False`` to record the
        entry without opening it (no metadata beyond what you pass in).
    reader_kwargs
        Extra keyword arguments forwarded to ``copernicusmarine.open_dataset`` (e.g.
        ``variables=``); ``dataset_id``/``service``/``dataset_version`` are supplied
        here.
    **metadata
        Extra metadata applied to every entry; caller values override derived ones.

    Returns
    -------
    dict[str, Any]
        The readers added, keyed by their suffixed entry name.
    """
    added: dict[str, Any] = {}
    for service in services:
        try:
            spec = COPERNICUS_SERVICES[service]
        except KeyError:
            raise ValueError(
                f"unknown Copernicus service {service!r}; expected one of "
                f"{sorted(COPERNICUS_SERVICES)}"
            ) from None
        entry_name = f"{name}_{spec['suffix']}"
        rk: dict[str, Any] = {
            "dataset_id": dataset_id,
            "service": service,
            **(reader_kwargs or {}),
        }
        if dataset_version is not None:
            rk["dataset_version"] = dataset_version
        added[entry_name] = add_source(
            cat,
            entry_name,
            reader="ocean_skill.readers:CopernicusMarineReader",
            reader_kwargs=rk,
            probe=probe,
            dataset_id=dataset_id,
            service=service,
            chunking=spec["chunking"],
            **metadata,
        )
    return added


# def add_erddap_catalog(
#     cat,
#     server: str,
#     *,
#     mask_failed_qartod: bool = True,
#     probe: bool = True,
#     **search_kwargs: Any,
# ) -> list[str]:
#     """Bulk-discover ERDDAP datasets into ``cat``.

#     A thin call into ``intake_erddap.ERDDAPCatalogReader`` (which builds the
#     catalog) followed by :func:`add_catalog` (which merges and enriches it); see
#     that function's docstring for what "enrich" means here.

#     **No ``search_kwargs`` at all** discovers every dataset the server has
#     (verified: a plain ``add_erddap_catalog(cat, SERVER)`` against OOI's ERDDAP
#     found all ~766 of its datasets, not just one array) — so this is equally "one
#     OOI catalog" or "one Papa catalog", depending only on whether you narrow the
#     search.

#     Parameters
#     ----------
#     **search_kwargs
#         Forwarded to ``ERDDAPCatalogReader`` — most usefully ``search_for=["papa"]``,
#         ``standard_names=[...]``, ``bbox=(...)``, ``start_time=``/``end_time=``. See
#         that class's docstring for the full set. Omit entirely for the whole server.

#     Returns
#     -------
#     list[str]
#         The dataset IDs added.
#     """
#     from intake_erddap.erddap_cat import ERDDAPCatalogReader

#     discovered = ERDDAPCatalogReader(
#         server, mask_failed_qartod=mask_failed_qartod, **search_kwargs
#     ).read()
#     return add_catalog(cat, discovered, probe=probe)
