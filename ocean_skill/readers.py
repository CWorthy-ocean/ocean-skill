"""Custom intake v2 readers for ocean-skill sources that intake can't express directly.

A native intake v2 catalog references a reader by import path (e.g.
``ocean_skill.readers:PoochTarNetCDF``), so any access pattern intake lacks — like
a remote tarball of NetCDFs — becomes a small ``BaseReader`` subclass here. The reader
must live in an installed module so a catalog can be reloaded elsewhere.
"""

from __future__ import annotations

from typing import ClassVar

from intake.readers.readers import BaseReader

__all__ = ["PoochTarNetCDF"]


class PoochTarNetCDF(BaseReader):
    """Fetch + untar a remote tarball of NetCDFs (pooch-cached) into a merged Dataset.

    Downloads ``url`` once via pooch (untarring with :class:`pooch.Untar`), caches under
    ``cache_dir`` (default :func:`ocean_skill.cache.obs_dir`, alongside every other
    downloaded source), keeps the members matching ``member_glob``, and opens + merges
    them with ``xarray.open_mfdataset``. Used for GLODAP (a NOAA .tar.gz of per-variable
    NetCDFs) but generic to any tarball-of-NetCDFs source.
    """

    output_instance = "xarray:Dataset"
    imports: ClassVar[set[str]] = {"pooch", "xarray"}

    def _read(
        self,
        url,
        known_hash=None,
        member_glob="*.nc",
        cache_dir=None,
        combine="by_coords",
        var_from_filename=False,
        keep_vars=(),
        **kwargs,
    ):
        import fnmatch
        import os

        import pooch
        import xarray as xr

        # Default to the same directory fsspec downloads into, resolved per call rather
        # than written into the catalog: a catalog that named an absolute path pinned
        # every install to one machine's home directory and ignored both
        # ``$OCEAN_SKILL_DIR`` and ``cache.enable()``.
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
        else:
            from ocean_skill import cache

            cache_dir = str(cache.obs_dir())
        paths = pooch.retrieve(
            url, known_hash=known_hash, processor=pooch.Untar(), path=cache_dir
        )
        if member_glob:
            paths = [
                p for p in paths if fnmatch.fnmatch(os.path.basename(p), member_glob)
            ]
        if not paths:
            raise FileNotFoundError(f"No members matched {member_glob!r} in {url}")
        paths = sorted(paths)

        if not var_from_filename:
            return xr.open_mfdataset(paths, combine=combine, **kwargs)

        # One-variable-per-file archives (GLODAP: GLODAPv2.2016b.TAlk.nc) repeat
        # generic diagnostics (Input_mean, SnR, ...) with *different values* in each
        # file, so a plain merge raises on conflicts. Keep only each file's own
        # variable, named by the second-to-last dot-separated token.
        pieces = []
        for p in paths:
            var = os.path.basename(p).split(".")[-2]
            ds = xr.open_dataset(p, **kwargs)
            wanted = [v for v in (var, *keep_vars) if v in ds.variables]
            if wanted:
                pieces.append(ds[wanted])
        if not pieces:
            raise ValueError(
                f"No filename-named variables found among {len(paths)} files"
            )
        return xr.merge(pieces, compat="override", join="override")
