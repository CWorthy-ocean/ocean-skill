"""Custom intake v2 readers for ocean-skill sources that intake can't express directly.

A native intake v2 catalog references a reader by import path (e.g.
``ocean_skill.readers:PoochTarNetCDF``), so any access pattern intake lacks — like
a remote tarball of NetCDFs — becomes a small ``BaseReader`` subclass here. The reader
must live in an installed module so a catalog can be reloaded elsewhere.
"""

from __future__ import annotations

from typing import ClassVar

from intake.readers.readers import BaseReader

__all__ = ["CopernicusMarineReader", "PoochTarNetCDF"]


class PoochTarNetCDF(BaseReader):
    """Fetch + untar a remote tarball of NetCDFs (pooch-cached) into a merged Dataset.

    Downloads ``url`` once via pooch (untarring with :class:`pooch.Untar`), caches under
    ``cache_dir`` (default :func:`ocean_skill.cache.obs_dir`, alongside every other
    downloaded source), keeps the members matching ``member_glob``, and opens + merges
    them with ``xarray.open_mfdataset``. Used for GLODAP (a NOAA .tar.gz of per-variable
    NetCDFs) but generic to any tarball-of-NetCDFs source.

    Pass ``local_dir`` instead of ``url`` when the tarball's contents are already
    sitting on disk (a cluster with the data pre-staged on a shared filesystem, say)
    -- the download/untar step is skipped and ``member_glob`` is matched straight
    against that directory. Everything downstream (the ``var_from_filename`` merge
    below) is identical either way, so a catalog built with ``local_dir`` and one
    built with ``url`` over the same files read out the same Dataset -- only the byte
    source differs.
    """

    output_instance = "xarray:Dataset"
    imports: ClassVar[set[str]] = {"pooch", "xarray"}

    def _read(
        self,
        url=None,
        known_hash=None,
        member_glob="*.nc",
        cache_dir=None,
        combine="by_coords",
        var_from_filename=False,
        keep_vars=(),
        local_dir=None,
        **kwargs,
    ):
        import fnmatch
        import glob
        import os

        import xarray as xr

        if (url is None) == (local_dir is None):
            raise ValueError(
                "Pass exactly one of url (a remote tarball) or local_dir (a directory "
                "of already-extracted files)."
            )

        if local_dir is not None:
            base = os.path.expanduser(local_dir)
            paths = sorted(glob.glob(os.path.join(base, member_glob or "*")))
            if not paths:
                raise FileNotFoundError(f"No files matched {member_glob!r} in {base}")
        else:
            import pooch

            # Default to the same directory fsspec downloads into, resolved per call
            # rather than written into the catalog: a catalog that named an absolute
            # path pinned every install to one machine's home directory and ignored
            # both ``$OCEAN_SKILL_DIR`` and ``cache.enable()``.
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
                    p
                    for p in paths
                    if fnmatch.fnmatch(os.path.basename(p), member_glob)
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


class CopernicusMarineReader(BaseReader):
    """Open a Copernicus Marine ARCO store by ``dataset_id`` via ``copernicusmarine``.

    Copernicus Marine's ARCO Zarr stores on CloudFerro are not anonymously readable at
    the *chunk* level -- only the Zarr metadata (``.zmetadata``/``.zarray``) is public.
    A plain anonymous ``xarray.open_dataset`` of the store URL therefore *opens* (it
    only reads metadata) but then 403s on the first data chunk: an empty land/ice tile
    comes back as ``403 AccessDenied`` rather than ``404`` (the anonymous caller lacks
    ``s3:ListBucket``), so zarr's missing-chunk-is-``fill_value`` path -- which triggers
    on a 404 -- never runs and the read raises. The toolbox holds the login
    (``copernicusmarine login`` -> ``~/.copernicusmarine``), mints the short-lived S3
    auth, and resolves the current dataset version, so a catalog stores only the stable
    ``dataset_id`` and delegates every read here (as roms-tools' GLORYS loader does).

    ``service`` selects which chunking layout to open -- CMEMS publishes each dataset in
    both:

    * ``"arco-time-series"`` (timeChunked): the full time axis per chunk with a small
      spatial footprint -- fast for a time series at a point/small area, slow for maps.
    * ``"arco-geo-series"`` (geoChunked): broad space per chunk with few time steps --
      fast for spatial maps at a few dates.

    ``copernicusmarine`` is an *optional* dependency imported here rather than declared
    in ``imports`` so that loading a catalog that merely *contains* Copernicus entries
    never requires it -- only actually reading one does, and then with a clear message.
    """

    output_instance = "xarray:Dataset"

    def _read(
        self,
        dataset_id,
        service="arco-time-series",
        dataset_version=None,
        check_login=True,
        **kwargs,
    ):
        from importlib.util import find_spec

        if find_spec("copernicusmarine") is None:
            raise RuntimeError(
                f"Reading the Copernicus Marine source {dataset_id!r} needs the "
                "'copernicusmarine' package. Install it (e.g. `pip install "
                "copernicusmarine`) and authenticate once with "
                "`copernicusmarine login`."
            )
        import copernicusmarine

        # Fail early and legibly on a missing/expired login rather than letting the
        # anonymous-style 403 resurface deeper in the stack (see the class docstring).
        if check_login and not copernicusmarine.login(check_credentials_valid=True):
            raise RuntimeError(
                f"Not authenticated with Copernicus Marine, so {dataset_id!r} cannot "
                "be read. Run `copernicusmarine login` (free CMEMS account) and retry."
            )
        opts = dict(kwargs)
        if dataset_version is not None:
            opts["dataset_version"] = dataset_version
        return copernicusmarine.open_dataset(
            dataset_id=dataset_id, service=service, **opts
        )
