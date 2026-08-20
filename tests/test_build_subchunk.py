"""``make_kerchunk``'s manifest subchunking: ``subchunk=`` and ``target_chunk_mb=``.

Splitting a stored chunk works only because its bytes are one contiguous run in C
order: everything before the split axis must already be a single chunk, and the
variable must be uncompressed (a compressed chunk is the unit its codec decompresses,
so there is no byte range within it that means anything on its own). These tests build
tiny netCDF fixtures with the two storage shapes that matter -- one record per file
(the ROMS norm, whose leading dim is already 1) and several records written
contiguously in one file (whose leading dim is not) -- and check the manifest that
comes out, not just that the build does not crash.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill.build import make_kerchunk

READ_KW = {"engine": "kerchunk", "chunks": {}, "decode_times": False}


def _roms_like_3d(path, fmt, *, nt=1, ns=4, ny=3, nx=5, t0=0.0, encoding=None):
    """A tracer on ``(ocean_time, s_rho, eta_rho, xi_rho)``.

    ``nt=1`` (the default) matches one-file-per-record ROMS output: whether the file
    is stored HDF5-chunked or contiguous, a single record already collapses
    ``ocean_time`` to a chunk length of 1 -- the shape most of these tests want.
    ``nt=2`` with no ``encoding`` gives the other real shape instead: several
    records written contiguously in one file, whose ``ocean_time`` chunk length is
    *not* yet 1 -- see :func:`test_subchunking_a_contiguous_file_needs_the_leading_dim`.
    """
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "temp": (
                ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
                rng.normal(15.0, 2.0, (nt, ns, ny, nx)),
            )
        },
        coords={
            "ocean_time": (
                "ocean_time",
                [t0 + 43200.0 * i for i in range(nt)],
                {"units": "second"},
            )
        },
    )
    ds.to_netcdf(path, format=fmt, encoding=encoding)
    return path


def test_subchunk_splits_the_vertical_and_round_trips(tmp_path):
    files = [
        _roms_like_3d(tmp_path / f"o.{i}.nc", "NETCDF4", t0=i * 86400.0)
        for i in range(2)
    ]

    out = make_kerchunk(
        files, tmp_path / "r.json", subchunk={"s_rho": 2}, target_chunk_mb=None
    )
    ds = xr.open_dataset(str(out), **READ_KW)
    assert ds.chunksizes["s_rho"] == (2, 2)
    assert ds.chunksizes["eta_rho"] == (3,), "the horizontal must be untouched"

    baseline = make_kerchunk(files, tmp_path / "base.json", target_chunk_mb=None)
    xr.testing.assert_allclose(
        ds["temp"].compute(), xr.open_dataset(str(baseline), **READ_KW)["temp"]
    )


def test_subchunk_needs_the_leading_dim_for_a_contiguous_file(tmp_path):
    """Several records in one file, written contiguously: ocean_time isn't 1 yet."""
    files = [_roms_like_3d(tmp_path / "o.nc", "NETCDF4", nt=2)]

    with pytest.raises(ValueError, match="ocean_time"):
        make_kerchunk(
            files, tmp_path / "bad.json", subchunk={"s_rho": 2}, target_chunk_mb=None
        )

    out = make_kerchunk(
        files,
        tmp_path / "ok.json",
        subchunk={"ocean_time": 1, "s_rho": 2},
        target_chunk_mb=None,
    )
    ds = xr.open_dataset(str(out), **READ_KW)
    assert ds.chunksizes["ocean_time"] == (1, 1)
    assert ds.chunksizes["s_rho"] == (2, 2)

    baseline = make_kerchunk(files, tmp_path / "base.json", target_chunk_mb=None)
    xr.testing.assert_allclose(
        ds["temp"].compute(), xr.open_dataset(str(baseline), **READ_KW)["temp"]
    )


@pytest.mark.parametrize("fmt", ["NETCDF4", "NETCDF3_CLASSIC"])
def test_subchunk_needs_the_leading_dim_regardless_of_container(tmp_path, fmt):
    """The same rule for netCDF3: a record dim not marked unlimited isn't 1 either."""
    files = [_roms_like_3d(tmp_path / f"o.{fmt}.nc", fmt, nt=2)]

    with pytest.raises(ValueError, match="ocean_time"):
        make_kerchunk(
            files, tmp_path / "bad.json", subchunk={"s_rho": 2}, target_chunk_mb=None
        )


def test_subchunk_rejects_a_non_divisor(tmp_path):
    files = [_roms_like_3d(tmp_path / "o.nc", "NETCDF4")]

    with pytest.raises(ValueError, match="divide"):
        make_kerchunk(
            files, tmp_path / "r.json", subchunk={"s_rho": 3}, target_chunk_mb=None
        )


def test_subchunk_skips_a_compressed_variable_with_a_warning(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "o.nc"
    xr.Dataset(
        {
            "temp": (
                ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
                rng.normal(15.0, 2.0, (1, 4, 3, 5)),
            ),
            "salt": (
                ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
                rng.normal(35.0, 1.0, (1, 4, 3, 5)),
            ),
        },
        coords={"ocean_time": ("ocean_time", [0.0], {"units": "second"})},
    ).to_netcdf(path, format="NETCDF4", encoding={"salt": {"zlib": True}})

    with pytest.warns(UserWarning, match="compressed"):
        out = make_kerchunk(
            [path], tmp_path / "r.json", subchunk={"s_rho": 2}, target_chunk_mb=None
        )

    ds = xr.open_dataset(str(out), **READ_KW)
    assert ds["temp"].chunksizes["s_rho"] == (2, 2)
    assert ds["salt"].chunksizes["s_rho"] == (4,), "compressed stays whole"

    orig = xr.open_dataset(path, decode_times=False)
    xr.testing.assert_allclose(ds["temp"].compute(), orig["temp"])
    xr.testing.assert_allclose(ds["salt"].compute(), orig["salt"])


def test_automatic_mode_splits_only_the_leading_dims(tmp_path):
    """A small ``target_chunk_mb`` reaches the vertical but never the horizontal."""
    files = [
        _roms_like_3d(tmp_path / f"o.{i}.nc", "NETCDF4", ns=8, t0=i * 86400.0)
        for i in range(2)
    ]
    # 1 * 8 * 3 * 5 * 8 bytes = 960 B per chunk; ask for a target far below that so
    # every divisor of s_rho gets tried, without needing a multi-megabyte fixture.
    target_mb = 200 / 1024**2  # 200 bytes

    out = make_kerchunk(files, tmp_path / "r.json", target_chunk_mb=target_mb)
    ds = xr.open_dataset(str(out), **READ_KW)
    assert len(ds.chunksizes["s_rho"]) > 1, "the vertical must have been split"
    assert ds.chunksizes["eta_rho"] == (3,)
    assert ds.chunksizes["xi_rho"] == (5,)

    baseline = make_kerchunk(files, tmp_path / "base.json", target_chunk_mb=None)
    xr.testing.assert_allclose(
        ds["temp"].compute(), xr.open_dataset(str(baseline), **READ_KW)["temp"]
    )


def test_target_chunk_mb_none_matches_the_unsplit_layout(tmp_path):
    """The default before this change: nothing split, whatever the store carries."""
    files = [_roms_like_3d(tmp_path / "o.nc", "NETCDF4")]

    out = make_kerchunk(files, tmp_path / "r.json", target_chunk_mb=None)
    ds = xr.open_dataset(str(out), **READ_KW)
    assert ds.chunksizes["s_rho"] == (4,)


def test_a_small_default_target_leaves_a_tiny_store_untouched(tmp_path):
    """The new default (``target_chunk_mb=128``) is a no-op far below that target."""
    files = [_roms_like_3d(tmp_path / "o.nc", "NETCDF4")]

    out = make_kerchunk(files, tmp_path / "r.json")  # target_chunk_mb defaults to 128
    ds = xr.open_dataset(str(out), **READ_KW)
    assert ds.chunksizes["s_rho"] == (4,)


def test_explicit_subchunk_overrides_automatic_for_the_dim_named(tmp_path):
    files = [
        _roms_like_3d(tmp_path / f"o.{i}.nc", "NETCDF4", ns=8, t0=i * 86400.0)
        for i in range(2)
    ]
    target_mb = 200 / 1024**2

    out = make_kerchunk(
        files, tmp_path / "r.json", target_chunk_mb=target_mb, subchunk={"s_rho": 4}
    )
    ds = xr.open_dataset(str(out), **READ_KW)
    assert ds.chunksizes["s_rho"] == (4, 4), "the explicit size must win, not auto's"


def test_subchunk_survives_the_parquet_writer_too(tmp_path):
    """``build_kerchunk``'s default format.

    Exercised separately from the JSON path every other test here uses, since the
    writer branches on ``out``'s extension.
    """
    files = [_roms_like_3d(tmp_path / "o.nc", "NETCDF4")]

    out = make_kerchunk(
        files, tmp_path / "r.parquet", subchunk={"s_rho": 2}, target_chunk_mb=None
    )
    ds = xr.open_dataset(str(out), **READ_KW)
    assert ds.chunksizes["s_rho"] == (2, 2)

    orig = xr.open_dataset(files[0], decode_times=False)
    xr.testing.assert_allclose(ds["temp"].compute(), orig["temp"])


def test_build_kerchunk_forwards_subchunking_kwargs(tmp_path):
    """``**kerchunk_kwargs`` needs no code change to carry these through."""
    from ocean_skill.build import build_kerchunk

    _roms_like_3d(tmp_path / "GOM_his.0.nc", "NETCDF4")

    refs = build_kerchunk(
        {"GOM_his": "GOM_his.*.nc"},
        root=tmp_path,
        out_dir=tmp_path,
        subchunk={"s_rho": 2},
        target_chunk_mb=None,
    )
    ds = xr.open_dataset(str(refs["GOM_his"]), **READ_KW)
    assert ds.chunksizes["s_rho"] == (2, 2)
