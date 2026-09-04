"""Tests for ocean_skill.readers: the small custom intake v2 reader classes."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from ocean_skill.readers import PoochTarNetCDF


@pytest.fixture
def glodap_like_files(tmp_path):
    """Loose per-variable NetCDFs shaped like GLODAP's tarball members.

    Each file carries its own filename-named variable, a shared ``Depth`` coordinate,
    and a generic diagnostic (``Input_mean``) whose *value differs per file* -- the
    case that breaks a naive ``xr.merge`` and is why ``PoochTarNetCDF`` keeps only
    each file's own variable instead of merging everything.
    """
    depth = np.array([0.0, 10.0])
    bases = {"temperature": 10.0, "salinity": 35.0}
    for i, (var, base) in enumerate(bases.items()):
        path = tmp_path / f"GLODAPv2.2016b.{var}.nc"
        xr.Dataset(
            {
                var: ("depth", np.array([base, base + 1])),
                "Input_mean": ("depth", np.full(2, float(i))),  # conflicts by design
            },
            coords={"Depth": ("depth", depth)},
        ).to_netcdf(path)
    return tmp_path


# -- local_dir mode -------------------------------------------------------------


def test_local_dir_merges_per_variable_files(glodap_like_files):
    ds = PoochTarNetCDF()._read(
        local_dir=str(glodap_like_files),
        member_glob="*.nc",
        var_from_filename=True,
        keep_vars=["Depth"],
    )
    assert {"temperature", "salinity"} <= set(ds.data_vars)
    assert "Depth" in ds.coords
    assert float(ds["temperature"].isel(depth=0)) == 10.0
    assert float(ds["salinity"].isel(depth=0)) == 35.0


def test_local_and_tarball_paths_produce_identical_datasets(glodap_like_files):
    """Internet and cluster-local catalogs must read out the same Dataset.

    The only difference between the two should be where the bytes come from -- the
    merge that follows has to be byte-for-byte identical either way.
    """
    paths = sorted(str(p) for p in glodap_like_files.glob("*.nc"))
    kwargs = dict(member_glob="*.nc", var_from_filename=True, keep_vars=["Depth"])

    with mock.patch("pooch.retrieve", return_value=paths):
        via_url = PoochTarNetCDF()._read(
            url="https://example.invalid/g.tar.gz", **kwargs
        )

    via_local_dir = PoochTarNetCDF()._read(local_dir=str(glodap_like_files), **kwargs)

    xr.testing.assert_identical(via_url, via_local_dir)


def test_local_dir_never_touches_pooch(glodap_like_files):
    with mock.patch(
        "pooch.retrieve", side_effect=AssertionError("should not download")
    ):
        ds = PoochTarNetCDF()._read(
            local_dir=str(glodap_like_files),
            member_glob="*.nc",
            var_from_filename=True,
        )
    assert "temperature" in ds


def test_local_dir_no_matches_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PoochTarNetCDF()._read(local_dir=str(tmp_path), member_glob="*.nc")


def test_requires_exactly_one_source(glodap_like_files):
    with pytest.raises(ValueError):
        PoochTarNetCDF()._read()  # neither url nor local_dir
    with pytest.raises(ValueError):
        PoochTarNetCDF()._read(
            url="https://example.invalid/g.tar.gz",
            local_dir=str(glodap_like_files),
        )


def test_local_dir_plain_mfdataset(tmp_path):
    """``var_from_filename=False`` takes the plain ``open_mfdataset`` path unchanged."""
    xr.Dataset({"temperature": ("depth", [10.0])}, coords={"depth": [0.0]}).to_netcdf(
        tmp_path / "a.nc"
    )
    xr.Dataset({"oxygen": ("depth", [200.0])}, coords={"depth": [0.0]}).to_netcdf(
        tmp_path / "b.nc"
    )

    ds = PoochTarNetCDF()._read(local_dir=str(tmp_path), member_glob="*.nc")
    assert {"temperature", "oxygen"} == set(ds.data_vars)


# -- build_catalog integration ---------------------------------------------------


def test_build_catalog_from_local_dir_probes_and_reads(glodap_like_files, tmp_path):
    import intake

    from ocean_skill.build import build_catalog

    out = build_catalog(
        {
            "glodap": {
                "reader": "ocean_skill.readers:PoochTarNetCDF",
                "reader_kwargs": {
                    "local_dir": str(glodap_like_files),
                    "member_glob": "*.nc",
                    "var_from_filename": True,
                    "keep_vars": ["Depth"],
                },
            }
        },
        tmp_path / "glodap.catalog.yaml",
        title="GLODAP (local)",
    )

    cat = intake.from_yaml_file(str(out))
    ds = cat["glodap"].read()
    assert {"temperature", "salinity"} <= set(ds.data_vars)
    assert cat.entries["glodap"].metadata.get("featureType")  # probing ran
