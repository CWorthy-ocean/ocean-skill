"""A roms-tools tidal forcing file, catalogued as a tidal-amplitude reference.

The forcing file (TPXO9 interpolated onto the ROMS grid) drops ``lat_rho``/``lon_rho``,
so the entry's reader chain merges them back in from the run's grid file -- built with
code, the same way as the Holte & Talley recipe, no custom reader and no hand-written
YAML. The README's "Tidal forcing as a reference" section is this recipe.
"""

import numpy as np
import xarray as xr
from intake.readers import datatypes, readers

from ocean_skill import build, sources

RE = "sea_surface_height_tidal_harmonic_real_part"
IM = "sea_surface_height_tidal_harmonic_imaginary_part"
NY, NX = 4, 5


def _write_files(tmp_path):
    """Write a tiny forcing file (no lon/lat) and a grid file (the lon/lat)."""
    dims = ("eta_rho", "xi_rho")
    lon = -20 + 0.1 * np.arange(NX)[None, :] + 0.01 * np.arange(NY)[:, None]
    lat = 63 + 0.1 * np.arange(NY)[:, None] + 0.01 * np.arange(NX)[None, :]
    xr.Dataset(
        {"lon_rho": (dims, lon), "lat_rho": (dims, lat), "h": (dims, lon * 0 + 100)}
    ).to_netcdf(tmp_path / "grid.nc")
    rng = np.random.default_rng(0)
    tide, label = ("ntides", *dims), {"long_name": "constituent label"}
    xr.Dataset(
        {
            "ssh_Re": (tide, rng.normal(size=(2, NY, NX)), {"units": "m"}),
            "ssh_Im": (tide, rng.normal(size=(2, NY, NX)), {"units": "m"}),
        },
        coords={
            "ntides": ("ntides", np.array(["M2", "K1"]), label),
            "omega": ("ntides", [1.405e-4, 7.292e-5]),
        },
    ).to_netcdf(tmp_path / "frc.nc")
    return lon, lat


def _reader(path):
    return readers.XArrayDatasetReader(datatypes.HDF5(url=str(path)), chunks={})


def test_forcing_file_gets_standard_names_labels_and_grid_coordinates(
    isolated_catalogs, tmp_path
):
    lon, lat = _write_files(tmp_path)
    # the README recipe: merge the grid's lon/lat into the forcing file, promote them
    # to coordinates, and name them the way osk recognises
    chain = (
        _reader(tmp_path / "frc.nc")
        .merge(other=_reader(tmp_path / "grid.nc")[["lon_rho", "lat_rho"]])
        .set_coords(["lon_rho", "lat_rho"])
        .rename({"lon_rho": "lon", "lat_rho": "lat"})
    )
    build.build_catalog(
        {
            "iceland_tides_frc": {
                "reader": chain,
                "standard_names": {"ssh_Re": RE, "ssh_Im": IM},
            }
        },
        isolated_catalogs / "tides.yaml",
        title="Tidal forcing references",
        name_map=None,  # not ROMS output: skip the ROMS name fallback
    )

    ds = sources.read("iceland_tides_frc")
    assert RE in ds and IM in ds
    assert list(ds["ntides"].values) == ["M2", "K1"]
    assert ds[RE].dims == ("ntides", "eta_rho", "xi_rho")
    np.testing.assert_allclose(ds["lon"].values, lon)
    np.testing.assert_allclose(ds["lat"].values, lat)
    assert {"lon", "lat"} <= set(ds.coords)

    ref = sources.resolve("iceland_tides_frc")
    assert ref.metadata["featureType"] == "grid"
    assert set(ref.metadata["variables"]) == {RE, IM}


def test_raw_tpxo_atlas_gets_constituent_labels_and_2d_lon_lat(
    isolated_catalogs, tmp_path
):
    """The README's atlas sub-recipe: ``con`` labels replace ``nc``, 2-D lon/lat."""
    dims = ("nx", "ny")
    lon = np.tile(np.linspace(216.8, 217.2, 3)[:, None], (1, 2))
    lat = np.tile(np.linspace(10.0, 10.5, 2)[None, :], (3, 1))
    xr.Dataset(
        {
            "hRe": (("nc", *dims), np.ones((2, 3, 2))),
            "hIm": (("nc", *dims), np.zeros((2, 3, 2))),
            "con": ("nc", np.array([b"m2  ", b"k1  "], dtype="S4")),
            "lon_z": (dims, lon),
            "lat_z": (dims, lat),
        }
    ).to_netcdf(tmp_path / "h_tpxo.nc")
    tpxo = (
        _reader(tmp_path / "h_tpxo.nc")
        .set_coords(["con", "lon_z", "lat_z"])
        .swap_dims({"nc": "con"})
        .rename({"lon_z": "lon", "lat_z": "lat"})
    )
    build.build_catalog(
        {"tpxo9_tides": {"reader": tpxo, "standard_names": {"hRe": RE, "hIm": IM}}},
        isolated_catalogs / "tpxo.yaml",
        title="TPXO",
        name_map=None,
    )
    ds = sources.read("tpxo9_tides")
    assert ds[RE].dims == ("con", "nx", "ny") and IM in ds
    assert [c.decode().strip() for c in ds["con"].values] == ["m2", "k1"]
    np.testing.assert_allclose(ds["lon"].values, lon)
    assert sources.resolve("tpxo9_tides").metadata["lon_convention"] == "0-360"
