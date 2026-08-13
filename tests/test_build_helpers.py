"""Tests for the one-call catalog builders.

These collapse the repeated "make a catalog, add sources one at a time repeating the
same options, save" sequence. What matters is that the collapse is faithful: shared
options really do reach every entry, a per-source override really does win, and a
long remote build can survive one bad URL without silently producing a short catalog
when you did not ask for that.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import intake
import numpy as np
import pytest
import xarray as xr

from ocean_skill.build import (
    _resolve_files,
    add_catalog,
    add_source,
    add_sources,
    build_catalog,
    build_kerchunk,
    detect_concat,
    new_catalog,
)

CHL = "mass_concentration_of_chlorophyll_a_in_sea_water"


@pytest.fixture
def netcdfs(tmp_path):
    """Two tiny CF-attributed NetCDFs standing in for remote products."""
    paths = {}
    for name in ("jan", "feb"):
        path = tmp_path / f"{name}.nc"
        xr.Dataset(
            {
                "chlor_a": (
                    ("lat", "lon"),
                    np.ones((4, 5)),
                    {"standard_name": CHL, "units": "mg m-3"},
                )
            },
            coords={"lat": np.linspace(10, 20, 4), "lon": np.linspace(-100, -90, 5)},
        ).to_netcdf(path)
        paths[name] = str(path)
    return paths


# -- build_catalog / add_sources ----------------------------------------------


def test_build_catalog_writes_every_entry(netcdfs, tmp_path):
    out = build_catalog(
        {"January": netcdfs["jan"], "February": netcdfs["feb"]},
        tmp_path / "modis.yaml",
        title="MODIS Aqua",
    )
    cat = intake.from_yaml_file(str(out))
    assert sorted(cat) == ["February", "January"]
    assert cat.metadata["title"] == "MODIS Aqua"


def test_shared_options_reach_every_entry(netcdfs, tmp_path):
    """The whole point: state `probe` (or storage_options) once, not per source."""
    out = build_catalog(
        {"January": netcdfs["jan"], "February": netcdfs["feb"]},
        tmp_path / "m.yaml",
        probe=True,
    )
    cat = intake.from_yaml_file(str(out))
    for name in cat:
        assert cat[name].metadata["variables"] == [CHL]


def test_per_source_options_override_the_shared_ones(netcdfs, tmp_path):
    out = build_catalog(
        {
            "January": netcdfs["jan"],
            "February": {"url": netcdfs["feb"], "probe": False},
        },
        tmp_path / "m.yaml",
        probe=True,
    )
    cat = intake.from_yaml_file(str(out))
    assert cat["January"].metadata["variables"] == [CHL]
    assert cat["February"].metadata.get("variables") is None


def test_extra_metadata_is_applied_to_entries(netcdfs, tmp_path):
    out = build_catalog(
        {"January": netcdfs["jan"]}, tmp_path / "m.yaml", institution="NASA OB.DAAC"
    )
    cat = intake.from_yaml_file(str(out))
    assert cat["January"].metadata["institution"] == "NASA OB.DAAC"


def test_a_bad_source_names_itself_in_the_error(netcdfs, tmp_path):
    """'failed adding X' beats a bare traceback when the list is twenty URLs long."""
    with pytest.raises(RuntimeError, match="February"):
        build_catalog(
            {"January": netcdfs["jan"], "February": str(tmp_path / "missing.nc")},
            tmp_path / "m.yaml",
        )


def test_skip_errors_keeps_the_good_entries(netcdfs, tmp_path):
    cat = new_catalog()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        added = add_sources(
            cat,
            {"January": netcdfs["jan"], "broken": str(tmp_path / "missing.nc")},
            skip_errors=True,
        )
    assert list(added) == ["January"]
    assert any("broken" in str(w.message) for w in caught)


def test_a_list_of_pairs_works_too(netcdfs, tmp_path):
    """Order matters for some catalogs, so a sequence has to be accepted."""
    out = build_catalog(
        [("January", netcdfs["jan"]), ("February", netcdfs["feb"])],
        tmp_path / "m.yaml",
    )
    assert len(list(intake.from_yaml_file(str(out)))) == 2


# -- model streams ------------------------------------------------------------


@pytest.fixture
def run_dir(tmp_path):
    """Build a run directory: two output streams plus a separate grid file."""
    run = tmp_path / "output_dt450"
    run.mkdir()
    for stream, n in {"bgc": 2, "his": 3}.items():
        for i in range(n):
            (run / f"output_{stream}.2012010{i + 1}000000.nc").touch()
    (tmp_path / "mygrid.nc").touch()
    return run


def test_globs_resolve_relative_to_root(run_dir):
    files = _resolve_files("output_bgc.*.nc", run_dir)
    assert len(files) == 2
    assert all(f.name.startswith("output_bgc.") for f in files)


def test_absolute_globs_need_no_root(run_dir):
    assert len(_resolve_files(str(run_dir / "output_his.*.nc"), None)) == 3


def test_an_explicit_file_list_is_accepted(run_dir):
    """Nothing should force a glob when the caller already has the paths."""
    chosen = sorted(run_dir.glob("output_bgc.*.nc"))[:1]
    assert _resolve_files(chosen, None) == chosen


def test_a_stream_matching_nothing_fails_loudly(run_dir):
    """Silently producing an empty catalog would look like a successful build."""
    with pytest.raises(FileNotFoundError, match="missing"):
        build_kerchunk({"missing": "output_nope.*.nc"}, root=run_dir)


# -- concat detection (replaces per-model presets) ----------------------------


def test_detects_time_axis_of_a_cf_compliant_model(tmp_path):
    """cf-xarray settles any CF-compliant model with no fallback involved."""
    path = tmp_path / "cf.nc"
    xr.Dataset(
        {"thetao": (("time", "lat", "lon"), np.ones((3, 2, 2)))},
        coords={
            "time": (
                "time",
                np.arange(3),
                {"units": "days since 2000-01-01", "axis": "T"},
            ),
            "lat": [1.0, 2.0],
            "lon": [1.0, 2.0],
        },
    ).to_netcdf(path)
    assert detect_concat(path) == ("time", ("time",))


def test_detects_a_time_variable_named_differently_from_its_dimension(tmp_path):
    """ROMS' shape: variable `ocean_time` on dimension `time`, and no coordinates.

    cf-xarray finds nothing here (no coords, non-CF `units="second"`), so this
    exercises the name fallback — and the dimension must come from the variable,
    not from the variable's own name.
    """
    path = tmp_path / "romsish.nc"
    xr.Dataset(
        {
            "ocean_time": (("time",), np.arange(3.0), {"units": "second"}),
            "temp": (("time", "eta_rho"), np.ones((3, 4))),
        }
    ).to_netcdf(path)
    assert detect_concat(path) == ("time", ("ocean_time",))


def test_undetectable_time_raises_rather_than_guessing(tmp_path):
    """Concatenating along the wrong axis silently scrambles record order."""
    path = tmp_path / "odd.nc"
    xr.Dataset({"x": (("record",), np.ones(3))}).to_netcdf(path)
    with pytest.raises(ValueError, match="cannot find a time variable"):
        detect_concat(path)


# -- custom readers -----------------------------------------------------------


def test_a_custom_reader_can_be_named_in_a_spec(tmp_path):
    """Not every source is a URL: a tarball or an ERDDAP table needs its own reader.

    Without this, GLODAP and OOI had to be hand-written or built through separate
    functions, so `build_catalog` covered only part of the catalog surface.
    """
    out = build_catalog(
        {
            "glodap": {
                "reader": "ocean_skill.readers:PoochTarNetCDF",
                "reader_kwargs": {
                    "url": "https://example.invalid/GLODAP.tar.gz",
                    "member_glob": "*.nc",
                    "var_from_filename": True,
                },
            }
        },
        tmp_path / "g.yaml",
        title="GLODAP",
        probe=False,  # never fetched: this asserts the entry, not the download
    )
    entry = intake.from_yaml_file(str(out))["glodap"]
    assert "PoochTarNetCDF" in str(entry.reader)
    assert entry.kwargs["member_glob"] == "*.nc"


def test_reader_and_url_sources_mix_in_one_catalog(netcdfs, tmp_path):
    """A catalog is a namespace, so it should not care how each entry is read."""
    out = build_catalog(
        {
            "plain": netcdfs["jan"],
            "custom": {
                "reader": "ocean_skill.readers:PoochTarNetCDF",
                "reader_kwargs": {"url": "https://example.invalid/x.tar.gz"},
                "probe": False,
            },
        },
        tmp_path / "mixed.yaml",
    )
    assert sorted(intake.from_yaml_file(str(out))) == ["custom", "plain"]


def test_a_source_with_neither_url_nor_reader_is_rejected(tmp_path):
    """add_sources wraps the failure so it names which entry was malformed."""
    cat = new_catalog()
    with pytest.raises(RuntimeError, match=r"'nothing'.*url or a reader"):
        add_sources(cat, {"nothing": {}})

    with pytest.raises(TypeError, match="url or a reader"):
        add_source(new_catalog(), "nothing")


# -- an already-built catalog as a source form --------------------------------


@pytest.fixture
def prebuilt(netcdfs, tmp_path):
    """Build a catalog elsewhere, standing in for a discovered ERDDAP catalog."""
    out = build_catalog(
        {"a": netcdfs["jan"], "b": netcdfs["feb"]}, tmp_path / "src.yaml", probe=False
    )
    return intake.from_yaml_file(str(out))


def test_build_catalog_accepts_an_already_built_catalog(prebuilt, tmp_path):
    """The point of unifying: a discovered catalog goes straight in, no adapter.

    discovered = ERDDAPCatalogReader(...).read()
    build_catalog(discovered, "catalogs/ooi_papa.yaml", title="OOI Papa")
    """
    out = build_catalog(prebuilt, tmp_path / "merged.yaml", title="Merged", probe=False)
    merged = intake.from_yaml_file(str(out))
    assert sorted(merged) == ["a", "b"]
    assert merged.metadata["title"] == "Merged"


def test_entries_from_a_catalog_are_probed_like_any_other(prebuilt, tmp_path):
    out = build_catalog(prebuilt, tmp_path / "m.yaml", probe=True)
    merged = intake.from_yaml_file(str(out))
    assert merged["a"].metadata["variables"] == [CHL]


def test_extra_metadata_reaches_catalog_entries(prebuilt, tmp_path):
    out = build_catalog(prebuilt, tmp_path / "m.yaml", probe=False, institution="OOI")
    assert intake.from_yaml_file(str(out))["a"].metadata["institution"] == "OOI"


def test_a_catalog_is_not_mistaken_for_a_spec_mapping(prebuilt):
    """Dispatch has to be unambiguous: a Catalog is not a dict and has no .items()."""
    from ocean_skill.build import _is_catalog

    assert _is_catalog(prebuilt)
    assert not _is_catalog({"a": "http://x/y.nc"})
    assert not _is_catalog([("a", "http://x/y.nc")])


def test_add_catalog_still_works_and_returns_names(prebuilt):
    cat = new_catalog()
    assert sorted(add_catalog(cat, prebuilt, probe=False)) == ["a", "b"]


# -- read failure vs probe failure --------------------------------------------


def test_an_unreadable_source_is_fatal_not_quietly_banked(netcdfs, tmp_path):
    """A dead path yields a dead entry; keeping it defers the error from its cause."""
    with pytest.raises(RuntimeError, match="broken"):
        build_catalog(
            {"ok": netcdfs["jan"], "broken": str(tmp_path / "missing.nc")},
            tmp_path / "m.yaml",
        )


def test_a_readable_but_unprobeable_source_is_kept_with_a_warning(tmp_path):
    """Undeciphered metadata is not an invalid entry -- it still reads fine."""
    from unittest import mock

    from ocean_skill import build

    path = tmp_path / "x.nc"
    xr.Dataset({"v": (("i",), np.ones(3))}).to_netcdf(path)
    with mock.patch.object(build, "_probe", side_effect=RuntimeError("no axes")):
        with pytest.warns(UserWarning, match="could not derive metadata"):
            out = build_catalog({"odd": str(path)}, tmp_path / "m.yaml")
    assert sorted(intake.from_yaml_file(str(out))) == ["odd"]


def test_add_catalog_tolerates_one_bad_entry_in_a_sweep(prebuilt, tmp_path):
    """A live server sweep always has a few duds; losing the other forty is useless."""
    from unittest import mock

    cat = new_catalog()
    real = type(prebuilt["a"]).read

    def flaky(self, *a, **k):
        raise RuntimeError("server hiccup")

    with mock.patch.object(type(prebuilt["a"]), "read", flaky):
        with pytest.warns(UserWarning):
            added = add_catalog(cat, prebuilt, probe=True)
    assert added == []  # both failed to read, but nothing raised
    type(prebuilt["a"]).read = real


def test_time_coverage_falls_back_to_global_attributes(tmp_path):
    """MODIS L3-mapped files carry no time axis but do declare their coverage.

    Reading only the time *axis* left every such entry untimed, so a `time=` search
    could never match them — even though the file says exactly when it is from.
    """
    from ocean_skill.build import _probe

    path = tmp_path / "l3m.nc"
    xr.Dataset(
        {"chlor_a": (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": [1.0, 2.0], "lon": [1.0, 2.0]},
        attrs={
            "time_coverage_start": "2003-01-01T00:50:01.000Z",
            "time_coverage_end": "2022-02-01T02:39:59.000Z",
        },
    ).to_netcdf(path)

    md = _probe(xr.open_dataset(path), None)
    assert md["time_coverage_start"] == "2003-01-01"
    assert md["time_coverage_end"] == "2022-02-01"


def test_a_real_time_axis_still_wins_over_global_attributes(tmp_path):
    """The data is the authority; the attribute is only a fallback."""
    from ocean_skill.build import _probe

    path = tmp_path / "withaxis.nc"
    xr.Dataset(
        {"v": (("time",), np.ones(3))},
        coords={"time": xr.date_range("2015-06-01", periods=3, freq="D")},
        attrs={"time_coverage_start": "1999-01-01", "time_coverage_end": "1999-12-31"},
    ).to_netcdf(path)

    md = _probe(xr.open_dataset(path), None)
    assert md["time_coverage_start"] == "2015-06-01"


# --------------------------------------------------------------- kerchunk targets


def test_the_extension_picks_the_kerchunk_format(tmp_path):
    """A ``.json`` path must not be written in parquet format.

    Parquet targets are *directories*, so the mismatch produces a directory named
    ``x.json`` and only fails much later, on read, as ``IsADirectoryError``.
    """
    from ocean_skill.build import _kerchunk_format

    assert _kerchunk_format(tmp_path / "refs.json") == "json"
    assert _kerchunk_format(tmp_path / "refs.JSON") == "json"
    assert _kerchunk_format(tmp_path / "refs.parquet") == "parquet"
    assert _kerchunk_format(tmp_path / "refs") == "parquet"


def test_a_json_reference_round_trips_as_one_file(tmp_path):
    """End-to-end: build a JSON reference over two files and read the values back."""
    from ocean_skill.build import make_kerchunk

    paths = []
    for i in range(2):
        path = tmp_path / f"day{i}.nc"
        xr.Dataset(
            {"chlor_a": (("time", "lat", "lon"), np.full((1, 3, 4), float(i)))},
            coords={
                "time": [np.datetime64("2012-01-01") + np.timedelta64(i, "D")],
                "lat": np.linspace(31, 18, 3),
                "lon": np.linspace(-98, -80, 4),
            },
        ).to_netcdf(path)
        paths.append(path)

    out = make_kerchunk(paths, out=tmp_path / "refs.json")

    assert out.is_file(), "a json target must be a file, not a parquet directory"
    ds = xr.open_dataset(str(out), engine="kerchunk", chunks={})
    assert ds.sizes["time"] == 2
    assert float(ds.chlor_a.isel(time=1, lat=0, lon=0)) == 1.0


def test_a_remote_url_survives_path_normalization():
    """``Path("http://h/f.nc")`` collapses the ``//``, which silently breaks URLs."""
    from ocean_skill.build import _is_remote, _store_for

    url = "https://example.org/data/f.nc"
    assert _is_remote(url)
    key, _store = _store_for(url)
    assert key == url, "the registry key doubles as the URL virtualizarr opens"


# ------------------------------------------------------------- container formats

# "NetCDF" is two unrelated formats sharing one extension: netCDF-4 is HDF5, netCDF-3
# is not. ROMS routinely writes netCDF-4 output beside a netCDF-3 grid file, so a
# single kerchunk store needs both readers. Using HDFParser on a netCDF-3 file fails
# with h5py's "file signature not found", which names neither the file nor the reason.


def _roms_like(path, fmt, value=1.0, t0=0.0):
    """One output file holding two half-daily records starting at ``t0``.

    ``t0`` matters whenever several of these are concatenated: real output files
    continue the series rather than restarting it, and make_kerchunk now warns about
    an axis that does not strictly increase.
    """
    xr.Dataset(
        {"NO3": (("ocean_time", "eta_rho", "xi_rho"), np.full((2, 4, 5), value))},
        coords={"ocean_time": ("ocean_time", [t0, t0 + 43200.0], {"units": "second"})},
    ).to_netcdf(path, format=fmt)
    return path


def _grid(path, fmt):
    xr.Dataset(
        {
            "lon_rho": (
                ("eta_rho", "xi_rho"),
                np.linspace(-160, -140, 20).reshape(4, 5),
            ),
            "lat_rho": (("eta_rho", "xi_rho"), np.linspace(50, 60, 20).reshape(4, 5)),
            "h": (("eta_rho", "xi_rho"), np.full((4, 5), 100.0)),
        }
    ).to_netcdf(path, format=fmt)
    return path


def test_the_container_format_is_sniffed_not_assumed(tmp_path):
    """``.nc`` says nothing about which of the two formats is inside."""
    from ocean_skill.build import _file_format

    assert _file_format(_roms_like(tmp_path / "n4.nc", "NETCDF4")) == "hdf5"
    assert _file_format(_roms_like(tmp_path / "n3.nc", "NETCDF3_CLASSIC")) == "netcdf3"
    assert _file_format(_roms_like(tmp_path / "n3b.nc", "NETCDF3_64BIT")) == "netcdf3"
    assert _file_format(tmp_path / "does_not_exist.nc") is None


def test_a_netcdf3_grid_merges_into_netcdf4_output(tmp_path):
    """The real ROMS combination, and the one that used to raise OSError."""
    from ocean_skill.build import make_kerchunk

    files = [
        _roms_like(tmp_path / f"out.{i}.nc", "NETCDF4", value=i + 1, t0=i * 86400.0)
        for i in range(2)
    ]
    grid = _grid(tmp_path / "grid.nc", "NETCDF3_CLASSIC")

    out = make_kerchunk(files, out=tmp_path / "refs.json", grid=grid)

    ds = xr.open_dataset(str(out), engine="kerchunk", chunks={}, decode_times=False)
    assert ds.sizes["ocean_time"] == 4
    assert {"lon_rho", "lat_rho", "h"} <= set(ds.variables)
    assert np.allclose(ds.lon_rho.values, xr.open_dataset(grid).lon_rho.values)


@pytest.mark.parametrize(
    ("out_fmt", "grid_fmt"),
    [
        ("NETCDF4", "NETCDF4"),
        ("NETCDF4", "NETCDF3_CLASSIC"),
        ("NETCDF3_CLASSIC", "NETCDF3_CLASSIC"),
        ("NETCDF3_64BIT", "NETCDF4"),
    ],
)
def test_every_format_combination_builds(tmp_path, out_fmt, grid_fmt):
    from ocean_skill.build import make_kerchunk

    files = [
        _roms_like(tmp_path / f"o.{i}.nc", out_fmt, t0=i * 86400.0) for i in range(2)
    ]
    out = make_kerchunk(
        files, out=tmp_path / "r.json", grid=_grid(tmp_path / "g.nc", grid_fmt)
    )

    ds = xr.open_dataset(str(out), engine="kerchunk", chunks={}, decode_times=False)
    assert ds.sizes["ocean_time"] == 4
    assert "h" in ds.variables


def _cdf5(path):
    """Write a CDF-5 file — netCDF-3's 64-bit-data variant (``ncdump -k``: cdf5)."""
    from netCDF4 import Dataset

    nc = Dataset(path, "w", format="NETCDF3_64BIT_DATA")
    nc.createDimension("eta_rho", 4)
    nc.createDimension("xi_rho", 5)
    nc.createVariable("h", "f8", ("eta_rho", "xi_rho"))[:] = np.full((4, 5), 100.0)
    nc.close()
    return path


def test_cdf5_is_distinguished_from_other_netcdf3(tmp_path):
    from ocean_skill.build import _file_format

    assert _file_format(_cdf5(tmp_path / "c5.nc")) == "cdf5"
    assert _file_format(_roms_like(tmp_path / "c1.nc", "NETCDF3_CLASSIC")) == "netcdf3"


def _reads_cdf5():
    from ocean_skill.build import _netcdf3_reads_cdf5

    return _netcdf3_reads_cdf5()


@pytest.mark.skipif(_reads_cdf5(), reason="this virtualizarr's parser reads CDF-5")
def test_cdf5_is_refused_with_the_conversion_command(tmp_path):
    """The kerchunk-backed NetCDF3Parser is scipy-backed, and scipy reads CDF-1/2 only.

    Left to itself it dies with ``IndexError: index 0 is out of bounds`` inside scipy,
    which names neither the file nor the format. A real ROMS grid file hit this.
    """
    from ocean_skill.build import make_kerchunk

    files = [
        _roms_like(tmp_path / f"o.{i}.nc", "NETCDF4", t0=i * 86400.0) for i in range(2)
    ]

    with pytest.raises(ValueError, match=r"CDF-5.*cannot kerchunk") as excinfo:
        make_kerchunk(files, out=tmp_path / "r.json", grid=_cdf5(tmp_path / "g.nc"))

    assert "nccopy -k netCDF-4" in str(excinfo.value), (
        "the error must say how to fix it"
    )


@pytest.mark.skipif(
    not _reads_cdf5(), reason="needs virtualizarr's native netCDF3 parser"
)
def test_cdf5_grid_values_survive_the_reference(tmp_path):
    """A CDF-5 grid must come back bit-identical, not merely build without raising.

    The parser's whole job is byte offsets, and a wrong offset does not raise — it
    returns plausible-looking numbers from the wrong part of the file. So this asserts
    values against what the C library reads from the same file, not just that a
    reference was produced.
    """
    from ocean_skill.build import make_kerchunk

    files = [
        _roms_like(tmp_path / f"o.{i}.nc", "NETCDF4", t0=i * 86400.0) for i in range(2)
    ]
    grid = _cdf5(tmp_path / "g.nc")

    out = make_kerchunk(files, out=tmp_path / "r.json", grid=grid)

    ds = xr.open_dataset(str(out), engine="kerchunk", chunks={}, decode_times=False)
    expected = xr.open_dataset(grid, engine="netcdf4")
    assert ds.h.dtype == expected.h.dtype
    np.testing.assert_array_equal(ds.h.values, expected.h.values)


# --------------------------------------------------- disordered concat axis

# `combine="nested"` joins in the order given and checks nothing, so globbing two
# output streams into one call yields an axis that runs forward, jumps back, and runs
# forward again — and the store reads back without complaint. The real case mixed ROMS
# `cdr` averages with `rst` restarts: the restarts hold two records each, and their
# second one repeated a cdr timestamp exactly. Averages and instantaneous snapshots
# then share a coordinate, which no later reader can untangle.


def test_a_well_ordered_series_says_nothing(tmp_path):
    """The warning is worthless if the ordinary multi-file build also trips it."""
    from ocean_skill.build import make_kerchunk

    files = [
        _roms_like(tmp_path / f"o.{i}.nc", "NETCDF4", t0=i * 86400.0) for i in range(3)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_kerchunk(files, out=tmp_path / "r.json")

    assert not [w for w in caught if "not strictly increasing" in str(w.message)]


def test_two_streams_in_one_reference_are_warned_about(tmp_path):
    """The cdr + rst shape: a second stream repeating the first stream's times."""
    from ocean_skill.build import make_kerchunk

    files = [
        _roms_like(tmp_path / f"cdr.{i}.nc", "NETCDF4", t0=i * 86400.0)
        for i in range(2)
    ]
    files += [
        _roms_like(tmp_path / f"rst.{i}.nc", "NETCDF4", t0=i * 86400.0)
        for i in range(2)
    ]

    with pytest.warns(UserWarning, match="not strictly increasing") as caught:
        out = make_kerchunk(files, out=tmp_path / "r.json")

    message = str(caught[0].message)
    assert "ocean_time" in message, "the offending variable must be named"
    assert "4 files" in message
    assert "one reference per stream" in message, "say what to do about it"

    # Warned about, not repaired: every record is still there, in file order.
    ds = xr.open_dataset(str(out), engine="kerchunk", chunks={}, decode_times=False)
    assert ds.sizes["ocean_time"] == 8


def test_the_warning_names_a_date_not_a_raw_roms_time(tmp_path):
    """ROMS times are seconds since an epoch stated only in the long_name.

    "first at index 4 (410313150.0)" tells you nothing about *when* the axis doubled
    back, which is the one thing you need to identify the offending stream.
    """
    from ocean_skill.build import make_kerchunk

    def roms_times(path, times):
        xr.Dataset(
            {"NO3": (("ocean_time", "eta_rho", "xi_rho"), np.ones((2, 4, 5)))},
            coords={
                "ocean_time": (
                    "ocean_time",
                    times,
                    {"units": "second", "long_name": "Time since 2000/01/01"},
                )
            },
        ).to_netcdf(path)
        return path

    day = 86400.0
    files = [
        roms_times(tmp_path / "a.nc", [0.0, day]),
        roms_times(tmp_path / "b.nc", [0.0, day]),
    ]

    with pytest.warns(UserWarning, match="not strictly increasing") as caught:
        make_kerchunk(files, out=tmp_path / "r.json")

    assert "2000-01-01T00:00:00" in str(caught[0].message)


# ------------------------------------------------------------- tilde expansion

# Path("~/x") keeps the tilde literally. obstore then reports "Unable to canonicalize
# filesystem root: ~/...", which does not hint that the path merely needed expanding.
# `root` and `out_dir` were already expanded; `grid` was not, so it was the one
# argument of the three that rejected a perfectly ordinary "~/..." path.


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point HOME at a temp dir, so ``~`` resolves somewhere assertable."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_file_format_expands_tilde(fake_home):
    """A None here would silently downgrade format detection to the HDF default."""
    from ocean_skill.build import _file_format

    _roms_like(fake_home / "f.nc", "NETCDF4")
    assert _file_format("~/f.nc") == "hdf5"


def test_store_for_expands_tilde(fake_home):
    from ocean_skill.build import _store_for

    _roms_like(fake_home / "f.nc", "NETCDF4")
    url, _store = _store_for("~/f.nc")
    assert "~" not in url
    assert str(fake_home) in url


def test_build_kerchunk_accepts_tilde_for_every_path(fake_home):
    """root, grid and out_dir all take ``~`` — grid used to be the odd one out."""
    from ocean_skill.build import build_kerchunk

    (fake_home / "runs").mkdir()
    for i in range(2):
        _roms_like(
            fake_home / "runs" / f"o.{i}.nc", "NETCDF4", value=i + 1, t0=i * 86400.0
        )
    _grid(fake_home / "grid.nc", "NETCDF4")

    # Building at all is the assertion: before the fix this raised obstore's
    # "Unable to canonicalize filesystem root: ~/grid.nc". Deliberately no read-back —
    # build_kerchunk writes parquet, and the parquet reference reader is intermittently
    # broken upstream ("boolean value of NA is ambiguous"), which would make this test
    # flaky for a reason with nothing to do with tilde expansion.
    refs = build_kerchunk(
        {"dev": "runs/o.*.nc"},
        root="~/",
        grid="~/grid.nc",
        out_dir="~/refs/",
    )

    built = refs["dev"]
    assert "~" not in str(built)
    assert built.exists()
    assert built.is_relative_to(fake_home / "refs")
