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
    _reader_for,
    _resolve_files,
    add_catalog,
    add_copernicus_source,
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


def test_save_prints_a_vocabulary_match_summary_not_a_persisted_map(
    tmp_path, capsys
):
    """The summary is stdout only -- nothing written into the catalog file itself.

    See ocean_skill.vocabulary.MatchReport on why: a persisted ``{nickname:
    [variables]}`` map would go stale the moment the vocabulary changes and this
    catalog isn't rebuilt.
    """
    path = tmp_path / "ctd.nc"
    xr.Dataset(
        {
            "Temperature_CTD": (
                ("lat", "lon"),
                np.ones((4, 5)),
                {
                    "standard_name": "sea_water_potential_temperature",
                    "units": "degC",
                },
            ),
            "Instrument_Type": (
                ("lat", "lon"),
                np.ones((4, 5)),
                {"standard_name": "instrument_type"},  # not a real CF name
            ),
        },
        coords={"lat": np.linspace(10, 20, 4), "lon": np.linspace(-100, -90, 5)},
    ).to_netcdf(path)

    out_path = build_catalog({"ctd": str(path)}, tmp_path / "ctd_catalog.yaml")

    printed = capsys.readouterr().out
    assert "2 variables across 1 sources: 1 matched, 1 unmatched" in printed
    assert "instrument_type" in printed
    assert "nickname" not in out_path.read_text()
    assert "instrument_type" in out_path.read_text()  # the raw standard_name, fine


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


# -- add_copernicus_source ----------------------------------------------------


def test_copernicus_source_adds_one_entry_per_chunking(tmp_path):
    """Each ARCO layout becomes its own suffixed, service-tagged entry."""
    cat = new_catalog()
    add_copernicus_source(
        cat,
        "chl_gapfree_my_daily",
        "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        probe=False,  # would need copernicusmarine + a login: assert the entries only
    )
    out = tmp_path / "cmems.yaml"
    from ocean_skill.build import save

    reloaded = intake.from_yaml_file(str(save(cat, out)))
    assert sorted(reloaded) == [
        "chl_gapfree_my_daily_geo",
        "chl_gapfree_my_daily_timeseries",
    ]

    ts = reloaded["chl_gapfree_my_daily_timeseries"]
    assert "CopernicusMarineReader" in str(ts.reader)
    # the chunking approach is recorded in metadata, and matched by the reader's service
    assert ts.metadata["chunking"] == "time-series"
    assert ts.metadata["service"] == "arco-time-series"
    assert ts.kwargs["service"] == "arco-time-series"
    assert (
        ts.metadata["dataset_id"]
        == "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
    )

    geo = reloaded["chl_gapfree_my_daily_geo"]
    assert geo.metadata["chunking"] == "geo"
    assert geo.kwargs["service"] == "arco-geo-series"


def test_copernicus_source_can_add_a_single_service(tmp_path):
    cat = new_catalog()
    added = add_copernicus_source(
        cat, "glorys_my_daily", "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        services=("arco-geo-series",), probe=False,
    )
    assert list(added) == ["glorys_my_daily_geo"]


def test_copernicus_source_rejects_an_unknown_service(tmp_path):
    with pytest.raises(ValueError, match="unknown Copernicus service"):
        add_copernicus_source(
            new_catalog(), "x", "some_id", services=("arco-bogus",), probe=False
        )


def test_copernicus_reader_without_the_toolbox_is_a_clear_error():
    """With copernicusmarine absent, reading says exactly what to install/do."""
    from importlib.util import find_spec

    if find_spec("copernicusmarine") is not None:
        pytest.skip("copernicusmarine is installed; cannot test the missing-dep path")

    from ocean_skill.readers import CopernicusMarineReader

    reader = CopernicusMarineReader(
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        service="arco-time-series",
    )
    with pytest.raises(RuntimeError, match="copernicusmarine"):
        reader.read()


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


def test_probe_retries_a_transient_read_then_succeeds(tmp_path):
    """A flaky server's read that recovers on a later try must not cost the entry."""
    from ocean_skill import build

    path = tmp_path / "x.nc"
    xr.Dataset({"v": (("i",), np.ones(3))}).to_netcdf(path)

    cat = new_catalog()
    reader = build._reader_for(str(path))
    real_read = reader.read
    calls = {"n": 0}

    def flaky_then_ok(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:  # first two attempts 500, third succeeds
            raise RuntimeError("server hiccup 500")
        return real_read(*a, **k)

    reader.read = flaky_then_ok
    with pytest.warns(UserWarning, match="retrying"):
        build._attach(cat, "flaky", reader, probe=True, name_map=None, metadata={})

    assert calls["n"] == 3  # two retries, then the read landed
    assert list(cat) == ["flaky"]  # and the entry is in the catalog


def test_probe_retries_are_bounded_then_give_up(tmp_path, monkeypatch):
    """A read that never recovers is retried PROBE_RETRIES times, then propagates."""
    from ocean_skill import build

    monkeypatch.setattr(build, "PROBE_RETRIES", 2)
    path = tmp_path / "x.nc"
    xr.Dataset({"v": (("i",), np.ones(3))}).to_netcdf(path)

    reader = build._reader_for(str(path))
    calls = {"n": 0}

    def always_fails(*a, **k):
        calls["n"] += 1
        raise RuntimeError("still down")

    reader.read = always_fails
    with pytest.warns(UserWarning, match="retrying"):
        with pytest.raises(RuntimeError, match="still down"):
            build._attach(new_catalog(), "dead", reader, probe=True, name_map=None, metadata={})
    assert calls["n"] == 3  # the initial attempt plus PROBE_RETRIES=2 more


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


# --------------------------------------------- variable -> standard_name resolution


def _glodap_like_dataset():
    """Return a GLODAP-shaped Dataset: raw per-variable names, no standard_name attrs.

    ``Cant``/``OmegaA`` stand in for GLODAP's genuinely un-nameable diagnostics
    (anthropogenic carbon, aragonite saturation state) -- CF has no single
    standard_name for either, so they must stay unmapped either way.
    """
    coord = ("x",)
    data = {
        name: (coord, np.ones(2))
        for name in (
            "NO3",
            "PO4",
            "TAlk",
            "TCO2",
            "oxygen",
            "salinity",
            "silicate",
            "temperature",
            "Cant",
            "OmegaA",
        )
    }
    return xr.Dataset(data, coords={"x": [0, 1]})


def test_probe_falls_back_to_the_vocabulary_when_name_map_misses():
    """GLODAP's raw names resolve via the shared vocabulary with no bespoke name_map.

    Regression guard for the silent partial-mapping bug: building GLODAP with the
    *default* ``ROMS_STANDARD_NAMES`` name_map used to capture only NO3/PO4 (the only
    two names that map coincidentally overlaps), since GLODAP's files carry no
    standard_name attrs and nothing else recognized the rest.
    """
    from ocean_skill.build import ROMS_STANDARD_NAMES, _probe

    md = _probe(_glodap_like_dataset(), ROMS_STANDARD_NAMES)

    assert md["standard_names"] == {
        "NO3": "mole_concentration_of_nitrate_in_sea_water",
        "PO4": "mole_concentration_of_phosphate_in_sea_water",
        "TAlk": "sea_water_alkalinity_expressed_as_mole_equivalent",
        "TCO2": "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water",
        "oxygen": "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
        "salinity": "sea_water_practical_salinity",
        "silicate": "mole_concentration_of_silicate_in_sea_water",
        "temperature": "sea_water_potential_temperature",
    }
    assert "Cant" not in md["standard_names"]
    assert "OmegaA" not in md["standard_names"]


def test_probe_falls_back_to_the_vocabulary_with_no_name_map_at_all():
    """``name_map=None`` skips straight to the vocabulary, not to attrs-only."""
    from ocean_skill.build import _probe

    md = _probe(_glodap_like_dataset(), None)
    assert md["standard_names"]["temperature"] == "sea_water_potential_temperature"


def test_name_map_still_wins_over_the_vocabulary_on_conflict():
    """The caller's name_map is checked before the shared vocabulary, not after."""
    from ocean_skill.build import _probe

    ds = xr.Dataset({"NO3": (("x",), np.ones(2))}, coords={"x": [0, 1]})
    md = _probe(ds, {"NO3": "a_custom_standard_name"})
    assert md["standard_names"]["NO3"] == "a_custom_standard_name"


def test_declared_attr_still_wins_over_the_vocabulary():
    """A file's own standard_name attr is authoritative over any fallback tier."""
    from ocean_skill.build import _probe

    ds = xr.Dataset(
        {"NO3": ("x", np.ones(2), {"standard_name": "a_declared_standard_name"})},
        coords={"x": [0, 1]},
    )
    md = _probe(ds, {"NO3": "would_be_from_name_map"})
    assert md["standard_names"]["NO3"] == "a_declared_standard_name"


# ----------------------------------------------------------- domain_outline (perimeter)


def test_probe_writes_a_domain_outline_for_a_curvilinear_grid():
    """A 2-D lon/lat grid gets its true grid-edge ring recorded, JSON-plain."""
    from ocean_skill.build import _probe

    ny, nx = 8, 10
    lon = np.linspace(150.0, 200.0, nx)[None, :] * np.ones((ny, 1))
    lat = np.linspace(-10.0, 10.0, ny)[:, None] * np.ones((1, nx))
    ds = xr.Dataset(
        {"temp": (("eta_rho", "xi_rho"), np.random.rand(ny, nx))},
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
        },
    )
    md = _probe(ds, None)
    outline = md["domain_outline"]
    assert isinstance(outline, list)
    assert all(
        isinstance(pt, list) and all(isinstance(v, float) for v in pt)
        for pt in outline
    )
    # plain floats/lists round-trip through YAML without a custom representer
    import yaml

    assert yaml.safe_load(yaml.safe_dump({"domain_outline": outline})) == {
        "domain_outline": outline
    }


def test_probe_writes_no_domain_outline_for_a_rectilinear_grid():
    """A 1-D lon/lat grid's bbox already is its perimeter -- no separate key needed."""
    from ocean_skill.build import _probe

    ds = xr.Dataset(
        {"temp": (("lat", "lon"), np.ones((4, 5)))},
        coords={"lon": np.linspace(0.0, 10.0, 5), "lat": np.linspace(0.0, 5.0, 4)},
    )
    md = _probe(ds, None)
    assert "domain_outline" not in md


def test_add_source_extracts_a_domain_outline_from_a_separate_roms_grid(tmp_path):
    """A ROMS entry whose grid never merges into the store still gets an outline.

    ``self_contained_grid`` is false whenever the grid stays a separate file (not
    merged in by ``make_kerchunk``'s ``grid=``) -- the raw probed output then has no
    ``lon_rho``/``lat_rho`` at all, so :func:`_probe` alone cannot derive one. This is
    exactly the ``pac_dt_ramp``-shaped case: the grid file is read again, once, for
    its shape.
    """
    from ocean_skill import build

    ny, nx = 8, 10
    lon = np.linspace(150.0, 200.0, nx)[None, :] * np.ones((ny, 1))
    lat = np.linspace(-10.0, 10.0, ny)[:, None] * np.ones((1, nx))
    grid_path = tmp_path / "grid.nc"
    xr.Dataset(
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
        }
    ).to_netcdf(grid_path)

    out_path = tmp_path / "out.nc"
    out = xr.Dataset(
        {
            "temp": (
                ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
                np.zeros((1, 2, ny, nx)),
            )
        },
        coords={
            "Cs_r": ("s_rho", np.linspace(-1.0, 0.0, 2)),
            "sigma_r": ("s_rho", np.linspace(-1.0, 0.0, 2)),
        },
    )
    out.attrs.update(theta_s=5.0, theta_b=1.0, hc=20.0)
    out.to_netcdf(out_path)

    cat = build.new_catalog(title="t")
    build.add_source(cat, "roms_separate_grid", out_path, grid=str(grid_path))
    md = cat["roms_separate_grid"].metadata
    assert md["self_contained_grid"] is False
    assert "domain_outline" in md
    assert len(md["domain_outline"]) >= 4


def test_a_missing_roms_grid_warns_but_keeps_the_entry(tmp_path):
    """No outline without a real grid file -- but the entry still lands, per pattern."""
    from ocean_skill import build

    ny, nx = 8, 10
    out_path = tmp_path / "out.nc"
    out = xr.Dataset(
        {
            "temp": (
                ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
                np.zeros((1, 2, ny, nx)),
            )
        },
        coords={
            "Cs_r": ("s_rho", np.linspace(-1.0, 0.0, 2)),
            "sigma_r": ("s_rho", np.linspace(-1.0, 0.0, 2)),
        },
    )
    out.attrs.update(theta_s=5.0, theta_b=1.0, hc=20.0)
    out.to_netcdf(out_path)

    cat = build.new_catalog(title="t")
    with pytest.warns(UserWarning, match="domain outline"):
        build.add_source(cat, "no_grid", out_path, grid=str(tmp_path / "missing.nc"))
    assert "no_grid" in list(cat)
    assert "domain_outline" not in cat["no_grid"].metadata


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


# --------------------------------------------------------------- engine selection


def test_reader_for_picks_h5netcdf_remote_and_netcdf4_local(tmp_path):
    """Default engine choice, unchanged by the override support below."""
    local = tmp_path / "f.nc"
    xr.Dataset({"chlor_a": (("lat",), np.ones(3))}).to_netcdf(local)

    remote_reader = _reader_for("https://example.org/data/f.nc")
    assert remote_reader.kwargs["engine"] == "h5netcdf"

    local_reader = _reader_for(str(local))
    assert local_reader.kwargs["engine"] == "netcdf4"


def test_an_explicit_engine_in_reader_kwargs_overrides_the_default(tmp_path):
    """A classic-format (netCDF3) file over a remote-shaped URL needs ``scipy``.

    ``h5netcdf`` only reads netCDF4/HDF5, so the auto-detected default fails on a
    classic-format file no matter the transport. Before the fix, passing
    ``engine`` through ``reader_kwargs`` collided with the ``engine=`` already
    baked into ``_reader_for``'s call and raised
    ``TypeError: got multiple values for keyword argument 'engine'``.
    """
    classic = tmp_path / "classic.nc"
    xr.Dataset({"chlor_a": (("lat", "lon"), np.ones((4, 5)))}).to_netcdf(
        classic, format="NETCDF3_CLASSIC"
    )
    url = "file://" + str(classic)

    cat = new_catalog()
    reader = add_source(cat, "classic", url, reader_kwargs={"engine": "scipy"}, probe=False)

    assert reader.kwargs["engine"] == "scipy"
    ds = cat["classic"].read()
    assert float(ds.chlor_a.isel(lat=0, lon=0)) == 1.0


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


# --------------------------------------------------------- keep="latest-per-file"

# A ROMS restart file writes more than one time record (typically two), and under
# cycling restarts (LcycleRST) the newest record is not always written to the last
# slot. Records carry distinct *values* here, not just distinct times, so a test that
# reads back the wrong record is caught, not just one that reads back the wrong count.


def _roms_restart_like(path, fmt, records):
    """A restart file with ``records`` as ``[(time, value), ...]``, one per record."""
    times, values = zip(*records)
    data = np.stack([np.full((4, 5), v) for v in values])
    xr.Dataset(
        {"NO3": (("ocean_time", "eta_rho", "xi_rho"), data)},
        coords={"ocean_time": ("ocean_time", list(times), {"units": "second"})},
    ).to_netcdf(path, format=fmt)
    return path


def test_keep_latest_per_file_drops_the_earlier_record(tmp_path):
    """Two two-record restart files -> one kept record each, from the latest time."""
    from ocean_skill.build import make_kerchunk

    day = 86400.0
    files = [
        _roms_restart_like(
            tmp_path / f"rst.{i}.nc",
            "NETCDF4",
            [(i * day, 10 * i + 1), (i * day + 43200.0, 10 * i + 2)],
        )
        for i in range(2)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = make_kerchunk(files, out=tmp_path / "r.json", keep="latest-per-file")

    assert not [w for w in caught if "not strictly increasing" in str(w.message)]

    ds = xr.open_dataset(str(out), engine="kerchunk", chunks={}, decode_times=False)
    assert ds.sizes["ocean_time"] == 2
    assert list(ds.ocean_time.values) == [43200.0, 129600.0]
    assert list(ds.NO3.isel(eta_rho=0, xi_rho=0).values) == [2.0, 12.0]


def test_keep_latest_per_file_uses_the_time_value_not_the_last_slot(tmp_path):
    """Cycling restarts can write the newest record to any slot, not just the last."""
    from ocean_skill.build import make_kerchunk

    files = [
        # the newer record (later time, value 2) sits in slot 0 here, not slot 1
        _roms_restart_like(tmp_path / "rst.0.nc", "NETCDF4", [(43200.0, 2), (0.0, 1)]),
        _roms_restart_like(
            tmp_path / "rst.1.nc", "NETCDF4", [(86400.0, 3), (129600.0, 4)]
        ),
    ]

    out = make_kerchunk(files, out=tmp_path / "r.json", keep="latest-per-file")

    ds = xr.open_dataset(str(out), engine="kerchunk", chunks={}, decode_times=False)
    assert list(ds.ocean_time.values) == [43200.0, 129600.0]
    assert list(ds.NO3.isel(eta_rho=0, xi_rho=0).values) == [2.0, 4.0]


def test_keep_latest_per_file_works_on_netcdf3(tmp_path):
    """The classic ROMS restart format, not just netCDF-4."""
    from ocean_skill.build import make_kerchunk

    files = [
        _roms_restart_like(
            tmp_path / f"rst.{i}.nc",
            "NETCDF3_64BIT",
            [(i * 86400.0, 10 * i + 1), (i * 86400.0 + 43200.0, 10 * i + 2)],
        )
        for i in range(2)
    ]

    out = make_kerchunk(files, out=tmp_path / "r.json", keep="latest-per-file")

    ds = xr.open_dataset(str(out), engine="kerchunk", chunks={}, decode_times=False)
    assert ds.sizes["ocean_time"] == 2
    assert list(ds.NO3.isel(eta_rho=0, xi_rho=0).values) == [2.0, 12.0]


def test_keep_latest_per_file_via_build_kerchunk(tmp_path):
    """The stream-dict wrapper forwards keep= like any other kerchunk kwarg."""
    from ocean_skill.build import build_kerchunk

    for i in range(2):
        _roms_restart_like(
            tmp_path / f"rst.{i}.nc",
            "NETCDF4",
            [(i * 86400.0, 10 * i + 1), (i * 86400.0 + 43200.0, 10 * i + 2)],
        )

    refs = build_kerchunk(
        {"GOM_rst": "rst.*.nc"},
        root=tmp_path,
        out_dir=tmp_path / "refs",
        keep="latest-per-file",
    )

    ds = xr.open_dataset(
        str(refs["GOM_rst"]), engine="kerchunk", chunks={}, decode_times=False
    )
    assert ds.sizes["ocean_time"] == 2


def test_an_unrecognized_keep_value_is_rejected(tmp_path):
    from ocean_skill.build import make_kerchunk

    files = [_roms_like(tmp_path / "o.nc", "NETCDF4")]

    with pytest.raises(ValueError, match="keep"):
        make_kerchunk(files, out=tmp_path / "r.json", keep="every-other")


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


# --------------------------------------------------------------- discover_opendap_files
#
# The two THREDDS families place data differently: a true THREDDS Data Server (TDS,
# e.g. NCEI) catalogs under /catalog/ but serves data under a separate OPeNDAP
# service base, marking each leaf with a urlPath attribute; Hyrax (e.g. NASA
# oceandata) serves catalog.xml alongside the data and its leaves carry no urlPath.
# Both shapes are canned here so neither can silently regress.

TDS_CATALOG = b"""<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
         xmlns:xlink="http://www.w3.org/1999/xlink" version="1.2">
  <service name="all" serviceType="compound" base="/thredds-ocean/">
    <service name="dap" serviceType="OPeNDAP" base="/thredds-ocean/dodsC/" />
    <service name="http" serviceType="HTTPServer" base="/thredds-ocean/fileServer/" />
  </service>
  <dataset name="DATA/WHOTS/">
    <dataset name="OS_WHOTS_2024_R_M-1.nc"
             urlPath="ndbc/oceansites/DATA/WHOTS/OS_WHOTS_2024_R_M-1.nc">
      <dataSize units="Kbytes">392.7</dataSize>
    </dataset>
    <dataset name="readme.txt" urlPath="ndbc/oceansites/DATA/WHOTS/readme.txt">
      <dataSize units="Kbytes">1.0</dataSize>
    </dataset>
    <catalogRef xlink:href="sub/catalog.xml" xlink:title="sub" name="" />
  </dataset>
</catalog>"""

TDS_SUBCATALOG = b"""<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
         xmlns:xlink="http://www.w3.org/1999/xlink" version="1.2">
  <service name="dap" serviceType="OPeNDAP" base="/thredds-ocean/dodsC/" />
  <dataset name="sub/">
    <dataset name="OS_WHOTS_2023_R_M-1.nc"
             urlPath="ndbc/oceansites/DATA/WHOTS/sub/OS_WHOTS_2023_R_M-1.nc">
      <dataSize units="Kbytes">100.0</dataSize>
    </dataset>
  </dataset>
</catalog>"""

HYRAX_CATALOG = b"""<?xml version="1.0" encoding="UTF-8"?>
<thredds:catalog xmlns:thredds="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
                 xmlns:xlink="http://www.w3.org/1999/xlink">
  <thredds:service name="dap" serviceType="OPeNDAP" base="/opendap/hyrax"/>
  <thredds:dataset name="/MODISA/L3SMI/2003/0101">
    <thredds:dataset name="AQUA_MODIS.20030101.L3m.DAY.CHL.chlor_a.9km.nc"
                     ID="/opendap/hyrax/MODISA/L3SMI/2003/0101/AQUA_MODIS.20030101.L3m.DAY.CHL.chlor_a.9km.nc">
      <thredds:dataSize units="bytes">4966941</thredds:dataSize>
      <thredds:access serviceName="dap"
                      urlPath="/MODISA/L3SMI/2003/0101/AQUA_MODIS.20030101.L3m.DAY.CHL.chlor_a.9km.nc"/>
    </thredds:dataset>
  </thredds:dataset>
</thredds:catalog>"""


@pytest.fixture
def fake_thredds(monkeypatch):
    """Serve canned catalog.xml bodies keyed by URL; record what was fetched."""
    import io
    import urllib.request

    pages: dict[str, bytes] = {}
    fetched: list[str] = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=None):
        fetched.append(url)
        if url not in pages:
            raise AssertionError(f"unexpected fetch: {url}")
        return _Resp(pages[url])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return pages, fetched


def test_discover_tds_joins_urlpath_onto_opendap_service(fake_thredds):
    """TDS leaves resolve under /dodsC/, not under the /catalog/ browse path."""
    from ocean_skill.build import discover_opendap_files

    pages, _ = fake_thredds
    pages[
        "https://www.ncei.noaa.gov/thredds-ocean/catalog/ndbc/oceansites/DATA/WHOTS/catalog.xml"
    ] = TDS_CATALOG

    # The browser URL (catalog.html) is what a person has on hand — pasted as-is.
    urls = discover_opendap_files(
        "https://www.ncei.noaa.gov/thredds-ocean/catalog/ndbc/oceansites/DATA/WHOTS/catalog.html"
    )
    assert urls == [
        "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ndbc/oceansites/DATA/WHOTS/OS_WHOTS_2024_R_M-1.nc"
    ]


def test_discover_tds_recurses_through_catalogref(fake_thredds):
    from ocean_skill.build import discover_opendap_files

    pages, _ = fake_thredds
    root = "https://www.ncei.noaa.gov/thredds-ocean/catalog/ndbc/oceansites/DATA/WHOTS/"
    pages[root + "catalog.xml"] = TDS_CATALOG
    pages[root + "sub/catalog.xml"] = TDS_SUBCATALOG

    urls = discover_opendap_files(root, recurse=True)
    assert (
        "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ndbc/oceansites/DATA/WHOTS/sub/OS_WHOTS_2023_R_M-1.nc"
        in urls
    )
    assert len(urls) == 2


def test_discover_hyrax_still_joins_directory_with_name(fake_thredds):
    """Hyrax's declared service base is untrustworthy — base+name must survive."""
    from ocean_skill.build import discover_opendap_files

    pages, _ = fake_thredds
    base = "http://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/2003/0101/"
    pages[base + "catalog.xml"] = HYRAX_CATALOG

    urls = discover_opendap_files(base, pattern="*.chlor_a.9km.nc")
    assert urls == [base + "AQUA_MODIS.20030101.L3m.DAY.CHL.chlor_a.9km.nc"]


# -- add_source: a caller-declared featureType is canonicalized and marked "declared" -


def test_add_source_featuretype_override_is_canonicalized_and_declared(tmp_path):
    """A repeat-visit station has ragged, position-varying rows -- the probe's own
    guess would land on trajectoryProfile. add_source(featureType=...) overrides
    that, and the override should read like any other declared featureType: the
    package's own canonical spelling, and featureType_source: "declared" rather
    than the stale "inferred" the probe's guess left behind.
    """
    from intake.readers import datatypes, readers

    csv = tmp_path / "station.csv"
    csv.write_text(
        "time,depth (m),lon,lat,Temperature (degC)\n"
        "2024-01-01,1,-21.987,64.2638,8.1\n"
        "2024-01-01,10,-21.988,64.2638,7.9\n"
        "2024-02-01,5,-21.9895,64.2638,8.3\n"
    )
    reader = readers.PandasCSV(datatypes.CSV(url=str(csv)))
    cat = new_catalog(title="t")
    add_source(
        cat,
        "hvalfjordur",
        reader=reader,
        name_map=None,
        featureType="timeseriesprofile",
    )
    md = cat["hvalfjordur"].metadata
    assert md["featureType"] == "timeSeriesProfile"
    assert md["featureType_source"] == "declared"
