"""Tests for catalog auto-discovery and name resolution."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

from ocean_skill import catalog


def test_discover_indexes_sources(isolated_catalogs):
    idx = catalog.discover()
    assert "foo" in idx
    assert idx["foo"].metadata["featureType"] == "grid"


def test_resolve_bare_and_qualified(isolated_catalogs):
    ref = catalog.resolve("foo")
    assert ref.name == "foo"
    # qualified form reaches the same entry
    assert catalog.resolve(ref.qualified).name == "foo"


def test_resolve_unknown_raises(isolated_catalogs):
    with pytest.raises(KeyError):
        catalog.resolve("does_not_exist")


def test_resolve_qualified_miss_suggests_within_that_catalog(isolated_catalogs):
    with pytest.raises(KeyError) as exc:
        catalog.resolve("example catalog:fo")
    message = exc.value.args[0]
    assert "Did you mean: foo?" in message
    assert "osk.find(" in message


def test_resolve_qualified_miss_unknown_catalog_has_no_suggestion(isolated_catalogs):
    with pytest.raises(KeyError) as exc:
        catalog.resolve("no_such_catalog:foo")
    assert "Did you mean" not in exc.value.args[0]


def test_find_by_standard_name_and_feature_type(isolated_catalogs):
    assert "foo" in catalog.find(variable="sea_water_temperature")
    assert "foo" in catalog.find(featureType="grid")
    assert "foo" not in catalog.find(featureType="timeSeries")


def test_catalogs_registry_membership(isolated_catalogs):
    from ocean_skill import catalogs

    assert "foo" in catalogs
    assert "foo" in catalogs.names()


# -- discover() caching --------------------------------------------------------


def test_discover_caches_until_files_change(isolated_catalogs, monkeypatch):
    import intake

    calls = []
    orig = intake.from_yaml_file

    def counting(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(intake, "from_yaml_file", counting)

    catalog.discover()
    catalog.discover()
    assert len(calls) == 1  # second call served from cache, no re-parse


def test_discover_invalidates_on_catalog_rewrite(isolated_catalogs, tmp_path):
    import intake
    from intake.readers import datatypes, readers

    idx = catalog.discover()
    assert idx["foo"].metadata["featureType"] == "grid"

    data = datatypes.HDF5(url=str(tmp_path / "foo.nc"))
    reader = readers.XArrayDatasetReader(data)
    reader.metadata.update({"featureType": "timeSeries"})
    cat = intake.entry.Catalog(metadata={"title": "example catalog"})
    cat["foo"] = reader
    cat.aliases["foo"] = "foo"
    cat.to_yaml_file(str(isolated_catalogs / "example.catalog.yaml"))

    idx2 = catalog.discover()
    assert idx2["foo"].metadata["featureType"] == "timeSeries"


def test_discover_invalidates_on_search_path_change(
    isolated_catalogs, tmp_path, monkeypatch
):
    import intake
    from intake.readers import datatypes, readers

    idx = catalog.discover()
    assert "foo" in idx

    other = tmp_path / "other_cats"
    other.mkdir()
    data = datatypes.HDF5(url=str(tmp_path / "bar.nc"))
    reader = readers.XArrayDatasetReader(data)
    reader.metadata.update({"featureType": "timeSeries"})
    other_cat = intake.entry.Catalog(metadata={"title": "other catalog"})
    other_cat["bar"] = reader
    other_cat.aliases["bar"] = "bar"
    other_cat.to_yaml_file(str(other / "other.catalog.yaml"))

    monkeypatch.setenv("OCEAN_SKILL_CATALOGS", str(other))
    idx2 = catalog.discover()
    assert "bar" in idx2
    assert "foo" not in idx2


def test_discover_returns_a_copy_each_time(isolated_catalogs):
    idx = catalog.discover()
    del idx["foo"]
    idx2 = catalog.discover()
    assert "foo" in idx2


def test_discover_does_not_instantiate_readers(isolated_catalogs):
    """A catalog entry naming an unimportable reader class still yields its metadata.

    ``cat[name]`` would import and instantiate the reader class (network-capable
    for ERDDAP entries); reading straight off ``cat.entries`` never does.
    """
    path = isolated_catalogs / "example.catalog.yaml"
    text = path.read_text()
    assert "reader: intake.readers.readers:XArrayDatasetReader" in text
    text = text.replace(
        "reader: intake.readers.readers:XArrayDatasetReader",
        "reader: not_a_module:Nope",
    )
    path.write_text(text)

    idx = catalog.discover()
    assert idx["foo"].metadata["featureType"] == "grid"


# -- search-path tiers ---------------------------------------------------------


def _write_catalog(directory, *, title, name, featureType="grid", filename=None):
    """Write a one-entry intake v2 catalog into ``directory``; returns its path."""
    import intake
    from intake.readers import datatypes, readers

    directory.mkdir(parents=True, exist_ok=True)
    data = datatypes.HDF5(url=str(directory / f"{name}.nc"))  # never actually read
    reader = readers.XArrayDatasetReader(data)
    reader.metadata.update(
        {"featureType": featureType, "variables": ["sea_water_temperature"]}
    )
    cat = intake.entry.Catalog(metadata={"title": title})
    cat[name] = reader
    cat.aliases[name] = name
    path = directory / (filename or f"{name}.catalog.yaml")
    cat.to_yaml_file(str(path))
    return path


def test_search_paths_tier_order(tmp_path, monkeypatch):
    shared_a, shared_b = tmp_path / "shared_a", tmp_path / "shared_b"
    added_c = tmp_path / "added_c"
    legacy, user = tmp_path / "legacy", tmp_path / "user"
    project = tmp_path / "project"
    for d in (shared_a, shared_b, added_c, legacy, user):
        d.mkdir()
    (project / "catalogs").mkdir(parents=True)
    (project / "pyproject.toml").touch()

    monkeypatch.setattr(catalog, "_added_dirs", [])
    monkeypatch.setenv(
        "OCEAN_SKILL_CATALOGS", os.pathsep.join([str(shared_a), str(shared_b)])
    )
    catalog.add_search_path(added_c)
    monkeypatch.setattr(catalog, "_legacy_user_dir", lambda: legacy)
    monkeypatch.setattr(catalog, "_user_dir", lambda: user)
    monkeypatch.chdir(project)

    packaged = Path(catalog.__file__).parent / "catalogs"
    assert catalog.search_paths() == [
        packaged,
        shared_a,
        shared_b,
        added_c,
        legacy,
        user,
        project / "catalogs",
    ]


def test_env_var_entries_are_expanded(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "_added_dirs", [])
    monkeypatch.setattr(catalog, "_legacy_user_dir", lambda: None)
    monkeypatch.setattr(catalog, "_user_dir", lambda: tmp_path / "user")
    monkeypatch.setenv("OCEAN_SKILL_CATALOGS", "~/team-catalogs")
    monkeypatch.chdir(tmp_path)

    paths = catalog.search_paths()
    assert Path.home() / "team-catalogs" in paths
    assert all("~" not in str(p) for p in paths)


def test_env_var_splits_on_os_pathsep_and_skips_empties(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "_added_dirs", [])
    monkeypatch.setattr(catalog, "_legacy_user_dir", lambda: None)
    monkeypatch.setattr(catalog, "_user_dir", lambda: tmp_path / "user")
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("OCEAN_SKILL_CATALOGS", f"{a}{os.pathsep}{os.pathsep}{b}")
    monkeypatch.chdir(tmp_path)

    paths = catalog.search_paths()
    assert a in paths and b in paths
    assert paths.index(a) < paths.index(b)


def test_add_search_path_expands_and_appends_in_call_order(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "_added_dirs", [])
    monkeypatch.setattr(catalog, "_legacy_user_dir", lambda: None)
    monkeypatch.setattr(catalog, "_user_dir", lambda: tmp_path / "user")
    env_dir = tmp_path / "env"
    monkeypatch.setenv("OCEAN_SKILL_CATALOGS", str(env_dir))
    monkeypatch.chdir(tmp_path)

    catalog.add_search_path("~/first")
    catalog.add_search_path(tmp_path / "second")

    paths = catalog.search_paths()
    first = Path.home() / "first"
    second = tmp_path / "second"
    user_dir = tmp_path / "user"
    assert paths.index(env_dir) < paths.index(first) < paths.index(second)
    assert paths.index(second) < paths.index(user_dir)
    assert all("~" not in str(p) for p in paths)


def test_add_search_path_takes_effect_without_cache_reset(isolated_catalogs, tmp_path):
    catalog.discover()  # warm the cache

    extra = tmp_path / "extra"
    _write_catalog(extra, title="extra catalog", name="bar")
    catalog.add_search_path(extra)

    assert "bar" in catalog.discover()


def test_user_dotdir_shadows_shared_env_dir(isolated_catalogs, tmp_path):
    user_dir = tmp_path / "user-catalogs"  # isolated_catalogs redirects _user_dir here
    _write_catalog(user_dir, title="user catalog", name="foo", featureType="timeSeries")

    idx = catalog.discover()
    assert idx["foo"].metadata["featureType"] == "timeSeries"

    with pytest.warns(UserWarning, match="shadows"):
        catalog.resolve("foo")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        catalog.resolve("user catalog:foo")  # qualified lookup never warns


def test_project_local_shadows_user_dir(isolated_catalogs, tmp_path):
    user_dir = tmp_path / "user-catalogs"
    _write_catalog(user_dir, title="user catalog", name="foo", featureType="timeSeries")

    (tmp_path / "pyproject.toml").touch()  # bound the project-local walk-up
    project_path = _write_catalog(
        tmp_path / "catalogs",
        title="project catalog",
        name="foo",
        featureType="profile",
    )

    idx = catalog.discover()
    assert idx["foo"].metadata["featureType"] == "profile"
    assert idx["foo"].path == project_path


def test_legacy_platformdirs_dir_is_scanned(isolated_catalogs, tmp_path):
    legacy_dir = tmp_path / "legacy-catalogs"
    _write_catalog(legacy_dir, title="legacy catalog", name="legacy_only")

    assert "legacy_only" in catalog.discover()


def test_legacy_ranks_below_dotdir(isolated_catalogs, tmp_path):
    legacy_dir = tmp_path / "legacy-catalogs"
    user_dir = tmp_path / "user-catalogs"
    _write_catalog(
        legacy_dir, title="legacy catalog", name="foo", featureType="profile"
    )
    _write_catalog(user_dir, title="user catalog", name="foo", featureType="timeSeries")

    idx = catalog.discover()
    assert idx["foo"].metadata["featureType"] == "timeSeries"


def test_legacy_dir_skipped_when_same_as_dotdir(monkeypatch, tmp_path):
    same = tmp_path / "same"
    monkeypatch.setattr(catalog, "_added_dirs", [])
    monkeypatch.delenv("OCEAN_SKILL_CATALOGS", raising=False)
    monkeypatch.setattr(catalog, "_legacy_user_dir", lambda: same)
    monkeypatch.setattr(catalog, "_user_dir", lambda: same)
    monkeypatch.chdir(tmp_path)

    assert catalog.search_paths().count(same) == 1


def test_duplicate_env_entries_appear_once(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "_added_dirs", [])
    monkeypatch.setattr(catalog, "_legacy_user_dir", lambda: None)
    monkeypatch.setattr(catalog, "_user_dir", lambda: tmp_path / "user")
    d = tmp_path / "shared"
    monkeypatch.setenv("OCEAN_SKILL_CATALOGS", os.pathsep.join([str(d), str(d)]))
    monkeypatch.chdir(tmp_path)

    assert catalog.search_paths().count(d) == 1


def test_missing_tier_dirs_are_skipped_silently(isolated_catalogs):
    # isolated_catalogs redirects the user/legacy tiers to tmp_path dirs that are
    # never created on disk; discovery must not choke on them.
    assert "foo" in catalog.discover()


def test_full_tier_precedence_end_to_end(isolated_catalogs, tmp_path):
    # shared (env) tier already has "foo"/grid, written by isolated_catalogs.
    legacy_path = _write_catalog(
        tmp_path / "legacy-catalogs",
        title="legacy catalog",
        name="foo",
        featureType="profile",
    )
    user_path = _write_catalog(
        tmp_path / "user-catalogs",
        title="user catalog",
        name="foo",
        featureType="timeSeries",
    )
    (tmp_path / "pyproject.toml").touch()  # bound the project-local walk-up
    project_path = _write_catalog(
        tmp_path / "catalogs",
        title="project catalog",
        name="foo",
        featureType="trajectory",
    )

    assert catalog.discover()["foo"].metadata["featureType"] == "trajectory"

    project_path.unlink()
    assert catalog.discover()["foo"].metadata["featureType"] == "timeSeries"

    user_path.unlink()
    assert catalog.discover()["foo"].metadata["featureType"] == "profile"

    legacy_path.unlink()
    assert catalog.discover()["foo"].metadata["featureType"] == "grid"


# -- find() filters -----------------------------------------------------------


def _fake_index(monkeypatch, entries):
    """Point discover() at a hand-built index so find() is tested in isolation."""
    from ocean_skill import catalog

    refs = {
        name: catalog.SourceRef(name=name, catalog=cat, path=None, metadata=meta)
        for name, (cat, meta) in entries.items()
    }
    monkeypatch.setattr(catalog, "discover", lambda *a, **k: refs)
    return catalog


GLOBAL = {
    "geospatial_lon_min": -180.0,
    "geospatial_lon_max": 180.0,
    "geospatial_lat_min": -90.0,
    "geospatial_lat_max": 90.0,
    "time_coverage_start": "2000-01-01",
    "time_coverage_end": "2020-01-01",
}
GULF = {
    "geospatial_lon_min": -98.0,
    "geospatial_lon_max": -80.0,
    "geospatial_lat_min": 18.0,
    "geospatial_lat_max": 31.0,
    "time_coverage_start": "2012-01-01",
    "time_coverage_end": "2012-01-21",
    "featureType": "grid",
}


@pytest.fixture
def index(monkeypatch):
    return _fake_index(
        monkeypatch,
        {
            "woa23_nitrate_month01": ("WOA23", GLOBAL),
            "woa23_nitrate_month02": ("WOA23", GLOBAL),
            "GOM_bgc": ("GOM offline run", GULF),
            "ooi-gp02hypm-rim01-02-ctdmog039": (
                "OOI Station Papa",
                {"featureType": "timeSeries"},
            ),
        },
    )


def test_name_matches_a_substring_case_insensitively(index):
    assert index.find(name="woa23") == index.find(name="WOA23")
    assert len(index.find(name="woa23")) == 2


def test_name_accepts_a_glob(index):
    assert index.find(name="woa23_nitrate_month0*") == [
        "woa23_nitrate_month01",
        "woa23_nitrate_month02",
    ]


def test_name_also_matches_the_catalog(index):
    """OOI source names are opaque ids; "papa" is only on the catalog."""
    assert index.find(name="papa") == ["ooi-gp02hypm-rim01-02-ctdmog039"]


def test_catalog_filters_only_on_the_catalog(index):
    assert len(index.find(catalog="OOI*")) == 1
    # "month01" appears in a source name but in no catalog name, so catalog= must
    # not match it -- that is the difference from the broader name= filter.
    assert index.find(catalog="month01") == []
    assert index.find(name="month01") == ["woa23_nitrate_month01"]


def test_filters_combine(index):
    assert index.find(name="papa", featureType="timeSeries") == [
        "ooi-gp02hypm-rim01-02-ctdmog039"
    ]
    assert index.find(name="papa", featureType="grid") == []


def test_bbox_tests_overlap_not_containment(index):
    """A global climatology must match a regional box, or the filter is useless."""
    gulf = index.find(bbox=(-98.0, 18.0, -80.0, 31.0))
    assert "GOM_bgc" in gulf
    assert "woa23_nitrate_month01" in gulf


def test_bbox_excludes_a_source_that_does_not_reach(index):
    """Regression: bbox was accepted and silently ignored, returning everything."""
    pacific = index.find(bbox=(-150.0, 40.0, -140.0, 50.0))
    assert "GOM_bgc" not in pacific
    assert "woa23_nitrate_month01" in pacific


def test_time_tests_overlap(index):
    assert "GOM_bgc" in index.find(time=("2012-01-05", "2012-01-10"))
    assert "GOM_bgc" not in index.find(time=("1990-01-01", "1990-02-01"))


def test_a_source_declaring_no_extent_is_kept(index):
    """An undeclared extent means unknown, not outside; keep such entries."""
    assert "ooi-gp02hypm-rim01-02-ctdmog039" in index.find(bbox=(0.0, 0.0, 1.0, 1.0))
    assert "ooi-gp02hypm-rim01-02-ctdmog039" in index.find(
        time=("1850-01-01", "1850-02-01")
    )


# -- lookup-miss messages: suggestions instead of a full dump -----------------


def test_resolve_unknown_suggests_close_match(index):
    """A typo gets a "Did you mean" nudge, not all 4+ names dumped."""
    with pytest.raises(KeyError) as exc:
        index.resolve("GOM_bgd")
    message = exc.value.args[0]
    assert "Did you mean: GOM_bgc?" in message
    assert "woa23_nitrate_month01" not in message  # no full dump


def test_resolve_unknown_alien_name_has_no_suggestion(index):
    """Nothing close enough -- no guess offered, but the self-search hint remains."""
    with pytest.raises(KeyError) as exc:
        index.resolve("zzzzqqqq")
    message = exc.value.args[0]
    assert "Did you mean" not in message
    assert "osk.find(" in message


def test_resolve_unknown_falls_back_to_substring_match(index):
    """A partial name that isn't edit-close still finds a suggestion by substring."""
    with pytest.raises(KeyError) as exc:
        index.resolve("month01")
    message = exc.value.args[0]
    assert "woa23_nitrate_month01" in message
    assert "woa23_nitrate_month02" not in message


def test_describe_unknown_suggests_across_sources_and_catalogs(index):
    with pytest.raises(KeyError) as exc:
        index.describe("GOM_bgd")
    message = exc.value.args[0]
    assert "Did you mean: GOM_bgc?" in message
    assert "Known catalogs:" in message


# -- climatologies ------------------------------------------------------------


@pytest.fixture
def climatologies(monkeypatch):
    return _fake_index(
        monkeypatch,
        {
            "woa23_nitrate_month01": (
                "WOA23",
                {"climatology": True, "climatology_period": "month01"},
            ),
            "woa23_nitrate_month07": (
                "WOA23",
                {"climatology": True, "climatology_period": "month07"},
            ),
            "woa23_nitrate_annual": (
                "WOA23",
                {"climatology": True, "climatology_period": "annual"},
            ),
            "GOM_bgc": ("GOM offline run", GULF),
        },
    )


def test_climatology_true_and_false_partition_the_index(climatologies):
    assert len(climatologies.find(climatology=True)) == 3
    assert climatologies.find(climatology=False) == ["GOM_bgc"]


@pytest.mark.parametrize(
    "spelling", ["January", "january", "jan", "01", "1", "month01"]
)
def test_a_month_climatology_is_findable_by_its_name(climatologies, spelling):
    """Catalogs record `month01`; people type "January". Both must work."""
    assert climatologies.find(climatology=spelling) == ["woa23_nitrate_month01"]


def test_periods_do_not_bleed_into_each_other(climatologies):
    assert climatologies.find(climatology="July") == ["woa23_nitrate_month07"]
    assert climatologies.find(climatology="annual") == ["woa23_nitrate_annual"]


def test_a_non_climatology_never_matches_a_period(climatologies):
    assert "GOM_bgc" not in climatologies.find(climatology="January")


def test_climatology_combines_with_other_filters(climatologies):
    assert climatologies.find(climatology=True, name="*month0*") == [
        "woa23_nitrate_month01",
        "woa23_nitrate_month07",
    ]


def test_a_glob_matches_the_whole_name_but_a_plain_string_is_a_substring(
    climatologies,
):
    """`*` means what a shell means by it; without one, the friendlier default wins."""
    assert climatologies.find(name="month01") == ["woa23_nitrate_month01"]  # substring
    assert climatologies.find(name="month0*") == []  # anchored glob: no match
    assert climatologies.find(name="*month0*") == [
        "woa23_nitrate_month01",
        "woa23_nitrate_month07",
    ]


# -- variable search ----------------------------------------------------------


NITRATE_PER_VOLUME = "mole_concentration_of_nitrate_in_sea_water"
NITRATE_PER_MASS = "moles_of_nitrate_per_unit_mass_in_sea_water"


@pytest.fixture
def mixed_spellings(monkeypatch):
    """Two sources holding the same variable under different CF names.

    Not contrived: WOA declares nitrate per unit mass while ROMS/MARBL and GLODAP
    declare it per unit volume.
    """
    return _fake_index(
        monkeypatch,
        {
            "GOM_bgc": ("GOM offline run", {"variables": [NITRATE_PER_VOLUME]}),
            "woa23_nitrate_month01": ("WOA23", {"variables": [NITRATE_PER_MASS]}),
            "GOM_his": (
                "GOM offline run",
                {"variables": ["sea_water_potential_temperature"]},
            ),
        },
    )


@pytest.mark.parametrize(
    "spelling",
    ["nitrate", "NITRATE", NITRATE_PER_VOLUME, NITRATE_PER_MASS],
)
def test_any_spelling_finds_every_source_holding_the_variable(
    mixed_spellings, spelling
):
    """Searching one exact standard_name used to miss half the relevant sources."""
    assert sorted(mixed_spellings.find(variable=spelling)) == [
        "GOM_bgc",
        "woa23_nitrate_month01",
    ]


def test_a_different_variable_does_not_match(mixed_spellings):
    assert mixed_spellings.find(variable="temperature") == ["GOM_his"]


def test_an_unknown_variable_matches_nothing(mixed_spellings):
    assert mixed_spellings.find(variable="not_a_real_variable") == []


def test_variable_combines_with_other_filters(mixed_spellings):
    assert mixed_spellings.find(variable="nitrate", catalog="WOA*") == [
        "woa23_nitrate_month01"
    ]


@pytest.fixture
def pattern_spellings(monkeypatch):
    """One source declares the canonical name, the other only a pattern spelling.

    SEANOE's CTD-export column style (`Temperature_CTD`) is not an enumerated
    alias -- only the vocabulary's regex tier recognizes it.
    """
    return _fake_index(
        monkeypatch,
        {
            "seanoe_ctd_mooring": (
                "SEANOE Hvalfjordur",
                {"variables": ["Temperature_CTD"]},
            ),
            "GOM_his": (
                "GOM offline run",
                {"variables": ["sea_water_potential_temperature"]},
            ),
        },
    )


@pytest.mark.parametrize("spelling", ["temperature", "Temperature_CTD"])
def test_a_pattern_recognized_spelling_finds_the_source(pattern_spellings, spelling):
    assert sorted(pattern_spellings.find(variable=spelling)) == [
        "GOM_his",
        "seanoe_ctd_mooring",
    ]


# -- free text ----------------------------------------------------------------


@pytest.fixture
def uneven_conventions(monkeypatch):
    """Catalogs describing the same idea three different ways.

    Real: WOA records `climatology_period: month01`, MODIS writes `jan` into the
    source name and `period: monthly_climatology`, and neither matches the other's
    spelling. Free text is what spans them.
    """
    return _fake_index(
        monkeypatch,
        {
            "modis_chl_climatology_jan": (
                "MODIS Aqua Chlorophyll",
                {"period": "monthly_climatology"},
            ),
            "modis_chl_climatology_sep": (
                "MODIS Aqua Chlorophyll",
                {"period": "monthly_climatology"},
            ),
            "modis_chl_daily_20120101": ("MODIS Aqua Chlorophyll", {}),
            "woa23_nitrate_month01": (
                "WOA23",
                {"climatology": True, "climatology_period": "month01"},
            ),
        },
    )


def test_terms_are_anded_not_ored(uneven_conventions):
    """A second word narrows; that is what it is for."""
    assert len(uneven_conventions.find(text="modis")) == 3
    assert uneven_conventions.find(text="modis jan") == ["modis_chl_climatology_jan"]


def test_free_text_reaches_metadata_not_just_names(uneven_conventions):
    """`monthly_climatology` appears only in a metadata value."""
    assert len(uneven_conventions.find(text="monthly_climatology")) == 2


def test_free_text_reaches_the_catalog_name(uneven_conventions):
    assert len(uneven_conventions.find(text="aqua")) == 3


@pytest.mark.parametrize("term", ["january", "jan", "month01"])
def test_a_month_matches_whichever_spelling_the_catalog_used(uneven_conventions, term):
    """MODIS writes `jan`, WOA writes `month01`; one term must find both."""
    found = uneven_conventions.find(text=f"chl {term}")
    assert found == ["modis_chl_climatology_jan"]
    assert uneven_conventions.find(text=f"nitrate {term}") == ["woa23_nitrate_month01"]


def test_a_bare_numeral_is_not_expanded(uneven_conventions):
    """Expanding "1" to every month spelling would match most of an index."""
    assert uneven_conventions.find(text="september") == ["modis_chl_climatology_sep"]
    assert "woa23_nitrate_month01" not in uneven_conventions.find(text="september")


def test_a_list_of_terms_is_equivalent_to_a_string(uneven_conventions):
    assert uneven_conventions.find(text=["modis", "jan"]) == uneven_conventions.find(
        text="modis jan"
    )


def test_free_text_combines_with_structured_filters(uneven_conventions):
    assert uneven_conventions.find(text="jan", climatology=True) == [
        "woa23_nitrate_month01"
    ]


def test_no_match_returns_empty(uneven_conventions):
    assert uneven_conventions.find(text="modis nitrate") == []


def test_a_climatology_is_excluded_from_a_time_search(monkeypatch):
    """Its lack of a date range is known, not unknown -- so it is not "kept".

    Returning a January climatology for a July query would be wrong, even though a
    genuinely un-probed source with no dates is deliberately kept.
    """
    cat = _fake_index(
        monkeypatch,
        {
            "jan_climatology": (
                "MODIS",
                {"climatology": True, "climatology_period": "month01"},
            ),
            "dated": (
                "MODIS",
                {
                    "time_coverage_start": "2012-07-01",
                    "time_coverage_end": "2012-07-31",
                },
            ),
            "unprobed": ("MODIS", {}),
        },
    )
    july = cat.find(time=("2012-07-01", "2012-08-01"))
    assert "dated" in july
    assert "unprobed" in july, "unknown coverage is kept"
    assert "jan_climatology" not in july, "a calendar slot is not a date range"


# -- vocabulary match report ---------------------------------------------------


@pytest.fixture
def match_report_sources(monkeypatch):
    """One source with a mix of matched and unmatched declared variables."""
    return _fake_index(
        monkeypatch,
        {
            "seanoe_ctd_mooring": (
                "SEANOE Hvalfjordur",
                {
                    "variables": [
                        "sea_water_potential_temperature",
                        "Instrument_Type",
                    ]
                },
            ),
            "GOM_bgc": (
                "GOM offline run",
                {"variables": [NITRATE_PER_VOLUME]},
            ),
        },
    )


def test_describe_source_includes_a_vocabulary_section(match_report_sources):
    text = match_report_sources.describe("seanoe_ctd_mooring")
    assert "vocabulary:" in text
    assert "temperature" in text
    assert "Instrument_Type" in text
    assert "unmatched" in text


def test_match_report_standalone_for_a_source(match_report_sources):
    text = match_report_sources.match_report("seanoe_ctd_mooring")
    assert "source: seanoe_ctd_mooring" in text
    assert "temperature" in text and "sea_water_potential_temperature" in text
    assert "Instrument_Type" in text


def test_match_report_standalone_for_a_catalog(match_report_sources):
    text = match_report_sources.match_report("GOM offline run")
    assert "catalog: GOM offline run" in text
    assert "nitrate" in text and NITRATE_PER_VOLUME in text


def test_match_report_unknown_name_raises_like_describe(match_report_sources):
    with pytest.raises(KeyError, match="neither a known source nor catalog"):
        match_report_sources.match_report("not_a_thing")


def test_describe_catalog_includes_a_vocabulary_section(isolated_catalogs):
    """"foo" declares sea_water_temperature -- a plain alias of "temperature"."""
    text = catalog.describe("example catalog")
    assert "vocabulary:" in text
    assert "temperature" in text
    assert "sea_water_temperature" in text


# -- coordinate report ---------------------------------------------------------


@pytest.fixture
def coord_report_sources(monkeypatch):
    """A healthy source, and one built before the "bottom" exclusion shipped."""
    return _fake_index(
        monkeypatch,
        {
            "seanoe_ctd_mooring": (
                "SEANOE Hvalfjordur",
                {
                    "standard_names": {
                        "Temperature_CTD": "sea_water_potential_temperature",
                        "Instrument_Type": "Instrument_Type",
                    },
                    "axes": {"X": "Longitude", "Y": "Latitude", "Z": "Depth"},
                },
            ),
            "stale_seanoe": (
                "SEANOE Hvalfjordur",
                {"standard_names": {}, "axes": {"Z": "Depth_bottom"}},
            ),
        },
    )


def test_describe_source_includes_a_coordinates_section(coord_report_sources):
    text = coord_report_sources.describe("seanoe_ctd_mooring")
    assert "coordinates:" in text
    assert "Z (vertical)" in text and "Depth" in text


def test_match_report_includes_a_coordinates_section(coord_report_sources):
    """A coordinate report whenever there's a match report -- the user's ask."""
    text = coord_report_sources.match_report("seanoe_ctd_mooring")
    assert "coordinates:" in text
    assert "Z (vertical)" in text and "Depth" in text


def test_coord_report_standalone_for_a_source(coord_report_sources):
    text = coord_report_sources.coord_report("seanoe_ctd_mooring")
    assert "source: seanoe_ctd_mooring" in text
    assert "Z (vertical)" in text and "Depth" in text


def test_coord_report_standalone_for_a_catalog(coord_report_sources):
    text = coord_report_sources.coord_report("SEANOE Hvalfjordur")
    assert "catalog: SEANOE Hvalfjordur" in text
    assert "Z (vertical)" in text and "Depth" in text


def test_coord_report_unknown_name_raises_like_describe(coord_report_sources):
    with pytest.raises(KeyError, match="neither a known source nor catalog"):
        coord_report_sources.coord_report("not_a_thing")


def test_coord_report_flags_a_stale_bottom_axis(coord_report_sources):
    """Regression guard: axes["Z"] = "Depth_bottom" is exactly the slow-comparison bug."""
    text = coord_report_sources.coord_report("stale_seanoe")
    assert "now excluded from the coordinate vocabulary" in text
    assert "Depth_bottom" in text
    assert "missing" in text and "Z" in text


def test_describe_flags_a_stale_bottom_axis_too(coord_report_sources):
    text = coord_report_sources.describe("stale_seanoe")
    assert "note:" in text
    assert "Depth_bottom" in text


def test_coord_report_for_a_dataframe():
    """A live report, straight from column names -- no catalog involved at all."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "Latitude": [1.0],
            "Longitude": [2.0],
            "Depth_bottom": [500.0],
            "Depth": [12.0],
        }
    )
    text = catalog.coord_report(df)
    assert "Z (vertical)" in text and "Depth" in text
    assert "Depth_bottom" not in text


def test_coord_report_for_an_xarray_dataset():
    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        {"temp": (("depth",), np.array([1.0, 2.0]))},
        coords={"depth": [10.0, 20.0], "depth_bottom": 500.0},
    )
    text = catalog.coord_report(ds)
    assert "Z (vertical)" in text and "depth" in text
    assert "depth_bottom" not in text


# -- overlap() ----------------------------------------------------------------------


def test_overlap_true_on_both_axes(monkeypatch):
    cat = _fake_index(monkeypatch, {"a": ("C", GULF), "b": ("C", GULF)})
    ov = cat.overlap("a", "b")
    assert (ov.space, ov.time) == (True, True)
    assert bool(ov) is True


def test_overlap_false_reports_the_offending_axis(monkeypatch):
    """The Anvil case: a station's own record predates the model's declared run."""
    far_future = {**GULF, "time_coverage_start": "2099-01-01", "time_coverage_end": "2099-02-01"}
    cat = _fake_index(monkeypatch, {"a": ("C", GULF), "b": ("C", far_future)})
    ov = cat.overlap("a", "b")
    assert (ov.space, ov.time) == (True, False)
    assert bool(ov) is False


def test_overlap_false_on_space_alone(monkeypatch):
    elsewhere = {**GULF, "geospatial_lon_min": 60.0, "geospatial_lon_max": 70.0}
    cat = _fake_index(monkeypatch, {"a": ("C", GULF), "b": ("C", elsewhere)})
    ov = cat.overlap("a", "b")
    assert (ov.space, ov.time) == (False, True)
    assert bool(ov) is False


def test_overlap_is_unknown_not_false_when_metadata_is_missing(monkeypatch):
    """An unprobed/undeclared source can't be checked -- that isn't a "no"."""
    cat = _fake_index(monkeypatch, {"a": ("C", GULF), "b": ("C", {})})
    ov = cat.overlap("a", "b")
    assert (ov.space, ov.time) == (None, None)
    assert bool(ov) is True


def test_overlap_handles_an_antimeridian_straddling_domain(monkeypatch):
    """The pac_dt_ramp stress case: a domain kept in 0-360 (see _domain_of) must
    still compare correctly against a source declared in plain +/-180.
    """
    straddling = {
        "geospatial_lon_min": 77.0,
        "geospatial_lon_max": 316.0,
        "geospatial_lat_min": 0.0,
        "geospatial_lat_max": 60.0,
    }
    inside = {  # -170 in +/-180 is 190 in 0-360, which the 77..316 arc covers
        "geospatial_lon_min": -171.0,
        "geospatial_lon_max": -169.0,
        "geospatial_lat_min": 10.0,
        "geospatial_lat_max": 20.0,
    }
    outside = {  # 0 falls in the 316..77 gap the domain does *not* cover
        "geospatial_lon_min": -1.0,
        "geospatial_lon_max": 1.0,
        "geospatial_lat_min": 10.0,
        "geospatial_lat_max": 20.0,
    }
    cat = _fake_index(
        monkeypatch,
        {"straddling": ("C", straddling), "inside": ("C", inside), "outside": ("C", outside)},
    )
    assert cat.overlap("straddling", "inside").space is True
    assert cat.overlap("straddling", "outside").space is False


def test_overlap_repr_and_bool():
    from ocean_skill.catalog import Overlap

    assert bool(Overlap(space=True, time=True)) is True
    assert bool(Overlap(space=False, time=True)) is False
    assert bool(Overlap(space=None, time=None)) is True
    assert repr(Overlap(space=True, time=False)) == "Overlap(space=yes, time=no)"
