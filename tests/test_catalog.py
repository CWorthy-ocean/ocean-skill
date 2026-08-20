"""Tests for catalog auto-discovery and name resolution."""

from __future__ import annotations

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
