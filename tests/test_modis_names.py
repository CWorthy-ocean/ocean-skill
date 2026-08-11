"""Tests for reading OB.DAAC Level-3 filenames into catalog-entry names.

The cases are the real filename shapes from NASA's MODIS Aqua OPeNDAP tree. What
matters is that each temporal code is *interpreted*, not echoed: an `MC` file
spanning 2003-2022 means "every January in those years", not "January 2003 through
January 2022", and getting that backwards would mislabel a whole catalog.
"""

from __future__ import annotations

import datetime

import pytest

from ocean_skill.obs.modis import catalog_entry, catalog_metadata, nickname, parse

BASE = "http://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/2003/0101/"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "AQUA_MODIS.20030101.L3m.DAY.CHL.chlor_a.9km.nc",
            "MODIS Aqua chlorophyll a daily 2003-01-01 9km",
        ),
        (
            "AQUA_MODIS.20030101_20030108.L3m.8D.CHL.chlor_a.9km.nc",
            "MODIS Aqua chlorophyll a 8-day 2003-01-01 to 2003-01-08 9km",
        ),
        (
            "AQUA_MODIS.20030101_20030131.L3m.MO.CHL.chlor_a.9km.nc",
            "MODIS Aqua chlorophyll a January 2003 9km",
        ),
        (
            "AQUA_MODIS.20030101_20030201.L3m.R32.CHL.chlor_a.9km.nc",
            "MODIS Aqua chlorophyll a 32-day rolling 2003-01-01 to 2003-02-01 9km",
        ),
        (
            "AQUA_MODIS.20030101_20031231.L3m.YR.CHL.chlor_a.9km.nc",
            "MODIS Aqua chlorophyll a annual 2003 9km",
        ),
        (
            "AQUA_MODIS.20030101_20220131.L3m.MC.CHL.chlor_a.9km.nc",
            "MODIS Aqua chlorophyll a January climatology 2003-2022 9km",
        ),
        (
            "AQUA_MODIS.20030201_20260228.L3m.MC.CHL.chlor_a.4km.nc",
            "MODIS Aqua chlorophyll a February climatology 2003-2026 4km",
        ),
        (
            "TERRA_MODIS.20030101.L3m.DAY.NSST.sst.4km.nc",
            "MODIS Terra night sea surface temperature daily 2003-01-01 4km",
        ),
        (
            "AQUA_MODIS.20030601_20260630.L3m.SCSU.CHL.chlor_a.4km.nc",
            "MODIS Aqua chlorophyll a summer climatology 2003-2026 4km",
        ),
    ],
)
def test_nickname_of_each_temporal_code(filename, expected):
    assert nickname(BASE + filename) == expected


def test_a_climatology_reads_its_range_as_years_not_a_span():
    """`MC` 2003-2022 is every January in those years, not 19 continuous years."""
    got = parse("AQUA_MODIS.20030101_20220131.L3m.MC.CHL.chlor_a.9km.nc")
    assert got["climatology"] == "month"
    assert got["start_date"] == datetime.date(2003, 1, 1)
    assert got["end_date"] == datetime.date(2022, 1, 31)


def test_an_unlisted_product_still_yields_a_usable_name():
    """New products appear; an unknown one must degrade, not break the sweep."""
    got = nickname("AQUA_MODIS.20030101.L3m.DAY.NEW.newvar.9km.nc")
    assert got == "MODIS Aqua NEW.newvar daily 2003-01-01 9km"


@pytest.mark.parametrize(
    "other", ["palette.nc", "checksums.txt", "", "AQUA_MODIS.nc", "some/dir/"]
)
def test_a_non_matching_name_returns_none(other):
    """Directory sweeps contain palettes and checksums; a miss is ordinary."""
    assert nickname(other) is None
    assert parse(other) is None


def test_a_bare_filename_and_a_full_url_agree():
    name = "AQUA_MODIS.20030101.L3m.DAY.CHL.chlor_a.9km.nc"
    assert nickname(name) == nickname(BASE + name)


def test_nicknames_are_unique_across_a_realistic_sweep():
    """They become catalog entry names, so collisions would silently drop entries."""
    files = [
        f"AQUA_MODIS.2003{m:02d}01_2026{m:02d}28.L3m.MC.CHL.chlor_a.4km.nc"
        for m in range(1, 13)
    ]
    names = [nickname(f) for f in files]
    assert len(set(names)) == len(names)


# -- rolling vs fixed ---------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected_label"),
    [("R3", "3-day rolling"), ("R8", "8-day rolling"), ("R32", "32-day rolling")],
)
def test_any_r_code_is_understood_as_rolling(code, expected_label):
    """Derived by rule, so a new R-code needs no table entry."""
    got = parse(f"AQUA_MODIS.20030101_20030201.L3m.{code}.CHL.chlor_a.9km.nc")
    assert got["rolling"] is True
    assert got["period_label"] == expected_label


def test_eight_day_is_a_fixed_bin_not_a_rolling_average():
    """NASA's 8-day products are fixed, non-overlapping bins restarting each year.

    They share the *formatting* need of a rolling product (both endpoints matter)
    but are not moving averages, so the two properties are tracked separately.
    """
    got = parse("AQUA_MODIS.20030101_20030108.L3m.8D.CHL.chlor_a.9km.nc")
    assert got["window"] is True
    assert got["rolling"] is False
    assert "rolling" not in nickname(
        "AQUA_MODIS.20030101_20030108.L3m.8D.CHL.chlor_a.9km.nc"
    )


@pytest.mark.parametrize("code", ["DAY", "MO", "YR", "MC"])
def test_calendar_periods_are_not_rolling(code):
    got = parse(f"AQUA_MODIS.20030101_20030131.L3m.{code}.CHL.chlor_a.9km.nc")
    assert got["rolling"] is False


# -- catalog metadata from the filename ---------------------------------------


def test_a_monthly_climatology_is_flagged_as_one():
    """The file itself says nothing; only the `MC` in the name does.

    Without this, MODIS climatologies were invisible to `find(climatology=...)`
    while WOA's were not — the same concept recorded two different ways.
    """
    md = catalog_metadata("AQUA_MODIS.20030101_20220131.L3m.MC.CHL.chlor_a.4km.nc")
    assert md["climatology"] is True
    assert md["climatology_period"] == "month01"


def test_a_climatology_carries_no_time_coverage():
    """Its global attrs span 2003-2022, which would answer a July 2012 query.

    A January climatology is a calendar slot, not two decades of data, so the
    averaging span is recorded separately and time_coverage is cleared.
    """
    md = catalog_metadata("AQUA_MODIS.20030101_20220131.L3m.MC.CHL.chlor_a.4km.nc")
    assert md["time_coverage_start"] is None
    assert md["time_coverage_end"] is None
    assert md["climatology_span_start"] == "2003-01-01"
    assert md["climatology_span_end"] == "2022-01-31"


def test_a_dated_product_keeps_its_time_coverage():
    md = catalog_metadata("AQUA_MODIS.20120101.L3m.DAY.CHL.chlor_a.4km.nc")
    assert md["time_coverage_start"] == "2012-01-01"
    assert md.get("climatology") is None


def test_a_seasonal_climatology_names_its_season():
    md = catalog_metadata("AQUA_MODIS.20030601_20260630.L3m.SCSU.CHL.chlor_a.4km.nc")
    assert md["climatology"] is True
    assert md["climatology_period"] == "summer"


def test_catalog_entry_is_a_ready_build_catalog_spec():
    url = "AQUA_MODIS.20030101_20220131.L3m.MC.CHL.chlor_a.4km.nc"
    entry = catalog_entry(url)
    assert entry["url"] == url
    assert entry["climatology_period"] == "month01"


def test_an_unrecognized_name_contributes_nothing():
    """Safe to map over a directory listing full of palettes and checksums."""
    assert catalog_metadata("palette.nc") == {}
    assert catalog_entry("palette.nc") == {"url": "palette.nc"}
