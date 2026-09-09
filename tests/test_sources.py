"""Tests for :mod:`ocean_skill.sources` — opening a catalog entry.

Offline throughout: a local CSV entry stands in for the remote table whose reader
keywords actually matter (an ERDDAP ``constraints=`` that subsets server-side).
"""

from __future__ import annotations

import intake
import pytest
from intake.readers import datatypes, readers

import ocean_skill as osk
from ocean_skill.catalog import SourceRef


@pytest.fixture
def csv_source(tmp_path) -> SourceRef:
    """Build a three-row CSV entry, referenced directly rather than discovered."""
    csv = tmp_path / "station.csv"
    csv.write_text("time,temp\n2015-01-01,1\n2015-01-02,2\n2015-01-03,3\n")
    reader = readers.PandasCSV(datatypes.CSV(url=str(csv)))
    reader.metadata.update({"featureType": "timeSeries", "axes": {"T": "time"}})
    cat = intake.entry.Catalog()
    cat["station"] = reader
    cat.aliases["station"] = "station"
    path = tmp_path / "station.catalog.yaml"
    cat.to_yaml_file(str(path))
    return SourceRef(
        name="station",
        catalog="station",
        path=path,
        metadata=dict(reader.metadata),
    )


def test_read_returns_the_whole_entry_by_default(csv_source):
    assert len(osk.read(csv_source)) == 3


def test_reader_keywords_reach_the_reader(csv_source):
    """``read`` used to declare ``**kwargs`` and discard them.

    The keyword that earns this is ``constraints=`` on an ERDDAP table: a mooring's
    whole record is a large download, and a later ``select={"time": ...}`` cannot subset
    it server-side. Silently ignoring the request meant paying for the full read anyway
    while believing it had been narrowed.
    """
    assert len(osk.read(csv_source, nrows=2)) == 2


# -- turning a select into server-side constraints ----------------------------

#: An ERDDAP tabledap entry's metadata, as `intake_erddap` writes it into a catalog.
TABLE = {
    "tabledap": "https://example.org/erddap/tabledap/mooring",
    "axes": {"T": "time (UTC)"},
}


@pytest.mark.parametrize(
    ("select", "expected"),
    [
        # a slice is the case the OOI moorings are read through
        (
            {"time": slice("2015-01-01", "2017-01-01")},
            {"time>=": "2015-01-01T00:00:00Z", "time<=": "2017-01-01T23:59:59Z"},
        ),
        # a partial date is a span, not an instant: all of January
        (
            {"time": "2012-01"},
            {"time>=": "2012-01-01T00:00:00Z", "time<=": "2012-01-31T23:59:59Z"},
        ),
        # the YAML-friendly spelling of a slice
        (
            {"time": {"min": "2015", "max": "2015"}},
            {"time>=": "2015-01-01T00:00:00Z", "time<=": "2015-12-31T23:59:59Z"},
        ),
        # an open end constrains only the end it names
        ({"time": slice("2015-01-01", None)}, {"time>=": "2015-01-01T00:00:00Z"}),
        ({"time": slice(None, "2015-01-01")}, {"time<=": "2015-01-01T23:59:59Z"}),
        # other spellings of the same axis, including the entry's own column name
        (
            {"T": "2012"},
            {"time>=": "2012-01-01T00:00:00Z", "time<=": "2012-12-31T23:59:59Z"},
        ),
        (
            {"time (UTC)": "2012"},
            {"time>=": "2012-01-01T00:00:00Z", "time<=": "2012-12-31T23:59:59Z"},
        ),
    ],
)
def test_a_time_select_becomes_constraints(select, expected):
    from ocean_skill.sources import erddap_constraints

    assert erddap_constraints(TABLE, select) == expected


@pytest.mark.parametrize(
    "select",
    [
        None,
        {},
        {"depth": 0},  # pd.Timestamp(0) is a perfectly good 1970; this is not a time
        {"depth": "surface"},
        {"lat": slice(20, 30)},
        {"time": "not a date"},
    ],
)
def test_a_select_naming_no_time_constrains_nothing(select):
    """Returning ``{}`` costs a bigger download; returning a wrong bound loses data."""
    from ocean_skill.sources import erddap_constraints

    assert erddap_constraints(TABLE, select) == {}


def test_a_transect_key_alongside_time_still_constrains_only_time():
    """A select={'transect': ...} key means nothing to ERDDAP -- it names a model
    grid dimension or a lon/lat path, neither of which a tabledap read narrows
    server-side. It should be ignored outright, not raise and not somehow leak
    into the constraints dict.
    """
    from ocean_skill.sources import erddap_constraints

    select = {"time": "2012", "transect": {"waypoints": [[-95.0, 24.0], [-94.0, 25.0]]}}
    assert erddap_constraints(TABLE, select) == {
        "time>=": "2012-01-01T00:00:00Z",
        "time<=": "2012-12-31T23:59:59Z",
    }


def test_only_tabledap_entries_are_constrained():
    """Every other source is opened lazily and narrows itself; see erddap_constraints.

    The gate is what keeps this from reaching the gridded catalogs — including ERDDAP's
    own griddap entries, which reach us as plain OPeNDAP URLs.
    """
    from ocean_skill.sources import erddap_constraints

    select = {"time": "2015"}
    assert erddap_constraints({}, select) == {}
    assert (
        erddap_constraints({"griddap": "https://example.org/x", "tabledap": ""}, select)
        == {}
    )


def test_a_derived_time_window_constrains_the_read():
    """The skill-map case: the window comes from the test lane, not from the caller."""
    import numpy as np

    from ocean_skill.sources import erddap_constraints

    window = (np.datetime64("2015-06-01"), np.datetime64("2015-08-31"))
    assert erddap_constraints(TABLE, None, window) == {
        "time>=": "2015-06-01T00:00:00Z",
        "time<=": "2015-08-31T00:00:00Z",
    }


def test_select_and_window_take_the_tighter_bound():
    """Both mean "only this much"; honouring the looser one would download the rest."""
    import numpy as np

    from ocean_skill.sources import erddap_constraints

    window = (np.datetime64("2015-06-01"), np.datetime64("2016-06-01"))
    assert erddap_constraints(
        TABLE, {"time": slice("2015-01-01", "2015-12-31")}, window
    ) == {
        "time>=": "2015-06-01T00:00:00Z",
        "time<=": "2015-12-31T23:59:59Z",
    }


def test_a_tz_aware_bound_is_converted_rather_than_refused():
    """`osk.read` decodes ERDDAP times as UTC-aware, so one can come back around."""
    import pandas as pd

    from ocean_skill.sources import erddap_constraints

    aware = pd.Timestamp("2015-01-01T06:00", tz="US/Central")
    assert erddap_constraints(TABLE, {"time": slice(aware, None)}) == {
        "time>=": "2015-01-01T12:00:00Z"
    }


# -- the in-process open memo -------------------------------------------------
#
# ``fresh_open_cache`` (tests/conftest.py) clears ``osk.read``'s memo around every
# test; ``isolated_cache`` (also autouse) already leaves caching *enabled* (just
# relocated to a temp dir), which is what lets this memo do anything in these tests
# at all.


def test_repeat_reads_reuse_one_open(csv_source, monkeypatch):
    """A second read of the same, unchanged entry never reopens the catalog file.

    The regression this guards: a station-fan comparison opens the *same* test
    source once per reference -- for a large gridded model, reopening +
    re-standardizing it per station is the dominant cost of an otherwise-cheap
    per-station point read.
    """
    import intake

    opens = []
    real_from_yaml_file = intake.from_yaml_file

    def spy(*args, **kwargs):
        opens.append(args)
        return real_from_yaml_file(*args, **kwargs)

    monkeypatch.setattr(intake, "from_yaml_file", spy)

    osk.read(csv_source)
    osk.read(csv_source)
    osk.read(csv_source)

    assert len(opens) == 1


def test_cached_reads_have_independent_attrs(csv_source):
    """Two reads share the same underlying object, but not the same attrs dict.

    A caller downstream (``_prepare``'s ``da.attrs["actual_depth"] = ...``, e.g.)
    writes into the result's attrs after extracting one variable -- if the memo
    handed out the identical object twice, that write would leak into every other
    caller holding "the same" cached read.
    """
    first = osk.read(csv_source)
    second = osk.read(csv_source)
    assert first is not second
    first.attrs["mutated_by"] = "first caller"
    assert "mutated_by" not in second.attrs


def test_editing_the_catalog_file_forces_a_reopen(csv_source, monkeypatch, tmp_path):
    """A rewritten catalog file (new mtime/size) is never served from the old memo."""
    import time

    import intake

    opens = []
    real_from_yaml_file = intake.from_yaml_file

    def spy(*args, **kwargs):
        opens.append(args)
        return real_from_yaml_file(*args, **kwargs)

    monkeypatch.setattr(intake, "from_yaml_file", spy)

    osk.read(csv_source)
    assert len(opens) == 1

    # Rewrite the underlying CSV with a fourth row, then touch the catalog file
    # itself (its own mtime/size is the memo key, not the CSV's) so the read that
    # follows is not served from before this edit.
    csv_path = tmp_path / "station.csv"
    csv_path.write_text(
        "time,temp\n2015-01-01,1\n2015-01-02,2\n2015-01-03,3\n2015-01-04,4\n"
    )
    time.sleep(0.01)
    csv_source.path.touch()

    result = osk.read(csv_source)
    assert len(opens) == 2
    assert len(result) == 4


def test_cache_clear_empties_the_open_memo(csv_source, monkeypatch):
    """``osk.cache.clear()`` also flushes the read memo, not just the on-disk one."""
    import intake

    from ocean_skill import cache

    opens = []
    real_from_yaml_file = intake.from_yaml_file

    def spy(*args, **kwargs):
        opens.append(args)
        return real_from_yaml_file(*args, **kwargs)

    monkeypatch.setattr(intake, "from_yaml_file", spy)

    osk.read(csv_source)
    osk.read(csv_source)
    assert len(opens) == 1

    cache.clear()

    osk.read(csv_source)
    assert len(opens) == 2


def test_cache_disable_bypasses_the_open_memo(csv_source, monkeypatch):
    """``osk.cache.disable()`` also turns off the read memo, like the on-disk one."""
    import intake

    from ocean_skill import cache

    opens = []
    real_from_yaml_file = intake.from_yaml_file

    def spy(*args, **kwargs):
        opens.append(args)
        return real_from_yaml_file(*args, **kwargs)

    monkeypatch.setattr(intake, "from_yaml_file", spy)

    cache.disable()
    try:
        osk.read(csv_source)
        osk.read(csv_source)
    finally:
        cache.enable()

    assert len(opens) == 2
