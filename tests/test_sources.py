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
