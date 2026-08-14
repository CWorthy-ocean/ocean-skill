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
