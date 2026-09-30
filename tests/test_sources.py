"""Tests for :mod:`ocean_skill.sources` — opening a catalog entry.

Offline throughout: a local CSV entry stands in for the remote table whose reader
keywords actually matter (an ERDDAP ``constraints=`` that subsets server-side), and a
tiny local NetCDF stands in for the product whose own ``standard_name`` attributes a
catalog's ``standard_names`` rename has to outrank.
"""

from __future__ import annotations

import intake
import numpy as np
import pytest
import xarray as xr
from intake.readers import datatypes, readers

import ocean_skill as osk
from ocean_skill.build import _reader_for
from ocean_skill.catalog import SourceRef

SIGMA_THETA = "ocean_mixed_layer_thickness_defined_by_sigma_theta"


def _saved_ref(tmp_path, name, reader, metadata) -> SourceRef:
    """Save ``reader`` as a one-entry catalog and return the ref that reads it."""
    reader.metadata.update(metadata)
    cat = intake.entry.Catalog()
    cat[name] = reader
    cat.aliases[name] = name
    path = tmp_path / f"{name}.catalog.yaml"
    cat.to_yaml_file(str(path))
    return SourceRef(name=name, catalog=name, path=path, metadata=dict(reader.metadata))


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


# -- the standard_names rename ------------------------------------------------
#
# The catalog's ``standard_names`` map renames variables at read time. For a Dataset it
# also has to make each renamed variable's own ``standard_name`` *attribute* agree,
# because a product like the Holte & Talley Argo mixed-layer-depth climatology sets
# that attribute to the variable's own name and the rename alone would leave it.


def _mld_dataset() -> xr.Dataset:
    """Return a tiny Holte & Talley-shaped Dataset with self-named attributes."""
    grid = ("iMONTH", "iLAT", "iLON")
    return xr.Dataset(
        {
            "mld_dt_mean": (
                grid,
                np.ones((2, 3, 4)),
                {"standard_name": "mld_dt_mean", "units": "m"},
            ),
            "mld_da_mean": (
                grid,
                np.ones((2, 3, 4)),
                {"standard_name": "mld_da_mean", "units": "m"},
            ),
            "lat": (("iLAT",), np.arange(3.0), {"standard_name": "latitude"}),
            "lon": (("iLON",), np.arange(4.0), {"standard_name": "longitude"}),
            "month": (("iMONTH",), np.array([1, 2]), {"standard_name": "Month"}),
        }
    )


@pytest.fixture
def mld_source(tmp_path) -> SourceRef:
    """Build a Holte & Talley-shaped NetCDF entry whose catalog renames ``mld_dt_mean``.

    The entry's ``standard_names`` is what a catalog built with
    ``standard_names={"mld_dt_mean": SIGMA_THETA}`` holds: the probed map (an identity
    map here, since every attribute is self-named) with that one entry corrected.
    ``mld_da_mean`` and ``month`` are deliberately *not* renamed to anything new.
    """
    nc = tmp_path / "mld_clim.nc"
    _mld_dataset().to_netcdf(nc)
    return _saved_ref(
        tmp_path,
        "mld_clim",
        _reader_for(str(nc)),
        {
            "featureType": "grid",
            "standard_names": {
                "mld_dt_mean": SIGMA_THETA,
                "mld_da_mean": "mld_da_mean",
                "lat": "latitude",
                "lon": "longitude",
            },
        },
    )


def test_a_renamed_variable_takes_the_catalogs_standard_name(mld_source):
    """The rename would otherwise leave a self-named attribute behind.

    Downstream code reads ``attrs["standard_name"] or name``, so a variable renamed to
    its CF name but still saying ``standard_name="mld_dt_mean"`` would be looked up,
    and reported in a mismatch warning, under the very name the catalog just replaced.
    """
    ds = osk.read(mld_source)

    assert SIGMA_THETA in ds
    assert "mld_dt_mean" not in ds
    assert ds[SIGMA_THETA].attrs["standard_name"] == SIGMA_THETA  # overwrote the bogus
    assert ds[SIGMA_THETA].attrs["units"] == "m"  # everything else rides along
    assert ds["latitude"].attrs["standard_name"] == "latitude"


def test_a_renamed_variable_with_no_attribute_gets_one(tmp_path):
    """A name that came from a name_map or the vocabulary has no attribute behind it.

    GLODAP-style files carry no ``standard_name`` at all, so their catalog's names were
    never read off an attribute; stamping puts the catalog's name where downstream
    lookups (``attrs["standard_name"] or name``) will see it as well.
    """
    nitrate = "mole_concentration_of_nitrate_in_sea_water"
    nc = tmp_path / "glodap.nc"
    xr.Dataset({"NO3": (("x",), np.ones(3))}, coords={"x": [0, 1, 2]}).to_netcdf(nc)
    ref = _saved_ref(
        tmp_path,
        "glodap",
        _reader_for(str(nc)),
        {"featureType": "grid", "standard_names": {"NO3": nitrate}},
    )

    ds = osk.read(ref)

    assert "NO3" not in ds
    assert ds[nitrate].attrs == {"standard_name": nitrate}


def test_variables_the_catalog_leaves_alone_keep_their_own_attrs(mld_source):
    """Only renamed variables are stamped; nothing else about the file is overruled."""
    ds = osk.read(mld_source)

    # not in the map at all: "Month" must not become "month" (its name)
    assert ds["month"].attrs == {"standard_name": "Month"}
    # in the map, but its target already exists, so it is never renamed
    assert ds["mld_da_mean"].attrs == {"standard_name": "mld_da_mean", "units": "m"}


def test_the_dataset_the_reader_returned_is_never_written_into(mld_source, monkeypatch):
    """Stamping builds new variables; it does not edit the reader's own object.

    The reader may hand back something it still holds, and the open memo serves
    shallow copies precisely so that one caller's edits never reach another's -- an
    in-place ``attrs`` write on the pre-rename object would leak the catalog's name
    into everything else sharing those variables.
    """
    opened = _mld_dataset()
    monkeypatch.setattr(
        readers.XArrayDatasetReader, "read", lambda self, *args, **kwargs: opened
    )

    ds = osk.read(mld_source)

    assert ds[SIGMA_THETA].attrs["standard_name"] == SIGMA_THETA
    assert "mld_dt_mean" in opened  # still under its own name ...
    assert opened["mld_dt_mean"].attrs["standard_name"] == "mld_dt_mean"  # ... and attr
    assert opened["lat"].attrs["standard_name"] == "latitude"
    assert SIGMA_THETA not in opened


def test_a_renamed_coordinate_is_stamped_and_stays_a_coordinate(tmp_path):
    """A rename target can be a coordinate, and a coordinate has to stay one.

    A dimension coordinate is the awkward case for anything that rebuilds a variable
    by hand: its index has to come along, and it must not turn into a data variable.
    """
    nc = tmp_path / "sst.nc"
    xr.Dataset(
        {"sst": (("lat", "lon"), np.ones((3, 4)), {"standard_name": "sst"})},
        coords={
            "lat": (
                "lat",
                [10.0, 20.0, 30.0],
                {"standard_name": "lat", "units": "degrees_north"},
            ),
            "lon": ("lon", [0.0, 90.0, 180.0, 270.0]),
        },
    ).to_netcdf(nc)
    ref = _saved_ref(
        tmp_path,
        "sst",
        _reader_for(str(nc)),
        {
            "featureType": "grid",
            "standard_names": {"sst": "sea_surface_temperature", "lat": "latitude"},
        },
    )

    ds = osk.read(ref)

    assert "latitude" in ds.coords
    assert "latitude" in ds.indexes  # still a dimension coordinate, index intact
    assert ds["latitude"].attrs == {
        "standard_name": "latitude",
        "units": "degrees_north",
    }
    assert "sea_surface_temperature" in ds.data_vars
    assert ds["sea_surface_temperature"].dims == ("latitude", "lon")
    assert ds["sea_surface_temperature"].attrs["standard_name"] == (
        "sea_surface_temperature"
    )
    assert "standard_name" not in ds["lon"].attrs  # not renamed, so not stamped


@pytest.fixture
def renamed_csv_source(tmp_path) -> SourceRef:
    """Build a CSV entry whose catalog renames its ``temp`` column."""
    csv = tmp_path / "station.csv"
    csv.write_text("time,temp\n2015-01-01,1\n2015-01-02,2\n2015-01-03,3\n")
    return _saved_ref(
        tmp_path,
        "station",
        readers.PandasCSV(datatypes.CSV(url=str(csv))),
        {
            "featureType": "timeSeries",
            "axes": {"T": "time"},
            "standard_names": {"temp": "sea_water_temperature"},
        },
    )


def test_a_tabular_source_still_renames_its_columns(renamed_csv_source):
    """A DataFrame's columns have no attrs, so the rename is all there is to do."""
    frame = osk.read(renamed_csv_source)

    assert list(frame.columns) == ["time", "sea_water_temperature"]
    assert frame["sea_water_temperature"].tolist() == [1, 2, 3]
    assert "standard_name" not in frame.attrs  # nothing is stamped onto the frame


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
