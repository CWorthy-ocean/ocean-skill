"""Tests for how a catalog entry says a source is *read*: zones, split times, depths.

Four declarations change the values a read produces, and each is stated per entry
(through :func:`ocean_skill.build.add_source`, ``add_sources`` or
``add_erddap_source``), validated at build time, and applied by the read path
(:mod:`ocean_skill.sources`, :mod:`ocean_skill.tabular`):

* ``time_zone`` / ``utc_offset_h`` -- what a *naive* timestamp means (it used to be UTC,
  silently, so an Alaska logger's ``12:00`` AKDT landed eight hours early);
* ``time_columns`` / ``time_format`` -- a time split over ``Date`` and ``Time`` columns;
* ``axes["Z"]`` -- which column is the instrument depth, declared rather than guessed;
* ``depth_convention`` -- where depth is measured from and which way it counts, with
  what the probe can read off the vertical coordinate kept apart under ``inferred``.

Everything is offline: a CSV written to ``tmp_path`` stands in for a table, a tiny
NetCDF for a gridded product, and the ERDDAP reader is swapped for the stand-in
:mod:`tests.test_qc` builds over a CSV.
"""

from __future__ import annotations

import warnings

import intake
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from intake.readers import datatypes, readers

import ocean_skill as osk
from ocean_skill import build, depth_convention, sources, tabular
from ocean_skill.catalog import SourceRef

# -- building entries ------------------------------------------------------------------


def _write_csv(tmp_path, text, filename="station.csv"):
    path = tmp_path / filename
    path.write_text(text)
    return path


def _csv_reader(tmp_path, text, filename="station.csv"):
    csv = _write_csv(tmp_path, text, filename)
    return readers.PandasCSV(datatypes.CSV(url=str(csv)))


def _refs(cat, tmp_path, stem="cat") -> dict[str, SourceRef]:
    """Save ``cat`` and return a ref per entry, with the metadata a YAML reload gives.

    The round trip is the point: what ``add_source`` stored has to survive being written
    as YAML and read back (floats, nested ``depth_convention`` dicts, lists), because
    that is the metadata a real read is handed.
    """
    path = tmp_path / f"{stem}.yaml"
    cat.to_yaml_file(str(path))
    loaded = intake.from_yaml_file(str(path))
    return {
        name: SourceRef(
            name=name, catalog=stem, path=path, metadata=dict(loaded[name].metadata)
        )
        for name in list(cat)
    }


def _entry(tmp_path, text, *, name="station", **metadata) -> SourceRef:
    """Add a CSV table to a new catalog through ``add_source``; return its ref."""
    cat = build.new_catalog(title="t")
    reader = _csv_reader(tmp_path, text, f"{name}.csv")
    build.add_source(cat, name, reader=reader, name_map=None, **metadata)
    return _refs(cat, tmp_path, name)[name]


def _series_csv(start="2024-07-01 12:00", n=3, *, time_column="time"):
    """Return a fixed station's hourly temperature, with naive timestamps."""
    times = pd.date_range(start, periods=n, freq="h").strftime("%Y-%m-%d %H:%M")
    rows = "".join(f"{t},-150.0,61.0,5.0,{8.0 + i}\n" for i, t in enumerate(times))
    return f"{time_column},lon,lat,depth (m),temp (degC)\n{rows}"


def _as_dataset(ref):
    """Read ``ref`` the way a comparison does: ``osk.read`` then ``to_dataset``."""
    return tabular.to_dataset(osk.read(ref), ref.metadata)


# -- time zones: naive timestamps ------------------------------------------------------


@pytest.mark.parametrize(
    ("stamp", "declared", "utc", "label"),
    [
        # the "+8 h" demo: a naive AKDT stamp is eight hours behind UTC in July ...
        (
            "2024-07-01 12:00",
            {"time_zone": "America/Anchorage"},
            "2024-07-01T20:00",
            "America/Anchorage",
        ),
        # ... and nine in January: the IANA zone follows daylight saving
        (
            "2024-01-15 12:00",
            {"time_zone": "America/Anchorage"},
            "2024-01-15T21:00",
            "America/Anchorage",
        ),
        # a logger that kept local *standard* time all year is the fixed offset: -9
        # shifts July by nine hours too, where the IANA zone would shift it by eight
        ("2024-07-01 12:00", {"utc_offset_h": -9}, "2024-07-01T21:00", "UTC-09:00"),
        ("2024-07-01 12:00", {"utc_offset_h": -8}, "2024-07-01T20:00", "UTC-08:00"),
        # both signs: UTC = stamp - utc_offset_h, so east of Greenwich goes *back*
        ("2024-07-01 12:00", {"utc_offset_h": 5.5}, "2024-07-01T06:30", "UTC+05:30"),
        # nothing declared: taken as UTC, exactly as before
        ("2024-07-01 12:00", {}, "2024-07-01T12:00", None),
    ],
)
def test_a_declared_zone_reads_naive_local_stamps_as_utc(
    tmp_path, stamp, declared, utc, label
):
    ref = _entry(tmp_path, _series_csv(stamp), **declared)

    frame = osk.read(ref)
    assert frame["time"].iloc[0] == pd.Timestamp(utc, tz="UTC")  # sources.read decodes

    ds = tabular.to_dataset(frame, ref.metadata)
    assert ds["time"].values[0] == np.datetime64(utc)
    assert ds["time"].values[1] - ds["time"].values[0] == np.timedelta64(1, "h")
    assert ds["time"].attrs["time_zone"] == "UTC"  # the values are UTC either way
    assert ds["time"].attrs.get("source_time_zone") == label  # and the source's clock


def test_a_table_given_straight_to_to_dataset_is_shifted_once(tmp_path):
    """No ``osk.read`` in front of it: ``to_dataset`` decodes the raw column itself."""
    frame = pd.read_csv(_write_csv(tmp_path, _series_csv(), "raw.csv"))
    ds = tabular.to_dataset(frame, {"utc_offset_h": -8})
    assert ds["time"].values[0] == np.datetime64("2024-07-01T20:00")


@pytest.mark.parametrize(
    ("declared", "first", "last"),
    [
        ({}, "2024-07-01", "2024-07-01"),
        # 20:00-23:00 AKDT is 04:00-07:00 UTC the next day: coverage is in UTC
        ({"time_zone": "America/Anchorage"}, "2024-07-02", "2024-07-02"),
        ({"utc_offset_h": -8}, "2024-07-02", "2024-07-02"),
    ],
)
def test_the_probed_time_coverage_is_in_utc(tmp_path, declared, first, last):
    ref = _entry(tmp_path, _series_csv("2024-07-01 20:00", 4), **declared)
    assert ref.metadata["time_coverage_start"] == first
    assert ref.metadata["time_coverage_end"] == last


def test_iso_stamps_with_their_own_zone_are_never_shifted(tmp_path):
    """A ``Z`` (or an offset) states what the stamp is; a zone is for naive ones."""
    text = "time,lon,lat,depth (m),temp (degC)\n" + "".join(
        f"2024-07-01T{h}:00:00{tail},-150.0,61.0,5.0,8.0\n"
        for h, tail in (("12", "Z"), ("13", "Z"))
    )
    with pytest.warns(UserWarning, match="own UTC offset"):
        ref = _entry(tmp_path, text, time_zone="America/Anchorage")
    with pytest.warns(UserWarning, match="own UTC offset"):
        frame = osk.read(ref)
    assert frame["time"].iloc[0] == pd.Timestamp("2024-07-01T12:00", tz="UTC")

    # to_dataset decodes the same column again, and must neither shift it a second time
    # nor repeat a warning about a conflict osk.read has already reported
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        ds = tabular.to_dataset(frame, ref.metadata)
    assert not [w for w in seen if "own UTC offset" in str(w.message)]
    assert ds["time"].values[0] == np.datetime64("2024-07-01T12:00")


def test_iso_stamps_with_an_offset_that_agrees_with_the_zone_are_converted(tmp_path):
    text = "time,lon,lat,depth (m),temp (degC)\n" + "".join(
        f"2024-07-01T{h}:00:00-08:00,-150.0,61.0,5.0,8.0\n" for h in ("12", "13")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # AKDT is -08:00 in July: no conflict to report
        ref = _entry(tmp_path, text, time_zone="America/Anchorage")
        ds = _as_dataset(ref)
    assert ds["time"].values[0] == np.datetime64("2024-07-01T20:00")


def test_the_cf_units_branch_reads_a_zone_free_reference_as_local(tmp_path):
    """``hours since <date>`` with no zone counts local hours; a ``Z`` is final."""
    local = pd.Series([12, 13])
    meta = {"utc_offset_h": -8}
    decoded = tabular.decode_time_column(
        local, "Time[hours_since_2024-07-01T00:00:00]", meta
    )
    assert decoded.iloc[0] == pd.Timestamp("2024-07-01T20:00", tz="UTC")

    explicit = tabular.decode_time_column(
        local, "Time[hours_since_2024-07-01T00:00:00Z]", meta
    )
    assert explicit.iloc[0] == pd.Timestamp("2024-07-01T12:00", tz="UTC")

    undeclared = tabular.decode_time_column(
        local, "Time[hours_since_2024-07-01T00:00:00]", {}
    )
    assert undeclared.iloc[0] == pd.Timestamp("2024-07-01T12:00", tz="UTC")


def test_an_already_tz_aware_column_is_not_localised_again():
    """``sources.read`` has converted it; ``to_dataset`` decodes it a second time."""
    aware = pd.Series(pd.to_datetime(["2024-07-01T20:00:00Z"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = tabular.decode_time_column(
            aware, "time", {"time_zone": "America/Anchorage"}
        )
    assert out.iloc[0] == pd.Timestamp("2024-07-01T20:00", tz="UTC")


def test_an_invalid_declaration_is_refused_when_a_column_is_decoded():
    with pytest.raises(ValueError, match="time_zone"):
        tabular.decode_time_column(
            pd.Series(["2024-07-01 12:00"]), "time", {"time_zone": "Mars/Phobos"}
        )


# -- time zones: one catalog, one convention per entry ---------------------------------


def test_add_sources_gives_each_entry_its_own_convention(tmp_path):
    """Visits logged in different conventions, plus one that says nothing."""
    cat = build.new_catalog(title="t")
    stamp = "2024-07-01 12:00"
    specs = {
        "akdt": {
            "reader": _csv_reader(tmp_path, _series_csv(stamp), "a.csv"),
            "utc_offset_h": -8,
        },
        "akst": {
            "reader": _csv_reader(tmp_path, _series_csv(stamp), "b.csv"),
            "utc_offset_h": -9,
        },
        "india": {
            "reader": _csv_reader(tmp_path, _series_csv(stamp), "c.csv"),
            "utc_offset_h": 5.5,
        },
        "anchorage_winter": {
            "reader": _csv_reader(tmp_path, _series_csv("2024-01-15 12:00"), "d.csv"),
            "time_zone": "America/Anchorage",
        },
        "undeclared": {"reader": _csv_reader(tmp_path, _series_csv(stamp), "e.csv")},
    }
    build.add_sources(cat, specs, name_map=None)
    refs = _refs(cat, tmp_path)

    first = {name: _as_dataset(ref)["time"].values[0] for name, ref in refs.items()}
    assert first == {
        "akdt": np.datetime64("2024-07-01T20:00"),  # -8: UTC = stamp + 8 h
        "akst": np.datetime64("2024-07-01T21:00"),
        "india": np.datetime64("2024-07-01T06:30"),  # +5.5: UTC = stamp - 5.5 h
        "anchorage_winter": np.datetime64("2024-01-15T21:00"),
        "undeclared": np.datetime64("2024-07-01T12:00"),
    }
    assert "time_zone" not in refs["undeclared"].metadata
    assert "utc_offset_h" not in refs["undeclared"].metadata
    assert refs["akdt"].metadata["utc_offset_h"] == -8.0


def test_a_per_source_zone_replaces_the_shared_one(tmp_path):
    """The two keys are alternatives, so naming one overrides a shared *other*."""
    cat = build.new_catalog(title="t")
    stamp = "2024-07-01 12:00"
    build.add_sources(
        cat,
        {
            "inherits": {"reader": _csv_reader(tmp_path, _series_csv(stamp), "a.csv")},
            "overrides": {
                "reader": _csv_reader(tmp_path, _series_csv(stamp), "b.csv"),
                "utc_offset_h": -9,
            },
            "clears": {
                "reader": _csv_reader(tmp_path, _series_csv(stamp), "c.csv"),
                "time_zone": None,
            },
        },
        name_map=None,
        time_zone="America/Anchorage",
    )
    refs = _refs(cat, tmp_path)
    first = {name: _as_dataset(ref)["time"].values[0] for name, ref in refs.items()}
    assert first == {
        "inherits": np.datetime64("2024-07-01T20:00"),
        "overrides": np.datetime64("2024-07-01T21:00"),
        "clears": np.datetime64("2024-07-01T12:00"),
    }
    assert refs["overrides"].metadata == {
        **refs["overrides"].metadata,
        "utc_offset_h": -9.0,
    }
    assert "time_zone" not in refs["overrides"].metadata


def test_add_erddap_source_takes_a_zone_too(monkeypatch, tmp_path):
    from tests.test_qc import _fake_table_dap_reader_factory

    csv = tmp_path / "erddap.csv"
    csv.write_text(
        "time (UTC),latitude (degrees_north),longitude (degrees_east),depth (m),"
        "TEMP (degree_Celsius)\n"
        "2024-07-01 12:00,61.0,-150.0,1.0,6.4\n"
        "2024-07-01 13:00,61.0,-150.0,1.0,6.5\n"
    )
    monkeypatch.setattr(
        "intake_erddap.erddap.TableDAPReader",
        _fake_table_dap_reader_factory(
            csv, lambda server, dataset_id: {"variables": {}}
        ),
    )
    cat = build.new_catalog(title="t")
    build.add_erddap_source(
        cat, "mooring", "https://example.org/erddap", "mooring_hourly", utc_offset_h=-8
    )
    ref = _refs(cat, tmp_path)["mooring"]
    assert ref.metadata["utc_offset_h"] == -8.0
    assert ref.metadata["time_coverage_start"] == "2024-07-01"

    with pytest.raises(ValueError, match="time_zone"):
        build.add_erddap_source(
            cat,
            "bad",
            "https://example.org/erddap",
            "mooring_hourly",
            time_zone="Nowhere/Land",
        )


# -- time zones: xarray sources ----------------------------------------------------


def _netcdf_series(tmp_path, name="series.nc", start="2024-07-01T18:00"):
    ds = xr.Dataset(
        {
            "temp": (
                "time",
                np.arange(3.0),
                {"standard_name": "sea_water_temperature", "units": "degC"},
            )
        },
        coords={
            "time": pd.date_range(start, periods=3, freq="h"),
            "lat": 61.0,
            "lon": -150.0,
        },
    )
    path = tmp_path / name
    ds.to_netcdf(path)
    return path


def test_a_netcdf_source_with_a_declared_zone_gets_its_time_shifted(tmp_path):
    cat = build.new_catalog(title="t")
    build.add_source(
        cat, "shifted", _netcdf_series(tmp_path, "a.nc"), time_zone="America/Anchorage"
    )
    build.add_source(cat, "plain", _netcdf_series(tmp_path, "b.nc"))
    refs = _refs(cat, tmp_path)

    shifted = osk.read(refs["shifted"])
    assert shifted["time"].values[0] == np.datetime64("2024-07-02T02:00")
    assert shifted["time"].attrs["source_time_zone"] == "America/Anchorage"
    # the probe's coverage is in UTC too: 18:00 AKDT is already tomorrow in UTC
    assert refs["shifted"].metadata["time_coverage_start"] == "2024-07-02"

    plain = osk.read(refs["plain"])
    assert plain["time"].values[0] == np.datetime64("2024-07-01T18:00")
    assert "source_time_zone" not in plain["time"].attrs
    assert refs["plain"].metadata["time_coverage_start"] == "2024-07-01"


def test_in_utc_leaves_an_undeclared_dataset_alone_and_warns_when_it_cannot_apply():
    ds = xr.Dataset(
        {"temp": ("time", [1.0, 2.0])},
        coords={"time": pd.date_range("2024-07-01", periods=2, freq="h")},
    )
    assert sources._in_utc(ds, {}, "time", subject="x") is ds
    assert sources._in_utc(ds, {"time_zone": "UTC"}, "time", subject="x") is ds

    # ROMS output is standardised to a `time` coordinate whatever the file called it
    out = sources._in_utc(ds, {"utc_offset_h": -8}, "time", subject="x")
    assert out["time"].values[0] == np.datetime64("2024-07-01T08:00")
    assert ds["time"].values[0] == np.datetime64(
        "2024-07-01T00:00"
    )  # not edited in place

    undecoded = xr.Dataset({"temp": ("time", [1.0, 2.0])}, coords={"time": [0, 1]})
    with pytest.warns(UserWarning, match="could not be applied"):
        same = sources._in_utc(undecoded, {"utc_offset_h": -8}, None, subject="x")
    assert same is undecoded


# -- time_columns: a time split over columns --------------------------------------


def _split_frame():
    return pd.DataFrame(
        {
            "Date": ["2024-07-01"] * 3,
            "Time": ["20:00", "21:00", "22:00"],
            "temp": [1.0, 2.0, 3.0],
        }
    )


def test_apply_table_options_joins_the_columns_into_one_naive_time():
    meta = {"time_columns": ["Date", "Time"]}
    out = tabular.apply_table_options(_split_frame(), meta)

    assert list(out.columns) == ["time", "temp"]  # first, and the pieces are gone
    assert out["time"].dt.tz is None  # naive: the zone is applied where it is decoded
    assert out["time"].tolist() == list(
        pd.date_range("2024-07-01 20:00", periods=3, freq="h")
    )
    assert tabular.joined_time_column(out, meta) == "time"
    # asked again (to_dataset does), it recognises its own work
    assert tabular.apply_table_options(out, meta) is out


def test_apply_table_options_is_a_no_op_without_time_columns():
    frame = _split_frame()
    assert tabular.apply_table_options(frame, {}) is frame
    assert tabular.joined_time_column(frame, {}) is None


def test_apply_table_options_keeps_attrs_and_never_overwrites_a_time_column():
    frame = _split_frame()
    frame["time"] = ["not", "the", "time"]
    frame.attrs["qc_applied"] = {"scheme": "x"}
    meta = {"time_columns": ["Date", "Time"]}
    out = tabular.apply_table_options(frame, meta)

    assert list(out.columns) == ["time_joined", "temp", "time"]
    assert out["time"].tolist() == ["not", "the", "time"]  # the other column untouched
    assert out.attrs["qc_applied"] == {"scheme": "x"}
    assert tabular.joined_time_column(out, meta) == "time_joined"
    assert (
        tabular.joined_time_column(out.copy().rename_axis(None), meta) == "time_joined"
    )


def test_a_row_missing_a_piece_has_no_time():
    frame = _split_frame()
    frame.loc[1, "Time"] = np.nan
    frame.loc[2, "Date"] = "  "
    out = tabular.apply_table_options(frame, {"time_columns": ["Date", "Time"]})
    assert out["time"].isna().tolist() == [False, True, True]


def test_time_format_reads_what_pandas_would_guess_wrong():
    frame = pd.DataFrame({"Date": ["03/07/2024"], "Time": ["20:00"]})
    guessed = tabular.apply_table_options(frame, {"time_columns": ["Date", "Time"]})
    assert guessed["time"].iloc[0].month == 3  # month first: the wrong reading
    right = tabular.apply_table_options(
        frame, {"time_columns": ["Date", "Time"], "time_format": "%d/%m/%Y %H:%M"}
    )
    assert right["time"].iloc[0] == pd.Timestamp("2024-07-03 20:00")


def test_apply_table_options_names_what_is_wrong():
    with pytest.raises(ValueError, match="Nope") as missing:
        tabular.apply_table_options(_split_frame(), {"time_columns": ["Date", "Nope"]})
    assert "Date" in str(missing.value) and "temp" in str(missing.value)  # what it has

    frame = pd.DataFrame({"Date": ["x", "y"], "Time": ["a", "b"]})
    with pytest.raises(ValueError, match=r"'x a'.*time_format"):
        tabular.apply_table_options(frame, {"time_columns": ["Date", "Time"]})


@pytest.mark.parametrize(
    ("time_columns", "time_format", "match"),
    [
        ("Date", None, "time_columns"),
        (["Date"], None, "time_columns"),
        (["Date", ""], None, "time_columns"),
        (["Date", 3], None, "time_columns"),
        (["Date", "Time"], 5, "time_format"),
        (["Date", "Time"], "  ", "time_format"),
        (None, "%d/%m/%Y", "time_format"),
    ],
)
def test_bad_time_options_are_refused(time_columns, time_format, match):
    with pytest.raises(ValueError, match=match):
        tabular.canonicalize_time_options(time_columns, time_format)


def test_canonical_time_options():
    assert tabular.canonicalize_time_options() == {}
    assert tabular.canonicalize_time_options(("Date", "Time")) == {
        "time_columns": ["Date", "Time"]
    }
    assert tabular.canonicalize_time_options(["Date", "Time"], "%d/%m/%Y %H:%M") == {
        "time_columns": ["Date", "Time"],
        "time_format": "%d/%m/%Y %H:%M",
    }


SPLIT_CSV = (
    "Date,Time,lon,lat,depth (m),temp (degC)\n"
    "2024-07-01,20:00,-150.0,61.0,5.0,8.0\n"
    "2024-07-01,21:00,-150.0,61.0,5.0,8.5\n"
    "2024-07-01,23:00,-150.0,61.0,5.0,9.0\n"
)


def test_time_columns_are_probed_read_and_localised(tmp_path):
    ref = _entry(
        tmp_path,
        SPLIT_CSV,
        time_columns=["Date", "Time"],
        time_zone="America/Anchorage",
    )
    md = ref.metadata

    assert md["time_columns"] == ["Date", "Time"]
    assert md["axes"]["T"] == "time"  # the joined column is the axis
    assert md["time_coverage_start"] == "2024-07-02"  # 20:00 AKDT is 04:00 UTC tomorrow
    assert md["time_coverage_end"] == "2024-07-02"
    assert "Date" not in md["standard_names"] and "Time" not in md["standard_names"]
    assert md["variables"] == ["temp"]
    assert md["featureType"] == "timeSeries"

    frame = osk.read(ref)
    assert "Date" not in frame.columns and "Time" not in frame.columns
    assert frame["time"].iloc[0] == pd.Timestamp("2024-07-02T04:00", tz="UTC")

    ds = tabular.to_dataset(frame, md)
    assert list(ds.data_vars) == ["temp"]
    assert ds["time"].values[0] == np.datetime64("2024-07-02T04:00")
    assert ds["time"].values[-1] == np.datetime64("2024-07-02T07:00")
    assert ds["time"].attrs["source_time_zone"] == "America/Anchorage"


def test_time_columns_with_a_format_and_no_zone(tmp_path):
    text = (
        "Date,Time,lon,lat,depth (m),temp (degC)\n"
        "03/07/2024,20:00,-150.0,61.0,5.0,8.0\n"
        "03/07/2024,21:00,-150.0,61.0,5.0,8.5\n"
    )
    ref = _entry(
        tmp_path, text, time_columns=["Date", "Time"], time_format="%d/%m/%Y %H:%M"
    )
    assert ref.metadata["time_format"] == "%d/%m/%Y %H:%M"
    assert ref.metadata["time_coverage_start"] == "2024-07-03"
    assert _as_dataset(ref)["time"].values[0] == np.datetime64("2024-07-03T20:00")


def test_time_columns_work_without_a_probe(tmp_path):
    """No probe means no recorded axes: the read still finds the joined column."""
    ref = _entry(
        tmp_path, SPLIT_CSV, time_columns=["Date", "Time"], utc_offset_h=-8, probe=False
    )
    assert "axes" not in ref.metadata
    frame = osk.read(ref)
    assert frame["time"].iloc[0] == pd.Timestamp("2024-07-02T04:00", tz="UTC")
    assert _as_dataset(ref)["time"].values[0] == np.datetime64("2024-07-02T04:00")


def test_to_dataset_joins_a_raw_table_itself():
    raw = pd.read_csv(pd.io.common.StringIO(SPLIT_CSV))
    ds = tabular.to_dataset(raw, {"time_columns": ["Date", "Time"], "utc_offset_h": -8})
    assert ds["time"].values[0] == np.datetime64("2024-07-02T04:00")
    assert list(ds.data_vars) == ["temp"]


def test_a_time_columns_declaration_the_table_cannot_satisfy_is_an_error(tmp_path):
    cat = build.new_catalog(title="t")
    reader = _csv_reader(tmp_path, SPLIT_CSV)
    with pytest.raises(ValueError, match="Nope"):
        build.add_source(
            cat, "bad", reader=reader, name_map=None, time_columns=["Date", "Nope"]
        )
    assert list(cat) == []


# -- the vertical axis: a declared column, and the wider bound ---------------------


def test_the_z_bound_keeps_either_sign_convention():
    values = pd.Series([-150.0, -9999.0, 12000.0, 500.0, -11001.0, 10000.0])
    kept = tabular.numeric_in_range(values, "Z")
    # -9999 is a fill sentinel, 12000 and -11001 are past the deepest trench
    assert kept.dropna().tolist() == [-150.0, 500.0, 10000.0]


def _two_depths_frame():
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-07-01", periods=4, freq="h").strftime(
                "%Y-%m-%d %H:%M"
            ),
            "lon": -150.0,
            "lat": 61.0,
            "depth (m)": 30.0,  # what some loggers call the water depth
            "bottom_depth (m)": 31.0,
            "sensor_depth (m)": 7.0,
            "temp (degC)": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_a_declared_z_column_is_the_depth_whatever_else_the_table_has():
    frame = _two_depths_frame()
    assert tabular.depth_of(frame, {}) == (30.0, "depth", False)  # the guess
    declared = {"axes": {"Z": "sensor_depth (m)"}}
    assert tabular.depth_of(frame, declared) == (7.0, "axes:sensor_depth (m)", False)

    ds = tabular.to_dataset(frame, declared)
    assert float(ds["depth"]) == 7.0
    assert ds.attrs["depth_source"] == "axes:sensor_depth (m)"
    assert ds["temp"].attrs["depth_m"] == 7.0
    assert "sensor_depth" not in ds  # the declared column is not a variable
    assert "depth" not in ds.data_vars


def test_a_declared_z_column_is_never_a_variable_whatever_it_is_called():
    frame = _two_depths_frame().rename(columns={"sensor_depth (m)": "inst_dm (m)"})
    ds = tabular.to_dataset(frame, {"axes": {"Z": "inst_dm (m)"}})
    assert float(ds["depth"]) == 7.0
    assert "inst_dm" not in ds.data_vars


def test_a_declared_z_that_is_all_nan_or_flat_zero_falls_through():
    """A catalog's axes are as often what the probe found as what a person wrote."""
    frame = _two_depths_frame()
    frame["z (m)"] = 0.0
    frame["sea_water_pressure (dbar)"] = 33.9
    with pytest.warns(UserWarning, match="placeholder"):
        depth, source, approximate = tabular.depth_of(
            frame.drop(columns=["depth (m)", "sensor_depth (m)"]),
            {"axes": {"Z": "z (m)"}},
        )
    assert (depth, source, approximate) == (33.9, "sea_water_pressure", True)

    # a declared column with nothing in it reads nothing: the search goes on without it
    frame["depth (m)"] = np.nan
    with pytest.warns(UserWarning, match="placeholder"):  # (z is still all zeros)
        _, source, _ = tabular.depth_of(frame, {"axes": {"Z": "depth (m)"}})
    assert source == "sea_water_pressure"


def test_attach_keeps_the_probed_axes_when_a_caller_declares_only_z(tmp_path):
    ref = _entry(
        tmp_path,
        _two_depths_frame().to_csv(index=False),
        axes={"Z": "sensor_depth (m)"},
    )
    md = ref.metadata

    assert md["axes"] == {
        "X": "lon",
        "Y": "lat",
        "T": "time",
        "Z": "sensor_depth (m)",
    }
    # and the extent comes from the declared column, not from the guessed `depth (m)`
    assert md["geospatial_vertical_min"] == md["geospatial_vertical_max"] == 7.0
    assert md["featureType"] == "timeSeries"
    assert float(_as_dataset(ref)["depth"]) == 7.0


def test_a_declared_axis_the_table_lacks_is_warned_about(tmp_path):
    with pytest.warns(UserWarning, match="no_such_depth"):
        ref = _entry(
            tmp_path,
            _two_depths_frame().to_csv(index=False),
            axes={"Z": "no_such_depth"},
        )
    # the probe fell back to the column names for the extents ...
    assert ref.metadata["geospatial_vertical_min"] == 30.0
    # ... and the read does the same rather than failing on the unusable declaration
    assert float(_as_dataset(ref)["depth"]) == 30.0


def test_axes_must_be_a_mapping(tmp_path):
    cat = build.new_catalog(title="t")
    reader = _csv_reader(tmp_path, _series_csv())
    with pytest.raises(ValueError, match="axes"):
        build.add_source(cat, "bad", reader=reader, name_map=None, axes="Z")


# -- depth conventions: normalised at the door -----------------------------------------


def _cast_csv(z_name="z (m)", z=(-1.0, -10.0, -50.0, -150.0)):
    """One CTD cast (one instant, fixed position) with depth stored as given."""
    rows = "".join(
        f"2024-07-01 12:00,-150.0,61.0,{v},{t}\n"
        for v, t in zip(z, (8.0, 7.5, 7.0, 6.0), strict=True)
    )
    return f"time,lon,lat,{z_name},temp (degC)\n{rows}"


@pytest.mark.parametrize(
    "declared",
    [{"depth_convention": {"positive": "up"}}, {}],
    ids=["declared positive up", "undeclared: read off the values"],
)
def test_a_positive_up_profile_becomes_depths_below_the_surface(tmp_path, declared):
    ref = _entry(tmp_path, _cast_csv(), **declared)
    assert ref.metadata["featureType"] == "profile"
    # the probe's extent is the raw column: whoever reads it converts via the convention
    assert ref.metadata["geospatial_vertical_min"] == -150.0

    ds = _as_dataset(ref)
    # 150 m on a positive-up axis survives (the old Z bound ended at -100)
    assert ds["depth"].values.tolist() == [1.0, 10.0, 50.0, 150.0]
    assert ds["temp"].values.tolist() == [8.0, 7.5, 7.0, 6.0]  # sorted along with it
    attrs = ds["depth"].attrs
    assert attrs["units"] == "m" and attrs["positive"] == "down"
    assert attrs["long_name"] == "depth"
    assert attrs["depth_origin"] == "surface"  # an undeclared profile is a cast
    assert attrs[depth_convention.NORMALIZED_ATTR] == 1
    assert "depth_datum_z_m" not in attrs  # only a fixed origin has one
    assert "depth_approximate" not in attrs


def test_a_positive_down_profile_is_unchanged(tmp_path):
    ds = _as_dataset(_entry(tmp_path, _cast_csv("depth (m)", (1.0, 10.0, 50.0, 150.0))))
    assert ds["depth"].values.tolist() == [1.0, 10.0, 50.0, 150.0]
    assert ds["depth"].attrs[depth_convention.NORMALIZED_ATTR] == 1


def test_a_fixed_origin_and_its_datum_ride_on_the_coordinate(tmp_path):
    declared = {"origin": "fixed", "positive": "up", "datum_z_m": 2.0}
    ds = _as_dataset(_entry(tmp_path, _cast_csv(), depth_convention=declared))
    assert ds["depth"].values.tolist() == [1.0, 10.0, 50.0, 150.0]
    assert ds["depth"].attrs["depth_origin"] == "fixed"
    assert ds["depth"].attrs["depth_datum_z_m"] == 2.0


def test_a_positive_up_station_depth_is_normalised_on_the_time_series_path(tmp_path):
    text = (
        "time,lon,lat,z (m),temp (degC)\n"
        "2024-07-01 12:00,-150.0,61.0,-8.0,8.0\n"
        "2024-07-01 13:00,-150.0,61.0,-8.0,8.5\n"
    )
    ds = _as_dataset(_entry(tmp_path, text))
    assert float(ds["depth"]) == 8.0
    assert ds["temp"].attrs["depth_m"] == 8.0  # the variables' attrs follow
    assert (
        ds["depth"].attrs["depth_origin"] == "fixed"
    )  # a time series is a fixed sensor
    assert ds["depth"].attrs["depth_datum_z_m"] == 0.0


def test_a_pressure_column_is_inferred_and_comes_out_in_metres(tmp_path):
    text = (
        "time,lon,lat,PRES (dbar),temp (degC)\n"
        "2024-07-01 12:00,-150.0,61.0,5.0,8.0\n"
        "2024-07-01 12:00,-150.0,61.0,10.0,7.5\n"
        "2024-07-01 12:00,-150.0,61.0,20.0,7.0\n"
    )
    ref = _entry(tmp_path, text)
    # the probe wrote what the column's name and units give away -- apart from anything
    # declared -- and said what it read it from
    assert ref.metadata["depth_convention"] == {
        "inferred": {
            "origin": "surface",
            "units": "dbar",
            "reason": "Z column 'PRES (dbar)'",
        }
    }
    resolved = depth_convention.resolve(ref.metadata)
    assert (resolved.origin, resolved.units) == ("surface", "dbar")
    assert not resolved.declared

    ds = _as_dataset(ref)
    assert ds["depth"].values.tolist() == [5.0, 10.0, 20.0]
    attrs = ds["depth"].attrs
    assert attrs["units"] == "m"
    assert (
        attrs["depth_approximate"] == 1
    )  # 1 dbar ~ 1 m: said so, not presented as measured
    assert attrs["depth_origin"] == "surface"


def test_a_pressure_station_is_surface_referenced_whatever_its_featuretype(tmp_path):
    text = (
        "time,lon,lat,PRES (dbar),temp (degC)\n"
        "2024-07-01 12:00,-150.0,61.0,33.9,8.0\n"
        "2024-07-01 13:00,-150.0,61.0,33.9,8.5\n"
    )
    ds = _as_dataset(_entry(tmp_path, text))
    assert float(ds["depth"]) == 33.9
    assert ds.attrs["depth_approximate"] is True  # the long-standing dataset attr
    assert ds["depth"].attrs["depth_origin"] == "surface"
    assert ds["depth"].attrs["depth_approximate"] == 1


def test_the_dbar_factor_is_applied_once(monkeypatch):
    """``depth_of`` converts pressure to metres; the convention must not do it again."""
    monkeypatch.setattr(depth_convention, "M_PER_DBAR", 1.02)  # not 1.0: a double shows
    frame = pd.DataFrame(
        {
            "time": ["2024-07-01 12:00", "2024-07-01 13:00"],
            "lon": -150.0,
            "lat": 61.0,
            "PRES (dbar)": [100.0, 100.0],
            "temp (degC)": [1.0, 2.0],
        }
    )
    assert tabular.depth_of(frame, {})[0] == pytest.approx(102.0)
    assert float(tabular.to_dataset(frame, {})["depth"]) == pytest.approx(102.0)

    cast = pd.DataFrame(
        {
            "time": ["2024-07-01 12:00"] * 2,
            "lon": -150.0,
            "lat": 61.0,
            "PRES (dbar)": [100.0, 200.0],
            "temp (degC)": [1.0, 2.0],
        }
    )
    ds = tabular.to_dataset(cast, {"featureType": "profile"})
    assert ds["depth"].values.tolist() == pytest.approx([102.0, 204.0])


def test_a_time_series_profile_is_normalised_too(tmp_path):
    text = "time,lon,lat,z (m),temp (degC)\n" + "".join(
        f"{t},-150.0,61.0,{z},{8.0 - i}\n"
        for i, (t, z) in enumerate(
            (t, z)
            for t in ("2024-07-01 12:00", "2024-07-02 12:00")
            for z in (-1.0, -5.0)
        )
    )
    ref = _entry(tmp_path, text)
    assert ref.metadata["featureType"] == "timeSeriesProfile"
    ds = _as_dataset(ref)
    assert ds["depth"].values.tolist() == [1.0, 5.0]
    assert ds["depth"].attrs["depth_origin"] == "fixed"  # not a cast by default
    assert ds["depth"].attrs[depth_convention.NORMALIZED_ATTR] == 1


def test_a_second_pass_leaves_normalised_depths_alone(tmp_path):
    """What ``comparison`` does next: the NORMALIZED_ATTR stops a second flip."""
    ds = _as_dataset(_entry(tmp_path, _cast_csv()))
    again = depth_convention.positive_down_values(ds["depth"].values, ds["depth"].attrs)
    assert again.tolist() == [1.0, 10.0, 50.0, 150.0]


# -- the probe: what the vertical coordinate gives away ----------------------------


def test_the_probe_reads_a_netcdf_vertical_coordinates_cf_attributes():
    ds = xr.Dataset(
        {"temp": ("depth", [1.0, 2.0, 3.0])},
        coords={
            "depth": ("depth", [-1.0, -5.0, -9.0], {"positive": "up", "units": "m"}),
            "time": pd.Timestamp("2024-07-01"),
        },
    )
    md = build._probe(ds, None)
    assert md["depth_convention"] == {
        "inferred": {
            "positive": "up",
            "units": "m",
            "reason": "vertical coordinate 'depth'",
        }
    }
    # the extent stays the raw values
    assert (md["geospatial_vertical_min"], md["geospatial_vertical_max"]) == (
        -9.0,
        -1.0,
    )

    # a vertical coordinate that says it is pressure: decibars, below the free surface
    pressure = ds.copy(deep=True)
    pressure["depth"].attrs = {"units": "dbar", "standard_name": "sea_water_pressure"}
    found = build._probe(pressure, None)["depth_convention"]["inferred"]
    assert (found["origin"], found["units"]) == ("surface", "dbar")


def test_the_probe_does_not_read_a_roms_s_rho_as_an_obs_depth():
    s = np.array([-0.75, -0.25])
    ds = xr.Dataset(
        {
            "Cs_r": ("s_rho", s),
            "sigma_r": ("s_rho", s),
            "temp": (("s_rho", "eta_rho", "xi_rho"), np.ones((2, 2, 2))),
            "lon_rho": (("eta_rho", "xi_rho"), np.zeros((2, 2))),
            "lat_rho": (("eta_rho", "xi_rho"), np.zeros((2, 2))),
        },
        coords={"s_rho": ("s_rho", s, {"positive": "up", "standard_name": "height"})},
        attrs={"hc": 10.0, "Vtransform": 2},
    )
    from ocean_skill.cf import find_coord

    assert find_coord(ds, "vertical") is not None  # so the skip below is not vacuous
    md = build._probe(ds, None)
    assert md["model"] == "roms"
    assert "depth_convention" not in md


def test_probe_signatures_stay_backward_compatible(tmp_path):
    frame = pd.read_csv(_write_csv(tmp_path, _series_csv(), "raw.csv"))
    assert build._probe(frame, None)["axes"]["T"] == "time"
    assert build._probe_dataframe(frame)["axes"]["T"] == "time"
    assert build._probe_dataframe(frame, declared=None, qc=None)["axes"]["T"] == "time"


# -- _attach: declared words win, and a bad one is an error ------------------------


PRESSURE_CAST = (
    "time,lon,lat,PRES (dbar),temp (degC)\n"
    "2024-07-01 12:00,-150.0,61.0,5.0,8.0\n"
    "2024-07-01 12:00,-150.0,61.0,10.0,7.5\n"
)


def test_a_declared_depth_convention_wins_over_what_the_probe_inferred(tmp_path):
    ref = _entry(tmp_path, PRESSURE_CAST, depth_convention={"origin": "fixed"})
    dc = ref.metadata["depth_convention"]

    assert dc["origin"] == "fixed"  # the person's word, at the top
    assert dc["inferred"]["origin"] == "surface"  # the probe's, kept apart
    assert dc["inferred"]["units"] == "dbar"
    resolved = depth_convention.resolve(ref.metadata)
    assert (resolved.origin, resolved.units) == ("fixed", "dbar")
    assert resolved.declared
    assert _as_dataset(ref)["depth"].attrs["depth_origin"] == "fixed"


def test_a_rebuild_replaces_a_stale_inferred_block_but_keeps_the_declared_words(
    tmp_path,
):
    stale = {
        "origin": "fixed",
        "inferred": {"units": "m", "reason": "an earlier build"},
    }
    ref = _entry(tmp_path, PRESSURE_CAST, depth_convention=stale)
    dc = ref.metadata["depth_convention"]
    assert dc["origin"] == "fixed"
    assert dc["inferred"]["units"] == "dbar"
    assert "an earlier build" not in dc["inferred"]["reason"]


def test_a_shorthand_depth_convention_is_canonicalised(tmp_path):
    ref = _entry(tmp_path, _series_csv(), depth_convention="Free_Surface")
    assert ref.metadata["depth_convention"]["origin"] == "surface"


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"depth_convention": {"origin": "sideways"}}, "depth_convention"),
        ({"depth_convention": {"orgin": "surface"}}, "depth_convention"),
        ({"depth_convention": {"origin": "surface", "datum_z_m": 2.0}}, "datum_z_m"),
        ({"time_zone": "Mars/Phobos"}, "time_zone"),
        ({"utc_offset_h": 99}, "utc_offset_h"),
        ({"utc_offset_h": -8, "time_zone": "America/Anchorage"}, "alternatives"),
        ({"time_columns": ["time"]}, "time_columns"),
        ({"time_format": "%Y"}, "time_format"),
    ],
)
def test_an_invalid_declaration_raises_value_error_out_of_add_source(
    tmp_path, metadata, match
):
    """It must not become the "could not derive metadata" warning and a kept entry."""
    cat = build.new_catalog(title="t")
    reader = _csv_reader(tmp_path, _series_csv())
    with pytest.raises(ValueError, match=match):
        build.add_source(cat, "bad", reader=reader, name_map=None, **metadata)
    assert list(cat) == []


def test_a_datum_that_only_conflicts_with_what_the_probe_found_is_an_error_too(
    tmp_path,
):
    """A pressure column is surface-referenced; a datum has no meaning there."""
    cat = build.new_catalog(title="t")
    reader = _csv_reader(tmp_path, PRESSURE_CAST)
    with pytest.raises(ValueError, match="datum_z_m"):
        build.add_source(
            cat,
            "bad",
            reader=reader,
            name_map=None,
            depth_convention={"datum_z_m": 2.0},
        )
    assert list(cat) == []


def test_the_zone_is_canonicalised_and_only_one_key_is_stored(tmp_path):
    cat = build.new_catalog(title="t")
    reader = _csv_reader(tmp_path, _series_csv())
    reader.metadata["utc_offset_h"] = -5.0  # left by an earlier build of this entry
    build.add_source(
        cat, "a", reader=reader, name_map=None, time_zone="america/anchorage"
    )
    md = dict(cat["a"].metadata)
    assert md["time_zone"] == "America/Anchorage"
    assert "utc_offset_h" not in md

    other = _csv_reader(tmp_path, _series_csv(), "other.csv")
    build.add_source(cat, "b", reader=other, name_map=None, utc_offset_h="-9")
    assert cat["b"].metadata["utc_offset_h"] == -9.0
    assert "time_zone" not in cat["b"].metadata


def test_add_sources_raises_a_bad_declaration_as_value_error(tmp_path):
    cat = build.new_catalog(title="t")
    specs = {
        "good": {"reader": _csv_reader(tmp_path, _series_csv(), "a.csv")},
        "bad": {
            "reader": _csv_reader(tmp_path, _series_csv(), "b.csv"),
            "utc_offset_h": 99,
        },
    }
    with pytest.raises(ValueError, match="utc_offset_h") as caught:
        build.add_sources(cat, specs, name_map=None)
    assert not isinstance(caught.value, RuntimeError)
    assert "'bad'" in str(caught.value)  # it says which entry

    skipped = build.new_catalog(title="t")
    with pytest.warns(UserWarning, match="skipping 'bad'"):
        added = build.add_sources(skipped, specs, name_map=None, skip_errors=True)
    assert list(added) == ["good"]


def test_a_bad_shared_declaration_fails_before_anything_is_opened(tmp_path):
    cat = build.new_catalog(title="t")
    never_opened = {"missing": str(tmp_path / "does_not_exist.nc")}
    with pytest.raises(ValueError, match="time_zone"):
        build.add_sources(cat, never_opened, time_zone="Mars/Phobos", skip_errors=True)
    assert list(cat) == []


def test_undeclared_entries_gain_no_new_keys(tmp_path):
    """Nothing declared, nothing recorded: an old-style entry reads as it always did."""
    md = _entry(tmp_path, _series_csv()).metadata
    for key in ("time_zone", "utc_offset_h", "time_columns", "time_format"):
        assert key not in md
    # (a unit-suffixed depth column records what it gave away, apart from the person)
    assert md["depth_convention"]["inferred"]["units"] == "m"
