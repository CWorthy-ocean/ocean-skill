"""The cast layer: each cast of a ``trajectoryProfile`` entry is a ``profile`` source.

``"<entry>[<cast id>]"`` resolves (:func:`ocean_skill.catalog.resolve`) to the cast's
own metadata and reads (:func:`ocean_skill.sources.read`) as its samples alone, so a
cast compares like a profile registered by itself. Offline: tiny CSV and NetCDF
transects, catalogs written into the ``isolated_catalogs`` directory.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from intake.readers import datatypes, readers

import ocean_skill as osk
from ocean_skill import build, casts, catalog, tabular

pytestmark = pytest.mark.usefixtures("isolated_catalogs")

# station, start, longitudes (a sample every 30 s), depths: A is visited twice (the
# second time down and up), B drifts, C is a single sample
TRANSECT = [
    ("A", "2024-07-01 00:00", [-150.0] * 3, [1, 2, 3]),
    ("B", "2024-07-01 02:00", [-150.1 + d for d in (2e-4, 0, 0, 3e-4)], [1, 2, 3, 4]),
    ("A", "2024-07-01 04:00", [-150.0] * 3, [2, 1, 2]),
    ("C", "2024-07-01 06:00", [-150.2], [5]),
]
NEAR = [  # 90 s apart, and 1 km
    ("A", "2024-07-01 00:00", [-150.0] * 2, [1, 2]),
    ("B", "2024-07-01 00:02", [-149.99] * 2, [1, 2]),
]
HOURLY = [(f"s{i}", f"2024-07-01 {i:02d}:00", [-150.0], [1]) for i in range(10)]
ID = {"casts": {"id": "station"}}
SPLIT = {"casts": {"gap": "15min"}}  # declared, with no id: split on pauses and moves


def _frame(spec):
    """Return the samples of ``spec`` in time order (``val`` counts them)."""
    rows = [
        (st, pd.Timestamp(t0) + pd.Timedelta(seconds=30 * i), lon, 61.0, d)
        for st, t0, lons, depths in spec
        for i, (lon, d) in enumerate(zip(lons, depths, strict=True))
    ]
    cols = ["station", "time", "lon", "lat", "depth (m)"]
    return pd.DataFrame(rows, columns=cols).assign(val=range(len(rows)))


def _nc(tmp_path, kind):
    """Write ``TRANSECT`` as NetCDF: ``flat`` (a row per sample) or ``multi``."""
    if kind == "flat":
        ds = _frame(TRANSECT).rename(columns={"depth (m)": "depth"}).to_xarray()
        ds = ds.set_coords(["time", "lon", "lat", "depth"])
    else:  # a column per level, short casts padded with NaN
        pad = lambda v: [*v, *[np.nan] * (4 - len(v))]  # noqa: E731
        ids = ("profile", [c[0] for c in TRANSECT], {"cf_role": "profile_id"})
        ds = xr.Dataset(
            {"val": (("profile", "z"), np.arange(16.0).reshape(4, 4)), "station": ids},
            {
                "time": ("profile", pd.to_datetime([c[1] for c in TRANSECT])),
                "lon": ("profile", [np.median(c[2]) for c in TRANSECT]),
                "lat": ("profile", [61.0] * 4),
                "depth": (("profile", "z"), [pad(c[3]) for c in TRANSECT]),
            },
        )
    ds.attrs["featureType"] = "trajectoryProfile"
    ds.to_netcdf(tmp_path / f"{kind}.nc")
    return tmp_path / f"{kind}.nc"


def _write(tmp_path, **entries):
    """Write catalog ``c`` of ``name=(netcdf path | csv samples, metadata)`` entries."""
    cat = build.new_catalog(title="c")
    for name, (source, meta) in entries.items():
        if isinstance(source, Path):
            url, reader = str(source), None
        else:  # a CSV table, its rows not in time order
            frame = source if isinstance(source, pd.DataFrame) else _frame(source)
            csv = tmp_path / f"{name}.csv"
            frame.sample(frac=1, random_state=3).to_csv(csv, index=False)
            url, reader = None, readers.PandasCSV(datatypes.CSV(url=str(csv)))
        meta = {"featureType": "trajectoryProfile", **meta}
        build.add_source(cat, name, url, reader=reader, name_map=None, **meta)
    return build.save(cat, Path(os.environ["OCEAN_SKILL_CATALOGS"]) / "c.yaml")


def test_canonicalize_stores_the_declaration_in_canonical_form():
    assert casts.canonicalize(None) is None and casts.canonicalize({}) is None
    spec = {"id": "station", "gap": " 15min ", "distance_m": 200, "position": "first"}
    assert casts.canonicalize(spec) == {**spec, "gap": "15min", "distance_m": 200.0}
    assert casts.canonicalize({"gap": 900}) == {"gap": 900}  # a bare number is seconds


@pytest.mark.parametrize(
    "bad",
    [
        {"bogus": 1}, "station", {"id": ""}, {"id": 3}, {"gap": "soon"}, {"gap": 0},
        {"gap": -5}, {"distance_m": 0}, {"distance_m": "far"}, {"position": "mode"},
    ],
)  # fmt: skip
def test_canonicalize_refuses_what_it_cannot_read(bad):
    with pytest.raises(ValueError, match=r"^casts: "):
        casts.canonicalize(bad)


def test_add_source_validates_and_stores_the_declaration(tmp_path):
    with pytest.raises(ValueError, match="casts: gap"):
        _write(tmp_path, t=(TRANSECT, {"casts": {"gap": "soon"}}))
    _write(tmp_path, t=(TRANSECT, {"casts": {"id": "station", "gap": " 1h"}}))
    assert catalog.resolve("t").metadata["casts"] == {"id": "station", "gap": "1h"}


def test_casts_by_an_id_column_in_time_order_with_a_revisit(tmp_path):
    _write(tmp_path, t=(TRANSECT, ID))
    found = casts.table("t")
    assert list(found.columns) == ["id", "name", "time", "lon", "lat", "n"]
    assert list(found["id"]) == ["A", "B", "A#2", "C"]
    assert list(found["n"]) == [3, 4, 3, 1]
    assert casts.names("c:t") == ["c:t[A]", "c:t[B]", "c:t[A#2]", "c:t[C]"]
    assert osk.cast_names("t") == ["t[A]", "t[B]", "t[A#2]", "t[C]"]
    assert found["time"][1] == pd.Timestamp("2024-07-01 02:00")
    assert found["lon"][1] == pytest.approx(-150.0999)  # B drifts: the median
    # the table is memoized, until the catalog file changes
    _write(tmp_path, t=(TRANSECT, SPLIT))
    assert list(casts.table("t")["id"]) == ["1", "2", "3", "4"]


@pytest.mark.parametrize(
    ("how", "lon"), [("median", -150.0999), ("first", -150.0998), ("mean", -150.099875)]
)
def test_a_cast_stands_where_the_position_option_says(tmp_path, how, lon):
    _write(tmp_path, t=(TRANSECT, {"casts": {"id": "station", "position": how}}))
    assert casts.table("t")["lon"][1] == pytest.approx(lon, abs=1e-9)


def test_without_an_id_casts_split_where_time_or_distance_jumps(tmp_path):
    _write(
        tmp_path,
        t=(TRANSECT, SPLIT),
        many=(HOURLY, SPLIT),
        slow=(HOURLY, {"casts": {"gap": "3h"}}),
        near=(NEAR, {"casts": {"position": "median"}}),  # any key opts in
        far=(NEAR, {"casts": {"distance_m": 5000}}),
    )
    assert list(casts.table("t")["id"]) == ["1", "2", "3", "4"]
    assert list(casts.table("many")["id"]) == [f"{i:02d}" for i in range(1, 11)]
    assert list(casts.table("slow")["n"]) == [10]  # an hour apart is within a 3 h gap
    assert list(casts.table("near")["n"]) == [2, 2]
    assert list(casts.table("far")["n"]) == [4]


def test_casts_nothing_identifies_are_not_guessed(tmp_path):
    _write(tmp_path, t=(TRANSECT, {}), nc=(_nc(tmp_path, "flat"), {}))
    for name in ("t", "nc"):
        with pytest.raises(casts.NoCasts, match=r"not identified.*casts=\{'id'"):
            osk.cast_names(name)
    with pytest.raises(KeyError, match="not identified"):
        catalog.resolve("t[1]")


def test_the_profile_dimension_identifies_the_casts_of_a_netcdf_file(tmp_path):
    ds = xr.open_dataset(_nc(tmp_path, "multi")).load()
    del ds["station"].attrs["cf_role"]
    ds.to_netcdf(tmp_path / "plain.nc")
    _write(tmp_path, t=(tmp_path / "plain.nc", {}))
    assert list(casts.table("t")["n"]) == [3, 4, 3, 1]


def test_casts_on_another_feature_type_is_ignored_with_a_warning(tmp_path):
    with pytest.warns(UserWarning, match="casts= is ignored.*a timeSeriesProfile"):
        _write(tmp_path, m=(TRANSECT, {"featureType": "timeSeriesProfile", **ID}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to say for a trajectoryProfile
        _write(tmp_path, t=(TRANSECT, ID))


def test_rows_with_no_time_belong_to_no_cast(tmp_path):
    _write(tmp_path, t=(TRANSECT, ID))
    csv = tmp_path / "t.csv"
    csv.write_text(csv.read_text().replace("2024-07-01 00:00:00", "", 1))
    with pytest.warns(UserWarning, match="1 of 11 samples have no time"):
        assert casts.table("t")["n"].sum() == 10


@pytest.mark.parametrize("kind", ["flat", "multi"])
def test_a_netcdf_transect_has_the_casts_of_the_csv_one(tmp_path, kind):
    # multi: a cast per profile, named by the variable with cf_role: profile_id
    _write(tmp_path, t=(_nc(tmp_path, kind), ID if kind == "flat" else {}))
    found = casts.table("t")
    assert list(found["id"]) == ["A", "B", "A#2", "C"]
    assert list(found["n"]) == [3, 4, 3, 1]
    assert list(found["lon"]) == pytest.approx([-150.0, -150.0999, -150.0, -150.2])
    cast = osk.read("t[A#2]")
    assert dict(cast.sizes) == {"depth": 2} and cast.attrs["featureType"] == "profile"
    assert list(cast["depth"]) == [1.0, 2.0]  # ascending; of 2 m (twice), the first
    assert list(cast["val"]) == {"flat": [8, 7], "multi": [9, 8]}[kind]
    assert [cast[n].ndim for n in ("time", "lon", "lat")] == [0, 0, 0]
    assert cast["time"] == np.datetime64("2024-07-01T04:00") and cast["lon"] == -150.0


def test_a_ragged_array_is_refused(tmp_path):
    ds = xr.open_dataset(_nc(tmp_path, "flat")).load()
    ds["rows"] = ("profile", [3, 4, 3, 1], {"sample_dimension": "index"})
    ds.to_netcdf(tmp_path / "ragged.nc")
    _write(tmp_path, ragged=(tmp_path / "ragged.nc", {}))
    with pytest.raises(ValueError, match="ragged"):
        casts.table("ragged")


def test_a_cast_resolves_to_the_metadata_of_a_profile(tmp_path):
    _write(tmp_path, t=(TRANSECT, ID))
    parent, ref = catalog.resolve("t"), catalog.resolve("c:t[B]")
    meta, lon, when = ref.metadata, pytest.approx(-150.0999), "2024-07-01T02:00:00"
    assert (ref.name, ref.cast, ref.qualified) == ("t", "B", "c:t[B]")
    assert meta["featureType"] == "profile" and "casts" not in meta
    assert parent.cast is None and "casts" in parent.metadata  # the parent as it was
    info = {"id": "B", "of": "c:t", "time": when, "lon": lon, "lat": 61.0, "n": 4}
    assert meta["cast"] == info
    assert meta["minTime"] == meta["maxTime"] == when
    assert meta["time_coverage_start"] == meta["time_coverage_end"] == "2024-07-01"
    assert meta["geospatial_lon_min"] == meta["geospatial_lon_max"] == lon
    assert meta["geospatial_lat_min"] == meta["geospatial_lat_max"] == 61.0
    assert catalog.fingerprint("t[B]") == catalog.fingerprint("t") != ""


def test_a_name_that_is_no_cast_is_a_key_error(tmp_path):
    _write(tmp_path, t=(TRANSECT, ID), grid=(TRANSECT, {"featureType": "grid"}))
    with pytest.raises(KeyError, match=r"no cast 'Z'.*'A', 'B', 'A#2', 'C'"):
        catalog.resolve("t[Z]")
    with pytest.raises(KeyError, match="has no casts"):
        catalog.resolve("grid[1]")
    with pytest.raises(KeyError, match="Unknown source 'nope'"):
        catalog.resolve("nope[1]")
    with pytest.raises(ValueError, match="not a trajectoryProfile"):
        casts.table("grid")


def test_reading_a_drifting_cast_gives_its_rows_and_one_position(tmp_path):
    _write(tmp_path, t=(TRANSECT, ID))
    ref = catalog.resolve("t[B]")
    frame = osk.read(ref)
    assert list(frame["station"]) == ["B"] * 4 and frame["time"].is_monotonic_increasing
    assert list(osk.read("t[C]")["depth (m)"]) == [5]  # not the memoized B
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "time varies across the cast"
        ds = tabular.to_dataset(frame, ref.metadata)
    assert float(ds["lon"]) == pytest.approx(-150.0999)
    assert ds["time"] == np.datetime64("2024-07-01T02:00")


@pytest.mark.parametrize(("cast", "rows"), [("A", slice(0, 3)), ("C", slice(10, 11))])
def test_a_cast_is_the_profile_its_rows_alone_would_be(tmp_path, cast, rows):
    own = (_frame(TRANSECT).iloc[rows], {"featureType": "profile"})
    _write(tmp_path, t=(TRANSECT, ID), own=own)
    ref = catalog.resolve(f"t[{cast}]")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a cast does not warn "time varies across it"
        made = tabular.to_dataset(osk.read(ref), ref.metadata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the same rows as an entry of their own do
        expected = tabular.to_dataset(osk.read("own"), catalog.resolve("own").metadata)
    xr.testing.assert_identical(made, expected)


def test_clearing_the_cache_reads_the_transect_again(tmp_path):
    _write(tmp_path, t=(TRANSECT, ID))
    assert len(casts.table("t")) == 4
    csv = tmp_path / "t.csv"
    csv.write_text("".join(line for line in csv.open() if not line.startswith("C,")))
    osk.cache.clear()
    assert list(casts.table("t")["id"]) == ["A", "B", "A#2"]
