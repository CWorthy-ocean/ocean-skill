"""Tests for :mod:`ocean_skill.tabular` — a station table becoming a 1-D Dataset.

The frames built here use the *real* column spellings an OOI Station Papa ERDDAP entry
produces, including the two states one frame can be in at once: data columns already
renamed by :func:`ocean_skill.sources.read` (bare) and coordinate columns left alone
(carrying their ``(units)`` suffix). Getting units from only one of those two places is
the easiest thing to break, so both are asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ocean_skill import tabular

#: The metadata an OOI tabledap entry carries, trimmed to what tabular reads. Keys of
#: `units`/`standard_names` are the *original* column names, as the catalog has them.
PAPA_META = {
    "axes": {
        "T": "time (UTC)",
        "X": "longitude (degrees_east)",
        "Y": "latitude (degrees_north)",
    },
    "datasetID": "ooi-gp03flma-rim01-02-ctdmog040",
    "featureType": "timeSeries",
    "institution": "Ocean Observatories Initiative (OOI)",
    "title": "Global Station Papa: Flanking Subsurface Mooring A: CTD (30 meters)",
    "standard_names": {
        "sea_water_temperature (degree_Celsius)": "sea_water_temperature",
        "sea_water_pressure (decibars)": "sea_water_pressure",
        "depth_reading (m)": "depth_reading",
    },
    "units": {
        "sea_water_temperature (degree_Celsius)": "degree_Celsius",
        "sea_water_pressure (decibars)": "decibars",
        "depth_reading (m)": "m",
    },
}


def papa_frame(
    n: int = 96,
    *,
    renamed: bool = True,
    pressure: float | np.ndarray | None = 33.9,
    depth_reading: float | np.ndarray | None = np.nan,
    z: float | np.ndarray | None = None,
    qc: bool = True,
    station: bool = True,
    lon: float | np.ndarray = -144.245227,
):
    """Return a Papa-shaped frame; ``renamed`` mimics what ``sources.read`` gives."""
    time = pd.date_range("2015-06-01", periods=n, freq="15min", tz="UTC")
    temperature = 9.0 + np.sin(np.linspace(0, 6.0, n))
    columns: dict[str, object] = {
        "time (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latitude (degrees_north)": np.full(n, 49.977563),
        "longitude (degrees_east)": np.full(n, lon) if np.isscalar(lon) else lon,
    }
    if z is not None:
        columns["z (m)"] = np.full(n, z) if np.isscalar(z) else z
    key = (
        "sea_water_temperature" if renamed else "sea_water_temperature (degree_Celsius)"
    )
    columns[key] = temperature
    if qc:
        columns["sea_water_temperature_qc_agg"] = np.ones(n, dtype="int64")
        columns["sea_water_temperature_qc_tests"] = np.zeros(n)
    if depth_reading is not None:
        name = "depth_reading" if renamed else "depth_reading (m)"
        columns[name] = (
            np.full(n, depth_reading) if np.isscalar(depth_reading) else depth_reading
        )
    if pressure is not None:
        name = "sea_water_pressure" if renamed else "sea_water_pressure (decibars)"
        columns[name] = np.full(n, pressure) if np.isscalar(pressure) else pressure
    if station:
        columns["station"] = np.full(n, np.nan)
    return pd.DataFrame(columns)


# -- the column vocabulary ------------------------------------------------------------


def test_split_units_reads_both_spellings():
    assert tabular.split_units("sea_water_temperature (degree_Celsius)") == (
        "sea_water_temperature",
        "degree_Celsius",
    )
    assert tabular.split_units("sea_water_temperature") == (
        "sea_water_temperature",
        None,
    )


def test_split_units_reads_the_bracketed_spelling_too():
    """A mooring CSV's own convention: no space before the unit, brackets not parens."""
    assert tabular.split_units("Salinity_qc[PSU]") == ("Salinity_qc", "PSU")
    assert tabular.split_units("Time[days_since_1950-01-01T00:00:00Z]") == (
        "Time",
        "days_since_1950-01-01T00:00:00Z",
    )
    assert tabular.split_units("Instrument_SN") == ("Instrument_SN", None)


@pytest.mark.parametrize(
    "column",
    [
        "temperature(degC)",  # no space -- the nominal "(units)" convention needs one
        "temperature (degC)",  # the documented single space
        "temperature  (degC)",  # extra space must not leak into the returned name
        "temperature[degC]",
        "temperature [degC]",
        "temperature  [degC]",
    ],
)
def test_split_units_tolerates_any_amount_of_whitespace_before_the_unit(column):
    """Real files are not consistent about the single space either convention assumes.

    A stray extra space used to survive into the returned name (``"temperature "``),
    which then silently fails every downstream exact-match lookup.
    """
    assert tabular.split_units(column) == ("temperature", "degC")


def test_build_and_tabular_share_one_parser():
    """The catalog description and the reader must agree about the convention.

    ``build._probe_dataframe`` writes the ``standard_names``/``units`` maps that
    ``tabular.to_dataset`` later reads back, so a second copy of the parse is a second
    chance for them to disagree — this asserts they are the same code.
    """
    from ocean_skill import build

    md = build._probe_dataframe(papa_frame(renamed=False, z=0.0))
    assert "sea_water_temperature" in md["variables"]
    # QARTOD companions describe a variable rather than being one...
    assert not [v for v in md["variables"] if "qc" in v]
    # ...and coordinate columns are not variables either.
    assert "z" not in md["variables"]
    assert md["units"]["sea_water_temperature (degree_Celsius)"] == "degree_Celsius"

    # A frame in the bracketed, capitalized convention agrees the same way.
    md = build._probe_dataframe(ctd_frame())
    assert md["axes"] == {
        "T": "Time[UTC]",
        "X": "Longitude[degrees_east]",
        "Y": "Latitude[degrees_north]",
        "Z": "Depth[m]",
    }
    assert md["variables"] == ["Temperature"]


# -- coordinate-column recognition: names beyond the ERDDAP convention ----------------


def ctd_frame(n: int = 5):
    """A non-ERDDAP CTD profile's own convention: capitalized, ``[units]``-suffixed,
    with a WOCE-style ``_flag`` companion column -- the pair-shaped tabular source
    :mod:`ocean_skill.qc` is built around. Flag values cycle through the full WOCE
    bottle set (2 good, 3 questionable, 4 bad, 9 missing) so any test reading this
    fixture through a resolved qc contract sees a mix, not a single flat value.
    """
    time = pd.date_range("2024-04-04", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Time[UTC]": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Latitude[degrees_north]": np.full(n, 27.5),
            "Longitude[degrees_east]": np.full(n, -96.5),
            "Depth[m]": np.full(n, 28.0),
            "Temperature[degC]": np.linspace(20.0, 21.0, n),
            "Temperature_flag": np.resize([2, 3, 4, 9], n),
        }
    )


def mooring_frame(n: int = 5):
    """The POST-RECIPE mooring shape: build-time rename already applied (plain
    names, no ``_qc``/``_raw``/``_flag`` triplet survives -- see
    ``docs/seanoe_qc_recipes.md``'s "Moorings" recipe), with the provider's 9999
    fill sentinel on every numeric column, position included -- exactly what
    :func:`ocean_skill.qc.mask_fill_values` (via a ``qc={"fill_values": [9999.0]}``
    contract) exists to clean up, and what a mooring recipe needs with no flag
    machinery at all.
    """
    time = pd.date_range("2024-04-04", periods=n, freq="D", tz="UTC")
    lat = np.full(n, 27.5)
    lon = np.full(n, -96.5)
    temperature = np.linspace(9.0, 10.0, n)
    lat[-1] = lon[-1] = temperature[-1] = 9999.0
    return pd.DataFrame(
        {
            "Time[UTC]": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Latitude[degrees_north]": lat,
            "Longitude[degrees_east]": lon,
            "Temperature[degC]": temperature,
        }
    )


def test_probe_recognizes_bracket_coordinates():
    from ocean_skill import build

    md = build._probe_dataframe(ctd_frame())
    assert md["axes"] == {
        "T": "Time[UTC]",
        "X": "Longitude[degrees_east]",
        "Y": "Latitude[degrees_north]",
        "Z": "Depth[m]",
    }
    assert (md["geospatial_lon_min"], md["geospatial_lon_max"]) == (-96.5, -96.5)
    assert (md["geospatial_lat_min"], md["geospatial_lat_max"]) == (27.5, 27.5)
    assert (md["geospatial_vertical_min"], md["geospatial_vertical_max"]) == (
        28.0,
        28.0,
    )
    assert md["time_coverage_start"] == "2024-04-04"
    assert md["time_coverage_end"] == "2024-04-08"
    assert md["featureType"] == "timeSeries"
    # The coordinates and the flag column are not data variables.
    assert md["variables"] == ["Temperature"]


def test_to_dataset_reads_bracket_frame_without_axes():
    """No catalog ``axes`` at all — the regex fallback alone must find everything."""
    ds = tabular.to_dataset(ctd_frame(), {})
    assert set(ds.dims) == {"time"}
    assert float(ds["lon"]) == pytest.approx(-96.5)
    assert float(ds["lat"]) == pytest.approx(27.5)
    assert float(ds["depth"]) == pytest.approx(28.0)
    assert ds["Temperature"].attrs["units"] == "degC"
    assert "Temperature_flag" in ds.data_vars


def seanoe_frame(n: int = 5, *, fill: bool = True):
    """A SEANOE CTD-mooring CSV: time as *numbers* under a ``days_since`` unit, and a
    ``9999`` fill row (its "not reported" marker) in lon/lat/pressure.

    Reproduces the two bugs reported against the built catalog: ``time_coverage`` both
    reading 1970-01-01 (the raw day numbers taken as Unix nanoseconds), and
    ``geospatial_lon/lat/vertical_max`` reading 9999 (the fill value taken as a real
    extreme).
    """
    # 2025-04-28 .. 2025-05-02 as "days since 1950-01-01"
    day0 = (pd.Timestamp("2025-04-28") - pd.Timestamp("1950-01-01")).days
    days = np.arange(day0, day0 + n, dtype="float64")
    lon = np.full(n, -21.5248)
    lat = np.full(n, 64.382)
    pres = np.full(n, 12.966)
    if fill:
        lon[-1] = lat[-1] = pres[-1] = 9999.0
    return pd.DataFrame(
        {
            "Time[days_since_1950-01-01T00:00:00Z]": days,
            "Longitude[degrees_east]": lon,
            "Latitude[degrees_north]": lat,
            "Pressure[dbar]": pres,
            "Temperature[degree_C]": np.linspace(6.0, 7.0, n),
        }
    )


def test_decode_time_column_honors_days_since_in_the_name():
    """The column's own ``days_since`` unit, not Unix nanoseconds, drives decoding."""
    frame = seanoe_frame(fill=False)
    decoded = tabular.decode_time_column(
        frame["Time[days_since_1950-01-01T00:00:00Z]"],
        "Time[days_since_1950-01-01T00:00:00Z]",
    )
    assert decoded.min().date().isoformat() == "2025-04-28"
    assert decoded.max().date().isoformat() == "2025-05-02"


def test_decode_time_column_falls_back_to_plain_parsing():
    """A column with no CF unit still parses as before (ISO timestamp strings)."""
    s = pd.Series(["2024-04-04T00:00:00Z", "2024-04-05T00:00:00Z"])
    decoded = tabular.decode_time_column(s, "Time[UTC]")
    assert decoded.min().date().isoformat() == "2024-04-04"


def test_numeric_in_range_drops_out_of_range_fill_values():
    s = pd.Series([-21.5, 9999.0, -20.0])
    cleaned = tabular.numeric_in_range(s, "X")
    assert cleaned.dropna().tolist() == [-21.5, -20.0]
    # Latitude's tighter bound catches a value longitude would accept.
    assert tabular.numeric_in_range(pd.Series([64.0, 100.0]), "Y").dropna().tolist() == [
        64.0
    ]


def test_probe_seanoe_frame_gets_real_times_and_extents():
    """The end-to-end reproduction: build's probe must not report 1970 or 9999."""
    from ocean_skill import build

    md = build._probe_dataframe(seanoe_frame())
    assert md["time_coverage_start"] == "2025-04-28"
    assert md["time_coverage_end"] == "2025-05-02"
    assert md["geospatial_lon_max"] == pytest.approx(-21.5248)
    assert md["geospatial_lat_max"] == pytest.approx(64.382)
    assert md["geospatial_vertical_max"] == pytest.approx(12.966)
    assert md["lon_convention"] == "-180-180"
    # A fixed mooring with a lone fill row is still a timeSeries, not a trajectory.
    assert md["featureType"] == "timeSeries"


def test_to_dataset_seanoe_frame_reads_times_and_ignores_fill():
    """to_dataset must agree with the probe: real dates, fixed position, no 9999."""
    ds = tabular.to_dataset(seanoe_frame(), {})
    assert pd.Timestamp(ds["time"].values.min()).date().isoformat() == "2025-04-28"
    assert pd.Timestamp(ds["time"].values.max()).date().isoformat() == "2025-05-02"
    assert float(ds["lon"]) == pytest.approx(-21.5248)
    assert float(ds["lat"]) == pytest.approx(64.382)
    assert float(ds["depth"]) == pytest.approx(12.966)


def test_to_dataset_masks_in_place_and_carries_flag_attrs_with_a_contract():
    """The contract-carrying sibling of the metadata-less pin just above.

    Same frame, but with a resolved qc contract in ``meta`` -- questionable/bad/
    missing rows are masked to NaN *in place* on ``Temperature`` (no new column,
    no rename), the flag column itself survives unmasked, and both carry the CF
    attrs :func:`ocean_skill.qc.resolve_contract`/:func:`ocean_skill.tabular
    .to_dataset` are meant to produce.
    """
    from ocean_skill import qc

    frame = ctd_frame()
    contract = qc.resolve_contract({"scheme": "woce_bottle"}, frame)
    meta = {"qc": contract}
    applied = qc.apply(frame, meta)
    ds = tabular.to_dataset(applied, meta)

    # flags = [2, 3, 4, 9, 2] -- keep=[GOOD] masks everything but the two 2s.
    assert ds["Temperature"].isnull().values.tolist() == [False, True, True, True, False]
    assert set(ds["Temperature_flag"].values.tolist()) == {2, 3, 4, 9}

    assert ds["Temperature"].attrs["ancillary_variables"] == "Temperature_flag"
    assert "qc_policy" in ds["Temperature"].attrs
    flag_attrs = ds["Temperature_flag"].attrs
    assert flag_attrs["flags_for"] == "Temperature"
    assert "flag_values" in flag_attrs and "flag_meanings" in flag_attrs


def test_probe_masks_a_moorings_9999_fill_before_extents_and_featuretype():
    """The mooring shape: no flags, only ``fill_values`` -- still needs masking.

    Without the fix, the poisoned last row would push the lat/lon extent to 9999
    and misclassify this fixed station as a moving trajectory.
    """
    from ocean_skill import build

    md = build._probe_dataframe(mooring_frame(), qc={"fill_values": [9999.0]})
    assert (md["geospatial_lat_min"], md["geospatial_lat_max"]) == (27.5, 27.5)
    assert (md["geospatial_lon_min"], md["geospatial_lon_max"]) == (-96.5, -96.5)
    assert md["featureType"] == "timeSeries"
    assert "qc" in md and md["qc"]["fill_values"] == [9999.0]
    assert md["qc"]["flags"] == {}  # no flag columns in a post-recipe mooring frame


def test_apply_masks_a_moorings_9999_fill_on_read_with_no_flags_involved():
    from ocean_skill import qc

    frame = mooring_frame()
    contract = qc.resolve_contract({"fill_values": [9999.0]}, frame)
    out = qc.apply(frame, {"qc": contract, "axes": {"T": "Time[UTC]"}})
    assert np.isnan(out["Latitude[degrees_north]"].iloc[-1])
    assert np.isnan(out["Longitude[degrees_east]"].iloc[-1])
    assert np.isnan(out["Temperature[degC]"].iloc[-1])
    # the time column itself is exempt from fill masking
    assert out["Time[UTC]"].notna().all()


def test_depth_of_recognizes_bracket_and_pressure():
    direct = pd.DataFrame({"Depth[m]": np.full(3, 28.0)})
    depth, source, approximate = tabular.depth_of(direct, {})
    assert depth == pytest.approx(28.0)
    assert source == "depth"
    assert approximate is False

    pressure_only = pd.DataFrame({"Pressure[dbar]": np.full(3, 28.2)})
    depth, source, approximate = tabular.depth_of(pressure_only, {})
    assert depth == pytest.approx(28.2)
    assert source == "sea_water_pressure"
    assert approximate is True

    both = pd.DataFrame(
        {"Depth[m]": np.full(3, 28.0), "PRES[dbar]": np.full(3, 28.2)}
    )
    depth, source, _ = tabular.depth_of(both, {})
    assert source == "depth"
    assert depth == pytest.approx(28.0)


@pytest.mark.parametrize(
    "columns",
    [
        ["month", "day_of_year", "year", "latency", "lateral_velocity"],
        ["along", "longwave", "zone", "zooplankton", "flagellate_abundance"],
    ],
)
def test_coordinate_matcher_refuses_lookalikes(columns):
    """Words that contain a coordinate word as a substring are not the coordinate."""
    from ocean_skill import build

    frame = pd.DataFrame({c: np.arange(3) for c in columns})
    for axis in ("T", "X", "Y", "Z"):
        assert tabular.coord_column(frame, axis) is None
    md = build._probe_dataframe(frame)
    assert set(md.get("variables", [])) == set(columns)


def test_a_lone_datetime_column_is_time_by_dtype():
    frame = pd.DataFrame(
        {
            "sample": pd.date_range("2024-01-01", periods=3, freq="D"),
            "temperature": [10.0, 10.5, 11.0],
        }
    )
    assert tabular.coord_column(frame, "T") == "sample"


def test_two_datetime_columns_refuse_to_guess_time():
    frame = pd.DataFrame(
        {
            "deployed": pd.date_range("2024-01-01", periods=3, freq="D"),
            "recovered": pd.date_range("2024-02-01", periods=3, freq="D"),
        }
    )
    assert tabular.coord_column(frame, "T") is None


def test_a_flag_column_is_a_qc_companion():
    assert tabular.is_qc_column("Temperature_flag")
    assert tabular.is_qc_column("QC_Flag[1]")
    assert not tabular.is_qc_column("flagellate_abundance")


def test_depth_of_column_search_is_case_insensitive():
    frame = pd.DataFrame({"DEPTH": np.full(3, 12.0)})
    depth, source, _ = tabular.depth_of(frame, {})
    assert depth == pytest.approx(12.0)
    assert source == "depth"


# -- units, variables, coordinates ----------------------------------------------------


def test_units_come_from_the_column_suffix():
    ds = tabular.to_dataset(papa_frame(renamed=False), {"axes": PAPA_META["axes"]})
    assert ds["sea_water_temperature"].attrs["units"] == "degree_Celsius"


def test_units_come_from_the_entry_metadata_when_the_column_was_renamed():
    """The half-renamed frame: bare data columns, units known only to the catalog."""
    frame = papa_frame(renamed=True)
    assert "sea_water_temperature" in frame.columns  # no suffix left to read
    ds = tabular.to_dataset(frame, PAPA_META)
    assert ds["sea_water_temperature"].attrs["units"] == "degree_Celsius"


def test_time_is_the_only_dimension_and_position_is_scalar():
    ds = tabular.to_dataset(papa_frame(), PAPA_META)
    assert set(ds.dims) == {"time"}
    assert ds["lon"].shape == () and ds["lat"].shape == ()
    assert float(ds["lon"]) == pytest.approx(-144.245227)


def test_qc_columns_are_kept_but_coordinate_columns_are_not_variables():
    """Flags are information (``qc()`` will want them); ``z`` is not a measurement."""
    ds = tabular.to_dataset(papa_frame(), PAPA_META)
    assert "sea_water_temperature_qc_agg" in ds.data_vars
    assert "z" not in ds.data_vars


def test_a_non_numeric_column_never_becomes_a_variable():
    """A bare ``.mean()`` over a string column raises, so the real assertion computes.

    Same failure ROMS' ``spherical`` flag causes on the gridded side: it is not enough
    for the column to be absent from ``data_vars``, the reduction has to work.
    """
    frame = papa_frame()
    frame["station"] = "GP03FLMA"
    ds = tabular.to_dataset(frame, PAPA_META)
    assert "station" not in ds.data_vars
    assert ds.attrs["station"] == "GP03FLMA"
    assert float(ds.mean()["sea_water_temperature"]) == pytest.approx(
        float(ds["sea_water_temperature"].mean())
    )


# -- time ------------------------------------------------------------------------------


def test_time_is_timezone_naive_and_survives_the_cache():
    """Tz-awareness breaks the *cache*, and the cache only warns when it fails.

    A ``datetime64[us, UTC]`` coordinate cannot be written to zarr; since
    ``cache.save`` treats a failed write as a warning, a tz-aware lane would silently
    re-read the whole remote record on every run. So the round trip is the assertion,
    not the dtype alone.
    """
    from ocean_skill import cache

    ds = tabular.to_dataset(papa_frame(), PAPA_META)
    assert np.issubdtype(ds["time"].dtype, np.datetime64)
    assert ds["time"].attrs["time_zone"] == "UTC"

    cache.save_field("test-tabular", ds["sea_water_temperature"], None)
    hit, _ = cache.load_field("test-tabular")
    assert hit is not None, "a tz-aware time coordinate silently defeats the cache"
    assert hit.sizes["time"] == ds.sizes["time"]


def test_duplicate_timestamps_are_collapsed_with_a_count():
    """Overlapping deployments repeat timestamps, which makes a later align raise."""
    frame = pd.concat([papa_frame(4), papa_frame(4)], ignore_index=True)
    with pytest.warns(UserWarning, match="4 duplicate timestamps"):
        ds = tabular.to_dataset(frame, PAPA_META)
    assert ds.sizes["time"] == 4
    assert ds.indexes["time"].is_monotonic_increasing


def test_unparseable_timestamps_are_dropped():
    frame = papa_frame(4)
    frame.loc[2, "time (UTC)"] = "not a time"
    ds = tabular.to_dataset(frame, PAPA_META)
    assert ds.sizes["time"] == 3


def test_a_table_without_a_time_column_says_so():
    frame = papa_frame(4).drop(columns=["time (UTC)"])
    with pytest.raises(ValueError, match="no time column"):
        tabular.to_dataset(frame, {})


# -- depth: one rung of the resolution order per test ---------------------------------


def test_depth_prefers_a_column_with_real_values():
    ds = tabular.to_dataset(papa_frame(depth_reading=30.0, pressure=30.4), PAPA_META)
    assert float(ds["depth"]) == pytest.approx(30.0)
    assert ds.attrs["depth_source"] == "depth_reading"
    assert ds.attrs["depth_approximate"] is False


def test_an_all_nan_column_falls_through_to_pressure():
    """The OOI case: ``depth_reading`` is all-NaN, so pressure is the only reading."""
    ds = tabular.to_dataset(papa_frame(pressure=33.9), PAPA_META)
    assert float(ds["depth"]) == pytest.approx(33.9)
    assert ds.attrs["depth_source"] == "sea_water_pressure"
    assert ds.attrs["depth_approximate"] is True


def test_a_flat_zero_column_is_a_placeholder_not_a_reading():
    """OOI writes ``z = 0.0`` for an instrument tens of metres down.

    Believing that zero is how a mid-water instrument comes to be compared as a surface
    one — the whole reason this comparison warns about depth at all.
    """
    with pytest.warns(UserWarning, match="placeholder"):
        ds = tabular.to_dataset(papa_frame(z=0.0, pressure=33.9), PAPA_META)
    assert ds.attrs["depth_source"] == "sea_water_pressure"
    assert float(ds["depth"]) == pytest.approx(33.9)


def test_two_readings_that_disagree_are_both_reported():
    with pytest.warns(UserWarning, match="depth_reading says 12 m"):
        ds = tabular.to_dataset(
            papa_frame(depth_reading=12.0, pressure=33.9), PAPA_META
        )
    assert ds.attrs["depth_source"] == "depth_reading"


def test_no_column_falls_through_to_a_metadata_attribute():
    meta = {**PAPA_META, "nominal_depth_m": 30.0}
    ds = tabular.to_dataset(papa_frame(pressure=None, depth_reading=None), meta)
    assert float(ds["depth"]) == pytest.approx(30.0)
    assert ds.attrs["depth_source"] == "metadata:nominal_depth_m"


def test_nothing_at_all_assumes_the_surface_and_says_so():
    """An assumption the user sees when it is made, rather than infers from a bias."""
    with pytest.warns(UserWarning, match="Assuming the surface"):
        ds = tabular.to_dataset(
            papa_frame(pressure=None, depth_reading=None), PAPA_META
        )
    assert "depth" not in ds.coords
    assert ds.attrs["depth_source"] == "assumed-surface"


def test_a_varying_depth_stays_on_time_and_describes_its_spread():
    """Papa's flanking moorings span deployments at ~9 m and ~34 m in one record."""
    pressure = np.r_[np.full(48, 9.2), np.full(48, 34.1)]
    with pytest.warns(UserWarning, match="depth is not constant"):
        ds = tabular.to_dataset(papa_frame(pressure=pressure), PAPA_META)
    assert ds["depth"].dims == ("time",)
    assert "depth" not in ds.dims  # a coordinate, never an axis to reduce


# -- position --------------------------------------------------------------------------


def test_a_moving_platform_is_reported_rather_than_averaged():
    lon = np.linspace(-144.3, -143.9, 96)
    with pytest.warns(UserWarning, match="trajectory"):
        tabular.to_dataset(papa_frame(lon=lon), PAPA_META)


def test_the_depth_rides_on_the_variables_attrs_as_well_as_a_coordinate():
    """A coordinate along time does not survive a reduction; the attrs do.

    Resampling a mooring to monthly means — which a comparison against a monthly product
    requires — drops a ``(time,)`` depth coordinate, and the depth would then be
    invisible to the caveat about comparing it against a surface field, which is exactly
    the record that needs it.
    """
    from ocean_skill import operators

    pressure = np.r_[np.full(48, 9.2), np.full(48, 34.1)]
    with pytest.warns(UserWarning, match="depth is not constant"):
        ds = tabular.to_dataset(papa_frame(pressure=pressure), PAPA_META)
    temperature = ds["sea_water_temperature"]
    assert temperature.attrs["depth_m"] == pytest.approx(np.median(pressure))
    assert temperature.attrs["depth_range_m"] == pytest.approx((9.2, 34.1))

    monthly = operators.aggregate(
        temperature, {"time": {"resample": "MS", "reduce": "mean"}}
    )
    assert "depth" not in monthly.coords, "the fixture no longer poses the question"
    assert monthly.attrs["depth_m"] == pytest.approx(np.median(pressure))


# -- profile: one instant, many depths, indexed on depth rather than time -------------


def profile_frame(n: int = 5, *, depth=None, time=None):
    """A single CTD cast: fixed position, one instant, varying depth."""
    if depth is None:
        depth = np.linspace(0.0, 50.0, n)
    if time is None:
        time = np.full(n, "2024-04-04T12:00:00Z")
    return pd.DataFrame(
        {
            "Time[UTC]": time,
            "Latitude[degrees_north]": np.full(n, 27.5),
            "Longitude[degrees_east]": np.full(n, -96.5),
            "Depth[m]": depth,
            "Temperature[degC]": np.linspace(20.0, 15.0, n),
            "Salinity[PSU]": np.linspace(35.0, 35.5, n),
        }
    )


def test_probe_detects_a_profile():
    """One instant, fixed position, depth varying -- the unambiguous cast shape."""
    from ocean_skill import build

    md = build._probe_dataframe(profile_frame())
    assert md["featureType"] == "profile"
    assert md["featureType_source"] == "inferred"


def test_probe_detects_timeseriesprofile_when_both_time_and_depth_vary():
    """A fixed station revisited more than once: both axes vary, unlike a single cast."""
    from ocean_skill import build

    time = pd.date_range("2024-04-01", periods=5, freq="D", tz="UTC").strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    md = build._probe_dataframe(profile_frame(time=time))
    assert md["featureType"] == "timeSeriesProfile"


def test_to_dataset_builds_a_depth_indexed_profile():
    """featureType: profile routes to the depth build, not the time build."""
    ds = tabular.to_dataset(profile_frame(), {"featureType": "profile"})
    assert set(ds.dims) == {"depth"}
    assert ds["time"].shape == () and ds["lon"].shape == () and ds["lat"].shape == ()
    assert ds["Temperature"].dims == ("depth",)
    assert ds["Temperature"].attrs["units"] == "degC"
    assert float(ds["lon"]) == pytest.approx(-96.5)
    assert ds.indexes["depth"].is_monotonic_increasing


def test_profile_duplicate_depths_are_collapsed_with_a_warning():
    """A mislabeled repeat-visit station is readable but flagged, not refused."""
    depth = np.array([0.0, 0.0, 10.0, 20.0, 30.0])
    with pytest.warns(UserWarning, match="1 duplicate depths"):
        ds = tabular.to_dataset(profile_frame(depth=depth), {"featureType": "profile"})
    assert ds.sizes["depth"] == 4
    assert ds.indexes["depth"].is_monotonic_increasing


def test_a_profile_with_no_depth_column_says_so():
    frame = profile_frame().drop(columns=["Depth[m]"])
    with pytest.raises(ValueError, match="no depth column"):
        tabular.to_dataset(frame, {"featureType": "profile"})


def test_a_profile_with_time_varying_across_the_cast_is_reported():
    """Depths sampled seconds apart during a real cast carry distinct timestamps --
    warn and use the earliest rather than refuse."""
    time = [
        "2024-04-04T12:00:00Z",
        "2024-04-04T12:00:05Z",
        "2024-04-04T12:00:10Z",
        "2024-04-04T12:00:15Z",
        "2024-04-04T12:00:20Z",
    ]
    with pytest.warns(UserWarning, match="time varies across the cast"):
        ds = tabular.to_dataset(profile_frame(time=time), {"featureType": "profile"})
    assert pd.Timestamp(ds["time"].values) == pd.Timestamp("2024-04-04T12:00:00")
