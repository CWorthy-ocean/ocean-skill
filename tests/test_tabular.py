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
