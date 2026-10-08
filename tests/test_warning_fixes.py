"""Warnings that were wrong or noise on real Iceland notebooks, and the fixes.

One test group per fix: the ``degree_C`` unit spelling, the GPS-wobble distance in
metres, the not-constant-depth warning on a declared mooring depth, cf-xarray's
dangling-``ancillary_variables`` warning, the free-surface advice for a declared
fixed-origin depth, and the ``cell_methods`` a ROMS file's time semantics become.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import roms, tabular, units
from ocean_skill.align import is_composite

# -- 1. degree_C -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling", ["degree_C", "deg_C", "degrees_C", "degC", "Celsius", "degree_Celsius"]
)
def test_every_celsius_spelling_parses(spelling):
    assert units.parse(spelling) is not None


def test_seanoe_degree_c_is_compatible_with_celsius():
    assert units.compatible("Celsius", "degree_C") is True


# -- 2. GPS wobble in metres, and the source name --------------------------------------


def _wobbly_station(lon, lat, n=6):
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n, freq="MS").astype(str),
            "depth (m)": [1, 5] * (n // 2),
            "lon": lon,
            "lat": lat,
            "Temperature (degC)": np.linspace(5, 8, n),
        }
    )


def _wobble_message(frame, meta):
    with pytest.warns(UserWarning, match="GPS/positioning wobble") as record:
        tabular.to_dataset(frame, meta)
    return next(
        str(w.message) for w in record if "GPS/positioning wobble" in str(w.message)
    )


def test_wobble_distance_shrinks_east_west_with_latitude():
    # 0.03 deg of longitude at 64N is 0.03 * 111 km * cos(64) = ~1.46 km, not 3.3 km
    lon = [-22.0, -21.97, -22.0, -21.98, -22.0, -21.99]
    frame = _wobbly_station(lon, [64.0] * 6)
    message = _wobble_message(
        frame, {"featureType": "timeSeriesProfile", "datasetID": "stn"}
    )
    metres = int(re.search(r"up to ~(\d+) m", message).group(1))
    assert metres == pytest.approx(0.03 * 111_000 * np.cos(np.radians(64.0)), abs=2)
    assert "0.0300" in message


def test_wobble_distance_takes_the_larger_of_the_two_ranges():
    lat = [64.0, 64.01, 64.0, 64.005, 64.0, 64.0]
    frame = _wobbly_station([-22.0, -21.999] * 3, lat)
    message = _wobble_message(frame, {"featureType": "timeSeriesProfile"})
    metres = int(re.search(r"up to ~(\d+) m", message).group(1))
    assert metres == pytest.approx(0.01 * 111_000, abs=2)


def test_the_warning_names_the_entry_when_the_reader_stamped_it():
    frame = _wobbly_station([-22.0, -21.97, -22.0, -21.98, -22.0, -21.99], [64.0] * 6)
    frame.attrs["source_name"] = "iceland_station"
    message = _wobble_message(frame, {"featureType": "timeSeriesProfile"})
    assert message.startswith("iceland_station:")


def test_subject_prefers_dataset_id_then_title_then_entry_name():
    frame = pd.DataFrame({"a": [1]})
    frame.attrs["source_name"] = "entry"
    assert tabular._subject_of({"datasetID": "id", "title": "t"}, frame) == "id"
    assert tabular._subject_of({"title": "t"}, frame) == "t"
    assert tabular._subject_of({}, frame) == "entry"
    assert tabular._subject_of({}, pd.DataFrame({"a": [1]})) == "this source"
    assert tabular._subject_of(None) == "this source"


# -- 3. depth is not constant, unless the entry declares one ---------------------------


def _tidal_pressure_mooring(n=48):
    time = pd.date_range("2024-01-01", periods=n, freq="h")
    pressure = 27.8 + 2.5 * np.sin(np.linspace(0, 8 * np.pi, n))  # a ~5 m tidal swing
    return pd.DataFrame(
        {
            "time": time.astype(str),
            "lon": -22.0,
            "lat": 64.0,
            "Pressure (dbar)": pressure,
            "Temperature (degC)": np.linspace(6, 7, n),
        }
    )


def _depth_warnings(meta):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = tabular.to_dataset(_tidal_pressure_mooring(), meta)
    messages = [str(w.message) for w in caught]
    return ds, [m for m in messages if "depth is not constant" in m]


def test_an_undeclared_varying_depth_still_warns():
    ds, caught = _depth_warnings({"featureType": "timeSeries", "datasetID": "m"})
    assert len(caught) == 1
    assert ds["depth"].dims == ("time",)


def test_a_declared_nominal_depth_silences_the_not_constant_warning():
    meta = {
        "featureType": "timeSeries",
        "datasetID": "m",
        "nominal_depth_m": 27.8,
        "depth_convention": {"origin": "fixed"},
    }
    ds, caught = _depth_warnings(meta)
    assert caught == []
    assert ds["depth"].dims == ("time",)  # the pressure-derived depth is still kept


# -- 4. cf-xarray's dangling ancillary_variables ---------------------------------------


def _qcd_dataset():
    temp = xr.DataArray(
        np.arange(3.0),
        dims="time",
        attrs={
            "standard_name": "sea_water_temperature",
            "units": "degC",
            "ancillary_variables": "TEMP_flag",
        },
    )
    return xr.Dataset({"TEMP": temp}, coords={"time": pd.date_range("2024", periods=3)})


def test_cf_lookup_is_quiet_about_a_dropped_flag_variable():
    from ocean_skill.cf import find_coord

    ds = _qcd_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert find_coord(ds, "time") is not None


def test_units_lookups_are_quiet_about_a_dropped_flag_variable():
    ds = _qcd_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert units._cf_name(ds, "temperature", "sea_water_temperature") == "TEMP"
        units._warn_if_only_a_flag_matched(ds, "sea_water_temperature")


def test_without_the_helper_cf_xarray_does_warn():
    """Guards the tests above: the warning they expect to be gone does exist."""
    import cf_xarray  # noqa: F401

    with pytest.warns(UserWarning, match="referred to in the CF attributes"):
        _qcd_dataset().cf["sea_water_temperature"]


def test_the_quiet_helper_lets_other_warnings_through():
    from ocean_skill.cf import quiet_dropped_ancillaries

    with pytest.warns(UserWarning, match="something else"), quiet_dropped_ancillaries():
        warnings.warn("something else", UserWarning)


# -- 5. the free-surface advice only when nothing declared the convention --------------


def _above_surface_messages(convention):
    from tests._tidal_roms import tidal_roms

    ds, meta = tidal_roms()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        roms.nearest_depth_levels(ds, meta, 1.0, convention=convention)
    messages = [str(w.message) for w in caught]
    return [m for m in messages if "above the free surface" in m]


def test_a_declared_fixed_origin_does_not_get_the_surface_advice():
    assert _above_surface_messages({"origin": "fixed", "source": "declared"}) == []
    assert _above_surface_messages({"origin": "fixed", "source": "inferred"}) == []


def test_an_undeclared_fixed_origin_still_does():
    assert len(_above_surface_messages({"origin": "fixed"})) == 1
    assert len(_above_surface_messages({"origin": "fixed", "source": "default"})) == 1


# -- 6. ROMS time semantics -> cell_methods --------------------------------------------


def _raw_roms(file_type=None, avg=False, existing=None):
    from tests.test_roms_classic_depth import _classic

    raw = _classic()
    if file_type is not None:
        raw.attrs["type"] = file_type
    if avg:
        raw["salt"] = raw["salt"].assign_attrs(long_name="avg_salt")
    if existing is not None:
        raw["temp"] = raw["temp"].assign_attrs(cell_methods=existing)
    return raw


def _standardized(**kwargs):
    from tests.test_roms_classic_depth import META

    return roms.standardize(_raw_roms(**kwargs), META)


def test_a_history_file_is_instantaneous():
    ds = _standardized(file_type="ROMS history file")
    for name in ("temp", "salt", "zeta"):
        assert ds[name].attrs["cell_methods"] == "time: point"
        assert is_composite(ds[name], "time") is False


def test_avg_long_names_are_period_means_whatever_the_file_type():
    ds = _standardized(avg=True)
    assert ds["salt"].attrs["cell_methods"] == "time: mean"
    assert is_composite(ds["salt"], "time") is True
    assert "cell_methods" not in ds["temp"].attrs  # nothing says what this one is


def test_an_averages_file_is_a_mean():
    ds = _standardized(file_type="ROMS average file")
    assert ds["temp"].attrs["cell_methods"] == "time: mean"


def test_an_existing_cell_methods_is_never_overwritten():
    ds = _standardized(file_type="ROMS history file", existing="time: mean")
    assert ds["temp"].attrs["cell_methods"] == "time: mean"
    assert ds["salt"].attrs["cell_methods"] == "time: point"


def test_a_file_that_says_nothing_is_left_alone():
    ds = _standardized()
    assert all("cell_methods" not in ds[name].attrs for name in ds.data_vars)


def test_the_cell_methods_survive_a_time_slice_and_a_depth_slice():
    ds = _standardized(file_type="ROMS history file")
    sliced = ds["temp"].isel(time=0, s_rho=0)
    assert sliced.attrs["cell_methods"] == "time: point"


# -- 7. declared units follow a column through the standard_names rename ---------------

ALK = "sea_water_alkalinity_expressed_as_mole_equivalent"


def _renamed_bottle_frame():
    """Return a frame as ``sources.read`` leaves it: ``TA`` renamed, Silicate not."""
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=4, freq="D").astype(str),
            "lon": -22.0,
            "lat": 64.0,
            "depth (m)": 5.0,
            ALK: [2300.0, 2301.0, 2302.0, 2303.0],
            "Silicate": [4.0, 4.1, 4.2, 4.3],
        }
    )


BOTTLE_META = {
    "featureType": "timeSeries",
    "datasetID": "bottle",
    "units": {"TA": "umol/kg", "Silicate": "umol/L"},
    "standard_names": {"TA": ALK},
}


def test_declared_units_follow_a_renamed_column():
    ds = tabular.to_dataset(_renamed_bottle_frame(), BOTTLE_META)
    assert ds[ALK].attrs["units"] == "umol/kg"
    assert ds["Silicate"].attrs["units"] == "umol/L"


def test_a_suffix_units_survive_the_rename_and_the_columns_own_suffix_wins():
    meta = {**BOTTLE_META, "units": {"TA": "mmol/kg"}}
    meta["standard_names"] = {"TA (umol/kg)": ALK}
    ds = tabular.to_dataset(_renamed_bottle_frame(), meta)
    assert ds[ALK].attrs["units"] == "umol/kg"
    frame = _renamed_bottle_frame().rename(columns={ALK: "TA (mmol/kg)"})
    assert tabular._units_map(frame, {"units": {"TA": "x"}})["TA (mmol/kg)"] == (
        "mmol/kg"
    )


def test_units_are_not_moved_onto_a_column_whose_rename_was_skipped():
    frame = _renamed_bottle_frame()
    frame["TA"] = 1.0  # the source has its own TA: sources.read skips the rename
    umap = tabular._units_map(frame, BOTTLE_META)
    assert umap["TA"] == "umol/kg"  # its own declared units, as before
    assert ALK not in umap


def test_the_first_claimant_of_a_target_wins():
    meta = {
        "units": {"A": "m", "B": "s"},
        "standard_names": {"A": "target", "B": "target"},
    }
    frame = pd.DataFrame({"target": [1.0]})
    assert tabular._units_map(frame, meta)["target"] == "m"
