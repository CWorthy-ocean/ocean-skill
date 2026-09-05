"""``ComparisonSet.average``: pooling stations together after ``compare()`` fans them out.

Builds aligned pairs by hand and injects them as ``Comparison._aligned``, mirroring
``tests/test_subtract_mean.py`` (gridded) and ``tests/test_profile_comparison.py``
(point/station) -- no catalog entry needs to exist for any of the station or model
names used here since ``Comparison.align`` is never called.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.comparison import Comparison, ComparisonSet

TEMPERATURE = "sea_water_potential_temperature"
SALINITY = "sea_water_practical_salinity"


def _series_pair(
    times, values_ref, values_test, *, lon: float, lat: float
) -> xr.Dataset:
    """A station's aligned pair: one position, a time axis, test/reference/difference."""
    reference = xr.DataArray(
        np.asarray(values_ref, dtype=float), dims=("time",), coords={"time": times}
    ).assign_coords(lon=lon, lat=lat)
    reference.attrs["units"] = "degC"
    test = xr.DataArray(
        np.asarray(values_test, dtype=float), dims=("time",), coords={"time": times}
    ).assign_coords(lon=lon, lat=lat)
    test.attrs["units"] = "degC"
    return xr.Dataset(
        {
            "test": test.rename("test"),
            "reference": reference.rename("reference"),
            "difference": (test - reference).rename("difference"),
        },
        attrs={"station_lon": lon, "station_lat": lat, "point_method": "nearest"},
    )


def _station_comparison(
    *,
    reference: str,
    test: str,
    variable: str,
    aligned: xr.Dataset,
    select: dict | None = None,
) -> Comparison:
    c = Comparison(
        reference=reference, test=test, variable=variable, select=select, cache=False
    )
    c._aligned = aligned.copy(deep=True)
    return c


TIMES = pd.date_range("2024-01-01", periods=4, freq="D")


def _two_station_set() -> ComparisonSet:
    """2 stations (HV1, HV5) x 2 variables (temperature, salinity), same time axis."""
    hv1_temp = _series_pair(TIMES, [10, 11, 12, 13], [10.5, 11.2, 12.1, 13.4], lon=-150.0, lat=20.0)
    hv5_temp = _series_pair(TIMES, [14, 15, 16, 17], [14.4, 15.3, 16.2, 17.1], lon=-152.0, lat=22.0)
    hv1_salt = _series_pair(TIMES, [34, 34.1, 34.2, 34.3], [34.1, 34.2, 34.3, 34.4], lon=-150.0, lat=20.0)
    hv5_salt = _series_pair(TIMES, [35, 35.1, 35.2, 35.3], [35.2, 35.3, 35.4, 35.5], lon=-152.0, lat=22.0)
    return ComparisonSet(
        [
            _station_comparison(reference="HV1", test="his", variable=TEMPERATURE, aligned=hv1_temp),
            _station_comparison(reference="HV5", test="his", variable=TEMPERATURE, aligned=hv5_temp),
            _station_comparison(reference="HV1", test="his", variable=SALINITY, aligned=hv1_salt),
            _station_comparison(reference="HV5", test="his", variable=SALINITY, aligned=hv5_salt),
        ]
    )


# -- basic grouping and test/reference separation ---------------------------------------


def test_average_by_variable_groups_stations_into_one_series_each():
    pooled = _two_station_set()
    averaged = pooled.average(by="variable")

    assert isinstance(averaged, ComparisonSet)
    assert len(averaged) == 2
    variables = {c.standard_name for c in averaged}
    assert variables == {TEMPERATURE, SALINITY}


def test_average_keeps_test_and_reference_separate():
    """test averages only with test, reference only with reference -- never mixed."""
    pooled = _two_station_set()
    averaged = pooled.average(by="variable")

    temp = next(c for c in averaged if c.standard_name == TEMPERATURE)
    hv1, hv5 = pooled.comparisons[0], pooled.comparisons[1]

    expected_test = (hv1.aligned["test"] + hv5.aligned["test"]) / 2
    expected_reference = (hv1.aligned["reference"] + hv5.aligned["reference"]) / 2

    np.testing.assert_allclose(temp.aligned["test"].values, expected_test.values)
    np.testing.assert_allclose(
        temp.aligned["reference"].values, expected_reference.values
    )


def test_average_recomputes_difference_rather_than_averaging_it():
    pooled = _two_station_set()
    averaged = pooled.average(by="variable")
    temp = next(c for c in averaged if c.standard_name == TEMPERATURE)

    np.testing.assert_allclose(
        temp.aligned["difference"].values,
        (temp.aligned["test"] - temp.aligned["reference"]).values,
    )


def test_average_names_the_composite_reference_and_test():
    pooled = _two_station_set()
    averaged = pooled.average(by="variable")
    temp = next(c for c in averaged if c.standard_name == TEMPERATURE)

    assert temp.reference_name == "HV1+HV5"
    assert temp.test_name == "his"


def test_average_reports_a_composite_station_position():
    pooled = _two_station_set()
    averaged = pooled.average(by="variable")
    temp = next(c for c in averaged if c.standard_name == TEMPERATURE)

    assert temp.aligned.attrs["station_lon"] == pytest.approx(-151.0)
    assert temp.aligned.attrs["station_lat"] == pytest.approx(21.0)


# -- mismatched axes ----------------------------------------------------------------


def test_average_unions_mismatched_time_axes_with_skipna():
    """A timestamp only one station sampled keeps that station's own value."""
    common = TIMES[:3]
    extra = TIMES  # HV5 has one extra day HV1 lacks
    hv1 = _series_pair(common, [10, 11, 12], [10, 11, 12], lon=-150.0, lat=20.0)
    hv5 = _series_pair(extra, [20, 21, 22, 23], [20, 21, 22, 23], lon=-152.0, lat=22.0)
    pooled = ComparisonSet(
        [
            _station_comparison(reference="HV1", test="his", variable=TEMPERATURE, aligned=hv1),
            _station_comparison(reference="HV5", test="his", variable=TEMPERATURE, aligned=hv5),
        ]
    )
    averaged = pooled.average(by="variable")
    temp = averaged.comparisons[0]

    # First 3 days: mean of both stations' reference values.
    np.testing.assert_allclose(
        temp.aligned["reference"].sel(time=common).values, [(10 + 20) / 2, (11 + 21) / 2, (12 + 22) / 2]
    )
    # The extra day: only HV5 has a value, so skipna gives HV5's own value.
    assert temp.aligned["reference"].sel(time=extra[-1]).item() == pytest.approx(23)


# -- group of one --------------------------------------------------------------------


def test_average_of_a_single_member_group_is_a_no_op():
    hv1_temp = _series_pair(TIMES, [10, 11, 12, 13], [10.5, 11.2, 12.1, 13.4], lon=-150.0, lat=20.0)
    c = _station_comparison(reference="HV1", test="his", variable=TEMPERATURE, aligned=hv1_temp)
    pooled = ComparisonSet([c])
    averaged = pooled.average(by="variable")

    assert len(averaged) == 1
    np.testing.assert_allclose(
        averaged.comparisons[0].aligned["reference"].values, c.aligned["reference"].values
    )
    assert averaged.comparisons[0].reference_name == "HV1"


# -- depth bands, pooled -------------------------------------------------------------


def test_pooled_depth_bands_keep_their_range_label_after_averaging():
    """The exact end-to-end shape ``compare(depths=[bands]).average(by=["variable",
    "depth"])`` produces: three bands, two stations each, pooled per band -- and each
    pooled comparison's ``as_item()`` must still say *which* band it is, even though
    ``.average()`` (:meth:`Comparison._averaged`) sets ``_actual_depth`` to ``None``.
    """
    bands = [{"min": 0, "max": 5}, {"min": 10, "max": 15}, {"min": 30, "max": 40}]
    comparisons = []
    for i, band in enumerate(bands):
        for station, lon, lat in (("HV1", -150.0, 20.0), ("HV5", -152.0, 22.0)):
            values = [10 + i, 11 + i, 12 + i, 13 + i]
            aligned = _series_pair(
                TIMES, values, [v + 0.5 for v in values], lon=lon, lat=lat
            )
            comparisons.append(
                _station_comparison(
                    reference=station,
                    test="his",
                    variable=TEMPERATURE,
                    aligned=aligned,
                    select={"Z": band},
                )
            )
    pooled = ComparisonSet(comparisons)
    averaged = pooled.average(by=["variable", "depth"])

    assert len(averaged) == 3  # bands stay distinct rather than averaging together
    labels = {c.as_item()["depth_band"] for c in averaged}
    assert labels == {"0-5 m", "10-15 m", "30-40 m"}
    # actual_depth is gone (a real consequence of pooling), yet the band survives.
    for c in averaged:
        assert c.aligned.attrs.get("actual_depth") is None
        assert c.as_item()["depth_band"] is not None


# -- downstream behavior --------------------------------------------------------------


def test_averaged_comparisons_produce_a_normal_metrics_table():
    pooled = _two_station_set()
    averaged = pooled.average(by="variable")
    df = averaged.metrics()

    assert len(df) == 2
    assert set(df["variable"]) == {TEMPERATURE, SALINITY}
    # A station's own position rides along, same as any other station comparison.
    assert df["station_lon"].tolist() == pytest.approx([-151.0, -151.0])


# -- grouping options and edge cases ---------------------------------------------------


def test_average_rejects_an_unknown_grouping_dimension():
    pooled = _two_station_set()
    with pytest.raises(ValueError, match="unknown dimension"):
        pooled.average(by="not_a_real_dimension")


def test_average_warns_when_a_group_spans_multiple_test_sources():
    hv1_temp = _series_pair(TIMES, [10, 11, 12, 13], [10.5, 11.2, 12.1, 13.4], lon=-150.0, lat=20.0)
    hv1_temp_other_model = _series_pair(
        TIMES, [10, 11, 12, 13], [9.5, 10.2, 11.1, 12.4], lon=-150.0, lat=20.0
    )
    pooled = ComparisonSet(
        [
            _station_comparison(reference="HV1", test="his_a", variable=TEMPERATURE, aligned=hv1_temp),
            _station_comparison(
                reference="HV1", test="his_b", variable=TEMPERATURE, aligned=hv1_temp_other_model
            ),
        ]
    )
    with pytest.warns(UserWarning, match="multiple test sources"):
        averaged = pooled.average(by="variable")
    assert len(averaged) == 1


def test_average_by_variable_and_test_keeps_models_distinct():
    hv1_temp = _series_pair(TIMES, [10, 11, 12, 13], [10.5, 11.2, 12.1, 13.4], lon=-150.0, lat=20.0)
    hv1_temp_other_model = _series_pair(
        TIMES, [10, 11, 12, 13], [9.5, 10.2, 11.1, 12.4], lon=-150.0, lat=20.0
    )
    pooled = ComparisonSet(
        [
            _station_comparison(reference="HV1", test="his_a", variable=TEMPERATURE, aligned=hv1_temp),
            _station_comparison(
                reference="HV1", test="his_b", variable=TEMPERATURE, aligned=hv1_temp_other_model
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        averaged = pooled.average(by=["variable", "test"])
    assert len(averaged) == 2
    assert {c.test_name for c in averaged} == {"his_a", "his_b"}


# -- generic over shape: gridded fields on a shared grid -------------------------------


LAT = np.linspace(10, 20, 5)
LON = np.linspace(-90, -80, 6)


def _grid_pair(offset: float, seed: int) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    reference = xr.DataArray(
        20 + rng.normal(0, 1, (5, 6)), dims=("lat", "lon"), coords={"lat": LAT, "lon": LON}
    )
    test = reference + offset
    return xr.Dataset(
        {
            "test": test.rename("test"),
            "reference": reference.rename("reference"),
            "difference": (test - reference).rename("difference"),
        }
    )


def test_average_of_gridded_comparisons_on_the_same_grid():
    a = _grid_pair(1.0, 0)
    b = _grid_pair(2.0, 1)
    pooled = ComparisonSet(
        [
            _station_comparison(reference="obsA", test="his", variable=TEMPERATURE, aligned=a),
            _station_comparison(reference="obsB", test="his", variable=TEMPERATURE, aligned=b),
        ]
    )
    averaged = pooled.average(by="variable")
    assert len(averaged) == 1
    field = averaged.comparisons[0]

    expected_reference = (a["reference"] + b["reference"]) / 2
    np.testing.assert_allclose(
        field.aligned["reference"].values, expected_reference.values
    )
