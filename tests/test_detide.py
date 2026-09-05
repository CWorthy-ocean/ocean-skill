"""``osk.detide``: the PL33 tidal low-pass filter, on every shape it takes.

A synthetic hourly signal -- a slow (10-day) trend plus M2 (12.42h) and K1 (24h)
tidal constituents -- stands in for a real record: PL33 should collapse the tidal
band and leave the trend standing, for a DataArray, a Dataset (including a
``time x depth`` timeSeriesProfile shape), a Series, and a DataFrame alike. See
``ocean_skill/detide.py``'s module docstring for the two ``pl33tn`` version quirks
this wrapper works around (a numpy datetime64 resolution bug in its ``dt``
calculation, and cf-xarray needing ``standard_name="time"`` to see a plain
``datetime64`` coordinate) -- these tests exercise both code paths, not just the
happy path they were found by testing directly against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.detide import detide

pytest.importorskip("oceans", reason="detide needs the 'oceans' package (pl33tn)")

N = 800  # hours -- long enough for PL33's ~67-sample window with room either side
EDGE = 40  # generously past PL33's own edge-NaN width at T=33 (about a day)


def _signal():
    """(t_hours, trend, tidal, combined) -- trend survives PL33, tidal does not."""
    t_hours = np.arange(0, N, 1.0)
    trend = 0.5 * np.sin(2 * np.pi * t_hours / (24 * 10))  # 10-day, below the band
    m2 = 1.0 * np.sin(2 * np.pi * t_hours / 12.42)
    k1 = 0.6 * np.sin(2 * np.pi * t_hours / 24.0)
    return t_hours, trend, m2 + k1, trend + m2 + k1


def _index():
    return pd.date_range("2020-01-01", periods=N, freq="h")


# -- DataArray -----------------------------------------------------------------------


def test_dataarray_subtidal_removes_tides_and_keeps_the_trend():
    _, trend, tidal, combined = _signal()
    da = xr.DataArray(
        combined, dims=["time"], coords={"time": _index()}, name="zeta",
        attrs={"units": "m", "long_name": "sea surface height"},
    )
    sub = detide(da)
    interior = slice(EDGE, -EDGE)
    # the tidal band is gone: residual std against the pure trend is small relative
    # to the tidal amplitude that would remain if nothing had been filtered
    assert np.nanstd(sub.values[interior] - trend[interior]) < 0.2 * np.nanstd(
        tidal[interior]
    )


def test_dataarray_edges_are_nan_interior_is_finite():
    _, _, _, combined = _signal()
    da = xr.DataArray(combined, dims=["time"], coords={"time": _index()})
    sub = detide(da)
    assert np.isnan(sub.values[:10]).all()
    assert np.isnan(sub.values[-10:]).all()
    assert np.isfinite(sub.values[EDGE:-EDGE]).all()


def test_dataarray_preserves_name_and_units_records_provenance():
    _, _, _, combined = _signal()
    da = xr.DataArray(
        combined, dims=["time"], coords={"time": _index()}, name="zeta",
        attrs={"units": "m"},
    )
    sub = detide(da)
    assert sub.name == "zeta"
    assert sub.attrs["units"] == "m"
    assert sub.attrs["detide_filter"] == "PL33"
    assert sub.attrs["detide_period_hours"] == 33.0
    assert sub.attrs["detide_component"] == "subtidal"


def test_dataarray_tidal_component_is_the_residual():
    _, _, _, combined = _signal()
    da = xr.DataArray(combined, dims=["time"], coords={"time": _index()}, name="zeta")
    sub = detide(da, component="subtidal")
    tidal = detide(da, component="tidal")
    np.testing.assert_allclose(
        tidal.values[EDGE:-EDGE], (da - sub).values[EDGE:-EDGE], atol=1e-10
    )
    assert tidal.attrs["detide_component"] == "tidal"


def test_dataarray_both_returns_the_pair():
    _, _, _, combined = _signal()
    da = xr.DataArray(combined, dims=["time"], coords={"time": _index()}, name="zeta")
    sub, tidal = detide(da, component="both")
    np.testing.assert_allclose(
        (sub + tidal).values[EDGE:-EDGE], da.values[EDGE:-EDGE], atol=1e-10
    )


def test_larger_T_also_removes_a_longer_period_oscillation():
    t_hours = np.arange(0, N, 1.0)
    two_day = 0.4 * np.sin(2 * np.pi * t_hours / 48.0)
    _, trend, tidal, _ = _signal()
    combined = trend + tidal + two_day
    da = xr.DataArray(combined, dims=["time"], coords={"time": _index()})

    sub_default = detide(da)  # T=33 -- the 2-day wave should mostly survive
    sub_3day = detide(da, T=72.0)  # a 3-day low-pass should remove it too

    interior = slice(EDGE * 2, -EDGE * 2)
    resid_default = np.nanstd(sub_default.values[interior] - trend[interior])
    resid_3day = np.nanstd(sub_3day.values[interior] - trend[interior])
    assert resid_3day < resid_default


def test_unknown_component_is_refused():
    da = xr.DataArray(np.zeros(100), dims=["time"], coords={"time": pd.date_range(
        "2020-01-01", periods=100, freq="h"
    )})
    with pytest.raises(ValueError, match="component"):
        detide(da, component="bogus")


def test_unsupported_type_is_refused():
    with pytest.raises(TypeError, match="DataArray"):
        detide([1, 2, 3])


# -- Dataset, including a timeSeriesProfile (time x depth) shape ---------------------


def test_dataset_filters_every_floating_time_variable_independently_per_depth():
    _, trend, tidal, combined = _signal()
    depths = [5.0, 10.0, 20.0]
    data = np.stack([combined + 0.01 * d for d in depths], axis=1)
    ds = xr.Dataset(
        {"temp": (["time", "depth"], data, {"units": "degC"})},
        coords={"time": _index(), "depth": depths},
    )
    out = detide(ds)
    interior = slice(EDGE, -EDGE)
    for i, d in enumerate(depths):
        expected = trend[interior] + 0.01 * d
        resid = out["temp"].isel(depth=i).values[interior] - expected
        assert np.nanstd(resid) < 0.2 * np.nanstd(tidal[interior])


def test_dataset_leaves_qc_flags_and_static_fields_untouched():
    _, _, _, combined = _signal()
    ds = xr.Dataset(
        {
            "temp": (["time"], combined, {"units": "degC"}),
            "temp_qc_agg": (["time"], np.ones(N)),
            "lon": ((), -122.5),
            "lat": ((), 45.0),
        },
        coords={"time": _index()},
    )
    with pytest.warns(UserWarning, match="temp_qc_agg"):
        out = detide(ds)
    np.testing.assert_array_equal(out["temp_qc_agg"].values, ds["temp_qc_agg"].values)
    assert float(out["lon"]) == -122.5
    assert float(out["lat"]) == 45.0


def test_dataset_both_component_suffixes_variable_names():
    _, _, _, combined = _signal()
    ds = xr.Dataset(
        {"temp": (["time"], combined, {"units": "degC"})}, coords={"time": _index()}
    )
    out = detide(ds, component="both")
    assert "temp" not in out.data_vars
    assert {"temp_subtidal", "temp_tidal"} <= set(out.data_vars)


def test_dataset_with_no_time_dimension_is_refused():
    ds = xr.Dataset({"x": (["depth"], [1.0, 2.0])}, coords={"depth": [1.0, 2.0]})
    with pytest.raises(ValueError, match="time"):
        detide(ds)


# -- Series ----------------------------------------------------------------------------


def test_series_subtidal_keeps_the_datetime_index_and_removes_tides():
    _, trend, tidal, combined = _signal()
    s = pd.Series(combined, index=_index(), name="zeta")
    sub = detide(s)
    assert isinstance(sub, pd.Series)
    assert sub.index.equals(s.index)
    interior = slice(EDGE, -EDGE)
    assert np.nanstd(sub.values[interior] - trend[interior]) < 0.2 * np.nanstd(
        tidal[interior]
    )


def test_series_without_datetime_index_is_refused():
    s = pd.Series(np.zeros(100))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        detide(s)


def test_series_tidal_and_both():
    _, _, _, combined = _signal()
    s = pd.Series(combined, index=_index(), name="zeta")
    sub = detide(s, component="subtidal")
    tidal = detide(s, component="tidal")
    interior = slice(EDGE, -EDGE)
    np.testing.assert_allclose(
        tidal.values[interior], (s - sub).values[interior], atol=1e-10
    )
    sub2, tidal2 = detide(s, component="both")
    pd.testing.assert_series_equal(sub2, sub)
    pd.testing.assert_series_equal(tidal2, tidal)


# -- DataFrame -------------------------------------------------------------------------


def test_dataframe_filters_numeric_columns_and_skips_qc():
    _, trend, tidal, combined = _signal()
    df = pd.DataFrame(
        {"zeta": combined, "temp_qc_agg": np.ones(N)}, index=_index()
    )
    with pytest.warns(UserWarning, match="temp_qc_agg"):
        out = detide(df)
    assert isinstance(out, pd.DataFrame)
    interior = slice(EDGE, -EDGE)
    assert np.nanstd(out["zeta"].values[interior] - trend[interior]) < 0.2 * np.nanstd(
        tidal[interior]
    )
    np.testing.assert_array_equal(out["temp_qc_agg"].values, df["temp_qc_agg"].values)


def test_dataframe_without_datetime_index_is_refused():
    df = pd.DataFrame({"zeta": np.zeros(100)})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        detide(df)


def test_dataframe_both_returns_two_frames():
    _, _, _, combined = _signal()
    df = pd.DataFrame({"zeta": combined}, index=_index())
    subtidal, tidal = detide(df, component="both")
    interior = slice(EDGE, -EDGE)
    np.testing.assert_allclose(
        (subtidal["zeta"] + tidal["zeta"]).values[interior],
        df["zeta"].values[interior],
        atol=1e-10,
    )


# -- meta=: a station DataFrame converts through tabular.to_dataset first ------------


def test_meta_routes_a_timeseriesprofile_dataframe_through_tabular_to_dataset():
    from tests.test_tabular import TSP_META, timeseriesprofile_frame

    with pytest.warns(UserWarning, match="duplicate"):
        out = detide(timeseriesprofile_frame(), meta=TSP_META, T=1.0)
    assert isinstance(out, xr.Dataset)
    assert set(out.dims) == {"time", "depth"}
    assert "Temperature" in out.data_vars
