"""Tests for :mod:`ocean_skill.extrema`: locating a field's min/max, then following
it through time.

Mirrors ``tests/test_field_series.py``'s stub pattern (``comparison.prepare_source``
swapped out) so these exercise the locator and the recipe-building logic without a
catalog. ``.series()``'s default time window additionally reads the source's native
time axis via ``extrema._native_time_index``, monkeypatched the same way.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

NITRATE = "nitrate"
SILICATE = "silicate"


# -- fixtures: prepared fields of each shape the locator has to handle -----------------


def _rectilinear_map(nt: int | None = None):
    """A map with a planted max (50.0) and min (-50.0), optionally faceted by time.

    The time-faceted variant plants its max at time index 1, so a test can assert
    the extremum reports *that* step's time -- not just a position.
    """
    lat = np.array([10.0, 20.0, 30.0, 40.0])
    lon = np.array([-100.0, -95.0, -90.0, -85.0, -80.0])
    if nt is None:
        values = np.full((4, 5), 5.0)
        values[1, 3] = 50.0
        values[3, 0] = -50.0
        return xr.DataArray(
            values,
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
            name=NITRATE,
            attrs={"units": "mmol m-3"},
        )
    time = pd.date_range("2012-01-01", periods=nt, freq="D")
    values = np.full((nt, 4, 5), 5.0)
    values[1, 1, 3] = 50.0
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


def _rectilinear_map_with_nan_peak():
    """A finite max of 20.0, with a larger value masked out by NaN nearby."""
    lat = np.array([10.0, 20.0, 30.0])
    lon = np.array([-100.0, -95.0, -90.0])
    values = np.array([[5.0, 5.0, 5.0], [5.0, np.nan, 5.0], [5.0, 5.0, 20.0]])
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


def _all_nan_map():
    lat = np.array([10.0, 20.0])
    lon = np.array([-100.0, -95.0])
    values = np.full((2, 2), np.nan)
    return xr.DataArray(
        values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name=NITRATE
    )


def _curvilinear_map():
    """A ROMS-shaped grid: 2-D lon_rho/lat_rho on (eta_rho, xi_rho)."""
    lat_1d = np.linspace(10.0, 40.0, 4)
    lon_1d = np.linspace(-100.0, -80.0, 5)
    lat2d = np.tile(lat_1d[:, None], (1, 5))
    lon2d = np.tile(lon_1d[None, :], (4, 1))
    values = np.full((4, 5), 5.0)
    values[2, 4] = 50.0
    return xr.DataArray(
        values,
        dims=("eta_rho", "xi_rho"),
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon2d),
            "lat_rho": (("eta_rho", "xi_rho"), lat2d),
        },
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


def _antimeridian_map():
    """A domain straddling the dateline, stored in ±180 -- the pac_dt_ramp shape."""
    lat = np.array([0.0, 10.0])
    lon = np.array([170.0, 180.0, -170.0, -160.0, -150.0])
    values = np.full((2, 5), 5.0)
    values[1, 2] = 50.0  # planted at lon=-170 (190 in 0-360)
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


def _depth_faceted_point(depths=(0.0, 50.0, 100.0), peak_index=1):
    """A map with a surviving vertical axis; the planted max sits at ``peak_index``."""
    lat = np.array([10.0, 20.0])
    lon = np.array([-100.0, -95.0])
    depth = np.array(depths)
    values = np.full((len(depths), 2, 2), 5.0)
    values[peak_index, 0, 0] = 50.0
    return xr.DataArray(
        values,
        dims=("depth", "lat", "lon"),
        coords={"depth": depth, "lat": lat, "lon": lon},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


def _point_series(n: int = 12):
    """A field already reduced to one place through time -- what a point select draws."""
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    values = 8.0 + np.sin(np.arange(n) / 3.0)
    da = xr.DataArray(
        values,
        dims="time",
        coords={"time": time},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )
    return da.assign_coords(lon=-144.245, lat=49.978)


@pytest.fixture
def stub(monkeypatch):
    """Return a setter that swaps ``comparison.prepare_source`` for one field."""
    from ocean_skill import comparison

    def use(field_da):
        monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (field_da, None))

    return use


def _make(**kwargs):
    from ocean_skill.field import field as make_field

    return make_field("stub", NITRATE, **kwargs)


# -- locating the extremum ---------------------------------------------------------------


def test_rectilinear_max_locates_value_and_position(stub):
    stub(_rectilinear_map())
    ext = _make().extremum("max")
    assert ext.kind == "max"
    assert ext.value == pytest.approx(50.0)
    assert ext.units == "mmol m-3"
    assert ext.indices == {"lat": 1, "lon": 3}
    assert ext.lat == pytest.approx(20.0)
    assert ext.lon == pytest.approx(-85.0)


def test_rectilinear_min_locates_value_and_position(stub):
    stub(_rectilinear_map())
    ext = _make().extremum("min")
    assert ext.kind == "min"
    assert ext.value == pytest.approx(-50.0)
    assert ext.indices == {"lat": 3, "lon": 0}
    assert ext.lat == pytest.approx(40.0)
    assert ext.lon == pytest.approx(-100.0)


def test_default_kind_is_max(stub):
    stub(_rectilinear_map())
    assert _make().extremum().kind == "max"


def test_curvilinear_indices_are_keyed_by_grid_dims(stub):
    stub(_curvilinear_map())
    ext = _make().extremum("max")
    assert ext.indices == {"eta_rho": 2, "xi_rho": 4}
    assert ext.lat == pytest.approx(np.linspace(10.0, 40.0, 4)[2])
    assert ext.lon == pytest.approx(np.linspace(-100.0, -80.0, 5)[4])


def test_nan_cells_are_excluded_from_the_search(stub):
    stub(_rectilinear_map_with_nan_peak())
    ext = _make().extremum("max")
    assert ext.value == pytest.approx(20.0)
    assert ext.indices == {"lat": 2, "lon": 2}


def test_all_nan_raises_naming_the_source(stub):
    stub(_all_nan_map())
    with pytest.raises(ValueError, match=r"'stub'.*NaN everywhere"):
        _make().extremum("max")


def test_bad_kind_is_refused(stub):
    stub(_rectilinear_map())
    with pytest.raises(ValueError, match='"max" or "min"'):
        _make().extremum("peak")


def test_dateline_straddling_lon_reports_0_360_convention(stub):
    stub(_antimeridian_map())
    ext = _make().extremum("max")
    assert ext.lon_convention == "0-360"
    assert ext.lon == pytest.approx(-170.0)  # reported as stored, not rewrapped


def test_time_facet_argmax_sets_snapshot_time(stub):
    da = _rectilinear_map(nt=3)
    stub(da)
    ext = _make().extremum("max")
    assert pd.Timestamp(ext.time) == pd.Timestamp(da["time"].values[1])
    assert ext.time_reason == "the time coordinate at the extremum"


def _ns_resolution(da):
    """Force ``da``'s time coordinate to datetime64[ns] -- the resolution whose
    ``.item()`` collapses to a bare ns-since-epoch int, reproducing the reported
    bug regardless of xarray's ambient default resolution.
    """
    return da.assign_coords(time=da["time"].values.astype("datetime64[ns]"))


def test_snapshot_time_is_a_datetime_not_a_raw_ns_int(stub):
    # A datetime64[ns] coordinate must not collapse to a bare ns-since-epoch int
    # via .item() -- that int cannot be nearest-matched against a native time
    # index of a different resolution in _window_select.
    stub(_ns_resolution(_rectilinear_map(nt=3)))
    ext = _make().extremum("max")
    assert not isinstance(ext.time, (int, np.integer))
    assert pd.Timestamp(ext.time) == pd.Timestamp("2012-01-02")


def test_default_window_survives_a_non_ns_native_index(stub, monkeypatch):
    # Regression: snapshot from a datetime64[ns] coord, native index at second
    # resolution -- previously raised "Cannot compare dtypes datetime64[s] and
    # int64" (then "'<' not supported between datetime.datetime and int") from
    # _window_select.
    stub(_ns_resolution(_rectilinear_map(nt=5)))
    ext = _make().extremum("max")  # planted max at time index 1
    index = pd.date_range("2012-01-01", periods=5, freq="D").as_unit("s")
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda src: index)
    fs = ext.series(pad=1)
    assert fs[0].select["time"] == {"min": str(index[0]), "max": str(index[2])}


def test_no_time_axis_leaves_time_none_with_a_reason(stub):
    stub(_rectilinear_map())
    ext = _make().extremum("max")
    assert ext.time is None
    assert "full record" in ext.time_reason


def test_a_select_time_entry_is_reflected_in_the_reason(stub):
    stub(_rectilinear_map())
    ext = _make(select={"time": "2012-01"}).extremum("max")
    assert ext.time is None
    assert "recipe's own time selection" in ext.time_reason


def test_point_reduced_field_refuses_extremum(stub):
    stub(_point_series())
    with pytest.raises(ValueError, match="no spatial extremum"):
        _make().extremum("max")


def test_fully_collapsed_point_also_refuses(stub):
    stub(_point_series().mean("time"))
    with pytest.raises(ValueError, match="no spatial extremum"):
        _make().extremum("max")


def test_repr_reports_value_position_and_indices(stub):
    stub(_rectilinear_map())
    text = repr(_make().extremum("max"))
    assert "max" in text
    assert "50" in text
    assert "grid indices" in text
    assert "lat" in text and "lon" in text
    assert "'stub'" in text


# -- the default time window --------------------------------------------------------------


def test_window_select_interior_snapshot():
    from ocean_skill.extrema import _window_select

    index = pd.date_range("2012-01-01", periods=100, freq="D")
    win = _window_select(index, index[50], pad=10)
    assert win == {"min": str(index[40]), "max": str(index[60])}


def test_window_select_clamps_at_record_start():
    from ocean_skill.extrema import _window_select

    index = pd.date_range("2012-01-01", periods=100, freq="D")
    win = _window_select(index, index[2], pad=10)
    assert win == {"min": str(index[0]), "max": str(index[12])}


def test_window_select_clamps_at_record_end():
    from ocean_skill.extrema import _window_select

    index = pd.date_range("2012-01-01", periods=100, freq="D")
    win = _window_select(index, index[97], pad=10)
    assert win == {"min": str(index[87]), "max": str(index[99])}


def test_window_select_pad_is_honored():
    from ocean_skill.extrema import _window_select

    index = pd.date_range("2012-01-01", periods=100, freq="D")
    win = _window_select(index, index[50], pad=3)
    assert win == {"min": str(index[47]), "max": str(index[53])}


# -- Extremum.series(): recipe construction ------------------------------------------------


def test_series_returns_a_fieldset(stub):
    from ocean_skill.field import FieldSet

    stub(_rectilinear_map())
    fs = _make().extremum("max").series(time="2012-01")
    assert isinstance(fs, FieldSet)
    assert len(fs) == 1


def test_series_pins_lon_lat_to_the_extremum(stub):
    stub(_rectilinear_map())
    ext = _make().extremum("max")
    fs = ext.series(time="2012-01")
    assert fs[0].select["lon"] == ext.lon
    assert fs[0].select["lat"] == ext.lat


def test_series_time_override_is_used_verbatim(stub, monkeypatch):
    stub(_rectilinear_map(nt=5))
    ext = _make().extremum("max")
    monkeypatch.setattr(
        "ocean_skill.extrema._native_time_index",
        lambda src: (_ for _ in ()).throw(AssertionError("should not reopen the source")),
    )
    fs = ext.series(time=slice("2012-01-01", "2012-01-03"))
    assert fs[0].select["time"] == slice("2012-01-01", "2012-01-03")


def test_series_default_window_is_padded_around_the_snapshot(stub, monkeypatch):
    stub(_rectilinear_map(nt=5))
    ext = _make().extremum("max")  # planted max at time index 1
    index = pd.date_range("2012-01-01", periods=5, freq="D")
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda src: index)
    fs = ext.series(pad=1)
    assert fs[0].select["time"] == {"min": str(index[0]), "max": str(index[2])}


def test_series_reuses_parent_time_selection_when_no_snapshot(stub):
    stub(_rectilinear_map())
    ext = _make(select={"time": "2012-01"}).extremum("max")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fs = ext.series()
    assert fs[0].select["time"] == "2012-01"


def test_series_warns_and_defaults_to_the_full_record(stub):
    stub(_rectilinear_map())
    ext = _make().extremum("max")
    with pytest.warns(UserWarning, match="full time record"):
        fs = ext.series()
    assert "time" not in fs[0].select


def test_series_keeps_a_scalar_depth_select(stub):
    stub(_rectilinear_map())
    ext = _make(select={"depth": "surface"}).extremum("max")
    fs = ext.series(time="2012-01")
    assert fs[0].select["depth"] == "surface"


def test_series_replaces_a_depth_list_with_the_argmax_level(stub):
    stub(_depth_faceted_point(depths=(0.0, 50.0, 100.0), peak_index=1))
    ext = _make(select={"depth": [0.0, 50.0, 100.0]}).extremum("max")
    assert ext.coords["depth"] == pytest.approx(50.0)
    fs = ext.series(time="2012-01")
    assert fs[0].select["depth"] == pytest.approx(50.0)


def test_series_pins_a_surviving_z_dim_when_no_vertical_select(stub):
    stub(_depth_faceted_point(depths=(0.0, 50.0, 100.0), peak_index=2))
    ext = _make().extremum("max")  # no select at all
    fs = ext.series(time="2012-01")
    assert fs[0].select["depth"] == pytest.approx(100.0)


def test_series_strips_time_aggregate_but_keeps_other_entries(stub):
    stub(_rectilinear_map())
    agg = {"time": {"resample": "1MS", "reduce": "mean"}, "Z": "mean"}
    ext = _make(aggregate=agg).extremum("max")
    fs = ext.series(time="2012-01")
    assert fs[0].aggregate == {"Z": "mean"}


def test_series_carries_cache_and_label(stub):
    stub(_rectilinear_map())
    ext = _make(label="run A", cache=False).extremum("max")
    fs = ext.series(time="2012-01")
    assert fs[0].label == "run A"
    assert fs[0].cache is False


def test_series_label_can_be_overridden(stub):
    stub(_rectilinear_map())
    ext = _make(label="run A").extremum("max")
    fs = ext.series(time="2012-01", label="custom")
    assert fs[0].label == "custom"


def test_series_extra_variable_adds_a_second_member(stub):
    stub(_rectilinear_map())
    ext = _make().extremum("max")
    fs = ext.series(variables=[SILICATE], time="2012-01")
    assert len(fs) == 2
    assert fs[0].standard_name == ext.standard_name
    assert fs[1].standard_name != ext.standard_name


def test_series_refuses_when_the_field_has_no_position(stub):
    da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims="x", name=NITRATE)
    stub(da)
    ext = _make().extremum("max")
    assert ext.lon is None
    with pytest.raises(ValueError, match="no lon/lat"):
        ext.series()


# -- .plot(): draws in both renderers, via the pre-existing series family ------------------


def test_series_plot_draws_one_line_in_both_renderers(stub):
    stub(_rectilinear_map())
    ext = _make(select={"time": "2012-01"}).extremum("max")
    stub(_point_series())  # the follow-on point select reduces to a series, as it would live
    fig = ext.series().plot()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 1

    import holoviews as hv

    obj = ext.series().plot(renderer="holoviews")
    assert len(obj.traverse(lambda x: x, [hv.Curve])) == 1


def test_plot_shortcut_delegates_to_series(stub):
    stub(_rectilinear_map())
    ext = _make(select={"time": "2012-01"}).extremum("max")
    stub(_point_series())
    fig = ext.plot()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 1
