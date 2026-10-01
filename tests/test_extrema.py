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


def test_series_accepts_a_single_variable_not_wrapped_in_a_list(stub):
    """A bare string must not be unpacked into ``s``, ``i``, ``l``, ... ."""
    stub(_rectilinear_map())
    ext = _make().extremum("max")
    fs = ext.series(variables=SILICATE, time="2012-01")
    assert len(fs) == 2
    assert fs[1].standard_name != ext.standard_name


def test_series_carries_qc_and_detide(stub):
    stub(_rectilinear_map())
    qc_spec = {"range": [-1000, 1000]}
    ext = _make(qc=qc_spec, detide=True).extremum("max")
    fs = ext.series(time="2012-01")
    assert fs[0].qc == qc_spec
    assert fs[0].detide == {"T": 33.0}


def test_series_cache_defaults_to_inheriting_the_parent(stub):
    stub(_rectilinear_map())
    ext = _make(cache=False).extremum("max")
    fs = ext.series(time="2012-01")
    assert fs[0].cache is False


def test_series_cache_can_be_overridden(stub):
    stub(_rectilinear_map())
    ext = _make(cache=True).extremum("max")
    fs = ext.series(time="2012-01", cache=False)
    assert fs[0].cache is False


def test_series_refuses_a_parent_transect(stub):
    stub(_rectilinear_map())
    ext = _make(select={"transect": {"lon": -90.0}, "time": "2012-01"}).extremum("max")
    with pytest.raises(ValueError, match="transect"):
        ext.series()


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


# -- default suptitle: the series says it follows an extremum ------------------------


def _suptitles(fs, **kwargs):
    """Return the figure title from each renderer, as ``(static, interactive)``."""
    fig = fs.plot(**kwargs)
    static = fig._suptitle.get_text() if fig._suptitle is not None else None
    obj = fs.plot(renderer="holoviews", **kwargs)
    return static, obj.opts.get("plot").kwargs.get("title")


def test_series_title_names_the_extremum_in_both_renderers(stub):
    stub(_rectilinear_map())
    ext = _make(select={"depth": "surface", "time": "2012-01"}).extremum("min")
    stub(_point_series())  # the follow-on point select reduces to a series, live
    fs = ext.series()
    expected = (
        "Time series at the surface nitrate minimum: -50 mmol m-3 at 40.0°N 100.0°W"
    )
    assert fs.title == expected
    assert _suptitles(fs) == (expected, expected)


def test_series_title_adds_the_snapshot_date_when_the_extremum_has_one(stub):
    stub(_rectilinear_map(nt=3))  # the planted max sits at the second day
    ext = _make().extremum("max")
    assert ext.time is not None
    stub(_point_series())
    fs = ext.series(time="2012-01")  # time= skips the native-index lookup
    assert fs.title == (
        "Time series at the nitrate maximum: 50 mmol m-3 at 20.0°N 85.0°W on 2012-01-02"
    )


def test_date_label_reads_every_time_type_an_extremum_carries():
    import cftime

    from ocean_skill.extrema import _date_label

    assert _date_label(pd.Timestamp("2010-10-31 06:00")) == "2010-10-31"
    assert _date_label(np.datetime64("2010-10-31T06:00:00")) == "2010-10-31"
    assert _date_label(cftime.DatetimeNoLeap(2010, 10, 30, 12)) == "2010-10-30"
    # An undecoded numeric axis has no date in it: omit it, don't print "3652.5".
    assert _date_label(3652.5) is None
    assert _date_label(None) is None


def test_series_title_depth_phrase_follows_the_parent_vertical_select(stub):
    stub(_rectilinear_map())

    def title_for(select):
        return _make(select=select).extremum("max").series(time="2012-01").title

    assert " at the surface nitrate maximum" in title_for({"depth": "surface"})
    assert " at the 100 m nitrate maximum" in title_for({"depth": 100})
    assert " at the 12.5 m nitrate maximum" in title_for({"Z": 12.5})
    # No single level named -> no depth phrase at all, rather than a guess.
    for select in (
        {},
        {"depth": {"min": 0, "max": 10}},
        {"depth": [0, 50]},
        {"depth": "column"},
    ):
        title = title_for(select)
        assert title.startswith("Time series at the nitrate maximum: "), (select, title)


def test_series_title_reads_a_single_isopycnal_as_sigma0_not_metres(stub):
    stub(_rectilinear_map())
    fs = _make(select={"sigma0": 26.0}).extremum("max").series(time="2012-01")
    assert "σ₀ = 26 kg/m³ nitrate maximum" in fs.title
    assert " m nitrate" not in fs.title


def test_series_title_marks_local_hits_and_ranks(stub):
    import dataclasses

    stub(_rectilinear_map())
    ext = _make(select={"depth": "surface"}).extremum("max")
    local = dataclasses.replace(ext, mode="local", rank=2)
    title = local.series(time="2012-01").title
    assert "surface nitrate local maximum (#2): 50 mmol m-3" in title
    # The first hit of a search is unranked, and a plain hit is not "local".
    assert "(#" not in ext.series(time="2012-01").title
    assert "local" not in ext.series(time="2012-01").title


def test_series_title_omits_what_it_does_not_know(stub):
    """No snapshot time -> no date; no units -> no trailing unit; both stay quiet."""
    import dataclasses

    stub(_rectilinear_map())
    ext = _make(select={"time": "2012-01"}).extremum("max")  # a map: no time to report
    assert ext.time is None
    bare = dataclasses.replace(ext, units=None, lon=None, lat=None)
    title = bare._series_title()
    assert title == "Time series at the nitrate maximum: 50"
    assert " on " not in title and " at " not in title.split(":", 1)[1]


def test_plot_title_overrides_the_default_in_both_renderers(stub):
    stub(_rectilinear_map())
    ext = _make(select={"time": "2012-01"}).extremum("min")
    stub(_point_series())
    fs = ext.series()
    assert _suptitles(fs, title="Where it bottoms out") == (
        "Where it bottoms out",
        "Where it bottoms out",
    )
    # An explicit empty title draws nothing rather than falling back to the default.
    static, interactive = _suptitles(fs, title="")
    assert not static and not interactive


def test_extremum_plot_shortcut_carries_the_default_title(stub):
    stub(_rectilinear_map())
    ext = _make(select={"time": "2012-01"}).extremum("max")
    stub(_point_series())
    fig = ext.plot()
    assert fig._suptitle.get_text().startswith("Time series at the nitrate maximum: 50")
    assert ext.plot(title="mine")._suptitle.get_text() == "mine"


def test_fieldset_title_survives_sel_and_the_usable_reentry(stub, monkeypatch):
    from ocean_skill import comparison
    from ocean_skill.field import FieldSet

    stub(_rectilinear_map())
    ext = _make(select={"time": "2012-01"}).extremum("max")
    stub(_point_series())
    fs = ext.series(variables=[SILICATE])
    assert fs.title is not None
    assert fs.sel(variable=NITRATE).title == fs.title
    assert FieldSet(list(fs)).title is None  # a hand-built set has no default

    # A member whose source lacks its variable is dropped and the set re-enters
    # plot() as a new FieldSet -- which has to keep the title it was given.
    monkeypatch.setattr(
        comparison, "_variable_available", lambda *a, **k: "silicate" not in str(a[1])
    )
    with pytest.warns(UserWarning, match="skipping 1 field"):
        fig = fs.plot()
    assert fig._suptitle.get_text() == fs.title


# -- n= and local=True: distinct places, departure from wet neighbors --------


def _ramp(nlat: int = 20, nlon: int = 30):
    """Build a smooth, linear background whose median departure is exactly 0."""
    lat = np.arange(nlat, dtype=float)
    lon = -120.0 + np.arange(nlon, dtype=float)
    ramp = 2300.0 + 0.5 * lat[:, None] + 0.3 * np.arange(nlon)[None, :]
    return lat, lon, ramp


def _speckled_map(nt: int | None = None):
    """Build a ramp with the features the local search must tell apart.

    * a NaN "land" block in the corner, with a very low cell (-300) on its coast
      at (3, 2), which has only 5 wet neighbors -- the global min;
    * a one-cell speck (-100) in the open ocean at (10, 12);
    * a two-cell speck (-60) at (15, 5) and (15, 6);
    * a +500 step front at lon index >= 22, which is large but straight, so it must
      not register as a local anomaly anywhere along it.

    With ``nt`` set, the open-ocean speck persists through every step and is deepest
    (-100, versus -50) at step 2, and the other features are left out.
    """
    lat, lon, ramp = _ramp()
    if nt is None:
        values = ramp.copy()
        values[:3, :5] = np.nan
        values[3, 2] -= 300.0
        values[10, 12] -= 100.0
        values[15, 5] -= 60.0
        values[15, 6] -= 60.0
        values[:, 22:] += 500.0
        return xr.DataArray(
            values,
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
            name=NITRATE,
            attrs={"units": "mmol m-3"},
        )
    values = np.tile(ramp, (nt, 1, 1))
    values[:, 10, 12] -= 50.0
    values[2, 10, 12] -= 50.0
    time = pd.date_range("2012-01-01", periods=nt, freq="D")
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


def _positions(hits):
    return [(h.indices["lat"], h.indices["lon"]) for h in hits]


def test_n_one_still_returns_a_single_extremum(stub):
    from ocean_skill.extrema import Extrema, Extremum

    stub(_speckled_map())
    assert type(_make().extremum("min")) is Extremum
    assert type(_make().extremum("min", n=1)) is Extremum
    assert type(_make().extremum("min", local=True)) is Extremum
    assert type(_make().extremum("min", n=2)) is Extrema


def test_global_min_is_the_coastal_cell_and_carries_no_local_fields(stub):
    stub(_speckled_map())
    ext = _make().extremum("min")
    assert ext.indices == {"lat": 3, "lon": 2}
    assert ext.mode == "global"
    assert ext.anomaly is None and ext.wet_neighbors is None and ext.window is None


def test_global_top_n_hits_are_distinct_places(stub):
    stub(_speckled_map())
    hits = _make().extremum("min", n=3, separation=4)
    assert [h.rank for h in hits] == [1, 2, 3]
    values = [h.value for h in hits]
    assert values == sorted(values)
    pos = _positions(hits)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            assert max(abs(pos[i][0] - pos[j][0]), abs(pos[i][1] - pos[j][1])) >= 4


def test_global_default_separation_merges_a_whole_neighborhood(stub):
    stub(_rectilinear_map())  # 4 x 5 grid: one place fills it at separation 10
    hits = _make().extremum("min", n=3)
    assert len(hits) == 1
    assert hits[0].value == pytest.approx(-50.0)


def test_separation_one_returns_adjacent_cells(stub):
    stub(_rectilinear_map())
    hits = _make().extremum("min", n=3, separation=1)
    assert [h.value for h in hits] == pytest.approx([-50.0, 5.0, 5.0])


def test_local_ranks_the_coast_then_the_open_ocean_speck_then_the_pair(stub):
    stub(_speckled_map())
    hits = _make().extremum("min", local=True, n=3)
    assert _positions(hits)[:2] == [(3, 2), (10, 12)]
    assert _positions(hits)[2] in {(15, 5), (15, 6)}
    assert [h.anomaly for h in hits] == pytest.approx([-300.0, -100.0, -60.0], abs=1.0)
    assert all(h.mode == "local" and h.window == 3 for h in hits)


def test_local_hit_reports_its_own_value_and_wet_neighbor_count(stub):
    stub(_speckled_map())
    coast, speck, _ = _make().extremum("min", local=True, n=3)
    ramp = _ramp()[2]
    # the field's own value, not the score
    assert speck.value == pytest.approx(ramp[10, 12] - 100.0)
    assert speck.wet_neighbors == 8
    assert speck.neighborhood == pytest.approx(ramp[10, 12], abs=0.1)
    assert speck.anomaly == pytest.approx(speck.value - speck.neighborhood)
    assert coast.wet_neighbors == 5
    assert (speck.lat, speck.lon) == (10.0, -108.0)


def test_local_max_finds_positive_departures_only_by_sign(stub):
    da = _speckled_map()
    da[8, 20] += 200.0
    stub(da)
    top = _make().extremum("max", local=True)
    assert (top.indices["lat"], top.indices["lon"]) == (8, 20)
    assert top.anomaly == pytest.approx(200.0, abs=1.0)


def test_a_straight_step_front_is_not_a_local_anomaly(stub):
    from ocean_skill.extrema import _horizontal_dims, _neighbor_departure

    da = _speckled_map()
    departure = _neighbor_departure(da, _horizontal_dims(da), 3)[0]
    # the two columns straddling the +500 step: the median ignores the far side, so
    # the step's size never shows up -- only the ramp's ~1-unit residue does
    front = departure.isel(lon=slice(21, 23))
    assert float(np.abs(front).max()) < 5.0
    # ... while a one-cell feature of the same order does
    assert float(np.abs(departure).max()) > 250.0


def test_a_two_cell_speck_is_one_hit_but_separation_one_splits_it(stub):
    stub(_speckled_map())
    merged = _positions(_make().extremum("min", local=True, n=4))
    assert sum(p in {(15, 5), (15, 6)} for p in merged) == 1
    split = _positions(_make().extremum("min", local=True, n=4, separation=1))
    assert {(15, 5), (15, 6)} <= set(split)


def test_interior_drops_cells_that_touch_land(stub):
    stub(_speckled_map())
    hits = _make().extremum("min", local=True, n=2, interior=True)
    assert _positions(hits)[0] == (10, 12)
    assert (3, 2) not in _positions(hits)
    assert all(h.wet_neighbors == 8 for h in hits)


def test_a_persistent_speck_is_reported_once_at_its_strongest_step(stub):
    da = _speckled_map(nt=4)
    stub(da)
    hits = _make().extremum("min", local=True, n=2)
    top = hits[0]
    assert top.indices == {"time": 2, "lat": 10, "lon": 12}
    assert pd.Timestamp(top.time) == pd.Timestamp(da["time"].values[2])
    assert top.anomaly == pytest.approx(-100.0, abs=1.0)
    second = hits[1]
    assert max(abs(second.indices["lat"] - 10), abs(second.indices["lon"] - 12)) >= 3


def test_local_on_a_curvilinear_grid_keys_indices_by_grid_dims(stub):
    stub(_curvilinear_map())  # planted +50 on the xi_rho edge at (2, 4)
    top = _make().extremum("max", local=True)
    assert top.indices == {"eta_rho": 2, "xi_rho": 4}
    assert top.wet_neighbors == 5
    assert top.anomaly == pytest.approx(45.0)


def test_local_ignores_dim_order(stub):
    da = _speckled_map(nt=4).transpose("lat", "time", "lon")
    stub(da)
    top = _make().extremum("min", local=True)
    assert top.indices == {"lat": 10, "time": 2, "lon": 12}
    assert list(top.indices) == list(da.dims)


def test_n_larger_than_the_scoreable_cells_returns_fewer(stub):
    stub(_curvilinear_map())
    hits = _make().extremum("max", local=True, n=1000, separation=1)
    assert 1 < len(hits) < 1000
    assert _make().extremum("max", n=1000, separation=1)[0].value == pytest.approx(50.0)


def test_all_nan_raises_for_local_and_for_n(stub):
    stub(_all_nan_map())
    with pytest.raises(ValueError, match=r"'stub'.*NaN everywhere"):
        _make().extremum("min", local=True)
    with pytest.raises(ValueError, match=r"'stub'.*NaN everywhere"):
        _make().extremum("min", n=3)


def test_interior_that_drops_every_cell_raises(stub):
    # a 2 x 2 field: every cell touches the grid's edge, so none has a full window
    stub(_rectilinear_map().isel(lat=slice(0, 2), lon=slice(0, 2)))
    with pytest.raises(ValueError, match="NaN everywhere"):
        _make().extremum("min", local=True, interior=True)


def test_neighbor_median_matches_a_brute_force_loop():
    # The sort-and-index median must agree with a plain per-cell nanmedian at every
    # wet-neighbor count (even and odd), around masked cells and at the edges.
    from ocean_skill.extrema import MIN_WET_NEIGHBORS, _slice_departure

    rng = np.random.default_rng(0)
    for window in (3, 5):
        sl = rng.normal(size=(9, 11))
        sl[rng.random(sl.shape) < 0.3] = np.nan
        departure, median, spread, wet = _slice_departure(sl, window)
        r = window // 2
        for i in range(sl.shape[0]):
            for j in range(sl.shape[1]):
                block = sl[max(i - r, 0) : i + r + 1, max(j - r, 0) : j + r + 1]
                centre = (i - max(i - r, 0)) * block.shape[1] + j - max(j - r, 0)
                around = np.delete(block.ravel(), centre)
                around = around[np.isfinite(around)]
                assert wet[i, j] == len(around)
                if len(around):
                    assert median[i, j] == pytest.approx(np.median(around))
                    mad = np.median(np.abs(around - np.median(around)))
                    scored = np.isfinite(sl[i, j]) and len(around) >= MIN_WET_NEIGHBORS
                    if scored:
                        assert spread[i, j] == pytest.approx(1.4826 * mad)
                if np.isfinite(sl[i, j]) and len(around) >= MIN_WET_NEIGHBORS:
                    expected = sl[i, j] - np.median(around)
                    assert departure[i, j] == pytest.approx(expected)
                else:
                    assert np.isnan(departure[i, j]) and np.isnan(spread[i, j])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"score": "z"}, "only apply to local=True"),
        ({"window": 5}, "only apply to local=True"),
        ({"local": True, "score": "bad"}, "score must be"),
        ({"interior": -1}, "interior must be"),
        ({"interior": 1.5}, "interior must be"),
        ({"interior": "yes"}, "interior must be"),
        ({"n": 0}, "n must be an integer"),
        ({"n": True}, "n must be an integer"),
        ({"n": 1.5}, "n must be an integer"),
        ({"separation": 0}, "separation must be an integer"),
        ({"local": True, "window": 4}, "odd integer >= 3"),
        ({"local": True, "window": 1}, "odd integer >= 3"),
        ({"local": True, "window": 3.0}, "odd integer >= 3"),
    ],
)
def test_bad_arguments_are_refused(stub, kwargs, match):
    stub(_speckled_map())
    with pytest.raises(ValueError, match=match):
        _make().extremum("min", **kwargs)


def test_local_needs_a_two_dimensional_horizontal_grid(stub):
    along = np.arange(6, dtype=float)
    line = xr.DataArray(
        np.arange(6, dtype=float),
        dims="along",
        coords={"lon": ("along", -100.0 + along), "lat": ("along", 10.0 + along)},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )
    stub(line)
    with pytest.raises(ValueError, match="2-D horizontal grid"):
        _make().extremum("min", local=True)
    assert _make().extremum("min", n=2, separation=1)[0].value == pytest.approx(0.0)


def test_a_wider_window_reads_a_wider_neighborhood(stub):
    stub(_speckled_map())
    top = _make().extremum("min", local=True, window=5)
    assert top.window == 5
    assert top.wet_neighbors <= 24
    assert "5x5" in repr(top)


def test_lone_wet_cells_are_never_scored(stub):
    da = _speckled_map()
    da[:, 27:] = np.nan  # leave the far columns dry ...
    da[18, 28] = -1e6  # ... except one lone, absurdly low wet pixel
    stub(da)
    top = _make().extremum("min", local=True)
    assert (top.indices["lat"], top.indices["lon"]) != (18, 28)


# -- the Extrema container ---------------------------------------------------


def test_extrema_behaves_like_a_tuple_of_extremum(stub):
    from ocean_skill.extrema import Extremum

    stub(_speckled_map())
    hits = _make().extremum("min", local=True, n=3)
    assert len(hits) == 3
    assert all(isinstance(h, Extremum) for h in hits)
    assert list(hits) == list(hits.items)
    assert hits[0] is hits.items[0]
    assert hits[-1].rank == 3


def test_local_dataframe_columns_and_index(stub):
    stub(_speckled_map())
    df = _make().extremum("min", local=True, n=3).to_dataframe()
    assert list(df.columns) == [
        "value", "anomaly", "neighborhood", "spread", "z", "wet_neighbors",
        "land_distance", "lon", "lat", "time", "i_lat", "i_lon",
    ]  # fmt: skip
    assert df.index.name == "rank" and list(df.index) == [1, 2, 3]
    assert df.loc[1, "wet_neighbors"] == 5
    assert len(df.query("wet_neighbors == 8")) == 2


def test_global_dataframe_has_no_local_columns_but_keeps_facet_coords(stub):
    stub(_depth_faceted_point())
    df = _make().extremum("max", n=2, separation=1).to_dataframe()
    assert "anomaly" not in df.columns
    assert {"value", "lon", "lat", "i_depth", "depth"} <= set(df.columns)
    assert df.loc[1, "depth"] == pytest.approx(50.0)


def test_extrema_repr_is_a_header_and_a_table(stub):
    stub(_speckled_map())
    text = repr(_make().extremum("min", local=True, n=3))
    assert text.splitlines()[0].startswith("3 local min nitrate")
    assert "3x3 wet-neighbor median" in text and "3 cells apart" in text
    assert "wet_neighbors" in text and "'stub'" in text
    assert "ranked by z" in text
    assert "no land within 1 cell" in repr(
        _make().extremum("min", local=True, n=2, interior=True)
    )
    assert "no land within 5 cells" in repr(
        _make().extremum("min", local=True, n=2, interior=5, separation=1)
    )
    assert repr(_make().extremum("min", n=2, separation=1)).splitlines()[0].startswith(
        "2 min nitrate"
    )


def test_a_local_extremum_repr_explains_the_contrast(stub):
    stub(_speckled_map())
    coast, speck, _ = _make().extremum("min", local=True, n=3)
    text = repr(speck)
    assert text.startswith("local min nitrate")
    assert "below the median of its 3x3 wet neighbors" in text and "8/8 wet" in text
    assert "5/8 wet" in repr(coast)


def test_a_local_hit_follows_through_time_like_any_extremum(stub, monkeypatch):
    stub(_speckled_map(nt=5))
    hits = _make().extremum("min", local=True, n=2)
    index = pd.date_range("2012-01-01", periods=5, freq="D")
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda src: index)
    fs = hits[0].series(pad=1)
    assert fs[0].select["lon"] == hits[0].lon
    assert fs[0].select["lat"] == hits[0].lat
    assert fs[0].select["time"] == {"min": str(index[1]), "max": str(index[3])}


def test_extrema_is_exported():
    import ocean_skill as osk

    assert osk.Extrema is not None and "Extrema" in osk.__all__


# -- z-score ranking, and interior= as a distance from land -----------------


def _plume_map():
    """Build a river-mouth blob a few cells off a coast and a lone speck offshore.

    The blob (-600, sigma 1.5 cells, centered at (14, 8), four cells from the land
    block along the left edge) departs from its neighbors by hundreds of units, but
    its neighbors differ from one another by nearly as much; the speck at (22, 25)
    departs by only 30, in water whose neighbors differ by a fraction of a unit.
    """
    lat = np.arange(30, dtype=float)
    lon = -120.0 + np.arange(40, dtype=float)
    rr, cc = np.mgrid[0:30, 0:40]
    values = 2300.0 + 0.5 * lat[:, None] + 0.3 * np.arange(40)[None, :]
    values = values - 600.0 * np.exp(-((rr - 14) ** 2 + (cc - 8) ** 2) / (2 * 1.5**2))
    values[22, 25] -= 30.0
    values[:, :5] = np.nan
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


def test_a_plume_outranks_a_speck_by_departure_even_with_interior_true(stub):
    # The reported failure: interior=True only clears the first ring of cells, and a
    # river plume a few cells further out still departs from its neighbors by more
    # than any speck does.
    stub(_plume_map())
    hits = _make().extremum(
        "min", local=True, n=2, interior=True, score="departure"
    )
    plume, speck = hits
    assert (plume.indices["lat"], plume.indices["lon"]) == (14, 8)
    assert 1 < plume.land_distance <= 5  # passes interior=True: not touching land
    assert (speck.indices["lat"], speck.indices["lon"]) == (22, 25)
    assert abs(plume.anomaly) > 4 * abs(speck.anomaly)


def test_z_ranks_the_speck_above_the_plume(stub):
    stub(_plume_map())
    hits = _make().extremum("min", local=True, n=2, interior=True)
    speck, plume = hits
    assert (speck.indices["lat"], speck.indices["lon"]) == (22, 25)
    assert speck.z < -20 and abs(plume.z) < 5
    assert plume.spread > 20 * speck.spread  # why: its neighbors vary just as much
    assert speck.anomaly == pytest.approx(-30.0, abs=0.5)  # anomaly stays in units


def test_z_is_the_default_and_score_z_is_explicit_z(stub):
    stub(_plume_map())
    default = _make().extremum("min", local=True, n=3)
    explicit = _make().extremum("min", local=True, n=3, score="z")
    assert _positions(default) == _positions(explicit)
    assert default.score == explicit.score == "z"
    assert all(h.score == "z" for h in default)
    assert _make().extremum("min", local=True, score="departure").score == "departure"


def test_global_hits_carry_no_score(stub):
    stub(_plume_map())
    assert _make().extremum("min").score is None
    assert _make().extremum("min", n=2).score is None


def test_interior_as_a_distance_clears_the_plume(stub):
    stub(_plume_map())
    hits = _make().extremum(
        "min", local=True, n=3, interior=6, score="departure", separation=1
    )
    assert (hits[0].indices["lat"], hits[0].indices["lon"]) == (22, 25)
    assert all(h.land_distance > 6 for h in hits)


def test_interior_keeps_a_global_search_off_the_coast(stub):
    stub(_speckled_map())
    assert _make().extremum("min").indices == {"lat": 3, "lon": 2}  # the coastal cell
    hits = _make().extremum("min", n=3, interior=4, separation=1)
    assert all(h.land_distance > 4 for h in hits)
    assert (3, 2) not in _positions(hits)
    assert isinstance(_make().extremum("min", interior=True).land_distance, int)


def test_interior_true_is_the_local_windows_own_reach(stub):
    stub(_speckled_map())
    for window in (3, 5):
        hits = _make().extremum(
            "min", local=True, n=6, window=window, interior=True, separation=1
        )
        assert hits.interior == window // 2
        assert all(h.land_distance > window // 2 for h in hits)
        assert all(h.wet_neighbors == window * window - 1 for h in hits)


def test_interior_zero_and_false_exclude_nothing(stub):
    stub(_speckled_map())
    base = _make().extremum("min", local=True, n=3)
    for off in (0, False):
        assert _positions(_make().extremum("min", local=True, n=3, interior=off)) == (
            _positions(base)
        )
    assert base.interior == 0


def test_near_land_matches_a_brute_force_loop():
    from ocean_skill.extrema import _near_land

    rng = np.random.default_rng(3)
    for reach in (1, 2, 3):
        sl = rng.normal(size=(12, 15))
        sl[rng.random(sl.shape) < 0.1] = np.nan
        padded = np.pad(~np.isfinite(sl), reach, constant_values=True)  # edge is dry
        expected = np.array(
            [
                [
                    padded[i : i + 2 * reach + 1, j : j + 2 * reach + 1].any()
                    for j in range(sl.shape[1])
                ]
                for i in range(sl.shape[0])
            ]
        )
        assert (_near_land(sl, reach) == expected).all()


def test_land_distance_counts_cells_to_land_and_the_edge(stub):
    stub(_speckled_map())
    coast, speck, _ = _make().extremum("min", local=True, n=3)
    assert coast.land_distance == 1  # touching the NaN block
    assert speck.land_distance == 8  # (10, 12) to the block's corner at (2, 4)
    stub(_curvilinear_map())
    edge = _make().extremum("max", local=True)
    assert edge.indices == {"eta_rho": 2, "xi_rho": 4}
    assert edge.land_distance == 1  # on the last column: the edge counts


def test_land_distance_is_reported_for_a_time_faceted_field(stub):
    stub(_speckled_map(nt=4))
    top = _make().extremum("min", local=True)
    assert top.indices["time"] == 2
    assert top.land_distance == min(10 + 1, 12 + 1, 20 - 10, 30 - 12)


def test_z_stays_finite_on_a_constant_field_with_a_lone_speck(stub):
    values = np.full((12, 12), 100.0)
    values[6, 6] = 70.0
    stub(
        xr.DataArray(
            values,
            dims=("lat", "lon"),
            coords={"lat": np.arange(12.0), "lon": np.arange(12.0)},
            name=NITRATE,
        )
    )
    top = _make().extremum("min", local=True)
    assert (top.indices["lat"], top.indices["lon"]) == (6, 6)
    assert np.isfinite(top.z) and top.z < 0


def test_the_spread_floor_stops_a_smooth_patch_inflating_z():
    from ocean_skill.extrema import _horizontal_dims, _neighbor_departure, _z_floor

    rng = np.random.default_rng(1)
    values = 2300.0 + rng.normal(0.0, 2.0, size=(30, 30))
    values[10:20, 10:20] = 2300.0  # a perfectly flat patch: its neighbors' spread is 0
    values[15, 15] = 2300.01  # ... with a wiggle of a hundredth of a unit
    da = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": np.arange(30.0), "lon": np.arange(30.0)},
    )
    departure, median, spread, _ = _neighbor_departure(da, _horizontal_dims(da), 3)
    assert float(spread.isel(lat=15, lon=15)) == 0.0
    floor = _z_floor(spread.values, median.values)
    assert floor > 0.5
    z = departure / np.maximum(spread, floor)
    assert abs(float(z.isel(lat=15, lon=15))) < 0.05  # not 0.01 / 0


def test_interior_needs_a_two_dimensional_horizontal_grid(stub):
    along = np.arange(6, dtype=float)
    stub(
        xr.DataArray(
            np.arange(6, dtype=float),
            dims="along",
            coords={"lon": ("along", -100.0 + along), "lat": ("along", 10.0 + along)},
            name=NITRATE,
        )
    )
    with pytest.raises(ValueError, match="interior needs a 2-D horizontal grid"):
        _make().extremum("min", interior=3)


def test_an_interior_that_leaves_nothing_names_the_distance(stub):
    stub(_speckled_map())  # 20 x 30: no cell is more than 10 cells from land/edge
    with pytest.raises(ValueError, match=r"NaN everywhere.*within 10 cells"):
        _make().extremum("min", local=True, interior=10)
    with pytest.raises(ValueError, match=r"NaN everywhere.*within 10 cells"):
        _make().extremum("min", interior=10)


def test_a_local_repr_carries_z_and_the_distance_to_land(stub):
    stub(_speckled_map())
    coast, speck, _ = _make().extremum("min", local=True, n=3)
    assert ", z = " in repr(speck)
    assert "8 cells from the nearest land or grid edge" in repr(speck)
    assert "1 cell from the nearest land or grid edge" in repr(coast)
    assert "from the nearest land" in repr(_make().extremum("min"))  # global, too
