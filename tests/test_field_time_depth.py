"""Tests for the ``time_depth`` path on a single, uncompared source (:class:`Field`).

Mirrors ``tests/test_field_series.py``'s stub pattern (``comparison.prepare_source``
swapped out) for the shape a select can leave standing at one place: both time *and*
depth surviving together is the default ``timeSeriesProfile`` station shape (see
``ctd_station_HV5`` in the Iceland catalog) and now draws as one depth-against-time
panel, colour = value, rather than being fanned into one line per ragged level (see
``Field.is_time_depth``/``Field.family``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

NITRATE = "nitrate"


def _point_time_depth(n: int = 6, depths=(0.0, 50.0, 100.0)):
    """A point with both time and depth standing, dense (no NaN) -- the mooring or
    model-column shape :func:`ocean_skill.plot.time_depth.default_mark` draws as a
    mesh.
    """
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    depth = np.array(depths)
    values = 8.0 + np.random.default_rng(1).normal(0, 1, (n, depth.size))
    da = xr.DataArray(
        values,
        dims=("time", "depth"),
        coords={"time": time, "depth": depth},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )
    return da.assign_coords(lon=-144.245, lat=49.978)


def _ragged_station(n_time: int = 8, n_depth: int = 20):
    """A repeat-visit station whose casts each reach a different, disjoint-ish
    subset of the union of every visit's own levels -- the shape
    :func:`ocean_skill.tabular._timeseriesprofile_dataset` builds (deeper or
    shallower depending on which visit), and the case
    :func:`~ocean_skill.plot.time_depth.default_mark` draws as scattered points.
    A prefix-only reach (every cast touching the same shallow levels) would trim
    away to a dense rectangle instead -- see ``default_mark``'s own note on
    trimming all-NaN rows/columns first.
    """
    time = pd.date_range("2024-04-01", periods=n_time, freq="2MS")
    depth = np.linspace(0.0, 100.0, n_depth)
    values = np.full((n_time, n_depth), np.nan)
    rng = np.random.default_rng(2)
    for i in range(n_time):
        reach = rng.integers(2, 5)
        start = rng.integers(0, n_depth - reach)
        idx = rng.choice(np.arange(start, start + reach), size=reach, replace=False)
        values[i, idx] = 2.0 + i + rng.normal(0, 0.1, reach)
    da = xr.DataArray(
        values,
        dims=("time", "depth"),
        coords={"time": time, "depth": depth},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )
    return da.assign_coords(lon=-21.8, lat=64.3)


def _point_month_climatology(n_years: int = 2, depths=(0.0, 50.0, 100.0)):
    """A station's month climatology -- the shape
    ``aggregate={"time": {"groupby": "month", "reduce": "mean"}}`` leaves
    standing with depth still surviving. Built through the real reduction
    (:func:`ocean_skill.operators.aggregate`), not hand-assembled, so the
    ``month`` dimension carries the marker
    :func:`~ocean_skill.operators.time_axis_dim` reads (see
    :func:`ocean_skill.operators._reduce_dim`) -- the same thing a stub built
    by hand (e.g. ``rename(season="month")``, see
    ``tests/test_field_series.py::test_a_non_season_extra_axis_on_a_profile_is_still_refused``)
    would not carry.
    """
    from ocean_skill.operators import aggregate

    n = 12 * n_years
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    depth = np.array(depths)
    values = 8.0 + np.sin(np.arange(n) / 3.0)[:, None] + 0.01 * depth[None, :]
    da = xr.DataArray(
        values,
        dims=("time", "depth"),
        coords={"time": time, "depth": depth},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    ).assign_coords(lon=-21.8, lat=64.3)
    return aggregate(da, {"time": {"groupby": "month", "reduce": "mean"}})


def _point_year_climatology(n_years: int = 4, depths=(0.0, 50.0, 100.0)):
    """The same shape one groupby key over: every year of the record averaged
    into one field, depth still standing. A plain integer axis -- unlike
    ``month`` it gets no spelled-out tick labels (see
    :func:`ocean_skill.plot.series.groupby_ticks`).
    """
    from ocean_skill.operators import aggregate

    n = 12 * n_years
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    depth = np.array(depths)
    values = 8.0 + np.arange(n)[:, None] * 0.05 + 0.01 * depth[None, :]
    da = xr.DataArray(
        values,
        dims=("time", "depth"),
        coords={"time": time, "depth": depth},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    ).assign_coords(lon=-21.8, lat=64.3)
    return aggregate(da, {"time": {"groupby": "year", "reduce": "mean"}})


def _native_s_rho_point(nt: int = 4, ns: int = 5, with_z_rho: bool = False):
    """A ROMS point with a bare native ``s_rho`` axis, mirroring
    ``tests/test_field_unreduced_vertical.py``'s own fixture for the fanned-line
    case.
    """
    da = xr.DataArray(
        np.random.default_rng(3).normal(15.0, 1.0, (nt, ns)),
        dims=("time", "s_rho"),
        coords={"time": pd.date_range("2020-01-01", periods=nt, freq="D")},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    ).assign_coords(lon=-144.0, lat=50.0)
    if with_z_rho:
        da = da.assign_coords(z_rho=(("s_rho",), -np.linspace(5.0, 100.0, ns)))
    return da


@pytest.fixture
def stub(monkeypatch):
    """Swap ``comparison.prepare_source`` for one hand-built field."""
    from ocean_skill import comparison

    def use(field_da):
        monkeypatch.setattr(
            comparison, "prepare_source", lambda *a, **k: (field_da, None)
        )

    return use


def _make(**kwargs):
    from ocean_skill.field import field as make_field

    return make_field("stub", NITRATE, **kwargs)


def _make_set(variables, **kwargs):
    from ocean_skill.field import field as make_field

    return make_field("stub", variables, **kwargs)


def _station_grid(monkeypatch, stations, variables=(NITRATE, "silicate")):
    """Monkeypatch ``comparison.prepare_source`` to a station x variable grid --
    one :func:`_point_time_depth` field per (station, variable), each station
    its own lon/lat -- the shape ``osk.field([stations...], [variables...])``
    fans, used by the ``rows=``/``cols=`` facet tests below.
    """
    from ocean_skill import comparison

    lon_lat = {"station_a": (-144.245, 49.978), "station_b": (-150.0, 55.0)}
    data = {
        (station, variable): _point_time_depth().assign_coords(
            lon=lon_lat[station][0], lat=lon_lat[station][1]
        )
        for station in stations
        for variable in variables
    }

    def fake_prepare_source(source, variable, *args, **kwargs):
        matched = next(v for v in variables if v in variable)
        return (data[(source, matched)], None)

    monkeypatch.setattr(comparison, "prepare_source", fake_prepare_source)


# -- family inference -------------------------------------------------------------------


def test_a_bare_point_with_time_and_depth_is_time_depth(stub):
    """``is_series`` legitimately stays true too (point + a surviving time axis) --
    :attr:`Field.family` is what resolves the overlap, by checking
    ``is_time_depth`` first.
    """
    stub(_point_time_depth())
    f = _make()
    assert f.is_time_depth
    assert f.family == "time_depth"
    assert "depth against time" in f.family_reason


def test_a_depth_band_select_is_still_time_depth(stub):
    stub(_point_time_depth())
    f = _make(select={"depth": {"min": 0, "max": 100}})
    assert f.is_time_depth
    assert f.family == "time_depth"


def test_an_explicit_depth_list_keeps_the_series_family(stub):
    """A named list of levels asks to tell them apart, not to see the whole
    record -- see ``Field.is_time_depth``.
    """
    stub(_point_time_depth())
    f = _make(select={"depth": [0.0, 50.0, 100.0]})
    assert not f.is_time_depth
    assert f.is_series
    assert f.family == "series"


def test_a_native_s_rho_point_with_z_rho_is_time_depth(stub):
    stub(_native_s_rho_point(with_z_rho=True))
    f = _make()
    assert f.family == "time_depth"


def test_a_labelless_native_s_rho_point_refuses_time_depth(stub):
    stub(_native_s_rho_point(with_z_rho=False))
    with pytest.raises(ValueError, match="native vertical axis"):
        _make().plot()


def test_a_third_axis_is_refused_with_its_name(stub):
    stub(_point_time_depth().expand_dims(member=[1, 2]))
    with pytest.raises(ValueError, match=r"\['member'\]"):
        _make().plot()


# -- a time-groupby's surviving dim still reads as "time" ------------------------------


def test_a_month_climatology_with_depth_is_time_depth(stub):
    """``aggregate={"time": {"groupby": "month", "reduce": "mean"}}`` leaves
    ``(month, depth)`` standing at a station -- the same ``time_depth`` shape a
    real time axis leaves, with ``month`` playing time's role (see
    :func:`ocean_skill.operators.time_axis_dim`).
    """
    stub(_point_month_climatology())
    f = _make(aggregate={"time": {"groupby": "month", "reduce": "mean"}})
    assert f.is_time_depth
    assert f.family == "time_depth"


def test_a_month_climatology_draws_jan_dec_ticks_in_both_renderers(stub):
    stub(_point_month_climatology())
    f = _make(aggregate={"time": {"groupby": "month", "reduce": "mean"}})
    fig = f.plot()
    ax = fig.axes[0]
    assert ax.get_xlabel() == "month"
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    assert "by month" in fig._suptitle.get_text()

    obj = f.plot(renderer="holoviews")
    assert obj.kdims[0].name == "month"


def test_a_year_climatology_gets_a_plain_integer_axis(stub):
    """Only ``month`` gets spelled-out ticks -- every other groupby key draws
    its own integer values on a plain numeric axis (see
    :func:`ocean_skill.plot.series.groupby_ticks`).
    """
    stub(_point_year_climatology())
    f = _make(aggregate={"time": {"groupby": "year", "reduce": "mean"}})
    fig = f.plot()
    ax = fig.axes[0]
    assert ax.get_xlabel() == "year"
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels == [str(v) for v in sorted(int(lb) for lb in labels)]


# -- default_mark: ragged vs dense -------------------------------------------------------


def test_default_mark_picks_scatter_for_a_ragged_record():
    from ocean_skill.plot.time_depth import default_mark

    da = _ragged_station()
    assert default_mark(da) == "scatter"


def test_default_mark_picks_pcolormesh_for_a_dense_record():
    from ocean_skill.plot.time_depth import default_mark

    da = _point_time_depth()
    assert default_mark(da) == "pcolormesh"


# -- .plot() smoke, both renderers, both marks -------------------------------------------


def test_a_dense_point_draws_a_mesh_in_both_renderers(stub):
    stub(_point_time_depth())
    fig = _make().plot()
    ax = fig.axes[0]
    from matplotlib.collections import QuadMesh

    assert any(isinstance(c, QuadMesh) for c in ax.collections)
    assert ax.yaxis_inverted()

    import holoviews as hv

    obj = _make().plot(renderer="holoviews")
    assert obj.traverse(lambda x: x, [hv.QuadMesh])


def test_a_ragged_station_draws_scatter_in_both_renderers(stub):
    stub(_ragged_station())
    fig = _make().plot()
    ax = fig.axes[0]
    from matplotlib.collections import PathCollection

    assert any(isinstance(c, PathCollection) for c in ax.collections)
    assert ax.yaxis_inverted()

    import holoviews as hv

    obj = _make().plot(renderer="holoviews")
    assert obj.traverse(lambda x: x, [hv.Points])


def test_mark_can_be_overridden(stub):
    stub(_point_time_depth())
    fig = _make().plot(mark="scatter")
    ax = fig.axes[0]
    from matplotlib.collections import PathCollection

    assert any(isinstance(c, PathCollection) for c in ax.collections)


def test_the_title_carries_place_and_period(stub):
    stub(_point_time_depth())
    fig = _make(label="run A").plot()
    title = fig._suptitle.get_text()
    assert "50.0°N" in title
    assert "144.2°W" in title
    assert "run A" in title


def test_a_colorbar_is_drawn(stub):
    stub(_point_time_depth())
    fig = _make().plot()
    assert len(fig.axes) > 1


# -- movie() and FieldSet.plot() refuse ---------------------------------------------------


def test_movie_refuses_a_time_depth_field(stub):
    stub(_point_time_depth())
    with pytest.raises(ValueError, match="depth against time"):
        _make().movie()


def test_a_time_depth_set_draws_a_stacked_column(stub):
    """Every member ``time_depth`` draws as a stacked column, one panel each --
    the ADCP-mooring shape ``osk.field(osk.find(...), variable).plot()`` gives
    :func:`~ocean_skill.plot.matplotlib_renderer.time_depth_grid`.
    """
    stub(_point_time_depth())
    fs = _make_set([NITRATE, "silicate"])
    fig = fs.plot()
    from matplotlib.collections import QuadMesh

    # panel axes are created (and registered in fig.axes) before any colorbar's
    # own axes, so the first two are the panels themselves
    panels = fig.axes[:2]
    assert all(
        any(isinstance(c, QuadMesh) for c in ax.collections) for ax in panels
    )
    # two panels, each with its own colorbar axes (four axes total)
    assert len(fig.axes) == 4

    obj = _make_set([NITRATE, "silicate"]).plot(renderer="holoviews")
    assert len(obj) == 2


def test_time_depth_grid_titles_overrides_one_panel_and_keeps_the_other_auto(stub):
    stub(_point_time_depth())
    auto = [
        ax.get_title()
        for ax in _make_set([NITRATE, "silicate"]).plot().axes
        if ax.get_title()
    ]
    override = [None, "My Silicate Panel"]
    static = [
        ax.get_title()
        for ax in _make_set([NITRATE, "silicate"]).plot(titles=override).axes
        if ax.get_title()
    ]
    assert static == [auto[0], "My Silicate Panel"]

    import holoviews as hv

    obj = _make_set([NITRATE, "silicate"]).plot(renderer="holoviews", titles=override)
    hv_titles = {
        el.opts.get("plot").kwargs.get("title")
        for el in obj.traverse(lambda x: x, [hv.QuadMesh])
    }
    assert hv_titles == {auto[0], "My Silicate Panel"}

    with pytest.raises(ValueError, match="needs one entry per panel"):
        _make_set([NITRATE, "silicate"]).plot(titles=["only one"])


def test_a_variable_fan_titles_each_panel_by_variable_in_both_renderers(stub):
    """One station, several variables (``osk.field("ctd_station_HV1", ["temp",
    "salt", ...])``) -- every panel used to carry the *identical* station ·
    place · period title, with nothing anywhere naming which variable was
    which. Each panel must now read its own variable, with the shared station
    identity lifted into one suptitle instead (see
    :func:`~ocean_skill.plot.matplotlib_renderer.time_depth_grid_titles`).
    """
    import holoviews as hv

    stub(_point_time_depth())
    fs = _make_set([NITRATE, "silicate"])
    fig = fs.plot()
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["nitrate", "silicate"]
    suptitle = fig._suptitle.get_text()
    assert "stub" in suptitle
    assert "50.0°N" in suptitle and "144.2°W" in suptitle
    assert "nitrate" not in suptitle
    assert "silicate" not in suptitle

    obj = fs.plot(renderer="holoviews")
    hv_titles = [
        el.opts.get("plot").kwargs.get("title")
        for el in obj.traverse(lambda x: x, [hv.QuadMesh])
    ]
    assert hv_titles == titles
    assert obj.opts.get("plot").kwargs.get("title") == suptitle


def test_a_station_fan_keeps_naming_the_shared_variable_up_top(monkeypatch):
    """The opposite fan -- one variable, several stations
    (``osk.field(osk.find(...), "temperature")``) -- must keep behaving as it
    already did: the shared variable named once, in the suptitle, and each
    panel titled by its own station identity (:func:`grid_suptitle`'s own
    case, preserved by :func:`time_depth_grid_titles`).
    """
    import holoviews as hv

    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    data = {
        "station_a": _point_time_depth(),
        "station_b": _point_time_depth().assign_coords(lon=-150.0, lat=55.0),
    }

    def fake_prepare_source(source, variable, *args, **kwargs):
        return (data[source], None)

    monkeypatch.setattr(comparison, "prepare_source", fake_prepare_source)
    fs = make_field(["station_a", "station_b"], NITRATE)

    fig = fs.plot()
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == [
        "station_a · 50.0°N 144.2°W",
        "station_b · 55.0°N 150.0°W",
    ]
    suptitle = fig._suptitle.get_text()
    assert "nitrate" in suptitle
    assert "station_a" not in suptitle
    assert "station_b" not in suptitle

    obj = fs.plot(renderer="holoviews")
    hv_titles = [
        el.opts.get("plot").kwargs.get("title")
        for el in obj.traverse(lambda x: x, [hv.QuadMesh])
    ]
    assert hv_titles == titles
    assert obj.opts.get("plot").kwargs.get("title") == suptitle


def test_a_time_depth_set_grid_layout_and_scale_options(stub):
    """``ncols=``/``nrows=`` wrap the panels; ``shared_limits=True`` warns once
    when the set's variables actually differ (:func:`ocean_skill.plot
    .matplotlib_renderer.time_depth_grid`'s own convention, mirroring
    ``field_grid``).
    """
    stub(_point_time_depth())
    fs = _make_set([NITRATE, "silicate", "oxygen"])
    fig = fs.plot(ncols=2)
    assert fig.axes[0].get_gridspec().ncols == 2

    with pytest.warns(UserWarning, match="shared_limits=True"):
        _make_set([NITRATE, "silicate"]).plot(shared_limits=True)


def test_a_time_depth_set_sharex_option(monkeypatch):
    """``sharex=False`` gives each panel its own time window instead of the
    default shared one -- the ADCP-mooring case, several disjoint deployments
    (one mooring's record ends where the next one's begins) rather than one
    long overlapping record. ``sharex=True``/``False`` is a real parameter of
    :func:`~ocean_skill.plot.matplotlib_renderer.time_depth_grid` (and its
    interactive twin), not silently dropped the way it was before this option
    existed.
    """
    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    data = {
        "nitrate": _point_time_depth(),
        "silicate": _point_time_depth().assign_coords(
            time=pd.date_range("2025-06-01", periods=6, freq="MS")
        ),
    }

    def fake_prepare_source(source, variable, *args, **kwargs):
        return (data["silicate" if "silicate" in variable else "nitrate"], None)

    monkeypatch.setattr(comparison, "prepare_source", fake_prepare_source)
    fs = make_field("stub", [NITRATE, "silicate"])

    fig = fs.plot()
    axes = fig.axes[:2]
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])

    fig2 = fs.plot(sharex=False)
    axes2 = fig2.axes[:2]
    assert not axes2[0].get_shared_x_axes().joined(axes2[0], axes2[1])

    obj = fs.plot(renderer="holoviews", sharex=False)
    assert len(obj) == 2


def test_a_time_depth_set_sharey_option(monkeypatch):
    """``sharey=True`` shares one depth range across every panel instead of the
    default per-panel one -- moorings at very different depths (a 20m
    instrument next to a 100m one, say) each keep the range their own readings
    reach by default (:func:`~ocean_skill.plot.matplotlib_renderer
    .time_depth_grid`'s own ``sharey=False`` default), and can be lined up on
    request instead, the same option :func:`profile` exposes for its own depth
    axis.
    """
    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    data = {
        "nitrate": _point_time_depth(depths=(0.0, 10.0, 20.0)),
        "silicate": _point_time_depth(depths=(0.0, 50.0, 100.0)),
    }

    def fake_prepare_source(source, variable, *args, **kwargs):
        return (data["silicate" if "silicate" in variable else "nitrate"], None)

    monkeypatch.setattr(comparison, "prepare_source", fake_prepare_source)
    fs = make_field("stub", [NITRATE, "silicate"])

    fig = fs.plot()
    axes = fig.axes[:2]
    assert not axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    assert axes[0].get_ylim() != axes[1].get_ylim()

    fig2 = fs.plot(sharey=True)
    axes2 = fig2.axes[:2]
    assert axes2[0].get_shared_y_axes().joined(axes2[0], axes2[1])
    assert axes2[0].get_ylim() == axes2[1].get_ylim()

    obj = fs.plot(renderer="holoviews", sharey=True)
    assert len(obj) == 2


def test_a_mixed_time_depth_and_series_set_refuses_to_plot(monkeypatch):
    """A ``time_depth`` panel and an overlaid line share no single figure -- see
    :meth:`~ocean_skill.field.FieldSet.plot`.
    """
    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    time_depth_da = _point_time_depth()
    series_da = time_depth_da.isel(depth=0, drop=True)

    def fake_prepare_source(source, variable, *args, **kwargs):
        return (series_da if "silicate" in variable else time_depth_da, None)

    monkeypatch.setattr(comparison, "prepare_source", fake_prepare_source)
    fs = make_field("stub", [NITRATE, "silicate"])
    with pytest.raises(ValueError, match="some fields draw as depth against time"):
        fs.plot()


# -- rows=/cols= facet time_depth_grid into a genuine two-axis grid --------------------


def test_the_reported_bug_cols_variable_on_several_stations_now_facets(monkeypatch):
    """``ctdprofiles_all.sel(variable=[...]).plot(cols="variable")`` on several
    stations x several variables used to raise ``TypeError: 'cols' is not an
    option of time_depth_grid()``. It now draws a genuine grid: one column per
    variable, one row per station, in both renderers.
    """
    from ocean_skill.field import field as make_field

    _station_grid(monkeypatch, ["station_a", "station_b"])
    fs = make_field(["station_a", "station_b"], [NITRATE, "silicate"])

    static = fs.plot(cols="variable")
    gridspec = static.axes[0].get_gridspec()
    assert (gridspec.nrows, gridspec.ncols) == (2, 2)
    # Every panel is drawn (a dense product, no missing combination), so
    # every one of the first 4 axes (before their own colorbars) is a real
    # panel -- not `if ax.get_title()`, which would also drop the one drawn
    # panel whose own computed title happens to be "" (see below).
    static_titles = [ax.get_title() for ax in static.axes[:4]]

    import holoviews as hv

    interactive = fs.plot(cols="variable", renderer="holoviews")
    assert len(interactive) == 4
    hv_titles = [
        el.opts.get("plot").kwargs.get("title")
        for el in interactive.traverse(lambda x: x, [hv.QuadMesh])
    ]
    assert static_titles == hv_titles
    # Row-major: cell (r, c) at index r*2+c holds station r, variable c -- the
    # top row heads each column with its variable, the left column heads each
    # row with its station; the one cell neither heads reads "" (its identity
    # is already unambiguous from its row + column position).
    assert "nitrate" in static_titles[0] and "station_a" in static_titles[0]
    assert static_titles[1] == "silicate"
    assert "station_b" in static_titles[2]
    assert static_titles[3] == ""


def test_a_single_station_cols_variable_reproduces_the_flat_titling(stub):
    """A degenerate 1-row facet grid (one station, several variables) must
    read exactly like the unfaceted default -- see
    ``test_a_variable_fan_titles_each_panel_by_variable_in_both_renderers``,
    which this reproduces via an explicit ``cols="variable"`` instead.
    """
    stub(_point_time_depth())
    fs = _make_set([NITRATE, "silicate"])

    auto = fs.plot()
    faceted = fs.plot(cols="variable")
    assert [ax.get_title() for ax in faceted.axes if ax.get_title()] == [
        ax.get_title() for ax in auto.axes if ax.get_title()
    ]
    assert faceted._suptitle.get_text() == auto._suptitle.get_text()


def test_a_single_variable_rows_source_reproduces_the_flat_titling(monkeypatch):
    """A degenerate 1-column facet grid (one variable, several stations) must
    read exactly like the unfaceted default -- see
    ``test_a_station_fan_keeps_naming_the_shared_variable_up_top``, which this
    reproduces via an explicit ``rows="source"`` instead.
    """
    from ocean_skill.field import field as make_field

    _station_grid(monkeypatch, ["station_a", "station_b"], variables=(NITRATE,))
    fs = make_field(["station_a", "station_b"], NITRATE)

    auto = fs.plot()
    faceted = fs.plot(rows="source")
    assert [ax.get_title() for ax in faceted.axes if ax.get_title()] == [
        ax.get_title() for ax in auto.axes if ax.get_title()
    ]
    assert faceted._suptitle.get_text() == auto._suptitle.get_text()


def test_cols_source_transposes_the_grid(monkeypatch):
    from ocean_skill.field import field as make_field

    _station_grid(monkeypatch, ["station_a", "station_b"])
    fs = make_field(["station_a", "station_b"], [NITRATE, "silicate"])

    by_variable = fs.plot(cols="variable")
    by_source = fs.plot(cols="source")
    assert by_variable.axes[0].get_gridspec().ncols == 2
    assert by_source.axes[0].get_gridspec().ncols == 2
    # a station-per-column grid is the transpose (2 rows x 2 cols either way,
    # but which fact heads the rows vs. the columns swaps).
    titles_by_variable = [ax.get_title() for ax in by_variable.axes if ax.get_title()]
    titles_by_source = [ax.get_title() for ax in by_source.axes if ax.get_title()]
    assert "station_a" in titles_by_source[0] and "nitrate" in titles_by_source[0]
    assert "station_b" in titles_by_source[1]
    assert "silicate" in titles_by_source[2]
    assert titles_by_variable != titles_by_source


def _time_depth_item(station, variable, lon, lat):
    return {
        "field": _point_time_depth().assign_coords(lon=lon, lat=lat),
        "units": "mmol m-3",
        "standard_name": variable,
        "label": station,
    }


def test_a_sparse_station_variable_product_hides_the_missing_cell():
    """A station missing one of the variables another one has draws a hidden
    blank panel in that cell, rather than shifting every later cell out of
    place. Built as hand-crafted items (rather than through ``osk.field()``,
    which fans the *full* cross product eagerly and so cannot leave a hole)
    -- the same idiom ``tests/test_profile_renderers.py``'s own ragged-grid
    test uses.
    """
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    items = [
        _time_depth_item("station_a", NITRATE, -144.245, 49.978),
        _time_depth_item("station_a", "silicate", -144.245, 49.978),
        _time_depth_item("station_b", NITRATE, -150.0, 55.0),
        # no station_b silicate
    ]

    static = render(
        PlotSpec(family="time_depth", items=items, options={"cols": "variable"}),
        renderer="matplotlib",
    )
    # 4 panel axes (the blank one included) + one colorbar per *drawn* panel
    # only -- the blank cell is never drawn, so it earns no colorbar.
    assert len(static.axes) == 4 + 3
    panel_axes = static.axes[:4]
    blank_index = 1 * 2 + 1  # row=station_b (index 1), col=silicate (index 1)
    for i, ax in enumerate(panel_axes):
        assert ax.get_visible() == (i != blank_index)

    import holoviews as hv

    interactive = render(
        PlotSpec(family="time_depth", items=items, options={"cols": "variable"}),
        renderer="holoviews",
    )
    hv_elements = list(interactive)
    assert isinstance(hv_elements[blank_index], hv.Empty)


def test_two_facet_ncols_is_refused(monkeypatch):
    _station_grid(monkeypatch, ["station_a", "station_b"])
    from ocean_skill.field import field as make_field

    fs = make_field(["station_a", "station_b"], [NITRATE, "silicate"])
    with pytest.raises(ValueError, match="already fix this grid's shape"):
        fs.plot(cols="variable", ncols=2)


def test_faceting_a_time_depth_grid_by_an_unknown_key_says_what_is_allowed(
    monkeypatch,
):
    _station_grid(monkeypatch, ["station_a", "station_b"])
    from ocean_skill.field import field as make_field

    fs = make_field(["station_a", "station_b"], [NITRATE, "silicate"])
    with pytest.raises(ValueError, match="expected one of variable, source"):
        fs.plot(cols="platform")


def test_faceting_a_time_depth_grid_by_depth_or_time_is_refused(monkeypatch):
    _station_grid(monkeypatch, ["station_a", "station_b"])
    from ocean_skill.field import field as make_field

    fs = make_field(["station_a", "station_b"], [NITRATE, "silicate"])
    with pytest.raises(ValueError, match="axis every panel already draws against"):
        fs.plot(rows="depth")
    with pytest.raises(ValueError, match="axis every panel already draws against"):
        fs.plot(rows="time")


def test_rows_and_cols_naming_the_same_fact_is_refused(monkeypatch):
    _station_grid(monkeypatch, ["station_a", "station_b"])
    from ocean_skill.field import field as make_field

    fs = make_field(["station_a", "station_b"], [NITRATE, "silicate"])
    with pytest.raises(ValueError, match="both name the same fact"):
        fs.plot(rows="variable", cols="standard_name")


def test_a_duplicate_variable_source_pair_is_refused():
    """Two members landing in the same (variable, source) cell can't be drawn
    -- a mesh panel has no second channel to overlay them onto, the way a
    line's colour would.
    """
    from ocean_skill.plot.matplotlib_renderer import time_depth_grid

    item = {
        "field": _point_time_depth(),
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "label": "station_a",
    }
    with pytest.raises(ValueError, match="land in the same cell"):
        time_depth_grid([item, item], cols="variable")


def test_a_single_panel_facet_is_refused_in_matplotlib_and_dropped_in_holoviews(stub):
    stub(_point_time_depth())
    with pytest.raises(TypeError, match="needs several panels to facet"):
        _make().plot(cols="variable")

    with pytest.warns(UserWarning, match="need several panels to facet"):
        obj = _make().plot(cols="variable", renderer="holoviews")
    import holoviews as hv

    assert isinstance(obj, hv.QuadMesh | hv.Overlay)


def test_facet_titles_override_in_row_major_grid_cell_order(monkeypatch):
    _station_grid(monkeypatch, ["station_a", "station_b"])
    from ocean_skill.field import field as make_field

    fs = make_field(["station_a", "station_b"], [NITRATE, "silicate"])
    override = [None, "My Panel", None, "Blank Cell Override"]
    static = fs.plot(cols="variable", titles=override)
    titles = [ax.get_title() for ax in static.axes if ax.get_title()]
    assert titles[1] == "My Panel"
    assert "Blank Cell Override" in titles

    with pytest.raises(ValueError, match="needs one entry per panel"):
        fs.plot(cols="variable", titles=["only one"])


def test_cols_variable_shares_the_x_axis_within_each_station_row(monkeypatch):
    """``sharex`` auto-shares within one station's own row (its several
    variables share one deployment window) but not across different stations
    -- which may have disjoint deployment windows -- see
    ``test_a_time_depth_set_sharex_option`` for the single-axis precedent.
    """
    from ocean_skill.field import field as make_field

    _station_grid(monkeypatch, ["station_a", "station_b"])
    fs = make_field(["station_a", "station_b"], [NITRATE, "silicate"])

    static = fs.plot(cols="variable")
    panel_axes = static.axes[:4]
    # row 0: station_a's two variable panels (indices 0, 1) share one x axis
    assert panel_axes[0].get_shared_x_axes().joined(panel_axes[0], panel_axes[1])
    # different stations (row 0 vs row 1) are not auto-shared
    assert not panel_axes[0].get_shared_x_axes().joined(panel_axes[0], panel_axes[2])

    obj = fs.plot(cols="variable", renderer="holoviews")
    assert len(obj) == 4


# -- save() --------------------------------------------------------------------------------


def test_save_writes_a_figure_for_a_time_depth_field(tmp_path, stub):
    from ocean_skill import outputs

    outputs.set_base(tmp_path)
    try:
        stub(_point_time_depth())
        paths = _make().save("proj")
        assert paths["figure"].exists()
    finally:
        outputs.set_base(None)


# -- spec registration -----------------------------------------------------------------


def test_time_depth_is_a_registered_family():
    from ocean_skill.plot.spec import FAMILIES

    assert "time_depth" in FAMILIES
