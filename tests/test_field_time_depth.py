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


def test_a_time_depth_set_refuses_to_plot(stub):
    stub(_point_time_depth())
    fs = _make_set([NITRATE, "silicate"])
    with pytest.raises(ValueError, match="depth against time"):
        fs.plot()


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
