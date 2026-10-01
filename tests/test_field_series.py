"""Tests for the point-series path on a single, uncompared source (:class:`Field`).

Mirrors ``tests/test_facet.py``'s stub pattern (``comparison.prepare_source`` swapped
out, so these exercise :class:`~ocean_skill.field.Field`'s own logic rather than a
catalog) but for the *other* shape a reduction can take: a select that narrows both
horizontal axes to one position has nothing left to lay out as columns, so it draws as
a line instead of map panels (see :attr:`Field.family`).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

NITRATE = "nitrate"
SILICATE = "silicate"


def _point_series(n: int = 12, *, lon_name: str = "lon", lat_name: str = "lat"):
    """A field already reduced to one place through time."""
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    values = 8.0 + np.sin(np.arange(n) / 3.0)
    da = xr.DataArray(
        values,
        dims="time",
        coords={"time": time},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )
    return da.assign_coords(**{lon_name: -144.245, lat_name: 49.978})


def _gridded_map(nt: int = 3):
    """An ordinary map with a surviving time facet -- the pre-existing behavior."""
    time = pd.date_range("2012-01-01", periods=nt, freq="MS")
    return xr.DataArray(
        np.random.default_rng(0).normal(5.0, 1.0, (nt, 8, 10)),
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": np.linspace(20, 30, 8),
            "lon": np.linspace(-100, -90, 10),
        },
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


def _point_with_season(depths=(0.0, 50.0, 100.0), seasons=("DJF", "MAM", "JJA", "SON")):
    """A point profile whose time axis was reduced to a season groupby -- the
    shape ``operators.aggregate({"time": {"groupby": "season", ...}})`` leaves
    standing on a model column.
    """
    depth = np.array(depths)
    values = 8.0 + np.arange(len(seasons))[:, None] + 0.01 * depth[None, :]
    da = xr.DataArray(
        values,
        dims=("season", "depth"),
        coords={"season": list(seasons), "depth": depth},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )
    return da.assign_coords(lon=-144.245, lat=49.978)


def _point_with_depth(n: int = 6, depths=(0.0, 50.0, 100.0)):
    """A point whose vertical axis also survives -- one line per level."""
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


def _make_set(variables, **kwargs):
    from ocean_skill.field import field as make_field

    return make_field("stub", variables, **kwargs)


# -- family inference: series vs field_facet -------------------------------------------


def test_a_point_with_time_is_a_series(stub):
    stub(_point_series())
    f = _make()
    assert f.is_series
    assert f.family == "series"
    assert "one place" in f.family_reason


def test_a_curvilinear_point_is_also_a_series(stub):
    """A ROMS point (scalar lon_rho/lat_rho) is recognized the same way."""
    stub(_point_series(lon_name="lon_rho", lat_name="lat_rho"))
    assert _make().is_series


def test_a_gridded_field_stays_field_facet(stub):
    """The pre-existing behavior: a real map with a time facet is unaffected."""
    stub(_gridded_map())
    f = _make()
    assert not f.is_series
    assert f.family == "field_facet"
    assert f.facet_dim == "time"


# -- .plot() smoke, both renderers ------------------------------------------------------


def test_a_point_series_draws_one_line_in_both_renderers(stub):
    stub(_point_series())
    fig = _make().plot()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 1
    assert fig.axes[0].lines[0].get_label() == "stub"

    import holoviews as hv

    obj = _make().plot(renderer="holoviews")
    assert len(obj.traverse(lambda x: x, [hv.Curve])) == 1


# -- a time-groupby's surviving dim, with no depth left, is a plain series --------------


def _point_month_climatology_no_depth(n_years: int = 2):
    """A station's month climatology with the vertical axis collapsed too --
    the shape ``aggregate={"time": {"groupby": "month", "reduce": "mean"},
    "depth": "mean"}`` leaves standing: one line, month on x. Built through
    the real reduction (:func:`ocean_skill.operators.aggregate`) so ``month``
    carries the marker :func:`~ocean_skill.operators.time_axis_dim` reads.
    """
    from ocean_skill.operators import aggregate

    n = 12 * n_years
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    values = 8.0 + np.sin(np.arange(n) / 3.0)
    da = xr.DataArray(
        values, dims="time", coords={"time": time}, name=NITRATE,
        attrs={"units": "mmol m-3"},
    ).assign_coords(lon=-21.8, lat=64.3)
    return aggregate(da, {"time": {"groupby": "month", "reduce": "mean"}})


def test_a_month_climatology_with_no_depth_is_a_series(stub):
    stub(_point_month_climatology_no_depth())
    f = _make(aggregate={"time": {"groupby": "month", "reduce": "mean"}})
    assert f.is_series
    assert f.family == "series"
    fig = f.plot()
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    assert ax.get_xlabel() == "month"
    assert list(ax.lines[0].get_xdata()) == list(range(1, 13))
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    import holoviews as hv

    obj = f.plot(renderer="holoviews")
    curves = obj.traverse(lambda x: x, [hv.Curve])
    assert len(curves) == 1
    assert curves[0].kdims[0].name == "month"


def test_a_plain_datetime_series_still_gets_the_date_axis(stub):
    """No-regression pin: an ordinary time series is untouched by any of this."""
    stub(_point_series())
    fig = _make().plot()
    assert fig.axes[0].get_xlabel() == "time"


# -- undrawable shapes: neither a map nor a line ----------------------------------------


def test_a_fully_collapsed_point_refuses_to_plot(stub):
    stub(_point_series().mean("time"))
    f = _make()
    assert not f.is_series
    assert not f.is_profile
    with pytest.raises(ValueError, match="no surviving time or depth axis"):
        f.plot()


def test_a_point_with_depth_and_no_time_draws_as_a_profile(stub):
    """Depth survives with no time standing: a profile, not a refusal."""
    stub(_point_with_depth().isel(time=0))
    f = _make()
    assert not f.is_series
    assert f.is_profile
    assert f.family == "profile"
    fig = f.plot()
    assert len(fig.axes[0].lines) == 1
    ydata = fig.axes[0].lines[0].get_ydata()
    assert sorted(ydata) == [0.0, 50.0, 100.0]


def test_a_point_with_neither_time_nor_depth_refuses_to_plot(stub):
    """Nothing survives at all: no map, no series, no profile to draw."""
    stub(_point_with_depth().isel(time=0, depth=0))
    f = _make()
    assert not f.is_series
    assert not f.is_profile
    with pytest.raises(ValueError, match="no surviving time or depth axis"):
        f.plot()


# -- a surviving vertical axis fans into one line per level -----------------------------


def test_a_point_with_depth_fans_into_one_item_per_level(stub):
    """An explicit list of levels keeps the per-level lines (see
    ``Field.is_time_depth``): a bare or banded select at this same point instead
    draws the new ``time_depth`` panel (tests/test_field_time_depth.py).
    """
    stub(_point_with_depth(depths=(0.0, 50.0, 100.0)))
    f = _make()
    items = f._series_items()
    assert len(items) == 3
    assert [item["aligned"].attrs["actual_depth"] for item in items] == [0.0, 50.0, 100.0]
    fig = _make(select={"depth": [0.0, 50.0, 100.0]}).plot()
    assert len(fig.axes[0].lines) == 3


def test_a_non_vertical_extra_axis_is_refused(stub):
    stub(_point_with_depth().expand_dims(member=[1, 2]))
    with pytest.raises(ValueError, match=r"\['member'\]"):
        _make().plot()


# -- a surviving season axis fans into one profile item per season ---------------------


def test_a_seasonal_point_column_is_a_profile_fanned_per_season(stub):
    """The one exception to _profile_items' "no axis beyond depth" rule: a
    surviving season axis fans into one item per season, in coordinate
    (chronological) order -- the same idiom as fanning depth levels for a
    series.
    """
    stub(_point_with_season())
    f = _make()
    assert not f.is_series
    assert f.is_profile
    assert f.family == "profile"
    items = f._profile_items()
    assert len(items) == 4
    assert [item["aligned"]["season"].item() for item in items] == [
        "DJF",
        "MAM",
        "JJA",
        "SON",
    ]
    for item in items:
        assert "season" not in item["aligned"].dims  # scalar, not a surviving axis
        assert list(item["aligned"]["value"].dims) == ["depth"]


def test_a_non_season_extra_axis_on_a_profile_is_still_refused(stub):
    """Fanning stays season- (or marked-month-) specific: an ordinary axis
    just named ``month`` -- one that never went through a real
    ``aggregate={"time": {"groupby": "month", ...}}`` reduction, so its
    coordinate carries none of :data:`~ocean_skill.operators.TIME_GROUPBY_ATTR`
    -- keeps today's error. See
    ``test_a_month_climatology_fans_into_one_item_per_selected_month`` for the
    marked case, name-only magic being deliberately excluded here.
    """
    stub(_point_with_season().rename(season="month"))
    with pytest.raises(ValueError, match=r"\['month'\]"):
        _make()._profile_items()


# -- an explicit month list turns a marked month axis into a profile fan too -----------


def _point_month_climatology_for_fan(n_years: int = 2, depths=(0.0, 50.0, 100.0)):
    """The same month climatology :mod:`tests.test_field_time_depth` builds,
    reused here for the escape hatch that routes it to the profile family
    instead: an explicit ``select={"month": [...]}`` list.
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


def test_a_month_climatology_fans_into_one_item_per_selected_month(stub):
    """A marked month axis, narrowed by an explicit ``select={"month": [...]}``
    list, fans into one profile item per month -- the same idiom
    ``fan_season`` already gives a season axis (see
    :func:`ocean_skill.plot.profile.fan_season`), and mirrors the depth-list
    exception :attr:`~ocean_skill.field.Field.is_time_depth` already makes.
    """
    climatology = _point_month_climatology_for_fan()
    stub(climatology.sel(month=[1, 4, 7]))
    f = _make(
        aggregate={"time": {"groupby": "month", "reduce": "mean"}},
        select={"month": [1, 4, 7]},
    )
    assert not f.is_time_depth
    assert f.is_profile
    assert f.family == "profile"
    items = f._profile_items()
    assert len(items) == 3
    assert [item["aligned"]["month"].item() for item in items] == [1, 4, 7]
    for item in items:
        assert "month" not in item["aligned"].dims  # scalar, not a surviving axis


def test_a_scalar_month_select_labels_a_single_profile(stub):
    climatology = _point_month_climatology_for_fan()
    stub(climatology.sel(month=4))
    f = _make(
        aggregate={"time": {"groupby": "month", "reduce": "mean"}},
        select={"month": 4},
    )
    assert f.family == "profile"
    fig = f.plot()
    assert "Apr" in fig.axes[0].get_title()


def test_a_month_fan_colours_and_legends_by_month_in_both_renderers(stub):
    climatology = _point_month_climatology_for_fan()
    stub(climatology.sel(month=[1, 4, 7]))
    f = _make(
        aggregate={"time": {"groupby": "month", "reduce": "mean"}},
        select={"month": [1, 4, 7]},
    )
    fig = f.plot()
    ax = fig.axes[0]
    legend = ax.get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == ["Jan", "Apr", "Jul"]
    # An overlay of several months claims no single "when" in the title --
    # they are told apart by the legend instead (mirrors the season rule).
    assert "Jan" not in ax.get_title()

    import holoviews as hv

    obj = f.plot(renderer="holoviews")
    assert obj.traverse(lambda x: x, [hv.Curve])


# -- .movie() has nothing to play for a point series ------------------------------------


def test_movie_refuses_a_point_series(stub):
    stub(_point_series())
    with pytest.raises(ValueError, match="nothing to play"):
        _make().movie()


# -- save() ------------------------------------------------------------------------------


def test_save_writes_a_figure_for_a_point_series(tmp_path, stub):
    from ocean_skill import outputs

    outputs.set_base(tmp_path)
    try:
        stub(_point_series())
        paths = _make().save("proj")
        assert paths["figure"].exists()
    finally:
        outputs.set_base(None)


# -- several variables from one source (FieldSet) ---------------------------------------


def test_a_list_of_variables_returns_a_fieldset(stub):
    from ocean_skill.field import Field, FieldSet

    stub(_point_series())
    fs = _make_set([NITRATE, SILICATE])
    assert isinstance(fs, FieldSet)
    assert len(fs) == 2
    assert all(isinstance(f, Field) for f in fs)
    assert fs[0].standard_name != fs[1].standard_name


def test_a_one_element_list_is_still_a_set(stub):
    from ocean_skill.field import FieldSet

    stub(_point_series())
    fs = _make_set([NITRATE])
    assert isinstance(fs, FieldSet)
    assert len(fs) == 1
    fig = fs.plot()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 1


def test_two_variables_share_a_panel_with_a_secondary_axis(stub):
    stub(_point_series())
    fs = _make_set([NITRATE, SILICATE])
    fig = fs.plot()
    assert len(fig.axes) == 2  # one panel plus its twin
    assert len([ax for ax in fig.axes if ax.get_title()]) == 1

    import holoviews as hv

    obj = _make_set([NITRATE, SILICATE]).plot(renderer="holoviews")
    assert len(obj.traverse(lambda x: x, [hv.Curve])) == 2


def test_secondary_y_false_stacks_two_variables(stub):
    stub(_point_series())
    fig = _make_set([NITRATE, SILICATE]).plot(secondary_y=False)
    assert len([ax for ax in fig.axes if ax.get_title()]) == 2

    obj = _make_set([NITRATE, SILICATE]).plot(
        secondary_y=False, renderer="holoviews"
    )
    import holoviews as hv

    assert len(obj.traverse(lambda x: x, [hv.Curve])) == 2


def test_three_variables_become_three_rows(stub):
    stub(_point_series())
    fig = _make_set([NITRATE, SILICATE, "oxygen"]).plot()
    assert len([ax for ax in fig.axes if ax.get_title()]) == 3

    obj = _make_set([NITRATE, SILICATE, "oxygen"]).plot(renderer="holoviews")
    import holoviews as hv

    titled = obj.traverse(lambda x: x.opts.get("plot").kwargs.get("title"), [hv.Overlay])
    assert len([t for t in titled if t]) == 3


def test_depth_fanout_multiplies_items_per_variable(stub):
    """An explicit list of levels keeps every member drawing as ``series`` -- a
    bare or banded select instead makes every member ``time_depth``, which
    ``FieldSet.plot`` refuses (see ``test_a_time_depth_set_refuses_to_plot``).
    """
    stub(_point_with_depth(depths=(0.0, 50.0, 100.0)))
    fs = _make_set([NITRATE, SILICATE], select={"depth": [0.0, 50.0, 100.0]})
    assert len(fs._items()) == 6
    fig = fs.plot()
    assert sum(len(ax.get_lines()) for ax in fig.axes) == 6


def test_a_list_passed_to_field_itself_is_refused():
    from ocean_skill.field import Field

    with pytest.raises(TypeError, match="list of variable specs"):
        Field("some_source", [NITRATE, SILICATE])


def test_an_empty_list_is_refused():
    with pytest.raises(ValueError, match="names nothing"):
        _make_set([])


def test_duplicate_variables_are_dropped(stub, capsys):
    stub(_point_series())
    fs = _make_set([NITRATE, NITRATE])
    assert len(fs) == 1
    assert "duplicate" in capsys.readouterr().out


# -- variable="all" (every variable a source declares) -----------------------------------


def _resolve_stub(monkeypatch, variables):
    """Mock ``catalog.resolve("stub")`` to declare ``variables`` in its metadata."""
    from ocean_skill import catalog

    def resolve(source):
        if source == "stub":
            return SimpleNamespace(metadata={"variables": list(variables)})
        raise KeyError(source)

    monkeypatch.setattr(catalog, "resolve", resolve, raising=True)


def test_variable_all_expands_to_every_declared_variable(stub, monkeypatch):
    from ocean_skill.field import FieldSet

    stub(_point_series())
    _resolve_stub(monkeypatch, [NITRATE, SILICATE, "oxygen"])
    fs = _make_set("all")
    assert isinstance(fs, FieldSet)
    assert len(fs) == 3
    assert {f.standard_name for f in fs} == {
        f.standard_name for f in _make_set([NITRATE, SILICATE, "oxygen"])
    }


def test_variable_all_deduplicates_aliases(stub, monkeypatch, capsys):
    stub(_point_series())
    _resolve_stub(monkeypatch, [NITRATE, "nitrate"])
    fs = _make_set("all")
    assert len(fs) == 1
    assert "duplicate" in capsys.readouterr().out


def test_variable_all_with_a_source_list_is_refused():
    from ocean_skill.field import field as make_field

    with pytest.raises(ValueError, match='variable="all"'):
        make_field(["stub_a", "stub_b"], "all")


def test_variable_all_on_a_source_with_no_declared_variables(monkeypatch):
    _resolve_stub(monkeypatch, [])
    with pytest.raises(ValueError, match="declares no variables"):
        _make_set("all")


def test_map_shaped_members_still_faceted_over_time_refuse_the_set_plot(stub):
    """A set whose members all draw as maps composes (see
    ``tests/test_field_map_grid.py``) -- but each map still has to reduce to
    *one* instant first, the same read-cheap/post-load refusal a solo
    ``Field.plot()`` would give this same field (see
    ``tests/test_field_grid_defaults.py``).
    """
    stub(_gridded_map())
    fs = _make_set([NITRATE, SILICATE])
    with pytest.raises(ValueError, match="no single default"):
        fs.plot()


def test_movie_refuses_a_fieldset(stub):
    stub(_point_series())
    with pytest.raises(ValueError, match="nothing to play"):
        _make_set([NITRATE, SILICATE]).movie()


def test_fieldset_save_writes_one_figure(tmp_path, stub):
    from ocean_skill import outputs

    outputs.set_base(tmp_path)
    try:
        stub(_point_series())
        paths = _make_set([NITRATE, SILICATE]).save("proj")
        assert paths["figure"].exists()
    finally:
        outputs.set_base(None)


def test_rows_facet_overrides_the_secondary_axis(stub):
    stub(_point_series())
    fig = _make_set([NITRATE, SILICATE]).plot(rows="variable")
    assert len([ax for ax in fig.axes if ax.get_title()]) == 2


# -- the same twin-axis merge, but for a profile FieldSet -------------------------------


def test_two_profile_variables_share_a_panel_with_a_top_axis(stub):
    stub(_point_with_depth().isel(time=0))
    fs = _make_set([NITRATE, SILICATE])
    fig = fs.plot()
    assert len(fig.axes) == 2  # one panel plus its twin
    assert len([ax for ax in fig.axes if ax.get_title()]) == 1

    import holoviews as hv

    obj = _make_set([NITRATE, SILICATE]).plot(renderer="holoviews")
    assert len(obj.traverse(lambda x: x, [hv.Curve])) == 2


def test_secondary_x_false_stacks_two_profile_variables(stub):
    stub(_point_with_depth().isel(time=0))
    fig = _make_set([NITRATE, SILICATE]).plot(secondary_x=False)
    assert len([ax for ax in fig.axes if ax.get_title()]) == 2


# -- several sources of one variable (FieldSet, fanned by source) -----------------------
#
# ``osk.find(...)`` always returns a list of source names, even a single match, so
# ``osk.field(osk.find(...), "temp")`` is the chain this exercises -- one Field per
# source, sharing this same stubbed data (the stub does not care what source name it
# is called with), pooled the same way a list of variables already is.


def _make_source_set(sources, **kwargs):
    from ocean_skill.field import field as make_field

    return make_field(sources, NITRATE, **kwargs)


def test_a_list_of_sources_returns_a_fieldset(stub):
    from ocean_skill.field import Field, FieldSet

    stub(_point_series())
    fs = _make_source_set(["stub_a", "stub_b"])
    assert isinstance(fs, FieldSet)
    assert len(fs) == 2
    assert all(isinstance(f, Field) for f in fs)
    assert [f.source for f in fs] == ["stub_a", "stub_b"]


def test_a_one_element_source_list_is_still_a_set(stub):
    from ocean_skill.field import FieldSet

    stub(_point_series())
    fs = _make_source_set(["stub_a"])
    assert isinstance(fs, FieldSet)
    assert len(fs) == 1
    fig = fs.plot()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 1


def test_two_sources_overlay_as_lines_in_one_panel(stub):
    """One variable, two sources -- no secondary axis (that's for two variables);
    both lines share the one panel, told apart by source.
    """
    stub(_point_series())
    fs = _make_source_set(["stub_a", "stub_b"])
    fig = fs.plot()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 2
    assert {line.get_label() for line in fig.axes[0].lines} == {"stub_a", "stub_b"}

    import holoviews as hv

    obj = _make_source_set(["stub_a", "stub_b"]).plot(renderer="holoviews")
    assert len(obj.traverse(lambda x: x, [hv.Curve])) == 2


def test_source_list_and_variable_list_fan_the_cross_product(stub):
    from ocean_skill.field import field as make_field

    stub(_point_series())
    fs = make_field(["stub_a", "stub_b"], [NITRATE, SILICATE])
    assert len(fs) == 4
    pairs = {(f.source, f.standard_name) for f in fs}
    assert len(pairs) == 4


def test_a_source_list_passed_to_field_itself_is_refused():
    from ocean_skill.field import Field

    with pytest.raises(TypeError, match="list of sources"):
        Field(["stub_a", "stub_b"], NITRATE)


def test_an_empty_source_list_is_refused():
    with pytest.raises(ValueError, match="names nothing"):
        _make_source_set([])


def test_duplicate_sources_are_dropped(stub, capsys):
    stub(_point_series())
    fs = _make_source_set(["stub_a", "stub_a"])
    assert len(fs) == 1
    assert "duplicate" in capsys.readouterr().out


def test_the_vocabulary_warning_fires_once_across_many_sources(stub):
    """The same alias resolves the same way for every source -- one warning for
    the whole fan-out, not one per source (mirrors ``compare()``'s identical
    up-front resolution).
    """
    stub(_point_series())
    with pytest.warns(UserWarning, match="resolved to standard_name") as record:
        _make_source_set(["stub_a", "stub_b", "stub_c"])
    hits = [r for r in record if "resolved to standard_name" in str(r.message)]
    assert len(hits) == 1


# -- FieldSet.sel() -- narrowing a set built once, without recomputing -------------------


def test_sel_by_variable_narrows_to_matching_members(stub):
    from ocean_skill.field import FieldSet

    stub(_point_series())
    fs = _make_set([NITRATE, SILICATE])
    narrowed = fs.sel(variable=NITRATE)
    assert isinstance(narrowed, FieldSet)
    assert len(narrowed) == 1
    assert narrowed[0].standard_name == fs[0].standard_name


def test_sel_by_variable_matches_through_vocabulary_aliases(stub):
    """``.sel(variable="temp")`` matches a member built from ``"temperature"``, and
    vice versa -- both resolve to the same standard_name."""
    stub(_point_series())
    fs = _make_set(["temp", "salt"])
    assert len(fs.sel(variable="temperature")) == 1
    assert fs.sel(variable="temperature")[0].variable == fs[0].variable


def test_sel_by_variable_accepts_a_list(stub):
    stub(_point_series())
    fs = _make_set([NITRATE, SILICATE])
    narrowed = fs.sel(variable=[NITRATE, SILICATE])
    assert len(narrowed) == 2


def test_sel_by_source_narrows_to_one_station(stub):
    stub(_point_series())
    fs = _make_source_set(["stub_a", "stub_b", "stub_c"])
    narrowed = fs.sel(source="stub_b")
    assert len(narrowed) == 1
    assert narrowed[0].source == "stub_b"


def test_sel_combines_variable_and_source(stub):
    from ocean_skill.field import field as make_field

    stub(_point_series())
    fs = make_field(["stub_a", "stub_b"], [NITRATE, SILICATE])
    narrowed = fs.sel(variable=NITRATE, source="stub_a")
    assert len(narrowed) == 1
    assert narrowed[0].source == "stub_a"
    assert narrowed[0].standard_name == fs.sel(variable=NITRATE)[0].standard_name


def test_sel_with_no_match_raises_and_lists_whats_present(stub):
    stub(_point_series())
    fs = _make_set([NITRATE, SILICATE])
    with pytest.raises(ValueError, match="no fields match"):
        fs.sel(variable="phosphate")


def test_sel_with_unknown_key_raises(stub):
    stub(_point_series())
    fs = _make_set([NITRATE, SILICATE])
    with pytest.raises(ValueError, match="does not know"):
        fs.sel(depth="surface")


def test_sel_never_touches_data(stub, monkeypatch):
    """.sel() reads only construction-time metadata -- narrowing a set never loads
    the members it drops (or the ones it keeps)."""
    from ocean_skill import comparison

    stub(_point_series())
    fs = _make_set([NITRATE, SILICATE])

    def boom(*a, **k):
        raise AssertionError(".sel() must not call prepare_source")

    monkeypatch.setattr(comparison, "prepare_source", boom)
    narrowed = fs.sel(variable=NITRATE)
    assert len(narrowed) == 1


def test_sel_then_plot_renders_only_the_selected_variable(stub):
    stub(_point_series())
    fs = _make_set([NITRATE, SILICATE])
    fig = fs.sel(variable=NITRATE).plot()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 1


def test_sel_then_plot_rows_by_source(stub):
    from ocean_skill.field import field as make_field

    stub(_point_series())
    fs = make_field(["stub_a", "stub_b"], [NITRATE, SILICATE])
    fig = fs.sel(variable=NITRATE).plot(rows="source")
    assert len(fig.axes) == 2


# -- a member whose source doesn't actually carry the requested variable ----------------
#
# Regression for `osk.field([...], "alkalinity").plot()` raising a bare KeyError when
# one station's source never carried the variable at all (`.sel()` matches by the
# *requested* name only, never touching data -- see FieldSet.sel's own docstring --
# so a station like that survives narrowing and only fails at .plot() time). Mirrors
# `compare()`'s own `skip_missing` precedent (tests/test_availability_probe.py) via
# the same `comparison._variable_available` probe.


@pytest.fixture
def unavailable_on(monkeypatch):
    """Report the given sources as not carrying whatever variable is asked for."""
    from ocean_skill import comparison

    def use(*missing_sources: str):
        real = comparison._variable_available

        def fake(source, variable, **kwargs):
            if source in missing_sources:
                return False
            return real(source, variable, **kwargs)

        monkeypatch.setattr(comparison, "_variable_available", fake)

    return use


def test_plot_skips_a_member_whose_source_lacks_the_variable(stub, unavailable_on):
    from ocean_skill.field import field as make_field

    stub(_point_series())
    unavailable_on("stub_b")
    fs = make_field(["stub_a", "stub_b"], NITRATE)
    with pytest.warns(UserWarning, match="stub_b"):
        fig = fs.plot()
    # only the surviving member (stub_a) drew -- one line, not two
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 1


def test_plot_raises_a_clear_error_when_nothing_is_available(stub, unavailable_on):
    from ocean_skill.field import field as make_field

    stub(_point_series())
    unavailable_on("stub_a", "stub_b")
    fs = make_field(["stub_a", "stub_b"], NITRATE)
    with pytest.raises(ValueError, match="none of this set's members"):
        fs.plot()


def test_plot_is_unchanged_and_silent_when_every_member_is_available(stub, unavailable_on):
    from ocean_skill.field import field as make_field

    stub(_point_series())
    fs = make_field(["stub_a", "stub_b"], NITRATE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = fs.plot()
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 2


# -- FieldSet.title: a default title the set carries ------------------------------
#
# Set by whatever built the set and knows something its panels cannot say (an
# Extremum's series names the extremum -- see tests/test_extrema.py). These pin the
# FieldSet half of that contract: it is a setdefault that every branch of plot() sees.


def _suptitles(fs, **kwargs):
    """Return the figure title from each renderer, as ``(static, interactive)``."""
    fig = fs.plot(**kwargs)
    static = fig._suptitle.get_text() if fig._suptitle is not None else None
    obj = fs.plot(renderer="holoviews", **kwargs)
    return static, obj.opts.get("plot").kwargs.get("title")


def test_a_set_without_a_title_draws_none(stub):
    stub(_point_series())
    fs = _make_set([NITRATE, SILICATE])
    assert fs.title is None
    assert _suptitles(fs) == (None, "")


def test_a_set_title_is_the_default_figure_title_in_both_renderers(stub):
    from ocean_skill.field import FieldSet

    stub(_point_series())
    fs = FieldSet(list(_make_set([NITRATE, SILICATE])), title="Why this is here")
    assert _suptitles(fs) == ("Why this is here", "Why this is here")
    assert _suptitles(fs, title="Mine") == ("Mine", "Mine")  # a caller's title wins
    static, interactive = _suptitles(fs, title="")  # ...and "" really means none
    assert not static and not interactive


def test_a_set_title_survives_sel(stub):
    from ocean_skill.field import FieldSet

    stub(_point_series())
    fs = FieldSet(list(_make_set([NITRATE, SILICATE])), title="Kept")
    assert fs.sel(variable=NITRATE).title == "Kept"
    assert _suptitles(fs.sel(variable=NITRATE)) == ("Kept", "Kept")


def test_a_set_title_reaches_the_lone_map_branch_too(stub):
    """A one-member set of maps hands off to Field.plot, which must see the title."""
    from ocean_skill.field import FieldSet

    stub(_gridded_map(nt=1))
    members = list(_make_set([NITRATE], select={"time": "2012-01"}))
    fs = FieldSet(members, title="Map set")
    assert fs.plot()._suptitle.get_text() == "Map set"
    assert fs.plot(title="Own")._suptitle.get_text() == "Own"


# -- domain-mean series: panel title and depth legend -----------------------------
#
# aggregate={"lon": "mean", "lat": "mean"} on a ROMS grid is one area-weighted mean of
# the whole domain (operators._horizontal_mean), which parks its scalar lon/lat at the
# bounding-box midpoint. That midpoint is bookkeeping, not a station -- on a dateline-
# straddling Pacific domain it is a spot nowhere near the data -- so the panel must not
# title it as one. And an explicit ["surface", 100, 200] list keeps a numeric z = 0 for
# the surface, spelled honestly in the coordinate's level_labels, which the legend must
# read instead of printing "0 m".

SPATIAL_MEAN = "area-weighted mean (cell_area)"
LEVEL_LABELS = ["surface", "100 m", "200 m"]


def _domain_mean_levels(
    *,
    labels=LEVEL_LABELS,
    depths=(0.0, -100.0, -200.0),
    n: int = 6,
    region=None,
    lon: float = 166.0,
    lat: float = 10.0,
):
    """Build what a whole-domain mean of a mixed-depth-list ROMS field leaves.

    ``z`` is the signed model coordinate (``comparison._surface_and_levels``), the
    scalar lon/lat the bounding-box midpoint ``_horizontal_mean`` assigns, and
    ``spatial_mean`` the description it leaves in attrs -- plus ``region`` only when
    a select box drove the mean.
    """
    time = pd.date_range("2010-07-01", periods=n, freq="MS")
    z_attrs = {} if labels is None else {"level_labels": list(labels)}
    values = 8.0 + np.random.default_rng(2).normal(0, 1, (n, len(depths)))
    attrs = {"units": "degC", "spatial_mean": SPATIAL_MEAN}
    if region is not None:
        attrs["region"] = list(region)
    da = xr.DataArray(
        values,
        dims=("time", "z"),
        coords={"time": time, "z": ("z", np.array(depths), z_attrs)},
        name=NITRATE,
        attrs=attrs,
    )
    return da.assign_coords(lon=lon, lat=lat)


def _mpl_panels(fig):
    """``[(title, [line labels]), ...]`` per drawn axes, twin axes folded in."""
    return [
        (ax.get_title(), [ln.get_label() for ln in ax.get_lines()])
        for ax in fig.axes
        if ax.get_lines() and not ax.get_label().startswith("_")
    ]


def _hv_panels(obj):
    """Return the same ``[(title, [curve labels]), ...]`` from the hv object."""
    import holoviews as hv

    return [
        (
            overlay.opts.get("plot").kwargs.get("title"),
            [c.label for c in overlay.traverse(lambda x: x, [hv.Curve])],
        )
        for overlay in obj.traverse(lambda x: x, [hv.Overlay])
    ]


def _legend_texts(ax):
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


def test_a_domain_mean_series_is_titled_a_domain_mean_not_a_station(stub):
    stub(_domain_mean_levels())
    fs = _make_set(
        [NITRATE],
        select={"depth": ["surface", 100, 200]},
        aggregate={"lon": "mean", "lat": "mean"},
    )
    ((title, _),) = _mpl_panels(fs.plot())
    ((hv_title, _),) = _hv_panels(fs.plot(renderer="holoviews"))
    assert title == hv_title
    assert "domain mean" in title
    assert "2010-07 to 2010-12" in title
    # The bounding-box midpoint is not a place anyone sampled; it must not be printed.
    assert "°N" not in title and "°E" not in title and "°W" not in title


def test_a_box_mean_keeps_reading_mean_over_its_region(stub):
    from ocean_skill.comparison import _region_label

    region = [165.0, 5.0, 175.0, 15.0]
    stub(_domain_mean_levels(region=region))
    fs = _make_set(
        [NITRATE],
        select={"depth": ["surface", 100, 200]},
        aggregate={"lon": "mean", "lat": "mean"},
    )
    ((title, _),) = _mpl_panels(fs.plot())
    ((hv_title, _),) = _hv_panels(fs.plot(renderer="holoviews"))
    assert title == hv_title
    assert f"mean over {_region_label(region)}" in title
    assert "domain mean" not in title


def test_a_real_station_still_titles_its_place(stub):
    """No ``spatial_mean`` attr -> unchanged: a point is a place."""
    stub(_point_series())
    ((title, _),) = _mpl_panels(_make_set([NITRATE]).plot())
    assert "50.0°N" in title and "144.2°W" in title
    assert "domain mean" not in title


def test_a_mixed_depth_list_legend_reads_surface_not_zero_m(stub):
    stub(_domain_mean_levels())
    fs = _make_set(
        [NITRATE],
        select={"depth": ["surface", 100, 200]},
        aggregate={"lon": "mean", "lat": "mean"},
    )
    fig = fs.plot()
    ((_, static_lines),) = _mpl_panels(fig)
    ((_, hv_curves),) = _hv_panels(fs.plot(renderer="holoviews"))
    assert static_lines == LEVEL_LABELS  # in the order asked for, labels exactly
    assert hv_curves == static_lines
    assert _legend_texts(fig.axes[0]) == LEVEL_LABELS
    assert "0 m" not in static_lines


def test_the_levels_are_told_apart_by_marker_and_by_colour_on_request(stub):
    """Same channels as a numeric list: colour is the variable, marker the depth."""
    stub(_domain_mean_levels())
    fs = _make_set(
        [NITRATE],
        select={"depth": ["surface", 100, 200]},
        aggregate={"lon": "mean", "lat": "mean"},
    )
    lines = _mpl_panels_lines(fs.plot())
    assert len({marker for _, _, marker in lines}) == 3
    # encode={"color": "depth"} keys colour on the labels, one hue per level.
    by_colour = _mpl_panels_lines(fs.plot(encode={"color": "depth", "marker": None}))
    assert len({colour for _, colour, _ in by_colour}) == 3
    assert [label for label, _, _ in by_colour] == LEVEL_LABELS


def _mpl_panels_lines(fig):
    return [
        (ln.get_label(), ln.get_color(), ln.get_marker())
        for ax in fig.axes
        for ln in ax.get_lines()
    ]


def test_several_variables_give_a_panel_each_with_the_three_depths(stub):
    """The motivating call: ``[temperature, salinity, ...]`` over three depths."""
    stub(_domain_mean_levels())
    fs = _make_set(
        [NITRATE, SILICATE, "oxygen"],
        select={"depth": ["surface", 100, 200]},
        aggregate={"lon": "mean", "lat": "mean"},
    )
    static = _mpl_panels(fs.plot())
    interactive = _hv_panels(fs.plot(renderer="holoviews"))
    assert len(static) == len(interactive) == 3
    assert static == interactive
    for title, labels in static:
        assert "domain mean" in title
        assert len(labels) == 3
        # the variable is still in each entry here (it varies across the figure);
        # what changed is that the surface line says "surface".
        assert [label.split(" · ")[-1] for label in labels] == LEVEL_LABELS
    # Faceting on variable drops it from the entries: exactly the three depths.
    faceted = _mpl_panels(fs.plot(rows="variable"))
    assert all(labels == LEVEL_LABELS for _, labels in faceted)
    assert all(labels == LEVEL_LABELS for _, labels in _hv_panels(
        fs.plot(rows="variable", renderer="holoviews")
    ))


def test_a_plain_numeric_depth_list_reads_as_it_always_did(stub):
    stub(_domain_mean_levels(labels=None, depths=(0.0, 100.0, 200.0)))
    fs = _make_set(
        [NITRATE],
        select={"depth": [0, 100, 200]},
        aggregate={"lon": "mean", "lat": "mean"},
    )
    ((_, static_lines),) = _mpl_panels(fs.plot())
    ((_, hv_curves),) = _hv_panels(fs.plot(renderer="holoviews"))
    assert static_lines == ["0 m", "100 m", "200 m"]
    assert hv_curves == static_lines


def test_labels_that_no_longer_match_the_axis_are_ignored(stub):
    """A stale ``level_labels`` (axis narrowed since) must not mislabel anything."""
    stub(_domain_mean_levels(labels=["surface", "100 m"]))  # three levels, two labels
    fs = _make_set(
        [NITRATE],
        select={"depth": ["surface", 100, 200]},
        aggregate={"lon": "mean", "lat": "mean"},
    )
    ((_, labels),) = _mpl_panels(fs.plot())
    assert labels == ["0 m", "100 m", "200 m"]


def test_each_series_item_carries_its_levels_label(stub):
    stub(_domain_mean_levels())
    fs = _make_set(
        [NITRATE],
        select={"depth": ["surface", 100, 200]},
        aggregate={"lon": "mean", "lat": "mean"},
    )
    assert [item["depth_label"] for item in fs._items()] == LEVEL_LABELS
    # the realized number stays on the item too (stats, other consumers)
    assert [item["aligned"].attrs["actual_depth"] for item in fs._items()] == [
        0.0, 100.0, 200.0
    ]


# -- ...and end to end, through the real vertical transform and spatial mean -------


def _roms_time_column(n: int = 5):
    """Build a tiny ROMS-shaped dataset with time (test_depth_average.py's grid)."""
    n_s, hc, theta_s, theta_b = 20, 250.0, 5.0, 2.0

    def stretch(s):
        c = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
        return (np.exp(theta_b * c) - 1) / (1 - np.exp(-theta_b))

    sigma_r = (np.arange(1, n_s + 1) - n_s - 0.5) / n_s
    sigma_w = np.linspace(-1, 0, n_s + 1)
    shape = (n, n_s, 2, 2)
    dims = ("time", "s_rho", "eta_rho", "xi_rho")
    rng = np.random.default_rng(3)
    ds = xr.Dataset(
        {
            "temp": (dims, 10 + rng.random(shape), {"units": "degC"}),
            "salt": (dims, 34 + rng.random(shape), {"units": "PSU"}),
            "h": (("eta_rho", "xi_rho"), np.array([[2e3, 3e3], [4e3, 5e3]])),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((2, 2))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), stretch(sigma_w)),
        },
        coords={
            "time": pd.date_range("2010-07-01", periods=n, freq="MS"),
            "lon": (("eta_rho", "xi_rho"), np.array([[-95.0, -94.0], [-95.0, -94.0]])),
            "lat": (("eta_rho", "xi_rho"), np.array([[25.0, 25.0], [26.0, 26.0]])),
        },
    )
    return ds, {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": hc}}


def test_a_roms_domain_mean_over_surface_100_200_end_to_end(monkeypatch):
    """Run a ROMS domain mean over surface/100/200 through the real pipeline.

    ``osk.field(src, [temp, salt], select={"depth": ["surface", 100, 200]},
    aggregate={"lon": "mean", "lat": "mean"}).plot()`` -- the real pipeline's attrs
    (spatial_mean, the z coordinate's level_labels) must be what the plot reads.
    """
    from ocean_skill import comparison

    ds, meta = _roms_time_column()
    monkeypatch.setattr(
        comparison,
        "prepare_source",
        lambda source, variable, select, aggregate, **k: comparison._prepare(
            ds, meta, variable, select, aggregate
        ),
    )
    monkeypatch.setattr(comparison, "_variable_available", lambda *a, **k: True)
    fs = _make_set(
        ["temp", "salt"],
        select={"depth": ["surface", 100, 200]},
        aggregate={"lon": "mean", "lat": "mean"},
    )
    assert fs[0].data["z"].attrs["level_labels"] == LEVEL_LABELS
    assert fs[0].data.attrs["spatial_mean"]
    assert "region" not in fs[0].data.attrs

    static = _mpl_panels(fs.plot(secondary_y=False))
    interactive = _hv_panels(fs.plot(secondary_y=False, renderer="holoviews"))
    assert static == interactive
    assert len(static) == 2
    for title, labels in static:
        assert "domain mean" in title and "°N" not in title
        assert [label.split(" · ")[-1] for label in labels] == LEVEL_LABELS
