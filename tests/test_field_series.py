"""Tests for the point-series path on a single, uncompared source (:class:`Field`).

Mirrors ``tests/test_facet.py``'s stub pattern (``comparison.prepare_source`` swapped
out, so these exercise :class:`~ocean_skill.field.Field`'s own logic rather than a
catalog) but for the *other* shape a reduction can take: a select that narrows both
horizontal axes to one position has nothing left to lay out as columns, so it draws as
a line instead of map panels (see :attr:`Field.family`).
"""

from __future__ import annotations

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
    standing on a model column."""
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
    stub(_point_with_depth(depths=(0.0, 50.0, 100.0)))
    f = _make()
    items = f._series_items()
    assert len(items) == 3
    assert [item["aligned"].attrs["actual_depth"] for item in items] == [0.0, 50.0, 100.0]
    fig = f.plot()
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
    series."""
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
    """Fanning is season-specific -- any other extra axis keeps today's error."""
    stub(_point_with_season().rename(season="month"))
    with pytest.raises(ValueError, match=r"\['month'\]"):
        _make()._profile_items()


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
    stub(_point_with_depth(depths=(0.0, 50.0, 100.0)))
    fs = _make_set([NITRATE, SILICATE])
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


def test_a_map_shaped_member_refuses_the_set_plot(stub):
    stub(_gridded_map())
    fs = _make_set([NITRATE, SILICATE])
    with pytest.raises(ValueError, match="overlaid lines"):
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
