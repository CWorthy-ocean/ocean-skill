"""Tests for the point-series path on a single, uncompared source (:class:`Field`).

Mirrors ``tests/test_facet.py``'s stub pattern (``comparison.prepare_source`` swapped
out, so these exercise :class:`~ocean_skill.field.Field`'s own logic rather than a
catalog) but for the *other* shape a reduction can take: a select that narrows both
horizontal axes to one position has nothing left to lay out as columns, so it draws as
a line instead of map panels (see :attr:`Field.family`).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

NITRATE = "nitrate"


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
    import ocean_skill.comparison as comparison

    def use(field_da):
        monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (field_da, None))

    return use


def _make(**kwargs):
    from ocean_skill.field import field as make_field

    return make_field("stub", NITRATE, **kwargs)


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
    with pytest.raises(ValueError, match="no surviving time axis"):
        f.plot()


def test_a_point_profile_with_no_time_refuses_to_plot(stub):
    """Depth survives, but there is no time to run a line along either."""
    stub(_point_with_depth().isel(time=0))
    f = _make()
    assert not f.is_series
    with pytest.raises(ValueError, match="no surviving time axis"):
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
