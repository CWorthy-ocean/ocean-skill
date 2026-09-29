"""A native s-level column under a raised free surface draws right side up.

ROMS's ``z_rho`` is a *height* (zero at mean sea level, positive up) that carries
the free surface in it, so on a shallow shelf at high tide (``zeta > 0``) the top
levels are genuinely above 0. Reading it as ``abs()`` mirrored those levels back
under the ones beneath them: the profile's top drew as a hook, its true surface
plotted ~1 m *deeper* than the level below it. Both vertical-profile families that
read a native column -- ``profile`` (either renderer) and ``time_depth`` -- must
negate it instead, the same as :func:`ocean_skill.plot.section.prepare_section`.

The same holds wherever else a native level's height is read back as a depth: the
``actual_depth`` label a fanned series line (:meth:`Field._series_items`) or the
surface default (:func:`ocean_skill.field._top_level`) records, and the level a
follow-up series is pinned to from an extremum
(:meth:`ocean_skill.extrema.Extremum.series`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.comparison import _prepare

ALKALINITY = "sea_water_alkalinity_expressed_as_mole_equivalent"
ZETA = 1.3  # metres above mean sea level: a high tide over a 14 m shelf


def _stretch(s, theta_s=5.0, theta_b=2.0):
    c = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
    return (np.exp(theta_b * c) - 1) / (1 - np.exp(-theta_b))


#: Bottom depth of each column a grid built by :func:`_raised_dataset` can carry.
_H = (14.0, 30.0, 45.0)
_NLEVELS = 20


def _raised_dataset(ncols=1, zetas=None):
    """Build a shallow ROMS grid with ``add_depth_coord`` applied, plus its meta.

    ``ncols`` columns along ``xi_rho`` (each a different bottom depth, ``_H``), and
    a free surface of ``ZETA`` (``+1.3 m``) at every one -- or, given ``zetas``, one
    per time step, adding a leading ``time`` axis. Built through the real
    :func:`ocean_skill.roms.add_depth_coord`, so ``z_rho`` is exactly what a model
    hands the plot: the top levels above 0 wherever ``zeta`` lifts them there.

    The tracer rises toward the surface, falls off toward the later columns, and
    grows with ``zeta``, so its maximum sits at the top level of the first column
    at the highest tide.
    """
    from ocean_skill import roms

    n = _NLEVELS
    sigma_r = (np.arange(1, n + 1) - n - 0.5) / n
    sigma_w = np.linspace(-1, 0, n + 1)
    profile = np.linspace(125.6, 126.1, n)[:, None, None]
    columns = (1.0 - 0.01 * np.arange(ncols))[None, None, :]
    tracer_dims = ("s_rho", "eta_rho", "xi_rho")
    zeta_dims = ("eta_rho", "xi_rho")
    tracer = profile * columns
    zeta = np.full((1, ncols), ZETA)
    coords = {}
    if zetas is not None:
        zetas = np.asarray(zetas, dtype="float64")
        tracer_dims, zeta_dims = ("time", *tracer_dims), ("time", *zeta_dims)
        tracer = (1.0 + 0.01 * zetas)[:, None, None, None] * tracer[None]
        zeta = zetas[:, None, None] * np.ones((1, 1, ncols))
        coords["time"] = pd.date_range("2010-10-31", periods=len(zetas), freq="h")
    ds = xr.Dataset(
        {
            "alk": (tracer_dims, tracer, {"units": "mmol/m^3"}),
            "zeta": (zeta_dims, zeta),
        },
        coords={
            **coords,
            "h": (("eta_rho", "xi_rho"), np.array([_H[:ncols]])),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((1, ncols))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
            "lon": (("eta_rho", "xi_rho"), 97.6 + 0.1 * np.arange(ncols)[None, :]),
            "lat": (("eta_rho", "xi_rho"), np.full((1, ncols), 16.1)),
        },
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": 250.0}}
    return roms.add_depth_coord(ds, meta), meta


def _raised_column(time=None) -> xr.DataArray:
    """One shallow ROMS column, prepared at its point, with ``zeta = +1.3 m``.

    Built through the real :func:`ocean_skill.roms.add_depth_coord` and
    :func:`ocean_skill.comparison._prepare`, so its ``z_rho`` is exactly what a
    bare ``select={"lon": ..., "lat": ..., "time": ...}`` hands the plot -- top
    two levels above 0, the rest below. Values increase monotonically toward the
    surface, so a correctly drawn line never doubles back.
    """
    ds, meta = _raised_dataset()
    da, _ = _prepare(ds, meta, "alk", {"eta_rho": 0, "xi_rho": 0})
    if time is not None:
        da = da.assign_coords(time=np.datetime64(time))
    return da


@pytest.fixture
def raised_field(monkeypatch):
    """Build a bare-vertical ``Field`` at one point and one time over the column."""
    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    da = _raised_column(time="2010-10-31T23:45")
    assert (da["z_rho"] > 0).sum() >= 2  # the case under test: levels above MSL
    monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (da, None))
    return make_field("stub", ALKALINITY)


def _assert_right_side_up(depth, z_rho):
    """Assert ``depth`` is ``-z_rho`` exactly, and so monotonic up the column.

    Above-MSL levels sit above the 0 m line rather than mirrored beneath it.
    """
    depth = np.asarray(depth, dtype="float64")
    np.testing.assert_allclose(depth, -np.asarray(z_rho))
    assert (np.diff(depth) < 0).all()  # s_rho runs bottom -> top
    assert depth.min() < 0


def test_the_profile_family_is_what_this_shape_draws(raised_field):
    assert raised_field.family == "profile"


def test_matplotlib_profile_keeps_above_msl_levels_above_the_line_below(raised_field):
    fig = raised_field.plot(renderer="matplotlib")
    (line,) = fig.axes[0].get_lines()
    _assert_right_side_up(line.get_ydata(), raised_field.data["z_rho"])
    bottom, top = fig.axes[0].get_ylim()
    assert bottom > top  # still inverted: deep at the bottom
    assert top == pytest.approx(-float(raised_field.data["z_rho"].max()))


def test_holoviews_profile_keeps_above_msl_levels_above_the_line_below(raised_field):
    import holoviews as hv

    obj = raised_field.plot(renderer="holoviews")
    (curve,) = obj.traverse(lambda x: x, [hv.Curve])
    depth = curve.dimension_values(curve.vdims[0])
    _assert_right_side_up(depth, raised_field.data["z_rho"])


def test_time_depth_negates_z_rho_rather_than_mirroring_it():
    from ocean_skill.plot.time_depth import prepare_time_depth

    column = _raised_column()
    da = xr.concat(
        [column.assign_coords(time=t) for t in pd.date_range("2010-10-31", periods=3)],
        dim="time",
    )
    result, geometry = prepare_time_depth(da)
    _assert_right_side_up(result[geometry.y_name].values, column["z_rho"])


def test_depth_and_to_depth_z_coordinates_still_read_positive_down():
    """Only heights are negated: an obs ``depth`` and ``to_depth``'s ``z`` are not."""
    from ocean_skill.plot.profile import positive_down

    depths = np.array([0.0, 5.0, 50.0])
    obs = xr.DataArray(depths, dims="depth", name="depth")
    z = xr.DataArray(-depths, dims="z", name="z")
    np.testing.assert_allclose(positive_down(obs), depths)
    np.testing.assert_allclose(positive_down(z), depths)


def test_a_cf_positive_up_coordinate_is_read_as_a_height():
    from ocean_skill.plot.profile import positive_down

    height = xr.DataArray(
        np.array([0.5, -5.0]), dims="lev", name="lev", attrs={"positive": "up"}
    )
    np.testing.assert_allclose(positive_down(height), [-0.5, 5.0])


# -- a native level's height read back as a depth label or a pin -----------------------


@pytest.fixture
def stub_source(monkeypatch):
    """Return a setter that swaps ``comparison.prepare_source`` for one field."""
    from ocean_skill import comparison

    def use(da):
        monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (da, None))

    return use


def _make(**kwargs):
    from ocean_skill.field import field as make_field

    return make_field("stub", ALKALINITY, **kwargs)


def _raised_series() -> xr.DataArray:
    """Stand the raised column over three times: a 1-D ``z_rho`` on ``s_rho``."""
    column = _raised_column()
    return xr.concat(
        [column.assign_coords(time=t) for t in pd.date_range("2010-10-31", periods=3)],
        dim="time",
    )


def test_a_fanned_native_level_is_labelled_with_its_height_negated(stub_source):
    """A level 0.9 m above mean sea level is -0.9 m deep, not a mirrored +0.9 m."""
    da = _raised_series()
    stub_source(da)
    z_rho = da["z_rho"].values
    assert (z_rho > 0).sum() >= 2  # the case under test: levels above MSL

    items = _make()._series_items()

    depths = [item["aligned"].attrs["actual_depth"] for item in items]
    np.testing.assert_allclose(depths, -z_rho)
    assert min(depths) < 0
    assert (np.diff(depths) < 0).all()  # s_rho runs bottom -> top: depth shrinks


def test_top_level_of_a_column_is_the_highest_not_the_nearest_zero():
    """Read a 1-D ``z_rho`` by value: its top level is not the one nearest 0.

    Once the surface is raised the top level (+0.94 m) and the one nearest 0
    (+0.21 m) are different levels.
    """
    from ocean_skill.field import _top_level

    column = _raised_column()
    z_rho = column["z_rho"].values
    assert int(np.argmin(np.abs(z_rho))) != int(np.argmax(z_rho))  # discriminates

    top, label = _top_level(column, "s_rho", source="stub")

    assert label == "surface"
    assert float(top["z_rho"]) == pytest.approx(z_rho.max())
    assert top.attrs["actual_depth"] == pytest.approx(-z_rho.max())
    assert top.attrs["actual_depth"] < 0


def test_top_level_of_a_grid_records_the_mean_height_negated():
    """A 2-D-plus ``z_rho`` picks its index by argmax, and labels it the same way."""
    from ocean_skill.field import _top_level

    ds, meta = _raised_dataset(ncols=2)
    grid, _ = _prepare(ds, meta, "alk", {})
    assert grid["z_rho"].ndim == 3  # a grid, not a point column

    top, label = _top_level(grid, "s_rho", source="stub")

    top_z = grid["z_rho"].isel(s_rho=_NLEVELS - 1)
    assert label == "surface"
    assert (top_z > 0).all()
    assert top.attrs["actual_depth"] == pytest.approx(-float(top_z.mean()))
    assert top.attrs["actual_depth"] < 0


ZETAS = (0.2, ZETA, 0.4)  # the tide the extremum is found at is the middle one


@pytest.fixture
def tidal_grid(stub_source):
    """Two shallow columns over three tides, prepared whole, with a bare Field on it.

    The tracer peaks at the top level of the first column at the high tide -- a
    level whose ``z_rho`` is positive, i.e. above mean sea level.
    """
    ds, meta = _raised_dataset(ncols=2, zetas=ZETAS)
    grid, _ = _prepare(ds, meta, "alk", {})
    stub_source(grid)
    return ds, meta, grid


def test_extremum_above_msl_pins_its_series_by_level_not_by_depth(tidal_grid):
    ext = _make().extremum("max")
    assert ext.coords["z_rho"] > 0  # the extremum's level is above mean sea level
    assert ext.indices["s_rho"] == _NLEVELS - 1

    fs = ext.series(time="2010-10-31")

    select = fs[0].select
    assert select["s_rho"] == {"index": _NLEVELS - 1}
    assert "depth" not in select  # no depth names a level above the 0 m line


def test_the_pinned_series_reads_back_the_extremum_at_every_tide(tidal_grid):
    """A fixed depth would leave the column as the tide moved; the level does not."""
    ds, meta, grid = tidal_grid
    ext = _make().extremum("max")
    fs = ext.series(time="2010-10-31")

    series, _ = _prepare(ds, meta, "alk", fs[0].select)

    assert series.dims == ("time",)
    assert series.sizes["time"] == len(ZETAS)
    assert not np.isnan(series.values).any()
    assert float(series.max()) == pytest.approx(ext.value)
    assert int(np.argmax(series.values)) == ext.indices["time"]
    # It followed one s-level up and down with the tide, top of the column throughout.
    np.testing.assert_allclose(
        series["z_rho"].values, grid["z_rho"].isel(s_rho=-1, eta_rho=0, xi_rho=0)
    )


def test_a_positionally_narrowed_parent_axis_maps_back_to_the_source_level(stub_source):
    """The extremum's ``s_rho`` index counts along the parent's cut, not the source."""
    ds, meta = _raised_dataset(ncols=2, zetas=ZETAS)
    grid, _ = _prepare(ds, meta, "alk", {})
    stub_source(grid.isel(s_rho=slice(10, 20)))

    ext = _make(select={"s_rho": {"index": {"min": 10, "max": 20}}}).extremum("max")
    assert ext.indices["s_rho"] == 9  # the top of the cut...

    fs = ext.series(time="2010-10-31")

    assert fs[0].select["s_rho"] == {"index": 19}  # ...is the top of the column


@pytest.mark.parametrize(
    "narrowed, k, expected",
    [
        (None, 4, 4),
        ({"index": {"min": 10, "max": 20}}, 3, 13),
        ({"index": {"min": None, "max": 5}}, 2, 2),
        ({"index": slice(2, 20, 4)}, 3, 14),
        ({"index": [3, 5, 7]}, 2, 7),
        ([3, 5, 7], 1, 5),
    ],
)
def test_source_index_maps_a_cut_position_back(narrowed, k, expected):
    from ocean_skill.extrema import _source_index

    assert _source_index(narrowed, k, source="stub") == expected


@pytest.mark.parametrize(
    "narrowed",
    [
        {"index": {"min": -5, "max": None}},  # counts from an end we cannot see
        {"index": slice(None, None, -1)},
        {"min": 0, "max": 3},  # a coordinate range: no coordinate to read it from
    ],
)
def test_source_index_refuses_what_it_cannot_map_back(narrowed):
    from ocean_skill.extrema import _source_index

    with pytest.raises(ValueError, match="cannot be mapped back"):
        _source_index(narrowed, 1, source="stub")
