"""A native s-level column under a raised free surface draws right side up.

ROMS's ``z_rho`` is a *height* (zero at mean sea level, positive up) that carries
the free surface in it, so on a shallow shelf at high tide (``zeta > 0``) the top
levels are genuinely above 0. Reading it as ``abs()`` mirrored those levels back
under the ones beneath them: the profile's top drew as a hook, its true surface
plotted ~1 m *deeper* than the level below it. Both vertical-profile families that
read a native column -- ``profile`` (either renderer) and ``time_depth`` -- must
negate it instead, the same as :func:`ocean_skill.plot.section.prepare_section`.
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


def _raised_column(time=None) -> xr.DataArray:
    """One shallow ROMS column, prepared at its point, with ``zeta = +1.3 m``.

    Built through the real :func:`ocean_skill.roms.add_depth_coord` and
    :func:`ocean_skill.comparison._prepare`, so its ``z_rho`` is exactly what a
    bare ``select={"lon": ..., "lat": ..., "time": ...}`` hands the plot -- top
    two levels above 0, the rest below. Values increase monotonically toward the
    surface, so a correctly drawn line never doubles back.
    """
    from ocean_skill import roms

    n = 20
    sigma_r = (np.arange(1, n + 1) - n - 0.5) / n
    sigma_w = np.linspace(-1, 0, n + 1)
    ds = xr.Dataset(
        {
            "alk": (
                ("s_rho", "eta_rho", "xi_rho"),
                np.linspace(125.6, 126.1, n)[:, None, None] * np.ones((n, 1, 1)),
                {"units": "mmol/m^3"},
            ),
            "zeta": (("eta_rho", "xi_rho"), np.array([[ZETA]])),
        },
        coords={
            "h": (("eta_rho", "xi_rho"), np.array([[14.0]])),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((1, 1))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
            "lon": (("eta_rho", "xi_rho"), np.array([[97.6]])),
            "lat": (("eta_rho", "xi_rho"), np.array([[16.1]])),
        },
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": 250.0}}
    ds = roms.add_depth_coord(ds, meta)
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
