"""Vertical matching on a moving free surface: frame, edge policy, per-step levels.

Built on ``tests/_tidal_roms.py`` -- 3 x 3 columns, ``h = 20``, ten 2 m layers,
``zeta = 3 * [1, 0, -1, 0]`` over four hourly steps, ``level`` = the ``s_rho`` index and
``height`` = ``z_rho`` -- whose answers are sums you can do by hand (its docstring has
the worked numbers). What is under test is :mod:`ocean_skill.roms` alone: the model side
of "a depth is measured below the instantaneous surface" vs "a position fixed in space".
Reading a source's convention and routing it here is the comparison layer's business,
not this file's: ``convention`` is a plain mapping.

The point of the exercise, in numbers: a CTD cast 1 m below the surface is compared
with the model's cell nearest ``z = -1`` -- a different cell at every state of the
tide, and no water at all at low tide -- instead of with the one that is really 1 m
below the moving surface. See the module docstring of :mod:`ocean_skill.roms`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms
from tests._tidal_roms import tidal_roms

# The warnings the matcher owes its caller are asserted where they matter (through
# `_captured`); the many tests that only read values do not need them in the report.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*lie above the free surface:UserWarning",
    "ignore:the free surface moves:UserWarning",
    "ignore:.*entirely NaN:UserWarning",
)

SURFACE = {"origin": "surface", "source": "declared"}
FIXED = {"origin": "fixed", "source": "declared"}
# Fixed by default, nobody having said so: the only fixed origin that is advised about
# the samples above the free surface (a declared one is a decision, not a guess).
FIXED_UNDECLARED = {"origin": "fixed"}
NAN = np.nan
COUNTS = (
    "depth_edge_top",
    "depth_edge_bottom",
    "depth_above_surface",
    "depth_below_bottom",
)


def _captured(fn, *args, **kwargs):
    """Call ``fn`` and return ``(result, [UserWarning message, ...])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]


def _per_step(da, z=0):
    """Return the per-step values at depth index ``z`` (NaN-aware).

    The fixture is horizontally uniform, so all 9 columns must agree; that is checked
    here and one column's time series is returned.
    """
    vals = da.isel(z=z).transpose("time", ...).values.reshape(da.sizes["time"], -1)
    for column in vals.T:
        np.testing.assert_array_equal(column, vals[:, 0])
    return vals[:, 0]


def _at(da, **index):
    """Return ``da`` at one column, with the time axis first."""
    return da.isel(eta_rho=1, xi_rho=1, **index).transpose("time", ...)


def _meta_with(meta, **vertical):
    return {**meta, "vertical": {**meta["vertical"], **vertical}}


# -- the worked numbers --------------------------------------------------------------


def test_nearest_level_one_metre_down_follows_the_tide_when_fixed_in_space():
    ds, meta = tidal_roms()
    out, caught = _captured(
        roms.nearest_depth_levels, ds, meta, 1.0, convention=FIXED_UNDECLARED
    )
    # zeta = [3, 0, -3, 0]: the cell nearest z = -1 is a different one at high tide,
    # and at low tide the free surface (-3) is below the target -- no water there.
    np.testing.assert_array_equal(_per_step(out["level"]), [8, 9, NAN, 9])
    assert out.attrs["depth_above_surface"] == 9
    assert any("above the free surface" in m for m in caught)


def test_nearest_level_one_metre_below_the_surface_is_the_same_cell_all_tide():
    ds, meta = tidal_roms()
    out, caught = _captured(
        roms.nearest_depth_levels, ds, meta, 1.0, convention=SURFACE
    )
    np.testing.assert_array_equal(_per_step(out["level"]), [9, 9, 9, 9])
    assert out.attrs["depth_above_surface"] == 0
    assert caught == []


def test_interp_height_below_the_surface_follows_zeta():
    ds, meta = tidal_roms()
    out = roms.to_depth(ds, meta, 3.0, convention=SURFACE)
    # `height` is z_rho itself, so a linear match returns the target's own height:
    # zeta - 3 at every step. No half-cell, no edge: all of it interior.
    np.testing.assert_allclose(_per_step(out["height"]), [0.0, -3.0, -6.0, -3.0])
    assert out.attrs["depth_edge_top"] == 0
    assert out.attrs["depth_edge_bottom"] == 0


def test_interp_height_fixed_in_space_edge_fills_the_top_half_cell():
    ds, meta = tidal_roms()
    out, caught = _captured(roms.to_depth, ds, meta, 3.0, convention=FIXED)
    # at step 2 the free surface is at z = -3: the target is exactly on it, past the
    # top cell centre (-3.85) -- the top half-cell, filled with the top cell's value
    np.testing.assert_allclose(_per_step(out["height"]), [-3.0, -3.0, -3.85, -3.0])
    assert out.attrs["depth_edge_top"] == 9, "the 9 columns of step 2"
    assert out.attrs["depth_above_surface"] == 0
    assert caught == [], "an edge-fill is counted, never a warning"


def test_the_two_origins_differ_by_zeta_wherever_the_target_is_interior():
    ds, meta = tidal_roms()
    fixed = roms.to_depth(ds, meta, 3.0, convention=FIXED)["height"]
    below = roms.to_depth(ds, meta, 3.0, convention=SURFACE)["height"]
    gap = _per_step(below) - _per_step(fixed)
    np.testing.assert_allclose(gap[[0, 1, 3]], [3.0, 0.0, 0.0])
    assert gap[2] != pytest.approx(-3.0), "step 2 is an edge-fill, not a plain shift"


# -- per-step matching, and the old static lookup -----------------------------------


def test_ref_time_restores_the_static_lookup_that_per_step_replaced():
    ds, meta = tidal_roms()
    first = ds["time"].values[0]
    static = roms.nearest_depth_levels(ds, meta, 1.0, ref_time=first, convention=FIXED)
    # One level, chosen at step 0 (zeta = 3 -> level 8), reported at every step --
    # including step 2, where there is no water that deep: the old, wrong answer.
    np.testing.assert_array_equal(_per_step(static["level"]), [8, 8, 8, 8])
    live = roms.nearest_depth_levels(ds, meta, 1.0, convention=FIXED)
    np.testing.assert_array_equal(_per_step(live["level"]), [8, 9, NAN, 9])


def test_ref_time_picks_the_model_time_nearest_and_keeps_the_edge_policy():
    ds, meta = tidal_roms()
    low_tide = ds["time"].values[2]
    # At the reference instant (zeta = -3) the target is above the surface in every
    # column, so the static lookup says so at *every* step: NaN throughout.
    out, caught = _captured(
        roms.nearest_depth_levels, ds, meta, 1.0, ref_time=low_tide, convention=FIXED
    )
    assert np.isnan(out["level"].values).all()
    assert any("entirely NaN" in m for m in caught)
    # counts are over the reference frame's own samples: its 9 columns, no time axis
    assert out.attrs["depth_above_surface"] == 9
    # a time that is not on the grid snaps to the nearest model step
    near = ds["time"].values[0] + np.timedelta64(10, "m")
    snapped = roms.nearest_depth_levels(ds, meta, 1.0, ref_time=near, convention=FIXED)
    np.testing.assert_array_equal(_per_step(snapped["level"]), [8, 8, 8, 8])


# -- the frame -------------------------------------------------------------------------


def test_the_surface_frame_is_z_rho_minus_zeta():
    ds, meta = tidal_roms()
    frame = roms.frame_coordinate(ds, meta, "surface")
    np.testing.assert_array_equal(frame.values, (ds["z_rho"] - ds["zeta"]).values)
    # = (zeta + h) * sigma_r for this Vtransform-2 grid
    expected = (ds["zeta"] + ds["h"]) * ds["sigma_r"]
    np.testing.assert_allclose(frame.transpose(*expected.dims).values, expected.values)
    assert "z_rho" not in frame.coords, "the shifted frame is not z_rho any more"


def test_the_fixed_frame_is_z_itself_or_z_minus_the_datum():
    ds, meta = tidal_roms()
    plain = roms.frame_coordinate(ds, meta, "fixed")
    np.testing.assert_array_equal(plain.values, ds["z_rho"].values)
    shifted = roms.frame_coordinate(ds, meta, "fixed", datum_z_m=1.5)
    np.testing.assert_allclose(shifted.values, ds["z_rho"].values - 1.5)
    # a surface frame has no datum to offset from
    ignored = roms.frame_coordinate(ds, meta, "surface", datum_z_m=7.0)
    surface = roms.frame_coordinate(ds, meta, "surface")
    np.testing.assert_array_equal(ignored.values, surface.values)


def test_the_frame_attaches_the_depth_coordinate_it_needs():
    ds, meta = tidal_roms()
    bare = ds.drop_vars("z_rho")
    assert "z_rho" not in bare.coords
    frame = roms.frame_coordinate(bare, meta, "surface")
    expected = roms.frame_coordinate(ds, meta, "surface")
    np.testing.assert_allclose(frame.values, expected.values)
    interfaces = roms.frame_coordinate(bare, meta, "surface", z="z_w")
    assert interfaces.sizes["s_w"] == 11


def test_the_interface_frame_spans_exactly_the_water_column():
    ds, meta = tidal_roms()
    z_w = roms.frame_coordinate(ds, meta, "surface", z="z_w")
    top, bottom = roms.water_column_bounds(ds, meta, "surface")
    # the top interface is the surface (0), the bottom the seafloor (-(h + zeta))
    at_top, at_bottom = z_w.isel(s_w=-1), z_w.isel(s_w=0)
    np.testing.assert_allclose(
        at_top.values, top.transpose(*at_top.dims).values, atol=1e-12
    )
    np.testing.assert_allclose(
        at_bottom.values, bottom.transpose(*at_bottom.dims).values
    )


def test_an_unknown_origin_or_vertical_names_what_is_allowed():
    ds, meta = tidal_roms()
    with pytest.raises(ValueError, match=r"\('surface', 'fixed'\)"):
        roms.frame_coordinate(ds, meta, "seafloor")
    with pytest.raises(ValueError, match="origin"):
        roms.water_column_bounds(ds, meta, "mid-water")
    with pytest.raises(ValueError, match="z_rho"):
        roms.frame_coordinate(ds, meta, "fixed", z="z_u")
    with pytest.raises(ValueError, match="origin"):
        roms.to_depth(ds, meta, 3.0, convention={"origin": "seafloor"})
    with pytest.raises(TypeError, match="mapping"):
        roms.to_depth(ds, meta, 3.0, convention="surface")


def test_a_datum_has_to_be_a_finite_number():
    ds, meta = tidal_roms()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            roms.frame_coordinate(ds, meta, "fixed", datum_z_m=bad)
        with pytest.raises(ValueError, match="finite"):
            roms.water_column_bounds(ds, meta, "fixed", datum_z_m=bad)
        with pytest.raises(ValueError, match="finite"):
            roms.to_depth(ds, meta, 3.0, convention={"datum_z_m": bad})


def test_water_column_bounds_in_each_frame():
    ds, meta = tidal_roms()
    zeta = ds["zeta"]
    top, bottom = roms.water_column_bounds(ds, meta, "surface")
    np.testing.assert_array_equal(top.values, np.zeros(zeta.shape))
    np.testing.assert_allclose(bottom.values, -(20.0 + zeta.values))
    assert top.dims == bottom.dims == zeta.dims, "one dim order for the pair"
    top, bottom = roms.water_column_bounds(ds, meta, "fixed")
    np.testing.assert_array_equal(top.values, zeta.values)
    np.testing.assert_array_equal(bottom.values, np.full(bottom.shape, -20.0))
    top, bottom = roms.water_column_bounds(ds, meta, "fixed", datum_z_m=2.0)
    np.testing.assert_array_equal(top.values, zeta.values - 2.0)
    np.testing.assert_array_equal(bottom.values, np.full(bottom.shape, -22.0))


def test_the_frame_and_bounds_are_nan_over_land():
    ds, meta = tidal_roms(land=True)
    land = {"eta_rho": 0, "xi_rho": 0}
    ocean = {"eta_rho": 1, "xi_rho": 1}
    for origin in ("surface", "fixed"):
        top, bottom = roms.water_column_bounds(ds, meta, origin)
        assert np.isnan(top.isel(**land)).all()
        assert np.isfinite(top.isel(**ocean)).all()
        assert np.isfinite(bottom.isel(**ocean)).all()
        assert np.isnan(roms.frame_coordinate(ds, meta, origin).isel(**land)).all()
    # the surface frame's seafloor is -(h + zeta): as masked as zeta is
    assert np.isnan(roms.water_column_bounds(ds, meta, "surface")[1].isel(**land)).all()


def _stretched(ds, meta, *, vtransform, hc):
    """Return the tidal dataset with a stretched grid, a given ``hc`` and Vtransform."""

    def stretch(s):  # -1 -> -1, 0 -> 0, monotone: a valid Cs
        return np.sinh(3.0 * s) / np.sinh(3.0)

    meta2 = _meta_with(meta, hc=hc, Vtransform=vtransform)
    ds2 = ds.assign_coords(Cs_r=("s_rho", stretch(ds["sigma_r"].values)))
    ds2 = ds2.assign(Cs_w=("s_w", stretch(ds["sigma_w"].values)))
    return roms.add_depth_coord(ds2, meta2), meta2


@pytest.mark.parametrize("vtransform", [1, 2])
def test_with_hc_and_stretching_the_interfaces_still_span_the_water_column(vtransform):
    """Valid for Vtransform 1 *and* 2, with a real ``hc`` and a stretched grid.

    ``z - zeta`` must run from ``-(h + zeta)`` at the seafloor interface to 0 at the
    surface one whatever the transform -- the invariant the edge policy's top/bottom
    bounds rest on.
    """
    ds, meta = tidal_roms()
    ds2, meta2 = _stretched(ds, meta, vtransform=vtransform, hc=5.0)
    z_w = roms.frame_coordinate(ds2, meta2, "surface", z="z_w")
    top, bottom = roms.water_column_bounds(ds2, meta2, "surface")
    np.testing.assert_allclose(z_w.isel(s_w=-1).values, top.values, atol=1e-12)
    np.testing.assert_allclose(z_w.isel(s_w=0).values, bottom.values, atol=1e-12)
    # and the centres lie strictly inside it, bottom -> top
    centres = roms.frame_coordinate(ds2, meta2, "surface")
    assert (centres.diff("s_rho") > 0).all()
    assert (centres.isel(s_rho=-1) < top).all()
    assert (centres.isel(s_rho=0) > bottom).all()


@pytest.mark.parametrize("vtransform", [1, 2])
@pytest.mark.parametrize(
    ("convention", "reference"),
    [
        ({"origin": "surface", "source": "declared"}, "zeta"),
        ({"origin": "fixed", "datum_z_m": 1.0, "source": "declared"}, "datum"),
    ],
)
def test_interpolating_z_rho_has_a_closed_form_on_a_stretched_grid(
    vtransform, convention, reference
):
    """``height`` *is* ``z_rho``, so linear matching returns ``target + reference``.

    ``-d + zeta`` below the surface, ``-d + datum`` fixed in space, wherever the target
    is interior -- on a stretched grid with ``hc != 0``, for both Vtransforms.
    """
    ds, meta = tidal_roms()
    ds2, meta2 = _stretched(ds, meta, vtransform=vtransform, hc=5.0)
    ds2 = ds2.assign(height=ds2["z_rho"].transpose("time", "s_rho", ...))
    depths = [6.0, 9.0]  # interior at every step, for both references
    out = roms.to_depth(ds2, meta2, depths, convention=convention)
    zeta = ds2["zeta"].isel(eta_rho=1, xi_rho=1).values
    shift = zeta if reference == "zeta" else np.full(4, 1.0)
    for j, depth in enumerate(depths):
        got = _at(out["height"], z=j).values
        np.testing.assert_allclose(got, -depth + shift, atol=1e-9)


def test_vertical_transform_1_gives_the_same_numbers_as_2_on_this_grid():
    """With hc = 0 and Cs = sigma the two transforms coincide.

    V1: ``z0 = Cs*h``, ``z = z0 + zeta*(1 + z0/h) = zeta + (zeta + h)*sigma`` -- exactly
    the V2 expression, so every worked number must come out the same.
    """
    ds, meta = tidal_roms()
    meta1 = _meta_with(meta, Vtransform=1)
    ds1 = roms.add_depth_coord(ds, meta1)
    z1 = ds1["z_rho"]
    np.testing.assert_allclose(z1.values, ds["z_rho"].values, atol=1e-12)
    height = ds["height"].transpose(*z1.dims).values
    np.testing.assert_allclose(z1.values, height, atol=1e-12)
    f1 = roms.frame_coordinate(ds1, meta1, "surface")
    f2 = roms.frame_coordinate(ds, meta, "surface")
    np.testing.assert_allclose(f1.values, f2.values, atol=1e-12)
    for convention, levels in ((SURFACE, [9, 9, 9, 9]), (FIXED, [8, 9, NAN, 9])):
        out = roms.nearest_depth_levels(ds1, meta1, 1.0, convention=convention)
        np.testing.assert_array_equal(_per_step(out["level"]), levels)
    out = roms.to_depth(ds1, meta1, 3.0, convention=SURFACE)
    expected = [0.0, -3.0, -6.0, -3.0]
    np.testing.assert_allclose(_per_step(out["height"]), expected, atol=1e-12)


def test_a_fixed_datum_shifts_the_target_and_is_recorded_on_z():
    ds, meta = tidal_roms()
    conv = {"origin": "fixed", "datum_z_m": 2.0, "source": "declared"}
    out = roms.to_depth(ds, meta, 6.0, convention=conv)
    # z = datum - d = -4: `height` is z_rho, so interior samples return exactly -4
    np.testing.assert_allclose(_per_step(out["height"])[[0, 1, 3]], -4.0)
    assert out["z"].attrs["depth_datum_z_m"] == 2.0
    assert out["z"].attrs["depth_origin"] == "fixed"
    # no datum is recorded when there is none, or for the surface origin
    for convention in (FIXED, SURFACE):
        plain = roms.to_depth(ds, meta, 6.0, convention=convention)
        assert "depth_datum_z_m" not in plain["z"].attrs


# -- the edge policy and its counts ---------------------------------------------------


def test_counts_a_target_above_the_surface_at_low_tide():
    ds, meta = tidal_roms()
    out = roms.to_depth(ds, meta, 1.0, convention=FIXED)
    assert out.attrs["depth_above_surface"] == 9, "step 2: z = -1 is above zeta = -3"
    values = _per_step(out["height"])
    np.testing.assert_allclose(values[[0, 1, 3]], -1.0)
    assert np.isnan(values[2])


def test_counts_a_target_past_the_seafloor_as_below_the_bottom():
    ds, meta = tidal_roms()
    out, caught = _captured(roms.to_depth, ds, meta, 25.0, convention=SURFACE)
    # 25 m below a surface over 20 m of water (+/- 3 m of tide): below the seafloor at
    # every step, so every sample of the one target is NaN -- and the target is
    # entirely NaN, which says so
    assert out.attrs["depth_below_bottom"] == 4 * 9
    assert out.attrs["depth_above_surface"] == 0
    assert np.isnan(out["height"].values).all()
    assert any("entirely NaN" in m for m in caught)


def test_below_the_seafloor_only_at_some_steps_counts_just_those():
    ds, meta = tidal_roms()
    # 22 m below the surface: the column is h + zeta = 23, 20, 17, 20 m deep
    out, caught = _captured(roms.to_depth, ds, meta, 22.0, convention=SURFACE)
    assert out.attrs["depth_below_bottom"] == 3 * 9
    values = _per_step(out["height"])
    assert np.isfinite(values[0]) and np.isnan(values[1:]).all()
    assert caught == [], "reachable at step 0: not entirely NaN"


def test_counts_a_target_in_the_bottom_half_cell_and_fills_it():
    ds, meta = tidal_roms()
    # Surface frame, zeta = 0 (steps 1 and 3): centres at -19 .. -1, seafloor -20, so
    # 19.5 m is in the bottom half-cell and the bottom cell (k = 0) is its value. At
    # high tide (23 m of water, bottom centre at 21.85 m) it is interior; at low tide
    # (17 m of water) it is below the seafloor.
    out = roms.nearest_depth_levels(ds, meta, 19.5, convention=SURFACE)
    values = _per_step(out["level"])
    assert values[1] == 0.0 and values[3] == 0.0
    assert out.attrs["depth_edge_bottom"] == 2 * 9
    assert out.attrs["depth_below_bottom"] == 9
    assert np.isnan(values[2])
    interp = roms.to_depth(ds, meta, 19.5, convention=SURFACE)
    assert _per_step(interp["level"])[1] == 0.0, "interp fills with the bottom cell"


def test_surface_origin_d_zero_edge_fills_the_top_cell_without_a_warning():
    ds, meta = tidal_roms()
    for fn in (roms.to_depth, roms.nearest_depth_levels):
        out, caught = _captured(fn, ds, meta, 0.0, convention=SURFACE)
        np.testing.assert_array_equal(_per_step(out["level"]), [9, 9, 9, 9])
        assert np.isfinite(out["height"].values).all()
        assert out.attrs["depth_edge_top"] == 4 * 9, "every sample is a top half-cell"
        assert caught == []


def test_a_target_exactly_on_a_cell_centre_is_interior_not_an_edge():
    ds, meta = tidal_roms()
    # step 1 (zeta = 0): the top centre is at -1.0 and a 1 m target sits on it
    top_centre = roms.to_depth(ds.isel(time=[1]), meta, 1.0, convention=SURFACE)
    assert top_centre.attrs["depth_edge_top"] == 0
    np.testing.assert_allclose(top_centre["level"].values, 9.0)
    bottom_centre = roms.to_depth(ds.isel(time=[1]), meta, 19.0, convention=SURFACE)
    assert bottom_centre.attrs["depth_edge_bottom"] == 0
    np.testing.assert_allclose(bottom_centre["level"].values, 0.0)


def test_counts_are_totals_over_every_sample_and_target():
    ds, meta = tidal_roms()
    # fixed, d = [1, 3, 25]: 1 -> above the surface at step 2 (9 columns); 3 -> top
    # half-cell at step 2 (9); 25 -> below the seafloor at all four steps (36)
    out = roms.to_depth(ds, meta, [1.0, 3.0, 25.0], convention=FIXED)
    assert out.attrs["depth_above_surface"] == 9
    assert out.attrs["depth_edge_top"] == 9
    assert out.attrs["depth_below_bottom"] == 36
    assert out.attrs["depth_edge_bottom"] == 0


def test_counts_and_origin_ride_on_the_dataset_and_every_variable_as_ints():
    ds, meta = tidal_roms()
    out = roms.to_depth(ds, meta, [1.0, 3.0], convention=FIXED)
    for holder in (out, out["level"], out["height"]):
        for key in COUNTS:
            assert type(holder.attrs[key]) is int
        assert holder.attrs["depth_origin"] == "fixed"
    assert out["level"].attrs["units"] == "1", "the source variable's attrs are kept"
    assert out["height"].attrs["units"] == "m"
    assert out["level"].attrs["depth_above_surface"] == out.attrs["depth_above_surface"]
    assert out.attrs["featureType"] == "grid", "the dataset's own attrs are kept"


def test_the_z_coordinate_is_negative_up_and_says_what_it_is():
    ds, meta = tidal_roms()
    out = roms.to_depth(ds, meta, [1.0, 3.0], convention=SURFACE)
    np.testing.assert_array_equal(out["z"].values, [-1.0, -3.0])
    assert out["z"].attrs == {"positive": "up", "units": "m", "depth_origin": "surface"}
    assert out["height"].dims[-1] == "z"
    assert {"lon", "lat", "time"} <= set(out.coords)


def test_counts_come_from_the_frame_not_the_data():
    ds, meta = tidal_roms()
    blank = ds.assign(level=ds["level"] * np.nan)
    a, _ = _captured(roms.nearest_depth_levels, ds, meta, 1.0, convention=FIXED)
    b, caught = _captured(roms.nearest_depth_levels, blank, meta, 1.0, convention=FIXED)
    for key in COUNTS:
        assert a.attrs[key] == b.attrs[key]
    assert np.isnan(b["level"].values).all()
    # reachability is geometry: NaN data does not make a reachable target "unreachable"
    assert not any("entirely NaN" in m for m in caught)


def test_a_land_column_is_nan_and_counted_as_neither_above_nor_below():
    ds, meta = tidal_roms(land=True)
    out = roms.nearest_depth_levels(ds, meta, 1.0, convention=FIXED)
    land = out["level"].isel(eta_rho=0, xi_rho=0, z=0).values
    assert np.isnan(land).all(), "masked in every step"
    # 8 ocean columns above the surface at step 2, not 9: the land column is neither
    assert out.attrs["depth_above_surface"] == 8
    assert out.attrs["depth_below_bottom"] == 0
    ocean = out["level"].isel(eta_rho=1, xi_rho=1, z=0).values
    np.testing.assert_array_equal(ocean, [8, 9, NAN, 9])
    # a target that is in the water for every ocean column is NaN over land and
    # counted as nothing at all
    deep = roms.to_depth(ds, meta, 3.0, convention=SURFACE)
    assert np.isnan(deep["height"].isel(eta_rho=0, xi_rho=0).values).all()
    assert deep.attrs["depth_below_bottom"] == 0
    assert deep.attrs["depth_above_surface"] == 0


def test_a_riding_spread_follows_the_same_per_step_levels_as_the_data():
    from ocean_skill.operators import SPREAD_COORD

    ds, meta = tidal_roms()
    ds = ds.assign_coords({SPREAD_COORD: 3.0 + 0.5 * ds["level"]})
    for fn in (roms.nearest_depth_levels, roms.to_depth):
        out = fn(ds, meta, [1.0, 3.0], convention=FIXED)
        assert SPREAD_COORD in out.coords
        assert SPREAD_COORD in out["level"].coords, "rides with the variable"
        assert "s_rho" not in out[SPREAD_COORD].dims
        expected = 3.0 + 0.5 * out["level"]
        np.testing.assert_allclose(
            out[SPREAD_COORD].transpose(*expected.dims).values,
            expected.values,
            equal_nan=True,
        )


def test_the_cell_area_coordinate_rides_along():
    ds, meta = tidal_roms()
    area = xr.DataArray(np.full((3, 3), 4.0), dims=("eta_rho", "xi_rho"))
    ds = ds.assign_coords({roms.AREA_COORD: area})
    out = roms.to_depth(ds, meta, 3.0, convention=SURFACE)
    assert roms.AREA_COORD in out.coords
    np.testing.assert_array_equal(out[roms.AREA_COORD].values, 4.0)


# -- the kernel ------------------------------------------------------------------------


def _column(values):
    return np.array([values], dtype=float)


def test_a_tie_between_two_levels_goes_to_the_deeper_one():
    frame = _column([-5.0, -3.0, -1.0])  # one column, ascending
    values = _column([10.0, 20.0, 30.0])
    top, bottom = np.array([0.0]), np.array([-6.0])
    targets = np.array([-2.0, -4.0])
    kwargs = {"targets": targets, "dtype": float}
    nearest = roms._match_columns(values, frame, top, bottom, mode="nearest", **kwargs)
    # -2 is midway between -3 and -1, -4 between -5 and -3: the deeper (lower) wins
    np.testing.assert_array_equal(nearest, [[20.0, 10.0]])
    linear = roms._match_columns(values, frame, top, bottom, mode="interp", **kwargs)
    np.testing.assert_allclose(linear, [[25.0, 15.0]])


def test_an_exact_centre_hit_returns_that_centre_even_beside_a_nan():
    frame = _column([-5.0, -3.0, -1.0])
    values = _column([10.0, np.nan, 30.0])
    got = roms._match_columns(
        values,
        frame,
        np.array([0.0]),
        np.array([-6.0]),
        targets=np.array([-5.0, -1.0, -4.0]),
        mode="interp",
        dtype=float,
    )
    # exact on a centre: its own value, whatever its neighbour is; between centres a
    # NaN neighbour makes it NaN, honestly
    np.testing.assert_array_equal(got, [[10.0, 30.0, NAN]])


def test_the_kernel_broadcasts_a_static_frame_under_a_time_series():
    frame = np.array([[[-5.0, -3.0, -1.0]]])  # (time-less, 1 column, 3 levels)
    values = np.arange(6.0).reshape(2, 1, 3)  # (2 times, 1 column, 3 levels)
    got = roms._match_columns(
        values,
        frame,
        np.zeros((1, 1)),
        np.full((1, 1), -6.0),
        targets=np.array([-3.0]),
        mode="nearest",
        dtype=float,
    )
    assert got.shape == (2, 1, 1)
    np.testing.assert_array_equal(got[:, 0, 0], [1.0, 4.0])


def _reference_match(values, frame, top, bottom, t, mode):
    """Match one column to one target the slow, obvious way: the contract's table."""
    if np.isnan(frame).all():
        return NAN, roms._MASKED
    if t > top:
        return NAN, roms._ABOVE_SURFACE
    if t < bottom:
        return NAN, roms._BELOW_BOTTOM
    if t < frame[0]:
        return values[0], roms._BOTTOM_CELL
    if t > frame[-1]:
        return values[-1], roms._TOP_CELL
    k = int(np.searchsorted(frame, t, side="right")) - 1  # the last centre <= t
    if k == len(frame) - 1:  # exactly on the top centre
        return values[k], roms._INTERIOR
    lo, hi = k, k + 1
    if mode == "nearest":
        closer_above = abs(frame[hi] - t) < abs(frame[lo] - t)  # a tie is the deeper
        return values[hi if closer_above else lo], roms._INTERIOR
    weight = (t - frame[lo]) / (frame[hi] - frame[lo])
    return values[lo] + weight * (values[hi] - values[lo]), roms._INTERIOR


def _random_columns(rng, shape, n=7):
    """Random frames, bounds and values for ``shape`` columns, with some land.

    The frame ascends bottom -> top; a quarter of its steps are exactly 2.0 so that
    midpoints land on whole numbers; the water column extends up to 2 above the top
    centre and down to 2 below the bottom one; one column in the first slice is land.
    """
    steps = rng.uniform(0.5, 3.0, (*shape, n))
    steps[rng.random(steps.shape) < 0.25] = 2.0
    frame = -30.0 + np.cumsum(steps, axis=-1)
    values = rng.normal(size=(*shape, n))
    top = frame[..., -1] + rng.uniform(0.0, 2.0, shape)
    bottom = frame[..., 0] - rng.uniform(0.0, 2.0, shape)
    land = (0,) * (len(shape) - 2) + (0, 1)
    frame[land] = np.nan
    top[land] = bottom[land] = np.nan
    return values, frame, top, bottom


@pytest.mark.parametrize("layout", ["same", "static_frame", "static_values"])
@pytest.mark.parametrize("mode", ["interp", "nearest"])
@pytest.mark.parametrize("seed", range(3))
def test_the_kernel_matches_a_slow_reference_over_random_columns(layout, mode, seed):
    """Random frames, bounds, land columns and targets -- including exact hits and ties.

    The reference is the contract's edge-policy table written column by column; the
    kernel must agree on every value and on every code (`_classify_columns` is the
    kernel's own code path, which the counts and the warnings are built from). Run
    with a leading axis on both, only the data, or only the frame: the broadcasting a
    static frame under a time series (or the reverse) relies on.
    """
    rng = np.random.default_rng(seed)
    steps, ny, nx = 3, 3, 4
    values, frame, top, bottom = _random_columns(rng, (steps, ny, nx))
    centres = frame[~np.isnan(frame)]
    midpoints = 0.5 * (frame[..., 1:] + frame[..., :-1])
    targets = np.concatenate(
        [
            rng.uniform(-36.0, 0.0, 12),  # anywhere, in or out of the water
            rng.choice(centres, 6),  # exact hits
            rng.choice(midpoints[~np.isnan(midpoints)], 6),  # ties
            [-100.0, 100.0],  # far beyond either bound
        ]
    )
    # what the kernel is handed: a size-1 leading axis stands for "the same at every
    # step" (what apply_ufunc gives a field that lacks the axis)
    first = slice(0, 1)
    k_values = values[first] if layout == "static_values" else values
    k_frame, k_top, k_bottom = (
        a[first] if layout == "static_frame" else a for a in (frame, top, bottom)
    )
    kernel = {"targets": targets, "dtype": float}
    got = roms._match_columns(k_values, k_frame, k_top, k_bottom, mode=mode, **kernel)
    codes = roms._classify_columns(k_frame, k_top, k_bottom, targets=targets)
    assert got.shape == (steps, ny, nx, len(targets))
    for s in range(steps):
        at_values = values[0 if layout == "static_values" else s]
        f, tp, bt = (
            a[0 if layout == "static_frame" else s] for a in (frame, top, bottom)
        )
        for i in range(ny):
            for j in range(nx):
                for k, t in enumerate(targets):
                    want, code = _reference_match(
                        at_values[i, j], f[i, j], tp[i, j], bt[i, j], t, mode
                    )
                    where = (s, i, j, k, mode, t)
                    c = codes[0 if layout == "static_frame" else s, i, j, k]
                    assert c == code, where
                    if np.isnan(want):
                        assert np.isnan(got[s, i, j, k]), where
                    else:
                        assert got[s, i, j, k] == pytest.approx(want, abs=1e-12), where


def test_every_target_is_classified_the_way_the_data_path_treats_it():
    frame = np.array([[-5.0, -3.0, -1.0], [np.nan] * 3])
    top = np.array([0.0, np.nan])
    bottom = np.array([-6.0, np.nan])
    codes = roms._classify_columns(
        frame, top, bottom, targets=np.array([-3.0, -0.5, -5.5, 0.5, -7.0])
    )
    expected = [
        roms._INTERIOR,
        roms._TOP_CELL,
        roms._BOTTOM_CELL,
        roms._ABOVE_SURFACE,
        roms._BELOW_BOTTOM,
    ]
    np.testing.assert_array_equal(codes[0], expected)
    assert (codes[1] == roms._MASKED).all()


# -- warnings ------------------------------------------------------------------------


def test_a_target_above_the_surface_at_every_step_warns_entirely_nan():
    ds, meta = tidal_roms(zeta=(-3.0, -3.0, -3.0, -3.0))
    out, caught = _captured(roms.to_depth, ds, meta, 1.0, convention=FIXED)
    assert np.isnan(out["height"].values).all()
    nan_warnings = [m for m in caught if "entirely NaN" in m]
    assert len(nan_warnings) == 1
    assert "target depth 1 m is" in nan_warnings[0]
    assert "use surface()" in nan_warnings[0], "a shallow target points at surface()"


def test_a_span_of_unreachable_targets_is_one_warning_not_one_each():
    ds, meta = tidal_roms()
    targets = [30.0, 40.0, 50.0]
    _, caught = _captured(
        roms.nearest_depth_levels, ds, meta, targets, convention=FIXED
    )
    nan_warnings = [m for m in caught if "entirely NaN" in m]
    assert len(nan_warnings) == 1
    assert "3 target depths (30-50 m) are entirely NaN" in nan_warnings[0]
    assert "use surface()" not in nan_warnings[0], "deep targets: no surface hint"


def test_fixed_origin_partial_above_surface_warns_and_names_the_fix():
    ds, meta = tidal_roms()
    _, caught = _captured(
        roms.nearest_depth_levels, ds, meta, 1.0, convention=FIXED_UNDECLARED
    )
    above = [m for m in caught if "above the free surface" in m]
    assert len(above) == 1
    assert "9 of 36" in above[0]
    assert "origin: surface" in above[0]
    assert 'depth_origin="surface"' in above[0]
    assert not any("entirely NaN" in m for m in caught), "reachable at 3 of 4 steps"


def test_a_surface_origin_never_warns_about_the_surface_or_the_tide():
    ds, meta = tidal_roms()
    for depths in ([0.0, 1.0, 3.0, 19.5], [1.0]):
        for fn in (roms.to_depth, roms.nearest_depth_levels):
            _, caught = _captured(fn, ds, meta, depths, convention=SURFACE)
            assert caught == []
    # the tide warning is about an *undeclared* fixed origin only
    _, caught = _captured(
        roms.to_depth, ds, meta, 1.0, convention={"origin": "surface"}
    )
    assert caught == []


def test_default_fixed_over_a_big_tide_warns_unless_the_origin_was_declared():
    ds, meta = tidal_roms()
    # 4 m is in the water at every step, so only the tide warning can fire
    defaults = (None, {}, {"origin": "fixed"}, {"origin": "fixed", "source": "default"})
    for convention in defaults:
        _, caught = _captured(roms.to_depth, ds, meta, 4.0, convention=convention)
        tide = [m for m in caught if "free surface moves" in m]
        assert len(tide) == 1, convention
        assert "6 m" in tide[0] and "declares no `depth_convention`" in tide[0]
        assert "fixed in space" in tide[0]
    for source in ("declared", "inferred", "data"):
        convention = {"origin": "fixed", "source": source}
        _, caught = _captured(roms.to_depth, ds, meta, 4.0, convention=convention)
        assert not any("free surface moves" in m for m in caught), source


def test_the_positional_call_shape_is_the_default_fixed_convention():
    ds, meta = tidal_roms()
    plain, caught = _captured(roms.nearest_depth_levels, ds, meta, 1.0)
    explicit = roms.nearest_depth_levels(ds, meta, 1.0, convention=FIXED)
    xr.testing.assert_identical(plain, explicit)  # attrs and all
    assert any("free surface moves" in m for m in caught)


def test_the_tide_is_each_columns_range_over_time_not_a_spatial_difference():
    """A full-domain lane's spatial sea-level differences are not "the tide"."""
    ds, meta = tidal_roms(zeta=(0.0, 0.0, 0.0, 0.0))
    # 0..8 m across the grid, identical at every step
    tilt = xr.DataArray(np.arange(9.0).reshape(3, 3), dims=("eta_rho", "xi_rho"))
    ds = roms.add_depth_coord(ds.assign(zeta=ds["zeta"] + tilt), meta)
    _, caught = _captured(roms.to_depth, ds, meta, 4.0, convention=None)
    assert not any("free surface moves" in m for m in caught)

    # ... whereas one column that does move over time warns, with its own range
    moving = ds["zeta"].copy()
    moving[{"time": 2, "eta_rho": 1, "xi_rho": 1}] += 5.0
    ds = roms.add_depth_coord(ds.assign(zeta=moving), meta)
    _, caught = _captured(roms.to_depth, ds, meta, 4.0, convention=None)
    tide = [m for m in caught if "free surface moves" in m]
    assert len(tide) == 1 and "5 m" in tide[0]


def test_a_free_surface_with_no_time_axis_does_not_move():
    ds, meta = tidal_roms()
    frozen = ds.isel(time=0, drop=True)  # zeta is (eta, xi) now, however large
    frozen = roms.add_depth_coord(frozen, meta)
    out, caught = _captured(roms.to_depth, frozen, meta, 4.0, convention=None)
    assert not any("free surface moves" in m for m in caught)
    assert "time" not in out["height"].dims
    flat = ds.drop_vars("zeta")  # no zeta at all: a flat surface
    _, caught = _captured(roms.to_depth, flat, meta, 4.0, convention=None)
    assert not any("free surface moves" in m for m in caught)


def test_the_tide_threshold_scales_with_the_shallowest_target():
    # a 6 m range: 0.1 * min(d) = 10 m for d = 100, so it is small beside the depth
    ds, meta = tidal_roms(h=500.0)
    _, caught = _captured(roms.to_depth, ds, meta, [100.0, 200.0], convention=None)
    assert not any("free surface moves" in m for m in caught)
    _, caught = _captured(roms.to_depth, ds, meta, [1.0, 200.0], convention=None)
    assert any("free surface moves" in m for m in caught)
    _, caught = _captured(roms.to_depth, ds, meta, [100.0, 1.0], convention=None)
    assert any("free surface moves" in m and "0.5 m" in m for m in caught), "min(d)"
    # a range at or below the 0.5 m floor never warns
    small, small_meta = tidal_roms(zeta=(0.2, 0.0, -0.2, 0.0))
    _, caught = _captured(roms.to_depth, small, small_meta, 4.0, convention=None)
    assert not any("free surface moves" in m for m in caught)


def test_edge_fills_are_counted_never_warned():
    ds, meta = tidal_roms()
    depths = [0.0, 0.5, 19.5]
    out, caught = _captured(roms.to_depth, ds, meta, depths, convention=SURFACE)
    assert caught == []
    assert out.attrs["depth_edge_top"] > 0
    assert out.attrs["depth_edge_bottom"] > 0


# -- lazy, named, broadcast -----------------------------------------------------------


@pytest.mark.parametrize("fn", [roms.to_depth, roms.nearest_depth_levels])
@pytest.mark.parametrize("convention", [SURFACE, FIXED])
def test_a_chunked_lane_stays_lazy_and_computes_to_the_eager_answer(fn, convention):
    pytest.importorskip("dask")
    ds, meta = tidal_roms()
    chunked, _ = tidal_roms(chunks={"time": 1})
    assert chunked["level"].chunks is not None
    eager = fn(ds, meta, [1.0, 3.0, 19.5], convention=convention)
    lazy = fn(chunked, meta, [1.0, 3.0, 19.5], convention=convention)
    for name in ("level", "height"):
        assert lazy[name].chunks is not None, f"{name} was computed eagerly"
        np.testing.assert_array_equal(lazy[name].compute().values, eager[name].values)
    # the counts are eager (they need the frame), and the same
    for key in COUNTS:
        assert lazy.attrs[key] == eager.attrs[key]


def test_a_source_chunked_in_the_vertical_still_matches():
    pytest.importorskip("dask")
    ds, meta = tidal_roms()
    chunked = ds.chunk({"s_rho": 3, "time": 2})
    level = chunked["level"]
    assert len(level.chunks[level.dims.index("s_rho")]) > 1
    out = roms.nearest_depth_levels(chunked, meta, 1.0, convention=FIXED)
    np.testing.assert_array_equal(_per_step(out["level"].compute()), [8, 9, NAN, 9])


def test_the_free_surface_may_be_called_sea_surface_height_above_geoid():
    ds, meta = tidal_roms(zeta_name="sea_surface_height_above_geoid")
    assert "zeta" not in ds.variables
    assert "sea_surface_height_above_geoid" in ds.variables
    out = roms.nearest_depth_levels(ds, meta, 1.0, convention=FIXED)
    np.testing.assert_array_equal(_per_step(out["level"]), [8, 9, NAN, 9])
    out = roms.to_depth(ds, meta, 3.0, convention=SURFACE)
    np.testing.assert_allclose(_per_step(out["height"]), [0.0, -3.0, -6.0, -3.0])
    # and it is the same surface the depth coordinate was built from
    reference = roms.frame_coordinate(tidal_roms()[0], meta, "surface")
    got = roms.frame_coordinate(ds, meta, "surface")
    np.testing.assert_array_equal(got.values, reference.values)


def test_a_field_with_no_time_axis_broadcasts_against_the_moving_frame():
    ds, meta = tidal_roms()
    static = ds["level"].isel(time=0, drop=True)  # (s_rho, eta, xi): same at all times
    out = roms.nearest_depth_levels(
        ds.assign(static_level=static), meta, 1.0, convention=FIXED
    )
    assert out["static_level"].dims == ("time", "eta_rho", "xi_rho", "z")
    np.testing.assert_array_equal(_per_step(out["static_level"]), [8, 9, NAN, 9])


def test_a_float32_field_stays_float32_when_snapped_and_widens_when_blended():
    ds, meta = tidal_roms()
    f32 = ds.assign(level=ds["level"].astype("float32"))
    snapped = roms.nearest_depth_levels(f32, meta, 3.0, convention=FIXED)
    assert snapped["level"].dtype == np.float32
    blended = roms.to_depth(f32, meta, 3.0, convention=FIXED)
    assert blended["level"].dtype == np.float64


def test_a_single_column_with_no_time_axis_matches_too():
    ds, meta = tidal_roms()
    # one column (the grid's h/lon/lat stay as scalar coordinates), no time axis
    column = ds.isel(eta_rho=1, xi_rho=1, time=0).drop_vars("time")
    out = roms.nearest_depth_levels(column, meta, 1.0, convention=SURFACE)
    assert float(out["level"].isel(z=0)) == 9.0


def test_non_vertical_variables_drop_and_the_grid_coordinates_stay():
    ds, meta = tidal_roms()
    out = roms.to_depth(ds, meta, 3.0, convention=FIXED)
    assert set(out.data_vars) == {"level", "height"}, "zeta, lon_rho, ocean_time drop"
    assert {"lon", "lat", "z", "time"} <= set(out.coords)
    assert "s_rho" not in out.dims and "z_rho" not in out.coords


def test_the_time_coordinates_own_attrs_survive():
    ds, meta = tidal_roms()
    ds["time"].attrs["source_time_zone"] = "America/Anchorage"
    for fn in (roms.to_depth, roms.nearest_depth_levels):
        out = fn(ds, meta, 3.0, convention=SURFACE)
        assert out["time"].attrs["source_time_zone"] == "America/Anchorage"


# -- the band, in each frame -----------------------------------------------------------


def test_a_depth_band_below_the_surface_always_spans_the_top_of_the_water():
    ds, meta = tidal_roms()
    band = roms.depth_band(ds, meta, 0.0, 5.0, convention=SURFACE)
    # 5 m of weight per column at *every* step: the band rides the tide
    total = band[roms.WEIGHT_COORD].sum("s_rho").transpose("time", ...)
    np.testing.assert_allclose(total.values, 5.0)
    assert band.sizes["s_rho"] == 3, "the top three cells"


def test_a_fixed_depth_band_loses_water_as_the_surface_drops_beneath_it():
    ds, meta = tidal_roms()
    band = roms.depth_band(ds, meta, 0.0, 5.0, convention=FIXED)
    total = _at(band[roms.WEIGHT_COORD].sum("s_rho"))
    # 0-5 m below mean sea level: all in the water at high tide and at the two mid
    # tides, but at low tide (surface at -3 m) only the 2 m between -3 and -5 remain
    np.testing.assert_allclose(total.values, [5.0, 5.0, 2.0, 5.0])


def test_the_two_frames_weight_a_band_differently_at_high_tide():
    ds, meta = tidal_roms()
    below = roms.depth_band(ds, meta, 0.0, 5.0, convention=SURFACE)
    fixed = roms.depth_band(ds, meta, 0.0, 5.0, convention=FIXED)
    # at high tide (zeta = +3) the fixed 0-5 m band starts 3 m under the surface and so
    # reaches deeper cells than the band that follows the surface
    assert fixed.sizes["s_rho"] > below.sizes["s_rho"], "the fixed band touches more"
    avg_fixed = roms.depth_average(ds, meta, 0.0, 5.0, convention=FIXED)["level"]
    avg_below = roms.depth_average(ds, meta, 0.0, 5.0, convention=SURFACE)["level"]
    fixed_avg, below_avg = _at(avg_fixed).values, _at(avg_below).values
    assert fixed_avg[0] < below_avg[0], "high tide: the fixed band sits deeper"
    assert fixed_avg[1] == pytest.approx(below_avg[1]), "zeta = 0: the frames coincide"
    assert fixed_avg[2] > below_avg[2], (
        "low tide: only the near-surface remnant is left"
    )


def test_a_band_with_no_convention_is_the_fixed_frame_at_datum_zero():
    ds, meta = tidal_roms()
    default = roms.depth_band(ds, meta, 0.0, 5.0)
    explicit = roms.depth_band(ds, meta, 0.0, 5.0, convention=FIXED)
    xr.testing.assert_identical(default, explicit)
    xr.testing.assert_identical(
        roms.depth_average(ds, meta, 0.0, 5.0),
        roms.depth_average(ds, meta, 0.0, 5.0, convention=FIXED),
    )


def test_a_fixed_band_is_measured_from_the_datum():
    ds, meta = tidal_roms()
    plain = roms.depth_band(ds, meta, 0.0, 5.0, convention=FIXED)
    # a datum 1 m above MSL: the band [0, 5] spans z = 1 .. -4, not 0 .. -5
    datum = {"origin": "fixed", "datum_z_m": 1.0}
    moved = roms.depth_band(ds, meta, 0.0, 5.0, convention=datum)
    total = _at(moved[roms.WEIGHT_COORD].sum("s_rho"))
    # high tide: all 5 m in the water; zeta = 0: z 0..-4 is water and z 1..0 is air, so
    # 4 m; low tide (surface at -3): only z -3..-4 is left, 1 m
    np.testing.assert_allclose(total.values, [5.0, 4.0, 1.0, 4.0])
    assert not np.array_equal(
        plain[roms.WEIGHT_COORD].values, moved[roms.WEIGHT_COORD].values
    )


def test_the_interface_axis_is_found_even_when_zeta_gives_z_w_a_time_axis():
    ds, meta = tidal_roms()
    with_zeta = roms.add_interface_coord(ds, meta)
    # the case the old "first dim that is not the centre dim" rule broke on
    assert with_zeta["z_w"].dims[0] == "time"
    assert roms._interface_dim(with_zeta, "s_rho") == "s_w"
    flat = roms.add_interface_coord(ds.drop_vars("zeta"), meta)
    assert roms._interface_dim(flat, "s_rho") == "s_w"


# -- bottom --------------------------------------------------------------------------


def test_bottom_returns_the_lowest_cell_like_surface_returns_the_top_one():
    ds, meta = tidal_roms()
    low = roms.bottom(ds, meta)
    assert "s_rho" not in low.dims and "z_rho" not in low.coords
    np.testing.assert_array_equal(low["level"].values, 0.0)
    np.testing.assert_array_equal(roms.surface(ds, meta)["level"].values, 9.0)
    # meta is optional, as for surface()
    np.testing.assert_array_equal(roms.bottom(ds)["level"].values, 0.0)
    # the cell's own height: the lowest centre sits half a layer above the seafloor
    zeta = _at(ds["zeta"]).values
    np.testing.assert_allclose(_at(low["height"]).values, zeta + (zeta + 20.0) * -0.95)
