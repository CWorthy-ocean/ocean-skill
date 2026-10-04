"""Depth conventions through the comparison pipeline, on a tide whose answers are known.

A free surface that moves ``+/-3 m`` over a ``20 m`` water column
(``tests/_tidal_roms``)
makes every vertical match a one-line sum, so what is asserted here is *numbers*, not
shapes: a CTD sample 1 m below the surface is the top cell at **every** step; the same
depth fixed in space is a different cell at high water and not in the water at all at
low water. The pieces are tested where they meet -- how a convention is resolved for a
lane (:func:`_lane_conventions`), what reaches the model (the frame, the kept free
surface, ``support``), how an observation's own levels are read (sign, units), what
the cache is keyed on, and the sign handling in ``align`` -- and end to end through
``Comparison.align`` for the headline case.

Worked answers (``zeta = [3, 0, -3, 0]``, ``level = k``, ``height = z_rho``):

* nearest ``level`` at 1 m: below the surface ``[9, 9, 9, 9]``; fixed
  ``[8, 9, NaN, 9]``.
* interp ``height`` at 1 m: below the surface ``[1.85, -1, -4, -1]`` (at the first step
  1 m lies in the top half-cell, 1.15 m down, so it takes the top cell's 1.85); fixed
  ``[-1, -1, NaN, -1]``. At 3 m below the surface it is exactly ``zeta - 3``.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import cache, catalog, comparison, depth_convention
from ocean_skill.align import ALONG_DIM, _align_along_path, _match_vertical
from ocean_skill.comparison import (
    DEPTH_ORIGIN_KEY,
    Comparison,
    _as_mean_aggregate,
    _identity,
    _lane_conventions,
    _prepare,
    _station_depth_from_metadata,
    compare,
    prepare_source,
)
from tests._tidal_roms import tidal_roms

SURFACE = {
    "origin": "surface",
    "datum_z_m": 0.0,
    "support": "point",
    "source": "declared",
}
FIXED = {"origin": "fixed", "datum_z_m": 0.0, "support": "point", "source": "declared"}
TIDE = np.array([3.0, 0.0, -3.0, 0.0])
NAN = np.nan
POINT = (200.01, 50.01)  # the middle cell of the fixture's 3 x 3 grid
SPEC = {"test": "level", "reference": "obs", "standard_name": "sea_water_temperature"}
HEIGHT_SPEC = {**SPEC, "test": "height"}


# -- the sources a comparison reads ----------------------------------------------------


def _install(monkeypatch, sources):
    """Stub the catalog and the reader for ``{name: (data, metadata)}``."""
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)


def _mooring(units="1", times=None, **meta):
    """Build a ``timeSeries`` obs at one point, 1 m down, at the fixture's 4 steps."""
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-07-01", periods=4, freq="h")
            if times is None
            else times,
            "lon": POINT[0],
            "lat": POINT[1],
            f"obs ({units})": 0.0,
        }
    )
    return frame, {"featureType": "timeSeries", "nominal_depth_m": 1.0, **meta}


def _series(monkeypatch, variable="level", depth=1.0, units="1", zeta=TIDE, **kw):
    """Build a ``Comparison`` of the fixture against a mooring, per ``kw``."""
    meta_extra = kw.pop("meta", {})
    ds, roms_meta = tidal_roms(zeta=zeta)
    # nanosecond stamps, like every obs frame: xarray's ``interp`` onto a
    # ``datetime64[ns]`` target over a ``datetime64[s]`` index (what the fixture's
    # decode leaves) is all-NaN
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))
    frame, obs_meta = _mooring(
        units=units, times=kw.pop("obs_times", None), **meta_extra
    )
    _install(monkeypatch, {"his": (ds, roms_meta), "pier": (frame, obs_meta)})
    spec = {**SPEC, "test": variable}
    return Comparison(
        reference="pier",
        test="his",
        variable=spec,
        select=kw.pop("select", {"depth": depth}),
        over="time",
        cache=False,
        **kw,
    )


def _test_lane(c):
    """Return the aligned test values along time, as a flat array."""
    return np.asarray(c.aligned["test"]).reshape(-1)


def _warnings_of(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in caught]


# -- the headline: a CTD below the surface, at the same depth through the tide ---------


def test_a_surface_referenced_obs_is_matched_at_the_top_cell_at_every_step(monkeypatch):
    c = _series(monkeypatch, meta={"depth_convention": {"origin": "surface"}})
    _, caught = _warnings_of(c.align)
    np.testing.assert_array_equal(_test_lane(c), [9, 9, 9, 9])
    assert not [m for m in caught if "free surface" in m or "entirely NaN" in m]


def test_the_same_depth_fixed_in_space_follows_the_tide_and_runs_dry(monkeypatch):
    c = _series(monkeypatch, meta={"depth_convention": {"origin": "fixed"}})
    _, caught = _warnings_of(c.align)
    np.testing.assert_array_equal(_test_lane(c), [8, 9, NAN, 9])
    # declared fixed: a decision, so the large-tide default warning stays quiet, but the
    # samples above the surface are counted and the fix named
    assert any("above the free surface" in m and "origin: surface" in m for m in caught)
    assert not any("declares no `depth_convention`" in m for m in caught)


def test_an_undeclared_obs_is_fixed_in_space_and_the_big_tide_is_flagged(monkeypatch):
    c = _series(monkeypatch)
    _, caught = _warnings_of(c.align)
    np.testing.assert_array_equal(_test_lane(c), [8, 9, NAN, 9])
    assert any("the free surface moves" in m and "declares no" in m for m in caught)


def test_depth_origin_overrides_an_undeclared_obs(monkeypatch):
    c = _series(monkeypatch, depth_origin="surface")
    _, caught = _warnings_of(c.align)
    np.testing.assert_array_equal(_test_lane(c), [9, 9, 9, 9])
    assert not [m for m in caught if "free surface" in m]


@pytest.mark.parametrize(
    ("meta", "depth", "expected"),
    [
        ({"depth_convention": {"origin": "surface"}}, 1.0, [1.85, -1.0, -4.0, -1.0]),
        # 3 m below the moving surface is exactly zeta - 3
        ({"depth_convention": {"origin": "surface"}}, 3.0, [0.0, -3.0, -6.0, -3.0]),
        ({"depth_convention": {"origin": "fixed"}}, 1.0, [-1.0, -1.0, NAN, -1.0]),
    ],
)
def test_interpolated_height_follows_the_declared_frame(
    monkeypatch, meta, depth, expected
):
    c = _series(
        monkeypatch,
        variable="height",
        units="m",
        depth=depth,
        depth_method="interp",
        meta={**meta, "nominal_depth_m": depth},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = _test_lane(c)
    np.testing.assert_allclose(values, expected, equal_nan=True)


def test_a_profile_obs_is_below_the_surface_by_default(monkeypatch):
    """An undeclared ``profile`` is surface-referenced; ``depth_origin`` undoes it."""
    ds, roms_meta = tidal_roms()
    cast = pd.DataFrame(
        {
            "time": pd.Timestamp("2024-07-01"),  # the first step: zeta = +3
            "lon": POINT[0],
            "lat": POINT[1],
            "depth (m)": [3.0, 5.0],
            "obs (m)": [0.0, 0.0],
        }
    )
    _install(
        monkeypatch,
        {"his": (ds, roms_meta), "cast": (cast, {"featureType": "profile"})},
    )

    def aligned(**kw):
        c = Comparison(
            reference="cast",
            test="his",
            variable=HEIGHT_SPEC,
            select={"depth": [3.0, 5.0]},
            over="Z",
            depth_method="interp",
            cache=False,
            **kw,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(c.aligned["test"]).reshape(-1)

    # 3 and 5 m below a surface at +3 m: z = 0 and -2; fixed in space, z = -3 and -5
    np.testing.assert_allclose(aligned(), [0.0, -2.0])
    np.testing.assert_allclose(aligned(depth_origin="fixed"), [-3.0, -5.0])


# -- how a lane's conventions are resolved ---------------------------------------------


def test_a_lane_resolves_its_own_convention_and_a_reference_frame_overrides_it():
    meta = {"depth_convention": {"origin": "surface", "units": "dbar"}}
    own, frame = _lane_conventions(meta, "temp", {})
    assert (own.origin, own.units) == ("surface", "dbar")
    assert frame == {
        "origin": "surface",
        "datum_z_m": 0.0,
        "support": "point",
        "source": "declared",
    }
    # the model follows the observation: a frame handed in replaces the lane's own...
    own, frame = _lane_conventions({}, "temp", {}, depth_convention=FIXED)
    assert own.origin == "fixed" and frame == FIXED
    # ... and a per-lane select entry outranks both
    _, frame = _lane_conventions(meta, "temp", {DEPTH_ORIGIN_KEY: "fixed"}, SURFACE)
    assert frame["origin"] == "fixed" and frame["source"] == "declared"


def test_a_select_override_replaces_only_what_it_names():
    base = {
        "origin": "fixed",
        "datum_z_m": 2.0,
        "support": "bottom",
        "source": "default",
    }
    # a datum alone is checked against the origin it lands on, and declares it
    _, frame = _lane_conventions({}, None, {DEPTH_ORIGIN_KEY: {"datum_z_m": 1.5}}, base)
    assert frame == {**base, "datum_z_m": 1.5, "source": "declared"}
    # a support alone leaves origin and its provenance alone
    _, frame = _lane_conventions(
        {}, None, {DEPTH_ORIGIN_KEY: {"support": "surface"}}, base
    )
    assert frame == {**base, "support": "surface"}
    # a surface origin has no datum, whatever was inherited
    _, frame = _lane_conventions({}, None, {DEPTH_ORIGIN_KEY: "surface"}, base)
    assert frame["datum_z_m"] == 0.0 and frame["origin"] == "surface"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("sideways", "depth_origin='sideways'"),
        ({"positive": "up"}, "only origin, datum_z_m and support"),
        ({"datum_z_m": 1.0}, "datum_z_m only applies to origin: fixed"),
    ],
)
def test_a_bad_depth_origin_is_refused_with_its_own_words(override, message):
    # on a lane's select (a datum is judged against the origin it lands on: surface)
    with pytest.raises(ValueError, match=message):
        _lane_conventions({}, None, {DEPTH_ORIGIN_KEY: override}, SURFACE)
    # and as a Comparison argument, before anything is read
    if not (
        isinstance(override, dict)
        and "origin" not in override
        and "datum_z_m" in override
    ):
        with pytest.raises(ValueError, match=message):
            Comparison(reference="a", test="b", variable="temp", depth_origin=override)


def test_a_comparison_writes_depth_origin_into_both_lanes_unless_one_names_its_own():
    plain = Comparison(
        reference="a",
        test="b",
        variable="temp",
        select={"depth": 5},
        depth_origin="surface",
    )
    assert plain.depth_origin == {"origin": "surface"}
    assert plain.select == {
        "depth": 5,
        DEPTH_ORIGIN_KEY: {"origin": "surface"},
    }  # still plain

    pair = Comparison(
        reference="a",
        test="b",
        variable="temp",
        select={"test": {"depth_origin": "fixed"}, "reference": {"depth": 5}},
        depth_origin={"origin": "surface"},
    )
    assert pair.select["test"] == {"depth_origin": "fixed"}  # its own, kept as written
    assert pair.select["reference"] == {
        "depth": 5,
        DEPTH_ORIGIN_KEY: {"origin": "surface"},
    }

    none = Comparison(reference="a", test="b", variable="temp", select={"depth": 5})
    assert none.depth_origin is None and none.select == {"depth": 5}


def test_compare_passes_depth_origin_to_every_comparison_it_builds():
    formed = []
    metas = {
        "ctd": {"featureType": "timeSeries", "variables": ["sea_water_temperature"]},
        "his": {"variables": ["sea_water_temperature"]},
    }
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: SimpleNamespace(metadata=metas[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self),
        ),
    ):
        compare(
            reference="ctd",
            test="his",
            variables=["sea_water_temperature"],
            depths=[1, 5],
            depth_origin="surface",
        )
    assert [c.depth_origin for c in formed] == [{"origin": "surface"}] * 2
    assert all(c.select[DEPTH_ORIGIN_KEY] == {"origin": "surface"} for c in formed)


# -- what reaches the model: the frame, the kept free surface, support -----------------


def _lane(
    variable="height", select=None, aggregate=None, frame=SURFACE, zeta=TIDE, **kw
):
    ds, meta = tidal_roms(zeta=zeta)
    return _prepare(
        ds,
        meta,
        variable,
        {"depth": 3.0, **(select or {})},
        aggregate,
        depth_method="interp",
        depth_convention=frame,
        obs_convention=depth_convention.resolve({}),
        **kw,
    )[0]


def _point(da):
    """Return the lane at the middle cell, as a flat array."""
    return np.asarray(da.isel(eta_rho=1, xi_rho=1)).reshape(-1)


def test_a_model_only_select_can_carry_depth_origin(monkeypatch):
    ds, meta = tidal_roms()
    _install(monkeypatch, {"his": (ds, meta)})
    select = {"depth": 3.0, DEPTH_ORIGIN_KEY: "surface"}
    da, caught = _warnings_of(
        lambda: prepare_source(
            "his", "height", select, None, use_cache=False, depth_method="interp"
        )[0]
    )
    np.testing.assert_allclose(_point(da), TIDE - 3.0)
    assert not [m for m in caught if "matched no axis" in m]
    assert select == {
        "depth": 3.0,
        DEPTH_ORIGIN_KEY: "surface",
    }  # the caller's own is untouched


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        # 3 m below the surface is the 9th cell from the bottom at every step...
        ("surface", [8, 8, 8, 8]),
        # ... fixed in space it is cell 7 at high water, and at low water the target
        # (z = -3) *is* the free surface, so the top half-cell: the top cell, 9
        ("fixed", [7, 8, 9, 8]),
    ],
)
def test_a_field_over_a_model_follows_its_own_depth_origin(
    monkeypatch, origin, expected
):
    from ocean_skill.field import Field

    ds, meta = tidal_roms()
    _install(monkeypatch, {"his": (ds, meta)})
    field = Field(
        "his", "level", select={"depth": 3.0, DEPTH_ORIGIN_KEY: origin}, cache=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        da = field.prepare()
    np.testing.assert_array_equal(_point(da), expected)


def test_a_direct_prepare_resolves_the_convention_from_meta_and_select():
    """A caller that passes no conventions gets the lane's own (catalog, select)."""
    ds, meta = tidal_roms()

    def lane(select, **meta_extra):
        return _prepare(
            ds, {**meta, **meta_extra}, "height", select, None, depth_method="interp"
        )[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        declared = lane({"depth": 3.0}, depth_convention={"origin": "surface"})
        asked = lane({"depth": 3.0, DEPTH_ORIGIN_KEY: "surface"})
        default = lane({"depth": 3.0})
    np.testing.assert_allclose(_point(declared), TIDE - 3.0)
    np.testing.assert_allclose(_point(asked), TIDE - 3.0)
    np.testing.assert_allclose(_point(default), [-3.0, -3.0, -3.85, -3.0])


def test_the_reference_frame_fails_open_but_a_lane_with_a_broken_declaration_raises(
    monkeypatch,
):
    broken = {"featureType": "timeSeries", "depth_convention": {"origin": "sideways"}}
    ds, roms_meta = tidal_roms()
    frame, _ = _mooring()
    _install(monkeypatch, {"his": (ds, roms_meta), "pier": (frame, broken)})
    c = Comparison(
        reference="pier",
        test="his",
        variable=SPEC,
        select={"depth": 1.0},
        over="time",
        cache=False,
    )
    assert c._reference_depth_frame() is None  # the test lane then resolves its own
    assert (
        "_depth_frame" not in repr(c._cache_key) and c._cache_key
    )  # and keying still works
    with pytest.raises(ValueError, match="depth_convention"):
        prepare_source("pier", "obs", {}, None, use_cache=False)
    # an unresolvable reference is no different
    monkeypatch.undo()
    assert (
        Comparison(
            reference="nowhere", test="nothing", variable="temp"
        )._reference_depth_frame()
        is None
    )


def test_time_method_interp_moves_the_whole_test_lane_onto_the_obs_times(monkeypatch):
    """Not only a repeat-visit station: an ordinary ``over="time"`` series too.

    The obs is sampled half an hour off the model's steps; with ``"interp"`` the model's
    free surface (so the depth frame) is interpolated onto the obs's own instants before
    the depth is matched, and ``match_axis`` then sees the two on the same times.
    """
    times = pd.date_range("2024-07-01 00:30", periods=3, freq="h")
    meta = {"depth_convention": {"origin": "surface"}, "nominal_depth_m": 3.0}
    c = _series(
        monkeypatch,
        variable="height",
        units="m",
        depth=3.0,
        depth_method="interp",
        time_method="interp",
        obs_times=times,
        meta=meta,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = _test_lane(c)
    # zeta between the steps: 1.5, -1.5, -1.5; 3 m below it is zeta - 3
    np.testing.assert_allclose(values, [-1.5, -4.5, -4.5])


def test_the_support_cell_is_read_instead_of_matching_a_depth():
    bottom = _lane("level", {"depth": 25.0}, frame={**SURFACE, "support": "bottom"})
    top = _lane("level", {"depth": 25.0}, frame={**FIXED, "support": "surface"})
    # 25 m is below the 20 m seafloor: a point match would be NaN, a support is never
    np.testing.assert_array_equal(_point(bottom), [0, 0, 0, 0])
    np.testing.assert_array_equal(_point(top), [9, 9, 9, 9])
    assert (
        bottom.attrs["depth_support"] == "bottom"
        and top.attrs["depth_support"] == "surface"
    )
    assert "z" not in bottom.dims and "z" not in top.dims


def test_a_support_on_a_list_or_a_band_is_ignored_with_a_warning():
    frame = {**FIXED, "support": "bottom"}
    for request in ([3.0, 5.0], {"min": 2.0, "max": 6.0}):
        with pytest.warns(UserWarning, match="only applies to a single-depth request"):
            da = _lane("level", {"depth": request}, frame=frame)
        assert "depth_support" not in da.attrs


def test_a_time_mean_keeps_the_mean_free_surface():
    """The frame of a time-mean field is the mean surface, not a flat one."""
    zeta = (3.0, 1.0, -1.0, 1.0)  # mean 1 m: the old zeta = 0 fallback cannot hide
    surface = _lane(aggregate={"time": "mean"}, zeta=zeta)
    fixed = _lane(aggregate={"time": "mean"}, zeta=zeta, frame=FIXED)
    np.testing.assert_allclose(_point(surface), [1.0 - 3.0])
    np.testing.assert_allclose(_point(fixed), [-3.0])


def test_a_std_aggregate_still_builds_its_frame_from_the_mean_surface():
    from ocean_skill import roms

    seen = {}
    real = roms.to_depth

    def spy(sub, meta, targets, **kwargs):
        seen["zeta"] = np.asarray(sub["zeta"])
        seen["z_rho"] = np.asarray(sub["z_rho"])
        return real(sub, meta, targets, **kwargs)

    with mock.patch.object(roms, "to_depth", spy):
        da = _lane(aggregate={"time": "std"}, zeta=(3.0, 1.0, -1.0, 1.0))
    assert da.ndim == 2  # the std field is still just the std field...
    assert "time" not in seen["zeta"].shape and np.allclose(
        seen["zeta"], 1.0
    )  # ... in a mean frame
    # z_rho = zeta + (zeta + h) * sigma on the mean surface: top cell 1 + 21 * -0.05
    np.testing.assert_allclose(seen["z_rho"][-1], 1.0 + 21.0 * -0.05)


def test_the_mean_aggregate_keeps_the_binning_and_drops_the_statistic():
    spec = {
        "time": {"groupby": "month", "reduce": "std", "spread": "std", "q": 0.5},
        "lat": [{"resample": "1D", "reduce": "max"}, {"reduce": "quantile", "q": 0.9}],
        "lon": "var",
    }
    assert _as_mean_aggregate(spec) == {
        "time": {"groupby": "month", "reduce": "mean"},
        "lat": [{"resample": "1D", "reduce": "mean"}, {"reduce": "mean"}],
        "lon": "mean",
    }
    assert _as_mean_aggregate(None) == {}


def test_detide_filters_the_free_surface_with_the_field():
    """The free surface is low-passed like the field: a stand-in proves it was."""

    def fake_detide(da, *, T, component):
        assert (T, component) == (33, "subtidal")
        return da * 0.0 if da.name == "zeta" else da  # a perfectly flat free surface

    with mock.patch("ocean_skill.detide.detide", fake_detide):
        da = _lane(detide={"T": 33})
    # frame = z_rho on a flat surface, so 3 m is the cell centre k = 8 at every step
    # and the (un-filtered) field there is zeta_t + (zeta_t + 20) * -0.15
    np.testing.assert_allclose(_point(da), 0.85 * TIDE - 3.0)


@pytest.mark.parametrize(
    "transect",
    [
        {"xi_rho": 1},  # a grid slice keeps z_rho but not the surface it was built on
        # a bilinear path drops z_rho/z_w and regrids zeta as a data variable
        {"points": [[200.005, 50.005], [200.015, 50.015]], "method": "bilinear"},
    ],
    ids=["grid slice", "bilinear"],
)
def test_a_transect_keeps_the_free_surface(transect):
    da = _lane(
        select={
            "depth": [3.0, 5.0],
            "transect": transect,
            "time": "2024-07-01T00:00:00",
        }
    )
    assert da.dims == (ALONG_DIM, "z")
    # 3 and 5 m below a surface at +3 m (the first step) are z = 0 and -2 at every
    # column -- on a flat surface they would be -3 and -5
    np.testing.assert_allclose(
        np.asarray(da), [[0.0, -2.0]] * da.sizes[ALONG_DIM], atol=1e-9
    )


def test_a_section_on_native_levels_keeps_its_flat_plot_mesh():
    """With no depth asked, z_rho is the zeta-free plot *mesh*, finite over land."""
    ds, meta = tidal_roms(land=True)
    da, _ = _prepare(ds, meta, "height", {"transect": {"xi_rho": 0}}, None)
    assert "time" not in da["z_rho"].dims
    assert np.isfinite(np.asarray(da["z_rho"])).all()  # the land column included
    np.testing.assert_allclose(np.asarray(da["z_rho"]).max(), 20.0 * -0.05)


# -- how an observation's own levels are read ------------------------------------------


def _cast(depth, positive=None, units=None, **attrs):
    """Build a profile obs as an xarray source, depth stored as the test says."""
    coord = {"axis": "Z", **attrs}
    if positive:
        coord["positive"] = positive
    if units:
        coord["units"] = units
    return xr.Dataset(
        {"obs": (("depth",), np.arange(len(depth), dtype=float), {"units": "m"})},
        coords={"depth": ("depth", np.asarray(depth, float), coord)},
    )


def _meta(**kw):
    return {"featureType": "profile", **kw}


def test_a_positive_up_profile_is_read_as_depths_in_a_literal_depth_request():
    cast = _cast([-1.0, -3.0, -5.0], positive="up")
    da, _ = _prepare(cast, _meta(), "obs", {"depth": [3.0, 5.0]}, literal_depths=True)
    # picked by *depth* (3 m, 5 m), stored as the heights they are, then stamped as
    # the positive-down metres they now are -- never read back as 3 m, 5 m up
    np.testing.assert_array_equal(np.asarray(da["depth"]), [3.0, 5.0])
    np.testing.assert_array_equal(np.asarray(da), [1.0, 2.0])
    attrs = da["depth"].attrs
    assert (attrs["positive"], attrs["units"]) == ("down", "m")
    assert attrs[depth_convention.NORMALIZED_ATTR] == 1


def test_a_positive_up_profile_keeps_its_own_axis_when_picking_levels():
    cast = _cast([-1.0, -3.0, -5.0], positive="up")
    da, _ = _prepare(cast, _meta(), "obs", {"depth": [3.0, 5.0]})
    np.testing.assert_array_equal(np.asarray(da), [1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(da["depth"]), [-3.0, -5.0])  # as stored
    assert da["depth"].attrs["positive"] == "up"  # and still saying so


def test_actual_depth_is_metres_positive_down_whatever_the_obs_stores():
    _, depth_up = _prepare(
        _cast([-1.0, -3.0, -5.0], positive="up"), _meta(), "obs", {"depth": 3.2}
    )
    _, depth_down = _prepare(_cast([1.0, 3.0, 5.0]), _meta(), "obs", {"depth": 3.2})
    assert depth_up == depth_down == 3.0
    # a declared convention reads values with no attrs of their own (dbar -> metres)
    meta = _meta(depth_convention={"units": "dbar"})
    _, depth_dbar = _prepare(_cast([1.0, 3.0, 5.0]), meta, "obs", {"depth": 3.2})
    assert depth_dbar == 3.0


def test_a_scalar_station_depth_is_read_through_its_sign():
    ds = xr.Dataset(
        {"obs": (("time",), [1.0, 2.0], {"units": "m"})},
        coords={
            "time": pd.date_range("2024-07-01", periods=2, freq="h"),
            "depth": ((), -30.0, {"positive": "up"}),
        },
    )
    _, actual = _prepare(ds, {"featureType": "timeSeries"}, "obs", {})
    assert actual == 30.0


def test_a_positive_up_profile_matches_the_model_at_the_right_depths(monkeypatch):
    ds, roms_meta = tidal_roms()
    cast = _cast([-3.0, -5.0], positive="up")
    cast = cast.assign_coords(
        time=pd.Timestamp("2024-07-01"), lon=POINT[0], lat=POINT[1]
    )
    _install(
        monkeypatch,
        {"his": (ds, roms_meta), "cast": (cast, {"featureType": "profile"})},
    )
    for literal in (True, False):
        c = Comparison(
            reference="cast",
            test="his",
            variable=HEIGHT_SPEC,
            select={"depth": [3.0, 5.0]},
            over="Z",
            depth_method="interp",
            literal_depths=literal,
            cache=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aligned = c.aligned
        # profile -> surface-referenced; zeta = +3 at the cast: z = 0 and -2
        np.testing.assert_allclose(np.asarray(aligned["test"]).reshape(-1), [0.0, -2.0])
        np.testing.assert_array_equal(
            np.asarray(aligned["reference"]).reshape(-1), [0.0, 1.0]
        )


def test_the_stations_depth_from_its_metadata_honours_a_declared_sign():
    def depth_of(**meta):
        base = {
            "featureType": "timeSeries",
            "geospatial_vertical_min": -30.0,
            "geospatial_vertical_max": -20.0,
        }
        with mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: SimpleNamespace(metadata={**base, **meta}),
        ):
            return _station_depth_from_metadata("m")

    assert depth_of() is None  # undeclared and negative: still "a units mixup"
    assert depth_of(depth_convention={"positive": "up"}) == 25.0
    assert depth_of(depth_convention={"inferred": {"positive": "up"}}) == 25.0
    # declared positive-down, and negative: nothing to flip
    assert depth_of(depth_convention={"positive": "down"}) is None
    # a broken declaration is the lane's to report, not this probe's
    assert depth_of(depth_convention={"positive": "sideways"}) is None
    # positive-down values are untouched
    assert depth_of(geospatial_vertical_min=20.0, geospatial_vertical_max=30.0) == 25.0


def _vertical_lane(values, dim, **attrs):
    return xr.DataArray(
        np.arange(len(values), dtype=float) * 10.0,
        dims=(dim,),
        coords={dim: (dim, np.asarray(values, float), attrs)},
    )


def test_match_vertical_reads_each_axis_by_its_own_sign():
    test = _vertical_lane([-1.0, -3.0, -5.0, -7.0], "z", positive="up")
    reference = _vertical_lane([-3.0, -5.0], "depth", positive="up")  # heights, too
    matched, _, report = _match_vertical(
        test, reference, "z", "depth", method="nearest"
    )
    np.testing.assert_array_equal(np.asarray(matched), [10.0, 20.0])  # 3 m and 5 m
    np.testing.assert_array_equal(np.asarray(matched["depth"]), [-3.0, -5.0])
    assert report["n_matched"] == 2

    # the same two columns with the reference stored positive-down give the same answer
    positive_down = _vertical_lane([3.0, 5.0], "depth", positive="down")
    matched_down, _, _ = _match_vertical(
        test, positive_down, "z", "depth", method="nearest"
    )
    np.testing.assert_array_equal(np.asarray(matched_down), [10.0, 20.0])


def test_observational_axes_become_negative_up_z_by_their_own_sign():
    from ocean_skill.align import _observational_vertical_to_z

    down = _observational_vertical_to_z(_vertical_lane([5.0, 50.0], "depth"))
    np.testing.assert_array_equal(np.asarray(down["z"]), [-5.0, -50.0])
    up = _observational_vertical_to_z(
        _vertical_lane([-5.0, -50.0], "depth", positive="up")
    )
    np.testing.assert_array_equal(
        np.asarray(up["z"]), [-5.0, -50.0]
    )  # already negative-up
    assert down["z"].attrs["positive"] == up["z"].attrs["positive"] == "up"
    # negative values and nobody saying which way is down: a height, renamed as is
    inferred = _observational_vertical_to_z(_vertical_lane([-5.0, -50.0], "lev"))
    np.testing.assert_array_equal(np.asarray(inferred["z"]), [-5.0, -50.0])
    # mixed signs with no word on it: not guessed at
    mixed = _vertical_lane([-5.0, 50.0], "depth")
    assert "z" not in _observational_vertical_to_z(mixed).dims


def _section_lane(vdim, values, **attrs):
    lon = np.array([-95.0, -94.9])
    along = np.array([0.0, 10.0])
    return xr.DataArray(
        np.ones((len(values), 2)),
        dims=(vdim, ALONG_DIM),
        coords={
            vdim: (vdim, np.asarray(values, float), attrs),
            ALONG_DIM: (ALONG_DIM, along, {"path_method": "nearest", "units": "km"}),
            "lon": (ALONG_DIM, lon),
            "lat": (ALONG_DIM, np.full(2, 24.0)),
        },
        attrs={"units": "degC"},
    )


@pytest.mark.parametrize(
    ("vdim", "values", "attrs"),
    [
        ("depth", [50.0, 200.0], {}),
        ("depth", [-50.0, -200.0], {"positive": "up"}),
        ("z", [-50.0, -200.0], {"positive": "up"}),
        ("z", [-50.0, -200.0], {}),
    ],
)
def test_a_sections_reference_levels_are_positive_down_whatever_the_axis(
    vdim, values, attrs
):
    test = _section_lane("z", [-50.0, -200.0], positive="up")
    reference = _section_lane(vdim, values, **attrs)
    out = _align_along_path(
        test,
        reference,
        convention="-180-180",
        test_name="test",
        reference_name="reference",
    )
    assert out.attrs["reference_levels"] == [50.0, 200.0]


# -- what the cache is keyed on --------------------------------------------------------


@pytest.fixture
def lane_keys(monkeypatch):
    """Record the select each lane's cache key is built from."""
    seen = []
    real = cache.key_for_prepared

    def spy(*, source, variable, select):
        seen.append(select)
        return real(source=source, variable=variable, select=select)

    monkeypatch.setattr(cache, "key_for_prepared", spy)
    return seen


def _key(lane_keys, name, meta, select, *, data, frame=None, aggregate=None):
    sources = {name: (data, meta)}
    with (
        mock.patch.object(osk, "read", lambda n, **kw: sources[n][0]),
        mock.patch.object(
            catalog, "resolve", lambda n: SimpleNamespace(metadata=sources[n][1])
        ),
    ):
        prepare_source(
            name,
            "height" if meta.get("model") == "roms" else "obs",
            select,
            aggregate,
            use_cache=True,
            depth_convention=frame,
        )
    return lane_keys[-1]


def test_a_model_lanes_key_carries_the_frame_only_when_a_depth_is_matched(lane_keys):
    ds, meta = tidal_roms()
    for request, expected in (
        (3.0, True),
        ([3.0, 5.0], True),
        (["surface", 5.0], True),
        ({"min": 0.0, "max": 10.0}, True),
        ("surface", False),
        ("column", False),
        (["surface"], False),
    ):
        select = _key(
            lane_keys, "his", meta, {"depth": request}, data=ds, frame=SURFACE
        )
        assert ("_depth_frame" in select) is expected, request
        assert "_depth_convention" not in select
    # no vertical request at all (a bare Field)
    assert "_depth_frame" not in _key(
        lane_keys, "his", meta, {}, data=ds, frame=SURFACE
    )

    surface = _key(lane_keys, "his", meta, {"depth": 3.0}, data=ds, frame=SURFACE)
    fixed = _key(lane_keys, "his", meta, {"depth": 3.0}, data=ds, frame=FIXED)
    shifted = _key(
        lane_keys,
        "his",
        meta,
        {"depth": 3.0},
        data=ds,
        frame={**FIXED, "datum_z_m": 1.0},
    )
    bottom = _key(
        lane_keys,
        "his",
        meta,
        {"depth": 3.0},
        data=ds,
        frame={**FIXED, "support": "bottom"},
    )
    assert surface["_depth_frame"] == {
        "origin": "surface",
        "datum_z_m": 0.0,
        "support": "point",
    }
    assert (
        len({repr(k["_depth_frame"]) for k in (surface, fixed, shifted, bottom)}) == 4
    )
    # how sure the frame is changes warnings, never values: it stays out of the key
    assert fixed == _key(
        lane_keys,
        "his",
        meta,
        {"depth": 3.0},
        data=ds,
        frame={**FIXED, "source": "default"},
    )


def test_an_observation_lane_keys_on_its_own_resolved_convention(lane_keys):
    cast = _cast([1.0, 3.0])
    plain = _key(lane_keys, "cast", _meta(), {"depth": 3.0}, data=cast)
    up = _key(
        lane_keys,
        "cast",
        _meta(depth_convention={"positive": "up"}),
        {"depth": 3.0},
        data=cast,
    )
    assert plain["_depth_convention"] == depth_convention.resolve(_meta()).key()
    assert up["_depth_convention"]["positive"] == "up"
    assert "_depth_frame" not in plain and "_time_zone" not in plain
    # a profile resolves surface-referenced by default, a mooring fixed
    assert plain["_depth_convention"]["origin"] == "surface"
    assert (
        _key(lane_keys, "m", {"featureType": "timeSeries"}, {}, data=cast)[
            "_depth_convention"
        ]["origin"]
        == "fixed"
    )


def test_a_matched_lane_survives_the_cache_round_trip(monkeypatch):
    """The edge counts and origin stamped on a matched lane are cacheable attrs."""
    ds, meta = tidal_roms()
    reads = []
    monkeypatch.setattr(osk, "read", lambda name, **kw: reads.append(name) or ds)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata=meta))

    def lane(frame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return prepare_source(
                "his", "level", {"depth": 1.0}, None, depth_convention=frame
            )[0]

    first = lane(SURFACE)
    again = lane(SURFACE)
    assert reads == ["his"], "the second call must be served from the cache"
    xr.testing.assert_identical(first, again)
    assert again.attrs["depth_origin"] == "surface"
    assert (
        again.attrs["depth_edge_top"] > 0
    )  # the first step's 1 m is the top half-cell
    # another frame is another entry, with other values
    fixed = lane(FIXED)
    assert reads == ["his", "his"]
    assert not np.array_equal(np.asarray(fixed), np.asarray(first), equal_nan=True)


def test_a_declared_time_zone_joins_the_lane_key(lane_keys):
    cast = _cast([1.0, 3.0])
    plain = _key(lane_keys, "cast", _meta(), {}, data=cast)
    zoned = _key(lane_keys, "cast", _meta(time_zone="America/Anchorage"), {}, data=cast)
    assert "_time_zone" not in plain
    assert zoned["_time_zone"] == "America/Anchorage"


def test_the_comparison_key_follows_the_references_convention(monkeypatch):
    def key(**meta):
        return _series(monkeypatch, meta=meta)._cache_key

    surface = key(depth_convention={"origin": "surface"})
    fixed = key(depth_convention={"origin": "fixed"})
    assert surface != fixed
    assert key(depth_convention={"origin": "surface"}) == surface  # and it is stable
    assert key(depth_convention={"origin": "fixed", "datum_z_m": 1.0}) != fixed
    # a catalog edit that declares the (same) default changes nothing a value depends on
    assert key() == key(
        depth_convention={"origin": "fixed", "inferred": {"origin": "surface"}}
    )


def test_a_declared_time_zone_on_either_side_changes_the_comparison_key(monkeypatch):
    plain = _series(monkeypatch)._cache_key
    zoned = _series(monkeypatch, meta={"time_zone": "America/Anchorage"})._cache_key
    assert plain != zoned
    assert (
        _series(monkeypatch, meta={"time_zone": "UTC"})._cache_key == plain
    )  # UTC is no zone


def test_pooling_tells_depth_origins_apart(monkeypatch):
    surface = _series(monkeypatch, depth_origin="surface")
    fixed = _series(monkeypatch, depth_origin="fixed")
    bare = _series(monkeypatch)
    assert len({_identity(c) for c in (surface, fixed, bare)}) == 3
    declared = _series(monkeypatch, meta={"depth_convention": {"origin": "surface"}})
    assert _identity(declared) != _identity(bare)  # the resolved frame counts, too
    assert _identity(
        SimpleNamespace(test_name="a")
    )  # a stub without any of it still works
