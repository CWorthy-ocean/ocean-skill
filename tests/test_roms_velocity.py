"""Tests for ROMS' true geographic east/north velocity (:mod:`ocean_skill.roms`).

Regression coverage for the bug where ``osk.compare(test="his", variables=
["eastward_sea_water_velocity"], ...)`` skipped every pair with `"No variable named
'sea_water_x_velocity'"`: ROMS' own ``u``/``v`` are grid-relative and staggered, and
``roms.to_depth`` deliberately drops any variable not already on rho points (see that
function's own comment). ``roms.standardize`` now derives true eastward/northward
velocity by averaging ``u``/``v`` onto rho points and rotating by the grid ``angle``
(:func:`ocean_skill.roms._add_geographic_velocity`), which reaches rho points and so
survives ``to_depth`` -- see the end-to-end section below for the direct regression
check.

Synthetic throughout, like ``tests/test_roms_chunking.py``: unlike ``test_roms.py``
this needs no local roms-tools example output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms

N = 8  # s_rho


def _roms_like(
    ny=4, nx=5, angle=0.0, with_angle=True, with_v=True, land_at=None, raw_names=False
) -> tuple[xr.Dataset, dict]:
    """Build a minimal self-contained ROMS Dataset, ready for :func:`roms.standardize`.

    ``u``/``v`` sit on the staggered grid (``xi_u`` has ``nx - 1`` points, ``eta_v``
    has ``ny - 1``), exactly as roms-tools writes them. ``angle`` is uniform, so a
    caller can check the rotation against a hand-computed value.

    ``raw_names=True`` names the velocity variables ``u``/``v`` -- a real ROMS file's
    own spelling -- and leaves ``angle`` as a plain data variable, exactly the
    pre-``standardize`` state: :func:`roms.standardize` renames u/v via
    ``meta["standard_names"]`` and promotes ``angle`` to a coordinate itself, so this
    is what the ``standardize``/``to_depth`` end-to-end tests below need. The default
    (``False``) instead names the velocity variables
    ``sea_water_x_velocity``/``sea_water_y_velocity`` directly and attaches ``angle``
    as a coordinate up front -- the already-``standardize``d state
    :func:`roms._add_geographic_velocity` itself expects -- for tests that call it (or
    ``_average_to_rho``) without going through ``standardize`` first.
    """
    rng = np.random.default_rng(0)
    sigma_r = np.linspace(-1.0 + 1.0 / (2 * N), -1.0 / (2 * N), N)
    h = np.full((ny, nx), 100.0)
    mask_rho = np.ones((ny, nx))
    if land_at is not None:
        mask_rho[land_at] = 0.0
    lon = np.tile(np.linspace(-90.0, -89.0, nx), (ny, 1))
    lat = np.tile(np.linspace(20.0, 21.0, ny)[:, None], (1, nx))

    x_name = "u" if raw_names else "sea_water_x_velocity"
    y_name = "v" if raw_names else "sea_water_y_velocity"
    data = {
        "lon_rho": (("eta_rho", "xi_rho"), lon),
        "lat_rho": (("eta_rho", "xi_rho"), lat),
        "h": (("eta_rho", "xi_rho"), h),
        "mask_rho": (("eta_rho", "xi_rho"), mask_rho),
        "sigma_r": ("s_rho", sigma_r),
        "Cs_r": ("s_rho", sigma_r),  # unstretched: only the horizontal math is tested
        x_name: (("s_rho", "eta_rho", "xi_u"), rng.normal(0.1, 0.02, (N, ny, nx - 1))),
    }
    if with_v:
        data[y_name] = (
            ("s_rho", "eta_v", "xi_rho"),
            rng.normal(0.05, 0.02, (N, ny - 1, nx)),
        )
    coords = {}
    if with_angle:
        angle_var = (("eta_rho", "xi_rho"), np.full((ny, nx), angle))
        (coords if not raw_names else data)["angle"] = angle_var

    ds = xr.Dataset(data, coords=coords)
    meta = {
        "self_contained_grid": True,
        "vertical": {"hc": 50.0, "s_dim": "s_rho"},
        "standard_names": {"u": "sea_water_x_velocity", "v": "sea_water_y_velocity"},
    }
    return ds, meta


# -- _average_to_rho: pure staggered-to-rho averaging -------------------------------


def test_average_to_rho_constant_field_stays_constant():
    """Interior and edges alike: averaging a constant changes nothing."""
    da = xr.DataArray(np.full((3, 4), 2.0), dims=("eta_rho", "xi_u"))
    rho = roms._average_to_rho(da, "xi_u", "xi_rho")
    assert rho.dims == ("eta_rho", "xi_rho")
    assert rho.shape == (3, 5)
    np.testing.assert_allclose(rho.data, 2.0)


def test_average_to_rho_interior_is_the_bracketing_mean_and_edges_are_nearest():
    """A ramp pins the exact rule: interior = 2-point mean, edges = nearest value."""
    row = np.array([0.0, 1.0, 2.0, 3.0])  # xi_u, 4 points -> 5 rho points
    da = xr.DataArray(row[None, :], dims=("eta_rho", "xi_u"))
    rho = roms._average_to_rho(da, "xi_u", "xi_rho")
    np.testing.assert_allclose(rho.data[0], [0.0, 0.5, 1.5, 2.5, 3.0])


def test_average_to_rho_stays_lazy():
    pytest.importorskip("dask")
    import dask.array as da_

    arr = da_.from_array(np.arange(12.0).reshape(3, 4), chunks=(2, 2))
    da = xr.DataArray(arr, dims=("eta_rho", "xi_u"))
    rho = roms._average_to_rho(da, "xi_u", "xi_rho")
    assert rho.chunks is not None


def test_average_to_rho_does_not_fragment_chunks():
    """A multi-chunk staggered dim must not produce size-1 chunks at every boundary.

    ``0.5*(left+right)`` adds two slices offset by one, which -- left to dask's own
    unification -- fragments the averaged dim into a ``(1, N, 1, N, ...)`` chunking.
    That fragmentation is what made the downstream rotation multiply
    (:func:`roms._add_geographic_velocity`) build a task graph into the millions over a
    real one-step-per-chunk history file, hanging the read for minutes. Guard the
    chunks stay coarse (no size-1 fragments) and the values stay correct.
    """
    pytest.importorskip("dask")
    import dask.array as da_

    # xi_u chunked in 3 pieces, like a real ROMS file's staggered velocity dim.
    row = np.tile(np.arange(9.0), (4, 1))  # (eta_rho=4, xi_u=9) -> xi_rho=10
    arr = da_.from_array(row, chunks=(4, 3))
    da = xr.DataArray(arr, dims=("eta_rho", "xi_u"))
    rho = roms._average_to_rho(da, "xi_u", "xi_rho")
    xi_chunks = rho.chunks[rho.dims.index("xi_rho")]
    assert 1 not in xi_chunks, f"fragmented into size-1 chunks: {xi_chunks}"
    # values still exact: interior = bracketing mean, edges = nearest
    expected = np.concatenate([[0.0], 0.5 * (row[0, :-1] + row[0, 1:]), [8.0]])
    np.testing.assert_allclose(np.asarray(rho.data)[0], expected)


# -- _add_geographic_velocity: rotation ----------------------------------------------


def test_rotation_angle_zero_is_the_identity():
    ds, _ = _roms_like(angle=0.0)
    out = roms._add_geographic_velocity(ds)
    u_rho = roms._average_to_rho(ds["sea_water_x_velocity"], "xi_u", "xi_rho")
    v_rho = roms._average_to_rho(ds["sea_water_y_velocity"], "eta_v", "eta_rho")
    np.testing.assert_allclose(out["eastward_sea_water_velocity"].values, u_rho.data)
    np.testing.assert_allclose(out["northward_sea_water_velocity"].values, v_rho.data)


def test_rotation_quarter_turn_swaps_and_flips_components():
    """angle=pi/2: the grid's x-axis points true north, so east=-v_rho, north=u_rho."""
    ds, _ = _roms_like(angle=np.pi / 2)
    out = roms._add_geographic_velocity(ds)
    u_rho = roms._average_to_rho(ds["sea_water_x_velocity"], "xi_u", "xi_rho")
    v_rho = roms._average_to_rho(ds["sea_water_y_velocity"], "eta_v", "eta_rho")
    np.testing.assert_allclose(
        out["eastward_sea_water_velocity"].values, -v_rho.data, atol=1e-10
    )
    np.testing.assert_allclose(
        out["northward_sea_water_velocity"].values, u_rho.data, atol=1e-10
    )


def test_rotation_arbitrary_angle_matches_hand_computation():
    theta = np.pi / 6
    ds, _ = _roms_like(angle=theta)
    out = roms._add_geographic_velocity(ds)
    u_rho = roms._average_to_rho(ds["sea_water_x_velocity"], "xi_u", "xi_rho").data
    v_rho = roms._average_to_rho(ds["sea_water_y_velocity"], "eta_v", "eta_rho").data
    expected_east = u_rho * np.cos(theta) - v_rho * np.sin(theta)
    expected_north = u_rho * np.sin(theta) + v_rho * np.cos(theta)
    np.testing.assert_allclose(
        out["eastward_sea_water_velocity"].values, expected_east
    )
    np.testing.assert_allclose(
        out["northward_sea_water_velocity"].values, expected_north
    )


def test_missing_angle_warns_and_leaves_only_grid_relative_components():
    ds, _ = _roms_like(with_angle=False)
    with pytest.warns(UserWarning, match="grid `angle`"):
        out = roms._add_geographic_velocity(ds)
    assert "eastward_sea_water_velocity" not in out
    assert "northward_sea_water_velocity" not in out
    assert "sea_water_x_velocity" in out  # untouched, unaffected by the no-op


def test_missing_velocity_is_a_silent_noop():
    """No u/v at all (e.g. a tracer-only read): nothing to derive, nothing to warn."""
    ds = xr.Dataset(
        {"h": (("eta_rho", "xi_rho"), np.full((3, 4), 100.0))},
        coords={"angle": (("eta_rho", "xi_rho"), np.zeros((3, 4)))},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        out = roms._add_geographic_velocity(ds)
    assert out.identical(ds)


def test_derived_velocity_stays_lazy():
    pytest.importorskip("dask")
    import dask.array as da_

    ds, _ = _roms_like()
    u, v = ds["sea_water_x_velocity"], ds["sea_water_y_velocity"]
    ds = ds.assign(
        sea_water_x_velocity=(u.dims, da_.from_array(u.values, chunks=(2, 2, 2))),
        sea_water_y_velocity=(v.dims, da_.from_array(v.values, chunks=(2, 2, 2))),
    )
    out = roms._add_geographic_velocity(ds)
    assert out["eastward_sea_water_velocity"].chunks is not None
    assert out["northward_sea_water_velocity"].chunks is not None


# -- add_geographic_velocity_windowed: byte-identical to a full-domain derive-then-crop
#
# Regression coverage for the ADCP-mooring hang: a full-domain
# _add_geographic_velocity's task graph, over a real ROMS history file's one-step-
# per-chunk record, still carries hundreds of thousands of tasks even after being
# culled down to one water column -- the confirmed cause of a multi-minute stall on
# a single point/station comparison (see prepare_source's own comment). The fix
# derives on a halo-cropped window instead (align._point_window supplies the halo
# and the trim it needs); this section checks that gives byte-identical values at
# every window position, including where the window touches a domain edge.


def _rotated_grid(ny=8, nx=9, land_at=None) -> tuple[xr.Dataset, dict]:
    """A ROMS-like Dataset with a spatially-varying ``angle`` and ``lon``/``lat``
    coords -- the post-``standardize``-coord-assignment, pre-``_add_geographic_
    velocity`` state :func:`ocean_skill.align._point_window` and
    :func:`roms.add_geographic_velocity_windowed` both operate on. A uniform angle
    (the rest of this file's fixture) would not distinguish a correct per-window
    rotation from one that accidentally used a neighboring cell's angle.
    """
    rng = np.random.default_rng(1)
    lon = np.tile(np.linspace(-90.0, -88.0, nx), (ny, 1))
    lat = np.tile(np.linspace(20.0, 22.0, ny)[:, None], (1, nx))
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    angle = 0.15 * i - 0.08 * j  # a ramp, radians -- different at every rho point
    mask_rho = np.ones((ny, nx))
    if land_at is not None:
        mask_rho[land_at] = 0.0
    ds = xr.Dataset(
        {
            "sea_water_x_velocity": (
                ("s_rho", "eta_rho", "xi_u"),
                rng.normal(0.1, 0.02, (N, ny, nx - 1)),
            ),
            "sea_water_y_velocity": (
                ("s_rho", "eta_v", "xi_rho"),
                rng.normal(0.05, 0.02, (N, ny - 1, nx)),
            ),
        },
        coords={
            "lon": (("eta_rho", "xi_rho"), lon),
            "lat": (("eta_rho", "xi_rho"), lat),
            "angle": (("eta_rho", "xi_rho"), angle),
            "mask_rho": (("eta_rho", "xi_rho"), mask_rho),
        },
    )
    meta = {"self_contained_grid": True, "vertical": {"hc": 50.0, "s_dim": "s_rho"}}
    return ds, meta


@pytest.mark.parametrize("iy,ix,cells", [(4, 4, 1), (0, 4, 1), (7, 4, 1), (4, 0, 1), (4, 8, 1), (4, 4, 3)])
def test_windowed_derive_matches_full_domain_derive_then_crop(iy, ix, cells):
    """Every window position -- interior, and each of the four edges -- agrees
    exactly with cropping a full-domain derive's own output to the same rho window.
    """
    from ocean_skill.align import _point_window

    ds, meta = _rotated_grid(land_at=(2, 3))

    # Path A: derive over the whole domain first, then take the same rho window
    # _point_window would (no halo needed -- the full-domain result already has
    # every value the crop below might want). _add_geographic_velocity itself does
    # not mask land (that is standardize's own separate loop) -- apply the same
    # mask_rho==1 rule here, matching what add_geographic_velocity_windowed does
    # for itself below, so both paths are compared on equal footing.
    full = roms._add_geographic_velocity(ds)
    land_mask = full["mask_rho"] == 1
    for name in ("eastward_sea_water_velocity", "northward_sea_water_velocity"):
        full[name] = full[name].where(land_mask)
    eta0, eta1 = max(iy - cells, 0), iy + cells + 1
    xi0, xi1 = max(ix - cells, 0), ix + cells + 1
    expected = full.isel(eta_rho=slice(eta0, eta1), xi_rho=slice(xi0, xi1))

    # Path B: crop first (with _point_window's own halo), re-derive on the window.
    lon = float(ds["lon"].values[iy, ix])
    lat = float(ds["lat"].values[iy, ix])
    cropped = _point_window(ds, "lon", "lat", lon, lat, cells=cells)
    assert "_roms_stagger_trim" in cropped.attrs  # the halo/trim machinery engaged
    windowed = roms.add_geographic_velocity_windowed(cropped, meta)

    for name in ("eastward_sea_water_velocity", "northward_sea_water_velocity"):
        np.testing.assert_allclose(
            windowed[name].values, expected[name].values, equal_nan=True
        )
    # And the window really did land on (iy, ix) as its center, as a sanity check
    # the two paths are comparing the same rho points at all.
    assert windowed.sizes["eta_rho"] == expected.sizes["eta_rho"]
    assert windowed.sizes["xi_rho"] == expected.sizes["xi_rho"]


def test_windowed_derive_whole_domain_matches_full_derive():
    """cells large enough to cover the whole grid: no halo left to trim at all."""
    from ocean_skill.align import _point_window

    ds, meta = _rotated_grid(ny=5, nx=6)
    full = roms._add_geographic_velocity(ds)
    cropped = _point_window(ds, "lon", "lat", float(ds["lon"][2, 2]), float(ds["lat"][2, 2]), cells=10)
    windowed = roms.add_geographic_velocity_windowed(cropped, meta)
    np.testing.assert_allclose(
        windowed["eastward_sea_water_velocity"].values,
        full["eastward_sea_water_velocity"].values,
    )
    np.testing.assert_allclose(
        windowed["northward_sea_water_velocity"].values,
        full["northward_sea_water_velocity"].values,
    )


def test_windowed_derive_only_ever_averages_the_halo_window():
    """The re-derive's own averaging step must see only the halo-cropped input,
    never the full-domain staggered arrays -- the whole mechanism this fix relies
    on to keep the graph bounded by the window rather than the domain. A raw task
    count or wall-clock comparison is not used here: at real (thousands of
    timesteps, hundreds of grid cells) scale the difference is dramatic (measured
    separately: a multi-minute full-domain compute vs sub-second on the window),
    but at a synthetic unit-test scale dask's own per-task overhead dominates and
    swamps the signal either way -- this instead checks the mechanism directly and
    deterministically: is the array _average_to_rho actually receives ever as
    large as the un-cropped domain?
    """
    from ocean_skill.align import _point_window

    ny, nx = 40, 41
    ds, meta = _rotated_grid(ny=ny, nx=nx)
    seen_shapes = []
    real_average = roms._average_to_rho

    def spy(da, stagger_dim, rho_dim):
        seen_shapes.append(da.shape)
        return real_average(da, stagger_dim, rho_dim)

    cropped = _point_window(
        ds, "lon", "lat", float(ds["lon"][20, 20]), float(ds["lat"][20, 20]), cells=1
    )
    # The crop alone already kept the staggered dims tiny -- nowhere near ny/nx.
    assert cropped.sizes["xi_u"] <= 5
    assert cropped.sizes["eta_v"] <= 5

    import unittest.mock as mock

    with mock.patch.object(roms, "_average_to_rho", side_effect=spy):
        roms.add_geographic_velocity_windowed(cropped, meta)

    assert seen_shapes, "the averaging step never ran"
    for shape in seen_shapes:
        # Every dim _average_to_rho actually saw is bounded by the halo window
        # (a handful of points), never anywhere close to the full ny/nx domain.
        assert all(d <= 5 for d in shape[-2:]), (
            f"averaged a shape touching the full domain: {shape} (ny={ny}, nx={nx})"
        )


def test_add_geographic_velocity_windowed_is_a_noop_without_velocity():
    """Mirrors _add_geographic_velocity's own no-op guard."""
    ds = xr.Dataset(
        {"h": (("eta_rho", "xi_rho"), np.full((3, 4), 100.0))},
        coords={"angle": (("eta_rho", "xi_rho"), np.zeros((3, 4)))},
    )
    out = roms.add_geographic_velocity_windowed(ds, {})
    assert out.identical(ds)


# -- end-to-end: standardize + to_depth (the direct regression check) ---------------


def test_standardize_masks_derived_velocity_on_land():
    ds, meta = _roms_like(land_at=(1, 2), raw_names=True)
    out = roms.standardize(ds, meta)
    east = out["eastward_sea_water_velocity"]
    assert np.isnan(east.isel(eta_rho=1, xi_rho=2).values).all()
    assert np.isfinite(east.isel(eta_rho=0, xi_rho=0).values).all()


def test_standardize_skips_geographic_velocity_without_v():
    """Only u present (a build that never wired v): no derived velocity, no crash."""
    ds, meta = _roms_like(with_v=False, raw_names=True)
    out = roms.standardize(ds, meta)
    assert "eastward_sea_water_velocity" not in out
    assert "sea_water_x_velocity" in out


def test_to_depth_includes_derived_velocity_but_still_skips_staggered_components():
    """The exact regression: compare()'s ROMS lane used to KeyError on this lookup.

    ``eastward_sea_water_velocity`` is now on rho dims and survives ``to_depth``;
    the raw staggered ``sea_water_x_velocity``/``sea_water_y_velocity`` are still
    deferred (unchanged from before this fix -- to_depth's own guard, not this one,
    is responsible for that).
    """
    ds, meta = _roms_like(angle=np.pi / 6, raw_names=True)
    standardized = roms.standardize(ds, meta)

    at_depth = roms.to_depth(standardized, meta, 50.0)

    assert "eastward_sea_water_velocity" in at_depth
    assert "northward_sea_water_velocity" in at_depth
    # to_depth keeps a size-1 "z" axis for a scalar depth (comparison.py's caller
    # squeezes it; this test calls to_depth directly, as compare()'s ROMS lane does
    # before that squeeze).
    east = at_depth["eastward_sea_water_velocity"]
    assert set(east.dims) == {"eta_rho", "xi_rho", "z"}
    assert east.sizes["z"] == 1
    assert np.isfinite(east.isel(z=0)).any()
    assert "sea_water_x_velocity" not in at_depth
    assert "sea_water_y_velocity" not in at_depth


# -- end-to-end: a grid constant requested directly (_prepare's own regression) -----


def test_prepare_draws_a_grid_constant_standardize_promoted_to_a_coordinate():
    """``osk.field(src, "h").plot()`` used to raise ``ValueError: cannot create a
    Dataset from a DataArray with the same name as one of its coordinates``.

    ``roms.standardize`` deliberately promotes ``h`` (and ``Cs_r``/``sigma_r``/
    ``mask_rho``/``angle``) out of ``data_vars`` into ``ds.coords`` -- grid
    geometry, not a comparable field (see its own comment). Resolving ``h`` by
    name then returns ``ds["h"]``, a DataArray that -- by ordinary xarray
    semantics for any coordinate pulled out of its own Dataset -- carries
    itself among its own ``.coords``. ``_prepare``'s ROMS-surface branch used
    to hand that self-referential DataArray straight to ``.to_dataset(name=...)``,
    which xarray refuses outright for exactly this shape.
    """
    from ocean_skill.comparison import _prepare

    ds, meta = _roms_like(raw_names=True)
    meta = {**meta, "model": "roms"}
    standardized = roms.standardize(ds, meta)
    assert "h" in standardized.coords and "h" not in standardized.data_vars

    da, depth = _prepare(standardized, meta, "h", {"depth": "surface"})

    assert set(da.dims) == {"eta_rho", "xi_rho"}
    np.testing.assert_allclose(da.values, 100.0)  # _roms_like's own constant h


def test_prepare_still_reduces_an_ordinary_field_on_the_same_dataset():
    """The unaffected case still works after the fix -- a real field with an
    actual ``s_rho`` level to drop, on the same standardized Dataset.
    """
    from ocean_skill.comparison import _prepare

    ds, meta = _roms_like(raw_names=True)
    meta = {**meta, "model": "roms"}
    standardized = roms.standardize(ds, meta)

    da, depth = _prepare(
        standardized, meta, "sea_water_x_velocity", {"depth": "surface"}
    )
    assert "s_rho" not in da.dims


# -- end-to-end: prepare_source's ROMS point-lane fast path --------------------------


def _standardized_rotated_grid(ny=12, nx=13):
    """A Dataset in the post-``standardize`` state (pre-derived east/north
    included, dask-backed) -- what :func:`ocean_skill.sources.read` hands
    ``prepare_source`` for a real ROMS source. Built directly from
    :func:`_rotated_grid` (already in the ``lon``/``lat``/``angle`` coord-attached,
    post-rename shape ``standardize`` produces) rather than by calling
    ``standardize`` itself, which expects raw pre-rename input
    (``lon_rho``/``h``/``mask_rho`` as a separate grid) that is not what this
    file's ``_rotated_grid`` fixture builds.
    """
    pytest.importorskip("dask")
    import dask.array as da_

    ds, meta = _rotated_grid(ny=ny, nx=nx)
    meta = {**meta, "model": "roms", "standard_names": {}}
    ds = ds.assign(
        sea_water_x_velocity=(
            ds["sea_water_x_velocity"].dims,
            da_.from_array(ds["sea_water_x_velocity"].values, chunks=(N, 4, 4)),
        ),
        sea_water_y_velocity=(
            ds["sea_water_y_velocity"].dims,
            da_.from_array(ds["sea_water_y_velocity"].values, chunks=(N, 4, 4)),
        ),
    )
    standardized = roms._add_geographic_velocity(ds)
    mask = standardized["mask_rho"] == 1
    for var in list(standardized.data_vars):
        da = standardized[var]
        if {"eta_rho", "xi_rho"} <= set(da.dims):
            standardized[var] = da.where(mask)
    standardized.attrs["ocean_skill_model"] = "roms"
    return standardized, meta


def test_prepare_source_takes_the_windowed_path_for_a_roms_velocity_point(monkeypatch):
    """``prepare_source`` for a ROMS point request drops the pre-derived east/north,
    crops the raw components with a halo, and re-derives on the window -- rather
    than loading the full-domain pair standardize() already built. Values must
    still match what the full-domain pair would have given at that same point.
    """
    from types import SimpleNamespace

    import ocean_skill as osk
    from ocean_skill import catalog
    from ocean_skill.comparison import prepare_source

    standardized, meta = _standardized_rotated_grid()
    iy, ix = 6, 6
    lon = float(standardized["lon"].values[iy, ix])
    lat = float(standardized["lat"].values[iy, ix])

    monkeypatch.setattr(osk, "read", lambda name, **kw: standardized)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata=meta))

    calls = []
    real_windowed = roms.add_geographic_velocity_windowed

    def spy(ds, meta_):
        calls.append(ds)
        return real_windowed(ds, meta_)

    monkeypatch.setattr(roms, "add_geographic_velocity_windowed", spy)

    da, _ = prepare_source(
        "his",
        "eastward_sea_water_velocity",
        {"depth": "surface"},
        None,
        use_cache=False,
        bbox=(lon, lat, lon, lat),
    )

    assert len(calls) == 1, "the windowed re-derive path did not engage"
    # And the pre-derived full-domain pair was actually dropped before the crop --
    # the Dataset the re-derive received had to build east/north itself.
    assert "eastward_sea_water_velocity" not in calls[0].variables

    # Correctness: matches the full-domain pair's own value over the same window
    # (the default POINT_WINDOW_CELLS crop, at the surface -- roms.surface's own
    # topmost-s_rho-level rule, same as _prepare applies to `da` above).
    from ocean_skill.align import POINT_WINDOW_CELLS, _point_window

    expected = _point_window(standardized, "lon", "lat", lon, lat, cells=POINT_WINDOW_CELLS)
    expected = expected["eastward_sea_water_velocity"].isel(s_rho=-1)
    xr.testing.assert_allclose(da.squeeze(drop=True), expected.squeeze(drop=True))


def test_prepare_source_temperature_point_does_not_take_the_velocity_path(monkeypatch):
    """A non-velocity point request on the same ROMS source must not touch the
    windowed re-derive machinery at all -- the gate is scoped to velocity only.
    """
    from types import SimpleNamespace

    import ocean_skill as osk
    from ocean_skill import catalog
    from ocean_skill.comparison import prepare_source

    standardized, meta = _standardized_rotated_grid()
    standardized = standardized.assign(
        sea_water_potential_temperature=(
            ("s_rho", "eta_rho", "xi_rho"),
            np.full((N, 12, 13), 15.0),
        )
    )
    meta = {**meta, "standard_names": {}}
    iy, ix = 6, 6
    lon = float(standardized["lon"].values[iy, ix])
    lat = float(standardized["lat"].values[iy, ix])

    monkeypatch.setattr(osk, "read", lambda name, **kw: standardized)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata=meta))
    calls = []
    monkeypatch.setattr(
        roms,
        "add_geographic_velocity_windowed",
        lambda ds, meta_: calls.append(ds) or ds,
    )

    da, _ = prepare_source(
        "his",
        "sea_water_potential_temperature",
        {"depth": "surface"},
        None,
        use_cache=False,
        bbox=(lon, lat, lon, lat),
    )
    assert not calls, "the velocity fast path engaged for a non-velocity variable"
    assert da is not None


# -- derived_geographic_velocities: the shared "what standardize adds" fact --------


def test_derived_geographic_velocities_needs_both_grid_relative_components():
    both = {"sea_water_x_velocity", "sea_water_y_velocity", "salt"}
    assert roms.derived_geographic_velocities(both) == [
        "eastward_sea_water_velocity",
        "northward_sea_water_velocity",
    ]


@pytest.mark.parametrize(
    "present",
    [
        {"sea_water_x_velocity", "salt"},
        {"sea_water_y_velocity", "salt"},
        {"salt"},
        set(),
    ],
)
def test_derived_geographic_velocities_needs_both_not_either(present):
    assert roms.derived_geographic_velocities(present) == []


# -- catalog-time advertisement: ocean_skill.build._probe --------------------------


def _roms_probe_dataset(with_v=True):
    """Build a minimal raw ROMS Dataset shaped for ocean_skill.build._probe.

    Not run through :func:`roms.standardize` -- ``Cs_r``/``sigma_r`` are what
    ``_roms_metadata`` keys off to set ``model: "roms"``; ``u``/``v`` are the
    literal ROMS names ``_probe`` renames via ``ROMS_STANDARD_NAMES`` before
    ``derived_geographic_velocities`` ever sees them.
    """
    ny, nx = 3, 4
    rng = np.random.default_rng(0)
    data = {
        "u": (("s_rho", "eta_rho", "xi_u"), rng.random((N, ny, nx - 1))),
        "temp": (("s_rho", "eta_rho", "xi_rho"), rng.random((N, ny, nx))),
        "Cs_r": ("s_rho", np.linspace(-1.0, 0.0, N)),
        "sigma_r": ("s_rho", np.linspace(-1.0, 0.0, N)),
    }
    if with_v:
        data["v"] = (
            ("s_rho", "eta_v", "xi_rho"),
            rng.random((N, ny - 1, nx)),
        )
    return xr.Dataset(data)


def test_probe_advertises_geographic_velocity_for_a_roms_source_with_both_components():
    from ocean_skill.build import ROMS_STANDARD_NAMES, _probe

    md = _probe(_roms_probe_dataset(with_v=True), ROMS_STANDARD_NAMES)
    assert md.get("model") == "roms"
    assert {
        "sea_water_x_velocity",
        "sea_water_y_velocity",
        "eastward_sea_water_velocity",
        "northward_sea_water_velocity",
    } <= set(md["variables"])


def test_probe_does_not_advertise_geographic_velocity_without_both_components():
    from ocean_skill.build import ROMS_STANDARD_NAMES, _probe

    md = _probe(_roms_probe_dataset(with_v=False), ROMS_STANDARD_NAMES)
    assert md.get("model") == "roms"
    assert "sea_water_x_velocity" in md["variables"]
    assert "eastward_sea_water_velocity" not in md["variables"]
    assert "northward_sea_water_velocity" not in md["variables"]


def test_probe_does_not_advertise_geographic_velocity_for_a_non_roms_source():
    """A plain gridded source that happens to declare u/v-shaped names is untouched.

    Guards the ROMS-only gate: without ``Cs_r``/``sigma_r`` (ROMS' own tell),
    ``_roms_metadata`` sets no ``model`` key, so the advertisement must not fire.
    """
    from ocean_skill.build import ROMS_STANDARD_NAMES, _probe

    ds = xr.Dataset(
        {
            "sea_water_x_velocity": (("lat", "lon"), np.ones((2, 2))),
            "sea_water_y_velocity": (("lat", "lon"), np.ones((2, 2))),
        },
        coords={"lat": [10.0, 11.0], "lon": [200.0, 201.0]},
    )
    md = _probe(ds, ROMS_STANDARD_NAMES)
    assert md.get("model") is None
    assert "eastward_sea_water_velocity" not in md.get("variables", [])
