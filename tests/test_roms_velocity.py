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
