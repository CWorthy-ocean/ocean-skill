"""Tests for classic Rutgers ROMS layout: time dim, sigma layout, Vtransform 1 or 2.

UCLA-ROMS (what ocean_skill.roms grew up on) puts ``sigma_r``/``sigma_w`` in data
variables over bare ``s_rho``/``s_w`` dims, ``hc`` in the catalog, the ``ocean_time``
*variable* on a dim called ``time``, and uses Vtransform 2. Classic Rutgers output
(xroms' example file) differs on every count: sigma values are the 1-D dimension
coordinates ``s_rho``/``s_w``, ``hc`` and ``Vtransform`` are 0-d data variables,
``ocean_time`` is itself the time *dimension*, and Vtransform is 1. These tests build a
small synthetic classic-layout dataset (no dependence on the real file) and check each
difference is bridged, against depth formulas written out independently from the raw
arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms

NS, NY, NX, NT = 4, 3, 5, 2
HC = 5.0
THETA_S, THETA_B = 5.0, 2.0
SECONDS = np.array([1.2586320e9, 1.2586464e9])  # 2009-11-19T12:00 and T16:00

META = {
    "model": "roms",
    "loader": "ocean_skill.roms",
    "self_contained_grid": True,
    "vertical": {"s_dim": "s_rho", "hc": HC, "Vtransform": 1},
    "time_coord": "ocean_time",
    "time_dim": "time",
    "time_units": "seconds",
    "reference_date": "1970-01-01",
}


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


def _meta(**vertical):
    """META with its ``vertical`` block replaced (an empty call -> an empty block)."""
    return {**META, "vertical": vertical}


def _classic(*, vtransform=1, with_hc=True, with_vtransform=True):
    """Build a small synthetic classic-Rutgers ROMS history dataset (raw output)."""
    rng = np.random.default_rng(0)
    s_rho = (np.arange(1, NS + 1) - NS - 0.5) / NS
    s_w = np.linspace(-1, 0, NS + 1)
    h = np.linspace(5.0, 200.0, NY * NX).reshape(NY, NX)
    mask = np.ones((NY, NX))
    mask[0, 0] = 0.0  # land
    mask[2, 4] = 0.0
    data_vars = {
        "Cs_r": ("s_rho", _stretch(s_rho)),
        "Cs_w": ("s_w", _stretch(s_w)),
        "h": (("eta_rho", "xi_rho"), h),
        "mask_rho": (("eta_rho", "xi_rho"), mask),
        "pm": (("eta_rho", "xi_rho"), np.full((NY, NX), 1 / 500.0)),
        "pn": (("eta_rho", "xi_rho"), np.full((NY, NX), 1 / 400.0)),
        "angle": (("eta_rho", "xi_rho"), np.zeros((NY, NX))),
        "zeta": (
            ("ocean_time", "eta_rho", "xi_rho"),
            rng.uniform(-0.5, 0.5, (NT, NY, NX)),
        ),
        "temp": (
            ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
            rng.uniform(5, 15, (NT, NS, NY, NX)),
        ),
        "salt": (
            ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
            rng.uniform(30, 35, (NT, NS, NY, NX)),
        ),
    }
    if with_hc:
        data_vars["hc"] = ((), HC)
    if with_vtransform:
        data_vars["Vtransform"] = ((), np.int32(vtransform))
    return xr.Dataset(
        data_vars,
        coords={
            "s_rho": ("s_rho", s_rho, {"long_name": "S-coordinate at RHO-points"}),
            "s_w": ("s_w", s_w, {"long_name": "S-coordinate at W-points"}),
            "ocean_time": (
                "ocean_time",
                SECONDS,
                {"long_name": "time since initialization"},
            ),
            "lon_rho": (
                ("eta_rho", "xi_rho"),
                np.linspace(-95, -90, NX) * np.ones((NY, 1)),
            ),
            "lat_rho": (
                ("eta_rho", "xi_rho"),
                np.linspace(25, 27, NY)[:, None] * np.ones((1, NX)),
            ),
        },
    )


def _ucla():
    """Build a minimal UCLA-layout dataset: ``sigma_*`` data variables, bare dims."""
    s_rho = (np.arange(1, NS + 1) - NS - 0.5) / NS
    s_w = np.linspace(-1, 0, NS + 1)
    return xr.Dataset(
        {
            "sigma_r": ("s_rho", s_rho),
            "Cs_r": ("s_rho", _stretch(s_rho)),
            "sigma_w": ("s_w", s_w),
            "Cs_w": ("s_w", _stretch(s_w)),
            "h": (("eta_rho", "xi_rho"), np.full((NY, NX), 100.0)),
        }
    )


def _z_v1(sigma, Cs, h, zeta):
    """Vtransform 1, written out from the raw numpy arrays (independent of roms.py)."""
    z0 = HC * (sigma - Cs) + Cs * h
    return z0 + zeta * (1 + z0 / h)


def _z_v2(sigma, Cs, h, zeta):
    """Vtransform 2, written out from the raw numpy arrays (independent of roms.py)."""
    s = (HC * sigma + h * Cs) / (HC + h)
    return zeta + (zeta + h) * s


def _expected(raw, zfunc, *, sigma_name, cs_name):
    """Ground-truth depths as a (time, s, eta, xi) numpy array, from the raw arrays."""
    sigma = raw[sigma_name].values[None, :, None, None]
    Cs = raw[cs_name].values[None, :, None, None]
    h = raw["h"].values[None, None]
    zeta = raw["zeta"].values[:, None]
    return zfunc(sigma, Cs, h, zeta)


def _ocean(raw):
    return raw["mask_rho"].values == 1


# -- time -----------------------------------------------------------------------


def test_classic_ocean_time_dim_becomes_the_time_dim():
    """The data must end up on ``time``, not stay on ``ocean_time`` beside a new dim."""
    ds = roms.standardize(_classic(), META)
    assert "time" in ds.dims
    assert "ocean_time" not in ds.dims
    assert ds["temp"].dims[0] == "time"
    assert ds["zeta"].dims[0] == "time"
    assert ds["time"].size == NT


def test_classic_time_values_are_decoded_from_seconds_since_the_reference_date():
    """Non-CF 'seconds since 1970-01-01' decodes to the file's real timestamps."""
    ds = roms.standardize(_classic(), META)
    np.testing.assert_array_equal(
        ds["time"].values,
        np.array(["2009-11-19T12:00", "2009-11-19T16:00"], dtype="datetime64[ns]"),
    )
    assert ds["time"].values[0] == np.datetime64("2009-11-19T12:00")
    assert ds["time"].values[1] == np.datetime64("2009-11-19T16:00")


def test_classic_time_can_be_selected_by_label_after_standardize():
    """``.sel(time=...)`` only works if ``time`` is an index on the data's own dim."""
    raw = _classic()
    ds = roms.standardize(raw, META)
    one = ds["temp"].sel(time=np.datetime64("2009-11-19T16:00"))
    assert "time" not in one.dims
    ocean = np.broadcast_to(_ocean(raw), one.shape)
    np.testing.assert_allclose(
        one.values[ocean], raw["temp"].isel(ocean_time=1).values[ocean]
    )


def test_classic_ocean_time_variable_rides_on_the_time_dim_as_a_plain_coord():
    """Like UCLA, ``ocean_time`` survives as a non-index coord on ``time``."""
    ds = roms.standardize(_classic(), META)
    assert ds["ocean_time"].dims == ("time",)
    assert "ocean_time" not in ds.indexes
    np.testing.assert_array_equal(ds["ocean_time"].values, SECONDS)


# -- sigma layout ---------------------------------------------------------------


def test_classic_s_rho_and_s_w_become_bare_dims_with_sigma_variables():
    """A valued ``s_rho`` would make plotting use sigma instead of ``z_rho``."""
    raw = _classic()
    ds = roms.standardize(raw, META)
    assert "s_rho" in ds.dims
    assert "s_rho" not in ds.variables
    assert "s_w" in ds.dims
    assert "s_w" not in ds.variables
    np.testing.assert_array_equal(ds["sigma_r"].values, raw["s_rho"].values)
    assert ds["sigma_r"].dims == ("s_rho",)
    assert ds["sigma_r"].attrs["long_name"] == "S-coordinate at RHO-points"
    np.testing.assert_array_equal(ds["sigma_w"].values, raw["s_w"].values)
    assert ds["sigma_w"].dims == ("s_w",)


def test_normalize_classic_layout_is_a_noop_on_a_ucla_style_dataset():
    """UCLA already has ``sigma_*`` over bare dims, so nothing must change."""
    ucla = _ucla()
    xr.testing.assert_identical(roms._normalize_classic_layout(ucla), ucla)


def test_normalize_classic_layout_leaves_an_existing_sigma_alone():
    """If ``sigma_r`` exists, a valued ``s_rho`` label is not second-guessed."""
    ucla = _ucla().assign_coords(s_rho=np.arange(NS, dtype=float))
    out = roms._normalize_classic_layout(ucla)
    xr.testing.assert_identical(out, ucla)


# -- Vtransform 1 depths ---------------------------------------------------------


def test_classic_z_rho_matches_the_vtransform_1_formula_with_zeta():
    """Independent ground truth: z0 = hc*(s - Cs) + Cs*h; z = z0 + zeta*(1 + z0/h)."""
    raw = _classic()
    ds = roms.standardize(raw, META)
    want = _expected(raw, _z_v1, sigma_name="s_rho", cs_name="Cs_r")
    got = ds["z_rho"].transpose("time", "s_rho", "eta_rho", "xi_rho").values
    ocean = np.broadcast_to(_ocean(raw), got.shape)
    np.testing.assert_allclose(got[ocean], want[ocean])
    # land is masked (zeta is NaN there), exactly as for UCLA
    assert np.isnan(got[~ocean]).all()


def test_classic_z_rho_has_the_standard_dim_order():
    """The ordering contract downstream code relies on, unchanged by the new path."""
    ds = roms.standardize(_classic(), META)
    assert ds["z_rho"].dims == ("time", "s_rho", "eta_rho", "xi_rho")


def test_classic_zero_zeta_z_rho_matches_the_vtransform_1_formula_with_zeta_0():
    """``zero_zeta=True`` gives the finite-everywhere mesh: land included."""
    raw = _classic()
    ds = roms.standardize(raw, META)
    z = roms.add_depth_coord(ds, META, zero_zeta=True)["z_rho"]
    sigma = raw["s_rho"].values[:, None, None]
    Cs = raw["Cs_r"].values[:, None, None]
    want = _z_v1(sigma, Cs, raw["h"].values[None], 0.0)
    np.testing.assert_allclose(z.transpose("s_rho", "eta_rho", "xi_rho").values, want)
    assert np.isfinite(z.values).all()


def test_classic_z_w_matches_the_vtransform_1_formula_on_the_interface_levels():
    """``add_interface_coord`` uses ``sigma_w``/``Cs_w`` with the same transform."""
    raw = _classic()
    ds = roms.standardize(raw, META)
    got = roms.add_interface_coord(ds, META)["z_w"]
    want = _expected(raw, _z_v1, sigma_name="s_w", cs_name="Cs_w")
    got = got.transpose("time", "s_w", "eta_rho", "xi_rho").values
    ocean = np.broadcast_to(_ocean(raw), got.shape)
    np.testing.assert_allclose(got[ocean], want[ocean])


def test_classic_vtransform_1_interfaces_span_the_bottom_to_the_free_surface():
    """Sanity on the physics: z_w runs from -h at s=-1 to zeta at s=0."""
    raw = _classic()
    ds = roms.standardize(raw, META)
    z_w = roms.add_interface_coord(ds, META)["z_w"]
    ocean = _ocean(raw)
    bottom = z_w.isel(s_w=0, time=0).values[ocean]
    top = z_w.isel(s_w=-1, time=0).values[ocean]
    np.testing.assert_allclose(bottom, -raw["h"].values[ocean])
    np.testing.assert_allclose(top, raw["zeta"].values[0][ocean])


# -- Vtransform 2 and the resolution order ----------------------------------------


def test_classic_layout_with_vtransform_2_in_meta_matches_the_vtransform_2_formula():
    """The catalog's Vtransform wins; the layout and the transform are independent."""
    raw = _classic(vtransform=1)  # the file's own variable says 1 ...
    meta = _meta(s_dim="s_rho", hc=HC, Vtransform=2)  # ... the catalog says 2
    ds = roms.standardize(raw, meta)
    want = _expected(raw, _z_v2, sigma_name="s_rho", cs_name="Cs_r")
    got = ds["z_rho"].transpose("time", "s_rho", "eta_rho", "xi_rho").values
    ocean = np.broadcast_to(_ocean(raw), got.shape)
    np.testing.assert_allclose(got[ocean], want[ocean])


def test_vtransform_falls_back_to_the_files_own_variable_when_meta_lacks_it():
    """No Vtransform in the catalog + a ``Vtransform`` variable of 2 -> Vtransform 2."""
    raw = _classic(vtransform=2)
    meta = _meta(s_dim="s_rho", hc=HC)
    ds = roms.standardize(raw, meta)
    want = _expected(raw, _z_v2, sigma_name="s_rho", cs_name="Cs_r")
    got = ds["z_rho"].transpose("time", "s_rho", "eta_rho", "xi_rho").values
    ocean = np.broadcast_to(_ocean(raw), got.shape)
    np.testing.assert_allclose(got[ocean], want[ocean])


def test_vtransform_file_variable_of_1_is_used_when_meta_lacks_it():
    """The fallback is the file's value, not a hardcoded 2."""
    raw = _classic(vtransform=1)
    meta = _meta(s_dim="s_rho", hc=HC)
    ds = roms.standardize(raw, meta)
    want = _expected(raw, _z_v1, sigma_name="s_rho", cs_name="Cs_r")
    got = ds["z_rho"].transpose("time", "s_rho", "eta_rho", "xi_rho").values
    ocean = np.broadcast_to(_ocean(raw), got.shape)
    np.testing.assert_allclose(got[ocean], want[ocean])


def test_vtransform_defaults_to_2_when_neither_meta_nor_the_file_names_it():
    """The UCLA/roms-tools convention is the default, as before this change."""
    raw = _classic(with_vtransform=False)
    meta = _meta(s_dim="s_rho", hc=HC)
    ds = roms.standardize(raw, meta)
    want = _expected(raw, _z_v2, sigma_name="s_rho", cs_name="Cs_r")
    got = ds["z_rho"].transpose("time", "s_rho", "eta_rho", "xi_rho").values
    ocean = np.broadcast_to(_ocean(raw), got.shape)
    np.testing.assert_allclose(got[ocean], want[ocean])


def test_hc_falls_back_to_the_files_own_variable_when_meta_lacks_it():
    """Classic output carries ``hc`` as a 0-d variable; no catalog entry is needed."""
    raw = _classic()
    meta = _meta(s_dim="s_rho", Vtransform=1)
    ds = roms.standardize(raw, meta)
    want = _expected(raw, _z_v1, sigma_name="s_rho", cs_name="Cs_r")
    got = ds["z_rho"].transpose("time", "s_rho", "eta_rho", "xi_rho").values
    ocean = np.broadcast_to(_ocean(raw), got.shape)
    np.testing.assert_allclose(got[ocean], want[ocean])


def test_ucla_style_depth_is_unchanged_bit_for_bit_from_the_old_inline_formula():
    """The ``_s_to_z`` refactor must not move a single UCLA (Vtransform 2) digit."""
    ds = _ucla().assign(
        zeta=(
            ("eta_rho", "xi_rho"),
            np.random.default_rng(1).uniform(-0.5, 0.5, (NY, NX)),
        )
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": 250.0}}
    got = roms.add_depth_coord(ds, meta)["z_rho"]
    s = (250.0 * ds["sigma_r"] + ds["h"] * ds["Cs_r"]) / (250.0 + ds["h"])
    old = ds["zeta"] + (ds["zeta"] + ds["h"]) * s
    np.testing.assert_array_equal(
        got.transpose("s_rho", "eta_rho", "xi_rho").values,
        old.transpose("s_rho", "eta_rho", "xi_rho").values,
    )


# -- errors ----------------------------------------------------------------------


def test_unsupported_vtransform_raises_a_clear_value_error():
    """Only the two ROMS transforms exist; 3 is a mistake, not a silent fallback."""
    raw = _classic()
    meta = _meta(s_dim="s_rho", hc=HC, Vtransform=3)
    with pytest.raises(ValueError, match="Unsupported ROMS Vtransform 3"):
        roms.standardize(raw, meta)


def test_unsupported_vtransform_in_the_file_variable_also_raises():
    """The validation applies wherever the value came from."""
    raw = _classic(vtransform=3)
    meta = _meta(s_dim="s_rho", hc=HC)
    with pytest.raises(ValueError, match="Unsupported ROMS Vtransform 3"):
        roms.standardize(raw, meta)


def test_missing_hc_raises_value_error_not_an_opaque_type_error():
    """Used to be ``TypeError: float(None)``; now names what is missing and where."""
    raw = _classic(with_hc=False)
    meta = _meta(s_dim="s_rho", Vtransform=1)
    with pytest.raises(ValueError, match="hc"):
        roms.standardize(raw, meta)
