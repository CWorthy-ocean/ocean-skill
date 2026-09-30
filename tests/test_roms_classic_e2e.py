"""End to end: a classic Rutgers ROMS file, from catalog build to mixed layer depth.

The unit tests next door (``test_roms_classic_detect.py`` and
``test_roms_classic_depth.py``) check each piece of classic-layout support on its
own: ``build._roms_metadata`` spots the file, ``roms.standardize`` reshapes it,
``add_depth_coord`` applies Vtransform 1.

These tests run the pieces *together*, through the public path a user takes, because
the failure that matters here is the silent one: a classic file that is catalogued but
never routed through :mod:`ocean_skill.roms` still opens and plots, just with no land
mask, no depth coordinate, and a sigma axis mistaken for metres.

Two kinds of ground truth:

* a synthetic file with a *known* stratification, written to disk, catalogued with
  :func:`ocean_skill.build.build_catalog`, read back with :func:`ocean_skill.read` and
  pushed through the MLD calculator -- the expected depths come from the profile the
  file was built with, not from the code under test;
* the real xroms example file, whose depths ``xroms`` computes independently (skipped
  unless it is already in pooch's cache; nothing here ever downloads).
"""

from __future__ import annotations

import intake
import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import build, roms
from ocean_skill.operators import resolve_variable

NS, NY, NX, NT = 60, 3, 4, 2
HC, THETA_S, THETA_B = 5.0, 5.0, 0.4
SECONDS = np.array([1.2586320e9, 1.2586464e9])  # 2009-11-19T12:00 and T16:00
LAND = [(0, 0), (2, 3)]  # (eta, xi) cells with mask_rho == 0

# The constructed stratification: uniform TEMP_TOP down to MIXED_LAYER metres, then
# cooling at LAPSE degC per metre. The 0.2 degC / 10 m criterion therefore trips
# (0.2 / LAPSE) metres below the mixed layer's base, exactly, wherever a model level
# sits inside that stretch (the fixture checks that they do).
TEMP_TOP, MIXED_LAYER, LAPSE, SALINITY = 10.0, 30.0, 0.025, 35.0
EXPECTED_TEMPERATURE_MLD = MIXED_LAYER + 0.2 / LAPSE  # 38 m

ENTRY = "classic_roms:classic"


def _stretch(s):
    """Return a Shchepetkin-style stretching curve: -1 at the bottom, 0 at the top."""
    surface = np.sinh(THETA_S * s) / np.sinh(THETA_S)
    bottom = (np.tanh(THETA_S * (s + 0.5)) - np.tanh(0.5 * THETA_S)) / (
        2 * np.tanh(0.5 * THETA_S)
    )
    return (1 - THETA_B) * surface + THETA_B * bottom


def _z_v1(sigma, Cs, h, zeta):
    """Vtransform 1 depths, written out from raw numpy arrays (not via roms.py)."""
    z0 = HC * (sigma - Cs) + Cs * h
    return z0 + zeta * (1 + z0 / h)


def _depths(raw):
    """Ground-truth z_rho, (time, s_rho, eta, xi), from the raw arrays only."""
    return _z_v1(
        raw["s_rho"].values[None, :, None, None],
        raw["Cs_r"].values[None, :, None, None],
        raw["h"].values[None, None],
        raw["zeta"].values[:, None],
    )


def _ocean(raw):
    return raw["mask_rho"].values == 1


def _classic_roms() -> xr.Dataset:
    """Build a classic-layout Rutgers ROMS history file, in memory.

    Valued ``s_rho``/``s_w`` coordinates, ``ocean_time`` as the time *dimension*, ``hc``
    / ``Vtransform`` (1) / ``theta_*`` as 0-d variables, and ROMS' habit of writing 0.0
    (not NaN) on land. Ocean cells are 100-300 m deep; the temperature is a function of
    the level's true depth (free surface included), so the MLD it implies is the same
    in every column at every time.
    """
    s_w = np.linspace(-1.0, 0.0, NS + 1)
    s_rho = 0.5 * (s_w[:-1] + s_w[1:])
    cs_r, cs_w = _stretch(s_rho), _stretch(s_w)

    mask = np.ones((NY, NX))
    for j, i in LAND:
        mask[j, i] = 0.0
    ocean = mask == 1
    h = np.where(ocean, np.linspace(100.0, 300.0, NY * NX).reshape(NY, NX), 5.0)
    zeta = np.where(
        ocean, np.linspace(-0.6, 0.6, NT * NY * NX).reshape(NT, NY, NX), 0.0
    )

    z = _z_v1(
        s_rho[None, :, None, None],
        cs_r[None, :, None, None],
        h[None, None],
        zeta[:, None],
    )
    depth = -z
    # The MLD is exact only if the two levels bracketing the criterion's crossing both
    # lie below the mixed layer's base, i.e. a level sits between the base and the
    # crossing. Guard that, so a tweak to the grid cannot quietly turn this into an
    # approximate test.
    window = (depth > MIXED_LAYER) & (depth < EXPECTED_TEMPERATURE_MLD)
    assert window.any(axis=1)[:, ocean].all(), "fixture: no level in the MLD window"

    temp = TEMP_TOP - LAPSE * np.clip(depth - MIXED_LAYER, 0.0, None)
    temp = np.where(ocean, temp, 0.0)
    salt = np.where(ocean, SALINITY, 0.0) * np.ones((NT, NS, NY, NX))

    lon, lat = np.meshgrid(np.linspace(-150.0, -148.0, NX), np.linspace(58.0, 60.0, NY))
    rho4 = ("ocean_time", "s_rho", "eta_rho", "xi_rho")
    rho2 = ("eta_rho", "xi_rho")
    return xr.Dataset(
        {
            "Cs_r": ("s_rho", cs_r),
            "Cs_w": ("s_w", cs_w),
            "hc": ((), HC),
            "Vtransform": ((), np.int32(1)),
            "theta_s": ((), THETA_S),
            "theta_b": ((), THETA_B),
            "h": (rho2, h),
            "mask_rho": (rho2, mask),
            "pm": (rho2, np.full((NY, NX), 1 / 500.0)),
            "pn": (rho2, np.full((NY, NX), 1 / 400.0)),
            "angle": (rho2, np.zeros((NY, NX))),
            "zeta": (("ocean_time", *rho2), zeta),
            "temp": (rho4, temp),
            "salt": (rho4, salt),
        },
        coords={
            "s_rho": ("s_rho", s_rho, {"long_name": "S-coordinate at RHO-points"}),
            "s_w": ("s_w", s_w, {"long_name": "S-coordinate at W-points"}),
            "ocean_time": (
                "ocean_time",
                SECONDS,
                {
                    "long_name": "time since initialization",
                    "units": "seconds since 1970-01-01",
                },
            ),
            "lon_rho": (rho2, lon),
            "lat_rho": (rho2, lat),
        },
    )


@pytest.fixture
def classic_catalog(tmp_path, isolated_catalogs):
    """Write the classic file to disk and catalog it the way a user would.

    Returns ``(raw, catalog_path)``. The catalog lands in the discovery directory
    ``isolated_catalogs`` points at, under the name ``classic_roms``, so
    ``osk.read("classic_roms:classic")`` resolves it like any other source.
    """
    raw = _classic_roms()
    data = tmp_path / "data"
    data.mkdir()
    nc = data / "ocean_his_0001.nc"
    raw.to_netcdf(nc)
    out = build.build_catalog(
        {"classic": str(nc)},
        isolated_catalogs / "classic_roms.yaml",
        title="Classic Rutgers ROMS",
    )
    return raw, out


def test_a_classic_file_is_catalogued_as_roms_with_its_vertical_grid(classic_catalog):
    """The entry routes through the ROMS adapter and records Vtransform 1 and ``hc``.

    Read back from the YAML on disk, so the plain-Python-types requirement (a numpy
    scalar would not survive ``to_yaml_file``) is exercised as well as the values.
    """
    _, out = classic_catalog
    md = intake.from_yaml_file(str(out))["classic"].metadata

    assert md["model"] == "roms"
    assert md["loader"] == "ocean_skill.roms"
    assert md["vertical"] == {
        "s_dim": "s_rho",
        "hc": HC,
        "Vtransform": 1,
        "theta_s": THETA_S,
        "theta_b": THETA_B,
    }
    assert md["time_units"] == "seconds"
    assert md["reference_date"] == "1970-01-01"
    assert md["standard_names"]["temp"] == "sea_water_potential_temperature"
    assert md["standard_names"]["salt"] == "sea_water_practical_salinity"


def test_a_classic_file_does_not_record_its_sigma_values_as_metres(classic_catalog):
    """A valued ``s_rho`` must not leak into the vertical-extent metadata as depths.

    The generic probe takes a valued vertical coordinate for depth in metres; for ROMS
    that would record -0.99..-0.008 as ``geospatial_vertical_min/max``. Depth comes from
    ``z_rho`` at read time, so the entry carries no vertical extent at all.
    """
    _, out = classic_catalog
    md = intake.from_yaml_file(str(out))["classic"].metadata

    for key in (
        "geospatial_vertical_min",
        "geospatial_vertical_max",
        "vertical_resolution_min",
        "vertical_resolution_max",
    ):
        assert key not in md


def test_reading_a_classic_entry_gives_a_time_dim_a_bare_s_rho_and_a_depth_coord(
    classic_catalog,
):
    """Reading reshapes classic layout: ``time`` dim, bare ``s_rho``, ``z_rho``."""
    ds = osk.read(ENTRY)

    assert ds.attrs["ocean_skill_model"] == "roms"
    assert "time" in ds.dims
    assert "ocean_time" not in ds.dims
    np.testing.assert_array_equal(
        ds["time"].values,
        np.array(["2009-11-19T12:00", "2009-11-19T16:00"], dtype="datetime64[ns]"),
    )
    # sigma values live on ``sigma_r``; the dim itself stays unlabelled, or plots and
    # profiles would take sigma for depth in metres
    assert "s_rho" in ds.dims
    assert "s_rho" not in ds.variables
    assert "sigma_r" in ds.variables
    assert "z_rho" in ds.coords
    assert ds["z_rho"].dims == ("time", "s_rho", "eta_rho", "xi_rho")


def test_the_depth_coordinate_follows_the_vtransform_1_formula_and_masks_land(
    classic_catalog,
):
    """``z_rho`` equals the Vtransform 1 depths written out from the raw arrays."""
    raw, _ = classic_catalog
    ds = osk.read(ENTRY)
    ocean = _ocean(raw)

    z = ds["z_rho"].values
    np.testing.assert_allclose(z[..., ocean], _depths(raw)[..., ocean], rtol=1e-12)
    assert np.isnan(z[..., ~ocean]).all()  # land: zeta is masked, so depth is too

    temp = ds["sea_water_potential_temperature"].values
    assert np.isfinite(temp[..., ocean]).all()
    assert np.isnan(temp[..., ~ocean]).all()  # ROMS wrote 0.0 there; read() masks it


def test_temperature_threshold_mld_recovers_the_constructed_mixed_layer(
    classic_catalog,
):
    """Catalog -> read -> MLD gives the depth the profile was built to cross at.

    With temperature flat to 30 m and cooling at 0.025 degC/m below, the 0.2 degC
    criterion trips at exactly 38 m in every ocean column, at both times and for every
    depth and free-surface height -- and nowhere on land.
    """
    raw, _ = classic_catalog
    ds = osk.read(ENTRY)
    ocean = _ocean(raw)

    out = resolve_variable(ds, {"calculate": "mld", "method": "temperature_threshold"})

    assert set(out.dims) == {"time", "eta_rho", "xi_rho"}
    mld = out.transpose("time", "eta_rho", "xi_rho").values
    np.testing.assert_allclose(mld[:, ocean], EXPECTED_TEMPERATURE_MLD, rtol=1e-9)
    assert np.isnan(mld[:, ~ocean]).all()
    assert out.attrs["units"] == "m"


def test_density_threshold_mld_matches_gsw_on_the_constructed_columns(classic_catalog):
    """The density path agrees with TEOS-10 worked independently from the raw arrays.

    TEOS-10 density has no hand-derivable answer, so the expectation is each column's
    sigma0 computed with gsw directly from the file's own temperature and the
    ground-truth depths, then run through the same crossing scan. That still pins
    everything the classic layout could break: which depths, which columns, where land
    is. The answer must also land below the mixed layer and (a 0.03 kg/m3 change
    needing less than a 0.2 degC one, here) above the temperature criterion's 38 m.
    """
    gsw = pytest.importorskip("gsw")
    from ocean_skill import mld

    raw, _ = classic_catalog
    ds = osk.read(ENTRY)
    ocean = _ocean(raw)
    z = _depths(raw)
    lon, lat = raw["lon_rho"].values, raw["lat_rho"].values

    expected = np.full((NT, NY, NX), np.nan)
    for t in range(NT):
        for j, i in zip(*np.nonzero(ocean), strict=True):
            depth = -z[t, :, j, i]
            pressure = gsw.p_from_z(-depth, lat[j, i])
            sa = gsw.SA_from_SP(SALINITY, pressure, lon[j, i], lat[j, i])
            ct = gsw.CT_from_pt(sa, raw["temp"].values[t, :, j, i])
            expected[t, j, i] = mld._mld_threshold_1d(
                gsw.sigma0(sa, ct), depth, threshold=0.03, ref_depth=10.0
            )

    out = resolve_variable(ds, {"calculate": "mld", "method": "density_threshold"})

    got = out.transpose("time", "eta_rho", "xi_rho").values
    np.testing.assert_allclose(got[:, ocean], expected[:, ocean], rtol=1e-6)
    assert np.isnan(got[:, ~ocean]).all()
    assert (got[:, ocean] > MIXED_LAYER).all()
    assert (got[:, ocean] < EXPECTED_TEMPERATURE_MLD).all()


# -- the real xroms example file ------------------------------------------------


@pytest.mark.integration
def test_classic_depths_match_xroms_on_the_xroms_example_file():
    """On xroms' own example file, ``z_rho`` and ``z_w`` agree with xroms'.

    The file is classic Rutgers output (Vtransform 1, ``hc`` 5) -- the reason this
    support exists. xroms computes the same two fields by its own route, so this is an
    independent check of the whole chain on real numbers: detection, layout
    normalization, time handling and the Vtransform 1 transform. Needs the file in
    pooch's cache (``~/Library/Caches/xroms`` on a Mac); never downloads it.
    """
    xroms = pytest.importorskip("xroms")
    pooch = pytest.importorskip("pooch")
    path = pooch.os_cache("xroms") / "ROMS_example_full_grid.nc"
    if not path.exists():
        pytest.skip(f"xroms example file not in the local pooch cache: {path}")

    # as the catalog reader opens it: raw times, lazy
    raw = xr.open_dataset(path, decode_times=False, chunks={})
    meta = build._roms_metadata(raw)
    assert meta["vertical"] == {"s_dim": "s_rho", "hc": 5.0, "Vtransform": 1}

    std = roms.standardize(raw, meta)
    std = roms.add_interface_coord(std, meta)
    # xroms only adds z_rho/z_w with its default ``include_3D_metrics=True``; it also
    # returns ``(dataset, xgcm_grid)`` and renames eta_u/xi_v, hence comparing values
    xds, _ = xroms.roms_dataset(xr.open_dataset(path, chunks={}))

    for name, sdim in (("z_rho", "s_rho"), ("z_w", "s_w")):
        ours = std[name].transpose("time", sdim, "eta_rho", "xi_rho").values
        theirs = xds[name].transpose("ocean_time", sdim, "eta_rho", "xi_rho").values
        valid = np.isfinite(ours)
        assert valid.mean() > 0.5  # most of the domain is water
        assert np.isfinite(theirs[valid]).all()
        np.testing.assert_allclose(ours[valid], theirs[valid], atol=1e-6)
