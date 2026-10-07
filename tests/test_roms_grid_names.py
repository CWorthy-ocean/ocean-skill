"""A ROMS store's grid and free surface keep the names the loader reads them under.

The build-time probe records ``standard_name`` *attributes* ahead of the fallback
``ROMS_STANDARD_NAMES`` table, and :func:`ocean_skill.roms.standardize` renames every
variable the catalog maps. A store whose ``h``/``mask_rho``/``zeta`` carry their own
CF-ish ``standard_name`` (``sea_floor_depth``, ``sea_surface_elevation_anomaly``, ...)
therefore got them renamed away from the names the loader looks them up by: the depth
coordinate raised ``KeyError: "No variable named 'h'"``, land was never masked (no
``mask_rho``) and a free surface it no longer recognised fell back to a flat
``zeta = 0``.

Two halves, tested separately and together:

* the probe never records a grid/vertical variable (``roms.GRID_VARIABLE_NAMES``) and
  always records ``zeta`` under the loader's own free-surface name;
* ``standardize`` is the safety net for catalogs already written with the bad map: it
  never renames those names and only renames ``zeta`` to a name the loader knows.

Synthetic throughout, like ``test_roms_classic_detect.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import build, roms

NS, NY, NX, NT = 4, 3, 5, 2
HC = 5.0
LAND = (0, 0)  # (eta, xi) cell with mask_rho == 0
ZETA = np.array([0.5, 1.5])  # one value per time step, uniform over the grid
FREE_SURFACE = "sea_surface_height_above_geoid"

# What a CF-annotated store says about its own grid -- and what the paper-era catalogs
# therefore recorded, which the loader then could not find.
BAD_ATTRS = {
    "h": "sea_floor_depth",
    "mask_rho": "land_binary_mask",
    "angle": "angle_of_rotation_from_east_to_x",
    "zeta": "sea_surface_elevation_anomaly",
}


def _annotated_roms(attrs: dict[str, str | None] | None = None) -> xr.Dataset:
    """Classic-layout ROMS output whose grid variables carry ``standard_name`` attrs.

    ``attrs`` maps a variable name to the ``standard_name`` it gets (``None``: none);
    the default is :data:`BAD_ATTRS`. One land cell, a ``zeta`` that differs by time
    (so a free surface that is silently ignored shows), Vtransform 1.
    """
    attrs = BAD_ATTRS if attrs is None else attrs

    def a(name: str) -> dict[str, str]:
        return {"standard_name": attrs[name]} if attrs.get(name) else {}

    rng = np.random.default_rng(0)
    s_w = np.linspace(-1.0, 0.0, NS + 1)
    s_rho = 0.5 * (s_w[:-1] + s_w[1:])
    lon, lat = np.meshgrid(np.linspace(-150.0, -148.0, NX), np.linspace(58.0, 60.0, NY))
    mask = np.ones((NY, NX))
    mask[LAND] = 0.0
    rho2, rho3 = ("eta_rho", "xi_rho"), ("ocean_time", "eta_rho", "xi_rho")
    return xr.Dataset(
        {
            "Cs_r": ("s_rho", np.linspace(-0.9, -0.1, NS)),
            "Cs_w": ("s_w", np.linspace(-1.0, 0.0, NS + 1)),
            "hc": ((), np.float64(HC)),
            "Vtransform": ((), np.int32(1)),
            "h": (rho2, np.linspace(20.0, 200.0, NY * NX).reshape(NY, NX), a("h")),
            "mask_rho": (rho2, mask, a("mask_rho")),
            "angle": (rho2, np.zeros((NY, NX)), a("angle")),
            "zeta": (
                rho3,
                np.broadcast_to(ZETA[:, None, None], (NT, NY, NX)).copy(),
                a("zeta"),
            ),
            "temp": (
                ("ocean_time", "s_rho", *rho2),
                1.0 + rng.random((NT, NS, NY, NX)),
            ),
        },
        coords={
            "s_rho": ("s_rho", s_rho),
            "s_w": ("s_w", s_w),
            "ocean_time": (
                "ocean_time",
                np.arange(NT) * 3600.0,
                {"units": "seconds since 1970-01-01"},
            ),
            "lon_rho": (rho2, lon),
            "lat_rho": (rho2, lat),
        },
    )


def _expected_z_rho(raw: xr.Dataset) -> np.ndarray:
    """Vtransform 1 depths ``(time, s_rho, eta, xi)``, from the raw arrays alone."""
    sigma = raw["s_rho"].values[None, :, None, None]
    cs = raw["Cs_r"].values[None, :, None, None]
    h = raw["h"].values[None, None]
    zeta = raw["zeta"].values[:, None]
    z0 = HC * (sigma - cs) + cs * h
    return z0 + zeta * (1 + z0 / h)


# ------------------------------------------------------------------------- the probe


def test_probe_leaves_the_grid_variables_out_of_the_standard_names_map():
    """A ``standard_name`` attribute on ``h``/``mask_rho``/``angle`` is not recorded.

    Attributes win over the fallback table in the probe, which is how these ever got in.
    """
    md = build._probe(_annotated_roms(), build.ROMS_STANDARD_NAMES)

    assert md["model"] == "roms"
    for name in ("h", "mask_rho", "angle"):
        assert name not in md["standard_names"]
    # ``variables`` is recomputed from the map, so none of their attribute names leak
    for leaked in ("sea_floor_depth", "land_binary_mask", BAD_ATTRS["angle"]):
        assert leaked not in md["variables"]
    # the data a user compares is untouched
    assert md["standard_names"]["temp"] == "sea_water_potential_temperature"
    assert "sea_water_potential_temperature" in md["variables"]


@pytest.mark.parametrize(
    "file_says", [None, "sea_surface_elevation_anomaly", "sea_surface_height"]
)
def test_probe_records_zeta_under_the_loaders_free_surface_name(file_says):
    """Whatever the file calls its free surface, the catalog says the loader's name."""
    md = build._probe(
        _annotated_roms({**BAD_ATTRS, "zeta": file_says}), build.ROMS_STANDARD_NAMES
    )

    assert md["standard_names"]["zeta"] == FREE_SURFACE
    assert FREE_SURFACE in md["variables"]
    assert "sea_surface_elevation_anomaly" not in md["variables"]


def test_probe_applies_the_grid_name_rule_with_no_fallback_table_too():
    """``name_map=None`` skips the table, not the rule: the loader's names hold."""
    md = build._probe(_annotated_roms(), None)

    assert "h" not in md["standard_names"]
    assert "mask_rho" not in md["standard_names"]
    assert md["standard_names"]["zeta"] == FREE_SURFACE


def test_probe_of_a_non_roms_store_still_trusts_its_own_standard_name_attributes():
    """The rule is ROMS-only: another model's ``h``/``zeta`` is its own business."""
    raw = _annotated_roms().drop_vars("Cs_r")  # no s-coordinate stretching: not ROMS
    md = build._probe(raw, build.ROMS_STANDARD_NAMES)

    assert "model" not in md
    assert md["standard_names"]["h"] == "sea_floor_depth"
    assert md["standard_names"]["zeta"] == "sea_surface_elevation_anomaly"


# ---------------------------------------------------------------------- standardize


def _old_style_meta(raw: xr.Dataset, **names: str) -> dict:
    """Return a catalog entry as written *before* the probe knew better (bad map)."""
    return {
        **build._roms_metadata(raw),
        "standard_names": {
            "h": "sea_floor_depth",
            "mask_rho": "land_binary_mask",
            "zeta": "sea_surface_elevation_anomaly",
            "temp": "sea_water_potential_temperature",
            **names,
        },
    }


def test_standardize_ignores_grid_names_a_catalog_maps_away():
    """Catalogs already built with the bad map load: ``h`` stays, land is masked."""
    raw = _annotated_roms()

    out = roms.standardize(raw, _old_style_meta(raw))

    assert "h" in out.coords
    assert "mask_rho" in out.coords
    assert "sea_floor_depth" not in out.variables
    assert "land_binary_mask" not in out.variables
    temp = out[
        "sea_water_potential_temperature"
    ]  # the one rename that is still applied
    assert "temp" not in out.variables
    land = temp.isel(eta_rho=LAND[0], xi_rho=LAND[1])
    assert land.isnull().all()  # ROMS writes finite values on land; the mask NaNs them
    assert int(np.isfinite(temp).sum()) == temp.size - NT * NS  # and nothing else


def test_standardize_renames_zeta_to_a_name_the_loader_knows_not_the_catalogs():
    """``zeta`` -> ``sea_surface_elevation_anomaly`` lands on the loader's name."""
    raw = _annotated_roms()

    out = roms.standardize(raw, _old_style_meta(raw))

    assert FREE_SURFACE in out.variables
    assert "sea_surface_elevation_anomaly" not in out.variables
    assert "zeta" not in out.variables


def test_the_depth_coordinate_rides_on_the_free_surface_not_a_flat_one():
    """``z_rho`` is the Vtransform 1 depth for the file's own ``zeta`` at every step."""
    raw = _annotated_roms()
    expected = _expected_z_rho(raw)

    z = roms.standardize(raw, _old_style_meta(raw))["z_rho"]

    assert z.dims == ("time", "s_rho", "eta_rho", "xi_rho")
    ocean = raw["mask_rho"].values == 1
    np.testing.assert_allclose(z.values[..., ocean], expected[..., ocean], rtol=1e-12)
    # not flat: the second step is a metre higher at the free surface than the first
    assert not np.allclose(z.values[1][..., ocean], z.values[0][..., ocean])


def test_the_identity_maps_of_the_old_workaround_still_load():
    """Catalogs that re-declared the grid as itself by hand (the fix-up) keep working.

    ``standard_names={h: h, mask_rho: mask_rho, ..., zeta: <free surface>}`` was how a
    store with annotated grid variables was made to load; it must neither break nor be
    needed now.
    """
    raw = _annotated_roms()
    meta = _old_style_meta(raw)
    meta["standard_names"] = {
        "h": "h",
        "mask_rho": "mask_rho",
        "angle": "angle",
        "zeta": FREE_SURFACE,
        "temp": "sea_water_potential_temperature",
    }

    out = roms.standardize(raw, meta)

    assert {"h", "mask_rho", "angle"} <= set(out.coords)
    ocean = raw["mask_rho"].values == 1
    np.testing.assert_allclose(
        out["z_rho"].values[..., ocean], _expected_z_rho(raw)[..., ocean], rtol=1e-12
    )
    assert (
        out["sea_water_potential_temperature"]
        .isel(eta_rho=LAND[0], xi_rho=LAND[1])
        .isnull()
        .all()
    )


@pytest.mark.parametrize("target", ["zeta", FREE_SURFACE])
def test_standardize_keeps_a_free_surface_name_the_loader_already_knows(target):
    """A catalog naming either of the loader's own free-surface names is taken as is."""
    raw = _annotated_roms()

    out = roms.standardize(raw, _old_style_meta(raw, zeta=target))

    other = ({"zeta", FREE_SURFACE} - {target}).pop()
    assert target in out.variables
    assert other not in out.variables
    ocean = raw["mask_rho"].values == 1
    np.testing.assert_allclose(
        out["z_rho"].values[..., ocean], _expected_z_rho(raw)[..., ocean], rtol=1e-12
    )


def test_every_grid_name_is_protected_from_the_rename():
    """The shared tuple both halves use: spot-check what is in it and what is not."""
    names = set(roms.GRID_VARIABLE_NAMES)
    assert {
        "h",
        "mask_rho",
        "mask_u",
        "mask_v",
        "mask_psi",
        "angle",
        "pm",
        "pn",
        "f",
    } <= names
    assert {
        "Cs_r",
        "Cs_w",
        "sigma_r",
        "sigma_w",
        "s_rho",
        "s_w",
        "hc",
        "Vtransform",
    } <= names
    assert {"lon_rho", "lat_rho", "lon_u", "lat_u", "lon_v", "lat_v"} <= names
    # a tracer or the free surface is *not* a grid variable: those are renamed
    assert not names & {"temp", "salt", "zeta", "u", "v"}


# ------------------------------------------------------------- catalog -> read, whole


def test_a_store_with_annotated_grid_variables_reads_without_any_workaround(
    tmp_path, isolated_catalogs
):
    """``add_source`` + ``osk.read`` on such a file, no ``standard_names=`` fix-up.

    The paper's workaround re-declared ``h``, ``mask_rho``, ... as their own names and
    ``zeta`` as the free surface by hand; the probe now does that by itself.
    """
    raw = _annotated_roms()
    nc = tmp_path / "annotated_his.nc"
    raw.to_netcdf(nc)
    out = build.build_catalog(
        {"run": str(nc)}, isolated_catalogs / "annotated.yaml", title="annotated grid"
    )

    import intake

    md = intake.from_yaml_file(str(out))["run"].metadata
    assert md["standard_names"]["zeta"] == FREE_SURFACE
    assert "h" not in md["standard_names"]

    ds = osk.read("annotated:run")
    assert "h" in ds.coords
    temp = ds["sea_water_potential_temperature"]
    assert temp.isel(eta_rho=LAND[0], xi_rho=LAND[1]).isnull().all()
    ocean = raw["mask_rho"].values == 1
    np.testing.assert_allclose(
        ds["z_rho"].values[..., ocean], _expected_z_rho(raw)[..., ocean], rtol=1e-12
    )
