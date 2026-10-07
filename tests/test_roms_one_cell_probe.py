"""A ROMS file with a single horizontal cell is still a grid.

:func:`ocean_skill.build.guess_feature_type` counts a horizontal dimension only when it
is longer than one, so a ROMS file cut down to one ``eta_rho`` x one ``xi_rho`` cell (a
point extraction, a one-column test case) looked like a fixed point -- a
``timeSeriesProfile`` in the classic layout (valued ``s_rho``), a ``timeSeries`` in the
UCLA one (bare ``s_rho``), a ``profile`` or a ``point`` with a single record. None of
those is what the model output is: it is the same rho grid on the same staggered
dimensions, only small, and everything downstream (the ROMS loader, the comparison's
grid handling) is written for it as one. So a ROMS store on ``eta_rho``/``xi_rho`` is
probed as ``grid`` whatever its size.

An explicit ``featureType=`` given to ``add_source`` still wins (the caller said so),
and so does a ``featureType`` the file itself declares -- only the *guess* is
overridden.

Synthetic throughout, like ``test_roms_classic_detect.py``.
"""

from __future__ import annotations

import intake
import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import build


def _classic_roms(ny: int, nx: int, ns: int, nt: int) -> xr.Dataset:
    """Return a classic-layout Rutgers ROMS dataset (valued ``s_rho``)."""
    rng = np.random.default_rng(0)
    s_w = np.linspace(-1.0, 0.0, ns + 1)
    lon, lat = np.meshgrid(np.linspace(-150.0, -148.0, nx), np.linspace(58.0, 60.0, ny))
    rho2 = ("eta_rho", "xi_rho")
    return xr.Dataset(
        {
            "Cs_r": ("s_rho", np.linspace(-0.9, -0.1, ns)),
            "hc": ((), np.float64(5.0)),
            "Vtransform": ((), np.int32(1)),
            "h": (rho2, np.full((ny, nx), 50.0)),
            "mask_rho": (rho2, np.ones((ny, nx))),
            "zeta": (("ocean_time", *rho2), rng.random((nt, ny, nx))),
            "temp": (("ocean_time", "s_rho", *rho2), rng.random((nt, ns, ny, nx))),
        },
        coords={
            "s_rho": ("s_rho", 0.5 * (s_w[:-1] + s_w[1:])),
            "ocean_time": (
                "ocean_time",
                np.arange(nt) * 3600.0,
                {"units": "seconds since 1970-01-01"},
            ),
            "lon_rho": (rho2, lon),
            "lat_rho": (rho2, lat),
        },
    )


def _ucla_roms(ny: int, nx: int, ns: int, nt: int) -> xr.Dataset:
    """Return a UCLA-layout ROMS dataset: ``sigma_r``, bare ``s_rho``, ``time`` dim."""
    rng = np.random.default_rng(0)
    rho2 = ("eta_rho", "xi_rho")
    lon, lat = np.meshgrid(np.linspace(-150.0, -148.0, nx), np.linspace(58.0, 60.0, ny))
    ds = xr.Dataset(
        {
            "Cs_r": ("s_rho", np.linspace(-0.9, -0.1, ns)),
            "sigma_r": ("s_rho", np.linspace(-0.9, -0.1, ns)),
            "h": (rho2, np.full((ny, nx), 50.0)),
            "mask_rho": (rho2, np.ones((ny, nx))),
            "lon_rho": (rho2, lon),
            "lat_rho": (rho2, lat),
            "ocean_time": (
                "time",
                np.arange(nt) * 3600.0,
                {"long_name": "time since 2012-01-01 00:00:00"},
            ),
            "zeta": (("time", *rho2), rng.random((nt, ny, nx))),
            "temp": (("time", "s_rho", *rho2), rng.random((nt, ns, ny, nx))),
        }
    )
    ds.attrs.update(hc=5.0, Vtransform=2, theta_s=5.0, theta_b=1.0)
    return ds


LAYOUTS = {"classic": _classic_roms, "ucla": _ucla_roms}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_one_cell_file_with_levels_and_times_is_guessed_as_a_point_not_a_grid(layout):
    """Teeth: the generic guess is the mislabel (``timeSeriesProfile``...)."""
    ds = LAYOUTS[layout](1, 1, 4, 3)

    ftype, source = build.guess_feature_type(ds)

    assert source == "inferred"
    assert ftype == {"classic": "timeSeriesProfile", "ucla": "timeSeries"}[layout]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    ("ns", "nt"), [(4, 3), (4, 1), (1, 3), (1, 1)], ids=["sxt", "s", "t", "cell"]
)
def test_a_one_cell_roms_file_probes_as_a_grid(layout, ns, nt):
    """Several levels and times, one of each, neither: always an inferred grid."""
    ds = LAYOUTS[layout](1, 1, ns, nt)

    md = build._probe(ds, build.ROMS_STANDARD_NAMES)

    assert md["model"] == "roms"
    assert md["featureType"] == "grid"
    assert md["featureType_source"] == "inferred"


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(("ny", "nx"), [(3, 5), (1, 5), (3, 1)])
def test_a_roms_file_with_more_than_one_cell_is_still_a_grid(layout, ny, nx):
    """A strip one cell wide is a grid too, as is a full one (as before)."""
    md = build._probe(LAYOUTS[layout](ny, nx, 4, 3), build.ROMS_STANDARD_NAMES)

    assert md["featureType"] == "grid"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_one_cell_roms_file_records_no_horizontal_resolution(layout):
    """Gridded, but one cell has no spacing: nothing is recorded, nothing raises."""
    md = build._probe(LAYOUTS[layout](1, 1, 4, 3), build.ROMS_STANDARD_NAMES)

    assert "grid_resolution_deg" not in md
    assert "grid_resolution_km" not in md


def test_only_a_roms_store_is_given_the_grid_rule():
    """The same one-cell layout without the s-coordinate stays a fixed point.

    A ``Cs_r``-less dataset is not ROMS output (see ``_roms_metadata``), so a one-cell
    ``eta_rho``/``xi_rho`` file of some other origin keeps the generic guess.
    """
    ds = _classic_roms(1, 1, 4, 3).drop_vars("Cs_r")

    md = build._probe(ds, None)

    assert "model" not in md
    assert md["featureType"] == "timeSeriesProfile"


def test_a_featuretype_the_file_declares_is_kept():
    """Only the guess is overridden: a file that says what it is, is believed."""
    ds = _classic_roms(1, 1, 4, 3)
    ds.attrs["featureType"] = "timeSeriesProfile"

    md = build._probe(ds, build.ROMS_STANDARD_NAMES)

    assert md["featureType"] == "timeSeriesProfile"
    assert md["featureType_source"] == "declared"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_add_source_records_a_one_cell_roms_file_as_a_grid(layout, tmp_path):
    """Through the public builder: the entry says ``grid``, inferred."""
    nc = tmp_path / "point_his.nc"
    LAYOUTS[layout](1, 1, 4, 3).to_netcdf(nc)
    cat = build.new_catalog(title="t")

    build.add_source(cat, "point", str(nc))

    md = cat["point"].metadata
    assert md["featureType"] == "grid"
    assert md["featureType_source"] == "inferred"


def test_a_callers_featuretype_still_wins_over_the_grid_rule(tmp_path):
    """``add_source(..., featureType=...)`` is the caller's word, saved as given."""
    nc = tmp_path / "point_his.nc"
    _classic_roms(1, 1, 4, 3).to_netcdf(nc)
    cat = build.new_catalog(title="t")

    build.add_source(cat, "point", str(nc), featureType="timeseriesprofile")

    md = cat["point"].metadata
    assert md["featureType"] == "timeSeriesProfile"  # canonicalized, as for any caller
    assert md["featureType_source"] == "declared"


def test_a_one_cell_roms_entry_reads_back_as_the_grid_it_is(
    tmp_path, isolated_catalogs
):
    """Catalog -> ``osk.read``: a ``grid`` on one cell, with its depth coordinate."""
    nc = tmp_path / "point_his.nc"
    _classic_roms(1, 1, 4, 3).to_netcdf(nc)
    out = build.build_catalog(
        {"run": str(nc)}, isolated_catalogs / "onecell.yaml", title="one cell"
    )

    md = intake.from_yaml_file(str(out))["run"].metadata
    assert md["featureType"] == "grid"

    ds = osk.read("onecell:run")

    assert ds.attrs["featureType"] == "grid"
    assert ds.sizes["eta_rho"] == 1
    assert ds.sizes["xi_rho"] == 1
    assert ds.sizes["time"] == 3
    assert ds["z_rho"].dims == ("time", "s_rho", "eta_rho", "xi_rho")
