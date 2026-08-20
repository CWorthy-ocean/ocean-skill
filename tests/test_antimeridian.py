"""A domain straddling the antimeridian must not come back painted across the globe.

The PACMED Pacific domain (longitudes 77°E to 316°E, crossing the dateline) forced
into ±180 has a bounding box the width of the planet, so the reference never got
cropped — and its 2-D longitudes folded mid-array (…179.9, -179.9…), so cell corners
derived across the fold averaged to ~0° and a conservative regrid painted the model
over the Atlantic. Regression tests for the whole chain: convention resolution, the
bbox crop, the corner derivation, and the map frame both renderers pick.
"""

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as A


def _pacific_test(ny=24, nx=40):
    """Build a curvilinear lane straddling the dateline, 0-360 native like ROMS."""
    lon = np.linspace(150.0, 250.0, nx)[None, :] * np.ones((ny, 1))
    lat = np.linspace(-20.0, 20.0, ny)[:, None] * np.ones((1, nx))
    return xr.DataArray(
        np.cos(np.deg2rad(lat)) * 10 + 5,
        dims=("eta", "xi"),
        coords={"lon": (("eta", "xi"), lon), "lat": (("eta", "xi"), lat)},
        attrs={"units": "mmol/m^3"},
    )


def _global_reference():
    """Build a 2-degree global grid in ±180, WOA-style."""
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(-179.0, 180.0, 2.0)
    return xr.DataArray(
        np.full((lat.size, lon.size), 5.0),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "mmol/m^3"},
    )


def test_natural_convention_follows_contiguity():
    assert A.natural_convention(_pacific_test()) == "0-360"
    assert A.natural_convention(_global_reference()) == "-180-180"
    # fits both conventions -> the ±180 default
    atlantic = _global_reference().sel(lon=slice(-40, 20))
    assert A.natural_convention(atlantic) == "-180-180"


@pytest.mark.parametrize("method", ["bilinear", "conservative_normed"])
def test_a_dateline_straddling_test_stays_inside_its_own_longitudes(method):
    """The bug in one line: a Pacific-only model must not fill a global map."""
    out = A.align(_pacific_test(), _global_reference(), method=method)
    assert out.attrs["lon_convention"] == "0-360"
    # the reference was cropped to the test's extent, not kept globe-wide
    assert float(out.lon.min()) >= 150.0 - 2 * A.DEFAULT_PAD - 2.0
    assert float(out.lon.max()) <= 250.0 + 2 * A.DEFAULT_PAD + 2.0
    # and the regridded lane holds sane values (the source spans 5..15)
    t = out["test"]
    assert np.isfinite(t).any()
    assert float(t.min()) >= 5.0 - 1e-6 and float(t.max()) <= 15.0 + 1e-6


def test_folded_corners_do_not_paint_the_far_side_of_the_planet():
    """Forced ±180, the fold is unavoidable — the corner unwrap must absorb it."""
    out = A.align(
        _pacific_test(),
        _global_reference(),
        method="conservative_normed",
        convention="-180-180",
    )
    # the ±180 bbox of a straddling domain is the whole globe, so the reference
    # stays wide — but nothing may land outside the test's own longitude band
    outside = out["test"].sel(lon=slice(-100.0, 140.0))
    assert int(np.isfinite(outside).sum()) == 0


def _fake_entry(monkeypatch, lon_min, lat_min, lon_max, lat_max):
    from types import SimpleNamespace

    entry = SimpleNamespace(
        metadata={
            "geospatial_lon_min": lon_min,
            "geospatial_lat_min": lat_min,
            "geospatial_lon_max": lon_max,
            "geospatial_lat_max": lat_max,
        }
    )
    monkeypatch.setattr("ocean_skill.catalog.resolve", lambda name: entry, raising=True)


def test_domain_box_stays_contiguous_for_a_straddling_domain(monkeypatch):
    from ocean_skill import comparison

    _fake_entry(monkeypatch, 77.4, -54.3, 316.2, 66.1)
    lon_min, _, lon_max, _ = comparison._domain_of("pac")
    assert lon_min < lon_max, "box endpoints must stay ordered"
    assert (lon_min, lon_max) == (77.4, 316.2)


def test_domain_box_still_normalizes_a_non_straddling_domain(monkeypatch):
    from ocean_skill import comparison

    _fake_entry(monkeypatch, 260.0, 18.0, 280.0, 31.0)
    lon_min, _, lon_max, _ = comparison._domain_of("gom")
    assert (lon_min, lon_max) == (-100.0, -80.0)


def test_static_maps_centre_on_180_for_a_straddling_field():
    ccrs = pytest.importorskip("cartopy.crs")
    from ocean_skill.plot.matplotlib_renderer import _map_projection

    aligned = A.align(_pacific_test(), _global_reference(), method="bilinear")
    proj = _map_projection(aligned)
    assert proj.proj4_params["lon_0"] == 180.0
    # a ±180 field keeps the default frame
    assert _map_projection(_global_reference()).proj4_params["lon_0"] == 0.0
    assert isinstance(proj, ccrs.PlateCarree)


def test_interactive_maps_centre_on_180_for_a_straddling_field():
    pytest.importorskip("cartopy.crs")
    from ocean_skill.plot.holoviews_renderer import _output_projection

    aligned = A.align(_pacific_test(), _global_reference(), method="bilinear")
    proj = _output_projection(aligned["test"])
    assert proj is not None and proj.proj4_params["lon_0"] == 180.0
    # a ±180 field keeps hvplot's own default (None)
    assert _output_projection(_global_reference()) is None
