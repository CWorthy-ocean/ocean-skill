"""A domain straddling the antimeridian must not come back painted across the globe.

The PACMED Pacific domain (longitudes 77°E to 316°E, crossing the dateline) forced
into ±180 has a bounding box the width of the planet, so the reference never got
cropped — and its 2-D longitudes folded mid-array (…179.9, -179.9…), so cell corners
derived across the fold averaged to ~0° and a conservative regrid painted the model
over the Atlantic. Regression tests for the whole chain: convention resolution, the
bbox crop, the corner derivation, and the map frame both renderers pick, and (for
interactive movies) the web-tile basemap and the offline coastline that stands in
for it.
"""

import warnings

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


# -- interactive movies: web tiles cannot show a straddling domain whole -------------


def _rectilinear_field(lon0: float, lon1: float, *, nx=24, ny=10):
    """Build a plain lat/lon field over ``[lon0, lon1]`` — a movie frame's grid."""
    return xr.DataArray(
        np.full((ny, nx), 5.0),
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(-15.0, 15.0, ny),
            "lon": np.linspace(lon0, lon1, nx),
        },
        attrs={"units": "mmol/m^3"},
    )


def _straddling_field():
    return _rectilinear_field(150.0, 250.0)  # crosses the dateline, 0-360 native


def _gom_like_field():
    return _rectilinear_field(-98.0, -80.0)  # a domain that fits ±180 whole


def test_tiles_for_downgrades_only_a_straddling_field():
    from ocean_skill.plot.holoviews_renderer import _tiles_for

    with pytest.warns(UserWarning, match="Web Mercator"):
        assert _tiles_for(True, _straddling_field()) is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _tiles_for(True, _gom_like_field()) is True
    assert not any("Web Mercator" in str(w.message) for w in caught)

    # already off never warns, straddling or not
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _tiles_for(False, _straddling_field()) is False
    assert not any("Web Mercator" in str(w.message) for w in caught)


def test_movie_coastline_lands_in_the_180_centred_frame_for_a_straddling_domain():
    pytest.importorskip("cartopy.feature")
    from ocean_skill.plot.holoviews_renderer import _extension, _movie_coastline

    _extension()
    path = _movie_coastline(_straddling_field())
    if path is None:  # pragma: no cover - depends on local Natural Earth cache
        pytest.skip("Natural Earth coastline data unavailable offline")
    xs = np.asarray(path.dimension_values(0))
    finite = xs[np.isfinite(xs)]
    assert finite.size, "the clipped coastline came back empty"
    # every point must land inside the 180-centred frame, not at the raw domain
    # longitudes (150..250) the mesh itself is projected away from
    assert np.all(np.abs(finite) <= 180.0 + 1e-6)
    assert not np.any(finite > 180.0)


def test_movie_coastline_is_unshifted_for_a_non_straddling_domain():
    pytest.importorskip("cartopy.feature")
    from ocean_skill.plot.holoviews_renderer import _extension, _movie_coastline

    _extension()
    field = _gom_like_field()
    path = _movie_coastline(field)
    if path is None:  # pragma: no cover - depends on local Natural Earth cache
        pytest.skip("Natural Earth coastline data unavailable offline")
    xs = np.asarray(path.dimension_values(0))
    finite = xs[np.isfinite(xs)]
    assert finite.size
    # unchanged behaviour: plain geographic degrees, within the padded clip box
    lon0, lon1 = float(field.lon.min()), float(field.lon.max())
    pad = 0.5 * (lon1 - lon0)
    assert finite.min() >= lon0 - pad - 1e-6
    assert finite.max() <= lon1 + pad + 1e-6


def _facet_item(lon0: float, lon1: float, *, days=3):
    """One ``facet_movie`` item, shaped as ``Field.as_item()`` builds it."""
    import pandas as pd

    rng = np.random.default_rng(0)
    ny, nx = 10, 24
    field = xr.DataArray(
        rng.normal(5.0, 1.0, (days, ny, nx)).astype("float32"),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2012-01-01", periods=days, freq="D"),
            "lat": np.linspace(-15.0, 15.0, ny),
            "lon": np.linspace(lon0, lon1, nx),
        },
        attrs={"units": "mmol/m^3"},
    )
    return {
        "field": field,
        "facet_dim": "time",
        "row_dim": None,
        "units": "mmol/m^3",
        "standard_name": "mole_concentration_of_nitrate_in_sea_water",
        "label": "pac_dt_ramp",
    }


def _first_movie_frame(movie):
    import holoviews as hv

    obj = getattr(movie, "object", movie)
    holomap = next(el for el in obj.traverse() if isinstance(el, hv.HoloMap))
    return holomap[next(iter(holomap.kdims[0].values))]


def test_a_straddling_facet_movie_drops_tiles_for_a_coastline():
    pytest.importorskip("geoviews")
    pytest.importorskip("cartopy.feature")
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    item = _facet_item(150.0, 250.0)
    with pytest.warns(UserWarning, match="Web Mercator"):
        movie = render(
            PlotSpec(family="facet_movie", items=[item], options={"domain": None}),
            renderer="holoviews",
        )
    kinds = [type(n).__name__ for n in _first_movie_frame(movie).traverse()]
    assert "WMTS" not in kinds, kinds
    assert "Path" in kinds, kinds


def test_a_non_straddling_facet_movie_keeps_tiles_without_a_warning():
    pytest.importorskip("geoviews")
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    item = _facet_item(-98.0, -80.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        movie = render(
            PlotSpec(family="facet_movie", items=[item], options={"domain": None}),
            renderer="holoviews",
        )
    assert not any("Web Mercator" in str(w.message) for w in caught)
    kinds = [type(n).__name__ for n in _first_movie_frame(movie).traverse()]
    assert "WMTS" in kinds, kinds
