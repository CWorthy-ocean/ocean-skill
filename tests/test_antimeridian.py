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


def test_natural_convention_is_not_flipped_by_float_noise_on_a_tied_span():
    """A domain nowhere near the dateline (e.g. -160..-125) is an exact tie: both
    conventions give the same span. The two wrap computations round differently at
    the ~1e-14 level, so a bare ``<`` used to flip roughly a quarter of realistic
    (irregular, curvilinear-style) grids to "0-360" -- dropping their tiles for no
    reason. Seed 7 is pinned because it reproduced the flip before the fix.
    """
    rng = np.random.default_rng(7)
    lon = np.linspace(-160.0, -125.0, 24) + rng.normal(0, 1e-9, 24)
    field = xr.DataArray(
        np.full((10, 24), 5.0),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-15.0, 15.0, 10), "lon": lon},
    )
    assert A.natural_convention(field) == "-180-180"


def test_natural_convention_treats_180_as_a_seam_not_a_wrap():
    """A domain that reaches, but does not cross, the dateline (e.g. 120..180E)
    does not straddle anything -- but wrapping +180 to -180 for the ±180 span
    inflates it to the whole globe, so the domain used to lose its tiles anyway.
    """
    edge = xr.DataArray(
        np.full((5, 41), 5.0),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-10.0, 10.0, 5), "lon": np.linspace(120.0, 180.0, 41)},
    )
    assert A.natural_convention(edge) == "-180-180"

    overshoot = edge.assign_coords(lon=np.linspace(120.0, 180.0000001, 41))
    assert A.natural_convention(overshoot) == "-180-180"

    # a genuine straddler running through 180 must still read as 0-360
    straddler = edge.assign_coords(lon=np.linspace(170.0, 190.0, 41))
    assert A.natural_convention(straddler) == "0-360"


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


# -- perimeter_of: the true grid-edge ring, not a bounding box -----------------------


def _rotated_grid(ny=20, nx=30, lon0=150.0, lat0=10.0, degrees=30.0):
    """Build a regular grid rotated ``degrees`` about its own origin.

    A stand-in for a ROMS domain whose rows/columns are not lines of constant
    lon/lat.
    """
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    theta = np.radians(degrees)
    x = i * np.cos(theta) - j * np.sin(theta)
    y = i * np.sin(theta) + j * np.cos(theta)
    return lon0 + x, lat0 + y


def test_perimeter_traces_the_rotated_grid_edge_not_its_bbox():
    lon, lat = _rotated_grid()
    ring = A.perimeter_of(lon, lat, max_points=40)
    assert np.allclose(ring[0], ring[-1]), "ring must close"
    # every vertex is an actual grid-edge point, not a rectangle interpolated from
    # the corners -- so lat varies along the "top" edge instead of staying constant
    edge = {
        tuple(np.round(pt, 6))
        for edge_lon, edge_lat in (
            (lon[0, :], lat[0, :]),
            (lon[:, -1], lat[:, -1]),
            (lon[-1, :], lat[-1, :]),
            (lon[:, 0], lat[:, 0]),
        )
        for pt in np.stack([edge_lon, edge_lat], axis=1)
    }
    assert all(tuple(np.round(pt, 6)) in edge for pt in ring)
    assert np.ptp(lat[0, :]) > 1.0, "a rotated top edge is not flat"
    for corner in (
        (lon[0, 0], lat[0, 0]),
        (lon[0, -1], lat[0, -1]),
        (lon[-1, -1], lat[-1, -1]),
        (lon[-1, 0], lat[-1, 0]),
    ):
        assert any(np.allclose(corner, pt) for pt in ring), corner


def test_perimeter_stays_contiguous_across_the_antimeridian():
    field = _pacific_test()
    ring = A.perimeter_of(np.asarray(field["lon"]), np.asarray(field["lat"]))
    assert np.max(np.abs(np.diff(ring[:, 0]))) < 30.0, "no 360-degree jump"


def test_perimeter_of_a_rectilinear_grid_is_its_bbox():
    lon1d = np.linspace(0.0, 10.0, 5)
    lat1d = np.linspace(0.0, 5.0, 4)
    ring = A.perimeter_of(lon1d, lat1d)
    lo0, la0, lo1, la1 = A.bbox_of(
        xr.DataArray(
            np.zeros((4, 5)), dims=("lat", "lon"), coords={"lon": lon1d, "lat": lat1d}
        )
    )
    assert np.allclose(
        ring, [[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]]
    )


def test_perimeter_thinning_still_keeps_every_corner():
    lon, lat = _rotated_grid()
    ring = A.perimeter_of(lon, lat, max_points=16)
    for corner in (
        (lon[0, 0], lat[0, 0]),
        (lon[0, -1], lat[0, -1]),
        (lon[-1, -1], lat[-1, -1]),
        (lon[-1, 0], lat[-1, 0]),
    ):
        assert any(np.allclose(corner, pt) for pt in ring), corner


def test_perimeter_of_returns_none_without_horizontal_coords():
    assert A.perimeter_of(np.array([]), np.array([])) is None
    assert A.perimeter_of(np.zeros((3, 3, 3)), np.zeros((3, 3, 3))) is None


# -- _outline_of: the stored ring re-grounded in the requested convention -----------


def _fake_outline_entry(monkeypatch, name, outline):
    from types import SimpleNamespace

    entry = SimpleNamespace(metadata={"domain_outline": outline})
    resolved = {name: entry}

    def fake_resolve(source):
        if source not in resolved:
            raise KeyError(source)
        return resolved[source]

    monkeypatch.setattr("ocean_skill.catalog.resolve", fake_resolve, raising=True)


def test_outline_of_a_straddling_domain_stays_in_0_360(monkeypatch):
    from ocean_skill import comparison

    ring = [
        [77.4 + t * (316.2 - 77.4), -10.0 + 20.0 * t] for t in np.linspace(0, 1, 12)
    ]
    _fake_outline_entry(monkeypatch, "pac", ring)

    auto = comparison._outline_of("pac")
    assert auto[:, 0].min() >= 0.0 and auto[:, 0].max() <= 360.0

    explicit = comparison._outline_of("pac", "0-360")
    assert np.allclose(auto, explicit)

    # -180/180 would split this ring at its own seam, so it falls back to 0-360
    # rather than fold it
    fallback = comparison._outline_of("pac", "-180-180")
    assert np.allclose(fallback, explicit)


def test_outline_of_a_non_straddling_domain_normalizes_both_ways(monkeypatch):
    from ocean_skill import comparison

    ring = [[260.0 + 20.0 * t, 18.0 + 13.0 * t] for t in np.linspace(0, 1, 8)]
    _fake_outline_entry(monkeypatch, "gom", ring)

    pm180 = comparison._outline_of("gom", "-180-180")
    assert pm180[:, 0].min() >= -180.0 and pm180[:, 0].max() <= 180.0
    assert np.isclose(pm180[:, 0].min(), -100.0)
    assert np.isclose(pm180[:, 0].max(), -80.0)

    zero360 = comparison._outline_of("gom", "0-360")
    assert np.isclose(zero360[:, 0].min(), 260.0)
    assert np.isclose(zero360[:, 0].max(), 280.0)

    # auto picks -180/180 here, since that does not split it
    auto = comparison._outline_of("gom")
    assert np.allclose(auto, pm180)


def test_outline_of_is_none_without_a_declared_outline(monkeypatch):
    from ocean_skill import comparison

    _fake_outline_entry(monkeypatch, "has_none", None)
    assert comparison._outline_of("has_none") is None
    assert comparison._outline_of("unresolvable") is None


# -- Comparison.plot()/ComparisonSet.plot() prefer the outline, fall back to the bbox


def _stub_comparison(monkeypatch, *, test_name="test_src", domain_outline=None):
    """Build a ``Comparison`` with ``.aligned`` set by hand, skipping align()/I/O.

    ``_implied_over`` (from ``Comparison.__init__``) resolves ``reference`` through
    the catalog too; leaving it unresolvable is fine -- ``over`` just stays ``None``,
    which is what a gridded pair wants anyway.
    """
    from types import SimpleNamespace

    from ocean_skill.comparison import Comparison

    comp = Comparison(reference="unresolvable_ref", test=test_name, variable="nitrate")
    field = _pacific_test()
    comp._aligned = xr.Dataset(
        {"test": field, "reference": field, "difference": field * 0},
        attrs={"lon_convention": "0-360"},
    )

    entry_meta = {"geospatial_lon_min": 77.4, "geospatial_lat_min": -20.0}
    entry_meta.update(
        {"geospatial_lon_max": 316.2, "geospatial_lat_max": 20.0}
    )
    if domain_outline is not None:
        entry_meta["domain_outline"] = domain_outline
    entry = SimpleNamespace(metadata=entry_meta)

    def fake_resolve(source):
        if source == test_name:
            return entry
        raise KeyError(source)

    monkeypatch.setattr("ocean_skill.catalog.resolve", fake_resolve, raising=True)
    return comp


def _capture_rendered_domain(monkeypatch):
    """Patch the registry's ``render`` to capture the spec's ``domain`` option.

    Renders nothing -- the plumbing under test is which value reaches the spec's
    options, not what a figure of it looks like.
    """
    captured = {}

    def fake_render(spec, **kwargs):
        captured["domain"] = spec.options.get("domain")
        return captured

    monkeypatch.setattr("ocean_skill.plot.registry.render", fake_render, raising=True)
    return captured


def test_comparison_plot_prefers_the_stored_outline_over_the_bbox(monkeypatch):
    ring = [[80.0, -15.0], [310.0, -10.0], [305.0, 15.0], [85.0, 12.0]]
    comp = _stub_comparison(monkeypatch, domain_outline=ring)
    captured = _capture_rendered_domain(monkeypatch)

    comp.plot()

    domain = captured["domain"]
    assert domain is not None and not isinstance(domain, tuple)
    assert np.asarray(domain).shape[1] == 2


def test_comparison_plot_falls_back_to_the_bbox_without_an_outline(monkeypatch):
    comp = _stub_comparison(monkeypatch, domain_outline=None)
    captured = _capture_rendered_domain(monkeypatch)

    comp.plot()

    assert captured["domain"] == (77.4, -20.0, 316.2, 20.0)


def test_comparison_plot_domain_none_still_suppresses_it(monkeypatch):
    comp = _stub_comparison(monkeypatch, domain_outline=[[1.0, 1.0], [2, 1], [2, 2]])
    captured = _capture_rendered_domain(monkeypatch)

    comp.plot(domain=None)

    assert captured["domain"] is None


def test_comparison_plot_a_user_bbox_overrides_the_outline(monkeypatch):
    comp = _stub_comparison(
        monkeypatch, domain_outline=[[1.0, 1.0], [2, 1], [2, 2]]
    )
    captured = _capture_rendered_domain(monkeypatch)

    comp.plot(domain=(0.0, 0.0, 5.0, 5.0))

    assert captured["domain"] == (0.0, 0.0, 5.0, 5.0)


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


def test_tiles_for_keeps_tiles_for_a_domain_nowhere_near_the_dateline():
    """A -160..-125 domain (e.g. a Pacific coast model) is a tied span, not a
    straddle -- it must keep its basemap, not get downgraded on float noise.
    """
    from ocean_skill.plot.holoviews_renderer import _tiles_for

    field = _rectilinear_field(-160.0, -125.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _tiles_for(True, field) is True
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
