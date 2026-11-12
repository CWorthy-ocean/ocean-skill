"""``horizontal_gradient``: |grad| of a 2-D field, per km, on any grid shape.

Fields are built linear in *distance* (``q = a * km``), so the exact answer is the
constant ``a`` wherever both centered-difference neighbors exist.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import units
from ocean_skill.horizontal import calculate_horizontal_gradient

KM_PER_DEG = 6371.0088 * np.pi / 180
A = 0.02  # mmol/m^3 per km


def _rect(lon, lat, *, field="north", name="oxygen", units_="mmol/m^3"):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    y_km = (lat - lat[0]) * KM_PER_DEG
    data = np.broadcast_to(A * y_km[:, None], (lat.size, lon.size)).copy()
    return xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        name=name,
        attrs={"units": units_},
    )


def _interior(out):
    return out.isel(lat=slice(1, -1), lon=slice(1, -1))


def test_rectilinear_linear_field_has_constant_gradient():
    da = _rect(np.linspace(-150, -140, 21), np.linspace(10, 20, 21))
    out = calculate_horizontal_gradient(da, variable="oxygen")
    inner = _interior(out)
    np.testing.assert_allclose(inner.values, A, rtol=5e-3)
    # the centered difference needs both neighbors: every edge cell is NaN
    assert out.isel(lat=0).isnull().all() and out.isel(lat=-1).isnull().all()
    assert out.isel(lon=0).isnull().all() and out.isel(lon=-1).isnull().all()


def test_curvilinear_rotated_grid_is_rotation_invariant():
    """A grid rotated 30 degrees from north: |grad| is still ``a``."""
    ny, nx = 25, 30
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    th = np.deg2rad(30)
    step = 0.02  # degrees: small, so the rotated grid stays near-orthogonal
    x = step * (i * np.cos(th) - j * np.sin(th))
    y = step * (i * np.sin(th) + j * np.cos(th))
    lon, lat = -125 + x / np.cos(np.deg2rad(10)), 10 + y
    # distance along true north from the first row, in km (small region: planar)
    q = A * (lat - lat[0, 0]) * KM_PER_DEG
    da = xr.DataArray(
        q,
        dims=("eta_rho", "xi_rho"),
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon, {"units": "degrees_east"}),
            "lat_rho": (("eta_rho", "xi_rho"), lat, {"units": "degrees_north"}),
        },
        name="oxygen",
        attrs={"units": "mmol/m^3"},
    )
    out = calculate_horizontal_gradient(da, variable="oxygen")
    np.testing.assert_allclose(out.values[1:-1, 1:-1], A, rtol=5e-3)


def test_land_cell_blanks_only_its_neighbors():
    da = _rect(np.linspace(-150, -140, 11), np.linspace(10, 20, 11))
    da[5, 5] = np.nan
    out = calculate_horizontal_gradient(da)
    bad = out.isnull().values
    expect = np.zeros_like(bad)
    expect[[0, -1], :] = expect[:, [0, -1]] = True  # edges
    # the land cell itself stays land, and each of its four neighbors loses a side
    expect[5, 5] = expect[4, 5] = expect[6, 5] = expect[5, 4] = expect[5, 6] = True
    np.testing.assert_array_equal(bad, expect)


def test_dateline_crossing_matches_in_both_conventions():
    lon180 = np.linspace(170, 190, 21)
    lat = np.linspace(0, 10, 11)
    pm180 = calculate_horizontal_gradient(_rect(((lon180 + 180) % 360) - 180, lat))
    e360 = calculate_horizontal_gradient(_rect(lon180 % 360, lat))
    np.testing.assert_allclose(pm180.values, e360.values, equal_nan=True)
    inner = _interior(pm180)
    np.testing.assert_allclose(inner.values, A, rtol=5e-3)  # no seam artifact


def test_dask_input_stays_lazy_and_broadcasts_over_time():
    pytest.importorskip("dask")
    da = _rect(np.linspace(-150, -140, 9), np.linspace(10, 20, 9))
    da = xr.concat([da, 2 * da, 3 * da], "time").assign_coords(time=[0, 1, 2])
    out = calculate_horizontal_gradient(da.chunk({"time": 1}))
    assert out.chunks is not None
    assert out.dims == da.dims
    got = _interior(out).compute()
    for t, k in enumerate((1, 2, 3)):
        np.testing.assert_allclose(got.isel(time=t).values, k * A, rtol=5e-3)


def test_transect_and_point_raise():
    n = 6
    transect = xr.DataArray(
        np.arange(n, dtype=float),
        dims="station",
        coords={
            "lon": ("station", np.linspace(-150, -140, n), {"units": "degrees_east"}),
            "lat": ("station", np.linspace(10, 20, n), {"units": "degrees_north"}),
        },
        name="oxygen",
    )
    with pytest.raises(ValueError, match="2-D horizontal field"):
        calculate_horizontal_gradient(transect)
    point = xr.DataArray(
        1.0,
        coords={
            "lon": ((), -145.0, {"units": "degrees_east"}),
            "lat": ((), 15.0, {"units": "degrees_north"}),
        },
        name="oxygen",
    )
    with pytest.raises(ValueError, match="2-D horizontal field"):
        calculate_horizontal_gradient(point)
    with pytest.raises(ValueError, match="longitude and latitude"):
        calculate_horizontal_gradient(xr.DataArray(np.ones((3, 3)), dims=("a", "b")))


@pytest.mark.parametrize(
    "given,expected",
    [
        ("mmol/m^3", "mmol/m^3/km"),
        ("mmol m-3", "(mmol m-3)/km"),
        ("micromoles_per_kilogram", "micromoles_per_kilogram/km"),
        ("", "1/km"),
    ],
)
def test_units_are_per_km_and_parse(given, expected):
    out = calculate_horizontal_gradient(
        _rect(np.linspace(0, 5, 6), np.linspace(0, 5, 6), units_=given),
        variable="oxygen",
    )
    assert out.attrs["units"] == expected
    assert units.parse(out.attrs["units"]) is not None
    assert "km" in units.display(out.attrs["units"])


def test_name_and_attrs():
    da = _rect(np.linspace(0, 5, 6), np.linspace(0, 5, 6))
    da.attrs["long_name"] = "dissolved oxygen"
    out = calculate_horizontal_gradient(da, variable="oxygen")
    assert out.name == "horizontal_gradient_of_oxygen"
    assert out.attrs["long_name"] == "horizontal gradient magnitude of dissolved oxygen"
    assert out.attrs["horizontal_gradient_variable"] == "oxygen"
    assert {"lon", "lat"} <= set(out.coords)
    # no spec string: labeled by the field's own name
    assert calculate_horizontal_gradient(da).name == "horizontal_gradient_of_oxygen"


def test_spec_names_reports_the_variable():
    from ocean_skill import operators

    spec = {"calculate": "horizontal_gradient", "variable": "oxygen"}
    assert operators.spec_names(spec) == operators.spec_names("oxygen")


# -- end to end: the real prepare_source / compare path, both renderers --------------

SPEC = {"calculate": "horizontal_gradient", "variable": "oxygen"}
NY, NX, NS = 30, 40, 20
LAT = np.linspace(-10.0, 10.0, NY)
LON = np.linspace(160.0, 200.0, NX)  # crosses the dateline in 0-360


def _roms_o2(slopes):
    """Build a ROMS run: O2 = 100 - depth/10 + slope * northward km, per snapshot."""
    from ocean_skill import roms

    nt = len(slopes)
    sig = (np.arange(NS) - NS + 0.5) / NS
    sigw = np.linspace(-1, 0, NS + 1)
    lon = np.tile(LON, (NY, 1))
    lat = np.tile(LAT[:, None], (1, NX))
    h = 2000.0
    z = sig[:, None, None] * h * np.ones((NY, NX))  # hc = 0: z = sigma * h
    y_km = (lat - LAT[0]) * KM_PER_DEG
    o2 = np.stack([100.0 + z / 10.0 + s * y_km for s in slopes])
    raw = xr.Dataset(
        {
            "zeta": (("time", "eta_rho", "xi_rho"), np.zeros((nt, NY, NX))),
            "O2": (("time", "s_rho", "eta_rho", "xi_rho"), o2, {"units": "mmol/m^3"}),
            "h": (("eta_rho", "xi_rho"), np.full((NY, NX), h)),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((NY, NX))),
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "Cs_r": ("s_rho", sig),
            "sigma_r": ("s_rho", sig),
            "Cs_w": ("s_w", sigw),
            "sigma_w": ("s_w", sigw),
            "ocean_time": ("time", np.arange(nt) * 86400.0 * 30),
        }
    )
    meta = {
        "model": "roms",
        "self_contained_grid": True,
        "vertical": {"s_dim": "s_rho", "hc": 0.0, "Vtransform": 2},
        "reference_date": "2011-01-01",
    }
    return roms.standardize(raw, meta), meta


def _woa_o2(slope):
    """Build a WOA-like 1-degree climatology in umol/kg on standard levels."""
    lat = np.arange(-12.0, 12.5, 1.0)
    lon = np.arange(158.0, 202.5, 1.0)
    depth = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    y_km = (lat - lat[0]) * KM_PER_DEG
    o2 = 100.0 - depth[:, None, None] / 10.0 + slope * y_km[None, :, None]
    o2 = np.broadcast_to(o2, (depth.size, lat.size, lon.size)).copy()
    return xr.Dataset(
        {
            "o_an": (
                ("depth", "lat", "lon"),
                o2,
                {
                    "units": "micromoles_per_kilogram",
                    "standard_name": (
                        "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"
                    ),
                },
            )
        },
        coords={
            "depth": ("depth", depth, {"units": "m", "positive": "down"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
    )


@pytest.fixture
def sources(monkeypatch):
    """Register ``{name: (Dataset, metadata)}`` as readable catalog sources."""
    from types import SimpleNamespace

    import ocean_skill as osk
    import ocean_skill.sources as sources_module
    from ocean_skill import catalog, comparison

    def install(table):
        monkeypatch.setattr(osk, "read", lambda n, **kw: table[n][0])
        monkeypatch.setattr(sources_module, "read", lambda n, **kw: table[n][0])
        monkeypatch.setattr(
            catalog, "resolve", lambda n: SimpleNamespace(metadata=table[n][1])
        )
        monkeypatch.setattr(comparison, "_domain_of", lambda n: None)
        monkeypatch.setattr(comparison, "_outline_of", lambda n, convention=None: None)

    return install


def _interior_roms(da):
    return da.isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1))


def test_roms_gradient_at_300_m_is_the_slope_of_the_time_mean(sources):
    from ocean_skill.comparison import prepare_source

    sources({"run": _roms_o2([3 * A, -A])})  # time mean slope = A
    da, _ = prepare_source(
        "run",
        SPEC,
        {"depth": 300},
        {"time": "mean"},
        use_cache=False,
        depth_method="interp",
    )
    assert da.name == "horizontal_gradient_of_oxygen"
    assert "time" not in da.dims
    np.testing.assert_allclose(_interior_roms(da), A, rtol=5e-3)


def test_snapshots_whose_slopes_cancel_give_no_gradient(sources):
    from ocean_skill.comparison import prepare_source

    sources({"run": _roms_o2([A, -A])})
    da, _ = prepare_source(
        "run",
        SPEC,
        {"depth": 300},
        {"time": "mean"},
        use_cache=False,
        depth_method="interp",
    )
    np.testing.assert_allclose(_interior_roms(da), 0.0, atol=1e-9)


def test_woa_umol_per_kg_is_converted_before_the_gradient(sources):
    from ocean_skill.comparison import prepare_source

    sources({"woa": (_woa_o2(A), {})})
    da, _ = prepare_source("woa", SPEC, {"depth": 300}, None, use_cache=False)
    assert "km" in da.attrs["units"] and "mmol" in da.attrs["units"]
    np.testing.assert_allclose(
        da.isel(lat=slice(1, -1), lon=slice(1, -1)), 1.025 * A, rtol=5e-3
    )


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_model_against_woa_compares_and_plots(sources, renderer):
    import warnings

    import ocean_skill as osk

    sources({"run": _roms_o2([A, A]), "woa": (_woa_o2(A), {})})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = osk.Comparison(
            test="run",
            reference="woa",
            variable=SPEC,
            select={"depth": 300},
            aggregate={"time": "mean"},
            depth_method="interp",
            cache=False,
        )
        out = c.plot(renderer=renderer)
    if renderer == "holoviews":
        import holoviews as hv

        hv.render(out)


def test_a_spec_without_a_variable_names_what_is_missing():
    from ocean_skill import operators

    with pytest.raises(KeyError, match="horizontal_gradient needs variable="):
        operators.layer_calculator({"calculate": "horizontal_gradient"})


def test_direct_use_on_native_s_levels_is_refused():
    from ocean_skill import operators

    ds, _ = _roms_o2([A])
    with pytest.raises(ValueError, match="native s-levels"):
        operators.resolve_variable(ds, SPEC)
    surface = ds.isel(s_rho=-1)  # one level: a layer, if a terrain-following one
    assert operators.resolve_variable(surface, SPEC) is not None
