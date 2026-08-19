"""Tests for thickness-weighted averaging over a depth band.

The motivating case is satellite chlorophyll: the sensor integrates roughly the first
optical depth, so the model counterpart is a fixed-depth average, not a model *level*.
A ROMS top cell ranges from ~0.2 m on the shelf to ~17 m offshore on a real grid, so
"surface" means a different depth in every column, while a 0-10 m band means the same
thing everywhere.

Interpolation cannot do this job: data lives at cell *centres*, and the shallowest
centre is metres below the surface in deep water, so a 0-10 m target grid is mostly
NaN offshore. Averaging over native cells with partial weights at the boundary has no
such gap, because interfaces start at the free surface by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms
from ocean_skill.comparison import _depth_label, _prepare, is_depth_band

N = 20
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


@pytest.fixture
def roms_column():
    """Build a grid spanning shelf to abyss, exercising both weighting regimes."""
    h = np.array([[20.0, 100.0], [1000.0, 5000.0]])
    sigma_r = (np.arange(1, N + 1) - N - 0.5) / N
    sigma_w = np.linspace(-1, 0, N + 1)
    ds = xr.Dataset(
        {
            "chl": (
                ("s_rho", "eta_rho", "xi_rho"),
                np.ones((N, 2, 2)),
                {"units": "mg/m^3"},
            ),
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((2, 2))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
        },
        coords={
            "lon": (("eta_rho", "xi_rho"), np.array([[-95.0, -94.0], [-95.0, -94.0]])),
            "lat": (("eta_rho", "xi_rho"), np.array([[25.0, 25.0], [26.0, 26.0]])),
        },
    )
    return ds, {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}


def test_interfaces_start_exactly_at_the_surface(roms_column):
    """The reason a band average has no NaN gap where interpolation does."""
    ds, meta = roms_column
    z_w = roms.add_interface_coord(ds, meta)["z_w"]
    assert float(z_w.isel(s_w=-1).max()) == pytest.approx(0.0, abs=1e-9)


def test_a_uniform_field_averages_to_itself(roms_column):
    """The weights must sum correctly, in every depth regime."""
    ds, meta = roms_column
    out = roms.depth_average(ds, meta, 0.0, 10.0)["chl"]
    assert np.allclose(out.values, 1.0)
    assert "s_rho" not in out.dims


def test_weights_are_the_overlap_with_the_band(roms_column):
    """A linear profile integrates to the band midpoint under exact weighting."""
    ds, meta = roms_column
    z_w = roms.add_interface_coord(ds, meta)["z_w"]
    depth = -0.5 * (
        z_w.isel(s_w=slice(1, None)).values + z_w.isel(s_w=slice(None, -1)).values
    )
    ds = ds.assign(chl=(("s_rho", "eta_rho", "xi_rho"), depth))  # value == its depth
    out = roms.depth_average(ds, meta, 0.0, 10.0)["chl"]
    # On the shelf many thin cells resolve the band, so the mean approaches 5 m.
    assert float(out.isel(eta_rho=0, xi_rho=0)) == pytest.approx(5.0, abs=0.6)


def test_the_band_is_finite_everywhere_including_deep_water(roms_column):
    """Interpolating to 0-10 m would be NaN offshore; the shallowest centre is deep."""
    ds, meta = roms_column
    band = roms.depth_average(ds, meta, 0.0, 10.0)["chl"]
    assert np.isfinite(band.values).all()

    interpolated = roms.to_depth(ds, meta, [0.0, 2.0, 5.0])["chl"]
    deep = interpolated.isel(eta_rho=1, xi_rho=1)
    assert np.isnan(deep.values).any(), "expected NaN above the shallowest cell centre"


def test_units_and_provenance_survive(roms_column):
    ds, meta = roms_column
    out = roms.depth_average(ds, meta, 0.0, 10.0)["chl"]
    assert out.attrs["units"] == "mg/m^3"
    assert "0.0-10.0 m" in out.attrs["depth_average"]


# -- through the compare layer ------------------------------------------------


def test_prepare_accepts_a_depth_band(roms_column):
    ds, meta = roms_column
    da, _ = _prepare(ds, meta, "chl", {"depth": {"min": 0, "max": 10}}, {"Z": "mean"})
    assert set(da.dims) == {"eta_rho", "xi_rho"}
    assert np.isfinite(da.values).all()


def test_a_band_survives_until_aggregate_collapses_it(roms_column):
    """Select narrows, aggregate collapses -- for depth exactly as for time.

    Without a Z entry the band is still standing, and align says so rather than the
    depth having been quietly averaged somewhere else.
    """
    ds, meta = roms_column
    da, _ = _prepare(ds, meta, "chl", {"depth": {"min": 0, "max": 10}}, {})
    assert "s_rho" in da.dims, "a band must not collapse itself"
    assert roms.WEIGHT_COORD in da.coords, "thickness weights must ride along"


def test_a_band_supports_reductions_other_than_mean(roms_column):
    """The payoff of carrying weights: max/std over a band become expressible."""
    ds, meta = roms_column
    for reduction in ("max", "std"):
        da, _ = _prepare(
            ds, meta, "chl", {"depth": {"min": 0, "max": 10}}, {"Z": reduction}
        )
        assert set(da.dims) == {"eta_rho", "xi_rho"}


def test_a_weighted_mean_differs_from_an_unweighted_one(roms_column):
    """Otherwise a 17 m cell would count the same as a 0.5 m one."""
    ds, meta = roms_column
    z_w = roms.add_interface_coord(ds, meta)["z_w"]
    depth = -0.5 * (
        z_w.isel(s_w=slice(1, None)).values + z_w.isel(s_w=slice(None, -1)).values
    )
    ds = ds.assign(chl=(("s_rho", "eta_rho", "xi_rho"), depth))
    band, _ = _prepare(ds, meta, "chl", {"depth": {"min": 0, "max": 60}}, {"Z": "mean"})
    raw, _ = _prepare(ds, meta, "chl", {"depth": {"min": 0, "max": 60}}, {})
    unweighted = raw.mean("s_rho")
    shelf = dict(eta_rho=0, xi_rho=0)
    assert float(band.isel(**shelf)) != pytest.approx(float(unweighted.isel(**shelf)))


def test_several_interpolated_levels_are_all_kept(roms_column):
    """Regression: _prepare used to isel(z=0) and silently drop every level but one."""
    ds, meta = roms_column
    da, _ = _prepare(ds, meta, "chl", {"depth": [20.0, 50.0, 80.0]}, {})
    assert da.sizes["z"] == 3


def test_a_single_interpolated_level_collapses_by_itself(roms_column):
    """A scalar selection drops the axis, as `.sel` does everywhere else."""
    ds, meta = roms_column
    da, _ = _prepare(ds, meta, "chl", {"depth": 50.0}, {})
    assert "z" not in da.dims


def test_surface_mixes_into_a_depth_list(roms_column):
    """``["surface", 50, 80]`` keeps three levels, in that order, honestly labelled.

    "surface" is the native top cell and the numbers are interpolated levels — no
    single vertical operation makes both, so this is the assembled case. The surface
    layer must be *identical* to what the scalar ``"surface"`` request gives, or the
    same word would mean two different fields depending on the spelling around it.
    """
    ds, meta = roms_column
    da, _ = _prepare(ds, meta, "chl", {"depth": ["surface", 50.0, 80.0]}, {})
    assert list(da.z.values) == [0.0, -50.0, -80.0], "requested order, surface first"
    assert da.z.attrs["level_labels"] == ["surface", "50 m", "80 m"]

    surf, _ = _prepare(ds, meta, "chl", {"depth": "surface"}, {})
    assert bool((da.isel(z=0) == surf).all())

    # requested order is kept even with the surface in the middle
    da, _ = _prepare(ds, meta, "chl", {"depth": [50.0, "surface", 80.0]}, {})
    assert list(da.z.values) == [-50.0, 0.0, -80.0]
    assert da.z.attrs["level_labels"] == ["50 m", "surface", "80 m"]


def test_a_mixed_list_survives_until_aggregate_collapses_it(roms_column):
    """Select narrows, aggregate collapses -- same contract as a band."""
    ds, meta = roms_column
    da, _ = _prepare(ds, meta, "chl", {"depth": ["surface", 50.0]}, {"Z": "mean"})
    assert set(da.dims) == {"eta_rho", "xi_rho"}


def test_a_bad_depth_spelling_names_the_accepted_forms(roms_column):
    """``float("bottom")``'s own error names neither the parameter nor the fix."""
    ds, meta = roms_column
    with pytest.raises(ValueError, match=r"cannot read .* as a depth selection"):
        _prepare(ds, meta, "chl", {"depth": ["bottom", 50]}, {})


def test_a_band_and_the_surface_are_different_operations(roms_column):
    """On the shelf the top cell is ~1 m, so a 0-10 m mean must not equal it."""
    ds, meta = roms_column
    z_w = roms.add_interface_coord(ds, meta)["z_w"]
    depth = -0.5 * (
        z_w.isel(s_w=slice(1, None)).values + z_w.isel(s_w=slice(None, -1)).values
    )
    ds = ds.assign(chl=(("s_rho", "eta_rho", "xi_rho"), depth))
    surf, _ = _prepare(ds, meta, "chl", {"depth": "surface"}, {})
    band, _ = _prepare(ds, meta, "chl", {"depth": {"min": 0, "max": 10}}, {"Z": "mean"})
    assert float(surf.isel(eta_rho=0, xi_rho=0)) < float(band.isel(eta_rho=0, xi_rho=0))


def test_observational_levels_inside_the_band_are_averaged():
    """A gridded product reports at standard levels; there are no thicknesses."""
    ds = xr.Dataset(
        {
            "v": (
                ("depth", "lat", "lon"),
                np.array([[[1.0]], [[3.0]], [[100.0]]]),
                {"units": "mg/m^3"},
            )
        },
        coords={"depth": [0.0, 10.0, 50.0], "lat": [1.0], "lon": [1.0]},
    )
    da, actual = _prepare(ds, {}, "v", {"depth": {"min": 0, "max": 10}}, {"Z": "mean"})
    assert float(da.squeeze()) == pytest.approx(2.0)  # the two levels inside, not 50 m
    assert actual == pytest.approx(5.0)


def test_a_band_between_levels_falls_back_to_the_nearest():
    ds = xr.Dataset(
        {"v": (("depth",), np.array([1.0, 9.0]), {"units": "m"})},
        coords={"depth": [0.0, 100.0]},
    )
    da, _ = _prepare(ds, {}, "v", {"depth": {"min": 40, "max": 45}}, {"Z": "mean"})
    assert float(da) == pytest.approx(1.0)


def test_surface_in_a_list_takes_the_nearest_observational_level():
    """On a product at standard levels, "surface" means what it means as a scalar."""
    ds = xr.Dataset(
        {"v": (("depth", "lat", "lon"), np.arange(12.0).reshape(3, 2, 2))},
        coords={"depth": [0.0, 10.0, 50.0], "lat": [1.0, 2.0], "lon": [1.0, 2.0]},
    )
    da, _ = _prepare(ds, {}, "v", {"depth": ["surface", 50]}, {})
    assert list(da.depth.values) == [0.0, 50.0]


@pytest.mark.parametrize(
    ("depth", "label"),
    [
        ("surface", "surface"),
        (100, "100 m"),
        ({"min": 0, "max": 10}, "0-10 m"),
        ([50, 100], "50 m, 100 m"),
        (["surface", 50, 100], "surface, 50 m, 100 m"),
    ],
)
def test_depth_labels_read_well(depth, label):
    assert _depth_label(depth) == label


def test_is_depth_band_only_matches_a_range():
    assert is_depth_band({"min": 0, "max": 10})
    assert not is_depth_band(10)
    assert not is_depth_band("surface")
    assert not is_depth_band({"min": 0})
