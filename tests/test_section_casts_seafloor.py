"""Cast marks and the model's seafloor on a section built from CTD casts.

A comparison section built from discrete casts
(``select={"transect": {"from": "reference"}}``) draws each cast as a labelled line
and the test model's bathymetry as a dense line along the cast path. The renderers
only draw; what they are handed is two keys on the ``section_row`` plot item,
``cast_labels`` and ``seafloor``, resolved by
:func:`ocean_skill.comparison._section_extras` from ``casts=``/``bathymetry=`` on
``plot()``. These tests pin that data side, in the house layers: the label stripping,
:meth:`Comparison.seafloor` against a synthetic ROMS-like run, and the item dict
``plot()`` builds (captured at the renderer registry, so no figure is drawn).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import roms
from ocean_skill.align import ALONG_DIM, _haversine_km
from ocean_skill.comparison import (
    BathymetryNotFound,
    _cast_labels,
    _normalize_seafloor,
    _section_extras,
)
from ocean_skill.field import field
from ocean_skill.plot.section import seafloor_line
from tests import test_section_comparison as _sc
from tests.test_section_comparison import (
    _CAST_LONLATS,
    _CAST_META,
    HC,
    N_S,
    VAR,
    _cast,
    _climatology,
    _stretch,
)

# the shared fixture, re-exported so this module's tests can ask for it by name
patched_sources = _sc.patched_sources

SELECT = {"transect": {"from": "reference"}, "depth": [50.0, 200.0]}
ROMS_META = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}


def _fine_roms_run(n: int = 120) -> xr.Dataset:
    """Build a ROMS-like run on a fine ``n`` x ``n`` rho grid, ``h`` rising northward.

    Fine enough (~4 km cells) that a nearest-cell sample of the seafloor sits within
    a fraction of a percent of the straight-line cumulative distance the section
    itself reports -- the comparison the along-coordinate test makes.
    """
    h = np.repeat(np.linspace(30.0, 2000.0, n)[:, None], n, axis=1)
    sigma_r = (np.arange(1, N_S + 1) - N_S - 0.5) / N_S
    sigma_w = np.linspace(-1, 0, N_S + 1)
    lon_2d, lat_2d = np.meshgrid(
        np.linspace(-95.5, -92.5, n), np.linspace(23.5, 28.5, n)
    )
    ds = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((n, n))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
        },
        coords={
            "lon": (("eta_rho", "xi_rho"), lon_2d),
            "lat": (("eta_rho", "xi_rho"), lat_2d),
        },
    )
    ds = roms.add_depth_coord(
        ds, {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    )
    ds = ds.assign({VAR: 20.0 + 0.002 * ds["z_rho"]})
    ds[VAR].attrs["units"] = "degC"
    return ds


def _register(patched_sources, *, test, test_meta, casts=True):
    sources = {"model": (test, test_meta)}
    for i, (lon, lat) in enumerate(_CAST_LONLATS, start=1):
        sources[f"ctd_station_HV{i}"] = (_cast(lon, lat, base=15.0 + i), _CAST_META)
    patched_sources(sources)
    return [f"ctd_station_HV{i}" for i in range(1, len(_CAST_LONLATS) + 1)]


def _cast_comparison(patched_sources, *, test=None, meta=None):
    test = _fine_roms_run() if test is None else test
    names = _register(
        patched_sources, test=test, test_meta=ROMS_META if meta is None else meta
    )
    out = osk.compare(
        reference=names,
        test="model",
        variables=[VAR],
        select=SELECT,
        aggregate={"time": "mean"},
        cache=False,
    )
    return out.comparisons[0], names


@pytest.fixture
def roms_casts(patched_sources):
    return _cast_comparison(patched_sources)


@pytest.fixture
def no_bathy_casts(patched_sources):
    """Casts against a rectilinear climatology that has no ``h``."""
    return _cast_comparison(patched_sources, test=_climatology(), meta={})


@pytest.fixture
def captured(monkeypatch):
    """Capture the spec ``plot()`` hands the renderer, drawing nothing."""
    seen = {}

    def fake_render(spec, renderer="matplotlib"):
        seen["spec"] = spec
        return "figure"

    monkeypatch.setattr("ocean_skill.plot.registry.render", fake_render)
    return seen


def _item(captured):
    (item,) = captured["spec"].items
    return item


# -- label stripping -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (
            ["ctd_station_HV1", "ctd_station_HV10", "ctd_station_HV12"],
            ["HV1", "HV10", "HV12"],
        ),
        (["ctd_station_HV1", "ctd_station_HV3"], ["HV1", "HV3"]),
        (["a-1", "a-2"], ["1", "2"]),
        (["cast 1", "cast 2"], ["1", "2"]),
        (["line.A", "line.B"], ["A", "B"]),
        # no common prefix: nothing to strip
        (["alpha", "beta"], ["alpha", "beta"]),
        # a prefix that stops mid-token backs off to the last separator
        (["st_abc", "st_abd"], ["abc", "abd"]),
        (["abc", "abd"], ["abc", "abd"]),
        # one name is a prefix of the other -> an empty label -> full names
        (["ctd_", "ctd_x"], ["ctd_", "ctd_x"]),
        # identical names strip to nothing -> full names
        (["st_", "st_"], ["st_", "st_"]),
        (["ctd_1", "ctd_1_b"], ["1", "1_b"]),
        (["only_one"], ["only_one"]),
    ],
)
def test_cast_labels(names, expected):
    assert _cast_labels(names) == expected


# -- casts= --------------------------------------------------------------------------


def test_casts_default_on_for_a_cast_built_section(roms_casts, captured):
    c, _ = roms_casts
    assert c.family == "section_row"
    c.plot()
    assert _item(captured)["cast_labels"] == ["HV1", "HV2", "HV3"]


def test_casts_false_omits_the_key(roms_casts, captured):
    c, _ = roms_casts
    c.plot(casts=False)
    assert "cast_labels" not in _item(captured)


def test_casts_true_is_the_default_labels(roms_casts, captured):
    c, _ = roms_casts
    c.plot(casts=True)
    assert _item(captured)["cast_labels"] == ["HV1", "HV2", "HV3"]


def test_casts_list_gives_the_labels(roms_casts, captured):
    c, _ = roms_casts
    c.plot(casts=["A", "B", "C"])
    assert _item(captured)["cast_labels"] == ["A", "B", "C"]


def test_casts_dict_relabels_some_and_keeps_the_rest(roms_casts, captured):
    c, names = roms_casts
    c.plot(casts={names[1]: "mid"})
    assert _item(captured)["cast_labels"] == ["HV1", "mid", "HV3"]


def test_casts_list_of_the_wrong_length_refused(roms_casts):
    c, _ = roms_casts
    with pytest.raises(ValueError, match=r"2 label.* for 3 cast"):
        c.plot(casts=["A", "B"])


def test_casts_dict_naming_an_unknown_cast_refused(roms_casts):
    c, _ = roms_casts
    with pytest.raises(ValueError, match="not among this section's casts"):
        c.plot(casts={"nope": "x"})


def test_casts_true_on_a_section_cut_from_a_grid_refused(patched_sources):
    patched_sources(
        {
            "model": (_fine_roms_run(), ROMS_META),
            "woa_ref": (_climatology(), {}),
        }
    )
    c = osk.Comparison(
        test="model",
        reference="woa_ref",
        variable=VAR,
        select={
            "transect": {"waypoints": [[-95.0, 24.0], [-93.0, 28.0]]},
            "depth": [50.0, 200.0],
        },
        cache=False,
    )
    with pytest.raises(ValueError, match="no casts to mark"):
        c.plot(casts=True)
    with pytest.raises(ValueError, match="no casts to mark"):
        c.plot(casts=["a", "b"])


def test_a_gridded_section_gets_no_extras_by_default(patched_sources, captured):
    patched_sources(
        {
            "model": (_fine_roms_run(), ROMS_META),
            "woa_ref": (_climatology(), {}),
        }
    )
    c = osk.Comparison(
        test="model",
        reference="woa_ref",
        variable=VAR,
        select={
            "transect": {"waypoints": [[-95.0, 24.0], [-93.0, 28.0]]},
            "depth": [50.0, 200.0],
        },
        cache=False,
    )
    c.plot()
    item = _item(captured)
    assert "cast_labels" not in item
    assert "seafloor" not in item
    # an explicit request still works on any path section
    c.plot(bathymetry=True)
    assert _item(captured)["seafloor"].dims == (ALONG_DIM,)


def test_a_non_section_adds_no_keys_and_refuses_explicit_requests():
    stub = SimpleNamespace(family="field_row")
    assert _section_extras(stub, None, None) == {}
    assert _section_extras(stub, False, False) == {}
    with pytest.raises(ValueError, match="draws as 'field_row'"):
        _section_extras(stub, True, None)
    with pytest.raises(ValueError, match="draws as 'field_row'"):
        _section_extras(stub, None, True)


# -- Comparison.seafloor() -----------------------------------------------------------


def test_seafloor_is_positive_down_along_the_cast_path(roms_casts):
    c, _ = roms_casts
    sea = c.seafloor()
    assert sea.dims == (ALONG_DIM,)
    # many more samples than the three casts: dense, not one per cast
    assert sea.sizes[ALONG_DIM] > 30
    # ROMS h is positive-down already and rises northward across this grid
    assert float(sea.min()) >= 30.0 - 1e-9
    assert float(sea.max()) <= 2000.0 + 1e-9
    assert np.all(np.diff(np.asarray(sea)) >= -1e-9)
    assert sea.attrs["positive"] == "down"
    assert sea.attrs["units"] == "m"


def test_seafloor_finds_h_kept_as_a_coordinate(patched_sources, captured):
    # the ROMS reader attaches ``h`` (and mask_rho) as coordinates, not data
    # variables; the default bathymetry= must still find it, not skip it silently
    run = _fine_roms_run()
    run = run.set_coords([v for v in ("h", "mask_rho") if v in run.data_vars])
    assert "h" not in run.data_vars
    c, _ = _cast_comparison(patched_sources, test=run)
    sea = c.seafloor()
    assert sea.sizes[ALONG_DIM] > 30
    assert float(sea.min()) >= 30.0 - 1e-9
    c.plot()
    assert "seafloor" in _item(captured)


def test_seafloor_along_starts_at_zero_and_carries_path_coords(roms_casts):
    c, _ = roms_casts
    sea = c.seafloor()
    along = np.asarray(sea[ALONG_DIM])
    assert along[0] == 0.0
    assert np.all(np.diff(along) > 0)
    for name in ("path_lon", "path_lat"):
        assert sea[name].dims == (ALONG_DIM,)
        assert np.isfinite(np.asarray(sea[name])).all()
    # no stray non-path dimension or coordinate survives
    assert set(sea.coords) == {ALONG_DIM, "path_lon", "path_lat"}


def test_seafloor_along_agrees_with_the_sections_own_along_at_each_cast(roms_casts):
    c, _ = roms_casts
    sea = c.seafloor()
    section_along = np.asarray(c.aligned[ALONG_DIM])
    cast_lon = np.asarray(c.aligned["reference"]["lon"])
    cast_lat = np.asarray(c.aligned["reference"]["lat"])
    assert len(section_along) == len(_CAST_LONLATS)
    tolerance = 0.01 * float(section_along[-1])
    for lon, lat, expected in zip(cast_lon, cast_lat, section_along, strict=True):
        km = _haversine_km(
            np.asarray(sea["path_lon"]), np.asarray(sea["path_lat"]), lon, lat
        )
        nearest = int(np.argmin(km))
        assert km[nearest] < 0.01 * float(section_along[-1])
        assert abs(float(sea[ALONG_DIM][nearest]) - expected) < tolerance
    # and the whole length is the section's
    assert abs(float(sea[ALONG_DIM][-1]) - section_along[-1]) < tolerance


def test_seafloor_spacing_km_sets_the_sampling(roms_casts):
    c, _ = roms_casts
    coarse = c.seafloor(spacing_km=60.0)
    fine = c.seafloor(spacing_km=10.0)
    assert fine.sizes[ALONG_DIM] > coarse.sizes[ALONG_DIM]
    # the same bottom either way, to the grid's own resolution
    assert abs(float(coarse[ALONG_DIM][-1]) - float(fine[ALONG_DIM][-1])) < 15.0


def test_seafloor_feeds_seafloor_line(roms_casts):
    """The contract is the renderers' own: seafloor_line accepts it unchanged."""
    from ocean_skill.plot.section import SectionGeometry, prepare_section_row

    c, _ = roms_casts
    sea = c.seafloor()
    _, geometry = prepare_section_row(c.aligned, "distance")
    assert isinstance(geometry, SectionGeometry)
    x, depth = seafloor_line(sea, geometry)
    np.testing.assert_allclose(x, np.asarray(sea[ALONG_DIM]))
    np.testing.assert_allclose(depth, np.asarray(sea))


def test_seafloor_without_bathymetry_raises_a_lookup_error(no_bathy_casts):
    c, _ = no_bathy_casts
    with pytest.raises(
        BathymetryNotFound, match=r"'model'.*bathymetry=<Field>.*bathymetry=False"
    ) as info:
        c.seafloor()
    assert isinstance(info.value, LookupError)


def test_seafloor_finds_a_cf_sea_floor_depth_variable(monkeypatch):
    from ocean_skill.comparison import _bathymetry_variable

    ds = _fine_roms_run(8).rename({"h": "sea_floor_depth_below_geoid"})
    odd = _fine_roms_run(8).rename({"h": "total_depth"})
    odd["total_depth"].attrs["standard_name"] = "sea_floor_depth_below_sea_surface"
    for source, expected in (
        (ds, "sea_floor_depth_below_geoid"),
        (odd, "total_depth"),  # found by standard_name, not by its name
    ):
        monkeypatch.setattr(osk, "read", lambda name, _ds=source, **kw: _ds)
        assert _bathymetry_variable("whatever") == expected


def test_seafloor_refused_on_a_slab(patched_sources):
    from tests.test_slab_section import MEAN_LON, ROMS_BOX

    patched_sources(
        {
            "model": (_fine_roms_run(30), ROMS_META),
            "woa_ref": (_climatology(), {}),
        }
    )
    c = osk.Comparison(
        test="model",
        reference="woa_ref",
        variable=VAR,
        select={**ROMS_BOX, "depth": [50.0, 200.0]},
        aggregate=MEAN_LON,
        cache=False,
    )
    assert c.is_section
    with pytest.raises(ValueError, match="slab"):
        c.seafloor()
    with pytest.raises(ValueError, match="slab"):
        _section_extras(c, None, True)


# -- bathymetry= ---------------------------------------------------------------------


def test_bathymetry_default_on_for_a_cast_built_section(roms_casts, captured):
    c, _ = roms_casts
    c.plot()
    sea = _item(captured)["seafloor"]
    xr.testing.assert_identical(sea, c.seafloor())


def test_bathymetry_false_omits_the_key(roms_casts, captured):
    c, _ = roms_casts
    c.plot(bathymetry=False)
    assert "seafloor" not in _item(captured)
    assert "cast_labels" in _item(captured)


def test_bathymetry_default_is_silently_skipped_without_an_h(no_bathy_casts, captured):
    import warnings

    c, _ = no_bathy_casts
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c.plot()
    item = _item(captured)
    assert "seafloor" not in item
    assert item["cast_labels"] == ["HV1", "HV2", "HV3"]
    assert not [w for w in caught if "bathymetry" in str(w.message)]


def test_bathymetry_true_raises_without_an_h(no_bathy_casts):
    c, _ = no_bathy_casts
    with pytest.raises(BathymetryNotFound):
        c.plot(bathymetry=True)


def test_bathymetry_field_is_used_as_given(roms_casts, captured):
    c, _ = roms_casts
    mine = field(
        "model",
        "h",
        select={
            "transect": {
                "waypoints": [list(p) for p in _CAST_LONLATS],
                "spacing_km": 25.0,
            }
        },
        cache=False,
    )
    c.plot(bathymetry=mine)
    sea = _item(captured)["seafloor"]
    assert sea.dims == (ALONG_DIM,)
    assert sea.sizes[ALONG_DIM] == mine.data.sizes[ALONG_DIM]
    assert sea[ALONG_DIM][0] == 0.0
    np.testing.assert_allclose(np.asarray(sea), np.asarray(mine.data))


def _bathy(values, *, along=None, **coords):
    values = np.asarray(values, dtype="float64")
    along = np.arange(values.size) * 10.0 if along is None else along
    return xr.DataArray(
        values,
        dims=(ALONG_DIM,),
        coords={ALONG_DIM: along, **{k: (ALONG_DIM, v) for k, v in coords.items()}},
    )


def test_bathymetry_dataarray_is_validated_and_normalized(roms_casts, captured):
    c, _ = roms_casts
    given = _bathy(
        [100.0, 200.0, 300.0],
        along=[5.0, 15.0, 25.0],
        path_lon=[-95.0, -94.0, -93.0],
        path_lat=[24.0, 26.0, 28.0],
    )
    c.plot(bathymetry=given)
    sea = _item(captured)["seafloor"]
    # rebased to start at the first cast
    np.testing.assert_allclose(sea[ALONG_DIM], [0.0, 10.0, 20.0])
    np.testing.assert_allclose(sea, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(sea["path_lon"], [-95.0, -94.0, -93.0])


def test_normalize_flips_an_elevation_style_variable_to_positive_down():
    below_sea_level = _bathy([-100.0, -250.0, -40.0])
    np.testing.assert_allclose(
        _normalize_seafloor(below_sea_level), [100.0, 250.0, 40.0]
    )
    up = _bathy([100.0, 250.0, 40.0])
    up.attrs["positive"] = "up"
    np.testing.assert_allclose(_normalize_seafloor(up), [-100.0, -250.0, -40.0])
    positive_down = _bathy([100.0, 250.0, 40.0])
    np.testing.assert_allclose(_normalize_seafloor(positive_down), [100.0, 250.0, 40.0])


def test_normalize_falls_back_to_lon_lat_and_squeezes_a_singleton():
    da = _bathy([1.0, 2.0], lon=[10.0, 11.0], lat=[0.0, 1.0]).expand_dims(time=[0])
    out = _normalize_seafloor(da)
    assert out.dims == (ALONG_DIM,)
    np.testing.assert_allclose(out["path_lon"], [10.0, 11.0])
    np.testing.assert_allclose(out["path_lat"], [0.0, 1.0])


def test_normalize_refuses_the_wrong_shape(roms_casts):
    two_d = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", ALONG_DIM),
        coords={ALONG_DIM: [0.0, 1.0, 2.0]},
    )
    with pytest.raises(ValueError, match="one-dimensional along 'along'"):
        _normalize_seafloor(two_d)
    other_dim = xr.DataArray([1.0, 2.0], dims="x", coords={"x": [0.0, 1.0]})
    with pytest.raises(ValueError, match="one-dimensional along 'along'"):
        _normalize_seafloor(other_dim)
    no_along = xr.DataArray([1.0, 2.0], dims=ALONG_DIM)
    with pytest.raises(ValueError, match="'along' coordinate"):
        _normalize_seafloor(no_along)
    with pytest.raises(TypeError, match="DataArray or a Field"):
        _normalize_seafloor([1.0, 2.0])
    c, _ = roms_casts
    with pytest.raises(ValueError, match="one-dimensional"):
        c.plot(bathymetry=two_d)


# -- ComparisonSet -------------------------------------------------------------------


def test_comparison_set_adds_the_keys_to_every_cast_row(roms_casts, captured):
    c, names = roms_casts
    # a second, distinct comparison (the casts in the opposite order) so the set
    # does not pool it away as a duplicate of the first
    (other,) = osk.compare(
        reference=names[::-1],
        test="model",
        variables=[VAR],
        select=SELECT,
        aggregate={"time": "mean"},
        cache=False,
    ).comparisons
    osk.ComparisonSet([c, other]).plot()
    first, second = captured["spec"].items
    assert first["cast_labels"] == ["HV1", "HV2", "HV3"]
    assert second["cast_labels"] == ["HV3", "HV2", "HV1"]
    for item in (first, second):
        assert item["seafloor"].dims == (ALONG_DIM,)


def test_comparison_set_forwards_the_options(roms_casts, captured):
    c, _ = roms_casts
    osk.ComparisonSet([c]).plot(casts=["A", "B", "C"], bathymetry=False)
    (item,) = captured["spec"].items
    assert item["cast_labels"] == ["A", "B", "C"]
    assert "seafloor" not in item
    osk.ComparisonSet([c]).plot(casts=False)
    (item,) = captured["spec"].items
    assert "cast_labels" not in item
    assert "seafloor" in item
