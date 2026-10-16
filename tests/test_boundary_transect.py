"""Boundary-row transects: land cells compare as NaN data, and boundary *strips*.

Two things about a ROMS grid's open boundaries:

* a boundary row whose cells include land has NaN ``lon``/``lat`` on those columns;
  :meth:`~ocean_skill.comparison.Comparison._resolved_path` re-spells the test lane's
  path as ``points`` for the reference lane and used to hand the sampler the NaNs, which
  it refuses (the end-to-end case lives in ``test_suite_smoke_fixture.py``);
* ``select={"transect": {"boundary": ["west", "south"]}}`` joins the named boundaries'
  outermost rows/columns into one along-boundary section, continuous around the corner.

The grid below is 6 (``eta_rho``) x 4 (``xi_rho``) with one-degree spacing and
``chl = 100 * eta + xi``, so which cell a column came from is readable off its value.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms
from ocean_skill.align import ALONG_DIM, _haversine_km
from ocean_skill.comparison import Comparison, _fill_missing_positions
from ocean_skill.transect import apply_transect, as_transect, boundary_strip

NY, NX, NS = 6, 4, 8
HC = 250.0
THETA_S, THETA_B = 5.0, 2.0


def _stretch(s):
    c = (1 - np.cosh(THETA_S * s)) / (np.cosh(THETA_S) - 1)
    return (np.exp(THETA_B * c) - 1) / (1 - np.exp(-THETA_B))


@pytest.fixture
def grid():
    sigma_r = (np.arange(1, NS + 1) - NS - 0.5) / NS
    sigma_w = np.linspace(-1, 0, NS + 1)
    lon, lat = np.meshgrid(-95.0 + np.arange(NX), 24.0 + np.arange(NY))
    cell = 100.0 * np.arange(NY)[:, None] + np.arange(NX)[None, :]
    ds = xr.Dataset(
        {
            "h": (("eta_rho", "xi_rho"), np.full((NY, NX), 1000.0)),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((NY, NX))),
            "sigma_r": (("s_rho",), sigma_r),
            "Cs_r": (("s_rho",), _stretch(sigma_r)),
            "sigma_w": (("s_w",), sigma_w),
            "Cs_w": (("s_w",), _stretch(sigma_w)),
            "chl": (
                ("s_rho", "eta_rho", "xi_rho"),
                np.broadcast_to(cell, (NS, NY, NX)).copy(),
            ),
            "u": (("s_rho", "eta_u", "xi_u"), np.zeros((NS, NY, NX - 1))),
        },
        coords={
            "lon": (("eta_rho", "xi_rho"), lon),
            "lat": (("eta_rho", "xi_rho"), lat),
        },
    )
    meta = {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}}
    return roms.add_depth_coord(ds, meta), meta


def _cells(strip):
    """Return each column's ``100 * eta + xi``, read from the first level."""
    return np.asarray(strip["chl"].isel(s_rho=0)).astype(int).tolist()


# -- the grammar ----------------------------------------------------------------------


def test_a_side_or_a_list_of_sides_is_a_boundary_request():
    assert as_transect({"boundary": "west"}) == {"kind": "boundary", "sides": ["west"]}
    assert as_transect({"boundary": ("west", "south")})["sides"] == ["west", "south"]


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"boundary": []}, "side or a list"),
        ({"boundary": [3]}, "side or a list"),
        ({"boundary": ["up"]}, "not boundary sides"),
        ({"boundary": ["west", "west"]}, "twice"),
        ({"boundary": ["west", "east"]}, "opposite sides"),
        ({"boundary": ["west", "south", "north"]}, "opposite sides"),
        ({"boundary": "west", "method": "bilinear"}, "do not apply"),
    ],
)
def test_a_bad_boundary_request_says_what_is_wrong(spec, match):
    with pytest.raises(ValueError, match=match):
        as_transect(spec)


# -- the strip ------------------------------------------------------------------------


def test_west_then_south_runs_down_the_west_side_and_along_the_south(grid):
    ds, _ = grid
    strip = apply_transect(ds, {"boundary": ["west", "south"]}, subject="t")
    # north end of the west column down to the south-west corner, then east
    assert _cells(strip) == [500, 400, 300, 200, 100, 0, 1, 2, 3]
    lon, lat = np.asarray(strip["lon"]), np.asarray(strip["lat"])
    assert lon[:6].tolist() == [-95.0] * 6
    assert lat[:6].tolist() == [29, 28, 27, 26, 25, 24]
    assert lat[5:].tolist() == [24.0] * 4 and lon[5:].tolist() == [-95, -94, -93, -92]


def test_the_strip_turns_other_corners_the_same_way(grid):
    ds, _ = grid
    assert _cells(apply_transect(ds, {"boundary": ["south", "east"]})) == [
        0, 1, 2, 3, 103, 203, 303, 403, 503
    ]  # fmt: skip
    assert _cells(apply_transect(ds, {"boundary": ["east", "south"]})) == [
        503, 403, 303, 203, 103, 3, 2, 1, 0
    ]  # fmt: skip
    assert _cells(apply_transect(ds, {"boundary": ["north", "west"]})) == [
        503, 502, 501, 500, 400, 300, 200, 100, 0
    ]  # fmt: skip


def test_three_sides_make_one_continuous_strip(grid):
    ds, _ = grid
    strip = apply_transect(ds, {"boundary": ["west", "south", "east"]})
    assert _cells(strip) == [
        500, 400, 300, 200, 100, 0, 1, 2, 3, 103, 203, 303, 403, 503
    ]  # fmt: skip
    # no repeated cell and no jump: neighbours are one grid step apart
    lon, lat = np.asarray(strip["lon"]), np.asarray(strip["lat"])
    assert np.all(np.hypot(np.diff(lon), np.diff(lat)) == 1.0)


def test_a_lone_side_runs_in_increasing_index_and_records_no_corner(grid):
    ds, _ = grid
    strip = apply_transect(ds, {"boundary": "north"})
    assert _cells(strip) == [500, 501, 502, 503]
    assert strip[ALONG_DIM].attrs["boundary_sides"] == "north"
    assert strip[ALONG_DIM].attrs["boundary_corner_km"] == []


def test_the_along_distance_runs_on_through_the_corner(grid):
    ds, _ = grid
    strip = apply_transect(ds, {"boundary": ["west", "south"]})
    lon, lat = np.asarray(strip["lon"]), np.asarray(strip["lat"])
    by_hand = np.concatenate(
        [[0.0], np.cumsum(_haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]))]
    )
    np.testing.assert_allclose(strip[ALONG_DIM], by_hand)
    assert np.all(np.diff(strip[ALONG_DIM]) > 0)
    assert strip[ALONG_DIM].attrs["units"] == "km"
    # the path coordinates every section carries are the strip's own
    np.testing.assert_allclose(strip["path_lon"], lon)
    np.testing.assert_allclose(strip["path_lat"], lat)


def test_the_corner_is_recorded_for_a_plot_to_mark(grid):
    ds, _ = grid
    strip = apply_transect(ds, {"boundary": ["west", "south", "east"]})
    attrs = strip[ALONG_DIM].attrs
    assert attrs["boundary_sides"] == "west,south,east"
    corners = attrs["boundary_corner_km"]
    assert len(corners) == 2
    along = np.asarray(strip[ALONG_DIM])
    lon, lat = np.asarray(strip["lon"]), np.asarray(strip["lat"])
    # the corners sit on the south-west and south-east grid corners
    at = [int(np.argmin(np.abs(along - c))) for c in corners]
    assert [(lon[k], lat[k]) for k in at] == [(-95.0, 24.0), (-92.0, 24.0)]


def test_variables_on_other_horizontal_dimensions_are_dropped(grid):
    ds, _ = grid
    strip = boundary_strip(ds, ["west", "south"])
    assert "u" not in strip
    assert {"chl", "h", "mask_rho", "Cs_r"} <= set(strip.variables)
    assert strip["chl"].dims == ("s_rho", ALONG_DIM)


def test_the_vertical_transform_still_works_on_the_strip(grid):
    ds, meta = grid
    strip = apply_transect(ds, {"boundary": ["west", "south"]})
    out = roms.to_depth(strip, meta, [50.0, 500.0])
    assert out["chl"].sizes == {"z": 2, ALONG_DIM: 9}


def test_a_boundary_the_grid_lacks_is_refused(grid):
    ds, _ = grid
    with pytest.raises(ValueError, match="not one of this source's dimensions"):
        apply_transect(ds.drop_dims("xi_rho"), {"boundary": ["west", "south"]})


# -- the strip through compare() ------------------------------------------------------


def test_a_boundary_strip_compares_against_a_gridded_reference(monkeypatch):
    from types import SimpleNamespace

    import ocean_skill as osk
    from ocean_skill import catalog
    from tests.test_section_comparison import VAR, _climatology, _roms_run

    sources = {
        "clim": (_climatology(), {"featureType": "grid"}),
        "roms_test": (
            _roms_run(),
            {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
        ),
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )
    cs = osk.compare(
        reference="clim",
        test="roms_test",
        variables=[VAR],
        select={"transect": {"boundary": ["west", "south"]}, "depth": [50.0, 200.0]},
    )
    (c,) = cs
    assert c.is_section
    aligned = c.aligned
    # the test lane is the 5 west + 3 south - 1 shared corner = 7 columns, around the
    # corner; the reference is paired onto them
    assert aligned.sizes[ALONG_DIM] >= 4
    assert np.all(np.diff(aligned[ALONG_DIM]) > 0)
    lat = np.asarray(aligned["path_lat"])
    assert lat[0] > lat[-1] or np.ptp(np.asarray(aligned["path_lon"])) > 0
    cs.plot()  # draws as a section


# -- land on a boundary row ----------------------------------------------------------


def test_missing_positions_are_interpolated_along_the_path():
    got = _fill_missing_positions([10.0, np.nan, np.nan, 13.0, np.nan])
    np.testing.assert_allclose(got, [10.0, 11.0, 12.0, 13.0, 13.0])


def test_missing_positions_interpolate_through_the_antimeridian():
    got = _fill_missing_positions([179.0, np.nan, -179.0])
    assert got[1] % 360.0 == pytest.approx(180.0)


def test_a_complete_path_is_returned_as_it_was():
    values = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_fill_missing_positions(values), values)
    assert np.isnan(_fill_missing_positions([np.nan, np.nan])).all()


def test_a_path_with_land_columns_still_resolves_to_finite_points():
    """The land columns of a boundary row carry NaN ``lon``/``lat``."""
    along = np.arange(5.0)
    lon = np.array([150.0, np.nan, np.nan, 156.0, 158.0])
    lat = np.array([50.0, np.nan, np.nan, 50.0, 50.0])
    lane = xr.Dataset(
        coords={
            "lon": (ALONG_DIM, lon),
            "lat": (ALONG_DIM, lat),
            ALONG_DIM: along,
            "path_lon": (ALONG_DIM, lon),
            "path_lat": (ALONG_DIM, lat),
        }
    )
    select, bbox = Comparison._resolved_path(None, lane, {"method": "nearest"})
    points = np.asarray(select["transect"]["points"])
    assert np.isfinite(points).all()
    np.testing.assert_allclose(points[:, 0], [150, 152, 154, 156, 158])
    assert bbox == (150.0, 50.0, 158.0, 50.0)
    as_transect(select["transect"])  # and the sampler accepts them
