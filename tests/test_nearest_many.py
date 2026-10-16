"""``_nearest_indices_many`` must snap where the per-point haversine argmin does.

The k-d tree replaces a full-grid haversine per path point; these tests hold it to the
old search (``_nearest_indices``) on the cases where a different answer would be
visible: a grid straddling the dateline in either longitude convention, NaN cells, and
points exactly midway between cells (the lowest-flat-index tie-break).
"""

from __future__ import annotations

import numpy as np
import pytest

from ocean_skill import align
from ocean_skill.align import _nearest_indices, _nearest_indices_many


@pytest.fixture(autouse=True)
def _always_the_tree(monkeypatch):
    """Send every query through the k-d tree, however few points it has.

    A short query normally takes the per-point haversine instead
    (``_TREE_MIN_POINTS``); these tests are about the tree.
    """
    monkeypatch.setattr(align, "_TREE_MIN_POINTS", 0)


def test_a_short_query_takes_the_per_point_search_and_agrees(monkeypatch):
    lon, lat = _curvilinear(180.0, "0-360")
    q_lon, q_lat = np.array([176.0, 181.5, 184.2]), np.array([12.0, 15.5, 20.1])
    tree = _nearest_indices_many(lon, lat, q_lon, q_lat)
    monkeypatch.setattr(align, "_TREE_MIN_POINTS", 32)
    short = _nearest_indices_many(lon, lat, q_lon, q_lat)
    np.testing.assert_array_equal(tree[0], short[0])
    np.testing.assert_array_equal(tree[1], short[1])


def _brute(lon_2d, lat_2d, lons, lats):
    pairs = [_nearest_indices(lon_2d, lat_2d, lo, la) for lo, la in zip(lons, lats)]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _curvilinear(lon0, convention, ny=40, nx=60, seed=0):
    """Build a rotated, slightly warped grid whose lon spans ``lon0`` +- 15 degrees."""
    rng = np.random.default_rng(seed)
    xi, eta = np.meshgrid(np.arange(nx), np.arange(ny))
    ang = np.deg2rad(20)
    lon = lon0 + 0.5 * (xi * np.cos(ang) - eta * np.sin(ang)) - 5
    lat = 10 + 0.4 * (xi * np.sin(ang) + eta * np.cos(ang))
    lon = lon + 0.01 * rng.standard_normal(lon.shape)
    lon = lon % 360 if convention == "0-360" else ((lon + 180) % 360) - 180
    return lon, lat


@pytest.mark.parametrize("convention", ["0-360", "-180-180"])
@pytest.mark.parametrize("lon0", [180.0, 100.0])
def test_matches_haversine_on_dateline_grid(convention, lon0):
    lon, lat = _curvilinear(lon0, convention)
    rng = np.random.default_rng(1)
    q_lon = lon0 + rng.uniform(-12, 12, 200)
    q_lat = rng.uniform(8, 30, 200)
    # Queries in either convention: the xyz search is periodic, so both must work.
    for q in (q_lon % 360, ((q_lon + 180) % 360) - 180):
        got = _nearest_indices_many(lon, lat, q, q_lat)
        want = _brute(lon, lat, q, q_lat)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


def test_nan_cells_are_excluded():
    lon, lat = _curvilinear(30.0, "-180-180")
    rng = np.random.default_rng(2)
    bad = rng.random(lon.shape) < 0.3
    lon = np.where(bad, np.nan, lon)
    lat = np.where(rng.random(lat.shape) < 0.1, np.nan, lat)
    q_lon = rng.uniform(15, 45, 150)
    q_lat = rng.uniform(10, 25, 150)
    got = _nearest_indices_many(lon, lat, q_lon, q_lat)
    want = _brute(lon, lat, q_lon, q_lat)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    assert np.isfinite(lon[got]).all() and np.isfinite(lat[got]).all()


def test_ties_pick_lowest_flat_index():
    lon, lat = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 8.0))
    # Midway between two cells along x (same latitude) and along y (same longitude)
    # are true great-circle ties: the lowest flat index (lower row / column) wins.
    q_lon = np.array([2.5, 6.5, 4.0, 9.0])
    q_lat = np.array([3.0, 0.0, 2.5, 6.5])
    got = _nearest_indices_many(lon, lat, q_lon, q_lat)
    np.testing.assert_array_equal(got[0], [3, 0, 2, 6])
    np.testing.assert_array_equal(got[1], [2, 6, 4, 9])


def test_midpoints_match_haversine_argmin():
    # Cell-corner and edge midpoints on a regular grid, off the equator where the
    # four cells are no longer exactly equidistant: the old search is the reference.
    lon, lat = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 8.0))
    q_lon, q_lat = np.meshgrid(np.arange(0.5, 9.0, 1.0), np.arange(0.5, 7.0, 1.0))
    got = _nearest_indices_many(lon, lat, q_lon.ravel(), q_lat.ravel())
    want = _brute(lon, lat, q_lon.ravel(), q_lat.ravel())
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_single_point_and_tiny_grid():
    lon, lat = np.meshgrid([10.0, 11.0], [5.0])  # one row: fewer cells than k
    iy, ix = _nearest_indices_many(lon, lat, [10.9], [5.0])
    assert (int(iy[0]), int(ix[0])) == (0, 1)


def test_all_nan_grid_raises():
    nan = np.full((3, 3), np.nan)
    with pytest.raises(ValueError):
        _nearest_indices_many(nan, nan, [1.0], [1.0])
