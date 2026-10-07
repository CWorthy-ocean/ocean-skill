"""A one-cell gridded source must not be skipped as "no declared overlap".

``compare()`` skips a pair whose catalog-declared extents provably never meet
(:func:`ocean_skill.catalog.overlap`, read-free). The extents come from the build
probe's cell-*centre* coordinates, so a model file that holds a single cell declares a
zero-width box -- a point -- and a station a few metres from that cell's centre, well
inside the cell, was reported as disjoint and never compared. The cell-centre extent of
a one-cell file says nothing about the cell's size, so a *gridded* source (featureType
``grid``, or a ``model`` entry such as ROMS) whose declared lon or lat extent has zero
width is padded by :data:`ocean_skill.catalog.POINT_EXTENT_TOLERANCE_DEG` on that axis.
Observation points are *not* padded: a station just outside a regional model is still
(correctly) skipped, and so is a one-point mooring entry against a station next to it.

``catalog.resolve`` is stubbed with real metadata keys, as in
``tests/test_compare_overlap_skip.py``, so the package's own ``_domain_of`` /
``overlap`` do the actual work.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocean_skill import catalog, comparison

TEMPERATURE = "sea_water_potential_temperature"

# The two ways an entry says "I am a model grid": a CF featureType, or the ``model``
# key a ROMS entry carries (a hand-written ROMS catalog need not set featureType).
GRIDDED_MARKERS = [
    pytest.param({"featureType": "grid"}, id="featureType-grid"),
    pytest.param({"model": "roms"}, id="model-roms"),
    pytest.param({"featureType": "grid", "model": "roms"}, id="both"),
]


def _extent(lon_min, lon_max, lat_min, lat_max, **extra):
    """Metadata declaring a lon/lat box and an overlapping time window."""
    return {
        "variables": [TEMPERATURE],
        "geospatial_lon_min": lon_min,
        "geospatial_lon_max": lon_max,
        "geospatial_lat_min": lat_min,
        "geospatial_lat_max": lat_max,
        "time_coverage_start": "2024-06-01",
        "time_coverage_end": "2024-06-02",
        **extra,
    }


def _point(lon, lat, **extra):
    """Metadata whose declared extent is a single point (what one cell centre gives)."""
    return _extent(lon, lon, lat, lat, **extra)


@pytest.fixture
def declare(monkeypatch):
    """Stub ``catalog.resolve`` over a name -> metadata table the test fills in."""
    table: dict[str, dict] = {}
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=table[name])
    )
    return table


# -- the bug: a one-cell grid vs a station inside the cell --------------------------


@pytest.mark.parametrize("marker", GRIDDED_MARKERS)
@pytest.mark.parametrize("flip", [False, True], ids=["model-first", "station-first"])
def test_a_one_cell_model_meets_a_station_a_few_metres_away(declare, marker, flip):
    declare["cell"] = _point(-150.0, 60.0, **marker)
    declare["station"] = _point(-150.0 + 1e-4, 60.0 - 1e-4, featureType="timeSeries")
    pair = ("station", "cell") if flip else ("cell", "station")
    ov = catalog.overlap(*pair)
    assert ov.space is True
    assert ov  # no known reason to refuse the pair


def test_a_station_beyond_the_tolerance_is_still_disjoint(declare):
    declare["cell"] = _point(-150.0, 60.0, featureType="grid")
    declare["station"] = _point(-150.0 + 0.2, 60.0, featureType="timeSeries")
    ov = catalog.overlap("cell", "station")
    assert ov.space is False
    assert not ov


def test_the_padding_is_the_documented_constant(declare):
    """Just inside the tolerance meets, just outside does not -- on either axis."""
    tol = catalog.POINT_EXTENT_TOLERANCE_DEG
    declare["cell"] = _point(-150.0, 60.0, featureType="grid")
    for axis in ("lon", "lat"):
        d_lon, d_lat = (1.0, 0.0) if axis == "lon" else (0.0, 1.0)
        declare["near"] = _point(-150.0 + 0.8 * tol * d_lon, 60.0 + 0.8 * tol * d_lat)
        declare["far"] = _point(-150.0 + 1.2 * tol * d_lon, 60.0 + 1.2 * tol * d_lat)
        assert catalog.overlap("cell", "near").space is True, axis
        assert catalog.overlap("cell", "far").space is False, axis


# -- what must not change -----------------------------------------------------------


@pytest.mark.parametrize(
    "station",
    [
        pytest.param(_point(-130.0 + 0.01, 58.0), id="east-of-the-box"),
        pytest.param(_point(-150.0 - 0.01, 58.0), id="west-of-the-box"),
        pytest.param(_point(-140.0, 62.0 + 0.01), id="north-of-the-box"),
        pytest.param(_point(-140.0, 55.0 - 0.01), id="south-of-the-box"),
    ],
)
def test_a_station_just_outside_a_regional_model_is_still_skipped(declare, station):
    """A real (non-degenerate) extent is exact -- the padding never reaches it."""
    declare["roms"] = _extent(
        -150.0, -130.0, 55.0, 62.0, model="roms", featureType="grid"
    )
    declare["station"] = {**station, "featureType": "timeSeries"}
    assert catalog.overlap("roms", "station").space is False


def test_a_station_inside_a_regional_model_still_meets_it(declare):
    declare["roms"] = _extent(
        -150.0, -130.0, 55.0, 62.0, model="roms", featureType="grid"
    )
    declare["station"] = _point(-140.0, 58.0, featureType="timeSeries")
    assert catalog.overlap("roms", "station").space is True


@pytest.mark.parametrize("feature_type", ["timeSeries", "profile", "point", None])
def test_a_one_point_observation_entry_is_not_padded(declare, feature_type):
    """Only gridded sources get the allowance -- a mooring's point is its position."""
    extra = {} if feature_type is None else {"featureType": feature_type}
    declare["mooring"] = _point(-150.0, 60.0, **extra)
    declare["station"] = _point(-150.0 + 1e-4, 60.0, featureType="timeSeries")
    assert catalog.overlap("mooring", "station").space is False


def test_two_coincident_points_still_meet(declare):
    """The closed-interval test is untouched: identical points overlap, unpadded."""
    declare["mooring"] = _point(-150.0, 60.0, featureType="timeSeries")
    declare["station"] = _point(-150.0, 60.0, featureType="profile")
    assert catalog.overlap("mooring", "station").space is True


# -- per-axis, and the longitude conventions ----------------------------------------


def test_only_the_zero_width_axis_is_padded(declare):
    """One row of cells: lon has a real extent (exact), lat is a point (padded)."""
    declare["row"] = _extent(-150.0, -140.0, 60.0, 60.0, featureType="grid")
    declare["inside_lon"] = _point(-145.0, 60.0 + 1e-4, featureType="timeSeries")
    declare["outside_lon"] = _point(-140.0 + 0.01, 60.0, featureType="timeSeries")
    declare["outside_lat"] = _point(-145.0, 60.0 + 0.2, featureType="timeSeries")
    assert catalog.overlap("row", "inside_lon").space is True
    assert catalog.overlap("row", "outside_lon").space is False
    assert catalog.overlap("row", "outside_lat").space is False


def test_a_one_cell_model_declared_0_360_meets_a_station_in_pm180(declare):
    """ROMS' native 0-360 longitude is normalised first; the padding follows it."""
    declare["cell"] = _point(200.0, 10.0, model="roms")
    declare["station"] = _point(-160.0 + 1e-4, 10.0, featureType="timeSeries")
    assert catalog.overlap("cell", "station").space is True


def test_a_one_cell_model_on_the_antimeridian_meets_a_station_across_it(declare):
    declare["cell"] = _point(180.0, 10.0, model="roms")  # normalises to -180
    declare["station"] = _point(179.9999, 10.0, featureType="timeSeries")
    assert catalog.overlap("cell", "station").space is True


# -- unchanged edges ----------------------------------------------------------------


def test_an_undeclared_extent_is_still_unknown_not_disjoint(declare):
    declare["cell"] = {"variables": [TEMPERATURE], "featureType": "grid"}
    declare["station"] = _point(-150.0, 60.0, featureType="timeSeries")
    assert catalog.overlap("cell", "station").space is None


def test_an_unresolvable_name_is_still_unknown(declare):
    declare["station"] = _point(-150.0, 60.0, featureType="timeSeries")
    assert catalog.overlap("ghost", "station").space is None  # the stub raises KeyError


def test_padding_does_not_mutate_the_catalogs_declared_metadata(declare):
    """The allowance is applied to a copy of the box, never written back."""
    cell = _point(-150.0, 60.0, featureType="grid")
    declare["cell"] = cell
    declare["station"] = _point(-150.0, 60.0, featureType="timeSeries")
    before = dict(cell)
    catalog.overlap("cell", "station")
    assert cell == before


# -- the symptom: compare() no longer skips the pair --------------------------------


def test_compare_does_not_skip_a_one_cell_model_against_a_nearby_station(capsys):
    from tests.test_compare_overlap_skip import _fan_recorded

    declared = {
        "cell": _point(-150.0, 60.0, featureType="grid", model="roms"),
        # a surface station: a mooring is refused "surface" until its entry says so
        "station": _point(
            -150.0 + 1e-4,
            60.0 + 1e-4,
            featureType="timeSeries",
            geospatial_vertical_min=0.0,
            geospatial_vertical_max=0.0,
        ),
    }
    with _fan_recorded(declared) as formed:
        comparison.compare(reference="station", test="cell", variables=[TEMPERATURE])
    assert formed == [("cell", "station")]
    assert "no declared overlap" not in capsys.readouterr().out


def test_compare_still_skips_a_one_point_mooring_against_a_nearby_station(capsys):
    from tests.test_compare_overlap_skip import _fan_recorded

    declared = {
        "mooring": _point(-150.0, 60.0, featureType="timeSeries"),
        "station": _point(-150.0 + 1e-4, 60.0 + 1e-4, featureType="timeSeries"),
    }
    with _fan_recorded(declared) as formed:
        comparison.compare(reference="station", test="mooring", variables=[TEMPERATURE])
    assert formed == []
    assert "no declared overlap in space" in capsys.readouterr().out
