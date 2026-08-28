"""``compare()`` against a fixed-station reference: its own depth drives the compare.

A moored instrument (``featureType`` in ``timeSeries``/``point``/``station``) sits at
one depth, not a whole column -- unlike a ``profile`` reference (see
``tests/test_profile_compare_depths.py``), which keeps its depth axis standing. A
mooring's own ``geospatial_vertical_min``/``_max`` (see
:func:`ocean_skill.build._extent`) names that one depth, and comparing it against the
model's default surface is comparing the wrong level unless the caller overrides it.

Mirrors ``tests/test_profile_compare_depths.py``'s fan-shape mocking:
``catalog.resolve`` is stubbed and ``Comparison.align`` records each fanned
comparison's select, so these assert the *shape* of the fan without a real
catalog, cache, or align().
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest import mock

import pytest

from ocean_skill import comparison
from ocean_skill.comparison import _profile_depth_plan, _station_depth_from_metadata

TEMPERATURE = "sea_water_potential_temperature"


@pytest.fixture
def stubbed_station_fan():
    """Record each fanned comparison's select against a fixed-station reference."""
    formed = []
    declared = {
        "ctd_mooring": {
            "featureType": "timeSeries",
            "geospatial_vertical_min": 20.0,
            "geospatial_vertical_max": 30.0,
            "variables": [TEMPERATURE],
        },
        "gridded_ref": {"variables": [TEMPERATURE]},
        "his": {"variables": [TEMPERATURE]},
    }
    with (
        mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: SimpleNamespace(metadata=declared[n]),
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(
                (self.reference_name, self.over, self.select, self.label)
            ),
        ),
    ):
        yield formed


# -- the feature: no depths given, the station's own metadata depth is used ---------


def test_no_depths_uses_the_stations_own_metadata_depth(stubbed_station_fan):
    with pytest.warns(UserWarning, match=r"~25 m"):
        comparison.compare(
            reference="ctd_mooring", test="his", variables=[TEMPERATURE]
        )
    assert len(stubbed_station_fan) == 1
    _, _, select, _ = stubbed_station_fan[0]
    assert select == {"depth": 25.0}


def test_explicit_depths_override_the_metadata_depth(stubbed_station_fan):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no depth-override warning may escape here
        comparison.compare(
            reference="ctd_mooring",
            test="his",
            variables=[TEMPERATURE],
            depths=[10],
        )
    assert len(stubbed_station_fan) == 1
    _, _, select, _ = stubbed_station_fan[0]
    assert select == {"depth": 10}


def test_explicit_select_depth_overrides_the_metadata_depth(stubbed_station_fan):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comparison.compare(
            reference="ctd_mooring",
            test="his",
            variables=[TEMPERATURE],
            select={"depth": 5.0},
        )
    assert len(stubbed_station_fan) == 1
    _, _, select, _ = stubbed_station_fan[0]
    assert select == {"depth": 5.0}


def test_a_gridded_reference_still_gets_the_surface_default(stubbed_station_fan):
    """The station branch must not touch an ordinary gridded map comparison."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comparison.compare(
            reference="gridded_ref", test="his", variables=[TEMPERATURE]
        )
    assert len(stubbed_station_fan) == 1
    _, _, select, _ = stubbed_station_fan[0]
    assert select == {"depth": "surface"}


# -- _station_depth_from_metadata: reading the metadata ------------------------------


def test_the_midpoint_of_min_and_max_is_used():
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(
            metadata={
                "featureType": "timeSeries",
                "geospatial_vertical_min": 20.0,
                "geospatial_vertical_max": 30.0,
            }
        ),
    ):
        assert _station_depth_from_metadata("m") == 25.0


def test_a_single_declared_depth_is_used_as_is():
    """A builder that only ever wrote one exact depth sets _min == _max."""
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(
            metadata={
                "featureType": "point",
                "geospatial_vertical_min": 8.0,
                "geospatial_vertical_max": 8.0,
            }
        ),
    ):
        assert _station_depth_from_metadata("m") == 8.0


@pytest.mark.parametrize(
    "meta",
    [
        {"featureType": "profile", "geospatial_vertical_min": 5.0},
        {"featureType": "timeSeriesProfile", "geospatial_vertical_min": 5.0},
        {"featureType": "trajectory", "geospatial_vertical_min": 5.0},
        {},
        {"featureType": "timeSeries"},  # no vertical metadata at all
        {"featureType": "timeSeries", "geospatial_vertical_min": float("nan")},
        {"featureType": "timeSeries", "geospatial_vertical_min": -5.0},
        {
            "featureType": "timeSeries",
            "geospatial_vertical_min": 5.0,
            "geospatial_vertical_max": -1.0,
        },
        {"featureType": "timeSeries", "geospatial_vertical_min": "not a number"},
    ],
)
def test_no_depth_is_derived_outside_its_narrow_scope(meta):
    with mock.patch(
        "ocean_skill.catalog.resolve", lambda n: SimpleNamespace(metadata=meta)
    ):
        assert _station_depth_from_metadata("m") is None


def test_an_unresolvable_source_derives_nothing():
    def raises(name):
        raise KeyError(name)

    with mock.patch("ocean_skill.catalog.resolve", raises):
        assert _station_depth_from_metadata("m") is None


# -- _profile_depth_plan: the station branch keeps its own scope --------------------


def test_a_vertical_select_on_the_reference_side_is_left_alone():
    """An explicit vertical select on the reference already answers the question;
    the metadata depth must not silently override it."""
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(
            metadata={
                "featureType": "timeSeries",
                "geospatial_vertical_min": 25.0,
                "geospatial_vertical_max": 25.0,
            }
        ),
    ):
        values, many = _profile_depth_plan(
            "m", {"depth": 5.0}, None, False, "depth", ("surface",), None, {}
        )
    assert values == ("surface",)
    assert many is False


def test_a_calculated_diagnostic_is_left_alone():
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(
            metadata={
                "featureType": "timeSeries",
                "geospatial_vertical_min": 25.0,
                "geospatial_vertical_max": 25.0,
            }
        ),
    ):
        values, many = _profile_depth_plan(
            "m", {}, None, True, "depth", ("surface",), None, {}
        )
    assert values == (None,)
    assert many is False


def test_a_sigma0_fan_is_left_alone():
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(
            metadata={
                "featureType": "timeSeries",
                "geospatial_vertical_min": 25.0,
                "geospatial_vertical_max": 25.0,
            }
        ),
    ):
        values, many = _profile_depth_plan(
            "m", {}, None, False, "sigma0", (1025.0,), None, {}
        )
    assert values == (1025.0,)
    assert many is False
