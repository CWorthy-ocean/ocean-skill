"""``profile``/``timeSeriesProfile`` featureTypes: which axis a comparison implies.

``over=`` inference happens in ``Comparison.__init__``, off the reference's
featureType and its own select/aggregate -- before any data is read -- so these
mock only ``comparison._feature_type`` (mirroring ``tests/test_series.py``'s
``station_lanes`` fixture) and check ``.over``/``.over_reason`` directly, with no
catalog, no cache, no alignment.
"""

from __future__ import annotations

import warnings

import pytest

from ocean_skill.comparison import PROFILE_FEATURE_TYPES, Comparison

TEMPERATURE = "sea_water_temperature"


def _feature(monkeypatch, name: str) -> None:
    import ocean_skill.comparison as comparison

    monkeypatch.setattr(comparison, "_feature_type", lambda source: name)


def _comparison(**kwargs) -> Comparison:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Comparison(
            reference="whots_station", test="run_new", variable=TEMPERATURE, **kwargs
        )


def test_profile_feature_types_constant():
    assert PROFILE_FEATURE_TYPES == frozenset({"profile", "timeSeriesProfile"})


# -- "profile": no time axis at all, unambiguous ----------------------------------------


def test_profile_featuretype_implies_over_Z(monkeypatch):
    _feature(monkeypatch, "profile")
    c = _comparison()
    assert c.over == "Z"
    assert "featureType is 'profile'" in c.over_reason


def test_profile_featuretype_implies_over_Z_even_with_a_depth_select(monkeypatch):
    """profile has no time axis by its own definition -- unambiguous regardless of
    what select/aggregate says, unlike timeSeriesProfile."""
    _feature(monkeypatch, "profile")
    c = _comparison(select={"depth": [0.0, 25.0, 50.0]})
    assert c.over == "Z"


# -- "timeSeriesProfile": both axes present, disambiguated by select/aggregate ----------


def test_timeseriesprofile_with_depth_collapsed_implies_over_time(monkeypatch):
    """The familiar mooring-at-a-depth reading -- must keep working unchanged."""
    _feature(monkeypatch, "timeSeriesProfile")
    c = _comparison(select={"depth": 50.0})
    assert c.over == "time"
    assert "depth is narrowed to one value" in c.over_reason


def test_timeseriesprofile_with_no_depth_select_implies_over_time(monkeypatch):
    """No depth key at all defaults to SURFACE -- a scalar, so still collapsed."""
    _feature(monkeypatch, "timeSeriesProfile")
    c = _comparison()
    assert c.over == "time"


def test_timeseriesprofile_with_time_collapsed_implies_over_Z(monkeypatch):
    """A cast at one instant: depth survives as the axis instead."""
    _feature(monkeypatch, "timeSeriesProfile")
    c = _comparison(
        select={"depth": [0.0, 25.0, 50.0], "time": "2015-06-15"},
    )
    assert c.over == "Z"
    assert "time is narrowed to one value" in c.over_reason


def test_timeseriesprofile_with_time_aggregate_mean_implies_over_Z(monkeypatch):
    _feature(monkeypatch, "timeSeriesProfile")
    c = _comparison(
        select={"depth": [0.0, 25.0, 50.0]},
        aggregate={"time": "mean"},
    )
    assert c.over == "Z"


def test_timeseriesprofile_with_neither_axis_collapsed_is_ambiguous(monkeypatch):
    """Both a depth list and a surviving time axis: genuinely ambiguous, left unset."""
    _feature(monkeypatch, "timeSeriesProfile")
    c = _comparison(select={"depth": [0.0, 25.0, 50.0]})
    assert c.over is None
    assert "ambiguous" in c.over_reason


def test_timeseriesprofile_column_request_also_survives(monkeypatch):
    _feature(monkeypatch, "timeSeriesProfile")
    c = _comparison(select={"depth": "column"}, aggregate={"time": "mean"})
    assert c.over == "Z"


def test_timeseriesprofile_band_with_a_vertical_mean_collapses(monkeypatch):
    """A band survives on its own, but a vertical aggregate on top of it collapses."""
    _feature(monkeypatch, "timeSeriesProfile")
    c = _comparison(
        select={"depth": {"min": 0.0, "max": 50.0}},
        aggregate={"Z": "mean"},
    )
    assert c.over == "time"


def test_timeseriesprofile_with_a_scalar_season_select_implies_over_Z(monkeypatch):
    """A season fan's deferred select={"season": "JJA"} collapses time the same
    way a scalar select={"time": "2015-06-15"} does -- one season's climatology
    field is no more a line than one time instant is, so depth is kept instead."""
    _feature(monkeypatch, "timeSeriesProfile")
    c = _comparison(
        select={"depth": [0.0, 25.0, 50.0], "season": "JJA"},
        aggregate={"time": {"groupby": "season", "seasons": ["JJA"], "reduce": "mean"}},
    )
    assert c.over == "Z"
    assert "time is narrowed to one value" in c.over_reason


# -- explicit over= always wins, whatever the featureType says --------------------------


def test_explicit_over_wins_over_profile_featuretype(monkeypatch):
    _feature(monkeypatch, "profile")
    c = _comparison(over="time")
    assert c.over == "time"
    assert c.over_reason == "over= as asked"


# -- other featureTypes are untouched ----------------------------------------------------


def test_timeseries_featuretype_still_implies_time(monkeypatch):
    _feature(monkeypatch, "timeSeries")
    c = _comparison()
    assert c.over == "time"


def test_gridded_featuretype_implies_nothing_by_itself(monkeypatch):
    _feature(monkeypatch, "grid")
    c = _comparison()
    assert c.over is None
    assert c.over_reason == "the reference is gridded"


# -- a scalar season select must not spuriously imply over="time" -----------------


def test_a_point_select_with_a_scalar_season_does_not_imply_over_time(monkeypatch):
    """Without the season guard, a point select plus a season groupby aggregate
    (which _collapses_time reads as "keeps the axis") would wrongly imply
    over="time" -- but a fanned season select has already narrowed it to one
    field, exactly like a scalar select={"time": ...} does."""
    _feature(monkeypatch, "grid")
    c = _comparison(
        select={"lon": -158.0, "lat": 22.0, "season": "JJA"},
        aggregate={"time": {"groupby": "season", "seasons": ["JJA"], "reduce": "mean"}},
    )
    assert c.over is None
    assert c.over_reason == "the reference is gridded"


def test_a_spatial_mean_with_a_scalar_season_does_not_imply_over_time(monkeypatch):
    _feature(monkeypatch, "grid")
    c = _comparison(
        select={
            "lon": {"min": -160.0, "max": -156.0},
            "lat": {"min": 20.0, "max": 24.0},
            "season": "JJA",
        },
        aggregate={
            "lon": "mean",
            "lat": "mean",
            "time": {"groupby": "season", "seasons": ["JJA"], "reduce": "mean"},
        },
    )
    assert c.over is None
