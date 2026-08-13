"""Tests for derived resolution metadata and the searches built on it.

Two things matter here. First that resolution is *derived from the axes* rather than
copied from what a product declares, because products misdeclare it: CoastWatch's
Metop-C ASCAT dataset advertises 0.25 degrees while its latitude axis steps 0.3333,
and AVISO's ``erdTAgeo1day`` is titled "1 Day Composite" while stepping about a week.
Second that grid spacing and effective resolution stay separate quantities, since a
1 km grid that resolves 10 km features will otherwise be searched as 1 km data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import catalog
from ocean_skill.build import _iso_duration, _probe, _spacing


def _grid(lat_step=0.25, lon_step=0.25, freq="D", periods=8):
    return xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.zeros((periods, 4, 5)))},
        coords={
            "time": pd.date_range("2020-01-01", periods=periods, freq=freq),
            "latitude": np.arange(4) * lat_step,
            "longitude": np.arange(5) * lon_step,
        },
    )


# -- derivation ---------------------------------------------------------------


def test_grid_resolution_comes_from_the_axis_not_the_attribute():
    """A product that lies about its spacing is recorded truthfully anyway.

    This is the ASCAT case: the file says 0.25 degrees and steps 0.3333.
    """
    ds = _grid(lat_step=1 / 3)
    ds.attrs["geospatial_lat_resolution"] = 0.25  # the product's own (wrong) claim
    md = _probe(ds, None)
    assert md["grid_resolution_deg"] == pytest.approx(0.3333, abs=1e-4)
    assert md["grid_resolution_km"] == pytest.approx(37.06, abs=0.1)


def test_time_resolution_comes_from_the_axis_not_the_title():
    """The AVISO case: titled "1 Day Composite", actually about weekly."""
    ds = _grid(freq="7D")
    md = _probe(ds, None)
    assert md["time_resolution"] == "P7D"
    assert md["time_resolution_s"] == pytest.approx(604800.0)


def test_median_spacing_survives_a_gap():
    """One missing interval must not drag the reported cadence off daily.

    MODIS 8-day bins restart each January 1 and hourly model runs have outages; a
    mean would report a cadence no file actually has.
    """
    times = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-03-01", "2020-03-02"]
    )
    ds = xr.Dataset(
        {"sst": ("time", np.zeros(5))},
        coords={"time": times, "latitude": 0.0, "longitude": 0.0},
    )
    md = _probe(ds, None)
    assert md["time_resolution"] == "P1D"


def test_evenly_spaced_axes_are_flagged_regular():
    md = _probe(_grid(), None)
    assert md["grid_regular"] is True


def test_lat_and_lon_collapse_to_one_number_when_they_agree():
    """Two near-identical numbers on every product is noise; disagreement is not."""
    md = _probe(_grid(lat_step=0.25, lon_step=0.25), None)
    assert md["grid_resolution_deg"] == pytest.approx(0.25)
    assert "grid_resolution_lat_deg" not in md
    assert "grid_resolution_lon_deg" not in md


def test_anisotropic_grid_records_both_axes():
    md = _probe(_grid(lat_step=0.25, lon_step=0.5), None)
    assert md["grid_resolution_lat_deg"] == pytest.approx(0.25)
    assert md["grid_resolution_lon_deg"] == pytest.approx(0.5)
    assert md["grid_resolution_deg"] == pytest.approx(
        0.25
    )  # latitude drives the scalar


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (3600, "PT1H"),
        (21600, "PT6H"),
        (86400, "P1D"),
        (8 * 86400, "P8D"),
        (30.4 * 86400, "P1M"),  # 28-31 days all mean "monthly"
        (28 * 86400, "P1M"),
        (365 * 86400, "P1Y"),
    ],
)
def test_iso_duration_snaps_to_the_period_the_product_means(seconds, expected):
    assert _iso_duration(seconds) == expected


def test_spacing_ignores_an_axis_too_short_to_have_one():
    assert _spacing(np.array([1.0])) is None
    assert _spacing(np.array([])) is None


# -- featureType gating -------------------------------------------------------


def test_a_point_time_series_gets_no_horizontal_resolution():
    """A mooring has a location, not a spacing.

    Recording one would let ``find(resolution=1)`` rank buoys against satellites.
    """
    ds = xr.Dataset(
        {"temp": ("time", np.arange(48.0))},
        coords={
            "time": pd.date_range("2020-01-01", periods=48, freq="h"),
            "latitude": 50.0,
            "longitude": -145.0,
        },
    )
    md = _probe(ds, None)
    assert md["featureType"] == "timeSeries"
    assert not any(key.startswith("grid_") for key in md)
    assert md["time_resolution"] == "PT1H"  # ...but cadence still applies


def test_vertical_resolution_is_a_range_because_levels_are_stretched():
    """One number would be wrong nearly everywhere on a stretched grid."""
    ds = xr.Dataset(
        {"temp": ("depth", np.arange(6.0))},
        coords={
            "depth": [0.0, 1.0, 5.0, 20.0, 100.0, 500.0],
            "latitude": 50.0,
            "longitude": -145.0,
        },
    )
    md = _probe(ds, None)
    assert md["vertical_levels"] == 6
    assert md["vertical_resolution_min"] == pytest.approx(1.0)
    assert md["vertical_resolution_max"] == pytest.approx(400.0)


# -- find() -------------------------------------------------------------------


@pytest.fixture
def resolution_index(monkeypatch):
    """Three sources spanning the distinctions the filters must draw."""
    entries = {
        "mur": (
            "sat",
            {
                "grid_resolution_km": 1.112,
                "time_resolution_s": 86400.0,
                "featureType": "grid",
            },
        ),
        # coarse grid
        "oisst": (
            "sat",
            {
                "grid_resolution_km": 27.8,
                "time_resolution_s": 86400.0,
                "featureType": "grid",
            },
        ),
        # a mooring: no spatial resolution at all, hourly, with depth
        "mooring": (
            "obs",
            {
                "time_resolution_s": 3600.0,
                "vertical_levels": 12,
                "featureType": "timeSeriesProfile",
            },
        ),
    }
    refs = {
        name: catalog.SourceRef(name=name, catalog=cat, path=None, metadata=meta)
        for name, (cat, meta) in entries.items()
    }
    monkeypatch.setattr(catalog, "discover", lambda: refs)
    return refs


def test_resolution_filters_on_grid_spacing(resolution_index):
    assert catalog.find(resolution=5) == ["mur", "mooring"]
    assert "oisst" in catalog.find(resolution=50)


def test_a_source_with_no_resolution_is_kept_not_dropped(resolution_index):
    """Unknown is not "too coarse" -- the same rule the extents follow."""
    assert "mooring" in catalog.find(resolution=1)


def test_cadence_accepts_a_spoken_period(resolution_index):
    assert sorted(catalog.find(cadence="hourly")) == ["mooring"]
    assert sorted(catalog.find(cadence="daily")) == ["mur", "oisst"]
    # 28- and 31-day months are both "monthly"; neither is daily
    assert catalog.find(cadence="monthly") == []


def test_cadence_rejects_a_period_it_does_not_know(resolution_index):
    with pytest.raises(ValueError, match="unknown cadence"):
        catalog.find(cadence="fortnightly")


def test_vertical_selects_sources_with_a_depth_axis(resolution_index):
    assert sorted(catalog.find(vertical=True)) == ["mooring"]
    assert sorted(catalog.find(vertical=False)) == ["mur", "oisst"]


def test_range_tuple_is_a_closed_interval(resolution_index):
    assert sorted(catalog.find(resolution=(20, 40))) == ["mooring", "oisst"]
