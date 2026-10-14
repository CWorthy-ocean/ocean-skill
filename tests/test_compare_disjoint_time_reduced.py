"""Two time-reduced states are compared at no shared date, so their dates may not meet.

A suite page can compare two model runs' initial conditions -- single restart
snapshots dated months apart -- with ``aggregate={"time": "mean"}``. The read-free
"no declared overlap in time" skip
(:func:`ocean_skill.comparison._provably_disjoint_axes`) used to drop every such
pair before it was read. When *both* lanes' aggregates reduce the time axis away,
the time check is moot -- as it already is for a climatology -- but *space* still
disqualifies, and one lane reducing time while the other keeps a series is still a
time-against-time comparison, so it is still skipped.

Sources are synthetic gridded fields with disjoint time stamps, stubbed the way
``tests/test_depth_convention_compare.py`` does, so a pair that is not skipped goes
all the way through ``align`` to metrics.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog, comparison
from ocean_skill.comparison import _both_lanes_reduce_time, compare

VARIABLE = "sea_water_potential_temperature"
LON = np.array([200.0, 200.5, 201.0])
LAT = np.array([50.0, 50.5, 51.0])
# Disjoint by months, not merely days: the catalog pads declared windows by a day.
TEST_TIMES = pd.to_datetime(["2010-12-31"])
REF_TIMES = pd.to_datetime(["2010-10-31"])

# A 3 x 3 grid of one snapshot each is deliberately tiny; its sparse-coverage and
# partial-seasonal-cycle warnings say nothing about what is under test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:only .* valid cells", "ignore:groupby 'month'"
)

MEAN = {"time": "mean"}
SERIES_KEPT = {"time": {"groupby": "month", "reduce": "mean"}}


def _field(times, offset, lon=LON, lat=LAT):
    """Build a ``(time, lat, lon)`` field: ``offset`` plus a fixed gradient."""
    grad = np.arange(lat.size)[:, None] + np.arange(lon.size)[None, :]
    values = offset + grad[None, :, :] * np.ones((len(times), 1, 1))
    return xr.Dataset(
        {
            VARIABLE: (
                ("time", "lat", "lon"),
                values,
                {"units": "degC", "standard_name": VARIABLE},
            )
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )


def _meta(times, lon=LON, lat=LAT):
    return {
        "variables": [VARIABLE],
        "featureType": "grid",
        "geospatial_lon_min": float(lon.min()),
        "geospatial_lon_max": float(lon.max()),
        "geospatial_lat_min": float(lat.min()),
        "geospatial_lat_max": float(lat.max()),
        "time_coverage_start": str(times.min().date()),
        "time_coverage_end": str(times.max().date()),
        "standard_names": {VARIABLE: VARIABLE},
    }


def _install(monkeypatch, *, ref_lon=LON, ref_lat=LAT):
    """Stub the catalog and reader for a test run and a time-disjoint reference."""
    sources = {
        "ic_test": (_field(TEST_TIMES, 10.0), _meta(TEST_TIMES)),
        "ic_ref": (
            _field(REF_TIMES, 8.0, ref_lon, ref_lat),
            _meta(REF_TIMES, ref_lon, ref_lat),
        ),
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )


def _run(aggregate, **kwargs):
    return compare(
        reference="ic_ref",
        test="ic_test",
        variables=[VARIABLE],
        aggregate=aggregate,
        **kwargs,
    )


def test_the_fixture_sources_really_are_time_disjoint(monkeypatch):
    _install(monkeypatch)
    assert catalog.overlap("ic_test", "ic_ref").time is False


def test_both_lanes_time_reduced_forms_a_comparison_with_finite_metrics(
    monkeypatch, capsys
):
    _install(monkeypatch)
    out = _run(MEAN)
    assert "no declared overlap" not in capsys.readouterr().out
    assert len(out) == 1
    m = out[0].metrics()
    # The reference is 2 degC colder everywhere, so the bias is exactly the offset.
    assert np.isfinite(m["bias"])
    assert m["bias"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "aggregate",
    [
        {"time": {"reduce": "mean"}},
        {"time": [SERIES_KEPT["time"], {"reduce": "std"}]},
        {"test": MEAN, "reference": {"time": {"reduce": "mean"}}},
    ],
    ids=["dict-reducer", "chain-ending-in-a-reduce", "per-lane-both-reduce"],
)
def test_every_spelling_of_a_time_reduction_is_exempt(aggregate, monkeypatch):
    assert _both_lanes_reduce_time(aggregate)
    _install(monkeypatch)
    assert len(_run(aggregate)) == 1


def test_without_a_time_reduction_the_pair_is_still_skipped(monkeypatch, capsys):
    _install(monkeypatch)
    out = _run(None)
    assert len(out) == 0
    assert "skipped 'ic_test' vs 'ic_ref': no declared overlap in time" in (
        capsys.readouterr().out
    )


def test_a_fold_that_keeps_the_time_axis_is_not_a_reduction(monkeypatch, capsys):
    """groupby/resample keep an axis, so the two time axes are still compared."""
    _install(monkeypatch)
    assert len(_run(SERIES_KEPT)) == 0
    assert "no declared overlap in time" in capsys.readouterr().out


@pytest.mark.parametrize("reducing_lane", ["test", "reference"])
def test_only_one_lane_reducing_time_is_still_skipped(
    reducing_lane, monkeypatch, capsys
):
    _install(monkeypatch)
    other = "reference" if reducing_lane == "test" else "test"
    out = _run({reducing_lane: MEAN, other: {}})
    assert len(out) == 0
    assert "no declared overlap in time" in capsys.readouterr().out


def test_skip_missing_false_still_raises_for_one_lane_reducing(monkeypatch):
    _install(monkeypatch)
    with pytest.raises(ValueError, match="do not overlap in time"):
        _run({"test": MEAN, "reference": {}}, skip_missing=False)


def test_a_space_disjoint_pair_is_still_skipped_even_with_time_reduced(
    monkeypatch, capsys
):
    _install(monkeypatch, ref_lon=LON + 60.0, ref_lat=LAT - 80.0)
    out = _run(MEAN)
    assert len(out) == 0
    assert "skipped 'ic_test' vs 'ic_ref': no declared overlap in space" in (
        capsys.readouterr().out
    )


def test_a_directly_built_comparison_warns_only_without_a_time_reduction(monkeypatch):
    _install(monkeypatch)
    quiet = comparison.Comparison(
        reference="ic_ref", test="ic_test", variable=VARIABLE, aggregate=MEAN
    )
    loud = comparison.Comparison(
        reference="ic_ref", test="ic_test", variable=VARIABLE, aggregate=None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet._warn_if_no_overlap()
    with pytest.warns(UserWarning, match="do not overlap in time"):
        loud._warn_if_no_overlap()
