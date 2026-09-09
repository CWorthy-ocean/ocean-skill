"""``compare()`` against a ``timeSeriesProfile`` reference whose time is pinned.

A repeat-visit station carries both axes, so ``featureType`` alone cannot say which
one a comparison keeps -- ordinarily that is read off select/aggregate
(``_implied_over``, unchanged by this feature). What this feature adds is the profile
treatment *for one visit*: pinning time to a single instant (``select={"time": ...}``,
or one entry of a ``times=[...]`` fan) makes that one comparison exactly a cast --
depth the only axis left to keep, and the reference's own (ragged) union of levels
filled in the same way a plain ``profile`` reference's are.

Mirrors ``tests/test_profile_compare_depths.py``'s stubbed-fan pattern (``over``/
``select`` recorded via a mocked ``Comparison.align``, no real catalog/cache/align())
and ``tests/test_compare_times.py``'s ``times=`` list-fan mocking.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from ocean_skill import comparison
from ocean_skill.comparison import _is_profile_reference

TEMPERATURE = "sea_water_potential_temperature"


def _tsp_source(times=("2024-01-01", "2024-02-01"), depths=(1.0, 10.0, 30.0)):
    """A tiny (time, depth) rectangle -- the shape tabular.to_dataset builds."""
    times = np.asarray(times, dtype="datetime64[ns]")
    depths = np.asarray(depths, dtype="float64")
    values = 20.0 - 0.1 * depths[None, :] + np.arange(len(times))[:, None]
    return xr.Dataset(
        {"TEMP": (("time", "depth"), values, {"units": "degC"})},
        coords={"time": times, "depth": depths},
    ).assign_coords(lon=-21.987, lat=64.2638)


@pytest.fixture
def stubbed_tsp_fan():
    """Record each fanned comparison's select/over against a timeSeriesProfile ref."""
    formed = []
    ref = _tsp_source()
    declared = {
        "hvalfjordur": {
            "featureType": "timeSeriesProfile",
            "axes": {"T": "time", "Z": "depth"},
            "standard_names": {"TEMP": TEMPERATURE},
            "variables": [TEMPERATURE],
        },
        "his": {"variables": [TEMPERATURE]},
    }
    lanes = {"hvalfjordur": ref}
    import ocean_skill as osk

    with (
        mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: SimpleNamespace(metadata=declared[n]),
        ),
        mock.patch.object(osk, "read", lambda name, **kw: lanes[name]),
        mock.patch("ocean_skill.sources.read", lambda name, **kw: lanes[name]),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append((self.over, self.select)),
        ),
    ):
        yield formed


def test_bare_compare_keeps_the_time_axis_unchanged(stubbed_tsp_fan):
    """No select at all: depth defaults to SURFACE (collapsed), time survives --
    the pre-existing timeSeriesProfile reading, untouched by this feature.
    """
    comparison.compare(reference="hvalfjordur", test="his", variables=[TEMPERATURE])
    assert len(stubbed_tsp_fan) == 1
    over, select = stubbed_tsp_fan[0]
    assert over == "time"
    assert select == {"depth": "surface"}


def test_time_pinned_by_select_reads_as_one_profile(stubbed_tsp_fan):
    """select={"time": <one visit>}, no vertical request: this one comparison keeps
    depth, filled with the reference's own (ragged) union of levels -- exactly the
    plain-profile treatment, for this one instant.
    """
    comparison.compare(
        reference="hvalfjordur",
        test="his",
        variables=[TEMPERATURE],
        select={"time": "2024-01-01"},
    )
    assert len(stubbed_tsp_fan) == 1
    over, select = stubbed_tsp_fan[0]
    assert over == "Z"
    assert select == {"time": "2024-01-01", "depth": [1.0, 10.0, 30.0]}


def test_explicit_depths_are_honored_over_the_references_own_levels(stubbed_tsp_fan):
    comparison.compare(
        reference="hvalfjordur",
        test="his",
        variables=[TEMPERATURE],
        select={"time": "2024-01-01"},
        depths=[5, 20],
    )
    assert len(stubbed_tsp_fan) == 1
    over, select = stubbed_tsp_fan[0]
    assert over == "Z"
    assert select == {"time": "2024-01-01", "depth": [5, 20]}


def test_times_fan_makes_one_profile_comparison_per_visit(stubbed_tsp_fan):
    """times=[...] (the "list" fan, one comparison per named visit) is the other
    route into a per-visit cast reading -- computed on the *base* select, before
    each iteration's own time entry is written in.
    """
    comparison.compare(
        reference="hvalfjordur",
        test="his",
        variables=[TEMPERATURE],
        times=["2024-01-01", "2024-02-01"],
    )
    assert len(stubbed_tsp_fan) == 2
    for over, select in stubbed_tsp_fan:
        assert over == "Z"
        assert select["depth"] == [1.0, 10.0, 30.0]
    assert [select["time"] for _, select in stubbed_tsp_fan] == [
        "2024-01-01",
        "2024-02-01",
    ]


def test_a_vertical_select_alongside_a_pinned_time_is_left_ordinary(stubbed_tsp_fan):
    """Both axes narrowed to one value: genuinely ambiguous, unchanged by this
    feature (_implied_over's own "neither, or both, collapsed" rule) -- and the
    caller's explicit scalar depth is never replaced by the reference's own list,
    since _profile_depth_plan's profile-list branch only ever fires when the
    reference's own select carries no vertical key at all.
    """
    comparison.compare(
        reference="hvalfjordur",
        test="his",
        variables=[TEMPERATURE],
        select={"time": "2024-01-01", "depth": 10},
    )
    assert len(stubbed_tsp_fan) == 1
    over, select = stubbed_tsp_fan[0]
    assert over is None
    assert select == {"time": "2024-01-01", "depth": 10}


def test_month_climatology_reads_as_one_profile_per_bin(stubbed_tsp_fan):
    """aggregate={"time": {"groupby": "month", ...}}, no select at all: time folds
    into monthly bins rather than one instant, but each bin is still exactly a
    cast -- depth is kept and filled with the reference's own levels, with no
    over= or depths= needed.
    """
    comparison.compare(
        reference="hvalfjordur",
        test="his",
        variables=[TEMPERATURE],
        aggregate={"time": {"groupby": "month", "reduce": "mean", "spread": "std"}},
    )
    assert len(stubbed_tsp_fan) == 1
    over, select = stubbed_tsp_fan[0]
    assert over == "Z"
    assert select == {"depth": [1.0, 10.0, 30.0]}


def test_month_climatology_pinned_to_one_depth_keeps_time(stubbed_tsp_fan):
    """A climatology *and* an explicit scalar depth (a seasonal cycle at one
    depth) is not the profile case -- the caller's own depth is honored, not
    replaced by the reference's own levels, and time is kept instead."""
    comparison.compare(
        reference="hvalfjordur",
        test="his",
        variables=[TEMPERATURE],
        select={"depth": 10.0},
        aggregate={"time": {"groupby": "month", "reduce": "mean", "spread": "std"}},
    )
    assert len(stubbed_tsp_fan) == 1
    over, select = stubbed_tsp_fan[0]
    assert over == "time"
    assert select == {"depth": 10.0}


# -- explicit depth bands: a fan, not a kept profile axis ------------------------------


def test_explicit_band_depths_fan_into_one_comparison_per_band(stubbed_tsp_fan):
    """A depths= list of bands collapses depth (aggregate's "Z": "mean" reduces
    each band to one value), so it is a genuine fan -- one comparison per band --
    not the profile's kept y-axis, even though the monthly resample otherwise makes
    this reference read as a climatology profile (see
    test_month_climatology_reads_as_one_profile_per_bin). Regression test for the
    band list reaching _prepare's depth handling as one unparseable list and
    raising ValueError.
    """
    comparison.compare(
        reference="hvalfjordur",
        test="his",
        variables=[TEMPERATURE],
        depths=[{"min": 0, "max": 5}, {"min": 10, "max": 15}],
        aggregate={"Z": "mean", "time": {"resample": "1MS", "reduce": "mean"}},
    )
    assert len(stubbed_tsp_fan) == 2
    for over, _select in stubbed_tsp_fan:
        assert over == "time"
    assert [select["depth"] for _, select in stubbed_tsp_fan] == [
        {"min": 0, "max": 5},
        {"min": 10, "max": 15},
    ]


def test_explicit_scalar_depths_still_stay_one_profile(stubbed_tsp_fan):
    """Regression guard: scalar/"surface" levels are the case that legitimately
    stays standing as one profile comparison -- unchanged by the band carve-out
    above (mirrors test_explicit_depths_are_honored_over_the_references_own_levels,
    but under the climatology route rather than a pinned single visit).
    """
    comparison.compare(
        reference="hvalfjordur",
        test="his",
        variables=[TEMPERATURE],
        depths=[0, 5, 10],
        aggregate={"time": {"groupby": "month", "reduce": "mean"}},
    )
    assert len(stubbed_tsp_fan) == 1
    over, select = stubbed_tsp_fan[0]
    assert over == "Z"
    assert select["depth"] == [0, 5, 10]


# -- _is_profile_reference: the extended scope ----------------------------------------


def test_is_profile_reference_needs_time_collapsed_true_for_tsp():
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata={"featureType": "timeSeriesProfile"}),
    ):
        assert not _is_profile_reference("tsp", None)  # default: unchanged
        assert not _is_profile_reference("tsp", None, time_collapsed=False)
        assert _is_profile_reference("tsp", None, time_collapsed=True)
        assert _is_profile_reference("tsp", "Z", time_collapsed=True)
        assert not _is_profile_reference("tsp", "time", time_collapsed=True)  # opts out


def test_is_profile_reference_still_ignores_trajectoryprofile():
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata={"featureType": "trajectoryProfile"}),
    ):
        assert not _is_profile_reference("traj", None, time_collapsed=True)


def test_is_profile_reference_climatology_param_for_tsp():
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata={"featureType": "timeSeriesProfile"}),
    ):
        assert not _is_profile_reference("tsp", None, climatology=False)
        assert _is_profile_reference("tsp", None, climatology=True)
        assert _is_profile_reference("tsp", "Z", climatology=True)
        assert not _is_profile_reference("tsp", "time", climatology=True)  # opts out
        # A trajectoryProfile still carries more than one candidate axis --
        # climatology alone does not resolve it either.


def test_is_profile_reference_still_ignores_trajectoryprofile_for_climatology():
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata={"featureType": "trajectoryProfile"}),
    ):
        assert not _is_profile_reference("traj", None, climatology=True)
