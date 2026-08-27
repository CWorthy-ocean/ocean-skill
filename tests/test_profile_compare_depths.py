"""``compare()`` against a ``profile`` reference: its own depths drive the compare.

A ``featureType == "profile"`` reference keeps its depth axis standing (over="Z" is
implied), so the comparison is *one* profile down that column -- not compare()'s
ordinary per-depth fan, which writes a scalar depth into the select and would collapse
the very axis the profile exists to keep. And there is a natural default no map has:
the reference's own levels, read straight off the source when the caller names none.

Mirrors ``tests/test_compare_times.py``'s fan-shape mocking: ``catalog.resolve`` and
``sources.read`` are stubbed and ``Comparison.align`` records each fanned comparison's
select, so these assert the *shape* of the fan without a real catalog, cache, or
align(). The vertical match itself is covered in ``tests/test_vertical_match.py``, and
``Comparison`` classification in ``tests/test_profile_comparison.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from ocean_skill import comparison
from ocean_skill.comparison import (
    _is_profile_reference,
    _profile_depth_plan,
    _profile_reference_depths,
)

TEMPERATURE = "sea_water_potential_temperature"


def _profile_source(depths=(5.0, 25.0, 50.0, 75.0, 100.0), *, zname="DEPTH"):
    depths = np.asarray(depths, dtype="float64")
    return xr.Dataset(
        {"TEMP": ((zname,), 20.0 - 0.1 * depths, {"units": "degC"})},
        coords={zname: depths},
    ).assign_coords(lon=-158.0, lat=22.75)


@pytest.fixture
def stubbed_profile_fan():
    """Record each fanned comparison's select against a profile reference."""
    formed = []
    ref = _profile_source()
    declared = {
        "ctd_profile": {
            "featureType": "profile",
            "axes": {"Z": "DEPTH"},
            "standard_names": {"TEMP": TEMPERATURE},
            "variables": [TEMPERATURE],
        },
        "gridded_ref": {"variables": [TEMPERATURE]},
        "his": {"variables": [TEMPERATURE]},
    }
    lanes = {"ctd_profile": ref}
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
            lambda self, refresh=False: formed.append(
                (self.reference_name, self.over, self.select, self.label)
            ),
        ),
    ):
        yield formed


# -- the feature: no depths given, the reference's own levels are used ---------------


def test_no_depths_uses_the_reference_profiles_own_levels(stubbed_profile_fan):
    comparison.compare(
        reference="ctd_profile", test="his", variables=[TEMPERATURE]
    )
    assert len(stubbed_profile_fan) == 1
    ref, over, select, _ = stubbed_profile_fan[0]
    assert over == "Z"
    # the whole column as one comparison's select, not a scalar surface default
    assert select == {"depth": [5.0, 25.0, 50.0, 75.0, 100.0]}


def test_it_is_one_comparison_not_a_per_depth_fan(stubbed_profile_fan):
    """The reference has five levels; a naive fan would make five comparisons."""
    comparison.compare(
        reference="ctd_profile", test="his", variables=[TEMPERATURE]
    )
    assert len(stubbed_profile_fan) == 1


def test_explicit_depths_become_the_one_profile_axis_not_a_fan(stubbed_profile_fan):
    """depths=[...] against a profile is the profile's y-axis, one comparison --
    not compare()'s ordinary one-map-per-depth fan (which collapses the axis)."""
    comparison.compare(
        reference="ctd_profile",
        test="his",
        variables=[TEMPERATURE],
        depths=[10, 20, 30],
    )
    assert len(stubbed_profile_fan) == 1
    _, _, select, _ = stubbed_profile_fan[0]
    assert select == {"depth": [10, 20, 30]}


def test_explicit_select_depth_list_is_left_untouched(stubbed_profile_fan):
    """An explicit select={"depth": [...]} already yields one whole-list comparison
    (the pre-existing working path) and must not be replaced by the reference's."""
    comparison.compare(
        reference="ctd_profile",
        test="his",
        variables=[TEMPERATURE],
        select={"depth": [200.0]},
    )
    assert len(stubbed_profile_fan) == 1
    _, _, select, _ = stubbed_profile_fan[0]
    assert select == {"depth": [200.0]}


def test_a_gridded_reference_still_gets_the_surface_default(stubbed_profile_fan):
    """The profile branch must not touch an ordinary gridded map comparison."""
    comparison.compare(
        reference="gridded_ref", test="his", variables=[TEMPERATURE]
    )
    assert len(stubbed_profile_fan) == 1
    _, over, select, _ = stubbed_profile_fan[0]
    assert over is None
    assert select == {"depth": "surface"}


# -- _profile_reference_depths: reading the axis --------------------------------------


def test_reference_depths_are_read_deduped_and_sorted():
    """A profile logged with duplicate depths (the '60 duplicate depths' case) still
    yields each level once, ascending."""
    src = _profile_source(depths=[50.0, 5.0, 25.0, 25.0, 5.0])
    meta = {"axes": {"Z": "DEPTH"}}
    with (
        mock.patch("ocean_skill.sources.read", lambda name, **kw: src),
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: SimpleNamespace(metadata=meta)
        ),
    ):
        assert _profile_reference_depths("ctd", {}) == [5.0, 25.0, 50.0]


def test_reference_depths_are_memoized_per_source():
    src = _profile_source()
    calls = {"n": 0}

    def counting_read(name, **kw):
        calls["n"] += 1
        return src

    cache: dict = {}
    with (
        mock.patch("ocean_skill.sources.read", counting_read),
        mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: SimpleNamespace(metadata={"axes": {"Z": "DEPTH"}}),
        ),
    ):
        _profile_reference_depths("ctd", cache)
        _profile_reference_depths("ctd", cache)
    assert calls["n"] == 1


def test_reference_depths_read_from_a_dataframe_profile():
    """The reported shape: a CTD cast comes back from intake as a DataFrame and is
    built into a depth-indexed Dataset by tabular.to_dataset -- the axis must still be
    found, and the duplicate-depth warning suppressed here (align() emits it for real).
    """
    import warnings

    import pandas as pd

    df = pd.DataFrame(
        {
            "depth (m)": [5.0, 5.0, 25.0, 50.0],  # a duplicate, as real casts carry
            "time": ["2024-06-01"] * 4,
            "longitude": [-150.0] * 4,
            "latitude": [60.0] * 4,
            "sea_water_temperature": [12.0, 12.0, 10.0, 8.0],
        }
    )
    meta = {
        "featureType": "profile",
        "axes": {"Z": "depth (m)", "T": "time", "X": "longitude", "Y": "latitude"},
    }
    with (
        mock.patch("ocean_skill.sources.read", lambda name, **kw: df),
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: SimpleNamespace(metadata=meta)
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")  # no duplicate-depth warning may escape here
        assert _profile_reference_depths("ctd", {}) == [5.0, 25.0, 50.0]


def test_reference_with_no_vertical_axis_is_a_clear_error():
    flat = xr.Dataset({"TEMP": (("x",), [1.0, 2.0])}, coords={"x": [0, 1]})
    with (
        mock.patch("ocean_skill.sources.read", lambda name, **kw: flat),
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: SimpleNamespace(metadata={})
        ),
    ):
        with pytest.raises(ValueError, match="no vertical axis"):
            _profile_reference_depths("ctd", {})


# -- _is_profile_reference: scope is exactly featureType == "profile" -----------------


def test_only_profile_featuretype_opts_in():
    metas = {
        "p": {"featureType": "profile"},
        "tsp": {"featureType": "timeSeriesProfile"},
        "traj": {"featureType": "trajectoryProfile"},
        "grid": {},
    }
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata=metas[n]),
    ):
        assert _is_profile_reference("p", None)
        assert not _is_profile_reference("tsp", None)  # deferred: two candidate axes
        assert not _is_profile_reference("traj", None)
        assert not _is_profile_reference("grid", None)


def test_a_non_vertical_explicit_over_opts_out():
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata={"featureType": "profile"}),
    ):
        # the caller named time as the scored axis themselves -- not our business
        assert not _is_profile_reference("p", "time")
        assert _is_profile_reference("p", "Z")


# -- align()'s guard: a collapsing vertical select fails clearly, not deep ------------


def test_direct_comparison_over_Z_without_depths_is_a_clear_error():
    with mock.patch(
        "ocean_skill.catalog.resolve",
        lambda n: SimpleNamespace(metadata={"variables": [TEMPERATURE]}),
    ):
        c = comparison.Comparison(
            reference="gridded_ref",
            test="his",
            variable=TEMPERATURE,
            over="Z",
            cache=False,
        )
        with pytest.raises(ValueError, match="no vertical axis left to score"):
            c.align()


def test_over_Z_with_a_depth_list_does_not_trip_the_guard():
    """The guard keys on collapse, not on over=: a real depth list survives it."""
    assert not comparison._collapses_vertical({"depth": [5, 25, 50]}, None)
    assert comparison._collapses_vertical({"depth": "surface"}, None)
    assert comparison._collapses_vertical({}, None)
