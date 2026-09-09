"""An explicit depth list, named by the caller, is honored literally -- even past
a ``profile``/pinned-cast ``timeSeriesProfile`` reference's own observed range.

Before this feature, a ``depths=[...]`` (or ``select={"depth": [...]}``) request
past the observation's deepest reported level was silently *clamped* to that
deepest level (the ordinary nearest-level ``isel`` always finds *some* real
level) -- so the model's own, separately-interpolated value at the requested
depth was discarded by ``align()``'s "always lands on the reference" rule
(``ocean_skill.align._match_vertical``) before a caller ever saw it. Naming a
depth is now honored: the observation reads ``NaN`` there instead of a repeated
shallower value, so the model -- interpolated onto the same literal targets --
is free to be finite past the data, exactly the "see the model below the data"
view a caller who deliberately asked for a deep level wants. This is the
mirror image of ``tests/test_vertical_match.py``'s
``test_a_reference_level_beyond_the_tests_range_is_nan``, one level up the
pipeline: this is what has to happen in ``_prepare`` for a reference level past
the *test's* range to even exist in the first place, on the *observational*
side.

An auto-filled depth list -- ``compare()`` writing a profile's own levels into
``select`` when the caller named none (see ``_profile_depth_plan``) -- is not
"literal" in this sense and keeps today's clamped behavior; see
``tests/test_profile_depth_matching.py`` (change A) for that path.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

TEMPERATURE = "sea_water_temperature"

# A cast with no NaN holes at all -- every declared level has a real reading --
# so any NaN that shows up past this axis is purely from a *requested* depth
# beyond what the cast ever reported, not a missing measurement at a real level
# (that case is change A's, covered in test_profile_depth_matching.py).
_CAST_DEPTHS = np.array([5.0, 25.0, 60.0, 100.0, 150.0])
_CAST_VALUES = np.array([18.0, 14.0, 10.0, 8.0, 5.0])


def _profile_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"TEMP": (("DEPTH",), _CAST_VALUES, {"units": "degC"})},
        coords={"DEPTH": _CAST_DEPTHS},
    ).assign_coords(lon=-158.0, lat=22.75)


def _profile_meta() -> dict:
    return {
        "axes": {"Z": "DEPTH"},
        "featureType": "profile",
        "standard_names": {"TEMP": "sea_water_temperature"},
    }


def _gridded_test_lane() -> xr.Dataset:
    """A gridded product whose depth axis reaches well past the cast's own
    150 m floor, so a literal request for 300 m has a real model value to show.
    """
    depth = np.array([0.0, 10.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 400.0])
    lon = np.array([-158.5, -157.5])
    lat = np.array([22.25, 23.25])
    base = 19.0 - 0.03 * depth
    values = base[:, None, None] + np.zeros((1, 2, 2))
    da = xr.DataArray(
        values,
        dims=("depth", "lat", "lon"),
        coords={"depth": depth, "lat": lat, "lon": lon},
        name="sea_water_temperature",
        attrs={"units": "degC"},
    )
    return da.to_dataset()


@pytest.fixture
def profile_and_model(monkeypatch):
    import ocean_skill as osk
    from ocean_skill import catalog, comparison, sources

    lanes = {"ctd_cast": _profile_dataset(), "run_new": _gridded_test_lane()}
    metas = {
        "ctd_cast": _profile_meta(),
        "run_new": {"standard_names": {"sea_water_temperature": "sea_water_temperature"}},
    }

    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(sources, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)
    return lanes


# -- _reindex_tolerance: the gap rule -------------------------------------------------


def test_reindex_tolerance_is_half_the_largest_gap():
    from ocean_skill.comparison import _reindex_tolerance

    # Gaps are 20, 35, 40, 50 -- half the largest (50) is 25.
    assert _reindex_tolerance(np.array([5.0, 25.0, 60.0, 100.0, 150.0])) == 25.0


def test_reindex_tolerance_floor_for_a_single_level():
    from ocean_skill.comparison import _reindex_tolerance

    assert _reindex_tolerance(np.array([42.0])) == 1.0


# -- _prepare: literal_depths honors a caller-named depth past the obs range ----------


def test_prepare_honors_a_literal_depth_past_the_obs_range():
    """300 m is past the cast's 150 m floor: with literal_depths=True the
    observation reads NaN there instead of being clamped to 150 m, and every
    in-range target still reads its own real value.
    """
    from ocean_skill.comparison import _prepare

    obj = _profile_dataset()
    meta = _profile_meta()
    targets = [5.0, 25.0, 60.0, 100.0, 150.0, 300.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        da, _ = _prepare(
            obj, meta, TEMPERATURE, {"depth": list(targets)}, literal_depths=True
        )
    depths = [float(v) for v in da["DEPTH"].values]
    assert depths == targets
    values = {d: float(v) for d, v in zip(depths, da.values, strict=True)}
    for d, expected in zip(_CAST_DEPTHS, _CAST_VALUES, strict=True):
        assert values[float(d)] == pytest.approx(expected)
    assert np.isnan(values[300.0])
    assert any(
        "below this reference's deepest observed level" in str(w.message)
        for w in caught
    )


def test_prepare_clamps_without_literal_depths():
    """The pre-existing behavior, unchanged: without literal_depths (the
    default), 300 m clamps to the nearest real level (150 m) instead of
    reading NaN -- silently duplicating that level's own reading.
    """
    from ocean_skill.comparison import _prepare

    obj = _profile_dataset()
    meta = _profile_meta()
    targets = [5.0, 25.0, 60.0, 100.0, 150.0, 300.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        da, _ = _prepare(obj, meta, TEMPERATURE, {"depth": list(targets)})
    # Clamped: two of the six requested targets (150 and 300) both land on the
    # same nearest real level, so the axis still reads 150 m twice, not NaN.
    depths = [float(v) for v in da["DEPTH"].values]
    assert depths == [5.0, 25.0, 60.0, 100.0, 150.0, 150.0]
    assert np.isfinite(da.values).all()


def test_band_request_with_literal_depths_warns_and_is_unaffected():
    """literal_depths only ever changes a discrete depth *list* -- a band still
    takes the levels inside it (or the nearest one), unchanged, with a warning
    that the caller's literal-depths intent does not apply to it.
    """
    from ocean_skill.comparison import _prepare

    obj = _profile_dataset()
    meta = _profile_meta()
    band = {"min": 0.0, "max": 30.0}
    with pytest.warns(UserWarning, match="only applies to a discrete depth list"):
        da, _ = _prepare(obj, meta, TEMPERATURE, {"depth": band}, literal_depths=True)
    assert sorted(float(v) for v in da["DEPTH"].values) == [5.0, 25.0]


# -- Comparison: literal_depths is inferred from an explicit select -------------------


def test_a_direct_comparison_with_an_explicit_depth_list_is_literal():
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="a",
            test="b",
            variable=TEMPERATURE,
            select={"depth": [5.0, 300.0]},
            cache=False,
        )
    assert c.literal_depths is True


def test_a_bare_direct_comparison_is_not_literal():
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(reference="a", test="b", variable=TEMPERATURE, cache=False)
    assert c.literal_depths is False


def test_an_explicit_literal_depths_argument_overrides_inference():
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="a",
            test="b",
            variable=TEMPERATURE,
            select={"depth": [5.0]},
            literal_depths=False,
            cache=False,
        )
    assert c.literal_depths is False


# -- _profile_depth_plan: provenance feeds compare()'s fan -----------------------------


def test_explicit_depths_through_compare_are_marked_literal():
    from ocean_skill.comparison import _profile_depth_plan

    with pytest_mock_profile_reference():
        values, many, literal = _profile_depth_plan(
            "ctd_cast",
            {},
            "Z",
            False,
            "depth",
            (None,),
            [5.0, 300.0],
            {},
        )
    assert values == ([5.0, 300.0],)
    assert literal is True


def test_auto_filled_reference_levels_through_compare_are_not_literal():
    from ocean_skill.comparison import _profile_depth_plan

    with pytest_mock_profile_reference():
        values, many, literal = _profile_depth_plan(
            "ctd_cast", {}, "Z", False, "depth", (None,), None, {}
        )
    assert literal is False


def pytest_mock_profile_reference():
    """Patch just enough of the catalog/sources layer for _profile_depth_plan's
    profile branch to see "ctd_cast" as a profile with the module's own levels.
    """
    from unittest import mock

    return mock.patch.multiple(
        "ocean_skill.comparison",
        _is_profile_reference=lambda *a, **k: True,
        _profile_reference_depths=lambda *a, **k: list(_CAST_DEPTHS),
    )


# -- end to end: a literal depth past the obs range survives to the aligned pair ------


def test_a_literal_depth_past_the_obs_range_survives_to_the_aligned_pair(
    profile_and_model,
):
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = Comparison(
            reference="ctd_cast",
            test="run_new",
            variable=TEMPERATURE,
            select={"depth": [5.0, 25.0, 60.0, 100.0, 150.0, 300.0]},
            cache=False,
        )
        assert c.literal_depths is True
        assert c.over == "Z"
        aligned = c.align()

    depths = sorted(float(v) for v in aligned["DEPTH"].values)
    assert depths == [5.0, 25.0, 60.0, 100.0, 150.0, 300.0]
    at_300 = aligned.sel(DEPTH=300.0)
    assert not np.isfinite(at_300["reference"].values)
    assert np.isfinite(at_300["test"].values)
    assert not np.isfinite(at_300["difference"].values)
    at_5 = aligned.sel(DEPTH=5.0)
    assert np.isfinite(at_5["reference"].values)
    assert np.isfinite(at_5["difference"].values)
    assert any(
        "below this reference's deepest observed level" in str(w.message)
        for w in caught
    )


# -- cache key: a literal-depths lane never collides with the default one -------------


def test_literal_depths_folds_into_the_cache_key():
    from ocean_skill.cache import key_for_prepared

    plain = key_for_prepared(
        source="ctd_cast", variable=TEMPERATURE, select={"depth": [5.0, 300.0]}
    )
    literal = key_for_prepared(
        source="ctd_cast",
        variable=TEMPERATURE,
        select={"depth": [5.0, 300.0], "_literal_depths": True},
    )
    assert plain != literal
