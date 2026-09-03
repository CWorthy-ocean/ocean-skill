"""``subtract_mean`` reaches every layer, the way ``min_coverage`` does.

Mirrors ``tests/test_min_coverage_plumbing.py`` in structure: defaults have to
match across :class:`~ocean_skill.comparison.Comparison` and
:func:`~ocean_skill.comparison.compare`, distinct requests must land distinct
cache entries and distinct pooling identities (a demeaned comparison and its raw
twin must never dedup into one point), and the normalizer's accepted shapes and
error messages are pinned down on their own.
"""

from __future__ import annotations

import inspect

import pytest

from ocean_skill.comparison import (
    Comparison,
    _identity,
    _normalize_subtract_mean,
    compare,
)


def _bare(**kwargs):
    with pytest.warns(UserWarning, match="resolved to standard_name"):
        return Comparison(
            reference="glodap", test="some_model", variable="temperature",
            cache=False, **kwargs,
        )


# -- normalization: accepted shapes ---------------------------------------------------


def test_default_is_neither_lane():
    assert _normalize_subtract_mean(None) == {"test": False, "reference": False}
    assert _normalize_subtract_mean(False) == {"test": False, "reference": False}


def test_true_is_both_lanes():
    assert _normalize_subtract_mean(True) == {"test": True, "reference": True}


def test_string_sugar_names_one_lane():
    assert _normalize_subtract_mean("test") == {"test": True, "reference": False}
    assert _normalize_subtract_mean("reference") == {"test": False, "reference": True}


def test_one_sided_dict_defaults_the_missing_side_to_false():
    assert _normalize_subtract_mean({"test": True}) == {
        "test": True,
        "reference": False,
    }
    assert _normalize_subtract_mean({"reference": True}) == {
        "test": False,
        "reference": True,
    }


def test_full_dict_is_unchanged():
    assert _normalize_subtract_mean({"test": True, "reference": False}) == {
        "test": True,
        "reference": False,
    }


# -- normalization: rejected shapes ----------------------------------------------------


def test_unknown_string_is_refused():
    with pytest.raises(ValueError, match=r"test.*reference"):
        _normalize_subtract_mean("bogus")


def test_extra_dict_key_is_named():
    with pytest.raises(ValueError, match="tset"):
        _normalize_subtract_mean({"tset": True})


def test_non_bool_value_is_refused():
    with pytest.raises(TypeError, match="test"):
        _normalize_subtract_mean({"test": "yes"})


def test_non_dict_non_bool_non_string_is_refused():
    with pytest.raises(TypeError, match="subtract_mean"):
        _normalize_subtract_mean(0.5)


# -- Comparison: stored and defaulted --------------------------------------------------


def test_default_matches_neither_lane():
    assert _bare().subtract_mean == {"test": False, "reference": False}


def test_an_explicit_request_is_stored_normalized():
    assert _bare(subtract_mean="test").subtract_mean == {
        "test": True,
        "reference": False,
    }


def test_subtract_mean_is_a_keyword_of_compare_and_defaults_to_the_same_value():
    compare_default = inspect.signature(compare).parameters["subtract_mean"].default
    init_default = inspect.signature(Comparison.__init__).parameters[
        "subtract_mean"
    ].default
    assert compare_default is False and init_default is False
    # And both normalize to "neither lane", the same as an explicit Comparison().
    assert _normalize_subtract_mean(compare_default) == _bare().subtract_mean


# -- cache keys: demeaning shares the raw pair's entry, never forks a new one ----------
# Demeaning is a scalar subtraction on the already-aligned pair, so the raw and the
# demeaned run of one comparison read the *same* cached regrid -- the key must not
# depend on subtract_mean at all (contrast pooling identity, below, which must).


def test_demeaning_does_not_change_the_cache_key():
    raw = _bare()
    for spec in (True, "test", "reference", {"test": True, "reference": False}):
        assert _bare(subtract_mean=spec)._cache_key == raw._cache_key


def test_every_subtract_mean_shape_shares_one_cache_key():
    keys = {
        _bare(subtract_mean=spec)._cache_key
        for spec in (False, True, "test", "reference", {"reference": True})
    }
    assert len(keys) == 1


# -- pooling identity: a demeaned comparison and its raw twin must stay distinct -------
# The counterpart to the cache-key sharing above: one shared entry on disk, but two
# distinct points on a pooled diagram (see _flatten's dedup).


def test_identity_distinguishes_demeaned_from_raw():
    raw = _bare()
    demeaned = _bare(subtract_mean=True)
    assert _identity(raw) != _identity(demeaned)


def test_identity_distinguishes_which_lane_was_demeaned():
    test_only = _bare(subtract_mean="test")
    reference_only = _bare(subtract_mean="reference")
    assert _identity(test_only) != _identity(reference_only)
