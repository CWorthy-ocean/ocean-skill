"""``detide`` reaches every layer, the way ``subtract_mean`` does.

Mirrors ``tests/test_subtract_mean_plumbing.py`` in structure -- defaults have to
match across :class:`~ocean_skill.comparison.Comparison`/:func:`~ocean_skill.
comparison.compare` and :class:`~ocean_skill.field.Field`/:func:`~ocean_skill.
field.field`, the normalizer's accepted shapes and error messages are pinned down
on their own, and pooling identity must distinguish a detided comparison from its
raw twin. One thing does **not** mirror ``subtract_mean``: ``detide`` runs *before*
alignment, on the lane itself (see ``ocean_skill.comparison._prepare``'s
``detide=`` paragraph), so unlike demeaning it *does* change the aligned-pair cache
key -- the opposite assertion from ``test_subtract_mean_plumbing.py``'s cache-key
tests.
"""

from __future__ import annotations

import inspect

import pytest

from ocean_skill.comparison import (
    Comparison,
    _identity,
    _normalize_detide,
    _normalize_detide_side,
    compare,
)
from ocean_skill.field import Field, field


def _bare(**kwargs):
    with pytest.warns(UserWarning, match="resolved to standard_name"):
        return Comparison(
            reference="glodap", test="some_model", variable="temperature",
            cache=False, **kwargs,
        )


# -- normalization (Comparison/compare's per-lane shape): accepted ---------------------


def test_default_is_neither_lane():
    assert _normalize_detide(None) == {"test": None, "reference": None}
    assert _normalize_detide(False) == {"test": None, "reference": None}


def test_true_is_both_lanes_at_pl33_default():
    assert _normalize_detide(True) == {
        "test": {"T": 33.0},
        "reference": {"T": 33.0},
    }


def test_string_sugar_names_one_lane():
    assert _normalize_detide("test") == {"test": {"T": 33.0}, "reference": None}
    assert _normalize_detide("reference") == {"test": None, "reference": {"T": 33.0}}


def test_one_sided_dict_defaults_the_missing_side_to_none():
    assert _normalize_detide({"test": True}) == {
        "test": {"T": 33.0},
        "reference": None,
    }


def test_pair_dict_gives_each_lane_its_own_cutoff():
    assert _normalize_detide({"test": {"T": 72}, "reference": True}) == {
        "test": {"T": 72.0},
        "reference": {"T": 33.0},
    }


def test_bare_cutoff_dict_applies_to_both_lanes():
    # No "test"/"reference" key -- the dict counterpart of plain `True`, with T set.
    assert _normalize_detide({"T": 72}) == {
        "test": {"T": 72.0},
        "reference": {"T": 72.0},
    }


# -- normalization: rejected shapes -------------------------------------------------


def test_unknown_string_is_refused():
    with pytest.raises(ValueError, match=r"test.*reference"):
        _normalize_detide("bogus")


def test_extra_pair_key_is_named():
    with pytest.raises(ValueError, match="tset"):
        _normalize_detide({"tset": True})


def test_extra_per_lane_key_is_named():
    with pytest.raises(ValueError, match="bogus"):
        _normalize_detide({"test": {"T": 72, "bogus": 1}})


def test_non_dict_non_bool_non_string_is_refused():
    with pytest.raises(TypeError, match="detide"):
        _normalize_detide(0.5)


def test_side_normalizer_matches_the_pair_normalizer_per_lane():
    # Field has one lane, not two -- _normalize_detide_side is what it uses directly.
    assert _normalize_detide_side(None) is None
    assert _normalize_detide_side(False) is None
    assert _normalize_detide_side(True) == {"T": 33.0}
    assert _normalize_detide_side({"T": 72}) == {"T": 72.0}


# -- Comparison: stored and defaulted -----------------------------------------------


def test_default_matches_neither_lane():
    assert _bare().detide == {"test": None, "reference": None}


def test_an_explicit_request_is_stored_normalized():
    assert _bare(detide="test").detide == {"test": {"T": 33.0}, "reference": None}


def test_detide_is_a_keyword_of_compare_and_defaults_to_the_same_value():
    compare_default = inspect.signature(compare).parameters["detide"].default
    init_default = inspect.signature(Comparison.__init__).parameters["detide"].default
    assert compare_default is False and init_default is False
    assert _normalize_detide(compare_default) == _bare().detide


def test_detide_is_a_keyword_of_field_and_defaults_to_the_same_value():
    compare_default = inspect.signature(field).parameters["detide"].default
    init_default = inspect.signature(Field.__init__).parameters["detide"].default
    assert compare_default is False and init_default is False
    assert _normalize_detide_side(init_default) is None


# -- cache keys: unlike subtract_mean, detide *does* change the aligned-pair key -------
# Detiding filters each lane before alignment ever runs (see _prepare's detide=
# paragraph), so the aligned pair a raw run and a detided run would produce are
# genuinely different data -- they must never share one cache entry (contrast
# subtract_mean, a post-align scalar shift that shares its raw twin's entry).


def test_detiding_changes_the_cache_key():
    raw = _bare()
    for spec in (True, "test", "reference", {"test": True, "reference": False}):
        assert _bare(detide=spec)._cache_key != raw._cache_key


def test_different_lanes_detided_land_different_keys():
    keys = {
        _bare(detide=spec)._cache_key
        for spec in (False, True, "test", "reference", {"reference": True})
    }
    # 4, not 5 -- "reference" and {"reference": True} normalize to the same request
    # (that lane detided at PL33's default cutoff) and so legitimately share a key.
    assert len(keys) == 4


def test_different_cutoffs_land_different_keys():
    assert _bare(detide={"T": 33})._cache_key != _bare(detide={"T": 72})._cache_key


# -- pooling identity: a detided comparison and its raw twin must stay distinct --------


def test_identity_distinguishes_detided_from_raw():
    raw = _bare()
    detided = _bare(detide=True)
    assert _identity(raw) != _identity(detided)


def test_identity_distinguishes_which_lane_was_detided():
    test_only = _bare(detide="test")
    reference_only = _bare(detide="reference")
    assert _identity(test_only) != _identity(reference_only)


def test_identity_distinguishes_different_cutoffs():
    t33 = _bare(detide={"T": 33})
    t72 = _bare(detide={"T": 72})
    assert _identity(t33) != _identity(t72)
