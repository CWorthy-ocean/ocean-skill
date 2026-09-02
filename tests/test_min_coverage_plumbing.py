"""``min_coverage`` reaches :func:`ocean_skill.align.align` from both entry points.

Fixing the NaN-contamination bug in a regrid's coverage accounting (see
``tests/test_regrid_target.py``) meant the threshold that decides which
partially-covered cells survive finally does something real -- so it has to be
reachable from :class:`~ocean_skill.comparison.Comparison`/:func:`~ocean_skill.
comparison.compare`, not only the lower-level :func:`~ocean_skill.align.align`, and
two different values must land two different aligned pairs, not share a cache entry
holding one or the other.
"""

import inspect

import pytest

from ocean_skill.comparison import Comparison, compare


def _bare(**kwargs):
    with pytest.warns(UserWarning, match="resolved to standard_name"):
        return Comparison(
            reference="glodap", test="some_model", variable="temperature",
            cache=False, **kwargs,
        )


def test_default_min_coverage_matches_aligns_own_default():
    import ocean_skill.align as align_module

    default = inspect.signature(align_module.align).parameters["min_coverage"].default
    assert _bare().min_coverage == default


def test_an_explicit_min_coverage_is_stored_and_forwarded():
    c = _bare(min_coverage=0.2)
    assert c.min_coverage == 0.2


def test_two_min_coverage_values_produce_different_cache_keys():
    strict = _bare(min_coverage=0.8)
    lenient = _bare(min_coverage=0.1)
    assert strict._cache_key != lenient._cache_key


def test_min_coverage_is_a_keyword_of_compare_and_defaults_to_the_same_value():
    sig = inspect.signature(compare)
    assert sig.parameters["min_coverage"].default == _bare().min_coverage
