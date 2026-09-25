"""Tests for :mod:`ocean_skill.workflows.pages`: expanding a suite into pages.

Pure and fast -- no catalog, no real data. ``extrema._native_time_index`` is
monkeypatched to a small, fixed index (Jan-Mar 2010), the same stub pattern
``tests/test_extrema.py`` uses, since resolving ``latest``/``month: run`` and the
injected time windows is the one place :func:`~ocean_skill.workflows.pages.expand`
reads anything at all -- and even that read is coordinate-only.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ocean_skill.config import SuiteConfig
from ocean_skill.workflows import pages as P

INDEX = pd.date_range("2010-01-05", periods=10, freq="7D")  # Jan 5 .. Mar 9, 2010


@pytest.fixture(autouse=True)
def _time_index(monkeypatch):
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: INDEX)


def _suite(pages, **top):
    return SuiteConfig.model_validate({"name": "t", "pages": pages, **top})


# -- schema validation ----------------------------------------------------------------


def test_no_pages_is_a_schema_error():
    with pytest.raises(Exception):  # pydantic ValidationError
        SuiteConfig.model_validate({"name": "t", "pages": []})


def test_a_page_needs_exactly_one_kind():
    with pytest.raises(Exception):
        SuiteConfig.model_validate(
            {"name": "t", "pages": [{"title": "x", "field": {}, "compare": {}}]}
        )
    with pytest.raises(Exception):
        SuiteConfig.model_validate({"name": "t", "pages": [{"title": "x"}]})


def test_defaults_test_must_be_a_single_source():
    with pytest.raises(Exception):
        SuiteConfig.model_validate(
            {
                "name": "t",
                "defaults": {"test": ["a", "b"]},
                "pages": [{"title": "x", "field": {"variables": ["temperature"]}}],
            }
        )


def test_catalog_search_paths_defaults_to_empty_list():
    suite = _suite([{"title": "x", "field": {"variables": ["temperature"]}}])
    assert suite.catalog_search_paths == []


def test_catalog_search_paths_accepts_absolute_relative_and_home_paths():
    suite = _suite(
        [{"title": "x", "field": {"variables": ["temperature"]}}],
        catalog_search_paths=["/abs/dir", "rel/dir", "~/dir"],
    )
    assert suite.catalog_search_paths == ["/abs/dir", "rel/dir", "~/dir"]


def test_catalog_search_paths_rejects_blank_entries():
    with pytest.raises(Exception):
        _suite(
            [{"title": "x", "field": {"variables": ["temperature"]}}],
            catalog_search_paths=[""],
        )


# -- for_each --------------------------------------------------------------------------


def test_for_each_product_order_and_count():
    suite = _suite(
        [
            {
                "title": "{a}-{b}",
                "for_each": {"a": [1, 2], "b": ["x", "y"]},
                "field": {"variables": ["temperature"], "select": {"depth": "surface"}},
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert [p.title for p in out] == ["1-x", "1-y", "2-x", "2-y"]


def test_whole_object_placeholder_keeps_the_list_type():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": ["temperature"],
                    "select": {"depth": "{depths}"},
                },
            }
        ],
        defaults={"test": "stub", "depths": ["surface", 100]},
    )
    out = P.expand(suite)
    assert out[0].kwargs["select"]["depth"] == ["surface", 100]
    assert isinstance(out[0].kwargs["select"]["depth"], list)


def test_for_each_defaults_collision_is_a_schema_error():
    suite = _suite(
        [
            {
                "title": "x",
                "for_each": {"depths": ["surface"]},
                "field": {"variables": ["temperature"], "select": {}},
            }
        ],
        defaults={"test": "stub", "depths": ["surface", 100]},
    )
    with pytest.raises(ValueError, match="collide"):
        P.expand(suite)


def test_unknown_placeholder_names_the_page():
    suite = _suite(
        [{"title": "{nope}", "field": {"variables": ["temperature"], "select": {}}}],
        defaults={"test": "stub"},
    )
    with pytest.raises(ValueError, match="nope"):
        P.expand(suite)


# -- month: run ------------------------------------------------------------------------


def test_month_run_lists_every_calendar_month():
    suite = _suite(
        [
            {
                "title": "{month.name} {month.year}",
                "for_each": {"month": "run"},
                "compare": {
                    "reference": ["woa23_nitrate_month{month.mm}"],
                    "variables": ["nitrate"],
                    "select": {
                        "test": {"time": "{month.window}"},
                        "reference": {},
                    },
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert [p.title for p in out] == ["January 2010", "February 2010", "March 2010"]
    assert out[0].kwargs["reference"] == ["woa23_nitrate_month01"]


def test_month_run_last_n_keeps_the_tail():
    suite = _suite(
        [
            {
                "title": "{month.name}",
                "for_each": {"month": {"run": {"last": 1}}},
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "select": {"test": {"time": "{month.window}"}, "reference": {}},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert [p.title for p in out] == ["March"]


def test_month_window_end_is_inclusive_last_day():
    suite = _suite(
        [
            {
                "title": "{month.window}",
                "for_each": {"month": {"run": {"last": 1}}},
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "select": {"test": {"time": "{month.window}"}, "reference": {}},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    window = out[0].kwargs["select"]["test"]["time"]
    assert window == {"min": "2010-03-01T00:00:00", "max": "2010-03-31T23:59:59"}


def test_depth_label_formats_with_units():
    suite = _suite(
        [
            {
                "title": "at {depth.label}",
                "for_each": {"depth": "{depths}"},
                "field": {"variables": ["temperature"], "select": {"depth": "{depth}"}},
            }
        ],
        defaults={"test": "stub", "depths": ["surface", 100]},
    )
    out = P.expand(suite)
    assert [p.title for p in out] == ["at surface", "at 100 m"]
    assert out[0].kwargs["select"]["depth"] == "surface"
    assert out[1].kwargs["select"]["depth"] == 100  # stays an int, not "100"


def test_literal_braces_survive():
    suite = _suite(
        [
            {
                "title": "{{literal}} {a}",
                "for_each": {"a": [1]},
                "field": {"variables": ["temperature"], "select": {}},
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].title == "{literal} 1"


# -- time: latest ----------------------------------------------------------------------


def test_latest_resolves_to_the_index_last_value_no_method_key():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": ["temperature"],
                    "select": {"depth": "surface", "time": "latest"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].kwargs["select"]["time"] == INDEX[-1].isoformat()
    assert "method" not in out[0].kwargs["select"]


def test_latest_page_is_never_cached():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": ["temperature"],
                    "select": {"depth": "surface", "time": "latest"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is False


# -- window injection & the cache flag -------------------------------------------------


def test_a_field_page_with_no_time_key_gets_the_whole_run_window():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {"variables": ["temperature"], "select": {"depth": "surface"}},
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].kwargs["select"]["time"] == {
        "min": INDEX[0].isoformat(),
        "max": INDEX[-1].isoformat(),
    }
    assert out[0].cache is False  # open-ended: reaches the run's latest step


def test_a_completed_woa_month_page_keeps_caching():
    suite = _suite(
        [
            {
                "title": "{month.name}",
                "for_each": {"month": "run"},
                "compare": {
                    "reference": ["woa23_nitrate_month{month.mm}"],
                    "variables": ["nitrate"],
                    "select": {"test": {"time": "{month.window}"}, "reference": {}},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    by_title = {p.title: p for p in out}
    assert by_title["January"].cache is True
    assert by_title["February"].cache is True
    assert by_title["March"].cache is False  # contains the run's latest step


def test_a_compare_page_with_no_select_gets_the_whole_run_window():
    suite = _suite(
        [
            {
                "title": "glodap",
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "aggregate": {"time": "mean"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].kwargs["select"]["test"]["time"] == {
        "min": INDEX[0].isoformat(),
        "max": INDEX[-1].isoformat(),
    }
    assert out[0].cache is False


def test_suite_level_cache_false_forces_every_page():
    suite = _suite(
        [
            {
                "title": "{month.name}",
                "for_each": {"month": "run"},
                "compare": {
                    "reference": ["woa23_nitrate_month{month.mm}"],
                    "variables": ["nitrate"],
                    "select": {"test": {"time": "{month.window}"}, "reference": {}},
                },
            }
        ],
        defaults={"test": "stub"},
        cache=False,
    )
    out = P.expand(suite)
    assert all(p.cache is False for p in out)


# -- semantic checks -------------------------------------------------------------------


def test_a_calculate_spec_cannot_sit_beside_a_real_depth():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": [{"calculate": "mld", "method": "density_threshold"}],
                    "select": {"depth": 100},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    with pytest.raises(ValueError, match="vertical axis"):
        P.expand(suite)


def test_a_calculate_spec_at_the_surface_is_fine():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": [{"calculate": "mld", "method": "density_threshold"}],
                    "select": {"depth": "surface"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    P.expand(suite)  # does not raise


# -- reproducibility -------------------------------------------------------------------


def test_expand_is_json_serializable_and_deterministic():
    suite = _suite(
        [
            {
                "title": "{month.name} {depth.label}",
                "for_each": {"month": "run", "depth": "{depths}"},
                "compare": {
                    "reference": ["woa23_nitrate_month{month.mm}"],
                    "variables": ["nitrate"],
                    "select": {
                        "test": {"depth": "{depth}", "time": "{month.window}"},
                        "reference": {"depth": "{depth}"},
                    },
                },
            }
        ],
        defaults={"test": "stub", "depths": ["surface", 100]},
    )
    out1 = P.expand(suite)
    out2 = P.expand(suite)
    dicts1 = [p.as_dict() for p in out1]
    dicts2 = [p.as_dict() for p in out2]
    assert json.dumps(dicts1, sort_keys=True) == json.dumps(dicts2, sort_keys=True)
