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


def test_cache_dir_defaults_to_none():
    suite = _suite([{"title": "x", "field": {"variables": ["temperature"]}}])
    assert suite.cache_dir is None


def test_cache_dir_accepts_absolute_relative_and_home_paths():
    for entry in ("/abs/dir", "rel/dir", "~/dir"):
        suite = _suite(
            [{"title": "x", "field": {"variables": ["temperature"]}}],
            cache_dir=entry,
        )
        assert suite.cache_dir == entry


def test_cache_dir_rejects_blank():
    with pytest.raises(Exception):
        _suite(
            [{"title": "x", "field": {"variables": ["temperature"]}}],
            cache_dir="   ",
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


def test_latest_page_is_cached_pinned_to_the_resolved_step():
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
    # cache=True is safe here precisely because the resolved value above is baked
    # into the cache key: a rerun against an unchanged run's-end hits, and a
    # rerun once the run has moved recomputes under a different key.
    assert out[0].cache is True


# -- window injection & the cache flag -------------------------------------------------
#
# The cache flag follows one rule (see "Caching" in docs/suites.md): cache whenever the
# test lane's time selection is *pinned* to the run's own last step -- so its key
# changes the moment that step does -- or *closed* against the run's current time axis:
# selects at least one step, none of them the run's current last step, and would select
# exactly the same steps with one more step appended or its own last step taken away.


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
    # pinned: the injected window's own "max" is the run's last step, so its key
    # changes the moment that step does.
    assert out[0].cache is True


def test_a_completed_woa_month_page_keeps_caching_and_the_open_month_now_does_too():
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
    # March's own window still only ever selects what the run actually has (Mar 2
    # and Mar 9) -- closed against a *contiguous* window would be wrong here, since
    # the window's nominal "max" (Mar 31) is well past the run's end, but the run's
    # last recorded step (Mar 9) is what is actually selected, so it stays uncached.
    assert by_title["March"].cache is False


def test_a_field_page_for_each_month_gets_the_same_open_month_treatment():
    suite = _suite(
        [
            {
                "title": "{month.name}",
                "for_each": {"month": "run"},
                "field": {
                    "variables": ["temperature"],
                    "select": {"time": "{month.window}"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    by_title = {p.title: p for p in out}
    assert by_title["January"].cache is True
    assert by_title["February"].cache is True
    assert by_title["March"].cache is False


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
    assert out[0].cache is True


def test_a_pair_spec_test_lane_explicitly_null_is_treated_as_no_time_key():
    suite = _suite(
        [
            {
                "title": "x",
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "select": {"test": None, "reference": {}},
                    "aggregate": {"time": "mean"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)  # does not raise
    assert out[0].kwargs["select"]["test"]["time"] == {
        "min": INDEX[0].isoformat(),
        "max": INDEX[-1].isoformat(),
    }
    assert out[0].cache is True


def test_a_flat_compare_select_with_no_time_key_is_never_cached():
    # a flat (non-paired) select narrows both lanes at once, so it is never
    # rewritten the way a pair-spec's test lane is -- only checked.
    suite = _suite(
        [
            {
                "title": "x",
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "select": {"depth": "surface"},
                    "aggregate": {"time": "mean"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].kwargs["select"] == {"depth": "surface"}  # untouched
    assert out[0].cache is False


def test_a_flat_compare_window_inside_the_run_is_cached():
    suite = _suite(
        [
            {
                "title": "x",
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "select": {"time": {"min": "2010-01-05", "max": "2010-01-26"}},
                    "aggregate": {"time": "mean"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is True


def test_a_flat_compare_window_reaching_the_last_step_is_not_cached():
    suite = _suite(
        [
            {
                "title": "x",
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "select": {"time": {"min": "2010-03-01", "max": "2010-03-09"}},
                    "aggregate": {"time": "mean"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is False


def test_a_period_containing_the_last_step_is_not_cached():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {"variables": ["temperature"], "select": {"time": "2010-03"}},
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is False


def test_a_period_well_before_the_last_step_is_cached():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {"variables": ["temperature"], "select": {"time": "2010-01"}},
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is True


def test_an_instant_nearest_matching_the_last_step_is_not_cached():
    # "2010-03-16" names no real step -- the nearest one is the run's own last
    # step (Mar 9), so this is exactly as unsafe to cache as asking for it by name.
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": ["temperature"],
                    "select": {"time": "2010-03-16"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is False


def test_an_instant_well_inside_the_run_is_cached():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": ["temperature"],
                    "select": {"time": "2010-01-12"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is True


def test_a_negative_index_tracking_the_tail_is_not_cached():
    # index=-2 names a *position*, which shifts to a different actual step once
    # the run gains (or loses) a step at the end.
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": ["temperature"],
                    "select": {"time": {"index": -2}},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is False


def test_a_positive_index_range_from_the_start_is_cached():
    # positions 0 and 1 name the same two steps regardless of what happens at the
    # tail.
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": ["temperature"],
                    "select": {"time": {"index": {"min": 0, "max": 2}}},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is True


def test_a_window_entirely_past_the_run_is_not_cached():
    suite = _suite(
        [
            {
                "title": "x",
                "field": {
                    "variables": ["temperature"],
                    "select": {"time": {"min": "2010-03-16", "max": "2010-03-23"}},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is False


def test_detide_pushes_the_cutoff_back_by_its_own_cutoff_period():
    # the same window, only the detide period differs: T=200h reaches back far
    # enough to catch Mar 2 (a week before the run's last step, Mar 9); a window
    # ending three weeks earlier (Feb 23) clears it either way.
    close_to_the_end = _suite(
        [
            {
                "title": "x",
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "select": {
                        "test": {"time": {"min": "2010-01-01", "max": "2010-03-02"}},
                        "reference": {},
                    },
                    "detide": {"T": 200},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    assert P.expand(close_to_the_end)[0].cache is False

    well_clear = _suite(
        [
            {
                "title": "x",
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "select": {
                        "test": {"time": {"min": "2010-01-01", "max": "2010-02-23"}},
                        "reference": {},
                    },
                    "detide": {"T": 200},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    assert P.expand(well_clear)[0].cache is True


def test_times_fan_is_never_cached_even_with_no_select():
    # times= replaces whatever time entry select carried with its own per-bin
    # value at draw time (comparison._fanned_time_select) -- what actually gets
    # keyed is not what expand() resolved above, so nothing here can vouch for it.
    suite = _suite(
        [
            {
                "title": "x",
                "compare": {
                    "reference": ["glodap"],
                    "variables": ["alkalinity"],
                    "times": {"resample": "1MS", "reduce": "mean"},
                },
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].cache is False


def test_a_shared_select_placeholder_is_not_mutated_across_pages():
    suite = _suite(
        [
            {"title": "a", "field": {"variables": ["temperature"], "select": "{sel}"}},
            {"title": "b", "field": {"variables": ["salinity"], "select": "{sel}"}},
        ],
        defaults={"test": "stub", "sel": {"depth": "surface", "time": "latest"}},
    )
    out = P.expand(suite)
    assert out[0].kwargs["select"]["time"] == INDEX[-1].isoformat()
    assert out[1].kwargs["select"]["time"] == INDEX[-1].isoformat()
    # each page resolved its own copy, not fighting over one shared dict
    assert out[0].kwargs["select"] is not out[1].kwargs["select"]
    # and the suite's own stored default was never written through
    assert suite.defaults["sel"]["time"] == "latest"


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


# -- pinning size/zoom/figsize to the page when pdf: true ------------------------------


def test_zoom_is_pinned_to_page_and_warned_once_when_pdf_is_on():
    suite = _suite(
        [
            {"title": "a", "field": {"variable": "temperature"}},
            {"title": "b", "field": {"variable": "salinity"}},
        ],
        defaults={"test": "stub", "plot": {"zoom": 1.5}},
    )
    assert suite.pdf  # default
    with pytest.warns(UserWarning, match=r"zoom=1\.5.*ignored"):
        out = P.expand(suite)

    assert [p.plot for p in out] == [{"size": "page"}, {"size": "page"}]


def test_zoom_is_untouched_when_pdf_is_false():
    suite = _suite(
        [{"title": "a", "field": {"variable": "temperature"}}],
        defaults={"test": "stub", "plot": {"zoom": 1.5}},
        pdf=False,
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = P.expand(suite)  # no warning raised

    assert out[0].plot == {"zoom": 1.5}


def test_per_page_figsize_is_pinned_and_warned_independently_of_defaults_zoom():
    suite = _suite(
        [
            {
                "title": "a",
                "field": {"variable": "temperature"},
                "plot": {"figsize": [12, 3]},
            },
            {"title": "b", "field": {"variable": "salinity"}},
        ],
        defaults={"test": "stub", "plot": {"zoom": 1.5}},
    )
    with pytest.warns(UserWarning) as caught:
        out = P.expand(suite)

    messages = {str(w.message) for w in caught.list}
    assert any("figsize=[12, 3]" in m for m in messages)
    assert any("zoom=1.5" in m for m in messages)
    # each distinct kwarg warns once across the whole suite, not once per page
    assert sum("zoom=1.5" in m for m in messages) == 1
    assert sum("figsize=" in m for m in messages) == 1
    assert out[0].plot == {"size": "page"}
    assert out[1].plot == {"size": "page"}


# -- then: a field page's method chain -------------------------------------------------


def _then_suite(then, *, select=None, aggregate=None, variables=None):
    field = {
        "variables": variables or ["alkalinity"],
        "select": select or {"depth": "surface", "time": "2010-01-19"},
    }
    if aggregate is not None:
        field["aggregate"] = aggregate
    return _suite(
        [{"title": "x", "field": field, "then": then}],
        defaults={"test": "stub"},
    )


def test_normalize_step_accepts_bare_scalar_and_dict_forms():
    assert P._normalize_step("extremum", title="t") == {
        "name": "extremum",
        "kwargs": {"kind": "max"},
    }
    assert P._normalize_step({"extremum": "min"}, title="t") == {
        "name": "extremum",
        "kwargs": {"kind": "min"},
    }
    assert P._normalize_step({"extremum": {"kind": "min"}}, title="t") == {
        "name": "extremum",
        "kwargs": {"kind": "min"},
    }
    assert P._normalize_step({"series": {"variables": ["a", "b"]}}, title="t") == {
        "name": "series",
        "kwargs": {"variables": ["a", "b"]},
    }


def test_normalize_step_rejects_an_unknown_step():
    with pytest.raises(ValueError, match="not a known step"):
        P._normalize_step("bogus", title="t")


def test_normalize_step_rejects_an_unknown_kwarg():
    with pytest.raises(ValueError, match="extra"):
        P._normalize_step({"extremum": {"kinds": "min"}}, title="t")


def test_normalize_step_rejects_time_latest_inside_series():
    with pytest.raises(ValueError, match="then: step"):
        P._normalize_step({"series": {"time": "latest"}}, title="t")


def test_config_rejects_an_unknown_step_name():
    with pytest.raises(Exception):
        SuiteConfig.model_validate(
            {
                "name": "t",
                "defaults": {"test": "stub"},
                "pages": [
                    {"title": "x", "field": {"variables": ["a"]}, "then": ["bogus"]}
                ],
            }
        )


def test_config_rejects_then_on_a_compare_page():
    with pytest.raises(Exception, match="field: pages"):
        SuiteConfig.model_validate(
            {
                "name": "t",
                "pages": [
                    {
                        "title": "x",
                        "compare": {
                            "test": "stub",
                            "reference": ["ref"],
                            "variables": ["x"],
                        },
                        "then": ["extremum"],
                    }
                ],
            }
        )


def test_expand_applies_placeholders_inside_then():
    suite = _suite(
        [
            {
                "title": "{variable}",
                "for_each": {"variable": ["alkalinity"]},
                "field": {
                    "variables": ["{variable}"],
                    "select": {"depth": "surface", "time": "2010-01-19"},
                },
                "then": [
                    {"extremum": "min"},
                    {"series": {"variables": ["{variable}"]}},
                ],
            }
        ],
        defaults={"test": "stub"},
    )
    out = P.expand(suite)
    assert out[0].steps[1]["kwargs"]["variables"] == ["alkalinity"]


def test_then_chain_out_of_order_is_refused():
    suite = _then_suite([{"series": {}}, {"extremum": "min"}])
    with pytest.raises(ValueError, match="check the step order"):
        P.expand(suite)


def test_then_refuses_a_multi_variable_page():
    suite = _then_suite(
        [{"extremum": "min"}], variables=["alkalinity", "dissolved_inorganic_carbon"]
    )
    with pytest.raises(ValueError, match="builds 2 members"):
        P.expand(suite)


def test_then_extremum_needs_an_explicit_time():
    suite = _then_suite([{"extremum": "min"}], select={"depth": "surface"})
    with pytest.raises(ValueError, match=r"explicit select\.time"):
        P.expand(suite)


def test_then_extremum_time_range_is_accepted_without_pad_resolution():
    suite = _then_suite(
        [{"extremum": "min"}, {"series": {}}],
        select={"depth": "surface", "time": {"min": "2010-01-01", "max": "2010-02-01"}},
    )
    out = P.expand(suite)
    assert out[0].steps[1]["kwargs"] == {}  # left for Extremum.series()'s own default


def test_then_extremum_needs_an_explicit_vertical_key():
    suite = _then_suite([{"extremum": "min"}], select={"time": "2010-01-19"})
    with pytest.raises(ValueError, match="explicit vertical key"):
        P.expand(suite)


def test_then_extremum_with_a_time_collapsing_aggregate_is_allowed():
    suite = _then_suite(
        [{"extremum": "min"}],
        select={"depth": "surface"},
        aggregate={"time": "mean"},
    )
    out = P.expand(suite)
    assert out[0].steps == [{"name": "extremum", "kwargs": {"kind": "min"}}]


def test_then_series_window_resolves_to_a_literal_for_a_fixed_snapshot():
    suite = _then_suite(
        [{"extremum": "min"}, {"series": {"pad": 1}}],
        select={"depth": "surface", "time": "2010-01-19"},
    )
    out = P.expand(suite)
    series_kwargs = out[0].steps[1]["kwargs"]
    assert series_kwargs["time"] == {
        "min": "2010-01-12 00:00:00",
        "max": "2010-01-26 00:00:00",
    }
    assert series_kwargs["cache"] is True  # well inside the record, not an open window


def test_then_series_window_reaching_latest_is_marked_uncached():
    suite = _then_suite(
        [{"extremum": "min"}, {"series": {}}],
        select={"depth": "surface", "time": "latest"},
    )
    out = P.expand(suite)
    # The locator field's own page-level flag caches: "latest" is pinned to the
    # resolved step (see "Caching" in docs/suites.md), so its key changes the
    # moment that step does. The nested series step is a different question --
    # its own padded window reaches the run's current end, so *that* stays
    # uncached regardless of the outer page's own flag.
    assert out[0].cache is True
    assert out[0].steps[1]["kwargs"]["cache"] is False


def test_then_series_explicit_time_is_left_alone():
    suite = _then_suite(
        [{"extremum": "min"}, {"series": {"time": "2010-02-02"}}],
        select={"depth": "surface", "time": "2010-01-19"},
    )
    out = P.expand(suite)
    assert out[0].steps[1]["kwargs"] == {"time": "2010-02-02"}


def test_expand_with_then_steps_is_json_serializable_and_deterministic():
    suite = _then_suite(
        [{"extremum": "min"}, {"series": {"variables": ["dissolved_inorganic_carbon"]}}]
    )
    out1 = P.expand(suite)
    out2 = P.expand(suite)
    dicts1 = [p.as_dict() for p in out1]
    dicts2 = [p.as_dict() for p in out2]
    assert json.dumps(dicts1, sort_keys=True) == json.dumps(dicts2, sort_keys=True)
    assert dicts1[0]["steps"][0] == {"name": "extremum", "kwargs": {"kind": "min"}}


def test_summary_pages_own_kwargs_are_pinned_too():
    # a summary page's sizing kwargs live directly on `summary:`, not `plot:` -- see
    # ocean_skill.workflows.pages.build, which forwards page.kwargs (not page.plot)
    # to osk.summary().
    suite = _suite(
        [
            {
                "title": "a",
                "compare": {"test": "stub", "reference": ["ref"], "variables": ["x"]},
            },
            {"title": "overview", "summary": {"zoom": 2.0}},
        ],
        defaults={"test": "stub"},
    )
    with pytest.warns(UserWarning, match=r"zoom=2\.0.*ignored"):
        out = P.expand(suite)

    summary_page = next(p for p in out if p.kind == "summary")
    assert summary_page.kwargs["size"] == "page"
    assert "zoom" not in summary_page.kwargs
