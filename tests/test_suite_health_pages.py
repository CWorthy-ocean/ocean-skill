"""``health:``/``rivers:`` suite pages, ``segment_starts``, ``kind: river`` refresh.

Pure and fast: the diagnostic objects (:func:`ocean_skill.run_health.health`,
:func:`ocean_skill.river_inputs.rivers`) are replaced by fakes, and
``extrema._native_time_index`` by a fixed index, so nothing here reads data.
"""

from __future__ import annotations

import sys
import types

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import yaml

from ocean_skill import build
from ocean_skill.config import HEALTH_DRAWS, RIVER_DRAWS, SuiteConfig
from ocean_skill.workflows import cover
from ocean_skill.workflows import pages as P

INDEX = pd.date_range("2010-01-05", periods=10, freq="7D")  # Jan 5 .. Mar 9, 2010


@pytest.fixture(autouse=True)
def _time_index(monkeypatch):
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: INDEX)


class FakeObject:
    """Stands in for RunHealth / RiverInputs: two figures per draw but ``empty``."""

    def __init__(self, label):
        self.label = label

    def figures(self, draw):
        if draw == "empty":
            return []
        return [("", plt.figure()), ("_b", plt.figure())]

    def table(self):
        return pd.DataFrame({"x": [1, 2]})

    def problems(self):
        return [{"kind": "fake", "label": self.label}]


@pytest.fixture
def fakes(monkeypatch):
    """Install fake ``run_health`` / ``river_inputs`` modules; return the call logs."""
    calls = {"health": [], "rivers": []}

    def health(source, *, time=None, joins=None, cache=True):
        calls["health"].append(
            {"source": source, "time": time, "joins": joins, "cache": cache}
        )
        return FakeObject("health")

    def rivers(source, *, time=None):
        calls["rivers"].append({"source": source, "time": time})
        return FakeObject("rivers")

    rh = types.ModuleType("ocean_skill.run_health")
    rh.health, rh.HEALTH_DRAWS = health, HEALTH_DRAWS
    ri = types.ModuleType("ocean_skill.river_inputs")
    ri.rivers, ri.RIVER_DRAWS = rivers, RIVER_DRAWS
    monkeypatch.setitem(sys.modules, "ocean_skill.run_health", rh)
    monkeypatch.setitem(sys.modules, "ocean_skill.river_inputs", ri)
    yield calls
    plt.close("all")


def _suite(pages, **top):
    return SuiteConfig.model_validate(
        {"name": "t", "defaults": {"test": "run"}, "pages": pages, **top}
    )


# -- schema --------------------------------------------------------------------------


def test_a_bad_draw_is_a_schema_error_and_a_placeholder_is_deferred():
    with pytest.raises(ValueError, match="not a known figure"):
        _suite([{"title": "x", "health": {"draw": "nope"}}])
    with pytest.raises(ValueError, match="not a known figure"):
        _suite([{"title": "x", "rivers": {"draw": "summary"}}])  # a health draw
    _suite([{"title": "x", "health": {"draw": "{which}"}}])  # filled in later
    filled = _suite(
        [{"title": "x", "health": {"draw": "{which}"}}],
        defaults={"test": "run", "which": "nope"},
    )
    with pytest.raises(ValueError, match="not a known figure"):
        P.expand(filled)


def test_then_and_contours_are_refused_and_unknown_keys_too():
    with pytest.raises(ValueError, match="then:"):
        _suite([{"title": "x", "health": {}, "then": ["extremum"]}])
    with pytest.raises(ValueError, match="contours:"):
        _suite([{"title": "x", "rivers": {}, "contours": True}])
    with pytest.raises(ValueError, match="no key"):
        P.expand(_suite([{"title": "x", "health": {"variable": "temp"}}]))


def test_module_constants_equal_the_configs():
    pytest.importorskip("ocean_skill.run_health")
    pytest.importorskip("ocean_skill.river_inputs")
    # the fakes shadow the modules only inside the ``fakes`` fixture
    from ocean_skill import river_inputs, run_health

    assert run_health.HEALTH_DRAWS == HEALTH_DRAWS
    assert river_inputs.RIVER_DRAWS == RIVER_DRAWS


# -- expand --------------------------------------------------------------------------


def test_expand_fills_the_source_and_pins_the_window():
    pages = P.expand(
        _suite(
            [
                {"title": "whole run", "health": {"draw": "integrity"}},
                {"title": "open", "health": {"time": {"min": "2010-02-01"}}},
                {
                    "title": "river",
                    "rivers": {
                        "source": "rivs",
                        "time": {"min": "2009", "max": "2010"},
                    },
                },
            ]
        )
    )
    whole, open_, river = pages
    assert whole.kind == "health" and whole.problem is None
    assert whole.kwargs == {
        "source": "run",
        "draw": "integrity",
        "time": {"min": INDEX[0].isoformat(), "max": INDEX[-1].isoformat()},
    }
    assert whole.cache is True
    assert open_.kwargs["draw"] == "summary"  # the first of HEALTH_DRAWS
    assert open_.kwargs["time"] == {
        "min": "2010-02-01",
        "max": INDEX[-1].isoformat(),
    }
    assert open_.cache is True  # an open window is pinned to the run's end
    assert river.kind == "rivers" and river.kwargs["source"] == "rivs"
    assert river.kwargs["time"] == {"min": "2009", "max": "2010"}  # as written
    assert river.kwargs["draw"] == "map" and river.cache is False


def test_expand_templates_and_fans_out_and_reports_unreadable_sources(monkeypatch):
    pages = P.expand(
        _suite(
            [
                {
                    "title": "Health {draw}",
                    "health": {"draw": "{draw}"},
                    "for_each": {"draw": ["summary", "plankton"]},
                }
            ]
        )
    )
    assert [p.title for p in pages] == ["Health summary", "Health plankton"]

    def boom(source):
        raise KeyError(source)

    monkeypatch.setattr("ocean_skill.extrema._native_time_index", boom)
    (page,) = P.expand(_suite([{"title": "x", "health": {}}]))
    assert "could not be read" in page.problem and page.cache is False


# -- build ---------------------------------------------------------------------------


def test_build_shares_one_computation_between_pages_with_the_same_key(fakes):
    pages = P.expand(
        _suite(
            [
                {"title": "a", "health": {"draw": "summary"}},
                {"title": "b", "health": {"draw": "integrity"}},
                {"title": "c", "health": {"draw": "inventory", "time": "2010-02"}},
            ]
        )
    )
    ctx = {"joins": lambda: ["a-join"], "objects": {}}
    out = [P.build(p, run_context=ctx) for p in pages]
    assert [len(o) for o in out] == [2, 2, 2]
    assert len(fakes["health"]) == 2  # a and b share; c has another window
    assert fakes["health"][0]["joins"] == ["a-join"]
    assert fakes["health"][0]["source"] == "run"
    assert ctx["joins"] == ["a-join"]  # the callable ran once and was replaced


def test_build_records_the_table_and_problems_but_no_metrics(fakes):
    summary, integrity = P.expand(
        _suite(
            [
                {"title": "a", "health": {"draw": "summary"}},
                {"title": "b", "health": {"draw": "integrity"}},
            ]
        )
    )
    P.build(summary)
    P.build(integrity)
    assert summary.summary_table is None  # only integrity/extremes write a table
    assert list(integrity.summary_table["x"]) == [1, 2]
    assert integrity.results == [{"kind": "fake", "label": "health"}]
    assert integrity.metrics_records == [] and summary.metrics_records == []


def test_rivers_page_builds_through_the_rivers_module(fakes):
    (page,) = P.expand(_suite([{"title": "r", "rivers": {"draw": "extremes"}}]))
    figures = P.build(page)
    assert len(figures) == 2
    assert fakes["rivers"] == [{"source": "run", "time": None}]
    assert page.summary_table is not None


def test_a_page_with_no_figures_is_skipped_with_a_reason(fakes):
    (page,) = P.expand(_suite([{"title": "x", "health": {"draw": "plankton"}}]))

    class Empty(FakeObject):
        def figures(self, draw):
            return []

    sys.modules["ocean_skill.run_health"].health = lambda *a, **k: Empty("h")
    with pytest.raises(P.PageSkipped, match="nothing to draw"):
        P.build(page)


def test_a_failed_computation_is_remembered_and_raised_for_each_page(fakes):
    pages = P.expand(
        _suite(
            [
                {"title": "a", "health": {"draw": "summary"}},
                {"title": "b", "health": {"draw": "inventory"}},
            ]
        )
    )
    n = []

    def broken(*a, **k):
        n.append(1)
        raise RuntimeError("unreadable")

    sys.modules["ocean_skill.run_health"].health = broken
    ctx = {"joins": [], "objects": {}}
    for page in pages:
        with pytest.raises(RuntimeError, match="unreadable"):
            P.build(page, run_context=ctx)
    assert n == [1]


# -- segment_starts ------------------------------------------------------------------


def _write_workplan(tmp_path, starts):
    """Write a workplan with one blueprint per start date; return the suite."""
    root = tmp_path / "cstar"
    run_dir = root / "tasks" / "step_a"
    (run_dir / "output").mkdir(parents=True)
    (run_dir / "work").mkdir()
    (run_dir / "work" / "directives.yaml").write_text("{}")
    steps = []
    for i, start in enumerate(starts):
        bp = root / f"bp_{i}.yaml"
        bp.write_text(yaml.safe_dump({"runtime_params": {"start_date": start}}))
        steps.append(
            {
                "name": f"s{i}",
                # the first is absolute, the rest relative to the workplan's folder
                "blueprint": str(bp) if i == 0 else bp.name,
                "working_dir": str(run_dir),
            }
        )
    (root / "wp.yaml").write_text(yaml.safe_dump({"name": "WP", "steps": steps}))
    src = {
        "name": "run",
        "files": str(run_dir / "output" / "output_rst.*.nc"),
        "ref": "r.json",
        "grid": "g",
    }
    return _suite(
        [{"title": "t", "health": {}}], refresh={"catalog": "c.yaml", "sources": [src]}
    )


def test_segment_starts_reads_the_workplans_blueprints(tmp_path):
    suite = _write_workplan(
        tmp_path,
        ["2010-01-01 00:00:00", "2010-07-01", "2010-04-01", "2010-07-01"],
    )
    assert cover.segment_starts(suite) == [
        pd.Timestamp("2010-04-01").to_pydatetime(),
        pd.Timestamp("2010-07-01").to_pydatetime(),
    ]  # sorted, de-duplicated, without the run's own start


def test_segment_starts_never_raises(tmp_path):
    assert cover.segment_starts(_suite([{"title": "t", "health": {}}])) == []
    suite = _write_workplan(tmp_path, ["2010-01-01", "2010-02-01"])
    (tmp_path / "cstar" / "bp_1.yaml").write_text("a: [unclosed")
    with pytest.warns(UserWarning):
        assert cover.segment_starts(suite) == []
    (tmp_path / "cstar" / "bp_1.yaml").write_text("runtime_params: {start_date: nope}")
    with pytest.warns(UserWarning, match="no segment start dates"):
        assert cover.segment_starts(suite) == []


# -- refresh: kind: river ------------------------------------------------------------


def test_the_river_refresh_kind_dispatches_to_add_river_source(
    tmp_path, monkeypatch
):
    from ocean_skill.workflows.run import _refresh_forcing

    calls = []

    def fake_add(cat, name, files, grid, **kwargs):
        calls.append((name, [str(f) for f in files], str(grid), kwargs))
        return [name]

    monkeypatch.setattr(build, "add_river_source", fake_add)
    (tmp_path / "rivers.nc").write_text("")
    catalog = tmp_path / "run.yaml"
    build.save(build.new_catalog(title="run"), catalog)
    entry = {
        "name": "rivs",
        "kind": "river",
        "files": str(tmp_path / "rivers.nc"),
        "grid": "grid.nc",
        "ref": str(tmp_path / "rivs.json"),
    }
    _refresh_forcing([entry], catalog)
    assert calls == [
        ("rivs", [str(tmp_path / "rivers.nc")], "grid.nc", {"ref": entry["ref"]})
    ]
    config = _suite(
        [{"title": "t", "health": {}}],
        refresh={"catalog": "c.yaml", "forcing": [{k: v for k, v in entry.items()}]},
    )
    assert config.refresh.forcing[0].kind == "river"
    with pytest.raises(ValueError, match="do not apply to kind: river"):
        _suite(
            [{"title": "t", "health": {}}],
            refresh={
                "catalog": "c.yaml",
                "forcing": [{**entry, "variables": ["x"]}],
            },
        )


def test_segment_joins_come_from_overrides_when_steps_share_a_blueprint(tmp_path):
    # C-Star workplans often point every step at one base blueprint and give each step
    # its own end date in apply-overrides: the joins are those end dates
    suite = _write_workplan(tmp_path, ["2013-01-01"] * 3)
    wp_path = tmp_path / "cstar" / "wp.yaml"
    wp = yaml.safe_load(wp_path.read_text())
    for step, end in zip(wp["steps"], ["2013-01-31", "2013-03-01", "2013-04-01"]):
        step["directives"] = {
            "apply-overrides": {"overrides": {"runtime_params": {"end_date": end}}}
        }
    # the second step also states its own start, a day after the first one's end:
    # one join, not two
    wp["steps"][1]["directives"]["apply-overrides"]["overrides"]["runtime_params"][
        "start_date"
    ] = "2013-02-01"
    wp_path.write_text(yaml.safe_dump(wp))
    assert cover.segment_starts(suite) == [
        pd.Timestamp("2013-02-01").to_pydatetime(),
        pd.Timestamp("2013-03-01").to_pydatetime(),
    ]
