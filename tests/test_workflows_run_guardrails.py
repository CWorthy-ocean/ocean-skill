"""``ocean-skill-run``'s guardrails: ``--smoke``, ``--self-check``, ``--only``, ...

Each flag exists so a bug in a suite shows up in minutes rather than after the whole
compute. These stay fast by stubbing the data (``stub_model``) or the comparison
(``_FakeComparison``), and test what the runner itself adds: the truncated read and its
separate output/cache directories, the self-check substitution and its verdict, the
title filter, the strict warning watch, the problems summary and the all-NaN guard.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import cache as _cache
from ocean_skill import comparison as _comparison
from ocean_skill import sources as _sources
from ocean_skill.workflows import pages as _pages
from ocean_skill.workflows import run as _run
from ocean_skill.workflows.run import PageProblem, main, run_suite
from tests.test_workflows_run import (
    _field_page,
    _model_only_suite,
    _section_page,
    _suite_with_pages,
    _write_suite,
    stub_model,  # noqa: F401  (used through usefixtures)
)

# -- --smoke ------------------------------------------------------------


def _model_dataset(n=6):
    return xr.Dataset(
        {"temp": (("time", "y"), np.arange(n * 2.0).reshape(n, 2))},
        coords={"time": pd.date_range("2020-01-01", periods=n, freq="D")},
    )


@pytest.fixture
def fake_sources(monkeypatch):
    """``osk.read`` of ``m`` (6 daily steps) and ``obs`` (6 steps) with no catalog."""
    monkeypatch.setattr(
        _sources,
        "resolve",
        lambda name: SimpleNamespace(name=name, metadata={}, path=None),
    )
    monkeypatch.setattr(
        _sources, "_read_uncached", lambda ref, meta, qc, kwargs: _model_dataset()
    )


def test_truncate_time_keeps_both_ends_of_the_named_sources_only(fake_sources):
    with _sources.truncate_time(["m"], 2):
        assert osk.read("m").sizes["time"] == 4
        assert list(osk.read("m").time.dt.day) == [1, 2, 5, 6]
        assert osk.read("obs").sizes["time"] == 6  # not named: untouched
    assert osk.read("m").sizes["time"] == 6


def test_other_sources_are_cut_to_end_where_the_anchor_does(monkeypatch):
    """A reference run continuing past the test's end must stay inside the window."""
    monkeypatch.setattr(
        _sources,
        "resolve",
        lambda name: SimpleNamespace(name=name, metadata={}, path=None),
    )
    datasets = {
        "test": _model_dataset(6),  # Jan 1-6
        "ref": xr.Dataset(  # Jan 1-10: runs past the test
            {"temp": (("time", "y"), np.zeros((10, 2)))},
            coords={"time": pd.date_range("2020-01-01", periods=10, freq="D")},
        ),
        "ic": xr.Dataset(  # a single-step restart
            {"temp": (("time", "y"), np.zeros((1, 2)))},
            coords={"time": pd.date_range("2020-01-01", periods=1, freq="D")},
        ),
        "late": xr.Dataset(  # nothing at or before the test's end: plain last-N
            {"temp": (("time", "y"), np.zeros((4, 2)))},
            coords={"time": pd.date_range("2021-01-01", periods=4, freq="D")},
        ),
    }
    monkeypatch.setattr(
        _sources, "_read_uncached", lambda ref, meta, qc, kwargs: datasets[ref.name]
    )
    with _sources.truncate_time(["test", "ref", "ic", "late"], 2, anchor="test"):
        assert list(osk.read("test").time.dt.day) == [1, 2, 5, 6]
        assert list(osk.read("ref").time.dt.day) == [1, 2, 5, 6]  # not 9, 10
        assert osk.read("ic").sizes["time"] == 1
        assert list(osk.read("late").time.dt.day) == [1, 2, 3, 4]
    assert osk.read("ref").sizes["time"] == 10


def test_truncate_time_is_reset_even_after_an_error(fake_sources):
    with pytest.raises(RuntimeError):
        with _sources.truncate_time(["m"], 2):
            raise RuntimeError("boom")
    assert _sources._TRUNCATE == {}
    assert osk.read("m").sizes["time"] == 6


def test_truncation_never_reaches_the_read_memo(fake_sources, monkeypatch, tmp_path):
    """The memo is keyed on the catalog file; it must keep the full record."""
    (tmp_path / "c.yaml").write_text("x")
    ref = SimpleNamespace(name="m", metadata={}, path=tmp_path / "c.yaml")
    monkeypatch.setattr(_sources, "resolve", lambda name: ref)
    with _sources.truncate_time(["m"], 2):
        assert osk.read("m").sizes["time"] == 4  # the first two and the last two
    assert osk.read("m").sizes["time"] == 6


def test_a_source_without_a_time_axis_is_returned_whole(monkeypatch):
    monkeypatch.setattr(
        _sources,
        "resolve",
        lambda name: SimpleNamespace(name=name, metadata={}, path=None),
    )
    static = xr.Dataset({"h": (("y",), [1.0, 2.0, 3.0])})
    monkeypatch.setattr(_sources, "_read_uncached", lambda *a, **k: static)
    with _sources.truncate_time(["m"], 2):
        assert osk.read("m").sizes["y"] == 3


def test_cache_scoped_moves_identity_keyed_kinds_but_not_weights(tmp_path):
    before = {k: _cache.path(k) for k in _cache.KINDS}
    with _cache.scoped("smoke-cache") as root:
        assert root == _cache.base_dir() / "smoke-cache"
        for kind in ("prepared", "aligned", "calculated"):
            assert _cache.path(kind) == root / kind
        assert _cache.path("weights") == before["weights"]
    assert {k: _cache.path(k) for k in _cache.KINDS} == before


def _smoke_suite(tmp_path, **extra):
    suite = _model_only_suite(
        tmp_path,
        refresh={
            "catalog": str(tmp_path / "catalogs" / "x.yaml"),
            "sources": [{"name": "m", "files": str(tmp_path / "none*.nc"), "ref": "r"}],
        },
        **extra,
    )
    suite["defaults"] = {"test": "m"}
    return suite


def test_smoke_reads_few_steps_into_its_own_report_and_cache(
    tmp_path, fake_sources, monkeypatch
):
    seen = {}

    def fake_prepare(source, variable, select, aggregate, **kwargs):
        seen["steps"] = osk.read("m").sizes["time"]
        seen["cache"] = _cache.path("prepared")
        da = xr.DataArray(
            np.full((2, 2), 5.0),
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [-100.0, -90.0]},
            attrs={"units": "degC"},
        )
        return da, None

    monkeypatch.setattr(_comparison, "prepare_source", fake_prepare)
    indexes = []
    real_index = osk.extrema._native_time_index

    def spy_index(source):
        indexes.append((bool(_sources._TRUNCATE), real_index(source)))
        return indexes[-1][1]

    monkeypatch.setattr("ocean_skill.extrema._native_time_index", spy_index)

    path = _write_suite(tmp_path, _smoke_suite(tmp_path))
    result = run_suite(path, refresh=False, smoke=2)

    assert result.exit_code == 0
    assert result.report_dir == (tmp_path / "out" / "quick_check-smoke")
    assert seen["steps"] == 4  # the first two and the last two
    # the time-index lookups made during expansion saw the same truncated record (the
    # throwaway pass that collects the pinned times runs before the cut, on all of it)
    assert any(cut for cut, _ in indexes)
    assert all(len(i) == 4 for cut, i in indexes if cut)
    assert all(len(i) == 6 for cut, i in indexes if not cut)
    smoke_dir = Path(json.loads(result.manifest.read_text())["modes"]["smoke_cache"])
    assert smoke_dir.parent == _cache.base_dir() / "smoke-cache"
    assert seen["cache"] == smoke_dir / "prepared"
    manifest = json.loads(result.manifest.read_text())
    assert manifest["modes"]["smoke"] == 2
    assert manifest["modes"]["smoke_cache_key"] == smoke_dir.name
    assert manifest["modes"]["smoke_cache_reused"] is False
    assert manifest["modes"]["smoke_sources"] == ["m"]
    # a smoke run is not "the latest report", and nothing it set outlives it
    assert not (tmp_path / "out" / "latest.txt").exists()
    assert _sources._TRUNCATE == {}
    assert _cache.path("prepared") == _cache.base_dir() / "cache" / "prepared"


def test_smoke_does_not_touch_the_real_report_dir(tmp_path, fake_sources, monkeypatch):
    monkeypatch.setattr(
        _comparison,
        "prepare_source",
        lambda *a, **k: stub_model_field(),
    )
    path = _write_suite(tmp_path, _smoke_suite(tmp_path))
    full = run_suite(path, refresh=False)
    smoke = run_suite(path, refresh=False, smoke=2)
    assert full.report_dir != smoke.report_dir
    assert (full.report_dir / "report.pdf").exists()
    assert (full.report_dir.parent / "latest.txt").read_text() == str(full.report_dir)


def stub_model_field():
    from tests.test_workflows_run import _stub_field

    return _stub_field()


def test_smoke_without_a_named_model_source_is_an_error(tmp_path):
    suite = _model_only_suite(tmp_path)
    suite["defaults"] = {}
    path = _write_suite(tmp_path, suite)
    with pytest.raises(ValueError, match="no model source"):
        run_suite(path, smoke=2, list_only=True)


def test_smoke_rejects_a_nonpositive_step_count(tmp_path, stub_model):  # noqa: F811
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    with pytest.raises(ValueError, match="at least 1"):
        run_suite(path, smoke=0)


@pytest.mark.usefixtures("stub_model")
def test_smoke_defaults_to_two_steps_on_the_command_line(tmp_path, monkeypatch):
    got = {}

    def fake_run(path, **kwargs):
        got.update(kwargs)
        return _run.SuiteResult()

    monkeypatch.setattr(_run, "run_suite", fake_run)
    main([str(tmp_path / "s.yaml"), "--smoke"])
    assert got["smoke"] == 2
    main([str(tmp_path / "s.yaml"), "--smoke", "5", "--only", "a*", "--strict"])
    assert got["smoke"] == 5 and got["only"] == ["a*"] and got["strict"] is True
    main([str(tmp_path / "s.yaml"), "--self-check"])
    assert got["smoke"] is None and got["self_check"] is True and got["only"] is None


def test_only_is_repeatable_on_the_command_line(tmp_path, monkeypatch):
    got = {}

    def fake_run(path, **kwargs):
        got.update(kwargs)
        return _run.SuiteResult()

    monkeypatch.setattr(_run, "run_suite", fake_run)
    main([str(tmp_path / "s.yaml"), "--only", "River inputs", "--only", "Nitrate*"])
    assert got["only"] == ["River inputs", "Nitrate*"]


# -- a fake comparison ------------------------------------------------------------


class _FakeComparison:
    """Just enough of a Comparison for ``pages.build`` and the runner."""

    family = "map"

    def __init__(self, *, test, reference, depth=None, offset=0.0, nan_lane=None):
        rng = np.random.default_rng(0)
        t = rng.normal(10.0, 2.0, (4, 5))
        r = t + offset
        if nan_lane == "reference":
            r = np.full_like(r, np.nan)
        if nan_lane == "test":
            t = np.full_like(t, np.nan)
        self.aligned = xr.Dataset(
            {
                "test": (("y", "x"), t),
                "reference": (("y", "x"), r),
            }
        )
        self.variable = "temperature"
        self.test_name = test
        self.reference_name = reference
        self.depth = depth
        ok = np.isfinite(t) & np.isfinite(r)
        diff = (t - r)[ok]
        self._record = {
            "variable": "temperature",
            "label": "temperature",
            "test": test,
            "reference": reference,
            "depth": depth,
            "n": int(ok.sum()),
            "bias": float(diff.mean()) if ok.any() else float("nan"),
            "rmse": float(np.sqrt((diff**2).mean())) if ok.any() else float("nan"),
            "corr": float(np.corrcoef(t[ok], r[ok])[0, 1]) if ok.sum() > 2 else np.nan,
            "mean_test": float(np.nanmean(t)) if np.isfinite(t).any() else np.nan,
            "std_test": float(np.nanstd(t)) if np.isfinite(t).any() else np.nan,
        }

    def metrics(self, **extra):
        return {**self._record, **extra}

    def _record_depth_time(self):
        return self.depth, None


class _FakeSet(list):
    def __init__(self, members=()):
        super().__init__(members)

    def plot(self, **kwargs):
        return plt.subplots()[0]


@pytest.fixture
def fake_compare(monkeypatch):
    """Route ``osk.compare`` to ``_FakeComparison``; ``calls`` records every call."""
    state = SimpleNamespace(calls=[], offset=0.0, nan=None)

    def compare(*, test, reference, **kwargs):
        state.calls.append({"test": test, "reference": reference, **kwargs})
        refs = [reference] if isinstance(reference, str) else list(reference)
        depths = kwargs.get("depths") or [None]
        members = []
        for ref in refs:
            for d in depths:
                nan_lane = state.nan.get(d) if isinstance(state.nan, dict) else None
                members.append(
                    _FakeComparison(
                        test=test,
                        reference=ref,
                        depth=d,
                        offset=state.offset,
                        nan_lane=nan_lane,
                    )
                )
        return _FakeSet(members)

    monkeypatch.setattr(osk, "compare", compare)
    monkeypatch.setattr(osk, "ComparisonSet", _FakeSet)
    return state


def _compare_page(title="Temp vs obs", **extra):
    return {
        "title": title,
        "compare": {
            "reference": ["obs_a", "obs_b"],
            "variables": [
                {"test": "temperature", "reference": "obs_temperature"},
                "salinity",
            ],
            "select": {"test": {"time": "2020-01"}, "reference": {"time": 1}},
            "aggregate": {"test": {"time": "mean"}, "reference": {"time": "mean"}},
            **extra,
        },
    }


# -- --self-check ------------------------------------------------------------


def test_self_check_page_collapses_every_pair_spec_to_the_test_side():
    page = _pages.ExpandedPage(
        title="T",
        kind="compare",
        kwargs={
            "test": "m",
            "reference": ["obs_a", "obs_b"],
            "variables": [
                {"test": "temperature", "reference": "obs_temperature"},
                "salinity",
            ],
            "select": {"test": {"time": "2020"}, "reference": {"time": 1}},
            "aggregate": [
                {"test": {"time": "mean"}, "reference": {"time": "max"}},
                {"lon": "mean"},
            ],
            "qc": {"test": None, "reference": "off"},
            "subtract_mean": "reference",
            "detide": {"reference": True},
            "depths": [0, 100],
        },
        plot={},
        cache=True,
        draw="plot",
    )
    out = _pages.self_check_page(page)
    assert out.title == "[self-check] T"
    k = out.kwargs
    assert k["reference"] == "m"
    assert k["variables"] == ["temperature", "salinity"]
    assert k["select"] == {"time": "2020"}
    assert k["aggregate"] == [{"time": "mean"}, {"lon": "mean"}]
    assert k["qc"] is None
    assert k["subtract_mean"] is False
    assert k["detide"] is False
    assert k["depths"] == [0, 100]  # shared by both lanes already
    assert out.cache is False
    assert page.kwargs["reference"] == ["obs_a", "obs_b"]  # the original is untouched


def test_self_check_page_leaves_other_pages_and_several_tests_alone():
    field = _pages.ExpandedPage(title="F", kind="field", kwargs={}, plot={}, cache=True)
    assert _pages.self_check_page(field) is None
    many = _pages.ExpandedPage(
        title="C",
        kind="compare",
        kwargs={"test": ["a", "b"], "reference": ["r"]},
        plot={},
        cache=True,
    )
    assert _pages.self_check_page(many) is None


@pytest.mark.usefixtures("stub_model")
def test_self_check_passes_when_both_lanes_are_treated_alike(tmp_path, fake_compare):
    path = _write_suite(
        tmp_path, _suite_with_pages(tmp_path, [_field_page(), _compare_page()])
    )
    result = run_suite(path, self_check=True)

    assert result.report_dir.name == "quick_check-selfcheck"
    assert result.problems == []
    assert result.exit_code == 0
    titles = [p.title for p in result.pages]
    assert titles == ["Physics latest", "[self-check] Temp vs obs"]
    call = fake_compare.calls[0]
    assert call["reference"] == "stub" and call["test"] == "stub"
    assert call["variables"] == ["temperature", "salinity"]
    assert call["select"] == {"time": "2020-01"}


@pytest.mark.usefixtures("stub_model")
def test_self_check_flags_a_lane_asymmetry(tmp_path, fake_compare, capsys):
    fake_compare.offset = 5.0  # the reference lane is cropped/shifted differently
    path = _write_suite(tmp_path, _suite_with_pages(tmp_path, [_compare_page()]))
    result = run_suite(path, self_check=True)

    [problem] = result.problems
    assert problem.kind == "self-check"
    assert problem.title == "[self-check] Temp vs obs"
    assert "bias" in problem.reason and "-5" in problem.reason
    assert problem.seconds is not None and problem.peak_mb > 0
    # the page itself still drew, but the run is not clean
    assert [p.status for p in result.pages] == ["ok"]
    assert result.exit_code == 4  # a lane bug, not a benign skip
    manifest = json.loads(result.manifest.read_text())
    assert manifest["problems"][0]["kind"] == "self-check"
    assert manifest["modes"]["self_check"] is True


@pytest.mark.usefixtures("stub_model")
def test_self_check_leaves_field_pages_running_normally(tmp_path, fake_compare):
    path = _write_suite(tmp_path, _suite_with_pages(tmp_path, [_field_page()]))
    result = run_suite(path, self_check=True)
    assert [p.title for p in result.pages] == ["Physics latest"]
    assert fake_compare.calls == []


def test_self_check_tolerance_scales_with_the_data():
    big = {"mean_test": 1e6, "std_test": 3.0, "bias": 1e-3, "rmse": 1e-3, "corr": 1.0}
    page = _pages.ExpandedPage(
        title="x", kind="compare", kwargs={}, plot={}, cache=False
    )
    page.metrics_records = [big]
    assert _pages.self_check_problem(page) is None  # 1e-3 << 1e-6 * 1e6
    page.metrics_records = [{**big, "corr": 0.999}]
    assert "corr 0.999" in _pages.self_check_problem(page)
    nan = float("nan")
    page.metrics_records = [{**big, "bias": nan, "rmse": nan}]
    assert "bias/rmse NaN although the lanes should be identical" in (
        _pages.self_check_problem(page)
    )
    # corr alone may be NaN: a constant field has no correlation
    page.metrics_records = [{**big, "corr": nan}]
    assert _pages.self_check_problem(page) is None
    page.metrics_records = [{"corr": 1.0}]  # absent metrics are not NaN metrics
    assert _pages.self_check_problem(page) is None


# -- --only ------------------------------------------------------------


def _three_pages(tmp_path):
    return _suite_with_pages(
        tmp_path,
        [
            _section_page("Physics", "notes"),
            _field_page("Temperature map"),
            _field_page("Salinity map"),
            _section_page("Biogeochemistry", "more"),
            _field_page("Nitrate map"),
        ],
    )


def test_only_draws_matching_pages_case_insensitively(tmp_path, stub_model):  # noqa: F811
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path, only="*SALINITY*")

    assert result.report_dir.name == "quick_check-only-salinity-" + _sha6("*salinity*")
    drawn = [(p.kind, p.title, p.status) for p in result.pages]
    # the divider over the match is kept (and written), the other one is dropped
    assert drawn == [
        ("section", "Physics", "ok"),
        ("field", "Salinity map", "ok"),
    ]
    assert len(result.figures) == 1
    assert result.exit_code == 0


def test_only_matching_nothing_names_the_pages(tmp_path, stub_model):  # noqa: F811
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    with pytest.raises(ValueError, match="Nitrate map"):
        run_suite(path, only="nope*")


def test_only_applies_to_list(tmp_path, stub_model, capsys):  # noqa: F811
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path, only="nitrate*", list_only=True)
    assert [p.title for p in result.pages] == ["Biogeochemistry", "Nitrate map"]
    out = capsys.readouterr().out
    assert "Nitrate map" in out and "Temperature map" not in out


def _titles(result):
    return [p.title for p in result.pages]


@pytest.mark.usefixtures("stub_model")
def test_only_a_section_glob_keeps_the_whole_section(tmp_path):
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path, only="physics")

    assert _titles(result) == ["Physics", "Temperature map", "Salinity map"]
    assert result.report_dir.name == "quick_check-only-physics"
    assert len(result.figures) == 2


@pytest.mark.usefixtures("stub_model")
def test_only_takes_several_globs_across_sections(tmp_path):
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path, only=["Biogeo*", "salinity map"])

    assert _titles(result) == [
        "Physics",
        "Salinity map",
        "Biogeochemistry",
        "Nitrate map",
    ]
    assert result.report_dir.name == (
        "quick_check-only-biogeo-salinity-map-" + _sha6("biogeo*\nsalinity map")
    )
    # --list shows the same selection
    listed = run_suite(path, only=["Biogeo*", "salinity map"], list_only=True)
    assert _titles(listed) == _titles(result)


@pytest.mark.usefixtures("stub_model")
def test_only_names_the_glob_that_matches_nothing_and_lists_the_sections(tmp_path):
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    with pytest.raises(ValueError, match="'nope\\*'") as err:
        run_suite(path, only=["physics", "nope*"])
    message = str(err.value)
    assert "'Physics'" in message and "'Biogeochemistry'" in message
    assert "Nitrate map" in message
    assert message.index("'Physics'") < message.index("Nitrate map")


def _sha6(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:6]


def test_only_tag_slugs_the_globs():
    # plain titles keep a readable folder name, and case does not matter
    assert _run._only_tag(["River inputs"]) == "-only-river-inputs"
    assert _run._only_tag(["river INPUTS"]) == "-only-river-inputs"
    assert _run._only_tag(["Boundary*", " River inputs"]) == (
        "-only-boundary-river-inputs-" + _sha6("boundary*\n river inputs")
    )
    # a glob with no letters or digits would leave an empty slug: hash instead
    assert _run._only_tag(["*"]) == "-only-" + _sha6("*")


def test_only_tag_never_gives_two_selections_one_folder():
    """A second selection would otherwise clear the first one's report."""
    tags = [
        _run._only_tag(g)
        for g in (
            ["River inputs"],
            ["river*inputs"],
            ["River-inputs"],
            ["river?inputs"],
            ["River", "inputs"],
        )
    ]
    assert len(set(tags)) == len(tags)


def test_only_tag_cuts_a_long_slug_and_keeps_it_unique():
    long_a = ["a very long section title that goes on and on", "x"]
    long_b = ["a very long section title that goes on and on", "y"]
    tag_a, tag_b = _run._only_tag(long_a), _run._only_tag(long_b)
    digest = _sha6("\n".join(long_a))
    assert tag_a == "-only-a-very-long-section-title-that-go-" + digest
    assert tag_a != tag_b
    assert len(tag_a) - len("-only-") <= 40


@pytest.mark.usefixtures("stub_model")
def test_only_parts_do_not_overwrite_each_other_or_the_full_report(tmp_path):
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    full = run_suite(path)
    a = run_suite(path, only="Physics")
    b = run_suite(path, only="Biogeochemistry")
    assert len({full.report_dir, a.report_dir, b.report_dir}) == 3
    assert (full.report_dir / "report.pdf").exists()
    assert len(full.figures) == 3


# -- --strict ------------------------------------------------------------


def _warning_build(monkeypatch, text, titles=None):
    real = _pages.build

    def build(page, *, pooled_records=None):
        if titles is None or page.title in titles:
            warnings.warn(text, stacklevel=2)
        return real(page, pooled_records=pooled_records)

    monkeypatch.setattr(_pages, "build", build)


CRAMPED = (
    "the maps are only 1.20in wide: a 4.00in canvas leaves too little width once the "
    "titles, colorbars and coordinate labels have taken theirs, and the type is "
    "already at its 6pt floor."
)


@pytest.mark.usefixtures("stub_model")
def test_strict_counts_the_cramped_maps_warning_as_a_problem(tmp_path, monkeypatch):
    _warning_build(monkeypatch, CRAMPED, titles={"Salinity map"})
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path, strict=True)

    [problem] = result.problems
    assert (problem.kind, problem.title) == ("strict", "Salinity map")
    assert "the maps are only 1.20in wide" in problem.reason
    # drawn all the same, but the run is no longer clean
    assert {p.status for p in result.pages} == {"ok"}
    assert result.exit_code == 3


@pytest.mark.usefixtures("stub_model")
def test_strict_reports_a_repeated_warning_on_every_page(tmp_path, monkeypatch):
    _warning_build(monkeypatch, CRAMPED)
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path, strict=True)
    assert [p.title for p in result.problems] == [
        "Temperature map",
        "Salinity map",
        "Nitrate map",
    ]


@pytest.mark.usefixtures("stub_model")
def test_without_strict_the_same_warning_is_not_a_problem(tmp_path, monkeypatch):
    _warning_build(monkeypatch, CRAMPED)
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path)
    assert result.problems == []
    assert result.exit_code == 0


@pytest.mark.usefixtures("stub_model")
def test_strict_ignores_ordinary_data_warnings(tmp_path, monkeypatch):
    _warning_build(monkeypatch, "only 3 valid cells: metrics are weakly constrained.")
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path, strict=True)
    assert result.problems == []


def test_strict_warning_text_matches_the_real_renderer_message():
    """The pattern must keep matching what ``_warn_if_cramped`` actually says."""
    import re

    from ocean_skill.plot import matplotlib_renderer as mr

    fig = plt.figure(figsize=(10.0, 4.0))
    ax = fig.add_axes([0.1, 0.1, 0.1, 0.8])  # one inch wide on a three-inch cell
    ax.projection = object()  # what marks a map panel
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mr._warn_if_cramped(fig, ncols=3, panels=[ax])
    plt.close(fig)
    texts = [str(w.message) for w in caught]
    assert texts, "the renderer no longer warns for a cramped figure"
    assert any(re.match(p, texts[0], re.DOTALL) for p in _run._STRICT_PATTERNS)


# -- the problems summary and exit code --------------------------------------------


@pytest.mark.usefixtures("stub_model")
def test_a_failed_page_is_a_problem_with_its_time_and_memory(tmp_path, monkeypatch):
    real = _pages.build

    def build(page, *, pooled_records=None):
        if page.title == "Salinity map":
            raise RuntimeError("regrid failed")
        return real(page, pooled_records=pooled_records)

    monkeypatch.setattr(_pages, "build", build)
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path)

    [problem] = result.problems
    assert problem.kind == "failed"
    assert "RuntimeError: regrid failed" in problem.reason
    assert problem.seconds >= 0 and problem.peak_mb > 0
    assert result.exit_code == 4
    manifest = json.loads(result.manifest.read_text())
    by_title = {p["title"]: p for p in manifest["pages"]}
    assert by_title["Temperature map"]["seconds"] >= 0
    assert by_title["Temperature map"]["peak_mb"] > 0
    assert manifest["problems"][0]["title"] == "Salinity map"


@pytest.mark.usefixtures("stub_model")
def test_an_expand_time_problem_page_is_listed_as_skipped(tmp_path, monkeypatch):
    real = _pages.expand

    def expand(suite, **kwargs):
        pages = real(suite, **kwargs)
        pages[1].problem = "source 'x' could not be read at expand time"
        return pages

    monkeypatch.setattr(_pages, "expand", expand)
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path)
    [problem] = result.problems
    assert problem.kind == "skipped" and "could not be read" in problem.reason


def test_a_clean_run_has_no_problems_and_exit_code_zero(tmp_path, stub_model):  # noqa: F811
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    result = run_suite(path, strict=True)
    assert result.problems == []
    assert result.exit_code == 0
    # the report directory and latest pointer are the ordinary ones
    assert result.report_dir.name == "quick_check"


def test_exit_code_is_nonzero_for_problems_even_when_every_page_drew():
    pages = [SimpleNamespace(kind="field", status="ok")]
    result = _run.SuiteResult(pages=pages)
    assert result.exit_code == 0
    result.problems.append(PageProblem("t", "strict", "r", 1.0, 10.0))
    assert result.exit_code == 3
    result.pages = [SimpleNamespace(kind="field", status="skipped")]
    assert result.exit_code == 1  # nothing drew: unchanged


@pytest.mark.usefixtures("stub_model")
def test_main_prints_and_logs_the_problem_list(tmp_path, monkeypatch, capsys):
    _warning_build(monkeypatch, CRAMPED, titles={"Salinity map"})
    path = _write_suite(tmp_path, _three_pages(tmp_path))
    code = main([str(path), "--strict"])
    out = capsys.readouterr().out

    assert code == 3
    assert "3 page(s) drawn, 0 skipped" in out
    assert "1 problem(s):" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("  [strict]"))
    assert "Salinity map" in line and "the maps are only" in line
    assert " MB" in line and "s," in line
    log = (tmp_path / "out" / "quick_check" / "run.log").read_text()
    assert "[strict] Salinity map" in log


# -- the all-NaN guard ------------------------------------------------------------


def _depth_page(depths=(0, 200)):
    return {
        "title": "Temp at depth",
        "compare": {
            "reference": ["obs"],
            "variables": ["temperature"],
            "depths": list(depths),
        },
    }


@pytest.mark.usefixtures("stub_model")
def test_an_all_nan_comparison_is_dropped_and_the_page_still_draws(
    tmp_path, fake_compare
):
    fake_compare.nan = {200: "reference"}  # a 200 m comparison on a shallow shelf
    path = _write_suite(tmp_path, _suite_with_pages(tmp_path, [_depth_page()]))
    with pytest.warns(UserWarning, match="dropped 1 of 2"):
        result = run_suite(path)

    [page] = result.pages
    assert page.status == "ok"
    assert [r["depth"] for r in page.metrics_records] == [0]
    [problem] = result.problems
    assert problem.kind == "skipped"
    assert "all-NaN: reference has no data at depth 200" in problem.reason
    assert result.exit_code == 3


@pytest.mark.usefixtures("stub_model")
def test_a_page_whose_comparisons_are_all_nan_is_skipped_with_the_reason(
    tmp_path, fake_compare
):
    fake_compare.nan = {200: "test"}
    path = _write_suite(tmp_path, _suite_with_pages(tmp_path, [_depth_page([200])]))
    result = run_suite(path)

    [page] = result.pages
    assert page.status == "skipped"
    assert page.reason.startswith("PageSkipped: all-NaN: test has no data at depth 200")
    assert result.figures == []
    [problem] = result.problems
    assert problem.kind == "skipped"
    assert result.exit_code == 1


@pytest.mark.usefixtures("stub_model")
def test_all_nan_guard_leaves_healthy_pages_alone(tmp_path, fake_compare):
    path = _write_suite(tmp_path, _suite_with_pages(tmp_path, [_depth_page()]))
    result = run_suite(path)
    assert result.problems == []
    assert len(result.pages[0].metrics_records) == 2


def test_problem_lines_truncate_long_reasons():
    lines = _run._problem_lines([PageProblem("t", "failed", "x" * 500, 1.5, 2048.0)])
    assert lines[0] == "1 problem(s):"
    assert len(lines[1]) < 280 and "..." in lines[1]
    assert "1.5s" in lines[1] and "2,048 MB" in lines[1]
    assert _run._problem_lines([]) == []


# -- review fixes ---------------------------------------------------------------


@pytest.mark.usefixtures("stub_model")
def test_strict_watches_the_save_not_just_the_draw(tmp_path, monkeypatch):
    """Layout warnings fire when the figure is rendered, i.e. in ``report.emit``."""

    def build(page, *, pooled_records=None):
        fig, ax = plt.subplots(figsize=(0.4, 0.4), layout="tight")  # no room
        ax.set_title("a title far too long for a figure this small " * 3)
        return [("", fig)]

    monkeypatch.setattr(_pages, "build", build)
    path = _write_suite(tmp_path, _suite_with_pages(tmp_path, [_field_page("One")]))
    result = run_suite(path, strict=True)

    [problem] = result.problems
    assert (problem.kind, problem.title) == ("strict", "One")
    assert "ight layout not applied" in problem.reason
    assert result.pages[0].status == "ok"


@pytest.mark.usefixtures("stub_model")
def test_contours_that_cannot_be_paired_after_an_all_nan_drop_raise(
    tmp_path, fake_compare
):
    fake_compare.nan = {200: "reference"}
    page = _depth_page()
    page["contours"] = {"depths": [0]}  # a one-member overlay for a two-member fill
    path = _write_suite(tmp_path, _suite_with_pages(tmp_path, [page]))
    result = run_suite(path)
    [problem] = result.problems
    assert problem.kind == "failed"
    assert "can no longer be paired" in problem.reason


@pytest.fixture
def smoke_code(monkeypatch):
    """Set the code identity, so the smoke-cache hash ignores the real git tree."""
    state = {"identity": "v1|abc"}
    monkeypatch.setattr(_run, "_code_identity", lambda: state["identity"])
    monkeypatch.setattr(
        _comparison, "prepare_source", lambda *a, **k: stub_model_field()
    )
    return state


def _smoke(tmp_path, n=2):
    result = run_suite(
        _write_suite(tmp_path, _smoke_suite(tmp_path)), refresh=False, smoke=n
    )
    modes = json.loads(result.manifest.read_text())["modes"]
    return Path(modes["smoke_cache"]), modes


@pytest.mark.usefixtures("fake_sources")
def test_same_code_and_truncation_reuse_the_smoke_cache(tmp_path, smoke_code, capsys):
    first, modes = _smoke(tmp_path)
    assert modes["smoke_cache_reused"] is False
    assert f"smoke cache: {first} (new)" in capsys.readouterr().out
    kept = first / "prepared" / "earlier_run.zarr"
    kept.mkdir(parents=True)  # not emptied at the start any more
    second, modes = _smoke(tmp_path)
    assert second == first and modes["smoke_cache_reused"] is True
    assert kept.exists()
    assert f"smoke cache: {first} (reused)" in capsys.readouterr().out


@pytest.mark.usefixtures("fake_sources")
def test_a_different_truncation_gets_a_new_smoke_cache(tmp_path, smoke_code):
    two, _ = _smoke(tmp_path, 2)
    three, modes = _smoke(tmp_path, 3)
    assert three != two and modes["smoke_cache_reused"] is False


@pytest.mark.usefixtures("fake_sources")
def test_different_kept_stamps_get_a_new_smoke_cache(tmp_path, smoke_code, monkeypatch):
    before, _ = _smoke(tmp_path)  # same N, but the model has grown by two steps
    monkeypatch.setattr(_sources, "_read_uncached", lambda *a: _model_dataset(8))
    after, modes = _smoke(tmp_path)
    assert after != before and modes["smoke_cache_reused"] is False


@pytest.mark.usefixtures("fake_sources")
def test_a_code_change_starts_a_cold_smoke_cache(tmp_path, smoke_code):
    clean, _ = _smoke(tmp_path)
    smoke_code["identity"] = "v1|abc|dirty-1"
    dirty, modes = _smoke(tmp_path)
    assert dirty != clean and modes["smoke_cache_reused"] is False
    smoke_code["identity"] = "v1|abc|dirty-2"  # edited again
    assert _smoke(tmp_path)[0] not in (clean, dirty)
    smoke_code["identity"] = "v1|abc"  # back to the committed code
    assert _smoke(tmp_path)[0] == clean


@pytest.mark.usefixtures("fake_sources")
def test_without_a_code_identity_every_smoke_run_is_cold(tmp_path, smoke_code, capsys):
    smoke_code["identity"] = None
    first, _ = _smoke(tmp_path)
    second, modes = _smoke(tmp_path)
    assert first != second and modes["smoke_cache_reused"] is False
    assert "every run starts cold" in capsys.readouterr().out


def _folders(root, ages_h):
    out = []
    for i, age in enumerate(ages_h):
        d = root / f"{i:012x}"
        (d / "prepared").mkdir(parents=True)
        t = time.time() - age * 3600
        os.utime(d / "prepared", (t, t))
        os.utime(d, (t, t))
        out.append(d)
    return out


def test_pruning_keeps_the_newest_and_never_the_current(tmp_path):
    root = tmp_path / "smoke-cache"
    current, *others = _folders(root, [90, 5, 4, 3, 2])  # current is the oldest
    removed = _run._prune_smoke_cache(root, current)
    assert current.exists()
    # the current one plus the newest KEEP - 1 others stay
    assert sorted(d for d in root.iterdir() if d != current) == sorted(
        others[-(_run.SMOKE_CACHE_KEEP - 1) :]
    )
    assert sorted(removed) == sorted(others[: -(_run.SMOKE_CACHE_KEEP - 1)])


def test_pruning_spares_anything_used_within_the_grace_period(tmp_path):
    root = tmp_path / "smoke-cache"
    current, *_ = _folders(root, [1, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert _run._prune_smoke_cache(root, current) == []  # all recent: none removed
    assert len(list(root.iterdir())) == 6
    # a concurrent run still writing into an old folder keeps it alive
    (old,) = _folders(tmp_path / "other", [10])
    (old / "calculated").mkdir()
    assert _run._prune_smoke_cache(old.parent, old) == []


def test_pruning_clears_the_old_one_folder_layout_after_the_grace_period(tmp_path):
    root = tmp_path / "smoke-cache"
    (current,) = _folders(root, [0])
    for kind in ("prepared", "aligned"):
        (root / kind / "x.zarr").mkdir(parents=True)
        os.utime(root / kind / "x.zarr", (1.0, 1.0))
        os.utime(root / kind, (1.0, 1.0))
    _run._prune_smoke_cache(root, current)
    assert [d.name for d in root.iterdir()] == [current.name]


@pytest.mark.usefixtures("fake_sources")
def test_a_smoke_run_prunes_old_caches_but_not_the_current_one(tmp_path, smoke_code):
    root = _cache.base_dir() / "smoke-cache"
    old = _folders(root, [50, 60, 70, 80])
    current, _ = _smoke(tmp_path)
    assert current.exists()
    assert sum(d.exists() for d in old) == _run.SMOKE_CACHE_KEEP - 1


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_code_identity_tracks_commits_and_uncommitted_edits(tmp_path, monkeypatch):
    import ocean_skill

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("x = 1\n")

    def git(*args):
        subprocess.run(
            ["git", "-C", str(pkg), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
        )

    monkeypatch.setattr(ocean_skill, "__file__", str(pkg / "__init__.py"))
    assert _run._code_identity() is None  # not a git checkout: always cold
    git("init")
    git("add", ".")
    git("commit", "-m", "one")
    clean = _run._code_identity()
    assert clean and "dirty" not in clean and _run._code_identity() == clean
    (pkg / "__init__.py").write_text("x = 2\n")
    edited = _run._code_identity()
    assert edited != clean and "dirty-" in edited and _run._code_identity() == edited
    (pkg / "__init__.py").write_text("x = 3\n")
    assert _run._code_identity() != edited  # a different edit
    (pkg / "__init__.py").write_text("x = 1\n")
    assert _run._code_identity() == clean  # edit undone
    (pkg / "new.py").write_text("y = 1\n")
    untracked = _run._code_identity()
    assert untracked != clean
    (pkg / "new.py").write_text("y = 2\n")
    assert _run._code_identity() != untracked
    git("add", ".")
    git("commit", "-m", "two")
    assert _run._code_identity() not in (clean, untracked)


@pytest.mark.usefixtures("fake_sources")
def test_list_prints_the_real_cache_and_a_separate_smoke_cache_line(tmp_path, capsys):
    path = _write_suite(tmp_path, _smoke_suite(tmp_path))
    run_suite(path, refresh=False, list_only=True)
    out = capsys.readouterr().out
    assert f"cache: {_cache.base_dir()}\n" in out and "smoke cache" not in out
    run_suite(path, refresh=False, list_only=True, smoke=2)
    out = capsys.readouterr().out
    assert f"cache: {_cache.base_dir()}\n" in out
    assert f"smoke cache: {_cache.base_dir() / 'smoke-cache'}/" in out
    assert "(new)" in out
    # listing creates nothing
    assert not (_cache.base_dir() / "smoke-cache").exists()


def test_peak_memory_is_unknown_without_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)  # makes the import fail
    with _run._PeakMemory() as mem:
        pass
    assert mem.peak_mb is None
    assert _run._problem_lines([PageProblem("t", "failed", "r", 1.0, None)]) == [
        "1 problem(s):",
        "  [failed] t: r (1.0s)",
    ]


def test_an_anchor_that_cannot_be_compared_warns_and_keeps_both_ends(
    monkeypatch,
):
    monkeypatch.setattr(
        _sources,
        "resolve",
        lambda name: SimpleNamespace(name=name, metadata={}, path=None),
    )
    datasets = {
        "test": _model_dataset(6),
        "noleap": xr.Dataset(  # cftime stamps do not compare with datetime64
            {"temp": (("time", "y"), np.zeros((5, 2)))},
            coords={
                "time": xr.date_range(
                    "2020-01-01", periods=5, calendar="noleap", use_cftime=True
                )
            },
        ),
    }
    monkeypatch.setattr(
        _sources, "_read_uncached", lambda ref, meta, qc, kwargs: datasets[ref.name]
    )
    with _sources.truncate_time(["test", "noleap"], 2, anchor="test"):
        with pytest.warns(UserWarning, match="could not bound 'noleap'"):
            got = osk.read("noleap")
    assert got.sizes["time"] == 4


def test_an_all_nan_check_that_cannot_run_says_so():
    class Broken:
        variable, test_name, reference_name = "temperature", "a", "b"

        @property
        def aligned(self):
            raise RuntimeError("cannot align")

    with pytest.warns(UserWarning, match="could not check 'temperature'"):
        assert _pages._all_nan_reason(Broken()) is None


def test_exit_code_4_is_for_failed_and_self_check_not_for_skips():
    ok = SimpleNamespace(kind="field", status="ok")
    result = _run.SuiteResult(pages=[ok])
    for kind, code in (("skipped", 3), ("strict", 3), ("failed", 4), ("self-check", 4)):
        result.problems = [PageProblem("t", kind, "r")]
        assert result.exit_code == code
    result.pages = [SimpleNamespace(kind="field", status="skipped")]
    assert result.exit_code == 1  # nothing drew: still 1


# -- --smoke keeps the steps the pages pin ------------------------------


def _ts(*days):
    return tuple(pd.Timestamp(d) for d in days)


def _win(lo, hi):
    """Return a window as the pins spell it: a date ends at the end of its period."""
    return pd.Timestamp(lo), pd.Period(hi).end_time


def _pins(*values):
    return _sources.TimePins.from_values(values)


def test_pins_an_instant_and_a_list_of_instants():
    pins = _pins("2010-01-31", ["2010-04-30T12:00", "2010-07-31"])
    assert pins.instants == _ts("2010-01-31", "2010-04-30T12:00", "2010-07-31")
    assert pins.windows == ()


def test_pins_a_date_object_from_unquoted_yaml():
    import datetime

    assert _pins(datetime.date(2010, 1, 31)).instants == _ts("2010-01-31")


def test_pins_a_period_string_as_the_window_it_spans():
    (window,) = _pins("2010-01").windows
    assert window[0] == pd.Timestamp("2010-01-01")
    assert window[1] > pd.Timestamp("2010-01-31")


def test_pins_a_short_window_and_ignores_a_long_one():
    short = {"min": "2010-01-01", "max": "2010-02-01T12:00"}
    long = {"min": "2010-01-01", "max": "2013-12-31"}
    pins = _pins(short, long)
    assert pins.windows == (_win("2010-01-01", "2010-02-01T12:00"),)
    assert pins.instants == ()


def test_a_window_without_both_ends_names_nothing():
    assert not _pins({"min": "2010-01-01"}, {"max": "2010-01-05"})


def test_pins_ignore_what_is_not_a_date():
    nothing = [
        "latest",
        "mean",
        "2010",  # a whole year is a long window
        {"resample": "1MS", "reduce": "mean"},
        {"season": "MAM"},
        {"months": [3, 4]},
        [{"groupby": "month"}, {"reduce": "std"}],
        5,
        None,
        True,
    ]
    assert not _pins(*nothing)


def test_pins_from_a_season_mask_with_a_short_window():
    # a mask carrying a short window still names the window
    pins = _pins({"season": "JJA", "min": "2010-06-01", "max": "2010-07-15"})
    assert pins.windows == (_win("2010-06-01", "2010-07-15"),)


def _expanded(**kwargs):
    return _pages.ExpandedPage(
        title="p", kind=kwargs.pop("kind", "compare"), kwargs=kwargs,
        plot={}, cache=False,
    )  # fmt: skip


def test_time_pins_reads_select_times_and_placeholder_lists():
    season_days = ["2010-01-31", "2010-04-30"]  # what "{season_days}" expands to
    pages = [
        _expanded(select={"test": {"time": "2010-01-31"}, "reference": {}}),
        _expanded(select={"depth": 0}, times=season_days),
        _expanded(times=["2010-01", "2010-07"]),
        _expanded(select={"time": {"min": "2010-01-01", "max": "2010-02-01T12:00"}}),
        _expanded(select={"time": {"min": "2010-01-01", "max": "2013-12-31"}}),
        _expanded(select={"time": "2010-09-15"}, kind="section"),  # a divider: skipped
    ]
    pins = _pages.time_pins(pages)
    assert pins.instants == _ts("2010-01-31", "2010-04-30")
    assert len(pins.windows) == 3  # January, July, and the first month
    assert _win("2010-01-01", "2010-02-01T12:00") in pins.windows
    assert pins.windows[-1][0] == pd.Timestamp("2010-07-01")


def test_time_pins_ignores_resampled_bins_and_aggregates():
    pages = [
        _expanded(
            select={"time": {"min": "2010-01-01", "max": "2013-12-31"}},
            times={"resample": "1MS", "reduce": "pool"},
            aggregate={"time": {"resample": "1MS", "reduce": "mean"}},
        ),
        _expanded(aggregate={"time": "mean"}),
    ]
    assert not _pages.time_pins(pages)


def test_time_pins_reads_steps_and_contours():
    page = _expanded(select={"time": "2010-01-31"})
    window = {"min": "2010-03-01", "max": "2010-03-05"}
    page.steps = [{"name": "series", "kwargs": {"time": window}}]
    page.contours = {"select": {"time": "2010-02-15"}}
    pins = _pages.time_pins([page])
    assert pins.instants == _ts("2010-01-31", "2010-02-15")
    assert pins.windows == (_win("2010-03-01", "2010-03-05"),)


def _long_model(n=200, start="2020-01-01"):
    return xr.Dataset(
        {"temp": (("time", "y"), np.arange(n * 2.0).reshape(n, 2))},
        coords={"time": pd.date_range(start, periods=n, freq="D")},
    )


@pytest.fixture
def long_sources(monkeypatch):
    monkeypatch.setattr(
        _sources,
        "resolve",
        lambda name: SimpleNamespace(name=name, metadata={}, path=None),
    )
    datasets = {"m": _long_model(), "ref": _long_model(300)}
    monkeypatch.setattr(
        _sources, "_read_uncached", lambda ref, meta, qc, kwargs: datasets[ref.name]
    )


def _days(name, steps=1, **kwargs):
    with _sources.truncate_time([name], steps, **kwargs):
        t = osk.read(name).time
    return list((t - np.datetime64("2020-01-01")).dt.days.values)


def test_truncation_keeps_the_nearest_step_to_a_pinned_instant(long_sources):
    pins = _pins("2020-03-10T00:00", "2020-05-02")
    assert _days("m", pins=pins) == [0, 69, 122, 199]


def test_a_pinned_instant_snaps_to_the_nearest_step_within_tolerance(long_sources):
    # 2020-03-10T18:00 is nearer the 11th than the 10th
    assert _days("m", pins=_pins("2020-03-10T18:00")) == [0, 70, 199]
    # exactly 2 days from a step still counts; further than that adds nothing
    assert _days("m", pins=_pins("2021-01-01")) == [0, 199]  # after the run ends
    assert _days("m", pins=_pins("2019-12-30")) == [0, 199]  # 2 days before: step 0


def test_an_instant_far_from_every_step_keeps_nothing_extra(monkeypatch):
    monkeypatch.setattr(
        _sources,
        "resolve",
        lambda name: SimpleNamespace(name=name, metadata={}, path=None),
    )
    sparse = _model_dataset(3).assign_coords(
        time=pd.to_datetime(["2020-01-01", "2020-03-01", "2020-06-01"])
    )
    monkeypatch.setattr(_sources, "_read_uncached", lambda *a, **k: sparse)
    # 2020-02-01 is 31 and 29 days from the nearest steps; 2020-02-29 is 1 day from one
    with _sources.truncate_time(["m"], 1, pins=_pins("2020-02-01")):
        assert osk.read("m").sizes["time"] == 2
    with _sources.truncate_time(["m"], 1, pins=_pins("2020-02-29")):
        assert osk.read("m").sizes["time"] == 3


def test_truncation_keeps_the_ends_of_a_pinned_window(long_sources):
    pins = _pins({"min": "2020-02-01", "max": "2020-02-10"})
    assert _days("m", 2, pins=pins) == [0, 1, 31, 32, 39, 40, 198, 199]
    assert _days("m", 1, pins=pins) == [0, 31, 40, 199]


def test_a_window_of_fewer_than_n_steps_keeps_them_all(long_sources):
    pins = _pins({"min": "2020-02-01", "max": "2020-02-02"})
    assert _days("m", 3, pins=pins) == [0, 1, 2, 31, 32, 197, 198, 199]


def test_a_long_window_adds_nothing_to_the_ends(long_sources):
    pins = _pins({"min": "2020-01-01", "max": "2020-12-31"})
    assert _days("m", 1, pins=pins) == [0, 199]


def test_pins_outside_the_anchors_end_are_not_kept_for_other_sources(long_sources):
    """The test source ends day 199; a reference runs on to day 299."""
    pins = _pins("2020-04-01", "2020-09-01", {"min": "2020-08-01", "max": "2020-08-10"})
    # ref is bounded by the anchor's end (day 199): Sep 1 (244) and Aug 1-10 are cut
    assert _days("ref", 1, anchor="m", pins=pins) == [0, 91, 199]
    # without the anchor, every pin is kept
    assert _days("ref", 1, pins=pins) == [0, 91, 213, 222, 244, 299]


def test_no_pins_is_the_plain_first_and_last(long_sources):
    assert _days("m", 2) == [0, 1, 198, 199]
    assert _days("m", 2, pins=_sources.TimePins()) == [0, 1, 198, 199]


def test_pins_on_a_calendar_that_does_not_compare_warn_and_keep_the_ends(monkeypatch):
    monkeypatch.setattr(
        _sources,
        "resolve",
        lambda name: SimpleNamespace(name=name, metadata={}, path=None),
    )
    noleap = xr.Dataset(
        {"temp": (("time", "y"), np.zeros((6, 2)))},
        coords={
            "time": xr.date_range(
                "2020-01-01", periods=6, calendar="noleap", use_cftime=True
            )
        },
    )
    monkeypatch.setattr(_sources, "_read_uncached", lambda *a, **k: noleap)
    with (
        _sources.truncate_time(["m"], 1, pins=_pins("2020-01-03")),
        pytest.warns(UserWarning, match="could not keep the steps the pages pin"),
    ):
        assert osk.read("m").sizes["time"] == 2


def test_smoke_keeps_the_pinned_steps_and_logs_them(
    tmp_path, long_sources, monkeypatch, capsys
):
    """A page naming a date in the middle still finds it; the log says what is kept."""
    seen = {}

    def fake_prepare(source, variable, select, aggregate, **kwargs):
        seen["days"] = _days_of(osk.read("m").time)
        da = xr.DataArray(
            np.full((2, 2), 5.0),
            dims=("lat", "lon"),
            coords={"lat": [10.0, 20.0], "lon": [-100.0, -90.0]},
            attrs={"units": "degC"},
        )
        return da, None

    monkeypatch.setattr(_comparison, "prepare_source", fake_prepare)
    suite = _smoke_suite(tmp_path)
    suite["pages"] = [
        _time_page("Mid", {"depth": "surface", "time": "2020-03-10"}),
        _time_page("Window", {"time": {"min": "2020-05-01", "max": "2020-05-04"}}),
    ]
    path = _write_suite(tmp_path, suite)
    result = run_suite(path, refresh=False, smoke=1)
    out = capsys.readouterr().out
    assert result.exit_code == 0
    assert seen["days"] == [0, 69, 121, 124, 199]
    assert "reading only" not in out
    assert "first and last 1 time step(s)" in out
    assert "1 instant(s), 1 short window(s)" in out
    assert "m 5 step(s)" in out
    modes = json.loads(result.manifest.read_text())["modes"]
    assert modes["smoke_pins"] == {"instants": 1, "windows": 1}


def _time_page(title, select):
    return {"title": title, "field": {"variables": ["temperature"], "select": select}}


def _days_of(t):
    return list((t - np.datetime64("2020-01-01")).dt.days.values)


@pytest.mark.usefixtures("long_sources")
def test_a_changed_pin_gets_a_new_smoke_cache(tmp_path, smoke_code):
    """The same ends but another step pinned in between is not the same cache."""

    def run(day):
        suite = _smoke_suite(tmp_path)
        suite["pages"] = [_time_page("Mid", {"time": day})]
        result = run_suite(_write_suite(tmp_path, suite), refresh=False, smoke=1)
        modes = json.loads(result.manifest.read_text())["modes"]
        return Path(modes["smoke_cache"]), modes

    first, _ = run("2020-03-10")
    again, modes = run("2020-03-10")
    assert again == first and modes["smoke_cache_reused"] is True
    other, modes = run("2020-04-10")
    assert other != first and modes["smoke_cache_reused"] is False


# -- the smoke cache key covers the data on disk, not just the kept time stamps ------


def _refresh_block(tmp_path: Path, **forcing_kw):
    from ocean_skill.config import RefreshConfig

    ref = tmp_path / "run.json"
    ref.write_text("{}")
    cat = tmp_path / "cat.yaml"
    cat.write_text("sources: {}")
    forcing = {
        "name": "bry",
        "kind": "surface",
        "files": str(tmp_path / "forcing" / "*.nc"),
        "grid": str(tmp_path / "grid.nc"),
        **forcing_kw,
    }
    (tmp_path / "forcing").mkdir()
    (tmp_path / "forcing" / "a.nc").write_bytes(b"x")
    return RefreshConfig(
        catalog=str(cat),
        sources=[{"name": "run", "files": "unused*.nc", "ref": str(ref)}],
        forcing=[forcing],
    ), ref, cat


def _bump(path: Path, *, size: bool = False) -> None:
    if size:
        path.write_bytes(path.read_bytes() + b"more")
    else:
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def test_smoke_cache_key_changes_when_a_source_ref_is_rewritten(tmp_path, monkeypatch):
    monkeypatch.setattr(_run, "_code_identity", lambda: "code-1")
    refresh, ref, cat = _refresh_block(tmp_path)
    stamps = [("run", 3, "a", "b", 6, "d")]  # the same kept time stamps throughout
    base, _ = _run._smoke_cache_key(stamps, _run._smoke_data_stamps(refresh))
    again, _ = _run._smoke_cache_key(stamps, _run._smoke_data_stamps(refresh))
    assert base == again
    # the reference rewritten in place (mtime moves, or just its size)
    _bump(ref)
    moved, _ = _run._smoke_cache_key(stamps, _run._smoke_data_stamps(refresh))
    assert moved != base
    _bump(ref, size=True)
    resized, _ = _run._smoke_cache_key(stamps, _run._smoke_data_stamps(refresh))
    assert resized not in (base, moved)
    # the catalog file too
    _bump(cat)
    catalog, _ = _run._smoke_cache_key(stamps, _run._smoke_data_stamps(refresh))
    assert catalog != resized


def test_smoke_cache_key_covers_forcing_files_and_a_parquet_ref_folder(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_run, "_code_identity", lambda: "code-1")
    refresh, _, _ = _refresh_block(tmp_path)  # forcing has no ref: its files count
    stamps = [("run", 3, "a", "b", 6, "d")]
    base, _ = _run._smoke_cache_key(stamps, _run._smoke_data_stamps(refresh))
    _bump(tmp_path / "forcing" / "a.nc", size=True)
    forced, _ = _run._smoke_cache_key(stamps, _run._smoke_data_stamps(refresh))
    assert forced != base

    folder = tmp_path / "forcing_ref.parq"
    folder.mkdir()
    (folder / "refs.0.parq").write_bytes(b"1")
    (tmp_path / "second").mkdir()
    refresh, _, _ = _refresh_block(tmp_path / "second", ref=str(folder))
    first, _ = _run._smoke_cache_key(stamps, _run._smoke_data_stamps(refresh))
    _bump(folder / "refs.0.parq", size=True)  # a file inside; the folder's mtime stays
    second, _ = _run._smoke_cache_key(stamps, _run._smoke_data_stamps(refresh))
    assert first != second
