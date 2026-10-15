"""``ocean-skill-run``'s guardrails: ``--smoke``, ``--self-check``, ``--only``, ...

Each flag exists so a bug in a suite shows up in minutes rather than after the whole
compute. These stay fast by stubbing the data (``stub_model``) or the comparison
(``_FakeComparison``), and test what the runner itself adds: the truncated read and its
separate output/cache directories, the self-check substitution and its verdict, the
title filter, the strict warning watch, the problems summary and the all-NaN guard.
"""

from __future__ import annotations

import json
import sys
import warnings
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


def test_truncate_time_keeps_the_last_steps_of_the_named_sources_only(fake_sources):
    with _sources.truncate_time(["m"], 2):
        assert osk.read("m").sizes["time"] == 2
        assert list(osk.read("m").time.dt.day) == [5, 6]
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
        assert list(osk.read("test").time.dt.day) == [5, 6]
        assert list(osk.read("ref").time.dt.day) == [5, 6]  # not 9, 10
        assert osk.read("ic").sizes["time"] == 1
        assert list(osk.read("late").time.dt.day) == [3, 4]
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
        assert osk.read("m").sizes["time"] == 2
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
        indexes.append(real_index(source))
        return indexes[-1]

    monkeypatch.setattr("ocean_skill.extrema._native_time_index", spy_index)

    path = _write_suite(tmp_path, _smoke_suite(tmp_path))
    result = run_suite(path, refresh=False, smoke=3)

    assert result.exit_code == 0
    assert result.report_dir == (tmp_path / "out" / "quick_check-smoke")
    assert seen["steps"] == 3
    # the time-index lookups made during expansion saw the same truncated record
    assert indexes and all(len(i) == 3 for i in indexes)
    assert seen["cache"] == _cache.base_dir() / "smoke-cache" / "prepared"
    manifest = json.loads(result.manifest.read_text())
    assert manifest["modes"]["smoke"] == 3
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
    assert got["smoke"] == 5 and got["only"] == "a*" and got["strict"] is True
    main([str(tmp_path / "s.yaml"), "--self-check"])
    assert got["smoke"] is None and got["self_check"] is True


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

    assert result.report_dir.name == "quick_check-only"
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

    def expand(suite):
        pages = real(suite)
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


@pytest.mark.usefixtures("fake_sources")
def test_every_smoke_run_starts_from_an_empty_smoke_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _comparison, "prepare_source", lambda *a, **k: stub_model_field()
    )
    stale = _cache.base_dir() / "smoke-cache" / "prepared" / "old_run.zarr"
    stale.mkdir(parents=True)
    path = _write_suite(tmp_path, _smoke_suite(tmp_path))
    run_suite(path, refresh=False, smoke=2)
    assert not stale.exists()


@pytest.mark.usefixtures("fake_sources")
def test_list_prints_the_real_cache_and_a_separate_smoke_cache_line(
    tmp_path, capsys
):
    path = _write_suite(tmp_path, _smoke_suite(tmp_path))
    run_suite(path, refresh=False, list_only=True)
    out = capsys.readouterr().out
    assert f"cache: {_cache.base_dir()}\n" in out and "smoke cache" not in out
    run_suite(path, refresh=False, list_only=True, smoke=2)
    out = capsys.readouterr().out
    assert f"cache: {_cache.base_dir()}\n" in out
    assert f"smoke cache: {_cache.base_dir() / 'smoke-cache'}" in out


def test_peak_memory_is_unknown_without_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)  # makes the import fail
    with _run._PeakMemory() as mem:
        pass
    assert mem.peak_mb is None
    assert _run._problem_lines([PageProblem("t", "failed", "r", 1.0, None)]) == [
        "1 problem(s):",
        "  [failed] t: r (1.0s)",
    ]


def test_an_anchor_that_cannot_be_compared_warns_and_keeps_the_last_steps(
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
    assert got.sizes["time"] == 2


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
