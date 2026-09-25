"""``ocean_skill.workflows.run._refresh_sources``: rebuilding a suite's kerchunk refs.

No suite ever had to say ``keep:`` for the ordinary case -- these check that leaving
it out really does let :func:`ocean_skill.build.make_kerchunk`'s own default apply,
rather than ``_refresh_sources`` silently pinning every suite to the old ``"all"``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import catalog as _catalog
from ocean_skill import comparison as _comparison
from ocean_skill.workflows import pages as _pages
from ocean_skill.workflows.run import _refresh_sources, main, run_suite
from tests.test_catalog import _write_catalog


def _segment(path, t0, value):
    """One weekly-segment-like file: two half-daily records starting at ``t0``."""
    xr.Dataset(
        {"NO3": (("ocean_time", "eta_rho", "xi_rho"), np.full((2, 4, 5), value))},
        coords={"ocean_time": ("ocean_time", [t0, t0 + 43200.0], {"units": "second"})},
    ).to_netcdf(path)
    return path


def test_no_keep_key_still_collapses_a_restart_boundary(tmp_path):
    """The exact suite shape that hit the reported bug: no ``keep:`` in the YAML."""
    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    # restarted from 43200.0 -- same value, a genuine repeat
    _segment(tmp_path / "out.1.nc", 43200.0, 1.0)

    _refresh_sources(
        [
            {
                "name": "GOM_bgc",
                "files": str(tmp_path / "out.*.nc"),
                "ref": str(tmp_path / "refs" / "gom_bgc.json"),
            }
        ],
        tmp_path / "catalog.yaml",
    )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.json"),
        engine="kerchunk",
        chunks={},
        decode_times=False,
    )
    assert ds.sizes["ocean_time"] == 3, "the shared 43200.0 record must be collapsed"
    assert list(ds.ocean_time.values) == sorted(ds.ocean_time.values)


def test_an_explicit_keep_key_is_still_forwarded(tmp_path):
    """A suite that opts back into 'all' (or any other keep=) still gets it."""
    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    _segment(tmp_path / "out.1.nc", 43200.0, 1.0)

    _refresh_sources(
        [
            {
                "name": "GOM_bgc",
                "files": str(tmp_path / "out.*.nc"),
                "ref": str(tmp_path / "refs" / "gom_bgc.json"),
                "keep": "all",
            }
        ],
        tmp_path / "catalog.yaml",
    )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.json"),
        engine="kerchunk",
        chunks={},
        decode_times=False,
    )
    assert ds.sizes["ocean_time"] == 4, "keep='all' must still keep every record"


def test_no_keep_key_collapses_a_genuine_disagreement_without_raising(tmp_path):
    """A real overlapping rerun that is not bit-reproducible must not block a build.

    A suite that names no ``keep:`` at all gets the ``"last"`` default: the newer
    segment wins and a loud warning names the divergence, but the refresh
    completes -- this is the exact shape that first surfaced as the Anvil/Iceland
    build raising on a legitimate rerun.
    """
    _segment(tmp_path / "cdr.nc", 0.0, 1.0)
    _segment(tmp_path / "rst.nc", 0.0, 999.0)  # same stamps, different data

    with pytest.warns(UserWarning, match="DISAGREE"):
        _refresh_sources(
            [
                {
                    "name": "GOM_bgc",
                    "files": str(tmp_path / "*.nc"),
                    "ref": str(tmp_path / "refs" / "gom_bgc.json"),
                }
            ],
            tmp_path / "catalog.yaml",
        )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.json"),
        engine="kerchunk",
        chunks={},
        decode_times=False,
    )
    assert float(ds.NO3.isel(ocean_time=0, eta_rho=0, xi_rho=0)) == 999.0, (
        "the last-globbed (rst) record must win"
    )


def test_a_suite_can_opt_into_the_strict_raise(tmp_path):
    """``keep: unique`` in the suite YAML still gets the mixed-stream tripwire."""
    _segment(tmp_path / "cdr.nc", 0.0, 1.0)
    _segment(tmp_path / "rst.nc", 0.0, 999.0)  # same stamps, different data

    with pytest.raises(ValueError, match="disagree"):
        _refresh_sources(
            [
                {
                    "name": "GOM_bgc",
                    "files": str(tmp_path / "*.nc"),
                    "ref": str(tmp_path / "refs" / "gom_bgc.json"),
                    "keep": "unique",
                }
            ],
            tmp_path / "catalog.yaml",
        )


def test_a_file_still_being_written_is_skipped_and_the_refresh_still_completes(
    tmp_path,
):
    """The exact case this exists for: refreshing against a run still writing output."""
    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    bad = tmp_path / "out.1.nc"
    _segment(bad, 43200.0, 1.0)
    data = bad.read_bytes()
    bad.write_bytes(data[: len(data) // 2])  # looks like it, mid-write

    _refresh_sources(
        [
            {
                "name": "GOM_bgc",
                "files": str(tmp_path / "out.*.nc"),
                "ref": str(tmp_path / "refs" / "gom_bgc.json"),
            }
        ],
        tmp_path / "catalog.yaml",
    )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.json"),
        engine="kerchunk",
        chunks={},
        decode_times=False,
    )
    assert ds.sizes["ocean_time"] == 2  # only out.0.nc's records


def test_a_stream_matching_only_unfinished_files_keeps_its_existing_entry(tmp_path):
    """Distinct from a real failure: a live run between output steps is routine.

    An entry whose only matched file currently looks unfinished must be treated the
    same as one whose glob matched nothing at all -- keep whatever the catalog
    already has for it, don't raise, and don't touch other entries.
    """
    import intake

    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    catalog_path = tmp_path / "catalog.yaml"
    spec = [
        {
            "name": "GOM_bgc",
            "files": str(tmp_path / "out.*.nc"),
            "ref": str(tmp_path / "refs" / "gom_bgc.json"),
        }
    ]
    _refresh_sources(spec, catalog_path)
    before = intake.from_yaml_file(str(catalog_path))["GOM_bgc"].read()

    # the run has moved on to a new segment that isn't finished yet
    bad = tmp_path / "out.0.nc"
    data = bad.read_bytes()
    bad.write_bytes(data[: len(data) // 2])

    _refresh_sources(spec, catalog_path)  # must not raise
    after = intake.from_yaml_file(str(catalog_path))["GOM_bgc"].read()
    xr.testing.assert_identical(before, after)


# ======================================================================================
# ``run_suite`` / ``main``: the page orchestrator and CLI.
#
# Field pages go through the real ``osk.field(...).plot()`` path with only
# ``comparison.prepare_source`` and ``extrema._native_time_index`` stubbed (the same
# minimal stub ``tests/test_field_map_grid.py`` uses) -- everything else, including
# ``FieldSet``'s own variable-availability pre-filter, runs for real. ``compare``/
# ``summary`` pages are exercised at the boundary this module owns:
# :func:`ocean_skill.workflows.pages.build` is the only place that calls
# ``osk.compare``/``osk.summary``, and those functions already have their own
# extensive test suites elsewhere in the repo, so here it is monkeypatched to
# isolate what ``run_suite`` itself is responsible for -- report layout, PNGs, the
# PDF, the metrics CSV, the manifest, and exit codes.
# ======================================================================================

_INDEX = pd.date_range("2010-01-05", periods=6, freq="7D")


def _stub_field(value: float = 5.0):
    lat = xr.DataArray([10.0, 20.0], dims="lat")
    lon = xr.DataArray([-100.0, -90.0], dims="lon")
    da = xr.DataArray(
        [[value, value], [value, value]],
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "degC"},
    )
    return da, None


@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(_comparison, "prepare_source", lambda *a, **k: _stub_field())
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: _INDEX)


def _write_suite(tmp_path, payload):
    import yaml

    path = tmp_path / "suite.yaml"
    path.write_text(yaml.dump(payload))
    return path


def _model_only_suite(tmp_path, **extra):
    return {
        "name": "quick_check",
        "output_dir": str(tmp_path / "out"),
        "defaults": {"test": "stub"},
        "pages": [
            {
                "title": "Physics latest",
                "field": {
                    "variables": ["temperature", "salinity"],
                    "select": {"depth": "surface", "time": "latest"},
                },
            },
        ],
        **extra,
    }


def test_model_only_report_writes_pdf_pngs_and_manifest(tmp_path, stub_model):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result = run_suite(path)

    assert result.exit_code == 0
    assert result.report_dir.exists()
    assert result.pdf is not None and result.pdf.exists()
    assert (result.report_dir / "suite.yaml").read_text()
    assert result.manifest.exists()
    manifest = json.loads(result.manifest.read_text())
    assert manifest["name"] == "quick_check"
    assert manifest["pages"][0]["status"] == "ok"
    assert len(result.figures) == 3  # title page, one drawn page, log page


def test_list_only_prints_and_draws_nothing(tmp_path, stub_model, capsys):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result = run_suite(path, list_only=True)
    out = capsys.readouterr().out
    assert "Physics latest" in out
    assert result.report_dir is None
    assert not (tmp_path / "out").exists()


def test_a_missing_variable_page_is_skipped_not_fatal(
    tmp_path, stub_model, monkeypatch
):
    monkeypatch.setattr(_comparison, "_variable_available", lambda *a, **k: False)
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result = run_suite(path)

    assert result.exit_code == 1  # the only page was skipped
    page = result.pages[0]
    assert page.status == "skipped"
    assert "none" in page.reason or "temperature" in page.reason


def test_two_runs_never_overwrite_each_other(tmp_path, stub_model):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result1 = run_suite(path)
    result2 = run_suite(path)
    assert result1.report_dir != result2.report_dir
    assert result1.report_dir.exists() and result2.report_dir.exists()
    latest = (tmp_path / "out" / "latest.txt").read_text()
    assert latest == str(result2.report_dir)


def test_suite_yaml_copy_is_byte_identical(tmp_path, stub_model):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result = run_suite(path)
    assert (result.report_dir / "suite.yaml").read_bytes() == path.read_bytes()


# -- catalog_search_paths: ---------------------------------------------------------


def test_catalog_search_paths_absolute_path_registered_and_recorded(
    tmp_path, stub_model, isolated_catalogs
):
    shared = tmp_path / "shared_abs"
    _write_catalog(shared, title="shared catalog", name="bar")

    path = _write_suite(
        tmp_path, _model_only_suite(tmp_path, catalog_search_paths=[str(shared)])
    )
    result = run_suite(path)

    assert shared in _catalog.search_paths()
    assert "bar" in _catalog.discover()
    manifest = json.loads(result.manifest.read_text())
    assert manifest["catalog_search_paths"] == [str(shared)]


def test_catalog_search_paths_relative_path_resolves_against_suite_file_not_cwd(
    tmp_path, stub_model, isolated_catalogs, monkeypatch
):
    import yaml

    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    shared = tmp_path / "shared"
    _write_catalog(shared, title="shared catalog", name="bar")

    payload = _model_only_suite(tmp_path, catalog_search_paths=["../shared"])
    path = suite_dir / "suite.yaml"
    path.write_text(yaml.dump(payload))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = run_suite(path)

    assert shared.resolve() in _catalog.search_paths()
    manifest = json.loads(result.manifest.read_text())
    assert manifest["catalog_search_paths"] == [str(shared.resolve())]


def test_catalog_search_paths_missing_directory_raises_naming_entry_and_resolved_path(
    tmp_path, stub_model
):
    import yaml

    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    payload = _model_only_suite(tmp_path, catalog_search_paths=["../nope"])
    path = suite_dir / "suite.yaml"
    path.write_text(yaml.dump(payload))

    with pytest.raises(FileNotFoundError) as exc_info:
        run_suite(path)
    message = str(exc_info.value)
    assert "../nope" in message
    assert str((tmp_path / "nope").resolve()) in message


def test_main_catalog_search_paths_missing_directory_is_a_usage_error(
    tmp_path, stub_model
):
    path = _write_suite(
        tmp_path,
        _model_only_suite(tmp_path, catalog_search_paths=[str(tmp_path / "nope")]),
    )
    assert main([str(path)]) == 2


def test_catalog_search_paths_registered_before_refresh_runs(
    tmp_path, stub_model, isolated_catalogs, monkeypatch
):
    shared = tmp_path / "shared_before_refresh"
    _write_catalog(shared, title="shared catalog", name="bar")

    seen_during_refresh = []

    def _fake_refresh(spec, cat):
        seen_during_refresh.append(shared in _catalog.search_paths())

    monkeypatch.setattr("ocean_skill.workflows.run._refresh_sources", _fake_refresh)

    suite = _model_only_suite(
        tmp_path,
        catalog_search_paths=[str(shared)],
        refresh={
            "catalog": "catalogs/x.yaml",
            "sources": [{"name": "stub", "files": "x/*.nc", "ref": "refs/x.parquet"}],
        },
    )
    path = _write_suite(tmp_path, suite)
    run_suite(path)

    assert seen_during_refresh == [True]


def test_catalog_search_paths_repeated_runs_do_not_duplicate_added_dirs(
    tmp_path, stub_model, isolated_catalogs
):
    shared = tmp_path / "shared_repeat"
    _write_catalog(shared, title="shared catalog", name="bar")

    path = _write_suite(
        tmp_path, _model_only_suite(tmp_path, catalog_search_paths=[str(shared)])
    )
    run_suite(path)
    run_suite(path)

    assert _catalog._added_dirs.count(shared) == 1


def test_refresh_block_still_calls_refresh_sources(tmp_path, stub_model, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "ocean_skill.workflows.run._refresh_sources",
        lambda spec, cat: calls.append((spec, cat)),
    )
    suite = _model_only_suite(
        tmp_path,
        refresh={
            "catalog": "catalogs/x.yaml",
            "sources": [{"name": "stub", "files": "x/*.nc", "ref": "refs/x.parquet"}],
        },
    )
    path = _write_suite(tmp_path, suite)
    run_suite(path)
    assert len(calls) == 1
    assert calls[0][1] == "catalogs/x.yaml"
    assert calls[0][0][0]["name"] == "stub"


def _obs_suite(tmp_path):
    return {
        "name": "with_obs",
        "output_dir": str(tmp_path / "out"),
        "defaults": {"test": "stub"},
        "pages": [
            {
                "title": "WOA",
                "compare": {
                    "reference": ["woa23_nitrate_month01"],
                    "variables": ["nitrate"],
                    "aggregate": {"time": "mean"},
                },
            },
            {"title": "Summary", "summary": {"kind": "portrait"}},
        ],
    }


def test_a_failing_compare_page_is_skipped_and_summary_is_skipped_too(
    tmp_path, stub_model, monkeypatch
):
    def fake_build(page, *, pooled_records=None):
        if page.kind == "compare":
            raise RuntimeError("regrid failed")
        # summary: still reached (every page is attempted), but with nothing
        # pooled to summarize -- the same shape the real build() raises.
        if not pooled_records:
            raise ValueError("no compare page produced results to summarize")
        raise AssertionError("unexpected pooled records in this test")

    monkeypatch.setattr(_pages, "build", fake_build)
    path = _write_suite(tmp_path, _obs_suite(tmp_path))
    result = run_suite(path)

    by_title = {p.title: p for p in result.pages}
    assert by_title["WOA"].status == "skipped"
    assert "regrid failed" in by_title["WOA"].reason
    assert by_title["Summary"].status == "skipped"
    assert result.metrics is None
    assert result.report_dir.exists()  # the report still completes and is written
    assert result.exit_code == 1  # both of this suite's two pages were skipped


def test_exit_code_is_3_when_some_pages_ok_and_some_skipped(
    tmp_path, stub_model, monkeypatch
):
    real_build = _pages.build

    def fake_build(page, *, pooled_records=None):
        if page.kind == "field":
            return real_build(page, pooled_records=pooled_records)
        raise RuntimeError("regrid failed")

    monkeypatch.setattr(_pages, "build", fake_build)
    suite = _model_only_suite(tmp_path)
    suite["pages"].append(
        {
            "title": "WOA",
            "compare": {
                "reference": ["woa23_nitrate_month01"],
                "variables": ["nitrate"],
                "aggregate": {"time": "mean"},
            },
        }
    )
    path = _write_suite(tmp_path, suite)
    result = run_suite(path)
    assert result.exit_code == 3
    assert result.report_dir.exists()


def test_an_unresolvable_obs_source_is_a_graceful_page_skip(
    tmp_path, stub_model, monkeypatch
):
    """A page skip driven by a real (unmocked) failure, not a controlled stand-in.

    A real ``osk.compare()`` call against a source no catalog on the search path
    declares must still be caught by ``run_suite``'s per-page try/except rather
    than aborting the report.
    """
    empty_cats = tmp_path / "empty_cats"
    empty_cats.mkdir()
    monkeypatch.setenv("OCEAN_SKILL_CATALOGS", str(empty_cats))

    path = _write_suite(tmp_path, _obs_suite(tmp_path))
    result = run_suite(path)

    assert result.report_dir.exists()
    assert result.pdf is not None and result.pdf.exists()
    by_title = {p.title: p for p in result.pages}
    assert by_title["WOA"].status == "skipped"
    assert by_title["Summary"].status == "skipped"
    assert result.metrics is None
    assert result.exit_code == 1


def test_compare_metrics_pool_into_a_summary_page_and_a_csv(
    tmp_path, stub_model, monkeypatch
):
    record = {
        "variable": "nitrate",
        "bias": 0.1,
        "rmse": 0.2,
        "corr": 0.9,
        "sigma_ratio": 1.0,
        "n": 10,
        "label": "nitrate",
        "units": "mmol m-3",
    }

    import matplotlib.pyplot as plt

    def fake_build(page, *, pooled_records=None):
        if page.kind == "compare":
            page.metrics_records = [record]
            return [("", plt.subplots()[0])]
        if page.kind == "summary":
            assert pooled_records == [record]
            return [("", plt.subplots()[0])]
        raise AssertionError("no field pages in this suite")

    monkeypatch.setattr(_pages, "build", fake_build)
    path = _write_suite(tmp_path, _obs_suite(tmp_path))
    result = run_suite(path)

    assert result.exit_code == 0
    assert result.metrics is not None and result.metrics.exists()
    df = pd.read_csv(result.metrics)
    assert df.iloc[0]["variable"] == "nitrate"


# -- CLI -------------------------------------------------------------------------------


def test_main_list_flag(tmp_path, stub_model, capsys):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    code = main([str(path), "--list"])
    assert code == 0
    assert "Physics latest" in capsys.readouterr().out


def test_main_no_args_is_a_usage_error():
    assert main([]) == 2


def test_main_bad_schema_is_a_usage_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: t\npages: []\n")
    assert main([str(path)]) == 2


def test_main_reports_exit_code_from_run(tmp_path, stub_model, capsys):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    code = main([str(path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "page(s) drawn" in out
