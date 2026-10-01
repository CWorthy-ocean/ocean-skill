"""``ocean_skill.workflows.run._refresh_sources``: rebuilding a suite's kerchunk refs.

No suite ever had to say ``keep:`` for the ordinary case -- these check that leaving
it out really does let :func:`ocean_skill.build.make_kerchunk`'s own default apply,
rather than ``_refresh_sources`` silently pinning every suite to the old ``"all"``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import cache as _cache
from ocean_skill import catalog as _catalog
from ocean_skill import comparison as _comparison
from ocean_skill.config import SuiteConfig
from ocean_skill.workflows import pages as _pages
from ocean_skill.workflows.run import _refresh_sources, main, run_suite
from tests.test_catalog import _write_catalog
from tests.test_workflows_report import _pagecount_by_bytes


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
    assert len(result.figures) == 1  # one drawn page


def test_a_one_variable_monthly_means_page_draws_instead_of_being_skipped(
    tmp_path, monkeypatch
):
    """A one-variable ``variables:`` list used to be refused and skipped.

    :meth:`~ocean_skill.field.Field._map_item` used to refuse a standing monthly
    time axis (see ``tests/test_field_map_grid.py``'s own coverage of the same
    fix), logging the page as ``skipped`` -- the exact shape of the shipped
    suites' own monthly-means pages (``suites/roms_marbl_quick.yaml``,
    ``suites/roms_marbl_diagnostic.yaml``).
    """
    months = pd.date_range("2010-01-01", periods=3, freq="MS")
    lat = xr.DataArray([10.0, 20.0], dims="lat")
    lon = xr.DataArray([-100.0, -90.0], dims="lon")

    def stub(*a, **k):
        da = xr.DataArray(
            np.full((3, 2, 2), 5.0),
            dims=("time", "lat", "lon"),
            coords={"time": months, "lat": lat, "lon": lon},
            attrs={"units": "degC"},
        )
        return da, None

    monkeypatch.setattr(_comparison, "prepare_source", stub)
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: months)

    suite = _model_only_suite(tmp_path)
    suite["pages"] = [
        {
            "title": "Monthly means -- temperature",
            "field": {
                "variables": ["temperature"],
                "aggregate": {"time": {"resample": "1MS", "reduce": "mean"}},
            },
        },
    ]
    path = _write_suite(tmp_path, suite)
    result = run_suite(path)

    page = result.pages[0]
    assert page.status == "ok", page.reason
    assert len(result.figures) == 1  # one drawn page


def test_list_only_prints_and_draws_nothing(tmp_path, stub_model, capsys):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result = run_suite(path, list_only=True)
    out = capsys.readouterr().out
    assert "Physics latest" in out
    assert f"latest step of stub: {_INDEX[-1]}" in out
    # time: latest is pinned to the resolved step above, so it caches.
    assert "(cache)" in out
    assert result.report_dir is None
    assert result.log is None
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.rglob("run.log"))


def test_list_only_notes_suite_level_cache_false_distinctly(
    tmp_path, stub_model, capsys
):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path, cache=False))
    run_suite(path, list_only=True)
    out = capsys.readouterr().out
    assert "no-cache (cache: false)" in out
    assert "no-cache (may change as the run grows)" not in out


def test_list_only_notes_a_growing_selection_distinctly(tmp_path, stub_model, capsys):
    suite = _model_only_suite(tmp_path)
    suite["pages"].append(
        {
            "title": "WOA",
            "compare": {
                "reference": ["woa23_nitrate_month01"],
                "variables": ["nitrate"],
                "aggregate": {"time": "mean"},
                "select": {"depth": "surface"},  # flat select, no time key
            },
        }
    )
    path = _write_suite(tmp_path, suite)
    run_suite(path, list_only=True)
    out = capsys.readouterr().out
    assert "no-cache (may change as the run grows)" in out
    assert "no-cache (cache: false)" not in out


# -- then: extremum -> series, end to end -----------------------------------------

_EXT_INDEX = pd.date_range("2010-01-05", periods=6, freq="7D")
_EXT_LAT = np.array([10.0, 20.0, 30.0])
_EXT_LON = np.array([-100.0, -95.0, -90.0])


def _extremum_stub(source, variable, select, aggregate, **kwargs):
    """Return a map with a planted min, or a point series at a pinned lon/lat.

    Which one depends on ``select``: a plain map until it names a lon/lat, the
    same shape :meth:`~ocean_skill.extrema.Extremum.series`'s re-entry into
    ``field()`` actually asks for -- so the ``then:`` chain draws a real
    ``series`` family figure rather than stalling on a map it cannot follow.
    """
    if select and "lon" in select and "lat" in select:
        values = 5.0 + np.sin(np.arange(len(_EXT_INDEX)))
        da = xr.DataArray(
            values,
            dims="time",
            coords={"time": _EXT_INDEX},
            attrs={"units": "mmol m-3"},
        )
        lon, lat = float(select["lon"]), float(select["lat"])
        return da.assign_coords(lon=lon, lat=lat), None
    values = np.full((3, 3), 5.0)
    values[1, 2] = -50.0
    da = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": _EXT_LAT, "lon": _EXT_LON},
        attrs={"units": "mmol m-3"},
    )
    return da, None


@pytest.fixture
def stub_extremum_model(monkeypatch):
    monkeypatch.setattr(_comparison, "prepare_source", _extremum_stub)
    monkeypatch.setattr(
        "ocean_skill.extrema._native_time_index", lambda source: _EXT_INDEX
    )


def _extremum_suite(tmp_path, **extra):
    return _model_only_suite(
        tmp_path,
        pages=[
            {
                "title": "Alkalinity minimum",
                "field": {
                    "variables": ["alkalinity"],
                    "select": {"depth": "surface", "time": "2010-01-19"},
                },
                "then": [
                    {"extremum": "min"},
                    {"series": {"variables": ["dissolved_inorganic_carbon"]}},
                ],
            },
        ],
        **extra,
    )


def test_then_extremum_series_draws_a_figure_and_records_results(
    tmp_path, stub_extremum_model
):
    path = _write_suite(tmp_path, _extremum_suite(tmp_path))
    result = run_suite(path)

    page = result.pages[0]
    assert page.status == "ok", page.reason
    assert len(result.figures) == 1
    assert len(page.results) == 1
    rec = page.results[0]
    assert rec["kind"] == "min"
    assert rec["value"] == pytest.approx(-50.0)
    assert rec["lon"] == pytest.approx(-90.0)
    assert rec["lat"] == pytest.approx(20.0)


def test_then_manifest_carries_steps_and_results(tmp_path, stub_extremum_model):
    path = _write_suite(tmp_path, _extremum_suite(tmp_path))
    result = run_suite(path)

    manifest = json.loads(result.manifest.read_text())
    page = manifest["pages"][0]
    assert page["steps"][0] == {"name": "extremum", "kwargs": {"kind": "min"}}
    assert page["steps"][1]["name"] == "series"
    assert page["results"][0]["value"] == pytest.approx(-50.0)


def test_then_extremum_repr_is_printed_to_run_log(tmp_path, stub_extremum_model):
    path = _write_suite(tmp_path, _extremum_suite(tmp_path))
    result = run_suite(path)
    log_text = result.log.read_text()
    assert "min " in log_text
    assert "lon -90.0000, lat 20.0000" in log_text


def test_list_only_shows_the_then_chain(tmp_path, stub_extremum_model, capsys):
    path = _write_suite(tmp_path, _extremum_suite(tmp_path))
    run_suite(path, list_only=True)
    out = capsys.readouterr().out
    assert "extremum(kind='min')" in out
    assert "series(variables=" in out


def test_field_page_forwards_qc_detide_and_label(tmp_path, monkeypatch):
    captured: dict = {}

    def capturing_stub(source, variable, select, aggregate, **kwargs):
        captured.update(kwargs)
        return _stub_field()

    monkeypatch.setattr(_comparison, "prepare_source", capturing_stub)
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: _INDEX)

    suite = _model_only_suite(
        tmp_path,
        pages=[
            {
                "title": "QC'd",
                "field": {
                    "variables": ["temperature"],
                    "select": {"depth": "surface", "time": "latest"},
                    "qc": {"range": [-2, 40]},
                    "detide": True,
                    "label": "custom label",
                },
            },
        ],
    )
    path = _write_suite(tmp_path, suite)
    result = run_suite(path)

    assert result.pages[0].status == "ok", result.pages[0].reason
    assert captured["qc"] == {"range": [-2, 40]}
    assert captured["detide"] == {"T": 33.0}


def test_field_page_rejects_a_cache_kwarg(tmp_path, stub_model):
    suite = _model_only_suite(tmp_path)
    suite["pages"][0]["field"]["cache"] = True
    path = _write_suite(tmp_path, suite)
    result = run_suite(path)

    page = result.pages[0]
    assert page.status == "skipped"
    assert "cache:" in page.reason


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


# -- XY: / TS: pages -----------------------------------------------------------------
#
# ``osk.XY``/``osk.TS`` are another worker's classes; what ``run_suite`` owns is
# building one member per ``members:`` entry, handing them over, and drawing the
# figure the page's ``.plot()`` returns. A stand-in class records what it was built
# with and draws a one-axes figure, so these hold whatever the real classes do.


@pytest.fixture
def fake_xy(monkeypatch):
    """Replace ``osk.XY``/``osk.TS`` with recorders; return their call log."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import ocean_skill as osk

    log = []

    class _Fake:
        def __init__(self, members, **kwargs):
            self.members = members
            self.kwargs = kwargs
            log.append((type(self).__name__, members, kwargs))

        def plot(self, **opts):
            log[-1] = (*log[-1], opts)
            fig, ax = plt.subplots()
            ax.plot([34.0, 35.0], [10.0, 20.0])
            return fig

    class FakeXY(_Fake):
        pass

    class FakeTS(_Fake):
        pass

    monkeypatch.setattr(osk, "XY", FakeXY, raising=False)
    monkeypatch.setattr(osk, "TS", FakeTS, raising=False)
    # regions are validated at expand time; what counts as valid is xy.py's business
    monkeypatch.setattr("ocean_skill.xy.normalize_regions", lambda r: {}, raising=False)
    return log


_BOX = {"lon": {"min": 155.24, "max": 156.33}, "lat": {"min": 20.51, "max": 21.60}}


def _ts_suite(tmp_path, kind="TS", **page_extra):
    page = {
        "title": "T-S diagrams",
        kind: {
            "members": {
                "ROMS": {},
                "WOA23": {"source": ["woa_temperature", "woa_salinity"]},
            },
            "regions": {"North West Pacific": _BOX},
            "at_center": ["WOA23"],
            **({"x": "phosphate", "y": "nitrate"} if kind == "XY" else {}),
        },
        "plot": {"ncols": 3},
        **page_extra,
    }
    return {
        "name": "quick_check",
        "output_dir": str(tmp_path / "out"),
        "defaults": {"test": "stub"},
        "pages": [page],
    }


def test_a_ts_page_draws_and_the_manifest_records_it(tmp_path, stub_model, fake_xy):
    result = run_suite(_write_suite(tmp_path, _ts_suite(tmp_path)))

    page = result.pages[0]
    assert page.status == "ok", page.reason
    assert result.exit_code == 0
    assert len(result.figures) == 1  # one figure, whatever the number of regions

    kind, members, kwargs, plot_opts = fake_xy[0]
    assert kind == "FakeTS"
    assert list(members) == ["ROMS", "WOA23"]  # one real osk.field() per member
    assert kwargs == {"regions": {"North West Pacific": _BOX}, "at_center": ["WOA23"]}
    assert plot_opts == {"ncols": 3, "size": "page"}  # pdf: true pins the canvas

    manifest = json.loads(result.manifest.read_text())
    (entry,) = manifest["pages"]
    assert entry["kind"] == "TS" and entry["status"] == "ok"
    assert entry["cache"] is True
    assert list(entry["kwargs"]["members"]) == ["ROMS", "WOA23"]
    assert entry["kwargs"]["members"]["ROMS"]["source"] == "stub"
    assert entry["kwargs"]["regions"] == {"North West Pacific": _BOX}
    assert entry["kwargs"]["at_center"] == ["WOA23"]


def test_an_xy_page_passes_x_and_y_to_osk_xy(tmp_path, stub_model, fake_xy):
    result = run_suite(_write_suite(tmp_path, _ts_suite(tmp_path, kind="XY")))

    assert result.pages[0].status == "ok", result.pages[0].reason
    kind, _, kwargs, _ = fake_xy[0]
    assert kind == "FakeXY"
    assert (kwargs["x"], kwargs["y"]) == ("phosphate", "nitrate")


def test_list_only_shows_an_xy_or_ts_page_with_its_cache_note(
    tmp_path, stub_model, fake_xy, capsys
):
    suite = _ts_suite(tmp_path)
    suite["pages"].append(_ts_suite(tmp_path, kind="XY")["pages"][0])
    suite["pages"][1]["title"] = "N-P"
    run_suite(_write_suite(tmp_path, suite), list_only=True)
    out = capsys.readouterr().out
    assert "[TS     ] T-S diagrams  (cache)" in out
    assert "[XY     ] N-P  (cache)" in out
    assert not fake_xy  # nothing was built or drawn


def test_an_xy_page_that_cannot_be_built_is_skipped_not_fatal(
    tmp_path, stub_model, fake_xy, monkeypatch
):
    import ocean_skill as osk

    def boom(*a, **k):
        raise ValueError("the 'WOA23' member has no salinity")

    monkeypatch.setattr(osk, "TS", boom, raising=False)
    result = run_suite(_write_suite(tmp_path, _ts_suite(tmp_path)))

    page = result.pages[0]
    assert page.status == "skipped" and "no salinity" in page.reason
    assert result.exit_code == 1  # the only page was skipped


def _ts_source(source, variable, select, aggregate, **kwargs):
    """Synthetic T and S for a real ``osk.TS`` run: a model box and a WOA profile.

    The model (``stub``) is a box of (time, s_rho, eta, xi) values with a height
    ``z_rho``, so it draws as dots; each WOA entry carries one variable and, sampled at
    the region's centre (``at_center``), is a profile -- a line.
    """
    from ocean_skill.vars import short_name

    name = short_name(variable)
    base = 10.0 if name == "temperature" else 34.5
    lon = select["lon"]
    lon = (lon["min"] + lon["max"]) / 2 if isinstance(lon, dict) else lon
    shift = (lon - 155.0) / 100.0  # each region a little different
    if source == "stub":
        depth = np.array([500.0, 200.0, 50.0, 5.0])  # s_rho: bottom to top
        shape = (len(_INDEX), 4, 2, 2)
        column = -depth[None, :, None, None] / 100
        values = base + shift + np.broadcast_to(column, shape)
        z_rho = np.broadcast_to(-depth[None, :, None, None], shape).copy()
        dims = ("time", "s_rho", "eta_rho", "xi_rho")
        da = xr.DataArray(
            values.copy(),
            dims=dims,
            coords={"time": _INDEX, "z_rho": (dims, z_rho)},
            attrs={"units": "degC" if name == "temperature" else "1"},
        )
    else:
        depth = np.array([0.0, 100.0, 1000.0])
        da = xr.DataArray(
            base + shift - depth / 200.0,
            dims="depth",
            coords={"depth": depth, "lon": lon, "lat": 21.0},
            attrs={"units": "degC" if name == "temperature" else "1"},
        )
    return da, None


def test_a_ts_page_runs_end_to_end_through_the_real_ts(tmp_path, monkeypatch):
    from ocean_skill import xy as _xy_module
    from ocean_skill.vars import short_name

    monkeypatch.setattr(_comparison, "prepare_source", _ts_source)
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: _INDEX)
    carries = {"woa_temperature": "temperature", "woa_salinity": "salinity"}
    monkeypatch.setattr(
        _comparison,
        "_variable_available",
        lambda source, variable, **k: (
            carries.get(source, short_name(variable)) == short_name(variable)
        ),
    )
    drawn = []
    real_items = _xy_module.XY._items

    def spy(self):
        drawn.append(real_items(self))
        return drawn[-1]

    monkeypatch.setattr(_xy_module.XY, "_items", spy)
    suite = _ts_suite(tmp_path)
    suite["pages"][0]["TS"]["regions"]["Subpolar Gyre"] = {
        "lon": {"min": 184.59, "max": 185.73},
        "lat": {"min": 47.76, "max": 48.59},
    }

    with pytest.warns(UserWarning, match="doesn't carry the requested variable"):
        result = run_suite(_write_suite(tmp_path, suite))

    page = result.pages[0]
    assert page.status == "ok", page.reason
    (png,) = result.figures
    assert Path(png).exists()
    (items,) = drawn
    assert [(i["region"], i["label"], i["mark"]) for i in items] == [
        ("North West Pacific", "ROMS", "points"),
        ("North West Pacific", "WOA23", "line"),
        ("Subpolar Gyre", "ROMS", "points"),
        ("Subpolar Gyre", "WOA23", "line"),
    ]
    # every cell, level and snapshot of the model box is a dot
    assert items[0]["x"].size == len(_INDEX) * 4 * 2 * 2


def test_a_bad_xy_page_is_a_schema_error_before_anything_is_drawn(
    tmp_path, stub_model, fake_xy
):
    suite = _ts_suite(tmp_path)
    suite["pages"][0]["TS"]["x"] = "salinity"  # TS: sets x and y itself
    with pytest.raises(ValueError, match="TS: sets x=salinity"):
        run_suite(_write_suite(tmp_path, suite), list_only=True)


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


# -- cache_dir: ---------------------------------------------------------------------


def test_cache_dir_absolute_path_relocates_cache_and_is_recorded(tmp_path, stub_model):
    pinned = tmp_path / "pinned_cache"
    path = _write_suite(tmp_path, _model_only_suite(tmp_path, cache_dir=str(pinned)))
    result = run_suite(path)

    assert _cache.base_dir() == pinned.resolve()
    assert _cache.obs_dir() == pinned.resolve() / "cache" / "obs"
    manifest = json.loads(result.manifest.read_text())
    assert manifest["cache_dir"] == str(pinned.resolve())


def test_cache_dir_relative_path_resolves_against_suite_file_not_cwd(
    tmp_path, stub_model, monkeypatch
):
    import yaml

    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    pinned = tmp_path / "pinned_relative"

    payload = _model_only_suite(tmp_path, cache_dir="../pinned_relative")
    path = suite_dir / "suite.yaml"
    path.write_text(yaml.dump(payload))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = run_suite(path)

    assert _cache.base_dir() == pinned.resolve()
    manifest = json.loads(result.manifest.read_text())
    assert manifest["cache_dir"] == str(pinned.resolve())


def test_no_cache_dir_manifest_records_whatever_cache_was_already_active(
    tmp_path, stub_model
):
    # The autouse ``isolated_cache`` fixture already pointed the cache at its own
    # tmp_path -- with no cache_dir: key, a suite must leave that alone.
    before = _cache.base_dir()
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result = run_suite(path)

    assert _cache.base_dir() == before
    manifest = json.loads(result.manifest.read_text())
    assert manifest["cache_dir"] == str(before)


def test_cache_dir_is_applied_under_list_only(tmp_path, stub_model):
    pinned = tmp_path / "pinned_list_only"
    path = _write_suite(tmp_path, _model_only_suite(tmp_path, cache_dir=str(pinned)))

    run_suite(path, list_only=True)

    assert _cache.base_dir() == pinned.resolve()
    assert not pinned.exists()  # applying it never creates the directory eagerly


# -- a page's resolved cache flag actually round-trips through the real cache -------
#
# ``stub_model`` replaces ``comparison.prepare_source`` wholesale, so it never
# exercises the cache layer prepare_source itself owns. These patch one level
# deeper -- ``comparison._prepare``, the read-and-reduce step *inside*
# prepare_source -- the same idiom ``tests/test_cache.py``'s own
# ``counted_pipeline`` fixture uses, so prepare_source's real cache-key/hit/miss
# logic runs for real (``isolated_cache``, autouse via conftest.py, already
# points it at a fresh temp dir).


@pytest.fixture
def counted_prepare(monkeypatch):
    """Patch out the expensive read/reduce step, counting how often it runs."""
    calls = {"n": 0}
    da = xr.DataArray(
        np.random.default_rng(0).normal(5.0, 1.0, (8, 10)),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(20, 30, 8), "lon": np.linspace(-100, -90, 10)},
        name="temperature",
        attrs={"units": "degC"},
    )

    def fake_prepare(obj, meta, variable, select, aggregate=None, **kwargs):
        calls["n"] += 1
        return da, None

    monkeypatch.setattr(_comparison, "_prepare", fake_prepare)
    monkeypatch.setattr(_catalog, "resolve", lambda name: mock.Mock(metadata={}))
    monkeypatch.setattr("ocean_skill.read", lambda n: None)
    return calls


def _expand_one_page(monkeypatch, index):
    suite = SuiteConfig.model_validate(
        {
            "name": "t",
            "defaults": {"test": "stub"},
            "pages": [
                {
                    "title": "x",
                    "field": {
                        "variables": ["temperature"],
                        "select": {"depth": "surface", "time": "latest"},
                    },
                }
            ],
        }
    )
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: index)
    return _pages.expand(suite)[0]


def test_a_latest_page_hits_the_real_cache_once_the_run_stops(
    monkeypatch, counted_prepare
):
    page = _expand_one_page(monkeypatch, _INDEX)
    assert page.cache is True

    from ocean_skill.comparison import prepare_source

    prepare_source(
        "stub",
        page.kwargs["variable"],
        page.kwargs["select"],
        page.kwargs.get("aggregate"),
        use_cache=page.cache,
    )
    prepare_source(
        "stub",
        page.kwargs["variable"],
        page.kwargs["select"],
        page.kwargs.get("aggregate"),
        use_cache=page.cache,
    )
    assert counted_prepare["n"] == 1, (
        "a rerun against an unchanged latest step should hit"
    )


def test_a_latest_page_recomputes_once_the_run_has_moved(monkeypatch, counted_prepare):
    from ocean_skill.comparison import prepare_source

    page = _expand_one_page(monkeypatch, _INDEX)
    prepare_source(
        "stub",
        page.kwargs["variable"],
        page.kwargs["select"],
        page.kwargs.get("aggregate"),
        use_cache=page.cache,
    )
    assert counted_prepare["n"] == 1

    grown = _INDEX.append(pd.DatetimeIndex([_INDEX[-1] + pd.Timedelta(days=7)]))
    page2 = _expand_one_page(monkeypatch, grown)
    assert page2.kwargs["select"]["time"] != page.kwargs["select"]["time"]
    prepare_source(
        "stub",
        page2.kwargs["variable"],
        page2.kwargs["select"],
        page2.kwargs.get("aggregate"),
        use_cache=page2.cache,
    )
    assert counted_prepare["n"] == 2, "a new latest step is a new key, so this misses"


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
    result = run_suite(path)
    assert len(calls) == 1
    assert calls[0][1] == "catalogs/x.yaml"
    assert calls[0][0][0]["name"] == "stub"

    # _refresh_sources was faked and never actually built refs/x.parquet, so the
    # snapshot step (which reads suite.refresh.sources directly, not this fake's
    # return value) has nothing to copy -- noted, not fatal.
    manifest = json.loads(result.manifest.read_text())
    assert manifest["refresh"]["catalog"] == str(Path("catalogs/x.yaml").resolve())
    assert manifest["refresh"]["sources"] == [
        {
            "name": "stub",
            "ref": str(Path("refs/x.parquet").resolve()),
            "snapshot": None,
        }
    ]
    assert result.refs == []


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
    # no page drew, so PdfPages never got a single savefig() call and never wrote
    # a file to disk at all
    assert result.pdf is None
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


def test_main_runs_a_suite_end_to_end(tmp_path, stub_model, capsys):
    """Everything one ``main([suite])`` run on a clean model-only suite has to do.

    Four claims, all against the same run — merged into one run because each used to
    build and throw away an identical suite just to check a different part of it.
    Kept as one function, not one assertion, so a failure still says which claim
    broke. ``main`` goes through ``run_suite`` and its ``_capture_terminal``, so the
    streams claim exercises the same swap-and-restore code as a direct ``run_suite``.
    """
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    stdout_before, stderr_before = sys.stdout, sys.stderr
    code = main([str(path)])

    # main reports the exit code from the run.
    assert code == 0

    # The summary line reaches the real terminal.
    out = capsys.readouterr().out
    assert "page(s) drawn" in out

    # run_suite's own tee is closed by the time main prints its summary, so main
    # appends those lines to run.log directly; they must end up there too.
    latest = Path((tmp_path / "out" / "latest.txt").read_text())
    log_text = (latest / "run.log").read_text()
    assert "page(s) drawn" in log_text

    # stdout/stderr are never left swapped out after a successful run.
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before


# -- run.log: the terminal transcript, persisted -------------------------------------
#
# ``run_suite`` mirrors stdout/stderr to ``<report_dir>/run.log`` for the whole run
# (see ``ocean_skill.workflows.run._capture_terminal``). These tests check that the
# file matches what the terminal actually showed, carries the tracebacks the terminal
# never shows (skipped pages, a fatal crash), stays attributable per page via headers
# and timing, is never written for ``--list``, and never leaves stdout/stderr swapped
# out after the run -- success, skip, or crash.


def test_run_log_of_a_skipped_page(tmp_path, stub_model, monkeypatch, capsys):
    """Everything the run log has to hold after a run whose page gets skipped.

    Two claims, both against the same run — merged into one run because each used to
    build and throw away an identical skipped-page suite just to check a different
    part of it. Kept as one function, not one assertion, so a failure still says
    which claim broke.
    """
    monkeypatch.setattr(_comparison, "_variable_available", lambda *a, **k: False)
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result = run_suite(path)

    # run.log is written, at the path the result reports, and matches the terminal.
    assert result.log == result.report_dir / "run.log"
    assert result.log.exists()
    log_text = result.log.read_text()
    out = capsys.readouterr().out
    # "SKIPPED after ...:" is a plain print(), so it reaches both the log file
    # and the real terminal; the warnings.warn() alongside it is not checked here
    # because pytest's own warning-capture plugin intercepts it before stderr
    # regardless of this tee (a test-harness artifact, not a run.py behavior).
    assert "SKIPPED after" in log_text
    assert "SKIPPED after" in out

    # The log also carries the traceback the terminal never shows for a skipped page.
    assert "Traceback" in log_text


def test_output_before_report_dir_exists_is_buffered_into_run_log(
    tmp_path, stub_model, monkeypatch
):
    def fake_refresh(spec, cat):
        print("refresh happened")

    monkeypatch.setattr("ocean_skill.workflows.run._refresh_sources", fake_refresh)
    suite = _model_only_suite(
        tmp_path,
        refresh={
            "catalog": "catalogs/x.yaml",
            "sources": [{"name": "stub", "files": "x/*.nc", "ref": "refs/x.parquet"}],
        },
    )
    path = _write_suite(tmp_path, suite)
    result = run_suite(path)

    assert "refresh happened" in result.log.read_text()


def test_a_fatal_crash_still_leaves_run_log_and_restores_streams(
    tmp_path, stub_model, monkeypatch
):
    """Everything a fatal crash mid-run has to leave behind.

    Two claims, both against the same crash — merged into one run because each used
    to build and throw away an identical crashing suite just to check a different
    part of it. Kept as one function, not one assertion, so a failure still says
    which claim broke.
    """
    from ocean_skill.workflows.report import PdfReport

    def boom(self, fig, stem):
        raise RuntimeError("boom")

    monkeypatch.setattr(PdfReport, "emit", boom)
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    stdout_before, stderr_before = sys.stdout, sys.stderr

    with pytest.raises(RuntimeError, match="boom"):
        run_suite(path)

    # stdout/stderr are never left swapped out, even when the run raises.
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before

    # The crash still leaves a run.log behind, with the traceback the terminal never
    # sees (main prints only a one-line "error: ...").
    report_dirs = list((tmp_path / "out").iterdir())
    assert len(report_dirs) == 1
    log_text = (report_dirs[0] / "run.log").read_text()
    assert "boom" in log_text
    assert "Traceback" in log_text


def test_run_log_has_page_headers_and_timing(tmp_path, stub_model, monkeypatch):
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

    log_text = result.log.read_text()
    assert "== page 1/2: Physics latest ==" in log_text
    assert "== page 2/2: WOA ==" in log_text
    assert "done in" in log_text
    assert "SKIPPED after" in log_text

    page2_idx = log_text.index("== page 2/2")
    assert log_text.index("regrid failed", page2_idx) > page2_idx

    assert result.pages[0].elapsed is not None and result.pages[0].elapsed >= 0
    assert result.pages[1].elapsed is not None and result.pages[1].elapsed >= 0


# ======================================================================================
# ``refs/`` snapshot: each report directory keeps a copy of the reference it was drawn
# from, since the shared reference at ``refresh.sources[].ref`` is rewritten in place
# by every later run's own refresh.
# ======================================================================================


def test_report_dir_gets_a_snapshot_of_the_refreshed_reference(tmp_path, stub_model):
    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    _segment(tmp_path / "out.1.nc", 43200.0, 1.0)
    ref = tmp_path / "refs" / "gom_bgc.json"
    catalog_path = tmp_path / "catalogs" / "x.yaml"
    suite = _model_only_suite(
        tmp_path,
        refresh={
            "catalog": str(catalog_path),
            "sources": [
                {
                    "name": "GOM_bgc",
                    "files": str(tmp_path / "out.*.nc"),
                    "ref": str(ref),
                }
            ],
        },
    )
    path = _write_suite(tmp_path, suite)
    result = run_suite(path)

    snapshot = result.report_dir / "refs" / "gom_bgc.json"
    assert snapshot.exists()
    assert snapshot.read_bytes() == ref.read_bytes()
    assert result.refs == [snapshot]

    manifest = json.loads(result.manifest.read_text())
    assert manifest["refresh"]["catalog"] == str(catalog_path.resolve())
    [record] = manifest["refresh"]["sources"]
    assert record["name"] == "GOM_bgc"
    assert record["ref"] == str(ref.resolve())
    assert record["snapshot"] == str(snapshot)


def test_snapshot_survives_the_next_refresh_overwriting_the_shared_ref(
    tmp_path, stub_model
):
    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    ref = tmp_path / "refs" / "gom_bgc.json"
    catalog_path = tmp_path / "catalogs" / "x.yaml"
    refresh_block = {
        "catalog": str(catalog_path),
        "sources": [
            {
                "name": "GOM_bgc",
                "files": str(tmp_path / "out.*.nc"),
                "ref": str(ref),
            }
        ],
    }
    path = _write_suite(tmp_path, _model_only_suite(tmp_path, refresh=refresh_block))

    result1 = run_suite(path)
    snapshot1 = result1.report_dir / "refs" / "gom_bgc.json"
    bytes1 = snapshot1.read_bytes()

    _segment(tmp_path / "out.1.nc", 43200.0, 1.0)  # the "run" grows
    result2 = run_suite(path)
    snapshot2 = result2.report_dir / "refs" / "gom_bgc.json"

    assert snapshot1.read_bytes() == bytes1, (
        "an earlier report's snapshot must not change"
    )
    assert snapshot2.read_bytes() != bytes1
    ds = xr.open_dataset(
        str(snapshot1), engine="kerchunk", chunks={}, decode_times=False
    )
    assert ds.sizes["ocean_time"] == 2, "the first report still reads only its own data"


def test_a_parquet_reference_directory_is_copied_as_a_tree(tmp_path):
    """A parquet kerchunk target is a directory; the copy must be a full tree.

    Built and compared byte-for-byte without ever opening it as parquet (see the
    module note at the top of tests/test_build_helpers.py: the parquet reference
    reader is intermittently broken upstream).
    """
    import filecmp

    from ocean_skill.config import RefreshConfig
    from ocean_skill.workflows.run import _snapshot_refs

    src = tmp_path / "refs" / "x.parquet"
    (src / "NO3").mkdir(parents=True)
    (src / ".zmetadata").write_text('{"zarr_consolidated_format": 1}')
    (src / "NO3" / "refs.0.parq").write_bytes(b"not-really-parquet-bytes")

    refresh = RefreshConfig.model_validate(
        {
            "catalog": str(tmp_path / "cat.yaml"),
            "sources": [{"name": "x", "files": "*.nc", "ref": str(src)}],
        }
    )
    report_dir = tmp_path / "report"
    report_dir.mkdir()

    records = _snapshot_refs(refresh, report_dir)

    dst = report_dir / "refs" / "x.parquet"
    assert dst.is_dir()
    cmp = filecmp.dircmp(src, dst)
    assert not cmp.diff_files and not cmp.left_only and not cmp.right_only
    for sub in cmp.subdirs.values():
        assert not sub.diff_files and not sub.left_only and not sub.right_only
    assert records == [{"name": "x", "ref": str(src.resolve()), "snapshot": str(dst)}]


def test_a_missing_reference_is_noted_not_fatal(tmp_path, stub_model):
    suite = _model_only_suite(
        tmp_path,
        refresh={
            "catalog": str(tmp_path / "catalogs" / "x.yaml"),
            "sources": [
                {
                    "name": "stub",
                    "files": str(tmp_path / "nothing" / "*.nc"),
                    "ref": str(tmp_path / "refs" / "x.json"),
                }
            ],
        },
    )
    path = _write_suite(tmp_path, suite)
    result = run_suite(path)

    assert not (result.report_dir / "refs").exists()
    assert result.refs == []
    manifest = json.loads(result.manifest.read_text())
    [record] = manifest["refresh"]["sources"]
    assert record["snapshot"] is None
    assert "no reference at" in result.log.read_text()


def test_manifest_refresh_is_null_without_a_refresh_block(tmp_path, stub_model):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result = run_suite(path)

    manifest = json.loads(result.manifest.read_text())
    assert manifest["refresh"] is None
    assert result.refs == []


# ======================================================================================
# ``section:`` divider pages: PDF-only, buffered until a figure follows, and ignored by
# the exit code and the "N page(s) drawn" count (a divider is not a figure).
# ======================================================================================


def _field_page(title="Physics latest"):
    return {
        "title": title,
        "field": {
            "variables": ["temperature", "salinity"],
            "select": {"depth": "surface", "time": "latest"},
        },
    }


def _section_page(title, text=""):
    return {"title": title, "section": text}


def _suite_with_pages(tmp_path, pages, **extra):
    suite = _model_only_suite(tmp_path, **extra)
    suite["pages"] = pages
    return suite


def _fail_titles(monkeypatch, titles):
    """Make ``_pages.build`` raise for pages titled in ``titles``; log the kinds."""
    real_build = _pages.build
    seen = []

    def fake_build(page, *, pooled_records=None):
        seen.append(page.kind)
        if page.title in titles:
            raise RuntimeError("regrid failed")
        return real_build(page, pooled_records=pooled_records)

    monkeypatch.setattr(_pages, "build", fake_build)
    return seen


def test_sections_are_pdf_only_and_settle_to_ok_or_skipped_after_the_run(
    tmp_path, stub_model, monkeypatch
):
    seen = _fail_titles(monkeypatch, set())
    path = _write_suite(
        tmp_path,
        _suite_with_pages(
            tmp_path,
            [
                _section_page("Part one", "Notes for part one."),
                _field_page(),
                _section_page("Nothing follows", "Orphan."),
            ],
        ),
    )
    result = run_suite(path)

    first, field_page, trailing = result.pages
    assert (first.status, first.reason) == ("ok", None)
    assert field_page.status == "ok"
    assert trailing.status == "skipped"
    assert trailing.reason == "no page after this section drew"
    assert result.exit_code == 0
    assert seen == ["field"]  # a section never reaches build()

    # divider + figure; the trailing divider was dropped
    assert _pagecount_by_bytes(result.pdf) == 2
    # PNG numbering is the figures' alone
    assert [p.name for p in result.figures] == ["01_Physics_latest.png"]
    assert [p.name for p in (result.report_dir / "figures").iterdir()] == [
        "01_Physics_latest.png"
    ]

    manifest = json.loads(result.manifest.read_text())
    assert [(p["kind"], p["status"]) for p in manifest["pages"]] == [
        ("section", "ok"),
        ("field", "ok"),
        ("section", "skipped"),
    ]
    assert manifest["pages"][0]["kwargs"] == {"text": "Notes for part one."}
    assert manifest["pages"][2]["reason"] == "no page after this section drew"

    log_text = result.log.read_text()
    assert "== section: Part one ==" in log_text
    assert "== page 2/3: Physics latest ==" in log_text
    assert "== section: Nothing follows ==" in log_text
    assert log_text.count("done in") == 1  # only the one figure page


def test_a_section_whose_pages_all_skipped_leaves_no_orphan_divider(
    tmp_path, stub_model, monkeypatch
):
    _fail_titles(monkeypatch, {"Bad page"})
    path = _write_suite(
        tmp_path,
        _suite_with_pages(
            tmp_path,
            [
                _section_page("Doomed part"),
                _field_page("Bad page"),
                _section_page("Good part"),
                _field_page("Good page"),
            ],
        ),
    )
    result = run_suite(path)

    doomed, bad, good, _ = result.pages
    assert doomed.status == "skipped"
    assert doomed.reason == "no page after this section drew"
    assert bad.status == "skipped"
    assert good.status == "ok"
    assert _pagecount_by_bytes(result.pdf) == 2  # "Good part" divider + "Good page"
    assert result.exit_code == 3  # one data page skipped, one drew


def test_pdf_false_leaves_every_section_skipped_with_its_own_reason(
    tmp_path, stub_model
):
    path = _write_suite(
        tmp_path,
        _suite_with_pages(
            tmp_path,
            [_section_page("Part one", "Notes"), _field_page()],
            pdf=False,
        ),
    )
    result = run_suite(path)

    assert result.pdf is None
    assert len(result.figures) == 1  # the PNG is still written
    section = result.pages[0]
    assert section.status == "skipped"
    assert section.reason == "pdf: false -- section pages appear only in report.pdf"
    assert result.exit_code == 0


def test_a_run_where_nothing_drew_writes_no_pdf_despite_its_sections(
    tmp_path, stub_model, monkeypatch
):
    _fail_titles(monkeypatch, {"Physics latest"})
    path = _write_suite(
        tmp_path,
        _suite_with_pages(
            tmp_path, [_section_page("Part one", "Notes"), _field_page()]
        ),
    )
    result = run_suite(path)

    assert result.pdf is None
    assert not (result.report_dir / "report.pdf").exists()
    assert result.pages[0].status == "skipped"
    assert result.exit_code == 1


@pytest.mark.parametrize(
    ("pages", "failing", "code", "counts"),
    [
        # a drawn data page: 0, whatever the sections around it did
        (["S", "F"], set(), 0, "1 page(s) drawn, 0 skipped"),
        (["S", "F", "S"], set(), 0, "1 page(s) drawn, 0 skipped"),
        # an undrawn data page: 1 -- the divider that *was* written does not rescue it
        (["S", "F"], {"F"}, 1, "0 page(s) drawn, 1 skipped"),
        # mixed data pages: 3
        (["S", "F", "S", "G"], {"G"}, 3, "1 page(s) drawn, 1 skipped"),
        # nothing but dividers: still no report to speak of
        (["S", "S"], set(), 1, "0 page(s) drawn, 0 skipped"),
    ],
    ids=["ok", "ok-trailing", "skipped", "mixed", "sections-only"],
)
def test_exit_code_and_drawn_counts_ignore_section_pages(
    tmp_path, stub_model, monkeypatch, capsys, pages, failing, code, counts
):
    _fail_titles(monkeypatch, failing)
    built = {
        "S": lambda n: _section_page(f"Divider {n}", "Notes"),
        "F": lambda n: _field_page("F"),
        "G": lambda n: _field_page("G"),
    }
    path = _write_suite(
        tmp_path,
        _suite_with_pages(tmp_path, [built[k](i) for i, k in enumerate(pages)]),
    )

    assert main([str(path)]) == code
    assert counts in capsys.readouterr().out


def test_list_only_prints_a_section_line_with_no_cache_note(
    tmp_path, stub_model, capsys
):
    path = _write_suite(
        tmp_path,
        _suite_with_pages(
            tmp_path, [_section_page("Part one", "Notes"), _field_page()]
        ),
    )
    result = run_suite(path, list_only=True)

    lines = capsys.readouterr().out.splitlines()
    assert " 1. [section] Part one" in lines  # the whole line: no cache note, no chain
    assert any(line.startswith(" 2. [field  ] Physics latest") for line in lines)
    assert [p.kind for p in result.pages] == ["section", "field"]
    assert not (tmp_path / "out").exists()  # --list still writes nothing


# -- pdf_images: jpeg ---------------------------------------------------------------


@pytest.fixture
def fake_compress(monkeypatch):
    """Replace ``compress_pdf_images`` with a recorder returning ``fake.result``."""
    calls = []

    def fake(path):
        calls.append(Path(path))
        return fake.result

    fake.calls = calls
    fake.result = (2_000_000, 1_000_000)
    monkeypatch.setattr("ocean_skill.workflows.report.compress_pdf_images", fake)
    return fake


def test_pdf_images_jpeg_compresses_after_the_pdf_closes_and_logs_the_sizes(
    tmp_path, stub_model, fake_compress
):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path, pdf_images="jpeg"))
    result = run_suite(path)

    assert fake_compress.calls == [result.pdf]
    assert (
        "report.pdf: 2.0 MB -> 1.0 MB (JPEG via Ghostscript)" in result.log.read_text()
    )
    manifest = json.loads(result.manifest.read_text())
    assert manifest["pdf_images"] == "jpeg"
    assert manifest["pdf_images_applied"] is True
    assert len(result.figures) == 1  # PNGs are untouched


def test_pdf_images_jpeg_that_fell_back_is_recorded_as_not_applied(
    tmp_path, stub_model, fake_compress
):
    fake_compress.result = None
    path = _write_suite(tmp_path, _model_only_suite(tmp_path, pdf_images="jpeg"))
    result = run_suite(path)

    assert result.pdf is not None and result.pdf.exists()  # the lossless PDF stays
    assert "JPEG via Ghostscript" not in result.log.read_text()
    manifest = json.loads(result.manifest.read_text())
    assert manifest["pdf_images"] == "jpeg"
    assert manifest["pdf_images_applied"] is False


def test_pdf_images_defaults_to_lossless_and_never_calls_ghostscript(
    tmp_path, stub_model, fake_compress
):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path))
    result = run_suite(path)

    assert fake_compress.calls == []
    manifest = json.loads(result.manifest.read_text())
    assert manifest["pdf_images"] == "lossless"
    assert manifest["pdf_images_applied"] is False


def test_pdf_images_jpeg_is_skipped_without_a_pdf_to_compress(
    tmp_path, stub_model, monkeypatch, fake_compress
):
    # pdf: false -- there is no report.pdf
    path = _write_suite(
        tmp_path, _model_only_suite(tmp_path, pdf_images="jpeg", pdf=False)
    )
    result = run_suite(path)
    assert fake_compress.calls == []
    assert json.loads(result.manifest.read_text())["pdf_images_applied"] is False

    # pdf: true, but no page drew -- PdfPages never wrote a file
    _fail_titles(monkeypatch, {"Physics latest"})
    path = _write_suite(tmp_path, _model_only_suite(tmp_path, pdf_images="jpeg"))
    result = run_suite(path)
    assert result.pdf is None
    assert fake_compress.calls == []


def test_pdf_images_jpeg_without_ghostscript_warns_and_keeps_the_lossless_pdf(
    tmp_path, stub_model, monkeypatch
):
    from ocean_skill.workflows import report as _report

    monkeypatch.setattr(_report.shutil, "which", lambda name: None)
    path = _write_suite(tmp_path, _model_only_suite(tmp_path, pdf_images="jpeg"))
    with pytest.warns(UserWarning, match="Ghostscript"):
        result = run_suite(path)

    assert result.exit_code == 0
    assert result.pdf is not None and result.pdf.exists()
    assert json.loads(result.manifest.read_text())["pdf_images_applied"] is False


def test_main_refuses_an_unknown_pdf_images_value(tmp_path):
    path = _write_suite(tmp_path, _model_only_suite(tmp_path, pdf_images="png"))
    assert main([str(path)]) == 2
