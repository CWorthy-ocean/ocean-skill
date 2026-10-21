"""Test mode (``ocean-skill-run --smoke``) beyond the time axis: dates, grid, dpi.

Test mode thins a page's ``times:`` list and month fan to first and last, strides the
truncated model grid (u/v staying one less than rho), and caps the figure dpi. None of
it may touch a real run, so these compare against the same call without test mode.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import comparison as _comparison
from ocean_skill import roms
from ocean_skill import sources as _sources
from ocean_skill.config import SuiteConfig
from ocean_skill.suites import pages as P
from ocean_skill.suites import report as _report
from ocean_skill.suites import run as _run
from ocean_skill.suites.run import main, run_suite
from tests.test_suites_run_guardrails import (
    _smoke_suite,
    _write_suite,
    fake_sources,  # noqa: F401  (used as a fixture)
    smoke_code,  # noqa: F401  (used as a fixture)
)

# -- thinning the dates of a page -----------------------------------------------------

INDEX = pd.date_range("2010-01-05", periods=10, freq="7D")  # Jan 5 .. Mar 9, 2010

SEASON_DAYS = [
    f"{year}-{md}"
    for year in (2010, 2011, 2012, 2013)
    for md in ("01-31", "04-30", "07-31", "10-31")
]
KEPT = ["2010-01-31", "2013-10-31"]


@pytest.fixture(autouse=True)
def _time_index(monkeypatch, request):
    """Fix the time axis for the page tests; a run on ``fake_sources`` reads its own."""
    if "fake_sources" not in request.fixturenames:
        monkeypatch.setattr(
            "ocean_skill.extrema._native_time_index", lambda source: INDEX
        )


def _suite(pages, **top):
    top.setdefault("defaults", {"test": "stub", "season_days": SEASON_DAYS})
    return SuiteConfig.model_validate({"name": "t", "pages": pages, **top})


def _seasons_page(**compare):
    return {
        "title": "Seasons",
        "compare": {
            "reference": ["ref"],
            "variables": ["temperature"],
            "times": "{season_days}",
            **compare,
        },
    }


def test_a_times_list_keeps_first_and_last_and_is_pinned_after_thinning():
    suite = _suite([_seasons_page()])
    [page] = P.expand(suite, thin_dates=True)
    assert page.kwargs["times"] == KEPT
    assert page.thinned == "test: 2 of 16 dates"
    assert suite.defaults["season_days"] == SEASON_DAYS  # shared list is not edited
    assert len(P.time_pins(P.expand(suite)).instants) == 16
    assert P.time_pins([page]).instants == tuple(pd.Timestamp(d) for d in KEPT)


def test_nothing_changes_without_test_mode_and_min_pairs_drops_only_in_it():
    suite = _suite([_seasons_page(), {"title": "s", "section": "text"}])
    plain = P.expand(suite)
    assert [p.as_dict() for p in plain] == [
        p.as_dict() for p in P.expand(suite, thin_dates=False)
    ]
    assert plain[0].kwargs["times"] == SEASON_DAYS
    assert all(p.thinned is None for p in plain)
    assert "min_pairs" not in plain[0].kwargs

    [thin] = P.expand(_suite([_seasons_page()]), thin_dates=True)
    assert thin.kwargs["min_pairs"] == P.TEST_MIN_PAIRS
    [own] = P.expand(_suite([_seasons_page(min_pairs=7)]), thin_dates=True)
    assert own.kwargs["min_pairs"] == 7  # the page's own value wins


@pytest.mark.parametrize(
    "times",
    [
        {"resample": "1MS", "reduce": "mean"},  # bins whatever is kept
        ["2010-01", "2010-07"],  # two dates: nothing to cut
        None,
    ],
)
def test_a_times_value_that_is_not_a_long_list_is_left_alone(times):
    compare = {} if times is None else {"times": times}
    page = {
        "title": "p",
        "compare": {"reference": ["ref"], "variables": ["t"], **compare},
    }
    [plain] = P.expand(_suite([page]))
    [thin] = P.expand(_suite([page]), thin_dates=True)
    assert {k: v for k, v in thin.kwargs.items() if k != "min_pairs"} == plain.kwargs
    assert thin.thinned is None


def test_other_fans_stay_complete_and_only_the_months_are_cut():
    suite = _suite(
        [
            {
                "title": "{variable} {month.name}",
                "for_each": {"month": "run", "variable": ["a", "b", "c"]},
                "compare": {
                    "reference": ["ref"],
                    "variables": ["{variable}"],
                    "select": {"test": {"time": "{month.window}"}, "reference": {}},
                },
            }
        ]
    )
    thin = P.expand(suite, thin_dates=True)
    assert len(P.expand(suite)) == 9  # three months x three variables
    assert len(thin) == 6  # January and March x the same three variables
    assert {p.title.split()[0] for p in thin} == {"a", "b", "c"}
    assert {p.title.split()[1] for p in thin} == {"January", "March"}
    assert all(p.thinned == "test: 2 of 3 months" for p in thin)


def test_contours_override_times_are_cut_like_the_fill():
    page = _seasons_page()
    page["contours"] = {"variables": ["salinity"]}
    [out] = P.expand(_suite([page]), thin_dates=True)
    assert out.contours["times"] == out.kwargs["times"] == KEPT


@pytest.mark.usefixtures("fake_sources")
def test_thinned_pages_are_marked_and_counted_in_the_page_list(tmp_path, capsys):
    suite = _smoke_suite(tmp_path)
    suite["defaults"]["season_days"] = SEASON_DAYS
    suite["pages"] = [_seasons_page()]
    path = _write_suite(tmp_path, suite)
    result = run_suite(path, refresh=False, list_only=True, smoke=2)
    out = capsys.readouterr().out
    assert "(test: 2 of 16 dates)" in out and "on 1 page(s)" in out
    assert result.pages[0].thinned == "test: 2 of 16 dates"
    run_suite(path, refresh=False, list_only=True)
    assert "of 16 dates" not in capsys.readouterr().out


# -- striding the model grid ----------------------------------------------------------

NY, NX = 10, 14


def _roms_like() -> xr.Dataset:
    """Return a standardized-ROMS look-alike (rho data, staggered u/v, grid coords)."""
    rng = np.random.default_rng(1)
    time = pd.date_range("2020-01-01", periods=6, freq="D")
    eta, xi = np.arange(NY), np.arange(NX)
    return xr.Dataset(
        {
            "temp": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                rng.random((6, 3, NY, NX)),
            ),
            "u": (("time", "s_rho", "eta_rho", "xi_u"), rng.random((6, 3, NY, NX - 1))),
            "v": (("time", "s_rho", "eta_v", "xi_rho"), rng.random((6, 3, NY - 1, NX))),
            "w": (("time", "s_w", "eta_rho", "xi_rho"), rng.random((6, 4, NY, NX))),
        },
        coords={
            "time": time,
            "lon": (("eta_rho", "xi_rho"), np.add.outer(0 * eta, xi * 1.0)),
            "lat": (("eta_rho", "xi_rho"), np.add.outer(eta * 1.0, 0 * xi)),
            "cell_area": (("eta_rho", "xi_rho"), np.ones((NY, NX))),
        },
    )


@pytest.fixture
def roms_sources(monkeypatch):
    datasets = {"m": _roms_like(), "other": _roms_like()}
    monkeypatch.setattr(
        _sources,
        "resolve",
        lambda name: SimpleNamespace(name=name, metadata={}, path=None),
    )
    monkeypatch.setattr(
        _sources, "_read_uncached", lambda ref, meta, qc, kwargs: datasets[ref.name]
    )
    return datasets


@pytest.mark.parametrize("stride", [2, 3, 4])
def test_stride_keeps_staggered_sizes_one_less_than_rho(roms_sources, stride):
    with _sources.truncate_time(["m"], 2, stride=stride):
        ds = osk.read("m")
    assert ds.sizes["xi_rho"] == len(range(0, NX, stride))
    assert ds.sizes["eta_rho"] == len(range(0, NY, stride))
    assert ds.sizes["xi_u"] == ds.sizes["xi_rho"] - 1
    assert ds.sizes["eta_v"] == ds.sizes["eta_rho"] - 1
    assert ds.sizes["s_rho"] == 3 and ds.sizes["s_w"] == 4  # the vertical is whole
    assert ds.sizes["time"] == 4  # the time cut still applies
    # the velocity code still averages the strided u/v onto rho
    u_rho = roms._average_to_rho(ds["u"], "xi_u", "xi_rho")
    v_rho = roms._average_to_rho(ds["v"], "eta_v", "eta_rho")
    assert u_rho.sizes["xi_rho"] == ds.sizes["xi_rho"]
    assert v_rho.sizes["eta_rho"] == ds.sizes["eta_rho"]


def test_stride_takes_the_data_and_grid_coordinates_together(roms_sources):
    full = roms_sources["m"]
    with _sources.truncate_time(["m"], 2, stride=4):
        ds = osk.read("m")
    np.testing.assert_array_equal(ds["lon"].values, full["lon"].values[::4, ::4])
    np.testing.assert_array_equal(ds["lat"].values, full["lat"].values[::4, ::4])
    assert ds["cell_area"].shape == (3, 4)
    t0 = {n: full[n].isel(time=0).values[:, ::4, ::4] for n in ("temp", "u", "v")}
    np.testing.assert_array_equal(ds["temp"].isel(time=0).values, t0["temp"])
    # a staggered point keeps the position it had in the full grid
    np.testing.assert_array_equal(ds["u"].isel(time=0).values, t0["u"][..., :3])
    np.testing.assert_array_equal(ds["v"].isel(time=0).values, t0["v"][:, :2])


def test_stride_one_unnamed_sources_and_each_stride_are_kept_apart(roms_sources):
    with _sources.truncate_time(["m"], 2):
        whole = osk.read("m")
    with _sources.truncate_time(["m"], 2, stride=1):
        xr.testing.assert_identical(whole, osk.read("m"))
    assert whole.sizes["xi_rho"] == NX
    with _sources.truncate_time(["m"], 2, stride=4):
        assert osk.read("other").sizes["xi_rho"] == NX  # not named: untouched
    with _sources.truncate_time(["m"], 2, stride=2):
        assert osk.read("m").sizes["xi_rho"] == 7
    with _sources.truncate_time(["m"], 2, stride=4):
        assert osk.read("m").sizes["xi_rho"] == 4  # memoized apart from stride 2
    assert osk.read("m").sizes["xi_rho"] == NX


def test_a_stride_that_leaves_too_few_cells_keeps_that_dimension_whole(roms_sources):
    with _sources.truncate_time(["m"], 2, stride=6):  # 3 rows would be 2 cells
        with pytest.warns(UserWarning, match="keeping that dimension whole"):
            ds = osk.read("m")
    assert ds.sizes["eta_rho"] == NY and ds.sizes["eta_v"] == NY - 1
    assert ds.sizes["xi_rho"] == 3 and ds.sizes["xi_u"] == 2  # this one still strides


def test_a_nonpositive_stride_is_refused(tmp_path):
    with pytest.raises(ValueError, match="stride"):
        with _sources.truncate_time(["m"], 2, stride=0):
            pass
    path = _write_suite(tmp_path, _smoke_suite(tmp_path))
    with pytest.raises(ValueError, match="stride must be at least 1"):
        run_suite(path, smoke=2, smoke_stride=0, list_only=True)


# -- the run: cache identity, options, dpi --------------------------------------------


def test_the_smoke_cache_key_differs_by_stride_and_stride_one_changes_nothing(
    monkeypatch,
):
    monkeypatch.setattr(_run, "_code_identity", lambda: "code-1")
    stamps = [("run", 3, "a", "b", 6, "d")]
    base, _ = _run._smoke_cache_key(stamps, [])
    assert _run._smoke_cache_key(stamps, [], 1)[0] == base  # as before the option
    keys = {base, _run._smoke_cache_key(stamps, [], 4)[0]}
    assert len(keys | {_run._smoke_cache_key(stamps, [], 2)[0]}) == 3


@pytest.mark.usefixtures("fake_sources")
def test_a_different_stride_gets_a_new_smoke_cache(tmp_path, smoke_code, capsys):  # noqa: F811
    def run(stride):
        result = run_suite(
            _write_suite(tmp_path, _smoke_suite(tmp_path)),
            refresh=False,
            smoke=2,
            smoke_stride=stride,
        )
        return json.loads(result.manifest.read_text())["modes"]

    four, two = run(4), run(2)
    assert four["smoke_stride"] == 4 and two["smoke_stride"] == 2
    assert four["smoke_cache"] != two["smoke_cache"]
    assert run(4)["smoke_cache_reused"] is True
    out = capsys.readouterr().out
    assert "test mode: model grid every 4th cell (stride 4)" in out
    assert f"figures at most {_report.SMOKE_DPI} dpi" in out


def test_smoke_stride_is_a_command_line_option(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(
        _run, "run_suite", lambda path, **kw: got.update(kw) or SimpleNamespace()
    )
    main([str(tmp_path / "s.yaml"), "--list"])
    assert got["smoke_stride"] == _run.SMOKE_STRIDE == 4
    main([str(tmp_path / "s.yaml"), "--smoke", "--smoke-stride", "2", "--list"])
    assert got["smoke"] == 2 and got["smoke_stride"] == 2


def _captured_report_args(tmp_path, monkeypatch, **kwargs):
    seen = {}
    real = _report.PdfReport

    def spy(*args, **kw):
        seen.update(kw)
        return real(*args, **kw)

    da = xr.DataArray(
        np.full((2, 2), 5.0),
        dims=("lat", "lon"),
        coords={"lat": [10.0, 20.0], "lon": [-100.0, -90.0]},
        attrs={"units": "degC"},
    )
    monkeypatch.setattr(_report, "PdfReport", spy)
    monkeypatch.setattr(_comparison, "prepare_source", lambda *a, **k: (da, None))
    tmp_path.mkdir(parents=True, exist_ok=True)
    run_suite(_write_suite(tmp_path, _smoke_suite(tmp_path)), refresh=False, **kwargs)
    return seen


@pytest.mark.usefixtures("fake_sources")
def test_test_mode_caps_the_figure_dpi_and_a_real_run_does_not(tmp_path, monkeypatch):
    smoke = _captured_report_args(tmp_path / "a", monkeypatch, smoke=2)
    assert smoke == {"png_dpi": _report.SMOKE_DPI, "pdf_dpi": _report.SMOKE_DPI}
    real = _captured_report_args(tmp_path / "b", monkeypatch)
    assert real == {"png_dpi": _report.PNG_DPI, "pdf_dpi": _report.PDF_DPI}


def test_a_lower_png_dpi_makes_a_smaller_png_of_the_same_figure(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from PIL import Image

    sizes = {}
    for dpi in (200, 100):
        fig = Figure(figsize=(4, 3))
        fig.add_subplot().plot([0, 1])
        with _report.PdfReport(None, tmp_path / str(dpi), png_dpi=dpi) as report:
            png = report.emit(fig, "f")
        with Image.open(png) as img:
            sizes[dpi] = img.size
    assert sizes[200][0] == pytest.approx(2 * sizes[100][0], abs=2)
