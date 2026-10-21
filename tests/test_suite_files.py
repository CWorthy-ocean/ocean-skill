"""Every suite template the package ships still loads and expands.

A suite is data, and nothing else exercises the real files: the other workflow tests
build their suites inline. This parametrizes over the four shipped suites under
``ocean_skill/suites/templates/`` (found from this file's own location, so it works
from any working directory) and, for each, does exactly what
:func:`ocean_skill.suites.run.run_suite` does before it touches any data --
``yaml.safe_load`` then ``SuiteConfig.model_validate`` (schema errors, e.g. a
``section:`` carrying a ``plot:``, surface here) -- followed by
:func:`ocean_skill.suites.pages.expand` (unresolvable placeholders, a ``then:``
chain out of order, a depth on a surface-only variable, ...).

No refresh, no catalog registration, no cache, no network. ``expand`` reads only one
thing: the test source's native time axis (``extrema._native_time_index``), used to
resolve ``time: latest``/``month: run`` and to bound the injected windows. That is
monkeypatched, as ``test_suites_pages.py`` does, to a stand-in index of four
month-end restart snapshots -- July to October 2010, 23:45 each -- for any source
name. Nothing else in ``expand`` reads data: ``_is_closed`` and
``_resolve_series_window`` work from that same index alone, so they need no patch.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from ocean_skill.config import SuiteConfig
from ocean_skill.suites import pages as P

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "ocean_skill/suites/templates"
SUITE_FILES = sorted(TEMPLATES_DIR.glob("*.yaml"))

#: Four month-end restart snapshots, standing in for any suite's test source.
RESTART_INDEX = pd.DatetimeIndex(
    [
        "2010-07-31 23:45",
        "2010-08-31 23:45",
        "2010-09-30 23:45",
        "2010-10-31 23:45",
    ]
)
RUN_WINDOW = {"min": "2010-07-31T23:45:00", "max": "2010-10-31T23:45:00"}


@pytest.fixture(autouse=True)
def _restart_index(monkeypatch):
    monkeypatch.setattr(
        "ocean_skill.extrema._native_time_index", lambda source: RESTART_INDEX
    )


def _expand(path: Path) -> tuple[SuiteConfig, list[P.ExpandedPage]]:
    """Load ``path`` the way ``run_suite`` does (parse + validate), then expand it."""
    suite = SuiteConfig.model_validate(yaml.safe_load(path.read_text()))
    return suite, P.expand(suite)


def test_suites_directory_is_found():
    # Guards the parametrization below: an empty glob would pass vacuously.
    assert SUITE_FILES, f"no suite YAML found under {TEMPLATES_DIR}"
    assert {p.name for p in TEMPLATES_DIR.glob("*.yaml")} == {
        "mid-run-check.yaml",
        "final-report.yaml",
        "model-comparison.yaml",
        "input-check.yaml",
    }


@pytest.mark.parametrize("path", SUITE_FILES, ids=lambda p: p.name)
def test_shipped_suite_loads_and_expands(path):
    _, expanded = _expand(path)
    assert expanded, f"{path.name} expanded to no pages"
    assert all(p.title for p in expanded)
    assert {p.kind for p in expanded} <= {
        "field",
        "compare",
        "summary",
        "section",
        "XY",
        "TS",
        "health",
        "rivers",
        "inputs",
    }
    # A suite of dividers alone would draw nothing at all.
    assert any(p.kind != "section" for p in expanded)


# -- the final-report and mid-run-check templates: grammar the pages rely on ----------


@pytest.fixture
def final_report():
    return _expand(TEMPLATES_DIR / "final-report.yaml")


@pytest.fixture
def mid_run_check():
    return _expand(TEMPLATES_DIR / "mid-run-check.yaml")


def _by_title(expanded, title):
    (page,) = [p for p in expanded if p.title == title]
    return page


@pytest.mark.parametrize("name", ["mid-run-check.yaml", "final-report.yaml"])
def test_section_pages_carry_only_their_text(name):
    _, expanded = _expand(TEMPLATES_DIR / name)
    sections = [p for p in expanded if p.kind == "section"]
    assert sections
    for page in sections:
        assert set(page.kwargs) <= {"text"}, page.title
        assert page.plot == {} and page.steps == []


def test_mid_run_check_domain_mean_keeps_its_depth_list_and_aggregate(mid_run_check):
    suite, expanded = mid_run_check
    page = _by_title(expanded, "Domain-mean time series")
    assert page.kwargs["source"] == suite.defaults["test"]
    assert page.kwargs["select"]["depth"] == ["surface", 100, 200]
    assert page.kwargs["select"]["time"] == suite.defaults["period"]
    assert page.kwargs["aggregate"] == {"lon": "mean", "lat": "mean"}
    assert page.plot["rows"] == "variable"
    assert page.plot["encode"] == {"color": "depth"}
    assert page.kwargs["variable"][-1]["sum"] == ["spChl", "diatChl", "diazChl"]


@pytest.mark.parametrize("kind", ["min", "max"])
def test_mid_run_check_extremum_pages_chain_extremum_then_series(mid_run_check, kind):
    suite, expanded = mid_run_check
    name = {"min": "minimum", "max": "maximum"}[kind]
    page = _by_title(expanded, f"Surface alkalinity {name} — time series")
    assert page.kwargs["select"] == {
        "depth": "surface",
        "time": suite.defaults["period"],
    }
    assert [s["name"] for s in page.steps] == ["extremum", "series"]
    assert page.steps[0]["kwargs"] == {"kind": kind}
    series = page.steps[1]["kwargs"]
    assert series["variables"] == [
        "dissolved_inorganic_carbon",
        "temperature",
        "salinity",
        "nitrate",
    ]
    assert series["time"] == suite.defaults["period"]


def test_final_report_ts_page_pins_roms_and_glorys_to_the_run(final_report):
    suite, expanded = final_report
    page = _by_title(expanded, "T-S diagrams — six Pacific regions")
    assert page.kind == "TS"
    members = page.kwargs["members"]
    assert list(members) == ["ROMS", "WOA23", "GLORYS12"]

    roms = members["ROMS"]
    assert roms["source"] == suite.defaults["test"]
    assert roms["variable"] == ["temperature", "salinity"]
    assert roms["select"] == {"time": RUN_WINDOW}
    assert roms["cache"] is True

    # WOA23 is a climatology: no run window, no select at all. GLORYS12 reanalysis
    # takes the run's window so it covers the dots' dates. All cache with the suite.
    woa, glorys = members["WOA23"], members["GLORYS12"]
    assert woa["source"] == ["woa23_temperature_annual", "woa23_salinity_annual"]
    assert "select" not in woa
    assert glorys["source"] == "glorys_my_daily_timeseries"
    assert glorys["select"] == {"time": RUN_WINDOW}
    assert "window" not in glorys
    assert glorys["aggregate"] == {"time": "mean", "lon": "mean", "lat": "mean"}
    assert woa["cache"] is True and glorys["cache"] is True
    assert page.cache is True


def test_final_report_ts_page_has_boxed_regions_with_matching_annotations(
    final_report,
):
    _, expanded = final_report
    page = _by_title(expanded, "T-S diagrams — six Pacific regions")

    regions = page.kwargs["regions"]
    assert len(regions) == 6
    for box in regions.values():
        assert set(box) == {"lon", "lat"}
        assert 0 <= box["lon"]["min"] < box["lon"]["max"] <= 360
        assert -90 <= box["lat"]["min"] < box["lat"]["max"] <= 90

    # WOA23's 1-degree grid has no cell centre in two of the boxes
    assert page.kwargs["at_center"] == ["WOA23"]

    assert page.plot["ncols"] == 3
    assert page.plot["colors"] == {
        "ROMS": "black",
        "WOA23": "tab:red",
        "GLORYS12": "tab:blue",
    }
    annotations = page.plot["annotations"]
    assert list(annotations) == list(regions)  # every panel is annotated
    for texts in annotations.values():
        for s_pos, t_pos in texts.values():
            assert 32 < s_pos < 37 and -2 < t_pos < 30  # (S, T) in plausible ranges
    assert "Northern\nsurface\nwaters" in annotations["Subpolar Gyre"]
    assert annotations["Peru Current"]["ESSW"] == [34.95, 10.0]


@pytest.mark.parametrize(
    "title,reference,lon",
    [
        ("Zonal velocity — 180-160W", "cravatte_u_179e_160w", {"min": 180, "max": 200}),
        ("Zonal velocity — 160-120W", "cravatte_u_160w_120w", {"min": 200, "max": 240}),
    ],
)
def test_final_report_zonal_current_pages_are_lon_mean_slabs_with_zero_contours(
    final_report, title, reference, lon
):
    """Only the model is box-averaged (the obs are a band mean already)."""
    _, expanded = final_report
    page = _by_title(expanded, title)
    assert page.kwargs["reference"] == [reference]
    assert page.kwargs["variables"] == ["east_velocity"]
    assert page.kwargs["select"]["lon"] == lon
    assert page.kwargs["aggregate"] == {
        "test": {"time": "mean", "lon": "mean"},
        "reference": {"lon": "mean"},
    }
    assert page.contours is True
    assert page.plot["contour_levels"] == [0]


# -- input-check and mid-run-check: plain words, boundary titles, optional pages -------


def _page_texts(path: Path) -> list[str]:
    raw = yaml.safe_load(path.read_text())["pages"]
    return [t for page in raw for t in (page.get("section"), page.get("caption")) if t]


@pytest.mark.parametrize("name", ["mid-run-check.yaml", "input-check.yaml"])
def test_section_texts_and_captions_use_no_yaml_jargon(name):
    for text in _page_texts(TEMPLATES_DIR / name):
        assert "defaults." not in text, text


def test_the_health_text_says_alternating_shading_marks_the_jobs():
    path = TEMPLATES_DIR / "mid-run-check.yaml"
    health = [t for t in _page_texts(path) if "shading" in t]
    assert health and all(
        "alternating shading marks the run's jobs" in t.lower() for t in health
    )


def test_input_check_expands_with_its_boundary_climatology_and_restoring_pages():
    _, expanded = _expand(TEMPLATES_DIR / "input-check.yaml")
    titles = [p.title for p in expanded]
    clim = [t for t in titles if "iron, ammonium, DOC" in t]
    assert [t.split(" boundary")[0] for t in clim] == ["South", "West"]
    assert any(t.startswith("Salinity restoring vs WOA23") for t in titles)
    assert not [p for p in expanded if p.problem]
    # every boundary figure names its side
    for p in expanded:
        if p.kind in ("compare", "field") and " boundary — " in p.title:
            assert p.plot["title"].startswith(
                p.title.split(" boundary")[0] + " boundary"
            )
            # the location is in the caption (a full title runs off the page)
            assert "(lon " not in p.plot["title"], p.plot["title"]
    section = next(p for p in expanded if p.title == "Boundary conditions")
    text = section.kwargs["text"]
    assert "2011-01" in text and "{" not in text
