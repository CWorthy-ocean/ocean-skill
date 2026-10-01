"""Every suite YAML shipped under ``suites/`` still loads and expands.

A suite is data, and nothing else exercises the real files: the other workflow tests
build their suites inline. This parametrizes over ``suites/*.yaml`` (found from this
file's own location, so it works from any working directory) and, for each, does
exactly what :func:`ocean_skill.workflows.run.run_suite` does before it touches any
data -- ``yaml.safe_load`` then ``SuiteConfig.model_validate`` (schema errors, e.g. a
``section:`` carrying a ``plot:``, surface here) -- followed by
:func:`ocean_skill.workflows.pages.expand` (unresolvable placeholders, a ``then:``
chain out of order, a depth on a surface-only variable, ...).

No refresh, no catalog registration, no cache, no network. ``expand`` reads only one
thing: the test source's native time axis (``extrema._native_time_index``), used to
resolve ``time: latest``/``month: run`` and to bound the injected windows. That is
monkeypatched, as ``test_workflows_pages.py`` does, to a stand-in index of four
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
from ocean_skill.workflows import pages as P

SUITES_DIR = Path(__file__).resolve().parents[1] / "suites"
SUITE_FILES = sorted(SUITES_DIR.glob("*.yaml"))

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
    assert SUITE_FILES, f"no suites/*.yaml found under {SUITES_DIR}"


#: Shipped suites that do not expand at HEAD, and why. ``strict`` xfail, so the moment
#: one is fixed this flags the stale entry here instead of silently staying green.
_KNOWN_BROKEN = {
    name: (
        "its 'Monthly means -- {variable}' page fans PH and pCO2 (surface-only "
        "variables) through select: {depth: '{depths}'}, which expand() refuses: "
        "move them to a surface-only page"
    )
    for name in ("roms_marbl_diagnostic.yaml", "roms_marbl_quick.yaml")
}


def _suite_params():
    for path in SUITE_FILES:
        marks = []
        if path.name in _KNOWN_BROKEN:
            marks.append(
                pytest.mark.xfail(
                    strict=True, raises=ValueError, reason=_KNOWN_BROKEN[path.name]
                )
            )
        yield pytest.param(path, id=path.name, marks=marks)


@pytest.mark.parametrize("path", _suite_params())
def test_shipped_suite_loads_and_expands(path):
    _, expanded = _expand(path)
    assert expanded, f"{path.name} expanded to no pages"
    assert all(p.title for p in expanded)
    assert {p.kind for p in expanded} <= {"field", "compare", "summary", "section"}
    # A suite of dividers alone would draw nothing at all.
    assert any(p.kind != "section" for p in expanded)


# -- suites/pacmed_review.yaml: the exact expanded sequence ---------------------------

_SNAPSHOT_VARIABLES = [
    "temperature",
    "salinity",
    "nitrate",
    "phosphate",
    "oxygen",
    "dissolved_inorganic_carbon",
    "alkalinity",
]
_MONTHS = ["July", "August", "September", "October"]

EXPECTED_PACMED = (
    [
        ("section", "Domain-wide trends"),
        ("field", "Domain-mean time series"),
        ("section", "Extreme values"),
        ("field", "Alkalinity minimum — time series"),
        ("field", "Alkalinity maximum — time series"),
        ("section", "Model snapshots — surface"),
    ]
    + [("field", f"Snapshot — {v} (surface)") for v in _SNAPSHOT_VARIABLES]
    + [("section", "Model snapshots — 100 m")]
    + [("field", f"Snapshot — {v} (100 m)") for v in _SNAPSHOT_VARIABLES]
    + [("section", "Model snapshots — 200 m")]
    + [("field", f"Snapshot — {v} (200 m)") for v in _SNAPSHOT_VARIABLES]
    + [
        ("section", "Chlorophyll and mixed layer depth vs observations"),
        ("compare", "Chlorophyll vs satellite — each snapshot"),
    ]
    + [("compare", f"Mixed layer depth vs Holte & Talley — {m} 2010") for m in _MONTHS]
    + [("section", "Nutrients vs WOA23")]
    + [
        ("compare", f"Nutrients vs WOA23 — {m} 2010 ({depth})")
        for m in _MONTHS
        for depth in ("surface", "100 m")
    ]
    + [
        ("section", "Carbon vs GLODAPv2"),
        ("compare", "Alkalinity & DIC vs GLODAPv2 — run mean (surface)"),
        ("compare", "Alkalinity & DIC vs GLODAPv2 — run mean (100 m)"),
    ]
)


@pytest.fixture
def pacmed():
    return _expand(SUITES_DIR / "pacmed_review.yaml")


def _by_title(expanded, title):
    (page,) = [p for p in expanded if p.title == title]
    return page


def test_pacmed_review_settings(pacmed):
    suite, _ = pacmed
    assert suite.name == "pacmed-review"
    assert suite.pdf_images == "jpeg"
    assert suite.defaults["test"] == "all_the_rest"
    assert suite.defaults["depths"] == ["surface", 100]


def test_pacmed_review_expands_to_the_exact_page_sequence(pacmed):
    _, expanded = pacmed
    assert len(expanded) == 47
    assert sum(p.kind == "section" for p in expanded) == 8
    assert [(p.kind, p.title) for p in expanded] == EXPECTED_PACMED


def test_pacmed_review_section_pages_carry_text_and_nothing_else(pacmed):
    _, expanded = pacmed
    for page in (p for p in expanded if p.kind == "section"):
        assert page.kwargs["text"].strip(), page.title
        assert page.plot == {} and page.steps == []


def test_pacmed_review_domain_mean_keeps_its_depth_list_and_aggregate(pacmed):
    _, expanded = pacmed
    page = _by_title(expanded, "Domain-mean time series")
    assert page.kwargs["select"]["depth"] == ["surface", 100, 200]
    assert page.kwargs["select"]["time"] == RUN_WINDOW
    assert page.kwargs["aggregate"] == {"lon": "mean", "lat": "mean"}
    assert page.plot["rows"] == "variable"
    assert page.plot["encode"] == {"color": "depth"}
    assert len(page.kwargs["variable"]) == 6
    assert page.kwargs["variable"][-1]["sum"] == ["spChl", "diatChl", "diazChl"]


@pytest.mark.parametrize("kind", ["min", "max"])
def test_pacmed_review_extremum_pages_chain_extremum_then_series(pacmed, kind):
    _, expanded = pacmed
    name = {"min": "minimum", "max": "maximum"}[kind]
    page = _by_title(expanded, f"Alkalinity {name} — time series")
    assert page.kwargs["select"] == {"depth": "surface", "time": "2010-10-31T23:45:00"}
    assert [s["name"] for s in page.steps] == ["extremum", "series"]
    assert page.steps[0]["kwargs"] == {"kind": kind}
    series = page.steps[1]["kwargs"]
    assert series["variables"] == [
        "dissolved_inorganic_carbon",
        "temperature",
        "salinity",
        "nitrate",
    ]
    assert "time" in series  # the fixed-snapshot pad window was resolved to a literal


def test_pacmed_review_snapshot_pages_share_variables_and_plot(pacmed):
    _, expanded = pacmed
    for depth_label, depth in (("surface", "surface"), ("100 m", 100), ("200 m", 200)):
        for v in _SNAPSHOT_VARIABLES:
            page = _by_title(expanded, f"Snapshot — {v} ({depth_label})")
            assert page.kwargs["variable"] == [v]
            assert page.kwargs["select"]["depth"] == depth
            assert page.kwargs["select"]["time"] == RUN_WINDOW
            assert page.plot["shared_limits"] is True
            assert page.plot["robust"] is True
            assert page.plot["colorbar_label_clipped"] is True


def test_pacmed_review_chlorophyll_page_keeps_times_and_gets_the_run_window(pacmed):
    _, expanded = pacmed
    page = _by_title(expanded, "Chlorophyll vs satellite — each snapshot")
    assert page.kwargs["reference"] == "chl_gapfree_my_daily_geo"
    assert page.kwargs["times"] == {"resample": "1D", "reduce": "mean"}
    assert page.kwargs["select"] == {
        "test": {"depth": "surface", "time": RUN_WINDOW},
        "reference": {},
    }
    # times= fans the page into per-bin comparisons, so it is never cacheable.
    assert page.cache is False


def test_pacmed_review_mld_pages_select_the_calendar_month_as_an_int(pacmed):
    _, expanded = pacmed
    for month, name in zip((7, 8, 9, 10), _MONTHS, strict=True):
        page = _by_title(expanded, f"Mixed layer depth vs Holte & Talley — {name} 2010")
        reference = page.kwargs["select"]["reference"]
        assert reference == {"month": month}
        assert type(reference["month"]) is int  # not the string "7"
        assert page.kwargs["select"]["test"]["time"] == {
            "min": f"2010-{month:02d}-01T00:00:00",
            "max": f"2010-{month:02d}-{31 if month in (7, 8, 10) else 30}T23:59:59",
        }
        assert page.kwargs["aggregate"] == {"test": {"time": "mean"}, "reference": {}}


def test_pacmed_review_woa_pages_are_month_major_with_month_matched_references(
    pacmed,
):
    _, expanded = pacmed
    woa = [p for p in expanded if p.title.startswith("Nutrients vs WOA23 — ")]
    assert len(woa) == 8
    expected = [
        (month, name, depth)
        for month, name in zip((7, 8, 9, 10), _MONTHS, strict=True)
        for depth in ("surface", 100)
    ]
    for page, (month, name, depth) in zip(woa, expected, strict=True):
        assert page.title.startswith(f"Nutrients vs WOA23 — {name} 2010")
        assert page.kwargs["reference"] == [
            f"woa23_{v}_month{month:02d}"
            for v in ("nitrate", "phosphate", "silicate", "oxygen")
        ]
        assert page.kwargs["select"]["test"]["depth"] == depth
        assert page.kwargs["select"]["reference"] == {"depth": depth}


def test_pacmed_review_glodap_pages_get_one_depth_each_and_the_run_window(pacmed):
    _, expanded = pacmed
    for label, depth in (("surface", "surface"), ("100 m", 100)):
        page = _by_title(expanded, f"Alkalinity & DIC vs GLODAPv2 — run mean ({label})")
        assert page.kwargs["reference"] == ["glodap"]
        assert page.kwargs["depths"] == [depth]
        assert page.kwargs["aggregate"] == {"time": "mean"}
        assert page.kwargs["select"]["test"]["time"] == RUN_WINDOW
