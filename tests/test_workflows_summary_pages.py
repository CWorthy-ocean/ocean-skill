"""``summary:`` pages: ``kind: [table, target]``, ``scope: section``, the table figure.

A summary page pools the metric records of the compare pages drawn so far (or only the
ones since the last ``section:`` divider), and draws each ``kind`` as one figure. The
table kind is the pooled metrics as a table (:meth:`MetricsTable.to_figures`), cut into
figures of ``max_rows`` rows, with its numbers also written beside ``metrics.csv``.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ocean_skill.config import SuiteConfig
from ocean_skill.workflows import pages as P
from ocean_skill.workflows.run import run_suite
from tests.test_workflows_run import (  # noqa: F401  (stub_model is a fixture)
    _write_suite,
    stub_model,
)

_VARS = [
    ("sea_water_potential_temperature", "degC"),
    ("nitrate", "mmol m-3"),
    ("sea_water_practical_salinity", "1"),
]


def _record(variable, units, depth="surface", **extra):
    return {
        "variable": variable,
        "test": "ROMS",
        "reference": "WOA23",
        "depth": depth,
        "units": units,
        "corr": 0.93,
        "bias": -0.2123,
        "sigma_ratio": 1.1,
        "std_test": 1.0,
        "std_reference": 1.0,
        "crmsd": 0.21,
        "rmse": 1.2345,
        "n": 1234,
        **extra,
    }


def _records(n=3):
    out = []
    for i in range(n):
        var, units = _VARS[i % len(_VARS)]
        out.append(_record(var, units, depth=str(10 * (i // 3)), bias=0.1 * (i + 1)))
    return out


def _page(summary, **top):
    suite = SuiteConfig.model_validate(
        {
            "name": "t",
            "defaults": {"test": "stub"},
            "pages": [{"title": "Summary", "summary": summary}],
            **top,
        }
    )
    (page,) = P.expand(suite)
    return page


def _table_of(fig):
    (tab,) = [t for ax in fig.axes for t in ax.tables]
    return tab


def _cell_text(fig):
    return {k: c.get_text().get_text() for k, c in _table_of(fig).get_celld().items()}


# -- grammar ---------------------------------------------------------------------------


def test_default_kind_is_the_diagram_it_always_drew_plus_a_table():
    page = _page({"color_by": "variable"})
    assert page.kwargs["kind"] == ["both", "table"]
    assert page.kwargs["scope"] == "all"


def test_a_single_kind_string_still_means_one_kind():
    assert _page({"kind": "portrait"}).kwargs["kind"] == ["portrait"]


@pytest.mark.parametrize(
    "summary, message",
    [
        ({"kind": ["table", "bogus"]}, "not a known kind"),
        ({"kind": []}, "non-empty list"),
        ({"kind": ["table", "table"]}, "names a kind twice"),
        ({"scope": "everything"}, "not a known scope"),
    ],
)
def test_bad_kind_or_scope_is_a_schema_error(summary, message):
    with pytest.raises(ValueError, match=message):
        _page(summary)


def test_placeholders_in_kind_are_resolved_before_they_are_checked():
    page = _page({"kind": "{k}"}, defaults={"test": "stub", "k": "target"})
    assert page.kwargs["kind"] == ["target"]


# -- build -----------------------------------------------------------------------------


def test_each_kind_is_one_figure_with_its_own_suffix():
    page = _page({"kind": ["table", "target"], "scope": "section"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = P.build(page, pooled_records=_records(3))
    assert [s for s, _ in results] == ["_table", "_target"]
    assert _table_of(results[0][1]) is not None
    assert not results[1][1].axes[0].tables


def test_one_kind_has_no_suffix():
    page = _page({"kind": "table"})
    ((suffix, _),) = P.build(page, pooled_records=_records(3))
    assert suffix == ""


def test_table_rows_are_the_pooled_comparisons_with_three_figure_numbers_and_units():
    page = _page({"kind": "table"})
    ((_, fig),) = P.build(page, pooled_records=_records(3))
    text = _cell_text(fig)
    header = [text[0, c] for c in range(8)]
    for name in ("bias", "RMSE", "CRMSD", "corr", "σ ratio", "n", "Units"):
        assert name in header
    body = [[text[r, c] for c in range(len(header))] for r in (1, 2, 3)]
    assert [row[header.index("Units")] for row in body] == ["°C", "mmol/m³", ""]
    assert {row[header.index("RMSE")] for row in body} == {"1.23"}
    assert {row[header.index("n")] for row in body} == {"1234"}
    assert fig.get_suptitle() == "Summary"
    assert fig.get_size_inches()[0] == pytest.approx(8.5)


def test_a_long_table_is_cut_into_continued_figures():
    page = _page({"kind": "table", "max_rows": 4})
    results = P.build(page, pooled_records=_records(9))
    assert [s for s, _ in results] == ["", "_part2", "_part3"]
    assert [f.get_suptitle() for _, f in results] == [
        "Summary",
        "Summary (cont.)",
        "Summary (cont.)",
    ]
    rows = [len(_table_of(f).get_celld()) // 8 - 1 for _, f in results]
    assert rows == [4, 4, 1]


def test_the_table_leaves_a_frame_with_units_for_the_csv():
    page = _page({"kind": "table"})
    P.build(page, pooled_records=_records(3))
    frame = page.summary_table
    assert list(frame["units"]) == ["°C", "mmol/m³", ""]
    assert {"bias", "rmse", "corr", "n"} <= set(frame.columns)
    assert len(frame) == 3


def test_a_diagram_only_summary_writes_no_table_frame():
    page = _page({"kind": "target"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P.build(page, pooled_records=_records(3))
    assert page.summary_table is None


def test_table_options_do_not_reach_the_diagrams(monkeypatch):
    import ocean_skill as osk

    seen = []
    monkeypatch.setattr(
        osk, "summary", lambda recs, **kw: seen.append(kw) or plt.figure()
    )
    page = _page({"kind": ["target", "table"], "max_rows": 5, "color_by": "variable"})
    P.build(page, pooled_records=_records(3))
    assert seen == [{"kind": "target", "color_by": "variable", "size": "page"}]


# -- the runner: scope, pool-once and the CSV ----------------------------------------


def _suite(tmp_path, summaries):
    def compare(title):
        return {
            "title": title,
            "compare": {
                "reference": ["woa23_nitrate_month01"],
                "variables": ["nitrate"],
            },
        }

    return {
        "name": "scoped",
        "output_dir": str(tmp_path / "out"),
        "defaults": {"test": "stub"},
        "cover": False,
        "pages": [
            compare("A1"),
            compare("A2"),
            {"title": "Part 2", "section": "second half"},
            compare("B1"),
            *summaries,
        ],
    }


def _patch_build(monkeypatch, seen):
    """Fake the compare pages (one record each, titled); summaries stay real."""
    real = P.build

    def fake_build(page, *, pooled_records=None):
        if page.kind == "compare":
            var, units = _VARS[len(seen["compare"]) % len(_VARS)]
            seen["compare"].append(page.title)
            page.metrics_records = [_record(var, units, label=page.title)]
            # a long page draws several figures, but its records still count once
            return [("", plt.figure()), ("_part2", plt.figure())]
        seen["pooled"][page.title] = [r["label"] for r in pooled_records]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return real(page, pooled_records=pooled_records)

    monkeypatch.setattr(P, "build", fake_build)


@pytest.mark.usefixtures("stub_model")
def test_scope_section_pools_only_records_since_the_last_divider(tmp_path, monkeypatch):
    seen = {"compare": [], "pooled": {}}
    _patch_build(monkeypatch, seen)
    path = _write_suite(
        tmp_path,
        _suite(
            tmp_path,
            [
                {"title": "Part", "summary": {"kind": "table", "scope": "section"}},
                {"title": "Whole", "summary": {"kind": "table"}},
            ],
        ),
    )
    result = run_suite(path)

    assert result.exit_code == 0
    assert seen["pooled"] == {"Part": ["B1"], "Whole": ["A1", "A2", "B1"]}
    metrics = pd.read_csv(result.metrics)
    assert list(metrics["label"]) == ["A1", "A2", "B1"]


@pytest.mark.usefixtures("stub_model")
def test_each_table_summary_writes_its_own_csv_beside_metrics_csv(
    tmp_path, monkeypatch
):
    seen = {"compare": [], "pooled": {}}
    _patch_build(monkeypatch, seen)
    path = _write_suite(
        tmp_path,
        _suite(
            tmp_path,
            [{"title": "Part", "summary": {"kind": "table", "scope": "section"}}],
        ),
    )
    result = run_suite(path)

    table_csv = result.metrics.parent / "scoped_part.csv"
    assert table_csv.exists()
    frame = pd.read_csv(table_csv)
    assert list(frame.columns[:2]) == ["row", "units"]
    assert len(frame) == 1
    assert any("part" in p.name for p in result.figures)


@pytest.mark.usefixtures("stub_model")
def test_a_diagram_only_summary_writes_no_table_csv(tmp_path, monkeypatch):
    seen = {"compare": [], "pooled": {}}
    _patch_build(monkeypatch, seen)
    path = _write_suite(
        tmp_path,
        _suite(tmp_path, [{"title": "Part", "summary": {"kind": "target"}}]),
    )
    result = run_suite(path)
    assert not (result.metrics.parent / "scoped_part.csv").exists()


def test_tall_rows_end_a_figure_before_it_outgrows_a_page():
    records = [
        _record(
            *_VARS[i % 3],
            depth=10 * i,
            bias=0.1 * i,
            test=f"pyesper_boundary_physics_south_{i}",
            reference=f"pacmed2015_boundary_physics_south_{i}",
        )
        for i in range(30)
    ]
    page = _page({"kind": "table", "max_rows": 0})
    results = P.build(page, pooled_records=records)
    assert len(results) > 1
    assert all(f.get_size_inches()[1] <= 11.0 for _, f in results)
    assert results[1][1].get_suptitle() == "Summary (cont.)"


def test_no_cell_text_runs_outside_its_cell():
    records = [
        _record(
            *_VARS[i % 3],
            depth=200,
            time="2011-01-01-2011-04-30",
            test=f"pyesper_boundary_physics_south_{i}",
            reference=f"pacmed2015_boundary_physics_south_{i}",
        )
        for i in range(6)
    ]
    ((_, fig),) = P.build(_page({"kind": "table"}), pooled_records=records)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for cell in _table_of(fig).get_celld().values():
        text, box = (
            cell.get_text().get_window_extent(renderer),
            cell.get_window_extent(renderer),
        )
        assert box.x0 - 0.5 <= text.x0 and text.x1 <= box.x1 + 0.5


# -- pooled labels ------------------------------------------------------------------


def test_a_label_no_other_record_shares_is_kept():
    records = [{"label": "temperature"}, {"label": "salinity"}]
    assert P._distinct_labels(records) == ["temperature", "salinity"]


def test_shared_labels_get_the_fields_that_tell_them_apart():
    records = [
        {"label": "temperature", "statistic": "mean", "depth": 0},
        {"label": "temperature", "statistic": "std", "depth": 0},
        {"label": "salinity", "statistic": "mean", "depth": 0},
    ]
    assert P._distinct_labels(records) == [
        "temperature · mean",
        "temperature · std",
        "salinity",
    ]


def test_a_depth_band_and_a_time_window_are_spelled_not_printed_as_dicts():
    window = {"min": "2010-01-01", "max": "2013-12-31"}
    records = [
        {"label": "temperature", "depth": {"min": 0, "max": 500}, "time": window},
        {"label": "temperature", "depth": [5, 50], "time": "2010-01"},
    ]
    labels = P._distinct_labels(records)
    assert labels == [
        "temperature · 0-500 m · 2010-01-01–2013-12-31",
        "temperature · 5 m, 50 m · 2010-01",
    ]
    assert not any("{" in label for label in labels)


def test_records_alike_in_every_field_are_numbered():
    records = [{"variable": "mld"}, {"variable": "mld"}]
    assert P._distinct_labels(records) == ["mld (1)", "mld (2)"]


def test_a_summary_table_draws_two_statistics_of_one_variable(tmp_path):
    page = P.ExpandedPage(
        title="Tracers — metrics",
        kind="summary",
        kwargs={"kind": ["table"], "scope": "all"},
        plot={},
        cache=False,
    )
    pooled = [
        {"label": "temperature", "variable": "temperature", "statistic": s,
         "bias": b, "rmse": 1.0, "crmsd": 0.9, "corr": 0.9, "sigma_ratio": 1.1,
         "n": 10, "units": "degC"}
        for s, b in (("mean", 0.1), ("std", 0.2))
    ]
    figs = P.build(page, pooled_records=pooled)
    assert figs and all(f is not None for _, f in figs)
    text = page.summary_table.to_string()
    assert "temperature · mean" in text and "temperature · std" in text
    for _, f in figs:
        plt.close(f)


def test_rows_of_one_page_that_differ_only_by_variable_are_named_by_it():
    records = [
        {"label": "surface mean", "variable": "temperature"},
        {"label": "surface mean", "variable": "salinity"},
    ]
    assert P._distinct_labels(records) == [
        "temperature surface mean",
        "salinity surface mean",
    ]


# -- row names: the figures' short variable names, wrapped at word boundaries ----------

_NITRATE = "mole_concentration_of_nitrate_in_sea_water"
_OXYGEN = "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"
_CHLOROPHYLL = "chlorophyll_a_in_sea_water"


def test_rows_named_by_a_standard_name_read_as_the_figures_name_the_variable():
    records = [
        {"variable": _NITRATE},
        {"variable": _OXYGEN},
        {"label": f"{_CHLOROPHYLL} · surface", "variable": _CHLOROPHYLL},
        {
            "label": "temperature · surface",
            "variable": "sea_water_potential_temperature",
        },
    ]
    assert P._distinct_labels(records) == [
        "nitrate",
        "oxygen",
        "chlorophyll a · surface",
        "temperature · surface",
    ]


def test_a_variable_leading_a_disambiguated_label_is_short_too():
    records = [
        {"label": "2010-01", "variable": _NITRATE},
        {"label": "2010-01", "variable": _OXYGEN},
    ]
    assert P._distinct_labels(records) == ["nitrate 2010-01", "oxygen 2010-01"]


def test_source_names_and_phrases_are_not_taken_for_standard_names():
    records = [
        {"label": "mld_bry_phys_strip", "variable": "x"},
        {"label": "pyesper_ic · mld_ic", "variable": "x"},
        {"label": "surface mean", "variable": "x"},
    ]
    assert P._distinct_labels(records) == [
        "mld_bry_phys_strip",
        "pyesper_ic · mld_ic",
        "surface mean",
    ]


def test_table_text_wraps_at_words_and_cuts_long_tokens_only_at_underscores():
    from ocean_skill.table import _wrap_cell

    assert _wrap_cell("temperature · surface · 2010-01", 20) == [
        "temperature ·",
        "surface · 2010-01",
    ]
    lines = _wrap_cell(f"{_OXYGEN} 2010-01 · mld_bry_bgc_strip", 24)
    for line in lines:
        # every line is whole words, or pieces cut right after an underscore
        assert not line.startswith("_")
    assert "".join(
        ln if ln.endswith("_") else ln + " " for ln in lines
    ).split() == f"{_OXYGEN} 2010-01 · mld_bry_bgc_strip".split()
    assert all(len(line) <= 24 for line in lines)
    # a token with nothing to cut at overflows rather than splitting
    assert _wrap_cell("a" * 30 + " b", 10) == ["a" * 30, "b"]
    assert _wrap_cell("", 10) == [""]


def test_a_long_standard_name_is_never_split_mid_word_in_the_table_figure():
    records = [
        _record(_OXYGEN, "mmol m-3", test="mld_bry_bgc_strip", reference="x"),
        _record(_NITRATE, "mmol m-3", test="mld_bry_bgc_strip", reference="y"),
    ]
    page = _page({"kind": "table"})
    ((_, fig),) = P.build(page, pooled_records=records)
    cells = [c.get_text().get_text() for c in _table_of(fig).get_celld().values()]
    flat = " ".join(cells)
    assert "mole_concentration" not in flat  # now the short name
    assert "oxygen" in flat and "nitrate" in flat
    plt.close(fig)
