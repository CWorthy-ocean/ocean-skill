"""The metrics table: ``osk.table`` / ``ComparisonSet.table`` and :class:`MetricsTable`.

What these defend: the table is the metric records pivoted (same pooling as
``osk.summary``, same duplicate-cell refusal as the portrait), that the one internal
layout -- merged cells included -- is what every writer renders, and that the
paper-table features (seasonal sub-rows, ``R² (surface / column)`` joined cells,
notes) land where they are asked to.

Offline: hand-built comparisons carrying a metrics record, as in ``test_pooling``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ocean_skill as osk
from ocean_skill.comparison import ComparisonSet
from ocean_skill.table import MetricsTable

TEMP = "sea_water_temperature"
SALT = "sea_water_salinity"
UEAST = "eastward_sea_water_velocity"
ALK = "sea_water_alkalinity"


class _Fake:
    """A comparison's specification plus a metrics record, and nothing else."""

    def __init__(
        self,
        label,
        *,
        variable=TEMP,
        depth="surface",
        time=None,
        test="roms",
        reference="obs",
        units="degC",
        combined=None,
        notes=None,
        demeaned="raw",
        **metrics,
    ):
        self.variable = variable
        self.standard_name = variable
        self.select = {"depth": depth}
        if time is not None:
            self.select["time"] = time
        self.test_name = test
        self.reference_name = reference
        self.label = label
        self.aggregate = None
        self.method = "conservative_normed"
        self.over = None
        self.time_method = "auto"
        self.tolerance = None
        self.bin_anchor = "auto"
        self.units = units
        self._combined = combined
        self.notes = notes
        corr = metrics.pop("corr", 0.9)
        self._record = {
            "bias": 0.1,
            "rmse": 0.5,
            "crmsd": 0.4,
            "corr": corr,
            "r2": corr**2,
            "sigma_ratio": 1.05,
            "n": 100,
            "variable": variable,
            "depth": depth,
            "time": time,
            "test": test,
            "reference": reference,
            "demeaned": demeaned,
            "detided": "raw",
            **metrics,
        }
        if notes:
            self._record["notes"] = notes

    def metrics(self):
        return self._record

    def as_item(self):
        return {"metrics": self._record, "label": self.label}


def _cs(*pairs):
    """A set from ``(label, _Fake)``-style fakes, each named by its own label."""
    return ComparisonSet(list(pairs), labels=[c.label for c in pairs])


def _target_set():
    """Build the target table's comparisons: seasonal for SST/Salinity only."""
    comps = [
        _Fake("SST", depth="surface", bias=0.10, rmse=0.50, corr=0.95, units="degC"),
        _Fake("SST", depth="column", bias=0.05, rmse=0.40, corr=0.92, units="degC"),
        _Fake("SST", depth="surface", time="MAM", bias=0.21, units="degC"),
        _Fake("SST", depth="surface", time="JJA", bias=0.32, units="degC"),
        _Fake("SST", depth="surface", time="SON", bias=-0.13, units="degC"),
        _Fake("Salinity", variable=SALT, bias=0.02, rmse=0.11, corr=0.8, units="1"),
        _Fake("Salinity", variable=SALT, time="MAM", bias=0.03, units="1"),
        _Fake("Salinity", variable=SALT, time="JJA", bias=0.04, units="1"),
        _Fake(
            "Velocity east", variable=UEAST, bias=0.01, rmse=0.07, corr=0.6, units="m/s"
        ),
        _Fake("Alk", variable=ALK, bias=-2.0, rmse=9.0, corr=0.7, units="umol/kg"),
    ]
    return _cs(*comps)


def _plain_set():
    """The same rows with one whole-record comparison each: nothing to split or join."""
    return _cs(
        *(
            c
            for c in _target_set()
            if c.select.get("depth") == "surface" and "time" not in c.select
        )
    )


def _target_table(**kw):
    return _target_set().table(
        metrics=("bias", "r2", "rmse"),
        split={"bias": "time"},
        join={"r2": "depth", "rmse": "depth"},
        info={"SST": {"Type": "CTD"}},
        notes={"SST": "ice-free months only"},
        **kw,
    )


# --- pooling and the basic pivot ----------------------------------------------------


def test_default_table_is_one_row_per_comparison_with_default_metrics():
    cs = _cs(_Fake("a", depth=0), _Fake("b", depth=5))
    t = cs.table()
    assert isinstance(t, MetricsTable)
    df = t.to_dataframe()
    assert list(df.columns) == ["bias", "rmse", "crmsd", "corr", "sigma_ratio", "n"]
    assert list(df.index) == ["a", "b"]


def test_osk_table_is_callable_and_still_the_module():
    # `import ocean_skill.table` makes the submodule a package attribute; the package
    # keeps it callable so the function and the module can share one name
    assert callable(osk.table)
    assert osk.table.MetricsTable is MetricsTable
    assert "table" in osk.__all__ and "MetricsTable" in osk.__all__


def test_osk_table_pools_sets_and_a_lone_comparison_like_summary():
    a = ComparisonSet(
        [_Fake("x", variable=TEMP, depth=0), _Fake("y", variable=SALT, depth=0)]
    )
    b = ComparisonSet([_Fake("x", variable=TEMP, depth=50)])
    lone = _Fake("z", variable=ALK, depth=0)
    t = osk.table([a, [b], lone], metrics=("bias",))
    labels = list(t.to_dataframe().index)
    assert len(labels) == 4
    assert len(set(labels)) == 4  # pooled names tell the members apart


def test_dict_keys_are_labels_and_the_group_field():
    hind = ComparisonSet(
        [_Fake("t", variable=TEMP, depth=0), _Fake("s", variable=SALT, depth=0)]
    )
    fore = ComparisonSet([_Fake("t", variable=TEMP, depth=0, test="fc", bias=0.9)])
    t = osk.table(
        {"hindcast": hind, "forecast": fore},
        rows="variable",
        cols="group",
        metrics=("bias",),
    )
    df = t.to_dataframe()
    assert set(df.columns) == {("bias", "hindcast"), ("bias", "forecast")}
    assert df[("bias", "forecast")].dropna().iloc[0] == pytest.approx(0.9)
    # without cols the keys name the rows
    assert list(
        osk.table({"hindcast": hind[0], "forecast": fore}, metrics=("bias",))
        .to_dataframe()
        .index
    ) == [
        "hindcast",
        "forecast",
    ]


def test_metric_selection_order_and_unknown_raises():
    cs = _cs(_Fake("a"))
    assert list(cs.table(metrics=("rmse", "bias")).to_dataframe().columns) == [
        "rmse",
        "bias",
    ]
    with pytest.raises(ValueError, match="no metric"):
        cs.table(metrics=("bias", "nonsense"))


def test_cols_pivots_a_field_into_metric_level_columns():
    cs = _cs(
        _Fake("A", test="r1", bias=1.0),
        _Fake("A", test="r2", bias=2.0),
        _Fake("B", variable=SALT, test="r1", bias=3.0),
    )
    t = cs.table(metrics=("bias", "rmse"), cols="test")
    df = t.to_dataframe()
    assert list(df.columns) == [
        ("bias", "r1"),
        ("bias", "r2"),
        ("rmse", "r1"),
        ("rmse", "r2"),
    ]
    assert df.loc["A", ("bias", "r2")] == 2.0
    assert np.isnan(df.loc["B", ("bias", "r2")])
    # two header rows: the metric groups over the levels
    assert len(t._layout.header) == 2
    assert "bias" in str(t).splitlines()[0]


def test_rows_by_a_record_field_uses_pretty_levels():
    cs = _cs(
        _Fake("a", depth={"min": 0, "max": 100}),
        _Fake("b", depth=200),
    )
    df = cs.table(rows="depth", metrics=("bias",)).to_dataframe()
    assert list(df.index) == ["0-100 m", "200 m"]


def test_two_comparisons_in_one_cell_raise_like_the_portrait():
    cs = _cs(_Fake("A", depth=0), _Fake("A", depth=5))
    with pytest.raises(ValueError, match="more than one comparison"):
        cs.table(metrics=("bias",))
    # telling them apart by cols resolves it
    assert cs.table(metrics=("bias",), cols="depth").to_dataframe().shape == (1, 2)


# --- split, join, layouts ------------------------------------------------------------


def test_split_rows_draws_the_level_as_a_sub_column_left_of_the_split_metric():
    t = _target_table()
    head = [c.text for c in t._layout.header[0]]
    # the level column sits immediately left of the first split metric (bias)
    assert head[:3] == ["", "Season", "bias"] or head[1] == "Season"
    assert head[2].startswith("bias")
    body = t._layout.body
    sst = body[:3]
    levels = [row[head.index("Season")].text for row in sst]
    assert levels == ["MAM", "JJA", "SON"]  # season order, not alphabetical
    # the row label spans all three sub-rows
    assert sst[0][0].rowspan == 3 and sst[1][0] is None and sst[2][0] is None


def test_split_level_column_sits_before_the_first_split_metric_not_at_the_far_left():
    cs = _target_set()
    t = cs.table(metrics=("r2", "bias"), split={"bias": "time"}, join={"r2": "depth"})
    heads = [c.path[0] for c in t._layout.cols]
    assert heads.index("Season") == heads.index("bias") - 1
    assert heads.index("Season") > heads.index(
        next(h for h in heads if h.startswith("R²"))
    )


def test_row_without_seasonal_records_gets_one_line_with_the_whole_value():
    t = _target_table()
    df = t.to_dataframe()
    assert df.index.names == ["row", "level"]
    assert list(df.loc["Alk"].index) == [""]
    assert df.loc[("Alk", ""), "bias"] == pytest.approx(-2.0)
    assert df.loc[("SST", "MAM"), "bias"] == pytest.approx(0.21)
    # Salinity only has MAM and JJA: its sub-rows are exactly those
    assert list(df.loc["Salinity"].index) == ["MAM", "JJA"]
    text = str(t).splitlines()
    assert sum(line.startswith("Alk") for line in text) == 1


def test_split_columns_layout_has_metric_level_columns_and_no_row_spans():
    t = _target_set().table(
        metrics=("bias", "rmse"),
        split={"bias": "time"},
        join={"rmse": "depth"},
        layout="columns",
    )
    paths = [c.path for c in t._layout.cols]
    assert ("bias (°C)", "MAM") not in paths  # mixed units: no shared suffix
    assert [p for p in paths if p[0] == "bias"] == [
        ("bias", "MAM"),
        ("bias", "JJA"),
        ("bias", "SON"),
        ("bias", "all"),  # rows with no seasonal records read their whole value here
    ]
    assert all(c is None or c.rowspan == 1 for line in t._layout.body for c in line)
    df = t.to_dataframe()
    assert df.loc["SST", ("bias", "MAM")] == pytest.approx(0.21)
    assert df.loc["Alk", ("bias", "all")] == pytest.approx(-2.0)
    assert np.isnan(df.loc["SST", ("bias", "all")])  # SST has seasons: no "all" value
    assert df.loc["Alk", ("rmse", "surface")] == pytest.approx(9.0)


def test_unsplit_metric_reads_the_whole_record_not_a_season():
    cs = _cs(
        _Fake("A", rmse=0.5),
        _Fake("A", time="MAM", rmse=9.0, bias=1.0),
        _Fake("A", time="JJA", rmse=8.0, bias=2.0),
    )
    df = cs.table(metrics=("rmse", "bias"), split={"bias": "time"}).to_dataframe()
    assert df.loc[("A", "MAM"), "rmse"] == pytest.approx(0.5)  # the whole-record value
    assert np.isnan(
        df.loc[("A", "JJA"), "rmse"]
    )  # first line only, like the merged cell
    assert df.loc[("A", "JJA"), "bias"] == pytest.approx(2.0)


def test_join_puts_levels_in_one_cell_with_a_missing_slot():
    t = _target_table()
    text = t.to_markdown()
    sst = next(
        line for line in text.splitlines() if "ice-free" in line or "SST" in line
    )
    assert "0.90 / 0.85" in sst  # corr 0.95**2, 0.92**2 -> surface / column
    sal = next(line for line in text.splitlines() if "Salinity" in line)
    assert "0.64 / –" in sal  # no column-depth Salinity record
    assert "R² (surface / column)" in text.splitlines()[0]
    assert "RMSE (surface / column)" in text.splitlines()[0]


def test_join_labels_rename_the_levels():
    cs = _target_set()
    t = cs.table(
        metrics=("r2", "bias"),
        split={"bias": "time"},
        join={"r2": "depth"},
        join_labels={"depth": ["top", "whole column"]},
    )
    assert "R² (top / whole column)" in t.to_markdown().splitlines()[0]
    with pytest.raises(ValueError, match="one name per level"):
        cs.table(
            metrics=("r2", "bias"),
            split={"bias": "time"},
            join={"r2": "depth"},
            join_labels={"depth": ["only"]},
        )


def test_split_and_join_misuse_raises():
    cs = _target_set()
    with pytest.raises(ValueError, match="not among"):
        cs.table(metrics=("bias",), split={"rmse": "time"})
    with pytest.raises(ValueError, match="the same field"):
        cs.table(metrics=("bias",), split={"bias": "time"}, join={"bias": "time"})
    with pytest.raises(ValueError, match="no record has"):
        cs.table(metrics=("bias",), split={"bias": "nope"})
    with pytest.raises(ValueError, match="layout="):
        cs.table(layout="diagonal")


def test_duplicate_cell_inside_a_split_raises():
    cs = _cs(
        _Fake("A", time="MAM", depth=0),
        _Fake("A", time="MAM", depth=5),
    )
    with pytest.raises(ValueError, match="more than one comparison"):
        cs.table(metrics=("bias",), split={"bias": "time"})


def test_depth_band_levels_get_labels_in_joins_and_notes():
    cs = _cs(
        _Fake("A", depth={"min": 0, "max": 5}),
        _Fake("A", depth={"min": 5, "max": 100}),
    )
    t = cs.table(metrics=("rmse",), join={"rmse": "depth"})
    assert "RMSE (°C; 0-5 m / 5-100 m)" in t.to_markdown().splitlines()[0]


# --- units, headers, round ----------------------------------------------------------


def test_shared_units_go_in_dimensional_headers_only():
    cs = _cs(_Fake("a"), _Fake("b", depth=5))
    head = cs.table(metrics=("bias", "corr", "n")).to_markdown().splitlines()[0]
    assert "bias (°C)" in head
    assert "corr |" in head + " |" and "corr (" not in head
    assert "n (" not in head


def test_mixed_units_become_a_per_row_units_column():
    t = _plain_set().table(metrics=("bias",))
    md = t.to_markdown().splitlines()
    assert "Units" in md[0] and "bias |" in md[0] + " |"
    assert any("umol/kg" in line and "Alk" in line for line in md)
    assert (
        "Units" not in _plain_set().table(metrics=("bias",), units=False).to_markdown()
    )


def test_headers_override_default_names():
    cs = _cs(_Fake("a"))
    head = (
        cs.table(
            metrics=("rmse", "r2"),
            units=False,
            auto_notes=False,
            headers={"rmse": "root MSE", "label": "Station"},
        )
        .to_markdown()
        .splitlines()[0]
    )
    assert head == "| Station | root MSE | R² |"


def test_round_formats_like_the_portrait_and_false_keeps_digits():
    cs = _cs(_Fake("a", bias=0.123456, corr=0.98765, n=12))
    row = (
        cs.table(metrics=("bias", "corr", "n"), units=False, auto_notes=False)
        .to_markdown()
        .splitlines()[2]
    )
    assert row == "| a | 0.123 | 0.99 | 12 |"
    raw = (
        cs.table(metrics=("bias",), units=False, round=False, auto_notes=False)
        .to_markdown()
        .splitlines()[2]
    )
    assert "0.123456" in raw


# --- notes and info -----------------------------------------------------------------


def test_auto_notes_describe_selection_pooling_and_demeaning():
    cs = _cs(
        _Fake(
            "deep",
            depth={"min": 100, "max": 200},
            combined=("pool", 8),
            demeaned="demeaned",
        ),
        _Fake("avg", depth=0, time="2020-01", combined=("average", 3)),
    )
    md = cs.table(metrics=("bias",), units=False).to_markdown().splitlines()
    assert "100-200 m; sample average (8); demeaned" in md[2]
    assert "0 m; 2020-01; station average (3)" in md[3]


def test_auto_notes_switches_and_overrides():
    cs = _cs(_Fake("a", combined=("pool", 4)), _Fake("b", depth=5))
    assert "Notes" not in cs.table(metrics=("bias",), auto_notes=False).to_markdown()
    custom = cs.table(
        metrics=("bias",), auto_notes=lambda rec: f"v={rec['variable'][:3]}"
    )
    assert "v=sea" in custom.to_markdown()
    over = cs.table(metrics=("bias",), auto_notes={"a": "my text"}).to_markdown()
    assert "my text" in over and "5 m" in over
    with pytest.raises(ValueError, match="auto_notes"):
        cs.table(auto_notes=3)


def test_user_notes_from_the_record_and_overridden_by_table():
    cs = _cs(_Fake("a", notes="from compare"), _Fake("b", depth=5, notes="also"))
    md = cs.table(metrics=("bias",), auto_notes=False).to_markdown().splitlines()
    assert "User notes" in md[0] and "from compare" in md[2]
    md = cs.table(
        metrics=("bias",), auto_notes=False, notes={"a": "mine"}
    ).to_markdown()
    assert "mine" in md and "from compare" not in md and "also" in md
    # no notes anywhere: the column is left out
    assert "User notes" not in _cs(_Fake("a")).table(metrics=("bias",)).to_markdown()


def test_info_columns_follow_the_metrics_and_span_the_row():
    t = _target_table()
    heads = [c.path[0] for c in t._layout.cols]
    assert heads.index("Type") > max(
        i for i, h in enumerate(heads) if h.startswith("RMSE")
    )
    j = heads.index("Type")
    assert t._layout.body[0][j].text == "CTD" and t._layout.body[0][j].rowspan == 3
    assert t._layout.body[3][j].text == ""  # the other rows have none


def test_notes_do_not_enter_the_cache_key_or_identity():
    from ocean_skill.comparison import Comparison, _identity

    kw = {"reference": "obs", "test": "roms", "variable": TEMP, "select": {"depth": 0}}
    a, b = Comparison(**kw), Comparison(**kw)
    b.notes = "ice-free months only"
    assert a.notes is None
    assert a._cache_key == b._cache_key
    assert _identity(a) == _identity(b)


def test_compare_notes_reaches_the_metrics_record(monkeypatch):
    """`compare(notes=)` sets the attribute and the record carries it.

    It is not a Comparison() constructor argument, so it cannot reach identity/keys.
    """
    import inspect

    from ocean_skill import comparison as cmp

    assert "notes" in inspect.signature(cmp.compare).parameters
    assert "notes" not in inspect.signature(cmp.Comparison.__init__).parameters
    src = inspect.getsource(cmp.Comparison.metrics)
    assert '"notes": self.notes' in src


# --- the writers ---------------------------------------------------------------------


def test_html_has_real_rowspans_and_escapes():
    t = _target_table()
    html = t._repr_html_()
    assert html == t.to_html()
    assert '<td rowspan="3"' in html
    assert "<table" in html and "R² (surface / column)" in html
    esc = _cs(_Fake("a<b>")).table(metrics=("bias",), units=False).to_html()
    assert "a&lt;b&gt;" in esc


def test_latex_uses_multirow_booktabs_and_escapes():
    tex = _target_table().to_latex()
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    assert r"\multirow{3}{*}{SST}" in tex
    assert r"R$^{2}$ (surface / column)" in tex
    assert r"$-$0.13" in tex  # a real minus sign
    assert r"\cmidrule" in tex
    special = (
        _cs(_Fake("50% & #1_x", sigma_ratio=1.2))
        .table(metrics=("sigma_ratio",), units=False)
        .to_latex()
    )
    assert r"50\% \& \#1\_x" in special
    assert r"$\sigma$ ratio" in special


def test_latex_two_level_header_uses_multicolumn():
    cs = _cs(_Fake("A", test="r1"), _Fake("A", test="r2"))
    tex = cs.table(metrics=("bias",), cols="test", units=False).to_latex()
    assert r"\multicolumn{2}{c}{bias}" in tex
    assert r"\cmidrule(lr){2-3}" in tex
    assert r"\multirow{2}{*}{}" in tex


def test_markdown_blanks_spanned_cells_and_is_a_pipe_table():
    md = _target_table().to_markdown().splitlines()
    assert md[1].startswith("| :---")
    sst_lines = [
        line for line in md if "MAM" in line or "JJA" in line or "SON" in line
    ][:3]
    assert (
        "SST" in sst_lines[0]
        and "SST" not in sst_lines[1]
        and "SST" not in sst_lines[2]
    )
    cells = [c.strip() for c in sst_lines[1].strip("|").split("|")]
    assert cells[0] == ""  # the row label is blank on the later sub-rows


def test_plain_text_is_aligned_with_spans_blank():
    text = str(_target_table())
    lines = text.splitlines()
    assert repr(_target_table()) == text
    assert "Season" in lines[0] and "bias" in lines[0]
    # columns are fixed width: "Season" starts at the same offset on every sub-row
    col = lines[0].index("Season")
    assert [
        line[col : col + 3] for line in lines[2:] if line.startswith(("SST", "   "))
    ][:3] == [
        "MAM",
        "JJA",
        "SON",
    ]
    assert sum(line.startswith("SST") for line in lines) == 1


def test_to_dataframe_is_numeric_and_to_tidy_is_long():
    t = _target_table()
    df = t.to_dataframe()
    assert df.loc[("SST", "JJA"), "bias"] == pytest.approx(0.32)
    # joined metrics expand to one column per level
    assert ("r2", "surface") in df.columns and ("r2", "column") in df.columns
    assert df.loc[("SST", "MAM"), ("r2", "surface")] == pytest.approx(0.95**2)
    assert np.isnan(df.loc[("SST", "JJA"), ("r2", "surface")])  # on the first line only
    tidy = t.to_tidy()
    assert {"row", "metric", "value"} <= set(tidy.columns)
    assert len(tidy) > len(df)


@pytest.mark.parametrize("suffix", [".tex", ".md", ".html", ".csv"])
def test_save_dispatches_on_suffix(tmp_path, suffix):
    t = _target_table()
    out = t.save(tmp_path / "sub" / f"tab{suffix}")
    text = out.read_text()
    assert out.exists() and text
    if suffix == ".tex":
        assert r"\multirow" in text
    elif suffix == ".md":
        assert text.startswith("|")
    elif suffix == ".html":
        assert "rowspan" in text
    else:
        csv = pd.read_csv(out)
        assert {"row", "metric", "value"} <= set(csv.columns)


def test_save_kwarg_and_unknown_suffix(tmp_path):
    cs = _plain_set()
    cs.table(metrics=("bias",), save=tmp_path / "t.md")
    assert (tmp_path / "t.md").exists()
    with pytest.raises(ValueError, match="suffix"):
        cs.table(metrics=("bias",), save=tmp_path / "t.docx")
    with pytest.raises(ValueError, match="suffix"):
        cs.table(metrics=("bias",)).save(tmp_path / "t")


def test_target_example_end_to_end():
    t = _target_table()
    md = t.to_markdown()
    lines = md.splitlines()
    # variable rows, a seasonal sub-column, joined R2/RMSE, info and notes columns
    assert "Type" in lines[0] and "User notes" in lines[0]
    assert (
        "| Notes |" not in lines[0]
    )  # nothing to say: depth/season are in the headers
    assert "CTD" in md and "ice-free months only" in md
    body_rows = len(lines) - 2
    assert body_rows == 3 + 2 + 1 + 1  # SST x3 seasons, Salinity x2, velocity, Alk
    assert isinstance(t._repr_html_(), str)


# --- r2 ------------------------------------------------------------------------------


def test_r2_is_corr_squared_and_registered_after_corr():
    from ocean_skill.metrics import REGISTRY, compute

    names = list(REGISTRY)
    assert names.index("r2") == names.index("corr") + 1
    assert REGISTRY["r2"].units == "1"

    import xarray as xr

    t = xr.DataArray(np.linspace(0, 1, 60) + 0.1 * np.sin(np.arange(60)), dims="x")
    r = xr.DataArray(np.linspace(0, 1, 60), dims="x")
    rec = compute(
        xr.Dataset({"test": t, "reference": r}), min_samples=1, weighted=False
    )
    assert rec["r2"] == pytest.approx(rec["corr"] ** 2)
    only = compute(
        xr.Dataset({"test": t, "reference": r}),
        names=["r2"],
        min_samples=1,
        weighted=False,
    )
    assert only["r2"] == pytest.approx(
        rec["corr"] ** 2
    )  # corr is computed to derive it


def test_labels_naming_split_and_join_levels_are_rebuilt_into_one_row_per_variable():
    # real pooled labels name every level ("temperature 0-5 m MAM"); with split= and
    # join= over those fields that would give each season and depth its own row
    comps = list(_target_set())
    labels = [
        " ".join(
            str(x)
            for x in (c.variable, c.select["depth"], c.select.get("time", "all"))
        )
        for c in comps
    ]
    t = ComparisonSet(comps, labels=labels).table(
        metrics=("r2", "rmse", "bias"),
        split={"bias": "time"},
        join={"r2": "depth", "rmse": "depth"},
    )
    rows = list(dict.fromkeys(t.to_dataframe().index.get_level_values(0)))
    assert len(rows) == 4
    assert not any("MAM" in r or "surface" in r for r in rows)


def test_row_units_fall_back_to_any_record_of_the_row():
    comps = [
        _Fake("SST", depth="surface", units=None),
        _Fake("SST", depth="column", units="degC"),
        _Fake("Alk", variable=ALK, units="umol/kg"),
    ]
    t = _cs(*comps).table(metrics=("bias",), join={"bias": "depth"})
    assert "°C" in t.to_markdown().splitlines()[2]


def test_a_split_metric_can_also_be_joined_by_another_field():
    # each season's bias at two depths: one line per season, "surface / column" in it
    cs = _cs(
        _Fake("T", depth="surface", bias=0.1),
        _Fake("T", depth="column", bias=0.2),
        _Fake("T", depth="surface", time="MAM", bias=0.31),
        _Fake("T", depth="column", time="MAM", bias=0.41),
        _Fake("T", depth="surface", time="JJA", bias=0.52),
        _Fake("A", variable=ALK, bias=-2.0),
    )
    t = cs.table(
        metrics=("rmse", "bias"),
        split={"bias": "time"},
        join={"rmse": "depth", "bias": "depth"},
    )
    md = t.to_markdown()
    assert "| MAM | 0.31 / 0.41 |" in md
    assert "| JJA | 0.52 / – |" in md
    assert "-2 / –" in md  # the unsplit row joins its whole-record value
    tidy = t.to_tidy()
    assert len(tidy[tidy.metric == "bias"]) == 4  # T's whole record shows no bias


def test_rows_ignore_names_that_only_change_with_the_season_or_depth():
    # a pooled reference reads "pool of 22 reference sources" in one season and
    # "pool of 20" in the next; that must not give each season its own row
    comps = [
        _Fake("x", depth=d, time=t, reference=f"pool of {n} reference sources")
        for n, (d, t) in enumerate(
            [(d, t) for d in ("surface", "column") for t in (None, "MAM", "JJA")]
        )
    ] + [
        _Fake("y", variable=SALT, depth=d, time=t, reference=f"pool of {9 - n}")
        for n, (d, t) in enumerate([("surface", None), ("surface", "MAM")])
    ]
    labels = [f"{c.variable} {c.reference_name}" for c in comps]
    t = ComparisonSet(comps, labels=labels).table(
        metrics=("rmse", "bias"),
        split={"bias": "time"},
        join={"rmse": "depth", "bias": "depth"},
    )
    rows = list(dict.fromkeys(t.to_dataframe().index.get_level_values(0)))
    assert rows == ["temperature", "salinity"]


def test_rows_by_variable_read_as_short_names():
    t = _plain_set().table(metrics=("bias",), rows="variable")
    rows = list(t.to_dataframe().index)
    assert "temperature" in rows[0] and "sea_water" not in rows[0]


def _user_pool():
    """The 40 comparisons of a real pooled set (2026-10-12), fields as metrics() gave.

    Labels name every level ("temperature 0-5 m MAM"); depth is the relabelled
    "Surface"/"Full column"; the whole record's time is a window ("Mar–Nov"); the
    pooled reference's name changes with season and depth; full-column temperature
    comes back without units.
    """
    refs = {
        TEMP: ([22, 20, 20, 22], [31, 32, 25, 36]),
        SALT: ([22, 20, 20, 22], [28, 27, 23, 31]),
        ALK: ([7] * 4, [7] * 4),
        "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water": (
            [7] * 4,
            [7] * 4,
        ),
        "mole_concentration_of_silicate_in_sea_water": ([7] * 4, [7] * 4),
    }
    short = {TEMP: "temperature", SALT: "salinity", ALK: "alkalinity"}
    comps, labels = [], []
    k = 0
    for var, by_depth in refs.items():
        for depth, band, ns in zip(("Surface", "Full column"), ("0-5", "0-65"), by_depth):
            for time, n in zip(("MAM", "JJA", "SON", "Mar–Nov"), ns):
                k += 1
                units = {TEMP: "degC", SALT: "PSU"}.get(var, "mmol/m3")
                if var == TEMP and depth == "Full column":
                    units = None
                comps.append(
                    _Fake(
                        f"x{k}",
                        variable=var,
                        depth=depth,
                        time=time,
                        reference=f"pool of {n} reference sources",
                        units=units,
                        bias=k / 100,
                    )
                )
                name = short.get(var, var.split("_of_")[-1].split("_in_")[0])
                tlabel = "MAMJJASON" if time == "Mar–Nov" else time
                labels.append(f"{name} {band} m {tlabel}")
    return ComparisonSet(comps, labels=labels)


def test_the_user_pool_gives_one_row_per_variable_with_seasons_and_depths_joined():
    t = _user_pool().table(
        metrics=("r2", "rmse", "bias"),
        split={"bias": "time"},
        join={"r2": "depth", "rmse": "depth", "bias": "depth"},
        join_labels={"depth": ["surface", "full column"]},
        headers={"label": "Variable", "bias": "Bias"},
    )
    df = t.to_dataframe()
    rows = list(dict.fromkeys(df.index.get_level_values(0)))
    assert len(rows) == 5
    assert rows[:3] == ["temperature", "salinity", "alkalinity"]
    assert len(df) == 15  # five variables x three seasons
    md = t.to_markdown()
    # temperature MAM: surface is comparison 1, full column comparison 5
    assert "| temperature | 0.81 / 0.81 | 0.5 / 0.5 | MAM | 0.01 / 0.05 |" in md
    # the whole-record (Mar–Nov) values feed R² and RMSE, not a fourth season line
    assert "Mar" not in md
    assert "°C" in md.splitlines()[2]  # full-column temperature borrows the units


# --- split_all: the whole-record value of a split metric as one more line ----------


def _season_whole_set():
    """Build a row with seasons and a whole-record window, and one with only a whole."""
    return _cs(
        _Fake("SST", time="Mar–Nov", bias=0.07, rmse=0.5, corr=0.9),
        _Fake("SST", time="MAM", bias=0.21),
        _Fake("SST", time="JJA", bias=0.32),
        _Fake("SST", time="SON", bias=-0.13),
        _Fake(
            "Alk", variable=ALK, time="Mar–Nov", bias=-2.0, rmse=9.0, units="umol/kg"
        ),
    )


def test_split_all_adds_a_whole_record_line_after_the_levels():
    t = _season_whole_set().table(
        metrics=("bias", "rmse"), split={"bias": "time"}, split_all="Mar–Nov"
    )
    head = [c.text for c in t._layout.header[0]]
    body = t._layout.body
    sst = body[:4]
    levels = [r[head.index("Season")].text for r in sst]
    assert levels == ["MAM", "JJA", "SON", "Mar–Nov"]
    bias_j = next(i for i, h in enumerate(head) if h.startswith("bias"))
    assert [r[bias_j].text for r in sst] == ["0.21", "0.32", "-0.13", "0.07"]
    # RMSE is one merged cell over all four lines, still the whole-record value
    rmse_j = next(i for i, h in enumerate(head) if h.startswith("RMSE"))
    assert sst[0][rmse_j].rowspan == 4 and sst[0][rmse_j].text == "0.5"
    assert sst[0][0].rowspan == 4
    df = t.to_dataframe()
    assert df.loc[("SST", "Mar–Nov"), "bias"] == pytest.approx(0.07)
    assert df.loc[("SST", "Mar–Nov"), "rmse"] == pytest.approx(0.5)  # whole-record line
    assert np.isnan(df.loc[("SST", "MAM"), "rmse"])
    tex = t.to_latex()
    assert r"\multirow{4}{*}{SST}" in tex
    assert r"\multirow{4}{*}{0.5}" in tex


def test_split_all_names_the_only_line_of_a_row_without_seasons():
    t = _season_whole_set().table(
        metrics=("bias", "rmse"), split={"bias": "time"}, split_all="All year"
    )
    df = t.to_dataframe()
    assert list(df.loc["Alk"].index) == ["All year"]
    assert df.loc[("Alk", "All year"), "bias"] == pytest.approx(-2.0)
    assert df.loc[("Alk", "All year"), "rmse"] == pytest.approx(9.0)
    assert sum(line.startswith("Alk") for line in str(t).splitlines()) == 1


def test_split_all_columns_layout_fills_every_row_and_replaces_the_all_column():
    t = _season_whole_set().table(
        metrics=("bias", "rmse"),
        split={"bias": "time"},
        split_all="Mar–Nov",
        layout="columns",
    )
    paths = [c.path for c in t._layout.cols if c.path[0] == "bias"]
    assert paths == [
        ("bias", "MAM"),
        ("bias", "JJA"),
        ("bias", "SON"),
        ("bias", "Mar–Nov"),
    ]
    df = t.to_dataframe()
    assert df.loc["SST", ("bias", "Mar–Nov")] == pytest.approx(0.07)
    assert df.loc["SST", ("bias", "MAM")] == pytest.approx(0.21)
    assert df.loc["Alk", ("bias", "Mar–Nov")] == pytest.approx(-2.0)


def test_split_all_line_keeps_a_joined_split_metric_joined():
    cs = _cs(
        _Fake("T", depth="surface", time="Mar–Nov", bias=0.1),
        _Fake("T", depth="column", time="Mar–Nov", bias=0.2),
        _Fake("T", depth="surface", time="MAM", bias=0.31),
        _Fake("T", depth="column", time="MAM", bias=0.41),
    )
    t = cs.table(
        metrics=("bias",),
        split={"bias": "time"},
        join={"bias": "depth"},
        split_all="Mar–Nov",
    )
    md = t.to_markdown()
    assert "| MAM | 0.31 / 0.41 |" in md
    assert "| Mar–Nov | 0.1 / 0.2 |" in md


def test_split_all_level_is_in_the_tidy_table():
    t = _season_whole_set().table(
        metrics=("bias", "rmse"), split={"bias": "time"}, split_all="Mar–Nov"
    )
    tidy = t.to_tidy()
    whole = tidy[(tidy.level == "Mar–Nov") & (tidy.metric == "bias")]
    assert sorted(whole.row) == ["Alk", "SST"]
    assert whole[whole.row == "SST"].value.iloc[0] == pytest.approx(0.07)


def test_split_all_files_merged_values_under_the_whole_record_line():
    # R²/RMSE are the whole-record values, so the data writers put them on the
    # split_all line -- not on the first season's, which they say nothing about
    cs = _cs(
        _Fake("T", depth="surface", time="Mar–Nov", bias=0.1, rmse=0.5, corr=0.9),
        _Fake("T", depth="column", time="Mar–Nov", bias=0.2, rmse=0.6, corr=0.8),
        _Fake("T", depth="surface", time="MAM", bias=0.31),
        _Fake("T", depth="column", time="MAM", bias=0.41),
    )
    t = cs.table(
        metrics=("r2", "rmse", "bias"),
        split={"bias": "time"},
        join={"r2": "depth", "rmse": "depth", "bias": "depth"},
        split_all="Mar–Nov",
    )
    df = t.to_dataframe()
    assert df.loc[("T", "Mar–Nov"), ("rmse", "surface")] == pytest.approx(0.5)
    assert df.loc[("T", "Mar–Nov"), ("r2", "surface")] == pytest.approx(0.81)
    assert np.isnan(df.loc[("T", "MAM"), ("rmse", "column")])
    assert np.isnan(df.loc[("T", "MAM"), ("r2", "surface")])
    tidy = t.to_tidy()
    assert set(tidy[tidy.metric.isin(["r2", "rmse"])].level) == {"Mar–Nov"}


def test_split_all_needs_split_and_a_name():
    cs = _season_whole_set()
    with pytest.raises(ValueError, match="split_all"):
        cs.table(metrics=("bias",), split_all="Mar–Nov")
    with pytest.raises(ValueError, match="split_all"):
        cs.table(metrics=("bias",), split={"bias": "time"}, split_all=3)


def test_split_all_default_leaves_the_table_as_it_was():
    cs = _season_whole_set()
    kw = {"metrics": ("bias", "rmse"), "split": {"bias": "time"}}
    assert str(cs.table(**kw)) == str(cs.table(split_all=None, **kw))
    assert "Mar–Nov" not in str(cs.table(**kw))
